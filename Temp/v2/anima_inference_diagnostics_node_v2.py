import json, math, time, uuid, threading, re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

_REC = {}

def _ints(s):
    out=set()
    for x in str(s).replace(' ','').split(','):
        if not x: continue
        if '-' in x:
            a,b=x.split('-',1); out.update(range(int(a),int(b)+1))
        else: out.add(int(x))
    return out

def _words(s): return [x.strip() for x in str(s).split(',') if x.strip()]
def _slug(s): return (re.sub(r'[^0-9A-Za-z가-힣_-]+','_',str(s)).strip('_')[:64] or 'word')
def _flat(tokens,key): return [x for g in tokens.get(key,[]) for x in g]
def _tid(x): return int(x[0]) if isinstance(x,(list,tuple)) else int(x)
def _txt(x): return str(x[1]) if isinstance(x,(list,tuple)) and len(x)>1 else str(x)
def _clean(s): return str(s).replace('▁','').replace('Ġ','')

def _detok(clip,pairs):
    try: return clip.tokenizer.t5xxl.untokenize(pairs)
    except Exception:
        try: return clip.tokenizer.untokenize(pairs)
        except Exception: return [(x[0],str(x[0])) for x in pairs]

def _word_ids(tm,word):
    clip=tm.get('_clip'); pieces=tm.get('token_texts',[])
    if clip is None: return []
    q=_detok(clip,_flat(clip.tokenize(word),'t5xxl'))
    needle=''.join(_clean(_txt(x)) for x in q if _txt(x) not in ('<pad>','</s>','')).lower()
    clean=[_clean(x) for x in pieces]
    for i in range(len(clean)):
        cur=''
        for j in range(i,len(clean)):
            if clean[j] in ('<pad>','</s>'): break
            cur+=clean[j]; low=cur.lower()
            if low==needle: return list(range(i,j+1))
            if len(low)>len(needle) or not needle.startswith(low): break
    return []

def _mapping(tm):
    lines=[f"prompt: {tm.get('prompt','')}",'T5 target tokens -> Anima cross-attention key indices:']
    for i,(tid,txt) in enumerate(zip(tm.get('token_ids',[]),tm.get('token_texts',[]))):
        lines.append(f'[{i:03d}] id={tid:<6d} token={txt!r}')
    return '\n'.join(lines)

def _grid(n,shape=None):
    ratio=1.0
    if shape and len(shape)>=2 and shape[-1]>0: ratio=float(shape[-2])/float(shape[-1])
    best=None; bs=1e99
    for h in range(1,int(n**0.5)+1):
        if n%h: continue
        w=n//h
        for a,b in ((h,w),(w,h)):
            sc=abs(math.log((a/b+1e-12)/(ratio+1e-12)))
            if sc<bs: bs,best=sc,(a,b)
    return best

def _spatial(vals,q,shape):
    a=np.asarray(vals,np.float32); sp=list(q.shape[1:-2])
    if len(sp)>=2:
        a=a.reshape(sp)
        while a.ndim>2: a=a.mean(0)
        return a
    g=_grid(a.size,shape); return None if not g else a.reshape(*g)

def _norm(a,lo=1,hi=99):
    a=np.asarray(a,np.float32); l,h=np.nanpercentile(a,[lo,hi])
    if not np.isfinite(l) or not np.isfinite(h) or h<=l: l,h=float(np.nanmin(a)),float(np.nanmax(a)+1e-8)
    return np.clip((a-l)/(h-l+1e-8),0,1)

def _rgb(norm,cmap='inferno'):
    norm=np.clip(norm,0,1)
    try:
        import matplotlib
        return (matplotlib.colormaps.get_cmap(cmap)(norm)[...,:3]*255).astype(np.uint8)
    except Exception: return np.repeat((norm[...,None]*255).astype(np.uint8),3,-1)

def _cond_only(x,opt):
    labs=list(opt.get('cond_or_uncond',[]) or [])
    if not labs or x.shape[0]%len(labs): return x
    per=x.shape[0]//len(labs); z=x.reshape(len(labs),per,*x.shape[1:]); keep=[i for i,v in enumerate(labs) if int(v)==0]
    return z[keep].reshape(-1,*x.shape[1:]) if keep else x

def _probs(q,k,tokens,conditional,opt,chunk=64):
    if conditional: q,k=_cond_only(q,opt),_cond_only(k,opt)
    qf=q.detach().reshape(q.shape[0],-1,q.shape[-2],q.shape[-1]).float()
    kf=k.detach().reshape(k.shape[0],-1,k.shape[-2],k.shape[-1]).float()
    valid=[i for i in sorted(tokens) if 0<=i<kf.shape[1]]
    if not valid: return {}
    scale=1/math.sqrt(qf.shape[-1]); rm=torch.full(qf.shape[:3],-float('inf'),device=qf.device); rs=torch.zeros_like(rm)
    for s in range(0,kf.shape[1],chunk):
        L=torch.einsum('bqhd,bkhd->bqhk',qf,kf[:,s:s+chunk])*scale; cm=L.amax(-1); nm=torch.maximum(rm,cm)
        rs=rs*torch.exp(rm-nm)+torch.exp(L-nm[...,None]).sum(-1); rm=nm
    den=rm+torch.log(rs.clamp_min(1e-30)); L=torch.einsum('bqhd,bkhd->bqhk',qf,kf[:,valid])*scale
    p=torch.exp(L-den[...,None]).mean(0).mean(1)  # EXACT V1: batch mean then head mean
    return {tok:p[:,j].cpu().numpy() for j,tok in enumerate(valid)}

def _append(st,r):
    with st['lock']:
        with open(st['records'],'a',encoding='utf-8') as f: f.write(json.dumps(r,ensure_ascii=False)+'\n')

def _record(q,k,opt):
    sid=opt.get('anima_diag_v2_session'); st=_REC.get(sid)
    if st is None: return
    ci=int(opt.get('anima_diag_v2_call_index',-1)); bi=int(opt.get('block_index',-1))
    if ci<0 or bi not in st['blocks'] or ci%st['stride']: return
    ids=sorted({i for xs in st['word_ids'].values() for i in xs}); conditional=st['batch_mode']=='conditional_only'
    ps=_probs(q,k,ids,conditional,opt)
    token_maps={}
    for tok,v in ps.items():
        a=_spatial(v,q,st['shapes'].get(ci))
        if a is None: continue
        token_maps[tok]=a
        raw=Path(st['token_raw'])/f'call{ci:04d}_block{bi:02d}_token{tok:03d}.npy'; np.save(raw,a)
        png=Path(st['token_png'])/f'call{ci:04d}_block{bi:02d}_token{tok:03d}.png'; Image.fromarray(_rgb(_norm(a),st['cmap']),'RGB').save(png)
        _append(st,{'kind':'token_map','call_index':ci,'block':bi,'text_token_index':tok,'raw_npy':str(raw),'png':str(png),'sigma':opt.get('anima_diag_v2_sigma')})
    for word,idxs in st['word_ids'].items():
        arr=[token_maps[i] for i in idxs if i in token_maps]
        if not arr: continue
        wmap=np.mean(np.stack(arr),0)
        raw=Path(st['word_raw'])/f'call{ci:04d}_block{bi:02d}_word-{_slug(word)}.npy'; np.save(raw,wmap)
        png=Path(st['word_png'])/f'call{ci:04d}_block{bi:02d}_word-{_slug(word)}.png'; Image.fromarray(_rgb(_norm(wmap),st['cmap']),'RGB').save(png)
        _append(st,{'kind':'word_map','call_index':ci,'block':bi,'attention_word':word,'token_indices':idxs,'raw_npy':str(raw),'png':str(png),'sigma':opt.get('anima_diag_v2_sigma')})

def _hook():
    try: from comfy.ldm.cosmos.predict2 import Attention as A
    except Exception as e: print('[AnimaDiagnosticsV2] hook unavailable',repr(e)); return
    if getattr(A,'_anima_diag_v2_hook_installed',False): return
    orig=A.compute_qkv
    def wrapped(self,x,context=None,rope_emb=None,transformer_options={}):
        q,k,v=orig(self,x,context=context,rope_emb=rope_emb,transformer_options=transformer_options)
        if not self.is_selfattn:
            try: _record(q,k,transformer_options or {})
            except Exception as e:
                sid=(transformer_options or {}).get('anima_diag_v2_session')
                if sid in _REC: _append(_REC[sid],{'kind':'error','error':repr(e)})
        return q,k,v
    A.compute_qkv=wrapped; A._anima_diag_v2_hook_installed=True; A._anima_diag_v2_original_compute_qkv=orig
_hook()

class AnimaTextEncodeWithTokenMapV2:
    @classmethod
    def INPUT_TYPES(c): return {'required':{'clip':('CLIP',),'text':('STRING',{'multiline':True,'dynamicPrompts':True})}}
    RETURN_TYPES=('CONDITIONING','ANIMA_TOKEN_MAP','STRING'); RETURN_NAMES=('conditioning','token_map','mapping_text'); FUNCTION='encode'; CATEGORY='diagnostics/anima'
    def encode(self,clip,text):
        t=clip.tokenize(text)
        if not isinstance(t,dict) or 't5xxl' not in t: raise RuntimeError('Anima t5xxl stream required')
        o=clip.encode_from_tokens(t,return_pooled=True,return_dict=True); cond=o.pop('cond'); pairs=_flat(t,'t5xxl'); dec=_detok(clip,pairs)
        tm={'prompt':text,'token_ids':[_tid(x) for x in pairs],'token_texts':[_txt(x) for x in dec],'_clip':clip}
        return ([[cond,o]],tm,_mapping(tm))

class AnimaTokenMapViewerV2:
    @classmethod
    def INPUT_TYPES(c): return {'required':{'token_map':('ANIMA_TOKEN_MAP',{'forceInput':True})}}
    RETURN_TYPES=('STRING',); RETURN_NAMES=('mapping_text',); FUNCTION='show'; OUTPUT_NODE=True; CATEGORY='diagnostics/anima'
    def show(self,token_map):
        s=_mapping(token_map); return {'ui':{'text':[s]},'result':(s,)}

class AnimaAttentionDiagnosticsV2:
    @classmethod
    def INPUT_TYPES(c):
        return {'required':{
            'model':('MODEL',),'token_map':('ANIMA_TOKEN_MAP',{'forceInput':True}),'attention_words':('STRING',{'multiline':True,'default':'arona'}),
            'selected_blocks':('STRING',{'default':'0,6,12,18,24,27'}),'batch_mode':(['v1_parity','conditional_only'],{'default':'v1_parity'}),
            'snapshot_every_n_calls':('INT',{'default':1,'min':1,'max':100}),'colormap':(['inferno','viridis','magma','plasma','gray'],{'default':'inferno'}),
            'output_root':('STRING',{'default':'/content/anima_diagnostics_v2'})}}
    RETURN_TYPES=('MODEL','STRING','STRING'); RETURN_NAMES=('model','diagnostic_directory','selected_word_mapping'); FUNCTION='patch'; CATEGORY='diagnostics/anima'
    def patch(self,model,token_map,attention_words,selected_blocks,batch_mode,snapshot_every_n_calls,colormap,output_root):
        ws=_words(attention_words); wi={w:_word_ids(token_map,w) for w in ws}; miss=[w for w,x in wi.items() if not x]
        if miss: raise ValueError('Could not map: '+', '.join(miss))
        sid=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]; out=Path(output_root)/sid
        for d in ('token_raw','token_maps','word_raw','word_maps'): (out/d).mkdir(parents=True,exist_ok=True)
        st={'blocks':_ints(selected_blocks),'batch_mode':batch_mode,'stride':int(snapshot_every_n_calls),'cmap':colormap,'word_ids':wi,
            'token_raw':str(out/'token_raw'),'token_png':str(out/'token_maps'),'word_raw':str(out/'word_raw'),'word_png':str(out/'word_maps'),
            'records':str(out/'records.jsonl'),'call':0,'shapes':{},'lock':threading.Lock()}; _REC[sid]=st
        public={k:v for k,v in token_map.items() if k!='_clip'}; (out/'token_map.json').write_text(json.dumps(public,ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'session.json').write_text(json.dumps({'session':sid,'attention_words':ws,'word_indices':wi,'selected_blocks':sorted(st['blocks']),
            'batch_mode':batch_mode,'baseline':'v1_parity reproduces V1 batch+head mean; maps remain separated by token/block/call'},ensure_ascii=False,indent=2),encoding='utf-8')
        m=model.clone(); old=m.model_options.get('model_function_wrapper')
        def wrap(apply_model,args):
            c=args['c'].copy()
            with st['lock']: ci=st['call']; st['call']+=1
            to=c.get('transformer_options',{}).copy(); to.update({'anima_diag_v2_session':sid,'anima_diag_v2_call_index':ci,'anima_diag_v2_sigma':float(args['timestep'].max().detach().cpu())})
            c['transformer_options']=to; st['shapes'][ci]=list(args['input'].shape)
            return old(apply_model,args|{'c':c}) if old is not None else apply_model(args['input'],args['timestep'],**c)
        m.set_model_unet_function_wrapper(wrap); mapping='\n'.join(f'{w!r} -> {x}' for w,x in wi.items()); return (m,str(out),mapping)

def _rows(session):
    p=Path(session)/'records.jsonl'; out=[]
    if p.exists():
        for s in p.read_text(encoding='utf-8').splitlines():
            try: out.append(json.loads(s))
            except: pass
    return out

def _arrays(session,kind,word=None,token=None,block=None,call=None):
    out=[]
    for r in _rows(session):
        if r.get('kind')!=kind: continue
        if word is not None and r.get('attention_word')!=word: continue
        if token is not None and int(r.get('text_token_index',-1))!=int(token): continue
        if block is not None and int(r.get('block',-1))!=int(block): continue
        if call is not None and int(r.get('call_index',-1))!=int(call): continue
        p=Path(r['raw_npy'])
        if p.exists(): out.append(np.load(p).astype(np.float32))
    return out

def _resize(rgb,h,w):
    t=torch.from_numpy(np.asarray(rgb,np.float32)/255).permute(2,0,1)[None]; t=F.interpolate(t,size=(h,w),mode='bicubic',align_corners=False)
    return t[0].permute(1,2,0).clamp(0,1)
def _cap(t,s):
    a=(t.numpy()*255).astype(np.uint8); im=Image.fromarray(a); d=ImageDraw.Draw(im); d.rectangle((4,im.height-22,min(im.width-4,10+7*len(s)),im.height-4),fill='black'); d.text((7,im.height-19),s,fill='white'); return torch.from_numpy(np.asarray(im).astype(np.float32)/255)

class AnimaAttentionOverlayV2:
    @classmethod
    def INPUT_TYPES(c):
        return {'required':{'images':('IMAGE',),'diagnostic_directory':('STRING',{'forceInput':True}),'attention_words':('STRING',{'multiline':True,'default':'arona'}),
            'view_mode':(['subtokens_global','single_token_map','word_global'],{'default':'subtokens_global'}),'block':('INT',{'default':18,'min':0,'max':100}),
            'call_index':('INT',{'default':0,'min':0,'max':10000}),'alpha':('FLOAT',{'default':0.5,'min':0,'max':1,'step':0.05}),
            'aggregation':(['mean','median'],{'default':'mean'}),'caption':('BOOLEAN',{'default':True})}}
    RETURN_TYPES=('IMAGE','IMAGE','STRING'); RETURN_NAMES=('overlay_images','heatmap_images','debug_info'); FUNCTION='overlay'; CATEGORY='diagnostics/anima'
    def overlay(self,images,diagnostic_directory,attention_words,view_mode,block,call_index,alpha,aggregation,caption):
        session=Path(diagnostic_directory); cfg=json.loads((session/'session.json').read_text()); wi=cfg.get('word_indices',{}); overs=[]; heats=[]; info=[]
        for b in range(images.shape[0]):
            base=images[b].detach().cpu().float().clamp(0,1); h,w=base.shape[:2]
            for word in _words(attention_words):
                series=[]
                if view_mode=='word_global':
                    aa=_arrays(session,'word_map',word=word)
                    if aa: series=[(word,np.mean(np.stack(aa),0) if aggregation=='mean' else np.median(np.stack(aa),0))]
                else:
                    for tok in wi.get(word,[]):
                        aa=_arrays(session,'token_map',token=tok,block=(block if view_mode=='single_token_map' else None),call=(call_index if view_mode=='single_token_map' else None))
                        if aa: series.append((f'{word} token {tok}',np.mean(np.stack(aa),0) if aggregation=='mean' else np.median(np.stack(aa),0)))
                if not series: raise RuntimeError(f'No maps for {word} / {view_mode}. Check block/call for single_token_map.')
                for label,a in series:
                    heat=_resize(_rgb(_norm(a),'inferno'),int(h),int(w)); over=((1-alpha)*base+alpha*heat).clamp(0,1)
                    if caption: over=_cap(over,label)
                    overs.append(over); heats.append(heat); info.append(label)
        return (torch.stack(overs),torch.stack(heats),'\n'.join(info))

NODE_CLASS_MAPPINGS={'AnimaTextEncodeWithTokenMapV2':AnimaTextEncodeWithTokenMapV2,'AnimaTokenMapViewerV2':AnimaTokenMapViewerV2,'AnimaAttentionDiagnosticsV2':AnimaAttentionDiagnosticsV2,'AnimaAttentionOverlayV2':AnimaAttentionOverlayV2}
NODE_DISPLAY_NAME_MAPPINGS={'AnimaTextEncodeWithTokenMapV2':'Anima Text Encode + Token Map V2','AnimaTokenMapViewerV2':'Anima Token Map Viewer V2','AnimaAttentionDiagnosticsV2':'Anima Attention Diagnostics V2','AnimaAttentionOverlayV2':'Anima Attention Overlay V2'}