import json, math, time, uuid, threading, re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

_RECORDER = {}


def _parse_ints(text):
    t = str(text).strip().lower()
    if t in ('', 'all', '*'):
        return None
    out = set()
    for p in t.replace(' ', '').split(','):
        if not p:
            continue
        if '-' in p:
            a, b = p.split('-', 1); out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(p))
    return out


def _parse_words(text):
    return [w.strip() for w in str(text).split(',') if w.strip()]


def _slug(text):
    s = re.sub(r'[^0-9A-Za-z가-힣_-]+', '_', str(text)).strip('_')
    return s[:64] or 'word'


def _flatten(tokens, key):
    return [x for group in (tokens.get(key, []) if isinstance(tokens, dict) else []) for x in group]


def _token_id(x):
    return int(x[0]) if isinstance(x, (tuple, list)) else int(x)


def _t5_detok(clip, pairs):
    try:
        return clip.tokenizer.t5xxl.untokenize(pairs)
    except Exception:
        try:
            return clip.tokenizer.untokenize(pairs)
        except Exception:
            return [(p[0], str(p[0])) for p in pairs]


def _decoded(x):
    return str(x[1]) if isinstance(x, (tuple, list)) and len(x) > 1 else str(x)


def _clean(x):
    return str(x).replace('▁', '').replace('Ġ', '')


def _find_word_indices(token_map, word):
    clip = token_map.get('_clip')
    if clip is None:
        return []
    pieces = token_map.get('token_texts', [])
    q = _t5_detok(clip, _flatten(clip.tokenize(word), 't5xxl'))
    q = [_clean(_decoded(x)) for x in q]
    target = ''.join(x for x in q if x not in ('', '<pad>', '</s>')).lower()
    clean = [_clean(x) for x in pieces]
    hits = []
    for i in range(len(clean)):
        if clean[i] in ('<pad>', '</s>'):
            continue
        cur = ''
        for j in range(i, len(clean)):
            if clean[j] in ('<pad>', '</s>'):
                break
            cur += clean[j]
            low = cur.lower()
            if low == target:
                hits.extend(range(i, j + 1)); break
            if len(low) > len(target) or not target.startswith(low):
                break
    return sorted(set(hits))


def _mapping_text(token_map):
    lines = [f"prompt: {token_map.get('prompt', '')}", 'T5 target tokens -> Anima cross-attention key indices:']
    for i, (tid, txt) in enumerate(zip(token_map.get('token_ids', []), token_map.get('token_texts', []))):
        lines.append(f'[{i:03d}] id={tid:<6d} token={txt!r}')
    n = len(token_map.get('token_ids', [])); k = int(token_map.get('cross_attention_key_count', 512))
    if k > n:
        lines.append(f'[{n:03d}..{k-1:03d}] model-side zero padding added by Anima preprocess_text_embeds')
    return '\n'.join(lines)


def _infer_grid(n, input_shape=None):
    ratio = 1.0
    if input_shape and len(input_shape) >= 2 and input_shape[-1] > 0:
        ratio = float(input_shape[-2]) / float(input_shape[-1])
    best, score = None, float('inf')
    for h in range(1, int(n ** 0.5) + 1):
        if n % h:
            continue
        for hh, ww in ((h, n // h), (n // h, h)):
            s = abs(math.log((hh / ww + 1e-12) / (ratio + 1e-12)))
            if s < score:
                score, best = s, (hh, ww)
    return best


def _relative_norm(arr, low=1.0, high=99.0):
    x = np.asarray(arr, np.float32)
    lo, hi = np.nanpercentile(x, [float(low), float(high)])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)


def _color_unit(x, cmap='inferno'):
    x = np.clip(np.asarray(x, np.float32), 0, 1)
    if cmap == 'gray':
        return np.repeat((x[..., None] * 255).astype(np.uint8), 3, -1)
    try:
        import matplotlib
        return (matplotlib.colormaps.get_cmap(cmap)(x)[..., :3] * 255).astype(np.uint8)
    except Exception:
        return np.repeat((x[..., None] * 255).astype(np.uint8), 3, -1)


def _color_abs(x, vmax, cmap):
    return _color_unit(np.asarray(x, np.float32) / max(float(vmax), 1e-8), cmap)


def _save_abs(x, path, vmax, cmap):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix('.npy'), np.asarray(x, np.float32))
    Image.fromarray(_color_abs(x, vmax, cmap), 'RGB').save(path)


def _append(st, rec):
    with st['lock']:
        with open(st['records_path'], 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def _conditional_only(x, opt):
    labels = list(opt.get('cond_or_uncond', []) or [])
    if not labels or x.shape[0] % len(labels) != 0:
        return x
    per = x.shape[0] // len(labels)
    chunks = x.reshape(len(labels), per, *x.shape[1:])
    keep = [i for i, v in enumerate(labels) if int(v) == 0]
    return chunks[keep].reshape(-1, *x.shape[1:]) if keep else x


def _selected_probs(q, k, tokens, opt, chunk=64):
    q = _conditional_only(q, opt); k = _conditional_only(k, opt)
    qf = q.detach().reshape(q.shape[0], -1, q.shape[-2], q.shape[-1]).float()
    kf = k.detach().reshape(k.shape[0], -1, k.shape[-2], k.shape[-1]).float()
    valid = [i for i in sorted(tokens) if 0 <= i < kf.shape[1]]
    if not valid:
        return {}, int(kf.shape[1])
    scale = 1 / math.sqrt(qf.shape[-1])
    rm = torch.full(qf.shape[:3], -float('inf'), device=qf.device); rs = torch.zeros_like(rm)
    for s in range(0, kf.shape[1], chunk):
        z = torch.einsum('bqhd,bkhd->bqhk', qf, kf[:, s:s+chunk]) * scale
        cm = z.amax(-1); nm = torch.maximum(rm, cm)
        rs = rs * torch.exp(rm - nm) + torch.exp(z - nm[..., None]).sum(-1); rm = nm
    denom = rm + torch.log(rs.clamp_min(1e-30))
    z = torch.einsum('bqhd,bkhd->bqhk', qf, kf[:, valid]) * scale
    p = torch.exp(z - denom[..., None]).mean(0).permute(2, 1, 0).contiguous()
    return {tok: p[j].cpu().numpy() for j, tok in enumerate(valid)}, int(kf.shape[1])


def _reshape_heads(v, q, input_shape):
    spatial = list(q.shape[1:-2]); out = []
    for h in range(v.shape[0]):
        a = np.asarray(v[h], np.float32)
        if len(spatial) >= 2:
            a = a.reshape(spatial)
            while a.ndim > 2:
                a = a.mean(0)
        else:
            g = _infer_grid(a.size, input_shape)
            if not g:
                return None
            a = a.reshape(*g)
        out.append(a)
    return np.stack(out, 0)


def _concentration(x):
    z = np.clip(np.asarray(x, np.float64).reshape(x.shape[0], -1), 0, None)
    p = z / np.maximum(z.sum(1, keepdims=True), 1e-30)
    e = -(p * np.log(np.maximum(p, 1e-30))).sum(1)
    return 1 - e / math.log(max(z.shape[1], 2))


def _pick_heads(st, maps):
    n = maps.shape[0]
    if st['selected_heads'] is not None:
        return [h for h in sorted(st['selected_heads']) if 0 <= h < n], None
    if st['head_selection_mode'] == 'concentration_topk':
        s = _concentration(maps); k = max(1, min(st['top_k_heads'], n))
        return np.argsort(s)[-k:][::-1].tolist(), s.tolist()
    return list(range(n)), None


def _record_cross(q, k, opt):
    sid = opt.get('anima_diag_v2_session')
    if not sid or sid not in _RECORDER:
        return
    st = _RECORDER[sid]; ci = int(opt.get('anima_diag_v2_call_index', -1)); bi = int(opt.get('block_index', -1))
    if ci < 0 or bi not in st['selected_blocks'] or ci % st['snapshot_every_n_calls']:
        return
    all_idx = sorted({i for ids in st['word_indices'].values() for i in ids})
    probs, nkeys = _selected_probs(q, k, all_idx, opt)
    for word, ids in st['word_indices'].items():
        subtokens, used = [], []
        for tok in ids:
            if tok not in probs:
                continue
            hm = _reshape_heads(probs[tok], q, st['input_shapes'].get(ci))
            if hm is not None:
                subtokens.append(hm * float(nkeys)); used.append(tok)
        if not subtokens:
            continue
        sub = np.stack(subtokens, 0)                 # [subtoken, head, H, W]
        ratio_heads = sub.mean(0)                   # raw subtoken aggregation, no normalization
        chosen, scores = _pick_heads(st, ratio_heads)
        if not chosen:
            continue
        aggregate = ratio_heads[chosen].mean(0)     # raw head aggregation, no normalization
        slug = _slug(word)
        raw = Path(st['raw_dir']) / f'call{ci:04d}_block{bi:02d}_word-{slug}.npz'
        np.savez_compressed(raw,
            aggregate_ratio=aggregate.astype(np.float32),
            attention_ratio_heads=ratio_heads.astype(np.float32),
            subtoken_attention_ratio=sub.astype(np.float32),
            token_indices=np.asarray(used, np.int32),
            selected_heads=np.asarray(chosen, np.int32),
            text_key_count=np.asarray([nkeys], np.int32))
        preview = Path(st['attention_maps_dir']) / f'absolute_call{ci:04d}_block{bi:02d}_word-{slug}.png'
        _save_abs(aggregate, preview, st['ratio_vmax'], st['colormap'])
        if st['save_head_pngs']:
            for h in chosen:
                _save_abs(ratio_heads[h], Path(st['head_maps_dir']) / f'absolute_call{ci:04d}_block{bi:02d}_word-{slug}_head{h:02d}.png', st['ratio_vmax'], st['colormap'])
        _append(st, {'kind':'cross_attention_word_v2','call_index':ci,'sigma':opt.get('anima_diag_v2_sigma'),'block':bi,
            'attention_word':word,'token_indices':used,'text_key_count':nkeys,'selected_heads':chosen,
            'head_selection_mode':st['head_selection_mode'],'concentration_scores':scores,'raw_npz':str(raw),
            'note':'raw attention only; final relative normalization happens after call/block aggregation'})


def _load_final(session_dir, word, aggregation='mean'):
    session = Path(session_dir); arrays = []
    for line in (session / 'records.jsonl').read_text(encoding='utf-8').splitlines():
        try: rec = json.loads(line)
        except Exception: continue
        if rec.get('kind') != 'cross_attention_word_v2' or rec.get('attention_word') != word:
            continue
        p = Path(rec.get('raw_npz', ''))
        if not p.exists(): p = session / 'raw' / p.name
        if p.exists():
            with np.load(p) as z: arrays.append(z['aggregate_ratio'].astype(np.float32))
    if not arrays:
        raise RuntimeError(f'No attention maps for {word!r}')
    stack = np.stack(arrays, 0)
    return np.median(stack, 0) if aggregation == 'median' else stack.mean(0)


def _resize_rgb(rgb, h, w):
    t = torch.from_numpy(np.asarray(rgb, np.float32) / 255).permute(2,0,1).unsqueeze(0)
    return F.interpolate(t, size=(h,w), mode='bicubic', align_corners=False).squeeze(0).permute(1,2,0).clamp(0,1)


def _caption(t, text):
    a = (t.detach().cpu().clamp(0,1).numpy()*255).astype(np.uint8); im = Image.fromarray(a,'RGB'); d = ImageDraw.Draw(im)
    try: b = d.textbbox((0,0), text); tw, th = b[2]-b[0], b[3]-b[1]
    except Exception: tw, th = 8*len(text), 12
    x=max(4,(im.width-tw)//2); y=max(4,im.height-th-12); d.rectangle((x-6,y-6,x+tw+6,y+th+6),fill=(0,0,0)); d.text((x,y),text,fill=(255,255,255))
    return torch.from_numpy(np.asarray(im).astype(np.float32)/255)


def _install_hook():
    try:
        from comfy.ldm.cosmos.predict2 import Attention as A
    except Exception as e:
        print('[AnimaDiagnosticsV2] hook unavailable:', repr(e)); return
    if getattr(A, '_anima_diag_v2_hook_installed', False): return
    original = A.compute_qkv
    def wrapped(self, x, context=None, rope_emb=None, transformer_options={}):
        q,k,v = original(self,x,context=context,rope_emb=rope_emb,transformer_options=transformer_options)
        if not self.is_selfattn:
            try: _record_cross(q,k,transformer_options or {})
            except Exception as e:
                sid=(transformer_options or {}).get('anima_diag_v2_session')
                if sid in _RECORDER: _append(_RECORDER[sid], {'kind':'diagnostic_error','error':repr(e)})
        return q,k,v
    A.compute_qkv = wrapped; A._anima_diag_v2_hook_installed=True; A._anima_diag_v2_original_compute_qkv=original
    print('[AnimaDiagnosticsV2] installed projected Q/K hook')

_install_hook()


class AnimaTextEncodeWithTokenMapV2:
    @classmethod
    def INPUT_TYPES(cls): return {'required': {'clip':('CLIP',), 'text':('STRING',{'multiline':True,'dynamicPrompts':True})}}
    RETURN_TYPES=('CONDITIONING','ANIMA_TOKEN_MAP','STRING'); RETURN_NAMES=('conditioning','token_map','mapping_text'); FUNCTION='encode'; CATEGORY='diagnostics/anima'
    def encode(self, clip, text):
        tokens=clip.tokenize(text)
        if not isinstance(tokens,dict) or 't5xxl' not in tokens: raise RuntimeError('Anima t5xxl token stream required.')
        out=clip.encode_from_tokens(tokens,return_pooled=True,return_dict=True); cond=out.pop('cond'); pairs=_flatten(tokens,'t5xxl'); dec=_t5_detok(clip,pairs)
        tm={'prompt':text,'stream':'t5xxl','token_ids':[_token_id(x) for x in pairs],'token_texts':[_decoded(x) for x in dec],
            'token_count':len(pairs),'cross_attention_key_count':max(512,len(pairs)),'_clip':clip}
        return ([[cond,out]],tm,_mapping_text(tm))


class AnimaTokenMapViewerV2:
    @classmethod
    def INPUT_TYPES(cls): return {'required': {'token_map':('ANIMA_TOKEN_MAP',{'forceInput':True})}}
    RETURN_TYPES=('STRING',); RETURN_NAMES=('mapping_text',); FUNCTION='show'; OUTPUT_NODE=True; CATEGORY='diagnostics/anima'
    def show(self,token_map):
        t=_mapping_text(token_map); return {'ui':{'text':[t]},'result':(t,)}


class AnimaAttentionDiagnosticsV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model':('MODEL',),'token_map':('ANIMA_TOKEN_MAP',{'forceInput':True}),'attention_words':('STRING',{'multiline':True,'default':'girl'}),
            'selected_blocks':('STRING',{'default':'0,6,12,18,24,27'}),'selected_heads':('STRING',{'default':'all'}),
            'head_selection_mode':(['all','concentration_topk'],{'default':'all'}),'top_k_heads':('INT',{'default':4,'min':1,'max':64}),
            'snapshot_every_n_calls':('INT',{'default':1,'min':1,'max':100}),'ratio_vmax':('FLOAT',{'default':6.0,'min':1.0,'max':50.0,'step':0.5}),
            'save_head_pngs':('BOOLEAN',{'default':True}),'colormap':(['inferno','viridis','magma','plasma','gray'],{'default':'inferno'}),
            'output_root':('STRING',{'default':'/content/anima_diagnostics_v2'})}}
    RETURN_TYPES=('MODEL','STRING','STRING'); RETURN_NAMES=('model','diagnostic_directory','selected_word_mapping'); FUNCTION='patch'; CATEGORY='diagnostics/anima'
    def patch(self,model,token_map,attention_words,selected_blocks,selected_heads,head_selection_mode,top_k_heads,snapshot_every_n_calls,ratio_vmax,save_head_pngs,colormap,output_root):
        words=_parse_words(attention_words); wi={w:_find_word_indices(token_map,w) for w in words}; missing=[w for w,v in wi.items() if not v]
        if missing: raise ValueError('Could not map words: '+', '.join(missing))
        sid=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]; out=Path(output_root)/sid
        for n in ('attention_maps','head_maps','raw'): (out/n).mkdir(parents=True,exist_ok=True)
        blocks=_parse_ints(selected_blocks) or set(); st={'session':sid,'attention_maps_dir':str(out/'attention_maps'),'head_maps_dir':str(out/'head_maps'),'raw_dir':str(out/'raw'),
            'records_path':str(out/'records.jsonl'),'selected_blocks':blocks,'token_map':token_map,'word_indices':wi,'selected_heads':_parse_ints(selected_heads),
            'head_selection_mode':head_selection_mode,'top_k_heads':int(top_k_heads),'snapshot_every_n_calls':int(snapshot_every_n_calls),'ratio_vmax':float(ratio_vmax),
            'save_head_pngs':bool(save_head_pngs),'colormap':colormap,'call_index':0,'input_shapes':{},'lock':threading.Lock()}
        _RECORDER[sid]=st; public={k:v for k,v in token_map.items() if k!='_clip'}
        (out/'token_map.json').write_text(json.dumps(public,ensure_ascii=False,indent=2),encoding='utf-8'); (out/'token_map.txt').write_text(_mapping_text(token_map),encoding='utf-8')
        (out/'session.json').write_text(json.dumps({'session':sid,'attention_words':words,'word_indices':wi,'selected_blocks':sorted(blocks),'head_selection_mode':head_selection_mode,
            'ratio_vmax':float(ratio_vmax),'colormap':colormap,'raw_aggregation':'subtokens -> heads -> calls/blocks, no per-map normalization',
            'relative_visualization':'normalize once after final call/block aggregation'},ensure_ascii=False,indent=2),encoding='utf-8')
        m=model.clone(); old=m.model_options.get('model_function_wrapper')
        def wrapper(apply_model,args):
            c=args['c'].copy(); ci=st['call_index']; st['call_index']+=1; to=c.get('transformer_options',{}).copy(); to.update({'anima_diag_v2_session':sid,'anima_diag_v2_call_index':ci,'anima_diag_v2_sigma':float(args['timestep'].max().detach().cpu())}); c['transformer_options']=to; st['input_shapes'][ci]=list(args['input'].shape)
            return old(apply_model,args|{'c':c}) if old is not None else apply_model(args['input'],args['timestep'],**c)
        m.set_model_unet_function_wrapper(wrapper); return (m,str(out),'\n'.join(f'{w!r} -> {ids}' for w,ids in wi.items()))


class AnimaAttentionOverlayV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images':('IMAGE',),'diagnostic_directory':('STRING',{'forceInput':True}),'attention_words':('STRING',{'multiline':True,'default':'girl'}),
            'alpha':('FLOAT',{'default':0.5,'min':0.0,'max':1.0,'step':0.05}),'aggregation':(['mean','median'],{'default':'mean'}),
            'visualization':(['relative_final','absolute'],{'default':'relative_final'}),'relative_low_percentile':('FLOAT',{'default':1.0,'min':0.0,'max':49.0,'step':0.5}),
            'relative_high_percentile':('FLOAT',{'default':99.0,'min':51.0,'max':100.0,'step':0.5}),'caption':('BOOLEAN',{'default':True})}}
    RETURN_TYPES=('IMAGE','IMAGE'); RETURN_NAMES=('overlay_images','heatmap_images'); FUNCTION='overlay'; CATEGORY='diagnostics/anima'
    DESCRIPTION='Aggregates raw attention first, then normalizes only the final word map for DAAM-style visualization.'
    def overlay(self,images,diagnostic_directory,attention_words,alpha,aggregation,visualization,relative_low_percentile,relative_high_percentile,caption):
        session=Path(diagnostic_directory); cfg=json.loads((session/'session.json').read_text(encoding='utf-8')); cmap=cfg.get('colormap','inferno'); vmax=float(cfg.get('ratio_vmax',6.0)); ovs=[]; hms=[]
        for b in range(images.shape[0]):
            base=images[b].detach().cpu().float().clamp(0,1); h,w=int(base.shape[0]),int(base.shape[1])
            for word in _parse_words(attention_words):
                raw=_load_final(session,word,aggregation)
                rgb=_color_unit(_relative_norm(raw,relative_low_percentile,relative_high_percentile),cmap) if visualization=='relative_final' else _color_abs(raw,vmax,cmap)
                heat=_resize_rgb(rgb,h,w); ov=((1-float(alpha))*base+float(alpha)*heat).clamp(0,1); ovs.append(_caption(ov,word) if caption else ov); hms.append(heat)
        if not ovs: raise RuntimeError('No overlay images produced.')
        return (torch.stack(ovs),torch.stack(hms))


NODE_CLASS_MAPPINGS={'AnimaTextEncodeWithTokenMapV2':AnimaTextEncodeWithTokenMapV2,'AnimaTokenMapViewerV2':AnimaTokenMapViewerV2,'AnimaAttentionDiagnosticsV2':AnimaAttentionDiagnosticsV2,'AnimaAttentionOverlayV2':AnimaAttentionOverlayV2}
NODE_DISPLAY_NAME_MAPPINGS={'AnimaTextEncodeWithTokenMapV2':'Anima Text Encode + Token Map V2','AnimaTokenMapViewerV2':'Anima Token Map Viewer V2','AnimaAttentionDiagnosticsV2':'Anima Attention Diagnostics V2','AnimaAttentionOverlayV2':'Anima Attention Overlay V2'}
