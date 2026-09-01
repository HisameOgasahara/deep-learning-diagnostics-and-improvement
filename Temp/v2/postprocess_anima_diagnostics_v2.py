from pathlib import Path
import json, re
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path('/content/anima_diagnostics_v2')
sessions=sorted([p for p in ROOT.glob('*') if p.is_dir()],key=lambda p:p.stat().st_mtime)
if not sessions: raise RuntimeError('No v2 diagnostic session found.')
session=sessions[-1]
rows=[]
p=session/'records.jsonl'
if p.exists():
    for line in p.read_text(encoding='utf-8').splitlines():
        try: rows.append(json.loads(line))
        except: pass
df=pd.DataFrame(rows)
if len(df): df.to_csv(session/'records.csv',index=False)

def make_gif(files,out,duration=250):
    ims=[Image.open(x).convert('RGB') for x in files if Path(x).exists()]
    if ims: ims[0].save(out,save_all=True,append_images=ims[1:],duration=duration,loop=0)

def norm(a,lo=1,hi=99):
    a=np.asarray(a,np.float32); l,h=np.nanpercentile(a,[lo,hi]); return np.clip((a-l)/(h-l+1e-8),0,1)
def rgb(a):
    try:
        import matplotlib
        return (matplotlib.colormaps.get_cmap('inferno')(norm(a))[...,:3]*255).astype(np.uint8)
    except: return np.repeat((norm(a)[...,None]*255).astype(np.uint8),3,-1)
def slug(s): return re.sub(r'[^0-9A-Za-z가-힣_-]+','_',str(s)).strip('_')[:64] or 'word'

# V1-parity token maps: one GIF per token x block over denoising calls.
tok=df[df['kind']=='token_map'].copy() if len(df) and 'kind' in df.columns else pd.DataFrame()
if len(tok):
    gifdir=session/'token_gifs'; gifdir.mkdir(parents=True,exist_ok=True)
    for (token,block),g in tok.groupby(['text_token_index','block']):
        g=g.sort_values('call_index')
        make_gif(g['png'].tolist(),gifdir/f'block{int(block):02d}_token{int(token):03d}.gif')

# Global token summaries are separate so subtoken behavior is never hidden by word averaging.
if len(tok):
    outdir=session/'token_global'; outdir.mkdir(parents=True,exist_ok=True)
    for token,g in tok.groupby('text_token_index'):
        arr=[np.load(x).astype(np.float32) for x in g['raw_npy'] if Path(x).exists()]
        if arr:
            m=np.mean(np.stack(arr),0); np.save(outdir/f'token{int(token):03d}_mean.npy',m)
            Image.fromarray(rgb(m),'RGB').save(outdir/f'token{int(token):03d}_mean_relative.png')

word=df[df['kind']=='word_map'].copy() if len(df) and 'kind' in df.columns else pd.DataFrame()
if len(word):
    gifdir=session/'word_gifs'; gifdir.mkdir(parents=True,exist_ok=True)
    for (name,block),g in word.groupby(['attention_word','block']):
        g=g.sort_values('call_index'); make_gif(g['png'].tolist(),gifdir/f'block{int(block):02d}_word-{slug(name)}.gif')
    outdir=session/'word_global'; outdir.mkdir(parents=True,exist_ok=True)
    for name,g in word.groupby('attention_word'):
        arr=[np.load(x).astype(np.float32) for x in g['raw_npy'] if Path(x).exists()]
        if arr:
            m=np.mean(np.stack(arr),0); np.save(outdir/f'word-{slug(name)}_mean.npy',m)
            Image.fromarray(rgb(m),'RGB').save(outdir/f'word-{slug(name)}_mean_relative.png')

print('session:',session)
print('token maps:',len(tok),'word maps:',len(word))
print('First inspect token_gifs/: token x block x timestep, then compare token_global/ and word_global/.')
