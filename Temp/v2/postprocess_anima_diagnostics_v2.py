from pathlib import Path
import json, re
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path('/content/anima_diagnostics_v2')
SESSIONS = sorted([p for p in ROOT.glob('*') if p.is_dir()], key=lambda p: p.stat().st_mtime)
if not SESSIONS:
    raise RuntimeError('No v2 diagnostic session found.')
session = SESSIONS[-1]
cfg = json.loads((session / 'session.json').read_text(encoding='utf-8'))
ratio_vmax = float(cfg.get('ratio_vmax', 6.0)); cmap = cfg.get('colormap', 'inferno')


def slug(text):
    s = re.sub(r'[^0-9A-Za-z가-힣_-]+', '_', str(text)).strip('_')
    return s[:64] or 'word'


def norm_final(x, lo=1, hi=99):
    a = np.asarray(x, np.float32); l, h = np.nanpercentile(a, [lo, hi])
    if not np.isfinite(l) or not np.isfinite(h) or h <= l:
        l, h = float(np.nanmin(a)), float(np.nanmax(a))
    return np.zeros_like(a) if h <= l else np.clip((a-l)/(h-l+1e-8), 0, 1)


def color_unit(x):
    try:
        import matplotlib
        return (matplotlib.colormaps.get_cmap(cmap)(np.clip(x,0,1))[..., :3] * 255).astype(np.uint8)
    except Exception:
        return np.repeat((np.clip(x,0,1)[...,None]*255).astype(np.uint8), 3, -1)


def color_abs(x):
    return color_unit(np.asarray(x, np.float32) / max(ratio_vmax, 1e-8))

rows=[]
for line in (session/'records.jsonl').read_text(encoding='utf-8').splitlines():
    try: rows.append(json.loads(line))
    except Exception: pass
if rows: pd.DataFrame(rows).to_csv(session/'records.csv', index=False)

for word in cfg.get('attention_words', []):
    maps=[]; sub_by_index={}
    for rec in rows:
        if rec.get('kind')!='cross_attention_word_v2' or rec.get('attention_word')!=word: continue
        p=Path(rec.get('raw_npz',''))
        if not p.exists(): p=session/'raw'/p.name
        if not p.exists(): continue
        with np.load(p) as z:
            maps.append(z['aggregate_ratio'].astype(np.float32))
            if 'subtoken_attention_ratio' in z:
                sub=z['subtoken_attention_ratio'].astype(np.float32).mean(axis=1)  # subtoken, H, W
                ids=z['token_indices'].astype(int).tolist()
                for i,tok in enumerate(ids): sub_by_index.setdefault(tok,[]).append(sub[i])
    if not maps: continue
    stack=np.stack(maps,0); mean_map=stack.mean(0); median_map=np.median(stack,0); s=slug(word)
    for name,raw in [('mean',mean_map),('median',median_map)]:
        np.save(session/f'word-{s}_{name}_raw_ratio.npy', raw)
        Image.fromarray(color_abs(raw),'RGB').save(session/f'word-{s}_{name}_absolute.png')
        rel=norm_final(raw); np.save(session/f'word-{s}_{name}_relative.npy',rel)
        Image.fromarray(color_unit(rel),'RGB').save(session/f'word-{s}_{name}_relative.png')
    for tok, vals in sorted(sub_by_index.items()):
        raw=np.stack(vals,0).mean(0); np.save(session/f'word-{s}_token{tok:03d}_raw_ratio.npy',raw)
        Image.fromarray(color_unit(norm_final(raw)),'RGB').save(session/f'word-{s}_token{tok:03d}_relative.png')

print('session:', session)
print('Final relative maps normalize ONCE after raw head/block/call/subtoken aggregation.')
print('Absolute maps are retained for quantitative comparison.')
