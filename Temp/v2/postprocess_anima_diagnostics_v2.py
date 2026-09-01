from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path('/content/anima_diagnostics_v2')
sessions = sorted([p for p in ROOT.glob('*') if p.is_dir()], key=lambda p: p.stat().st_mtime)
if not sessions:
    raise RuntimeError('No v2 diagnostic session found.')
session = sessions[-1]

rows = []
records_path = session / 'records.jsonl'
if records_path.exists():
    for line in records_path.read_text(encoding='utf-8').splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

df = pd.DataFrame(rows)
if len(df):
    df.to_csv(session / 'records.csv', index=False)

cfg = json.loads((session / 'session.json').read_text(encoding='utf-8')) if (session / 'session.json').exists() else {}
cmap = cfg.get('colormap', 'turbo')


def minmax(a):
    a = np.asarray(a, np.float32)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)


def rgb(a):
    n = minmax(a)
    try:
        import matplotlib
        return (matplotlib.colormaps.get_cmap(cmap)(n)[..., :3] * 255).astype(np.uint8)
    except Exception:
        return np.repeat((n[..., None] * 255).astype(np.uint8), 3, -1)


def resize_np(a, h, w):
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.asarray(a, np.float32))[None, None]
    return F.interpolate(t, size=(h, w), mode='bicubic', align_corners=False)[0, 0].numpy()


tok = df[df['kind'] == 'timestep_token_map'].copy() if len(df) and 'kind' in df.columns else pd.DataFrame()
outdir = session / 'global_tokens'
outdir.mkdir(parents=True, exist_ok=True)
summary = []

if len(tok):
    for (batch, token), g in tok.groupby(['batch_index', 'text_token_index']):
        arrays = []
        for p in g.sort_values('call_index')['raw_npy']:
            p = Path(p)
            if not p.exists():
                p = session / 'timestep_raw' / p.name
            if p.exists():
                arrays.append(np.load(p).astype(np.float32))
        if not arrays:
            continue
        h = max(a.shape[0] for a in arrays)
        w = max(a.shape[1] for a in arrays)
        aligned = [resize_np(a, h, w) if a.shape != (h, w) else a for a in arrays]
        global_map = np.stack(aligned, axis=0).sum(axis=0)
        stem = f'batch{int(batch):02d}_token{int(token):03d}'
        np.save(outdir / f'{stem}_raw.npy', global_map)
        Image.fromarray(rgb(global_map), 'RGB').save(outdir / f'{stem}_relative.png')
        summary.append({
            'batch_index': int(batch),
            'text_token_index': int(token),
            'num_timestep_maps': len(arrays),
            'shape': list(global_map.shape),
        })

if summary:
    pd.DataFrame(summary).to_csv(session / 'global_token_index.csv', index=False)

print('session:', session)
print('timestep token maps:', len(tok))
print('global token maps:', len(summary))
print('aggregation: layer-mean per timestep/resolution was done during sampling; here timestep/resolution maps are summed, matching DAAM GlobalHeatMap semantics.')
print('word maps are intentionally produced later by Anima Attention Overlay V2 from arbitrary attention_words.')
