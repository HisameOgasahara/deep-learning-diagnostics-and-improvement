from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path('/content/anima_diagnostics_v2')
SESSIONS = sorted([p for p in ROOT.glob('*') if p.is_dir()], key=lambda p: p.stat().st_mtime)
if not SESSIONS:
    raise RuntimeError('No v2 diagnostic session found.')

session = SESSIONS[-1]
rows = []
records_path = session / 'records.jsonl'
if records_path.exists():
    for line in records_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

df = pd.DataFrame(rows)
if len(df):
    df.to_csv(session / 'records.csv', index=False)

cfg = json.loads((session / 'session.json').read_text(encoding='utf-8')) if (session / 'session.json').exists() else {}
ratio_vmax = float(cfg.get('ratio_vmax', 6.0))
colormap = cfg.get('colormap', 'inferno')


def colorize_fixed(arr, vmax):
    arr = np.asarray(arr, np.float32)
    norm = np.clip(arr / max(vmax, 1e-8), 0.0, 1.0)
    try:
        import matplotlib
        rgb = matplotlib.colormaps.get_cmap(colormap)(norm)[..., :3]
        return (rgb * 255).astype(np.uint8)
    except Exception:
        return np.repeat((norm[..., None] * 255).astype(np.uint8), 3, -1)


def slug(text):
    import re
    s = re.sub(r'[^0-9A-Za-z가-힣_-]+', '_', str(text)).strip('_')
    return s[:64] or 'word'


word_indices = cfg.get('word_indices', {})
raw_files = sorted((session / 'raw').glob('*.npz'))
by_word = {}
meta = []

for word in cfg.get('attention_words', []):
    s = slug(word)
    matching = [p for p in raw_files if f'_word-{s}' in p.stem]
    for p in matching:
        z = np.load(p)
        agg = z['aggregate_ratio'].astype(np.float32)
        parts = p.stem.split('_')
        call = int(parts[0].replace('call', ''))
        block = int(parts[1].replace('block', ''))
        by_word.setdefault(word, []).append((call, block, agg))
        meta.append({
            'attention_word': word,
            'token_indices': word_indices.get(word, []),
            'call_index': call,
            'block': block,
            'raw_npz': str(p),
        })

if meta:
    pd.DataFrame(meta).sort_values(['attention_word', 'call_index', 'block']).to_csv(
        session / 'aggregate_index.csv', index=False
    )

for word, items in by_word.items():
    shapes = {x[2].shape for x in items}
    if len(shapes) != 1:
        print('skip', word, 'because map shapes differ:', shapes)
        continue
    stack = np.stack([x[2] for x in items], axis=0)
    mean_map = stack.mean(axis=0)
    median_map = np.median(stack, axis=0)
    s = slug(word)
    np.save(session / f'word-{s}_mean_ratio.npy', mean_map)
    np.save(session / f'word-{s}_median_ratio.npy', median_map)
    Image.fromarray(colorize_fixed(mean_map, ratio_vmax), 'RGB').save(session / f'word-{s}_mean_ratio.png')
    Image.fromarray(colorize_fixed(median_map, ratio_vmax), 'RGB').save(session / f'word-{s}_median_ratio.png')

print('session:', session)
print('records:', len(df))
print('raw maps:', len(raw_files))
print('attention words:', list(by_word))
print('word -> token indices:', word_indices)
print('scale: attention_ratio = token_probability * text_key_count; 1.0 = uniform attention')
print('vmax:', ratio_vmax)
