from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path('/content/anima_diagnostics')
sessions = sorted([p for p in ROOT.glob('*') if p.is_dir()], key=lambda p: p.stat().st_mtime)
if not sessions:
    raise RuntimeError('No diagnostic session found. Generate once in ComfyUI with Anima Inference Diagnostics connected.')

session = sessions[-1]
records_path = session / 'records.jsonl'
if not records_path.exists():
    raise RuntimeError(f'Missing {records_path}')

rows = []
with open(records_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

df = pd.DataFrame(rows)
csv_path = session / 'records.csv'
df.to_csv(csv_path, index=False)
print('session:', session)
print('rows:', len(df))
print('csv:', csv_path)

mc = df[df['kind'] == 'model_call'].copy() if 'kind' in df.columns else pd.DataFrame()
if len(mc):
    mc = mc.sort_values('call_index')
    fig = plt.figure(figsize=(9, 4.5))
    plt.plot(mc['call_index'], mc['input_rms'], marker='o', linewidth=1)
    plt.xlabel('model call index')
    plt.ylabel('sampled input RMS')
    plt.title('Anima inference trajectory: model input RMS')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out = session / 'model_input_rms_by_call.png'
    plt.savefig(out, dpi=160)
    plt.close(fig)
    print('saved:', out)

a1 = df[df['kind'] == 'attn1'].copy() if 'kind' in df.columns else pd.DataFrame()
if len(a1) and 'q_rms' in a1.columns:
    pivot = a1.pivot_table(index='block', columns='call_index', values='q_rms', aggfunc='mean')
    if pivot.size:
        fig = plt.figure(figsize=(11, 5))
        plt.imshow(pivot.values, aspect='auto', origin='lower')
        plt.colorbar(label='sampled q RMS')
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90, fontsize=7)
        plt.xlabel('model call index')
        plt.ylabel('DiT block')
        plt.title('Self-attention input RMS across Anima blocks')
        plt.tight_layout()
        out = session / 'attn1_q_rms_block_call_heatmap.png'
        plt.savefig(out, dpi=160)
        plt.close(fig)
        print('saved:', out)

rank_df = a1.dropna(subset=['q_effective_rank']) if len(a1) and 'q_effective_rank' in a1.columns else pd.DataFrame()
if len(rank_df):
    fig = plt.figure(figsize=(9, 5))
    for block, g in rank_df.groupby('block'):
        g = g.sort_values('call_index')
        plt.plot(g['call_index'], g['q_effective_rank'], marker='o', label=f'block {block}')
    plt.xlabel('model call index')
    plt.ylabel('approx. effective rank')
    plt.title('Selected-block representation effective rank')
    plt.legend(ncol=2, fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out = session / 'effective_rank_by_call.png'
    plt.savefig(out, dpi=160)
    plt.close(fig)
    print('saved:', out)

cols = [c for c in ['kind','call_index','block','q_dtype','k_dtype','v_dtype','q_device','k_device','v_device'] if c in df.columns]
if cols and 'kind' in df.columns:
    mismatch = df[df['kind'].isin(['attn1','attn2'])][cols].drop_duplicates()
    out = session / 'dtype_device_table.csv'
    mismatch.to_csv(out, index=False)
    print('saved:', out)

# DAAM-like cross-attention map index: one row per denoising call/block/text-token.
if 'kind' in df.columns:
    ca = df[df['kind'] == 'cross_attention_map'].copy()
    if len(ca):
        ca_cols = [c for c in [
            'call_index','sigma','block','text_token_index','text_key_count',
            'map_shape','map_path','gif_key'
        ] if c in ca.columns]
        out = session / 'cross_attention_maps.csv'
        ca[ca_cols].sort_values(['block','text_token_index','call_index']).to_csv(out, index=False)
        print('saved:', out)

print('\nGIFs:')
gifs = sorted((session / 'gifs').glob('*.gif')) if (session / 'gifs').exists() else []
if gifs:
    for p in gifs:
        print(p.relative_to(session))
else:
    print('(none)')

print('\nFiles:')
for p in sorted(session.rglob('*')):
    if p.is_file():
        print(p.relative_to(session))
