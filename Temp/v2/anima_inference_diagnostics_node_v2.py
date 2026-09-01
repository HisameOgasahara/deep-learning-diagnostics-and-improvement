import json, math, time, uuid, threading, re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_RECORDER = {}


def _parse_ints(text):
    text = str(text).strip().lower()
    if text in ('', 'all', '*'):
        return None
    out = set()
    for part in text.replace(' ', '').split(','):
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _parse_words(text):
    return [w.strip() for w in str(text).split(',') if w.strip()]


def _slug(text):
    s = re.sub(r'[^0-9A-Za-z가-힣_-]+', '_', str(text)).strip('_')
    return s[:64] or 'word'


def _flatten_stream(tokens, key):
    stream = tokens.get(key, []) if isinstance(tokens, dict) else []
    return [item for group in stream for item in group]


def _token_id(item):
    return int(item[0]) if isinstance(item, (tuple, list)) else int(item)


def _t5_detokenize(clip, pairs):
    try:
        return clip.tokenizer.t5xxl.untokenize(pairs)
    except Exception:
        try:
            return clip.tokenizer.untokenize(pairs)
        except Exception:
            return [(p[0], str(p[0])) for p in pairs]


def _decoded_text(item):
    if isinstance(item, (tuple, list)) and len(item) > 1:
        return str(item[1])
    return str(item)


def _clean_piece(text):
    return str(text).replace('▁', '').replace('Ġ', '')


def _find_word_indices_from_map(token_map, word):
    pieces = token_map.get('token_texts', [])
    clip = token_map.get('_clip')
    if clip is None:
        return []
    try:
        q_tokens = clip.tokenize(word)
        q_pairs = _flatten_stream(q_tokens, 't5xxl')
        q_decoded = _t5_detokenize(clip, q_pairs)
        query_pieces = [_clean_piece(_decoded_text(x)) for x in q_decoded]
        query_pieces = [x for x in query_pieces if x not in ('', '<pad>', '</s>')]
    except Exception:
        query_pieces = [_clean_piece(word)]
    joined_query = ''.join(query_pieces).lower()
    if not joined_query:
        return []

    clean = [_clean_piece(x) for x in pieces]
    matches = []
    for i in range(len(clean)):
        if clean[i] in ('<pad>', '</s>'):
            continue
        current = ''
        for j in range(i, len(clean)):
            if clean[j] in ('<pad>', '</s>'):
                break
            current += clean[j]
            low = current.lower()
            if low == joined_query:
                matches.extend(range(i, j + 1))
                break
            if len(low) > len(joined_query) or not joined_query.startswith(low):
                break
    return sorted(set(matches))


def _mapping_text(token_map):
    lines = [f"prompt: {token_map.get('prompt', '')}", 'T5 target tokens -> Anima cross-attention key indices:']
    ids = token_map.get('token_ids', [])
    texts = token_map.get('token_texts', [])
    for i, (tid, txt) in enumerate(zip(ids, texts)):
        lines.append(f'[{i:03d}] id={tid:<6d} token={txt!r}')
    pad_to = int(token_map.get('cross_attention_key_count', 512))
    if pad_to > len(ids):
        lines.append(f'[{len(ids):03d}..{pad_to-1:03d}] model-side zero padding added by Anima preprocess_text_embeds')
    return '\n'.join(lines)


def _infer_grid(n, input_shape=None):
    if n <= 0:
        return None
    ratio = 1.0
    if input_shape and len(input_shape) >= 2 and input_shape[-1] > 0:
        ratio = float(input_shape[-2]) / float(input_shape[-1])
    best, best_score = None, float('inf')
    for h in range(1, int(n ** 0.5) + 1):
        if n % h:
            continue
        w = n // h
        for hh, ww in ((h, w), (w, h)):
            score = abs(math.log((hh / ww + 1e-12) / (ratio + 1e-12)))
            if score < best_score:
                best_score, best = score, (hh, ww)
    return best


def _colorize_fixed(arr, vmax, cmap='inferno'):
    arr = np.asarray(arr, np.float32)
    norm = np.clip(arr / max(float(vmax), 1e-8), 0.0, 1.0)
    if cmap == 'gray':
        return np.repeat((norm[..., None] * 255).astype(np.uint8), 3, -1)
    try:
        import matplotlib
        return (matplotlib.colormaps.get_cmap(cmap)(norm)[..., :3] * 255).astype(np.uint8)
    except Exception:
        return np.repeat((norm[..., None] * 255).astype(np.uint8), 3, -1)


def _save_map_fixed(arr, path, vmax, cmap):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix('.npy'), np.asarray(arr, np.float32))
    Image.fromarray(_colorize_fixed(arr, vmax, cmap), 'RGB').save(path)


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
    keep = [i for i, label in enumerate(labels) if int(label) == 0]
    if not keep:
        return x
    return chunks[keep].reshape(-1, *x.shape[1:])


def _selected_probs_per_head(q, k, tokens, opt, chunk=64):
    q = _conditional_only(q, opt)
    k = _conditional_only(k, opt)
    qf = q.detach().reshape(q.shape[0], -1, q.shape[-2], q.shape[-1]).float()
    kf = k.detach().reshape(k.shape[0], -1, k.shape[-2], k.shape[-1]).float()
    valid = [i for i in sorted(tokens) if 0 <= i < kf.shape[1]]
    if not valid:
        return {}, int(kf.shape[1])
    scale = 1.0 / math.sqrt(qf.shape[-1])
    running_max = torch.full(qf.shape[:3], -float('inf'), device=qf.device)
    running_sum = torch.zeros_like(running_max)
    for s in range(0, kf.shape[1], chunk):
        logits = torch.einsum('bqhd,bkhd->bqhk', qf, kf[:, s:s + chunk]) * scale
        chunk_max = logits.amax(-1)
        new_max = torch.maximum(running_max, chunk_max)
        running_sum = running_sum * torch.exp(running_max - new_max) + torch.exp(logits - new_max[..., None]).sum(-1)
        running_max = new_max
    log_denom = running_max + torch.log(running_sum.clamp_min(1e-30))
    logits = torch.einsum('bqhd,bkhd->bqhk', qf, kf[:, valid]) * scale
    probs = torch.exp(logits - log_denom[..., None]).mean(0).permute(2, 1, 0).contiguous()
    return {tok: probs[j].cpu().numpy() for j, tok in enumerate(valid)}, int(kf.shape[1])


def _reshape_heads(values_hq, q, input_shape):
    values_hq = np.asarray(values_hq, np.float32)
    spatial = list(q.shape[1:-2])
    maps = []
    for h in range(values_hq.shape[0]):
        arr = values_hq[h]
        if len(spatial) >= 2:
            arr = arr.reshape(spatial)
            while arr.ndim > 2:
                arr = arr.mean(0)
        else:
            grid = _infer_grid(arr.size, input_shape)
            if not grid:
                return None
            arr = arr.reshape(*grid)
        maps.append(arr)
    return np.stack(maps, axis=0)


def _spatial_concentration_scores(head_maps):
    x = np.asarray(head_maps, np.float64).reshape(head_maps.shape[0], -1)
    x = np.clip(x, 0.0, None)
    p = x / np.maximum(x.sum(axis=1, keepdims=True), 1e-30)
    entropy = -(p * np.log(np.maximum(p, 1e-30))).sum(axis=1)
    return 1.0 - entropy / math.log(max(x.shape[1], 2))


def _pick_heads(st, head_maps):
    nheads = int(head_maps.shape[0])
    explicit = st['selected_heads']
    if explicit is not None:
        return [h for h in sorted(explicit) if 0 <= h < nheads], None
    if st['head_selection_mode'] == 'concentration_topk':
        scores = _spatial_concentration_scores(head_maps)
        k = max(1, min(int(st['top_k_heads']), nheads))
        chosen = np.argsort(scores)[-k:][::-1].tolist()
        return chosen, scores.tolist()
    return list(range(nheads)), None


def _record_cross(q, k, opt):
    sid = opt.get('anima_diag_v2_session')
    if not sid or sid not in _RECORDER:
        return
    st = _RECORDER[sid]
    ci = int(opt.get('anima_diag_v2_call_index', -1))
    bi = int(opt.get('block_index', -1))
    if ci < 0 or bi not in st['selected_blocks'] or ci % st['snapshot_every_n_calls'] != 0:
        return

    all_indices = sorted({i for idxs in st['word_indices'].values() for i in idxs})
    probs, nkeys = _selected_probs_per_head(q, k, all_indices, opt)
    expected = int(st['token_map'].get('cross_attention_key_count', 512))
    if nkeys != expected and not st.get('key_count_warning_written'):
        st['key_count_warning_written'] = True
        _append(st, {'kind': 'token_mapping_warning', 'expected_key_count': expected, 'observed_key_count': nkeys})

    for word, idxs in st['word_indices'].items():
        available = [probs[i] for i in idxs if i in probs]
        if not available:
            continue
        values_hq = np.stack(available, axis=0).mean(axis=0)
        head_maps = _reshape_heads(values_hq, q, st['input_shapes'].get(ci))
        if head_maps is None or head_maps.ndim != 3:
            continue
        ratio_maps = head_maps * float(nkeys)
        chosen, concentration_scores = _pick_heads(st, ratio_maps)
        if not chosen:
            continue
        aggregate = ratio_maps[chosen].mean(axis=0)
        slug = _slug(word)
        raw_base = Path(st['raw_dir']) / f'call{ci:04d}_block{bi:02d}_word-{slug}'
        np.savez_compressed(
            str(raw_base) + '.npz',
            attention_probability=head_maps.astype(np.float32),
            attention_ratio=ratio_maps.astype(np.float32),
            aggregate_ratio=aggregate.astype(np.float32),
            selected_heads=np.asarray(chosen, dtype=np.int32),
            token_indices=np.asarray(idxs, dtype=np.int32),
            text_key_count=np.asarray([nkeys], dtype=np.int32),
        )
        agg_png = Path(st['attention_maps_dir']) / f'agg_call{ci:04d}_block{bi:02d}_word-{slug}.png'
        _save_map_fixed(aggregate, agg_png, st['ratio_vmax'], st['colormap'])
        if st['save_head_pngs']:
            for h in chosen:
                hp = Path(st['head_maps_dir']) / f'call{ci:04d}_block{bi:02d}_word-{slug}_head{h:02d}.png'
                _save_map_fixed(ratio_maps[h], hp, st['ratio_vmax'], st['colormap'])
        _append(st, {
            'time': time.time(), 'session': sid, 'kind': 'cross_attention_word_v2',
            'call_index': ci, 'sigma': opt.get('anima_diag_v2_sigma'), 'block': bi,
            'attention_word': word, 'token_indices': idxs, 'text_key_count': nkeys,
            'head_count': int(head_maps.shape[0]), 'selected_heads': chosen,
            'head_selection_mode': st['head_selection_mode'], 'concentration_scores': concentration_scores,
            'map_shape': list(aggregate.shape), 'ratio_vmax': st['ratio_vmax'],
            'aggregate_png': str(agg_png), 'raw_npz': str(raw_base) + '.npz',
        })


def _install_hook():
    try:
        from comfy.ldm.cosmos.predict2 import Attention as A
    except Exception as e:
        print('[AnimaDiagnosticsV2] Cosmos Predict2 Attention hook unavailable:', repr(e))
        return
    if getattr(A, '_anima_diag_v2_hook_installed', False):
        return
    original = A.compute_qkv
    def wrapped(self, x, context=None, rope_emb=None, transformer_options={}):
        q, k, v = original(self, x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
        if not self.is_selfattn:
            try:
                _record_cross(q, k, transformer_options or {})
            except Exception as e:
                sid = (transformer_options or {}).get('anima_diag_v2_session')
                if sid in _RECORDER:
                    _append(_RECORDER[sid], {'kind': 'diagnostic_error', 'error': 'cross:' + repr(e)})
        return q, k, v
    A.compute_qkv = wrapped
    A._anima_diag_v2_hook_installed = True
    A._anima_diag_v2_original_compute_qkv = original
    print('[AnimaDiagnosticsV2] installed projected Q/K hook')


_install_hook()


class AnimaTextEncodeWithTokenMapV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'clip': ('CLIP',),
            'text': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
        }}
    RETURN_TYPES = ('CONDITIONING', 'ANIMA_TOKEN_MAP', 'STRING')
    RETURN_NAMES = ('conditioning', 'token_map', 'mapping_text')
    FUNCTION = 'encode'
    CATEGORY = 'diagnostics/anima'

    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        if not isinstance(tokens, dict) or 't5xxl' not in tokens:
            raise RuntimeError('This node requires the Anima tokenizer with a t5xxl token stream.')
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop('cond')
        pairs = _flatten_stream(tokens, 't5xxl')
        decoded = _t5_detokenize(clip, pairs)
        token_ids = [_token_id(x) for x in pairs]
        token_texts = [_decoded_text(x) for x in decoded]
        token_map = {
            'prompt': text,
            'stream': 't5xxl',
            'token_ids': token_ids,
            'token_texts': token_texts,
            'token_count': len(token_ids),
            'cross_attention_key_count': max(512, len(token_ids)),
            '_clip': clip,
        }
        mapping = _mapping_text(token_map)
        return ([[cond, output]], token_map, mapping)


class AnimaTokenMapViewerV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'token_map': ('ANIMA_TOKEN_MAP', {'forceInput': True})}}
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('mapping_text',)
    FUNCTION = 'show'
    OUTPUT_NODE = True
    CATEGORY = 'diagnostics/anima'

    def show(self, token_map):
        text = _mapping_text(token_map)
        return {'ui': {'text': [text]}, 'result': (text,)}


class AnimaAttentionDiagnosticsV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('MODEL',),
            'token_map': ('ANIMA_TOKEN_MAP', {'forceInput': True}),
            'attention_words': ('STRING', {'multiline': True, 'default': 'girl'}),
            'selected_blocks': ('STRING', {'default': '0,6,12,18,24,27'}),
            'selected_heads': ('STRING', {'default': 'all'}),
            'head_selection_mode': (['all', 'concentration_topk'], {'default': 'concentration_topk'}),
            'top_k_heads': ('INT', {'default': 4, 'min': 1, 'max': 64}),
            'snapshot_every_n_calls': ('INT', {'default': 1, 'min': 1, 'max': 100}),
            'ratio_vmax': ('FLOAT', {'default': 6.0, 'min': 1.0, 'max': 50.0, 'step': 0.5}),
            'save_head_pngs': ('BOOLEAN', {'default': True}),
            'colormap': (['inferno', 'viridis', 'magma', 'plasma', 'gray'], {'default': 'inferno'}),
            'output_root': ('STRING', {'default': '/content/anima_diagnostics_v2'}),
        }}
    RETURN_TYPES = ('MODEL', 'STRING', 'STRING')
    RETURN_NAMES = ('model', 'diagnostic_directory', 'selected_word_mapping')
    FUNCTION = 'patch'
    CATEGORY = 'diagnostics/anima'

    def patch(self, model, token_map, attention_words, selected_blocks, selected_heads,
              head_selection_mode, top_k_heads, snapshot_every_n_calls, ratio_vmax,
              save_head_pngs, colormap, output_root):
        words = _parse_words(attention_words)
        if not words:
            raise ValueError('attention_words is empty.')
        word_indices = {w: _find_word_indices_from_map(token_map, w) for w in words}
        missing = [w for w, ids in word_indices.items() if not ids]
        if missing:
            raise ValueError('Could not map attention_words to T5 token positions: ' + ', '.join(missing))
        selected_mapping = '\n'.join(f'{w!r} -> {ids}' for w, ids in word_indices.items())

        sid = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        out = Path(output_root) / sid
        for name in ('attention_maps', 'head_maps', 'raw'):
            (out / name).mkdir(parents=True, exist_ok=True)
        blocks = _parse_ints(selected_blocks) or set()
        st = {
            'session': sid, 'out_dir': str(out), 'attention_maps_dir': str(out / 'attention_maps'),
            'head_maps_dir': str(out / 'head_maps'), 'raw_dir': str(out / 'raw'),
            'records_path': str(out / 'records.jsonl'), 'selected_blocks': blocks,
            'token_map': token_map, 'word_indices': word_indices,
            'selected_heads': _parse_ints(selected_heads), 'head_selection_mode': head_selection_mode,
            'top_k_heads': int(top_k_heads), 'snapshot_every_n_calls': int(snapshot_every_n_calls),
            'ratio_vmax': float(ratio_vmax), 'save_head_pngs': bool(save_head_pngs), 'colormap': colormap,
            'call_index': 0, 'input_shapes': {}, 'lock': threading.Lock(), 'key_count_warning_written': False,
        }
        _RECORDER[sid] = st
        public_map = {k: v for k, v in token_map.items() if k != '_clip'}
        (out / 'token_map.json').write_text(json.dumps(public_map, ensure_ascii=False, indent=2), encoding='utf-8')
        (out / 'token_map.txt').write_text(_mapping_text(token_map), encoding='utf-8')
        (out / 'session.json').write_text(json.dumps({
            'session': sid, 'attention_words': words, 'word_indices': word_indices,
            'selected_blocks': sorted(blocks),
            'selected_heads': None if st['selected_heads'] is None else sorted(st['selected_heads']),
            'head_selection_mode': head_selection_mode, 'top_k_heads': int(top_k_heads),
            'snapshot_every_n_calls': int(snapshot_every_n_calls), 'ratio_vmax': float(ratio_vmax),
            'save_head_pngs': bool(save_head_pngs), 'colormap': colormap,
            'scale_definition': 'attention_ratio = token_probability * text_key_count; 1.0 = uniform attention',
        }, ensure_ascii=False, indent=2), encoding='utf-8')

        m = model.clone()
        old = m.model_options.get('model_function_wrapper')
        def wrapper(apply_model, args):
            c = args['c'].copy()
            with st['lock']:
                ci = st['call_index']; st['call_index'] += 1
            sigma = float(args['timestep'].max().detach().cpu())
            to = c.get('transformer_options', {}).copy()
            to.update({'anima_diag_v2_session': sid, 'anima_diag_v2_call_index': ci, 'anima_diag_v2_sigma': sigma})
            c['transformer_options'] = to
            st['input_shapes'][ci] = list(args['input'].shape)
            return old(apply_model, args | {'c': c}) if old is not None else apply_model(args['input'], args['timestep'], **c)
        m.set_model_unet_function_wrapper(wrapper)
        print(f'[AnimaDiagnosticsV2] session={sid} words={word_indices} blocks={sorted(blocks)}')
        return (m, str(out), selected_mapping)


NODE_CLASS_MAPPINGS = {
    'AnimaTextEncodeWithTokenMapV2': AnimaTextEncodeWithTokenMapV2,
    'AnimaTokenMapViewerV2': AnimaTokenMapViewerV2,
    'AnimaAttentionDiagnosticsV2': AnimaAttentionDiagnosticsV2,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'AnimaTextEncodeWithTokenMapV2': 'Anima Text Encode + Token Map V2',
    'AnimaTokenMapViewerV2': 'Anima Token Map Viewer V2',
    'AnimaAttentionDiagnosticsV2': 'Anima Attention Diagnostics V2',
}
