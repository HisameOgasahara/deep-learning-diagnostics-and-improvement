import json, math, time, uuid, threading, re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

_REC = {}


def _words(s):
    return [x.strip() for x in str(s).split(',') if x.strip()]


def _slug(s):
    return re.sub(r'[^0-9A-Za-z가-힣_-]+', '_', str(s)).strip('_')[:64] or 'word'


def _flat(tokens, key):
    return [x for group in tokens.get(key, []) for x in group]


def _tid(x):
    return int(x[0]) if isinstance(x, (tuple, list)) else int(x)


def _txt(x):
    return str(x[1]) if isinstance(x, (tuple, list)) and len(x) > 1 else str(x)


def _clean(s):
    return str(s).replace('▁', '').replace('Ġ', '')


def _detok(clip, pairs):
    try:
        return clip.tokenizer.t5xxl.untokenize(pairs)
    except Exception:
        try:
            return clip.tokenizer.untokenize(pairs)
        except Exception:
            return [(x[0], str(x[0])) for x in pairs]


def _mapping(tm):
    lines = [f"prompt: {tm.get('prompt', '')}", 'T5 target tokens -> Anima cross-attention key indices:']
    for i, (tid, txt) in enumerate(zip(tm.get('token_ids', []), tm.get('token_texts', []))):
        lines.append(f'[{i:03d}] id={tid:<6d} token={txt!r}')
    return '\n'.join(lines)


def _word_ids(tm, word):
    clip = tm.get('_clip')
    pieces = tm.get('token_texts', [])
    if clip is None:
        return []
    query_pairs = _flat(clip.tokenize(word), 't5xxl')
    query_decoded = _detok(clip, query_pairs)
    needle = ''.join(_clean(_txt(x)) for x in query_decoded if _txt(x) not in ('', '<pad>', '</s>')).lower()
    if not needle:
        return []
    clean = [_clean(x) for x in pieces]
    for i in range(len(clean)):
        if clean[i] in ('<pad>', '</s>'):
            continue
        cur = ''
        for j in range(i, len(clean)):
            if clean[j] in ('<pad>', '</s>'):
                break
            cur += clean[j]
            low = cur.lower()
            if low == needle:
                return list(range(i, j + 1))
            if len(low) > len(needle) or not needle.startswith(low):
                break
    return []


def _active_token_indices(tm):
    out = []
    for i, text in enumerate(tm.get('token_texts', [])):
        if text == '</s>':
            break
        if text != '<pad>':
            out.append(i)
    return out


def _grid(n, input_shape=None):
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


def _spatial(values, q, input_shape):
    a = np.asarray(values, np.float32)
    spatial = list(q.shape[1:-2])
    if len(spatial) >= 2:
        a = a.reshape(spatial)
        while a.ndim > 2:
            a = a.mean(0)
        return a
    g = _grid(a.size, input_shape)
    return None if g is None else a.reshape(*g)


def _canonical_hw(h, w, max_side=64):
    if h >= w:
        return max_side, max(1, int(round(w * max_side / h)))
    return max(1, int(round(h * max_side / w))), max_side


def _resize_np(a, h, w):
    t = torch.from_numpy(np.asarray(a, np.float32))[None, None]
    return F.interpolate(t, size=(h, w), mode='bicubic', align_corners=False)[0, 0].cpu().numpy()


def _minmax(a):
    a = np.asarray(a, np.float32)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    return np.clip((a - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def _rgb(norm, cmap='turbo'):
    norm = np.clip(norm, 0, 1)
    try:
        import matplotlib
        return (matplotlib.colormaps.get_cmap(cmap)(norm)[..., :3] * 255).astype(np.uint8)
    except Exception:
        return np.repeat((norm[..., None] * 255).astype(np.uint8), 3, -1)


def _append(st, rec):
    with st['lock']:
        with open(st['records'], 'a', encoding='utf-8') as f:
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


def _selected_attention(q, k, token_indices, opt, key_chunk=64, token_chunk=24):
    q = _conditional_only(q, opt)
    k = _conditional_only(k, opt)
    qf = q.detach().reshape(q.shape[0], -1, q.shape[-2], q.shape[-1]).float()
    kf = k.detach().reshape(k.shape[0], -1, k.shape[-2], k.shape[-1]).float()
    valid = [i for i in token_indices if 0 <= i < kf.shape[1]]
    if not valid:
        return {}, int(kf.shape[1])

    scale = 1.0 / math.sqrt(qf.shape[-1])
    running_max = torch.full(qf.shape[:3], -float('inf'), device=qf.device)
    running_sum = torch.zeros_like(running_max)
    for s in range(0, kf.shape[1], key_chunk):
        logits = torch.einsum('bqhd,bkhd->bqhk', qf, kf[:, s:s + key_chunk]) * scale
        cm = logits.amax(-1)
        nm = torch.maximum(running_max, cm)
        running_sum = running_sum * torch.exp(running_max - nm) + torch.exp(logits - nm[..., None]).sum(-1)
        running_max = nm
    log_denom = running_max + torch.log(running_sum.clamp_min(1e-30))

    out = {}
    for s in range(0, len(valid), token_chunk):
        ids = valid[s:s + token_chunk]
        logits = torch.einsum('bqhd,bkhd->bqhk', qf, kf[:, ids]) * scale
        probs = torch.exp(logits - log_denom[..., None]).sum(2)
        probs = probs.permute(2, 0, 1).contiguous().cpu().numpy()
        for j, tok in enumerate(ids):
            out[tok] = probs[j]
    return out, int(kf.shape[1])


def _record_cross(q, k, opt):
    sid = opt.get('anima_daam_v2_session')
    st = _REC.get(sid)
    if st is None:
        return
    ci = int(opt.get('anima_daam_v2_call_index', -1))
    if ci < 0 or ci % st['call_stride'] != 0:
        return

    probs, nkeys = _selected_attention(q, k, st['active_tokens'], opt)
    if not probs:
        return

    for tok, batch_values in probs.items():
        for batch_index in range(batch_values.shape[0]):
            arr = _spatial(batch_values[batch_index], q, st['input_shapes'].get(ci))
            if arr is None or arr.ndim != 2:
                continue
            h, w = arr.shape
            ch, cw = _canonical_hw(h, w, st['max_map_side'])
            arr = _resize_np(arr, ch, cw)
            key = (ci, batch_index, ch, cw, tok)
            with st['lock']:
                if key not in st['call_accum']:
                    st['call_accum'][key] = [arr.astype(np.float64), 1]
                else:
                    st['call_accum'][key][0] += arr
                    st['call_accum'][key][1] += 1
            st['observed_key_count'] = nkeys


def _flush_call(st, ci):
    rows = []
    with st['lock']:
        keys = [k for k in st['call_accum'] if k[0] == ci]
        for key in keys:
            call_idx, batch_idx, h, w, tok = key
            total, count = st['call_accum'].pop(key)
            rows.append((batch_idx, h, w, tok, count, (total / max(count, 1)).astype(np.float32)))

    for batch_idx, h, w, tok, layer_count, arr in rows:
        stem = f'call{ci:04d}_batch{batch_idx:02d}_res{h}x{w}_token{tok:03d}'
        raw = Path(st['timestep_raw']) / f'{stem}.npy'
        png = Path(st['timestep_maps']) / f'{stem}.png'
        np.save(raw, arr)
        Image.fromarray(_rgb(_minmax(arr), st['colormap']), 'RGB').save(png)
        _append(st, {
            'kind': 'timestep_token_map', 'call_index': ci, 'batch_index': batch_idx,
            'resolution': [h, w], 'text_token_index': tok, 'layer_count': layer_count,
            'sigma': st['sigmas'].get(ci), 'raw_npy': str(raw), 'png': str(png),
        })


def _install_hook():
    try:
        from comfy.ldm.cosmos.predict2 import Attention as A
    except Exception as e:
        print('[AnimaDAAMV2] Cosmos Predict2 hook unavailable:', repr(e))
        return
    if getattr(A, '_anima_daam_v2_hook_installed', False):
        return
    original = A.compute_qkv

    def wrapped(self, x, context=None, rope_emb=None, transformer_options={}):
        q, k, v = original(self, x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
        if not self.is_selfattn:
            try:
                _record_cross(q, k, transformer_options or {})
            except Exception as e:
                sid = (transformer_options or {}).get('anima_daam_v2_session')
                if sid in _REC:
                    _append(_REC[sid], {'kind': 'diagnostic_error', 'error': repr(e)})
        return q, k, v

    A.compute_qkv = wrapped
    A._anima_daam_v2_hook_installed = True
    A._anima_daam_v2_original_compute_qkv = original
    print('[AnimaDAAMV2] installed Cosmos Predict2 cross-attention hook')


_install_hook()


class AnimaTextEncodeWithTokenMapV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'clip': ('CLIP',), 'text': ('STRING', {'multiline': True, 'dynamicPrompts': True})}}

    RETURN_TYPES = ('CONDITIONING', 'ANIMA_TOKEN_MAP', 'STRING')
    RETURN_NAMES = ('conditioning', 'token_map', 'mapping_text')
    FUNCTION = 'encode'
    CATEGORY = 'diagnostics/anima'

    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        if not isinstance(tokens, dict) or 't5xxl' not in tokens:
            raise RuntimeError('This node requires the Anima tokenizer with a t5xxl stream.')
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop('cond')
        pairs = _flat(tokens, 't5xxl')
        decoded = _detok(clip, pairs)
        tm = {'prompt': text, 'token_ids': [_tid(x) for x in pairs], 'token_texts': [_txt(x) for x in decoded], '_clip': clip}
        return ([[cond, output]], tm, _mapping(tm))


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
        text = _mapping(token_map)
        return {'ui': {'text': [text]}, 'result': (text,)}


class AnimaAttentionDiagnosticsV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('MODEL',), 'token_map': ('ANIMA_TOKEN_MAP', {'forceInput': True}),
            'snapshot_every_n_calls': ('INT', {'default': 1, 'min': 1, 'max': 100}),
            'max_map_side': ('INT', {'default': 64, 'min': 16, 'max': 256}),
            'colormap': (['turbo', 'inferno', 'viridis', 'magma', 'plasma', 'gray'], {'default': 'turbo'}),
            'output_root': ('STRING', {'default': '/content/anima_diagnostics_v2'}),
        }}

    RETURN_TYPES = ('MODEL', 'STRING')
    RETURN_NAMES = ('model', 'diagnostic_directory')
    FUNCTION = 'patch'
    CATEGORY = 'diagnostics/anima'

    def patch(self, model, token_map, snapshot_every_n_calls, max_map_side, colormap, output_root):
        active = _active_token_indices(token_map)
        if not active:
            raise RuntimeError('No active T5 prompt tokens found.')
        sid = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        out = Path(output_root) / sid
        for d in ('timestep_raw', 'timestep_maps', 'global', 'individual'):
            (out / d).mkdir(parents=True, exist_ok=True)
        public = {k: v for k, v in token_map.items() if k != '_clip'}
        (out / 'token_map.json').write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding='utf-8')
        (out / 'token_map.txt').write_text(_mapping(token_map), encoding='utf-8')
        (out / 'session.json').write_text(json.dumps({
            'session': sid, 'active_token_indices': active,
            'snapshot_every_n_calls': int(snapshot_every_n_calls), 'max_map_side': int(max_map_side),
            'colormap': colormap,
            'aggregation': 'global: sum heads -> mean layers within call/resolution -> sum calls/resolutions -> mean word subtokens -> resize -> min-max once; individual: same pre-aggregation call/resolution maps -> mean word subtokens -> resize -> min-max per map',
            'reference': 'nisaruj/comfyui-daam semantics adapted to Anima/Cosmos Predict2',
        }, ensure_ascii=False, indent=2), encoding='utf-8')

        st = {
            'active_tokens': active, 'call_stride': int(snapshot_every_n_calls), 'max_map_side': int(max_map_side),
            'colormap': colormap, 'timestep_raw': str(out / 'timestep_raw'), 'timestep_maps': str(out / 'timestep_maps'),
            'records': str(out / 'records.jsonl'), 'call_index': 0, 'call_accum': {}, 'input_shapes': {},
            'sigmas': {}, 'observed_key_count': None, 'lock': threading.Lock(),
        }
        _REC[sid] = st
        m = model.clone()
        old = m.model_options.get('model_function_wrapper')

        def wrapper(apply_model, args):
            c = args['c'].copy()
            with st['lock']:
                ci = st['call_index']
                st['call_index'] += 1
            st['input_shapes'][ci] = list(args['input'].shape)
            st['sigmas'][ci] = float(args['timestep'].max().detach().cpu())
            to = c.get('transformer_options', {}).copy()
            to.update({'anima_daam_v2_session': sid, 'anima_daam_v2_call_index': ci})
            c['transformer_options'] = to
            result = old(apply_model, args | {'c': c}) if old is not None else apply_model(args['input'], args['timestep'], **c)
            if ci % st['call_stride'] == 0:
                _flush_call(st, ci)
            return result

        m.set_model_unet_function_wrapper(wrapper)
        print(f'[AnimaDAAMV2] session={sid} active_tokens={len(active)} output={out}')
        return (m, str(out))


def _records(session):
    p = Path(session) / 'records.jsonl'
    rows = []
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _load_record_array(session, rec):
    p = Path(rec.get('raw_npy', ''))
    if not p.exists():
        p = Path(session) / 'timestep_raw' / p.name
    return np.load(p).astype(np.float32) if p.exists() else None


def _global_token_map(session, batch_index, token_index):
    arrays = []
    for rec in _records(session):
        if rec.get('kind') != 'timestep_token_map':
            continue
        if int(rec.get('batch_index', -1)) != int(batch_index) or int(rec.get('text_token_index', -1)) != int(token_index):
            continue
        a = _load_record_array(session, rec)
        if a is not None:
            arrays.append(a)
    if not arrays:
        return None
    target_h = max(a.shape[0] for a in arrays)
    target_w = max(a.shape[1] for a in arrays)
    aligned = [_resize_np(a, target_h, target_w) if a.shape != (target_h, target_w) else a for a in arrays]
    return np.stack(aligned, axis=0).sum(axis=0)


def _word_global_map(session, token_map, word, batch_index):
    ids = _word_ids(token_map, word)
    if not ids:
        return None, []
    maps, used = [], []
    for tok in ids:
        m = _global_token_map(session, batch_index, tok)
        if m is not None:
            maps.append(m)
            used.append(tok)
    if not maps:
        return None, ids
    return np.stack(maps, axis=0).mean(axis=0), used


def _word_individual_maps(session, token_map, word, batch_index):
    """Return pre-global-aggregation word maps grouped by denoising call and resolution."""
    ids = _word_ids(token_map, word)
    if not ids:
        return [], []
    wanted = set(ids)
    groups = {}
    for rec in _records(session):
        if rec.get('kind') != 'timestep_token_map' or int(rec.get('batch_index', -1)) != int(batch_index):
            continue
        tok = int(rec.get('text_token_index', -1))
        if tok not in wanted:
            continue
        a = _load_record_array(session, rec)
        if a is None:
            continue
        res = tuple(int(x) for x in rec.get('resolution', list(a.shape)))
        key = (int(rec.get('call_index', -1)), res, rec.get('sigma'))
        groups.setdefault(key, {})[tok] = a

    out = []
    for (call_index, res, sigma), token_arrays in sorted(groups.items(), key=lambda kv: kv[0][0]):
        used = [tok for tok in ids if tok in token_arrays]
        if not used:
            continue
        arrays = [token_arrays[tok] for tok in used]
        th = max(a.shape[0] for a in arrays)
        tw = max(a.shape[1] for a in arrays)
        aligned = [_resize_np(a, th, tw) if a.shape != (th, tw) else a for a in arrays]
        out.append({'call_index': call_index, 'resolution': res, 'sigma': sigma, 'token_ids': used,
                    'raw': np.stack(aligned, axis=0).mean(axis=0)})
    return out, ids


def _resize_to_image(a, h, w):
    t = torch.from_numpy(np.asarray(a, np.float32))[None, None]
    return F.interpolate(t, size=(h, w), mode='bicubic', align_corners=False)[0, 0]


def _caption(t, text):
    arr = (t.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    im = Image.fromarray(arr)
    draw = ImageDraw.Draw(im)
    try:
        box = draw.textbbox((0, 0), text)
        tw, th = box[2] - box[0], box[3] - box[1]
    except Exception:
        tw, th = len(text) * 7, 12
    x = max(4, (im.width - tw) // 2)
    y = max(4, im.height - th - 10)
    draw.rectangle((x - 5, y - 4, x + tw + 5, y + th + 4), fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255))
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)


def _make_overlay(base, raw, h, w, cmap, alpha, caption, label):
    resized = _resize_to_image(raw, h, w).cpu().numpy()
    norm = _minmax(resized)
    heat = torch.from_numpy(_rgb(norm, cmap).astype(np.float32) / 255.0)
    over = ((1.0 - float(alpha)) * base + float(alpha) * heat).clamp(0, 1)
    if caption:
        over = _caption(over, label)
    return over, heat, norm


class AnimaAttentionOverlayV2:
    """DAAM-style global overlay plus optional pre-aggregation call/resolution overlays."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'images': ('IMAGE',), 'token_map': ('ANIMA_TOKEN_MAP', {'forceInput': True}),
            'diagnostic_directory': ('STRING', {'forceInput': True}),
            'attention_words': ('STRING', {'multiline': True, 'default': 'arona'}),
            'view_mode': (['global', 'individual', 'global + individual'], {'default': 'global'}),
            'alpha': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.05}),
            'caption': ('BOOLEAN', {'default': True}),
        }}

    RETURN_TYPES = ('IMAGE', 'IMAGE', 'STRING')
    RETURN_NAMES = ('overlay_images', 'heatmap_images', 'debug_info')
    FUNCTION = 'overlay'
    CATEGORY = 'diagnostics/anima'

    def overlay(self, images, token_map, diagnostic_directory, attention_words, view_mode, alpha, caption):
        session = Path(diagnostic_directory)
        cfg = json.loads((session / 'session.json').read_text(encoding='utf-8'))
        cmap = cfg.get('colormap', 'turbo')
        overlays, heatmaps, info = [], [], []
        show_global = view_mode in ('global', 'global + individual')
        show_individual = view_mode in ('individual', 'global + individual')

        for batch_index in range(images.shape[0]):
            base = images[batch_index].detach().cpu().float().clamp(0, 1)
            h, w = int(base.shape[0]), int(base.shape[1])
            for word in _words(attention_words):
                if show_global:
                    raw, ids = _word_global_map(session, token_map, word, batch_index)
                    if raw is None:
                        raise RuntimeError(f'No DAAM maps found for word {word!r}; mapped token ids={ids}')
                    label = f'GLOBAL {word} tokens {ids}'
                    over, heat, norm = _make_overlay(base, raw, h, w, cmap, alpha, caption, label)
                    overlays.append(over)
                    heatmaps.append(heat)
                    info.append(label)
                    global_dir = session / 'global'
                    global_dir.mkdir(exist_ok=True)
                    np.save(global_dir / f'batch{batch_index:02d}_word-{_slug(word)}_raw.npy', raw.astype(np.float32))
                    Image.fromarray(_rgb(norm, cmap), 'RGB').save(global_dir / f'batch{batch_index:02d}_word-{_slug(word)}_heatmap.png')

                if show_individual:
                    items, ids = _word_individual_maps(session, token_map, word, batch_index)
                    if not items:
                        raise RuntimeError(f'No individual DAAM maps found for word {word!r}; mapped token ids={ids}')
                    individual_dir = session / 'individual'
                    individual_dir.mkdir(exist_ok=True)
                    for item in items:
                        ci = item['call_index']
                        rh, rw = item['resolution']
                        sigma = item['sigma']
                        used = item['token_ids']
                        label = f'CALL {ci} res {rh}x{rw} sigma {sigma:.4g} {word} tokens {used}' if sigma is not None else f'CALL {ci} res {rh}x{rw} {word} tokens {used}'
                        over, heat, norm = _make_overlay(base, item['raw'], h, w, cmap, alpha, caption, label)
                        overlays.append(over)
                        heatmaps.append(heat)
                        info.append(label)
                        stem = f'batch{batch_index:02d}_word-{_slug(word)}_call{ci:04d}_res{rh}x{rw}'
                        np.save(individual_dir / f'{stem}_raw.npy', item['raw'].astype(np.float32))
                        Image.fromarray(_rgb(norm, cmap), 'RGB').save(individual_dir / f'{stem}_heatmap.png')

        if not overlays:
            raise RuntimeError('No overlay images produced.')
        return (torch.stack(overlays), torch.stack(heatmaps), '\n'.join(info))


NODE_CLASS_MAPPINGS = {
    'AnimaTextEncodeWithTokenMapV2': AnimaTextEncodeWithTokenMapV2,
    'AnimaTokenMapViewerV2': AnimaTokenMapViewerV2,
    'AnimaAttentionDiagnosticsV2': AnimaAttentionDiagnosticsV2,
    'AnimaAttentionOverlayV2': AnimaAttentionOverlayV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'AnimaTextEncodeWithTokenMapV2': 'Anima Text Encode + Token Map V2',
    'AnimaTokenMapViewerV2': 'Anima Token Map Viewer V2',
    'AnimaAttentionDiagnosticsV2': 'Anima Attention Diagnostics V2',
    'AnimaAttentionOverlayV2': 'Anima Attention Overlay V2',
}
