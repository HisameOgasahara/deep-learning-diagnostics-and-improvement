import json, math, time, uuid, threading, re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_RECORDER = {}


def _parse_ints(text):
    out = set()
    for part in str(text).replace(' ', '').split(','):
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _sample_stats(x, max_values=262144):
    z = x.detach().reshape(-1)
    if z.numel() == 0:
        return {'mean': None, 'std': None, 'rms': None, 'zero_frac': None}
    if z.numel() > max_values:
        idx = torch.linspace(0, z.numel() - 1, max_values, device=z.device).long()
        z = z.index_select(0, idx)
    z = z.float()
    return {
        'mean': float(z.mean()),
        'std': float(z.std(unbiased=False)),
        'rms': float(torch.sqrt((z * z).mean())),
        'zero_frac': float((z == 0).float().mean()),
    }


def _effective_rank(x, max_tokens=64):
    z = x.detach().reshape(-1, x.shape[-1])
    if z.shape[0] < 2:
        return None
    if z.shape[0] > max_tokens:
        idx = torch.linspace(0, z.shape[0] - 1, max_tokens, device=z.device).long()
        z = z.index_select(0, idx)
    z = z.float() - z.float().mean(0, keepdim=True)
    try:
        s = torch.linalg.svdvals(z)
        p = s * s
        d = p.sum()
        if d <= 0:
            return 0.0
        p = p / d
        p = p[p > 0]
        return float(torch.exp(-(p * torch.log(p)).sum()))
    except Exception:
        return None


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


def _normalize(arr):
    arr = np.asarray(arr, np.float32)
    lo, hi = np.nanpercentile(arr, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr) + 1e-8)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)


def _colorize(arr, cmap='inferno'):
    norm = _normalize(arr)
    if cmap == 'gray':
        return np.repeat((norm[..., None] * 255).astype(np.uint8), 3, -1)
    try:
        import matplotlib
        return (matplotlib.colormaps.get_cmap(cmap)(norm)[..., :3] * 255).astype(np.uint8)
    except Exception:
        return np.repeat((norm[..., None] * 255).astype(np.uint8), 3, -1)


def _save_map(arr, path, cmap='inferno'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_colorize(arr, cmap), 'RGB').save(path)
    np.save(path.with_suffix('.npy'), np.asarray(arr, np.float32))


def _append(state, rec):
    with state['lock']:
        with open(state['records_path'], 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def _save_rms(x, path, input_shape, cmap):
    z = x.detach()
    if z.ndim == 3:
        grid = _infer_grid(int(z.shape[1]), input_shape)
        if not grid:
            return None
        arr = z[0].float().pow(2).mean(-1).sqrt().reshape(*grid).cpu().numpy()
    elif z.ndim == 5:
        arr = z[0].float().pow(2).mean(-1).sqrt().mean(0).cpu().numpy()
    elif z.ndim == 4:
        arr = z[0].float().pow(2).mean(-1).sqrt().cpu().numpy()
    else:
        return None
    if arr.ndim != 2:
        return None
    _save_map(arr, path, cmap)
    return list(arr.shape)


def _record_pre(kind, q, k, v, opt):
    sid = opt.get('anima_diag_session')
    if not sid or sid not in _RECORDER:
        return
    st = _RECORDER[sid]
    ci = int(opt.get('anima_diag_call_index', -1))
    bi = int(opt.get('block_index', -1))
    if ci < 0 or ci % st['record_every_n_calls'] != 0:
        return
    rec = {
        'time': time.time(), 'session': sid, 'kind': kind, 'call_index': ci,
        'block': bi, 'sigma': opt.get('anima_diag_sigma'),
        'q_shape': list(q.shape), 'k_shape': list(k.shape), 'v_shape': list(v.shape),
        'q_dtype': str(q.dtype), 'k_dtype': str(k.dtype), 'v_dtype': str(v.dtype),
        'q_device': str(q.device), 'k_device': str(k.device), 'v_device': str(v.device),
    }
    rec.update({f'q_{a}': b for a, b in _sample_stats(q).items()})
    if bi in st['selected_blocks'] and ci % st['snapshot_every_n_calls'] == 0:
        rec['q_effective_rank'] = _effective_rank(q, st['max_rank_tokens'])
        if st['map_mode'] in ('representation_rms', 'both'):
            p = Path(st['maps_dir']) / f'{kind}_call{ci:04d}_block{bi:02d}_q_rms.png'
            shp = _save_rms(q, p, st['input_shapes'].get(ci), st['colormap'])
            rec['q_rms_map_shape'] = shp
            if shp:
                rec['q_rms_map'] = str(p)
    _append(st, rec)


def _make_patch(kind):
    def patch(q, k, v, pe=None, attn_mask=None, extra_options=None):
        opt = extra_options or {}
        try:
            _record_pre(kind, q, k, v, opt)
        except Exception as e:
            sid = opt.get('anima_diag_session')
            if sid in _RECORDER:
                _append(_RECORDER[sid], {
                    'kind': 'diagnostic_error', 'error': repr(e),
                    'call_index': opt.get('anima_diag_call_index', -1),
                    'block': opt.get('block_index', -1),
                })
        return {'q': q, 'k': k, 'v': v, 'pe': pe}
    return patch


def _selected_probs(q, k, tokens, chunk=64):
    qf = q.detach().reshape(q.shape[0], -1, q.shape[-2], q.shape[-1]).float()
    kf = k.detach().reshape(k.shape[0], -1, k.shape[-2], k.shape[-1]).float()
    valid = [i for i in sorted(tokens) if 0 <= i < kf.shape[1]]
    if not valid:
        return {}, int(kf.shape[1])
    scale = 1 / math.sqrt(qf.shape[-1])
    rm = torch.full(qf.shape[:3], -float('inf'), device=qf.device)
    rs = torch.zeros_like(rm)
    for s in range(0, kf.shape[1], chunk):
        logits = torch.einsum('bqhd,bkhd->bqhk', qf, kf[:, s:s + chunk]) * scale
        cm = logits.amax(-1)
        nm = torch.maximum(rm, cm)
        rs = rs * torch.exp(rm - nm) + torch.exp(logits - nm[..., None]).sum(-1)
        rm = nm
    denom = rm + torch.log(rs.clamp_min(1e-30))
    logits = torch.einsum('bqhd,bkhd->bqhk', qf, kf[:, valid]) * scale
    probs = torch.exp(logits - denom[..., None]).mean(0).mean(1)
    return {tok: probs[:, j].cpu().numpy() for j, tok in enumerate(valid)}, int(kf.shape[1])


def _reshape_q(values, q, input_shape):
    spatial = list(q.shape[1:-2])
    arr = np.asarray(values, np.float32)
    if len(spatial) >= 2:
        arr = arr.reshape(spatial)
        while arr.ndim > 2:
            arr = arr.mean(0)
        return arr
    grid = _infer_grid(arr.size, input_shape)
    return None if not grid else arr.reshape(*grid)


def _update_raw_gif(st, key):
    if not st['make_gif']:
        return
    files = st['gif_frames'].get(key, [])
    ims = [Image.open(p).convert('RGB') for p in files if Path(p).exists()]
    if ims:
        ims[0].save(
            Path(st['gifs_dir']) / (key + '.gif'), save_all=True,
            append_images=ims[1:], duration=st['gif_duration_ms'], loop=0,
        )


def _record_cross(q, k, opt):
    sid = opt.get('anima_diag_session')
    if not sid or sid not in _RECORDER:
        return
    st = _RECORDER[sid]
    if st['map_mode'] not in ('cross_attention', 'both'):
        return
    ci = int(opt.get('anima_diag_call_index', -1))
    bi = int(opt.get('block_index', -1))
    if ci < 0 or bi not in st['selected_blocks'] or ci % st['snapshot_every_n_calls'] != 0:
        return
    probs, nkeys = _selected_probs(q, k, st['text_token_indices'])
    for tok, vals in probs.items():
        arr = _reshape_q(vals, q, st['input_shapes'].get(ci))
        if arr is None or arr.ndim != 2:
            continue
        p = Path(st['attention_maps_dir']) / f'crossattn_call{ci:04d}_block{bi:02d}_token{tok:03d}.png'
        _save_map(arr, p, st['colormap'])
        key = f'crossattn_block{bi:02d}_token{tok:03d}'
        with st['lock']:
            st['gif_frames'].setdefault(key, []).append(str(p))
        _update_raw_gif(st, key)
        _append(st, {
            'time': time.time(), 'session': sid, 'kind': 'cross_attention_map',
            'call_index': ci, 'block': bi, 'text_token_index': tok,
            'text_key_count': nkeys, 'map_shape': list(arr.shape),
            'map_path': str(p), 'sigma': opt.get('anima_diag_sigma'),
        })


def _install_hook():
    try:
        from comfy.ldm.cosmos.predict2 import Attention as A
    except Exception as e:
        print('[AnimaDiagnostics] QK hook unavailable', repr(e))
        return
    if getattr(A, '_anima_diag_hook_installed', False):
        return
    original = A.compute_qkv
    def wrapped(self, x, context=None, rope_emb=None, transformer_options={}):
        q, k, v = original(self, x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
        if not self.is_selfattn:
            try:
                _record_cross(q, k, transformer_options or {})
            except Exception as e:
                sid = (transformer_options or {}).get('anima_diag_session')
                if sid in _RECORDER:
                    _append(_RECORDER[sid], {'kind': 'diagnostic_error', 'error': 'cross:' + repr(e)})
        return q, k, v
    A.compute_qkv = wrapped
    A._anima_diag_hook_installed = True
    A._anima_diag_original_compute_qkv = original
    print('[AnimaDiagnostics] installed projected Q/K hook')


_install_hook()


class AnimaInferenceDiagnostics:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('MODEL',),
            'map_mode': (['representation_rms', 'cross_attention', 'both'], {'default': 'both'}),
            'selected_blocks': ('STRING', {'default': '0,6,12,18,24,27'}),
            'text_token_indices': ('STRING', {'default': '0'}),
            'snapshot_every_n_calls': ('INT', {'default': 1, 'min': 1, 'max': 100}),
            'make_gif': ('BOOLEAN', {'default': True}),
            'save_step_latents': ('BOOLEAN', {'default': True}),
            'gif_duration_ms': ('INT', {'default': 250, 'min': 50, 'max': 5000}),
            'colormap': (['inferno', 'viridis', 'magma', 'plasma', 'gray'], {'default': 'inferno'}),
            'output_root': ('STRING', {'default': '/content/anima_diagnostics'}),
            'record_every_n_calls': ('INT', {'default': 1, 'min': 1, 'max': 100}),
            'max_rank_tokens': ('INT', {'default': 64, 'min': 8, 'max': 512}),
        }}
    RETURN_TYPES = ('MODEL', 'STRING')
    RETURN_NAMES = ('model', 'diagnostic_directory')
    FUNCTION = 'patch'
    CATEGORY = 'diagnostics/anima'

    def patch(self, model, map_mode, selected_blocks, text_token_indices,
              snapshot_every_n_calls, make_gif, save_step_latents, gif_duration_ms,
              colormap, output_root, record_every_n_calls, max_rank_tokens):
        sid = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        out = Path(output_root) / sid
        for name in ('maps', 'attention_maps', 'gifs', 'daam', 'latents'):
            (out / name).mkdir(parents=True, exist_ok=True)
        st = {
            'session': sid, 'out_dir': str(out), 'maps_dir': str(out / 'maps'),
            'attention_maps_dir': str(out / 'attention_maps'), 'gifs_dir': str(out / 'gifs'),
            'daam_dir': str(out / 'daam'), 'latents_dir': str(out / 'latents'),
            'records_path': str(out / 'records.jsonl'), 'map_mode': map_mode,
            'selected_blocks': _parse_ints(selected_blocks),
            'text_token_indices': _parse_ints(text_token_indices),
            'record_every_n_calls': int(record_every_n_calls),
            'snapshot_every_n_calls': int(snapshot_every_n_calls),
            'max_rank_tokens': int(max_rank_tokens), 'make_gif': bool(make_gif),
            'save_step_latents': bool(save_step_latents),
            'gif_duration_ms': int(gif_duration_ms), 'colormap': colormap,
            'call_index': 0, 'input_shapes': {}, 'gif_frames': {}, 'lock': threading.Lock(),
        }
        _RECORDER[sid] = st
        (out / 'session.json').write_text(json.dumps(
            {k: v for k, v in st.items() if k not in ('lock', 'gif_frames', 'input_shapes')},
            indent=2, default=list), encoding='utf-8')
        m = model.clone()
        old = m.model_options.get('model_function_wrapper')

        def wrapper(apply_model, args):
            c = args['c'].copy()
            with st['lock']:
                ci = st['call_index']
                st['call_index'] += 1
            sigma = float(args['timestep'].max().detach().cpu())
            to = c.get('transformer_options', {}).copy()
            to.update({'anima_diag_session': sid, 'anima_diag_call_index': ci, 'anima_diag_sigma': sigma})
            c['transformer_options'] = to
            inp = args['input']
            st['input_shapes'][ci] = list(inp.shape)
            latent_path = None
            if st['save_step_latents'] and ci % st['snapshot_every_n_calls'] == 0:
                latent_path = Path(st['latents_dir']) / f'call{ci:04d}.pt'
                torch.save(inp.detach().to(device='cpu', dtype=torch.float16), latent_path)
            rec = {
                'kind': 'model_call', 'time': time.time(), 'session': sid,
                'call_index': ci, 'sigma': sigma, 'block': -1,
                'input_shape': list(inp.shape), 'input_dtype': str(inp.dtype),
                'input_device': str(inp.device),
            }
            if latent_path is not None:
                rec['latent_path'] = str(latent_path)
            rec.update({f'input_{a}': b for a, b in _sample_stats(inp).items()})
            _append(st, rec)
            return old(apply_model, args | {'c': c}) if old is not None else apply_model(args['input'], args['timestep'], **c)

        m.set_model_unet_function_wrapper(wrapper)
        m.set_model_attn1_patch(_make_patch('attn1'))
        m.set_model_attn2_patch(_make_patch('attn2'))
        print(f'[AnimaDiagnostics] session={sid} output={out} mode={map_mode} tokens={sorted(st["text_token_indices"])}')
        return (m, str(out))


def _extract_qwen_ids(tokenized):
    seq = tokenized.get('qwen3_06b', tokenized) if isinstance(tokenized, dict) else tokenized
    if isinstance(seq, list) and seq and isinstance(seq[0], list):
        seq = seq[0]
    ids = []
    for x in seq:
        ids.append(int(x[0]) if isinstance(x, (tuple, list)) else int(x))
    while ids and ids[-1] == 151643:
        ids.pop()
    return ids


def _find_subsequence(seq, sub):
    if not sub or len(sub) > len(seq):
        return []
    out = []
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            out.append(list(range(i, i + len(sub))))
    return out


class AnimaDAAMKeywordSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'clip': ('CLIP',),
            'prompt': ('STRING', {'multiline': True, 'default': ''}),
            'keyword': ('STRING', {'default': 'arona'}),
        }}
    RETURN_TYPES = ('STRING', 'STRING')
    RETURN_NAMES = ('token_indices', 'token_report')
    FUNCTION = 'select'
    CATEGORY = 'diagnostics/anima'

    def select(self, clip, prompt, keyword):
        pids = _extract_qwen_ids(clip.tokenize(prompt))
        # BPE tokenization depends on the left boundary. In a prompt, a word is
        # commonly tokenized with its preceding space/newline rather than as a
        # standalone string. Try those boundary forms and use whichever sequence
        # actually occurs in the full prompt.
        variants = []
        for text in (keyword, ' ' + keyword, '\n' + keyword, '\t' + keyword):
            ids = _extract_qwen_ids(clip.tokenize(text))
            if ids and ids not in [x[1] for x in variants]:
                variants.append((text, ids))

        hits = []
        matched_variants = []
        for text, ids in variants:
            groups = _find_subsequence(pids, ids)
            if groups:
                matched_variants.append((text, ids, groups))
                for g in groups:
                    hits.extend(g)

        # Fallback for punctuation-adjacent BPE merges: tokenize exact small
        # contexts that occur in the raw prompt and search those token IDs.
        if not hits:
            for m in re.finditer(re.escape(keyword), prompt, flags=re.IGNORECASE):
                for left in (1, 2, 4):
                    snippet = prompt[max(0, m.start() - left):m.end()]
                    ids = _extract_qwen_ids(clip.tokenize(snippet))
                    groups = _find_subsequence(pids, ids)
                    if groups:
                        matched_variants.append((snippet, ids, groups))
                        for g in groups:
                            # Context form may contain punctuation as a separate
                            # token; keep only the trailing keyword-like tokens by
                            # intersecting with the leading-space form when possible.
                            hits.extend(g[-min(len(g), 4):])
                        break
                if hits:
                    break

        hits = sorted(set(hits))
        variant_report = '; '.join(
            f'{text!r}:{ids}->' + str(groups) for text, ids, groups in matched_variants
        ) or 'none'
        report = (
            f'keyword={keyword!r} matches={hits} prompt_token_count={len(pids)} '
            f'matched_variants={variant_report}'
        )
        print('[AnimaDAAMKeywordSelector]', report)
        return (','.join(map(str, hits)), report)


def _load_daam_arrays(diag_dir, tokens):
    root = Path(diag_dir) / 'attention_maps'
    rows = []
    pat = re.compile(r'crossattn_call(\d+)_block(\d+)_token(\d+)\.npy$')
    for p in root.glob('*.npy'):
        m = pat.match(p.name)
        if not m:
            continue
        ci, bi, ti = map(int, m.groups())
        if tokens and ti not in tokens:
            continue
        try:
            rows.append((ci, bi, ti, np.load(p)))
        except Exception:
            pass
    return rows


def _resize_arr(arr, h, w):
    t = torch.from_numpy(np.asarray(arr, np.float32))[None, None]
    return F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)[0, 0].numpy()


def _aggregate(arrs, mode):
    s = np.stack(arrs, 0)
    if mode == 'sum':
        return s.sum(0)
    if mode == 'max':
        return s.max(0)
    return s.mean(0)


def _image_tensor_to_uint8(image):
    x = image.detach().cpu().float().clamp(0, 1).numpy()
    return (x * 255).round().astype(np.uint8)


def _uint8_to_tensor(x):
    return torch.from_numpy(np.asarray(x, np.float32) / 255.0)


def _decode_latent_frame(vae, latent_path, batch_index, out_h, out_w):
    try:
        latent = torch.load(latent_path, map_location='cpu')
        decoded = vae.decode(latent)
        if decoded.ndim == 5:
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
        arr = _image_tensor_to_uint8(decoded)
        if arr.shape[0] == 0:
            return None
        frame = arr[min(batch_index, arr.shape[0] - 1)]
        if frame.shape[0] != out_h or frame.shape[1] != out_w:
            frame = np.asarray(Image.fromarray(frame).resize((out_w, out_h), Image.Resampling.LANCZOS))
        return frame
    except Exception as e:
        print('[AnimaDAAMOverlay] latent decode failed:', latent_path, repr(e))
        return None


class AnimaDAAMOverlay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'diagnostic_directory': ('STRING', {'default': '/content/anima_diagnostics/...'}),
                'text_token_indices': ('STRING', {'default': '0'}),
                'aggregation': (['mean', 'sum', 'max'], {'default': 'mean'}),
                'colormap': (['inferno', 'viridis', 'magma', 'plasma', 'gray'], {'default': 'inferno'}),
                'overlay_alpha': ('FLOAT', {'default': 0.45, 'min': 0.0, 'max': 1.0, 'step': 0.05}),
                'make_step_gif': ('BOOLEAN', {'default': True}),
                'gif_duration_ms': ('INT', {'default': 250, 'min': 50, 'max': 5000}),
            },
            'optional': {
                'vae': ('VAE',),
            },
        }
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'STRING')
    RETURN_NAMES = ('overlay', 'heatmap', 'daam_output_directory')
    FUNCTION = 'render'
    CATEGORY = 'diagnostics/anima'

    def render(self, image, diagnostic_directory, text_token_indices, aggregation,
               colormap, overlay_alpha, make_step_gif, gif_duration_ms, vae=None):
        tokens = _parse_ints(text_token_indices)
        if not tokens:
            raise RuntimeError('text_token_indices is empty. Check Anima DAAM Keyword Selector token_report.')
        rows = _load_daam_arrays(diagnostic_directory, tokens)
        if not rows:
            raise RuntimeError('No matching cross-attention .npy maps. Run diagnostics with cross_attention/both and matching token indices first.')
        base = _image_tensor_to_uint8(image)
        h, w = base.shape[1], base.shape[2]
        outdir = Path(diagnostic_directory) / 'daam'
        outdir.mkdir(parents=True, exist_ok=True)
        overlays, heats = [], []

        for b in range(base.shape[0]):
            all_arr = [_resize_arr(r[3], h, w) for r in rows]
            agg = _aggregate(all_arr, aggregation)
            heat = _colorize(agg, colormap)
            ov = ((1 - overlay_alpha) * base[b].astype(np.float32) + overlay_alpha * heat.astype(np.float32)).clip(0, 255).astype(np.uint8)
            Image.fromarray(heat).save(outdir / f'daam_heatmap_batch{b:02d}.png')
            Image.fromarray(ov).save(outdir / f'daam_overlay_batch{b:02d}.png')
            np.save(outdir / f'daam_aggregate_batch{b:02d}.npy', agg.astype(np.float32))
            overlays.append(ov)
            heats.append(heat)

            if make_step_gif:
                calls = sorted(set(r[0] for r in rows))
                frames = []
                for ci in calls:
                    rr = [_resize_arr(r[3], h, w) for r in rows if r[0] == ci]
                    if not rr:
                        continue
                    step_attn = _aggregate(rr, aggregation)
                    step_heat = _colorize(step_attn, colormap)

                    # If VAE is connected, decode the noisy/intermediate latent
                    # captured at this exact model call. The GIF background then
                    # evolves from noisy latent toward the final image instead of
                    # reusing the final image for every attention frame.
                    step_base = None
                    if vae is not None:
                        latent_path = Path(diagnostic_directory) / 'latents' / f'call{ci:04d}.pt'
                        if latent_path.exists():
                            step_base = _decode_latent_frame(vae, latent_path, b, h, w)
                    if step_base is None:
                        step_base = base[b]

                    frame = ((1 - overlay_alpha) * step_base.astype(np.float32) + overlay_alpha * step_heat.astype(np.float32)).clip(0, 255).astype(np.uint8)
                    frames.append(Image.fromarray(frame))

                if frames:
                    name = 'daam_denoising_steps' if vae is not None else 'daam_steps'
                    frames[0].save(
                        outdir / f'{name}_batch{b:02d}.gif', save_all=True,
                        append_images=frames[1:], duration=gif_duration_ms, loop=0,
                    )

        return (_uint8_to_tensor(np.stack(overlays)), _uint8_to_tensor(np.stack(heats)), str(outdir))


NODE_CLASS_MAPPINGS = {
    'AnimaInferenceDiagnostics': AnimaInferenceDiagnostics,
    'AnimaDAAMKeywordSelector': AnimaDAAMKeywordSelector,
    'AnimaDAAMOverlay': AnimaDAAMOverlay,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'AnimaInferenceDiagnostics': 'Anima Inference Diagnostics',
    'AnimaDAAMKeywordSelector': 'Anima DAAM Keyword Selector',
    'AnimaDAAMOverlay': 'Anima DAAM Overlay',
}
