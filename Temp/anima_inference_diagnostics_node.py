import json
import math
import time
import uuid
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_RECORDER = {}


def _parse_ints(text):
    out = set()
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _sample_stats(x, max_values=262144):
    x = x.detach().reshape(-1)
    n = x.numel()
    if n == 0:
        return {"mean": None, "std": None, "rms": None, "zero_frac": None}
    if n > max_values:
        idx = torch.linspace(0, n - 1, max_values, device=x.device).long()
        x = x.index_select(0, idx)
    xf = x.float()
    return {
        "mean": float(xf.mean().item()),
        "std": float(xf.std(unbiased=False).item()),
        "rms": float(torch.sqrt(torch.mean(xf * xf)).item()),
        "zero_frac": float((xf == 0).float().mean().item()),
    }


def _effective_rank(x, max_tokens=64):
    z = x.detach().reshape(-1, x.shape[-1])
    if z.shape[0] < 2:
        return None
    if z.shape[0] > max_tokens:
        idx = torch.linspace(0, z.shape[0] - 1, max_tokens, device=z.device).long()
        z = z.index_select(0, idx)
    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(z)
        p = s * s
        denom = p.sum()
        if denom <= 0:
            return 0.0
        p = p / denom
        p = p[p > 0]
        return float(torch.exp(-(p * torch.log(p)).sum()).item())
    except Exception:
        return None


def _infer_grid(seq_len, input_shape):
    if seq_len <= 0:
        return None
    target_ratio = 1.0
    if input_shape and len(input_shape) >= 2:
        h, w = int(input_shape[-2]), int(input_shape[-1])
        if h > 0 and w > 0:
            target_ratio = h / w
    best = None
    best_score = float("inf")
    for h_tok in range(1, int(seq_len ** 0.5) + 1):
        if seq_len % h_tok:
            continue
        w_tok = seq_len // h_tok
        for hh, ww in ((h_tok, w_tok), (w_tok, h_tok)):
            score = abs(np.log((hh / ww + 1e-12) / (target_ratio + 1e-12)))
            if score < best_score:
                best_score = score
                best = (hh, ww)
    return best


def _grid_from_projected_q(q, input_shape):
    # projected q is [B, ..., heads, head_dim]. If spatial axes survive,
    # use them directly; otherwise infer HxW from the flattened sequence.
    spatial = list(q.shape[1:-2])
    if len(spatial) >= 2:
        # Cosmos image path often has T,H,W; average T later.
        return spatial
    seq_len = int(np.prod(spatial)) if spatial else 0
    if seq_len:
        grid = _infer_grid(seq_len, input_shape)
        return list(grid) if grid else None
    return None


def _array_to_png(arr, path, colormap="inferno"):
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.nanpercentile(arr, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr) + 1e-8)
    norm = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)

    if colormap == "gray":
        img = Image.fromarray((norm * 255).astype(np.uint8), mode="L").convert("RGB")
    else:
        try:
            import matplotlib
            cmap = matplotlib.colormaps.get_cmap(colormap)
            rgba = cmap(norm)
            img = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8), mode="RGB")
        except Exception:
            img = Image.fromarray((norm * 255).astype(np.uint8), mode="L").convert("RGB")
    img.save(path)
    return img


def _save_spatial_rms(x, path, input_shape=None, colormap="inferno"):
    with torch.no_grad():
        z = x.detach()
        if z.ndim == 3:  # [B,S,C]
            grid = _infer_grid(int(z.shape[1]), input_shape)
            if grid is None:
                return None
            token_rms = z[0].float().pow(2).mean(dim=-1).sqrt()
            arr = token_rms.reshape(*grid).cpu().numpy()
        elif z.ndim == 5:  # [B,T,H,W,C]
            arr = z[0].float().pow(2).mean(dim=-1).sqrt().mean(dim=0).cpu().numpy()
        elif z.ndim == 4:  # [B,H,W,C]
            arr = z[0].float().pow(2).mean(dim=-1).sqrt().cpu().numpy()
        else:
            return None
        if arr.ndim != 2:
            return None
        _array_to_png(arr, path, colormap)
        np.save(str(path).replace(".png", ".npy"), arr.astype(np.float32))
        return list(arr.shape)


def _append_jsonl(state, rec):
    with state["lock"]:
        with open(state["records_path"], "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _update_gif(state, key):
    if not state["make_gif"]:
        return
    frames = state["gif_frames"].get(key, [])
    if not frames:
        return
    images = []
    for p in frames:
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    if not images:
        return
    gif_path = Path(state["gifs_dir"]) / f"{key}.gif"
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=state["gif_duration_ms"],
        loop=0,
    )


def _record_preprojection(kind, q, k, v, extra_options):
    sid = extra_options.get("anima_diag_session")
    if not sid or sid not in _RECORDER:
        return
    state = _RECORDER[sid]
    call_index = int(extra_options.get("anima_diag_call_index", -1))
    if call_index < 0 or call_index % state["record_every_n_calls"] != 0:
        return
    block_index = int(extra_options.get("block_index", -1))
    sigma = extra_options.get("anima_diag_sigma")
    rec = {
        "time": time.time(), "session": sid, "call_index": call_index,
        "sigma": None if sigma is None else float(sigma), "block": block_index,
        "kind": kind, "q_shape": list(q.shape), "k_shape": list(k.shape),
        "v_shape": list(v.shape), "q_dtype": str(q.dtype), "k_dtype": str(k.dtype),
        "v_dtype": str(v.dtype), "q_device": str(q.device),
        "k_device": str(k.device), "v_device": str(v.device),
    }
    rec.update({f"q_{kk}": vv for kk, vv in _sample_stats(q).items()})
    rec.update({f"k_{kk}": vv for kk, vv in _sample_stats(k).items()})
    rec.update({f"v_{kk}": vv for kk, vv in _sample_stats(v).items()})

    selected = block_index in state["selected_blocks"]
    snapshot = selected and call_index % state["snapshot_every_n_calls"] == 0
    if snapshot:
        rec["q_effective_rank"] = _effective_rank(q, state["max_rank_tokens"])
        if state["map_mode"] in ("representation_rms", "both"):
            map_name = f"{kind}_call{call_index:04d}_block{block_index:02d}_q_rms.png"
            map_path = Path(state["maps_dir"]) / map_name
            input_shape = state["input_shapes"].get(call_index)
            rec["q_rms_map_shape"] = _save_spatial_rms(
                q, map_path, input_shape=input_shape, colormap=state["colormap"]
            )
            if rec["q_rms_map_shape"] is not None:
                rec["q_rms_map"] = str(map_path)
    _append_jsonl(state, rec)


def _make_patch(kind):
    def patch(q, k, v, pe=None, attn_mask=None, extra_options=None):
        extra_options = extra_options or {}
        try:
            _record_preprojection(kind, q, k, v, extra_options)
        except Exception as e:
            sid = extra_options.get("anima_diag_session")
            if sid and sid in _RECORDER:
                _append_jsonl(_RECORDER[sid], {
                    "time": time.time(), "session": sid, "kind": "diagnostic_error",
                    "error": repr(e), "block": int(extra_options.get("block_index", -1)),
                    "call_index": int(extra_options.get("anima_diag_call_index", -1)),
                })
        return {"q": q, "k": k, "v": v, "pe": pe}
    return patch


def _cross_attention_selected_probabilities(q, k, token_indices, chunk=64):
    """Exact softmax probabilities for selected text keys without materializing QxK all at once.

    q,k are the actual projected+normalized tensors returned by Cosmos Predict2
    compute_qkv, with shape [B,...,H,D]. Query spatial axes are flattened only
    for this diagnostic calculation. The softmax denominator is accumulated over
    every key token, so selected-token values are true attention probabilities.
    """
    qf = q.detach().reshape(q.shape[0], -1, q.shape[-2], q.shape[-1]).float()
    kf = k.detach().reshape(k.shape[0], -1, k.shape[-2], k.shape[-1]).float()
    key_count = kf.shape[1]
    valid = [i for i in sorted(token_indices) if 0 <= i < key_count]
    if not valid:
        return {}, key_count

    scale = 1.0 / math.sqrt(qf.shape[-1])
    # Streaming logsumexp over the whole text-token axis.
    running_max = torch.full(qf.shape[:3], -float("inf"), device=qf.device, dtype=torch.float32)
    running_sum = torch.zeros_like(running_max)
    for start in range(0, key_count, chunk):
        kc = kf[:, start:start + chunk]
        logits = torch.einsum("bqhd,bkhd->bqhk", qf, kc) * scale
        cmax = logits.amax(dim=-1)
        new_max = torch.maximum(running_max, cmax)
        running_sum = running_sum * torch.exp(running_max - new_max) + torch.exp(logits - new_max[..., None]).sum(dim=-1)
        running_max = new_max
    log_denom = running_max + torch.log(running_sum.clamp_min(1e-30))

    ks = kf[:, valid]
    selected_logits = torch.einsum("bqhd,bkhd->bqhk", qf, ks) * scale
    probs = torch.exp(selected_logits - log_denom[..., None])
    # Average batch and heads -> [Q, selected_tokens]
    probs = probs.mean(dim=0).mean(dim=1)
    return {tok: probs[:, j].detach().cpu().numpy() for j, tok in enumerate(valid)}, key_count


def _reshape_query_map(values, q, input_shape):
    spatial = list(q.shape[1:-2])
    arr = np.asarray(values, dtype=np.float32)
    if len(spatial) >= 2:
        shaped = arr.reshape(spatial)
        # Image models may retain T,H,W. Average temporal axis to produce HxW.
        while shaped.ndim > 2:
            shaped = shaped.mean(axis=0)
        return shaped
    grid = _infer_grid(arr.size, input_shape)
    if grid is None:
        return None
    return arr.reshape(*grid)


def _record_projected_cross_attention(q, k, transformer_options):
    sid = transformer_options.get("anima_diag_session")
    if not sid or sid not in _RECORDER:
        return
    state = _RECORDER[sid]
    if state["map_mode"] not in ("cross_attention", "both"):
        return
    call_index = int(transformer_options.get("anima_diag_call_index", -1))
    block_index = int(transformer_options.get("block_index", -1))
    if call_index < 0 or block_index not in state["selected_blocks"]:
        return
    if call_index % state["snapshot_every_n_calls"] != 0:
        return

    probs, key_count = _cross_attention_selected_probabilities(q, k, state["text_token_indices"])
    input_shape = state["input_shapes"].get(call_index)
    sigma = transformer_options.get("anima_diag_sigma")

    for token_index, values in probs.items():
        arr = _reshape_query_map(values, q, input_shape)
        if arr is None or arr.ndim != 2:
            continue
        name = f"crossattn_call{call_index:04d}_block{block_index:02d}_token{token_index:03d}.png"
        path = Path(state["attention_maps_dir"]) / name
        _array_to_png(arr, path, state["colormap"])
        np.save(str(path).replace(".png", ".npy"), arr.astype(np.float32))

        key = f"crossattn_block{block_index:02d}_token{token_index:03d}"
        with state["lock"]:
            state["gif_frames"].setdefault(key, []).append(str(path))
        _update_gif(state, key)
        _append_jsonl(state, {
            "time": time.time(), "session": sid, "kind": "cross_attention_map",
            "call_index": call_index, "sigma": None if sigma is None else float(sigma),
            "block": block_index, "text_token_index": token_index,
            "text_key_count": int(key_count), "map_shape": list(arr.shape),
            "map_path": str(path), "gif_key": key,
        })


def _install_projected_qk_hook():
    """Wrap Cosmos Predict2 Attention.compute_qkv once.

    ComfyUI's attn1/attn2 patch API runs before Q/K/V projection, so it cannot
    expose the actual attention logits. This wrapper observes the tensors after
    the model's own projection, norm and RoPE code has run. It is a global no-op
    unless transformer_options contains an active anima diagnostic session.
    """
    try:
        from comfy.ldm.cosmos.predict2 import Attention as CosmosAttention
    except Exception as e:
        print(f"[AnimaDiagnostics] projected Q/K hook unavailable: {e!r}")
        return

    if getattr(CosmosAttention, "_anima_diag_hook_installed", False):
        return
    original = CosmosAttention.compute_qkv

    def wrapped(self, x, context=None, rope_emb=None, transformer_options={}):
        q, k, v = original(self, x, context=context, rope_emb=rope_emb, transformer_options=transformer_options)
        # Cross-attention only: this is the DAAM-like image-query/text-key path.
        if not self.is_selfattn:
            try:
                _record_projected_cross_attention(q, k, transformer_options or {})
            except Exception as e:
                sid = (transformer_options or {}).get("anima_diag_session")
                if sid and sid in _RECORDER:
                    _append_jsonl(_RECORDER[sid], {
                        "time": time.time(), "session": sid, "kind": "diagnostic_error",
                        "error": f"projected_cross_attention: {e!r}",
                        "block": int((transformer_options or {}).get("block_index", -1)),
                        "call_index": int((transformer_options or {}).get("anima_diag_call_index", -1)),
                    })
        return q, k, v

    CosmosAttention.compute_qkv = wrapped
    CosmosAttention._anima_diag_hook_installed = True
    CosmosAttention._anima_diag_original_compute_qkv = original
    print("[AnimaDiagnostics] installed projected Q/K hook")


_install_projected_qk_hook()


class AnimaInferenceDiagnostics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "map_mode": (["representation_rms", "cross_attention", "both"], {"default": "both"}),
                "selected_blocks": ("STRING", {"default": "0,6,12,18,24,27"}),
                "text_token_indices": ("STRING", {"default": "0"}),
                "snapshot_every_n_calls": ("INT", {"default": 1, "min": 1, "max": 100}),
                "make_gif": ("BOOLEAN", {"default": True}),
                "gif_duration_ms": ("INT", {"default": 250, "min": 50, "max": 5000}),
                "colormap": (["inferno", "viridis", "magma", "plasma", "gray"], {"default": "inferno"}),
                "output_root": ("STRING", {"default": "/content/anima_diagnostics"}),
                "record_every_n_calls": ("INT", {"default": 1, "min": 1, "max": 100}),
                "max_rank_tokens": ("INT", {"default": 64, "min": 8, "max": 512}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "diagnostic_directory")
    FUNCTION = "patch"
    CATEGORY = "diagnostics/anima"

    def patch(self, model, map_mode, selected_blocks, text_token_indices,
              snapshot_every_n_calls, make_gif, gif_duration_ms, colormap,
              output_root, record_every_n_calls, max_rank_tokens):
        sid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        out_dir = Path(output_root) / sid
        maps_dir = out_dir / "maps"
        attention_maps_dir = out_dir / "attention_maps"
        gifs_dir = out_dir / "gifs"
        for d in (maps_dir, attention_maps_dir, gifs_dir):
            d.mkdir(parents=True, exist_ok=True)

        state = {
            "session": sid, "out_dir": str(out_dir), "maps_dir": str(maps_dir),
            "attention_maps_dir": str(attention_maps_dir), "gifs_dir": str(gifs_dir),
            "records_path": str(out_dir / "records.jsonl"),
            "map_mode": map_mode, "selected_blocks": _parse_ints(selected_blocks),
            "text_token_indices": _parse_ints(text_token_indices),
            "record_every_n_calls": int(record_every_n_calls),
            "snapshot_every_n_calls": int(snapshot_every_n_calls),
            "max_rank_tokens": int(max_rank_tokens), "make_gif": bool(make_gif),
            "gif_duration_ms": int(gif_duration_ms), "colormap": colormap,
            "call_index": 0, "input_shapes": {}, "gif_frames": {},
            "lock": threading.Lock(),
        }
        _RECORDER[sid] = state

        with open(out_dir / "session.json", "w", encoding="utf-8") as f:
            json.dump({
                "session": sid, "map_mode": map_mode,
                "selected_blocks": sorted(state["selected_blocks"]),
                "text_token_indices": sorted(state["text_token_indices"]),
                "record_every_n_calls": state["record_every_n_calls"],
                "snapshot_every_n_calls": state["snapshot_every_n_calls"],
                "make_gif": state["make_gif"], "gif_duration_ms": state["gif_duration_ms"],
                "colormap": colormap,
                "note": "representation_rms uses pre-projection representations. cross_attention hooks Cosmos Predict2 compute_qkv after projection/norm/rope and computes exact selected text-token softmax probabilities over all text keys. GIFs show denoising-call evolution per block/token. Token selection is by text-encoder sequence index, not word string."
            }, f, indent=2)

        m = model.clone()
        old_wrapper = m.model_options.get("model_function_wrapper")

        def model_function_wrapper(apply_model, args):
            c = args["c"].copy()
            with state["lock"]:
                call_index = state["call_index"]
                state["call_index"] += 1
            timestep = args["timestep"]
            sigma = float(timestep.max().detach().cpu().item())
            transformer_options = c.get("transformer_options", {}).copy()
            transformer_options["anima_diag_session"] = sid
            transformer_options["anima_diag_call_index"] = call_index
            transformer_options["anima_diag_sigma"] = sigma
            c["transformer_options"] = transformer_options

            inp = args["input"]
            with state["lock"]:
                state["input_shapes"][call_index] = list(inp.shape)
            call_rec = {
                "time": time.time(), "session": sid, "call_index": call_index,
                "sigma": sigma, "block": -1, "kind": "model_call",
                "input_shape": list(inp.shape), "input_dtype": str(inp.dtype),
                "input_device": str(inp.device),
            }
            call_rec.update({f"input_{kk}": vv for kk, vv in _sample_stats(inp).items()})
            _append_jsonl(state, call_rec)
            if old_wrapper is not None:
                return old_wrapper(apply_model, args | {"c": c})
            return apply_model(args["input"], args["timestep"], **c)

        m.set_model_unet_function_wrapper(model_function_wrapper)
        m.set_model_attn1_patch(_make_patch("attn1"))
        m.set_model_attn2_patch(_make_patch("attn2"))

        print(f"[AnimaDiagnostics] session={sid}")
        print(f"[AnimaDiagnostics] output={out_dir}")
        print(f"[AnimaDiagnostics] mode={map_mode} tokens={sorted(state['text_token_indices'])} gif={make_gif}")
        return (m, str(out_dir))


NODE_CLASS_MAPPINGS = {"AnimaInferenceDiagnostics": AnimaInferenceDiagnostics}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaInferenceDiagnostics": "Anima Inference Diagnostics"}
