import os
import json
import time
import uuid
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_RECORDER = {}


def _parse_blocks(text):
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


def _save_spatial_rms(x, path):
    if x.ndim < 4:
        return None
    with torch.no_grad():
        z = x.detach()
        if z.ndim == 5:
            z = z[0]
            z = z.float().pow(2).mean(dim=-1).sqrt()
            z = z.mean(dim=0)
        elif z.ndim == 4:
            z = z[0].float().pow(2).mean(dim=-1).sqrt()
            if z.ndim == 3:
                z = z.mean(dim=0)
        else:
            return None
        arr = z.cpu().numpy()
        if arr.ndim != 2:
            return None
        lo, hi = np.nanpercentile(arr, [1.0, 99.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr) + 1e-8)
        norm = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
        Image.fromarray((norm * 255).astype(np.uint8), mode="L").save(path)
        np.save(str(path).replace(".png", ".npy"), arr.astype(np.float32))
        return list(arr.shape)


def _append_jsonl(state, rec):
    with state["lock"]:
        with open(state["records_path"], "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _record_attention(kind, q, k, v, extra_options):
    sid = extra_options.get("anima_diag_session")
    if not sid or sid not in _RECORDER:
        return

    state = _RECORDER[sid]
    call_index = int(extra_options.get("anima_diag_call_index", -1))
    if call_index < 0 or call_index % state["record_every_n_calls"] != 0:
        return

    block_index = int(extra_options.get("block_index", -1))
    sigma = extra_options.get("anima_diag_sigma", None)

    rec = {
        "time": time.time(),
        "session": sid,
        "call_index": call_index,
        "sigma": None if sigma is None else float(sigma),
        "block": block_index,
        "kind": kind,
        "q_shape": list(q.shape),
        "k_shape": list(k.shape),
        "v_shape": list(v.shape),
        "q_dtype": str(q.dtype),
        "k_dtype": str(k.dtype),
        "v_dtype": str(v.dtype),
        "q_device": str(q.device),
        "k_device": str(k.device),
        "v_device": str(v.device),
    }
    rec.update({f"q_{kk}": vv for kk, vv in _sample_stats(q).items()})
    rec.update({f"k_{kk}": vv for kk, vv in _sample_stats(k).items()})
    rec.update({f"v_{kk}": vv for kk, vv in _sample_stats(v).items()})

    selected = block_index in state["selected_blocks"]
    snapshot = selected and call_index % state["snapshot_every_n_calls"] == 0
    if snapshot:
        rec["q_effective_rank"] = _effective_rank(q, state["max_rank_tokens"])
        map_name = f"{kind}_call{call_index:04d}_sigma{(sigma if sigma is not None else -1):.6f}_block{block_index:02d}_q_rms.png"
        map_path = Path(state["maps_dir"]) / map_name
        rec["q_rms_map_shape"] = _save_spatial_rms(q, map_path)
        rec["q_rms_map"] = str(map_path)

    _append_jsonl(state, rec)


def _make_patch(kind):
    def patch(q, k, v, pe=None, attn_mask=None, extra_options=None):
        extra_options = extra_options or {}
        try:
            _record_attention(kind, q, k, v, extra_options)
        except Exception as e:
            sid = extra_options.get("anima_diag_session")
            if sid and sid in _RECORDER:
                _append_jsonl(_RECORDER[sid], {
                    "time": time.time(),
                    "session": sid,
                    "kind": "diagnostic_error",
                    "error": repr(e),
                    "block": int(extra_options.get("block_index", -1)),
                    "call_index": int(extra_options.get("anima_diag_call_index", -1)),
                })
        return {"q": q, "k": k, "v": v, "pe": pe}
    return patch


class AnimaInferenceDiagnostics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "output_root": ("STRING", {"default": "/content/anima_diagnostics"}),
                "selected_blocks": ("STRING", {"default": "0,6,12,18,24,27"}),
                "record_every_n_calls": ("INT", {"default": 1, "min": 1, "max": 100}),
                "snapshot_every_n_calls": ("INT", {"default": 5, "min": 1, "max": 100}),
                "max_rank_tokens": ("INT", {"default": 64, "min": 8, "max": 512}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "diagnostic_directory")
    FUNCTION = "patch"
    CATEGORY = "diagnostics/anima"

    def patch(self, model, output_root, selected_blocks, record_every_n_calls, snapshot_every_n_calls, max_rank_tokens):
        sid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        out_dir = Path(output_root) / sid
        maps_dir = out_dir / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "session": sid,
            "out_dir": str(out_dir),
            "maps_dir": str(maps_dir),
            "records_path": str(out_dir / "records.jsonl"),
            "selected_blocks": _parse_blocks(selected_blocks),
            "record_every_n_calls": int(record_every_n_calls),
            "snapshot_every_n_calls": int(snapshot_every_n_calls),
            "max_rank_tokens": int(max_rank_tokens),
            "call_index": 0,
            "lock": threading.Lock(),
        }
        _RECORDER[sid] = state

        with open(out_dir / "session.json", "w", encoding="utf-8") as f:
            json.dump({
                "session": sid,
                "selected_blocks": sorted(state["selected_blocks"]),
                "record_every_n_calls": state["record_every_n_calls"],
                "snapshot_every_n_calls": state["snapshot_every_n_calls"],
                "max_rank_tokens": state["max_rank_tokens"],
                "note": "attn1/attn2 patch inputs are recorded before q/k/v projection; maps are representation RMS maps, not DAAM attention probabilities."
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
            call_rec = {
                "time": time.time(),
                "session": sid,
                "call_index": call_index,
                "sigma": sigma,
                "block": -1,
                "kind": "model_call",
                "input_shape": list(inp.shape),
                "input_dtype": str(inp.dtype),
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
        return (m, str(out_dir))


NODE_CLASS_MAPPINGS = {"AnimaInferenceDiagnostics": AnimaInferenceDiagnostics}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaInferenceDiagnostics": "Anima Inference Diagnostics"}
