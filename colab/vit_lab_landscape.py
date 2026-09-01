from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from vit_lab_model_optim import SmallViT
from vit_lab_train import evaluate
from vit_lab_repr import load_checkpoint, extract_features_and_logits, expected_calibration_error


def flatten_tensors(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def unflatten_like(vector, params):
    outputs = []
    offset = 0
    for p in params:
        n = p.numel()
        outputs.append(vector[offset:offset + n].view_as(p))
        offset += n
    return outputs


def hessian_vector_product(model, x, y, vector):
    params = [p for p in model.parameters() if p.requires_grad]
    vec_parts = unflatten_like(vector, params)
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    dot = sum((g * v).sum() for g, v in zip(grads, vec_parts))
    hv = torch.autograd.grad(dot, params)
    return flatten_tensors([h.detach() for h in hv])


def full_batch_gradient(model, x, y):
    params = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    grads = torch.autograd.grad(loss, params)
    return flatten_tensors([g.detach() for g in grads])


def lanczos_hessian_ritz(model, x, y, device, steps=16):
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    q = torch.randn(n_params, device=device)
    q = q / q.norm()
    q_prev = torch.zeros_like(q)
    beta_prev = 0.0
    alphas, betas, basis_cpu = [], [], []

    for j in range(steps):
        basis_cpu.append(q.detach().float().cpu())
        z = hessian_vector_product(model, x, y, q)
        if j > 0:
            z = z - beta_prev * q_prev
        alpha = torch.dot(q, z)
        z = z - alpha * q
        beta = z.norm()
        alphas.append(float(alpha))
        if j < steps - 1:
            betas.append(float(beta))
        if beta.item() < 1e-8:
            break
        q_prev, q = q, z / beta
        beta_prev = float(beta)

    k = len(alphas)
    T = np.diag(alphas)
    for i in range(k - 1):
        T[i, i + 1] = betas[i]
        T[i + 1, i] = betas[i]
    ritz_values, ritz_vectors = np.linalg.eigh(T)

    gradient_cpu = full_batch_gradient(model, x, y).float().cpu()
    gradient_cpu = gradient_cpu / (gradient_cpu.norm() + 1e-12)
    Q = torch.stack(basis_cpu[:k], dim=1)
    top_direction = Q @ torch.from_numpy(ritz_vectors[:, -1]).float()
    min_direction = Q @ torch.from_numpy(ritz_vectors[:, 0]).float()
    top_direction /= top_direction.norm() + 1e-12
    min_direction /= min_direction.norm() + 1e-12
    align_top = float(torch.dot(gradient_cpu, top_direction).abs())
    align_min = float(torch.dot(gradient_cpu, min_direction).abs())
    return ritz_values, align_top, align_min


def run_hessian_diagnostics(
    optimizer_names,
    final_epoch,
    initial_state,
    hessian_batch,
    ckpt_dir,
    csv_dir,
    device,
    steps=16,
):
    hx, hy = hessian_batch
    hx, hy = hx.to(device), hy.to(device)
    rows = []
    checkpoints = [("init", None)] + [(name, final_epoch) for name in optimizer_names]

    for run_name, epoch in checkpoints:
        print("Hessian Lanczos:", run_name)
        if run_name == "init":
            model = SmallViT().to(device)
            model.load_state_dict(initial_state)
        else:
            model = load_checkpoint(run_name, epoch, ckpt_dir, device)
        ritz, align_top, align_min = lanczos_hessian_ritz(
            model, hx, hy, device, steps=steps
        )
        for rank, value in enumerate(np.sort(ritz), start=1):
            rows.append({
                "run": run_name,
                "ritz_rank": rank,
                "ritz_value": float(value),
                "gradient_alignment_top": align_top,
                "gradient_alignment_min": align_min,
            })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(csv_dir / "hessian_lanczos_ritz.csv", index=False)
    summary = (
        df.groupby("run")
        .agg(
            min_ritz=("ritz_value", "min"),
            max_ritz=("ritz_value", "max"),
            grad_align_top=("gradient_alignment_top", "first"),
            grad_align_min=("gradient_alignment_min", "first"),
        )
        .reset_index()
    )
    summary["spectral_spread"] = summary["max_ritz"] - summary["min_ritz"]
    summary.to_csv(csv_dir / "hessian_summary.csv", index=False)
    return df, summary


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits_list, labels_list = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits_list.append(model(x).float().cpu())
        labels_list.append(y.cpu())
    return torch.cat(logits_list), torch.cat(labels_list)


def run_function_space_comparison(
    optimizer_names,
    final_epoch,
    loader,
    ckpt_dir,
    csv_dir,
    device,
):
    final_logits = {}
    rows = []
    labels = None
    for run_name in optimizer_names:
        model = load_checkpoint(run_name, final_epoch, ckpt_dir, device)
        logits, labels = collect_logits(model, loader, device)
        final_logits[run_name] = logits
        rows.append({
            "run": run_name,
            "ece": expected_calibration_error(logits, labels),
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pair_rows = []
    for a, b in combinations(optimizer_names, 2):
        la, lb = final_logits[a], final_logits[b]
        pair_rows.append({
            "run_a": a,
            "run_b": b,
            "logit_mse": float(F.mse_loss(la, lb)),
            "prediction_disagreement": float(
                (la.argmax(dim=1) != lb.argmax(dim=1)).float().mean()
            ),
        })

    df = pd.DataFrame(rows)
    pair_df = pd.DataFrame(pair_rows)
    df.to_csv(csv_dir / "function_calibration.csv", index=False)
    pair_df.to_csv(csv_dir / "function_pairwise.csv", index=False)
    return df, pair_df


def interpolate_state(state_a, state_b, alpha):
    out = {}
    for key in state_a:
        a, b = state_a[key], state_b[key]
        out[key] = (1 - alpha) * a + alpha * b if torch.is_floating_point(a) else a
    return out


def run_mode_connectivity(
    optimizer_names,
    final_epoch,
    loader,
    ckpt_dir,
    csv_dir,
    device,
    max_samples=1000,
):
    final_states = {
        name: torch.load(
            ckpt_dir / f"{name}_epoch{final_epoch}.pt", map_location="cpu"
        )
        for name in optimizer_names
    }
    alphas = np.linspace(0.0, 1.0, 11)
    rows, barrier_rows = [], []

    for a, b in combinations(optimizer_names, 2):
        print("Mode connectivity:", a, "<->", b)
        model = SmallViT().to(device)
        losses, accs = [], []
        for alpha in alphas:
            model.load_state_dict(interpolate_state(final_states[a], final_states[b], float(alpha)))
            loss, acc = evaluate(model, loader, device, max_samples=max_samples)
            losses.append(loss)
            accs.append(acc)
            rows.append({
                "run_a": a,
                "run_b": b,
                "alpha": float(alpha),
                "val_loss": loss,
                "val_accuracy": acc,
            })
        middle = int(np.argmin(np.abs(alphas - 0.5)))
        barrier_rows.append({
            "run_a": a,
            "run_b": b,
            "barrier_height": float(max(losses) - max(losses[0], losses[-1])),
            "midpoint_loss": float(losses[middle]),
            "midpoint_accuracy": float(accs[middle]),
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    barrier_df = pd.DataFrame(barrier_rows)
    df.to_csv(csv_dir / "mode_connectivity.csv", index=False)
    barrier_df.to_csv(csv_dir / "mode_connectivity_barriers.csv", index=False)
    return df, barrier_df
