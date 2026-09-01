from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from vit_lab_model_optim import SmallViT
from vit_lab_repr import (
    expected_calibration_error,
    load_checkpoint,
)
from vit_lab_train import evaluate


def flatten_tensors(tensors):
    return torch.cat(
        [tensor.reshape(-1) for tensor in tensors]
    )


def unflatten_like(vector, params):
    outputs = []
    offset = 0

    for parameter in params:
        size = parameter.numel()
        outputs.append(
            vector[offset:offset + size].view_as(parameter)
        )
        offset += size

    return outputs


def hessian_vector_product(model, x, y, vector):
    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]
    vector_parts = unflatten_like(vector, params)

    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)

    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
    )
    directional_derivative = sum(
        (grad * direction).sum()
        for grad, direction in zip(grads, vector_parts)
    )
    hv = torch.autograd.grad(
        directional_derivative,
        params,
    )

    return flatten_tensors(
        [item.detach() for item in hv]
    )


def full_batch_gradient(model, x, y):
    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    grads = torch.autograd.grad(loss, params)

    return flatten_tensors(
        [grad.detach() for grad in grads]
    )


def lanczos_hessian_ritz(model, x, y, device, steps=16):
    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]
    n_params = sum(p.numel() for p in params)

    q = torch.randn(n_params, device=device)
    q = q / q.norm()

    q_previous = torch.zeros_like(q)
    beta_previous = 0.0

    alphas = []
    betas = []
    basis_cpu = []

    for step in range(steps):
        basis_cpu.append(q.detach().float().cpu())

        z = hessian_vector_product(
            model,
            x,
            y,
            q,
        )

        if step > 0:
            z = z - beta_previous * q_previous

        alpha = torch.dot(q, z)
        z = z - alpha * q
        beta = z.norm()

        alphas.append(float(alpha))

        if step < steps - 1:
            betas.append(float(beta))

        if beta.item() < 1e-8:
            break

        q_previous = q
        q = z / beta
        beta_previous = float(beta)

    k = len(alphas)
    tridiagonal = np.diag(alphas)

    for index in range(k - 1):
        tridiagonal[index, index + 1] = betas[index]
        tridiagonal[index + 1, index] = betas[index]

    ritz_values, ritz_vectors = np.linalg.eigh(
        tridiagonal
    )

    gradient = full_batch_gradient(
        model,
        x,
        y,
    ).float().cpu()
    gradient = gradient / (gradient.norm() + 1e-12)

    basis = torch.stack(
        basis_cpu[:k],
        dim=1,
    )

    top_direction = basis @ torch.from_numpy(
        ritz_vectors[:, -1]
    ).float()
    min_direction = basis @ torch.from_numpy(
        ritz_vectors[:, 0]
    ).float()

    top_direction /= top_direction.norm() + 1e-12
    min_direction /= min_direction.norm() + 1e-12

    align_top = float(
        torch.dot(gradient, top_direction).abs()
    )
    align_min = float(
        torch.dot(gradient, min_direction).abs()
    )

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
    x, y = hessian_batch
    x = x.to(device)
    y = y.to(device)

    rows = []
    checkpoints = [("init", None)] + [
        (name, final_epoch)
        for name in optimizer_names
    ]

    for run_name, epoch in checkpoints:
        print("Hessian Lanczos:", run_name)

        if run_name == "init":
            model = SmallViT().to(device)
            model.load_state_dict(initial_state)
        else:
            model = load_checkpoint(
                run_name,
                epoch,
                ckpt_dir,
                device,
            )

        ritz_values, align_top, align_min = (
            lanczos_hessian_ritz(
                model,
                x,
                y,
                device,
                steps=steps,
            )
        )

        for rank, value in enumerate(
            np.sort(ritz_values),
            start=1,
        ):
            rows.append(
                {
                    "run": run_name,
                    "ritz_rank": rank,
                    "ritz_value": float(value),
                    "gradient_alignment_top": align_top,
                    "gradient_alignment_min": align_min,
                }
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    hessian_df = pd.DataFrame(rows)
    hessian_df.to_csv(
        csv_dir / "hessian_lanczos_ritz.csv",
        index=False,
    )

    summary_df = (
        hessian_df
        .groupby("run")
        .agg(
            min_ritz=("ritz_value", "min"),
            max_ritz=("ritz_value", "max"),
            grad_align_top=(
                "gradient_alignment_top",
                "first",
            ),
            grad_align_min=(
                "gradient_alignment_min",
                "first",
            ),
        )
        .reset_index()
    )
    summary_df["spectral_spread"] = (
        summary_df["max_ritz"]
        - summary_df["min_ritz"]
    )
    summary_df.to_csv(
        csv_dir / "hessian_summary.csv",
        index=False,
    )

    return hessian_df, summary_df


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()

    logits_list = []
    labels_list = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits_list.append(
            model(x).float().cpu()
        )
        labels_list.append(y.cpu())

    return (
        torch.cat(logits_list),
        torch.cat(labels_list),
    )


def run_function_space_comparison(
    optimizer_names,
    final_epoch,
    loader,
    ckpt_dir,
    csv_dir,
    device,
):
    final_logits = {}
    summary_rows = []
    labels = None

    for run_name in optimizer_names:
        model = load_checkpoint(
            run_name,
            final_epoch,
            ckpt_dir,
            device,
        )
        logits, labels = collect_logits(
            model,
            loader,
            device,
        )

        final_logits[run_name] = logits
        summary_rows.append(
            {
                "run": run_name,
                "ece": expected_calibration_error(
                    logits,
                    labels,
                ),
            }
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pair_rows = []

    for run_a, run_b in combinations(
        optimizer_names,
        2,
    ):
        logits_a = final_logits[run_a]
        logits_b = final_logits[run_b]

        pair_rows.append(
            {
                "run_a": run_a,
                "run_b": run_b,
                "logit_mse": float(
                    F.mse_loss(logits_a, logits_b)
                ),
                "prediction_disagreement": float(
                    (
                        logits_a.argmax(dim=1)
                        != logits_b.argmax(dim=1)
                    )
                    .float()
                    .mean()
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    pair_df = pd.DataFrame(pair_rows)

    summary_df.to_csv(
        csv_dir / "function_calibration.csv",
        index=False,
    )
    pair_df.to_csv(
        csv_dir / "function_pairwise.csv",
        index=False,
    )

    return summary_df, pair_df


def interpolate_state(state_a, state_b, alpha):
    output = {}

    for key in state_a:
        value_a = state_a[key]
        value_b = state_b[key]

        if torch.is_floating_point(value_a):
            output[key] = (
                (1 - alpha) * value_a
                + alpha * value_b
            )
        else:
            output[key] = value_a

    return output


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
            ckpt_dir / f"{name}_epoch{final_epoch}.pt",
            map_location="cpu",
        )
        for name in optimizer_names
    }

    alphas = np.linspace(0.0, 1.0, 11)
    rows = []
    barrier_rows = []

    for run_a, run_b in combinations(
        optimizer_names,
        2,
    ):
        print(
            "Mode connectivity:",
            run_a,
            "<->",
            run_b,
        )

        model = SmallViT().to(device)
        losses = []
        accuracies = []

        for alpha in alphas:
            state = interpolate_state(
                final_states[run_a],
                final_states[run_b],
                float(alpha),
            )
            model.load_state_dict(state)

            loss, accuracy = evaluate(
                model,
                loader,
                device,
                max_samples=max_samples,
            )

            losses.append(loss)
            accuracies.append(accuracy)

            rows.append(
                {
                    "run_a": run_a,
                    "run_b": run_b,
                    "alpha": float(alpha),
                    "val_loss": loss,
                    "val_accuracy": accuracy,
                }
            )

        midpoint = int(
            np.argmin(np.abs(alphas - 0.5))
        )

        barrier_rows.append(
            {
                "run_a": run_a,
                "run_b": run_b,
                "barrier_height": float(
                    max(losses)
                    - max(losses[0], losses[-1])
                ),
                "midpoint_loss": float(
                    losses[midpoint]
                ),
                "midpoint_accuracy": float(
                    accuracies[midpoint]
                ),
            }
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    connectivity_df = pd.DataFrame(rows)
    barrier_df = pd.DataFrame(barrier_rows)

    connectivity_df.to_csv(
        csv_dir / "mode_connectivity.csv",
        index=False,
    )
    barrier_df.to_csv(
        csv_dir / "mode_connectivity_barriers.csv",
        index=False,
    )

    return connectivity_df, barrier_df
