import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def velocity_diagnostics(model, loader, device, t_values, max_batches=8):
    rows = []
    model.eval()

    for t_value in t_values:
        mse_values = []
        norm_pred = []
        norm_target = []
        cosine_values = []

        for batch_index, (x1, y) in enumerate(loader):
            if batch_index >= max_batches:
                break

            x1 = x1.to(device)
            y = y.to(device)
            x0 = torch.randn_like(x1)
            t = torch.full((x1.shape[0],), float(t_value), device=device)
            tv = t[:, None, None, None]
            xt = (1 - tv) * x0 + tv * x1
            target = x1 - x0

            pred = model(xt, t, y)

            flat_pred = pred.flatten(1)
            flat_target = target.flatten(1)

            mse_values.extend(
                ((flat_pred - flat_target) ** 2).mean(dim=1).cpu().tolist()
            )
            norm_pred.extend(flat_pred.norm(dim=1).cpu().tolist())
            norm_target.extend(flat_target.norm(dim=1).cpu().tolist())
            cosine_values.extend(
                F.cosine_similarity(flat_pred, flat_target, dim=1).cpu().tolist()
            )

        rows.append({
            "t": float(t_value),
            "velocity_mse": float(np.mean(mse_values)),
            "pred_velocity_norm": float(np.mean(norm_pred)),
            "target_velocity_norm": float(np.mean(norm_target)),
            "velocity_cosine": float(np.mean(cosine_values)),
            "velocity_mse_std": float(np.std(mse_values)),
        })

    return pd.DataFrame(rows)


def hutchinson_divergence(model, x, t, y, probes=1):
    """
    Estimate div_x v(x,t,y) = tr(dv/dx) with Rademacher probes.
    Returns one divergence estimate per sample.
    """
    x = x.detach().requires_grad_(True)
    v = model(x, t, y)

    estimates = []
    for _ in range(probes):
        eps = torch.empty_like(v).bernoulli_(0.5).mul_(2).sub_(1)
        scalar = (v * eps).sum()
        grad = torch.autograd.grad(
            scalar,
            x,
            create_graph=False,
            retain_graph=True,
        )[0]
        estimates.append((grad * eps).flatten(1).sum(dim=1))

    return torch.stack(estimates).mean(dim=0).detach()


def divergence_diagnostics(
    model,
    loader,
    device,
    t_values,
    max_batches=2,
    probes=1,
):
    rows = []
    model.eval()

    for t_value in t_values:
        values = []

        for batch_index, (x1, y) in enumerate(loader):
            if batch_index >= max_batches:
                break

            x1 = x1.to(device)
            y = y.to(device)
            x0 = torch.randn_like(x1)
            t = torch.full((x1.shape[0],), float(t_value), device=device)
            tv = t[:, None, None, None]
            xt = (1 - tv) * x0 + tv * x1

            div = hutchinson_divergence(
                model,
                xt,
                t,
                y,
                probes=probes,
            )
            values.extend(div.cpu().tolist())

        rows.append({
            "t": float(t_value),
            "divergence_mean": float(np.mean(values)),
            "divergence_abs_mean": float(np.mean(np.abs(values))),
            "divergence_std": float(np.std(values)),
        })

    return pd.DataFrame(rows)


@torch.no_grad()
def collect_layer_features(
    model,
    loader,
    device,
    t_value,
    max_samples=4000,
):
    model.eval()
    feature_chunks = {}
    labels = []
    seen = 0

    for x1, y in loader:
        x1 = x1.to(device)
        y = y.to(device)
        x0 = torch.randn_like(x1)
        t = torch.full((x1.shape[0],), float(t_value), device=device)
        tv = t[:, None, None, None]
        xt = (1 - tv) * x0 + tv * x1

        _, aux = model(
            xt,
            t,
            y,
            return_features=True,
        )

        for layer_name, feature in aux["features"].items():
            feature_chunks.setdefault(layer_name, []).append(
                feature.float().cpu().numpy()
            )
        labels.append(y.cpu().numpy())

        seen += x1.shape[0]
        if seen >= max_samples:
            break

    features = {
        layer_name: np.concatenate(chunks, axis=0)[:max_samples]
        for layer_name, chunks in feature_chunks.items()
    }
    labels = np.concatenate(labels, axis=0)[:max_samples]
    return features, labels


def _fit_probe(train_x, train_y, val_x, val_y):
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)

    classifier = RidgeClassifier(alpha=1.0)
    classifier.fit(train_x, train_y)
    return float(classifier.score(val_x, val_y))


def _subspace_energy_alignment(train_x, train_y, val_x, val_y, rank=8):
    classes = np.unique(train_y)
    class_bases = {}
    class_means = {}

    for class_id in classes:
        x_c = train_x[train_y == class_id]
        mean = x_c.mean(axis=0, keepdims=True)
        x_c = x_c - mean
        n_components = min(rank, x_c.shape[0] - 1, x_c.shape[1])
        pca = PCA(n_components=n_components, svd_solver="randomized")
        pca.fit(x_c)
        class_bases[class_id] = pca.components_.T
        class_means[class_id] = mean.reshape(-1)

    scores = []
    for x, class_id in zip(val_x, val_y):
        basis = class_bases[class_id]
        centered = x - class_means[class_id]
        numerator = np.linalg.norm(basis.T @ centered) ** 2
        denominator = np.linalg.norm(centered) ** 2 + 1e-12
        scores.append(numerator / denominator)

    return float(np.mean(scores))


def representation_grid(
    model,
    train_loader,
    val_loader,
    device,
    t_values,
    max_train_samples=3000,
    max_val_samples=2000,
    subspace_rank=8,
):
    probe_rows = []
    alignment_rows = []

    for t_value in t_values:
        train_features, train_labels = collect_layer_features(
            model,
            train_loader,
            device,
            t_value,
            max_samples=max_train_samples,
        )
        val_features, val_labels = collect_layer_features(
            model,
            val_loader,
            device,
            t_value,
            max_samples=max_val_samples,
        )

        for layer_name in train_features:
            probe = _fit_probe(
                train_features[layer_name],
                train_labels,
                val_features[layer_name],
                val_labels,
            )
            alignment = _subspace_energy_alignment(
                train_features[layer_name],
                train_labels,
                val_features[layer_name],
                val_labels,
                rank=subspace_rank,
            )

            probe_rows.append({
                "t": float(t_value),
                "layer": layer_name,
                "probe_accuracy": probe,
            })
            alignment_rows.append({
                "t": float(t_value),
                "layer": layer_name,
                "class_subspace_energy": alignment,
            })

    return pd.DataFrame(probe_rows), pd.DataFrame(alignment_rows)


@torch.no_grad()
def attention_locality_grid(
    model,
    loader,
    device,
    t_values,
    max_batches=2,
):
    rows = []
    model.eval()

    grid = model.grid
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(grid),
            torch.arange(grid),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2).float()
    pairwise_distance = torch.cdist(coords, coords)

    for t_value in t_values:
        layer_accumulator = {}

        for batch_index, (x1, y) in enumerate(loader):
            if batch_index >= max_batches:
                break

            x1 = x1.to(device)
            y = y.to(device)
            x0 = torch.randn_like(x1)
            t = torch.full((x1.shape[0],), float(t_value), device=device)
            tv = t[:, None, None, None]
            xt = (1 - tv) * x0 + tv * x1

            _, aux = model(
                xt,
                t,
                y,
                return_attention=True,
            )

            for layer_name, attn in aux["attention"].items():
                a = attn.float().cpu()
                entropy = -(a.clamp_min(1e-12) * a.clamp_min(1e-12).log()).sum(dim=-1)
                entropy = entropy.mean().item()

                distance = (
                    a
                    * pairwise_distance[None, None, :, :]
                ).sum(dim=-1).mean().item()

                effective_tokens = math.exp(entropy)
                layer_accumulator.setdefault(layer_name, []).append(
                    (entropy, distance, effective_tokens)
                )

        for layer_name, values in layer_accumulator.items():
            arr = np.asarray(values)
            rows.append({
                "t": float(t_value),
                "layer": layer_name,
                "attention_entropy": float(arr[:, 0].mean()),
                "mean_attention_distance": float(arr[:, 1].mean()),
                "effective_attended_tokens": float(arr[:, 2].mean()),
            })

    return pd.DataFrame(rows)


def _orthonormalize(columns):
    q, _ = torch.linalg.qr(columns, mode="reduced")
    return q


def approximate_jacobian_singular_values(
    model,
    x,
    t,
    y,
    rank=4,
    iterations=4,
):
    """
    Randomized subspace iteration on J^T J using JVP/VJP.
    Designed for a single FashionMNIST sample.
    """
    model.eval()
    x = x.detach().requires_grad_(True)

    def f(inp):
        return model(inp, t, y)

    dimension = x.numel()
    vectors = torch.randn(dimension, rank, device=x.device)
    vectors = _orthonormalize(vectors)

    for _ in range(iterations):
        output_columns = []

        for j in range(rank):
            direction = vectors[:, j].view_as(x)
            _, jv = torch.autograd.functional.jvp(
                f,
                (x,),
                (direction,),
                create_graph=False,
            )
            output_columns.append(jv.reshape(-1))

        outputs = torch.stack(output_columns, dim=1)
        outputs = _orthonormalize(outputs)

        input_columns = []
        fx = f(x)

        for j in range(rank):
            u = outputs[:, j].view_as(fx)
            scalar = (fx * u).sum()
            jt_u = torch.autograd.grad(
                scalar,
                x,
                retain_graph=True,
            )[0]
            input_columns.append(jt_u.reshape(-1))

        vectors = _orthonormalize(
            torch.stack(input_columns, dim=1)
        )

    singular_values = []
    for j in range(rank):
        direction = vectors[:, j].view_as(x)
        _, jv = torch.autograd.functional.jvp(
            f,
            (x,),
            (direction,),
            create_graph=False,
        )
        singular_values.append(float(jv.norm().detach()))

    return np.sort(np.asarray(singular_values))[::-1]


def jacobian_spectrum_over_time(
    model,
    sample,
    label,
    device,
    t_values,
    rank=4,
    iterations=4,
):
    rows = []
    x1 = sample.to(device).unsqueeze(0)
    y = torch.tensor([int(label)], device=device)
    x0 = torch.randn_like(x1)

    for t_value in t_values:
        t = torch.tensor([float(t_value)], device=device)
        tv = t[:, None, None, None]
        xt = (1 - tv) * x0 + tv * x1

        values = approximate_jacobian_singular_values(
            model,
            xt,
            t,
            y,
            rank=rank,
            iterations=iterations,
        )

        for index, value in enumerate(values, start=1):
            rows.append({
                "t": float(t_value),
                "rank": index,
                "singular_value": float(value),
            })

    return pd.DataFrame(rows)


@dataclass
class SampleResult:
    samples: torch.Tensor
    trajectories: torch.Tensor
    velocities: torch.Tensor
    nfe: int


@torch.no_grad()
def sample_flow(
    model,
    labels,
    device,
    solver="euler",
    nfe=16,
):
    model.eval()
    labels = labels.to(device)
    x = torch.randn(labels.shape[0], 1, 28, 28, device=device)

    cost = {"euler": 1, "heun": 2, "rk4": 4}[solver]
    if nfe % cost != 0:
        raise ValueError(f"{solver} requires NFE divisible by {cost}")

    steps = nfe // cost
    dt = 1.0 / steps

    trajectory = [x.detach().cpu()]
    velocity_history = []

    def field(state, time_value):
        t = torch.full(
            (state.shape[0],),
            float(time_value),
            device=device,
        )
        return model(state, t, labels)

    for step in range(steps):
        t0 = step / steps

        if solver == "euler":
            k1 = field(x, t0)
            x = x + dt * k1
            used_velocity = k1

        elif solver == "heun":
            k1 = field(x, t0)
            predictor = x + dt * k1
            k2 = field(predictor, min(1.0, t0 + dt))
            x = x + 0.5 * dt * (k1 + k2)
            used_velocity = 0.5 * (k1 + k2)

        elif solver == "rk4":
            k1 = field(x, t0)
            k2 = field(x + 0.5 * dt * k1, t0 + 0.5 * dt)
            k3 = field(x + 0.5 * dt * k2, t0 + 0.5 * dt)
            k4 = field(x + dt * k3, min(1.0, t0 + dt))
            used_velocity = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            x = x + dt * used_velocity
        else:
            raise ValueError(solver)

        velocity_history.append(used_velocity.detach().cpu())
        trajectory.append(x.detach().cpu())

    return SampleResult(
        samples=x.detach().cpu().clamp(-1, 1),
        trajectories=torch.stack(trajectory, dim=1),
        velocities=torch.stack(velocity_history, dim=1),
        nfe=nfe,
    )


def trajectory_metrics(sample_result):
    trajectory = sample_result.trajectories.float()
    deltas = trajectory[:, 1:] - trajectory[:, :-1]

    segment_lengths = deltas.flatten(2).norm(dim=2)
    path_length = segment_lengths.sum(dim=1)

    displacement = (
        trajectory[:, -1] - trajectory[:, 0]
    ).flatten(1).norm(dim=1)

    straightness_ratio = path_length / (displacement + 1e-12)

    velocities = sample_result.velocities.float().flatten(2)
    if velocities.shape[1] > 1:
        turning_cosine = F.cosine_similarity(
            velocities[:, 1:],
            velocities[:, :-1],
            dim=2,
        ).mean(dim=1)
    else:
        turning_cosine = torch.ones_like(path_length)

    return pd.DataFrame({
        "path_length": path_length.numpy(),
        "endpoint_displacement": displacement.numpy(),
        "straightness_ratio": straightness_ratio.numpy(),
        "mean_velocity_cosine": turning_cosine.numpy(),
    })


@torch.no_grad()
def classifier_features(classifier, images, device, batch_size=256):
    chunks = []
    logits = []

    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device)
        output, feature = classifier(batch, return_features=True)
        chunks.append(feature.float().cpu())
        logits.append(output.float().cpu())

    return torch.cat(chunks), torch.cat(logits)


def frechet_distance(features_real, features_fake):
    x = features_real.double().numpy()
    y = features_fake.double().numpy()

    mu_x, mu_y = x.mean(0), y.mean(0)
    cov_x = np.cov(x, rowvar=False)
    cov_y = np.cov(y, rowvar=False)

    product = cov_x @ cov_y
    eigvals = np.linalg.eigvals(product)
    eigvals = np.clip(eigvals.real, 0, None)
    sqrt_trace = np.sqrt(eigvals).sum()

    diff = mu_x - mu_y
    return float(
        diff @ diff
        + np.trace(cov_x)
        + np.trace(cov_y)
        - 2 * sqrt_trace
    )


def generated_distribution_summary(
    classifier,
    real_images,
    fake_images,
    fake_labels,
    device,
):
    real_features, _ = classifier_features(
        classifier,
        real_images,
        device,
    )
    fake_features, fake_logits = classifier_features(
        classifier,
        fake_images,
        device,
    )

    class_accuracy = float(
        (
            fake_logits.argmax(dim=1)
            == fake_labels.cpu()
        ).float().mean()
    )

    distance = frechet_distance(real_features, fake_features)

    return {
        "conditional_class_accuracy": class_accuracy,
        "feature_frechet_distance": distance,
    }
