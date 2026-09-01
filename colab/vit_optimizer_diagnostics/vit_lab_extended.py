from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.svm import LinearSVC

from vit_lab_model_optim import SmallViT
from vit_lab_repr import covariance_spectrum, effective_rank_from_eig, load_checkpoint


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _flatten_named_parameters(model, prefixes=None):
    tensors = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if prefixes is not None and not any(name.startswith(prefix) for prefix in prefixes):
            continue

        tensors.append(parameter)

    if not tensors:
        raise ValueError("No trainable parameters matched the requested scope.")

    return tensors


def _center_gram(matrix):
    n = matrix.shape[0]
    center = np.eye(n) - np.ones((n, n)) / n
    return center @ matrix @ center


def _kernel_target_alignment(kernel, labels):
    labels = np.asarray(labels)
    classes = np.unique(labels)

    one_hot = np.zeros((len(labels), len(classes)), dtype=np.float64)
    class_to_index = {class_id: index for index, class_id in enumerate(classes)}

    for row, class_id in enumerate(labels):
        one_hot[row, class_to_index[class_id]] = 1.0

    target_kernel = one_hot @ one_hot.T
    centered_kernel = _center_gram(kernel)
    centered_target = _center_gram(target_kernel)

    numerator = np.sum(centered_kernel * centered_target)
    denominator = (
        np.linalg.norm(centered_kernel)
        * np.linalg.norm(centered_target)
        + 1e-12
    )

    return float(numerator / denominator)


# -----------------------------------------------------------------------------
# 1. Empirical tangent-kernel spectrum and target alignment
# -----------------------------------------------------------------------------


def empirical_tangent_kernel_diagnostics(
    model,
    x,
    y,
    parameter_prefixes=("blocks.5.", "norm.", "head."),
):
    """Compute a T4-safe empirical tangent-kernel diagnostic.

    For each sample, the true-class logit is differentiated with respect to a
    late-network parameter subset. The resulting tangent-feature rows form G,
    and K = G G^T is the empirical tangent kernel used here.

    This is deliberately labelled as a *partial empirical tangent kernel* rather
    than a full multiclass NTK because the notebook is intended to remain usable
    on Colab T4-class hardware.
    """

    model.eval()
    parameters = _flatten_named_parameters(model, prefixes=parameter_prefixes)

    tangent_rows = []

    for sample_index in range(x.shape[0]):
        model.zero_grad(set_to_none=True)

        logits = model(x[sample_index:sample_index + 1])
        score = logits[0, y[sample_index]]

        gradients = torch.autograd.grad(
            score,
            parameters,
            retain_graph=False,
            create_graph=False,
        )

        flat_gradient = torch.cat(
            [gradient.detach().float().reshape(-1).cpu() for gradient in gradients]
        )
        tangent_rows.append(flat_gradient)

    tangent_matrix = torch.stack(tangent_rows).numpy().astype(np.float64)
    kernel = tangent_matrix @ tangent_matrix.T

    eigenvalues = np.linalg.eigvalsh(kernel)
    eigenvalues = np.maximum(eigenvalues[::-1], 0.0)
    total = eigenvalues.sum() + 1e-12

    summary = {
        "kernel_target_alignment": _kernel_target_alignment(kernel, y.cpu().numpy()),
        "kernel_effective_rank": effective_rank_from_eig(eigenvalues),
        "kernel_trace": float(eigenvalues.sum()),
        "kernel_top_eigenvalue": float(eigenvalues[0]),
    }

    spectrum_rows = []
    for rank, eigenvalue in enumerate(eigenvalues, start=1):
        spectrum_rows.append(
            {
                "rank": rank,
                "eigenvalue": float(eigenvalue),
                "explained_fraction": float(eigenvalue / total),
            }
        )

    return summary, pd.DataFrame(spectrum_rows)


def run_tangent_kernel_diagnostics(
    optimizer_names,
    final_epoch,
    diagnostic_batch,
    ckpt_dir,
    csv_dir,
    device,
    sample_count=24,
):
    x, y = diagnostic_batch
    x = x[:sample_count].to(device)
    y = y[:sample_count].to(device)

    summary_rows = []
    spectrum_rows = []

    for run_name in optimizer_names:
        print("Tangent kernel:", run_name)

        model = load_checkpoint(run_name, final_epoch, ckpt_dir, device)
        summary, spectrum = empirical_tangent_kernel_diagnostics(model, x, y)

        summary_rows.append({"run": run_name, **summary})

        spectrum.insert(0, "run", run_name)
        spectrum_rows.append(spectrum)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows)
    spectrum_df = pd.concat(spectrum_rows, ignore_index=True)

    summary_df.to_csv(csv_dir / "tangent_kernel_summary.csv", index=False)
    spectrum_df.to_csv(csv_dir / "tangent_kernel_spectrum.csv", index=False)

    return summary_df, spectrum_df


# -----------------------------------------------------------------------------
# 2. Input-output Jacobian singular spectrum
# -----------------------------------------------------------------------------


def input_output_jacobian_spectrum(model, x):
    """Return singular values of d(logits)/d(input) for each sample."""

    model.eval()
    rows = []

    for sample_index in range(x.shape[0]):
        sample = x[sample_index:sample_index + 1].detach().clone()
        sample.requires_grad_(True)

        logits = model(sample)[0]
        jacobian_rows = []

        for class_index in range(logits.numel()):
            gradient = torch.autograd.grad(
                logits[class_index],
                sample,
                retain_graph=class_index < logits.numel() - 1,
                create_graph=False,
            )[0]
            jacobian_rows.append(gradient.reshape(-1))

        jacobian = torch.stack(jacobian_rows)
        singular_values = torch.linalg.svdvals(jacobian.float()).detach().cpu().numpy()

        squared = singular_values ** 2
        participation_rank = (
            squared.sum() ** 2
            / (np.sum(squared ** 2) + 1e-12)
        )

        for rank, singular_value in enumerate(singular_values, start=1):
            rows.append(
                {
                    "sample": sample_index,
                    "rank": rank,
                    "singular_value": float(singular_value),
                    "spectral_norm": float(singular_values[0]),
                    "frobenius_norm": float(np.sqrt(squared.sum())),
                    "jacobian_participation_rank": float(participation_rank),
                }
            )

    return pd.DataFrame(rows)


def run_jacobian_diagnostics(
    optimizer_names,
    final_epoch,
    diagnostic_batch,
    ckpt_dir,
    csv_dir,
    device,
    sample_count=8,
):
    x, _ = diagnostic_batch
    x = x[:sample_count].to(device)

    all_rows = []

    for run_name in optimizer_names:
        print("Input-output Jacobian:", run_name)

        model = load_checkpoint(run_name, final_epoch, ckpt_dir, device)
        frame = input_output_jacobian_spectrum(model, x)
        frame.insert(0, "run", run_name)
        all_rows.append(frame)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    jacobian_df = pd.concat(all_rows, ignore_index=True)

    summary_df = (
        jacobian_df
        .groupby("run")
        .agg(
            spectral_norm=("spectral_norm", "mean"),
            frobenius_norm=("frobenius_norm", "mean"),
            participation_rank=("jacobian_participation_rank", "mean"),
        )
        .reset_index()
    )

    jacobian_df.to_csv(csv_dir / "jacobian_spectrum.csv", index=False)
    summary_df.to_csv(csv_dir / "jacobian_summary.csv", index=False)

    return jacobian_df, summary_df


# -----------------------------------------------------------------------------
# 3. Relative sharpness at the classifier layer
# -----------------------------------------------------------------------------


@torch.no_grad()
def relative_sharpness_classifier(model, loader, device, max_samples=512):
    """Exact cross-entropy relative sharpness for the classifier weight.

    For logits z = W h + b and softmax probabilities p, the trace of the
    cross-entropy Hessian with respect to W is

        E[(1 - ||p||_2^2) ||h||_2^2].

    The reported quantity is ||W||_F^2 times that trace, matching the
    layerwise relative-sharpness form used in the referenced flatness work.
    """

    model.eval()

    total_trace = 0.0
    total_count = 0

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        logits, features = model(x, return_features=True)

        h = features["penultimate"].float()
        probabilities = logits.float().softmax(dim=1)

        probability_term = 1.0 - probabilities.square().sum(dim=1)
        feature_norm_sq = h.square().sum(dim=1)

        batch_trace = (probability_term * feature_norm_sq).sum().item()
        total_trace += batch_trace
        total_count += x.shape[0]

        if total_count >= max_samples:
            break

    trace = total_trace / max(1, total_count)
    weight_norm_sq = model.head.weight.detach().float().square().sum().item()
    relative_sharpness = weight_norm_sq * trace

    return {
        "head_weight_norm_sq": weight_norm_sq,
        "head_hessian_trace": trace,
        "relative_sharpness": relative_sharpness,
    }


def run_relative_sharpness_diagnostics(
    optimizer_names,
    final_epoch,
    loader,
    ckpt_dir,
    csv_dir,
    device,
    max_samples=512,
):
    rows = []

    for run_name in optimizer_names:
        print("Relative sharpness:", run_name)

        model = load_checkpoint(run_name, final_epoch, ckpt_dir, device)
        stats = relative_sharpness_classifier(
            model,
            loader,
            device,
            max_samples=max_samples,
        )

        rows.append({"run": run_name, **stats})

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    frame.to_csv(csv_dir / "relative_sharpness.csv", index=False)
    return frame


# -----------------------------------------------------------------------------
# 4. Neural-manifold axis geometry and empirical dichotomy capacity
# -----------------------------------------------------------------------------


def manifold_axis_geometry(features, labels):
    X = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    classes = np.unique(labels)

    global_mean = X.mean(axis=0)
    centers = []
    axes = []
    radii = []
    dimensions = []

    for class_id in classes:
        class_points = X[labels == class_id]
        center = class_points.mean(axis=0)
        centered = class_points - center

        eigenvalues = covariance_spectrum(class_points)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        top_axis = vh[0]

        radius = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
        participation_dimension = (
            eigenvalues.sum() ** 2
            / (np.sum(eigenvalues ** 2) + 1e-12)
        )

        centers.append(center - global_mean)
        axes.append(top_axis)
        radii.append(radius)
        dimensions.append(participation_dimension)

    centers = np.stack(centers)
    axes = np.stack(axes)

    normalized_centers = centers / (
        np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
    )
    normalized_axes = axes / (
        np.linalg.norm(axes, axis=1, keepdims=True) + 1e-12
    )

    center_axis_alignment = np.abs(
        np.sum(normalized_centers * normalized_axes, axis=1)
    )

    axis_gram = np.abs(normalized_axes @ normalized_axes.T)
    mask = ~np.eye(len(classes), dtype=bool)

    return {
        "mean_class_radius": float(np.mean(radii)),
        "mean_class_participation_dim": float(np.mean(dimensions)),
        "mean_center_axis_alignment": float(np.mean(center_axis_alignment)),
        "mean_axis_axis_alignment": float(np.mean(axis_gram[mask])),
    }


def empirical_dichotomy_capacity(
    features,
    labels,
    trials=64,
    random_state=0,
):
    """Estimate class-manifold separability with random class dichotomies.

    This is an empirical capacity proxy, not the asymptotic replica-theory
    manifold capacity from the Chung lab papers. Each trial assigns every class
    to one of two random super-classes and asks whether one linear hyperplane can
    separate all sample points according to that class-level dichotomy.
    """

    X = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    classes = np.unique(labels)

    X = X - X.mean(axis=0, keepdims=True)
    scale = X.std(axis=0, keepdims=True) + 1e-8
    X = X / scale

    rng = np.random.default_rng(random_state)
    separable = []

    for _ in range(trials):
        signs = rng.choice([-1, 1], size=len(classes))

        if np.all(signs == signs[0]):
            signs[0] *= -1

        class_sign = {
            class_id: signs[index]
            for index, class_id in enumerate(classes)
        }
        targets = np.array([class_sign[class_id] for class_id in labels])

        classifier = LinearSVC(
            C=1e4,
            dual="auto",
            max_iter=5000,
            random_state=random_state,
        )
        classifier.fit(X, targets)
        predictions = classifier.predict(X)

        separable.append(float(np.all(predictions == targets)))

    return float(np.mean(separable))


def run_manifold_diagnostics(
    optimizer_names,
    final_epoch,
    rep_val_loader,
    ckpt_dir,
    csv_dir,
    device,
    max_samples=1000,
    dichotomy_trials=64,
):
    rows = []

    for run_name in optimizer_names:
        print("Manifold geometry:", run_name)

        model = load_checkpoint(run_name, final_epoch, ckpt_dir, device)
        model.eval()

        features = []
        labels = []
        count = 0

        with torch.no_grad():
            for x, y in rep_val_loader:
                x = x.to(device, non_blocking=True)
                _, layer_features = model(x, return_features=True)

                features.append(layer_features["penultimate"].float().cpu().numpy())
                labels.append(y.numpy())

                count += y.numel()
                if count >= max_samples:
                    break

        X = np.concatenate(features, axis=0)[:max_samples]
        y = np.concatenate(labels, axis=0)[:max_samples]

        geometry = manifold_axis_geometry(X, y)
        capacity = empirical_dichotomy_capacity(
            X,
            y,
            trials=dichotomy_trials,
            random_state=0,
        )

        rows.append(
            {
                "run": run_name,
                **geometry,
                "empirical_dichotomy_capacity": capacity,
            }
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    frame.to_csv(csv_dir / "manifold_geometry_extended.csv", index=False)
    return frame


# -----------------------------------------------------------------------------
# 5. Parameter-trajectory PCA and path geometry
# -----------------------------------------------------------------------------


def _flatten_state_dict(state_dict):
    pieces = []

    for value in state_dict.values():
        if torch.is_floating_point(value):
            pieces.append(value.detach().float().reshape(-1).cpu())

    return torch.cat(pieces)


def trajectory_pca_and_path_metrics(
    run_name,
    epochs,
    ckpt_dir,
):
    vectors = []

    for epoch in epochs:
        state = torch.load(
            Path(ckpt_dir) / f"{run_name}_epoch{epoch}.pt",
            map_location="cpu",
        )
        vectors.append(_flatten_state_dict(state))

    trajectory = torch.stack(vectors).double()
    centered = trajectory - trajectory.mean(dim=0, keepdim=True)

    gram = centered @ centered.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)

    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = torch.clamp(eigenvalues[order], min=0.0)
    eigenvectors = eigenvectors[:, order]

    coordinates = eigenvectors * torch.sqrt(eigenvalues + 1e-12)
    total_variance = eigenvalues.sum() + 1e-12
    explained = eigenvalues / total_variance

    steps = trajectory[1:] - trajectory[:-1]
    step_norms = torch.linalg.vector_norm(steps, dim=1)
    path_length = step_norms.sum()
    chord_length = torch.linalg.vector_norm(trajectory[-1] - trajectory[0])

    consecutive_cosines = []
    for index in range(len(steps) - 1):
        cosine = F.cosine_similarity(
            steps[index].float(),
            steps[index + 1].float(),
            dim=0,
        )
        consecutive_cosines.append(float(cosine))

    coordinate_rows = []
    for index, epoch in enumerate(epochs):
        coordinate_rows.append(
            {
                "run": run_name,
                "epoch": epoch,
                "pc1": float(coordinates[index, 0]) if coordinates.shape[1] > 0 else 0.0,
                "pc2": float(coordinates[index, 1]) if coordinates.shape[1] > 1 else 0.0,
            }
        )

    summary = {
        "run": run_name,
        "pc1_explained": float(explained[0]) if len(explained) > 0 else 0.0,
        "pc2_explained": float(explained[1]) if len(explained) > 1 else 0.0,
        "path_length": float(path_length),
        "chord_length": float(chord_length),
        "path_to_chord_ratio": float(path_length / (chord_length + 1e-12)),
        "mean_consecutive_step_cosine": (
            float(np.mean(consecutive_cosines))
            if consecutive_cosines
            else np.nan
        ),
    }

    return pd.DataFrame(coordinate_rows), summary


def run_trajectory_diagnostics(
    optimizer_names,
    diag_epochs,
    ckpt_dir,
    csv_dir,
):
    coordinate_frames = []
    summary_rows = []

    for run_name in optimizer_names:
        print("Parameter trajectory:", run_name)
        coordinates, summary = trajectory_pca_and_path_metrics(
            run_name,
            diag_epochs,
            ckpt_dir,
        )

        coordinate_frames.append(coordinates)
        summary_rows.append(summary)

    coordinates_df = pd.concat(coordinate_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    coordinates_df.to_csv(csv_dir / "trajectory_pca.csv", index=False)
    summary_df.to_csv(csv_dir / "trajectory_path_summary.csv", index=False)

    return coordinates_df, summary_df
