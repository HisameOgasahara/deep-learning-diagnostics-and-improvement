import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from vit_lab_model_optim import FEATURE_LAYERS, SmallViT


@torch.no_grad()
def extract_features_and_logits(model, loader, device):
    model.eval()
    feature_chunks = {name: [] for name in FEATURE_LAYERS}
    logits_chunks, label_chunks = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits, features = model(x, return_features=True)
        for name in FEATURE_LAYERS:
            feature_chunks[name].append(features[name].float().cpu())
        logits_chunks.append(logits.float().cpu())
        label_chunks.append(y.cpu())
    features = {
        name: torch.cat(chunks, dim=0).numpy()
        for name, chunks in feature_chunks.items()
    }
    logits = torch.cat(logits_chunks, dim=0).numpy()
    labels = torch.cat(label_chunks, dim=0).numpy()
    return features, logits, labels


def load_checkpoint(run_name, epoch, ckpt_dir, device):
    model = SmallViT().to(device)
    state = torch.load(ckpt_dir / f"{run_name}_epoch{epoch}.pt", map_location=device)
    model.load_state_dict(state)
    return model


def covariance_spectrum(X):
    X = X.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    eig = (s ** 2) / max(1, len(X) - 1)
    return np.maximum(eig, 0.0)


def effective_rank_from_eig(eig, eps=1e-12):
    p = eig / (eig.sum() + eps)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p + eps)).sum()))


def linear_cka(X, Y, eps=1e-12):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    xty = X.T @ Y
    hsic = np.sum(xty ** 2)
    norm_x = np.sqrt(np.sum((X.T @ X) ** 2))
    norm_y = np.sqrt(np.sum((Y.T @ Y) ** 2))
    return float(hsic / (norm_x * norm_y + eps))


def ridge_linear_probe(train_X, train_y, val_X, val_y):
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(train_X, train_y)
    return float(clf.score(val_X, val_y))


def class_geometry_metrics(features, logits, labels, head_weight):
    X = features.astype(np.float64)
    classes = np.unique(labels)
    global_mean = X.mean(axis=0)

    class_means = []
    within_trace = 0.0
    class_radii = []
    class_dims = []

    for c in classes:
        Xc = X[labels == c]
        mu = Xc.mean(axis=0)
        class_means.append(mu)
        centered = Xc - mu
        within_trace += np.sum(centered ** 2) / max(1, len(Xc))
        class_radii.append(float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))))
        eig = covariance_spectrum(Xc)
        participation = (eig.sum() ** 2) / (np.sum(eig ** 2) + 1e-12)
        class_dims.append(float(participation))

    class_means = np.stack(class_means)
    centered_means = class_means - global_mean
    between_trace = np.mean(np.sum(centered_means ** 2, axis=1))
    nc1 = within_trace / (between_trace * len(classes) + 1e-12)

    normalized = centered_means / (
        np.linalg.norm(centered_means, axis=1, keepdims=True) + 1e-12
    )
    gram = normalized @ normalized.T
    target_offdiag = -1.0 / (len(classes) - 1)
    mask = ~np.eye(len(classes), dtype=bool)
    nc2_etf_error = float(np.sqrt(np.mean((gram[mask] - target_offdiag) ** 2)))
    center_abs_corr = float(np.mean(np.abs(gram[mask])))

    W = head_weight.astype(np.float64)
    W = W - W.mean(axis=0, keepdims=True)
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    nc3_alignment = float(np.mean(np.sum(W * normalized, axis=1)))

    true_logits = logits[np.arange(len(labels)), labels]
    masked = logits.copy()
    masked[np.arange(len(labels)), labels] = -np.inf
    margins = true_logits - masked.max(axis=1)

    nn_model = NearestNeighbors(n_neighbors=11, metric="euclidean")
    nn_model.fit(X)
    indices = nn_model.kneighbors(return_distance=False)[:, 1:]
    purity = float(np.mean(labels[indices] == labels[:, None]))

    return {
        "nc1_within_between": float(nc1),
        "nc2_etf_error": nc2_etf_error,
        "nc3_classifier_alignment": nc3_alignment,
        "margin_mean": float(np.mean(margins)),
        "margin_median": float(np.median(margins)),
        "margin_p10": float(np.quantile(margins, 0.10)),
        "margin_positive_fraction": float(np.mean(margins > 0)),
        "knn_purity": purity,
        "mean_class_radius": float(np.mean(class_radii)),
        "mean_class_participation_dim": float(np.mean(class_dims)),
        "class_center_abs_correlation": center_abs_corr,
    }


def expected_calibration_error(logits, labels, bins=15):
    logits = torch.as_tensor(logits)
    labels = torch.as_tensor(labels)
    probs = logits.softmax(dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels)
    edges = torch.linspace(0, 1, bins + 1)
    ece = torch.tensor(0.0)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (conf > low) & (conf <= high)
        if mask.any():
            acc = correct[mask].float().mean()
            avg_conf = conf[mask].mean()
            ece += mask.float().mean() * torch.abs(acc - avg_conf)
    return float(ece)


def representation_diagnostics(
    optimizer_names,
    diag_epochs,
    initial_state,
    init_train_features,
    init_train_labels,
    init_val_features,
    init_val_logits,
    init_val_labels,
    rep_train_loader,
    rep_val_loader,
    ckpt_dir,
    csv_dir,
    device,
):
    representation_rows, spectrum_rows, class_rows = [], [], []

    for run_name in optimizer_names:
        for epoch in diag_epochs:
            if epoch == 0:
                train_features = init_train_features
                val_features = init_val_features
                train_labels = init_train_labels
                val_labels = init_val_labels
                val_logits = init_val_logits
                head_weight = initial_state["head.weight"].numpy()
            else:
                model = load_checkpoint(run_name, epoch, ckpt_dir, device)
                train_features, _, train_labels = extract_features_and_logits(
                    model, rep_train_loader, device
                )
                val_features, val_logits, val_labels = extract_features_and_logits(
                    model, rep_val_loader, device
                )
                head_weight = model.head.weight.detach().cpu().numpy()
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            for layer in FEATURE_LAYERS:
                eig = covariance_spectrum(val_features[layer])
                representation_rows.append({
                    "run": run_name,
                    "epoch": epoch,
                    "layer": layer,
                    "effective_rank": effective_rank_from_eig(eig),
                    "cka_to_init": linear_cka(init_val_features[layer], val_features[layer]),
                    "linear_probe_accuracy": ridge_linear_probe(
                        train_features[layer], train_labels,
                        val_features[layer], val_labels,
                    ),
                })
                total = eig.sum() + 1e-12
                for index, value in enumerate(eig[:32], start=1):
                    spectrum_rows.append({
                        "run": run_name,
                        "epoch": epoch,
                        "layer": layer,
                        "rank": index,
                        "eigenvalue": float(value),
                        "explained_fraction": float(value / total),
                    })

            class_rows.append({
                "run": run_name,
                "epoch": epoch,
                **class_geometry_metrics(
                    val_features["penultimate"], val_logits, val_labels, head_weight
                ),
            })

    representation_df = pd.DataFrame(representation_rows)
    spectrum_df = pd.DataFrame(spectrum_rows)
    class_df = pd.DataFrame(class_rows)
    representation_df.to_csv(csv_dir / "representation_summary.csv", index=False)
    spectrum_df.to_csv(csv_dir / "covariance_spectrum.csv", index=False)
    class_df.to_csv(csv_dir / "class_geometry.csv", index=False)
    return representation_df, spectrum_df, class_df
