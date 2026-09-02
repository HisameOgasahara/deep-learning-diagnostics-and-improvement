from pathlib import Path
import math
import random

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from vit_lab_model_optim import SmallViT
from vit_lab_train import train_one_optimizer
from vit_lab_repr import extract_features_and_logits, representation_diagnostics
from vit_lab_landscape import run_function_space_comparison, run_hessian_diagnostics
from vit_lab_extended import (
    run_jacobian_diagnostics,
    run_manifold_diagnostics,
    run_relative_sharpness_diagnostics,
    run_tangent_kernel_diagnostics,
    run_trajectory_diagnostics,
)


class CIFAR10FromHF(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        image = row["img"].convert("RGB")
        label = int(row["label"])
        return self.transform(image), label


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets(seed, train_samples, validation_mode):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    raw = load_dataset("uoft-cs/cifar10")
    train_augmented = CIFAR10FromHF(raw["train"], train_transform)
    train_fixed = CIFAR10FromHF(raw["train"], eval_transform)

    if validation_mode == "old_train_holdout_5k":
        permutation = torch.randperm(
            len(train_augmented),
            generator=torch.Generator().manual_seed(seed),
        ).tolist()

        train_indices = permutation[:train_samples]
        val_indices = permutation[40_000:45_000]

        train_dataset = Subset(train_augmented, train_indices)
        rep_train_dataset = Subset(train_fixed, train_indices)
        val_dataset = Subset(train_fixed, val_indices)

    elif validation_mode == "current_test_2k":
        train_split = raw["train"].shuffle(seed=seed).select(range(train_samples))
        val_split = raw["test"].shuffle(seed=seed).select(range(2_000))

        train_dataset = CIFAR10FromHF(train_split, train_transform)
        rep_train_dataset = CIFAR10FromHF(train_split, eval_transform)
        val_dataset = CIFAR10FromHF(val_split, eval_transform)

    else:
        raise ValueError(f"unknown validation_mode: {validation_mode}")

    return train_dataset, rep_train_dataset, val_dataset


def build_loaders(config, device, num_workers=4):
    train_dataset, rep_train_dataset, val_dataset = build_datasets(
        seed=config["seed"],
        train_samples=config["train_samples"],
        validation_mode=config["validation_mode"],
    )

    loader_kwargs = dict(
        batch_size=config["batch_size"],
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    rep_train_loader = DataLoader(rep_train_dataset, shuffle=False, **loader_kwargs)
    rep_val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    return (
        train_dataset,
        rep_train_dataset,
        val_dataset,
        train_loader,
        rep_train_loader,
        rep_val_loader,
    )


def choose_diag_epochs(epochs):
    return sorted(set([
        0,
        max(1, round(epochs * 0.2)),
        max(1, round(epochs * 0.5)),
        epochs,
    ]))


def run_muon_regime(config, output_root, device=None, dynamics_every=10):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"

    run_root = Path(output_root) / config["name"]
    ckpt_dir = run_root / "checkpoints"
    csv_dir = run_root / "csv"
    tb_dir = run_root / "tensorboard"

    for directory in [ckpt_dir, csv_dir, tb_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    seed_everything(config["seed"])

    (
        train_dataset,
        rep_train_dataset,
        val_dataset,
        train_loader,
        rep_train_loader,
        rep_val_loader,
    ) = build_loaders(config, device)

    seed_everything(config["seed"])
    initial_model = SmallViT().to(device)
    initial_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in initial_model.state_dict().items()
    }

    init_train_features, _, init_train_labels = extract_features_and_logits(
        initial_model,
        rep_train_loader,
        device,
    )
    init_val_features, init_val_logits, init_val_labels = extract_features_and_logits(
        initial_model,
        rep_val_loader,
        device,
    )
    diagnostic_batch = next(iter(rep_val_loader))

    del initial_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    epochs = config["epochs"]
    diag_epochs = choose_diag_epochs(epochs)

    history, dynamics, gradient_noise = train_one_optimizer(
        run_name="muon",
        initial_state=initial_state,
        train_loader=train_loader,
        val_loader=rep_val_loader,
        device=device,
        epochs=epochs,
        diag_epochs=diag_epochs,
        dynamics_every=dynamics_every,
        ckpt_dir=ckpt_dir,
        csv_dir=csv_dir,
        tb_dir=tb_dir,
        amp_enabled=amp_enabled,
    )

    representation_df, spectrum_df, class_df = representation_diagnostics(
        optimizer_names=["muon"],
        diag_epochs=diag_epochs,
        initial_state=initial_state,
        init_train_features=init_train_features,
        init_train_labels=init_train_labels,
        init_val_features=init_val_features,
        init_val_logits=init_val_logits,
        init_val_labels=init_val_labels,
        rep_train_loader=rep_train_loader,
        rep_val_loader=rep_val_loader,
        ckpt_dir=ckpt_dir,
        csv_dir=csv_dir,
        device=device,
    )

    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        hessian_df, hessian_summary = run_hessian_diagnostics(
            optimizer_names=["muon"],
            final_epoch=epochs,
            initial_state=initial_state,
            hessian_batch=diagnostic_batch,
            ckpt_dir=ckpt_dir,
            csv_dir=csv_dir,
            device=device,
            steps=16,
        )

    function_summary, _ = run_function_space_comparison(
        optimizer_names=["muon"],
        final_epoch=epochs,
        loader=rep_val_loader,
        ckpt_dir=ckpt_dir,
        csv_dir=csv_dir,
        device=device,
    )

    tangent_summary, tangent_spectrum = run_tangent_kernel_diagnostics(
        optimizer_names=["muon"],
        final_epoch=epochs,
        diagnostic_batch=diagnostic_batch,
        ckpt_dir=ckpt_dir,
        csv_dir=csv_dir,
        device=device,
        sample_count=24,
    )

    jacobian_df, jacobian_summary = run_jacobian_diagnostics(
        optimizer_names=["muon"],
        final_epoch=epochs,
        diagnostic_batch=diagnostic_batch,
        ckpt_dir=ckpt_dir,
        csv_dir=csv_dir,
        device=device,
        sample_count=8,
    )

    relative_sharpness_df = run_relative_sharpness_diagnostics(
        optimizer_names=["muon"],
        final_epoch=epochs,
        loader=rep_val_loader,
        ckpt_dir=ckpt_dir,
        csv_dir=csv_dir,
        device=device,
        max_samples=512,
    )

    manifold_df = run_manifold_diagnostics(
        optimizer_names=["muon"],
        final_epoch=epochs,
        rep_val_loader=rep_val_loader,
        ckpt_dir=ckpt_dir,
        csv_dir=csv_dir,
        device=device,
        max_samples=1000,
        dichotomy_trials=64,
    )

    trajectory_coordinates, trajectory_summary = run_trajectory_diagnostics(
        optimizer_names=["muon"],
        diag_epochs=diag_epochs,
        ckpt_dir=ckpt_dir,
        csv_dir=csv_dir,
    )

    final_rep = representation_df.query(
        f"epoch == {epochs} and layer == 'penultimate'"
    ).iloc[0]
    final_class = class_df.query(f"epoch == {epochs}").iloc[0]

    summary = {
        **config,
        "steps_per_epoch": len(train_loader),
        "approx_updates": len(train_loader) * epochs,
        "train_accuracy": float(history.iloc[-1]["train_accuracy"]),
        "val_accuracy": float(history.iloc[-1]["val_accuracy"]),
        "val_loss": float(history.iloc[-1]["val_loss"]),
        "penultimate_effective_rank": float(final_rep["effective_rank"]),
        "penultimate_cka_to_init": float(final_rep["cka_to_init"]),
        "penultimate_linear_probe": float(final_rep["linear_probe_accuracy"]),
        "nc1": float(final_class["nc1_within_between"]),
        "knn_purity": float(final_class["knn_purity"]),
        "margin_mean": float(final_class["margin_mean"]),
        "hessian_min_ritz": float(hessian_summary.query("run == 'muon'").iloc[0]["min_ritz"]),
        "hessian_max_ritz": float(hessian_summary.query("run == 'muon'").iloc[0]["max_ritz"]),
        "ece": float(function_summary.iloc[0]["ece"]),
        "tangent_target_alignment": float(tangent_summary.iloc[0]["kernel_target_alignment"]),
        "tangent_effective_rank": float(tangent_summary.iloc[0]["kernel_effective_rank"]),
        "jacobian_spectral_norm": float(jacobian_summary.iloc[0]["spectral_norm"]),
        "jacobian_participation_rank": float(jacobian_summary.iloc[0]["participation_rank"]),
        "relative_sharpness": float(relative_sharpness_df.iloc[0]["relative_sharpness"]),
        "mean_class_radius": float(manifold_df.iloc[0]["mean_class_radius"]),
        "mean_class_participation_dim": float(manifold_df.iloc[0]["mean_class_participation_dim"]),
        "mean_center_axis_alignment": float(manifold_df.iloc[0]["mean_center_axis_alignment"]),
        "mean_axis_axis_alignment": float(manifold_df.iloc[0]["mean_axis_axis_alignment"]),
        "empirical_dichotomy_capacity": float(manifold_df.iloc[0]["empirical_dichotomy_capacity"]),
        "path_to_chord_ratio": float(trajectory_summary.iloc[0]["path_to_chord_ratio"]),
    }

    pd.DataFrame([summary]).to_csv(run_root / "summary.csv", index=False)

    return {
        "summary": summary,
        "history": history,
        "dynamics": dynamics,
        "gradient_noise": gradient_noise,
        "representation": representation_df,
        "spectrum": spectrum_df,
        "class_geometry": class_df,
        "hessian": hessian_df,
        "hessian_summary": hessian_summary,
        "tangent_summary": tangent_summary,
        "tangent_spectrum": tangent_spectrum,
        "jacobian": jacobian_df,
        "jacobian_summary": jacobian_summary,
        "relative_sharpness": relative_sharpness_df,
        "manifold": manifold_df,
        "trajectory_coordinates": trajectory_coordinates,
        "trajectory_summary": trajectory_summary,
    }
