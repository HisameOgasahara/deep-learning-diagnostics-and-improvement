from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dit_rf_model import rectified_flow_batch


def train_dit(
    model,
    train_loader,
    device,
    artifact_dir,
    epochs=50,
    checkpoint_epochs=(0, 10, 25, 50),
    learning_rate=3e-4,
    weight_decay=1e-2,
):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    t_edges = torch.linspace(0, 1, 11)
    checkpoint_epochs = set(checkpoint_epochs)
    history_rows = []
    timestep_rows = []

    if 0 in checkpoint_epochs:
        torch.save(
            model.state_dict(),
            artifact_dir / "dit_epoch_0.pt",
        )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        sample_count = 0
        bin_loss_sum = np.zeros(10, dtype=np.float64)
        bin_count = np.zeros(10, dtype=np.int64)

        for x1, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            xt, t, y, target_v, _, _ = rectified_flow_batch(
                x1,
                y,
                device,
            )

            with torch.amp.autocast(
                "cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                pred_v = model(xt, t, y)
                per_sample_loss = (
                    (pred_v - target_v)
                    .flatten(1)
                    .square()
                    .mean(dim=1)
                )
                loss = per_sample_loss.mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )
            scaler.step(optimizer)
            scaler.update()

            batch_size = x1.shape[0]
            epoch_loss_sum += float(loss.detach()) * batch_size
            sample_count += batch_size

            bin_ids = torch.bucketize(
                t.detach().cpu(),
                t_edges[1:-1],
            ).numpy()
            loss_values = per_sample_loss.detach().float().cpu().numpy()

            for bin_id in range(10):
                mask = bin_ids == bin_id
                if mask.any():
                    bin_loss_sum[bin_id] += loss_values[mask].sum()
                    bin_count[bin_id] += mask.sum()

        scheduler.step()
        epoch_loss = epoch_loss_sum / sample_count

        history_rows.append({
            "epoch": epoch,
            "train_flow_loss": epoch_loss,
            "lr": scheduler.get_last_lr()[0],
        })

        for bin_id in range(10):
            timestep_rows.append({
                "epoch": epoch,
                "t_mid": float(
                    (t_edges[bin_id] + t_edges[bin_id + 1]) / 2
                ),
                "flow_loss": (
                    bin_loss_sum[bin_id]
                    / max(1, bin_count[bin_id])
                ),
            })

        if epoch in checkpoint_epochs:
            torch.save(
                model.state_dict(),
                artifact_dir / f"dit_epoch_{epoch}.pt",
            )

        print(
            f"epoch {epoch:02d}/{epochs} | "
            f"flow loss {epoch_loss:.5f}"
        )

    return (
        pd.DataFrame(history_rows),
        pd.DataFrame(timestep_rows),
    )


class FashionFeatureEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.feature = nn.Linear(64 * 7 * 7, feature_dim)
        self.head = nn.Linear(feature_dim, 10)

    def forward(self, x, return_features=False):
        h = self.conv(x).flatten(1)
        feature = self.feature(h)
        logits = self.head(F.gelu(feature))
        if return_features:
            return logits, feature
        return logits


def train_feature_encoder(
    train_loader,
    val_loader,
    device,
    epochs=3,
):
    model = FashionFeatureEncoder().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    for epoch in range(epochs):
        model.train()
        correct = 0
        count = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            count += labels.numel()

        print(
            f"encoder epoch {epoch + 1}/{epochs} | "
            f"train acc {correct / count:.4f}"
        )

    model.eval()
    correct = 0
    count = 0

    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images.to(device))
            correct += (
                logits.argmax(dim=1).cpu() == labels
            ).sum().item()
            count += labels.numel()

    print(f"encoder validation accuracy: {correct / count:.4f}")
    return model
