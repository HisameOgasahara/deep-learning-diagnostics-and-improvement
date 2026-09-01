import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from vit_lab_model_optim import (
    OptimizerBundle,
    SmallViT,
    TRACKED_PARAM_NAMES,
    make_optimizer,
)


def cosine_flat(a, b, eps=1e-12):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return (
        torch.dot(a, b)
        / (a.norm() * b.norm() + eps)
    ).item()


class GradientNoiseAccumulator:
    def __init__(self):
        self.count = 0
        self.sum_grad = {}
        self.sum_sq_norm = defaultdict(float)

    def add(self, named_params):
        self.count += 1

        for name in TRACKED_PARAM_NAMES:
            grad = named_params[name].grad
            if grad is None:
                continue

            grad_cpu = grad.detach().float().cpu()

            if name not in self.sum_grad:
                self.sum_grad[name] = torch.zeros_like(grad_cpu)

            self.sum_grad[name].add_(grad_cpu)
            self.sum_sq_norm[name] += grad_cpu.square().sum().item()

    def summarize(self):
        result = {}

        if self.count == 0:
            return result

        for name, sum_grad in self.sum_grad.items():
            mean_grad = sum_grad / self.count
            mean_sq_norm = self.sum_sq_norm[name] / self.count
            signal = mean_grad.square().sum().item()
            variance_trace = max(0.0, mean_sq_norm - signal)

            result[name] = {
                "grad_mean_norm": math.sqrt(signal),
                "grad_variance_trace": variance_trace,
                "gradient_noise_ratio": (
                    variance_trace / (signal + 1e-12)
                ),
            }

        return result


@torch.no_grad()
def evaluate(model, loader, device, max_samples=None):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)

        total_loss += F.cross_entropy(
            logits,
            y,
            reduction="sum",
        ).item()
        total_correct += (
            logits.argmax(dim=1) == y
        ).sum().item()
        total_count += y.numel()

        if max_samples is not None and total_count >= max_samples:
            break

    return (
        total_loss / total_count,
        total_correct / total_count,
    )


def train_one_optimizer(
    run_name,
    initial_state,
    train_loader,
    val_loader,
    device,
    epochs,
    diag_epochs,
    dynamics_every,
    ckpt_dir,
    csv_dir,
    tb_dir,
    amp_enabled=True,
):
    model = SmallViT().to(device)
    model.load_state_dict(initial_state)

    optimizer = make_optimizer(model, run_name)
    writer = SummaryWriter(str(tb_dir / run_name))
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    history_rows = []
    dynamics_rows = []
    noise_rows = []

    named_params = dict(model.named_parameters())
    initial_tracked = {
        name: named_params[name].detach().cpu().clone()
        for name in TRACKED_PARAM_NAMES
    }

    previous_grad = {}
    previous_update = {}
    global_step = 0

    torch.save(
        model.state_dict(),
        ckpt_dir / f"{run_name}_epoch0.pt",
    )

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        noise_acc = GradientNoiseAccumulator()

        loss_sum = 0.0
        correct = 0
        count = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()

            if isinstance(optimizer, OptimizerBundle):
                for opt in optimizer.optimizers:
                    scaler.unscale_(opt)
            else:
                scaler.unscale_(optimizer)

            should_track = global_step % dynamics_every == 0
            before_update = {}

            if should_track:
                named_params = dict(model.named_parameters())
                noise_acc.add(named_params)

                for param_name in TRACKED_PARAM_NAMES:
                    parameter = named_params[param_name]
                    grad = parameter.grad.detach()

                    before_update[param_name] = (
                        parameter.detach().clone()
                    )

                    dynamics_rows.append(
                        {
                            "run": run_name,
                            "epoch": epoch,
                            "step": global_step,
                            "parameter": param_name,
                            "grad_norm": grad.norm().item(),
                            "grad_cosine_prev": (
                                cosine_flat(
                                    grad,
                                    previous_grad[param_name],
                                )
                                if param_name in previous_grad
                                else np.nan
                            ),
                        }
                    )

                    previous_grad[param_name] = grad.clone()

            if isinstance(optimizer, OptimizerBundle):
                for opt in optimizer.optimizers:
                    scaler.step(opt)
                scaler.update()
            else:
                scaler.step(optimizer)
                scaler.update()

            if should_track:
                named_params = dict(model.named_parameters())
                recent_rows = dynamics_rows[-len(TRACKED_PARAM_NAMES):]

                for row in recent_rows:
                    param_name = row["parameter"]
                    parameter = named_params[param_name]

                    update = (
                        parameter.detach()
                        - before_update[param_name]
                    )

                    row["update_norm"] = update.norm().item()
                    row["update_to_weight"] = (
                        update.norm()
                        / (parameter.detach().norm() + 1e-12)
                    ).item()
                    row["update_cosine_prev"] = (
                        cosine_flat(
                            update,
                            previous_update[param_name],
                        )
                        if param_name in previous_update
                        else np.nan
                    )
                    row["parameter_displacement"] = (
                        parameter.detach().cpu()
                        - initial_tracked[param_name]
                    ).norm().item()

                    previous_update[param_name] = update.clone()

            batch_size = y.numel()
            loss_sum += loss.item() * batch_size
            correct += (
                logits.argmax(dim=1) == y
            ).sum().item()
            count += batch_size
            global_step += 1

        train_loss = loss_sum / count
        train_acc = correct / count
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            device,
        )
        elapsed = time.time() - start

        history_rows.append(
            {
                "run": run_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "seconds": elapsed,
            }
        )

        for param_name, stats in noise_acc.summarize().items():
            noise_rows.append(
                {
                    "run": run_name,
                    "epoch": epoch,
                    "parameter": param_name,
                    **stats,
                }
            )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("accuracy/train", train_acc, epoch)
        writer.add_scalar("accuracy/val", val_acc, epoch)

        print(
            f"[{run_name:7s}] epoch {epoch:02d}/{epochs} | "
            f"train {train_acc:.4f} | "
            f"val {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if epoch in diag_epochs:
            torch.save(
                model.state_dict(),
                ckpt_dir / f"{run_name}_epoch{epoch}.pt",
            )

    writer.close()

    history_df = pd.DataFrame(history_rows)
    dynamics_df = pd.DataFrame(dynamics_rows)
    noise_df = pd.DataFrame(noise_rows)

    history_df.to_csv(
        csv_dir / f"{run_name}_history.csv",
        index=False,
    )
    dynamics_df.to_csv(
        csv_dir / f"{run_name}_dynamics.csv",
        index=False,
    )
    noise_df.to_csv(
        csv_dir / f"{run_name}_gradient_noise.csv",
        index=False,
    )

    return history_df, dynamics_df, noise_df
