import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from prodigyopt import Prodigy


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=192, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class EncoderBlock(nn.Module):
    def __init__(self, dim=192, heads=3, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        z = self.norm1(x)
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SmallViT(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=192,
        depth=6,
        heads=3,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(3, dim, patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            EncoderBlock(dim, heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self, x, return_layers=False):
        x = self.patch_embed(x)
        features = {"patch": x.mean(dim=1)}

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed

        for index, block in enumerate(self.blocks, start=1):
            x = block(x)
            if index in (2, 4, 6):
                features[f"block{index}"] = x[:, 0]

        penultimate = self.norm(x)[:, 0]
        features["penultimate"] = penultimate
        return (penultimate, features) if return_layers else penultimate

    def forward(self, x, return_features=False):
        penultimate, features = self.forward_features(x, return_layers=True)
        logits = self.head(penultimate)
        return (logits, features) if return_features else logits


FEATURE_LAYERS = ["patch", "block2", "block4", "block6", "penultimate"]
TRACKED_PARAM_NAMES = [
    "patch_embed.proj.weight",
    "blocks.0.attn.in_proj_weight",
    "blocks.3.mlp.fc1.weight",
    "head.weight",
]


@torch.no_grad()
def zeropower_via_newton_schulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class LocalMuon(torch.optim.Optimizer):
    """Fallback for Colab runtimes that do not yet expose torch.optim.Muon."""

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01,
                 nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("LocalMuon expects only 2D parameters")
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                update = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                update = zeropower_via_newton_schulz5(update, steps=group["ns_steps"])
                update = update * math.sqrt(max(1.0, p.shape[0] / p.shape[1]))
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
        return loss


class OptimizerBundle:
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)


def split_muon_parameters(model):
    muon_params, aux_params = [], []
    for name, p in model.named_parameters():
        eligible = (
            p.ndim == 2
            and not name.startswith("head.")
            and "pos_embed" not in name
            and "cls_token" not in name
        )
        (muon_params if eligible else aux_params).append(p)
    return muon_params, aux_params


def make_optimizer(model, name):
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    if name == "prodigy":
        return Prodigy(
            model.parameters(), lr=1.0, weight_decay=0.05,
            decouple=True, safeguard_warmup=True,
        )
    if name == "muon":
        muon_params, aux_params = split_muon_parameters(model)
        if hasattr(torch.optim, "Muon"):
            muon_opt = torch.optim.Muon(
                muon_params, lr=0.02, momentum=0.95, weight_decay=0.01
            )
        else:
            muon_opt = LocalMuon(
                muon_params, lr=0.02, momentum=0.95, weight_decay=0.01
            )
        aux_opt = torch.optim.AdamW(aux_params, lr=3e-4, weight_decay=0.05)
        return OptimizerBundle([muon_opt, aux_opt])
    raise ValueError(name)
