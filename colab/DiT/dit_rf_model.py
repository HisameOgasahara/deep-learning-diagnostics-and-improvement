import math
import torch
import torch.nn as nn


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal embedding for scalar t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=t.device)
        / max(1, half)
    )
    args = t[:, None].float() * freqs[None] * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class SelfAttention(nn.Module):
    """Readable multi-head self-attention using explicit matmul + softmax."""

    def __init__(self, dim=192, heads=6):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, return_attn=False):
        b, n, d = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(
            b,
            n,
            3,
            self.heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = scores.softmax(dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(b, n, d)
        out = self.proj(out)

        if return_attn:
            return out, attn
        return out


class DiTBlock(nn.Module):
    def __init__(self, dim=192, heads=6, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

        self.ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, cond, return_attn=False):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(cond).chunk(6, dim=-1)

        h = self.norm1(x)
        h = h * (1 + scale1[:, None, :]) + shift1[:, None, :]

        if return_attn:
            attn_out, attn_weights = self.attn(h, return_attn=True)
        else:
            attn_out = self.attn(h)
            attn_weights = None

        x = x + gate1[:, None, :] * attn_out

        h = self.norm2(x)
        h = h * (1 + scale2[:, None, :]) + shift2[:, None, :]
        x = x + gate2[:, None, :] * self.mlp(h)

        if return_attn:
            return x, attn_weights
        return x


class MiniConditionalDiT(nn.Module):
    """
    Pixel-space conditional DiT for FashionMNIST.

    Input/output:
        x: [B, 1, 28, 28]
        t: [B]
        y: [B] with classes 0..9
    """

    def __init__(
        self,
        image_size=28,
        patch_size=4,
        in_channels=1,
        dim=192,
        depth=6,
        heads=6,
        mlp_ratio=4,
        num_classes=10,
    ):
        super().__init__()
        assert image_size % patch_size == 0

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dim = dim
        self.depth = depth

        grid = image_size // patch_size
        self.grid = grid
        self.num_tokens = grid * grid
        patch_dim = in_channels * patch_size * patch_size

        self.patch_embed = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, dim)
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
        self.class_embed = nn.Embedding(num_classes, dim)

        self.blocks = nn.ModuleList([
            DiTBlock(dim, heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(
            dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.final_ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim),
        )
        self.final_linear = nn.Linear(dim, patch_dim)

        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def _unpatchify(self, patches):
        b = patches.shape[0]
        p = self.patch_size
        g = self.grid
        c = self.in_channels

        x = patches.view(b, g, g, c, p, p)
        x = torch.einsum("bhwcpq->bchpwq", x)
        return x.reshape(b, c, g * p, g * p)

    def forward(
        self,
        x,
        t,
        y,
        return_features=False,
        return_attention=False,
    ):
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed

        t_emb = timestep_embedding(t, self.dim)
        cond = self.time_mlp(t_emb) + self.class_embed(y)

        features = {}
        attentions = {}

        for index, block in enumerate(self.blocks):
            if return_attention:
                tokens, attn = block(
                    tokens,
                    cond,
                    return_attn=True,
                )
                attentions[f"block_{index + 1}"] = attn
            else:
                tokens = block(tokens, cond)

            if return_features:
                features[f"block_{index + 1}"] = tokens.mean(dim=1)

        shift, scale = self.final_ada(cond).chunk(2, dim=-1)
        h = self.final_norm(tokens)
        h = h * (1 + scale[:, None, :]) + shift[:, None, :]
        patches = self.final_linear(h)
        velocity = self._unpatchify(patches)

        if return_features or return_attention:
            aux = {}
            if return_features:
                aux["features"] = features
            if return_attention:
                aux["attention"] = attentions
            return velocity, aux

        return velocity


def rectified_flow_batch(x_data, labels, device):
    """
    Straight interpolation from Gaussian noise x0 to data x1.

    x_t = (1-t)x0 + t x1
    target velocity = x1 - x0
    """
    x1 = x_data.to(device)
    y = labels.to(device)
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=device)

    t_view = t[:, None, None, None]
    xt = (1.0 - t_view) * x0 + t_view * x1
    target_v = x1 - x0

    return xt, t, y, target_v, x0, x1
