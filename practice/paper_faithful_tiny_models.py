import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(q, k):
    d = q.size(-1)
    assert d % 2 == 0
    pos = torch.arange(q.size(-2), device=q.device, dtype=q.dtype)
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device=q.device, dtype=q.dtype) / d))
    ang = pos[:, None] * inv[None, :]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    def rot(x):
        xe, xo = x[..., 0::2], x[..., 1::2]
        return torch.stack((xe*cos-xo*sin, xe*sin+xo*cos), dim=-1).flatten(-2)
    return rot(q), rot(k)


class SelfAttention(nn.Module):
    def __init__(self, d, heads, rope=False):
        super().__init__()
        assert d % heads == 0
        self.heads, self.dh, self.rope = heads, d // heads, rope
        self.qkv = nn.Linear(d, 3*d)
        self.out = nn.Linear(d, d)

    def forward(self, x, allowed_mask=None):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.rope:
            q, k = apply_rope(q, k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        if allowed_mask is not None:
            if allowed_mask.ndim == 2:
                allowed_mask = allowed_mask[None, None]
            elif allowed_mask.ndim == 3:
                allowed_mask = allowed_mask[:, None]
            scores = scores.masked_fill(~allowed_mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        y = (weights @ v).transpose(1, 2).contiguous().reshape(b, n, d)
        return self.out(y), weights


class PreNormBlock(nn.Module):
    def __init__(self, d, heads, mlp_ratio=4, rope=False):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = SelfAttention(d, heads, rope=rope)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio*d), nn.GELU(), nn.Linear(mlp_ratio*d, d))

    def forward(self, x, allowed_mask=None):
        a, weights = self.attn(self.n1(x), allowed_mask)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x, weights


class TinyGPT(nn.Module):
    """GPT-2-like tiny decoder: learned token/position embeddings, causal MHA, pre-norm residual blocks, tied LM head."""
    def __init__(self, vocab=32, max_len=16, d=24, heads=3, depth=2):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([PreNormBlock(d, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens, return_attn=False):
        _, n = tokens.shape
        positions = torch.arange(n, device=tokens.device)
        x = self.tok(tokens) + self.pos(positions)[None]
        causal = torch.tril(torch.ones(n, n, dtype=torch.bool, device=tokens.device))
        maps = []
        for block in self.blocks:
            x, a = block(x, causal)
            maps.append(a)
        logits = F.linear(self.norm(x), self.tok.weight)
        return (logits, maps) if return_attn else logits


class TinyViT(nn.Module):
    """ViT-like tiny classifier: non-overlapping patches + CLS + learned position + encoder."""
    def __init__(self, image=16, patch=4, d=24, heads=3, depth=2, classes=3):
        super().__init__()
        self.patch = patch
        n = (image // patch) ** 2
        self.patch_proj = nn.Linear(3*patch*patch, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, 1+n, d))
        self.blocks = nn.ModuleList([PreNormBlock(d, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, classes)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, image, return_attn=False):
        p = self.patch
        patches = F.unfold(image, kernel_size=p, stride=p).transpose(1, 2)
        x = self.patch_proj(patches)
        x = torch.cat([self.cls.expand(image.size(0), -1, -1), x], dim=1)
        x = x + self.pos[:, :x.size(1)]
        maps = []
        for block in self.blocks:
            x, a = block(x)
            maps.append(a)
        logits = self.head(self.norm(x)[:, 0])
        return (logits, maps) if return_attn else logits


def sinusoidal_1d(pos, dim):
    assert dim % 2 == 0
    pos = pos.float().reshape(-1, 1)
    omega = torch.arange(dim//2, device=pos.device, dtype=torch.float32) / (dim//2)
    omega = 1.0 / (10000 ** omega)
    out = pos * omega[None]
    return torch.cat([out.sin(), out.cos()], dim=-1)


def sincos_grid_2d(h, w, d, device):
    assert d % 4 == 0
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    return torch.cat([sinusoidal_1d(yy.reshape(-1), d//2), sinusoidal_1d(xx.reshape(-1), d//2)], dim=-1)


def sincos_grid_3d(z, y, x, d, device):
    assert d % 6 == 0
    zz, yy, xx = torch.meshgrid(torch.arange(z, device=device), torch.arange(y, device=device), torch.arange(x, device=device), indexing='ij')
    e = d // 3
    return torch.cat([sinusoidal_1d(zz.reshape(-1), e), sinusoidal_1d(yy.reshape(-1), e), sinusoidal_1d(xx.reshape(-1), e)], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, d, freq_dim=32):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t):
        return self.mlp(sinusoidal_1d(t * 1000.0, self.freq_dim))


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    """DiT adaLN-Zero block following the official facebookresearch/DiT structure."""
    def __init__(self, d, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(d, heads)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(approximate='tanh'), nn.Linear(4*d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6*d))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c, return_attn=False):
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = self.ada(c).chunk(6, dim=-1)
        a, weights = self.attn(modulate(self.n1(x), sh_a, sc_a))
        x = x + g_a[:, None] * a
        x = x + g_m[:, None] * self.mlp(modulate(self.n2(x), sh_m, sc_m))
        return (x, weights) if return_attn else x


class DiTFinal(nn.Module):
    def __init__(self, d, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2*d))
        self.out = nn.Linear(d, out_dim)
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, c):
        shift, scale = self.ada(c).chunk(2, dim=-1)
        return self.out(modulate(self.norm(x), shift, scale))


class Tiny2DDiT(nn.Module):
    """DiT backbone with Flow-Matching velocity output instead of diffusion epsilon training target."""
    def __init__(self, channels=2, size=8, patch=2, d=24, heads=3, depth=2):
        super().__init__()
        self.channels, self.size, self.patch, self.d = channels, size, patch, d
        self.in_proj = nn.Linear(channels*patch*patch, d)
        self.time = TimestepEmbedder(d)
        self.blocks = nn.ModuleList([DiTBlock(d, heads) for _ in range(depth)])
        self.final = DiTFinal(d, channels*patch*patch)

    def forward(self, x, t, return_attn=False):
        p = self.patch
        tok = F.unfold(x, kernel_size=p, stride=p).transpose(1, 2)
        g = self.size // p
        z = self.in_proj(tok) + sincos_grid_2d(g, g, self.d, x.device).to(x.dtype)[None]
        c = self.time(t.reshape(-1))
        maps = []
        for block in self.blocks:
            if return_attn:
                z, a = block(z, c, True)
                maps.append(a)
            else:
                z = block(z, c)
        patches = self.final(z, c).transpose(1, 2)
        out = F.fold(patches, output_size=(self.size, self.size), kernel_size=p, stride=p)
        return (out, maps) if return_attn else out


def patchify3d(x, p):
    b, c, z, y, w = x.shape
    assert z%p == y%p == w%p == 0
    x = x.reshape(b, c, z//p, p, y//p, p, w//p, p)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return x.reshape(b, (z//p)*(y//p)*(w//p), c*p**3)


def unpatchify3d(tok, c, size, p):
    b = tok.size(0)
    g = size // p
    x = tok.reshape(b, g, g, g, c, p, p, p)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return x.reshape(b, c, size, size, size)


class Tiny3DDiT(nn.Module):
    """Educational 3D extension of DiT: 3D non-overlapping patch tokens + 3D sin-cos position + adaLN-Zero."""
    def __init__(self, channels=1, size=4, patch=2, d=24, heads=3, depth=2):
        super().__init__()
        self.channels, self.size, self.patch, self.d = channels, size, patch, d
        self.in_proj = nn.Linear(channels*patch**3, d)
        self.time = TimestepEmbedder(d)
        self.blocks = nn.ModuleList([DiTBlock(d, heads) for _ in range(depth)])
        self.final = DiTFinal(d, channels*patch**3)

    def forward(self, x, t):
        p, g = self.patch, self.size // self.patch
        z = self.in_proj(patchify3d(x, p)) + sincos_grid_3d(g, g, g, self.d, x.device).to(x.dtype)[None]
        c = self.time(t.reshape(-1))
        for block in self.blocks:
            z = block(z, c)
        return unpatchify3d(self.final(z, c), self.channels, self.size, p)


def make_pi0_like_mask(prefix_len, state_len, action_len, device):
    n = prefix_len + state_len + action_len
    m = torch.zeros(n, n, dtype=torch.bool, device=device)
    m[:prefix_len, :prefix_len] = True
    s = prefix_len
    m[s:s+state_len, :s+state_len] = True
    a = s + state_len
    m[a:, :] = True
    return m


class TinyVLAFlowPolicy(nn.Module):
    """π0/openpi-like tiny VLA: vision+language prefix, state+noisy-action suffix, joint masked Transformer, action flow."""
    def __init__(self, image=16, patch=8, vocab=32, d=24, heads=3, depth=2, action_dim=4, horizon=4):
        super().__init__()
        self.patch, self.horizon = patch, horizon
        self.vision = nn.Linear(3*patch*patch, d)
        self.language = nn.Embedding(vocab, d)
        self.state = nn.Linear(action_dim, d)
        self.action_in = nn.Linear(action_dim, d)
        self.time = TimestepEmbedder(d)
        self.blocks = nn.ModuleList([PreNormBlock(d, heads, rope=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.action_out = nn.Linear(d, action_dim)

    def forward(self, image, language_ids, state, noisy_actions, t, return_attn=False):
        p = self.patch
        v = self.vision(F.unfold(image, kernel_size=p, stride=p).transpose(1, 2))
        l = self.language(language_ids)
        prefix = torch.cat([v, l], dim=1)
        s = self.state(state).unsqueeze(1)
        a = self.action_in(noisy_actions) + self.time(t.reshape(-1))[:, None]
        x = torch.cat([prefix, s, a], dim=1)
        mask = make_pi0_like_mask(prefix.size(1), 1, self.horizon, x.device)
        maps = []
        for block in self.blocks:
            x, w = block(x, mask)
            maps.append(w)
        velocity = self.action_out(self.norm(x[:, -self.horizon:]))
        return (velocity, maps, mask) if return_attn else velocity
