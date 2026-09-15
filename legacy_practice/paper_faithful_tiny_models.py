import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(q, k):
    head_dim = q.size(-1)
    assert head_dim % 2 == 0

    positions = torch.arange(
        q.size(-2),
        device=q.device,
        dtype=q.dtype,
    )
    frequencies = torch.arange(
        0,
        head_dim,
        2,
        device=q.device,
        dtype=q.dtype,
    )
    inverse_frequencies = 1.0 / (10000 ** (frequencies / head_dim))

    angles = positions[:, None] * inverse_frequencies[None, :]
    cos = angles.cos()[None, None]
    sin = angles.sin()[None, None]

    def rotate(x):
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated = torch.stack(
            (
                even * cos - odd * sin,
                even * sin + odd * cos,
            ),
            dim=-1,
        )
        return rotated.flatten(-2)

    return rotate(q), rotate(k)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, use_rope=False):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_rope = use_rope

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, allowed_mask=None):
        batch_size, sequence_length, hidden_dim = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.use_rope:
            q, k = apply_rope(q, k)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        if allowed_mask is not None:
            if allowed_mask.ndim == 2:
                allowed_mask = allowed_mask[None, None]
            elif allowed_mask.ndim == 3:
                allowed_mask = allowed_mask[:, None]

            scores = scores.masked_fill(
                ~allowed_mask,
                torch.finfo(scores.dtype).min,
            )

        weights = scores.softmax(dim=-1)
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.reshape(batch_size, sequence_length, hidden_dim)

        return self.out_projection(attended), weights


class PreNormBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4, use_rope=False):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = SelfAttention(
            hidden_dim,
            num_heads,
            use_rope=use_rope,
        )

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_ratio * hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_dim, hidden_dim),
        )

    def forward(self, x, allowed_mask=None):
        attention_output, weights = self.attention(
            self.norm1(x),
            allowed_mask,
        )
        x = x + attention_output

        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output

        return x, weights


class TinyGPT(nn.Module):
    """GPT-2-like decoder with the important computation graph preserved."""

    def __init__(
        self,
        vocab_size=32,
        max_length=16,
        hidden_dim=24,
        num_heads=3,
        depth=2,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                PreNormBlock(hidden_dim, num_heads)
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens, return_attn=False):
        sequence_length = tokens.size(1)
        positions = torch.arange(sequence_length, device=tokens.device)

        x = self.token_embedding(tokens)
        x = x + self.position_embedding(positions)[None]

        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=tokens.device,
            )
        )

        attention_maps = []
        for block in self.blocks:
            x, weights = block(x, causal_mask)
            attention_maps.append(weights)

        x = self.final_norm(x)

        # GPT-style weight tying: the token embedding matrix is reused as LM head.
        logits = F.linear(x, self.token_embedding.weight)

        if return_attn:
            return logits, attention_maps
        return logits


class TinyViT(nn.Module):
    """ViT-like classifier: patches + CLS + learned positions + encoder."""

    def __init__(
        self,
        image_size=16,
        patch_size=4,
        hidden_dim=24,
        num_heads=3,
        depth=2,
        num_classes=3,
    ):
        super().__init__()

        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2

        self.patch_projection = nn.Linear(
            3 * patch_size * patch_size,
            hidden_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, 1 + num_patches, hidden_dim)
        )

        self.blocks = nn.ModuleList(
            [
                PreNormBlock(hidden_dim, num_heads)
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, image, return_attn=False):
        patches = F.unfold(
            image,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        patches = patches.transpose(1, 2)

        x = self.patch_projection(patches)
        cls = self.cls_token.expand(image.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.position_embedding[:, : x.size(1)]

        attention_maps = []
        for block in self.blocks:
            x, weights = block(x)
            attention_maps.append(weights)

        cls_hidden = self.final_norm(x)[:, 0]
        logits = self.classifier(cls_hidden)

        if return_attn:
            return logits, attention_maps
        return logits


def sinusoidal_1d(positions, embedding_dim):
    assert embedding_dim % 2 == 0

    positions = positions.float().reshape(-1, 1)
    omega = torch.arange(
        embedding_dim // 2,
        device=positions.device,
        dtype=torch.float32,
    )
    omega = omega / (embedding_dim // 2)
    omega = 1.0 / (10000 ** omega)

    angles = positions * omega[None]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def sincos_grid_2d(height, width, hidden_dim, device):
    assert hidden_dim % 4 == 0

    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )

    embedding_y = sinusoidal_1d(
        grid_y.reshape(-1),
        hidden_dim // 2,
    )
    embedding_x = sinusoidal_1d(
        grid_x.reshape(-1),
        hidden_dim // 2,
    )

    return torch.cat([embedding_y, embedding_x], dim=-1)


def sincos_grid_3d(depth, height, width, hidden_dim, device):
    assert hidden_dim % 6 == 0

    grid_z, grid_y, grid_x = torch.meshgrid(
        torch.arange(depth, device=device),
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )

    axis_dim = hidden_dim // 3
    embedding_z = sinusoidal_1d(grid_z.reshape(-1), axis_dim)
    embedding_y = sinusoidal_1d(grid_y.reshape(-1), axis_dim)
    embedding_x = sinusoidal_1d(grid_x.reshape(-1), axis_dim)

    return torch.cat(
        [embedding_z, embedding_y, embedding_x],
        dim=-1,
    )


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, frequency_dim=32):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        frequency_embedding = sinusoidal_1d(
            t * 1000.0,
            self.frequency_dim,
        )
        return self.mlp(frequency_embedding)


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    """DiT adaLN-Zero block following the official DiT structure."""

    def __init__(self, hidden_dim, num_heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            hidden_dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.attention = SelfAttention(hidden_dim, num_heads)

        self.norm2 = nn.LayerNorm(
            hidden_dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        self.ada_ln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )

        nn.init.zeros_(self.ada_ln_modulation[-1].weight)
        nn.init.zeros_(self.ada_ln_modulation[-1].bias)

    def forward(self, x, condition, return_attn=False):
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.ada_ln_modulation(condition).chunk(6, dim=-1)

        attention_input = modulate(
            self.norm1(x),
            shift_attn,
            scale_attn,
        )
        attention_output, weights = self.attention(attention_input)
        x = x + gate_attn[:, None] * attention_output

        mlp_input = modulate(
            self.norm2(x),
            shift_mlp,
            scale_mlp,
        )
        mlp_output = self.mlp(mlp_input)
        x = x + gate_mlp[:, None] * mlp_output

        if return_attn:
            return x, weights
        return x


class DiTFinalLayer(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()

        self.norm = nn.LayerNorm(
            hidden_dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.ada_ln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        nn.init.zeros_(self.ada_ln_modulation[-1].weight)
        nn.init.zeros_(self.ada_ln_modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x, condition):
        shift, scale = self.ada_ln_modulation(condition).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.output_projection(x)


class Tiny2DDiT(nn.Module):
    """DiT backbone with Flow-Matching velocity regression."""

    def __init__(
        self,
        channels=2,
        image_size=8,
        patch_size=2,
        hidden_dim=24,
        num_heads=3,
        depth=2,
    ):
        super().__init__()

        self.channels = channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        patch_dim = channels * patch_size * patch_size

        self.input_projection = nn.Linear(patch_dim, hidden_dim)
        self.time_embedding = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_dim, num_heads)
                for _ in range(depth)
            ]
        )
        self.final_layer = DiTFinalLayer(hidden_dim, patch_dim)

    def forward(self, x, t, return_attn=False):
        patches = F.unfold(
            x,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        patches = patches.transpose(1, 2)

        grid_size = self.image_size // self.patch_size
        position_embedding = sincos_grid_2d(
            grid_size,
            grid_size,
            self.hidden_dim,
            x.device,
        ).to(x.dtype)

        hidden = self.input_projection(patches)
        hidden = hidden + position_embedding[None]
        condition = self.time_embedding(t.reshape(-1))

        attention_maps = []
        for block in self.blocks:
            if return_attn:
                hidden, weights = block(
                    hidden,
                    condition,
                    return_attn=True,
                )
                attention_maps.append(weights)
            else:
                hidden = block(hidden, condition)

        output_patches = self.final_layer(hidden, condition)
        output_patches = output_patches.transpose(1, 2)

        output = F.fold(
            output_patches,
            output_size=(self.image_size, self.image_size),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        if return_attn:
            return output, attention_maps
        return output


def patchify_3d(x, patch_size):
    batch_size, channels, depth, height, width = x.shape

    assert depth % patch_size == 0
    assert height % patch_size == 0
    assert width % patch_size == 0

    grid_depth = depth // patch_size
    grid_height = height // patch_size
    grid_width = width // patch_size

    x = x.reshape(
        batch_size,
        channels,
        grid_depth,
        patch_size,
        grid_height,
        patch_size,
        grid_width,
        patch_size,
    )
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()

    num_patches = grid_depth * grid_height * grid_width
    patch_dim = channels * patch_size**3

    return x.reshape(batch_size, num_patches, patch_dim)


def unpatchify_3d(tokens, channels, volume_size, patch_size):
    batch_size = tokens.size(0)
    grid_size = volume_size // patch_size

    x = tokens.reshape(
        batch_size,
        grid_size,
        grid_size,
        grid_size,
        channels,
        patch_size,
        patch_size,
        patch_size,
    )
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()

    return x.reshape(
        batch_size,
        channels,
        volume_size,
        volume_size,
        volume_size,
    )


class Tiny3DDiT(nn.Module):
    """Educational 3D extension of the DiT block design."""

    def __init__(
        self,
        channels=1,
        volume_size=4,
        patch_size=2,
        hidden_dim=24,
        num_heads=3,
        depth=2,
    ):
        super().__init__()

        self.channels = channels
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        patch_dim = channels * patch_size**3

        self.input_projection = nn.Linear(patch_dim, hidden_dim)
        self.time_embedding = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_dim, num_heads)
                for _ in range(depth)
            ]
        )
        self.final_layer = DiTFinalLayer(hidden_dim, patch_dim)

    def forward(self, x, t):
        grid_size = self.volume_size // self.patch_size

        patches = patchify_3d(x, self.patch_size)
        position_embedding = sincos_grid_3d(
            grid_size,
            grid_size,
            grid_size,
            self.hidden_dim,
            x.device,
        ).to(x.dtype)

        hidden = self.input_projection(patches)
        hidden = hidden + position_embedding[None]
        condition = self.time_embedding(t.reshape(-1))

        for block in self.blocks:
            hidden = block(hidden, condition)

        output_patches = self.final_layer(hidden, condition)

        return unpatchify_3d(
            output_patches,
            self.channels,
            self.volume_size,
            self.patch_size,
        )


def make_pi0_like_mask(
    prefix_length,
    state_length,
    action_length,
    device,
):
    total_length = prefix_length + state_length + action_length

    allowed = torch.zeros(
        total_length,
        total_length,
        dtype=torch.bool,
        device=device,
    )

    # Image/language prefix is bidirectional inside the prefix block.
    allowed[:prefix_length, :prefix_length] = True

    # State can read the prefix and itself.
    state_start = prefix_length
    state_end = state_start + state_length
    allowed[state_start:state_end, :state_end] = True

    # Action suffix can read all prefix/state/action tokens.
    action_start = state_end
    allowed[action_start:, :] = True

    return allowed


class TinyVLAFlowPolicy(nn.Module):
    """Tiny pi0/openpi-like joint-attention flow policy."""

    def __init__(
        self,
        image_size=16,
        patch_size=8,
        vocab_size=32,
        hidden_dim=24,
        num_heads=3,
        depth=2,
        action_dim=4,
        action_horizon=4,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.action_horizon = action_horizon

        self.vision_projection = nn.Linear(
            3 * patch_size * patch_size,
            hidden_dim,
        )
        self.language_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.state_projection = nn.Linear(action_dim, hidden_dim)
        self.action_input_projection = nn.Linear(action_dim, hidden_dim)
        self.time_embedding = TimestepEmbedder(hidden_dim)

        self.blocks = nn.ModuleList(
            [
                PreNormBlock(
                    hidden_dim,
                    num_heads,
                    use_rope=True,
                )
                for _ in range(depth)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.action_output_projection = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        image,
        language_ids,
        state,
        noisy_actions,
        t,
        return_attn=False,
    ):
        image_patches = F.unfold(
            image,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        image_patches = image_patches.transpose(1, 2)
        image_tokens = self.vision_projection(image_patches)

        language_tokens = self.language_embedding(language_ids)
        prefix = torch.cat([image_tokens, language_tokens], dim=1)

        state_token = self.state_projection(state).unsqueeze(1)

        action_tokens = self.action_input_projection(noisy_actions)
        action_tokens = action_tokens + self.time_embedding(
            t.reshape(-1)
        )[:, None]

        x = torch.cat(
            [prefix, state_token, action_tokens],
            dim=1,
        )

        allowed_mask = make_pi0_like_mask(
            prefix_length=prefix.size(1),
            state_length=1,
            action_length=self.action_horizon,
            device=x.device,
        )

        attention_maps = []
        for block in self.blocks:
            x, weights = block(x, allowed_mask)
            attention_maps.append(weights)

        action_hidden = x[:, -self.action_horizon :]
        action_hidden = self.final_norm(action_hidden)
        velocity = self.action_output_projection(action_hidden)

        if return_attn:
            return velocity, attention_maps, allowed_mask
        return velocity
