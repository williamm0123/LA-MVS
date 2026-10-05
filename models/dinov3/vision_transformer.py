"""Minimal DINOv3 ViT-B/16 feature extractor.

This file keeps only the inference path used by this project:
`vit_base(...).get_intermediate_layers(x, n=..., reshape=True, norm=True)`.
It intentionally omits training, heads, datasets, eval code, ConvNeXt, and hub
helpers from the upstream DINOv3 repository.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return x.reshape(-1, height, width, x.shape[-1])


def rope_rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return (x * cos) + (rope_rotate_half(x) * sin)


class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        base: float = 100.0,
        normalize_coords: str = "separate",
        dtype: torch.dtype = torch.bfloat16,
        device=None,
    ):
        super().__init__()
        if embed_dim % (4 * num_heads) != 0:
            raise ValueError("embed_dim must be divisible by 4 * num_heads")
        self.base = base
        self.normalize_coords = normalize_coords
        self.dtype = dtype
        self.d_head = embed_dim // num_heads
        self.register_buffer(
            "periods",
            torch.empty(self.d_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        periods = self.base ** (
            2
            * torch.arange(self.d_head // 4, device=self.periods.device, dtype=self.dtype)
            / (self.d_head // 2)
        )
        self.periods.data = periods

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        dtype = self.dtype
        device = self.periods.device
        if self.normalize_coords == "max":
            max_hw = max(H, W)
            coords_h = torch.arange(0.5, H, device=device, dtype=dtype) / max_hw
            coords_w = torch.arange(0.5, W, device=device, dtype=dtype) / max_hw
        elif self.normalize_coords == "min":
            min_hw = min(H, W)
            coords_h = torch.arange(0.5, H, device=device, dtype=dtype) / min_hw
            coords_w = torch.arange(0.5, W, device=device, dtype=dtype) / min_hw
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, device=device, dtype=dtype) / H
            coords_w = torch.arange(0.5, W, device=device, dtype=dtype) / W
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).flatten(0, 1)
        coords = 2.0 * coords - 1.0
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return torch.sin(angles), torch.cos(angles)


def _rope_apply_to_qk(q: Tensor, k: Tensor, rope: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    q_dtype = q.dtype
    k_dtype = k.dtype
    sin, cos = rope
    rope_dtype = sin.dtype
    q = q.to(dtype=rope_dtype)
    k = k.to(dtype=rope_dtype)
    num_tokens = q.shape[-2]
    prefix = num_tokens - sin.shape[-2]
    if prefix < 0:
        raise ValueError("RoPE sequence is longer than q/k sequence")
    q_prefix = q[:, :, :prefix, :]
    k_prefix = k[:, :, :prefix, :]
    q = rope_apply(q[:, :, prefix:, :], sin, cos)
    k = rope_apply(k[:, :, prefix:, :], sin, cos)
    q = torch.cat((q_prefix, q), dim=-2).to(dtype=q_dtype)
    k = torch.cat((k_prefix, k), dim=-2).to(dtype=k_dtype)
    return q, k


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, proj_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        batch, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, num_tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = torch.unbind(qkv, dim=2)
        q, k, v = [tensor.transpose(1, 2) for tensor in (q, k, v)]
        if rope is not None:
            q, k = _rope_apply_to_qk(q, k, rope)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(batch, num_tokens, channels)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class LayerScale(nn.Module):
    """Per-channel residual-branch scale (``blocks.*.ls{1,2}.gamma``).

    Not optional: the released DINOv3 ViT-B/16 ships these and they are nowhere
    near 1 — block 11's ls1 has median 6.20 and range [-28.1, 25.0], block 0's
    median is ~0. Stubbing them out as nn.Identity() (as this file used to) made
    load_state_dict(strict=False) drop all 24 tensors silently and produced
    patch features with ~0 cosine similarity to real DINOv3 output and almost no
    patch-to-patch contrast (0.026 vs 0.440) — i.e. a near-constant feature map.
    """

    def __init__(self, dim: int, init_values: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)
        self.ls1 = LayerScale(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * ffn_ratio), bias=ffn_bias)
        self.ls2 = LayerScale(dim)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        device=None,
        **ignored_kwargs,
    ):
        super().__init__()
        del ignored_kwargs
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.n_storage_tokens = n_storage_tokens

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))
        self.rope_embed = RopePositionEmbedding(embed_dim=embed_dim, num_heads=num_heads, device=device)
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Identity()

    def prepare_tokens_with_masks(self, x: Tensor, masks: Tensor | None = None) -> tuple[Tensor, tuple[int, int]]:
        x = self.patch_embed(x)
        batch, height, width, _ = x.shape
        x = x.flatten(1, 2)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(1, 0, cls_token.shape[-1], dtype=cls_token.dtype, device=cls_token.device)
        x = torch.cat(
            [
                cls_token.expand(batch, -1, -1),
                storage_tokens.expand(batch, -1, -1),
                x,
            ],
            dim=1,
        )
        return x, (height, width)

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int | Sequence[int] = 1) -> list[Tensor]:
        x, (height, width) = self.prepare_tokens_with_masks(x)
        if isinstance(n, int):
            blocks_to_take = range(len(self.blocks) - n, len(self.blocks))
        else:
            blocks_to_take = set(n)
        outputs = []
        for index, block in enumerate(self.blocks):
            rope = self.rope_embed(H=height, W=width)
            x = block(x, rope)
            if index in blocks_to_take:
                outputs.append(x)
        return outputs

    def get_intermediate_layers(
        self,
        x: Tensor,
        *,
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> tuple:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]

        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        patch_tokens = [out[:, self.n_storage_tokens + 1 :] for out in outputs]

        if reshape:
            batch, _, height, width = x.shape
            patch_tokens = [
                out.reshape(batch, height // self.patch_size, width // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in patch_tokens
            ]

        if not return_class_token and not return_extra_tokens:
            return tuple(patch_tokens)
        if return_class_token and not return_extra_tokens:
            return tuple(zip(patch_tokens, class_tokens))
        if not return_class_token and return_extra_tokens:
            return tuple(zip(patch_tokens, extra_tokens))
        return tuple(zip(patch_tokens, class_tokens, extra_tokens))

    def forward(self, x: Tensor) -> Tensor:
        output = self.get_intermediate_layers(x, n=1, reshape=False, norm=True)[0]
        return self.head(output[:, 0])


def vit_base(patch_size: int = 16, **kwargs) -> DinoVisionTransformer:
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4.0,
        **kwargs,
    )
