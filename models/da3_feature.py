"""Depth Anything 3 as the frozen ViT in front of the SVA, plus the online monocular depth.

MonoMVSNet takes its monocular model's own encoder features instead of a separate
semantic backbone (reference/MonoMVSNet: Depth Anything V2's last-layer tokens are
added to the reference FPN). Here the same idea is applied to the MVSFormer++-style
pipeline this repo already has: DA3MONO-LARGE (DINOv2 ViT-L/14, out_layers 4/11/17/23)
replaces DINOv3 ViT-B/16, and everything downstream of the tokens is unchanged —
``SVAFusion`` (reference self-attention + source cross-attention over the layers),
``SVADeconv`` to the FPN width, injection at the 1/8 bottleneck, then the FMT pathway.

One DA3 forward per step serves two consumers:

* every view's intermediate tokens -> SVA -> FPN (matching features), and
* the DPT head on the **reference** view only -> monocular depth for LAPE, online,
  at the same images the network sees (augmentation, crop and scale included), so
  there is no offline cache to build, align or keep in sync any more.

DA3 is frozen and always in eval mode; its weights are reloaded from the pretrained
directory, never written into checkpoints (train_moa.FROZEN_PREFIXES).

Preprocessing follows DA3's ``InputProcessor`` (``upper_bound_resize``): the longest
side goes to ``process_res``, each side is rounded to the nearest multiple of 14,
ImageNet normalisation. Done here with one antialiased bicubic resize on the GPU
instead of DA3's PIL/cv2 two-step on the CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def da3_grid(H: int, W: int, process_res: int, patch: int = 14) -> tuple[int, int]:
    """DA3 ``upper_bound_resize``: longest side -> process_res, sides -> nearest multiple of patch."""
    s = float(process_res) / float(max(H, W))
    h, w = max(1, int(round(H * s))), max(1, int(round(W * s)))

    def nearest(x: int) -> int:
        lo = (x // patch) * patch
        hi = lo + patch
        return max(patch, lo if (x - lo) <= (hi - x) else hi)

    return nearest(h), nearest(w)


@dataclass
class DA3Output:
    tokens: list            # per selected layer [B, V, N, C] float32 (LayerNorm'ed, patch tokens only)
    grid: tuple             # (gh, gw) token grid
    depth: torch.Tensor     # [B, 1, H, W] float32 reference-view relative depth (DA3 units)
    valid: torch.Tensor     # [B, 1, H, W] bool: finite, > 0 and not sky
    process_hw: tuple       # (h, w) DA3 input size


class DA3Backbone(nn.Module):
    """Frozen DA3 (backbone + DPT head). ``net`` may be injected (tests)."""

    def __init__(self, weights_dir=None, process_res: int = 518, sky_threshold: float = 0.3,
                 layers: tuple[int, ...] = (0, 1, 2, 3), net: nn.Module | None = None) -> None:
        super().__init__()
        if net is None:
            if weights_dir is None or not Path(weights_dir).exists():
                raise FileNotFoundError(f"DA3 weights directory not found: {weights_dir} "
                                        f"(cfg.paths.da3_weights_file)")
            from depth_anything_3.api import DepthAnything3
            net = DepthAnything3.from_pretrained(str(weights_dir)).model
        self.net = net
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()
        self.process_res = int(process_res)
        self.sky_threshold = float(sky_threshold)
        self.layers = tuple(int(i) for i in layers)
        self.patch = int(getattr(self.net.head, "patch_size", 14))
        self.embed_dim = int(self.net.backbone.pretrained.embed_dim)
        n_out = len(self.net.backbone.out_layers)
        if any(i < 0 or i >= n_out for i in self.layers):
            raise ValueError(f"feat.sva_layers {self.layers} must index DA3's {n_out} out_layers")
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.net.eval()          # frozen: never leaves eval (no dropout / droppath)
        return self

    def _amp_dtype(self, device: torch.device):
        if device.type != "cuda":
            return None
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> DA3Output:
        """``images``: [B, V, 3, H, W] in [0, 1]. View 0 is the reference."""
        B, V, _, H, W = images.shape
        h, w = da3_grid(H, W, self.process_res, self.patch)
        x = images.reshape(B * V, 3, H, W).float()
        if (h, w) != (H, W):
            x = F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False,
                              antialias=(h < H or w < W)).clamp(0.0, 1.0)
        x = ((x - self.mean) / self.std).view(B, V, 3, h, w)
        amp = self._amp_dtype(images.device)
        with torch.autocast(device_type=images.device.type, dtype=amp or torch.float32,
                            enabled=amp is not None):
            feats, _ = self.net.backbone(x, cam_token=None, export_feat_layers=[])
        gh, gw = h // self.patch, w // self.patch
        # head on the reference view only, fp32 (as DA3 itself runs its heads)
        with torch.autocast(device_type=images.device.type, enabled=False):
            out = self.net.head([(f[:, :1].float(),) for f, _ in feats], h, w, patch_start_idx=0)
            depth = out["depth"].reshape(B, 1, h, w).float()
            sky = out["sky"].reshape(B, 1, h, w).float() if "sky" in out else None
        if (h, w) != (H, W):
            depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)
            if sky is not None:
                sky = F.interpolate(sky, size=(H, W), mode="bilinear", align_corners=False)
        valid = torch.isfinite(depth) & (depth > 0)
        if sky is not None:
            valid &= sky < self.sky_threshold
        depth = torch.where(valid, depth, torch.ones_like(depth))
        tokens = [feats[i][0].float() for i in self.layers]           # [B, V, N, C]
        return DA3Output(tokens=tokens, grid=(gh, gw), depth=depth, valid=valid, process_hw=(h, w))


class DA3SVA(nn.Module):
    """Frozen DA3 + SVA fusion + projection to the FPN width (``DinoSVA``'s replacement).

    Same interface as ``models.spre.DinoSVA`` where the network uses it
    (``forward`` -> fused tokens, ``fpn_feature``), plus the online mono depth.
    """

    def __init__(self, feat_cfg, fusion_cfg, weights_dir, fpn_channels: int, sva_cfg,
                 net: nn.Module | None = None) -> None:
        super().__init__()
        from models.spre import SVAFusion

        self.da3 = DA3Backbone(weights_dir, feat_cfg.process_res, feat_cfg.sky_threshold,
                               tuple(feat_cfg.sva_layers), net=net)
        self.fusion = SVAFusion(self.da3.embed_dim, fusion_cfg, n_layers=len(self.da3.layers))
        self.out_dim = self.fusion.out_dim
        self.full_sva = bool(sva_cfg is not None and sva_cfg.full)
        self.to_fpn = self.deconv = None
        if self.full_sva:
            from models.sva import SVADeconv
            self.deconv = SVADeconv(self.out_dim, fpn_channels)
        else:
            self.to_fpn = nn.Conv2d(self.out_dim, fpn_channels, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        self.da3.eval()
        return self

    def forward(self, images: torch.Tensor):
        """``images`` [B, V, 3, H, W] in [0, 1] -> (fused [B, V, N, dim], grid, DA3Output)."""
        d = self.da3(images)
        return self.fusion(d.tokens), d.grid, d

    def fpn_feature(self, fused: torch.Tensor, grid: tuple[int, int],
                    target_hw: tuple[int, int]) -> torch.Tensor:
        """[B, V, N, dim] -> [B, V, fpn_channels, *target_hw] for the 1/8 FPN injection."""
        B, V, _, C = fused.shape
        gh, gw = grid
        x = fused.reshape(B * V, gh, gw, C).permute(0, 3, 1, 2)
        x = self.deconv(x) if self.full_sva else self.to_fpn(x)
        if tuple(x.shape[-2:]) != tuple(target_hw):
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return x.reshape(B, V, -1, *target_hw)
