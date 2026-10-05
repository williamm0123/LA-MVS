"""Plane-sweep cost volume that *returns* its per-voxel support statistics.

``models/cost_volume.CostVolumeBuilder`` only returns the cost and parks
n_valid / source stats on mutable ``last_*`` attributes, which is fragile across
stages and batches (MoA.md §3). This builder returns them explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.cost_volume import group_wise_correlation
from utils.geometry import homography_warp_features


@dataclass
class CostVolumeOutput:
    cv: torch.Tensor         # [B, G, D, H, W] valid-weighted mean over sources
    n_valid: torch.Tensor    # [B, D, H, W] number of sources that see the voxel
    src_mean: torch.Tensor   # [B, D, H, W] mean over valid sources of the group-mean correlation
    src_std: torch.Tensor    # [B, D, H, W] std over valid sources (0 where < 2 sources)
    num_src: int

    def decoder_input(self) -> torch.Tensor:
        """cv plus the n_valid/S channel for the regularizer: [B, G+1, D, H, W]."""
        frac = (self.n_valid / max(self.num_src, 1)).unsqueeze(1).to(self.cv.dtype)
        return torch.cat([self.cv, frac], dim=1)


class MoACostVolume(nn.Module):
    def __init__(self, in_channels: int, warp_channels: int, num_groups: int = 8,
                 use_half: bool = True) -> None:
        super().__init__()
        if warp_channels % num_groups != 0:
            raise ValueError(f"warp_channels {warp_channels} must be divisible by num_groups {num_groups}")
        self.proj = nn.Conv2d(in_channels, warp_channels, kernel_size=1, bias=False)
        self.num_groups = num_groups
        self.use_half = use_half

    def _sample_dtype(self, ref: torch.Tensor) -> torch.dtype:
        if not (self.use_half and ref.is_cuda):
            return ref.dtype
        if torch.is_autocast_enabled():
            return torch.get_autocast_dtype("cuda")
        return torch.float16

    def forward(self, ref_feat: torch.Tensor, src_feats: torch.Tensor, K_ref: torch.Tensor,
                K_src: torch.Tensor, E_ref: torch.Tensor, E_src: torch.Tensor,
                depth_hypos: torch.Tensor, feature_stride: int) -> CostVolumeOutput:
        """ref_feat [B,C,H,W], src_feats [B,S,C,H,W], depth_hypos [B,D,H,W] (ascending)."""
        B, S = src_feats.shape[:2]
        dt = self._sample_dtype(ref_feat)
        ref_p = self.proj(ref_feat).to(dt)
        agg = None
        wsum = None
        gms, valids = [], []
        for s in range(S):
            src_p = self.proj(src_feats[:, s]).to(dt)
            warped, valid = homography_warp_features(
                src_p, K_ref, K_src[:, s], E_ref, E_src[:, s], depth_hypos, feature_stride,
                return_valid=True)
            cv_s = group_wise_correlation(ref_p, warped, self.num_groups).float()
            v = valid.float()                                              # [B,D,H,W]
            agg = cv_s * v.unsqueeze(1) if agg is None else agg + cv_s * v.unsqueeze(1)
            wsum = v if wsum is None else wsum + v
            gms.append(cv_s.detach().mean(dim=1))
            valids.append(v)
        cv = agg / wsum.clamp_min(1.0).unsqueeze(1)
        with torch.no_grad():
            gm = torch.stack(gms, dim=0)                                   # [S,B,D,H,W]
            vv = torch.stack(valids, dim=0)
            n = vv.sum(dim=0)
            mean = (gm * vv).sum(dim=0) / n.clamp_min(1.0)
            var = (((gm - mean) ** 2) * vv).sum(dim=0) / n.clamp_min(1.0)
            std = torch.where(n >= 2, var.clamp_min(0.0).sqrt(), torch.zeros_like(var))
        return CostVolumeOutput(cv=cv, n_valid=wsum.detach(), src_mean=mean, src_std=std, num_src=S)
