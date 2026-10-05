"""MVS evidence ``G = [N(CV), PV, |PV''|, N_valid, S_src]`` and its confidence (MoA.md §3-4).

Everything here runs in FP32. PV curvature is the second difference along the
depth index, which equals the second derivative only because MoA axes are
uniform in inverse depth; it is set to 0 at d=0 and d=D-1 (replicate padding
would fabricate a second-order response at the boundary).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def pv_curvature(prob: torch.Tensor) -> torch.Tensor:
    p = prob.float()
    c = torch.zeros_like(p)
    if p.shape[1] >= 3:
        c[:, 1:-1] = (p[:, 2:] - 2.0 * p[:, 1:-1] + p[:, :-2]).abs()
    return c


def posterior_stats(prob: torch.Tensor, u_hyp: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-pixel posterior summaries, all [B,1,H,W] (``mode_idx`` long)."""
    p = prob.float()
    D = p.shape[1]
    pmax, mode_idx = p.max(dim=1, keepdim=True)
    if D >= 2:
        top2 = p.topk(2, dim=1).values
        gap = top2[:, :1] - top2[:, 1:2]
    else:
        gap = pmax
    ent = -(p * p.clamp_min(1e-12).log()).sum(dim=1, keepdim=True) / math.log(max(D, 2))
    u = u_hyp.float()
    mu = (p * u).sum(dim=1, keepdim=True)
    sigma_u = ((p * (u - mu) ** 2).sum(dim=1, keepdim=True)).clamp_min(1e-12).sqrt()
    return {"pmax": pmax, "mode_idx": mode_idx, "gap": gap, "entropy": ent, "sigma_u": sigma_u}


def gather_depth(volume: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """volume [B,D,H,W] -> [B,1,H,W] or [B,C,D,H,W] -> [B,C,H,W] at index ``idx`` [B,1,H,W]."""
    if volume.dim() == 4:
        return volume.gather(1, idx)
    C = volume.shape[1]
    return volume.gather(2, idx.unsqueeze(1).expand(-1, C, -1, -1, -1)).squeeze(2)


def normalize_cost(cv: torch.Tensor) -> torch.Tensor:
    """z-score each group's correlation along D (stage-to-stage magnitudes differ ~200x)."""
    cv = cv.float()
    mu = cv.mean(dim=2, keepdim=True)
    sd = cv.std(dim=2, keepdim=True, unbiased=False)
    return (cv - mu) / (sd + 1e-4)


def normalize_src_std(src_std: torch.Tensor, cv: torch.Tensor) -> torch.Tensor:
    """Cross-source disagreement relative to the along-D spread of the matching signal."""
    gm = cv.float().mean(dim=1)                                  # [B,D,H,W]
    ref = gm.std(dim=1, keepdim=True, unbiased=False) + 1e-4
    return (src_std.float() / ref).clamp(0.0, 10.0)


class MVSEvidenceEncoder(nn.Module):
    """Conv3d x2 over the evidence stack, then PV-weighted pooling over depth."""

    def __init__(self, num_groups: int, dim: int = 16, hidden: int = 16) -> None:
        super().__init__()
        cin = num_groups + 4
        self.net = nn.Sequential(
            nn.Conv3d(cin, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, hidden), hidden), nn.SiLU(),
            nn.Conv3d(hidden, dim, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, dim), dim), nn.SiLU(),
        )
        self.dim = dim

    def forward(self, cv_norm: torch.Tensor, prob: torch.Tensor, curv: torch.Tensor,
                n_valid_frac: torch.Tensor, src_std_n: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (V [B,Ce,D,H,W], F_mvs [B,Ce,H,W])."""
        g = torch.cat([cv_norm, prob.unsqueeze(1), curv.unsqueeze(1),
                       n_valid_frac.unsqueeze(1), src_std_n.unsqueeze(1)], dim=1).float()
        V = self.net(g)
        F_mvs = (V * prob.float().unsqueeze(1)).sum(dim=2)
        return V, F_mvs


class MVSConfidenceHead(nn.Module):
    """r_mvs = sigmoid(h([F_mvs, Pmax, H, P1-P2, curv_peak, sigma/du, n_valid_peak, S_src_peak]))."""

    NUM_STATS = 7

    def __init__(self, feat_dim: int, hidden: int = 32) -> None:
        super().__init__()
        cin = feat_dim + self.NUM_STATS
        self.net = nn.Sequential(
            nn.Conv2d(cin, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, hidden), hidden), nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, F_mvs: torch.Tensor, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logit = self.net(torch.cat([F_mvs, stats], dim=1).float())
        return logit, torch.sigmoid(logit)
