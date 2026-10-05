"""Mixture over {MVS, global mono, 3x3, 5x5, 7x7} + deterministic MVS override (MoA.md §8).

Expert 0 is always the MVS centre. The override is not a learned gate:

    alpha = stopgrad(r_mvs * conflict)
    pi~_mvs = pi_mvs + alpha (1 - pi_mvs),   pi~_j = (1 - alpha) pi_j

so reliable MVS that disagrees with the monocular proposal wins outright,
and the network cannot lower ``r_mvs`` or ``conflict`` to escape it.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.moa.geometry import spatial_grad

NUM_EXPERTS = 5


class MixtureHead(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 32, mvs_bias_init: float = 1.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, hidden), hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, hidden), hidden), nn.SiLU(),
            nn.Conv2d(hidden, NUM_EXPERTS, 1),
        )
        with torch.no_grad():
            last = self.net[-1]
            last.bias.zero_()
            last.bias[0] = float(mvs_bias_init)

    def forward(self, feats: torch.Tensor, expert_valid: torch.Tensor) -> torch.Tensor:
        """feats [B,C,H,W], expert_valid [B,5,H,W] (column 0 forced valid) -> pi [B,5,H,W]."""
        logits = self.net(feats.float())
        valid = expert_valid.bool().clone()
        valid[:, 0] = True
        logits = logits.masked_fill(~valid, float("-inf"))
        return torch.softmax(logits, dim=1)


def mono_proposal(pi: torch.Tensor, experts: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """Mixture of the monocular experts only, renormalized; ``fallback`` where all are off."""
    rest = 1.0 - pi[:, :1]
    num = (pi[:, 1:] * experts[:, 1:]).sum(dim=1, keepdim=True)
    has_mono = rest > 1e-6
    return torch.where(has_mono, num / torch.where(has_mono, rest, torch.ones_like(rest)), fallback)


def depth_conflict(x_prop: torch.Tensor, y: torch.Tensor, sigma_u: torch.Tensor, du: torch.Tensor,
                   t_d: float, T_d: float) -> torch.Tensor:
    gap = (x_prop - y).abs() / (sigma_u + du + 1e-8)
    return torch.sigmoid((gap - t_d) / T_d)


def prob_conflict(pv_at_prop: torch.Tensor, pmax: torch.Tensor) -> torch.Tensor:
    """1 - PV(x_prop)/Pmax; a proposal outside the parent axis has PV = 0 -> conflict 1."""
    return (1.0 - pv_at_prop / (pmax + 1e-8)).clamp(0.0, 1.0)


def shape_conflict(x_prop: torch.Tensor, y: torch.Tensor, edge: torch.Tensor, du: torch.Tensor,
                   tau_s: float) -> torch.Tensor:
    """Surface-gradient disagreement, only in the interior (0 on and next to edges)."""
    gxp, gyp, vx, vy = spatial_grad(x_prop)
    gxm, gym, _, _ = spatial_grad(y)
    g = (gxp - gxm).abs() * vx + (gyp - gym).abs() * vy
    c = 1.0 - torch.exp(-(g / (du + 1e-8)) / tau_s)
    return c * (edge < 0.5).float()


def soft_or(*terms: torch.Tensor) -> torch.Tensor:
    keep = torch.ones_like(terms[0])
    for t in terms:
        keep = keep * (1.0 - t)
    return 1.0 - keep


def scale_mono_weights(pi: torch.Tensor, gain: float) -> torch.Tensor:
    """Cap the mass the monocular experts may hold: pi_j *= gain (j != mvs), rest to MVS.

    A per-transition authority knob. At stage 4 the search window is already
    narrow, so a local affine's residual shape error is no longer a useful
    correction but noise around a nearly-correct MVS centre — hence a small
    gain there. It does NOT change the mono proposal used for conflict
    detection (all monocular experts are scaled by the same factor, so their
    renormalized mixture is unchanged), only how far the centre may move.
    """
    if gain >= 1.0:
        return pi
    mono = pi[:, 1:] * gain
    return torch.cat([1.0 - mono.sum(dim=1, keepdim=True), mono], dim=1)


def apply_mvs_override(pi: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Redistribute mixture mass toward expert 0 by ``alpha`` [B,1,H,W]; sums stay 1."""
    alpha = alpha.detach()
    out = pi * (1.0 - alpha)
    return torch.cat([pi[:, :1] + alpha * (1.0 - pi[:, :1]), out[:, 1:]], dim=1)
