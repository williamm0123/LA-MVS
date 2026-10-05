"""Monotone sigma calibrator for the LAPE monocular experts.

    log sigma = log s_base + h(f),   s_base = sqrt(s_ho^2 + sigma_par^2)

``f`` = [log(s_ho / D1), log(sigma_par / D1), log1p(delta / delta0) | log1p(n_eff), edge, flag,
r, r_img] (D1 = stage-1 bin in u; r = MoA confidence of the MVS at the pixel, r_img = its mean
over the sample's valid monocular pixels). The confidence matters: the residual scales are
measured against the MVS anchors, so early in training — or wherever the MVS is wrong
but self-consistent — they are small for a prior that is off by a lot; only "how much is
the MVS to be trusted here" can tell the calibrator that.

``h`` is a one-hidden-layer MLP that is non-decreasing in
the first three inputs: their first-layer weights and every second-layer weight go
through softplus, and the activation (softplus) is increasing. So a larger held-out
residual, a larger parameter variance or a farther anchor can never make the expert
*more* confident, whatever the training data says; the free inputs only shift it.

The last layer starts at ~0, i.e. sigma starts at the analytic s_base, and the
Gaussian NLL in losses/moa_loss.py moves it to the observed error scale. Inputs are
detached by the caller: the calibrator learns sigma, it cannot move the experts.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_FEAT = 8
N_MONO = 3


class SigmaCalibrator(nn.Module):
    def __init__(self, n_in: int = N_FEAT, n_mono: int = N_MONO, hidden: int = 16) -> None:
        super().__init__()
        self.n_mono = int(n_mono)
        self.w1_mono = nn.Parameter(torch.full((hidden, n_mono), -1.0))
        self.w1_free = nn.Parameter(torch.randn(hidden, n_in - n_mono) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = nn.Parameter(torch.full((hidden,), -6.0))
        self.b2 = nn.Parameter(torch.zeros(()))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats [B, n_in, H, W] -> h [B, 1, H, W] (added to log s_base)."""
        f = feats.float()
        w1 = torch.cat([F.softplus(self.w1_mono), self.w1_free], dim=1)       # [hid, n_in]
        z = torch.einsum("kc,bchw->bkhw", w1, f) + self.b1.view(1, -1, 1, 1)
        h = torch.einsum("k,bkhw->bhw", F.softplus(self.w2), F.softplus(z)) + self.b2
        return h.unsqueeze(1)


def calibrated_sigma(cal: SigmaCalibrator, s_ho: torch.Tensor, sigma_par: torch.Tensor,
                     delta: torch.Tensor, n_eff: torch.Tensor, edge: torch.Tensor, flag: torch.Tensor,
                     du_global: float, delta0: float, floor_bins: float, r: torch.Tensor | None = None,
                     r_img: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """All [B,1,H,W] (detached inputs). Returns (sigma, features) in u units."""
    if r is None:
        r = torch.ones_like(s_ho)
    if r_img is None:
        r_img = torch.ones_like(s_ho)
    d1 = float(du_global)
    s_ho = s_ho.detach().float().clamp_min(1e-4 * d1)
    sp = sigma_par.detach().float().clamp_min(1e-4 * d1)
    feats = torch.cat([
        (s_ho / d1).log(), (sp / d1).log(), torch.log1p(delta.detach().float() / delta0),
        torch.log1p(n_eff.detach().float()), edge.detach().float(), flag.detach().float(),
        r.detach().float().expand_as(s_ho), r_img.detach().float().expand_as(s_ho)], dim=1)
    base = (s_ho * s_ho + sp * sp).sqrt()
    sig = base * torch.exp(cal(feats).clamp(-6.0, 6.0))
    sig = (sig * sig + (floor_bins * d1) ** 2).sqrt()
    return sig, feats
