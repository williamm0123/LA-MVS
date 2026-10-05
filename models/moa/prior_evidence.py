"""LAPE (5) candidate evidence: the monocular prior as probability mass on the current axis.

The prior at each pixel is the mixture of the monocular experts N(mu_j, sigma_j^2)
with the mixture head's monocular weights. On the stage's candidate axis u_i
(descending, uniform per pixel) each candidate receives the mass between its bin
edges (midpoints to the neighbours, half a spacing beyond the ends):

    q_i = sum_j pi_j [Phi((b_{i-1/2} - mu_j)/sigma_j) - Phi((b_{i+1/2} - mu_j)/sigma_j)],   M = sum_i q_i

M is the in-window mass. Below ``mass_min`` the prior is off at that pixel: a prior
that lies outside the window must not be renormalised into the window, where its
Gaussian tail would look like a confident vote for the edge candidate. Otherwise

    q~ = (1 - eps) q / M + eps / D,     l_i = clip(log q~_i - max log q~, -B, 0)

and the prior enters twice:

* before the 3D UNet: E_i = [l_i, (u_i - mu)/sigma, log(sigma/spacing), M] -> zero-init
  1x1x1 conv -> G channels added to the correlation channels, gated by g_P;
* before the softmax: L* = L_raw + gamma * l, gamma = gamma_max * sigmoid(h(.)), h a
  small conv head on detached statistics of the raw posterior and of the prior.

Both are zero where the matching is *verified* reliable (v_M: posterior mass within
+-1 bin of the argmax >= reliable_mass, normalised source disagreement at the argmax
<= reliable_src_std, and >= reliable_nvalid of the sources see that voxel). A sharp
peak alone is not a guard: two candidates' fused difference is (L_i - L_j) +
gamma (l_i - l_j), which a margin below gamma * B cannot defend.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_EVIDENCE = 4
N_GAMMA_IN = 14


def bin_edges(u_hyp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Upper / lower u edge of every candidate bin, [B,D,H,W] each (u_hyp descending)."""
    u = u_hyp.float()
    mid = 0.5 * (u[:, :-1] + u[:, 1:])
    top = u[:, :1] + 0.5 * (u[:, :1] - u[:, 1:2])
    bot = u[:, -1:] - 0.5 * (u[:, -2:-1] - u[:, -1:])
    return torch.cat([top, mid], 1), torch.cat([mid, bot], 1)


def normalized_weights(pi: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    w = pi.float() * valid.float()
    s = w.sum(dim=1, keepdim=True)
    return torch.where(s > 1e-6, w / s.clamp_min(1e-6), torch.zeros_like(w))


def bin_mass(u_hyp: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
             w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """u_hyp [B,D,H,W]; mu / sigma / w [B,J,H,W] (w normalised, 0 for invalid experts)."""
    up, lo = bin_edges(u_hyp)
    q = torch.zeros_like(up)
    for j in range(mu.shape[1]):
        m, s = mu[:, j:j + 1].float(), sigma[:, j:j + 1].float().clamp_min(1e-8)
        q = q + w[:, j:j + 1] * (torch.special.ndtr((up - m) / s) - torch.special.ndtr((lo - m) / s))
    return q.clamp_min(0.0), q.sum(dim=1, keepdim=True)


def mixture_moments(mu: torch.Tensor, sigma: torch.Tensor, w: torch.Tensor):
    m = (w * mu.float()).sum(1, keepdim=True)
    second = (w * (sigma.float() ** 2 + mu.float() ** 2)).sum(1, keepdim=True)
    s = (second - m * m).clamp_min(1e-12).sqrt()
    return m, s


def log_prior(q: torch.Tensor, M: torch.Tensor, mass_min: float, eps: float, clip: float):
    D = q.shape[1]
    active = M >= mass_min
    qt = (1.0 - eps) * q / M.clamp_min(1e-6) + eps / D
    lq = qt.clamp_min(1e-12).log()
    ell = (lq - lq.amax(dim=1, keepdim=True)).clamp(-clip, 0.0)
    return ell * active.float(), active


def evidence_channels(u_hyp, ell, mbar, sbar, M, active, du, clip: float) -> torch.Tensor:
    """[B,4,D,H,W]: log prior / B, (u - mu)/sigma / 8, log(sigma / spacing) / 4, in-window mass."""
    z = ((u_hyp.float() - mbar) / sbar.clamp_min(1e-8)).clamp(-8.0, 8.0) / 8.0
    D = u_hyp.shape[1]
    ls = torch.log(sbar / du.clamp_min(1e-8)).clamp(-4.0, 4.0).expand(-1, D, -1, -1) / 4.0
    E = torch.stack([ell / clip, z, ls, M.clamp(0.0, 1.0).expand(-1, D, -1, -1)], dim=1)
    return E * active.float().unsqueeze(1)


class PriorAdapter(nn.Module):
    """E [B,4,D,H,W] -> G channels added to the correlation volume (zero-init)."""

    def __init__(self, num_groups: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(N_EVIDENCE, num_groups, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        return self.conv(E)


class GammaHead(nn.Module):
    def __init__(self, n_in: int = N_GAMMA_IN, hidden: int = 32, gamma_max: float = 2.0,
                 bias_init: float = -4.0) -> None:
        super().__init__()
        self.gamma_max = float(gamma_max)
        self.net = nn.Sequential(
            nn.Conv2d(n_in, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, hidden), hidden), nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, float(bias_init))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.gamma_max * torch.sigmoid(self.net(feats.float()))


@torch.no_grad()
def posterior_features(logits_raw: torch.Tensor, u_hyp: torch.Tensor, src_std_n: torch.Tensor | None,
                       nvalid_frac: torch.Tensor | None) -> dict:
    """Detached statistics of the raw (pre-prior) posterior, all [B,1,H,W]."""
    p = F.softmax(logits_raw.detach().float(), dim=1)
    D = p.shape[1]
    pmax, idx = p.max(dim=1, keepdim=True)
    top2 = p.topk(min(2, D), dim=1).values
    gap = top2[:, :1] - top2[:, -1:] if D >= 2 else pmax
    ent = -(p * p.clamp_min(1e-12).log()).sum(1, keepdim=True) / math.log(max(D, 2))
    lo = (idx - 1).clamp(0, D - 1)
    hi = (idx + 1).clamp(0, D - 1)
    mass1 = p.gather(1, idx) + torch.where(lo != idx, p.gather(1, lo), torch.zeros_like(pmax)) \
        + torch.where(hi != idx, p.gather(1, hi), torch.zeros_like(pmax))
    u = u_hyp.float()
    mu = (p * u).sum(1, keepdim=True)
    sd = ((p * (u - mu) ** 2).sum(1, keepdim=True)).clamp_min(1e-12).sqrt()
    src = src_std_n.float().gather(1, idx) if src_std_n is not None else torch.zeros_like(pmax)
    nv = nvalid_frac.float().gather(1, idx) if nvalid_frac is not None else torch.ones_like(pmax)
    return {"pmax": pmax, "gap": gap, "entropy": ent, "mass1": mass1, "mu_u": mu, "sd_u": sd,
            "src": src, "nvalid": nv, "argmax": idx}


def reliable_mvs(pf: dict, cfg) -> torch.Tensor:
    return (pf["mass1"] >= cfg.reliable_mass) & (pf["src"] <= cfg.reliable_src_std) \
        & (pf["nvalid"] >= cfg.reliable_nvalid)
