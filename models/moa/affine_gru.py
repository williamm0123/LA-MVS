"""LAPE-GRU: recurrent refinement of the monocular local affine field (alpha, b~).

The window experts (models/moa/lape_affine.py) solve, once and in closed form, the
weighted least squares

    min_{alpha, b~}  sum_q w_q (d_q - alpha x~_q - b~)^2,   d = y - x,  x~_q = x_q - x_p

per pixel p, and the mixture can only pick a convex combination of their centres.
Here the same centred parameters are *iterated* by a ConvGRU:

    fit -> look up the matching evidence at the current mono depth -> update -> refit

* the state is (alpha, b~) per pixel, the mono prediction at p is x_p + b~ and on the
  child grid ``x_c + b~ + alpha (x_c - x_p)`` (``LAPECascade.lift``), so the detail
  below the working resolution still comes from DA3. Centred, not raw (a, b): with
  x in [0, 1] and a window variance of ~1e-5, a and b are nearly collinear;
* "refit" costs nothing: the objective is quadratic in (alpha, b~), so its gradient
  and residual at any state are functions of the experts' window sums
  (Sw, Sx, Sxx, Sd, Sxd, Sdd), which ``WindowAffineExperts`` already computes;
* the matching evidence the closed-form fit never sees: the parent stage's
  normalised correlation and raw posterior, interpolated on the parent axis at
  x_p + b~ + k * spacing (k = -r..r), plus whether those points lie inside the axis;
* initial state = inverse-variance mix of {E_RAC (alpha = b~ = 0), w3, w7, w11};
  the update head is zero-initialised, so at step 0 the GRU expert is that mix.

Every input is detached (the LAPE convention): the GRU's losses train only LAPE.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moa.geometry import interp_along_axis, spatial_grad

N_FIT = 12          # 3 windows x (mean residual, slope residual, rms, support)
N_GEO = 12


class SepConvGRU(nn.Module):
    """RAFT's separable ConvGRU (1x5 then 5x1)."""

    def __init__(self, hidden: int, inp: int) -> None:
        super().__init__()
        c = hidden + inp
        self.convz1 = nn.Conv2d(c, hidden, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(c, hidden, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(c, hidden, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(c, hidden, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(c, hidden, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(c, hidden, (5, 1), padding=(2, 0))

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        for cz, cr, cq in ((self.convz1, self.convr1, self.convq1), (self.convz2, self.convr2, self.convq2)):
            hx = torch.cat([h, x], dim=1)
            z = torch.sigmoid(cz(hx))
            r = torch.sigmoid(cr(hx))
            q = torch.tanh(cq(torch.cat([r * h, x], dim=1)))
            h = (1.0 - z) * h + z * q
        return h


def wls_residual_feats(sums: torch.Tensor, a: torch.Tensor, b: torch.Tensor, dg: float):
    """sums [B,3,6,H,W] = (Sw, Sx, Sxx, Sd, Sxd, Sdd) per window; a, b [B,1,H,W].

    Returns (features [B,9,H,W], rms [B,3,H,W] in u): the mean residual (offset
    gradient / Sw, stage-1 bins), the slope residual (alpha gradient / Sxx) and the
    weighted rms of the current model in every window."""
    Sw, Sx, Sxx, Sd, Sxd, Sdd = sums.float().unbind(2)
    sw = Sw.clamp_min(1e-6)
    g_b = (Sd - a * Sx - b * Sw) / sw / dg
    g_a = (Sxd - a * Sxx - b * Sx) / Sxx.clamp_min(1e-10)
    sse = (Sdd - 2 * a * Sxd - 2 * b * Sd + a * a * Sxx + 2 * a * b * Sx + b * b * Sw).clamp_min(0.0)
    rms = torch.where(Sw > 1e-6, (sse / sw).sqrt(), torch.zeros_like(Sw))
    feats = torch.cat([g_b.clamp(-20, 20) / 20, g_a.clamp(-10, 10) / 10,
                       (rms / dg).clamp(max=20) / 20], dim=1)
    return feats, rms


@dataclass
class AffineGRUOutput:
    alpha: torch.Tensor           # [B,1,h,w] final slope correction (grad -> GRU)
    bt: torch.Tensor              # [B,1,h,w] final centre offset (grad -> GRU)
    sigma: torch.Tensor           # [B,1,h,w] std of the GRU expert in u (grad -> GRU)
    seq: list                     # x + b~ after every iteration (grad through that step)


def inverse_variance_init(alphas: torch.Tensor, bts: torch.Tensor, sigma: torch.Tensor,
                          valid: torch.Tensor, floor: float):
    """Mix of the experts' (alpha, b~) with weights valid / sigma^2. [B,J,H,W] each."""
    iv = valid.float() / sigma.float().clamp_min(1e-6) ** 2
    den = iv.sum(1, keepdim=True)
    ok = den > 0
    dsafe = torch.where(ok, den, torch.ones_like(den))
    a0 = torch.where(ok, (iv * alphas).sum(1, keepdim=True) / dsafe, torch.zeros_like(den))
    b0 = torch.where(ok, (iv * bts).sum(1, keepdim=True) / dsafe, torch.zeros_like(den))
    s0 = torch.where(ok, dsafe.rsqrt(), torch.full_like(den, 1.0))
    return a0, b0, s0.clamp_min(floor)


class AffineGRU(nn.Module):
    def __init__(self, num_groups: int, ctx_in: int, hidden: int = 64, radius: int = 2,
                 alpha_step: float = 0.1, b_step_bins: float = 2.0) -> None:
        super().__init__()
        self.ks = tuple(range(radius, -radius - 1, -1))
        n_look = len(self.ks) * (num_groups + 2)
        self.ctx = nn.Sequential(nn.Conv2d(ctx_in, 2 * hidden, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(2 * hidden, 2 * hidden, 3, padding=1))
        self.enc = nn.Sequential(nn.Conv2d(N_FIT + n_look + N_GEO, 96, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(96, hidden, 3, padding=1), nn.SiLU())
        self.gru = SepConvGRU(hidden, 2 * hidden)
        self.head = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(hidden, 3, 3, padding=1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.alpha_step = float(alpha_step)
        self.b_step = float(b_step_bins)

    def _inputs(self, x, xv, y, du, dg, sums, support, vol, u_hyp, a, b, edge, r, mu_e, sig_e, val_e):
        xh = x + b
        fit, _ = wls_residual_feats(sums, a, b, dg)
        look = []
        for k in self.ks:
            v, ins = interp_along_axis(vol, u_hyp, xh + k * du)
            look += [v, ins.float()]
        gx1, gy1, _, _ = spatial_grad(xh)
        gx2, gy2, _, _ = spatial_grad(y)
        rel = ((mu_e - xh) / sig_e.clamp_min(1e-8)).clamp(-8, 8) / 8 * val_e
        geo = [((xh - y) / du).clamp(-20, 20) / 20, a, (b / dg).clamp(-10, 10) / 10,
               ((gx1 - gx2) / du).clamp(-10, 10) / 10, ((gy1 - gy2) / du).clamp(-10, 10) / 10,
               edge, xv, r, rel]
        return torch.cat([fit, support] + look + geo, dim=1)

    def forward(self, iters: int, *, x, xv, y, du, dg: float, sums, support, vol, u_hyp, ctx, a0, b0,
                s0, edge, r, mu_e, sig_e, val_e, a_lim: tuple[float, float], b_max: float,
                floor: float) -> AffineGRUOutput:
        """All maps [B,C,h,w] at the parent resolution, every input detached; ``vol``
        [B,G+1,D,h,w] = (normalised correlation, raw posterior) on ``u_hyp``."""
        h, c = self.ctx(ctx).chunk(2, dim=1)
        h, c = torch.tanh(h), F.relu(c)
        mv = xv > 0.5
        a, b = a0, b0
        seq = []
        o = None
        for _ in range(iters):
            a, b = a.detach(), b.detach()
            feat = self._inputs(x, xv, y, du, dg, sums, support, vol, u_hyp, a, b, edge, r, mu_e, sig_e, val_e)
            h = self.gru(h, torch.cat([self.enc(feat), c], dim=1))
            o = self.head(h).float()
            a = torch.where(mv, (a + self.alpha_step * torch.tanh(o[:, 0:1])).clamp(*a_lim), a)
            b = torch.where(mv, (b + self.b_step * du * torch.tanh(o[:, 1:2])).clamp(-b_max, b_max), b)
            seq.append(x + b)
        # sigma: residual of the final model in the 7x7 window, rescaled by the head
        _, rms = wls_residual_feats(sums, a.detach(), b.detach(), dg)
        base = (rms[:, 1:2] ** 2 + s0 ** 2).sqrt()
        scale = torch.exp(o[:, 2:3].clamp(-6.0, 6.0)) if o is not None else torch.ones_like(base)
        sigma = (base * scale).clamp_min(floor)
        return AffineGRUOutput(alpha=a, bt=b, sigma=sigma, seq=seq)
