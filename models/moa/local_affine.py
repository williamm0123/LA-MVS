"""Shape-conditioned 3x3 / 5x5 / 7x7 local affine fields (MoA.md §7).

Per pixel p and scale s, solve in normalized inverse depth

    min_{a,b} sum_q w_pq (a x_q + b - y_q)^2 + lam_a (a-1)^2 + lam_b b^2

with ``x`` the globally aligned monocular map and ``y`` the parent-stage MVS
map. The ridge is centred on the identity (a=1, b=0), so in the spec's normal
equations the first right-hand side is ``Sxy + lam_a`` — not ``Sxy``.

Numerics: written as in MoA.md §7.3 (raw sums of x, x^2, xy) the 2x2 system
cancels catastrophically in FP32, because x ~ 0.5 while its spread inside a
window is ~1e-3: det = Sw*Sxx - Sx^2 subtracts two ~10 numbers to get ~1e-4.
``_solve`` minimizes the *same* objective, re-parameterized around the
identity and the centre pixel (alpha = a - 1, x~ = x - x_p, d = y - x), so all
sums are small differences. ``spec_closed_form`` is the literal §7.3 formula,
kept for the equivalence test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from models.moa.edge import shifted_views


class _ResBlock(nn.Module):
    def __init__(self, ch: int, dilation: int) -> None:
        super().__init__()
        g = min(8, ch)
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(g, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(g, ch),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class ShapeEncoder(nn.Module):
    """Monocular shape features -> unit-norm embedding e(p) [B, Ce, H, W]."""

    def __init__(self, in_ch: int, width: int = 32, out_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, width), width), nn.SiLU(),
            _ResBlock(width, 1),
            _ResBlock(width, 2),
            nn.Conv2d(width, out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.net(x)
        return e / e.norm(dim=1, keepdim=True).clamp_min(1e-6)


@dataclass
class LocalAffineResult:
    a: torch.Tensor            # [B, S, H, W]
    b: torch.Tensor            # [B, S, H, W]
    valid: torch.Tensor        # [B, S, H, W] float, solve accepted (incl. offset-only)
    offset_only: torch.Tensor  # [B, S, H, W] float, flat region -> a fixed to 1
    support: torch.Tensor      # [B, S, H, W] n_eff / window area
    x_fit: torch.Tensor        # [B, S, H, W] a * x + b


class MultiScaleLocalAffine(nn.Module):
    """Closed-form 2x2 weighted least squares at every pixel, S scales at once.

    Weights ``w_pq = r(q) * pass(p,q) * K_s(p-q) * exp(-(1 - e_p.e_q) / tau_s^2)``.
    ``e`` is unit-norm, so ``||e_p - e_q||^2 / 2 = 1 - e_p.e_q`` and the
    Gaussian affinity of MoA.md §7.2 becomes a dot product — no per-offset
    Ce-channel difference tensor is kept for backward.
    """

    def __init__(self, barrier_offsets: list[tuple[int, int]], radius: int = 3,
                 spatial_sigma: tuple[float, ...] = (1.0, 2.0, 3.0),
                 tau_init: float = 0.5, lambda_a: float = 1e-6, lambda_b: float = 1e-6,
                 tau_var: float = 1e-6, min_neff: tuple[float, ...] = (2.0, 4.0, 6.0),
                 a_range: tuple[float, float] = (0.2, 5.0), b_max: float = 0.25,
                 det_eps: float = 1e-12) -> None:
        super().__init__()
        self.offsets = list(barrier_offsets)
        self.radius = int(radius)
        self.num_scales = self.radius
        if len(spatial_sigma) != self.num_scales or len(min_neff) != self.num_scales:
            raise ValueError("spatial_sigma / min_neff need one value per scale (1..radius)")
        self.spatial_sigma = tuple(float(s) for s in spatial_sigma)
        self.log_tau = nn.Parameter(torch.full((self.num_scales,), math.log(tau_init)))
        self.lambda_a = float(lambda_a)
        self.lambda_b = float(lambda_b)
        self.tau_var = float(tau_var)
        self.min_neff = tuple(float(m) for m in min_neff)
        self.a_min, self.a_max = float(a_range[0]), float(a_range[1])
        self.b_max = float(b_max)
        self.det_eps = float(det_eps)

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_valid: torch.Tensor,
                y_valid: torch.Tensor, r_anchor: torch.Tensor, emb: torch.Tensor,
                pass_mask: torch.Tensor) -> LocalAffineResult:
        """x, y, x_valid, y_valid, r_anchor: [B,1,H,W]; emb [B,Ce,H,W]; pass_mask [B,K,H,W].

        ``r_anchor`` must already be detached (MoA.md §4): the centre loss may
        not learn to switch anchors off by lowering the MVS confidence.
        """
        x = x.float()
        y = y.float()
        emb = emb.float()
        vx = x_valid.float()
        vy = y_valid.float()
        anchor = (r_anchor.float() * vx * vy)
        R = self.radius
        xq = shifted_views(x * vx, self.offsets, R)
        yq = shifted_views(y * vy, self.offsets, R)
        aq = shifted_views(anchor, self.offsets, R)
        eq = shifted_views(emb, self.offsets, R)
        tau2 = torch.exp(2.0 * self.log_tau)

        zeros = torch.zeros_like(x)
        S = self.num_scales
        # centred sums: xt = x_q - x_p, d = y_q - x_q (see module docstring)
        Sw = [zeros] * S
        Sx = [zeros] * S
        Sxx = [zeros] * S
        Sd = [zeros] * S
        Sxd = [zeros] * S
        Sww = [zeros] * S
        for k, (dy, dx) in enumerate(self.offsets):
            ring = max(abs(dy), abs(dx))
            base = aq[k] * pass_mask[:, k:k + 1]
            cos = (emb * eq[k]).sum(dim=1, keepdim=True)
            d2 = float(dy * dy + dx * dx)
            xt = xq[k] - x
            d = yq[k] - xq[k]
            for s in range(S):
                if ring > s + 1:
                    continue
                spatial = math.exp(-d2 / (2.0 * self.spatial_sigma[s] ** 2))
                w = base * spatial * torch.exp(-(1.0 - cos) / tau2[s])
                wxt = w * xt
                Sw[s] = Sw[s] + w
                Sx[s] = Sx[s] + wxt
                Sxx[s] = Sxx[s] + wxt * xt
                Sd[s] = Sd[s] + w * d
                Sxd[s] = Sxd[s] + wxt * d
                Sww[s] = Sww[s] + w * w

        outs = {k: [] for k in ("a", "b", "valid", "offset_only", "support", "x_fit")}
        for s in range(S):
            a, b, off, ok, flat, n_eff = self._solve(Sw[s], Sx[s], Sxx[s], Sd[s], Sxd[s], Sww[s], x,
                                                     self.min_neff[s], vx)
            area = float((2 * (s + 1) + 1) ** 2)
            outs["a"].append(a)
            outs["b"].append(b)
            outs["valid"].append(ok.float())
            outs["offset_only"].append((flat & ok).float())
            outs["support"].append((n_eff / area).detach())
            outs["x_fit"].append(x + off)
        return LocalAffineResult(**{k: torch.cat(v, dim=1) for k, v in outs.items()})

    def _solve(self, Sw, Sxt, Sxx, Sd, Sxd, Sww, t, min_neff, center_valid):
        """Minimize sum w (alpha xt + bt - d)^2 + la alpha^2 + lb (bt - alpha t)^2,
        i.e. the §7.3 objective with a = 1 + alpha, b = bt - alpha t, t = x_p.
        Returns (a, b, offset, ok, flat, n_eff) with fitted value x_p + offset."""
        la, lb = self.lambda_a, self.lambda_b
        # Every denominator is replaced by a safe value before dividing: the
        # rejected branch of a torch.where still backpropagates, and 0/0 there
        # would put NaN into the gradient of the accepted branch.
        sw_ok = Sw > 1e-6
        Sw_s = torch.where(sw_ok, Sw, torch.ones_like(Sw))
        Sww_s = torch.where(Sww > 1e-12, Sww, torch.ones_like(Sww))
        n_eff = torch.where(sw_ok, Sw * Sw / Sww_s, torch.zeros_like(Sw))
        mean_xt = Sxt / Sw_s
        var_x = (Sxx / Sw_s - mean_xt * mean_xt).clamp_min(0.0)

        A11 = Sxx + la + lb * t * t
        A12 = Sxt - lb * t
        A22 = Sw + lb
        det = A11 * A22 - A12 * A12
        det_ok = det > self.det_eps
        det_s = torch.where(det_ok, det, torch.ones_like(det))
        alpha = (Sxd * A22 - A12 * Sd) / det_s
        bt = (A11 * Sd - A12 * Sxd) / det_s
        a_full = 1.0 + alpha
        b_full = bt - alpha * t

        flat = var_x < self.tau_var
        b_off = Sd / Sw_s
        a = torch.where(flat, torch.ones_like(a_full), a_full)
        b = torch.where(flat, b_off, b_full)
        off = torch.where(flat, b_off, bt)

        enough = sw_ok & (n_eff >= min_neff) & (center_valid > 0.5)
        full_ok = det_ok & (a_full > self.a_min) & (a_full < self.a_max)
        ok = enough & (flat | full_ok) & torch.isfinite(a) & torch.isfinite(b) & (b.abs() <= self.b_max)
        a = torch.where(ok, a, torch.ones_like(a))
        b = torch.where(ok, b, torch.zeros_like(b))
        off = torch.where(ok, off, torch.zeros_like(off))
        return a, b, off, ok, flat, n_eff


def spec_closed_form(Sw, Sx, Sy, Sxx, Sxy, lambda_a, lambda_b):
    """MoA.md §7.3 verbatim (raw sums):
    [[Sxx+la, Sx], [Sx, Sw+lb]] [a, b] = [Sxy+la, Sy]. Use in float64 only."""
    A11 = Sxx + lambda_a
    A22 = Sw + lambda_b
    r1 = Sxy + lambda_a
    det = A11 * A22 - Sx * Sx
    return (r1 * A22 - Sx * Sy) / det, (A11 * Sy - Sx * r1) / det
