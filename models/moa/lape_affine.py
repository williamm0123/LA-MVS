"""LAPE local experts: 3x3 / 7x7 / 11x11 weighted least squares with honest uncertainty.

Same objective and re-parameterisation as ``local_affine.MultiScaleLocalAffine``
(regress d = y - x on x~ = x_q - x_p around the centre pixel, ridge toward the
identity, offset-only on flat windows), with three changes:

* windows 3x3 (dense), 7x7 (dense) and 11x11 on offsets {0, +-1, +-3, +-5} — the
  same 49 samples as 7x7 but a radius of 5, so the largest window still covers
  ~10 full-resolution pixels at the last transition (1/2 res) instead of 6;
* the sums are kept separately for the two checkerboard parities of the offsets
  ((dy + dx) mod 2). Each parity is solved on its own and predicts the other one,
  which gives a held-out residual ``s_ho`` from the same single pass — a fitted
  residual on few anchors is optimistic (two anchors fit any line exactly);
* sum(w^2), sum(w^2 x~), sum(w^2 x~^2) give the sandwich variance of the centre
  prediction (``sigma_par``) and the exact Kish n_eff = (sum w)^2 / sum w^2;
* an out-of-range slope falls back to offset-only instead of rejecting the window
  (the centre prediction barely depends on the slope); validity = enough effective
  anchors and |b~| <= b_max (the offset at the centre, in u).

The residual sum of squares uses the full quadratic expansion, valid with the ridge:

    SSE(alpha, b~; S) = Sdd - 2 alpha Sxd - 2 b~ Sd + alpha^2 Sxx + 2 alpha b~ Sx + b~^2 Sw
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from models.moa.edge import shifted_views

E11_STEPS = (-5, -3, -1, 0, 1, 3, 5)
RADII = (1, 3, 5)


def expert_offsets() -> tuple[list[tuple[int, int]], list[tuple[bool, bool, bool]]]:
    """Union of the three windows' offsets and, per offset, which windows use it."""
    e3 = {(dy, dx) for dy in range(-1, 2) for dx in range(-1, 2)}
    e7 = {(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4)}
    e11 = {(dy, dx) for dy in E11_STEPS for dx in E11_STEPS}
    union = sorted(e3 | e7 | e11)
    member = [(o in e3, o in e7, o in e11) for o in union]
    return union, member


@dataclass
class WindowExpertResult:
    mu: torch.Tensor          # [B,3,H,W] prediction in u (x where invalid)
    alpha: torch.Tensor       # [B,3,H,W] slope correction (0 if flat / invalid)
    bt: torch.Tensor          # [B,3,H,W] offset at the centre pixel (mu = x + bt)
    valid: torch.Tensor       # [B,3,H,W] float
    offset_only: torch.Tensor  # [B,3,H,W] float
    support: torch.Tensor     # [B,3,H,W] n_eff / samples
    n_eff: torch.Tensor       # [B,3,H,W]
    s_ho: torch.Tensor        # [B,3,H,W] held-out residual scale (u)
    sigma_par: torch.Tensor   # [B,3,H,W] std of the centre prediction from the fit (u)
    delta: torch.Tensor       # [B,3,H,W] radius * (1 - support), a distance proxy (px)


def _solve(S, t, lam_a, lam_b, tau_var, det_eps=1e-12):
    """S = (Sw, Sx, Sxx, Sd, Sxd). Returns (alpha, bt, flat, det_ok, A11, A12, A22, det)."""
    Sw, Sx, Sxx, Sd, Sxd = S
    sw_ok = Sw > 1e-6
    Sw_s = torch.where(sw_ok, Sw, torch.ones_like(Sw))
    mean_x = Sx / Sw_s
    var_x = (Sxx / Sw_s - mean_x * mean_x).clamp_min(0.0)
    A11 = Sxx + lam_a + lam_b * t * t
    A12 = Sx - lam_b * t
    A22 = Sw + lam_b
    det = A11 * A22 - A12 * A12
    det_ok = det > det_eps
    det_s = torch.where(det_ok, det, torch.ones_like(det))
    alpha = (Sxd * A22 - A12 * Sd) / det_s
    bt = (A11 * Sd - A12 * Sxd) / det_s
    flat = (var_x < tau_var) | ~det_ok
    alpha = torch.where(flat, torch.zeros_like(alpha), alpha)
    bt = torch.where(flat, Sd / Sw_s, bt)
    return alpha, bt, flat, det_ok, A11, A12, A22, det


def _sse(alpha, bt, S6):
    Sw, Sx, Sxx, Sd, Sxd, Sdd = S6
    return (Sdd - 2 * alpha * Sxd - 2 * bt * Sd + alpha * alpha * Sxx
            + 2 * alpha * bt * Sx + bt * bt * Sw).clamp_min(0.0)


class WindowAffineExperts(nn.Module):
    def __init__(self, spatial_sigma=(1.0, 2.0, 4.0), tau_init: float = 0.5,
                 lambda_a: float = 1e-6, lambda_b: float = 1e-6, tau_var: float = 1e-6,
                 min_neff=(4.0, 8.0, 12.0), a_range=(0.2, 5.0), b_max: float = 0.25) -> None:
        super().__init__()
        self.offsets, self.member = expert_offsets()
        self.radius = max(RADII)
        self.n_samples = tuple(float(sum(m[s] for m in self.member)) for s in range(3))
        self.spatial_sigma = tuple(float(s) for s in spatial_sigma)
        self.log_tau = nn.Parameter(torch.full((3,), math.log(tau_init)))
        self.lambda_a, self.lambda_b, self.tau_var = float(lambda_a), float(lambda_b), float(tau_var)
        self.min_neff = tuple(float(m) for m in min_neff)
        self.a_min, self.a_max = float(a_range[0]), float(a_range[1])
        self.b_max = float(b_max)
        # static index sets: offsets of window s with checkerboard parity p (no .nonzero() per call)
        d2 = [float(dy * dy + dx * dx) for dy, dx in self.offsets]
        self.register_buffer("d2", torch.tensor(d2), persistent=False)
        for si in range(3):
            for p in (0, 1):
                idx = [k for k, ((dy, dx), m) in enumerate(zip(self.offsets, self.member))
                       if m[si] and ((dy + dx) & 1) == p]
                self.register_buffer(f"sel_{si}_{p}", torch.tensor(idx, dtype=torch.long), persistent=False)
        self.register_buffer("member_f", torch.tensor(self.member, dtype=torch.float32), persistent=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_valid: torch.Tensor, anchor_w: torch.Tensor,
                emb: torch.Tensor, pass_mask: torch.Tensor) -> WindowExpertResult:
        """x, y, x_valid, anchor_w: [B,1,H,W]; emb [B,Ce,H,W] unit-norm; pass_mask [B,K,H,W]
        on ``self.offsets``. ``anchor_w`` must be detached (MVS reliability)."""
        x, y, emb = x.float(), y.float(), emb.float()
        vx = x_valid.float()
        anchor = anchor_w.float() * vx
        R = self.radius
        # all 73 offsets at once ([B, K, H, W]); only the embedding affinity is looped, so the
        # Ce-channel neighbour stack is never materialised
        X = torch.cat(shifted_views(x * vx, self.offsets, R), dim=1)
        Y = torch.cat(shifted_views(y * vx, self.offsets, R), dim=1)
        A = torch.cat(shifted_views(anchor, self.offsets, R), dim=1) * pass_mask.float()
        cos = torch.cat([(emb * e).sum(dim=1, keepdim=True) for e in shifted_views(emb, self.offsets, R)], dim=1)
        xt = X - x
        d = Y - X
        tau2 = torch.exp(2.0 * self.log_tau)
        acc = []
        for s in range(3):
            sp = torch.exp(-self.d2 / (2.0 * self.spatial_sigma[s] ** 2)) * self.member_f[:, s]
            w = A * sp.view(1, -1, 1, 1) * torch.exp(-(1.0 - cos) / tau2[s])
            per = []
            for p_ in (0, 1):
                sel = getattr(self, f"sel_{s}_{p_}")
                ws, xs, ds = w.index_select(1, sel), xt.index_select(1, sel), d.index_select(1, sel)
                wx, ww = ws * xs, ws * ws
                per.append([ws.sum(1, keepdim=True), wx.sum(1, keepdim=True), (wx * xs).sum(1, keepdim=True),
                            (ws * ds).sum(1, keepdim=True), (wx * ds).sum(1, keepdim=True),
                            (ws * ds * ds).sum(1, keepdim=True), ww.sum(1, keepdim=True),
                            (ww * xs).sum(1, keepdim=True), (ww * xs * xs).sum(1, keepdim=True)])
            acc.append(per)

        outs = {k: [] for k in ("mu", "alpha", "bt", "valid", "offset_only", "support", "n_eff",
                                "s_ho", "sigma_par", "delta")}
        la, lb, tv = self.lambda_a, self.lambda_b, self.tau_var
        for s in range(3):
            Se, So = acc[s]
            S = [e + o for e, o in zip(Se, So)]
            Sw, Sx, Sxx, Sd, Sxd, Sdd, S2w, S2x, S2xx = S
            alpha, bt, flat, det_ok, A11, A12, A22, det = _solve(S[:5], x, la, lb, tv)
            # held-out residual: each parity predicts the other
            ae, be, *_ = _solve(Se[:5], x, la, lb, tv)
            ao, bo, *_ = _solve(So[:5], x, la, lb, tv)
            both = (Se[0] > 1e-6) & (So[0] > 1e-6)
            sw_s = Sw.clamp_min(1e-12)
            s2_cross = (_sse(ae, be, So[:6]) + _sse(ao, bo, Se[:6])) / sw_s
            S2w_s = S2w.clamp_min(1e-12)
            n_eff = torch.where(Sw > 1e-6, Sw * Sw / S2w_s, torch.zeros_like(Sw))
            s2_in = _sse(alpha, bt, S[:6]) / (sw_s * (1.0 - 2.0 / n_eff.clamp_min(2.5)).clamp_min(0.1))
            s2 = torch.where(both, s2_cross, s2_in)
            # sandwich variance of b~ (the centre prediction)
            det_s = torch.where(det_ok, det, torch.ones_like(det))
            p21, p22 = -A12 / det_s, A11 / det_s
            v_full = p21 * p21 * S2xx + 2.0 * p21 * p22 * S2x + p22 * p22 * S2w
            v_flat = S2w / (sw_s * sw_s)
            var_b = torch.where(flat, v_flat, v_full).clamp_min(0.0)
            sig_par = (s2 * var_b).clamp_min(0.0).sqrt()

            # An implausible slope (small windows: x varies by ~1e-3 and d is noisy) does not
            # invalidate the expert — the centre prediction x_p + b~ hardly depends on alpha
            # (x~_p = 0) — it falls back to offset-only, where only the mean offset is used.
            a_full = 1.0 + alpha
            wild = ~flat & ~((a_full > self.a_min) & (a_full < self.a_max))
            sw_s1 = torch.where(Sw > 1e-6, Sw, torch.ones_like(Sw))
            alpha = torch.where(wild, torch.zeros_like(alpha), alpha)
            bt = torch.where(wild, Sd / sw_s1, bt)
            var_b = torch.where(wild, v_flat, var_b).clamp_min(0.0)
            sig_par = (s2 * var_b).clamp_min(0.0).sqrt()
            flat = flat | wild
            enough = (Sw > 1e-6) & (n_eff >= self.min_neff[s]) & (vx > 0.5)
            ok = enough & torch.isfinite(bt) & torch.isfinite(alpha) & (bt.abs() <= self.b_max)
            okf = ok.float()
            outs["mu"].append(torch.where(ok, x + bt, x))
            outs["alpha"].append(alpha * okf)
            outs["bt"].append(bt * okf)
            outs["valid"].append(okf)
            outs["offset_only"].append((flat & ok).float())
            sup = (n_eff / self.n_samples[s]).clamp(0.0, 1.0)
            outs["support"].append(sup.detach())
            outs["n_eff"].append(n_eff.detach())
            outs["s_ho"].append(s2.clamp_min(0.0).sqrt())
            outs["sigma_par"].append(sig_par)
            outs["delta"].append((RADII[s] * (1.0 - sup)).detach())
        return WindowExpertResult(**{k: torch.cat(v, dim=1) for k, v in outs.items()})
