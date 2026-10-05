"""Per-sample robust global affine ``a * Z_mono + b ~ Z_mvs`` (MoA.md §6).

The DA3 cache holds raw relative depth (no SfM pre-scaling), so this fit is the
only thing that puts the monocular map into metric depth. The spec's
fallback (a=1, b=0) presumes an SfM-metric input; on raw DA3 it would inject
an arbitrary scale. A failed fit therefore returns ``ok=False`` and the caller
disables every monocular expert for that sample (mixture -> MVS).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def robust_global_affine(z_mono: torch.Tensor, z_mvs: torch.Tensor, weight: torch.Tensor,
                         n_iter: int = 3, huber_k: float = 1.345, min_eff: float = 64.0,
                         max_points: int = 200_000) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """z_mono, z_mvs, weight: [B,1,H,W] (weight in [0,1], already 0 on invalid pixels).

    ``min_eff`` is a reliability-weighted anchor count (sum of weights): a map
    of uniformly unreliable MVS must not pass as "plenty of anchors" just
    because the weights are uniform. Each sample is solved independently in
    FP32 with Huber/MAD IRLS; nothing is shared across the batch.
    Returns (a [B], b [B], ok [B] bool).
    """
    B = z_mono.shape[0]
    dev = z_mono.device
    a_out = torch.ones(B, device=dev)
    b_out = torch.zeros(B, device=dev)
    ok_out = torch.zeros(B, dtype=torch.bool, device=dev)
    for i in range(B):
        x = z_mono[i].reshape(-1).float()
        y = z_mvs[i].reshape(-1).float()
        w = weight[i].reshape(-1).float()
        m = (w > 0) & torch.isfinite(x) & torch.isfinite(y) & (x > 0) & (y > 0)
        if int(m.sum()) < 3:
            continue
        x, y, w = x[m], y[m], w[m]
        if float(w.sum()) < min_eff:
            continue
        if x.numel() > max_points:
            idx = torch.linspace(0, x.numel() - 1, max_points, device=dev).long()
            x, y, w = x[idx], y[idx], w[idx]
        w = w / w.max()
        h = torch.ones_like(w)
        a = b = None
        for _ in range(n_iter + 1):
            ww = w * h
            sw = ww.sum().clamp_min(1e-12)
            mx = (ww * x).sum() / sw
            my = (ww * y).sum() / sw
            dx = x - mx
            vxx = (ww * dx * dx).sum()
            if float(vxx) <= 1e-12 * float(sw):
                a = None
                break
            a = (ww * dx * (y - my)).sum() / vxx
            b = my - a * mx
            r = y - (a * x + b)
            scale = 1.4826 * r.abs().median() + 1e-6
            h = (huber_k * scale / r.abs().clamp_min(1e-12)).clamp(max=1.0)
        if a is None or not (torch.isfinite(a) and torch.isfinite(b)) or float(a) <= 0.0:
            continue
        a_out[i], b_out[i], ok_out[i] = a, b, True
    return a_out, b_out, ok_out


def _resid_bins(a, b, x, y, inv_bin):
    """|u(a x + b) - u(y)| in bins of width 1 / inv_bin (u is uniform in inverse depth)."""
    return (1.0 / (a * x + b).clamp_min(1e-3) - 1.0 / y).abs() * inv_bin


@torch.no_grad()
def ransac_tukey_global_affine(z_mono: torch.Tensor, z_mvs: torch.Tensor, weight: torch.Tensor,
                               inv_bin: torch.Tensor, tau: float = 0.5, n_hyp: int = 512,
                               tukey_c: float = 4.685, n_irls: int = 5, s_floor: float = 0.1,
                               min_eff: float = 64.0, max_points: int = 50_000, seed: int = 0,
                               return_weights: bool = False):
    """Same interface as ``robust_global_affine`` plus ``inv_bin`` [B] = 1 / ((vmax - vmin) * du_bin).

    Huber is a monotone M-estimator: anchors in DA3's far tail are leverage points
    in x, pull the line onto themselves and are never down-weighted, and the first
    iteration is a contaminated LS. Here a weighted RANSAC (2-point hypotheses
    drawn by anchor weight, score = inlier weight within ``tau`` bins) picks the
    dominant consistent set, then Tukey biweight IRLS refines it (zero weight
    beyond ``tukey_c * s``). Each sample uses its own generator seeded with
    ``seed``, so the result does not depend on batch composition.

    Returns (a [B], b [B], ok [B] bool), plus with ``return_weights`` the final
    anchor weight map ``weight * tukey(residual)`` shaped like ``weight`` (0 where
    the fit failed), evaluated on every anchor, not only the subsampled ones.
    """
    B = z_mono.shape[0]
    dev = z_mono.device
    a_out = torch.ones(B, device=dev)
    b_out = torch.zeros(B, device=dev)
    ok_out = torch.zeros(B, dtype=torch.bool, device=dev)
    w_out = torch.zeros_like(weight, dtype=torch.float32) if return_weights else None
    for i in range(B):
        x = z_mono[i].reshape(-1).float()
        y = z_mvs[i].reshape(-1).float()
        w = weight[i].reshape(-1).float()
        m = (w > 0) & torch.isfinite(x) & torch.isfinite(y) & (x > 0) & (y > 0)
        if int(m.sum()) < 3:
            continue
        x, y, w = x[m], y[m], w[m]
        xf, yf, wf = x, y, w
        if float(w.sum()) < min_eff:
            continue
        if x.numel() > max_points:
            idx = torch.linspace(0, x.numel() - 1, max_points, device=dev).long()
            x, y, w = x[idx], y[idx], w[idx]
        ib = float(inv_bin[i])
        gen = torch.Generator(device=dev).manual_seed(seed)
        p = w / w.sum()
        hi = torch.multinomial(p, n_hyp, replacement=True, generator=gen)
        hj = torch.multinomial(p, n_hyp, replacement=True, generator=gen)
        dx = x[hj] - x[hi]
        good = dx.abs() > 1e-3 * x.std()
        a = (y[hj] - y[hi]) / torch.where(good, dx, torch.ones_like(dx))
        b = y[hi] - a * x[hi]
        good &= (a > 0) & torch.isfinite(a) & torch.isfinite(b)
        if not bool(good.any()):
            continue
        r = _resid_bins(a[:, None], b[:, None], x[None], y[None], ib)
        score = (w[None] * (r < tau)).sum(1).masked_fill(~good, -1.0)
        k = int(score.argmax())
        inl = r[k] < tau

        xd, yd, wd = x.double(), y.double(), w.double()
        a, b = a[k].double(), b[k].double()
        rb = _resid_bins(a, b, xd, yd, ib)
        s = max(1.4826 * float(rb[inl].median()), s_floor) if bool(inl.any()) else s_floor
        sig2_inv = (yd.median() / yd) ** 4
        failed = False
        for _ in range(n_irls):
            h = (1.0 - (rb / (tukey_c * s)).clamp(max=1.0) ** 2) ** 2
            ww = wd * h * sig2_inv
            sw = float(ww.sum())
            if sw <= 0:
                failed = True
                break
            mx, my = (ww * xd).sum() / sw, (ww * yd).sum() / sw
            vxx = (ww * (xd - mx) ** 2).sum()
            if float(vxx) <= 1e-12 * sw:
                failed = True
                break
            a = (ww * (xd - mx) * (yd - my)).sum() / vxx
            b = my - a * mx
            rb = _resid_bins(a, b, xd, yd, ib)
            keep = rb < tukey_c * s
            if not bool(keep.any()):
                failed = True
                break
            s = max(1.4826 * float(rb[keep].median()), s_floor)
        if failed or not (torch.isfinite(a) and torch.isfinite(b)) or float(a) <= 0.0:
            continue
        h = (1.0 - (rb / (tukey_c * s)).clamp(max=1.0) ** 2) ** 2
        if float((wd * h).sum()) < min_eff:
            continue
        a_out[i], b_out[i], ok_out[i] = a.float(), b.float(), True
        if return_weights:
            rf = _resid_bins(a, b, xf.double(), yf.double(), ib)
            hf = (1.0 - (rf / (tukey_c * s)).clamp(max=1.0) ** 2) ** 2
            w_out[i].view(-1)[m] = (wf.double() * hf).float()
    if return_weights:
        return a_out, b_out, ok_out, w_out
    return a_out, b_out, ok_out
