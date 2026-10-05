"""LAPE (4) low-frequency recall: where MVS has collapsed on a textureless smooth
surface, put the next stage's candidates around the monocular depth.

Per pixel p of the parent stage, W_p = the 7x7 neighbourhood restricted by the DA3
edge barrier, D = parent spacing, everything in normalised inverse depth u:

    C1  RGB low-frequency      mean Sobel magnitude in W_p / image median   < tau_rgb
    C2  mono smooth            plane-fit residual of x (RAC-aligned DA3) in W_p   < tau_m * D
    C3  out of the next window |y - mu_RAC| > max(k_s * D, g * D1)
                               (k_s = next stage's half window in D; g stage-1 bins D1 as a floor)
    C4  MVS failed, either way:
        C4a rougher            plane-fit residual of y in W_p > rho * R_mono + 0.5 D   (noisy matches)
        C4b collapsed          y does not follow x: |slope of y on x| < slope_min in W_p while x
                               varies by >= mono_var * D there (smoothed onto a wrong, flat surface)
    C5  regional               >= tau_r of W_p's same-surface pixels satisfy C3 and C4
    C6  supported              p's RAC region has anchors and its nearest anchor is < delta_max
    C0  untrusted matching     7x7 mean of the MoA confidence r < conf_max

``F = C0 & C1 & ... & C6``. A plane is affine in u over image coordinates, so the u-domain
plane residual measures "not a plane" directly (it covers the depth gradient and
variance checks); C4b is the collapse a plane test cannot see — a constant depth is a
perfect plane — and the same signature RAC rejects globally (slope ratio 0.01-0.3).
The reference is ``mu_RAC``, not a local expert: the local experts are fitted to the
nearby MVS, i.e. to the very values whose failure is being tested.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.moa.edge import shifted_views

OFFSETS7 = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4)]


@dataclass
class LFRResult:
    flag: torch.Tensor        # [B,1,h,w] bool
    soft: torch.Tensor        # [B,1,h,w] product of the soft conditions (feature)
    rgb_ratio: torch.Tensor   # [B,1,h,w]
    r_mono: torch.Tensor      # [B,1,h,w] plane residual of x / D
    r_mvs: torch.Tensor       # [B,1,h,w] plane residual of y / D
    gap: torch.Tensor         # [B,1,h,w] |y - mu_RAC| / D


def texture_ratio(gray: torch.Tensor, hw: tuple[int, int], win: int = 7) -> torch.Tensor:
    """Sobel magnitude of the full-res image, averaged into ``hw`` cells and a win x win
    window, divided by the image median. [B,1,H,W] -> [B,1,h,w]."""
    g = gray.float()
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=g.device).view(1, 1, 3, 3)
    gx = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), kx)
    gy = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), kx.transpose(2, 3))
    mag = (gx * gx + gy * gy).sqrt()
    cell = F.adaptive_avg_pool2d(mag, hw)
    loc = F.avg_pool2d(cell, win, stride=1, padding=win // 2, count_include_pad=False)
    med = cell.flatten(1).median(dim=1).values.clamp_min(1e-6).view(-1, 1, 1, 1)
    return loc / med


def plane_residual(f: torch.Tensor, w: torch.Tensor, pass7: torch.Tensor) -> torch.Tensor:
    """RMS residual of a weighted plane fit f ~ c0 + c1 dx + c2 dy over the 7x7 window.

    ``f`` is centred on the window's own centre value before summing (fp32 would lose
    the residual otherwise: f ~ 0.5, its variation ~ 1e-3). [B,1,h,w] each; pass7 [B,49,h,w].
    """
    f = f.float()
    fq = shifted_views(f, OFFSETS7, 3)
    wq = shifted_views(w.float(), OFFSETS7, 3)
    S = [torch.zeros_like(f) for _ in range(10)]   # 1, x, y, xx, xy, yy, f, fx, fy, ff
    for k, (dy, dx) in enumerate(OFFSETS7):
        wk = wq[k] * pass7[:, k:k + 1]
        fk = (fq[k] - f) * wk
        S[0] = S[0] + wk
        S[1] = S[1] + wk * dx
        S[2] = S[2] + wk * dy
        S[3] = S[3] + wk * dx * dx
        S[4] = S[4] + wk * dx * dy
        S[5] = S[5] + wk * dy * dy
        S[6] = S[6] + fk
        S[7] = S[7] + fk * dx
        S[8] = S[8] + fk * dy
        S[9] = S[9] + fk * (fq[k] - f)
    A = torch.stack([torch.stack([S[0], S[1], S[2]], -1),
                     torch.stack([S[1], S[3], S[4]], -1),
                     torch.stack([S[2], S[4], S[5]], -1)], -2)[:, 0]      # [B,h,w,3,3]
    rhs = torch.stack([S[6], S[7], S[8]], -1)[:, 0].unsqueeze(-1)       # [B,h,w,3,1]
    A = A + 1e-6 * torch.eye(3, device=f.device)
    theta = torch.linalg.solve(A, rhs)
    rss = S[9][:, 0] - (theta * rhs).sum(dim=(-2, -1))
    n = S[0][:, 0].clamp_min(1.0)
    r = (rss.clamp_min(0.0) / n).sqrt()
    r = torch.where(S[0][:, 0] >= 6.0, r, torch.full_like(r, float("inf")))  # too few pixels: undecided
    return r.unsqueeze(1)


def follow_slope(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, pass7: torch.Tensor):
    """Weighted regression of y on x over the 7x7 window: (slope, std of x), [B,1,h,w] each."""
    x, y = x.float(), y.float()
    xq = shifted_views(x, OFFSETS7, 3)
    yq = shifted_views(y, OFFSETS7, 3)
    wq = shifted_views(w.float(), OFFSETS7, 3)
    S = [torch.zeros_like(x) for _ in range(5)]   # w, wx, wy, wxx, wxy (centred on the centre pixel)
    for k in range(len(OFFSETS7)):
        wk = wq[k] * pass7[:, k:k + 1]
        xc, yc = xq[k] - x, yq[k] - y
        S[0] = S[0] + wk
        S[1] = S[1] + wk * xc
        S[2] = S[2] + wk * yc
        S[3] = S[3] + wk * xc * xc
        S[4] = S[4] + wk * xc * yc
    n = S[0].clamp_min(1e-6)
    mx, my = S[1] / n, S[2] / n
    vx = (S[3] / n - mx * mx).clamp_min(0.0)
    cxy = S[4] / n - mx * my
    slope = torch.where(vx > 1e-12, cxy / vx.clamp_min(1e-12), torch.ones_like(vx))
    return slope, vx.sqrt()


@torch.no_grad()
def low_freq_recall(gray: torch.Tensor, x: torch.Tensor, xv: torch.Tensor, y: torch.Tensor,
                    du: torch.Tensor, pass7: torch.Tensor, supported: torch.Tensor, delta: torch.Tensor,
                    k_s: float, cfg, du_global: float = 1.0, conf: torch.Tensor | None = None) -> LFRResult:
    """gray [B,1,H,W] full-res in [0,1]; the rest [B,1,h,w] at the parent resolution;
    pass7 [B,49,h,w] same-surface masks on ``OFFSETS7``."""
    hw = tuple(x.shape[-2:])
    D = du.float().clamp_min(1e-8)
    rgb = texture_ratio(gray, hw)
    valid = xv.float()
    r_mono = plane_residual(x, valid, pass7) / D
    r_mvs = plane_residual(y, torch.ones_like(valid), pass7) / D
    gap = (y.float() - x.float()).abs() / D
    c1 = rgb < cfg.lfr_tau_rgb
    c2 = r_mono < cfg.lfr_tau_mono
    floor = float(cfg.lfr_gap_floor_bins) * float(du_global) / D           # in parent spacings
    c3 = gap > torch.clamp(floor, min=float(k_s))
    slope, sx = follow_slope(x, y, valid, pass7)
    c4 = (r_mvs > cfg.lfr_rho * r_mono + 0.5) | \
        ((sx / D >= cfg.lfr_mono_var_bins) & (slope.abs() < cfg.lfr_slope_min))
    c34 = (c3 & c4 & xv.bool()).float()
    num = torch.zeros_like(c34)
    den = torch.zeros_like(c34)
    cq = shifted_views(c34, OFFSETS7, 3)
    vq = shifted_views(valid, OFFSETS7, 3)
    for k in range(len(OFFSETS7)):
        wk = pass7[:, k:k + 1] * vq[k]
        num = num + wk * cq[k]
        den = den + wk
    c5 = (num / den.clamp_min(1.0)) >= cfg.lfr_tau_region
    c6 = supported.bool() & (delta < cfg.lfr_delta_max) & xv.bool()
    if conf is not None:
        rq = shifted_views(conf.float(), OFFSETS7, 3)
        rs = torch.zeros_like(c34)
        rn = torch.zeros_like(c34)
        for k in range(len(OFFSETS7)):
            wk = pass7[:, k:k + 1]
            rs = rs + wk * rq[k]
            rn = rn + wk
        c0 = (rs / rn.clamp_min(1.0)) < cfg.lfr_conf_max
        c6 = c6 & c0
    flag = c1 & c2 & c3 & c4 & c5 & c6
    sig = torch.sigmoid
    finite = lambda t: torch.nan_to_num(t, nan=1e3, posinf=1e3, neginf=-1e3)
    soft = (sig((cfg.lfr_tau_rgb - rgb) * 8.0) * sig((cfg.lfr_tau_mono - finite(r_mono)) * 8.0)
            * sig((gap - torch.clamp(floor, min=float(k_s))) * 2.0) * c4.float() * c6.float())
    return LFRResult(flag=flag, soft=soft, rgb_ratio=rgb, r_mono=finite(r_mono), r_mvs=finite(r_mvs), gap=gap)
