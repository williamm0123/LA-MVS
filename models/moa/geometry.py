"""Normalized inverse-depth axis ``u = (1/Z - vmin) / (vmax - vmin)`` in [0, 1].

u = 1 is the near bound (depth_min), u = 0 the far bound (depth_max). Every
hypothesis tensor in the MoA network is sorted by **ascending depth**, i.e.
descending u — the order ``soft_label_cross_entropy`` and ``DepthDecoder``
assume. Axes are uniform in u, so a plain second difference along the index is
the true second derivative (MoA.md §13.5).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def inverse_bounds(depth_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``depth_values`` [B, D] -> (vmin, vmax), each [B, 1, 1, 1] in 1/mm."""
    dv = depth_values.float()
    vmin = 1.0 / dv.amax(dim=1)
    vmax = 1.0 / dv.amin(dim=1)
    return vmin.view(-1, 1, 1, 1), vmax.view(-1, 1, 1, 1)


def depth_to_u(depth: torch.Tensor, vmin: torch.Tensor, vmax: torch.Tensor) -> torch.Tensor:
    return (1.0 / depth.float().clamp_min(1e-6) - vmin) / (vmax - vmin)


def u_to_depth(u: torch.Tensor, vmin: torch.Tensor, vmax: torch.Tensor) -> torch.Tensor:
    return 1.0 / (vmin + u.float().clamp(0.0, 1.0) * (vmax - vmin))


def stage1_u_hypotheses(batch: int, num: int, hw: tuple[int, int],
                        device: torch.device) -> torch.Tensor:
    """Full-range uniform inverse-depth axis [B, D, H, W], descending u."""
    u = torch.linspace(1.0, 0.0, num, device=device, dtype=torch.float32)
    return u.view(1, num, 1, 1).expand(batch, num, *hw).contiguous()


def window_u_hypotheses(center: torch.Tensor, half: torch.Tensor, num: int) -> torch.Tensor:
    """Uniform window around ``center`` [B,1,H,W] with half-width ``half`` [B,1,H,W].

    A window that pokes out of [0, 1] is slid back inside at constant width
    rather than clipped, so the stage keeps all ``num`` candidates in range.
    """
    half = half.clamp(min=1e-6, max=0.5)
    c = torch.minimum(torch.maximum(center, half), 1.0 - half)
    steps = torch.linspace(1.0, -1.0, num, device=center.device, dtype=torch.float32)
    return c + half * steps.view(1, num, 1, 1)


def axis_spacing(u_hyp: torch.Tensor) -> torch.Tensor:
    """Local spacing of a uniform u axis, [B, 1, H, W]."""
    return (u_hyp[:, :1] - u_hyp[:, 1:2]).abs()


def sample_at_feature_pixels(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Pick, for each cell of an ``hw`` feature map, the image pixel its plane sweep uses.

    ``homography_warp_features`` scales the whole K (principal point included)
    by 1/stride, so feature pixel i looks at image pixel i*stride — the same
    pixel ``F.interpolate(mode="nearest")`` picks. Exact values, no averaging:
    averaging across a depth discontinuity would invent a surface on neither side.
    """
    H, W = x.shape[-2:]
    h, w = hw
    if (H, W) == (h, w):
        return x
    ys = (torch.arange(h, device=x.device) * H // h).clamp(max=H - 1)
    xs = (torch.arange(w, device=x.device) * W // w).clamp(max=W - 1)
    return x.index_select(-2, ys).index_select(-1, xs)


def upsample_nearest(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(hw):
        return x
    return F.interpolate(x, size=tuple(hw), mode="nearest")


def interp_along_axis(volume: torch.Tensor, u_hyp: torch.Tensor,
                      u_query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear interpolation of ``volume`` along its depth axis at ``u_query``.

    ``volume`` [B, D, H, W] or [B, C, D, H, W]; ``u_hyp`` [B, D, H, W] strictly
    monotone (descending u), not necessarily uniform — positions come from the
    real hypothesis values, never from treating a depth as an index.
    ``u_query`` [B, 1, H, W]. Returns (value — [B, 1, H, W] for a 4-D volume,
    [B, C, H, W] for a 5-D one — and inside [B, 1, H, W]); outside the axis the
    value is 0.
    """
    D = u_hyp.shape[1]
    hs = (-u_hyp.float()).permute(0, 2, 3, 1).contiguous()          # ascending
    q = (-u_query.float()).permute(0, 2, 3, 1).contiguous()          # [B,H,W,1]
    inside = (q >= hs[..., :1]) & (q <= hs[..., -1:])
    r = torch.searchsorted(hs, q).clamp(1, D - 1)
    left = r - 1
    h_l = hs.gather(-1, left)
    h_r = hs.gather(-1, r)
    t = ((q - h_l) / (h_r - h_l).clamp_min(1e-12)).clamp(0.0, 1.0)
    left = left.permute(0, 3, 1, 2)                                   # [B,1,H,W]
    r = r.permute(0, 3, 1, 2)
    t = t.permute(0, 3, 1, 2)
    inside = inside.permute(0, 3, 1, 2)
    if volume.dim() == 4:
        v = volume.float()
        val = (1.0 - t) * v.gather(1, left) + t * v.gather(1, r)
    else:
        v = volume.float()
        C = v.shape[1]
        il = left.unsqueeze(1).expand(-1, C, -1, -1, -1)
        ir = r.unsqueeze(1).expand(-1, C, -1, -1, -1)
        val = (1.0 - t) * v.gather(2, il).squeeze(2) + t * v.gather(2, ir).squeeze(2)
    return val * inside.float(), inside


def spatial_grad(x: torch.Tensor, valid: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward differences (gx, gy) [B,1,H,W] and their pair-validity masks.

    The last column/row has no forward neighbour and gets 0 / invalid.
    """
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    gy[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    vx = torch.zeros_like(x, dtype=torch.bool)
    vy = torch.zeros_like(x, dtype=torch.bool)
    if valid is None:
        vx[..., :, :-1] = True
        vy[..., :-1, :] = True
    else:
        v = valid.bool()
        vx[..., :, :-1] = v[..., :, 1:] & v[..., :, :-1]
        vy[..., :-1, :] = v[..., 1:, :] & v[..., :-1, :]
    return gx * vx, gy * vy, vx, vy


def laplacian(x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """5-point Laplacian; 0 wherever any stencil pixel is invalid or off-image."""
    p = F.pad(x, (1, 1, 1, 1))
    lap = p[..., 1:-1, :-2] + p[..., 1:-1, 2:] + p[..., :-2, 1:-1] + p[..., 2:, 1:-1] - 4.0 * x
    ok = torch.ones_like(x, dtype=torch.bool) if valid is None else valid.bool()
    okp = F.pad(ok.float(), (1, 1, 1, 1))
    ok = ok & (okp[..., 1:-1, :-2] > 0) & (okp[..., 1:-1, 2:] > 0) \
        & (okp[..., :-2, 1:-1] > 0) & (okp[..., 2:, 1:-1] > 0)
    return lap * ok
