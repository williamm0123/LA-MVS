"""DA3 depth edges and the edge barrier for 3x3 / 5x5 / 7x7 local affine (MoA.md §5).

Two pixels p, q may share an affine field only if nothing on the discrete
segment p -> q is an edge (``M_edge``) and their monocular log depths are close
(``M_jump``). Checking only the endpoints is not enough: p and q can both be
off-edge while a foreground/background boundary runs between them.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moa.geometry import spatial_grad


def log_depth_edge(depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """``E = max(|d/dx log Z|, |d/dy log Z|)`` with forward differences, [B,1,H,W].

    Log-depth makes the edge magnitude independent of the global scale of the
    (uncalibrated) DA3 output. Pairs touching an invalid pixel contribute 0.
    """
    l = torch.where(valid.bool(), depth.float().clamp_min(1e-6).log(), torch.zeros_like(depth, dtype=torch.float32))
    gx, gy, _, _ = spatial_grad(l, valid)
    return torch.maximum(gx.abs(), gy.abs())


def downsample_edge(edge: torch.Tensor, hw: tuple[int, int], dilate: bool = False) -> torch.Tensor:
    """Max-pool (never bilinear/area): an edge anywhere in a cell marks the cell."""
    e = edge if tuple(edge.shape[-2:]) == tuple(hw) else F.adaptive_max_pool2d(edge, tuple(hw))
    if dilate:
        e = F.max_pool2d(e, 3, stride=1, padding=1)
    return e


def supercover_path(dy: int, dx: int) -> list[tuple[int, int]]:
    """Every cell the closed segment (0,0)->(dy,dx) touches, including corner touches.

    Cell (i, j) is the closed square [i-.5, i+.5] x [j-.5, j+.5]; a cell is on
    the path when the segment intersects it (Liang-Barsky clipping). Corner
    touches count, so a diagonal step can never slip between two edge cells.
    """
    eps = 1e-9
    cells = []
    for i in range(min(0, dy), max(0, dy) + 1):
        for j in range(min(0, dx), max(0, dx) + 1):
            t0, t1 = 0.0, 1.0
            ok = True
            for d, c in ((dy, i), (dx, j)):
                lo, hi = c - 0.5 - eps, c + 0.5 + eps
                if d == 0:
                    if not (lo <= 0.0 <= hi):
                        ok = False
                        break
                    continue
                a, b = lo / d, hi / d
                if a > b:
                    a, b = b, a
                t0, t1 = max(t0, a), min(t1, b)
                if t0 > t1:
                    ok = False
                    break
            if ok:
                cells.append((t0, i, j))
    cells.sort()
    return [(i, j) for _, i, j in cells]


def shifted_views(x: torch.Tensor, offsets: list[tuple[int, int]], radius: int,
                  fill: float = 0.0) -> list[torch.Tensor]:
    """``out[k][..., i, j] = x[..., i+dy_k, j+dx_k]``; off-image reads give ``fill``.

    Pad + slice, never ``torch.roll`` — roll would wrap the right border onto
    the left one and let a corner pixel borrow anchors from the opposite corner.
    """
    H, W = x.shape[-2:]
    r = radius
    p = F.pad(x, (r, r, r, r), value=fill)
    return [p[..., r + dy:r + dy + H, r + dx:r + dx + W] for dy, dx in offsets]


class NeighborhoodBarrier(nn.Module):
    """One 7x7 expansion shared by the 3x3 / 5x5 / 7x7 scales.

    Buffers: ``offsets`` [K, 2], ``ring`` [K] (Chebyshev radius) and the
    padded supercover path table ``path_idx`` [K, L] (indices into offsets).
    """

    def __init__(self, radius: int = 3, subset: list[tuple[int, int]] | None = None) -> None:
        """``subset``: only these offsets get a pass mask (LAPE's 3/7/11 union, NCE's dilated
        5x5). Paths are still traced through every cell of the full square, so a skipped
        pixel's edge still blocks the offsets behind it. ``None`` = the whole square."""
        super().__init__()
        self.radius = int(radius)
        r = self.radius
        full = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
        index = {o: k for k, o in enumerate(full)}
        offsets = full if subset is None else [tuple(int(v) for v in o) for o in subset]
        bad = [o for o in offsets if o not in index]
        if bad:
            raise ValueError(f"offsets {bad[:4]} lie outside radius {r}")
        paths = [[index[c] for c in supercover_path(dy, dx)] for dy, dx in offsets]
        L = max(len(p) for p in paths)
        # Pad with the endpoint itself: repeating a path cell never changes the max.
        padded = [p + [p[-1]] * (L - len(p)) for p in paths]
        self.full_offsets = full
        self.offset_list = offsets
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.register_buffer("ring", torch.tensor([max(abs(a), abs(b)) for a, b in offsets],
                                                  dtype=torch.long), persistent=False)
        self.register_buffer("path_idx", torch.tensor(padded, dtype=torch.long), persistent=False)

    @property
    def num_offsets(self) -> int:
        return len(self.offset_list)

    @torch.no_grad()
    def forward(self, edge: torch.Tensor, log_z: torch.Tensor, valid: torch.Tensor,
                tau_edge: float, tau_jump: float) -> dict[str, torch.Tensor]:
        """edge / log_z / valid: [B,1,H,W] at the working resolution.

        Returns ``pass`` [B,K,H,W] (1 = q may anchor p's affine), ``pair_valid``
        [B,K,H,W] (in-bounds and both endpoints valid, barrier ignored) and
        ``blocked_rate`` [B, radius] — share of valid neighbour pairs the barrier
        removed, per scale (1..radius).
        """
        r = self.radius
        v = valid.float()
        e = edge.float()
        lz = log_z.float()
        e_views = torch.stack(shifted_views(e, self.full_offsets, r, 0.0), dim=0)  # [Kfull,B,1,H,W]
        v_views = shifted_views(v, self.offset_list, r, 0.0)
        lz_views = shifted_views(lz, self.offset_list, r, 0.0)
        pas, pair = [], []
        for k in range(self.num_offsets):
            path_max = e_views[self.path_idx[k]].amax(dim=0)                       # [B,1,H,W]
            pv = (v * v_views[k]) > 0.5
            ok = pv & (path_max < tau_edge) & ((lz_views[k] - lz).abs() < tau_jump)
            pas.append(ok)
            pair.append(pv)
        pas = torch.cat(pas, dim=1).float()
        pair = torch.cat(pair, dim=1).float()
        rates = []
        for s in range(1, r + 1):
            sel = ((self.ring >= 1) & (self.ring <= s)).float().view(1, -1, 1, 1)
            n_pair = (pair * sel).sum(dim=(1, 2, 3))
            n_block = ((pair - pas) * sel).sum(dim=(1, 2, 3))
            rates.append(n_block / n_pair.clamp_min(1.0))
        return {"pass": pas, "pair_valid": pair, "blocked_rate": torch.stack(rates, dim=1)}
