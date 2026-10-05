"""RAC (reliable anchor cascade): multi-model global alignment of DA3 to the MVS depth.

``ReliableAnchor`` (models/moa/reliable_anchor.py, test22c) fits up to ``max_models``
global affines ``a_k * Z_mono + b_k ~ Z_mvs`` by sequential RANSAC + Tukey: each
accepted model's anchors are removed and the rest is fitted again; a secondary model
is kept only if its slope is 0.5-2x the primary's (a stage-1 region collapsed to a
near-constant depth fits DA3 with a tiny slope) and it has >= 10% of the primary's
anchors. It returns which anchor belongs to which model — not what every *pixel*
should use. That is the part added here:

* regions = connected components of ``mono_valid & not dilated DA3 edge`` (4-conn.);
* a region takes the model its anchors vote for (Tukey weights); a region without
  anchors takes the primary model and is flagged ``supported = False``;
* pixels in the dilated edge band take the model of the nearest region pixel;
* ``delta`` = Chebyshev distance to the nearest anchor of the pixel's own model
  (capped at ``DELTA_CAP``), an input to the experts' sigma and to the LFR gate.

Labelling runs on the CPU (scipy) on the stage-resolution mask, everything else on
the device. No gradients: RAC chooses anchors, it is not trained.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moa.geometry import depth_to_u
from models.moa.reliable_anchor import ReliableAnchor

DELTA_CAP = 32


@dataclass
class RACResult:
    z_rac: torch.Tensor        # [B,1,H,W] metric depth of the region model (1 where invalid)
    valid: torch.Tensor        # [B,1,H,W] bool: mono valid and the sample's primary model exists
    a: torch.Tensor            # [B,K] per-model slope (1 for absent models)
    b: torch.Tensor            # [B,K]
    ok: torch.Tensor           # [B,K] bool: model k accepted
    n_anchor: torch.Tensor     # [B,K] anchors per model
    scale_u: torch.Tensor      # [B,K] robust residual scale of each model's anchors, in u
    model_map: torch.Tensor    # [B,1,H,W] long, model index 0..K-1 used by each pixel
    supported: torch.Tensor    # [B,1,H,W] bool: the pixel's region has anchors
    delta: torch.Tensor        # [B,1,H,W] float: distance to the nearest anchor of its model (px)
    split: torch.Tensor        # [B,1,H,W] float: 1 - winning vote share of the region
    anchor_w: torch.Tensor     # [B,1,H,W] final anchor weight (q * Tukey), 0 = not an anchor
    primary_ok: torch.Tensor   # [B] bool


def _regions(region_ok: np.ndarray, model_id: np.ndarray, weight: np.ndarray, K: int):
    """One sample, numpy. Returns (model_map, supported, split), each [H, W]."""
    from scipy import ndimage

    H, W = region_ok.shape
    lab, n = ndimage.label(region_ok, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if n == 0:
        return np.zeros((H, W), np.int64), np.zeros((H, W), bool), np.zeros((H, W), np.float32)
    anc = model_id > 0
    votes = np.bincount(lab[anc] * K + (model_id[anc] - 1), weights=weight[anc],
                        minlength=(n + 1) * K).reshape(n + 1, K)
    tot = votes.sum(1)
    reg_sup = tot > 0
    reg_model = np.where(reg_sup, votes.argmax(1), 0)
    reg_split = np.where(reg_sup, 1.0 - votes.max(1) / np.maximum(tot, 1e-12), 0.0)
    mm, sup, spl = reg_model[lab], reg_sup[lab], reg_split[lab].astype(np.float32)
    outside = lab == 0
    if outside.any():
        _, (iy, ix) = ndimage.distance_transform_edt(outside, return_indices=True)
        mm, sup, spl = mm[iy, ix], sup[iy, ix], spl[iy, ix]
    return mm.astype(np.int64), sup.astype(bool), spl.astype(np.float32)


def chebyshev_distance(mask: torch.Tensor, cap: int = DELTA_CAP) -> torch.Tensor:
    """Distance (in pixels, Chebyshev) to the nearest True of ``mask`` [B,1,H,W], capped."""
    d = torch.full(mask.shape, float(cap), device=mask.device)
    cur = mask.float()
    d = torch.where(cur > 0, torch.zeros_like(d), d)
    for r in range(1, cap):          # no early exit: a .all() check would sync every step
        nxt = F.max_pool2d(cur, 3, stride=1, padding=1)
        d = torch.where((nxt > 0) & (cur <= 0), torch.full_like(d, float(r)), d)
        cur = nxt
    return d


class RACAligner(nn.Module):
    def __init__(self, tau: float = 0.5, min_eff: float = 64.0, max_models: int = 3,
                 a_ratio: tuple[float, float] = (0.5, 2.0), min_frac: float = 0.1) -> None:
        super().__init__()
        self.K = int(max_models)
        self.ra = ReliableAnchor(tau=tau, min_eff=min_eff, max_models=self.K,
                                 a_ratio=a_ratio, min_frac=min_frac)

    @torch.no_grad()
    def forward(self, z_mono: torch.Tensor, mono_valid: torch.Tensor, z_mvs: torch.Tensor,
                conf: torch.Tensor, edge_bin: torch.Tensor, inv_bin: torch.Tensor,
                vmin: torch.Tensor, vmax: torch.Tensor, du_global: float) -> RACResult:
        """All maps [B,1,H,W] at the working resolution; ``inv_bin`` [B] = 1 / (stage-1 bin width in 1/mm)."""
        B = z_mono.shape[0]
        dev = z_mono.device
        K = self.K
        mv = mono_valid.bool()
        ra = self.ra(z_mono, mv.float(), z_mvs, conf, edge_bin, inv_bin)

        a = torch.ones(B, K, device=dev)
        b = torch.zeros(B, K, device=dev)
        ok = torch.zeros(B, K, dtype=torch.bool, device=dev)
        n_anc = torch.zeros(B, K, device=dev)
        for i, models in enumerate(ra.models):
            for k, (ak, bk, nk) in enumerate(models[:K]):
                a[i, k], b[i, k], ok[i, k], n_anc[i, k] = ak, bk, True, float(nk)

        edge_dil = F.max_pool2d(edge_bin.float(), 3, stride=1, padding=1)
        region_ok = (mv & (edge_dil < 0.5))[:, 0].cpu().numpy()
        mid = ra.model_id[:, 0].cpu().numpy()
        wt = ra.weight[:, 0].float().cpu().numpy()
        mm_l, sup_l, spl_l = [], [], []
        for i in range(B):
            mm, sup, spl = _regions(region_ok[i], mid[i], wt[i], K)
            mm_l.append(mm)
            sup_l.append(sup)
            spl_l.append(spl)
        model_map = torch.from_numpy(np.stack(mm_l)).to(dev).unsqueeze(1)
        supported = torch.from_numpy(np.stack(sup_l)).to(dev).unsqueeze(1)
        split = torch.from_numpy(np.stack(spl_l)).to(dev).unsqueeze(1)
        # a rejected model can only be voted for by its own anchors, which do not exist
        model_map = torch.where(ok.gather(1, model_map.flatten(1)).view_as(model_map), model_map,
                                torch.zeros_like(model_map))

        a_px = a.gather(1, model_map.flatten(1)).view_as(z_mono)
        b_px = b.gather(1, model_map.flatten(1)).view_as(z_mono)
        z = a_px * z_mono.float() + b_px
        primary_ok = ok[:, 0]
        valid = mv & primary_ok.view(B, 1, 1, 1) & torch.isfinite(z) & (z > 0)
        z = torch.where(valid, z, torch.ones_like(z))

        delta = torch.full_like(z, float(DELTA_CAP))
        scale_u = torch.full((B, K), 10.0 * du_global, device=dev)
        anchors = ra.model_id                                   # [B,1,H,W] 0 = none, k+1 = model k
        u_mvs = depth_to_u(z_mvs, vmin, vmax)
        for k in range(K):
            if not bool(ok[:, k].any()):
                continue
            anc_k = anchors == (k + 1)
            dk = chebyshev_distance(anc_k)
            delta = torch.where(model_map == k, dk, delta)
            zk = a[:, k].view(B, 1, 1, 1) * z_mono.float() + b[:, k].view(B, 1, 1, 1)
            rk = (depth_to_u(zk.clamp_min(1e-3), vmin, vmax) - u_mvs).abs()
            for i in range(B):
                m = anc_k[i]
                if bool(ok[i, k]) and int(m.sum()) >= 8:
                    scale_u[i, k] = (1.4826 * rk[i][m].median()).clamp_min(0.1 * du_global)
        return RACResult(z_rac=z, valid=valid, a=a, b=b, ok=ok, n_anchor=n_anc, scale_u=scale_u,
                         model_map=model_map, supported=supported & valid, delta=delta, split=split,
                         anchor_w=ra.weight, primary_ok=primary_ok)
