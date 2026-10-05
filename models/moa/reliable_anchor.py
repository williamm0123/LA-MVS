"""RA (reliable anchor) module: pick the MVS pixels a monocular map may be anchored to.

Factored out of ``MoA._forward`` (the q + RANSAC+Tukey block) so every stage can
reuse the same selection. Inputs live at the stage resolution:

    q      = r * mono_valid * (1 - dilate3x3(edge))      candidate weight
    a, b   = ransac_tukey_global_affine(z_mono, z_mvs, q) global DA3 -> MVS fit
    weight = q * tukey(residual)                         0 where the fit failed
    anchor = weight > 0

With ``max_models > 1`` further affines are fitted to the candidates the previous
models rejected (see ``ReliableAnchor``). The module only selects anchors and fits
the affines; applying them to the monocular map is left to the caller.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moa.global_affine import ransac_tukey_global_affine


@dataclass
class RAOutput:
    a: torch.Tensor          # [B] primary (model 1) affine
    b: torch.Tensor          # [B]
    ok: torch.Tensor         # [B] bool
    q: torch.Tensor          # [B,1,H,W] candidate weight
    weight: torch.Tensor     # [B,1,H,W] final anchor weight (q * tukey of the anchor's own model)
    anchor: torch.Tensor     # [B,1,H,W] bool
    model_id: torch.Tensor   # [B,1,H,W] long, 0 = not an anchor, k = anchor of model k
    models: list             # per sample: [(a, b, n_anchors), ...] accepted models, primary first


class ReliableAnchor(nn.Module):
    """Parameter-free; ``tau`` / ``min_eff`` as ``MoAConfig.global_ransac_tau / global_min_eff``.

    ``max_models > 1``: sequential RANSAC. DA3's foreground and background need not share
    one affine, so a single fit keeps only the dominant (usually table) surface and
    rejects correct MVS on the object. After each accepted model its anchors are removed
    from q and RANSAC+Tukey runs again on the rest. A secondary model k is accepted only if
    ``a_k / a_1`` lies in ``a_ratio`` (a collapsed, near fronto-parallel wrong MVS region
    fits DA3 with a tiny slope) and it has at least ``min_frac`` x model 1's anchors.
    ``max_models = 1`` is exactly the single-model RA.
    """

    def __init__(self, tau: float = 0.5, min_eff: float = 64.0, max_models: int = 1,
                 a_ratio: tuple[float, float] = (0.5, 2.0), min_frac: float = 0.1):
        super().__init__()
        self.tau = float(tau)
        self.min_eff = float(min_eff)
        self.max_models = int(max_models)
        self.a_ratio = (float(a_ratio[0]), float(a_ratio[1]))
        self.min_frac = float(min_frac)

    @torch.no_grad()
    def forward(self, z_mono: torch.Tensor, mono_valid: torch.Tensor, z_mvs: torch.Tensor,
                conf: torch.Tensor, edge_bin: torch.Tensor, inv_bin: torch.Tensor) -> RAOutput:
        """z_mono / mono_valid / z_mvs / conf / edge_bin: [B,1,H,W]; inv_bin [B] = 1 / ((vmax-vmin) * du_bin)."""
        edge_dil = F.max_pool2d(edge_bin.float(), 3, stride=1, padding=1)
        q = conf.float() * mono_valid.float() * (1.0 - edge_dil)
        zm, zv = z_mono.float(), z_mvs.float()
        a, b, ok, w = ransac_tukey_global_affine(zm, zv, q, inv_bin, tau=self.tau,
                                                 min_eff=self.min_eff, return_weights=True)
        B = q.shape[0]
        bv = lambda x: x.view(B, 1, 1, 1)
        anc = w > 0
        model_id = anc.long()
        n1 = anc.flatten(1).sum(1)
        models = [[(float(a[i]), float(b[i]), int(n1[i]))] if bool(ok[i]) else [] for i in range(B)]
        rem = q * (~anc)
        alive = ok.clone()
        for k in range(2, self.max_models + 1):
            if not bool(alive.any()):
                break
            ak, bk, okk, wk = ransac_tukey_global_affine(zm, zv, rem * bv(alive.float()), inv_bin, tau=self.tau,
                                                         min_eff=self.min_eff, return_weights=True)
            nk = (wk > 0).flatten(1).sum(1)
            ratio = ak / a.clamp_min(1e-12)
            acc = alive & okk & (ratio >= self.a_ratio[0]) & (ratio <= self.a_ratio[1]) & (nk >= self.min_frac * n1)
            new = (wk > 0) & bv(acc)
            w = torch.where(new, wk, w)
            model_id = torch.where(new, torch.full_like(model_id, k), model_id)
            for i in range(B):
                if bool(acc[i]):
                    models[i].append((float(ak[i]), float(bk[i]), int(nk[i])))
            rem = rem * (~new)
            alive = acc          # stop a sample at its first rejected model
        return RAOutput(a=a, b=b, ok=ok, q=q, weight=w, anchor=w > 0, model_id=model_id, models=models)
