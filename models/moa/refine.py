"""Final recurrent refinement of the stage-4 depth (IterMVS / RAFT style).

Stage 4 searches 4 candidates in a window of ~0.3 stage-1 bins and regresses within
+-1 bin of the argmax, so the inlier precision that DTU's chamfer measures is fixed
there. This module works on the 1/2-resolution grid (the stride-2 features):

    u_0 = stage-4 depth (sampled at the plane-sweep pixels), state h from the reference
    for t = 1..T:
        plane-sweep 2r+1 candidates u_t + k*delta (delta = spacing/2) with a light
        32-channel projection of the same FPN features -> normalised correlation
        + the LAPE prior at this grid (mixture mean / std of the level-3 experts)
        ConvGRU -> du_t (bounded by one stage-4 spacing), convex-upsampling mask
    full-res depth = stage-4 full-res u + convex-up(u_T - u_0)

Only the *residual* is upsampled, so the stage-4 full-resolution detail is kept.
Zero-initialised update head: at step 0 the output is the stage-4 depth. Features and
the stage-4 depth are detached, so the refinement losses never reach the cascade.
The head also gives a confidence (P(|error| < spacing)) saved for fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moa.affine_gru import SepConvGRU
from models.moa.cost_volume import MoACostVolume
from models.moa.evidence import normalize_cost
from models.moa.geometry import axis_spacing, depth_to_u, sample_at_feature_pixels, u_to_depth
from models.moa.prior_evidence import mixture_moments


def convex_upsample(x: torch.Tensor, mask: torch.Tensor, f: int) -> torch.Tensor:
    """RAFT convex upsampling. x [B,1,h,w], mask [B,9*f*f,h,w] -> [B,1,f*h,f*w]."""
    B, _, h, w = x.shape
    m = mask.view(B, 1, 9, f, f, h, w).softmax(dim=2)
    up = F.unfold(x, 3, padding=1).view(B, 1, 9, 1, 1, h, w)
    up = (m * up).sum(dim=2)                                           # [B,1,f,f,h,w]
    return up.permute(0, 1, 4, 2, 5, 3).reshape(B, 1, f * h, f * w)


class DepthRefiner(nn.Module):
    factor = 2

    def __init__(self, fpn_channels: int, num_groups: int = 8, warp_channels: int = 32, hidden: int = 64,
                 radius: int = 2, iters: int = 4, use_half: bool = True) -> None:
        super().__init__()
        self.ks = tuple(range(radius, -radius - 1, -1))     # descending u = ascending depth
        self.iters = int(iters)
        D = len(self.ks)
        self.cv = MoACostVolume(fpn_channels, warp_channels, num_groups, use_half)
        self.ctx = nn.Sequential(nn.Conv2d(fpn_channels + 1, 2 * hidden, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(2 * hidden, 2 * hidden, 3, padding=1))
        n_in = num_groups * D + D + 4
        self.enc = nn.Sequential(nn.Conv2d(n_in, 96, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(96, hidden, 3, padding=1), nn.SiLU())
        self.gru = SepConvGRU(hidden, 2 * hidden)
        self.delta = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
                                   nn.Conv2d(hidden, 2, 3, padding=1))       # du, confidence logit
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.mask = nn.Sequential(nn.Conv2d(hidden, 2 * hidden, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(2 * hidden, 9 * self.factor ** 2, 1))

    def forward(self, feat2: torch.Tensor, K: torch.Tensor, E: torch.Tensor, stage4: dict, lape4,
                vmin: torch.Tensor, vmax: torch.Tensor) -> dict:
        """feat2 [B,V,C,h,w] stride-2 features; stage4: the decoded stage-4 dict; lape4: the
        level-3 LAPEOutput (same grid as feat2) or None."""
        feat2 = feat2.detach()
        hw = tuple(feat2.shape[-2:])
        u_full = depth_to_u(stage4["depth"].detach(), vmin, vmax)                # [B,1,H,W]
        du_full = axis_spacing(stage4["u_hypos"].detach())
        H, W = u_full.shape[-2:]
        u0 = sample_at_feature_pixels(u_full, hw)
        du = sample_at_feature_pixels(du_full, hw).clamp_min(1e-6)
        delta = 0.5 * du
        pmax = sample_at_feature_pixels(stage4["prob"].detach().float().amax(1, keepdim=True), hw)

        if lape4 is not None and tuple(lape4.mu.shape[-2:]) == hw:
            mbar, sbar = mixture_moments(lape4.mu.detach(), lape4.sigma.detach(), lape4.prior_w.detach())
            act = (lape4.prior_w.sum(1, keepdim=True) > 0).float()
        else:
            mbar, sbar, act = u0, torch.ones_like(u0), torch.zeros_like(u0)

        h, c = self.ctx(torch.cat([feat2[:, 0], pmax], dim=1)).chunk(2, dim=1)
        h, c = torch.tanh(h), F.relu(c)
        u = u0
        seq, logits = [], None
        for _ in range(self.iters):
            u = u.detach()
            hyp = torch.cat([u + k * delta for k in self.ks], dim=1)
            cvo = self.cv(feat2[:, 0], feat2[:, 1:], K[:, 0], K[:, 1:], E[:, 0], E[:, 1:],
                          u_to_depth(hyp, vmin, vmax), feature_stride=2)
            cvn = normalize_cost(cvo.cv).flatten(1, 2)
            nv = cvo.n_valid / float(max(cvo.num_src, 1))
            x = torch.cat([cvn, nv, ((u - u0) / du).clamp(-8, 8) / 8,
                           ((mbar - u) / du).clamp(-8, 8) / 8 * act,
                           (torch.log(sbar / du).clamp(-4, 4) / 4) * act, act], dim=1)
            h = self.gru(h, torch.cat([self.enc(x.float()), c], dim=1))
            o = self.delta(h).float()
            u = u + du * torch.tanh(o[:, :1])
            logits = o[:, 1:2]
            res = convex_upsample(u - u0, 0.25 * self.mask(h).float(), self.factor)
            if tuple(res.shape[-2:]) != (H, W):
                res = F.interpolate(res, size=(H, W), mode="bilinear", align_corners=False)
            seq.append(u_full + res)
        conf_logit = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        u_out = seq[-1] if seq else u_full
        return {"u": u_out, "depth": u_to_depth(u_out, vmin, vmax), "seq": seq, "u0": u_full,
                "du": du_full, "conf_logit": conf_logit, "conf": torch.sigmoid(conf_logit)}
