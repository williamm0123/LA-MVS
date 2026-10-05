"""MoA_s for s in {2, 3, 4}: next-stage hypothesis centre from stage s-1 (MoA.md §8 + 修正).

Shared across the three transitions: edge barrier, WLS solver, shape-encoder
trunk. Per transition: reference-feature projection, MVS evidence encoder,
confidence head, mixture head and the conflict thresholds (tighter at finer
stages). Everything runs at the *parent* resolution and is re-solved at every
transition — affine parameters are never upsampled from a coarser stage.

Every MVS-derived input is detached here, unconditionally: MoA losses train
only MoA parameters and never reshape the matching features, cost volume or
regularizer (MoA.md §12). This is an architectural invariant, not an option.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.moa.edge import NeighborhoodBarrier, downsample_edge
from models.moa.evidence import (
    MVSConfidenceHead, MVSEvidenceEncoder, gather_depth, normalize_cost,
    normalize_src_std, posterior_stats, pv_curvature,
)
from models.moa.geometry import (
    axis_spacing, depth_to_u, interp_along_axis, laplacian, sample_at_feature_pixels,
    spatial_grad, u_to_depth,
)
from models.moa.global_affine import ransac_tukey_global_affine, robust_global_affine
from models.moa.local_affine import MultiScaleLocalAffine, ShapeEncoder
from models.moa.mixture import (
    NUM_EXPERTS, MixtureHead, apply_mvs_override, depth_conflict, mono_proposal,
    prob_conflict, scale_mono_weights, shape_conflict, soft_or,
)

NUM_TRANSITIONS = 3
NUM_SCALES = 3          # 3x3 / 5x5 / 7x7


@dataclass
class MoAOutput:
    center_depth: torch.Tensor        # [B,1,H,W] metric depth of the next-stage centre
    center_u: torch.Tensor            # [B,1,H,W]
    mvs_u: torch.Tensor               # [B,1,H,W] parent MVS centre y
    du: torch.Tensor                  # [B,1,H,W] parent axis spacing (u)
    aligned_mono: torch.Tensor        # [B,1,H,W] globally aligned DA3 depth (0 = invalid)
    mono_valid: torch.Tensor          # [B,1,H,W] float
    mvs_confidence: torch.Tensor      # [B,1,H,W]
    mvs_conf_logit: torch.Tensor      # [B,1,H,W]
    conflict: torch.Tensor            # [B,1,H,W]
    alpha: torch.Tensor               # [B,1,H,W] override strength (detached)
    mixture_weights: torch.Tensor     # [B,5,H,W] after override
    mixture_weights_raw: torch.Tensor  # [B,5,H,W] before override
    experts_u: torch.Tensor           # [B,5,H,W]
    affine_a: torch.Tensor            # [B,3,H,W]
    affine_b: torch.Tensor
    affine_valid: torch.Tensor
    affine_support: torch.Tensor
    affine_offset_only: torch.Tensor
    edge: torch.Tensor                # [B,1,H,W] binary DA3 edge at this resolution
    edge_snap_mono: torch.Tensor      # [B,1,H,W] 1 = edge pixel snapped to the monocular surface
    edge_barrier_rate: torch.Tensor   # [B,3]
    global_a: torch.Tensor            # [B]
    global_b: torch.Tensor            # [B]
    global_ok: torch.Tensor           # [B] bool


class _StageAdapter(nn.Module):
    def __init__(self, cfg, fpn_channels: int, num_groups: int, mix_in: int) -> None:
        super().__init__()
        self.feat_proj = nn.Conv2d(fpn_channels, cfg.feat_dim, 1)
        self.evidence = MVSEvidenceEncoder(num_groups, dim=cfg.evidence_dim, hidden=cfg.evidence_hidden)
        self.conf = MVSConfidenceHead(cfg.evidence_dim, hidden=cfg.conf_hidden)
        self.mixture = MixtureHead(mix_in, hidden=cfg.mix_hidden, mvs_bias_init=cfg.mvs_bias_init)


class MoACascade(nn.Module):
    def __init__(self, cfg, fpn_channels: int, num_groups: int, num_depths_stage1: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.du_global = 1.0 / float(num_depths_stage1 - 1)
        self.barrier = NeighborhoodBarrier(radius=NUM_SCALES)
        shape_in = 6 + NUM_TRANSITIONS + cfg.feat_dim
        self.shape = ShapeEncoder(shape_in, width=cfg.shape_width, out_dim=cfg.emb_dim)
        self.local = MultiScaleLocalAffine(
            self.barrier.offset_list, radius=NUM_SCALES,
            spatial_sigma=tuple(cfg.spatial_sigma), tau_init=cfg.tau_shape_init,
            lambda_a=cfg.lambda_a, lambda_b=cfg.lambda_b, tau_var=cfg.tau_var,
            min_neff=tuple(cfg.min_neff), a_range=tuple(cfg.a_range), b_max=cfg.b_max,
        )
        # emb + F_mvs + 4 expert offsets + 4 posterior stats + 3x(support, valid,
        # offset_only) + 4x(PV ratio, inside, evidence distance) + mono_valid + edge
        mix_in = cfg.emb_dim + cfg.evidence_dim + 4 + 4 + 3 * NUM_SCALES + 3 * 4 + 2
        self.adapters = nn.ModuleList(
            _StageAdapter(cfg, fpn_channels, num_groups, mix_in) for _ in range(NUM_TRANSITIONS))

    def _maybe_ckpt(self, fn, *args):
        # The 49-offset WLS loop and the 3D evidence convs dominate MoA's saved
        # activations and are cheap to recompute.
        if self.training and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, level: int, *, mono_depth: torch.Tensor, mono_valid: torch.Tensor,
                mono_edge: torch.Tensor, z_prev: torch.Tensor, prob: torch.Tensor,
                u_hyp: torch.Tensor, cv: torch.Tensor, n_valid: torch.Tensor,
                src_std: torch.Tensor, num_src: int, ref_feat: torch.Tensor,
                vmin: torch.Tensor, vmax: torch.Tensor) -> MoAOutput:
        """level 0/1/2 -> centre for stage 2/3/4.

        mono_depth / mono_valid / mono_edge: full-resolution [B,1,H,W].
        z_prev [B,1,h,w], prob / u_hyp / n_valid / src_std [B,D,h,w],
        cv [B,G,D,h,w], ref_feat [B,C,h,w] — all at the parent resolution.
        """
        dev = z_prev.device
        with torch.autocast(device_type=dev.type, enabled=False):
            return self._forward(
                level, mono_depth.detach().float(), mono_valid.detach().float(),
                mono_edge.detach().float(), z_prev.detach().float(), prob.detach().float(),
                u_hyp.detach().float(), cv.detach().float(), n_valid.detach().float(),
                src_std.detach().float(), int(num_src), ref_feat.detach().float(),
                vmin.float(), vmax.float())

    def _forward(self, level, mono_depth, mono_valid, mono_edge, z_prev, prob, u_hyp, cv,
                 n_valid, src_std, num_src, ref_feat, vmin, vmax) -> MoAOutput:
        cfg = self.cfg
        ad = self.adapters[level]
        B = z_prev.shape[0]
        hw = tuple(z_prev.shape[-2:])

        # ---- DA3 at the parent resolution -------------------------------------
        zm = sample_at_feature_pixels(mono_depth, hw)
        mv = (sample_at_feature_pixels(mono_valid, hw) > 0.5) & torch.isfinite(zm) & (zm > 0)
        zm = torch.where(mv, zm, torch.ones_like(zm))
        em = downsample_edge(mono_edge, hw, dilate=cfg.edge_dilate)
        edge_bin = (em >= cfg.tau_edge).float()

        # ---- MVS evidence + confidence -----------------------------------------
        y = depth_to_u(z_prev, vmin, vmax)
        du = axis_spacing(u_hyp)
        st = posterior_stats(prob, u_hyp)
        idx = st["mode_idx"]
        curv = pv_curvature(prob)
        nvf = n_valid / float(max(num_src, 1))
        ssn = normalize_src_std(src_std, cv)
        V, F_mvs = self._maybe_ckpt(ad.evidence, normalize_cost(cv), prob, curv, nvf, ssn)
        sig_bins = st["sigma_u"] / (du + 1e-8)
        stats = torch.cat([st["pmax"], st["entropy"], st["gap"], gather_depth(curv, idx),
                           sig_bins.clamp(max=50.0), gather_depth(nvf, idx),
                           gather_depth(ssn, idx)], dim=1)
        r_logit, r = ad.conf(F_mvs, stats)
        r_anchor = r.detach()

        # ---- global affine (per sample, reliable non-edge anchors) -------------
        edge_dil = F.max_pool2d(edge_bin, 3, stride=1, padding=1)
        q = r_anchor * mv.float() * (1.0 - edge_dil)
        solver = cfg.global_solver[level]
        if solver == "ransac":
            # stage-1 bins at every level: DA3's shape error does not shrink with the window
            inv_bin = 1.0 / ((vmax - vmin).reshape(B).float() * self.du_global)
            a_g, b_g, ok_g = ransac_tukey_global_affine(zm, z_prev, q, inv_bin, tau=cfg.global_ransac_tau,
                                                        min_eff=cfg.global_min_eff)
        elif solver == "huber":
            a_g, b_g, ok_g = robust_global_affine(zm, z_prev, q, n_iter=cfg.global_iters,
                                                  huber_k=cfg.global_huber_k, min_eff=cfg.global_min_eff)
        else:
            raise ValueError(f"unknown global_solver {solver!r}")
        zg = a_g.view(B, 1, 1, 1) * zm + b_g.view(B, 1, 1, 1)
        xv = mv & ok_g.view(B, 1, 1, 1) & torch.isfinite(zg) & (zg > 0)
        x = depth_to_u(torch.where(xv, zg, torch.ones_like(zg)), vmin, vmax)
        xv = xv & (x > -0.25) & (x < 1.25)
        x = torch.where(xv, x, y)

        # ---- edge barrier + shape embedding -----------------------------------
        bar = self.barrier(em, zm.log(), mv.float(), cfg.tau_edge, cfg.tau_jump)
        dg = self.du_global
        gx, gy, _, _ = spatial_grad(x, xv)
        lap = laplacian(x, xv)
        onehot = torch.zeros(B, NUM_TRANSITIONS, *hw, device=x.device)
        onehot[:, level] = 1.0
        shape_in = torch.cat([
            x * xv, (gx / dg).clamp(-10, 10), (gy / dg).clamp(-10, 10), (lap / dg).clamp(-10, 10),
            em.clamp(0.0, 1.0), xv.float(), onehot, ad.feat_proj(ref_feat),
        ], dim=1)
        emb = self.shape(shape_in)

        # ---- local affine experts ---------------------------------------------
        la = self._maybe_ckpt(self.local, x, y, xv.float(), torch.ones_like(y), r_anchor, emb, bar["pass"])
        max_shift = float(cfg.max_shift_bins[level]) * du
        experts = torch.cat([y, x, la.x_fit], dim=1)
        experts = y + torch.maximum(torch.minimum(experts - y, max_shift), -max_shift)
        xvf = xv.float()
        expert_valid = torch.cat([torch.ones_like(xvf)] + [xvf] * (NUM_EXPERTS - 1), dim=1)

        # ---- MVS evidence at each monocular expert ----------------------------
        pmax = st["pmax"]
        V_peak = gather_depth(V, idx)
        pv_r, ins, dist = [], [], []
        for j in range(1, NUM_EXPERTS):
            e_j = experts[:, j:j + 1]
            pv_j, in_j = interp_along_axis(prob, u_hyp, e_j)
            V_j, _ = interp_along_axis(V, u_hyp, e_j)
            pv_r.append(pv_j / (pmax + 1e-8))
            ins.append(in_j.float())
            dist.append((V_j - V_peak).norm(dim=1, keepdim=True) / math.sqrt(V.shape[1]) * in_j.float())
        mix_in = torch.cat([
            emb, F_mvs, ((experts[:, 1:] - y) / (du + 1e-8)).clamp(-50, 50),
            pmax, st["entropy"], st["gap"], sig_bins.clamp(max=50.0),
            la.support, la.valid, la.offset_only,
            *pv_r, *ins, *dist, xvf, edge_bin,
        ], dim=1)
        pi = ad.mixture(mix_in, expert_valid)

        # ---- conflict + deterministic MVS override ----------------------------
        x_prop = mono_proposal(pi, experts, y)
        pv_prop, _ = interp_along_axis(prob, u_hyp, x_prop.detach())
        conflict = soft_or(
            depth_conflict(x_prop, y, st["sigma_u"], du, cfg.conflict_t_d[level], cfg.conflict_T_d[level]),
            prob_conflict(pv_prop, pmax),
            shape_conflict(x_prop, y, edge_dil, du, cfg.conflict_tau_s),
        )
        alpha = (r_anchor * conflict).detach()
        pi_t = scale_mono_weights(apply_mvs_override(pi, alpha), float(cfg.moa_gain[level]))
        u_mix = (pi_t * experts).sum(dim=1, keepdim=True)

        # ---- depth-edge snap: pick a surface, never the average of two -------
        snap_mono = torch.zeros_like(u_mix)
        u_c = u_mix
        if bool(cfg.edge_snap[level]):
            to_mono = (u_mix - x_prop).abs() < (u_mix - y).abs()
            at_edge = edge_bin > 0.5
            u_c = torch.where(at_edge, torch.where(to_mono, x_prop, y), u_mix)
            snap_mono = (at_edge & to_mono).float()

        return MoAOutput(
            center_depth=u_to_depth(u_c, vmin, vmax), center_u=u_c, mvs_u=y, du=du,
            aligned_mono=torch.where(xv, zg, torch.zeros_like(zg)), mono_valid=xvf,
            mvs_confidence=r, mvs_conf_logit=r_logit, conflict=conflict.detach(), alpha=alpha,
            mixture_weights=pi_t, mixture_weights_raw=pi, experts_u=experts,
            affine_a=la.a, affine_b=la.b, affine_valid=la.valid, affine_support=la.support,
            affine_offset_only=la.offset_only, edge=edge_bin, edge_snap_mono=snap_mono,
            edge_barrier_rate=bar["blocked_rate"], global_a=a_g, global_b=b_g, global_ok=ok_g,
        )
