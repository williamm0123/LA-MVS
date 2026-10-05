"""LAPE cascade: the monocular prior for stage 1 (second pass) and for every transition.

Built on the MoA transition (models/moa/moa.py), with the changes of the LAPE plan:

* (1) global alignment = RAC (models/moa/rac.py): up to three global affines and a
  per-region model instead of one affine for the whole image;
* (2) monocular experts = [E_RAC, E3, E7, E11] (models/moa/lape_affine.py), each with
  a calibrated sigma (models/moa/calibrator.py); the mixture still has the MVS centre
  as expert 0, so it keeps its five outputs;
* (4) low-frequency recall (models/moa/low_freq_recall.py) on transitions: where MVS
  has collapsed, the next centre is the region model's depth, the MVS override and
  the edge snap are skipped, and the window is not stretched back to the MVS centre;
* outputs the prior (mu, sigma, mixture weight, validity) that the network turns
  into candidate evidence for the next stage (models/moa/prior_evidence.py).

``level`` 0 = stage-1 prior (same resolution, no centre); 1..3 = the transitions
that produce the centres of stages 2..4. Every MVS-derived input is detached, as in
MoA: the MoA losses and the sigma NLL train only these parameters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.moa.calibrator import SigmaCalibrator, calibrated_sigma
from models.moa.edge import NeighborhoodBarrier, downsample_edge
from models.moa.evidence import (
    MVSConfidenceHead, MVSEvidenceEncoder, gather_depth, normalize_cost, normalize_src_std,
    posterior_stats, pv_curvature,
)
from models.moa.geometry import (
    axis_spacing, depth_to_u, interp_along_axis, laplacian, sample_at_feature_pixels, spatial_grad,
    u_to_depth, upsample_nearest,
)
from models.moa.lape_affine import WindowAffineExperts, expert_offsets
from models.moa.local_affine import ShapeEncoder
from models.moa.low_freq_recall import OFFSETS7, low_freq_recall
from models.moa.mixture import (
    NUM_EXPERTS, MixtureHead, apply_mvs_override, depth_conflict, mono_proposal, prob_conflict,
    scale_mono_weights, shape_conflict, soft_or,
)
from models.moa.prior_evidence import normalized_weights
from models.moa.rac import RACAligner

NUM_LEVELS = 4
NUM_MONO = 4          # E_RAC, E3, E7, E11
MONO_NAMES = ("rac", "w3", "w7", "w11")


@dataclass
class LAPEOutput:
    level: int
    # prior (parent resolution)
    mu: torch.Tensor              # [B,4,h,w] monocular experts in u (unclamped)
    sigma: torch.Tensor           # [B,4,h,w] calibrated std in u (grad -> calibrators only)
    expert_valid: torch.Tensor    # [B,4,h,w] float
    prior_w: torch.Tensor         # [B,4,h,w] normalised monocular mixture weights (detached)
    sigma_feats: torch.Tensor     # [B,4,8,h,w] calibrator inputs (diagnostics)
    # alignment state needed to lift the prior to the child resolution
    x: torch.Tensor               # [B,1,h,w] RAC-aligned DA3 in u (y where invalid)
    x_valid: torch.Tensor         # [B,1,h,w] bool
    win_alpha: torch.Tensor       # [B,3,h,w]
    win_bt: torch.Tensor          # [B,3,h,w]
    win_valid: torch.Tensor       # [B,3,h,w]
    rac_a: torch.Tensor           # [B,K]
    rac_b: torch.Tensor           # [B,K]
    rac_ok: torch.Tensor          # [B,K] bool
    rac_n: torch.Tensor           # [B,K]
    rac_model_map: torch.Tensor   # [B,1,h,w] long
    rac_supported: torch.Tensor   # [B,1,h,w] bool
    rac_delta: torch.Tensor       # [B,1,h,w]
    rac_split: torch.Tensor       # [B,1,h,w]
    global_ok: torch.Tensor       # [B] bool (RAC primary model exists)
    # MVS side (same names as MoAOutput where the loss reads them)
    mvs_u: torch.Tensor
    du: torch.Tensor
    mvs_confidence: torch.Tensor
    mvs_conf_logit: torch.Tensor
    edge: torch.Tensor            # [B,1,h,w] binary DA3 edge
    mixture_weights_raw: torch.Tensor
    experts_u: torch.Tensor       # [B,5,h,w] clamped experts for the centre
    # centre (transitions only)
    center_u: torch.Tensor | None = None
    center_depth: torch.Tensor | None = None
    mixture_weights: torch.Tensor | None = None
    conflict: torch.Tensor | None = None
    alpha: torch.Tensor | None = None
    edge_snap_mono: torch.Tensor | None = None
    lfr_flag: torch.Tensor | None = None      # [B,1,h,w] bool
    lfr_soft: torch.Tensor | None = None
    mono_valid: torch.Tensor | None = None


class _LevelAdapter(nn.Module):
    def __init__(self, cfg, fpn_channels: int, num_groups: int, mix_in: int) -> None:
        super().__init__()
        self.feat_proj = nn.Conv2d(fpn_channels, cfg.feat_dim, 1)
        self.evidence = MVSEvidenceEncoder(num_groups, dim=cfg.evidence_dim, hidden=cfg.evidence_hidden)
        self.conf = MVSConfidenceHead(cfg.evidence_dim, hidden=cfg.conf_hidden)
        self.mixture = MixtureHead(mix_in, hidden=cfg.mix_hidden, mvs_bias_init=cfg.mvs_bias_init)


class LAPECascade(nn.Module):
    def __init__(self, cfg, lcfg, cascade_cfg, fpn_channels: int, num_groups: int,
                 num_depths_stage1: int) -> None:
        super().__init__()
        self.cfg, self.lcfg = cfg, lcfg
        self.halfwidth = tuple(float(h) for h in cascade_cfg.window_halfwidth_bins)
        self.du_global = 1.0 / float(num_depths_stage1 - 1)
        offs, _ = expert_offsets()
        self.barrier = NeighborhoodBarrier(radius=5, subset=offs)
        self.register_buffer("idx7", torch.tensor([offs.index(o) for o in OFFSETS7], dtype=torch.long),
                             persistent=False)
        self.shape = ShapeEncoder(6 + NUM_LEVELS + cfg.feat_dim, width=cfg.shape_width, out_dim=cfg.emb_dim)
        self.windows = WindowAffineExperts(
            spatial_sigma=tuple(lcfg.spatial_sigma), tau_init=cfg.tau_shape_init, lambda_a=cfg.lambda_a,
            lambda_b=cfg.lambda_b, tau_var=cfg.tau_var, min_neff=tuple(lcfg.min_neff),
            a_range=tuple(cfg.a_range), b_max=cfg.b_max)
        self.rac = RACAligner(tau=cfg.global_ransac_tau, min_eff=cfg.global_min_eff,
                              max_models=lcfg.rac_max_models, a_ratio=tuple(lcfg.rac_a_ratio),
                              min_frac=lcfg.rac_min_frac)
        # emb + F_mvs + 4 offsets + 4 posterior stats + 3x(support, valid, offset_only)
        # + 4x(PV ratio, inside, evidence distance) + mono_valid + edge + 4 log(sigma/du)
        # + RAC supported + RAC split
        mix_in = cfg.emb_dim + cfg.evidence_dim + 4 + 4 + 9 + 12 + 2 + 4 + 2
        self.adapters = nn.ModuleList(
            _LevelAdapter(cfg, fpn_channels, num_groups, mix_in) for _ in range(NUM_LEVELS))
        self.calibrators = nn.ModuleList(
            nn.ModuleList(SigmaCalibrator() for _ in range(NUM_MONO)) for _ in range(NUM_LEVELS))

    def _maybe_ckpt(self, fn, *args):
        if self.training and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, level: int, *, mono_depth, mono_valid, mono_edge, gray, z_prev, prob, u_hyp, cv,
                n_valid, src_std, num_src: int, ref_feat, vmin, vmax) -> LAPEOutput:
        dev = z_prev.device
        with torch.autocast(device_type=dev.type, enabled=False):
            return self._forward(
                int(level), mono_depth.detach().float(), mono_valid.detach().float(), mono_edge.detach().float(),
                gray.detach().float(), z_prev.detach().float(), prob.detach().float(), u_hyp.detach().float(),
                cv.detach().float(), n_valid.detach().float(), src_std.detach().float(), int(num_src),
                ref_feat.detach().float(), vmin.float(), vmax.float())

    def _forward(self, level, mono_depth, mono_valid, mono_edge, gray, z_prev, prob, u_hyp, cv,
                 n_valid, src_std, num_src, ref_feat, vmin, vmax) -> LAPEOutput:
        cfg, lc = self.cfg, self.lcfg
        ad = self.adapters[level]
        want_center = level > 0
        t = max(level - 1, 0)                       # transition index for per-transition knobs
        B = z_prev.shape[0]
        hw = tuple(z_prev.shape[-2:])
        dg = self.du_global

        # ---- DA3 at this resolution ---------------------------------------
        zm = sample_at_feature_pixels(mono_depth, hw)
        mv = (sample_at_feature_pixels(mono_valid, hw) > 0.5) & torch.isfinite(zm) & (zm > 0)
        zm = torch.where(mv, zm, torch.ones_like(zm))
        em = downsample_edge(mono_edge, hw, dilate=cfg.edge_dilate)
        edge_bin = (em >= cfg.tau_edge).float()

        # ---- MVS evidence + confidence --------------------------------------
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
                           sig_bins.clamp(max=50.0), gather_depth(nvf, idx), gather_depth(ssn, idx)], dim=1)
        r_logit, r = ad.conf(F_mvs, stats)
        r_anchor = r.detach()

        # ---- (1) RAC ------------------------------------------------------------
        inv_bin = 1.0 / ((vmax - vmin).reshape(B).float() * dg)
        rac = self.rac(zm, mv, z_prev, r_anchor, edge_bin, inv_bin, vmin, vmax, dg)
        xv = rac.valid
        x = depth_to_u(torch.where(xv, rac.z_rac, torch.ones_like(rac.z_rac)), vmin, vmax)
        xv = xv & (x > -0.25) & (x < 1.25)
        x = torch.where(xv, x, y)
        xvf = xv.float()

        # ---- edge barrier + shape embedding -----------------------------------
        bar = self.barrier(em, zm.log(), mv.float(), cfg.tau_edge, cfg.tau_jump)
        gx, gy, _, _ = spatial_grad(x, xv)
        lap = laplacian(x, xv)
        onehot = torch.zeros(B, NUM_LEVELS, *hw, device=x.device)
        onehot[:, level] = 1.0
        shape_in = torch.cat([
            x * xvf, (gx / dg).clamp(-10, 10), (gy / dg).clamp(-10, 10), (lap / dg).clamp(-10, 10),
            em.clamp(0.0, 1.0), xvf, onehot, ad.feat_proj(ref_feat)], dim=1)
        emb = self.shape(shape_in)

        # ---- (2) window experts + E_RAC -------------------------------------------
        we = self._maybe_ckpt(self.windows, x, y, xvf, r_anchor, emb, bar["pass"])
        mm = rac.model_map.flatten(1)
        s_rac = rac.scale_u.gather(1, mm).view_as(x)
        n_rac = rac.n_anchor.gather(1, mm).view_as(x)
        sp_rac = s_rac / n_rac.clamp_min(1.0).sqrt()
        unsup = (~rac.supported).float()
        mu = torch.cat([x, we.mu], dim=1)
        valid4 = torch.cat([xvf, we.valid * xvf], dim=1)
        s_list = [s_rac] + [we.s_ho[:, s:s + 1] for s in range(3)]
        p_list = [sp_rac] + [we.sigma_par[:, s:s + 1] for s in range(3)]
        d_list = [rac.delta] + [we.delta[:, s:s + 1] for s in range(3)]
        n_list = [n_rac] + [we.n_eff[:, s:s + 1] for s in range(3)]
        f_list = [unsup] + [we.offset_only[:, s:s + 1] for s in range(3)]
        r_img = (r_anchor * xvf).sum(dim=(2, 3), keepdim=True) / xvf.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        sig, feats = [], []
        for j in range(NUM_MONO):
            sj, fj = calibrated_sigma(self.calibrators[level][j], s_list[j], p_list[j], d_list[j],
                                      n_list[j], edge_bin, f_list[j], dg, lc.delta0, lc.sigma_floor_bins,
                                      r=r_anchor, r_img=r_img)
            sig.append(sj)
            feats.append(fj)
        sigma = torch.cat(sig, dim=1)
        sig_d = sigma.detach()

        # ---- mixture --------------------------------------------------------------
        max_shift = float(cfg.max_shift_bins[t]) * du
        experts = torch.cat([y, mu], dim=1)
        experts = y + torch.maximum(torch.minimum(experts - y, max_shift), -max_shift)
        expert_valid = torch.cat([torch.ones_like(xvf), valid4], dim=1)
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
        log_sig = torch.log(sig_d / (du + 1e-8)).clamp(-6.0, 6.0) * valid4
        mix_in = torch.cat([
            emb, F_mvs, ((experts[:, 1:] - y) / (du + 1e-8)).clamp(-50, 50),
            pmax, st["entropy"], st["gap"], sig_bins.clamp(max=50.0),
            we.support, we.valid, we.offset_only, *pv_r, *ins, *dist, xvf, edge_bin,
            log_sig, rac.supported.float(), rac.split], dim=1)
        pi = ad.mixture(mix_in, expert_valid)
        prior_w = normalized_weights(pi[:, 1:].detach(), valid4)

        out = LAPEOutput(
            level=level, mu=mu, sigma=sigma, expert_valid=valid4, prior_w=prior_w,
            sigma_feats=torch.stack(feats, dim=1).detach(),
            x=x, x_valid=xv, win_alpha=we.alpha.detach(), win_bt=we.bt.detach(), win_valid=we.valid.detach(),
            rac_a=rac.a, rac_b=rac.b, rac_ok=rac.ok, rac_n=rac.n_anchor, rac_model_map=rac.model_map,
            rac_supported=rac.supported, rac_delta=rac.delta, rac_split=rac.split, global_ok=rac.primary_ok,
            mvs_u=y, du=du, mvs_confidence=r, mvs_conf_logit=r_logit, edge=edge_bin,
            mixture_weights_raw=pi, experts_u=experts, mono_valid=xvf)
        if not want_center:
            return out

        # ---- conflict + deterministic MVS override (MoA) ----------------------------
        x_prop = mono_proposal(pi, experts, y)
        pv_prop, _ = interp_along_axis(prob, u_hyp, x_prop.detach())
        edge_dil = torch.nn.functional.max_pool2d(edge_bin, 3, stride=1, padding=1)
        conflict = soft_or(
            depth_conflict(x_prop, y, st["sigma_u"], du, cfg.conflict_t_d[t], cfg.conflict_T_d[t]),
            prob_conflict(pv_prop, pmax),
            shape_conflict(x_prop, y, edge_dil, du, cfg.conflict_tau_s))
        alpha = (r_anchor * conflict).detach()

        # ---- (4) low-frequency recall -----------------------------------------------
        if lc.lfr and bool(lc.lfr_levels[t]):
            lfr = low_freq_recall(gray, x, xv, y, du, bar["pass"].index_select(1, self.idx7),
                                  rac.supported, rac.delta, self.halfwidth[t], lc, dg, conf=r_anchor)
            F_ = lfr.flag
            lfr_soft = lfr.soft
        else:
            F_ = torch.zeros_like(xv)
            lfr_soft = torch.zeros_like(xvf)
        alpha = torch.where(F_, torch.zeros_like(alpha), alpha)
        pi_t = scale_mono_weights(apply_mvs_override(pi, alpha), float(cfg.moa_gain[t]))
        u_mix = (pi_t * experts).sum(dim=1, keepdim=True)

        snap_mono = torch.zeros_like(u_mix)
        u_c = u_mix
        if bool(cfg.edge_snap[t]):
            to_mono = (u_mix - x_prop).abs() < (u_mix - y).abs()
            at_edge = (edge_bin > 0.5) & ~F_
            u_c = torch.where(at_edge, torch.where(to_mono, x_prop, y), u_mix)
            snap_mono = (at_edge & to_mono).float()
        u_c = torch.where(F_, x, u_c)

        out.center_u = u_c
        out.center_depth = u_to_depth(u_c, vmin, vmax)
        out.mixture_weights = pi_t
        out.conflict = conflict.detach()
        out.alpha = alpha
        out.edge_snap_mono = snap_mono
        out.lfr_flag = F_
        out.lfr_soft = lfr_soft
        return out

    @torch.no_grad()
    def lift(self, out: LAPEOutput, child_hw: tuple[int, int], mono_depth: torch.Tensor,
             mono_valid: torch.Tensor, vmin: torch.Tensor, vmax: torch.Tensor) -> dict:
        """The prior on the child grid: the region model and the window experts' (alpha, b~)
        are applied to the child-resolution DA3, so the detail comes from DA3, not from
        upsampling the parent prediction. Sigma / weights / flags are nearest-upsampled."""
        B = mono_depth.shape[0]
        zc = sample_at_feature_pixels(mono_depth.float(), child_hw)
        mvc = (sample_at_feature_pixels(mono_valid.float(), child_hw) > 0.5) & torch.isfinite(zc) & (zc > 0)
        mm = upsample_nearest(out.rac_model_map.float(), child_hw).long()
        a_px = out.rac_a.gather(1, mm.flatten(1)).view(B, 1, *child_hw)
        b_px = out.rac_b.gather(1, mm.flatten(1)).view(B, 1, *child_hw)
        z = a_px * zc + b_px
        vc = mvc & out.global_ok.view(B, 1, 1, 1) & torch.isfinite(z) & (z > 0)
        z = torch.where(vc, z, torch.ones_like(z))
        xc = depth_to_u(z, vmin, vmax)
        vc = vc & (xc > -0.25) & (xc < 1.25)
        up = lambda t_: upsample_nearest(t_.float(), child_hw)
        x_up = up(out.x)
        mu_w = xc + up(out.win_bt) + up(out.win_alpha) * (xc - x_up)
        mu = torch.cat([xc, mu_w], dim=1)
        valid = torch.cat([vc.float(), up(out.win_valid) * vc.float()], dim=1)
        w = normalized_weights(up(out.prior_w), valid)
        return {"mu": mu, "sigma": up(out.sigma.detach()), "w": w, "valid": valid,
                "z_rac": z, "z_valid": vc, "delta": up(out.rac_delta), "split": up(out.rac_split),
                "lfr_flag": up(out.lfr_flag) if out.lfr_flag is not None else torch.zeros_like(xc),
                "lfr_soft": up(out.lfr_soft) if out.lfr_soft is not None else torch.zeros_like(xc)}
