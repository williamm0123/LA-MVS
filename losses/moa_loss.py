"""Loss for MoAMVSNet (MoA.md §11).

    L = sum_s w_s CE_s  +  sum_{s=2..4} lam_s (w_c L_center + w_sh L_shape + w_r L_conf)

CE_s is the two-bin soft-label CE of each MVS stage, restricted to pixels whose
GT lies inside that stage's window (a GT outside the window has no correct bin;
getting it inside is the job of the centre, i.e. of L_center). GT is used only
here, never in the forward pass.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from base.config_moa import MoALossConfig
from losses.depth_loss import soft_label_cross_entropy
from models.moa.geometry import depth_to_u, sample_at_feature_pixels, u_to_depth


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    m = m.float()
    return (x * m).sum() / m.sum().clamp_min(1.0)


def _gt_at(gt: torch.Tensor, mask: torch.Tensor, hw) -> tuple[torch.Tensor, torch.Tensor]:
    g = sample_at_feature_pixels(gt.unsqueeze(1), hw)
    m = sample_at_feature_pixels(mask.float().unsqueeze(1), hw) > 0.5
    return g, m & (g > 0)


class MoALoss:
    def __init__(self, cfg: MoALossConfig, num_depths_stage1: int) -> None:
        self.cfg = cfg
        self.du_global = 1.0 / float(num_depths_stage1 - 1)

    def __call__(self, outputs: dict, batch: dict, diagnostics: bool = False
                 ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns (loss, logs). Log values are detached 0-d tensors (no host sync);
        ``diagnostics`` adds coverage / error / MoA statistics — meant for log steps."""
        cfg = self.cfg
        gt = batch["depth_gt"].float()
        dv = batch["depth_values"].float()
        mask = batch["mask"].float() > 0.5
        mask &= (gt >= dv.amin(dim=1).view(-1, 1, 1)) & (gt <= dv.amax(dim=1).view(-1, 1, 1))
        vmin, vmax = outputs["vmin"], outputs["vmax"]
        total = gt.new_zeros(())
        logs: dict[str, torch.Tensor] = {}

        for s in range(1, 5):
            st = outputs[f"stage{s}"]
            hyp = st["depth_hypos"]
            g, m = _gt_at(gt, mask, tuple(hyp.shape[-2:]))
            inside = (g >= hyp[:, :1]) & (g <= hyp[:, -1:])
            ce = soft_label_cross_entropy(st["logits"], hyp, g[:, 0], (m & inside)[:, 0])
            total = total + cfg.stage_weights[s - 1] * ce
            logs[f"ce_s{s}"] = ce.detach()
            if diagnostics:
                with torch.no_grad():
                    logs[f"cover_s{s}"] = _masked_mean(inside.float(), m)
                    logs[f"err_s{s}_mm"] = _masked_mean((st["depth"] - g).abs(), m)
                    fu = outputs.get(f"fusion{s}")
                    if fu is not None:
                        logs.update(fusion_diagnostics(st, fu, g, m & inside, s))

        s1a = outputs.get("stage1a")
        if s1a is not None:
            hyp = s1a["depth_hypos"]
            g, m = _gt_at(gt, mask, tuple(hyp.shape[-2:]))
            inside = (g >= hyp[:, :1]) & (g <= hyp[:, -1:])
            ce = soft_label_cross_entropy(s1a["logits"], hyp, g[:, 0], (m & inside)[:, 0])
            total = total + cfg.w_ce_stage1a * ce
            logs["ce_s1a"] = ce.detach()
            if diagnostics:
                with torch.no_grad():
                    logs["err_s1a_mm"] = _masked_mean((s1a["depth"] - g).abs(), m)

        # LAPE sigma calibration (Gaussian NLL, residual capped) + level-0 confidence
        for key in ("lape1", "moa2", "moa3", "moa4"):
            lo = outputs.get(key)
            if lo is None or not hasattr(lo, "sigma"):
                continue
            hw = tuple(lo.mu.shape[-2:])
            g, m = _gt_at(gt, mask, hw)
            u_gt = depth_to_u(torch.where(m, g, torch.ones_like(g)), vmin, vmax)
            dg = self.du_global
            res = ((lo.mu.detach() - u_gt) / dg).clamp(-cfg.sigma_nll_cap_bins, cfg.sigma_nll_cap_bins)
            sb = lo.sigma / dg
            z = (res / sb).abs()
            c = cfg.sigma_nll_huber
            # Gaussian NLL with Huber tails beyond c sigma: still pushes sigma up for outliers
            # (d/dsigma of c|res|/sigma < 0), but grows linearly, so a prior that is far off
            # early in training cannot dominate the loss.
            nll = torch.log(sb) + torch.where(z <= c, 0.5 * z * z, c * z - 0.5 * c * c)
            vm = (lo.expert_valid > 0.5) & m
            l_nll = (nll * vm).sum() / vm.sum().clamp_min(1)
            total = total + cfg.w_sigma_nll * l_nll
            logs[f"{key}_nll"] = l_nll.detach()
            if key == "lape1":
                target = torch.exp(-(lo.mvs_u - u_gt).abs() / (lo.du + 1e-8)).detach()
                l_conf = _masked_mean(F.binary_cross_entropy_with_logits(
                    lo.mvs_conf_logit, target, reduction="none"), m)
                total = total + cfg.moa_stage_weights[0] * cfg.w_conf * l_conf
                logs["lape1_conf"] = l_conf.detach()
            if diagnostics:
                logs.update(lape_diagnostics(lo, g, m, u_gt, vmin, vmax, key, dg))

        for s in range(2, 5):
            mo = outputs.get(f"moa{s}")
            if mo is None:
                continue
            hw = tuple(mo.center_u.shape[-2:])
            g, m = _gt_at(gt, mask, hw)
            u_gt = depth_to_u(torch.where(m, g, torch.ones_like(g)), vmin, vmax)
            dg = self.du_global

            l_center = _masked_mean(
                F.smooth_l1_loss((mo.center_u - u_gt) / dg, torch.zeros_like(u_gt), reduction="none"), m)

            gxc = mo.center_u[..., :, 1:] - mo.center_u[..., :, :-1]
            gyc = mo.center_u[..., 1:, :] - mo.center_u[..., :-1, :]
            gxg = u_gt[..., :, 1:] - u_gt[..., :, :-1]
            gyg = u_gt[..., 1:, :] - u_gt[..., :-1, :]
            lg = torch.where(m, g, torch.ones_like(g)).log()
            mx = m[..., :, 1:] & m[..., :, :-1] & ((lg[..., :, 1:] - lg[..., :, :-1]).abs() < cfg.gt_edge_tau)
            my = m[..., 1:, :] & m[..., :-1, :] & ((lg[..., 1:, :] - lg[..., :-1, :]).abs() < cfg.gt_edge_tau)
            hx = F.smooth_l1_loss((gxc - gxg) / dg, torch.zeros_like(gxg), reduction="none")
            hy = F.smooth_l1_loss((gyc - gyg) / dg, torch.zeros_like(gyg), reduction="none")
            l_shape = ((hx * mx).sum() + (hy * my).sum()) / (mx.sum() + my.sum()).clamp_min(1)

            target = torch.exp(-(mo.mvs_u - u_gt).abs() / (mo.du + 1e-8)).detach()
            l_conf = _masked_mean(F.binary_cross_entropy_with_logits(
                mo.mvs_conf_logit, target, reduction="none"), m)

            lam = cfg.moa_stage_weights[s - 2]
            total = total + lam * (cfg.w_center * l_center + cfg.w_shape * l_shape + cfg.w_conf * l_conf)
            logs[f"moa{s}_center"] = l_center.detach()
            logs[f"moa{s}_shape"] = l_shape.detach()
            logs[f"moa{s}_conf"] = l_conf.detach()
            if diagnostics and not hasattr(mo, "sigma"):
                logs.update(moa_diagnostics(mo, g, m, vmin, vmax, s))
            elif diagnostics:
                with torch.no_grad():
                    p = f"moa{s}_"
                    e_c = (mo.center_depth - g).abs()
                    e_m = (u_to_depth(mo.mvs_u, vmin, vmax) - g).abs()
                    logs[p + "err_center_mm"] = _masked_mean(e_c, m)
                    logs[p + "err_mvs_mm"] = _masked_mean(e_m, m)
                    logs[p + "better_frac"] = _masked_mean((e_c < e_m).float(), m)
                    logs[p + "override_rate"] = (mo.alpha > 0.5).float().mean()
                    for j, name in enumerate(("mvs", "rac", "w3", "w7", "w11")):
                        logs[p + f"pi_{name}"] = mo.mixture_weights[:, j].mean()
                    if mo.lfr_flag is not None:
                        f = mo.lfr_flag & m
                        logs[p + "lfr_rate"] = mo.lfr_flag.float().mean()
                        logs[p + "lfr_good"] = ((e_c < e_m) & f).float().sum() / f.float().sum().clamp_min(1)

        logs["loss"] = total.detach()
        return total, logs


@torch.no_grad()
def lape_diagnostics(lo, g, m, u_gt, vmin, vmax, key: str, dg: float) -> dict[str, torch.Tensor]:
    """Per monocular expert: error, sigma calibration (mean z^2, share within 1 sigma); RAC state."""
    out = {}
    for j, name in enumerate(("rac", "w3", "w7", "w11")):
        vm = (lo.expert_valid[:, j:j + 1] > 0.5) & m
        e = (u_to_depth(lo.mu[:, j:j + 1], vmin, vmax) - g).abs()
        z = (lo.mu[:, j:j + 1] - u_gt) / lo.sigma[:, j:j + 1].clamp_min(1e-8)
        out[f"{key}_{name}_err_mm"] = _masked_mean(e, vm)
        out[f"{key}_{name}_valid"] = _masked_mean(vm.float(), m)
        out[f"{key}_{name}_sigma_bins"] = _masked_mean(lo.sigma[:, j:j + 1] / dg, vm)
        out[f"{key}_{name}_in1sigma"] = _masked_mean((z.abs() < 1.0).float(), vm)
    out[f"{key}_rac_models"] = lo.rac_ok.float().sum(1).mean()
    out[f"{key}_rac_supported"] = _masked_mean(lo.rac_supported.float(), m)
    out[f"{key}_global_ok"] = lo.global_ok.float().mean()
    return out


@torch.no_grad()
def fusion_diagnostics(st: dict, fu: dict, g: torch.Tensor, m: torch.Tensor, s: int) -> dict[str, torch.Tensor]:
    """Where the prior changed the winner: did it move toward the GT (flip_good) or away?"""
    out = {}
    hyp = st["depth_hypos"]
    a_raw = st["logits_raw"].argmax(1, keepdim=True)
    a_fus = st["logits"].argmax(1, keepdim=True)
    d_raw = hyp.gather(1, a_raw)
    d_fus = hyp.gather(1, a_fus)
    flip = (a_raw != a_fus) & m
    better = (d_fus - g).abs() < (d_raw - g).abs()
    n = m.float().sum().clamp_min(1)
    out[f"fuse_s{s}_flip_good"] = (flip & better).float().sum() / n
    out[f"fuse_s{s}_flip_bad"] = (flip & ~better).float().sum() / n
    for k in ("gamma", "M"):
        if k in fu:
            out[f"fuse_s{s}_{k}"] = fu[k].float().mean()
    if "vM" in fu:
        out[f"fuse_s{s}_vM"] = fu["vM"].float().mean()
    if "active" in fu:
        out[f"fuse_s{s}_active"] = fu["active"].float().mean()
    if "nce_gate" in fu:
        out[f"fuse_s{s}_nce_gate"] = fu["nce_gate"].float().mean()
    return out


@torch.no_grad()
def moa_diagnostics(mo, g: torch.Tensor, m: torch.Tensor, vmin, vmax, s: int) -> dict[str, torch.Tensor]:
    """Is MoA moving the centre closer to GT than the MVS centre it started from?"""
    p = f"moa{s}_"
    z_mvs = u_to_depth(mo.mvs_u, vmin, vmax)
    e_c = (mo.center_depth - g).abs()
    e_m = (z_mvs - g).abs()
    out = {
        p + "err_center_mm": _masked_mean(e_c, m),
        p + "err_mvs_mm": _masked_mean(e_m, m),
        p + "better_frac": _masked_mean((e_c < e_m).float(), m),
        p + "override_rate": (mo.alpha > 0.5).float().mean(),
        p + "conflict": mo.conflict.mean(),
        p + "r_mvs": mo.mvs_confidence.mean(),
        p + "mono_valid": mo.mono_valid.mean(),
        p + "global_ok": mo.global_ok.float().mean(),
        p + "edge_frac": mo.edge.mean(),
        # share of edge pixels the snap sent to the monocular surface (rest -> MVS)
        p + "snap_mono_frac": mo.edge_snap_mono.sum() / mo.edge.sum().clamp_min(1.0),
    }
    for j, name in enumerate(("mvs", "glob", "l3", "l5", "l7")):
        out[p + f"pi_{name}"] = mo.mixture_weights[:, j].mean()
    for j, name in enumerate(("3", "5", "7")):
        out[p + f"aff_valid_{name}"] = mo.affine_valid[:, j].mean()
        out[p + f"offset_only_{name}"] = mo.affine_offset_only[:, j].mean()
        out[p + f"barrier_{name}"] = mo.edge_barrier_rate[:, j].mean()
    return out
