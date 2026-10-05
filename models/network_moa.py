"""MoAMVSNet: frozen-ViT + FPN + SVA features, 4-stage cascade, monocular prior.

    images -> DA3SVA (or legacy DinoSVA) -> MultiViewFPN(out0) -> SVAPathway -> feats {8,4,2,1}
    cfg.feat.backbone = "da3" (default): Depth Anything 3's encoder tokens feed the SVA in
      place of DINOv3 (MonoMVSNet's recipe), and its DPT head gives the reference view's
      monocular depth online — no DA3 cache. "dinov3": moa1 / moa2 checkpoints.

    LAPE (cfg.lape.enabled, default):
      stage 1, pass A: full-range axis, cost volume + 3D UNet (pure matching)
      LAPE level 0 (stage-1 resolution): RAC + 4 monocular experts -> prior
      stage 1, pass B: same cost volume + normal-consistent evidence + prior adapter,
                       same 3D UNet, logits + gamma * log q
      for s in 2..4:
          LAPE level s-1: RAC + experts + MoA mixture + low-frequency recall -> centre
          window around the centre (not forced to hold the MVS centre where LFR fired)
          cost volume (+ normal evidence for s <= 3) + prior adapter -> 3D UNet -> + gamma*log q
    MoA (cfg.lape.enabled = False): the previous cascade — the prior only sets centres.

Every stage's depth still comes from its own posterior; the prior shifts which candidate
wins only where the matching is not verified reliable (gamma = 0 there, see
models/moa/prior_evidence.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.config_moa import MoAMVSConfig
from models.decoder import DepthDecoder
from models.fpn import MultiViewFPN
from models.moa.cost_volume import MoACostVolume
from models.moa.edge import NeighborhoodBarrier, downsample_edge, log_depth_edge
from models.moa.evidence import normalize_src_std
from models.moa.geometry import (
    axis_spacing, depth_to_u, inverse_bounds, stage1_u_hypotheses, u_to_depth,
    upsample_nearest, window_u_hypotheses,
)
from models.sva import SVAPathway

FROZEN_PREFIXES = ("dino_sva.dino.", "da3_sva.da3.")


class MoAMVSNet(nn.Module):
    strides: tuple[int, int, int, int] = (8, 4, 2, 1)

    def __init__(self, cfg: MoAMVSConfig | None = None, da3_net: nn.Module | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MoAMVSConfig()
        cc = self.cfg.cascade
        if len(cc.num_depths) != 4 or len(cc.warp_channels) != 4:
            raise ValueError("cascade.num_depths / warp_channels need 4 entries")
        fpn_c = self.cfg.fpn.out_channels
        self.full_sva = bool(self.cfg.sva.full)
        self.backbone_kind = str(self.cfg.feat.backbone)

        self.dino_sva = self.da3_sva = None
        if self.backbone_kind == "da3":
            from models.da3_feature import DA3SVA
            self.da3_sva = DA3SVA(self.cfg.feat, self.cfg.dino_fusion, self.cfg.paths.da3_weights_file,
                                  fpn_c, self.cfg.sva, net=da3_net)
        elif self.backbone_kind == "dinov3":
            from models.spre import DinoSVA
            self.dino_sva = DinoSVA(self.cfg.dino_fusion, self.cfg.dino, self.cfg.paths.dinov3_weights_file,
                                    fpn_channels=fpn_c, sva_cfg=self.cfg.sva)
        else:
            raise ValueError(f"feat.backbone must be da3 / dinov3, got {self.backbone_kind!r}")
        self.fpn = MultiViewFPN(out_channels=fpn_c, base_channel=self.cfg.fpn.base_channel,
                                stage1_head="out0" if self.full_sva else "smooth")
        self.sva_pathway = None
        if self.full_sva:
            sva = self.cfg.sva
            self.sva_pathway = SVAPathway(
                fpn_c, heads=int(sva.hr_heads),
                layer_names=tuple(x.strip() for x in str(sva.hr_layers).split(",")),
                mlp_ratio=float(sva.hr_mlp_ratio), pe_max_shape=tuple(sva.pe_max_shape),
                strides=self.strides)

        self.cost_volumes = nn.ModuleList(
            MoACostVolume(fpn_c, wc, cc.num_groups, cc.warp_use_half) for wc in cc.warp_channels)
        self.decoders = nn.ModuleList(
            DepthDecoder(in_channels=cc.num_groups + 1, base=cc.unet_base_channels,
                         depth=cc.unet_depth, mode_window=mw, head_mode="expect")
            for mw in cc.mode_windows)
        for i, d in enumerate(self.decoders):
            d.tag = f"stage{i + 1}"

        self.lape_on = bool(self.cfg.moa.enabled and self.cfg.lape.enabled)
        self.moa = None
        if self.lape_on:
            from models.moa.lape import LAPECascade
            from models.moa.normal_evidence import NormalEvidence
            from models.moa.prior_evidence import GammaHead, PriorAdapter
            lc = self.cfg.lape
            self.moa = LAPECascade(self.cfg.moa, lc, cc, fpn_c, cc.num_groups, cc.num_depths[0])
            nce, bars = [], []
            for s in range(4):
                if lc.nce_stages[s]:
                    m = NormalEvidence(cc.num_groups, radius=lc.nce_radius, dilation=lc.nce_dilation[s],
                                       kappa_init=lc.nce_kappa_init)
                    nce.append(m)
                    bars.append(NeighborhoodBarrier(radius=m.reach, subset=m.offsets))
                else:
                    nce.append(nn.Identity())
                    bars.append(nn.Identity())
            self.nce = nn.ModuleList(nce)
            self.nce_barriers = nn.ModuleList(bars)
            self.prior_adapters = nn.ModuleList(PriorAdapter(cc.num_groups) for _ in range(4))
            self.gamma_heads = nn.ModuleList(
                GammaHead(gamma_max=lc.gamma_max, bias_init=lc.gamma_bias_init) for _ in range(4))
        elif self.cfg.moa.enabled:
            from models.moa.moa import MoACascade
            self.moa = MoACascade(self.cfg.moa, fpn_c, cc.num_groups, cc.num_depths[0])

    # ------------------------------------------------------------------ helpers
    @property
    def uses_mono(self) -> bool:
        return self.moa is not None

    @property
    def needs_mono_cache(self) -> bool:
        """True only for legacy (DINOv3) checkpoints: the mono depth then comes from log/da3_cache."""
        return self.moa is not None and self.da3_sva is None

    @property
    def da3_process_res(self):
        return int(self.cfg.feat.process_res) if self.da3_sva is not None else None

    def _features(self, images: torch.Tensor):
        coarse_hw = (images.shape[-2] // self.strides[0], images.shape[-1] // self.strides[0])
        if self.da3_sva is not None:
            fused, grid, d = self.da3_sva(images)
            fpn_in = self.da3_sva.fpn_feature(fused, grid, coarse_hw)
        else:
            fused, _, grid = self.dino_sva(images)
            fpn_in = self.dino_sva.fpn_feature(fused, grid, coarse_hw)
            d = None
        feats = self.fpn(images, dino=fpn_in)
        if self.sva_pathway is not None:
            feats = self.sva_pathway(feats)
        return feats, d

    def _cost(self, k, feat, K, E, u_hyp, vmin, vmax):
        hypos = u_to_depth(u_hyp, vmin, vmax)
        cvo = self.cost_volumes[k](feat[:, 0], feat[:, 1:], K[:, 0], K[:, 1:], E[:, 0], E[:, 1:],
                                   hypos, feature_stride=self.strides[k])
        return cvo, hypos

    def _decode(self, k, x_in, hypos, u_hyp, cvo, branch_prior=None) -> dict:
        depth, sigma, prob, logits, mode_idx, logits_raw = self.decoders[k](x_in, hypos, branch_prior)
        return {"depth": depth.unsqueeze(1), "sigma": sigma.unsqueeze(1), "prob": prob,
                "logits": logits, "logits_raw": logits_raw, "mode_idx": mode_idx,
                "depth_hypos": hypos, "u_hypos": u_hyp, "_cv": cvo}

    def _run_stage(self, k, feat, K, E, u_hyp, vmin, vmax) -> dict:
        cvo, hypos = self._cost(k, feat, K, E, u_hyp, vmin, vmax)
        return self._decode(k, cvo.decoder_input(), hypos, u_hyp, cvo)

    def _augment_prior(self, md: torch.Tensor, mv: torch.Tensor):
        """Train-time: drop the whole prior of a sample, or perturb one 64x64 DA3 block."""
        t = self.cfg.train
        B, _, H, W = md.shape
        dev = md.device
        keep = (torch.rand(B, device=dev) >= float(t.prior_dropout)).view(B, 1, 1, 1)
        mv = mv & keep
        pert = torch.rand(B, device=dev) < float(t.mono_perturb)
        if bool(pert.any()) and H >= 64 and W >= 64:
            md = md.clone()
            for i in torch.nonzero(pert).flatten().tolist():
                y0 = int(torch.randint(0, H - 63, (1,)))
                x0 = int(torch.randint(0, W - 63, (1,)))
                blk = md[i, :, y0:y0 + 64, x0:x0 + 64]
                s = 1.0 + (torch.rand((), device=dev) * 2 - 1) * 0.1
                sh = (torch.rand((), device=dev) * 2 - 1) * 0.02 * blk.median()
                md[i, :, y0:y0 + 64, x0:x0 + 64] = (blk * s + sh).clamp_min(1e-3)
        return md, mv

    def _mono(self, batch: dict, d):
        if self.moa is None:
            return None
        if d is not None:
            md, mv = d.depth, d.valid
        else:
            if "mono_depth" not in batch:
                raise KeyError("legacy (dinov3) MoA needs batch['mono_depth'] from the DA3 cache")
            md = batch["mono_depth"].float().unsqueeze(1)
            mv = torch.isfinite(md) & (md > 0)
            md = torch.where(mv, md, torch.ones_like(md))
        if self.training and self.lape_on:
            md, mv = self._augment_prior(md, mv)
        return {"depth": md, "valid": mv.float(), "edge": log_depth_edge(md, mv)}

    # ------------------------------------------------------------------ LAPE stage inputs
    def _reliability(self, stage: dict, cvo) -> torch.Tensor:
        """v_M of a decoded stage from its raw (pre-prior) posterior, [B,1,h,w] bool."""
        from models.moa.prior_evidence import posterior_features, reliable_mvs
        src_n = normalize_src_std(cvo.src_std, cvo.cv)
        nvf = cvo.n_valid / float(max(cvo.num_src, 1))
        pf = posterior_features(stage["logits_raw"], stage["u_hypos"], src_n, nvf)
        return reliable_mvs(pf, self.cfg.lape)

    def _lape_inputs(self, k, cvo, u_hyp, pri, K, mono, gray, vmin, vmax, vM_parent):
        """Decoder input with (3) normal evidence and (5) the prior adapter, plus the logit
        fusion callback. ``pri``: the prior lifted to this stage's grid."""
        from models.moa.low_freq_recall import texture_ratio
        from models.moa.normal_evidence import camera_normals
        from models.moa.prior_evidence import (
            bin_mass, evidence_channels, log_prior, mixture_moments, posterior_features, reliable_mvs,
        )
        lc, mcfg = self.cfg.lape, self.cfg.moa
        G = self.cfg.cascade.num_groups
        cv = cvo.cv.float()
        hw = tuple(cv.shape[-2:])
        du_k = axis_spacing(u_hyp)
        store: dict = {}

        # (3) normal-consistent evidence
        delta_n = torch.zeros_like(cv)
        if lc.nce_stages[k]:
            stride = self.strides[k]
            Ks = K[:, 0].float().clone()
            Ks[:, :2, :] = Ks[:, :2, :] / float(stride)
            edge_k = downsample_edge(mono["edge"], hw)
            n, ok_n, cosv = camera_normals(pri["z_rac"], pri["z_valid"], edge_k, Ks, mcfg.tau_edge)
            edge_dil = F.max_pool2d((edge_k >= mcfg.tau_edge).float(), 3, stride=1, padding=1)
            c_n = ok_n.float() * (1.0 - edge_dil) * (cosv / 0.25).clamp(0.0, 1.0)
            lz = torch.where(pri["z_valid"], pri["z_rac"], torch.ones_like(pri["z_rac"])).log()
            bar = self.nce_barriers[k](edge_k, lz, pri["z_valid"].float(), mcfg.tau_edge, mcfg.tau_jump)
            # the layer checkpoints each neighbour's term itself in training
            delta_n, gate_n = self.nce[k](cv, u_hyp, n, c_n, Ks, bar["pass"], vmin, vmax)
            store["nce_gate"] = gate_n.detach()

        # (5) prior mass on this axis
        q, M = bin_mass(u_hyp, pri["mu"], pri["sigma"], pri["w"])
        ell, active = log_prior(q, M, lc.mass_min, lc.eps_floor, lc.log_prior_clip)
        mbar, sbar = mixture_moments(pri["mu"], pri["sigma"], pri["w"])
        E = evidence_channels(u_hyp, ell, mbar, sbar, M, active, du_k, lc.log_prior_clip)
        g_p = (active & ~vM_parent).float()
        dC = self.prior_adapters[k](E) * g_p.unsqueeze(1)
        frac = (cvo.n_valid / max(cvo.num_src, 1)).unsqueeze(1).to(cv.dtype)
        x_in = torch.cat([cv + delta_n + dC, frac], dim=1)

        src_n = normalize_src_std(cvo.src_std, cvo.cv)
        nvf = cvo.n_valid / float(max(cvo.num_src, 1))
        tex = texture_ratio(gray, hw).clamp_min(1e-3).log().clamp(-4.0, 4.0) / 4.0
        lsig = torch.log(sbar / du_k.clamp_min(1e-8)).clamp(-4.0, 4.0) / 4.0
        dlt = torch.log1p(pri["delta"] / lc.delta0) / 2.0
        head = self.gamma_heads[k]

        def fuse(logits_raw: torch.Tensor) -> torch.Tensor:
            pf = posterior_features(logits_raw, u_hyp, src_n, nvf)
            vM = reliable_mvs(pf, lc)
            md = ((pf["mu_u"] - mbar) / (pf["sd_u"] ** 2 + sbar ** 2).sqrt().clamp_min(1e-8)).clamp(-8.0, 8.0) / 8.0
            gin = torch.cat([pf["pmax"], pf["entropy"], pf["gap"], pf["mass1"], pf["src"].clamp(0, 10) / 10.0,
                             pf["nvalid"], lsig, M.clamp(0, 1), dlt, pri["split"], pri["lfr_soft"],
                             pri["lfr_flag"], md, tex], dim=1).detach()
            gamma = head(gin) * (active & ~vM).float()
            store.update(gamma=gamma.detach(), vM=vM, M=M.detach(), active=active, g_p=g_p)
            return logits_raw + gamma * ell

        return x_in, fuse, store

    # ------------------------------------------------------------------ forward
    def forward(self, batch: dict) -> dict:
        images = batch["images"].float() / 255.0
        K = batch["intrinsics"].float()
        E = batch["extrinsics"].float()
        B = images.shape[0]
        vmin, vmax = inverse_bounds(batch["depth_values"])
        feats, d = self._features(images)
        mono = self._mono(batch, d)
        gray = (0.299 * images[:, 0, 0:1] + 0.587 * images[:, 0, 1:2] + 0.114 * images[:, 0, 2:3])
        cc = self.cfg.cascade
        out: dict = {"vmin": vmin, "vmax": vmax}
        if d is not None:
            out["mono_depth"] = d.depth
            out["mono_valid"] = d.valid

        f1 = feats[self.strides[0]]
        u1 = stage1_u_hypotheses(B, cc.num_depths[0], tuple(f1.shape[-2:]), f1.device)
        cvo1, hyp1 = self._cost(0, f1, K, E, u1, vmin, vmax)
        s1a = self._decode(0, cvo1.decoder_input(), hyp1, u1, cvo1)
        if self.lape_on and self.cfg.lape.stage1_two_pass:
            s1a.pop("_cv")
            vM1a = self._reliability(s1a, cvo1)
            lp0 = self.moa(0, mono_depth=mono["depth"], mono_valid=mono["valid"], mono_edge=mono["edge"],
                           gray=gray, z_prev=s1a["depth"].detach(), prob=s1a["prob"].detach(), u_hyp=u1,
                           cv=cvo1.cv.detach(), n_valid=cvo1.n_valid.detach(), src_std=cvo1.src_std.detach(),
                           num_src=cvo1.num_src, ref_feat=f1[:, 0].detach(), vmin=vmin, vmax=vmax)
            pri = self.moa.lift(lp0, tuple(f1.shape[-2:]), mono["depth"], mono["valid"], vmin, vmax)
            x_in, fuse, store = self._lape_inputs(0, cvo1, u1, pri, K, mono, gray, vmin, vmax, vM1a)
            prev = self._decode(0, x_in, hyp1, u1, cvo1, branch_prior=fuse)
            prev["vM"] = store.get("vM", vM1a)
            out.update(stage1a=s1a, lape1=lp0, fusion1=store)
        else:
            prev = s1a
            if self.lape_on:
                prev["vM"] = self._reliability(prev, cvo1)
        out["stage1"] = prev

        for k in (1, 2, 3):
            feat = feats[self.strides[k]]
            du = axis_spacing(prev["u_hypos"])
            y = depth_to_u(prev["depth"], vmin, vmax).detach()
            cvo = prev.pop("_cv")
            mo = None
            lfr = None
            if self.lape_on:
                mo = self.moa(k, mono_depth=mono["depth"], mono_valid=mono["valid"], mono_edge=mono["edge"],
                              gray=gray, z_prev=prev["depth"].detach(), prob=prev["prob"].detach(),
                              u_hyp=prev["u_hypos"].detach(), cv=cvo.cv.detach(), n_valid=cvo.n_valid.detach(),
                              src_std=cvo.src_std.detach(), num_src=cvo.num_src,
                              ref_feat=feats[self.strides[k - 1]][:, 0].detach(), vmin=vmin, vmax=vmax)
                center = mo.center_u.detach()
                lfr = mo.lfr_flag
            elif self.moa is not None:
                mo = self.moa(
                    k - 1, mono_depth=mono["depth"], mono_valid=mono["valid"], mono_edge=mono["edge"],
                    z_prev=prev["depth"].detach(), prob=prev["prob"].detach(),
                    u_hyp=prev["u_hypos"].detach(), cv=cvo.cv.detach(), n_valid=cvo.n_valid.detach(),
                    src_std=cvo.src_std.detach(), num_src=cvo.num_src,
                    ref_feat=feats[self.strides[k - 1]][:, 0].detach(), vmin=vmin, vmax=vmax)
                center = mo.center_u.detach()
            else:
                center = y
            if mo is not None:
                out[f"moa{k + 1}"] = mo
            h_base = float(cc.window_halfwidth_bins[k - 1]) * du
            h_keep = float(cc.keep_margin_bins[k - 1]) * du
            half = torch.maximum(h_base, (center - y).abs() + h_keep)
            if lfr is not None:
                # low-frequency recall: centre on the region model, do not stretch the window
                # back to the collapsed MVS centre; the width follows the prior's own sigma
                half = torch.where(lfr, torch.maximum(h_base, 2.0 * mo.sigma[:, :1].detach()), half)
            hw = tuple(feat.shape[-2:])
            u_k = window_u_hypotheses(upsample_nearest(center, hw), upsample_nearest(half, hw),
                                      cc.num_depths[k]).detach()
            vM_prev = prev.get("vM")
            del cvo
            cvo_k, hyp_k = self._cost(k, feat, K, E, u_k, vmin, vmax)
            if self.lape_on:
                pri = self.moa.lift(mo, hw, mono["depth"], mono["valid"], vmin, vmax)
                vM_up = upsample_nearest(vM_prev.float(), hw) > 0.5
                x_in, fuse, store = self._lape_inputs(k, cvo_k, u_k, pri, K, mono, gray, vmin, vmax, vM_up)
                cur = self._decode(k, x_in, hyp_k, u_k, cvo_k, branch_prior=fuse)
                cur["vM"] = store["vM"]
                out[f"fusion{k + 1}"] = store
            else:
                cur = self._decode(k, cvo_k.decoder_input(), hyp_k, u_k, cvo_k)
            out[f"stage{k + 1}"] = cur
            prev = cur
        prev.pop("_cv")
        out["depth_full"] = out["stage4"]["depth"][:, 0]
        return out
