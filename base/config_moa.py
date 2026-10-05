"""Configuration for the MoA network (models/network_moa.py, train_moa.py, test_moa.py).

A separate tree from ``base/config.MVSConfig`` on purpose: the new network has no
prior / SPRE / depth-range sections, so nothing in here can switch them back on.
Machine-independent building blocks (paths, FPN, SVA, DINO, augmentation) are
reused from ``base/config`` unchanged.
"""
from __future__ import annotations

import dataclasses
import os
import typing
from dataclasses import dataclass, field

from base.config import (
    AugmentConfig, DINOConfig, FPNConfig, ProjectPaths, SPREConfig, SVAConfig,
)


@dataclass(frozen=True)
class CascadeConfig:
    """Four stages at strides 8/4/2/1, all axes uniform in inverse depth."""

    num_depths: tuple[int, int, int, int] = (48, 16, 8, 4)
    # v2: 四级全 128 (v1 是 128/64/32/16)。加宽的代价几乎全在 warp 张量
    # [B,C,D,H,W] 上 —— 它每个 source 都要留给反向。本地实测每样本 26.4 ->
    # 42.2 GiB/Mpx (+60%), 每步 +43%; 按集群 v1 的 98%@batch4 反推, 640x896
    # 上 batch 2 约 63 GiB (79%), batch 3 放不下。
    warp_channels: tuple[int, int, int, int] = (128, 128, 128, 128)
    num_groups: int = 8
    warp_use_half: bool = True
    unet_base_channels: int = 16
    unet_depth: int = 3
    # DepthDecoder regression window (+-bins around the argmax) per stage.
    mode_windows: tuple[int, int, int, int] = (2, 2, 1, 1)
    # Stage 2/3/4 window half-width, in units of the parent stage's spacing.
    window_halfwidth_bins: tuple[float, float, float] = (3.0, 2.0, 1.0)
    # MoA.md §9: h' = max(h_base, |c - y| + h_keep) so the window still holds the MVS centre.
    keep_margin_bins: tuple[float, float, float] = (1.0, 1.0, 0.5)


@dataclass(frozen=True)
class MoAConfig:
    enabled: bool = True
    # widths
    emb_dim: int = 16
    shape_width: int = 32
    feat_dim: int = 16
    evidence_dim: int = 16
    evidence_hidden: int = 16
    conf_hidden: int = 32
    mix_hidden: int = 32
    # DA3 edge barrier. tau_edge: log-depth jump between adjacent full-res pixels;
    # tau_jump: endpoint log-depth difference at the working resolution.
    tau_edge: float = 0.03
    tau_jump: float = 0.10
    edge_dilate: bool = False
    # local WLS (3x3 / 5x5 / 7x7)
    spatial_sigma: tuple[float, float, float] = (1.0, 2.0, 3.0)
    tau_shape_init: float = 0.5
    # Ridge toward the identity. x lives in [0,1], so a 3x3 window on a typical slope
    # has Sw*Var(x) ~ 1e-4: anything near that would dominate the fit. Flat windows are
    # handled by the offset-only branch (tau_var), not by the ridge.
    lambda_a: float = 1e-6
    lambda_b: float = 1e-6
    tau_var: float = 1e-6
    min_neff: tuple[float, float, float] = (2.0, 4.0, 6.0)
    a_range: tuple[float, float] = (0.2, 5.0)
    b_max: float = 0.25
    # global affine (per sample). Per transition (->stage 2/3/4): "ransac" = weighted
    # RANSAC + Tukey IRLS with residuals in stage-1 bins (tau = inlier threshold);
    # "huber" = Huber/MAD IRLS (global_iters / global_huber_k), which DA3's far-tail
    # leverage anchors drag off the object (test19: stage1 mono error 14.7 -> 5.7 mm).
    global_solver: tuple[str, str, str] = ("ransac", "ransac", "ransac")
    global_ransac_tau: float = 0.5
    global_iters: int = 3
    global_huber_k: float = 1.345
    global_min_eff: float = 64.0
    # Per transition (->stage 2/3/4): largest centre move, in parent spacings.
    max_shift_bins: tuple[float, float, float] = (8.0, 4.0, 2.0)
    # Per transition: ceiling on the mixture mass the monocular experts may hold.
    # Stage 4 searches an already narrow window, where a local affine cannot be
    # exact on the true surface, so its residual shape error is noise rather
    # than a correction — MoA keeps the coarse stages and steps back at the end.
    moa_gain: tuple[float, float, float] = (1.0, 0.7, 0.3)
    # Per transition: at a DA3 depth edge, snap the centre to whichever of
    # {MVS centre, monocular surface} the mixture already leans to, instead of
    # blending them. A blend of a foreground and a background estimate lands on
    # neither surface, which is exactly the failure the barrier cannot prevent
    # (the two experts disagree *at* the same pixel, not across a neighbourhood).
    edge_snap: tuple[bool, bool, bool] = (True, True, True)
    # depth conflict c_d = sigmoid((gap/(sigma+du) - t_d)/T_d); tighter at finer stages
    conflict_t_d: tuple[float, float, float] = (3.0, 2.0, 1.5)
    conflict_T_d: tuple[float, float, float] = (0.5, 0.5, 0.5)
    conflict_tau_s: float = 1.0
    # mixture head starts leaning on MVS (softmax logit bias of expert 0)
    mvs_bias_init: float = 1.0


@dataclass(frozen=True)
class FeatureConfig:
    """Which frozen ViT feeds the SVA, and where the monocular depth comes from.

    ``da3``: Depth Anything 3 (DA3MONO-LARGE, DINOv2 ViT-L/14) replaces DINOv3 — the
    MonoMVSNet recipe of taking the monocular model's own encoder features. One DA3
    forward per step serves both consumers: the intermediate tokens of every view go
    through ``SVAFusion`` into the FPN exactly where the DINOv3 tokens used to, and
    the DPT head on the reference view gives the online monocular depth (no cache).
    ``dinov3``: the old MVSFormer++-style DINOv3 ViT-B/16 + the offline DA3 cache —
    only for checkpoints trained before this section existed (moa1 / moa2).
    """

    backbone: str = "da3"                       # da3 | dinov3 (legacy)
    # DA3 input: longest side -> process_res, then each side rounded to a multiple of 14
    # (DA3's own upper_bound_resize). 518 is where its shape is best (test18).
    process_res: int = 518
    sva_layers: tuple[int, ...] = (0, 1, 2, 3)  # indices into DA3's out_layers (4, 11, 17, 23)
    sky_threshold: float = 0.3                  # DA3 sky probability above this -> no mono depth


@dataclass(frozen=True)
class LAPEConfig:
    """Local-Affine Prior Evidence (docs: LAPE 终版方案). ``enabled=False`` = the MoA cascade.

    (1) RAC: sequential multi-model RANSAC + Tukey global alignment, each pixel takes
    the model voted by the anchors of its DA3-edge-bounded region; (2) four monocular
    experts (RAC region model + 3x3 / 7x7 / 11x11 local WLS), each with a calibrated
    sigma; (3) normal-consistent evidence: neighbours' matching cost carried along the
    monocular plane (GoMVS-style), stages 1-3; (4) low-frequency recall: re-centre the
    next window on the monocular depth where MVS has visibly collapsed; (5) the prior's
    bin mass enters the cost volume (gated adapter) and the logits (bounded gamma*log q).
    Stage 1 is regularised twice on the same cost volume.
    """

    enabled: bool = True
    stage1_two_pass: bool = True
    # (1) RAC
    rac_max_models: int = 3
    rac_a_ratio: tuple[float, float] = (0.5, 2.0)
    rac_min_frac: float = 0.1
    # (2) local experts: 3x3 dense, 7x7 dense, 11x11 on offsets {0, +-1, +-3, +-5}
    spatial_sigma: tuple[float, float, float] = (1.0, 2.0, 4.0)
    min_neff: tuple[float, float, float] = (3.0, 6.0, 8.0)   # Kish effective anchors per window
    sigma_floor_bins: float = 0.1               # c0, in stage-1 bins
    delta0: float = 4.0                         # anchor-distance scale, parent pixels
    # (3) normal-consistent evidence, per stage 1..4
    nce_stages: tuple[bool, bool, bool, bool] = (True, True, True, False)
    nce_dilation: tuple[int, int, int, int] = (1, 1, 2, 1)
    nce_radius: int = 2                         # 5x5 neighbourhood
    nce_kappa_init: float = 0.25
    # (4) low-frequency recall, per transition (->stage 2/3/4). Off at the last one: its
    # window is ~0.2 stage-1 bins wide, finer than DA3 is accurate after any alignment.
    lfr: bool = True
    lfr_levels: tuple[bool, bool, bool] = (True, True, False)
    lfr_tau_rgb: float = 0.5
    lfr_tau_mono: float = 0.25
    lfr_rho: float = 4.0
    lfr_gap_floor_bins: float = 2.0             # C3 also needs |y - mu_RAC| > this many stage-1 bins
    lfr_slope_min: float = 0.3                  # C4b: MVS ignores the mono shape (collapsed) ...
    lfr_mono_var_bins: float = 0.25             # ... where the mono depth varies >= this (parent bins)
    lfr_tau_region: float = 0.5
    lfr_delta_max: float = 16.0
    # C0: the matching is not trusted there (7x7 mean of the MoA confidence head). "y does not
    # follow x" alone cannot tell a collapsed MVS from a wrongly shaped DA3; low confidence can.
    lfr_conf_max: float = 0.5
    # (5) candidate evidence + bounded fusion
    mass_min: float = 0.2
    eps_floor: float = 0.05
    log_prior_clip: float = 4.0
    gamma_max: float = 2.0
    gamma_bias_init: float = -4.0
    reliable_mass: float = 0.9                  # v_M: posterior mass within +-1 bin of the argmax
    reliable_src_std: float = 1.0               # v_M: normalised source disagreement at the argmax
    reliable_nvalid: float = 0.5                # v_M: fraction of sources that see the argmax voxel


@dataclass(frozen=True)
class MoALossConfig:
    stage_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.5, 2.0)
    moa_stage_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    w_center: float = 1.0
    w_shape: float = 0.5
    w_conf: float = 0.5
    # GT depth discontinuity (log-depth jump between neighbours) excluded from L_shape
    gt_edge_tau: float = 0.03
    # LAPE: deep supervision of stage-1 pass A, and the Gaussian NLL that calibrates sigma
    w_ce_stage1a: float = 0.5
    w_sigma_nll: float = 0.1
    sigma_nll_cap_bins: float = 20.0            # residuals beyond this (stage-1 bins) are capped
    sigma_nll_huber: float = 3.0                # NLL tails become linear beyond this many sigma


@dataclass(frozen=True)
class MoATrainConfig:
    profile: str = "umhpc"
    epochs: int = 10
    max_steps: int = 0                 # 0 = epochs x steps_per_epoch
    lr_schedule_steps: int = 0         # 0 = 跟随实际停止步数
    batch_size: int = 2
    val_batch_size: int = 4
    num_views: int = 5
    num_workers: int = 12
    lr: float = 3e-4
    weight_decay: float = 1.0e-4
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    amp: bool = True
    amp_dtype: str = "bf16"
    seed: int = 20260526
    log_interval: int = 20
    val_interval: int = 2000           # 0 = only at epoch ends
    ckpt_interval: int = 1000
    multi_scale: bool = True
    height: int = 512                  # used when multi_scale is off
    width: int = 640
    da3_missing: str = "error"         # error / skip (legacy offline DA3 cache only)
    nan_grad_patience: int = 10
    # LAPE training-time prior augmentation (per sample, train mode only):
    # drop the whole monocular prior, or perturb one 64x64 block of the DA3 depth
    # (scale +-10%, shift +-2% of the block median) before RAC sees it.
    prior_dropout: float = 0.3
    mono_perturb: float = 0.3


def _train_profile(profile: str) -> MoATrainConfig:
    if profile == "umhpc":
        return MoATrainConfig(profile="umhpc")
    if profile == "local":
        return MoATrainConfig(profile="local", epochs=2, batch_size=1, val_batch_size=1,
                              num_views=3, num_workers=4, lr=1e-4, warmup_steps=500,
                              val_interval=1000, ckpt_interval=500, multi_scale=False,
                              da3_missing="skip")
    raise ValueError(f"unknown profile {profile!r}")


@dataclass(frozen=True)
class MoAMVSConfig:
    paths: ProjectPaths = field(default_factory=ProjectPaths)
    fpn: FPNConfig = field(default_factory=FPNConfig)
    sva: SVAConfig = field(default_factory=lambda: SVAConfig(full=True))
    dino: DINOConfig = field(default_factory=DINOConfig)
    # DinoSVA's constructor takes a SPREConfig; only its SVAFusion widths are read.
    dino_fusion: SPREConfig = field(default_factory=SPREConfig)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    moa: MoAConfig = field(default_factory=MoAConfig)
    feat: FeatureConfig = field(default_factory=FeatureConfig)
    lape: LAPEConfig = field(default_factory=LAPEConfig)
    loss: MoALossConfig = field(default_factory=MoALossConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: MoATrainConfig = field(default_factory=lambda: _train_profile(
        os.environ.get("UPRMVS_PROFILE", "umhpc")))


def build_moa_config(profile: str | None = None) -> MoAMVSConfig:
    prof = profile or os.environ.get("UPRMVS_PROFILE", "umhpc")
    return MoAMVSConfig(train=_train_profile(prof))


# Sections that define the network. Saved in every checkpoint and restored by
# test_moa.py, so inference rebuilds exactly what was trained.
ARCH_SECTIONS = ("fpn", "sva", "dino", "dino_fusion", "cascade", "moa", "feat", "lape")

# What a section means when a checkpoint predates it. A whole missing section is
# not "today's default": moa1 / moa2 were trained with DINOv3 features, the offline
# DA3 cache and no LAPE, so that is what they must be rebuilt with.
LEGACY_SECTIONS = {
    "feat": FeatureConfig(backbone="dinov3"),
    "lape": LAPEConfig(enabled=False),
}


def _to_plain(v):
    if dataclasses.is_dataclass(v):
        return {f.name: _to_plain(getattr(v, f.name)) for f in dataclasses.fields(v)}
    if isinstance(v, (tuple, list)):
        return [_to_plain(x) for x in v]
    if isinstance(v, os.PathLike):
        return str(v)
    return v


def arch_snapshot(cfg: MoAMVSConfig) -> dict:
    return {name: _to_plain(getattr(cfg, name)) for name in ARCH_SECTIONS}


def config_snapshot(cfg: MoAMVSConfig) -> dict:
    return _to_plain(cfg)


def _to_tuple(v):
    return tuple(_to_tuple(x) for x in v) if isinstance(v, list) else v


def _from_dict(cls, data: dict):
    hints = typing.get_type_hints(cls)
    kw = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        if dataclasses.is_dataclass(hints.get(f.name)) and isinstance(v, dict):
            v = _from_dict(hints[f.name], v)
        kw[f.name] = _to_tuple(v)
    unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        print(f"[config] {cls.__name__}: ignoring unknown keys {sorted(unknown)}")
    return cls(**kw)


def apply_arch_snapshot(cfg: MoAMVSConfig, snapshot: dict) -> MoAMVSConfig:
    """Replace the architecture sections of ``cfg`` with a checkpoint's snapshot.

    Fields the snapshot does not carry take today's defaults. That is silent by
    construction — a config-only field (no weights attached) changes behaviour
    without changing the state dict — so they are reported: a checkpoint from
    before such a field existed would otherwise be evaluated under semantics it
    was never trained with.
    """
    hints = typing.get_type_hints(MoAMVSConfig)
    upd = {name: _from_dict(hints[name], snapshot[name]) for name in ARCH_SECTIONS if name in snapshot}
    legacy = {name: LEGACY_SECTIONS[name] for name in LEGACY_SECTIONS if name not in snapshot}
    if legacy:
        print(f"[config] 快照里没有 {sorted(legacy)} 段 (旧 checkpoint): 按旧语义重建 "
              f"(feat.backbone=dinov3 + 离线 DA3 缓存, lape.enabled=False)")
    out = dataclasses.replace(cfg, **upd, **legacy)
    missing = [f"{sec}.{f.name}={getattr(getattr(out, sec), f.name)!r}"
               for sec in ARCH_SECTIONS if sec in snapshot
               for f in dataclasses.fields(getattr(out, sec)) if f.name not in snapshot[sec]]
    if missing:
        print("[config] WARNING 这份 checkpoint 的快照里没有以下字段, 将使用当前代码的默认值 "
              "(训练时的行为可能与此不同):\n  " + "\n  ".join(missing))
    return out
