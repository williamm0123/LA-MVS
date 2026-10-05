"""Semantic Prior Reliability Estimator (SPRE).

A learned, supervised, DINOv3-witnessed successor to the hand-engineered
offline ``conf`` (``models/conf.py``). SPRE predicts a per-pixel reliability
``r in [0, 1]`` for the (possibly corrupted) depth prior from:

  * frozen DINOv3 semantic features of every view, fused by ``CrossViTFusion``
    (ported from MVSFormer++) — an *independent* witness that does not depend
    on the prior's own depth values, and
  * cheap online statistics of the depth prior (local MAD, gradient, validity)
    computed AFTER prior-corruption, so injected failures are visible.

``r`` replaces the cached ``conf_prior`` fed to ``build_stage1_hypotheses``:
low ``r`` widens the local search (leaning on the prior-independent guard),
high ``r`` keeps it tight. The head is supervised by the corruption mask and
the prior's true error (see ``losses/composite.py``). DINOv3 is frozen; only
the fusion + projection + head train.

The module emits raw LOGITS (not the sigmoid) so the loss can use the
autocast-safe ``binary_cross_entropy_with_logits``; the network applies the
sigmoid to obtain ``r`` for the hypothesis builder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.config import DINOConfig, SPREConfig, SVAConfig
from models.dinov3.extractor import compute_patch_aligned_size, load_dinov3_vit_base
from utils.geometry import make_pixel_grid


def _masked_standardize(d: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per-sample zero-mean/unit-std over valid pixels; invalid pixels -> 0."""
    v = valid.float()
    n = v.flatten(1).sum(1).clamp_min(1.0)
    m = (d * v).flatten(1).sum(1) / n
    var = (((d - m[:, None, None]) ** 2) * v).flatten(1).sum(1) / n
    s = var.sqrt().clamp_min(1e-3)
    return ((d - m[:, None, None]) / s[:, None, None]) * v


def _local_mad_rel(d: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """3x3 median-absolute-deviation, made scale-free by dividing by depth."""
    B, H, W = d.shape
    nan = float("nan")
    dd = torch.where(valid, d, torch.full_like(d, nan))
    patches = F.unfold(dd.unsqueeze(1), kernel_size=3, padding=1).view(B, 9, H, W)
    med = patches.nanmedian(dim=1).values
    mad = (patches - med.unsqueeze(1)).abs().nanmedian(dim=1).values
    mad = torch.nan_to_num(mad, nan=0.0)
    return mad / (d.abs() + 1.0)


def _grad_mag_rel(d: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Scale-free forward-difference gradient magnitude."""
    dd = torch.nan_to_num(torch.where(valid, d, torch.zeros_like(d)), nan=0.0)
    gx = F.pad(dd[:, :, 1:] - dd[:, :, :-1], (0, 1, 0, 0))
    gy = F.pad(dd[:, 1:, :] - dd[:, :-1, :], (0, 0, 0, 1))
    return (gx.abs() + gy.abs()) / (d.abs() + 1.0)


def prior_view_consistency(
    feats: torch.Tensor,
    depth_prior: torch.Tensor,
    K: torch.Tensor,
    E: torch.Tensor,
    image_hw: tuple[int, int],
    grid: tuple[int, int],
) -> torch.Tensor:
    """Does the prior's geometry survive reprojection into the source views?

    Warps every source view's fused DINO token map into the reference frame
    *using the prior depth*, and measures how well it lines up with the
    reference tokens. Asking "if the prior were right, would the views agree?"
    is exactly the question SPRE has to answer, and it is not circular: the
    verdict comes from source-view content the prior never saw.

    This replaces the ref-queries-source attention branch that the switch to
    MVSFormer++'s SVA direction removed. Measuring the geometry directly is both
    cheaper (one grid_sample on the 26x32 token grid, no extra blocks) and
    harder evidence than asking attention to discover occlusion implicitly.

    ``feats`` should be the **frozen** DINOv3 tokens, not the SVA-fused ones.
    The result is detached either way, so fused features buy no gradient — they
    would only make the statistic drift as the fusion trains, turning a fixed
    measurement into a moving one. Frozen tokens keep it stationary and genuinely
    independent of everything SPRE's loss can touch.

    Returns ``[B, 3, gh, gw]``: mean agreement, worst-source agreement, and the
    fraction of sources where the point lands in front of the camera and inside
    the frame — the last one is a direct occlusion / field-of-view cue.

    Features are detached: this is a measurement, so SPRE's loss must not reshape
    the matching features through it.
    """
    B, V, _, C = feats.shape
    gh, gw = grid
    H, W = image_hw
    if V < 2:                                   # no sources: nothing to check
        return feats.new_zeros(B, 3, gh, gw)

    with torch.autocast(device_type=feats.device.type, enabled=False):
        f = F.normalize(feats.detach().float().reshape(B, V, gh, gw, C).permute(0, 1, 4, 2, 3), dim=2)
        ref_f, src_f = f[:, 0], f[:, 1:]                       # [B,C,gh,gw], [B,V-1,C,gh,gw]

        # Intrinsics from image pixels to the token grid (each axis separately —
        # the patch-aligned resize does not preserve the aspect ratio exactly).
        Kt = K.float().clone()
        Kt[:, :, 0, :] *= gw / W
        Kt[:, :, 1, :] *= gh / H

        d = F.interpolate(depth_prior.unsqueeze(1).float(), size=(gh, gw),
                          mode="nearest").reshape(B, 1, gh * gw)
        valid_d = (d > 0) & torch.isfinite(d)

        # Same convention as utils.geometry.homography_warp_features.
        Eg = E.float()
        R, t = Eg[:, :, :3, :3], Eg[:, :, :3, 3:4]
        R_rel = R[:, :1] @ R[:, 1:].transpose(2, 3)             # [B,V-1,3,3]
        t_rel = t[:, :1] - R_rel @ t[:, 1:]

        px = make_pixel_grid(gh, gw, feats.device, torch.float32).view(3, -1).unsqueeze(0).expand(B, -1, -1)
        rays = torch.bmm(torch.inverse(Kt[:, 0]), px) * d       # [B,3,gh*gw]

        agree, visible = [], []
        for v in range(V - 1):
            cam = torch.bmm(R_rel[:, v].transpose(1, 2), rays - t_rel[:, v])
            pix = torch.bmm(Kt[:, v + 1], cam)
            z = pix[:, 2:3]
            uv = pix[:, :2] / z.clamp(min=1e-6)
            inb = ((uv[:, 0] >= 0) & (uv[:, 0] <= gw - 1)
                   & (uv[:, 1] >= 0) & (uv[:, 1] <= gh - 1) & (z[:, 0] > 0) & valid_d[:, 0])
            gx = (uv[:, 0] / (gw - 1) * 2 - 1)
            gy = (uv[:, 1] / (gh - 1) * 2 - 1)
            g = torch.nan_to_num(torch.stack([gx, gy], -1), nan=2.0, posinf=2.0, neginf=-2.0)
            warped = F.grid_sample(src_f[:, v], g.clamp(-2, 2).view(B, gh, gw, 2),
                                   mode="bilinear", padding_mode="zeros", align_corners=True)
            cos = (ref_f * warped).sum(1)                        # [B,gh,gw] cosine, features are L2-normalised
            m = inb.view(B, gh, gw)
            agree.append(torch.where(m, cos, torch.zeros_like(cos)))
            visible.append(m.float())

        vis = torch.stack(visible, 1)                            # [B,V-1,gh,gw]
        agr = torch.stack(agree, 1)
        n = vis.sum(1).clamp_min(1.0)
        mean_a = agr.sum(1) / n
        # invisible sources must not win the min; push them to +1 first
        worst_a = torch.where(vis.bool(), agr, torch.ones_like(agr)).amin(1)
        return torch.stack([mean_a, worst_a, vis.mean(1)], dim=1)


class _AttnBlock(nn.Module):
    """Pre-norm transformer block; self-attention when ``kv`` is None, else cross.

    Uses scaled_dot_product_attention so the [B, heads, N_q, N_kv] matrix is
    never materialised — with all source views as keys N_kv is (V-1)*N, which
    would otherwise cost hundreds of MB per block just to hold for backward.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, cross: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        # self-attention reuses norm_q for the keys; allocating an unused
        # norm_kv would leave parameters without gradients, which DDP rejects.
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6) if cross else None
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        B, N, C = t.shape
        return t.view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

    def forward(self, x: torch.Tensor, kv: torch.Tensor | None = None) -> torch.Tensor:
        q_in = self.norm_q(x)
        kv_in = q_in if kv is None or self.norm_kv is None else self.norm_kv(kv)
        q = self._heads(self.to_q(q_in))
        k, v = self.to_kv(kv_in).chunk(2, dim=-1)
        a = F.scaled_dot_product_attention(q, self._heads(k), self._heads(v))
        B, _, N, _ = a.shape
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, -1))
        return x + self.mlp(self.norm2(x))


class SVAFusion(nn.Module):
    """MVSFormer++'s Side View Attention (their ``CrossVITDecoder``), verbatim in
    direction: the reference stream is built by progressive AAS self-attention
    over the DINOv3 layers, and each source view is a *query* against that
    stream.

    Reference branch: the shallowest layer seeds the stream, then each further
    layer folds in as ``norm(aas * self_attn(prev) + layer_i)``. That residual —
    not a plain concatenation of the three layers — is what their ablation
    credits.

    Source branch: ``cross_attn(query=src_layer_i, key/value=ref_stage_i)``. This
    direction (src asks ref, not the reverse) is what yields one feature *per
    view*, which is what the cost volume needs — every view is warped, so every
    view needs its own descriptor.

    Returns ``[B, V, N, dim]``: view 0 is the fused reference, the rest are the
    cross-attended sources.
    """

    def __init__(self, in_dim: int, cfg: SPREConfig, n_layers: int) -> None:
        super().__init__()
        dim, heads = int(cfg.attn_dim), int(cfg.num_heads)
        self.in_proj = nn.Linear(in_dim, dim)
        self.out_dim = dim

        self.self_blocks = nn.ModuleList(_AttnBlock(dim, heads) for _ in range(n_layers - 1))
        self.self_norms = nn.ModuleList(nn.LayerNorm(dim, eps=1e-6) for _ in range(n_layers - 1))
        self.self_aas = nn.ParameterList(
            nn.Parameter(torch.tensor(float(cfg.aas_init))) for _ in range(n_layers - 1)
        )
        self.cross_blocks = nn.ModuleList(_AttnBlock(dim, heads, cross=True) for _ in range(n_layers))
        self.cross_norms = nn.ModuleList(nn.LayerNorm(dim, eps=1e-6) for _ in range(n_layers - 1))
        self.cross_aas = nn.ParameterList(
            nn.Parameter(torch.tensor(float(cfg.aas_init))) for _ in range(n_layers - 1)
        )

    def forward(self, layers: list[torch.Tensor]) -> torch.Tensor:
        """``layers``: per DINOv3 layer, [B, V, N, in_dim]. Returns [B, V, N, dim]."""
        x = [self.in_proj(t) for t in layers]

        ref = [x[0][:, 0]]
        for i, (blk, norm, aas) in enumerate(zip(self.self_blocks, self.self_norms, self.self_aas)):
            ref.append(norm(aas * blk(ref[-1]) + x[i + 1][:, 0]))

        V = x[0].shape[1]
        outs = [ref[-1]]
        for v in range(1, V):
            ctx = None
            for i, blk in enumerate(self.cross_blocks):
                if i == 0:
                    q = x[i][:, v]
                else:
                    q = self.cross_norms[i - 1](self.cross_aas[i - 1] * ctx + x[i][:, v])
                ctx = blk(q, kv=ref[i])
            outs.append(ctx)
        return torch.stack(outs, dim=1)


class DinoSVA(nn.Module):
    """Frozen DINOv3 + SVA fusion — one forward, two consumers.

    Runs once per step over all views and hands the same fused tokens to
    (a) the FPN bottleneck, as per-view matching features, and (b) the SPRE
    head, as the reference view's semantic descriptor. Before this split the
    ViT was paid for and then used only to predict one scalar per pixel.

    DINOv3 stays frozen: gradients start at ``fusion.in_proj``.

    ``sva_cfg.full`` switches the projection to the FPN width from a 1x1 conv to
    MVSFormer++'s proj + two deconvs (``models/sva.SVADeconv``). The rest of the
    complete SVA — FMT on the 1/8 FPN output plus the second pathway — lives in
    ``models/sva.SVAPathway`` and runs after the FPN, as in MVSFormer++.
    """

    def __init__(self, cfg: SPREConfig, dino_cfg: DINOConfig, weights_file,
                 fpn_channels: int | None = None, sva_cfg: SVAConfig | None = None) -> None:
        super().__init__()
        self.layers = tuple(dino_cfg.layers)
        self.patch_size = int(dino_cfg.patch_size)
        self.max_side = int(dino_cfg.max_side)

        self.dino = load_dinov3_vit_base("cpu", weights_file, patch_size=self.patch_size)
        for p in self.dino.parameters():
            p.requires_grad_(False)
        self.register_buffer("dino_mean", torch.tensor(dino_cfg.mean).view(1, 3, 1, 1))
        self.register_buffer("dino_std", torch.tensor(dino_cfg.std).view(1, 3, 1, 1))

        self.fusion = SVAFusion(768, cfg, n_layers=len(self.layers))
        self.out_dim = self.fusion.out_dim
        self.full_sva = bool(sva_cfg is not None and sva_cfg.full)
        if self.full_sva and not fpn_channels:
            raise ValueError("sva.full 需要 feed_fpn (第二段 SVA 作用在融合了 DINO 的 1/8 特征上)")
        self.to_fpn = self.deconv = None
        if self.full_sva:
            from models.sva import SVADeconv
            self.deconv = SVADeconv(self.out_dim, fpn_channels)
        elif fpn_channels:
            # 1x1 to the FPN's width, applied on the patch grid: a pointwise conv
            # commutes exactly with bilinear upsampling, so projecting 26x32 instead
            # of the target grid is identical arithmetic and far cheaper.
            self.to_fpn = nn.Conv2d(self.out_dim, fpn_channels, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()          # frozen backbone never leaves eval
        return self

    def _tokens(self, images: torch.Tensor) -> tuple[list[torch.Tensor], tuple[int, int]]:
        B, V, _, H, W = images.shape
        dh, dw = compute_patch_aligned_size(H, W, self.max_side, self.patch_size)
        self.dino.eval()
        with torch.no_grad():
            x = images.reshape(B * V, 3, H, W)
            x = F.interpolate(x, size=(dh, dw), mode="bilinear", align_corners=False)
            x = (x - self.dino_mean) / self.dino_std
            feats = self.dino.get_intermediate_layers(x, n=self.layers, reshape=True, norm=True)
        gh, gw = feats[0].shape[-2:]
        return [f.float().flatten(2).transpose(1, 2).view(B, V, gh * gw, -1) for f in feats], (gh, gw)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        """``images``: [B, V, 3, H, W] in [0, 1].

        Returns ``(fused, raw_deepest, grid)``: the SVA-fused per-view features
        [B, V, N, dim], the frozen deepest DINOv3 layer [B, V, N, 768] for the
        consistency measurement, and the token grid.
        """
        tokens, grid = self._tokens(images)
        return self.fusion(tokens), tokens[-1], grid

    def fpn_feature(self, fused: torch.Tensor, grid: tuple[int, int],
                    target_hw: tuple[int, int]) -> torch.Tensor:
        """[B, V, N, dim] -> [B, V, fpn_channels, target_hw] for FPN injection."""
        assert self.to_fpn is not None or self.deconv is not None, \
            "DinoSVA built without fpn_channels"
        B, V, _, C = fused.shape
        gh, gw = grid
        x = fused.reshape(B * V, gh, gw, C).permute(0, 3, 1, 2)
        x = self.deconv(x) if self.full_sva else self.to_fpn(x)
        if tuple(x.shape[-2:]) != tuple(target_hw):
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return x.reshape(B, V, -1, *target_hw)


class SPRE(nn.Module):
    """Prior-reliability head on top of ``DinoSVA``'s reference stream.

    Consumes the fused reference tokens plus four online statistics of the
    (possibly corrupted) depth prior, and emits reliability LOGITS. Raw logits,
    not the sigmoid, so the loss can use autocast-safe
    ``binary_cross_entropy_with_logits``; the network applies the sigmoid to get
    ``r`` for the hypothesis builder.
    """

    # depth_norm, mad_rel, grad_rel, valid | consist_mean, consist_worst, visible
    N_STAT = 7

    # Fixed per-channel scales, not a norm layer. Measured on real DTU depth the
    # raw channels span 300x (mad_rel std 0.0033 vs the semantic channels' ~0.8),
    # so they need rebalancing — but a GroupNorm here would divide the nearly
    # constant channels (``valid`` sits at ~0.99, ``visible`` often at 1.0) by a
    # near-zero std and amplify pure noise. Constants keep the transform
    # deterministic and safe on degenerate channels.
    # Measured on real DTU with frozen DINOv3 tokens: consist_mean 0.885 +- 0.072,
    # consist_worst 0.842 +- 0.109 — a large constant offset carrying no signal
    # and a variation 10x smaller than the semantic channels. Centring then
    # scaling puts the informative part at std 0.58 / 0.87, matching everything
    # else. Correlation with the prior's true error is -0.51, and it separates
    # above ~10mm (below that the disparity is sub-token, so it cannot and need
    # not discriminate — the window width only cares about tens of mm).
    STAT_SCALE = (1.0, 100.0, 2.0, 1.0, 8.0, 8.0, 1.0)
    STAT_OFFSET = (0.0, 0.0, 0.0, 0.0, 0.85, 0.85, 0.0)

    def __init__(self, cfg: SPREConfig, in_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Conv2d(in_dim, cfg.proj_dim, kernel_size=1)
        self.proj_post = nn.Sequential(
            nn.GroupNorm(min(8, cfg.proj_dim), cfg.proj_dim),
            nn.GELU(),
        )
        self.register_buffer("stat_scale", torch.tensor(self.STAT_SCALE).view(1, -1, 1, 1))
        self.register_buffer("stat_offset", torch.tensor(self.STAT_OFFSET).view(1, -1, 1, 1))
        in_ch = cfg.proj_dim + self.N_STAT
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, cfg.hidden, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, cfg.hidden), cfg.hidden),
            nn.GELU(),
            nn.Conv2d(cfg.hidden, cfg.hidden, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, cfg.hidden), cfg.hidden),
            nn.GELU(),
            nn.Conv2d(cfg.hidden, 1, kernel_size=1),
        )

    def forward(self, ref_tokens: torch.Tensor, grid: tuple[int, int],
                depth_prior: torch.Tensor, target_hw: tuple[int, int],
                consistency: torch.Tensor | None = None) -> torch.Tensor:
        """``ref_tokens``: [B, N, dim] (view 0 of DinoSVA). Returns [B, *target_hw].

        ``consistency``: [B, 3, gh, gw] from ``prior_view_consistency`` — the
        cross-view evidence. Zeros when unavailable (single view).
        """
        B, _, C = ref_tokens.shape
        gh, gw = grid
        f_sem = ref_tokens.reshape(B, gh, gw, C).permute(0, 3, 1, 2)
        f_sem = self.proj(f_sem)                                   # project on the patch grid
        f_sem = F.interpolate(f_sem, size=target_hw, mode="bilinear", align_corners=False)
        f_sem = self.proj_post(f_sem)

        d = F.interpolate(depth_prior.unsqueeze(1).float(), size=target_hw, mode="nearest").squeeze(1)
        valid = torch.isfinite(d) & (d > 0)
        d = torch.nan_to_num(d, nan=0.0)
        prior_stats = torch.stack(
            [_masked_standardize(d, valid), _local_mad_rel(d, valid),
             _grad_mag_rel(d, valid), valid.float()],
            dim=1,
        )
        if consistency is None:
            consistency = prior_stats.new_zeros(B, 3, *prior_stats.shape[-2:])
        elif consistency.shape[-2:] != target_hw:
            consistency = F.interpolate(consistency.float(), size=target_hw,
                                        mode="bilinear", align_corners=False)
        stats = (torch.cat([prior_stats, consistency.float()], dim=1)
                 - self.stat_offset) * self.stat_scale
        return self.head(torch.cat([f_sem, stats.to(f_sem.dtype)], dim=1)).squeeze(1)
