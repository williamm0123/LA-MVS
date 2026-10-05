"""Synthetic unit tests for LAPE (docs: LAPE 终版方案) and the DA3 feature path.

    pytest tests/test_lape_unit.py -q

Everything runs on the CPU with a stub DA3 network, so no weights are needed.
"""
from __future__ import annotations

import dataclasses
import math

import pytest
import torch
import torch.nn as nn

from base.config_moa import LAPEConfig, MoAMVSConfig, build_moa_config
from models.moa.calibrator import SigmaCalibrator, calibrated_sigma
from models.moa.edge import NeighborhoodBarrier
from models.moa.geometry import depth_to_u, u_to_depth
from models.moa.lape_affine import WindowAffineExperts, _sse, expert_offsets
from models.moa.low_freq_recall import OFFSETS7, low_freq_recall
from models.moa.normal_evidence import NormalEvidence, camera_normals
from models.moa.prior_evidence import bin_mass, log_prior, normalized_weights
from models.moa.rac import RACAligner

VMIN = torch.tensor(1.0 / 935.0).view(1, 1, 1, 1)
VMAX = torch.tensor(1.0 / 425.0).view(1, 1, 1, 1)
DG = 1.0 / 47.0


def _ramp(H=24, W=32, lo=0.3, hi=0.6):
    xs = torch.linspace(lo, hi, W).view(1, 1, 1, W).expand(1, 1, H, W)
    ys = torch.linspace(0.0, 0.05, H).view(1, 1, H, 1)
    return (xs + ys).contiguous()


def _experts(x, y, anchor=None, emb=None, pass_mask=None, **kw):
    we = WindowAffineExperts(**kw)
    B, _, H, W = x.shape
    if anchor is None:
        anchor = torch.ones_like(x)
    if emb is None:
        emb = torch.zeros(B, 4, H, W)
        emb[:, 0] = 1.0
    if pass_mask is None:
        pass_mask = torch.ones(B, len(we.offsets), H, W)
    return we(x, y, torch.ones_like(x), anchor, emb, pass_mask)


# --------------------------------------------------------------------------- window experts
def test_window_experts_recover_exact_affine():
    x = _ramp()
    y = 1.2 * x + 0.03
    r = _experts(x, y)
    inner = (slice(None), slice(None), slice(6, -6), slice(6, -6))
    for s in range(3):
        assert torch.allclose(r.mu[:, s:s + 1][inner], y[inner], atol=1e-5), s
        assert r.valid[:, s:s + 1][inner].min() == 1.0
        assert float(r.s_ho[:, s:s + 1][inner].max()) < 1e-4


def test_window_experts_heldout_residual_tracks_noise():
    torch.manual_seed(0)
    x = _ramp(40, 48)
    noise = 0.002
    y = x + 0.01 + noise * torch.randn_like(x)
    r = _experts(x, y)
    inner = (slice(None), slice(None), slice(8, -8), slice(8, -8))
    s7 = float(r.s_ho[:, 1:2][inner].median())
    assert 0.7 * noise < s7 < 1.6 * noise
    # more anchors -> smaller parameter variance of the centre prediction
    assert float(r.sigma_par[:, 1:2][inner].median()) < float(r.sigma_par[:, 0:1][inner].median())


def test_sse_expansion_matches_direct_sum():
    torch.manual_seed(1)
    w = torch.rand(30)
    xt = torch.randn(30) * 1e-2
    d = 0.5 * xt + 0.01 + torch.randn(30) * 1e-3
    alpha, bt = torch.tensor(0.3), torch.tensor(0.012)
    S6 = [w.sum(), (w * xt).sum(), (w * xt * xt).sum(), (w * d).sum(), (w * xt * d).sum(), (w * d * d).sum()]
    direct = (w * (d - alpha * xt - bt) ** 2).sum()
    assert torch.allclose(_sse(alpha, bt, S6), direct, rtol=1e-4, atol=1e-10)


def test_wild_slope_falls_back_to_offset_only():
    torch.manual_seed(2)
    x = torch.full((1, 1, 16, 16), 0.5) + 1e-4 * torch.randn(1, 1, 16, 16)   # x barely varies
    y = x * 0 + 0.52 + 1e-3 * torch.randn(1, 1, 16, 16)                       # unrelated to x
    r = _experts(x, y)
    inner = (slice(None), slice(None), slice(6, -6), slice(6, -6))
    assert float(r.valid[inner].mean()) == 1.0
    assert torch.all(r.alpha[inner] == 0)
    assert float((r.mu[inner] - 0.52).abs().max()) < 5e-3


def test_offsets_union_and_e11_sparsity():
    offs, member = expert_offsets()
    assert len(offs) == 73
    assert sum(m[2] for m in member) == 49 and sum(m[1] for m in member) == 49 and sum(m[0] for m in member) == 9
    assert (0, 2) not in [o for o, m in zip(offs, member) if m[2]]
    assert (5, -3) in [o for o, m in zip(offs, member) if m[2]]


def test_barrier_subset_blocks_skipped_edge():
    offs, _ = expert_offsets()
    bar = NeighborhoodBarrier(radius=5, subset=offs)
    H, W = 16, 24
    edge = torch.zeros(1, 1, H, W)
    edge[..., 12] = 1.0                       # vertical edge at column 12
    out = bar(edge, torch.zeros(1, 1, H, W), torch.ones(1, 1, H, W), 0.5, 1.0)
    k = offs.index((0, 5))                    # sampled offset, path crosses column 12 from column 9
    assert out["pass"][0, k, 8, 9] == 0       # blocked although column 12 itself is never sampled
    assert out["pass"][0, k, 8, 13] == 1


# --------------------------------------------------------------------------- RAC
def test_rac_two_surfaces_two_models_and_regions():
    torch.manual_seed(3)
    H, W = 48, 64
    z_mono = 1.0 + torch.linspace(0, 1, W).view(1, 1, 1, W).expand(1, 1, H, W).clone()
    fg = torch.zeros(1, 1, H, W, dtype=torch.bool)
    fg[..., 10:38, 8:30] = True
    z_mono[fg] += 2.0                                    # DA3: object in front of a slanted wall
    z_mvs = torch.where(fg, 300.0 * z_mono + 50.0, 250.0 * z_mono + 200.0)   # different affine per surface
    edge = torch.zeros(1, 1, H, W)
    ring = fg ^ torch.nn.functional.max_pool2d(fg.float(), 3, 1, 1).bool()
    edge[ring] = 1.0
    rac = RACAligner(tau=0.5, min_eff=16.0, max_models=3)
    inv_bin = torch.tensor([1.0 / ((1 / 425 - 1 / 935) * DG)])
    vmin = torch.tensor(1.0 / 2000).view(1, 1, 1, 1)
    vmax = torch.tensor(1.0 / 300).view(1, 1, 1, 1)
    res = rac(z_mono, torch.ones_like(z_mono, dtype=torch.bool), z_mvs, torch.ones_like(z_mono), edge,
              inv_bin, vmin, vmax, DG)
    assert int(res.ok.sum()) == 2
    err = (res.z_rac - z_mvs).abs() / z_mvs
    assert float(err[fg].median()) < 1e-3 and float(err[~fg].median()) < 1e-3
    assert res.model_map[fg].unique().numel() == 1 and res.model_map[~fg & ~ring].unique().numel() == 1
    assert int(res.model_map[fg][0]) != int(res.model_map[~fg & ~ring][0])


# --------------------------------------------------------------------------- prior evidence
def test_bin_mass_inside_and_outside_window():
    D, H, W = 8, 4, 4
    u = torch.linspace(0.6, 0.5, D).view(1, D, 1, 1).expand(1, D, H, W)
    mu = torch.full((1, 1, H, W), 0.55)
    sig = torch.full((1, 1, H, W), 0.003)
    w = torch.ones(1, 1, H, W)
    q, M = bin_mass(u, mu, sig, w)
    assert torch.allclose(M, torch.ones_like(M), atol=1e-3)
    ell, act = log_prior(q, M, 0.2, 0.05, 4.0)
    assert act.all() and torch.allclose(ell.amax(1), torch.zeros(1, H, W)) and float(ell.min()) >= -4.0
    q2, M2 = bin_mass(u, torch.full_like(mu, 0.3), sig, w)       # prior far outside: off, not renormalised
    ell2, act2 = log_prior(q2, M2, 0.2, 0.05, 4.0)
    assert float(M2.max()) < 1e-6 and not act2.any() and float(ell2.abs().max()) == 0.0


def test_normalized_weights_ignore_invalid():
    pi = torch.tensor([0.5, 0.3, 0.2]).view(1, 3, 1, 1)
    valid = torch.tensor([1.0, 0.0, 1.0]).view(1, 3, 1, 1)
    w = normalized_weights(pi, valid)
    assert torch.allclose(w.flatten(), torch.tensor([0.5 / 0.7, 0.0, 0.2 / 0.7]))
    assert normalized_weights(pi, torch.zeros_like(valid)).abs().sum() == 0


# --------------------------------------------------------------------------- normal evidence
def _K(H, W, f=50.0):
    return torch.tensor([[[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]]])


def test_camera_normals_of_a_slanted_plane():
    H, W = 20, 24
    K = _K(H, W)
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    n_true = torch.tensor([0.3, -0.2, -1.0])
    n_true = n_true / n_true.norm()
    rays = torch.stack([(xs - W / 2) / 50.0, (ys - H / 2) / 50.0, torch.ones_like(xs)], 0)
    # camera-facing plane through (0, 0, 600): n.X = 600 n_z  ->  z = 600 n_z / (n . ray) > 0
    z = (600.0 * n_true[2] / (n_true.view(3, 1, 1) * rays).sum(0)).view(1, 1, H, W)
    assert float(z.min()) > 0
    n, ok, _ = camera_normals(z, torch.ones_like(z, dtype=torch.bool), torch.zeros_like(z), K, 0.03)
    inner = n[0, :, 2:-2, 2:-2].reshape(3, -1)
    cos = (inner * n_true.view(3, 1)).sum(0)
    assert ok[0, 0, 2:-2, 2:-2].all() and float(cos.min()) > 0.999


def test_nce_zero_init_and_fronto_parallel_propagation():
    torch.manual_seed(4)
    B, G, D, H, W = 1, 4, 6, 10, 12
    K = _K(H, W)
    u = torch.linspace(0.8, 0.3, D).view(1, D, 1, 1).expand(B, D, H, W).contiguous()
    base = torch.randn(1, G, D, 1, 1)
    cv = base.expand(B, G, D, H, W).contiguous()             # every pixel has the same cost curve
    n = torch.zeros(B, 3, H, W)
    n[:, 2] = -1.0                                          # fronto-parallel: r = 1 for every neighbour
    nce = NormalEvidence(G, radius=1, dilation=1, kappa_init=1e-6)
    pm = torch.ones(B, len(nce.offsets), H, W)
    delta, gate = nce(cv, u, n, torch.ones(B, 1, H, W), K, pm, VMIN, VMAX)
    assert float(delta.abs().max()) == 0.0                  # W = 0 at init
    with torch.no_grad():
        nce.W.copy_(torch.eye(G).expand_as(nce.W) / 1.0)
    delta, gate = nce(cv, u, n, torch.ones(B, 1, H, W), K, pm, VMIN, VMAX)
    from models.moa.normal_evidence import normalize_cost
    target = normalize_cost(cv)
    inner = (slice(None), slice(None), slice(1, -1), slice(2, -2), slice(2, -2))
    # tolerance band of +-0.5 bin around an exact hit: log-sum-exp of neighbours >= the hit itself
    assert float((delta[inner] - target[inner]).abs().mean()) < 0.6
    assert float(gate[..., 2:-2, 2:-2].min()) > 0.5


# --------------------------------------------------------------------------- low-frequency recall
def _lfr_scene(collapse: bool, conf_value: float, mvs_correct: bool = False):
    H, W = 40, 48
    xs = torch.linspace(0.40, 0.60, W).view(1, 1, 1, W).expand(1, 1, H, W).clone()
    truth = xs + 0.0
    x = truth.clone()                                        # aligned mono == truth (smooth plane)
    y = truth.clone()
    blk = (slice(None), slice(None), slice(12, 30), slice(14, 34))
    if collapse and not mvs_correct:
        y[blk] = truth[blk].mean() + 0.15                   # collapsed: constant, far from truth
    du = torch.full((1, 1, H, W), DG)
    gray = torch.full((1, 1, H * 8, W * 8), 0.5)            # textureless image
    pass7 = torch.ones(1, len(OFFSETS7), H, W)
    conf = torch.full((1, 1, H, W), conf_value)
    cfg = LAPEConfig()
    res = low_freq_recall(gray, x, torch.ones_like(x, dtype=torch.bool), y, du, pass7,
                          torch.ones_like(x, dtype=torch.bool), torch.zeros_like(x), 3.0, cfg, DG, conf=conf)
    inside = torch.zeros_like(x, dtype=torch.bool)
    inside[blk] = True
    return res.flag, inside


def test_lfr_fires_on_collapsed_untrusted_region_only():
    flag, inside = _lfr_scene(collapse=True, conf_value=0.1)
    core = inside.clone()
    core[..., :15, :] = False
    core[..., 27:, :] = False
    core[..., :, :17] = False
    core[..., :, 31:] = False
    assert float(flag[core].float().mean()) > 0.9
    assert float(flag[~inside].float().mean()) == 0.0


def test_lfr_silent_when_matching_is_trusted_or_correct():
    flag, _ = _lfr_scene(collapse=True, conf_value=0.9)
    assert not flag.any()
    flag, _ = _lfr_scene(collapse=False, conf_value=0.1)
    assert not flag.any()


# --------------------------------------------------------------------------- calibrator
def test_calibrator_monotone_in_residual_and_distance():
    torch.manual_seed(5)
    cal = SigmaCalibrator()
    with torch.no_grad():
        cal.w2.fill_(0.5)                                   # make h non-trivial
    s = torch.linspace(1e-4, 1e-2, 50).view(1, 1, 1, 50)
    one = torch.ones_like(s)
    sig, _ = calibrated_sigma(cal, s, 1e-3 * one, 2.0 * one, 9.0 * one, 0 * one, 0 * one, DG, 4.0, 0.1)
    assert torch.all(sig[..., 1:] >= sig[..., :-1] - 1e-9)
    d = torch.linspace(0, 30, 50).view(1, 1, 1, 50)
    sig2, _ = calibrated_sigma(cal, 1e-3 * one, 1e-3 * one, d, 9.0 * one, 0 * one, 0 * one, DG, 4.0, 0.1)
    assert torch.all(sig2[..., 1:] >= sig2[..., :-1] - 1e-9)


# --------------------------------------------------------------------------- network with a stub DA3
class _StubHead(nn.Module):
    patch_size = 14

    def forward(self, feats, H, W, patch_start_idx=0):
        t = feats[-1][0]                                     # [B,S,N,C]
        B, S = t.shape[:2]
        ph, pw = H // 14, W // 14
        m = t.mean(-1).view(B * S, 1, ph, pw)
        d = torch.nn.functional.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
        return {"depth": (1.0 + 0.1 * d.sigmoid()).view(B, S, H, W), "sky": torch.zeros(B, S, H, W)}


class _StubViT(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.embed_dim = dim
        self.proj = nn.Conv2d(3, dim, 14, stride=14)


class _StubBackbone(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.out_layers = [0, 1, 2, 3]
        self.pretrained = _StubViT(dim)

    def forward(self, x, cam_token=None, export_feat_layers=()):
        B, S = x.shape[:2]
        t = self.pretrained.proj(x.flatten(0, 1)).flatten(2).transpose(1, 2)
        t = t.view(B, S, t.shape[1], t.shape[2])
        return tuple((t * (1 + 0.1 * i), t[:, :, 0]) for i in range(4)), []


class _StubDA3(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _StubBackbone()
        self.head = _StubHead()


def _tiny_cfg(**lape_kw) -> MoAMVSConfig:
    cfg = build_moa_config("local")
    return dataclasses.replace(
        cfg, feat=dataclasses.replace(cfg.feat, process_res=112),
        lape=dataclasses.replace(cfg.lape, **lape_kw),
        moa=dataclasses.replace(cfg.moa, global_min_eff=4.0))


def _batch(B=1, V=3, H=64, W=80):
    torch.manual_seed(6)
    dmin, interval, nd = 425.0, 2.5, 192
    dv = torch.arange(nd, dtype=torch.float32) * interval + dmin
    b = {"images": torch.rand(B, V, 3, H, W) * 255.0,
         "intrinsics": torch.tensor([[[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]]]).repeat(B, V, 1, 1),
         "extrinsics": torch.eye(4).repeat(B, V, 1, 1), "depth_values": dv.unsqueeze(0).repeat(B, 1)}
    b["extrinsics"][:, 1:, 0, 3] = 5.0
    return b


def test_network_lape_forward_backward_cpu():
    from losses.moa_loss import MoALoss
    from models.network_moa import MoAMVSNet
    cfg = _tiny_cfg()
    net = MoAMVSNet(cfg, da3_net=_StubDA3())
    net.train()
    b = _batch()
    b["depth_gt"] = torch.full((1, 64, 80), 600.0)
    b["mask"] = torch.ones(1, 64, 80)
    out = net(b)
    for k in ("stage1a", "stage1", "stage2", "stage3", "stage4", "lape1", "moa2", "moa3", "moa4",
              "fusion1", "fusion2", "fusion3", "fusion4"):
        assert k in out, k
    loss, logs = MoALoss(cfg.loss, cfg.cascade.num_depths[0])(out, b, diagnostics=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert net.da3_sva.fusion.in_proj.weight.grad is not None
    assert all(p.grad is None for p in net.da3_sva.da3.parameters())


def test_stage1_pass_b_equals_pass_a_at_init():
    """Zero-init NCE / adapter and a closed gamma: pass B must reproduce pass A exactly."""
    from models.network_moa import MoAMVSNet
    cfg = _tiny_cfg()
    net = MoAMVSNet(cfg, da3_net=_StubDA3())
    with torch.no_grad():
        for h in net.gamma_heads:
            h.net[-1].bias.fill_(-60.0)
    net.eval()
    with torch.no_grad():
        out = net(_batch())
    assert torch.equal(out["stage1a"]["logits_raw"], out["stage1"]["logits_raw"])
    assert torch.allclose(out["stage1a"]["depth"], out["stage1"]["depth"])


def test_legacy_snapshot_rebuilds_without_lape():
    from base.config_moa import apply_arch_snapshot, arch_snapshot
    snap = arch_snapshot(build_moa_config("local"))
    snap.pop("feat")
    snap.pop("lape")
    cfg = apply_arch_snapshot(build_moa_config("local"), snap)
    assert cfg.feat.backbone == "dinov3" and cfg.lape.enabled is False


def test_lape_losses_do_not_touch_the_matching_network():
    """With every CE weight at zero, the centre / shape / confidence / sigma-NLL losses must
    train only the LAPE cascade (calibrators, mixture, confidence heads), as in MoA."""
    from losses.moa_loss import MoALoss
    from models.network_moa import MoAMVSNet
    cfg = _tiny_cfg()
    net = MoAMVSNet(cfg, da3_net=_StubDA3())
    net.train()
    b = _batch()
    b["depth_gt"] = torch.full((1, 64, 80), 600.0)
    b["mask"] = torch.ones(1, 64, 80)
    out = net(b)
    lcfg = dataclasses.replace(cfg.loss, stage_weights=(0.0, 0.0, 0.0, 0.0), w_ce_stage1a=0.0)
    loss, _ = MoALoss(lcfg, cfg.cascade.num_depths[0])(out, b)
    loss.backward()
    for mod in (net.fpn, net.sva_pathway, net.cost_volumes, net.decoders, net.da3_sva, net.nce,
                net.prior_adapters, net.gamma_heads):
        for p in mod.parameters():
            assert p.grad is None or float(p.grad.abs().max()) == 0.0
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in net.moa.parameters())
