from __future__ import annotations

from dataclasses import dataclass

import math

import torch
import torch.nn.functional as F

from base.config import DepthRangeConfig
from utils.geometry import make_depth_hypotheses_global


@dataclass
class Stage1Hypotheses:
    """Sorted stage-1 hypothesis axis with branch bookkeeping.

    hypos       [B, D, H, W]  merged sorted depth axis (D = num_global + num_local)
    is_local    [B, D, H, W]  1.0 where the bin came from the local branch
    interval    [B, D, H, W]  per-bin physical spacing (h_{i+1}-h_{i-1})/2
    global_idx  [B, Dg, H, W] positions of the global bins on the sorted axis,
                              ascending in depth (gathers a monotone sub-axis)
    local_idx   [B, Dl, H, W] same for the local bins
    global_lo/hi [B]          per-image global-branch bounds (the guard range)
    local_lo/hi [B, H, W]     per-pixel local-branch bounds
    prior/conf  [B, H, W]     prior depth & confidence resampled to stage res
    edge        [B, H, W]     rule-based edge/unreliable map in [0, 1]
    branch_active [B, H, W]   True 时 local 分支确实携带先验候选; False 时那些
                              bin 是硬门触发后的 guard 加密网格, 应按 global 处理
    """

    hypos: torch.Tensor
    is_local: torch.Tensor
    interval: torch.Tensor
    global_idx: torch.Tensor
    local_idx: torch.Tensor
    global_lo: torch.Tensor
    global_hi: torch.Tensor
    local_lo: torch.Tensor
    local_hi: torch.Tensor
    prior: torch.Tensor
    conf: torch.Tensor
    edge: torch.Tensor
    branch_active: torch.Tensor


def _resize_map(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(hw):
        return x
    return F.interpolate(x.unsqueeze(1), size=hw, mode="bilinear", align_corners=False).squeeze(1)


def edge_map_from_prior(prior: torch.Tensor, valid: torch.Tensor, edge_grad_rel: float) -> torch.Tensor:
    """Relative depth-gradient edge/unreliable map in [0, 1], band-widened 3x3.

    Depth (not RGB) gradients: texture edges are not depth edges. Invalid prior
    pixels count as fully unreliable.
    """
    d = torch.where(valid, prior, torch.zeros_like(prior))
    pad = F.pad(d.unsqueeze(1), (1, 1, 1, 1), mode="replicate")
    gx = (pad[:, :, 1:-1, 2:] - pad[:, :, 1:-1, :-2]).abs() * 0.5
    gy = (pad[:, :, 2:, 1:-1] - pad[:, :, :-2, 1:-1]).abs() * 0.5
    grad = torch.maximum(gx, gy).squeeze(1)
    rel = grad / (prior.abs() + 1.0)
    e = (rel / max(edge_grad_rel, 1e-6)).clamp(0.0, 1.0)
    e = torch.where(valid, e, torch.ones_like(e))
    return F.max_pool2d(e.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)


def _robust_global_bounds(
    prior: torch.Tensor,
    valid: torch.Tensor,
    depth_min: torch.Tensor,
    depth_max: torch.Tensor,
    cfg: DepthRangeConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-image guard bounds: prior quantiles + margin, clamped to the physical
    range and never narrower than global_min_span_frac of it."""
    B = prior.shape[0]
    flat = torch.where(valid, prior, torch.full_like(prior, float("nan"))).view(B, -1)
    q = torch.nanquantile(
        flat.float(),
        torch.tensor([cfg.global_quantile_lo, cfg.global_quantile_hi], device=prior.device, dtype=torch.float32),
        dim=1,
    )  # [2, B]
    q_lo, q_hi = q[0], q[1]
    bad = ~(torch.isfinite(q_lo) & torch.isfinite(q_hi) & (q_hi > q_lo))
    q_lo = torch.where(bad, depth_min, q_lo)
    q_hi = torch.where(bad, depth_max, q_hi)

    margin = cfg.global_margin_ratio * (q_hi - q_lo)
    lo = q_lo - margin
    hi = q_hi + margin

    # Fixed-width window placement: widen to at least min_span, cap at the
    # physical span, then slide the whole window inside [depth_min, depth_max].
    # (A naive expand-then-clamp silently eats the expansion on the clamped
    # side without compensating on the other, so an offset quantile range
    # never reaches the far bound even at min_span_frac=1.0.)
    span_phys = (depth_max - depth_min).clamp_min(1e-3)
    width = torch.maximum(hi - lo, cfg.global_min_span_frac * span_phys)
    width = torch.minimum(width, span_phys)
    center = 0.5 * (lo + hi)
    center = torch.maximum(center, depth_min + 0.5 * width)
    center = torch.minimum(center, depth_max - 0.5 * width)
    lo = (center - 0.5 * width).clamp_min(1e-3)
    hi = center + 0.5 * width
    # degenerate (physical range itself tiny/inverted) -> physical
    degen = hi - lo < 1e-3
    lo = torch.where(degen, depth_min, lo)
    hi = torch.where(degen, depth_max, hi)
    return lo, hi


def _global_branch(
    lo: torch.Tensor,
    hi: torch.Tensor,
    num: int,
    hw: tuple[int, int],
    inverse_depth: bool,
) -> torch.Tensor:
    """[B, num, H, W] confidence-independent guard bins, unique and monotone."""
    B = lo.shape[0]
    t = torch.linspace(0.0, 1.0, num, device=lo.device, dtype=torch.float32).view(1, num)
    if inverse_depth:
        inv = (1.0 / hi).view(B, 1) + ((1.0 / lo) - (1.0 / hi)).view(B, 1) * t
        bins = (1.0 / inv).flip(dims=[1])  # ascending depth
    else:
        bins = lo.view(B, 1) + (hi - lo).view(B, 1) * t
    return bins.view(B, num, 1, 1).expand(B, num, hw[0], hw[1])


def _neighborhood_median_mad(prior: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """3x3 robust stats over valid neighbors: (median, MAD, has_any_valid)."""
    B, H, W = prior.shape
    nan = float("nan")
    d = torch.where(valid, prior, torch.full_like(prior, nan))
    patches = F.unfold(d.unsqueeze(1), kernel_size=3, padding=1).view(B, 9, H, W)
    med = patches.nanmedian(dim=1).values
    mad = (patches - med.unsqueeze(1)).abs().nanmedian(dim=1).values
    has_valid = torch.isfinite(med)
    med = torch.nan_to_num(med, nan=0.0)
    mad = torch.nan_to_num(mad, nan=0.0)
    return med, mad, has_valid


def _local_branch(
    prior: torch.Tensor,
    conf: torch.Tensor,
    valid: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    global_interval: torch.Tensor,
    num: int,
    cfg: DepthRangeConfig,
    gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[B, num, H, W] dense bins around a spike-robust prior center.

    Width scales with (1 - conf) between a floor and a ceiling measured in
    global-interval units: the floor keeps a confidently-wrong prior from
    locking the search to a point; the ceiling keeps this branch dense (wide
    coverage is the global branch's job). Pixels with no usable prior fall back
    to a half-offset uniform grid over the guard range (raising its effective
    resolution instead of duplicating it).
    """
    B, H, W = prior.shape
    med, mad, has_nbr = _neighborhood_median_mad(prior, valid)

    mad_floor = cfg.spike_min_mad_rel * med.abs().clamp_min(1.0)
    is_spike = valid & has_nbr & ((prior - med).abs() > cfg.spike_k * torch.maximum(mad, mad_floor))
    center = torch.where(is_spike, med, prior)

    gi = global_interval.view(B, 1, 1)
    half_min = cfg.local_half_min_gi * gi
    half_max = cfg.local_half_max_gi * gi
    c = conf.clamp(0.0, 1.0)
    half = half_min + (1.0 - c) * (half_max - half_min)
    half = torch.where(is_spike, half_max.expand_as(half), half)

    lo_b = lo.view(B, 1, 1)
    hi_b = hi.view(B, 1, 1)
    # keep the whole window inside the guard range so clamping never collapses
    # several bins onto the same boundary value
    half = torch.minimum(half, 0.5 * (hi_b - lo_b) - 1e-4)
    center = center.clamp(min=lo_b + half + 1e-4, max=hi_b - half - 1e-4)

    steps = torch.linspace(-1.0, 1.0, num, device=prior.device, dtype=prior.dtype).view(1, num, 1, 1)
    local = center.unsqueeze(1) + half.unsqueeze(1) * steps

    # no-prior fallback: half-offset uniform grid across the guard range
    t = (torch.arange(num, device=prior.device, dtype=prior.dtype) + 0.5) / num
    fb = lo_b.unsqueeze(1) + (hi_b - lo_b).unsqueeze(1) * t.view(1, num, 1, 1)
    usable = valid | has_nbr
    # 硬门: 可靠度低于 gate_hard_conf 时整条 local 分支退化成 guard 范围内的
    # 半偏移网格, 而不是"围绕错误 prior 的宽窗口"。
    # 实测依据 (experiments/out/coloc_val_*.log): stage1 尾巴里 local 获胜的
    # 像素中 91.5% 的真值根本不在 local 跨度内, 而 global 分支 100% 含有
    # <20mm 的候选。只放宽宽度救不了 —— prior 差 500mm 时放宽十几毫米无意义。
    if gate is not None:
        usable = usable & (gate > 0.5)
    local = torch.where(usable.unsqueeze(1), local, fb.expand_as(local))
    l_lo = local.amin(dim=1)
    l_hi = local.amax(dim=1)
    # usable == False 的像素上这 num 个 bin 其实是 guard 范围的加密网格, 不是
    # 先验候选。调用方据此把它们当 global 处理 —— 否则它们会同时背上
    # is_local=1、log(q≈0.02) 的压制和 local aux loss, 白白浪费掉。
    return local, l_lo, l_hi, usable


def build_stage1_hypotheses(
    depth_prior: torch.Tensor,
    confidence: torch.Tensor,
    depth_min: torch.Tensor,
    depth_max: torch.Tensor,
    config: DepthRangeConfig,
    target_hw: tuple[int, int],
    prior_valid: torch.Tensor | None = None,
) -> Stage1Hypotheses:
    """Dual-branch stage-1 axis: prior-independent global guard + prior-guided
    dense local bins, merged and sorted, with branch identity preserved."""
    with torch.no_grad():
        depth_min = depth_min.float()
        depth_max = depth_max.float()
        prior = _resize_map(depth_prior.float(), target_hw)
        conf = _resize_map(confidence.float(), target_hw)

        valid = torch.isfinite(prior) & (prior > 0) & torch.isfinite(conf) & (conf >= 0)
        conf = torch.where(valid, conf.clamp(0.0, 1.0), torch.zeros_like(conf))
        # 样本级标尺校验 (data/dtu.py 的 prior_valid): 标尺失败的先验整批作废
        if prior_valid is not None:
            pv = prior_valid.float().view(-1, 1, 1) > 0.5
            valid = valid & pv
            conf = torch.where(pv, conf, torch.zeros_like(conf))

        lo, hi = _robust_global_bounds(prior, valid, depth_min, depth_max, config)
        Dg, Dl = config.num_global, config.num_local
        global_interval = (hi - lo) / max(Dg - 1, 1)  # [B]

        g_bins = _global_branch(lo, hi, Dg, target_hw, config.inverse_depth_global)
        gate = (conf >= config.gate_hard_conf) if config.gate_local_branch else None
        l_bins, l_lo, l_hi, branch_active = _local_branch(
            prior, conf, valid, lo, hi, global_interval, Dl, config, gate=gate)

        B, _, H, W = g_bins.shape
        hypos = torch.cat([g_bins, l_bins], dim=1)
        branch = torch.cat(
            [hypos.new_zeros(B, Dg, H, W), hypos.new_ones(B, Dl, H, W)], dim=1
        )
        hypos, order = hypos.sort(dim=1)
        is_local = branch.gather(1, order)

        # positions of each branch on the sorted axis (stable sort keeps them in
        # ascending depth order because the merged axis is already sorted)
        _, pos = is_local.to(torch.int8).sort(dim=1, stable=True)
        global_idx = pos[:, :Dg].contiguous()
        local_idx = pos[:, Dg:].contiguous()

        # per-bin spacing (h_{i+1} - h_{i-1}) / 2 with edge replication
        d_next = torch.cat([hypos[:, 1:], hypos[:, -1:]], dim=1)
        d_prev = torch.cat([hypos[:, :1], hypos[:, :-1]], dim=1)
        interval = 0.5 * (d_next - d_prev)
        interval[:, 0] = hypos[:, 1] - hypos[:, 0]
        interval[:, -1] = hypos[:, -1] - hypos[:, -2]
        interval = interval.clamp_min(1e-4)

        edge = edge_map_from_prior(prior, valid, config.edge_grad_rel)

    return Stage1Hypotheses(
        hypos=hypos,
        is_local=is_local,
        interval=interval,
        global_idx=global_idx,
        local_idx=local_idx,
        global_lo=lo,
        global_hi=hi,
        local_lo=l_lo,
        local_hi=l_hi,
        prior=prior,
        conf=conf,
        edge=edge,
        branch_active=branch_active,
    )


def mode_centered_regression(
    prob: torch.Tensor,
    depth_hypos: torch.Tensor,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expectation restricted to +-window bins around the argmax mode.

    A global soft-argmin over a bimodal posterior (wrong local peak + correct
    global peak) lands between the peaks, on no surface at all; restricting the
    expectation to the winning mode keeps the estimate on a real candidate.
    Returns (depth, sigma_within_mode, argmax_idx [B,1,H,W]).

    The window is *shifted* to stay in bounds rather than clamped. Clamping
    repeats the edge index, so a mode at bin 0 of a 4-bin axis would gather bin 0
    three times and drag the expectation onto the boundary. The fine stages of
    the 4-level cascade have as few as 4 hypotheses, where that bias is severe.
    """
    D = prob.shape[1]
    idx = prob.argmax(dim=1, keepdim=True)
    w = min(2 * window + 1, D)                       # window never exceeds the axis
    start = (idx - (w // 2)).clamp(0, D - w)         # slide, do not clamp indices
    offs = torch.arange(w, device=prob.device).view(1, -1, 1, 1)
    nbr = start + offs
    p = prob.gather(1, nbr)
    h = depth_hypos.gather(1, nbr)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-8)
    depth = (p * h).sum(dim=1)
    var = (p * (h - depth.unsqueeze(1)) ** 2).sum(dim=1)
    sigma = var.clamp_min(1e-12).sqrt()
    return depth, sigma, idx


def refine_range_from_posterior(
    center: torch.Tensor,
    winner_interval: torch.Tensor,
    prob: torch.Tensor,
    edge: torch.Tensor,
    config: DepthRangeConfig,
    num_depths: int,
    global_interval: torch.Tensor,
    depth_min: torch.Tensor,
    depth_max: torch.Tensor,
    stage_idx: int = 0,
) -> torch.Tensor:
    """Next-stage hypotheses sized by the *winning candidate's* sampling
    precision: a local winner shrinks the search, a global winner keeps a
    correction-sized range, and entropy / edge uncertainty widen it.

    ``stage_idx`` selects the per-stage ``range_k`` / ``range_min_gi`` (0 for the
    stage-1 -> stage-2 step, 1 for 2 -> 3, 2 for 3 -> 4).

    All geometry is detached: hypothesis placement carries no gradient, each
    stage is trained by its own losses.

    Returns ``(hypos, stats)``; ``stats`` records which of the three width terms
    actually决定了窗宽 (见函数末尾的注释)。
    """
    with torch.no_grad():
        D = prob.shape[1]
        p = prob.float().clamp_min(1e-8)
        entropy = -(p * p.log()).sum(dim=1) / float(torch.log(torch.tensor(float(D))))
        # range_k / range_min_gi are per-stage: a shared value makes the cascade
        # coarsen as the hypothesis count drops (see DepthRangeConfig).
        half = config.range_k[stage_idx] * winner_interval * (
            1.0 + config.range_entropy_a * entropy + config.range_edge_b * edge
        )
        gi = global_interval.view(-1, 1, 1)
        # 深度自适应: 三角测量的深度分辨率 ~ d^2/(f*B), 而窗宽是常数毫米的, 于是
        # 覆盖率随深度单调下降。scale 只放宽远端 (近端恒为 1), 上限由
        # depth_adaptive_max 封住, 避免远端窗口大到把分辨率全吃掉。
        # 参考深度用 depth_min —— 推理时可得, 不依赖 GT。
        ascale = 1.0
        if (config.depth_adaptive_range
                and config.depth_adaptive_stages[stage_idx]):
            dref = depth_min.view(-1, 1, 1).clamp_min(1e-3)
            ascale = ((center.detach() / dref) ** 2).clamp(1.0, config.depth_adaptive_max)
            if str(config.depth_adaptive_apply).lower() == "both":
                half = half * ascale
        half_raw = half
        floor = config.range_min_gi[stage_idx] * gi * ascale
        ceil = config.range_max_gi * gi
        half = torch.maximum(half_raw, floor)
        half = torch.minimum(half, ceil)
        steps = torch.linspace(-1.0, 1.0, num_depths, device=center.device, dtype=center.dtype)
        hypos = center.detach().unsqueeze(1) + half.unsqueeze(1) * steps.view(1, num_depths, 1, 1)
        clamped = (hypos < depth_min.view(-1, 1, 1, 1)) | (hypos > depth_max.view(-1, 1, 1, 1))
        hypos = hypos.clamp(
            min=depth_min.view(-1, 1, 1, 1),
            max=depth_max.view(-1, 1, 1, 1),
        )
        # 窗宽归因诊断: "stage2-4 窗口缩了 21%" 到底是 floor 主导还是
        # range_k*winner_interval 主导, 只有这三个比例能定。half_raw 低于 floor
        # 的比例高 = floor 说了算 = 窗宽由 global_interval 决定 (于是改 num_global
        # 会整体缩放级联); 反之则由 winner_interval 决定。
        stats = {
            # 逐像素的 half_raw / floor / center 交给 loss 侧按 GT 深度分桶 ——
            # 这里没有 GT, 而"窗口该开多宽"只有对着 GT 才能回答。
            "maps": {"half_raw": half_raw, "half": half,
                     "floor": floor.expand_as(half_raw) if floor.shape != half_raw.shape else floor,
                     "center": center.detach()},
            "half_raw_mm": float(half_raw.mean()),
            "half_mm": float(half.mean()),
            "floor_binding": float((half_raw < floor).float().mean()),
            "cap_binding": float((half_raw > ceil).float().mean()),
            "bound_clamp_frac": float(clamped.float().mean()),
            "wint_p10": float(winner_interval.float().quantile(0.10)),
            "wint_p50": float(winner_interval.float().quantile(0.50)),
            "wint_p90": float(winner_interval.float().quantile(0.90)),
            "adapt_scale": float(ascale.mean()) if torch.is_tensor(ascale) else 1.0,
        }
    return hypos, stats




def second_mode(
    prob: torch.Tensor,
    hypos: torch.Tensor,
    mode_idx: torch.Tensor,
    guard: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage-1 的次峰深度和它的概率质量。

    把 winner 及其 ``guard`` 个相邻 bin 屏蔽掉再取 argmax。返回
    ``(second_depth [B,H,W], second_mass [B,H,W])``。

    为什么需要它: 共位测试显示 stage1 的失败 100% 是选择失败 —— 轴上一直
    存在 <20mm 的候选 (中位 2.43mm), 只是没赢得 argmax。而 stage2-4 的候选
    完全围绕 winner 构造, 所以一旦 stage1 选错, 正确的模态就永久消失了。
    """
    B, D, H, W = prob.shape
    idx = torch.arange(D, device=prob.device).view(1, D, 1, 1)
    far = (idx - mode_idx).abs() > max(int(round(guard)), 1)
    masked = torch.where(far, prob, torch.zeros_like(prob))
    j = masked.argmax(dim=1, keepdim=True)
    depth = hypos.gather(1, j).squeeze(1)
    # 次峰**邻域**的质量, 而不是 winner 邻域之外的总质量: 平坦分布的"其余总质量"
    # 很容易超过任何阈值, 拿它当"存在第二个模态"的判据会一路假阳。
    g = max(int(round(guard)), 1)
    near_j = (idx - j).abs() <= g
    mass = torch.where(near_j & far, prob, torch.zeros_like(prob)).sum(dim=1)
    return depth, mass


def merge_dual_mode(
    hypos: torch.Tensor,
    second_depth: torch.Tensor,
    second_mass: torch.Tensor,
    winner_interval: torch.Tensor,
    cfg: DepthRangeConfig,
) -> torch.Tensor:
    """把 stage2 的候选拆成 winner-centered 一半 + 次峰 centered 一半。

    只在次峰概率质量 >= ``dual_mode_min_mass`` 的像素上启用; 其余像素保持
    原来的单峰候选, 所以深度分辨率不会为全图买单 (MonoMVSNet Table 3 的
    教训: 不门控的先验候选会让 Overall 从 0.288 退化到 0.292)。

    ``hypos`` [B,D,h,w] 升序; 返回同形状、同样升序的候选轴。
    """
    B, D, h, w = hypos.shape
    half = D // 2
    if half < 2:
        return hypos
    sd = _resize_map(second_depth, (h, w))
    sm = _resize_map(second_mass, (h, w))
    wi = _resize_map(winner_interval, (h, w)).clamp_min(1e-4)

    span = (hypos[:, -1] - hypos[:, 0]).clamp_min(1e-4)
    # winner 位于原轴中心, 所以 winner 侧必须取*中间* half 个 bin —— 取 [:half]
    # 会把 winner 上方的候选整段丢掉。
    lo_k = (hypos.shape[1] - half) // 2
    keep = hypos[:, lo_k:lo_k + half]
    # 次峰侧: 以 sd 为中心, 宽度沿用 winner 的窗口宽度的一半
    t = torch.linspace(-1.0, 1.0, D - half, device=hypos.device,
                       dtype=hypos.dtype).view(1, D - half, 1, 1)
    half_w = (0.5 * span / max(half - 1, 1) * (D - half - 1)).clamp_min(wi)
    alt = sd.unsqueeze(1) + half_w.unsqueeze(1) * t

    # alt 必须夹回物理范围, 否则次峰靠近边界时会采到负深度/超出场景的候选
    alt = alt.clamp(min=float(hypos.min()), max=float(hypos.max()))
    dual = torch.cat([keep, alt], dim=1).sort(dim=1).values
    use = (sm >= cfg.dual_mode_min_mass).unsqueeze(1)
    return torch.where(use, dual, hypos)


def apply_branch_prior(
    logits: torch.Tensor,
    global_idx: torch.Tensor,
    local_idx: torch.Tensor,
    q: torch.Tensor,
    branch_active: torch.Tensor,
    q_min: float,
    mode: str = "hard",
    beta: float = 1.0,
) -> torch.Tensor:
    """把 ``P(d) = q P(d|local) + (1-q) P(d|global)`` 正确地加到 stage-1 logits 上。

    直接给 local 候选加 ``log q``、给 global 加 ``log(1-q)`` 是错的: 两个分支的
    候选数不同 (40 vs 8), flat logits 下 local 的实际质量是
    ``Dl*q / (Dl*q + Dg*(1-q))`` —— q=0.5 时只有 16.7%, 要 q>0.833 才过半。
    正确做法是先在各分支内部归一化, 再乘分支先验:

        z_g = logits_g - logsumexp(logits_g) + log(1-q)
        z_l = logits_l - logsumexp(logits_l) + log(q)

    ``branch_active`` 为 False 的像素 (硬门触发, local bin 其实是 guard 网格)
    保持原始 48 路 softmax, 不做分支划分。

    ``q`` 必须带梯度, 否则 branch loss 只能训练 decoder 去抵消一个常数,
    传不回 SPRE。
    """
    qc = q.clamp(q_min, 1.0 - q_min).unsqueeze(1)                  # [B,1,H,W]
    lg = logits.gather(1, global_idx)
    ll = logits.gather(1, local_idx)
    if str(mode).lower() == "soft":
        # 计数校正的加性偏置, **不**做分支内归一化 —— 于是 cost volume 的跨分支
        # 证据被保留下来, 先验只是把天平推一下。flat logits 下这个偏置仍能还原出
        # (1-q, q) 的分支质量 (log Dg / log Dl 就是为此), 但有信息的 logits 可以
        # 推翻它。beta=0 等价于完全不加先验。
        Dg = float(global_idx.shape[1])
        Dl = float(local_idx.shape[1])
        bg = (beta * ((1.0 - qc).log() - math.log(Dg))).to(lg.dtype)
        bl = (beta * (qc.log() - math.log(Dl))).to(ll.dtype)
        zg, zl = lg + bg, ll + bl
    else:
        zg = lg - torch.logsumexp(lg.float(), dim=1, keepdim=True).to(lg.dtype) + (1.0 - qc).log().to(lg.dtype)
        zl = ll - torch.logsumexp(ll.float(), dim=1, keepdim=True).to(ll.dtype) + qc.log().to(ll.dtype)
    out = torch.empty_like(logits)
    out.scatter_(1, global_idx, zg)
    out.scatter_(1, local_idx, zl)
    return torch.where(branch_active.unsqueeze(1), out, logits)


# ===========================================================================
#  W1: 逆深度轴 / 非均匀轴记账 / 物理口径的第二峰
#  下面这一组是 refine_range_from_posterior 的替代路径, 旧函数原样保留 ——
#  它是 --axis-space legacy_depth 的实现, 也是实现校验时的对照。
# ===========================================================================


def local_intervals(hypos: torch.Tensor, min_interval: float = 1e-4) -> torch.Tensor:
    """逐 bin 的局部间距 (h_{i+1} - h_{i-1})/2, 端点用单边差分。

    ``hypos`` [B, D, H, W] 沿 dim=1 单调。返回同形状。

    ``min_interval`` 是**有量纲**的下限, 默认 1e-4 只对**深度(毫米)轴**成立。
    逆深度轴上典型间距是 1e-5 量级, 1e-4 会把整张图钉在同一个常数上 ——
    w_bar 恒等于 1e-4, 窗口宽出 3-20 倍, 而且完全静默。逆深度轴的调用方一律
    传 0.0, 退化由 ``robust_local_scale`` 的 ``> eps`` 过滤 + 回退 g_v 处理。

    为什么必须逐 bin: 现在 network.py 用 ``span / (D-1)`` 当下一级的
    ``winner_interval``, 也当 Huber 的 scale。深度均匀轴上这是对的, 逆深度
    均匀轴上就不对了 —— 远端 bin 在深度空间里宽得多。不改这里, 前面所有
    逆深度的工作都会被一个错误的归一化尺度吃掉。
    """
    d_next = torch.cat([hypos[:, 1:], hypos[:, -1:]], dim=1)
    d_prev = torch.cat([hypos[:, :1], hypos[:, :-1]], dim=1)
    interval = 0.5 * (d_next - d_prev)
    interval = interval.clone()
    if hypos.shape[1] >= 2:
        interval[:, 0] = hypos[:, 1] - hypos[:, 0]
        interval[:, -1] = hypos[:, -1] - hypos[:, -2]
    return interval.abs().clamp_min(min_interval)


def robust_local_scale(
    intervals: torch.Tensor,
    mode_idx: torch.Tensor,
    fallback: torch.Tensor,
    half: int = 2,
    eps: float = 1e-9,
    return_fallback: bool = False,
):
    """winner 邻域内正间距的中位数 —— 范围损失的归一化尺度。

    为什么不能直接用 ``intervals.gather(1, mode_idx)``: stage1 的 16 个 local
    bin 挤在先验附近, winner 落在两个近重复候选之间时局部间距会趋近 0, 而
    范围损失要除以它 —— 少量像素就能产生极大梯度。中位数对近重复稳健。

    ``fallback`` 广播到输出形状 (通常是 g_v), 邻域内没有有效正间距时用它。
    返回 [B, H, W]。
    """
    D = intervals.shape[1]
    offs = torch.arange(-half, half + 1, device=intervals.device).view(1, -1, 1, 1)
    idx = (mode_idx + offs).clamp(0, D - 1)
    w = intervals.gather(1, idx).float()
    w = torch.where(w > eps, w, torch.full_like(w, float("nan")))
    wbar = w.nanmedian(dim=1).values
    fb = fallback.expand_as(wbar)
    good = torch.isfinite(wbar) & (wbar > eps)
    out = torch.where(good, wbar, fb)
    if return_fallback:
        # 回退比例偏高 = stage1 的近重复候选比预想严重 (wbar_fallback_frac)
        return out, (~good).float().mean()
    return out


def to_inverse(depth: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """深度 -> 逆深度。深度升序的轴换过去就是逆深度降序, 调用方注意。"""
    return 1.0 / depth.clamp_min(eps)


def build_axis_inverse(
    v_c: torch.Tensor,
    h_lo: torch.Tensor,
    h_hi: torch.Tensor,
    num_depths: int,
    v_min: torch.Tensor,
    v_max: torch.Tensor,
) -> torch.Tensor:
    """在逆深度空间均匀布点, 转回深度并**按深度升序**返回。

    左右可以不等宽 (控制器输出非对称半宽)。夹持由调用方在 v 空间用整体缩放 +
    平移完成 (range_controller 里做), 这里只做最后一道安全 clamp。
    """
    t = torch.linspace(-1.0, 1.0, num_depths, device=v_c.device, dtype=torch.float32)
    t = t.view(1, num_depths, 1, 1)
    h = torch.where(t < 0, h_lo.unsqueeze(1), h_hi.unsqueeze(1))
    v = v_c.unsqueeze(1) + h * t
    v = v.clamp(min=v_min.unsqueeze(1), max=v_max.unsqueeze(1))
    d = 1.0 / v.clamp_min(1e-8)
    return d.flip(dims=[1])                     # v 升序 -> 深度降序 -> 翻转成升序


def blend_axes(d_legacy: torch.Tensor, d_inverse: torch.Tensor, lam: float) -> torch.Tensor:
    """legacy(深度均匀) -> inverse(逆深度均匀) 的凸组合迁移。

    零初始化控制器只保证 "控制器不动手", 挡不住 "深度均匀 -> 逆深度均匀" 这个
    坐标变化 —— 即使端点相同, 中间候选也必然不同。凸组合让 step 0 的候选轴
    **真正等于** f10, 再在 warm-up 期内平滑迁移。两条轴都按深度升序, 凸组合
    仍然升序。
    """
    if lam >= 1.0:
        return d_inverse
    if lam <= 0.0:
        return d_legacy
    return (1.0 - lam) * d_legacy + lam * d_inverse


def second_mode_physical(
    prob: torch.Tensor,
    v_axis: torch.Tensor,
    mode_idx: torch.Tensor,
    h_sep: torch.Tensor,
    h_mass: torch.Tensor,
    sep_k: float = 2.0,
):
    """物理口径的 **deployable** 第二峰。

    与旧 ``second_mode`` 的三点区别:
      1. 分离性用 |v_j - v_m| 的**物理逆深度距离**, 不用 bin 索引距离 ——
         stage1 是 global+local 混合轴, 相邻 local bin 可能只差零点几毫米。
      2. 返回的是**峰邻域质量**, 不是 winner 邻域之外的总质量 (平坦后验一路假阳)。
      3. 选峰用 "winner 之外、物理分离的局部极大里概率最高的那个", 不用 GT ——
         这才是部署时拿得到的量。离线诊断脚本里另有一个用 GT 的 oracle 口径,
         两者的差距就是 "峰存在但选不出来" 的部分。

    返回 ``(v_second, mass, dv_peak, found, j2)``, 前四个各 [B, H, W],
    ``j2`` 是 [B, 1, H, W] 的峰索引 (给 "回归窗口是否跨两个表面" 的诊断用)。
    """
    p_prev = torch.cat([prob[:, :1], prob[:, :-1]], dim=1)
    p_next = torch.cat([prob[:, 1:], prob[:, -1:]], dim=1)
    is_peak = (prob >= p_prev) & (prob >= p_next)

    v_m = v_axis.gather(1, mode_idx)                                  # [B,1,H,W]
    sep = (v_axis - v_m).abs() > (sep_k * h_sep).unsqueeze(1)
    cand = is_peak & sep
    score = torch.where(cand, prob.float(), torch.full_like(prob, -1.0, dtype=torch.float32))
    j2 = score.argmax(dim=1, keepdim=True)
    found = score.gather(1, j2) >= 0.0

    v2 = v_axis.gather(1, j2)
    near = (v_axis - v2).abs() <= h_mass.unsqueeze(1)
    mass = torch.where(near, prob.float(), torch.zeros_like(prob, dtype=torch.float32)).sum(1, keepdim=True)
    dv2 = (v2 - v_m).abs()

    zero = torch.zeros_like(mass)
    return (torch.where(found, v2, v_m).squeeze(1),
            torch.where(found, mass, zero).squeeze(1),
            torch.where(found, dv2, zero).squeeze(1),
            found.squeeze(1),
            j2)
