from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.depth_range import mode_centered_regression
from models.probe import Probe


class ConvGN3d(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(min(8, out_c), out_c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class UpConv3d(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2, bias=False)
        self.norm = nn.GroupNorm(min(8, out_c), out_c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.up(x)))


class CostVolumeUNet(nn.Module):
    def __init__(self, in_channels: int = 8, base: int = 16, depth: int = 3) -> None:
        super().__init__()
        chs = [base * (2 ** i) for i in range(depth + 1)]
        self.input = ConvGN3d(in_channels, chs[0])
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(nn.Sequential(
                ConvGN3d(chs[i], chs[i + 1], stride=2),
                ConvGN3d(chs[i + 1], chs[i + 1]),
            ))
        self.ups = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.ups.append(nn.ModuleList([
                UpConv3d(chs[i], chs[i - 1]),
                ConvGN3d(chs[i - 1] * 2, chs[i - 1]),
            ]))
        self.head = nn.Conv3d(chs[0], 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.input(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        feat = skips[-1]
        for i, (up, smooth) in enumerate(self.ups):
            feat = up(feat)
            skip = skips[-2 - i]
            if feat.shape != skip.shape:
                feat = F.interpolate(feat, size=skip.shape[2:], mode="trilinear", align_corners=False)
            feat = smooth(torch.cat([feat, skip], dim=1))
        return self.head(feat).squeeze(1)


def _posterior_sigma(prob: torch.Tensor, hypos: torch.Tensor) -> torch.Tensor:
    """整条轴上的后验标准差。

    MAP 模式下不再有 mode-centered 的 sigma, 但这个量仍然是有用的**特征**
    (残差头和融合置信度头都要它), 只是不再当深度估计用。
    """
    p = prob.float()
    h = hypos.float()
    mu = (p * h).sum(dim=1, keepdim=True)
    var = (p * (h - mu) ** 2).sum(dim=1)
    return var.clamp_min(1e-12).sqrt()


class DepthDecoder(nn.Module):
    """``head_mode``:

    ``expect`` —— 现有行为: 在 argmax 附近 +-mode_window 个 bin 上做期望。
    ``map``    —— 硬 MAP 选面, 深度 = argmax 对应的候选, 亚 bin 修正交给
                  外面的逐候选残差头 (models/range_controller.Stage4ResidualHead)。

    为什么 stage4 要 ``map``: ``mode_centered_regression`` 取
    ``w = min(2*mode_window+1, D)``, 而 mode_window=2、num_depths_stage4=4 ->
    w = 4 = **全轴**。stage4 的深度实际上是一次普通 soft-argmax; 窗口跨度
    (f15 下 4.9mm) 在遮挡边界上常常同时含两个表面, 输出就落在两者之间。
    """

    def __init__(self, in_channels: int = 8, base: int = 16, depth: int = 3,
                 mode_window: int = 2, head_mode: str = "expect") -> None:
        super().__init__()
        self.unet = CostVolumeUNet(in_channels=in_channels, base=base, depth=depth)
        self.mode_window = mode_window
        self.head_mode = str(head_mode).lower()
        if self.head_mode not in ("expect", "map"):
            raise ValueError(f"head_mode 只能是 expect / map, 收到 {head_mode!r}")
        self.tag = "stage?"          # network.py 构造后写入, 只给探针用

    def forward(
        self,
        cost_volume: torch.Tensor,
        depth_hypos: torch.Tensor,
        branch_prior=None,
    ) -> tuple[torch.Tensor, ...]:
        Probe.log(self.tag, "unet_in", cost_volume)
        logits_raw = self.unet(cost_volume)
        Probe.log(self.tag, "logits_raw", logits_raw)
        # 分支先验: P(d) = q P(d|local) + (1-q) P(d|global)。必须在各分支内部
        # 先归一化再乘先验 —— 见 depth_range.apply_branch_prior 的推导。
        # logits_raw 是加先验之前的纯 matching evidence, 原样返回给 loss 做
        # "加先验前/后哪个 argmax 更接近 GT" 的诊断。分支先验会把两个分支的总
        # 概率质量硬指派成 (1-q, q), cost volume 学到的跨分支证据被整段抹掉,
        # 这个诊断是判断它有没有害的唯一直接证据。
        logits = branch_prior(logits_raw) if branch_prior is not None else logits_raw
        # max-shift keeps softmax finite under AMP; log_softmax downstream is
        # shift-invariant so the loss sees the same distribution.
        Probe.log(self.tag, "logits_bp", logits)
        logits = logits - logits.amax(dim=1, keepdim=True).detach()
        prob = F.softmax(logits.float(), dim=1)
        Probe.log(self.tag, "prob", prob)
        # Mode-centered regression instead of a global soft-argmin: over a
        # bimodal posterior (wrong local peak + correct global peak) the global
        # expectation lands between the peaks, on no real surface.
        if self.head_mode == "map":
            # 硬选面: 不做任何跨 bin 的期望, 所以永远不会落在两个表面之间。
            # argmax 不可导, 深度对后验没有梯度 —— 这是有意的: CE 训练选择,
            # 逐候选残差训练亚 bin 修正, 两条监督互不污染。
            mode_idx = prob.argmax(dim=1, keepdim=True)
            depth = depth_hypos.float().gather(1, mode_idx).squeeze(1)
            sigma = _posterior_sigma(prob, depth_hypos)
        else:
            depth, sigma, mode_idx = mode_centered_regression(
                prob, depth_hypos.float(), self.mode_window)
        Probe.log(self.tag, "depth", depth)
        return depth, sigma, prob, logits, mode_idx, logits_raw
