from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F




class ConvBnReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_bn_relu(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.convbn1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.convbn2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.convbn1(x)
        out = self.convbn2(out)
        return F.relu(out + x, inplace=True)


class ScaleTower(nn.Module):
    """自底向上单个尺度塔：stride-2 降采样 + 一层同尺度 conv + 残差块。

    每尺度共 4 个 3x3 卷积层（2 个 ConvBnReLU + 1 个残差块），比原来的
    (stride conv + 残差) 更深，扩大感受野；下探一级（多一次 stride-2）
    对感受野的贡献是翻倍级的。
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.down = ConvBnReLU(in_ch, out_ch, stride=2)   # 降采样并换通道
        self.conv = ConvBnReLU(out_ch, out_ch)            # 同尺度加深
        self.res = ResidualBlock(out_ch)                  # 同尺度残差增强

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.conv(self.down(x)))


class FPNFeatureExtractor(nn.Module):
    """从头训练的特征塔 + FPN，产出 1/8、1/4、1/2、全分辨率四级特征。

    自底向上：原图 -> 1/2 塔 -> 1/4 塔 -> 1/8 塔（每塔第 1 层 stride=2 降采样，
    同尺度再叠加 conv + 残差块加深）。下探到 1/8 主要是为了扩感受野——1/8
    分辨率的特征图很便宜（显存/参数增量都很小），四级级联把最粗的那一级放在
    这里：绝大多数深度假设在 1/8 上评估，代价只有 stride-1 的 1/64。

    自顶向下：p8(1/8) -> 上采样加到 p4(1/4) -> 加到 p2(1/2) -> 加到 p1(全分辨率)。
    全分辨率分支用 2 层 Conv-GN-ReLU-Conv 而不是单层线性投影，至少经过一次
    非线性组合来补高频细节（精细 stage 的判别力主力仍来自 top-down 语义）。
    """

    def __init__(
        self,
        out_channels: int = 128,
        base_channel: int = 32,
        stage1_head: str = "smooth",
    ) -> None:
        """``stage1_head``: 1/8 那一级普通 FPN 输出的头。

        * ``smooth`` —— 3x3 conv (原有行为, 构造顺序与改动前逐位一致)。
        * ``out0``   —— MVSFormer++ ``FPNDecoder.out0``: 1x1 conv + BN + SiLU。完整
          SVA 用它: 第二段注意力 (FMT) 的输入就是这一级的 out0 输出, 而普通 top-down
          仍从 out0 **之前**的 p8 出发 (他们的 ``intra_feat``)。
        """
        super().__init__()
        if stage1_head not in ("smooth", "out0"):
            raise ValueError(f"stage1_head 只能是 smooth / out0, 收到 {stage1_head!r}")
        self.stage1_head = stage1_head
        c_half = base_channel * 2      # 1/2 尺度内部通道
        c_quarter = base_channel * 4   # 1/4 尺度内部通道
        c_eighth = base_channel * 8    # 1/8 尺度内部通道

        # ---- 自底向上：三级特征塔 (1/2, 1/4, 1/8) ----
        self.tower_half = ScaleTower(3, c_half)             # 原图 -> 1/2
        self.tower_quarter = ScaleTower(c_half, c_quarter)  # 1/2 -> 1/4
        self.tower_eighth = ScaleTower(c_quarter, c_eighth)  # 1/4 -> 1/8

        # ---- FPN 横向连接：统一到 out_channels 才能逐元素相加 ----
        self.lateral_half = nn.Conv2d(c_half, out_channels, 1)
        self.lateral_quarter = nn.Conv2d(c_quarter, out_channels, 1)
        self.lateral_eighth = nn.Conv2d(c_eighth, out_channels, 1)

        # ---- 全分辨率分支：2 层 conv（Conv-BN-ReLU-Conv），非单层线性投影 ----
        self.input_proj = nn.Sequential(
            ConvBnReLU(3, out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        # ---- 输出平滑：消除上采样混叠 ----
        # out0 占 smooth_p8 的构造位置: 不用的那个干脆不建 (建了不前向 = 要梯度
        # 却收不到), smooth 模式下构造顺序与改动前完全相同。
        if stage1_head == "out0":
            self.out0 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1),
                                      nn.BatchNorm2d(out_channels), nn.SiLU())
        else:
            self.smooth_p8 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, dino: torch.Tensor | None = None) -> dict[int, torch.Tensor]:
        """``dino``: 可选的 [B, out_channels, H/8, W/8] 语义特征, 注入 1/8 瓶颈。

        注入点选在 top-down 之前而不是之后 —— 这样 DINO 的语义会沿 p8->p4->p2->p1
        传遍所有尺度, 和 MVSFormer++ 的 ``conv31 = conv31 + vit_feat`` 位置一致
        (他们也是加在 encoder 的 1/8 输出上, 再进 decoder)。

        返回的是**普通** FPN 的四级输出。完整 SVA 的第二段 (FMT + 第二条逐级融合
        路径) 不在这里, 在 ``models/sva.SVAPathway``, 由网络接在本函数之后 ——
        与 MVSFormer++ ``FPNDecoder`` -> ``FMT_with_pathway`` 的分工相同。
        """
        # 自底向上
        f_half = self.tower_half(x)          # [B, c_half, H/2, W/2]
        f_quarter = self.tower_quarter(f_half)  # [B, c_quarter, H/4, W/4]
        f_eighth = self.tower_eighth(f_quarter)  # [B, c_eighth, H/8, W/8]

        # 自顶向下：从 1/8 开始逐级上采样相加
        p8 = self.lateral_eighth(f_eighth)      # [B, out_channels, H/8, W/8]
        if dino is not None:
            if dino.shape[-2:] != p8.shape[-2:]:
                dino = F.interpolate(dino, size=p8.shape[-2:], mode="bilinear", align_corners=False)
            p8 = p8 + dino
        p4 = self.lateral_quarter(f_quarter) + F.interpolate(
            p8, size=f_quarter.shape[-2:], mode="bilinear", align_corners=False
        )                                        # [B, out_channels, H/4, W/4]
        p2 = self.lateral_half(f_half) + F.interpolate(
            p4, size=f_half.shape[-2:], mode="bilinear", align_corners=False
        )                                        # [B, out_channels, H/2, W/2]
        p1 = self.input_proj(x) + F.interpolate(
            p2, size=x.shape[-2:], mode="bilinear", align_corners=False
        )                                        # [B, out_channels, H, W]

        return {
            8: self.out0(p8) if self.stage1_head == "out0" else self.smooth_p8(p8),
            4: self.smooth_p4(p4),
            2: self.smooth_p2(p2),
            1: self.smooth_p1(p1),
        }


class MultiViewFPN(nn.Module):
    def __init__(
        self,
        out_channels: int = 128,
        base_channel: int = 32,
        stage1_head: str = "smooth",
    ) -> None:
        super().__init__()
        self.fpn = FPNFeatureExtractor(
            out_channels=out_channels,
            base_channel=base_channel,
            stage1_head=stage1_head,
        )
        self.out_channels = out_channels

    def forward(self, imgs: torch.Tensor, dino: torch.Tensor | None = None) -> dict[int, torch.Tensor]:
        """``dino``: 可选 [B, V, out_channels, h8, w8], 每个视角一份。"""
        B, V, C, H, W = imgs.shape
        d = dino.reshape(B * V, *dino.shape[2:]) if dino is not None else None
        feats = self.fpn(imgs.view(B * V, C, H, W), dino=d)
        return {s: f.view(B, V, f.shape[1], f.shape[2], f.shape[3]) for s, f in feats.items()}
