"""MVSFormer++ 完整 SVA 的后半段: deconv 上采样 + FMT (1/8 尺度的 normalized 2D-PE
与 self/cross 注意力) + 第二条逐级融合路径。前半段 (DINOv3 token 上的 SVA) 在
``models/spre.py`` 的 ``SVAFusion``。

数据流与 MVSFormer++ (``DINOv2MVSNet.forward``) 一一对应:

    DINO 多层 token --SVAFusion (self: ref / cross: src)--> SVADeconv (proj + 2x deconv)
        --> 加到 FPN 的 1/8 特征上                         (conv31 = conv31 + vit_feat)
    普通 FPN (models/fpn.py, stage1_head="out0"):
        1/8 输出 = out0(p8) = 1x1 conv + BN + SiLU          (FPNDecoder.out0)
        普通 top-down 从 out0 **之前**的 p8 出发, 产出 1/4、1/2、1/1   (intra_feat 链)
    SVAPathway (FMT_with_pathway):
        s8 = FMT(1/8 输出)       FMT = normalized 2D-PE + (self, cross, self, cross)
        s4 = smooth(up(reduce(s8)) + 普通 1/4 输出)
        s2 = smooth(up(reduce(s4)) + 普通 1/2 输出)
        s1 = smooth(up(reduce(s2)) + 普通 1/1 输出)
    四级 cost volume 用的是 (s8, s4, s2, s1)。

所以细尺度拿到的是**两条**路径的和: 不经过注意力的普通 FPN top-down, 加上从 FMT
输出逐级传下来的第二条路径。2026-09-18 的第一版 (cdaa5d5) 把 FMT 直接插在唯一的
top-down 之前改 p8, 既没有 out0, 也没有第二条路径 —— 那是另一个结构, 由
``SVA_LAYOUT`` 进 fingerprint 区分开。

代码对应关系 (reference/MVSFormerPlusPlus):
  * ``SVADeconv``      <- ``CrossVITDecoder.proj / upsampler0 / upsampler1``
  * ``NormalizedPE2D`` <- ``position_encoding.PositionEncodingSineNorm``
  * ``SVABlock``       <- ``dino/layers/block.CrossBlock`` (pre-norm, LayerScale,
                          ``pre_norm_query=False``: key/value 与 query 共用 norm1)
  * ``_linear_attn``   <- ``dino/layers/attention.CrossLinearAttention``
  * ``HighResSVA``     <- ``FMT.FMT`` (``FMT_config.layer_names = self,cross,self,cross``)
  * ``SVAPathway``     <- ``FMT.FMT_with_pathway``

与 MVSFormer++ 的差别 (都是适配, 不是改设计):
  * 宽度: 它们的四级是 64/32/16/8 通道, pathway 的 1x1 顺带逐级减半; 这里 FPN
    四级都是 128 通道 (cost volume 的 warp 投影在各级 builder 里做), 所以
    reduce 是 128->128 的 1x1。FMT 的 d_model=128, 头数沿用 nhead=4。
  * 普通 FPN 的 1/4、1/2、1/1 输出头仍是本仓库 FPN 自己的 3x3 conv, 不是它们
    FPNDecoder 的 3x3 + BN + SiLU —— 那属于 backbone, 不属于 SVA。
  * 线性注意力的 key/value 长度可以与 query 不同。cross 把 V-1 个 source 视角
    **拼在 token 维上**一次查 reference —— 注意力对每个 query 独立, 与逐视角循环
    逐位同义; pathway 的卷积按样本独立, 把视角并进 batch 维同样逐位同义。
  * 注意力的数值部分固定 fp32 并关掉 autocast。线性注意力要对全部 key 求和
    (0.8 整幅推理时 1/8 有 120x160 = 19200 个 token), fp16 会溢出 (job 415038
    就是 CVPE 的同一个问题: 22 个 scan 全 nan)。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 进 fingerprint 的结构标签。实现的数据流一变就改它, 旧 checkpoint 会在 test 端
# 被明确拒绝, 而不是 load 失败时只报一串 missing/unexpected key。
SVA_LAYOUT = "fpn_out0+fmt_pathway"


class SVADeconv(nn.Module):
    """SVA 输出 (patch 网格) -> 1/8 尺度的 FPN 宽度特征。

    ``CrossVITDecoder`` 的 proj(3x3) + 两层 ConvTranspose2d(4, s2, p1), 每层
    BN + SiLU。x4 上采样: 它们的 1/32 token 网格 -> 1/8; 这里 DINOv3 patch16 +
    max_side 512 下 token 网格约为图像的 1/28 (训练最大尺度 640x896), x4 后约 1/7,
    最后由调用方双线性对齐到 FPN 的 p8 尺寸 —— 与它们 ``F.interpolate(vit_feat,
    size=conv31)`` 同一处理。
    """

    def __init__(self, in_dim: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_dim, out_ch * 4, 3, padding=1),
                                  nn.BatchNorm2d(out_ch * 4), nn.SiLU())
        self.up0 = nn.Sequential(nn.ConvTranspose2d(out_ch * 4, out_ch * 2, 4, stride=2, padding=1),
                                 nn.BatchNorm2d(out_ch * 2), nn.SiLU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(out_ch * 2, out_ch, 4, stride=2, padding=1),
                                 nn.BatchNorm2d(out_ch), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up1(self.up0(self.proj(x)))


class NormalizedPE2D(nn.Module):
    """``PositionEncodingSineNorm``: 正弦 2D 位置编码, 坐标先线性归一化到
    ``max_shape`` —— 无论特征图多大, 行/列坐标的最大值都是 128。

    这是它们高分辨率泛化的关键 (Tab. 7/11): 训练在 1/8 的 64x80 上, 推理在
    120x160 上, 不归一化的话推理时一半的位置都是训练没见过的频率相位。

    编码是确定性的, 不是参数, 按 (h, w, device, dtype) 缓存, 不进 state_dict。
    """

    def __init__(self, d_model: int, max_shape: tuple[int, int] = (128, 128)) -> None:
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"NormalizedPE2D 需要 d_model 能被 4 整除, 收到 {d_model}")
        self.d_model = int(d_model)
        self.max_shape = (float(max_shape[0]), float(max_shape[1]))
        self._cache: dict = {}

    def encoding(self, h: int, w: int, device, dtype) -> torch.Tensor:
        key = (h, w, str(device), dtype)
        pe = self._cache.get(key)
        if pe is None:
            c = self.d_model
            # ones.cumsum(0) = 1..H —— 与原实现一样从 1 开始
            y = torch.arange(1, h + 1, dtype=torch.float32).view(h, 1).expand(h, w) * (self.max_shape[0] / h)
            x = torch.arange(1, w + 1, dtype=torch.float32).view(1, w).expand(h, w) * (self.max_shape[1] / w)
            div = torch.exp(torch.arange(0, c // 2, 2, dtype=torch.float32)
                            * (-math.log(10000.0) / (c // 2))).view(-1, 1, 1)
            pe = torch.zeros(c, h, w, dtype=torch.float32)
            pe[0::4] = torch.sin(x.unsqueeze(0) * div)
            pe[1::4] = torch.cos(x.unsqueeze(0) * div)
            pe[2::4] = torch.sin(y.unsqueeze(0) * div)
            pe[3::4] = torch.cos(y.unsqueeze(0) * div)
            pe = pe.unsqueeze(0).to(device=device, dtype=dtype)
            self._cache[key] = pe
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, C, H, W]"""
        return x + self.encoding(x.shape[-2], x.shape[-1], x.device, x.dtype)


def _linear_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int,
                 eps: float = 1e-6) -> torch.Tensor:
    """``CrossLinearAttention`` 的数值部分: phi(x) = elu(x) + 1。

    q: [B, N, C], k/v: [B, M, C] (M 可以 != N)。fp32 且关掉 autocast。
    """
    B, N, C = q.shape
    M = k.shape[1]
    d = C // heads
    with torch.autocast(device_type=q.device.type, enabled=False):
        q = F.elu(q.float().view(B, N, heads, d)) + 1.0
        k = F.elu(k.float().view(B, M, heads, d)) + 1.0
        v = v.float().view(B, M, heads, d)
        kv = torch.einsum("bmhd,bmhe->bhde", k, v)                      # [B,h,d,d]
        z = 1.0 / (torch.einsum("bnhd,bhd->bnh", q, k.sum(dim=1)) + eps)
        out = torch.einsum("bnhd,bhde,bnh->bnhe", q, kv, z)
    return out.reshape(B, N, C)


class SVABlock(nn.Module):
    """``CrossBlock`` (pre-norm + LayerScale init 1.0, qkv 无 bias)。

    self-attention 时 ``kv=None``; cross 时 key/value 也过 ``norm1`` ——
    FMT 配置 ``pre_norm_query=False`` 的行为。
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0,
                 init_values: float = 1.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} 不能被 heads={heads} 整除")
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.ls1 = nn.Parameter(init_values * torch.ones(dim))
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.ls2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor, kv: torch.Tensor | None = None) -> torch.Tensor:
        xn = self.norm1(x)
        kvn = xn if kv is None else self.norm1(kv)
        a = _linear_attn(self.q_proj(xn), self.k_proj(kvn), self.v_proj(kvn), self.heads)
        x = x + self.ls1 * self.proj(a.to(xn.dtype))
        return x + self.ls2 * self.mlp(self.norm2(x))


class HighResSVA(nn.Module):
    """1/8 尺度上的第二段 SVA —— ``FMT`` 的移植。

    layer_names = (self, cross, self, cross), 各层权重在 reference 与 source 之间
    **共享** (FMT 里只有一个 ``self.layers``):

        reference:  PE -> self_0 -> r1 -> self_2 -> r2           (只走 self 层)
        source:     PE -> self_0 -> cross_1(ref=r1) -> self_2 -> cross_3(ref=r2)

    输出 view 0 = r2 (最后一个 reference 状态), 其余 = 各 source 的最终状态。
    与原实现一样, PE 进入残差流, 所以输出特征里含有 PE —— 这是它们的设计。
    """

    def __init__(self, dim: int, heads: int = 4, layer_names: tuple[str, ...] = ("self", "cross", "self", "cross"),
                 mlp_ratio: float = 4.0, pe_max_shape: tuple[int, int] = (128, 128)) -> None:
        super().__init__()
        names = tuple(str(n).strip() for n in layer_names)
        # FMT 的 ref_idx = i // 2 只对 (self, cross) 交替、以 self 开头的排布成立。
        if not names or len(names) % 2 or names != ("self", "cross") * (len(names) // 2):
            raise ValueError(f"layer_names 必须是 self,cross 交替 (如 self,cross,self,cross), 收到 {names}")
        self.layer_names = names
        self.layers = nn.ModuleList(SVABlock(dim, heads, mlp_ratio) for _ in names)
        self.pe = NormalizedPE2D(dim, pe_max_shape)
        self._reset_parameters()
        self.last_stats: dict | None = None

    def _reset_parameters(self) -> None:
        # FMT._reset_parameters: 所有 >1 维的参数 xavier_uniform
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: [B, V, C, h, w] (view 0 = reference) -> 同形状。"""
        B, V, C, h, w = feat.shape
        x = self.pe(feat.reshape(B * V, C, h, w)).view(B, V, C, h, w)
        tok = x.flatten(3).transpose(2, 3).contiguous()       # [B, V, N, C]
        N = h * w

        ref = tok[:, 0]
        ref_states = []
        for layer, name in zip(self.layers, self.layer_names):
            if name == "self":
                ref = layer(ref)
                ref_states.append(ref)

        out = [ref]
        if V > 1:
            src = tok[:, 1:].reshape(B * (V - 1), N, C)
            for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
                if name == "self":
                    src = layer(src)                          # 每个 source 视角内部
                else:
                    r = ref_states[i // 2]
                    # 注意力对每个 query 独立: V-1 个视角拼在 token 维上查同一个 ref
                    # 与逐视角循环逐位同义, 但 ref 的 k/v 只投影一次。
                    q = src.reshape(B, (V - 1) * N, C)
                    src = layer(q, kv=r).reshape(B * (V - 1), N, C)
            out.append(src.reshape(B, V - 1, N, C))
            y = torch.cat([out[0].unsqueeze(1), out[1]], dim=1)
        else:
            y = out[0].unsqueeze(1)

        with torch.no_grad():
            # 纯诊断: 注意力块相对 (特征 + PE) 的改动量。恒 ~0 = 什么都没学到;
            # >>1 = 在覆盖 FPN 特征而不是修正它。
            d = y.float() - tok.float()
            self.last_stats = {
                "delta_rel": float(d.norm() / tok.float().norm().clamp_min(1e-6)),
                "ls1_mean": float(torch.stack([l.ls1.abs().mean() for l in self.layers]).mean()),
            }
        return y.transpose(2, 3).reshape(B, V, C, h, w)


class SVAPathway(nn.Module):
    """``FMT_with_pathway``: 1/8 上跑 FMT, 再建一条从 FMT 输出出发的逐级融合路径,
    叠加到普通 FPN 的 1/4、1/2、1/1 输出上。

    每一级: ``smooth(upsample(reduce(上一级)) + 普通 FPN 这一级)``, reduce 为
    无 bias 的 1x1, smooth 为无 bias 的 3x3 —— 与原实现相同 (那两组卷积用 PyTorch
    默认初始化, 只有 FMT 自己做 xavier)。
    """

    def __init__(self, channels: int, heads: int = 4,
                 layer_names: tuple[str, ...] = ("self", "cross", "self", "cross"),
                 mlp_ratio: float = 4.0, pe_max_shape: tuple[int, int] = (128, 128),
                 strides: tuple[int, ...] = (8, 4, 2, 1)) -> None:
        super().__init__()
        self.strides = tuple(int(s) for s in strides)
        self.fmt = HighResSVA(channels, heads=heads, layer_names=layer_names,
                              mlp_ratio=mlp_ratio, pe_max_shape=pe_max_shape)
        n = len(self.strides) - 1
        self.reduce = nn.ModuleList(nn.Conv2d(channels, channels, 1, bias=False) for _ in range(n))
        self.smooth = nn.ModuleList(nn.Conv2d(channels, channels, 3, padding=1, bias=False)
                                    for _ in range(n))

    def forward(self, feats: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """feats: 普通 FPN 输出 {stride: [B, V, C, h, w]} -> 同结构的 SVA 输出。"""
        s0 = self.strides[0]
        prev = self.fmt(feats[s0])
        out = {s0: prev}
        for stride, red, sm in zip(self.strides[1:], self.reduce, self.smooth):
            f = feats[stride]
            B, V, C, h, w = f.shape
            x = F.interpolate(red(prev.flatten(0, 1)), size=(h, w), mode="bilinear",
                              align_corners=False)
            prev = sm(x + f.flatten(0, 1)).view(B, V, C, h, w)
            out[stride] = prev
        return out
