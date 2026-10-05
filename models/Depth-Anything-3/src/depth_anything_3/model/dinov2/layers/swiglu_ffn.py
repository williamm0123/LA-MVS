# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional
import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


try:
    from xformers.ops import SwiGLU

    XFORMERS_AVAILABLE = True
except Exception:
    # 2026-09-20: umhpc 上 torch 被别的安装悄悄升到了 2.8.0+cu128, 而装的 xformers 还是
    # 编译给 2.3.1+cu121 的, C++/CUDA 扩展加载不了, 退到它自己的 triton 后端 —— 那条路又和
    # 装的 triton 版本不兼容, 在 `xformers.ops.fmha.triton_splitk` 模块导入时就直接抛
    # AttributeError (不是 ImportError), 原来这里只 except ImportError 接不住, 整个脚本
    # 直接崩掉。SwiGLU 只是个可选的融合算子, 退回下面纯 PyTorch 的 SwiGLUFFN 功能等价、
    # 只是慢一点, 所以任何导入期异常都应该退化, 不止 ImportError。
    SwiGLU = SwiGLUFFN
    XFORMERS_AVAILABLE = False


class SwiGLUFFNFused(SwiGLU):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )
