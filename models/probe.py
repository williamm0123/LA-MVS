"""逐样本的张量健康探针 —— 用来定位 stage4 前向非有限值的第一处源头。

为什么需要它: 看门狗只能看到 ``outputs`` 里的顶层张量, 所以事故报告里只有
``depth_full finite_frac=0.5``。真正要知道的是这条链上**哪一环先坏**::

    ref_p -> src_p -> warped -> cv_s -> cost -> logits_raw -> logits -> prob -> depth

而且必须**逐样本**统计: 五次崩溃的 ``finite_frac`` 都恰好是 0.500000, 与
"batch=2 里只有一个样本整幅坏掉"一致 —— 全局均值会把这件事抹平。

用法是"默认关闭、按需打开"::

    from models.probe import Probe
    with Probe.session():
        out = model(batch)
    rows = Probe.rows()

关闭时 ``Probe.log`` 只做一次布尔判断, 训练路径上的开销可以忽略。
"""

from __future__ import annotations

import contextlib

import torch


class Probe:
    """进程级的探针记录器。单线程前向下使用, 不做并发保护。"""

    active: bool = False
    _rows: list[dict] = []

    @classmethod
    @contextlib.contextmanager
    def session(cls):
        """打开探针并在退出时关掉; 每次进入都清空上一轮的记录。"""
        prev, cls.active, cls._rows = cls.active, True, []
        try:
            yield cls
        finally:
            cls.active = prev

    @classmethod
    def rows(cls) -> list[dict]:
        return list(cls._rows)

    @classmethod
    def log(cls, stage: str, name: str, t: torch.Tensor, src: int | None = None) -> None:
        """记一个张量的逐样本健康度。``t`` 的第 0 维必须是 batch。"""
        if not cls.active or not isinstance(t, torch.Tensor):
            return
        with torch.no_grad():
            x = t.detach()
            b = x.shape[0]
            flat = x.reshape(b, -1).float()
            fin = torch.isfinite(flat)
            frac = fin.float().mean(dim=1)
            # max|x| 只在有限元素上取, 否则永远是 inf, 看不出"坏之前有多大"
            big = torch.where(fin, flat.abs(), torch.zeros_like(flat)).amax(dim=1)
            cls._rows.append({
                "stage": stage,
                "src": src,
                "name": name,
                "dtype": str(x.dtype).replace("torch.", ""),
                "shape": tuple(x.shape),
                "finite_frac": [round(float(v), 6) for v in frac],
                "max_abs": [round(float(v), 3) for v in big],
            })

    @classmethod
    def first_bad(cls, tol: float = 1.0) -> dict | None:
        """按记录顺序返回第一个存在非有限元素的张量。"""
        for r in cls._rows:
            if min(r["finite_frac"]) < tol:
                return r
        return None

    @classmethod
    def format(cls, only_bad: bool = False) -> str:
        rows = cls._rows
        if only_bad:
            rows = [r for r in rows if min(r["finite_frac"]) < 1.0]
        if not rows:
            return "  (无记录)"
        w = max(len(f"{r['stage']}.{r['name']}") + (4 if r["src"] is not None else 0) for r in rows)
        out = [f"  {'张量'.ljust(w)}  {'dtype':<9}{'finite_frac (逐样本)':<30}max|x| (逐样本)"]
        for r in rows:
            tag = f"{r['stage']}.{r['name']}" + (f"[s{r['src']}]" if r["src"] is not None else "")
            ff = " ".join(f"{v:.3f}" for v in r["finite_frac"])
            mx = " ".join(f"{v:g}" for v in r["max_abs"])
            flag = "   <<<" if min(r["finite_frac"]) < 1.0 else ""
            out.append(f"  {tag.ljust(w)}  {r['dtype']:<9}{ff:<30}{mx}{flag}")
        return "\n".join(out)
