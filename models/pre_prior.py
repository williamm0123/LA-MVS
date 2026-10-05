"""Offline prior precomputation & caching.

For each training sample this computes, **once**, the geometry priors the
network consumes and stores them to disk:

    {depth_prior, conf_prior, norm_depth_fill, src_weights}

Pipeline per sample (all at the *pre-crop* resolution so random-crop stays
valid -- the cached full-frame maps are simply sliced alongside the image at
load time):

    sfm.generate_sparse_depth_from_sample  -> sparse_depth (metric) + source_weights
        └─ sparse_depth fed to norm_fill for metric scale calibration
    norm_fill.generate_priors_from_sample  -> depth_filled / conf_map / normals

Heavy models (VGGT + DA3) are loaded **once** by ``PriorPrecomputer`` and
reused across all samples. This module's top level stays import-light (numpy
only) so ``data.dtu`` can import :func:`load_prior` without pulling in the VGGT
stack; the heavy deps are imported lazily inside the compute path.
"""

from __future__ import annotations

from pathlib import Path

import os

import numpy as np
import torch

from base.config import ProjectPaths
# Keys stored in every prior cache file (and expected by the network/loss).
PRIOR_KEYS = ("depth_prior", "conf_prior", "norm_depth_fill", "src_weights")
# 标尺质量元数据。旧缓存没有这些键, load_prior 会补默认值 —— 但默认是
# ``sfm_valid=0``(未知), 这样"没有元数据"和"标尺失败"都会走保守路径。
META_KEYS = ("sfm_valid", "sfm_scale", "sfm_num_pairs", "pipeline_version",
             "num_views", "target_w", "target_h", "prior_h", "prior_w",
             # 尺度的来源, 由 scripts/build_prior_cache_all.py 的回退链写入:
             #   0 = 本样本自己的 SfM (默认, 旧缓存读到的也是 0)
             #   1 = 同视角换了光照重跑 SfM (几何没变, 稀疏点更多; 尺度仍是精确的
             #       —— 它是拿*本样本自己的*稠密深度和那批稀疏点求的比值中位数)
             #   2 = 借了同 scan 相邻视角的尺度 (近似, 实测邻视角之间差 ~5%)
             # scale_light / scale_ref_view 记下尺度是从哪个光照/视角来的。
             # 只在 source>0 时有意义。**不 bump PIPELINE_VERSION**: 这几个键是
             # 增量的, 旧缓存缺了按 0.0 读 = "自有尺度", 语义正确; bump 会让全部
             # 28k 个缓存被判过期, 触发一次没必要的整轮重建。
             "scale_source", "scale_light", "scale_ref_view")
PIPELINE_VERSION = 3


def cache_signature(prior: dict) -> dict:
    """从缓存里读出它是"怎么生成的"。

    ``auto`` 模式过去只判断文件是否存在, 于是旧缓存被当成完整的。实测现有
    train/val 缓存全部是 ``pipeline_version=0`` 且 ``src_weights`` 形状 (2,)
    —— 它们是用 1 ref + 2 source 生成的, 而现在训练喂 1 ref + 4 source。
    VGGT 是多视图模型, 视角集变了先验就不是同一个东西; SfM 标尺也因为可
    三角化的点更少而更容易失败 (这多半就是 8% 未标尺的来源)。
    """
    def g(k, d=0.0):
        v = prior.get(k, d)
        return float(np.asarray(v).reshape(-1)[0]) if np.size(v) else d
    return {
        "pipeline_version": int(g("pipeline_version")),
        "num_views": int(g("num_views")),
        "target_w": int(g("target_w")),
        "target_h": int(g("target_h")),
    }


def cache_signature_from_file(path: str | Path) -> dict | None:
    """只读元数据标量版的 :func:`cache_signature` —— 不解压大数组。

    ``load_prior`` 会把 depth_prior / conf_prior / norm_depth_fill 三个大数组
    全部解压, 用它做"缓存是否过期"的判定就等于**每次启动训练都把整个缓存读一遍**
    (train split 27097 个文件 × 3.7 MB ≈ 100 GB)。本地 SSD 上 4.2 ms/个 (约
    114 秒), 集群的共享 /scr 上要慢一个量级, 而且每次 launch 都来一次 —— 表现
    就是卡在 "[pre_prior] ensuring priors for train split" 不动。

    npz 就是个 zip: 只取几个 0 维标量不会碰大数组, 实测 0.35 ms/个 (快 12 倍,
    I/O 从 100 GB 降到几十 MB)。文件读不了 (截断/损坏) 返回 None, 调用方按
    "过期"处理, 和原来 ``except Exception: return True`` 的语义一致。
    """
    try:
        with np.load(path) as z:
            def g(k: str) -> float:
                return float(z[k]) if k in z.files else 0.0
            return {"pipeline_version": int(g("pipeline_version")),
                    "num_views": int(g("num_views")),
                    "target_w": int(g("target_w")),
                    "target_h": int(g("target_h"))}
    except Exception:
        return None


def signature_is_current(sig: dict, target_wh: tuple[int, int]) -> bool:
    """判据本体, 见 :func:`cache_is_current` 的说明。"""
    return (sig["pipeline_version"] >= PIPELINE_VERSION
            and sig["target_w"] == int(target_wh[0])
            and sig["target_h"] == int(target_wh[1]))


def cache_is_current(prior: dict, num_views: int, target_wh: tuple[int, int]) -> bool:
    """缓存是否需要重建。

    **只看 pipeline_version 和 target_wh。**

    ``num_views`` 是*出处*信息, 不是兼容性判据: 缓存的键是 (scan, ref_view,
    light), 存的是参考视角的一张深度图; 训练时 FPN / cost volume / 位姿全部
    用 dataloader 的图像现算, ``depth_prior`` 只是被读进来当 stage1 的候选
    中心。所以"生成先验时用了几个 source"只影响这张先验的*质量*, 不影响它
    能不能被 V=3 或 V=5 的训练使用。

    (曾经把 num_views 当判据, 那会让本地 3 视角建的缓存在 5 视角的集群上被
    判定过期, 触发一次毫无必要的 25 小时重建。)

    ``target_wh`` 则是真的判据 —— 它决定 VGGT/DA3 实际跑的分辨率, 也就是
    先验的真实细节量, ``_match_hw`` 的重采样造不出来。
    """
    return signature_is_current(cache_signature(prior), target_wh)


# --------------------------------------------------------------------------- #
# Cache IO (import-light: numpy only)
# --------------------------------------------------------------------------- #
def save_prior(path: str | Path, prior: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {k: np.asarray(prior[k], dtype=np.float32) for k in PRIOR_KEYS}
    for k in META_KEYS:
        out[k] = np.asarray(prior.get(k, 0.0), dtype=np.float32)
    # 临时文件 + os.replace: 中途崩溃不会留下半个 npz 被后续当成有效缓存。
    # 注意必须传*文件句柄*: np.savez_compressed 收到不以 .npz 结尾的*路径*时
    # 会自己追加 .npz, 于是 "a.npz.tmp" 变成 "a.npz.tmp.npz", os.replace 找不到。
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **out)
    os.replace(tmp, path)


def load_prior(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"prior cache missing: {path}. Run the prior precompute first "
            f"(train.py does this automatically unless --build-priors skip)."
        )
    with np.load(path) as data:
        out = {k: data[k] for k in PRIOR_KEYS}
        for k in META_KEYS:
            out[k] = data[k] if k in data.files else np.float32(0.0)
    return out


# --------------------------------------------------------------------------- #
# Per-sample computation (heavy deps imported lazily)
# --------------------------------------------------------------------------- #
class PriorPrecomputer:
    """Loads VGGT + DA3 once, computes priors for a pre-crop multi-view sample."""

    def __init__(
        self,
        device,
        image_mode: str = "resize",
        conf_percentile: float = 10.0,
        image_target_wh: tuple[int, int] = (518, 420),
    ) -> None:


        # 懒加载: models.norm_fill 顶层 import 了 vggt 和 depth_anything_3, 放在模块
        # 顶层会让**训练进程**也把这两个栈拖进来 (train.py -> data.dtu -> pre_prior),
        # 于是启动脚本必须替它们设 PYTHONPATH。训练只读缓存, 从不现算先验 —— 只有
        # 真要建缓存时才会走到这里。放在这里, 顶层就回到了 docstring 声称的
        # "import-light (numpy only)"。
        import models.norm_fill as norm_fill

        self._nf = norm_fill
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        paths = ProjectPaths()
        self.vggt_model = norm_fill.load_vggt_model(paths.vggt_weights_path, self.device)
        self.da3_model = norm_fill.load_da3_model(paths.da3_weights_file, self.device)
        self.image_mode = image_mode
        self.conf_percentile = conf_percentile
        self.image_target_wh = image_target_wh

    def compute(self, precrop_sample: dict) -> dict:
        """precrop_sample needs ``images`` [V,C,H,W], ``intrinsics`` [V,3,3],
        ``extrinsics`` [V,4,4] at the pre-crop resolution."""
        import models.sfm as sfm

        # 1) SfM: metric sparse depth (for scale) + per-source weights.
        sfm_out = sfm.generate_sparse_depth_from_sample(precrop_sample, ref_idx=0)

        sample = dict(precrop_sample)
        sample["sfm_depth"] = sfm_out["sparse_depth"]  # reused by norm_fill for scale

        # 2) Dense fill + confidence + normals (models reused, not reloaded).
        priors = self._nf.generate_priors_from_sample(
            sample,
            self.device,
            image_mode=self.image_mode,
            conf_percentile=self.conf_percentile,
            image_target_wh=self.image_target_wh,
            vggt_model=self.vggt_model,
            da3_model=self.da3_model,
        )
        info = priors.get("sfm_info", {}) or {}
        return {
            "depth_prior": np.asarray(priors["depth_filled"], np.float32),
            "conf_prior": np.asarray(priors["conf_map"], np.float32),
            "norm_depth_fill": np.asarray(priors["normal"], np.float32),
            "src_weights": np.asarray(sfm_out["source_weights"], np.float32),
            "sfm_valid": float(bool(info.get("valid", False))),
            "sfm_scale": float(priors.get("sfm_scale", 1.0)),
            "sfm_num_pairs": float(info.get("num_pairs", 0)),
            "pipeline_version": float(PIPELINE_VERSION),
            "num_views": float(len(precrop_sample["images"])),
            "target_w": float(self.image_target_wh[0]),
            "target_h": float(self.image_target_wh[1]),
            "prior_h": float(priors["depth_filled"].shape[0]),
            "prior_w": float(priors["depth_filled"].shape[1]),
        }




def _check_target_wh(image_target_wh: tuple[int, int]) -> None:
    """VGGT/DA3 use a 14px patch; both dims must be multiples of 14 or the DPT
    head reassembly (H//14 patches -> *14) truncates/misaligns the depth map."""
    w, h = image_target_wh
    bad = [d for d in (w, h) if d % 14 != 0]
    if bad:
        raise ValueError(
            f"prior image_target_wh={image_target_wh} must have both dims divisible "
            f"by the backbone patch size 14 (offending: {bad}). Nearest multiples: "
            f"w={round(w / 14) * 14}, h={round(h / 14) * 14}."
        )


def build_prior_cache(
    dataset,
    device,
    overwrite: bool = False,
    verbose: bool = True,
    image_target_wh: tuple[int, int] = (518, 420),
    fail_open: bool = False,
) -> int:
    """Populate the prior cache for every meta in ``dataset`` (run once, main process).

    ``dataset`` must expose ``precrop_inputs(idx)`` (pre-crop multi-view sample)
    and ``prior_cache_path_for(idx)``. ``image_target_wh`` is the resolution VGGT
    + DA3 run at (the prior's true resolution before it is resampled up to the
    working image size); both dims must be multiples of 14.
    """
    _check_target_wh(image_target_wh)
    n = len(dataset)
    # skip loading the heavy models entirely if everything is already cached
    def _stale(i: int) -> bool:
        f = Path(dataset.prior_cache_path_for(i))
        if not f.exists():
            return True
        sig = cache_signature_from_file(f)      # 只读元数据, 不解压大数组
        return sig is None or not signature_is_current(sig, image_target_wh)

    pending = [i for i in range(n) if overwrite or _stale(i)]
    if pending and not overwrite and verbose:
        print(f"[pre_prior] {len(pending)}/{n} 需要重建 (缺失, 或 pipeline_version/"
              f"num_views/target_wh 与当前不符)")
        # 视角数不符是最常见也最贵的误触发: 本地脚本和集群脚本的 NUM_VIEWS
        # 曾经一个 3 一个 5, 缓存按其中一个建、训练按另一个跑, 就会闷头重建
        # 24206 个样本 (约 21 小时)。这里大声报出来。
        mism = {}
        for i in pending[:200]:
            f = Path(dataset.prior_cache_path_for(i))
            if not f.exists():
                continue
            sig = cache_signature_from_file(f)
            if sig and sig["num_views"] and sig["num_views"] != dataset.nviews:
                mism[sig["num_views"]] = mism.get(sig["num_views"], 0) + 1
        if mism:
            print(f"[pre_prior] 提示: 部分缓存的 num_views={mism}, 当前训练 "
                  f"nviews={dataset.nviews}。这*不会*触发重建 —— 先验只是参考"
                  f"视角的一张深度图, 与训练的 source 数无关。视角数只影响先验"
                  f"质量 (source 越多 SfM 三角化点越多、标尺越容易成功), 跨代"
                  f"比较实验时需要知道这一点。")
    if not pending:
        if verbose:
            print(f"[pre_prior] cache already complete: {n} priors")
        return 0

    if verbose:
        print(f"[pre_prior] building {len(pending)}/{n} priors at target_wh={image_target_wh} "
              f"(loading VGGT + DA3 once) ...")
    precomputer = PriorPrecomputer(device, image_target_wh=image_target_wh)
    built = failed = 0
    quarantine_log = []
    for idx in pending:
        pc = dataset.precrop_inputs(idx)
        prior = precomputer.compute(pc)
        dst = Path(dataset.prior_cache_path_for(idx))
        # fail-closed: 标尺无效时**照样写主缓存**, 但 sfm_valid=0。
        # 不删文件的三个理由:
        #   1. 删了 dataloader 会 FileNotFoundError —— 除非同时更新 exclude 表,
        #      而那是 audit 脚本的事, 两者不同步就炸。
        #   2. build_prior_cache 的 pending 判据是"文件不存在", 删了会导致每次
        #      auto 都重跑同一批失败样本 (VGGT+DA3 白算)。
        #   3. data/dtu.py 读到 sfm_valid=0 会把 prior_valid 置 0, 网络自动退回
        #      global-only —— 这恰好是"先验失败时 guard 分支兜底"这一主张的
        #      真实检验, 比把样本删掉更有研究价值。
        # 同时在 *_quarantine/ 留一份副本供审计。
        if not (fail_open or prior["sfm_valid"] > 0.5):
            qdst = dst.parent.parent.with_name(dst.parent.parent.name + "_quarantine") / dst.parent.name / dst.name
            save_prior(qdst, prior)
            save_prior(dst, prior)          # sfm_valid=0, 网络会自己绕开
            quarantine_log.append((str(dst.relative_to(dst.parent.parent)),
                                   int(prior["sfm_num_pairs"])))
            failed += 1
            continue
        save_prior(dst, prior)
        built += 1
        if verbose and (built + failed) % 20 == 0:
            print(f"[pre_prior]   {built + failed}/{len(pending)} done ({failed} quarantined)")
    if verbose:
        print(f"[pre_prior] cache ready: {built} newly built, {failed} 标尺无效 "
              f"(已写主缓存但 sfm_valid=0, 副本在 *_quarantine/), {n} total")
        for name, npairs in quarantine_log[:10]:
            print(f"[pre_prior]   quarantined {name}  num_pairs={npairs}")
    return built
