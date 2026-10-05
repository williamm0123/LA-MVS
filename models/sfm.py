"""Lightweight two-view SfM for DTU samples with known camera poses.

DTU provides metric (millimetre) intrinsics/extrinsics, so we do not need to
solve for camera poses. We only need correspondences: detect + match features
between the reference view and each source view, reject outliers with RANSAC,
then triangulate the surviving matches with the *known* projection matrices.
The triangulated world points are projected into the reference camera to form a
sparse, metric-scale depth map that can later anchor the (scale-free) VGGT
prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import data.camera_utils as C


@dataclass
class SfMConfig:
    max_features: int = 8000
    ratio_test: float = 0.75
    max_reproj_error: float = 2.0
    min_depth: float = 1e-3
    max_depth: float = 2000.0
    # per-point confidence (conf = f_reproj * f_angle * w_pair, all in [0,1])
    conf_tau_e: float = 1.0          # reprojection-error decay scale (px)
    conf_theta_sat_deg: float = 10.0 # triangulation-angle saturation (deg)
    conf_pair_n0: float = 100.0      # pair-inlier soft-saturation constant


def _to_uint8_rgb(image) -> np.ndarray:
    """sample["images"][i] is a [C, H, W] float tensor in 0-255 RGB."""
    arr = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _build_detector(max_features: int):
    """Prefer SIFT (metric-friendly float descriptors); fall back to ORB."""
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=int(max_features)), "SIFT", cv2.NORM_L2
    return cv2.ORB_create(nfeatures=int(max_features)), "ORB", cv2.NORM_HAMMING


def _projection_matrix(K: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    return (np.asarray(K, np.float64) @ np.asarray(extrinsic, np.float64)[:3, :4])


def _camera_depth(extrinsic: np.ndarray, points_h: np.ndarray) -> np.ndarray:
    """Z of homogeneous world points in the given camera frame."""
    return (points_h @ np.asarray(extrinsic, np.float64)[:3, :4].T)[:, 2]


def _reproj_error(P: np.ndarray, points_h: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    proj = points_h @ P.T
    uv = proj[:, :2] / np.clip(proj[:, 2:3], 1e-12, None)
    return np.linalg.norm(uv - pixels, axis=1)


def _camera_center(extrinsic: np.ndarray) -> np.ndarray:
    """World-frame camera centre from a world->camera extrinsic: C = -R^T t."""
    R = np.asarray(extrinsic, np.float64)[:3, :3]
    t = np.asarray(extrinsic, np.float64)[:3, 3]
    return -R.T @ t


def _point_confidence(points, err_ref, err_src, E_ref, E_src, n_pair, cfg) -> np.ndarray:
    """Per-point triangulation confidence in [0, 1].

        conf = f_reproj * f_angle * w_pair
          f_reproj : reprojection consistency, taken over the *worse* of the two
                     views  -> exp(-(max(err_ref, err_src) / tau_e)^2)
          f_angle  : parallax angle at the 3D point between the rays to each
                     camera; small angle -> uncertain depth. conf ~ sin(theta),
                     saturating at theta_sat.
          w_pair   : pair-level reliability from the RANSAC inlier count
                     (soft-saturating: N / (N + N0)).
    """
    if len(points) == 0:
        return np.empty((0,), np.float32)

    e = np.maximum(err_ref, err_src)
    f_reproj = np.exp(-((e / max(cfg.conf_tau_e, 1e-6)) ** 2))

    C_ref = _camera_center(E_ref)
    C_src = _camera_center(E_src)
    v_ref = C_ref[None, :] - points
    v_src = C_src[None, :] - points
    denom = np.clip(np.linalg.norm(v_ref, axis=1) * np.linalg.norm(v_src, axis=1), 1e-12, None)
    cos_theta = np.clip(np.sum(v_ref * v_src, axis=1) / denom, -1.0, 1.0)
    sin_theta = np.sin(np.arccos(cos_theta))
    sin_sat = max(np.sin(np.radians(cfg.conf_theta_sat_deg)), 1e-6)
    f_angle = np.clip(sin_theta / sin_sat, 0.0, 1.0)

    w_pair = float(n_pair) / (float(n_pair) + max(cfg.conf_pair_n0, 1e-6))

    return (f_reproj * f_angle * w_pair).astype(np.float32)


def _triangulate_pair(gray_ref, gray_src, K_ref, E_ref, K_src, E_src, detector, norm_type, cfg):
    """Return (world_points [M,3], ref_pixels [M,2], stats dict) for one pair."""
    stats = {"matches": 0, "ransac_matches": 0, "triangulated_points": 0}

    kp_ref, des_ref = detector.detectAndCompute(gray_ref, None)
    kp_src, des_src = detector.detectAndCompute(gray_src, None)
    if des_ref is None or des_src is None or len(kp_ref) < 2 or len(kp_src) < 2:
        return np.empty((0, 3), np.float32), np.empty((0, 2), np.float32), np.empty((0,), np.float32), stats

    matcher = cv2.BFMatcher(norm_type)
    knn = matcher.knnMatch(des_ref, des_src, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < cfg.ratio_test * n.distance]
    stats["matches"] = len(good)
    if len(good) < 8:
        return np.empty((0, 3), np.float32), np.empty((0, 2), np.float32), np.empty((0,), np.float32), stats

    pts_ref = np.float64([kp_ref[m.queryIdx].pt for m in good])
    pts_src = np.float64([kp_src[m.trainIdx].pt for m in good])

    F, mask = cv2.findFundamentalMat(
        pts_ref, pts_src, cv2.FM_RANSAC, cfg.max_reproj_error, 0.99
    )
    if F is None or mask is None:
        return np.empty((0, 3), np.float32), np.empty((0, 2), np.float32), np.empty((0,), np.float32), stats
    inliers = mask.ravel().astype(bool)
    pts_ref, pts_src = pts_ref[inliers], pts_src[inliers]
    stats["ransac_matches"] = int(inliers.sum())
    if len(pts_ref) < 1:
        return np.empty((0, 3), np.float32), np.empty((0, 2), np.float32), np.empty((0,), np.float32), stats

    P_ref = _projection_matrix(K_ref, E_ref)
    P_src = _projection_matrix(K_src, E_src)
    pts4d = cv2.triangulatePoints(P_ref, P_src, pts_ref.T, pts_src.T)
    w = pts4d[3:4]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)  # guard zero only; keep sign of w
    points = (pts4d[:3] / w).T  # [M, 3] world
    points_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)

    z_ref = _camera_depth(E_ref, points_h)
    z_src = _camera_depth(E_src, points_h)
    err_ref = _reproj_error(P_ref, points_h, pts_ref)
    err_src = _reproj_error(P_src, points_h, pts_src)
    keep = (
        (z_ref > cfg.min_depth) & (z_ref < cfg.max_depth)
        & (z_src > cfg.min_depth)
        & (err_ref <= cfg.max_reproj_error) & (err_src <= cfg.max_reproj_error)
    )
    stats["triangulated_points"] = int(keep.sum())
    conf = _point_confidence(
        points[keep], err_ref[keep], err_src[keep], E_ref, E_src, stats["ransac_matches"], cfg
    )
    return points[keep].astype(np.float32), pts_ref[keep].astype(np.float32), conf, stats


def generate_sparse_depth_from_sample(sample, ref_idx: int = 0, config: SfMConfig | None = None):
    """Build a metric sparse depth map for the reference view via two-view SfM.

    Returns a dict with ``sparse_depth`` [H, W], ``sparse_conf`` [H, W] in [0,1]
    (per-pixel confidence of the point that won each pixel), ``valid_mask``
    [H, W] bool, ``points_world`` [N, 3], ``points_conf`` [N] in [0,1],
    ``source_weights`` [num_views-1] in [0,1] (one per-view cost-volume weight,
    aligned to the non-ref source order), ``points_color`` [N, 3] uint8 and an
    ``info`` dict. ``sparse_depth`` is identical to before (nearest-point
    z-buffer); confidence is additive only.
    """
    cfg = config or SfMConfig()
    images = sample["images"]
    intrinsics = np.asarray(sample["intrinsics"], np.float64)
    extrinsics = np.asarray(sample["extrinsics"], np.float64)
    num_views = len(images)

    rgb = [_to_uint8_rgb(images[i]) for i in range(num_views)]
    gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in rgb]
    H, W = gray[ref_idx].shape[:2]

    detector, feature_type, norm_type = _build_detector(cfg.max_features)
    K_ref, E_ref = intrinsics[ref_idx], extrinsics[ref_idx]

    all_points, all_pixels, all_conf, source_weights, pairs = [], [], [], [], []
    for src_idx in range(num_views):
        if src_idx == ref_idx:
            continue
        points, pixels, conf, stats = _triangulate_pair(
            gray[ref_idx], gray[src_idx], K_ref, E_ref,
            intrinsics[src_idx], extrinsics[src_idx], detector, norm_type, cfg,
        )
        all_points.append(points)
        all_pixels.append(pixels)
        all_conf.append(conf)
        # per-view weight for cost-volume fusion: median of this source's point
        # confidences (0.0 if the pair produced no triangulated points).
        w = float(np.median(conf)) if len(conf) else 0.0
        source_weights.append(w)
        pairs.append({"src_idx": src_idx, "weight": w, **stats})

    points_world = np.concatenate(all_points, axis=0) if all_points else np.empty((0, 3), np.float32)
    pixels = np.concatenate(all_pixels, axis=0) if all_pixels else np.empty((0, 2), np.float32)
    points_conf = np.concatenate(all_conf, axis=0) if all_conf else np.empty((0,), np.float32)
    # aligned to the non-ref source order (src_idx ascending, ref skipped)
    source_weights = np.asarray(source_weights, np.float32)

    # Sample reference-image colours at the matched keypoints for the PLY.
    if len(pixels):
        u = np.clip(np.rint(pixels[:, 0]).astype(np.int32), 0, W - 1)
        v = np.clip(np.rint(pixels[:, 1]).astype(np.int32), 0, H - 1)
        points_color = rgb[ref_idx][v, u]
    else:
        points_color = np.empty((0, 3), np.uint8)

    # conf is per-point; the z-buffer inside keeps nearest-point-wins, so
    # sparse_depth is unchanged vs passing None -- we only additionally get the
    # per-pixel confidence of whichever point won each pixel.
    sparse_depth, sparse_conf = C.project_world_points_to_depth(
        points_world, points_conf, K_ref, E_ref, (W, H)
    )
    valid_mask = sparse_depth > 0

    info = {
        "feature_type": feature_type,
        "num_points_world": int(len(points_world)),
        "pairs": pairs,
    }
    return {
        "sparse_depth": sparse_depth,
        "sparse_conf": sparse_conf,
        "valid_mask": valid_mask,
        "points_world": points_world,
        "points_conf": points_conf,
        "source_weights": source_weights,
        "points_color": points_color,
        "info": info,
    }


def load_or_compute_sparse_depth(
    images,
    intrinsics,
    extrinsics,
    cache_path,
    ref_idx: int = 0,
    config: SfMConfig | None = None,
    save_vis: bool = True,
):
    """Return the ref-view SfM sparse depth at the input image resolution.

    Loads ``cache_path`` (an ``.npy``) if it exists, otherwise runs two-view SfM
    on the given (multi-view) arrays, caches the result and an optional ``.png``
    visualisation next to it. ``images`` may be ``[V, H, W, 3]`` uint8 or
    ``[V, C, H, W]``; ``intrinsics`` ``[V, 3, 3]``; ``extrinsics`` ``[V, 4, 4]``.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        return np.load(cache_path).astype(np.float32)

    sfm_sample = {"images": images, "intrinsics": intrinsics, "extrinsics": extrinsics}
    out = generate_sparse_depth_from_sample(sfm_sample, ref_idx=ref_idx, config=config)
    sparse_depth = out["sparse_depth"].astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, sparse_depth)
    if save_vis:
        C.save_depth_png(sparse_depth, cache_path.with_suffix(".png"), valid=out["valid_mask"])
    return sparse_depth


def metric_scale_from_sparse(depth, sparse_depth, sparse_valid=None, min_pairs: int = 20):
    """Global VGGT->metric scale s.t. ``depth * scale`` matches ``sparse_depth``.

    VGGT depth is metric-consistent up to a single global scale, so the scale is
    the median of the per-pixel ratio ``sparse_depth / depth`` over the pixels
    where both are valid. Using the (零稀释的) mean of ``sparse_depth`` is wrong.
    Returns ``(scale, info)``; ``scale`` falls back to 1.0 when too few overlaps.
    """
    depth = np.asarray(depth, np.float32)
    sparse_depth = np.asarray(sparse_depth, np.float32)
    if sparse_valid is None:
        sparse_valid = sparse_depth > 0
    mask = (
        np.asarray(sparse_valid, bool)
        & np.isfinite(depth) & (depth > 0)
        & np.isfinite(sparse_depth) & (sparse_depth > 0)
    )
    num_pairs = int(mask.sum())
    if num_pairs < min_pairs:
        return 1.0, {"num_pairs": num_pairs, "valid": False, "scale": 1.0}
    ratio = sparse_depth[mask] / depth[mask]
    scale = float(np.median(ratio))
    return scale, {"num_pairs": num_pairs, "valid": True, "scale": scale}


def metric_affine_from_sparse(depth, sparse_depth, sparse_valid=None, *,
                              min_pairs: int = 200, holdout_frac: float = 0.3,
                              block: int = 64, trim_mad: float = 3.0, iters: int = 3,
                              depth_min: float | None = None, depth_max: float | None = None):
    """在**逆深度域**拟合 ``1/z_metric = a * (1/z_prior) + b``。

    为什么不是纯缩放: ``metric_scale_from_sparse`` 只拟合一个乘性 scale, 依据是
    "VGGT depth is metric-consistent up to a single global scale"。实测这条假设
    不完全成立 —— 先验误差对 GT 深度呈 U 形, 最低点在 ~620mm, 往两边发散, 而用
    GT 做的 oracle 逆深度仿射能把 log-log 斜率从 1.99 压到 0.70、最远档误差从
    9.62mm 降到 4.55mm。这正是"输出本质是仿射-in-逆深度、却只拟合了乘性 scale"
    留下的残差形状。

    守卫 (任一不满足就退回 scale-only, 由调用方处理):
      * ``a > 0``, 且全物理深度范围内 ``a*rho + b > 0`` —— 否则映射会翻转或产生
        负深度;
      * 逆深度跨度和条件数足够 —— 稀疏点挤在一个深度上时 b 无法辨识;
      * held-out 残差必须优于 scale-only —— 拟合点上的改善不算数。

    fit/held-out 按**空间块**划分而不是随机像素: SfM 点在空间上高度聚集, 随机
    划分会让同一个 track 的邻域同时进两边, held-out 就失去意义。

    返回 ``(ok, params, info)``；``params = (a, b)``, ``info`` 含两侧的残差分位数。
    """
    depth = np.asarray(depth, np.float32)
    sparse_depth = np.asarray(sparse_depth, np.float32)
    if sparse_valid is None:
        sparse_valid = sparse_depth > 0
    m = (np.asarray(sparse_valid, bool)
         & np.isfinite(depth) & (depth > 0)
         & np.isfinite(sparse_depth) & (sparse_depth > 0))
    info = {"num_pairs": int(m.sum()), "mode": "scale", "reason": ""}
    if int(m.sum()) < min_pairs:
        info["reason"] = f"点数不足 ({int(m.sum())} < {min_pairs})"
        return False, (1.0, 0.0), info

    ys, xs = np.nonzero(m)
    rho_p = 1.0 / depth[m].astype(np.float64)          # 先验逆深度
    rho_t = 1.0 / sparse_depth[m].astype(np.float64)   # 目标逆深度

    span = float(rho_p.max() - rho_p.min())
    rel_span = span / max(float(np.median(rho_p)), 1e-12)
    info["inv_span_rel"] = rel_span
    if rel_span < 0.05:
        info["reason"] = f"逆深度跨度太小 ({rel_span:.4f}) —— b 无法辨识"
        return False, (1.0, 0.0), info

    # 按空间块分 fit / held-out
    bid = (ys // block).astype(np.int64) * 100003 + (xs // block).astype(np.int64)
    ub = np.unique(bid)
    if ub.size < 4:
        info["reason"] = f"空间块太少 ({ub.size})"
        return False, (1.0, 0.0), info
    rs = np.random.default_rng(0)
    ho = set(rs.choice(ub, size=max(1, int(round(ub.size * holdout_frac))), replace=False).tolist())
    is_ho = np.array([b in ho for b in bid])
    fit, hold = ~is_ho, is_ho
    info["blocks"] = int(ub.size)
    info["n_fit"], info["n_hold"] = int(fit.sum()), int(hold.sum())
    if info["n_fit"] < min_pairs // 2 or info["n_hold"] < 20:
        info["reason"] = "分块后任一侧点数不足"
        return False, (1.0, 0.0), info

    # 迭代裁剪的最小二乘
    x, y = rho_p[fit], rho_t[fit]
    keep = np.ones_like(x, dtype=bool)
    a, b = 1.0, 0.0
    for _ in range(iters):
        A = np.stack([x[keep], np.ones(int(keep.sum()))], axis=1)
        cond = float(np.linalg.cond(A))
        if not np.isfinite(cond) or cond > 1e8:
            info["reason"] = f"条件数过大 ({cond:.3g})"
            return False, (1.0, 0.0), info
        a, b = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        r = y - (a * x + b)
        med = float(np.median(r[keep]))
        mad = float(np.median(np.abs(r[keep] - med)))
        sig = max(1.4826 * mad, 1e-12)
        nk = np.abs(r - med) <= trim_mad * sig
        if int(nk.sum()) < min_pairs // 2 or int(nk.sum()) == int(keep.sum()):
            break
        keep = nk
    info["cond"] = cond
    info["a"], info["b"] = float(a), float(b)

    if a <= 0:
        info["reason"] = f"a <= 0 ({a:.4g}) —— 映射会翻转"
        return False, (1.0, 0.0), info
    # 全物理范围内必须映到正深度: rho in [1/dmax, 1/dmin], a>0 时最小值在 1/dmax
    if depth_min and depth_max:
        lo_rho = 1.0 / float(depth_max)
        if a * lo_rho + b <= 1e-9:
            info["reason"] = f"远端会映到非正深度 (a/dmax + b = {a * lo_rho + b:.3g})"
            return False, (1.0, 0.0), info

    # held-out 上跟 scale-only 比
    scale = float(np.median(sparse_depth[m] / depth[m]))
    z_aff = 1.0 / np.maximum(a * rho_p + b, 1e-9)
    z_sca = depth[m].astype(np.float64) * scale
    tgt = sparse_depth[m].astype(np.float64)
    for nm, sel in (("fit", fit), ("hold", hold)):
        for lbl, z in (("affine", z_aff), ("scale", z_sca)):
            e = np.abs(z[sel] - tgt[sel])
            info[f"{nm}_{lbl}_median"] = float(np.median(e))
            info[f"{nm}_{lbl}_mean"] = float(e.mean())
            info[f"{nm}_{lbl}_p95"] = float(np.quantile(e, 0.95))
    better = info["hold_affine_median"] < info["hold_scale_median"]
    info["scale_only"] = scale
    if not better:
        info["reason"] = (f"held-out 未改善 (affine {info['hold_affine_median']:.3f} "
                          f"vs scale {info['hold_scale_median']:.3f})")
        return False, (1.0, 0.0), info
    info["mode"] = "affine"
    return True, (float(a), float(b)), info


def apply_affine_inverse(depth, a: float, b: float):
    """``z -> 1 / (a/z + b)``, 非有限或非正的结果置 0 (下游按无效处理)。"""
    d = np.asarray(depth, np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(d > 0, 1.0 / d.astype(np.float64), np.nan)
        out = 1.0 / (a * rho + b)
    out = np.where(np.isfinite(out) & (out > 0), out, 0.0)
    return out.astype(np.float32)


def calibrate_depth_to_metric(sample, depth, ref_idx: int = 0, config: SfMConfig | None = None,
                              mode: str = "scale", depth_min: float | None = None,
                              depth_max: float | None = None):
    """Rescale ``depth`` to metric using the sample's SfM sparse depth.

    Prefers ``sample["sfm_depth"]`` (precomputed/cropped by the dataset) and only
    falls back to running SfM when it is absent. ``depth`` must already be at the
    sample's reference-view resolution. Returns ``(depth_metric, scale, sfm_out)``.
    """
    cached = sample.get("sfm_depth") if hasattr(sample, "get") else None
    if cached is not None:
        sparse_depth = np.asarray(cached, np.float32)
        valid_mask = sparse_depth > 0
        sfm_out = {"sparse_depth": sparse_depth, "valid_mask": valid_mask, "info": {"source": "sample"}}
    else:
        sfm_out = generate_sparse_depth_from_sample(sample, ref_idx=ref_idx, config=config)
        sfm_out["info"]["source"] = "computed"

    scale, scale_info = metric_scale_from_sparse(depth, sfm_out["sparse_depth"], sfm_out["valid_mask"])
    sfm_out["info"]["scale"] = scale_info

    # mode="affine": 先尝试逆深度域 scale+shift, 任一守卫不过就静默退回 scale-only。
    # 默认仍是 "scale", 所以不显式打开时行为与旧缓存完全一致。
    if str(mode).lower() == "affine":
        ok, (a, b), aff_info = metric_affine_from_sparse(
            depth, sfm_out["sparse_depth"], sfm_out["valid_mask"],
            depth_min=depth_min, depth_max=depth_max)
        sfm_out["info"]["affine"] = aff_info
        if ok:
            return apply_affine_inverse(depth, a, b), scale, sfm_out

    return (np.asarray(depth, np.float32) * scale).astype(np.float32), scale, sfm_out
