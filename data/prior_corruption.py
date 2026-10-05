"""Synthetic prior-failure augmentation.

The cached VGGT/DA3/norm-fill priors are *mostly right* on the training set, so
a network trained on them alone learns the local-branch shortcut and the global
guard's rescue path is never exercised. This module injects the failure modes
actually observed in the real pipeline — edge fill ramps, fly-points, local
scale/bias drift, holes, confidently-wrong regions — into a fraction of
training samples. GT is never touched, so the unified stage-1 loss then
actively teaches: press the wrong local candidates down, find GT among the
global bins, keep a correction-sized stage-2 range.

All ops are numpy/cv2 on the cropped [H, W] prior; returns the corrupted
(depth, conf) plus a bool mask of materially-changed pixels so training can
report the rescue rate (err on corrupted vs clean pixels) separately.
"""

from __future__ import annotations

import cv2
import numpy as np


def _valid(depth: np.ndarray) -> np.ndarray:
    return np.isfinite(depth) & (depth > 0)


def _rand_blob_mask(shape: tuple[int, int], rng: np.random.Generator,
                    frac_lo: float = 0.02, frac_hi: float = 0.15) -> np.ndarray:
    """Random smooth elliptical blob covering roughly frac of the image."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    area_frac = rng.uniform(frac_lo, frac_hi)
    radius = int(np.sqrt(area_frac * h * w / np.pi))
    cx = rng.integers(radius, max(w - radius, radius + 1))
    cy = rng.integers(radius, max(h - radius, radius + 1))
    ax = max(int(radius * rng.uniform(0.6, 1.6)), 4)
    ay = max(int(radius * rng.uniform(0.6, 1.6)), 4)
    angle = float(rng.uniform(0, 180))
    cv2.ellipse(mask, (int(cx), int(cy)), (ax, ay), angle, 0, 360, 1, thickness=-1)
    return mask.astype(bool)


def _edge_band(depth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Dilated band around depth discontinuities of the prior itself."""
    v = _valid(depth)
    d = np.where(v, depth, 0.0).astype(np.float32)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    scale = np.median(depth[v]) if v.any() else 1.0
    band = grad > 0.02 * max(scale, 1.0)
    k = int(rng.integers(2, 6))
    band = cv2.dilate(band.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
    return band & v


def _op_edge_ramp(depth, conf, rng):
    """Norm-fill style fore/background ramps: heavy blur inside the edge band,
    reported with HIGH confidence (the nastiest observed failure)."""
    band = _edge_band(depth, rng)
    if not band.any():
        return depth, conf, band
    ksize = int(rng.integers(4, 11)) * 2 + 1
    blurred = cv2.GaussianBlur(depth, (ksize, ksize), 0)
    depth = np.where(band, blurred, depth)
    conf = np.where(band, np.maximum(conf, rng.uniform(0.6, 0.9)), conf)
    return depth, conf, band


def _op_fly_points(depth, conf, rng):
    """Isolated spikes on surfaces (VGGT residual noise after denoising)."""
    v = _valid(depth)
    frac = rng.uniform(0.001, 0.01)
    hit = (rng.random(depth.shape) < frac) & v
    if not hit.any():
        return depth, conf, hit
    mag = rng.uniform(20.0, 150.0, size=depth.shape).astype(np.float32)
    sign = np.where(rng.random(depth.shape) < 0.5, -1.0, 1.0).astype(np.float32)
    depth = np.where(hit, np.maximum(depth + sign * mag, 1.0), depth)
    conf = np.where(hit, np.maximum(conf, rng.uniform(0.5, 0.9)), conf)
    return depth, conf, hit


def _op_region_drift(depth, conf, rng):
    """Local scale/bias drift with UNCHANGED confidence: confidently wrong."""
    blob = _rand_blob_mask(depth.shape, rng) & _valid(depth)
    if not blob.any():
        return depth, conf, blob
    if rng.random() < 0.5:
        factor = 1.0 + float(rng.uniform(0.03, 0.12)) * (1 if rng.random() < 0.5 else -1)
        depth = np.where(blob, depth * factor, depth)
    else:
        bias = float(rng.uniform(15.0, 60.0)) * (1 if rng.random() < 0.5 else -1)
        depth = np.where(blob, np.maximum(depth + bias, 1.0), depth)
    return depth, conf, blob


def _op_block_missing(depth, conf, rng):
    """Contiguous prior dropout (denoising removed a whole region)."""
    blob = _rand_blob_mask(depth.shape, rng, 0.02, 0.10)
    depth = np.where(blob, 0.0, depth)
    conf = np.where(blob, 0.0, conf)
    return depth, conf, blob


def _op_wrong_high_conf(depth, conf, rng):
    """Perturb depth inside a blob and *raise* its confidence."""
    blob = _rand_blob_mask(depth.shape, rng, 0.01, 0.08) & _valid(depth)
    if not blob.any():
        return depth, conf, blob
    noise = rng.normal(0.0, rng.uniform(10.0, 30.0), size=depth.shape).astype(np.float32)
    depth = np.where(blob, np.maximum(depth + noise, 1.0), depth)
    conf = np.where(blob, np.maximum(conf, 0.9), conf)
    return depth, conf, blob


def _op_edge_mixing(depth, conf, rng):
    """3x3 fore/background mixing at edges (min/max filter in the band)."""
    band = _edge_band(depth, rng)
    if not band.any():
        return depth, conf, band
    kernel = np.ones((3, 3), np.uint8)
    mixed = cv2.erode(depth, kernel) if rng.random() < 0.5 else cv2.dilate(depth, kernel)
    depth = np.where(band, mixed, depth)
    return depth, conf, band


_OPS = (
    _op_edge_ramp,
    _op_fly_points,
    _op_region_drift,
    _op_block_missing,
    _op_wrong_high_conf,
    _op_edge_mixing,
)


def corrupt_prior(
    depth_prior: np.ndarray,
    conf_prior: np.ndarray,
    sample_prob: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply 1-3 random failure modes with probability ``sample_prob``.

    Returns (depth, conf, corrupt_mask). corrupt_mask marks pixels whose depth
    changed by >1mm or whose validity/confidence was materially altered — the
    denominator of the training-time rescue-rate diagnostic.
    """
    rng = rng or np.random.default_rng()
    depth = depth_prior.astype(np.float32).copy()
    conf = conf_prior.astype(np.float32).copy()
    corrupt = np.zeros(depth.shape, dtype=bool)

    if float(rng.random()) >= sample_prob or not _valid(depth).any():
        return depth, conf, corrupt

    before = depth.copy()
    n_ops = int(rng.integers(1, 4))
    ops = rng.choice(len(_OPS), size=n_ops, replace=False)
    for i in ops:
        depth, conf, touched = _OPS[i](depth, conf, rng)
        corrupt |= touched

    corrupt &= (np.abs(depth - before) > 1.0) | (_valid(before) != _valid(depth))
    return depth, conf, corrupt
