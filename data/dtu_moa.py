"""DTU dataset for MoAMVSNet: no prior cache, plus the DA3 monocular depth.

Reuses ``DTUMVSDataset``'s pairing, resize, crop and per-sample seeding, but
replaces ``__getitem__`` — the parent always loads the old VGGT/DA3 prior cache,
which the MoA network never uses. The DA3 map (raw relative depth, cached at
native 1200x1600 by scripts/build_da3_cache_all.py) goes through exactly the
same resize and crop as the images, so it stays pixel-aligned with them.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from data.dtu import DTUMVSDataset

NATIVE_HW = (1200, 1600)


def da3_cache_file(root: Path, scan: str, view: int, light: int) -> Path:
    return Path(root) / scan / f"da3_{view:04d}_{light}.npz"


class MoADTUDataset(DTUMVSDataset):
    def __init__(self, datapath, listfile, *, da3_root=None, load_mono: bool = True,
                 da3_missing: str = "error", **kwargs) -> None:
        kwargs.setdefault("exclude_file", None)
        super().__init__(datapath, listfile, **kwargs)
        # lists/dtu/test.txt starts with a blank line -> metas with an empty scan
        self.metas = [m for m in self.metas if m[0]]
        self.load_mono = bool(load_mono)
        self.da3_root = Path(da3_root) if da3_root is not None else None
        self.da3_process_res: int | None = None
        if self.load_mono:
            if self.da3_root is None:
                raise ValueError("load_mono=True needs da3_root")
            self._check_da3(da3_missing)

    def _check_da3(self, policy: str) -> None:
        if policy not in ("error", "skip"):
            raise ValueError(f"da3_missing must be error/skip, got {policy!r}")
        keep, missing = [], []
        for m in self.metas:
            scan, light, ref, _ = m
            (keep if self.da3_file(scan, ref, light).is_file() else missing).append(m)
        if missing:
            ex = ", ".join(f"{s} v{r} l{l}" for s, l, r, _ in missing[:5])
            msg = (f"dataset {self.mode}: {len(missing)}/{len(self.metas)} samples have no DA3 cache "
                   f"under {self.da3_root} (e.g. {ex})")
            if policy == "error":
                raise FileNotFoundError(msg + " — build/download it or use --da3-missing skip")
            print(f"[data] {msg}; skipped")
            self.metas = keep
        if not self.metas:
            raise RuntimeError(f"dataset {self.mode}: no samples left with a DA3 cache in {self.da3_root}")
        scan, light, ref, _ = self.metas[0]
        with np.load(self.da3_file(scan, ref, light)) as z:
            if "process_res" in z.files:
                self.da3_process_res = int(np.asarray(z["process_res"]).reshape(-1)[0])
        print(f"[data] {self.mode}: {len(self.metas)} samples, DA3 cache {self.da3_root} "
              f"(process_res={self.da3_process_res})")

    # Frame size the DA3 cache is stored at and the images are resized from.
    native_hw: tuple[int, int] = NATIVE_HW
    # Training-time source selection (MVSFormer++ datasets/*_ms.py): 0 = first nviews-1
    # of pair.txt (moa1's DTU training); k > 0 = random nviews-1 out of the top k;
    # -1 = random out of all. Drawn from the per-sample rng, so still reproducible.
    src_shuffle_top: int = 0
    # Re-draw a random crop up to this many times while its 1/8-resolution GT mask is
    # empty (MVSFormer++ loops until non-empty; BlendedMVS has sky-only crops).
    crop_retry: int = 0

    def _pick_sources(self, srcs: list, rng) -> list:
        k = self.src_shuffle_top
        if self.mode != "train" or k == 0:
            return srcs
        pool = list(srcs if k < 0 else srcs[:k])
        rng.shuffle(pool)
        return pool + [v for v in srcs if v not in pool]

    def da3_file(self, scan: str, view: int, light: int) -> Path:
        return da3_cache_file(self.da3_root, scan, view, light)

    def _load_mono(self, scan: str, view: int, light: int, resize_scale: float) -> np.ndarray:
        with np.load(self.da3_file(scan, view, light)) as z:
            d = np.asarray(z["depth"], dtype=np.float32)
        nh, nw = self.native_hw
        if d.shape != (nh, nw):
            d = cv2.resize(d, (nw, nh), interpolation=cv2.INTER_NEAREST)
        # invalid -> 0 *before* any resampling, so a negative or NaN never gets
        # blended into a plausible-looking near depth
        d[~np.isfinite(d) | (d <= 0)] = 0.0
        if resize_scale != 1.0:
            h, w = d.shape
            d = cv2.resize(d, (int(w * resize_scale), int(h * resize_scale)),
                           interpolation=cv2.INTER_NEAREST)
        return d

    def __getitem__(self, idx):
        rng = self._rng(idx)
        crop_h, crop_w, resize_scale = self.sample_geometry(idx, rng)
        aug_params = self.aug.draw(rng) if self.aug is not None else None
        meta = self.metas[idx]
        picked = self._pick_sources(meta[3], rng)
        # precrop_inputs reads the sources from self.metas; swap in the draw for this call only
        self.metas[idx] = (*meta[:3], picked)
        try:
            pc = self.precrop_inputs(idx, resize_scale=resize_scale, aug_params=aug_params,
                                     load_src_depth=False)
        finally:
            self.metas[idx] = meta
        imgs_np, Ks, Es = pc["views_np"], pc["intrinsics"], pc["extrinsics"]
        h0, w0 = imgs_np[0].shape[:2]
        crop_x, crop_y = self.pick_crop_origin(h0, w0, crop_h, crop_w, rng=rng)
        if self.random_crop and pc["mask_hr"] is not None:
            for _ in range(self.crop_retry):
                if (pc["mask_hr"][crop_y:crop_y + crop_h:8, crop_x:crop_x + crop_w:8] > 0).any():
                    break
                crop_x, crop_y = self.pick_crop_origin(h0, w0, crop_h, crop_w, rng=rng)

        images, intrinsics = [], []
        depth_gt = mask_gt = None
        for i in range(len(imgs_np)):
            img, K, depth, mask = self.crop_at(
                imgs_np[i], Ks[i], crop_x, crop_y,
                pc["depth_hr"] if i == 0 else None, pc["mask_hr"] if i == 0 else None,
                crop_h=crop_h, crop_w=crop_w)
            if i == 0:
                depth_gt, mask_gt = depth, mask
            images.append(img)
            intrinsics.append(K)

        scan, light, ref, _ = meta
        src_views = picked[: self.nviews - 1]
        sample = {
            "sample_index": int(idx),
            "scan": str(scan),
            "ref_view": int(ref),
            "light_idx": int(light),
            "src_views": [int(v) for v in src_views],
            "crop_xy": np.asarray([crop_x, crop_y], dtype=np.int32),
            "crop_hw": np.asarray([crop_h, crop_w], dtype=np.int32),
            "resize_scale": np.asarray(resize_scale, dtype=np.float32),
            "images": torch.from_numpy(np.stack(images, axis=0)).permute(0, 3, 1, 2).float(),
            "intrinsics": np.stack(intrinsics, axis=0).astype(np.float32),
            "extrinsics": np.stack(Es, axis=0).astype(np.float32),
            "depth_gt": np.ascontiguousarray(depth_gt, dtype=np.float32),
            "mask": np.ascontiguousarray(mask_gt, dtype=np.float32),
            "depth_values": pc["depth_values"],
        }
        if self.load_mono:
            mono = self._load_mono(scan, ref, light, resize_scale)
            if mono.shape != (h0, w0):
                raise RuntimeError(f"DA3 map {mono.shape} vs image {(h0, w0)} for {scan} v{ref} l{light}")
            sample["mono_depth"] = np.ascontiguousarray(
                mono[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], dtype=np.float32)
        return sample
