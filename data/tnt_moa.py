"""Tanks-and-Temples scene for MoAMVSNet inference (no ground truth).

    <root>/<split>/<Scene>/images/{view:08d}.jpg
    <root>/<split>/<Scene>/cams_1/{view:08d}_cam.txt     (line 12 = depth_min depth_max)
    <root>/<split>/<Scene>/pair.txt

Line 12 of the T&T cameras is (min, MAX), not DTU's (min, INTERVAL): read_cam
branches on the field count. The frame is resized by ``resize_scale`` and then
centre-cropped to a multiple of 8 (1920x1080 already is one). The DA3 map comes
from scripts/build_da3_cache_mvs.py --dataset tnt and goes through the same
resize + crop.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.blended_moa import read_cam, read_pair

STRIDE = 8


class MoATnTScene(Dataset):
    def __init__(self, root, scene: str, nviews: int = 5, resize_scale: float = 1.0,
                 da3_root=None, max_refs: int = 0, fusion_src_views: int = 10) -> None:
        self.scene = scene                              # "<split>/<Scene>"
        self.dir = Path(root) / scene
        self.nviews, self.resize_scale = int(nviews), float(resize_scale)
        self.da3_dir = Path(da3_root) / scene if da3_root is not None else None
        self.fusion_src_views = int(fusion_src_views)
        have = {int(f.stem) for f in (self.dir / "images").glob("*.jpg") if f.stem.isdigit()}
        self.metas = []
        for ref, srcs in read_pair(self.dir / "pair.txt"):
            srcs = [s for s in srcs if s in have]
            if ref in have and srcs:
                self.metas.append((ref, srcs))
        if max_refs:
            self.metas = self.metas[:max_refs]
        if not self.metas:
            raise RuntimeError(f"{self.dir}: no usable reference view")
        self.da3_process_res = None
        if self.da3_dir is not None:
            miss = [r for r, _ in self.metas if not self.da3_file(r).is_file()]
            if miss:
                raise FileNotFoundError(f"{scene}: {len(miss)} views have no DA3 cache under {self.da3_dir} "
                                        f"(e.g. {miss[:5]}) — run scripts/build_da3_cache_mvs.py --dataset tnt")
            with np.load(self.da3_file(self.metas[0][0])) as z:
                self.da3_process_res = int(z["process_res"])

    def __len__(self) -> int:
        return len(self.metas)

    def da3_file(self, view: int) -> Path:
        return self.da3_dir / f"da3_{view:08d}.npz"

    def _geometry(self, h0: int, w0: int):
        h1, w1 = int(round(h0 * self.resize_scale)), int(round(w0 * self.resize_scale))
        ch, cw = (h1 // STRIDE) * STRIDE, (w1 // STRIDE) * STRIDE
        return h1, w1, (h1 - ch) // 2, (w1 - cw) // 2, ch, cw

    def _load_view(self, view: int):
        img = np.asarray(Image.open(self.dir / "images" / f"{view:08d}.jpg").convert("RGB"))
        K, E, dmin, dmax = read_cam(self.dir / "cams_1" / f"{view:08d}_cam.txt")
        h0, w0 = img.shape[:2]
        h1, w1, y0, x0, ch, cw = self._geometry(h0, w0)
        K = K.copy()
        if (h1, w1) != (h0, w0):
            img = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_AREA)
            K[0, :] *= w1 / w0
            K[1, :] *= h1 / h0
        img = img[y0:y0 + ch, x0:x0 + cw]
        K[0, 2] -= x0
        K[1, 2] -= y0
        return img, K, E, dmin, dmax, (h0, w0)

    def __getitem__(self, idx: int) -> dict:
        ref, srcs = self.metas[idx]
        views = [ref] + srcs[: self.nviews - 1]
        # fewer sources than nviews-1: repeat the best one (same rule as BlendedMVS)
        views += [srcs[0]] * (self.nviews - len(views))
        imgs, Ks, Es = [], [], []
        dmin = dmax = hw0 = None
        for i, v in enumerate(views):
            img, K, E, d0, d1, shp = self._load_view(v)
            if i == 0:
                dmin, dmax, hw0 = d0, d1, shp
            imgs.append(img)
            Ks.append(K)
            Es.append(E)
        sample = {
            "scan": self.scene, "ref_view": int(ref),
            "images": torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float(),
            "intrinsics": np.stack(Ks).astype(np.float32),
            "extrinsics": np.stack(Es).astype(np.float32),
            "depth_values": np.linspace(dmin, dmax, 192, dtype=np.float32),
            "src_views": np.asarray(srcs[: self.fusion_src_views], dtype=np.int64),
        }
        if self.da3_dir is not None:
            with np.load(self.da3_file(ref)) as z:
                d = np.asarray(z["depth"], dtype=np.float32)
            h0, w0 = hw0
            if d.shape != (h0, w0):
                d = cv2.resize(d, (w0, h0), interpolation=cv2.INTER_NEAREST)
            d[~np.isfinite(d) | (d <= 0)] = 0.0
            h1, w1, y0, x0, ch, cw = self._geometry(h0, w0)
            if (h1, w1) != (h0, w0):
                d = cv2.resize(d, (w1, h1), interpolation=cv2.INTER_NEAREST)
            sample["mono_depth"] = np.ascontiguousarray(d[y0:y0 + ch, x0:x0 + cw])
        return sample
