"""BlendedMVS (low-res, 576x768) dataset for MoAMVSNet fine-tuning.

    <root>/<scene>/blended_images/{view:08d}.jpg      (the *_masked.jpg twins are not used)
    <root>/<scene>/cams/{view:08d}_cam.txt
    <root>/<scene>/cams/pair.txt
    <root>/<scene>/rendered_depth_maps/{view:08d}.pfm

Scene folders are named by their 24-hex-digit object id (``5a3ca9cb270f0e3f14d0eddb``,
``000000000000000000000001``, ...). That *is* the official naming — the scene
lists of every BlendedMVS repo refer to scenes by it — so it must not be renamed.

Everything after the file reading is ``MoADTUDataset``'s: the same per-sample
seeding, multi-scale buckets, photometric augmentation, crop and DA3 alignment.
Differences from DTU:

* one lighting per scene (``light_idx`` is fixed to 0 and never used on disk);
* no mask file — valid GT is ``depth > 0``;
* line 12 of ``_cam.txt`` is ``depth_min interval num depth_max``; parsed by
  field count (see :func:`read_cam`);
* hypotheses follow MVSFormer++'s blended_dataset_ms.py: interval =
  num * interval_cam / 192 * 1.06, values = arange(min, min + interval * 191.5)
  — the camera's range stretched by the same 1.06 as DTU;
* depth units are per-scene arbitrary. Every sample carries ``metric_scale =
  1 / interval`` and the training metrics multiply errors by it, i.e. Blended
  errors are counted in hypothesis intervals — MVSFormer++'s Blended validation
  convention (its "thres2mm" = 2 intervals). DTU samples keep 1 (mm; MVSFormer++
  divides DTU's 2.65mm interval by 2.65). Network and loss are unaffected;
* pair.txt may list fewer than ``nviews-1`` sources; the best one is repeated
  (MVSFormer++ does the same). A reference without any source is dropped.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from data.augment import PhotometricAug, resize_scale_for_crop
from data.dtu_moa import MoADTUDataset
from data.io import read_pfm

BLENDED_HW = (576, 768)
# Multi-scale crops for 576x768 frames: DTU's list tops out at 640x896, which a
# 576x768 source cannot provide without upsampling.
BLENDED_SCALES = ((384, 512), (448, 576), (448, 640), (512, 640), (512, 704), (576, 768))


def read_cam(filename) -> tuple[np.ndarray, np.ndarray, float, float]:
    """MVSNet ``_cam.txt`` -> (K, E, depth_min, depth_max), branching on line 12's field count.

        ``min interval num max``  BlendedMVS / MVSNet     -> max = field 4
        ``min interval num``                              -> max = min + interval * num
        ``min max``               T&T (cams_1)            -> max = field 2

    Two fields are read as (min, MAX): that is the T&T convention. DTU's two-field
    (min, INTERVAL) cameras must keep going through data/dtu.py — this reader
    rejects them only when the result is not increasing.
    """
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    E = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape((4, 4))
    K = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ").reshape((3, 3))
    f12 = lines[11].split()
    dmin = float(f12[0])
    if len(f12) >= 4:
        dmax = float(f12[3])
    elif len(f12) == 3:
        dmax = dmin + float(f12[1]) * float(f12[2])
    elif len(f12) == 2:
        dmax = float(f12[1])
    else:
        raise ValueError(f"{filename}: cannot parse the depth range from {lines[11]!r}")
    if not (np.isfinite(dmin) and np.isfinite(dmax)) or dmax <= dmin or dmin <= 0:
        raise ValueError(f"{filename}: depth range ({dmin}, {dmax}) from {lines[11]!r} is not valid")
    return K, E, dmin, dmax


def read_pair(filename) -> list[tuple[int, list[int]]]:
    with open(filename) as f:
        lines = [line.strip() for line in f if line.strip()]
    n = int(lines[0])
    out = []
    for i in range(n):
        ref = int(lines[1 + 2 * i])
        out.append((ref, [int(x) for x in lines[2 + 2 * i].split()[1::2]]))
    return out


def read_scene_list(path) -> list[str]:
    with open(path) as f:
        return [s.strip() for s in f if s.strip() and not s.lstrip().startswith("#")]


def blended_interval(cam_file, ndepths: int = 192, interval_scale: float = 1.06) -> float:
    """MVSFormer++ blended_dataset_ms.read_cam_file: (num * interval) / ndepths * 1.06."""
    with open(cam_file) as f:
        f12 = f.read().splitlines()[11].split()
    interval = float(f12[1])
    if len(f12) >= 3:
        interval = int(float(f12[2])) * interval / ndepths
    return interval * interval_scale


def blended_da3_file(root: Path, scene: str, view: int) -> Path:
    return Path(root) / scene / f"da3_{view:08d}.npz"


class MoABlendedDataset(MoADTUDataset):
    native_hw = BLENDED_HW

    def __init__(self, datapath, listfile, **kwargs) -> None:
        # val/test: the whole native frame, no resize
        kwargs.setdefault("height", BLENDED_HW[0])
        kwargs.setdefault("width", BLENDED_HW[1])
        super().__init__(datapath, listfile, **kwargs)
        self.resize_scale = 1.0

    # ---- listing ----------------------------------------------------------- #
    def build_list(self):
        metas, n_padded, n_dropped = [], 0, 0
        scenes = read_scene_list(self.listfile)
        for scene in scenes:
            for ref, srcs in read_pair(os.path.join(self.datapath, scene, "cams", "pair.txt")):
                if not srcs:
                    n_dropped += 1
                    continue
                if len(srcs) < self.nviews - 1:
                    n_padded += 1
                    srcs = srcs + [srcs[0]] * (self.nviews - 1 - len(srcs))
                metas.append((scene, 0, ref, srcs))
        print(f"dataset {self.mode}: {len(scenes)} scenes, metas: {len(metas)} "
              f"(padded {n_padded} refs with < {self.nviews - 1} sources, dropped {n_dropped} without any)")
        return metas

    def da3_file(self, scan: str, view: int, light: int) -> Path:
        return blended_da3_file(self.da3_root, scan, view)

    # ---- geometry ---------------------------------------------------------- #
    def sample_geometry(self, idx, rng=None):
        if not self.scales:
            return self.height, self.width, self.resize_scale
        crop_h, crop_w = self.scales[self._barrel.get(int(idx), int(idx)) % len(self.scales)]
        lo, hi = self.resize_range
        draw = rng.random() if rng is not None else np.random.rand()
        enlarge = lo + float(draw) * (hi - lo)
        return crop_h, crop_w, resize_scale_for_crop(crop_h, crop_w, *BLENDED_HW, enlarge)

    def precrop_inputs(self, idx, resize_scale=None, aug_params=None, load_src_depth=None):
        scene, _light, ref_view, src_views = self.metas[idx]
        root = Path(self.datapath) / scene
        resize_scale = self.resize_scale if resize_scale is None else resize_scale
        imgs, Ks, Es = [], [], []
        depth_hr = mask_hr = depth_values = None
        dmin = dmax = None
        for i, v in enumerate([ref_view] + src_views[: self.nviews - 1]):
            img = np.asarray(Image.open(root / "blended_images" / f"{v:08d}.jpg").convert("RGB"))
            if aug_params is not None:
                img = PhotometricAug.apply(img, aug_params)
            K, E, d0, d1 = read_cam(root / "cams" / f"{v:08d}_cam.txt")
            depth = None
            if i == 0:
                dmin, dmax = d0, d1
                interval = blended_interval(root / "cams" / f"{v:08d}_cam.txt", self.ndepths)
                depth = np.asarray(read_pfm(str(root / "rendered_depth_maps" / f"{v:08d}.pfm")),
                                   dtype=np.float32)
                depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)
            if img.shape[:2] != BLENDED_HW:
                # the low-res release is 576x768 throughout; anything else is brought
                # to that frame so the scale plan and the DA3 cache stay aligned
                K = K.copy()
                K[0, :] *= BLENDED_HW[1] / img.shape[1]
                K[1, :] *= BLENDED_HW[0] / img.shape[0]
                img = cv2.resize(img, BLENDED_HW[::-1], interpolation=cv2.INTER_AREA)
                if depth is not None:
                    depth = cv2.resize(depth, BLENDED_HW[::-1], interpolation=cv2.INTER_NEAREST)
            if i == 0:
                mask = (depth > 0).astype(np.float32)
                if resize_scale != 1.0:
                    img, depth, K, mask = self.pre_resize(img, depth, K, mask, resize_scale)
                depth_hr, mask_hr = depth, mask
            elif resize_scale != 1.0:
                img, _, K, _ = self.pre_resize(img, None, K, None, resize_scale)
            imgs.append(img)
            Ks.append(np.asarray(K, np.float32))
            Es.append(np.asarray(E, np.float32))
        depth_values = np.arange(dmin, dmin + interval * (self.ndepths - 0.5), interval, dtype=np.float32)
        return {"views_np": imgs, "intrinsics": np.stack(Ks), "extrinsics": np.stack(Es),
                "depth_hr": depth_hr, "mask_hr": mask_hr, "depth_values": depth_values,
                "scan": scene, "ref_view": ref_view, "light_idx": 0}

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        dv = sample["depth_values"]
        sample["metric_scale"] = np.asarray(1.0 / float(dv[1] - dv[0]), dtype=np.float32)
        return sample


class BalancedMixDataset(torch.utils.data.Dataset):
    """DTU + BlendedMVS for balanced fine-tuning (MVSFormer++ ``--balanced_training``).

    Each epoch draws ``min(len)`` samples from every child (a fresh random subset
    of the larger ones) and shuffles them together, so the two datasets contribute
    equally no matter how large each is. Batches are mixed; that works because
    both children use the same multi-scale list (BLENDED_SCALES — DTU can supply
    any crop up to its frame, Blended cannot exceed 576x768) and the scale plan
    assigns one crop size per batch across children.

    DTU samples get ``metric_scale = 1`` (already mm); Blended ones carry their own.
    """

    def __init__(self, children: dict, seed: int) -> None:
        self.names = list(children)
        self.children = [children[n] for n in self.names]
        scales = {tuple(c.scales) for c in self.children}
        if len(scales) != 1:
            raise ValueError(f"children need one shared scale list for mixed batches, got {scales}")
        self.offsets = np.cumsum([0] + [len(c) for c in self.children]).tolist()
        self.seed = int(seed)
        self.per_child = min(len(c) for c in self.children)
        res = {n: c.da3_process_res for n, c in children.items()}
        self.da3_process_res = res
        print(f"[mix] {', '.join(f'{n}={len(c)}' for n, c in children.items())} -> "
              f"{self.per_child} per dataset per epoch (DA3 process_res {res})")

    def __len__(self) -> int:
        return self.offsets[-1]

    def _locate(self, idx: int) -> tuple[int, int]:
        k = int(np.searchsorted(self.offsets, idx, side="right")) - 1
        return k, idx - self.offsets[k]

    def __getitem__(self, idx):
        k, j = self._locate(int(idx))
        s = self.children[k][j]
        s.setdefault("metric_scale", np.asarray(1.0, dtype=np.float32))    # DTU: already mm
        s["dataset"] = self.names[k]
        return s

    def set_epoch(self, epoch: int) -> None:
        for c in self.children:
            c.set_epoch(epoch)

    def reset_scale_plan(self, order, batch_size: int) -> None:
        plans = [dict() for _ in self.children]
        for i, g in enumerate(order):
            k, j = self._locate(int(g))
            plans[k][j] = i // max(batch_size, 1)
        for c, p in zip(self.children, plans):
            c._barrel = p

    def make_sampler(self, seed: int) -> "BalancedEpochSampler":
        return BalancedEpochSampler(self, seed)


class BalancedEpochSampler(torch.utils.data.Sampler):
    def __init__(self, ds: BalancedMixDataset, seed: int) -> None:
        self.ds, self.seed = ds, int(seed)
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        rng = np.random.default_rng(self.seed + int(epoch))
        parts = [o + rng.permutation(n)[: self.ds.per_child]
                 for o, n in zip(self.ds.offsets[:-1], np.diff(self.ds.offsets))]
        self.order = rng.permutation(np.concatenate(parts)).tolist()

    def __iter__(self):
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)
