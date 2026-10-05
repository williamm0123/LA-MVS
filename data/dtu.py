from __future__ import annotations
from pathlib import Path
from typing import Iterator
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from .io import read_pfm
import models.sfm as sfm
from models import pre_prior
from base.config import ProjectPaths
from .augment import PhotometricAug, resize_scale_for_crop
from .prior_corruption import corrupt_prior

class DTUMVSDataset(Dataset):
    def __init__(
        self, datapath, listfile, nviews = 3, ndepths = 192, mode= "train",random_crop = False,
        resize_scale = 1.0,
        **kwargs):
        
        self.datapath = Path(datapath)
        self.listfile = listfile
        self.ndepths = ndepths
        self.nviews = nviews
        self.mode = mode
        # 逐 (scan, ref_view, light) 排除表, 由 scripts/audit_prior_cache.py 产出。
        # 按整个 scan 剔除太粗: 全量审计里 51 个 scan 至少有一个坏 prior, 但多数
        # 只有 1-3%。这里做精确过滤, *_clean.txt 只负责剔掉重灾区 scan。
        # 必须在 build_list() 之前赋值。
        self.exclude_file = kwargs.get('exclude_file', None)

        self.metas = self.build_list()
        self.resize_scale = kwargs.get('resize_scale', 0.5)
        self.height = kwargs.get('height', 512)
        self.width  = kwargs.get('width', 640)
        self.sfm_cache_dir = Path(ProjectPaths().sfm_cache_path)
        self.prior_cache_dir = Path(ProjectPaths().prior_cache_path)
        # 训练默认随机裁剪做增广, 其余模式居中裁剪保证可复现; 可用 kwarg 覆盖
        self.random_crop = kwargs.get('random_crop', mode == 'train')
        # prior 失败模式增强 (仅训练): 见 data/prior_corruption.py。默认关闭,
        # train.py 从 cfg.train.prior_corruption_prob 传入。
        self.prior_corruption_prob = float(kwargs.get('prior_corruption_prob', 0.0)) \
            if mode == 'train' else 0.0
        # src_weights 是离线 SfM 按 src 视角数算出的变长向量; 当离线缓存与训练
        # nviews 不一致时长度对不上, collate 会报错。默认忽略 (网络 use_src_weights
        # 也默认 False), 需要时显式打开并保证缓存 view 数一致。
        self.use_src_weights = bool(kwargs.get('use_src_weights', False))
        # W3-B: 是否连源视图的 GT 深度/mask 一起读。默认 False —— 打开等于每样本
        # 多读 N-1 张 PFM + mask, 而且它们必须跑**完全一致**的 resize / crop /
        # 增强路径, 否则遮挡标签会和 warp 出来的几何对不上。
        self.load_src_depth = bool(kwargs.get('load_src_depth', False))

        # --- 逐样本确定性随机 ---
        # 增强、多尺度 resize、随机裁剪、prior 腐蚀原本各自调 np.random / 无种子
        # default_rng(), 于是同一份配置跑两次的增广序列不同。实测同配置两次跑的
        # val abs_err 单点差异达 0.109mm, 而消融要分辨的效应就是 0.17mm 量级 ——
        # 不播种就无法做单变量比较。这里用 SeedSequence([seed, epoch, idx]) 给每个
        # 样本生成独立且可复现的 Generator, 四个随机入口共用它。
        self.base_seed = int(kwargs.get('seed', 20260526))
        self.epoch = 0

        # --- 训练增强 (仅 train; val/test 必须确定性, 否则 best.pth 不可比) ---
        # aug: data.augment.PhotometricAug 或 None
        # scales / resize_range: 多尺度; 空 scales = 关闭, 固定 height x width
        self.aug = kwargs.get('aug') if mode == 'train' else None
        self.scales = tuple(kwargs.get('scales') or ()) if mode == 'train' else ()
        self.resize_range = tuple(kwargs.get('resize_range', (1.0, 1.0)))
        # idx -> barrel id; 同一个 batch 的样本共用一个 barrel, 因而共用一个尺度
        # (collate 要求 batch 内 H x W 一致)。由 reset_scale_plan 每个 epoch 重建。
        self._barrel: dict[int, int] = {}
        self.kwargs = kwargs
        # self.center_crop_size = kwargs.get('center_crop_size', None)
        # if mode != 'train':
        #     self.random_crop = False
        #     self.augment = False
        # else:
        #     self.random_crop = random_crop
        #     self.augment = augment
        
    def __iter__(self) -> Iterator:
        """
        显式实现迭代器协议，解决 Pylance 无法识别 enumerate 的报错
        """
        for i in range(len(self)):
            yield self[i]
    def _load_exclude(self) -> set:
        if not self.exclude_file or not os.path.exists(self.exclude_file):
            return set()
        import csv as _csv
        with open(self.exclude_file) as fh:
            rows = list(_csv.DictReader(fh))
        out = {(r["scan"], int(r["view"]), int(r["light"])) for r in rows}
        print(f"dataset {self.mode}: exclude list {self.exclude_file} -> {len(out)} 个坏 prior 样本")
        return out

    def build_list(self):
        metas = []
        excluded = self._load_exclude()
        n_dropped = 0
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    if self.mode == "train":
                        for light_idx in range(7):
                            if (scan, ref_view, light_idx) in excluded:
                                n_dropped += 1
                                continue
                            metas.append((scan, light_idx, ref_view, src_views))
                    else:
                        if (scan, ref_view, 3) in excluded:
                            n_dropped += 1
                            continue
                        metas.append((scan, 3, ref_view, src_views))
        if n_dropped:
            print(f"dataset {self.mode}: 剔除 {n_dropped} 个坏 prior 样本")
        print("dataset", self.mode, "metas:", len(metas))
        return metas
    def read_camera_file(self,filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape((4, 4))
        intrinsics = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ").reshape((3, 3))
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * 1.06

        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self,filename):
        img=Image.open(filename).convert('RGB')
        return img
    def __len__(self) -> int:
        return len(self.metas)

    def read_mask(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        # np_img = self.prepare_img(np_img)
        return np_img

    # def generate_stage_depth(self, depth):
    #     h, w = depth.shape
    #     depth_ms = {
    #         "stage1": cv2.resize(depth, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST),
    #         "stage2": cv2.resize(depth, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
    #         "stage3": cv2.resize(depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
    #         "stage4": depth
    #     }
    #     return depth_ms

    def pre_resize(self, img, depth, intrinsic, mask, resize_scale):
        ori_h, ori_w, _ = img.shape
        img = cv2.resize(img, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_AREA)
        h, w, _ = img.shape

        output_intrinsics = intrinsic.copy()
        output_intrinsics[0, :] *= resize_scale
        output_intrinsics[1, :] *= resize_scale

        if depth is not None:
            depth = cv2.resize(depth, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_NEAREST)

        if mask is not None:
            mask = cv2.resize(mask, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_NEAREST)

        return img, depth, output_intrinsics, mask

    def reset_scale_plan(self, order, batch_size: int) -> None:
        """按 sampler 这一 epoch 的取样顺序把样本分桶, 每 batch_size 个一桶。

        DataLoader 把 sampler 连续吐出的 batch_size 个索引组成一个 batch, 所以
        同桶 = 同 batch, 桶号决定尺度就保证了 batch 内分辨率一致 (collate 的硬
        要求)。这是 MVSFormer++ ``reset_dataset`` 的做法。DDP 下每个 rank 各自
        用自己那份索引建表, 但桶号是 rank 内的序号, 因此同一步各 rank 拿到同一
        个尺度, 不会出现负载倾斜。
        """
        if not self.scales:
            return
        self._barrel = {int(sid): i // max(batch_size, 1) for i, sid in enumerate(order)}

    def set_epoch(self, epoch: int) -> None:
        """每个 epoch 换一组样本级种子; 不调用则所有 epoch 复用同一组增广。"""
        self.epoch = int(epoch)

    def _rng(self, idx) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence([self.base_seed, self.epoch, int(idx)])
        )

    def sample_geometry(self, idx, rng: np.random.Generator | None = None):
        """该样本的 (crop_h, crop_w, resize_scale)。

        无多尺度时退化为固定的 self.height/width/resize_scale。
        """
        if not self.scales:
            return self.height, self.width, self.resize_scale
        crop_h, crop_w = self.scales[self._barrel.get(int(idx), int(idx)) % len(self.scales)]
        lo, hi = self.resize_range
        draw = rng.random() if rng is not None else np.random.rand()
        enlarge = lo + float(draw) * (hi - lo)
        # DTU Rectified_raw 恒为 1200x1600
        return crop_h, crop_w, resize_scale_for_crop(crop_h, crop_w, 1200, 1600, enlarge)

    def pick_crop_origin(self, h, w, crop_h=None, crop_w=None, rng: np.random.Generator | None = None):
        """选裁剪左上角 (x0, y0): 训练随机, 其余居中。同一 sample 内只调用一次,
        让 ref/src/depth/sfm_depth/mask 共用同一窗口。"""
        crop_h = self.height if crop_h is None else crop_h
        crop_w = self.width if crop_w is None else crop_w
        max_y, max_x = max(h - crop_h, 0), max(w - crop_w, 0)
        if self.random_crop:
            if rng is not None:
                return int(rng.integers(0, max_x + 1)), int(rng.integers(0, max_y + 1))
            return int(np.random.randint(0, max_x + 1)), int(np.random.randint(0, max_y + 1))
        return max_x // 2, max_y // 2

    def crop_at(self, img, intrinsic, x0, y0, depth=None, mask=None, crop_h=None, crop_w=None):
        """在给定左上角 (x0, y0) 处裁剪到 (crop_w, crop_h), 并平移主点。"""
        height = self.height if crop_h is None else crop_h
        width = self.width if crop_w is None else crop_w
        img = img[y0:y0 + height, x0:x0 + width]
        K = intrinsic.copy().astype(np.float32)
        K[0, 2] -= x0   # cx：裁剪只平移主点
        K[1, 2] -= y0   # cy：焦距 fx,fy 不变

        if depth is not None:
            depth = depth[y0:y0 + height, x0:x0 + width]
        if mask is not None:
            mask = mask[y0:y0 + height, x0:x0 + width]

        return img, K, depth, mask

    def prior_cache_path_for(self, idx):
        """Prior cache file for a meta (mirrors the SfM cache naming)."""
        scan, light_idx, ref_view, _ = self.metas[idx]
        return self.prior_cache_dir / scan / f"prior_{ref_view:0>4}_{light_idx}.npz"

    def precrop_inputs(self, idx, resize_scale=None, aug_params=None, load_src_depth=None):
        """Pre-crop multi-view inputs (before random crop), shared by both
        __getitem__ and the offline prior precompute so the two stay aligned.

        ``resize_scale`` / ``aug_params`` default to off so the prior builder
        keeps producing one deterministic cache entry per (scan, view) — only
        __getitem__ varies them per sample.
        """
        scan, light_idx, ref_view, src_views = self.metas[idx]
        view_ids = [ref_view] + src_views[:(self.nviews - 1)]
        resize_scale = self.resize_scale if resize_scale is None else resize_scale

        if load_src_depth is None:
            load_src_depth = getattr(self, "load_src_depth", False)
        resized_imgs, resized_intrinsics, extrinsics = [], [], []
        depth_hr = mask_hr = None
        src_depth_hr, src_mask_hr = [], []
        depth_values = []
        for i, view_id in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, 'Rectified_raw/{}/rect_{:0>3}_{}_r5000.png'.format(scan, view_id + 1, light_idx))
            mask_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, view_id))
            depth_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, view_id))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/{:0>8}_cam.txt').format(view_id)

            img = np.asarray(self.read_img(img_filename))
            # 光度增强在 resize 之前、每个视角用同一组参数: plane sweep 依赖视角
            # 间的光度一致性, 逐视角独立抖动会直接破坏 cost volume 要测的信号。
            if aug_params is not None:
                img = PhotometricAug.apply(img, aug_params)
            intrinsic, extrinsic, depth_min, depth_interval = self.read_camera_file(proj_mat_filename)

            if i == 0:
                depth_hr = np.asarray(read_pfm(depth_filename_hr), dtype=np.float32)
                mask_hr = self.read_mask(mask_filename_hr)
                depth_max = depth_min + depth_interval * self.ndepths
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)
                if resize_scale != 1.0:
                    img, depth_hr, intrinsic, mask_hr = self.pre_resize(img, depth_hr, intrinsic, mask_hr, resize_scale)
            elif load_src_depth:
                # 源视图的 GT 必须走与参考视图**同一条** resize 路径 —— 差一个
                # 插值方式或半个像素, 遮挡标签就落到相邻表面上了。
                d_s = np.asarray(read_pfm(depth_filename_hr), dtype=np.float32)
                m_s = self.read_mask(mask_filename_hr)
                if resize_scale != 1.0:
                    img, d_s, intrinsic, m_s = self.pre_resize(
                        img, d_s, intrinsic, m_s, resize_scale)
                src_depth_hr.append(d_s)
                src_mask_hr.append(m_s)
            elif resize_scale != 1.0:
                img, _, intrinsic, _ = self.pre_resize(img, None, intrinsic, None, resize_scale)

            resized_imgs.append(img)
            resized_intrinsics.append(np.asarray(intrinsic, dtype=np.float32))
            extrinsics.append(np.asarray(extrinsic, dtype=np.float32))

        imgs = torch.from_numpy(np.stack(resized_imgs, axis=0)).permute(0, 3, 1, 2).float()  # [V,C,H,W]
        return {
            "images": imgs,                                    # for norm_fill / sfm
            "views_np": resized_imgs,                          # HWC list, for cropping
            "intrinsics": np.stack(resized_intrinsics, axis=0),
            "extrinsics": np.stack(extrinsics, axis=0),
            "depth_hr": depth_hr,
            "mask_hr": mask_hr,
            "src_depth_hr": src_depth_hr or None,
            "src_mask_hr": src_mask_hr or None,
            "depth_values": depth_values,
            "scan": scan, "ref_view": ref_view, "light_idx": light_idx,
        }

    def _match_hw(self, arr, hw, is_depth):
        h, w = hw
        if arr.shape[:2] == (h, w):
            return arr
        interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
        return cv2.resize(arr, (w, h), interpolation=interp)

    def __getitem__(self, idx):
        rng = self._rng(idx)
        crop_h, crop_w, resize_scale = self.sample_geometry(idx, rng)
        aug_params = self.aug.draw(rng) if self.aug is not None else None
        pc = self.precrop_inputs(idx, resize_scale=resize_scale, aug_params=aug_params)
        resized_imgs = pc["views_np"]
        resized_intrinsics = pc["intrinsics"]
        extrinsics = pc["extrinsics"]
        depth_hr, mask_hr, depth_values = pc["depth_hr"], pc["mask_hr"], pc["depth_values"]
        # scan, ref_view, light_idx = pc["scan"], pc["ref_view"], pc["light_idx"]
        num_v = len(resized_imgs)
        h0, w0 = resized_imgs[0].shape[:2]

        # --- SfM 稀疏深度 (pre-crop, 带磁盘缓存) ---
        # cache_path = self.sfm_cache_dir / scan / f"sfm_{ref_view:0>4}_{light_idx}.npy"
        # sfm_depth = sfm.load_or_compute_sparse_depth(
        #     images=np.stack(resized_imgs, axis=0),
        #     intrinsics=resized_intrinsics,
        #     extrinsics=extrinsics,
        #     cache_path=cache_path,
        #     ref_idx=0,
        # )

        # --- 预计算好的 prior (pre-crop 全帧, 由 pre_prior 离线缓存) ---
        prior = pre_prior.load_prior(self.prior_cache_path_for(idx))
        depth_prior_full = self._match_hw(prior["depth_prior"], (h0, w0), is_depth=True)
        conf_prior_full = self._match_hw(prior["conf_prior"], (h0, w0), is_depth=False)
        norm_full = self._match_hw(prior["norm_depth_fill"], (h0, w0), is_depth=False)
        # 先验标尺有效性。两个来源:
        #   1) sfm_valid  —— pipeline_version>=2 的缓存才有 (旧缓存默认 0)
        #   2) 物理范围检查 —— 对旧缓存也有效: 未标尺先验的中位数是 ~1 而不是
        #      cam.txt 给出的几百 mm, 一查就出来
        # 二者取或, 因为旧缓存的 sfm_valid=0 只代表"未知"而非"失败"。
        _pv = float(np.asarray(prior.get("pipeline_version", 0.0)).reshape(-1)[0])
        _sv = float(np.asarray(prior.get("sfm_valid", 0.0)).reshape(-1)[0])
        _dp = depth_prior_full[np.isfinite(depth_prior_full) & (depth_prior_full > 0)]
        _med = float(np.median(_dp)) if _dp.size else 0.0
        _dmin, _dmax = float(depth_values[0]), float(depth_values[-1])
        in_range = (_dmin * 0.3) <= _med <= (_dmax * 3.0)
        prior_valid = np.asarray(1.0 if (in_range and (_pv < 2.0 or _sv > 0.5)) else 0.0, dtype=np.float32)

        # 离线 src_weights 长度 = 缓存时的 src 视角数, 可能 != 当前 nviews;
        # 默认忽略, 避免变长向量在 collate 时 stack 失败。
        src_weights = prior["src_weights"] if self.use_src_weights else None

        # --- 裁剪到 (crop_w x crop_h): 所有视角/深度/prior 共用同一窗口 ---
        crop_x, crop_y = self.pick_crop_origin(h0, w0, crop_h, crop_w, rng=rng)
        y1, x1 = crop_y + crop_h, crop_x + crop_w

        images, intrinsics, projection_matrices = [], [], []
        # depth_gt = mask_gt = sfm_depth_crop = None
        depth_gt = mask_gt =  None
        src_d_hr, src_m_hr = pc.get("src_depth_hr"), pc.get("src_mask_hr")
        src_depth_gt, src_mask_gt = [], []
        for i in range(num_v):
            _d_in = depth_hr if i == 0 else (src_d_hr[i - 1] if src_d_hr else None)
            _m_in = mask_hr if i == 0 else (src_m_hr[i - 1] if src_m_hr else None)
            img, K, depth, mask = self.crop_at(
                resized_imgs[i], resized_intrinsics[i], crop_x, crop_y,
                _d_in, _m_in, crop_h=crop_h, crop_w=crop_w)
            if i == 0:
                depth_gt, mask_gt = depth, mask
            elif src_d_hr:
                src_depth_gt.append(depth)
                src_mask_gt.append(mask)
                # _, _, sfm_depth_crop, _ = self.crop_at(
                #     resized_imgs[0], resized_intrinsics[0], crop_x, crop_y, sfm_depth)

            projection_matrix = np.eye(4, dtype=np.float32)
            projection_matrix[:3, :4] = K @ extrinsics[i][:3, :4]
            images.append(img)
            intrinsics.append(K)
            projection_matrices.append(projection_matrix)

        imgs = torch.from_numpy(np.stack(images, axis=0)).permute(0, 3, 1, 2).float()  # [V, C, H, W]

        depth_prior_crop = depth_prior_full[crop_y:y1, crop_x:x1]
        conf_prior_crop = conf_prior_full[crop_y:y1, crop_x:x1]
        # 先验失败模式增强: GT 不变, 只腐蚀 prior/conf; corrupt_mask 用于训练日志
        # 分别统计 corrupted / clean 像素误差 (global 分支救回率的直接度量)。
        if self.prior_corruption_prob > 0.0:
            depth_prior_crop, conf_prior_crop, corrupt_mask = corrupt_prior(
                depth_prior_crop, conf_prior_crop, self.prior_corruption_prob, rng=rng
            )
        else:
            corrupt_mask = np.zeros(depth_prior_crop.shape, dtype=bool)

        scan_id, light_id, ref_id, _srcs = self.metas[idx]
        sample = {
            # --- 样本身份 ---
            # 看门狗报"某个样本坏了"是没法离线复现的。这几个字段加上 crop/scale
            # 就能唯一确定一次 __getitem__ 的输出 (随机部分已由 _rng(idx) 固定)。
            # _collate 对非张量值原样收进 list, 所以字符串也能安全过 DataLoader。
            "sample_index": int(idx),
            "scan": str(scan_id),
            "ref_view": int(ref_id),
            "light_idx": int(light_id),
            "crop_xy": np.asarray([crop_x, crop_y], dtype=np.int32),
            "crop_hw": np.asarray([crop_h, crop_w], dtype=np.int32),
            "resize_scale": np.asarray(resize_scale, dtype=np.float32),
            "images": imgs,
            "intrinsics": np.stack(intrinsics, axis=0),       # [V, 3, 3]
            "extrinsics": np.stack(extrinsics, axis=0),       # [V, 4, 4]
            "depth_gt": depth_gt,
            "mask": mask_gt,
            "depth_values": depth_values,
            "projection_matrices": np.stack(projection_matrices, axis=0),
            # "sfm_depth": sfm_depth_crop,
            # priors consumed by the network / loss (cropped to the same window)
            "depth_prior": depth_prior_crop,
            "conf_prior": conf_prior_crop,
            "prior_corrupt_mask": corrupt_mask,
            # 0 = 该样本的 prior 没有通过标尺校验 -> 网络禁用 local 分支
            "prior_valid": prior_valid,
            "norm_depth_fill": norm_full[crop_y:y1, crop_x:x1],
        }
        if src_weights is not None:
            sample["src_weights"] = src_weights
        if src_depth_gt:
            # W3-B 的遮挡标签原料。与参考视图共用同一个 crop 窗口和同一组
            # 光度增强参数, 所以 warp 出来的 uv 可以直接在这上面采样。
            sample["src_depth_gt"] = np.stack(src_depth_gt, axis=0).astype(np.float32)
            sample["src_mask_gt"] = np.stack(src_mask_gt, axis=0).astype(np.float32)
        return sample
