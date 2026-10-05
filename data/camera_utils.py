from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
















def backproject_depth_to_world_points(
    depth: np.ndarray,
    K: np.ndarray,
    extrinsic: np.ndarray | None = None,
) -> np.ndarray:
    """Back-project valid z-depth pixels to camera or world coordinates.

    DTU extrinsics are world-to-camera. When ``extrinsic`` is provided, the
    returned points are transformed back to the world frame.
    """
    height, width = depth.shape
    yy, xx = np.indices((height, width))
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float32)

    z = depth[valid].astype(np.float64)
    x = (xx[valid].astype(np.float64) - float(K[0, 2])) * z / max(float(K[0, 0]), 1e-12)
    y = (yy[valid].astype(np.float64) - float(K[1, 2])) * z / max(float(K[1, 1]), 1e-12)
    points_cam = np.stack((x, y, z), axis=1)
    if extrinsic is None:
        return points_cam.astype(np.float32)

    ext = np.asarray(extrinsic, dtype=np.float64)
    if ext.shape == (3, 4):
        ext4 = np.eye(4, dtype=np.float64)
        ext4[:3, :4] = ext
        ext = ext4
    points_h = np.concatenate((points_cam, np.ones((len(points_cam), 1), dtype=np.float64)), axis=1)
    points_world = (np.linalg.inv(ext) @ points_h.T).T[:, :3]
    return points_world.astype(np.float32)




def project_world_points_to_depth(
    points_world: np.ndarray, 
    conf: np.ndarray | None,    # (N,) 或 None
    K: np.ndarray,              
    extrinsic: np.ndarray,      
    image_size: tuple,          
) -> tuple[np.ndarray, np.ndarray]: # 明确声明返回值永远是两个数组
    W, H = image_size

    # 世界坐标 -> 相机坐标
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    pts_cam = (R @ points_world.T).T + t   

    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 透视投影
    u = np.round(fx * X / Z + cx).astype(np.int32)
    v = np.round(fy * Y / Z + cy).astype(np.int32)

    # 过滤越界和背面点
    mask = (Z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, Z = u[mask], v[mask], Z[mask]
    
    # 【修复 Bug】分别初始化两个矩阵
    depth_map = np.zeros((H, W), dtype=np.float32)
    
    # 如果没有置信度，默认生成全为 1.0 的置信度图（代表 100% 置信）
    if conf is None:
        conf_map = np.ones((H, W), dtype=np.float32) 
    else:
        conf_map = np.zeros((H, W), dtype=np.float32)
        conf = conf[mask]

    # 近点优先
    order = np.argsort(Z)[::-1]
    depth_map[v[order], u[order]] = Z[order]
    
    # 只有传入了 conf 时才去刷 conf_map 的值
    if conf is not None:
        conf_map[v[order], u[order]] = conf[order]

    return depth_map, conf_map  
def save_depth_png(
    depth: np.ndarray,                 # [H, W] float, 单视图深度 (来自 pred["depth"][i])
    path,
    valid: np.ndarray | None = None,   # [H, W] bool, 可选; 无效像素不参与归一化也不上色
    colored: bool = True,              # True: 伪彩色可视化 PNG; False: 16-bit 灰度原值
    depth_scale: float = 1000.0,       # 仅 colored=False 时用: 米 -> 毫米存成 uint16
):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = np.asarray(depth, dtype=np.float32)

    # 有效区域: 给定 valid 就用它, 否则按 有限且 > 0
    if valid is None:
        valid = np.isfinite(d) & (d > 0)
    else:
        valid = valid.astype(bool) & np.isfinite(d)

    if not colored:
        out = np.zeros_like(d, dtype=np.float32)
        out[valid] = d[valid] * depth_scale
        out = np.clip(out, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(path), out)          # 16-bit 单通道 PNG
        return path

    # 伪彩色: 在有效像素范围内做 min-max 归一化
    vis = np.zeros_like(d, dtype=np.float32)
    if valid.any():
        dmin = float(d[valid].min())
        dmax = float(d[valid].max())
        rng = max(dmax - dmin, 1e-8)
        vis[valid] = (d[valid] - dmin) / rng        # 0~1
    vis_u8 = (vis * 255.0).clip(0, 255).astype(np.uint8)
    color = cv2.applyColorMap(vis_u8, cv2.COLORMAP_TURBO)   # [H,W,3] BGR
    color[~valid] = 0                                       # 无效区涂黑
    cv2.imwrite(str(path), color)



"""把若干张测试用的中间结果(法向图 / 差值图 / 深度图 / RGB)拼到一张图保存。"""




def save_pointcloud_ply(
    points: np.ndarray,                # [N, 3] 或 [V, H, W, 3] (来自 pred["world_points"])
    path: str | Path,
    colors: np.ndarray | None = None,  # [N, 3] 或 [V, H, W, 3] uint8, 可选
    conf: np.ndarray | None = None,    # [N] 或 [V, H, W] 置信度, 给了就按 conf_percentile 过滤
    conf_percentile: float = 0.0,      # >0 时丢弃最低的这一百分比 (例如 10 表示丢最低 10%)
) -> Path:
    """把点云存成 PLY (ascii)。points/colors/conf 形状要一致(同为展平或同为网格)。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    cols = None
    if colors is not None:
        cols = np.asarray(colors).reshape(-1, 3).astype(np.uint8)

    # 有效性 + 置信度过滤
    keep = np.isfinite(pts).all(axis=1)
    if conf is not None and conf_percentile > 0:
        c = np.asarray(conf, dtype=np.float32).reshape(-1)
        keep &= np.isfinite(c)
        if keep.any():
            thr = float(np.percentile(c[keep], conf_percentile))
            keep &= c >= thr
    pts = pts[keep]
    if cols is not None:
        cols = cols[keep]

    n = len(pts)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x", "property float y", "property float z",
    ]
    if cols is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")

    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        if cols is None:
            for p in pts:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        else:
            for p, c in zip(pts, cols):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
    print(f"Saved {n} points to {path}")
    return path






