"""Download one prior per DTU scan from umhpc and export prior / GT point clouds.

    1) rsync  <remote>/scan*/prior_{view:04d}_{light}.npz  ->  <out>/npz/scan*/
       (one connection, one password prompt; scans without that file are skipped)
    2) for every downloaded scan: back-project ``depth_prior`` with the DTU camera of
       that view (K rescaled to the prior's resolution) -> blue points; the GT depth
       Depths_raw/scan*/depth_map_{view:04d}.pfm -> white points. World frame = DTU's
       (cam.txt extrinsic is world -> camera), so the clouds overlay the STL GT.

    <out>/ply/scan{N}_prior.ply   blue  (0, 0, 255)
    <out>/ply/scan{N}_gt.ply      white (255, 255, 255)
    <out>/ply/scan{N}_both.ply    both in one file

Usage:
    python scripts/prior_cache_to_ply.py                       # download + convert
    python scripts/prior_cache_to_ply.py --skip-download       # convert what is already local
    python scripts/prior_cache_to_ply.py --mask gt --stride 2  # prior only inside the GT mask, every 2nd pixel

numpy only (no torch / cv2), so it runs in any Python.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
REMOTE_HOST = "qinglong@login01.dicc.um.edu.my"
REMOTE_DIR = "/scr/user/qinglong/projects/upr-mvs01/log/prior_cache"
DTU_ROOT = PROJECT.parent / "dataset/DTU/dtu_training"
FULL_W, FULL_H = 1600, 1200          # resolution the Cameras/*_cam.txt intrinsics refer to
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)


def download(host: str, remote_dir: str, view: int, light: int, dst: Path) -> None:
    name = f"prior_{view:04d}_{light}.npz"
    dst.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-av", "--prune-empty-dirs",
           "--include=scan*/", f"--include=scan*/{name}", "--exclude=*",
           f"{host}:{remote_dir.rstrip('/')}/", f"{dst}/"]
    print("$", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(f"rsync failed with exit code {rc}")


def read_cam(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text().split("\n")
    E = np.array(" ".join(lines[1:5]).split(), dtype=np.float64).reshape(4, 4)
    K = np.array(" ".join(lines[7:10]).split(), dtype=np.float64).reshape(3, 3)
    return K, E


def read_pfm(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().decode().strip()
        if header not in ("Pf", "PF"):
            raise ValueError(f"{path}: not a PFM file")
        w, h = map(int, f.readline().decode().split())
        scale = float(f.readline().decode().strip())
        data = np.fromfile(f, dtype="<f4" if scale < 0 else ">f4")
    ch = 3 if header == "PF" else 1
    data = data.reshape(h, w, ch) if ch == 3 else data.reshape(h, w)
    return np.flipud(data).astype(np.float32)


def read_mask(path: Path) -> np.ndarray | None:
    """depth_visual_*.png > 10, read without cv2/PIL when possible."""
    try:
        from PIL import Image
    except ImportError:
        return None
    if not path.exists():
        return None
    m = np.asarray(Image.open(path), dtype=np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    return m > 10


def nearest_resize(a: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if a.shape[:2] == (h, w):
        return a
    ys = (np.arange(h) * a.shape[0] // h).clip(max=a.shape[0] - 1)
    xs = (np.arange(w) * a.shape[1] // w).clip(max=a.shape[1] - 1)
    return a[ys][:, xs]


def backproject(depth: np.ndarray, K: np.ndarray, E: np.ndarray, valid: np.ndarray,
                stride: int) -> np.ndarray:
    """depth [H,W] (mm), K at that resolution, E world->camera. Returns [N,3] world points."""
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    d = depth[ys, xs]
    ok = valid[ys, xs] & np.isfinite(d) & (d > 0)
    x, y, d = xs[ok].astype(np.float64), ys[ok].astype(np.float64), d[ok].astype(np.float64)
    cam = np.linalg.inv(K) @ np.stack([x * d, y * d, d])
    world = np.linalg.inv(E) @ np.vstack([cam, np.ones_like(d)])
    return world[:3].T.astype(np.float32)


def write_ply(path: Path, pts: np.ndarray, rgb: np.ndarray) -> None:
    v = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    v["red"], v["green"], v["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(v)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(v.tobytes())


def colored(n: int, color: tuple[int, int, int]) -> np.ndarray:
    return np.tile(np.array(color, dtype=np.uint8), (n, 1))


def scalar(z, key: str, default: float = 0.0) -> float:
    return float(np.asarray(z[key]).reshape(-1)[0]) if key in z.files else default


def convert_scan(npz: Path, scan: str, view: int, dtu_root: Path, out: Path, mask_mode: str,
                 stride: int) -> None:
    K_full, E = read_cam(dtu_root / "Cameras" / f"{view:08d}_cam.txt")
    with np.load(npz) as z:
        depth = np.asarray(z["depth_prior"], dtype=np.float32).squeeze()
        sfm_valid = scalar(z, "sfm_valid")
    h, w = depth.shape
    # the prior lives at the pre-crop resolution (DTU image * resize_scale): rescale K
    K = K_full.copy()
    K[0] *= w / FULL_W
    K[1] *= h / FULL_H

    gt_path = dtu_root / "Depths_raw" / scan / f"depth_map_{view:04d}.pfm"
    gt = read_pfm(gt_path) if gt_path.exists() else None
    gt_mask = read_mask(dtu_root / "Depths_raw" / scan / f"depth_visual_{view:04d}.png")

    valid = np.ones_like(depth, dtype=bool)
    if mask_mode == "gt":
        if gt_mask is not None:
            valid = nearest_resize(gt_mask, (h, w))
        elif gt is not None:
            valid = nearest_resize(gt, (h, w)) > 0
    p_pts = backproject(depth, K, E, valid, stride)

    med = float(np.median(depth[np.isfinite(depth) & (depth > 0)])) if np.any(depth > 0) else 0.0
    note = "" if med > 50 else "  [!] median depth looks unscaled (SfM scale failed?)"
    print(f"{scan}: prior {w}x{h}, median {med:.1f} mm, sfm_valid={sfm_valid:.0f}, "
          f"{len(p_pts)} pts{note}", flush=True)

    write_ply(out / f"{scan}_prior.ply", p_pts, colored(len(p_pts), BLUE))
    if gt is None:
        print(f"  {scan}: no GT depth at {gt_path}, GT cloud skipped")
        return
    gh, gw = gt.shape
    Kg = K_full.copy()
    Kg[0] *= gw / FULL_W
    Kg[1] *= gh / FULL_H
    g_valid = gt > 0
    if gt_mask is not None:
        g_valid &= nearest_resize(gt_mask, (gh, gw))
    g_pts = backproject(gt, Kg, E, g_valid, stride)
    write_ply(out / f"{scan}_gt.ply", g_pts, colored(len(g_pts), WHITE))
    write_ply(out / f"{scan}_both.ply", np.concatenate([p_pts, g_pts]),
              np.concatenate([colored(len(p_pts), BLUE), colored(len(g_pts), WHITE)]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", type=int, default=17, help="DTU view index (0-based, as in cam.txt / prior names)")
    ap.add_argument("--light", type=int, default=6)
    ap.add_argument("--host", default=REMOTE_HOST)
    ap.add_argument("--remote-dir", default=REMOTE_DIR)
    ap.add_argument("--dtu-root", type=Path, default=DTU_ROOT)
    ap.add_argument("--out", type=Path, default=PROJECT / "log/prior_ply")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--mask", choices=("none", "gt"), default="none",
                    help="gt: keep only prior pixels inside the GT depth mask")
    ap.add_argument("--stride", type=int, default=1, help="pixel subsampling step")
    args = ap.parse_args()

    npz_root = args.out / "npz"
    ply_root = args.out / "ply"
    if not args.skip_download:
        download(args.host, args.remote_dir, args.view, args.light, npz_root)
    ply_root.mkdir(parents=True, exist_ok=True)

    name = f"prior_{args.view:04d}_{args.light}.npz"
    files = sorted(npz_root.glob(f"scan*/{name}"),
                   key=lambda p: int(re.sub(r"\D", "", p.parent.name) or 0))
    if not files:
        sys.exit(f"no {name} under {npz_root}")
    print(f"{len(files)} scans -> {ply_root}")
    for f in files:
        try:
            convert_scan(f, f.parent.name, args.view, args.dtu_root, ply_root, args.mask, args.stride)
        except Exception as e:      # one broken cache file must not stop the batch
            print(f"  {f.parent.name}: FAILED ({e})")


if __name__ == "__main__":
    main()
