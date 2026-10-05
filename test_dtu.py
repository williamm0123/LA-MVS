#!/usr/bin/env python
"""MoAMVSNet DTU 测试: test_moa.py 推理 + MonoMVSNet 式点云融合 (纯 Python, 不需要 fusibile)。

    # 一条龙: 推理 (与 test_moa.py 完全相同, 多余参数原样转给它) -> 融合
    python test_dtu.py --out log/moa2/depth_cache_test --ply-dir log/moa2/ply_dypcd \
        --ckpt log/moa2/checkpoint/best.pth --profile local --split test \
        --full-image --resize-scale 0.8 --num-views 5

    # 只融合已有的深度缓存 (换阈值重融不必重跑推理)
    python test_dtu.py --phase fuse --no-skip-existing --out log/moa2/depth_cache_test --ply-dir log/moa2/ply_dypcd

    # 中断后原命令重跑: 默认跳过已有的逐视角 NPZ 和逐 scan PLY。
    # 需要全部重算时加 --no-skip-existing。

    # MonoMVSNet 的固定门槛消融版 (test_dtu_pcd.sh)
    python test_dtu.py --phase fuse --filter fixed --out ... --ply-dir ...

推理
    --phase all/infer 时, 本脚本自己不认识的参数全部转给 test_moa.main(),
    ``--out`` 两边共用。缓存格式见 test_moa.py 的文档。

融合 (对齐 MonoMVSNet 0485818 的 test_dtu_dypcd.py::filter_depth, 即 README
推荐的 DTU 入口 scripts/test_dtu_dypcd.sh)
    对每个参考视图:
    1. 光度掩码: conf > 0.55。conf 取最后一级概率体的最大概率 —— 缓存里的
       ``conf_last``。它是 2026-09-25 才加进 test_moa.py 的; 更早的缓存
       (例如 log/moa2/depth_cache_test) 只有四级 mode-mass 连乘的 ``conf``,
       --conf-key auto 会退回用它并在日志和 manifest 里注明。两者分布不同,
       0.55/0.75 这两个门槛是按 conf_last 定的, 严格对齐请重跑推理。
    2. 对 pair.txt 里该视图的**全部**源视图 (缓存里 src_views 存的就是 pair.txt
       的完整 10 个, 与推理用了几个视角无关): 参考像素 -> 源视图 -> 采样源深度
       -> 投回参考视图, 得回投影像素距离 dist 和相对深度误差 rel。
    3. 动态门槛: 对 i = 1..10, 第 i 档要求 dist < 0.5 i 且
       rel < 0.001 log10(max(i, 1.05)); 存在某个 i 使至少 i 个源视图过第 i 档,
       就通过几何过滤。
    4. 深度 = (参考深度 + 过最宽一档 (i=10) 的源视图回投影深度) 的算术平均;
       参考 conf > 0.75 的像素直接用参考深度。
    5. 光度 & 几何都过的像素按 K, E 反投影到世界坐标, 颜色取参考图,
       各参考视图汇总写 <ply-dir>/mvsnet<scan:03d>_l3.ply。
    --filter fixed 对应 test_dtu_pcd.py: conf > 0.6, dist < 0.25, rel < 0.001,
    至少 2 个源视图一致, 深度同样取一致视图平均, 但没有 0.75 的回退。

    与原实现唯一有意的差别: 源视图深度缓存缺失时 (--max-refs 之类的子集推理)
    跳过该源视图并计数, 原实现会直接读文件报错。

打分仍然是独立的第三步 (Fast-DTU-Evaluation), 与 points_fusibile.py 的产物同名同格式。
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

DYNAMIC = {"conf": 0.55, "conf_keep_ref": 0.75, "dist_base": 0.5, "rel_base": 1e-3, "levels": 10}
FIXED = {"conf": 0.6, "dist": 0.25, "rel": 1e-3, "thres_view": 2}


def parse_args(argv=None) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser("MoAMVSNet DTU test + MonoMVSNet fusion",
                                description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["all", "infer", "fuse"], default="all")
    p.add_argument("--out", required=True, help="深度缓存根目录 (含 depth/<scan>/*.npz)")
    p.add_argument("--ply-dir", default=None, help="默认 <out>/../ply_dypcd")
    p.add_argument("--scans", type=int, nargs="+", default=None, help="只融合这些 scan (推理也只跑它们)")
    p.add_argument("--filter", choices=["dynamic", "fixed"], default="dynamic",
                   help="dynamic = test_dtu_dypcd.py (默认); fixed = test_dtu_pcd.py")
    p.add_argument("--conf", type=float, default=None, help="光度门槛, 默认 dynamic 0.55 / fixed 0.6")
    p.add_argument("--conf-key", choices=["auto", "conf_last", "conf"], default="auto",
                   help="auto: 有 conf_last 用它, 否则退回四级连乘 conf")
    p.add_argument("--workers", type=int, default=4, help="并行融合的 scan 数")
    p.add_argument("--save-masks", action="store_true", help="写 <out>/mask/<scan>/<ref>_{photo,geo,final}.png")
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True,
                   help="跳过已有的有效逐视角 NPZ 和逐 scan PLY (默认开启; --no-skip-existing 重算)")
    args, rest = p.parse_known_args(argv)
    if args.phase == "fuse" and rest:
        p.error(f"--phase fuse 不认识这些参数: {' '.join(rest)}")
    if args.conf is None:
        args.conf = DYNAMIC["conf"] if args.filter == "dynamic" else FIXED["conf"]
    return args, rest


# --------------------------------------------------------------------------- fusion

def load_view(path: Path) -> dict:
    z = np.load(path)
    v = {k: z[k] for k in z.files}
    v["depth"] = np.nan_to_num(v["depth"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return v


def reproject(ref: dict, src: dict, pix: np.ndarray, ones: np.ndarray) -> tuple[np.ndarray, ...]:
    """test_dtu_dypcd.py::reproject_with_depth。

    精度照抄原实现: K/E 保持 float32, 求逆也在 float32 里做, 与 float64 的像素坐标
    相乘时才升到 float64 —— 换成全 float64 会让门槛边界上约 3e-5 的像素翻转。
    """
    h, w = ref["depth"].shape
    Kr, Er, Ks, Es = ref["K"], ref["E"], src["K"], src["E"]
    xyz_ref = np.linalg.inv(Kr) @ (pix * ref["depth"].reshape(-1))
    xyz_src = ((Es @ np.linalg.inv(Er)) @ np.vstack((xyz_ref, ones)))[:3]
    k = Ks @ xyz_src
    xy_src = k[:2] / k[2:3]
    x_src = xy_src[0].reshape(h, w).astype(np.float32)
    y_src = xy_src[1].reshape(h, w).astype(np.float32)
    d_src = cv2.remap(src["depth"], x_src, y_src, interpolation=cv2.INTER_LINEAR)

    xyz_src = np.linalg.inv(Ks) @ (np.vstack((xy_src, ones)) * d_src.reshape(-1))
    xyz_rp = ((Er @ np.linalg.inv(Es)) @ np.vstack((xyz_src, ones)))[:3]
    depth_rp = xyz_rp[2].reshape(h, w).astype(np.float32)
    k = Kr @ xyz_rp
    k[2][k[2] == 0] += 1e-5
    xy_rp = k[:2] / k[2:3]
    return (depth_rp, xy_rp[0].reshape(h, w).astype(np.float32),
            xy_rp[1].reshape(h, w).astype(np.float32))


def geo_check(ref: dict, src: dict, pix, ones, x_ref, y_ref, mode: str):
    """返回 (逐档掩码列表, 用于平均的那一档掩码, 该档外置 0 的回投影深度)。"""
    depth_rp, x_rp, y_rp = reproject(ref, src, pix, ones)
    dist = np.sqrt((x_rp - x_ref) ** 2 + (y_rp - y_ref) ** 2)
    rel = np.abs(depth_rp - ref["depth"]) / ref["depth"]
    if mode == "dynamic":
        masks = [(dist < i * DYNAMIC["dist_base"])
                 & (rel < math.log(max(i, 1.05), 10) * DYNAMIC["rel_base"])
                 for i in range(1, DYNAMIC["levels"] + 1)]
    else:
        masks = [(dist < FIXED["dist"]) & (rel < FIXED["rel"])]
    widest = masks[-1]
    depth_rp[~widest] = 0
    return masks, widest, depth_rp


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    v = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    v["red"], v["green"], v["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(v)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    tmp = path.with_suffix(".ply.part")
    with open(tmp, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(v.tobytes())
    tmp.replace(path)


def fuse_scan(scan_dir: Path, ply_path: Path, mask_dir: Path | None, mode: str,
              conf_thr: float, conf_key: str) -> dict:
    t0 = time.time()
    views = {int(f.stem): load_view(f) for f in sorted(scan_dir.glob("*.npz"))}
    if not views:
        raise RuntimeError(f"{scan_dir} 下没有 npz")
    first = next(iter(views.values()))
    key = conf_key if conf_key != "auto" else ("conf_last" if "conf_last" in first else "conf")
    if key not in first:
        raise RuntimeError(f"{scan_dir}: 缓存里没有 {key} (旧缓存只有 conf, 用 --conf-key conf 或重跑推理)")
    h, w = first["depth"].shape
    x_ref, y_ref = np.meshgrid(np.arange(w), np.arange(h))
    ones = np.ones(h * w, dtype=np.int64)
    pix = np.vstack((x_ref.reshape(-1), y_ref.reshape(-1), ones))

    pts, cols, stats, missing = [], [], [], 0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for rv, ref in views.items():
            conf = ref[key]
            photo = conf > conf_thr
            n_lvl = DYNAMIC["levels"] if mode == "dynamic" else 1
            lvl_sums = [np.zeros((h, w), np.int32) for _ in range(n_lvl)]
            geo_sum = np.zeros((h, w), np.int32)
            depth_sum = np.zeros((h, w), np.float32)
            for sv in ref["src_views"].tolist():
                src = views.get(int(sv))
                if src is None:
                    missing += 1
                    continue
                masks, widest, depth_rp = geo_check(ref, src, pix, ones, x_ref, y_ref, mode)
                geo_sum += widest
                depth_sum += depth_rp
                for s, m in zip(lvl_sums, masks):
                    s += m
            depth = (depth_sum + ref["depth"]) / (geo_sum + 1)
            if mode == "dynamic":
                keep_ref = conf > DYNAMIC["conf_keep_ref"]
                depth[keep_ref] = ref["depth"][keep_ref]
                # 原实现的 geo_mask_sum >= dy_range(=11) 在只有 10 个源视图时恒假, 照抄
                geo = geo_sum >= DYNAMIC["levels"] + 1
                for i, s in enumerate(lvl_sums, start=1):
                    geo |= s >= i
            else:
                geo = geo_sum >= FIXED["thres_view"]
            final = photo & geo & (ref["depth"] > 0)
            if mask_dir is not None:
                mask_dir.mkdir(parents=True, exist_ok=True)
                for tag, m in (("photo", photo), ("geo", geo), ("final", final)):
                    cv2.imwrite(str(mask_dir / f"{rv:08d}_{tag}.png"), m.astype(np.uint8) * 255)
            stats.append((photo.mean(), geo.mean(), final.mean()))

            xs, ys, d = x_ref[final], y_ref[final], depth[final]
            xyz_cam = np.linalg.inv(ref["K"]) @ (np.vstack((xs, ys, np.ones_like(xs))) * d)
            xyz = (np.linalg.inv(ref["E"]) @ np.vstack((xyz_cam, np.ones_like(xs))))[:3]
            pts.append(xyz.T.astype(np.float32))
            cols.append(ref["image"][final])

    xyz = np.concatenate(pts)
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    write_ply(ply_path, xyz, np.concatenate(cols))
    s = np.asarray(stats)
    return {"scan": scan_dir.name, "views": len(views), "points": int(len(xyz)), "conf_key": key,
            "photo": float(s[:, 0].mean()), "geo": float(s[:, 1].mean()), "final": float(s[:, 2].mean()),
            "missing_src": missing, "seconds": round(time.time() - t0, 1)}


def _fuse_one(job):
    return fuse_scan(*job)


def fuse(args) -> None:
    cache = Path(args.out) / "depth"
    if not cache.is_dir():
        raise SystemExit(f"{cache} 不存在 —— 先跑推理 (--phase all / infer)")
    ply_dir = Path(args.ply_dir) if args.ply_dir else Path(args.out).parent / "ply_dypcd"
    scans = sorted((d for d in cache.iterdir() if d.is_dir() and d.name.startswith("scan")),
                   key=lambda d: int(d.name[4:]))
    if args.scans:
        want = {f"scan{s}" for s in args.scans}
        scans = [d for d in scans if d.name in want]
        if missing := want - {d.name for d in scans}:
            raise SystemExit(f"缓存里没有: {sorted(missing)}")
    jobs = []
    skipped = []
    for d in scans:
        ply = ply_dir / f"mvsnet{int(d.name[4:]):03d}_l3.ply"
        if args.skip_existing and ply.is_file():
            print(f"[fuse] skip {d.name} (已有 {ply.name})")
            skipped.append(d.name)
            continue
        mask_dir = Path(args.out) / "mask" / d.name if args.save_masks else None
        jobs.append((d, ply, mask_dir, args.filter, args.conf, args.conf_key))
    print(f"[fuse] {len(jobs)} 个 scan, filter={args.filter} conf>{args.conf} conf_key={args.conf_key} "
          f"-> {ply_dir}", flush=True)

    manifest_path = ply_dir / "fusion_manifest.json"
    if not jobs and manifest_path.is_file():
        print(f"[fuse] 全部已有 PLY，保留现有 {manifest_path}")
        return

    results = []
    with ProcessPoolExecutor(max_workers=max(args.workers, 1)) as ex:
        for r in ex.map(_fuse_one, jobs):
            results.append(r)
            print(f"[fuse] {r['scan']}: {r['points']:,} 点  photo/geo/final={r['photo']:.3f}/"
                  f"{r['geo']:.3f}/{r['final']:.3f}  conf={r['conf_key']}"
                  f"{'  缺源视图 ' + str(r['missing_src']) if r['missing_src'] else ''}  {r['seconds']}s",
                  flush=True)
    keys = {r["conf_key"] for r in results}
    if "conf" in keys:
        print("[fuse] ⚠ 用的是四级连乘 conf (缓存没有 conf_last); 与 MonoMVSNet 的光度置信度"
              "不是同一个量, 严格对齐需要用新版 test_moa.py 重跑推理")
    manifest = {
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "depth_cache": str(Path(args.out).resolve()),
        "fusion": {"method": "monomvsnet_" + args.filter, "conf": args.conf, "conf_key": sorted(keys),
                   "params": DYNAMIC if args.filter == "dynamic" else FIXED},
        "scans": results,
        "skipped_existing_scans": skipped,
    }
    ply_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[fuse] 新融合 {len(results)} 个 scan，跳过已有 {len(skipped)} 个 -> {ply_dir}")


def main(argv=None) -> None:
    args, rest = parse_args(argv)
    if args.phase in ("all", "infer"):
        import test_moa
        targv = rest + ["--out", args.out]
        if args.scans:
            targv += ["--scans", *map(str, args.scans)]
        if args.skip_existing:
            targv.append("--skip-existing")
        test_moa.main(targv)
    if args.phase in ("all", "fuse"):
        fuse(args)


if __name__ == "__main__":
    main(sys.argv[1:])
