#!/usr/bin/env python3
"""Raw DA3 monocular depth cache for BlendedMVS and Tanks-and-Temples (MoAMVSNet input).

The DTU counterpart is build_da3_cache_all.py; this one only differs in how
images are enumerated and where files go:

    --dataset blended   <root>/<scene>/blended_images/{view:08d}.jpg   (*_masked.jpg skipped)
                        -> <out>/<scene>/da3_{view:08d}.npz
    --dataset tnt       <root>/<split>/<Scene>/images/{view:08d}.jpg
                        -> <out>/<split>/<Scene>/da3_{view:08d}.npz

Same file format as the DTU cache: ``depth`` float16 at the image's native size,
``process_res`` int. ``--process-res native`` (default) runs DA3 at the image's
long side — the DTU cache moa1 was trained on is native (1600) too — i.e. 768 on
BlendedMVS and 1920 on T&T. One cache root must hold one process_res; a
mismatch with files already there is refused.

Resumable (existing files are skipped), ``--shard i/N`` splits the to-do list
across processes (the GPU is not the bottleneck: JPEG decode + npz compression).

    python scripts/build_da3_cache_mvs.py --dataset blended --root .../BlendedMVS_plus \
        --scenes-file lists/blended/training_list.txt lists/blended/validation_list.txt \
        --out log/da3_cache_blended
    python scripts/build_da3_cache_mvs.py --dataset tnt --root .../TankandTemples \
        --scenes intermediate/Family advanced/Temple --out log/da3_cache_tnt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
TNT_SCENES = [f"intermediate/{s}" for s in
              ("Family", "Francis", "Horse", "Lighthouse", "M60", "Panther", "Playground", "Train")] + \
             [f"advanced/{s}" for s in ("Auditorium", "Ballroom", "Courtroom", "Museum", "Palace", "Temple")]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["blended", "tnt"], required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scenes", nargs="*", default=None,
                   help="blended: scene ids; tnt: split/Scene (default: every scene)")
    p.add_argument("--scenes-file", nargs="+", default=None,
                   help="files with one scene per line (blended lists/, tnt split/Scene); merged")
    p.add_argument("--process-res", default="native", help="'native' (image long side) or an int")
    p.add_argument("--shard", default=None, metavar="i/N")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def scene_images(dataset: str, root: Path, scene: str) -> list[tuple[int, Path]]:
    d = root / scene / ("blended_images" if dataset == "blended" else "images")
    out = []
    for f in d.glob("*.jpg"):
        if f.stem.isdigit():                 # drops 00000000_masked.jpg
            out.append((int(f.stem), f))
    return sorted(out)


def cache_file(out: Path, scene: str, view: int) -> Path:
    return out / scene / f"da3_{view:08d}.npz"


def existing_res(out: Path) -> tuple[int, Path] | None:
    if not out.is_dir():
        return None
    for f in out.rglob("da3_*.npz"):
        try:
            with np.load(f) as z:
                return int(z["process_res"]), f
        except Exception:  # noqa: BLE001
            continue
    return None


def save(path: Path, depth: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex[:8]}.tmp")
    try:
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, depth=depth, **meta)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    root, out = Path(args.root), Path(args.out)
    if args.scenes:
        scenes = args.scenes
    elif args.scenes_file:
        scenes = list(dict.fromkeys(s.strip() for f in args.scenes_file
                                    for s in Path(f).read_text().splitlines() if s.strip()))
    elif args.dataset == "blended":
        scenes = sorted(d.name for d in root.iterdir() if d.is_dir())
    else:
        scenes = [s for s in TNT_SCENES if (root / s).is_dir()]
    missing = [s for s in scenes if not (root / s).is_dir()]
    if missing:
        raise SystemExit(f"scenes not under {root}: {missing[:10]}")

    native = args.process_res == "native"
    fixed_res = None if native else int(args.process_res)
    todo, n_done = [], 0
    for s in scenes:
        for v, f in scene_images(args.dataset, root, s):
            if cache_file(out, s, v).is_file():
                n_done += 1
            else:
                todo.append((s, v, f))
    print(f"[da3-mvs] {args.dataset} root={root} out={out} scenes={len(scenes)} "
          f"done={n_done} todo={len(todo)} process_res={args.process_res}", flush=True)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        todo = todo[i::n]
        print(f"[da3-mvs] shard {args.shard}: {len(todo)}", flush=True)
    if args.dry_run or not todo:
        return

    found = existing_res(out)
    import torch
    import models.norm_fill as nf
    from base.config import ProjectPaths

    if not torch.cuda.is_available():
        raise SystemExit("DA3 needs CUDA")
    model = nf.load_da3_model(ProjectPaths().da3_weights_file, torch.device(args.device))
    t0, n_ok, n_fail = time.time(), 0, 0
    for k, (s, v, f) in enumerate(todo, 1):
        try:
            img = np.asarray(Image.open(f).convert("RGB"))
            h, w = img.shape[:2]
            res = max(h, w) if native else fixed_res
            if found is None:
                found = (res, f)
            elif found[0] != res:
                raise SystemExit(f"{f}: process_res {res} != {found[0]} already in {out} ({found[1]}); "
                                 f"one cache root holds one resolution — use another --out")
            d = nf._get_depth_da3(img, model, res)
            if d.shape != (h, w):
                d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
            save(cache_file(out, s, v), d.astype(np.float16),
                 {"scene": np.asarray(s), "view": np.asarray(v, np.int32),
                  "process_res": np.asarray(res, np.int32), "hw": np.asarray((h, w), np.int32)})
            n_ok += 1
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad image must not kill an hours-long job
            n_fail += 1
            print(f"  !! {s} v{v}: {type(exc).__name__}: {exc}", flush=True)
        if k % args.log_every == 0 or k == len(todo):
            dt = time.time() - t0
            print(f"[da3-mvs] {k}/{len(todo)} ok={n_ok} fail={n_fail} {k / dt:.2f}/s "
                  f"eta {(len(todo) - k) / max(k / dt, 1e-9) / 60:.0f}min", flush=True)
    if n_fail:
        raise SystemExit(f"[da3-mvs] {n_fail} failures (rerun to retry them)")


if __name__ == "__main__":
    main()
