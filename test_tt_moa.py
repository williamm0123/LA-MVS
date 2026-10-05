"""Tanks-and-Temples inference + fusion for MoAMVSNet.

    python test_tt_moa.py --ckpt log/experiments/MOA1_BLD_30K/model/latest.pth \
        --tt-root /scr/user/qinglong/dataset/TankandTemples --num-views 7 --resize-scale 1.0 \
        --out log/tnt/MOA1_BLD_30K

    # re-fuse with another threshold without re-running the network
    python test_tt_moa.py --phase fuse --out log/tnt/MOA1_BLD_30K --conf 0.3 --ply-dir log/tnt/MOA1_BLD_30K/ply_c03

Outputs
    <out>/depth/<split>/<Scene>/<ref:08d>.npz   test_moa.py's cache format
    <ply-dir>/<Scene>.ply                        (default <out>/ply) — upload these, together with the
                                                 dataset's <Scene>.log, to the T&T site; there is no local GT
    <out>/run_manifest.json, <ply-dir>/fusion_manifest.json

Network: rebuilt from the checkpoint's architecture snapshot. Behaviour fields the
snapshot predates (global_solver / moa_gain / edge_snap — added after moa1) take
moa1's semantics (huber, 1,1,1, off), not today's defaults; a checkpoint from
train_blended.py carries them explicitly.

Fusion: test_dtu.py's MonoMVSNet dynamic geometric consistency (stage-4 max
posterior > --conf, dynamic reprojection levels over pair.txt's top-10 sources).
The thresholds are DTU's; T&T usually wants per-scene tuning — sweep --conf with
--phase fuse. Scenes fuse in parallel (--workers); each worker holds a whole
scene in RAM (~30 MB per 1920x1080 view, Palace ~16 GB).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from base.config_moa import apply_arch_snapshot, build_moa_config
from data.tnt_moa import MoATnTScene
from models.network_moa import MoAMVSNet
from scripts.build_da3_cache_mvs import TNT_SCENES
from test_dtu import DYNAMIC, FIXED, fuse_scan
from test_moa import cascade_confidence, last_stage_confidence, valid_depth_cache
from train_moa import collate, git_state, load_checkpoint, load_model_state

REPO = Path(__file__).resolve().parent
MOA1_DEFAULTS = {"global_solver": ("huber",) * 3, "moa_gain": (1.0,) * 3, "edge_snap": (False,) * 3}


def parse_args(argv=None):
    p = argparse.ArgumentParser("MoAMVSNet Tanks-and-Temples", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["all", "infer", "fuse"], default="all")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--profile", choices=["local", "umhpc"], default="umhpc")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tt-root", default="/scr/user/qinglong/dataset/TankandTemples")
    p.add_argument("--scenes", nargs="*", default=None, help="split/Scene (default: all 14 found)")
    p.add_argument("--num-views", type=int, default=7)
    p.add_argument("--resize-scale", type=float, default=1.0)
    p.add_argument("--max-refs", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--da3-root", default=str(REPO / "log/da3_cache_tnt"),
                   help="DA3 cache, only for legacy (DINOv3) checkpoints; DA3-backbone models run it online")
    p.add_argument("--conf-window", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    # fusion
    p.add_argument("--ply-dir", default=None, help="default <out>/ply")
    p.add_argument("--filter", choices=["dynamic", "fixed"], default="dynamic")
    p.add_argument("--conf", type=float, default=None, help="default dynamic 0.55 / fixed 0.6")
    p.add_argument("--conf-key", choices=["auto", "conf_last", "conf"], default="conf_last")
    p.add_argument("--workers", type=int, default=3, help="scenes fused in parallel")
    args = p.parse_args(argv)
    if args.phase != "fuse" and not args.ckpt:
        p.error("--ckpt is required unless --phase fuse")
    if args.conf is None:
        args.conf = DYNAMIC["conf"] if args.filter == "dynamic" else FIXED["conf"]
    return args


def scene_list(args) -> list[str]:
    root = Path(args.tt_root)
    scenes = args.scenes or [s for s in TNT_SCENES if (root / s).is_dir()]
    bad = [s for s in scenes if not (root / s / "pair.txt").is_file()]
    if bad or not scenes:
        raise SystemExit(f"T&T scenes not found under {root}: {bad or 'none'}")
    return scenes


def build_model(args, device):
    ck = load_checkpoint(args.ckpt, map_location="cpu")
    if ck.get("kind") != "moa_mvsnet":
        raise SystemExit(f"{args.ckpt} is not a MoAMVSNet checkpoint")
    cfg = apply_arch_snapshot(build_moa_config(args.profile), ck["arch"])
    snap = ck["arch"].get("moa", {})
    pre = {k: v for k, v in MOA1_DEFAULTS.items() if k not in snap}
    if pre:
        cfg = dataclasses.replace(cfg, moa=dataclasses.replace(cfg.moa, **pre))
        print(f"[tnt] snapshot predates {sorted(pre)} -> moa1 semantics {pre}")
    m = cfg.moa
    print(f"[tnt] {args.ckpt} step {ck.get('step')}: warp={cfg.cascade.warp_channels} "
          f"feat={cfg.feat.backbone} lape={'on' if cfg.lape.enabled else 'off'} "
          f"global_solver={m.global_solver} moa_gain={m.moa_gain} edge_snap={m.edge_snap}")
    model = MoAMVSNet(cfg).to(device)
    load_model_state(model, ck["model"])
    model.eval()
    tcfg = ck.get("config", {}).get("train", {})
    amp_dtype = torch.bfloat16 if tcfg.get("amp_dtype", "bf16") == "bf16" else torch.float16
    return model, ck, amp_dtype


@torch.no_grad()
def infer(args, scenes) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, ck, amp_dtype = build_model(args, device)
    use_amp = device.type == "cuda"
    out_root = Path(args.out)
    report = {}
    for scene in scenes:
        ds = MoATnTScene(args.tt_root, scene, nviews=args.num_views, resize_scale=args.resize_scale,
                         da3_root=args.da3_root if model.needs_mono_cache else None, max_refs=args.max_refs)
        d = out_root / "depth" / scene
        d.mkdir(parents=True, exist_ok=True)
        todo = [i for i, (ref, _) in enumerate(ds.metas)
                if not (args.skip_existing and valid_depth_cache(d / f"{ref:08d}.npz"))]
        da3_res = model.da3_process_res if model.da3_sva is not None else ds.da3_process_res
        print(f"[tnt] {scene}: {len(ds)} refs, {len(ds) - len(todo)} cached, {len(todo)} to run, "
              f"{args.num_views} views, DA3 {'online' if model.da3_sva is not None else 'cache'} "
              f"process_res={da3_res}", flush=True)
        t0, gok = time.time(), []
        loader = DataLoader(torch.utils.data.Subset(ds, todo), batch_size=1, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
        for k, batch in enumerate(loader):
            ref = int(batch["ref_view"][0])
            batch = {n: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for n, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(batch)
            pred = out["depth_full"].float()
            if not torch.isfinite(pred).any():
                raise SystemExit(f"{scene} ref {ref}: depth entirely non-finite")
            if "moa2" in out:
                gok.append(float(out["moa2"].global_ok.float().mean()))
            tmp = None
            try:
                with tempfile.NamedTemporaryFile(mode="wb", dir=d, prefix=f".{ref:08d}.",
                                                 suffix=".npz.part", delete=False) as fh:
                    tmp = Path(fh.name)
                    np.savez_compressed(
                        fh, depth=pred[0].cpu().numpy(),
                        conf=cascade_confidence(out, args.conf_window)[0].cpu().numpy().astype(np.float32),
                        conf_last=last_stage_confidence(out)[0].cpu().numpy().astype(np.float32),
                        K=batch["intrinsics"][0, 0].float().cpu().numpy(),
                        E=batch["extrinsics"][0, 0].float().cpu().numpy(),
                        image=batch["images"][0, 0].permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy(),
                        src_views=batch["src_views"][0].cpu().numpy().astype(np.int64))
                tmp.replace(d / f"{ref:08d}.npz")
            finally:
                if tmp is not None:
                    tmp.unlink(missing_ok=True)
            if (k + 1) % 25 == 0 or k + 1 == len(todo):
                dt = time.time() - t0
                print(f"[tnt] {scene} {k + 1}/{len(todo)}  {dt / (k + 1):.2f}s/view  "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30 if use_amp else 0:.1f}G", flush=True)
        report[scene] = {"refs": len(ds), "inferred": len(todo), "da3_process_res": da3_res,
                         "moa2_global_ok": float(np.mean(gok)) if gok else None}
    manifest = {
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "checkpoint": {"path": str(Path(args.ckpt).resolve()), "step": ck.get("step"),
                       "arch": ck.get("arch"), "git": ck.get("git")},
        "code_git": git_state(),
        "inference": {"num_views": args.num_views, "resize_scale": args.resize_scale,
                      "da3": "online" if model.da3_sva is not None else args.da3_root,
                      "conf_window": args.conf_window, "tt_root": args.tt_root},
        "scenes": report,
    }
    path = out_root / "run_manifest.json"
    if path.is_file():
        old = json.loads(path.read_text())
        manifest["scenes"] = {**old.get("scenes", {}), **report}
    path.write_text(json.dumps(manifest, indent=2, default=str))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def _fuse_one(job):
    return fuse_scan(*job)


def fuse(args, scenes) -> None:
    out_root = Path(args.out)
    ply_dir = Path(args.ply_dir) if args.ply_dir else out_root / "ply"
    ply_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for scene in scenes:
        sd = out_root / "depth" / scene
        if not any(sd.glob("*.npz")):
            raise SystemExit(f"no depth cache in {sd} — run inference first")
        ply = ply_dir / f"{scene.split('/')[-1]}.ply"
        if args.skip_existing and ply.is_file():
            print(f"[fuse] skip {scene} (have {ply.name})")
            continue
        jobs.append((sd, ply, None, args.filter, args.conf, args.conf_key))
    print(f"[fuse] {len(jobs)} scenes, filter={args.filter} conf>{args.conf} ({args.conf_key}) -> {ply_dir}",
          flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=max(args.workers, 1)) as ex:
        for r in ex.map(_fuse_one, jobs):
            results.append(r)
            print(f"[fuse] {r['scan']}: {r['points']:,} pts  photo/geo/final={r['photo']:.3f}/"
                  f"{r['geo']:.3f}/{r['final']:.3f}  {r['seconds']}s", flush=True)
    mpath = ply_dir / "fusion_manifest.json"
    old = json.loads(mpath.read_text()).get("scenes", []) if mpath.is_file() else []
    done = {r["scan"] for r in results}
    mpath.write_text(json.dumps({
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "depth_cache": str(out_root.resolve()),
        "fusion": {"method": "monomvsnet_" + args.filter, "conf": args.conf, "conf_key": args.conf_key,
                   "params": DYNAMIC if args.filter == "dynamic" else FIXED},
        "scenes": [r for r in old if r.get("scan") not in done] + results}, indent=2))


def main(argv=None) -> None:
    args = parse_args(argv)
    scenes = scene_list(args) if args.phase != "fuse" or args.scenes else \
        [s for s in TNT_SCENES if (Path(args.out) / "depth" / s).is_dir()]
    if args.phase in ("all", "infer"):
        infer(args, scenes)
    if args.phase in ("all", "fuse"):
        fuse(args, scenes)


if __name__ == "__main__":
    main()
