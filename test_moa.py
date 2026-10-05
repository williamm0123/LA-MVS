"""Inference for MoAMVSNet: depth metrics + per-view depth cache for fusion.

    python test_moa.py --ckpt log/experiments/MOA_v1/model/best.pth --split test \
        --full-image --resize-scale 0.8 --num-views 5 --out log/depth_cache/MOA_v1_test
    python points_fusibile.py --out log/depth_cache/MOA_v1_test --ply-dir log/pred_points_MOA_v1 ...

Writes ``<out>/metrics.json``, ``<out>/run_manifest.json`` and
``<out>/depth/<scan>/<ref:08d>.npz`` (depth, conf, conf_last float32, K, E, image, src_views) —
the layout test.py produces and points_fusibile.py consumes. The network is
rebuilt from the architecture snapshot stored in the checkpoint.
With --skip-existing, structurally complete per-view caches are skipped before
dataset loading and model inference. Metrics then cover newly inferred views only.
"""
from __future__ import annotations

import argparse
import datetime
import json
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from base.config_moa import apply_arch_snapshot, build_moa_config
from data.dtu_moa import MoADTUDataset
from models.network_moa import MoAMVSNet
from train_moa import collate, git_state, load_checkpoint, load_model_state, metric_mask


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser("MoAMVSNet test / DTU inference")
    p.add_argument("--profile", choices=["local", "umhpc"], default="umhpc")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--list", default=None, help="override the split's scan list")
    p.add_argument("--num-views", type=int, default=5)
    p.add_argument("--resize-scale", type=float, default=0.8)
    p.add_argument("--full-image", action="store_true",
                   help="whole resized frame instead of the 512x640 centre crop")
    p.add_argument("--scans", type=int, nargs="+", default=None)
    p.add_argument("--max-scans", type=int, default=0)
    p.add_argument("--max-refs", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--da3-root", default=None, help="DA3 cache, only for legacy (DINOv3) checkpoints")
    p.add_argument("--out", default=None, help="default log/depth_cache/<ckpt run>_<split>")
    p.add_argument("--skip-existing", action="store_true",
                   help="跳过已有的有效逐视角 NPZ，适合中断后续跑")
    p.add_argument("--fuse", action=argparse.BooleanOptionalAction, default=True,
                   help="write the per-view npz cache (--no-fuse = metrics only)")
    p.add_argument("--da3-missing", choices=["error", "skip"], default="error",
                   help="DA3 缓存缺样本时: error (默认, 启动即报错) / skip (丢掉那些视角 —— "
                        "注意融合出来的点云会因此缺一块, 但照样会被打分)")
    p.add_argument("--conf-window", type=int, default=1,
                   help="+-bins around each stage's argmax for the fusion confidence")
    p.add_argument("--moa-gain", default=None, metavar="G2,G3,G4",
                   help="覆盖 checkpoint 的 moa_gain, 例如 1,1,1 (在该字段存在之前训练的权重)")
    p.add_argument("--edge-snap", choices=["on", "off"], default=None,
                   help="覆盖 checkpoint 的 edge_snap (在该字段存在之前训练的权重用 off)")
    return p.parse_args(argv)


_CACHE_MEMBERS = {f"{key}.npy" for key in ("depth", "conf", "K", "E", "image", "src_views")}


def valid_depth_cache(path: Path) -> bool:
    """Check the ZIP directory without decompressing each cached image."""
    if not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            return _CACHE_MEMBERS <= names and all(
                z.getinfo(name).file_size > 0 for name in _CACHE_MEMBERS)
    except (OSError, zipfile.BadZipFile):
        return False


def mode_mass(prob: torch.Tensor, window: int, hw) -> torch.Tensor:
    """Posterior mass within +-window bins of the argmax, resized to ``hw`` [B,H,W]."""
    D = prob.shape[1]
    w = min(2 * window + 1, D)
    idx = prob.argmax(dim=1, keepdim=True)
    start = (idx - w // 2).clamp(0, D - w)
    offs = torch.arange(w, device=prob.device).view(1, -1, 1, 1)
    mass = prob.float().gather(1, start + offs).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    if tuple(mass.shape[-2:]) != tuple(hw):
        mass = torch.nn.functional.interpolate(mass, size=tuple(hw), mode="bilinear", align_corners=False)
    return mass[:, 0]


def last_stage_confidence(outputs: dict) -> torch.Tensor:
    """Stage-4 max posterior, the photometric confidence MonoMVSNet fuses with (test_dtu.py)."""
    hw = outputs["depth_full"].shape[-2:]
    pmax = outputs["stage4"]["prob"].float().amax(dim=1, keepdim=True)
    if tuple(pmax.shape[-2:]) != tuple(hw):
        pmax = torch.nn.functional.interpolate(pmax, size=tuple(hw), mode="bilinear", align_corners=False)
    return pmax[:, 0]


def cascade_confidence(outputs: dict, window: int) -> torch.Tensor:
    """Product over the four stages. Stage 4 alone (4 bins) would saturate at 1."""
    hw = outputs["depth_full"].shape[-2:]
    conf = torch.ones_like(outputs["depth_full"])
    for s in range(1, 5):
        conf = conf * mode_mass(outputs[f"stage{s}"]["prob"], window, hw)
    return conf


class ScanMeter:
    def __init__(self) -> None:
        self.sums = defaultdict(lambda: np.zeros(6, dtype=np.float64))
        self.pool: dict[str, list] = defaultdict(list)

    def update(self, scan: str, err: torch.Tensor) -> None:
        e = err.detach().float()
        s = self.sums[scan]
        s[0] += e.sum().item()
        s[1] += e.numel()
        for i, t in enumerate((1.0, 2.0, 4.0, 8.0)):
            s[2 + i] += (e < t).sum().item()
        if e.numel():
            self.pool[scan].append(e[:: max(e.numel() // 4096, 1)].cpu().numpy())

    @staticmethod
    def _summ(s, pool) -> dict:
        n = max(s[1], 1.0)
        pool = np.concatenate(pool) if pool else np.zeros(1)
        return {"abs_err": s[0] / n, "median": float(np.median(pool)), "p90": float(np.percentile(pool, 90)),
                "acc_1mm": s[2] / n, "acc_2mm": s[3] / n, "acc_4mm": s[4] / n, "acc_8mm": s[5] / n,
                "pixels": int(s[1])}

    def per_scan(self) -> dict:
        return {k: self._summ(self.sums[k], self.pool[k]) for k in self.sums}

    def overall(self) -> dict:
        tot = np.sum([self.sums[k] for k in self.sums], axis=0) if self.sums else np.zeros(6)
        return self._summ(tot, [v for vs in self.pool.values() for v in vs])


def build_dataset(cfg, args, load_mono: bool, da3_root: Path) -> MoADTUDataset:
    base = cfg.paths.val_list_file if args.split == "val" else cfg.paths.test_list_file
    ds = MoADTUDataset(cfg.paths.dtu_train_root, args.list or str(base), nviews=args.num_views,
                       mode=args.split, da3_root=da3_root, load_mono=False, da3_missing=args.da3_missing)
    # the parent __init__ swallows a resize_scale keyword; set the attribute
    ds.resize_scale = args.resize_scale
    if args.full_image:
        ds.height = int(round(1200 * args.resize_scale))
        ds.width = int(round(1600 * args.resize_scale))
    per_scan: dict[str, list] = defaultdict(list)
    for m in ds.metas:
        per_scan[m[0]].append(m)
    scans = list(per_scan)
    if args.scans:
        want = {f"scan{s}" for s in args.scans}
        missing = want - set(scans)
        if missing:
            raise SystemExit(f"--scans: {sorted(missing)} not in the {args.split} list")
        scans = [s for s in scans if s in want]
    if args.max_scans > 0:
        scans = scans[: args.max_scans]
    ds.metas = [m for s in scans for m in (per_scan[s][: args.max_refs] if args.max_refs > 0 else per_scan[s])]
    if load_mono:
        ds.load_mono = True
        ds.da3_root = Path(da3_root)
        ds._check_da3(args.da3_missing)
    for side in (ds.height, ds.width):
        if side % 8:
            raise SystemExit(f"crop {ds.height}x{ds.width} is not a multiple of 8 (resize {args.resize_scale})")
    return ds


@torch.no_grad()
def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = load_checkpoint(args.ckpt, map_location="cpu")
    if ck.get("kind") != "moa_mvsnet":
        raise SystemExit(f"{args.ckpt} is not a MoAMVSNet checkpoint (use test.py for UprMVSNet)")
    cfg = apply_arch_snapshot(build_moa_config(args.profile), ck["arch"])
    moa_over = {}
    if args.moa_gain:
        g = tuple(float(x) for x in args.moa_gain.split(","))
        if len(g) != 3:
            raise SystemExit("--moa-gain 需要三个值, 例如 1,1,1")
        moa_over["moa_gain"] = g
    if args.edge_snap:
        moa_over["edge_snap"] = (args.edge_snap == "on",) * 3
    if moa_over:
        import dataclasses as _dc
        cfg = _dc.replace(cfg, moa=_dc.replace(cfg.moa, **moa_over))
        print(f"[test] 覆盖 MoA 推理行为: {moa_over}")
    model = MoAMVSNet(cfg).to(device)
    load_model_state(model, ck["model"])
    model.eval()
    tcfg = ck.get("config", {}).get("train", {})
    amp_dtype = torch.bfloat16 if tcfg.get("amp_dtype", "bf16") == "bf16" else torch.float16
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"

    da3_root = Path(args.da3_root) if args.da3_root else Path(cfg.paths.da3_cache_path)
    ds = build_dataset(cfg, args, model.needs_mono_cache, da3_root)
    if model.needs_mono_cache and ck.get("da3_process_res") not in (None, ds.da3_process_res):
        print(f"[test] WARNING DA3 cache process_res {ds.da3_process_res} != training "
              f"{ck.get('da3_process_res')}")
    da3_info = ({"source": "online", "process_res": model.da3_process_res} if model.da3_sva is not None
                else {"source": "cache", "root": str(da3_root) if model.uses_mono else None,
                      "process_res": ds.da3_process_res})
    run = Path(args.ckpt).resolve().parent.parent.name
    out_root = Path(args.out) if args.out else Path(cfg.paths.depth_cache_path) / f"{run}_{args.split}"
    out_root.mkdir(parents=True, exist_ok=True)
    selected_views = len(ds)
    todo_indices = list(range(selected_views))
    cached_views = invalid_cache = 0
    if args.fuse and args.skip_existing:
        todo_indices = []
        for idx, (scan, _light, ref_view, _src_views) in enumerate(ds.metas):
            cache_path = out_root / "depth" / scan / f"{ref_view:08d}.npz"
            if valid_depth_cache(cache_path):
                cached_views += 1
            else:
                todo_indices.append(idx)
                invalid_cache += cache_path.is_file()
    print(f"[test] {args.ckpt} step {ck.get('step')}  moa={'on' if model.uses_mono else 'off'}  "
          f"lape={'on' if model.lape_on else 'off'}  da3={da3_info['source']}  "
          f"{selected_views} samples  {ds.height}x{ds.width}  amp={amp_dtype if use_amp else 'off'}  -> {out_root}")
    if args.fuse and args.skip_existing:
        print(f"[test] 续跑检查: 跳过已有 NPZ {cached_views}/{selected_views}，"
              f"待推理 {len(todo_indices)}，无效缓存重算 {invalid_cache}", flush=True)
        if todo_indices:
            scan, _light, ref_view, _src_views = ds.metas[todo_indices[0]]
            print(f"[test] 从 {todo_indices[0] + 1}/{selected_views} ({scan} ref {ref_view}) 继续", flush=True)
    if not todo_indices:
        print("[test] 所选视角均已有缓存，无需推理；已有 metrics.json 保持不变", flush=True)
        return

    loader = DataLoader(Subset(ds, todo_indices), batch_size=1, shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate, pin_memory=True)
    meter = ScanMeter()
    moa_stats: dict[str, float] = defaultdict(float)
    for i, batch in enumerate(loader):
        selected_idx = todo_indices[i]
        scan, _light, ref_view, src_views = ds.metas[selected_idx]
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(batch)
        pred = out["depth_full"].float()
        if i == 0 and not torch.isfinite(pred).any():
            raise SystemExit("first depth map is entirely non-finite — check the checkpoint / amp dtype")
        conf = cascade_confidence(out, args.conf_window)
        m = metric_mask(batch) & (batch["depth_gt"] > 0)
        if m.any():
            meter.update(scan, (pred[m] - batch["depth_gt"].float()[m]).abs())
        for s in (2, 3, 4):
            mo = out.get(f"moa{s}")
            if mo is not None:
                moa_stats[f"moa{s}_override_rate"] += float((mo.alpha > 0.5).float().mean())
                moa_stats[f"moa{s}_pi_mvs"] += float(mo.mixture_weights[:, 0].mean())
                moa_stats[f"moa{s}_global_ok"] += float(mo.global_ok.float().mean())
        if args.fuse:
            d = out_root / "depth" / scan
            d.mkdir(parents=True, exist_ok=True)
            cache_path = d / f"{ref_view:08d}.npz"
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(mode="wb", dir=d, prefix=f".{ref_view:08d}.",
                                                 suffix=".npz.part", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                    np.savez_compressed(
                        tmp,
                        depth=pred[0].cpu().numpy().astype(np.float32),
                        conf=conf[0].cpu().numpy().astype(np.float32),
                        conf_last=last_stage_confidence(out)[0].cpu().numpy().astype(np.float32),
                        K=batch["intrinsics"][0, 0].float().cpu().numpy(),
                        E=batch["extrinsics"][0, 0].float().cpu().numpy(),
                        image=batch["images"][0, 0].permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy(),
                        src_views=np.asarray(src_views, dtype=np.int64),
                    )
                tmp_path.replace(cache_path)
            finally:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)
        if (i + 1) % 20 == 0 or i + 1 == len(todo_indices):
            print(f"[test] 新推理 {i + 1}/{len(todo_indices)}，"
                  f"总进度 {selected_idx + 1}/{selected_views} ({scan} ref {ref_view})", flush=True)

    n = max(len(todo_indices), 1)
    summary = {"overall": meter.overall(), "per_scan": meter.per_scan(),
               "moa": {k: v / n for k, v in moa_stats.items()}}
    if cached_views:
        summary["coverage"] = {"selected_views": selected_views,
                               "newly_inferred_views": len(todo_indices),
                               "cached_views_skipped": cached_views,
                               "invalid_cache_recomputed": invalid_cache,
                               "metric_scope": "newly_inferred_views_only"}
    # 只跑了一部分 scan 时不要抹掉已有的: 合并逐 scan 结果, 并按像素数重新加权出
    # 总体的 abs_err / acc (这两个可以精确合并)。median / p90 需要误差池, 跨 run
    # 无法重建, 所以只报本次这批 scan 的。
    path = out_root / "metrics.json"
    if cached_views and path.is_file():
        path = out_root / "metrics_resume.json"
    if not cached_views and path.is_file():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            old = {}
        merged = {**old.get("per_scan", {}), **summary["per_scan"]}
        if len(merged) > len(summary["per_scan"]):
            kept = sorted(set(merged) - set(summary["per_scan"]))
            print(f"[test] 合并已有的 {len(kept)} 个 scan 的结果: {', '.join(kept)}")
            tot = sum(v["pixels"] for v in merged.values()) or 1
            summary["overall_this_run"] = summary.pop("overall")
            summary["overall"] = {
                k: sum(v[k] * v["pixels"] for v in merged.values()) / tot
                for k in ("abs_err", "acc_1mm", "acc_2mm", "acc_4mm", "acc_8mm")}
            summary["overall"].update(pixels=tot, scans=len(merged),
                                      note="median/p90 仅在 overall_this_run 里, 跨 run 无法合并")
            summary["per_scan"] = merged
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "checkpoint": {"path": str(Path(args.ckpt).resolve()), "step": ck.get("step"),
                       "best_metric": ck.get("best_metric"), "arch": ck.get("arch"), "git": ck.get("git")},
        "code_git": git_state(),
        "da3": da3_info,
        "inference": {"split": args.split, "num_views": args.num_views, "resize_scale": args.resize_scale,
                      "full_image": bool(args.full_image), "max_scans": args.max_scans,
                      "max_refs": args.max_refs, "conf_window": args.conf_window,
                      "skip_existing": args.skip_existing,
                      "selected_views": selected_views, "newly_inferred_views": len(todo_indices),
                      "cached_views_skipped": cached_views,
                      "invalid_cache_recomputed": invalid_cache},
    }
    (out_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    o = summary.get("overall_this_run", summary["overall"])
    print(f"[test] 本次新推理 {len(todo_indices)} 视角: "
          f"abs_err={o['abs_err']:.4f} median={o['median']:.4f} "
          f"acc_2mm={o['acc_2mm']:.4f} acc_4mm={o['acc_4mm']:.4f}")
    if "overall_this_run" in summary:
        m = summary["overall"]
        print(f"[test] 合并 {m['scans']} 个 scan: abs_err={m['abs_err']:.4f} acc_2mm={m['acc_2mm']:.4f} "
              f"acc_4mm={m['acc_4mm']:.4f}")
    if cached_views:
        print(f"[test] 指标仅覆盖本次新推理的 {len(todo_indices)} 个视角；"
              f"跳过的 {cached_views} 个缓存视角未重新计算指标")
    print(f"[test] -> {path}")


if __name__ == "__main__":
    main()
