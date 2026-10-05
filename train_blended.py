"""Fine-tune a DTU-trained MoAMVSNet on BlendedMVS, balanced with DTU (MVSFormer++ recipe).

    python train_blended.py --name LAPE_BLD_E10 --init-from log/experiments/LAPE_DTU_E10/model/latest.pth \
        --blended-root /scr/user/qinglong/dataset/BlendedMVS_lowres --epochs 10 --lr 1e-4

Same training loop as train_moa.py (train_moa.run); only the config source and the
dataset change:

* The architecture comes from the ``--init-from`` checkpoint's snapshot, not from
  today's defaults, and the init is strict (every trainable weight must be found).
  A LAPE checkpoint (feat.backbone = da3) runs DA3 online: no DA3 cache is read or
  built for DTU or Blended. Legacy moa1 checkpoints still work: their snapshot lacks
  the ``feat`` / ``lape`` sections, so they are rebuilt with DINOv3 + the DA3 cache, and
  ``--moa1-semantics auto`` pins the three behaviour fields moa1 predates
  (global_solver = huber, moa_gain = 1,1,1, edge_snap = off).
* Weights only: a fresh optimizer and a fresh warmup + cosine schedule. Requeues
  resume from this run's own latest.pth as usual.
* Data: data/blended_moa.py — the original low-res BlendedMVS (576x768) with the
  official 106/7 split MVSFormer++ uses (lists/blended/), multi-scale crops up to the
  native frame.
* ``--mix-dtu on`` (default) = MVSFormer++'s ``--balanced_training``
  (reference/MVSFormerPlusPlus datasets/balanced_sampling.py + config/mvsformer++_ft.json):
  each epoch takes min(len) samples from DTU (lists/dtu/trainval.txt) and from
  Blended-train and shuffles them together; one crop size per batch across both
  (their CustomConcatDataset.reset_dataset); 10 epochs, lr 1e-4, warmup 500.
  Validation: Blended-val. ``--mix-dtu off`` = Blended only.
* ``--mvsformer-sampling on`` (default), as their *_dataset_ms.py in training:
  sources = random nviews-1 of pair.txt's top 7 (Blended) / all 10 (DTU), and a
  random crop is re-drawn while its 1/8 GT mask is empty.
* Metrics: errors are multiplied by each sample's ``metric_scale``: 1 on DTU (mm),
  1/interval on Blended — ``acc_2mm`` there = within 2 hypothesis intervals, the
  convention of MVSFormer++'s Blended validation. The per-stage ``err_s*_mm`` /
  ``moa*_err_*_mm`` diagnostics stay in raw scene units.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

import train_moa
from base.config_moa import apply_arch_snapshot
from data.augment import PhotometricAug
from data.blended_moa import BLENDED_SCALES, BalancedMixDataset, MoABlendedDataset
from data.dtu_moa import MoADTUDataset

REPO = Path(__file__).resolve().parent
MOA1_SEMANTICS = {"global_solver": ("huber",) * 3, "moa_gain": (1.0,) * 3, "edge_snap": (False,) * 3}


def parse_args(argv=None):
    p = train_moa.build_parser()
    p.description = __doc__
    p.add_argument("--blended-root", default="/scr/user/qinglong/dataset/BlendedMVS_lowres")
    p.add_argument("--mix-dtu", choices=["on", "off"], default="on",
                   help="on: balanced DTU + Blended training (MVSFormer++ --balanced_training)")
    p.add_argument("--dtu-da3-root", default=None, help="DTU DA3 cache (default cfg.paths.da3_cache_path)")
    p.add_argument("--dtu-train-list", default=str(REPO / "lists/dtu/trainval.txt"),
                   help="DTU scans in the mix (MVSFormer++ fine-tuning uses trainval)")
    p.add_argument("--mvsformer-sampling", choices=["on", "off"], default="on",
                   help="random sources (Blended top-7 / DTU all) + re-draw crops with an empty GT mask")
    p.add_argument("--arch-from", default=None,
                   help="checkpoint whose architecture snapshot builds the net (default: --init-from)")
    p.add_argument("--moa1-semantics", choices=["auto", "on", "off"], default="auto",
                   help="auto: pin global_solver=huber, moa_gain=1,1,1, edge_snap=off only when the "
                        "checkpoint's snapshot predates those fields (moa1); on: always; off: never")
    p.set_defaults(train_list=str(REPO / "lists/blended/training_list.txt"),
                   val_list=str(REPO / "lists/blended/validation_list.txt"),
                   da3_root=str(REPO / "log/da3_cache_blended"), lr=1e-4, warmup_steps=500, epochs=10)
    args = p.parse_args(argv)
    if args.warp_channels or args.moa_dim:
        p.error("--warp-channels / --moa-dim come from the checkpoint's architecture here")
    if not (args.arch_from or args.init_from):
        p.error("need --init-from (or --arch-from): the architecture is taken from that checkpoint")
    return args


def build_config(args):
    cfg = train_moa.build_config(args)
    src = args.arch_from or args.init_from
    arch = torch.load(src, map_location="cpu", weights_only=False)["arch"]
    cfg = apply_arch_snapshot(cfg, arch)
    predates = [k for k in MOA1_SEMANTICS if k not in arch.get("moa", {})]
    if args.moa1_semantics == "on" or (args.moa1_semantics == "auto" and predates):
        cfg = dataclasses.replace(cfg, moa=dataclasses.replace(cfg.moa, **MOA1_SEMANTICS))
    cfg = dataclasses.replace(cfg, moa=dataclasses.replace(cfg.moa, enabled=args.moa == "on"),
                              augment=dataclasses.replace(cfg.augment, scales=BLENDED_SCALES))
    m = cfg.moa
    print(f"[blended] mix_dtu={args.mix_dtu}  mvsformer_sampling={args.mvsformer_sampling}  "
          f"scales={BLENDED_SCALES}")
    print(f"[blended] arch from {src}: warp={cfg.cascade.warp_channels} depths={cfg.cascade.num_depths} "
          f"feat={cfg.feat.backbone} lape={'on' if cfg.lape.enabled else 'off'} "
          f"moa={'on' if m.enabled else 'off'} global_solver={m.global_solver} "
          f"moa_gain={m.moa_gain} edge_snap={m.edge_snap}")
    return cfg


def build_datasets(cfg, args):
    t, aug = cfg.train, cfg.augment
    load_mono = train_moa.mono_from_cache(cfg)        # False with the DA3 backbone: mono depth is online
    common = dict(nviews=t.num_views, seed=t.seed, da3_root=Path(args.da3_root),
                  load_mono=load_mono, da3_missing=t.da3_missing)
    train_kw = dict(
        mode="train",
        aug=PhotometricAug(brightness=aug.brightness, contrast=aug.contrast, saturation=aug.saturation,
                           hue=aug.hue, min_gamma=aug.min_gamma, max_gamma=aug.max_gamma)
        if aug.photometric else None,
        scales=aug.scales if t.multi_scale else (), resize_range=aug.resize_range,
        height=t.height, width=t.width)
    ms = args.mvsformer_sampling == "on"
    train_ds = MoABlendedDataset(args.blended_root, args.train_list, **train_kw, **common)
    train_ds.src_shuffle_top, train_ds.crop_retry = (7, 50) if ms else (0, 0)
    if args.mix_dtu == "on":
        dtu = MoADTUDataset(
            cfg.paths.dtu_train_root, args.dtu_train_list, **train_kw,
            **{**common, "da3_root": Path(args.dtu_da3_root or cfg.paths.da3_cache_path)})
        dtu.src_shuffle_top, dtu.crop_retry = (-1, 50) if ms else (0, 0)
        train_ds = BalancedMixDataset({"dtu": dtu, "blended": train_ds}, seed=t.seed)
    val_ds = MoABlendedDataset(args.blended_root, args.val_list, mode="val", **common)
    if args.max_val_samples and len(val_ds.metas) > args.max_val_samples:
        idx = np.linspace(0, len(val_ds.metas) - 1, args.max_val_samples).round().astype(int)
        val_ds.metas = [val_ds.metas[i] for i in idx]
    return train_ds, val_ds


def main(argv=None) -> None:
    args = parse_args(argv)
    args.init_strict = True
    train_moa.run(args, config_fn=build_config, datasets_fn=build_datasets)


if __name__ == "__main__":
    main()
