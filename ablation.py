"""Pure-MVS ablation: the LAPE-GRU training run with the monocular prior removed.

    python ablation.py --name ABL_MVS_DA3_E10 --variant mvs_da3 --epochs 10      # umhpc, DTU
    python ablation.py --name abl_smoke --variant mvs_fpn --profile local --smoke

Same training loop, data, loss, schedule and checkpoints as train_moa.py
(train_moa.run); only the architecture is pinned, so the comparison with a LAPE /
LAPE-GRU run differs in the prior and nothing else:

* ``mvs_da3`` (default): frozen DA3 encoder tokens -> SVA -> FPN, 4-stage cascade,
  every window centred on the parent stage's own MVS depth. No MoA / LAPE / LAPE-GRU,
  no prior fusion, no 1/2-res refinement; DA3's DPT head is not even run, so no
  monocular depth exists anywhere in the graph. Isolates what the prior adds.
* ``mvs_fpn``: additionally drops the frozen ViT (feat.backbone = none): plain FPN +
  FMT pathway + cascade, i.e. the matching network with no foundation features.

The loss reduces to the per-stage CE terms (MoALoss skips the absent prior /
GRU / refine outputs). Evaluate with test_dtu.py as usual: the checkpoint's
architecture snapshot carries moa.enabled = False and feat.backbone.
"""
from __future__ import annotations

import dataclasses

import train_moa

VARIANTS = {
    "mvs_da3": "DA3 tokens -> SVA -> FPN + 4-stage cascade, no monocular prior",
    "mvs_fpn": "plain FPN + 4-stage cascade, no frozen ViT, no monocular prior",
}


def parse_args(argv=None):
    p = train_moa.build_parser()
    p.description = __doc__
    p.add_argument("--variant", choices=sorted(VARIANTS), default="mvs_da3",
                   help="; ".join(f"{k}: {v}" for k, v in VARIANTS.items()))
    p.set_defaults(name="ABL_MVS")
    args = p.parse_args(argv)
    # the prior is what is being ablated: refuse switches that would bring it back or need it
    if args.moa != "on" or args.lape is not None:
        p.error("--moa / --lape are fixed by --variant here (both off)")
    if args.feat_backbone is not None:
        p.error("--feat-backbone is fixed by --variant (mvs_da3 -> da3, mvs_fpn -> none)")
    if args.freeze_backbone or args.moa_dim or args.da3_root:
        p.error("--freeze-backbone / --moa-dim / --da3-root only apply to the prior")
    args.moa = "off"
    args.feat_backbone = "da3" if args.variant == "mvs_da3" else "none"
    return args


def build_config(args):
    cfg = train_moa.build_config(args)
    cfg = dataclasses.replace(cfg, lape=dataclasses.replace(cfg.lape, enabled=False))
    assert not train_moa.mono_from_cache(cfg)        # no DA3 cache is read either
    print(f"[ablation] variant={args.variant}: {VARIANTS[args.variant]}  "
          f"(moa={cfg.moa.enabled} lape={cfg.lape.enabled} feat={cfg.feat.backbone} "
          f"warp={cfg.cascade.warp_channels} depths={cfg.cascade.num_depths})")
    return cfg


def main(argv=None) -> None:
    train_moa.run(parse_args(argv), config_fn=build_config)


if __name__ == "__main__":
    main()
