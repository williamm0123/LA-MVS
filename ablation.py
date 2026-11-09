"""Retrained ablations of LAPEmvs (paper Tab. "ablation"): one preset per row.

    python ablation.py --exp A0 --name ABL_A0_E10 --epochs 10          # umhpc, DTU
    python ablation.py --exp B5 --profile local --name abl_smoke --smoke
    python ablation.py --list                                          # the preset table

Same training loop, data, loss, schedule and checkpoints as train_moa.py
(train_moa.run); a preset only pins the architecture, so every row differs from the
full model (FULL = today's defaults: LAPE + LAPE-GRU + 1/2-res refine, LFR off,
moa_gain 1,1,1) in the stated component and nothing else. The switches live in
base/config_moa.py (LAPEConfig section (8)) and are recorded in the checkpoint's
architecture snapshot, so test_moa.py / test_dtu.py rebuild each variant as trained.
Batch scripts, evaluation and the metric table: scripts/ablation/README.md.
"""
from __future__ import annotations

import dataclasses

NO_PRIOR = {"gru": False, "refine": False}

# id -> (paper row, priority, description, moa on, feat backbone, lape overrides, cascade overrides)
EXPERIMENTS = {
    "FULL": ("A4 / full", 1, "full model (extra seeds; seed 1 = LAPE_GRU_DTU_E10)", True, "da3", {}, {}),
    "A0": ("A0", 1, "cascade with DA3 features, no monocular prior", False, "da3", NO_PRIOR, {}),
    "A1": ("A1", 1, "MoA cascade: prior only sets the window centres (no LAPE)", True, "da3",
           NO_PRIOR | {"enabled": False}, {}),
    "A1D": ("A1-D", 1, "simple fusion: pdf at the candidates, renormalised, no mass gate, no v_M gate",
            True, "da3", {"prior_mass": "density", "reliability_gate": False}, {}),
    "B4": ("B4", 1, "analytic sigma, no calibrator", True, "da3", {"sigma_calibrator": False}, {}),
    "B5": ("B5", 1, "pdf at the candidates, renormalised (no bin mass, no mass gate)", True, "da3",
           {"prior_mass": "density"}, {}),
    "B6": ("B6", 1, "no matching-reliability gate (v_M = 0)", True, "da3", {"reliability_gate": False}, {}),
    "B7": ("B7", 2, "logit injection only (no cost adapter)", True, "da3", {"cost_adapter": False}, {}),
    "B8": ("B8", 2, "cost adapter only (no logit injection, gamma_max = 0)", True, "da3", {"gamma_max": 0.0}, {}),
    "B9": ("B9", 2, "single-pass stage 1 (prior evidence from stage 2 on)", True, "da3",
           {"stage1_two_pass": False}, {}),
    "B10": ("B10", 2, "no normal-consistent evidence", True, "da3", {"nce_stages": (False,) * 4}, {}),
    "NOGRU": ("new", 2, "no LAPE-GRU expert", True, "da3", {"gru": False}, {}),
    "NOREF": ("new", 2, "no 1/2-res refinement", True, "da3", {"refine": False}, {}),
    "A0R": ("new", 2, "A0 + the 1/2-res refinement (refine needs no prior)", False, "da3",
            {"gru": False, "refine": True, "refine_without_prior": True}, {}),
    "A0W": ("A0-W", 2, "A0 with 1.5x base half-widths and +50% candidates", False, "da3", NO_PRIOR,
            {"num_depths": (72, 24, 12, 6), "window_halfwidth_bins": (4.5, 3.0, 1.5)}),
    "A0FPN": ("extra", 3, "A0 without the frozen ViT (plain FPN)", False, "none", NO_PRIOR, {}),
}


def describe() -> str:
    rows = [f"{'id':6s} {'paper':10s} tier  description"]
    rows += [f"{k:6s} {v[0]:10s} {v[1]:>4d}  {v[2]}" for k, v in EXPERIMENTS.items()]
    return "\n".join(rows)


def parse_args(argv=None):
    import train_moa        # torch: kept out of module scope so the table imports on a login node

    p = train_moa.build_parser()
    p.description = __doc__
    p.add_argument("--exp", choices=list(EXPERIMENTS), default=None, help="ablation preset (see --list)")
    p.add_argument("--list", action="store_true", help="print the preset table and exit")
    args = p.parse_args(argv)
    if args.list:
        print(describe())
        raise SystemExit(0)
    if args.exp is None:
        p.error("--exp is required (see --list)")
    # the preset is the architecture: refuse switches that would silently change it
    if args.moa != "on" or args.lape is not None or args.feat_backbone is not None:
        p.error("--moa / --lape / --feat-backbone are fixed by --exp")
    if args.freeze_backbone or args.moa_dim or args.init_from:
        p.error("--freeze-backbone / --moa-dim / --init-from: ablations train from scratch at the default width")
    _, _, _, moa_on, feat, _, _ = EXPERIMENTS[args.exp]
    args.moa = "on" if moa_on else "off"
    args.feat_backbone = feat
    if args.name == "MOA":
        args.name = f"ABL_{args.exp}"
    return args


def build_config(args):
    import train_moa

    paper, _, desc, _, _, lape_upd, cascade_upd = EXPERIMENTS[args.exp]
    cfg = train_moa.build_config(args)
    cfg = dataclasses.replace(cfg, lape=dataclasses.replace(cfg.lape, **lape_upd),
                              cascade=dataclasses.replace(cfg.cascade, **cascade_upd))
    lc = cfg.lape
    print(f"[ablation] {args.exp} ({paper}): {desc}")
    print(f"[ablation] moa={cfg.moa.enabled} lape={lc.enabled} feat={cfg.feat.backbone} gru={lc.gru} "
          f"refine={lc.refine}/{lc.refine_without_prior} two_pass={lc.stage1_two_pass} nce={lc.nce_stages} "
          f"prior_mass={lc.prior_mass} v_M={lc.reliability_gate} adapter={lc.cost_adapter} "
          f"gamma_max={lc.gamma_max} calibrator={lc.sigma_calibrator} depths={cfg.cascade.num_depths} "
          f"half={cfg.cascade.window_halfwidth_bins} warp={cfg.cascade.warp_channels}")
    return cfg


def main(argv=None) -> None:
    import train_moa

    train_moa.run(parse_args(argv), config_fn=build_config)


if __name__ == "__main__":
    main()
