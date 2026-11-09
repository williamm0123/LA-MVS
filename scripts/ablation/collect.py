"""Collect the DTU results of every ablation run into one table (scripts/ablation/README.md).

    python scripts/ablation/collect.py                      # -> stdout + log/ablation/summary.{md,csv}
    python scripts/ablation/collect.py --full-seed1 LAPE_GRU_DTU_E10

Reads, per run directory log/ablation/<RUN>/ (written by scripts/ablation/test_umhpc.sh):
    depth_cache/metrics.json     overall.{abs_err, median, acc_1mm, acc_2mm, acc_4mm} and
                                 window.{cov2, cov3, cov4, e_in_mm, e_out_mm, out_pixel_share, out_error_share}
    ply_dypcd/fast_dtu_eval.log  "mean acc / comp / overall" (Fast-DTU-Evaluation), or result.txt
Run names map to presets as ABL_<EXP>_E<epochs>[_S<seed>]; --full-seed1 names the run that is
FULL's seed 1. Rows with several seeds are reported as mean +- std (sample std) over seeds.
No torch needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import ablation  # noqa: E402  (torch-free table)

# (key, header, scale, digits, lower is better)
COLUMNS = [
    ("acc", "Acc.", 1.0, 4, True),
    ("comp", "Comp.", 1.0, 4, True),
    ("overall", "Overall", 1.0, 4, True),
    ("abs_err", "MAE", 1.0, 3, True),
    ("median", "Med.", 1.0, 3, True),
    ("acc_1mm", "<1mm%", 100.0, 2, False),
    ("acc_2mm", "<2mm%", 100.0, 2, False),
    ("acc_4mm", "<4mm%", 100.0, 2, False),
    ("cov2", "Cov2%", 100.0, 2, False),
    ("cov3", "Cov3%", 100.0, 2, False),
    ("cov4", "Cov4%", 100.0, 2, False),
    ("e_in_mm", "E_in", 1.0, 3, True),
    ("e_out_mm", "E_out", 1.0, 2, True),
    ("out_pixel_share", "out px%", 100.0, 2, True),
    ("out_error_share", "out err%", 100.0, 2, True),
]
RUN_RE = re.compile(r"^ABL_(?P<exp>[A-Z0-9]+)_E\d+(?:_S(?P<seed>\d+))?$")
MEAN_RE = re.compile(r"mean\s+(acc|comp|overall)\s*:?\s*([0-9.]+)")


def point_cloud(ply_dir: Path) -> dict[str, float]:
    for name in ("fast_dtu_eval.log", "result.txt", "python_eval_result.txt"):
        f = ply_dir / name
        if f.is_file():
            found = {k: float(v) for k, v in MEAN_RE.findall(f.read_text(errors="ignore"))}
            if len(found) == 3:
                return found
    return {}


def load_run(root: Path) -> dict[str, float]:
    row: dict[str, float] = {}
    mj = root / "depth_cache" / "metrics.json"
    if mj.is_file():
        m = json.loads(mj.read_text())
        row.update({k: v for k, v in m.get("overall", {}).items() if isinstance(v, (int, float))})
        row.update(m.get("window", {}))
    row.update(point_cloud(root / "ply_dypcd"))
    return row


def identify(run: str, full_seed1: str) -> tuple[str, int] | None:
    if run == full_seed1:
        return "FULL", 1
    m = RUN_RE.match(run)
    if not m or m["exp"] not in ablation.EXPERIMENTS:
        return None
    return m["exp"], int(m["seed"] or 1)


def mean_std(vals: list[float]) -> tuple[float, float]:
    mu = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else float("nan")
    return mu, sd


def fmt(mu: float | None, sd: float, scale: float, digits: int) -> str:
    if mu is None:
        return "-"
    s = f"{mu * scale:.{digits}f}"
    return s + (f"±{sd * scale:.{digits}f}" if not math.isnan(sd) else "")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(REPO / "log/ablation"))
    p.add_argument("--full-seed1", default="LAPE_GRU_DTU_E10", help="run name that is FULL seed 1")
    p.add_argument("--out", default=None, help="default <root>/summary")
    args = p.parse_args(argv)
    root = Path(args.root)
    runs: dict[str, dict[int, dict]] = {}
    per_run = []
    for d in sorted(root.glob("*/")) if root.is_dir() else []:
        ident = identify(d.name, args.full_seed1)
        if ident is None:
            continue
        row = load_run(d)
        if not row:
            continue
        exp, seed = ident
        runs.setdefault(exp, {})[seed] = row
        per_run.append({"run": d.name, "exp": exp, "seed": seed, **row})
    if not per_run:
        raise SystemExit(f"没有找到结果: {root}/<ABL_*>/depth_cache/metrics.json (先跑 scripts/ablation/test_umhpc.sh)")

    header = ["ID", "paper", "n"] + [c[1] for c in COLUMNS] + ["ΔOverall", "ΔMAE"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    agg_rows = []
    full = {k: mean_std([r[k] for r in runs["FULL"].values() if k in r])[0]
            for k in ("overall", "abs_err") if "FULL" in runs and any(k in r for r in runs["FULL"].values())}
    for exp, spec in ablation.EXPERIMENTS.items():
        if exp not in runs:
            continue
        seeds = runs[exp]
        cells, agg = [], {"ID": exp, "paper": spec[0], "n": len(seeds)}
        for key, _, scale, digits, _ in COLUMNS:
            vals = [r[key] for r in seeds.values() if key in r]
            if vals:
                mu, sd = mean_std(vals)
                agg[key], agg[key + "_std"] = mu, sd
                cells.append(fmt(mu, sd, scale, digits))
            else:
                cells.append("-")
        for key, digits in (("overall", 4), ("abs_err", 3)):
            if key in agg and key in full and exp != "FULL":
                cells.append(f"{agg[key] - full[key]:+.{digits}f}")
            else:
                cells.append("-")
        lines.append("| " + " | ".join([exp, spec[0], str(len(seeds))] + cells) + " |")
        agg_rows.append(agg)

    table = "\n".join(lines)
    note = ("Acc/Comp/Overall: DTU point cloud (mm, Fast-DTU-Evaluation). MAE/Med.: depth (mm) over valid GT "
            "in the depth range; <k mm: share of those pixels. CovS: GT inside the stage-S candidate interval. "
            "E_in/E_out: final-depth MAE of pixels inside/outside the stage-4 window; out px/err: their share of "
            "pixels / of the total abs error. Δ = row - FULL (negative = better for both). ± = std over seeds.")
    print(table)
    print("\n" + note)
    out = Path(args.out) if args.out else root / "summary"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(table + "\n\n" + note + "\n", encoding="utf-8")
    keys = ["run", "exp", "seed"] + [c[0] for c in COLUMNS]
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_run)
    print(f"\n-> {out.with_suffix('.md')}  {out.with_suffix('.csv')} (per run)")


if __name__ == "__main__":
    main()
