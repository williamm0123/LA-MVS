#!/usr/bin/env python3
"""Check a BlendedMVS(+/++) folder and write the scene lists train_blended.py reads.

    python scripts/blended_lists.py --root /scr/user/qinglong/dataset/BlendedMVS_plus

The folder names (``5a3ca9cb270f0e3f14d0eddb``, ``000000000000000000000001``, ...)
are BlendedMVS's own scene ids, not corrupted names: every published scene list
refers to scenes by them. Nothing is renamed here.

Per scene it checks that ``cams/pair.txt`` exists and that every view it lists
has an image, a camera and a depth map, and records the image size. Scenes with
problems are reported and left out of the lists. Output (``--out``, default
lists/blended_plus/):

    all.txt    every usable scene
    val.txt    --num-val scenes, evenly spaced over the sorted list (deterministic)
    train.txt  the rest
    report.tsv scene, #refs, #views, image size, problems

An existing val.txt is kept (so the split cannot drift between runs) unless
--force is given.

``--check-lists A B ...`` instead only checks the scenes named in existing list
files (e.g. the official lists/blended/{training,validation}_list.txt) and exits
non-zero if any is missing or incomplete; nothing is written.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.blended_moa import read_pair  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def check_scene(d: Path) -> tuple[dict, list[str]]:
    probs: list[str] = []
    info = {"refs": 0, "views": 0, "hw": None}
    pair = d / "cams" / "pair.txt"
    if not pair.is_file():
        return info, ["no cams/pair.txt"]
    try:
        pairs = read_pair(pair)
    except Exception as exc:  # noqa: BLE001
        return info, [f"bad pair.txt: {exc}"]
    views = sorted({r for r, _ in pairs} | {s for _, ss in pairs for s in ss})
    info["refs"] = sum(1 for _, ss in pairs if ss)
    info["views"] = len(views)
    miss = Counter()
    for v in views:
        if not (d / "blended_images" / f"{v:08d}.jpg").is_file():
            miss["image"] += 1
        if not (d / "cams" / f"{v:08d}_cam.txt").is_file():
            miss["cam"] += 1
        if not (d / "rendered_depth_maps" / f"{v:08d}.pfm").is_file():
            miss["depth"] += 1
    probs += [f"{n} views without {k}" for k, n in miss.items()]
    if views and not miss["image"]:
        with Image.open(d / "blended_images" / f"{views[0]:08d}.jpg") as im:
            info["hw"] = (im.height, im.width)
    if info["refs"] == 0:
        probs.append("no reference view with sources")
    return info, probs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="/scr/user/qinglong/dataset/BlendedMVS_plus")
    p.add_argument("--out", default=str(REPO / "lists/blended_plus"))
    p.add_argument("--num-val", type=int, default=7)
    p.add_argument("--force", action="store_true", help="re-draw val.txt even if it exists")
    p.add_argument("--check-lists", nargs="+", default=None, metavar="LIST",
                   help="only check the scenes in these list files; write nothing")
    args = p.parse_args()

    root, out = Path(args.root), Path(args.out)
    if args.check_lists:
        names = list(dict.fromkeys(s.strip() for f in args.check_lists
                                   for s in Path(f).read_text().splitlines() if s.strip()))
        bad, n_refs, hws = [], 0, Counter()
        for name in names:
            d = root / name
            info, probs = check_scene(d) if d.is_dir() else ({"refs": 0, "hw": None}, ["folder missing"])
            n_refs += info["refs"]
            hws[info["hw"]] += 1
            if probs:
                bad.append(name)
                print(f"  !! {name}: {'; '.join(probs)}")
        print(f"[blended] {len(names)} listed scenes under {root}: {len(names) - len(bad)} ok, "
              f"{len(bad)} bad, {n_refs} reference views; image sizes {dict(hws)}")
        raise SystemExit(1 if bad else 0)
    scenes = sorted(d for d in root.iterdir() if d.is_dir())
    good, rows = [], []
    for d in scenes:
        info, probs = check_scene(d)
        rows.append((d.name, info["refs"], info["views"], info["hw"], "; ".join(probs)))
        if probs:
            print(f"  !! {d.name}: {'; '.join(probs)}")
        else:
            good.append(d.name)
    hws = Counter(r[3] for r in rows if r[3])
    print(f"[blended] {len(scenes)} scene folders, {len(good)} usable, "
          f"{sum(r[1] for r in rows if r[0] in set(good))} reference views; image sizes {dict(hws)}")
    if not good:
        raise SystemExit("no usable scene")

    out.mkdir(parents=True, exist_ok=True)
    val_path = out / "val.txt"
    if val_path.is_file() and not args.force:
        val = [s.strip() for s in val_path.read_text().splitlines() if s.strip()]
        gone = [s for s in val if s not in good]
        if gone:
            raise SystemExit(f"existing {val_path} names unusable scenes {gone}; fix them or pass --force")
        print(f"[blended] keeping existing {val_path} ({len(val)} scenes)")
    else:
        n = min(args.num_val, len(good) // 2)
        step = len(good) / n
        val = [good[int(step * i + step / 2)] for i in range(n)]
        val_path.write_text("\n".join(val) + "\n")
    train = [s for s in good if s not in set(val)]
    (out / "train.txt").write_text("\n".join(train) + "\n")
    (out / "all.txt").write_text("\n".join(good) + "\n")
    with (out / "report.tsv").open("w") as fh:
        fh.write("scene\trefs\tviews\thw\tproblems\n")
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    print(f"[blended] train {len(train)} / val {len(val)} scenes -> {out}")
    print(f"[blended] val: {' '.join(val)}")


if __name__ == "__main__":
    main()
