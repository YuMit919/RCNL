#!/usr/bin/env python
"""Wall-to-wall map of one scene with RCNL.

References come either from a CSV (row,col,class; class = legend index starting at 0) or, for the benchmark
setting, from the spatially independent split (--protocol D) with about 30 % of classes out of the legend.

  python scripts/map_scene.py --scene "Salinas=Salinas_corrected.mat,Salinas_gt.mat,400,2500" --features features
  python scripts/map_scene.py --scene ... --refs my_references.csv --threshold 0.5

Outputs <out>/<scene>_map.npz with the class map, the legend-membership map s(x) = max_k sigmoid(f_k(x)) and the
references. With --threshold tau, pixels with s(x) < tau are set to -1 (unclassified). The paper reports
threshold-free metrics (OSCR, AUROC); choosing tau for deployment is left to the user."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rcnl import RCNL, build_views, load_scene, make_split, parse_scene, select_out_of_legend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--features", default="features")
    ap.add_argument("--fms", nargs="+", default=["lvd", "sat"])
    ap.add_argument("--refs", default="", help="CSV with row,col,class (header optional)")
    ap.add_argument("--protocol", default="D")
    ap.add_argument("--shots", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=None, help="membership below this value -> unclassified (-1)")
    ap.add_argument("--out", default="maps")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=None, help="override training steps (quick tests only; paper: 1500)")
    a = ap.parse_args()
    name, dpath, gpath, lo, hi = parse_scene(a.scene)
    cube, gt = load_scene(dpath, gpath)
    H, W, _ = cube.shape
    views = build_views(cube, lo, hi, {k: np.load(os.path.join(a.features, f"fm_{name}_{k}.npy")) for k in a.fms})
    if a.refs:
        R = np.loadtxt(a.refs, delimiter=",", ndmin=2, skiprows=0 if open(a.refs).readline()[0].isdigit() else 1).astype(int)
        tr, ytr = R[:, 0] * W + R[:, 1], R[:, 2]
        K = int(ytr.max()) + 1
    else:
        unk = select_out_of_legend(gt, name)
        gt_legend = gt.copy(); gt_legend[np.isin(gt, unk)] = 0
        y, tr, _, K = make_split(gt_legend, a.protocol, a.shots, np.random.default_rng(1000 * a.shots + a.seed))
        ytr = y[tr]
    model = RCNL(a.device, **({"steps": a.steps} if a.steps else {})).fit(views, tr, ytr, K, img_w=W, seed=a.seed)
    pred, score = model.predict(np.arange(H * W))
    cls = pred.copy()
    if a.threshold is not None:
        cls[score < a.threshold] = -1
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, f"{name}_map.npz"), cls=cls.reshape(H, W).astype(np.int16),
                        membership=score.reshape(H, W).astype(np.float32), refs=tr, ref_classes=ytr)
    msg = f"; unclassified share {(cls == -1).mean():.3f}" if a.threshold is not None else ""
    print(f"saved {os.path.join(a.out, name + '_map.npz')}{msg}")


if __name__ == "__main__":
    main()
