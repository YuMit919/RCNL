#!/usr/bin/env python
"""Evaluate RCNL on open-legend scenes (10 references per class, about 30 % of classes out of the legend).

  python scripts/run_benchmark.py --features features --out results/rcnl.json \
      --scenes "PaviaU=PaviaU.mat,PaviaU_gt.mat,430,860" "Salinas=Salinas_corrected.mat,Salinas_gt.mat,400,2500"

Protocols: D = spatially independent 4 x 4 checkerboard with a 7-px buffer (main), R = random.
Reports OA (known test pixels), AUROC, OSCR and, at the diagnostic 90 % known-acceptance point, silent commission
and area inflation. Results are averaged over seeds 0..n_seeds-1."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rcnl import (RCNL, area_bias, build_views, evaluation_pixels, load_scene, make_split,  # noqa: E402
                  open_set_metrics, parse_scene, select_out_of_legend)
from rcnl.data import checkerboard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--features", default="features", help="directory with fm_<scene>_<model>.npy")
    ap.add_argument("--fms", nargs="+", default=["lvd", "sat"])
    ap.add_argument("--protocols", nargs="+", default=["D", "R"])
    ap.add_argument("--shots", type=int, default=10)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--unknown_frac", type=float, default=0.3)
    ap.add_argument("--unk_seed", type=int, default=777)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--buffer", type=float, default=7.0)
    ap.add_argument("--out", default="results/rcnl.json")
    ap.add_argument("--dump", default="", help="optional directory for per-run predictions (npz)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=None, help="override training steps (quick tests only; paper: 1500)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for s in a.scenes:
        name, dpath, gpath, lo, hi = parse_scene(s)
        cube, gt = load_scene(dpath, gpath)
        H, W, _ = cube.shape
        views = build_views(cube, lo, hi, {k: np.load(os.path.join(a.features, f"fm_{name}_{k}.npy")) for k in a.fms})
        del cube
        unk = select_out_of_legend(gt, name, a.unknown_frac, a.unk_seed)
        gt_legend = gt.copy(); gt_legend[np.isin(gt, unk)] = 0
        print(f"== {name}: {H}x{W}, out-of-legend classes {unk}", flush=True)
        res[name] = {}
        for proto in a.protocols:
            rows = []
            for sd in range(a.n_seeds):
                t0 = time.time()
                rng = np.random.default_rng(1000 * a.shots + sd)
                y, tr, te_full, K = make_split(gt_legend, proto, a.shots, rng, a.grid, a.buffer)
                te, ut = evaluation_pixels(gt, unk, proto, te_full, rng, sd, a.grid, a.buffer)
                ev = np.r_[te, ut]
                model = RCNL(a.device, **({"steps": a.steps} if a.steps else {})).fit(views, tr, y[tr], K, img_w=W, seed=sd)
                pred, score = model.predict(ev)
                r = open_set_metrics(pred[:len(te)], y[te], score[:len(te)], score[len(te):])
                test_o = checkerboard(gt, a.grid, a.buffer)[2] if proto == "D" else np.where(gt.ravel() > 0)[0]
                n_unk_full = int(np.isin(gt.ravel()[test_o], unk).sum())
                r.update(area_bias(pred, score, y[te], len(te), K, len(te_full) / len(te), n_unk_full / max(1, len(ut))))
                r["threshold_t"] = model.threshold_t
                rows.append(r)
                if a.dump:
                    os.makedirs(a.dump, exist_ok=True)
                    np.savez_compressed(os.path.join(a.dump, f"{name}_{proto}{a.shots}_s{sd}.npz"), ev=ev, n_te=len(te),
                                        pred=pred, score=score.astype(np.float32), yt=y[te], tr=tr)
                print(f"   [{proto}, seed {sd}] OA={r['OA']:.4f} AUROC={r['AUROC']:.4f} OSCR={r['OSCR']:.4f} "
                      f"area inflation={r['area_inflation']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            res[name][proto] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        json.dump(res, open(a.out, "w"), indent=2)
    for proto in a.protocols:
        sc = [s for s in res if proto in res[s]]
        if sc:
            m = {k: np.mean([res[s][proto][k] for s in sc]) for k in ("OA", "AUROC", "OSCR", "area_inflation")}
            print(f"[{proto}] {len(sc)} scenes: " + "  ".join(f"{k} {100 * v:.1f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
