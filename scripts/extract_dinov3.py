#!/usr/bin/env python
"""Extract and cache the true-colour DINOv3 views of each scene.

  python scripts/extract_dinov3.py \
      --fm lvd=/path/dinov3-vitl16-pretrain-lvd1689m sat=/path/dinov3-vitl16-pretrain-sat493m \
      --scenes "PaviaU=PaviaU.mat,PaviaU_gt.mat,430,860" ... --out features

Writes features/fm_<scene>_<model>.npy (N x 64, float32); existing files are skipped."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rcnl import dinov3_view, load_dinov3, load_scene, parse_scene, spectral_pca_and_rgb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True, help="Name=data.mat,gt.mat,first_band_nm,last_band_nm")
    ap.add_argument("--fm", nargs="+", required=True, help="name=path to a Hugging Face DINOv3 checkpoint")
    ap.add_argument("--out", default="features")
    ap.add_argument("--scale", type=float, default=4.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fms = {}
    for s in a.scenes:
        name, dpath, gpath, lo, hi = parse_scene(s)
        todo = [n for n, _ in (x.split("=", 1) for x in a.fm) if not os.path.exists(os.path.join(a.out, f"fm_{name}_{n}.npy"))]
        if not todo:
            print(f"[{name}] cached, skipped", flush=True); continue
        cube, _ = load_scene(dpath, gpath)
        _, rgb = spectral_pca_and_rgb(cube, lo, hi, 30, 0)
        for n, p in (x.split("=", 1) for x in a.fm):
            if n not in todo:
                continue
            if n not in fms:
                fms[n] = load_dinov3(p, dev)
            model, mean, std = fms[n]
            np.save(os.path.join(a.out, f"fm_{name}_{n}.npy"), dinov3_view(rgb, model, mean, std, dev, scale=a.scale))
            print(f"[{name}] {n} done", flush=True)


if __name__ == "__main__":
    main()
