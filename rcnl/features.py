"""Frozen representation: a spectral view and true-colour DINOv3 views (native RGB interface).

spectral view  : PCA-30 of band-standardised spectra + window means (3, 5, 7, 15, 31 px)  -> 180-d
DINOv3 view(s) : true-colour composite (~640 / 550 / 470 nm, 2-98 % stretch), upsampled x4,
                 last-layer patch tokens (overlapping 896-px tiles), L2-normalised, PCA-64,
                 bilinearly interpolated back to pixels                                      -> 64-d each
Every view is standardised over the scene."""
import json
import os

import numpy as np
import torch
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

TRUE_COLOUR_NM = (640, 550, 470)


def _window_means(A, H, W, sizes):
    Ai = A.reshape(H, W, -1)
    return [ndimage.uniform_filter(Ai, size=(k, k, 1), mode="reflect").reshape(H * W, -1) for k in sizes]


def spectral_pca_and_rgb(cube, lo, hi, d=30, seed=0):
    """Return PCA scores A (N, d) and the stretched true-colour image (H, W, 3)."""
    H, W, B = cube.shape
    X = np.nan_to_num(cube.reshape(-1, B).astype(np.float32))
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    rng = np.random.default_rng(seed)
    sub = Xs[rng.choice(len(Xs), min(50000, len(Xs)), replace=False)]
    _, _, Vt = np.linalg.svd(sub - sub.mean(0), full_matrices=False)
    A = ((Xs - sub.mean(0)) @ Vt[:d].T).astype(np.float32)
    wl = np.linspace(lo, hi, B)
    idx = [int(np.argmin(np.abs(wl - t))) for t in TRUE_COLOUR_NM]
    rgb = X[:, idx].copy()
    for c in range(3):
        p2, p98 = np.percentile(rgb[:, c], [2, 98])
        rgb[:, c] = np.clip((rgb[:, c] - p2) / (p98 - p2 + 1e-12), 0, 1)
    return A, rgb.reshape(H, W, 3).astype(np.float32)


def spectral_view(cube, lo, hi, d=30, seed=0):
    H, W, _ = cube.shape
    A, _ = spectral_pca_and_rgb(cube, lo, hi, d, seed)
    S = np.concatenate([A] + _window_means(A, H, W, (3, 5, 7)), 1).astype(np.float32)
    return np.concatenate([S] + _window_means(S[:, :d], H, W, (15, 31)), 1)


# ----------------------------------------------------------------------------- DINOv3
def load_dinov3(path, dev):
    """Load a DINOv3 checkpoint in Hugging Face format (e.g. dinov3-vitl16-pretrain-lvd1689m / -sat493m)."""
    from transformers import AutoModel
    m = AutoModel.from_pretrained(path).to(dev).eval()
    cfg = json.load(open(os.path.join(path, "preprocessor_config.json")))
    mean = np.array(cfg.get("image_mean", [0.485, 0.456, 0.406]), np.float32)
    std = np.array(cfg.get("image_std", [0.229, 0.224, 0.225]), np.float32)
    return m, mean, std


def _origins(L, T, s):
    if L <= T:
        return [0]
    o = list(range(0, L - T + 1, s))
    if o[-1] != L - T:
        o.append(L - T)
    return o


@torch.no_grad()
def dinov3_view(rgb, model, mean, std, dev, scale=4, tile=896, stride=672, out_dim=64, seed=0):
    """Dense per-pixel DINOv3 features of a true-colour image (H, W, 3) in [0, 1]."""
    H, W, _ = rgb.shape
    p = model.config.patch_size
    n_pre = 1 + int(getattr(model.config, "num_register_tokens", 0))
    img = torch.tensor(((rgb - mean) / std).transpose(2, 0, 1)[None], dtype=torch.float32)
    Hs, Ws = int(round(H * scale / p)) * p, int(round(W * scale / p)) * p
    img = torch.nn.functional.interpolate(img, size=(Hs, Ws), mode="bicubic", align_corners=False)
    th, tw = min(tile, Hs), min(tile, Ws)
    D = model.config.hidden_size
    acc = torch.zeros(D, Hs // p, Ws // p); cnt = torch.zeros(1, Hs // p, Ws // p)
    for y0 in _origins(Hs, th, stride):
        for x0 in _origins(Ws, tw, stride):
            t = img[:, :, y0:y0 + th, x0:x0 + tw].to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                h = model(pixel_values=t).last_hidden_state[:, n_pre:]
            g = h.float()[0].T.reshape(D, th // p, tw // p).cpu()
            acc[:, y0 // p:(y0 + th) // p, x0 // p:(x0 + tw) // p] += g
            cnt[:, y0 // p:(y0 + th) // p, x0 // p:(x0 + tw) // p] += 1
    G = (acc / cnt.clamp_min(1))[:, :Hs // p, :Ws // p]
    Gf = G.reshape(D, -1).T.numpy()
    Gf /= np.linalg.norm(Gf, axis=1, keepdims=True) + 1e-8
    sub = np.random.default_rng(seed).choice(len(Gf), min(50000, len(Gf)), replace=False)
    pca = PCA(out_dim, random_state=seed).fit(Gf[sub])
    Gp = torch.tensor(pca.transform(Gf).T.reshape(out_dim, Hs // p, Ws // p)[None], dtype=torch.float32)
    F = torch.nn.functional.interpolate(Gp, size=(H, W), mode="bilinear", align_corners=False)[0]
    return F.reshape(out_dim, -1).T.numpy().astype(np.float32)


def build_views(cube, lo, hi, dino_feats, seed=0):
    """dino_feats: {"lvd": (N, 64), "sat": (N, 64)} (cached by scripts/extract_dinov3.py).
    Returns the ordered dict of standardised views {"s", "lvd", "sat"}."""
    views = {"s": spectral_view(cube, lo, hi, 30, seed)}
    for k, F in dino_feats.items():
        views[k] = F
    return {k: StandardScaler().fit_transform(v).astype(np.float32) for k, v in views.items()}
