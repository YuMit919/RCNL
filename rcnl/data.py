"""Scene loading, open-legend class selection and the two split protocols."""
import zlib

import numpy as np
import scipy.io as sio
from scipy import ndimage


# ----------------------------------------------------------------------------- loading
def _flatten_cells(v, prefix, out):
    """Flatten MATLAB structs (loadmat(simplify_cells=True)) into {"a/b/c": numeric array}."""
    if isinstance(v, dict):
        for k, x in v.items():
            _flatten_cells(x, f"{prefix}/{k}", out)
    elif isinstance(v, np.ndarray) and v.dtype != object and v.dtype.names is None and np.issubdtype(v.dtype, np.number):
        out[prefix] = v


def load_mat(path):
    """Return {name: ndarray} for a .mat file (v5 or v7.3/HDF5)."""
    try:
        d = sio.loadmat(path)
        out = {k: v for k, v in d.items() if not k.startswith("__") and isinstance(v, np.ndarray)
               and v.dtype != object and v.dtype.names is None}
        nested = [k for k, v in d.items() if not k.startswith("__") and isinstance(v, np.ndarray)
                  and (v.dtype == object or v.dtype.names is not None)]
        if nested:
            dc = sio.loadmat(path, simplify_cells=True)
            for k in nested:
                _flatten_cells(dc[k], k, out)
        return out
    except (NotImplementedError, ValueError):
        import h5py
        out = {}
        with h5py.File(path, "r") as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                    out[name] = np.array(obj).T          # HDF5 is column-major
            f.visititems(visit)
        return out


def load_scene(data_path, gt_path, data_key=None, gt_key=None):
    """Return cube (H, W, B) float32 and ground truth (H, W) int64 (0 = unlabelled)."""
    d, g = load_mat(data_path), load_mat(gt_path)
    if data_key:
        cube = d[data_key]
    else:
        cands = [v for v in d.values() if v.ndim == 3]
        if not cands:
            raise SystemExit(f"no 3-D array in {data_path}; keys: {list(d)}")
        cube = max(cands, key=lambda a: a.size)
    if gt_key:
        gt = g[gt_key]
    else:
        cands = [(k, v) for k, v in g.items() if v.ndim == 2]
        if not cands:
            raise SystemExit(f"no 2-D array in {gt_path}; keys: {list(g)}")
        big = max(v.size for _, v in cands)
        top = [(k, v) for k, v in cands if v.size == big]
        named = [(k, v) for k, v in top if "label" in k.lower() or "gt" in k.lower()]
        gt = (named or top)[0][1]
    cube = np.asarray(cube, dtype=np.float32)
    gt = np.asarray(gt).astype(np.int64)
    gt[gt < 0] = 0
    if cube.shape[:2] != gt.shape:
        if cube.shape[1:] == gt.shape:
            cube = np.moveaxis(cube, 0, -1)
        else:
            raise SystemExit(f"shape mismatch: cube {cube.shape}, gt {gt.shape}")
    return cube, gt


def parse_scene(s):
    """'Name=data.mat,gt.mat,lo_nm,hi_nm' -> (name, data, gt, lo, hi)."""
    name, rest = s.split("=", 1)
    data, gt, lo, hi = rest.split(",")
    return name, data, gt, float(lo), float(hi)


# ----------------------------------------------------------------------------- open legend
def select_out_of_legend(gt, name, frac=0.3, seed=777):
    """About `frac` of the labelled classes (round(frac * n)) are withheld from the legend; fixed per scene name."""
    cls_all = np.unique(gt[gt > 0])
    rng = np.random.default_rng(seed + zlib.crc32(name.encode()) % 1000)
    return sorted(rng.choice(cls_all, max(1, int(round(frac * len(cls_all)))), replace=False).tolist())


# ----------------------------------------------------------------------------- protocols
def checkerboard(gt, grid=4, buffer=7.0):
    """Spatially independent split: references from one colour of a grid x grid checkerboard,
    test pixels from the other colour at least `buffer` pixels away."""
    H, W = gt.shape
    y = gt.ravel().astype(np.int64) - 1
    bh, bw = int(np.ceil(H / grid)), int(np.ceil(W / grid))
    rr, cc = np.mgrid[0:H, 0:W]
    A = ((rr // bh + cc // bw) % 2 == 0)
    dA = ndimage.distance_transform_edt(~A).ravel()
    pool = np.where((y >= 0) & A.ravel())[0]
    test = np.where((y >= 0) & (~A.ravel()) & (dA >= buffer))[0]
    return y, pool, test


def make_split(gt_legend, protocol, shots, rng, grid=4, buffer=7.0):
    """gt_legend: ground truth with out-of-legend classes set to 0.
    Returns y (legend index per pixel, -1 otherwise), reference indices, known test indices, K."""
    if protocol == "D":
        y_all, pool, test = checkerboard(gt_legend, grid, buffer)
    else:
        y_all = gt_legend.ravel().astype(np.int64) - 1
        pool = np.where(y_all >= 0)[0]; test = pool
    cls = [c for c in np.unique(y_all[pool]) if (y_all[pool] == c).sum() >= shots + (protocol == "R")
           and (y_all[test] == c).sum() > 0]
    K = len(cls); cmap = np.full(y_all.max() + 1, -1); cmap[cls] = np.arange(K)
    y = np.where(y_all >= 0, cmap[np.clip(y_all, 0, None)], -1)
    pl = pool[y[pool] >= 0]
    tr = np.concatenate([rng.choice(pl[y[pl] == k], shots, replace=False) for k in range(K)])
    te = np.setdiff1d(test[y[test] >= 0], tr)
    return y, tr, te, K


def evaluation_pixels(gt, unk, protocol, te, rng, seed, grid=4, buffer=7.0, max_known=50000, max_unknown=20000):
    """Known test pixels (subsampled) followed by out-of-legend test pixels (subsampled), as in the paper."""
    if len(te) > 200000:
        te = np.sort(rng.choice(te, 200000, replace=False))
    if len(te) > max_known:
        te = np.sort(np.random.default_rng(seed + 12345).choice(te, max_known, replace=False))
    test_o = checkerboard(gt, grid, buffer)[2] if protocol == "D" else np.where(gt.ravel() > 0)[0]
    ut = test_o[np.isin(gt.ravel()[test_o], unk)]
    if len(ut) > max_unknown:
        ut = np.random.default_rng(seed).choice(ut, max_unknown, replace=False)
    return te, ut
