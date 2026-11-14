"""OA, AUROC, OSCR and area bias at a known-acceptance operating point."""
import numpy as np
from sklearn.metrics import roc_auc_score

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def oscr(s_known, correct, s_unk):
    """Area under the open-set classification rate curve (Dhamija et al., 2018): CCR vs. FPR as the
    membership threshold sweeps. s_*: membership scores (higher = more likely in the legend)."""
    s = np.r_[s_known, s_unk]; isk = np.r_[np.ones(len(s_known)), np.zeros(len(s_unk))]
    cor = np.r_[correct, np.zeros(len(s_unk))]
    o = np.argsort(-s, kind="stable"); ss = s[o]
    ccr = np.cumsum(cor[o]) / max(1, len(s_known)); fpr = np.cumsum(1 - isk[o]) / max(1, len(s_unk))
    last = np.r_[ss[1:] != ss[:-1], True]                   # tied scores are accepted together (linear interpolation within ties)
    return float(_trapz(np.r_[0, ccr[last]], np.r_[0, fpr[last]]))


def open_set_metrics(pred_known, y_known, s_known, s_unk):
    return {"OA": float((pred_known == y_known).mean()),
            "AUROC": float(roc_auc_score(np.r_[np.zeros(len(s_known)), np.ones(len(s_unk))], -np.r_[s_known, s_unk])),
            "OSCR": oscr(s_known, pred_known == y_known, s_unk)}


def area_bias(pred, score, y_known, n_known, K, w_known, w_unk, accept=0.90):
    """Silent commission and area inflation of the legend classes.

    pred, score : predictions / memberships on [known test pixels, out-of-legend test pixels]
    w_known, w_unk : weights mapping the two subsamples back to the true pixel counts of the test region
    accept : the threshold accepts this fraction of known test pixels (a diagnostic operating point that uses
             test labels; it is for comparing methods, not a deployable threshold). None = no rejection.
    """
    thr = None if accept is None else np.quantile(score[:n_known], 1 - accept)
    acc = np.ones(len(pred), bool) if thr is None else score >= thr
    kn = np.arange(len(pred)) < n_known; un = ~kn
    A = np.array([w_known * (y_known == k).sum() for k in range(K)])
    Ahat = np.array([w_known * (acc & kn & (pred == k)).sum() + w_unk * (acc & un & (pred == k)).sum() for k in range(K)])
    U = np.array([w_unk * (acc & un & (pred == k)).sum() for k in range(K)])
    return {"silent_commission": float(U.sum() / max(Ahat.sum(), 1e-9)),
            "area_inflation": float(U.sum() / max(A.sum(), 1e-9))}
