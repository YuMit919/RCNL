"""Reference-certified negative learning (RCNL).

1. Reference support threshold (spatial cross-fitting)
   The references of each class are clustered by image coordinates (k-means, 5 clusters) and the clusters are
   assigned to 5 folds. A reference model A (same network, cross-entropy only) is trained on 4 folds and predicts the
   held-out fold; the true-class probabilities of all held-out references are pooled and their alpha-quantile
   (alpha = 0.1) is the threshold t, shared by all classes.
2. Certification
   A trained on all references gives P_A(k | x) for every unlabeled pixel x; C(x) = {k : P_A(k | x) >= t}.
   The negative loss of x for class k is masked (w_k(x) = 0) only if C(x) = {k}; otherwise w_k(x) = 1.
3. Membership learning
   R = 1/2 sum_k [ mean_{refs of k} softplus(-f_k) + mean_{U} w_k softplus(f_k) ]
   trained jointly with the known-class head (cross-entropy on references, auxiliary per-view heads x 0.5, L2).
Inference: class = argmax of the known-class head; legend membership s(x) = max_k sigmoid(f_k(x)).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from .model import RCNLNet

CFG = dict(steps=1500, lr=1e-3, wd=1e-2, hidden=128, dropout=0.3, view_drop=0.2, C=1.0,
           n_unlabeled=4000, alpha=0.1, folds=5)


def membership_risk(f_ref, y_ref, f_unl, w, K):
    """Logistic PU risk with certification weights."""
    oh = Fnn.one_hot(y_ref, K).float()
    n = oh.sum(0).clamp(min=1)
    pos = ((-Fnn.logsigmoid(f_ref)) * oh).sum(0) / n
    neg = (w * Fnn.softplus(f_unl)).mean(0)
    return 0.5 * (pos + neg).sum()


def train_network(X, tr, ytr, K, seed, dev, unl=None, W=None, cfg=CFG):
    """X: {view: tensor (N, d)} on dev. Without unl / W this trains the reference model A (cross-entropy only)."""
    torch.manual_seed(seed)
    dims = {k: v.shape[1] for k, v in X.items()}
    net = RCNLNet(dims, K, cfg["hidden"], cfg["dropout"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    ttr, yl = torch.tensor(tr, device=dev), torch.tensor(ytr, device=dev)
    tun = torch.tensor(unl, device=dev) if unl is not None and len(unl) else None
    l2 = 0.5 / (cfg["C"] * len(tr))
    for _ in range(cfg["steps"]):
        net.train()
        lf, lv, fl = net({k: v[ttr] for k, v in X.items()}, cfg["view_drop"])
        loss = Fnn.cross_entropy(lf, yl) + 0.5 * sum(Fnn.cross_entropy(l, yl) for l in lv.values()) \
            + l2 * sum((m_.weight ** 2).sum() for m_ in net.modules() if isinstance(m_, nn.Linear))
        if tun is not None and W is not None:
            _, _, fu = net({k: v[tun] for k, v in X.items()}, cfg["view_drop"])
            loss = loss + membership_risk(fl, yl, fu, W, K)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def predict(net, X, idx, dev, batch=65536):
    """Return known-class probabilities (n, K) and membership probabilities sigmoid(f) (n, K)."""
    P, S = [], []
    for i in range(0, len(idx), batch):
        b = torch.tensor(idx[i:i + batch], device=dev)
        lf, _, f = net({k: v[b] for k, v in X.items()})
        P.append(torch.softmax(lf, 1).cpu().numpy()); S.append(torch.sigmoid(f).cpu().numpy())
    return np.concatenate(P), np.concatenate(S)


def spatial_folds(tr, ytr, K, img_w, seed, n_folds=5):
    from sklearn.cluster import KMeans
    g = np.random.default_rng(seed + 77)
    fold = np.empty(len(tr), int)
    for k in range(K):
        ix = np.where(ytr == k)[0]
        nf = min(n_folds, len(ix))
        xy = np.stack(np.divmod(tr[ix], img_w), 1).astype(float)
        lab = KMeans(nf, n_init=5, random_state=seed).fit_predict(xy)
        fold[ix] = g.permutation(n_folds)[:nf][lab]
    return fold


def certify(X, tr, ytr, unl, K, img_w, seed, dev, cfg=CFG):
    """Return certification weights W (n_unl, K), the shared threshold t and the reference model A."""
    fold = spatial_folds(tr, ytr, K, img_w, seed, cfg["folds"])
    oof = np.zeros(len(tr))
    for f in range(cfg["folds"]):
        a, b = fold != f, fold == f
        if b.sum() == 0:
            continue
        net = train_network(X, tr[a], ytr[a], K, seed + 100 + f, dev, cfg=cfg)
        P, _ = predict(net, X, tr[b], dev)
        oof[b] = P[np.arange(b.sum()), ytr[b]]
    t = float(np.quantile(oof, cfg["alpha"]))
    ref = train_network(X, tr, ytr, K, seed, dev, cfg=cfg)
    PA, _ = predict(ref, X, unl, dev)
    support = PA >= t
    unique = support & (support.sum(1, keepdims=True) == 1)
    W = (~unique).astype(np.float32)
    return W, t, ref


def sample_unlabeled(N, tr, seed, n):
    """Unlabeled pool: n random non-reference pixels of the whole scene (their labels are never used)."""
    g = np.random.default_rng(seed + 4242)
    pool = np.setdiff1d(np.arange(N), tr)
    return np.sort(g.choice(pool, min(n, len(pool)), replace=False))


class RCNL:
    """Fit on one scene; predict any pixels.

        model = RCNL(device="cuda").fit(views, tr, ytr, K, img_w=W, seed=0)
        cls, s = model.predict(idx)          # class index and legend membership in [0, 1]
    """

    def __init__(self, device=None, **cfg):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg = {**CFG, **cfg}

    def fit(self, views, tr, ytr, K, img_w, seed=0):
        self.X = {k: torch.tensor(v, device=self.dev) for k, v in views.items()}
        N = len(next(iter(views.values())))
        self.unl = sample_unlabeled(N, tr, seed, self.cfg["n_unlabeled"])
        self.W, self.threshold_t, _ = certify(self.X, tr, ytr, self.unl, K, img_w, seed, self.dev, self.cfg)
        self.net = train_network(self.X, tr, ytr, K, seed, self.dev, self.unl,
                                 torch.tensor(self.W, device=self.dev), self.cfg)
        return self

    def predict(self, idx):
        P, S = predict(self.net, self.X, np.asarray(idx), self.dev)
        return P.argmax(1), S.max(1)
