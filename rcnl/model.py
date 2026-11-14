"""Network: one linear encoder per view, concatenated; a known-class head g and K membership heads f_k."""
import torch
import torch.nn as nn


class ViewEncoder(nn.Module):
    def __init__(self, d, h, K, p):
        super().__init__()
        self.f = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, h), nn.Dropout(p))
        self.head = nn.Linear(h, K)                       # auxiliary per-view classifier

    def forward(self, x):
        z = self.f(x)
        return z, self.head(z)


class _Backbone(nn.Module):
    def __init__(self, dims, K, h, p):
        super().__init__()
        self.names = list(dims)
        self.enc = nn.ModuleDict({k.replace(".", "_"): ViewEncoder(d, h, K, p) for k, d in dims.items()})
        self.fuse_in = h * len(dims)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(self.fuse_in, K))   # known-class head g

    def forward(self, xs, view_drop=0.0):
        zs, logits = [], {}
        for k in self.names:
            z, l = self.enc[k.replace(".", "_")](xs[k]); zs.append(z); logits[k] = l
        Z = torch.stack(zs, 1)
        if self.training and view_drop > 0 and len(self.names) > 1:          # drop whole views at random
            keep = (torch.rand(Z.shape[0], Z.shape[1], device=Z.device) > view_drop).float()
            keep[keep.sum(1) == 0, 0] = 1.0
            Z = Z * keep[:, :, None]
        fused = Z.flatten(1)
        return self.head(fused), logits, fused


class RCNLNet(nn.Module):
    """forward -> (known-class logits, per-view logits, membership logits f (n, K))."""

    def __init__(self, dims, K, h=128, p=0.3):
        super().__init__()
        self.m = _Backbone(dims, K, h, p)
        self.f = nn.Linear(self.m.fuse_in, K)

    def forward(self, xs, view_drop=0.0):
        lf, lv, fused = self.m(xs, view_drop)
        return lf, lv, self.f(fused)
