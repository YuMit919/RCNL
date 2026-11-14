"""RCNL: reference-certified negative learning for few-shot open-legend hyperspectral mapping."""
from .core import CFG, RCNL, certify, membership_risk, predict, train_network
from .data import evaluation_pixels, load_scene, make_split, parse_scene, select_out_of_legend
from .features import build_views, dinov3_view, load_dinov3, spectral_pca_and_rgb, spectral_view
from .metrics import area_bias, open_set_metrics, oscr

__version__ = "1.0.0"
