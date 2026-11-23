"""CPU-only learned scorer for synthetic, within-source fibre continuation.

Coordinates are global native XYZ. Both arrays are TIP FIRST: index zero is
the endpoint facing the proposed gap. Scores are ranking scores learned from
silver labels, not calibrated probabilities of physical fibre identity.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

FEATURE_VERSION = 1
VOXEL_MICRONS = 8.64
SCALES_MICRONS = np.array([4., 12., 28.]) * VOXEL_MICRONS


def clean_curve(points):
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError("A spline needs at least two finite XYZ points")
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]]
    if len(p) < 2:
        raise ValueError("A spline needs nonzero length")
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    return p, s


def sample_curve(points, distances):
    p, s = clean_curve(points)
    return np.stack([np.interp(distances, s, p[:, k]) for k in range(3)], axis=1)


def endpoint_snippet(points, n=24, extent=48.):
    p, s = clean_curve(points)
    return sample_curve(p, np.linspace(0., min(extent, s[-1]), n)).astype(np.float32)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


_base_names = ["gap_um_div_259", "log_gap_um", "family"]
for _scale in (4, 12, 28):
    _base_names += [f"alignment_min_{_scale}", f"alignment_mean_{_scale}",
                   f"tangent_agreement_{_scale}", f"lateral_min_{_scale}", f"lateral_max_{_scale}"]
_base_names += ["support_min", "support_max", "curvature_min", "curvature_max",
                "tangent_stability_min", "tangent_stability_max"]
for _family in ("same", "cross"):
    _base_names += [f"context_{_family}_{name}" for name in
                   ("n", "near4", "near12", "near32", "dmin", "d25", "dmedian",
                    "align_mean", "align_max", "tip_a", "tip_b", "midpoint")]
FEATURE_NAMES = tuple(_base_names)
FEATURE_DIM = len(FEATURE_NAMES)


def geometry_features(a, b, context, voxel_microns=VOXEL_MICRONS, family=0):
    """Return a 48-D, translation/rotation/pair-order invariant float32 vector.

    Uses fixed physical scales. Context dictionaries contain points and family;
    callers must remove both candidate sources and overlapping aliases. At most
    64 context curves are consumed, sorted by distance to bridge midpoint.
    """
    if not np.isfinite(voxel_microns) or voxel_microns <= 0:
        raise ValueError("voxel_microns must be positive")
    pa, sa = clean_curve(a)
    pb, sb = clean_curve(b)
    delta = (pb[0] - pa[0]) * voxel_microns
    distance = float(np.linalg.norm(delta))
    if distance < 1e-5:
        raise ValueError("Coincident endpoints have no continuation direction")
    direction = delta / distance
    scales = SCALES_MICRONS / voxel_microns
    ta = _unit(pa[0] - sample_curve(pa, scales))
    tb = _unit(pb[0] - sample_curve(pb, scales))
    aa, ab = ta @ direction, tb @ -direction
    vals = [distance / (30 * VOXEL_MICRONS), np.log1p(distance) / 6., float(family)]
    for i in range(3):
        lateral = distance * np.sqrt(np.maximum(0., 1. - np.array([aa[i], ab[i]]) ** 2)) / (8 * VOXEL_MICRONS)
        vals += [min(aa[i], ab[i]), (aa[i] + ab[i]) / 2., float(ta[i] @ -tb[i]),
                 min(lateral), max(lateral)]
    support = np.minimum([sa[-1] * voxel_microns, sb[-1] * voxel_microns], 40 * VOXEL_MICRONS) / (40 * VOXEL_MICRONS)
    curvature = [1 - float(ta[0] @ ta[-1]), 1 - float(tb[0] @ tb[-1])]
    stability = [1 - float(ta[1] @ ta[-1]), 1 - float(tb[1] @ tb[-1])]
    vals += [min(support), max(support), min(curvature), max(curvature), min(stability), max(stability)]
    midpoint = (pa[0] + pb[0]) / 2
    bridge = np.linspace(pa[0], pb[0], 9)
    collected = [[], []]
    nearby = []
    for item in context or []:
        try:
            p, s = clean_curve(item["points"])
            nearby.append((float(np.min(np.linalg.norm(p - midpoint, axis=1))), p, s, int(item.get("family", family))))
        except (KeyError, ValueError, TypeError):
            continue
    for _, p, s, cf in sorted(nearby, key=lambda x: x[0])[:64]:
        ps = sample_curve(p, np.linspace(0, s[-1], min(64, max(2, int(s[-1] / 3) + 1))))
        dm = np.linalg.norm(ps[:, None, :] - bridge[None, :, :], axis=-1) * voxel_microns / VOXEL_MICRONS
        near = int(np.argmin(np.linalg.norm(ps - midpoint, axis=1)))
        tang = _unit(ps[min(near + 1, len(ps) - 1)] - ps[max(near - 1, 0)])
        collected[0 if cf == int(family) else 1].append((dm.min(), abs(float(tang @ direction)), dm[:, 0].min(), dm[:, -1].min(), dm[:, 4].min()))
    for rows in collected:
        if not rows:
            vals += [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
            continue
        r = np.asarray(rows)
        d = r[:, 0]
        endpoint_mins = sorted([r[:, 2].min(), r[:, 3].min()])
        vals += [min(len(r), 32) / 32., np.count_nonzero(d < 4) / 32.,
                 np.count_nonzero(d < 12) / 32., np.count_nonzero(d < 32) / 32.,
                 min(d.min(), 64) / 64., min(np.quantile(d, .25), 64) / 64.,
                 min(np.median(d), 64) / 64., r[:, 1].mean(), r[:, 1].max(),
                 min(endpoint_mins[0], 64) / 64., min(endpoint_mins[1], 64) / 64.,
                 min(r[:, 4].min(), 64) / 64.]
    result = np.asarray(vals, np.float32)
    if result.shape != (FEATURE_DIM,) or not np.isfinite(result).all():
        raise ValueError("Invalid geometry feature vector")
    return result


def baseline_score(features):
    """Fixed multiscale distance+tangent score, larger means better match."""
    x = np.atleast_2d(np.asarray(features))
    return -(x[:, 0] + 3 * (1 - x[:, 9]) + 2 * (1 - x[:, 10]) + x[:, 12])


def make_network(input_dim):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(input_dim, 96), torch.nn.SiLU(),
                               torch.nn.Linear(96, 48), torch.nn.SiLU(),
                               torch.nn.Linear(48, 1))


class Predictor:
    """Requires a trained checkpoint; never falls back to random weights."""
    def __init__(self, checkpoint_path):
        import torch
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Trained continuation checkpoint missing: {path}")
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.metadata = state["metadata"]
        if state.get("feature_version") != FEATURE_VERSION or state["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("Checkpoint feature schema mismatch")
        self.ct_dim = int(state.get("ct_dim", 0))
        self.feature_dim = FEATURE_DIM
        self.mean = np.asarray(state["mean"], np.float32)
        self.scale = np.asarray(state["scale"], np.float32)
        self.net = make_network(FEATURE_DIM + self.ct_dim)
        self.net.load_state_dict(state["state_dict"])
        self.net.eval()
        self.checkpoint_path = str(path.resolve())

    def score(self, features, ct_features=None):
        import torch
        x = np.atleast_2d(np.asarray(features, np.float32))
        if x.shape[1] != FEATURE_DIM or not np.isfinite(x).all():
            raise ValueError("Invalid geometry features")
        if self.ct_dim:
            if ct_features is None:
                raise ValueError("This hybrid checkpoint requires valid CT features; use geometry_model.pt when CT is unavailable")
            ct = np.atleast_2d(np.asarray(ct_features, np.float32))
            if ct.shape != (len(x), self.ct_dim) or not np.isfinite(ct).all():
                raise ValueError("Invalid CT features")
            x = np.concatenate([x, ct], axis=1)
        elif ct_features is not None:
            raise ValueError("Geometry-only checkpoint does not consume CT features")
        if not len(x):
            return np.empty(0, np.float32)
        x = np.clip((x - self.mean) / self.scale, -12, 12)
        with torch.inference_mode():
            return torch.sigmoid(self.net(torch.from_numpy(x))).numpy().reshape(-1)
