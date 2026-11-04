"""Fresh-location weights that sample heavily bent fibers and compressed sheets more often (but not predominantly).

Per source, a cache (built by scripts/build_location_weights.py) holds for every annotated fiber, every ``STRIDE``
vox of arclength:
  bend         the largest of: the fiber's bend over +-12 vox and its bend over the next 32 vox in either direction
               (the annotation smoothed at sigma 4 vox, ``smoothed``), in degrees
  compression  sheetness and sheet modulation (``compression``) at one fiber point per occupied ``CELL``-vox grid cell
               (compression varies slowly; this keeps CT reads to a few thousand per source)
Flags (per source): bend at or above its p90; compression score max(1 - pct(sheetness), 1 - pct(modulation)) at or
above its p90, so each condition marks 10% of locations. A location's weight is
1 + bend_boost * bent + compression_boost * compressed; fresh windows and positions within them are drawn in proportion
(FollowDataset.fresh_location). With boosts of 1 each condition rises from ~10% to ~17-18% of fresh crops.
"""
import hashlib
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fibers_digest
from vesuvius.neural_tracing.tracers.shared.geometry import interp_at

STRIDE = 8.
CELL = 32.
VERSION = 1
DEFAULT_DIR = Path(__file__).resolve().parents[1]/'output'/'location_weights'


def cache_path(source, directory=None):
    """The cache file of a dataset source: its name and the identity of its annotations (a fiber store's content
    digest)."""
    if source['kind'] == 'paris4':
        identity = hashlib.sha256(str(Path(source['fibers']).resolve()).encode()).hexdigest()[:12]
    else:
        identity = fibers_digest(source['path'])[:12]
    return Path(directory or DEFAULT_DIR)/f"{source['name']}_{identity}_v{VERSION}.npz"


def load(path, fibers, bend_boost=1., compression_boost=1.):
    """Per fiber (aligned with ``fibers`` by name, or by UUID for a fiber store): (arclength positions, location
    weights), or None for a fiber the cache does not hold (uniform)."""
    with np.load(path) as data:  # read each array once: an npz decompresses on every access
        names, starts, arcs = data['names'], data['starts'], data['arcs'].astype(np.float64)
        weight = 1.+bend_boost*data['bent']+compression_boost*data['compressed']
    index = {str(n): i for i, n in enumerate(names)}
    wanted = [row['id'] for row in fibers.catalog] if hasattr(fibers, 'catalog') else [f.name for f in fibers]
    out = []
    for name in wanted:
        i = index.get(str(name))
        out.append(None if i is None else (arcs[starts[i]:starts[i+1]], weight[starts[i]:starts[i+1]]))
    return out


# ------------------------------------------------- measures the caches are built from (scripts/build_location_weights.py)
def smoothed(annotation, s, sigma=4.):
    """The annotation resampled every 1 vox of arclength ``s`` and smoothed (Gaussian, ``sigma`` vox), and those arcs."""
    arcs = np.arange(s[0], s[-1]+1e-9, 1.)
    return gaussian_filter1d(interp_at(annotation, s, arcs), sigma, axis=0, mode='nearest'), arcs


def sample_ct(ct, spec, pts):
    """CT values (raw, trilinear) at trace-coordinate points of any shape (..., 3); the fill value outside the volume."""
    pts = np.asarray(pts, float)
    return ct.sample(pts[..., ::-1]*spec.input_scale, order=1, dtype=np.float64).astype(np.float32)


def sheet_normal(ct, spec, centre, half=6.):
    """Largest eigenvector of the CT gradient structure tensor over a (2*half+1)^3 vox block, and sheetness."""
    g = np.arange(-half, half+.01, 1.)
    block = sample_ct(ct, spec, centre+np.stack(np.meshgrid(g, g, g, indexing='ij'), -1))
    grads = np.stack(np.gradient(block), -1).reshape(-1, 3)
    w, v = np.linalg.eigh(grads.T @ grads)
    return v[:, -1], float((w[-1]-w[-2])/max(w[-1], 1e-9))


def sheet_modulation(ct, spec, centre, normal, half=20.):
    """(p90 - p10)/mean of the CT profile along the sheet normal (3x3 vox patch mean): how clearly sheets separate."""
    e1 = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    offsets = np.arange(-half, half+.01, .5)
    profile = np.stack([sample_ct(ct, spec, centre+u*e1+v*e2+offsets[:, None]*normal)
                        for u in (-1., 0., 1.) for v in (-1., 0., 1.)]).mean(0)
    return float(np.percentile(profile, 90)-np.percentile(profile, 10))/max(float(profile.mean()), 1e-6)


def compression(ct, spec, a):
    """Sheetness and sheet modulation of the CT at point ``a``."""
    normal, sheetness = sheet_normal(ct, spec, a)
    return dict(sheetness=sheetness, sheet_modulation=sheet_modulation(ct, spec, a, normal))


_VOLUMES, _DOCUMENT = {}, None


def init_volumes(config):
    """Worker initializer: the dataset configuration whose sources ``volume`` opens."""
    global _DOCUMENT
    from vesuvius.neural_tracing.tracers.fibers.data.datasets import read_dataset_config
    _DOCUMENT = read_dataset_config(config)[0]


def volume(name):
    """The CT reader and volume spec of source ``name`` (opened once per process)."""
    if name not in _VOLUMES:
        from vc_reader.reader import open_zarr
        from vesuvius.neural_tracing.tracers.fibers.data.datasets import ct_source_spec, primary_source_spec
        source = next(s for s in _DOCUMENT['sources'] if s['name'] == name)
        spec = primary_source_spec(_DOCUMENT) if source['kind'] == 'paris4' else ct_source_spec(source, _DOCUMENT['cache_dir'])
        _VOLUMES[name] = open_zarr(spec.ct_zarr, level=spec.ct_level, cache_dir=spec.cache_dir, memory_cache=3 << 30), spec
    return _VOLUMES[name]

