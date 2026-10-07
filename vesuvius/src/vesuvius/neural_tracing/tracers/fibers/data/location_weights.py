"""Fresh-location weights that sample heavily bent fibers and compressed sheets more often (but not predominantly).

Per source, a cache (built by scripts/build_location_weights.py) holds for every annotated fiber, every ``STRIDE``
vox of arclength:
  bend         the largest of: the fiber's bend over +-12 vox and its bend over the next 32 vox in either direction
               (the annotation smoothed at sigma 4 vox, evaluation.audit.failure_scoring.smoothed), in degrees
  compression  sheetness and sheet modulation (evaluation.audit.failure_scoring.compression) at one fiber point per occupied
               ``CELL``-vox grid cell (compression varies slowly; this keeps CT reads to a few thousand per source)
Flags (per source): bend at or above its p90; compression score max(1 - pct(sheetness), 1 - pct(modulation)) at or
above its p90, so each condition marks 10% of locations, as in evaluation.audit.failure_scoring. A location's weight is
1 + bend_boost * bent + compression_boost * compressed; fresh windows and positions within them are drawn in proportion
(FollowDataset.fresh_location). With boosts of 1 each condition rises from ~10% to ~17-18% of fresh crops.
"""
import hashlib
from pathlib import Path

import numpy as np

STRIDE = 8.
CELL = 32.
VERSION = 1
DEFAULT_DIR = Path(__file__).resolve().parents[1]/'output'/'location_weights'


def cache_path(source, directory=None):
    """The cache file of a dataset source: its name and the identity of its annotations."""
    if source['kind'] == 'paris4':
        identity = hashlib.sha256(str(Path(source['fibers']).resolve()).encode()).hexdigest()[:12]
    else:
        identity = str(source['sha256'])[:12]
    return Path(directory or DEFAULT_DIR)/f"{source['name']}_{identity}_v{VERSION}.npz"


def load(path, fibers, bend_boost=1., compression_boost=1.):
    """Per fiber (aligned with ``fibers`` by name): (arclength positions, location weights), or None for a fiber the
    cache does not hold (uniform)."""
    with np.load(path) as data:  # read each array once: an npz decompresses on every access
        names, starts, arcs = data['names'], data['starts'], data['arcs'].astype(np.float64)
        weight = 1.+bend_boost*data['bent']+compression_boost*data['compressed']
    index = {str(n): i for i, n in enumerate(names)}
    wanted = [row[1] for row in fibers.catalog] if hasattr(fibers, 'catalog') else [f.name for f in fibers]
    out = []
    for name in wanted:
        i = index.get(str(name))
        out.append(None if i is None else (arcs[starts[i]:starts[i+1]], weight[starts[i]:starts[i+1]]))
    return out
