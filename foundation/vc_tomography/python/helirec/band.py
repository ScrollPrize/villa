"""Surface composites straight from the projections: the composite's sample points reconstructed where they lie
(the exact backprojector at the pixels they need, sampled trilinearly), without forming slices or a volume:
helirec::surface::BandComposite (helirec/surface_composite.hpp), which this module binds; from a shell,
helirec-composite --projections.

    from helirec.recon import HelicalRecon
    from helirec.band import BandComposite
    r = HelicalRecon(projections, cache_dir=..., device="cuda")
    bc = BandComposite(r, pts, valid)           # pts (L, H, W, 3) XYZ in volume voxels
    bc.run(chunk=64)                            # single parameter set (the reconstructor's)
    mx, mean = bc.composites()

    bc = BandComposite(r, pts, valid, n_sets=3)  # sweep: filter chain per set, everything before it computed once
    bc.run(chunk=40, sweep=[{"paganin.delta_beta": v} for v in (250, 500, 1000)])
    mx, mean = bc.composites(set=1)

Points are XYZ voxel coordinates of the volume: x, y = slice column, row; z = volume slice index
(view height h = z + r.lo). A pixel is kept when all its layers can be sampled inside [h_range) and the slice.
"""
from __future__ import annotations

import json

import numpy as np

from . import _core


class BandComposite:
    def __init__(self, recon, pts, valid, h_range=None, n_sets=1):
        """recon: a helirec.recon.HelicalRecon; pts: (L, H, W, 3) float XYZ; valid: (H, W) bool; h_range: [h0, h1) view
        heights to reconstruct (default: the span the points need); n_sets: value sets (parameter sweeps)."""
        self.r = recon
        self._b = _core.BandComposite(recon._r, np.ascontiguousarray(pts, np.float32), np.ascontiguousarray(valid, np.uint8),
                                      None if h_range is None else (int(h_range[0]), int(h_range[1])), int(n_sets))
        self.band, self.n_points, self.valid, self.h_range, self.box = self._b.band, self._b.n_points, self._b.valid, self._b.h_range, self._b.box

    def run(self, chunk=64, timing=False, progress=None, sweep=None, prefetch=True):
        """Reconstruct h_range in chunks; returns per-stage seconds (+ 'wall'). sweep: one entry per value set, a dict
        of filter overrides ({"paganin.delta_beta": 500}, applied to the reconstructor's chain) or a whole chain."""
        chains = [json.dumps(c if isinstance(c, list) else self.r.chains([c])[0]) for c in (sweep or [])]
        return self._b.run(int(chunk), bool(timing), chains, bool(prefetch), progress)

    def layers(self, set=0):
        """(L, H, W) float32 samples of value set `set`, NaN where a pixel is not valid."""
        return self._b.layers(int(set))

    def composites(self, set=0):
        """(max, mean) float32 (H, W) of value set `set`, NaN where a pixel is not valid."""
        return self._b.composites(int(set))
