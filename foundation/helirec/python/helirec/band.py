"""Surface composites straight from the projections: band reconstruction (exact backprojector at the pixels the
composite's sample points need) + trilinear sampling + max / mean, without forming slices or a volume.

    from helirec.recon import HelicalRecon
    from helirec.band import BandComposite
    r = HelicalRecon(projections, cache_dir=..., device="cuda")
    bc = BandComposite(r, pts, valid)           # pts (L, H, W, 3) XYZ in volume voxels
    bc.run(chunk=64)                            # single parameter set (the reconstructor's)
    mx, mean = bc.composites()

    bc = BandComposite(r, pts, valid, n_sets=3)  # sweep: filter chain per set, everything before it computed once
    bc.run(chunk=40, sweep=[{"filters": f"paganin(delta_beta={v})"} for v in (250, 500, 1000)])
    mx, mean = bc.composites(set=1)

Points are XYZ voxel coordinates of the volume: x, y = slice column, row; z = volume slice index
(view height h = z + r.lo). A pixel is kept when all its layers can be sampled inside [h_range) and the slice.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from . import _core


class BandComposite:
    def __init__(self, recon, pts, valid, h_range=None, n_sets=1):
        """pts: (L, H, W, 3) float XYZ; valid: (H, W) bool; h_range: [h0, h1) view heights to reconstruct
        (default: the span the points need); n_sets: value sets (parameter sweeps)."""
        self.r = recon
        L, H, W, _ = pts.shape
        N = recon.info["grid_size"]
        h = pts[..., 2].astype(np.float64) + recon.lo
        y, x = pts[..., 1], pts[..., 0]
        ok = valid[None] & np.isfinite(pts).all(-1) & (x >= 0) & (y >= 0) & (x <= N - 1) & (y <= N - 1)
        if h_range is None:
            hv = h[np.broadcast_to(ok, h.shape)]
            h_range = (int(np.floor(hv.min())), int(np.floor(hv.max())) + 2)
        h0, h1 = int(h_range[0]), int(h_range[1])
        h0, h1 = max(h0, recon.lo), min(h1, recon.hi)
        ok &= (h >= h0) & (h <= h1 - 1)
        self.valid = ok.all(0)                                   # every layer samplable
        self.L, self.H, self.W, self.h_range = L, H, W, (h0, h1)
        sel = np.nonzero(self.valid.ravel())[0]
        hh = h.reshape(L, -1)[:, sel].ravel()
        zlo = np.minimum(np.floor(hh), h1 - 2).astype(np.int32)
        order = np.argsort(zlo, kind="stable")
        self.order = order                                       # sampler index -> (layer, valid pixel) index
        self.sel = sel
        self.n_sets = n_sets
        p_zlo = np.ascontiguousarray(zlo[order])
        p_fz = np.ascontiguousarray((hh - zlo)[order], np.float32)
        p_y = np.ascontiguousarray(y.reshape(L, -1)[:, sel].ravel()[order], np.float32)
        p_x = np.ascontiguousarray(x.reshape(L, -1)[:, sel].ravel()[order], np.float32)
        from .recon import backprojector_config
        self.band = backprojector_config(recon.params["backprojector"])["backprojector"] == "fbp"
        if self.band:   # exact backprojector at the band pixels, sampled on the reconstructor's device
            self.sampler = _core.BandSampler(p_zlo, p_fz, p_y, p_x, N, device=recon.device, n_sets=n_sets)
        else:           # whole-slice methods (ghbp, sirt, mlem, gridrec, python): ROI slices, sampled on the host
            self.sampler = None
            self.pts = (p_zlo, p_fz, p_y, p_x)
            iy, ix = np.minimum(np.floor(p_y).astype(int), N - 2), np.minimum(np.floor(p_x).astype(int), N - 2)
            self.box = (int(ix.min()), int(ix.max()) + 2, int(iy.min()), int(iy.max()) + 2)   # x0, x1, y0, y1
            self.vals = np.full((n_sets, len(p_zlo)), np.nan, np.float32)

    @property
    def n_points(self):
        return self.sampler.size if self.band else len(self.pts[0])

    def _run_slices(self, chunk, sweep, progress):
        """Whole-slice backprojectors: reconstruct the box of the points slice by slice (chunks), trilinear samples."""
        h0, h1 = self.h_range
        zlo, fz, py, px = self.pts
        x0, x1, y0, y1 = self.box
        N = self.r.info["grid_size"]
        iy = np.minimum(np.floor(py).astype(int), N - 2) - y0
        ix = np.minimum(np.floor(px).astype(int), N - 2) - x0
        wy, wx = py - (iy + y0), px - (ix + x0)
        base = [dict(f) for f in self.r.params["filters"]]
        t0 = time.time()
        for k, chain in enumerate(sweep or [None]):
            if chain is not None:
                self.r.set_params(filters=chain)
            prev = None
            for c0 in range(h0, h1, chunk):
                c1 = min(h1, c0 + chunk)
                vol = self.r.reconstruct_heights(c0, c1, roi=(x0, x1, y0, y1))
                lo = c0
                if prev is not None:
                    vol = np.concatenate([prev[None], vol])
                    lo -= 1
                a, b = np.searchsorted(zlo, [lo, lo + vol.shape[0] - 1])   # points whose two slices are here
                if b > a:
                    z, yy, xx, fy, fx = zlo[a:b] - lo, iy[a:b], ix[a:b], wy[a:b], wx[a:b]
                    def bil(zz):
                        v00, v01 = vol[zz, yy, xx], vol[zz, yy, xx + 1]
                        v10, v11 = vol[zz, yy + 1, xx], vol[zz, yy + 1, xx + 1]
                        top = v00 + fx * (v01 - v00)
                        return top + fy * (v10 + fx * (v11 - v10) - top)
                    lo_v, hi_v = bil(z), bil(z + 1)
                    self.vals[k, a:b] = lo_v + fz[a:b] * (hi_v - lo_v)
                prev = vol[-1].copy()
                if progress:
                    progress(c1 - h0, h1 - h0, time.time() - t0)
        if sweep:
            self.r.set_params(filters=base)

    def run(self, chunk=64, timing=False, progress=None, sweep=None, prefetch=True):
        """Reconstruct [h0, h1) in chunks through the band; returns per-stage seconds (+ 'wall').

        sweep: one entry per value set (CUDA): a dict of filter overrides ({"paganin.delta_beta": 500, ...}, applied
        to the reconstructor's chain) or a whole chain (list of filter dicts); everything before the filters is
        computed once per chunk. prefetch: read the next chunk's raw data in the background (CUDA)."""
        h0, h1 = self.h_range
        if sweep is not None:
            sweep = [c if isinstance(c, list) else self.r.chains([c])[0] for c in sweep]
        if sweep is not None and len(sweep) != self.n_sets:
            raise ValueError(f"{len(sweep)} parameter sets for a sampler with {self.n_sets} value sets")
        if sweep is None and self.n_sets != 1:
            raise ValueError("this sampler has several value sets: pass sweep=[...]")
        if not self.band:
            t0 = time.time()
            self._run_slices(chunk, sweep, progress)
            return {"wall": time.time() - t0}
        self.sampler.reset()
        sec = defaultdict(float)
        t0 = time.time()
        for c0 in range(h0, h1, chunk):
            c1 = min(h1, c0 + chunk)
            nxt = (c1, min(h1, c1 + chunk)) if prefetch and c1 < h1 else None
            for k, v in self.r.rec.reconstruct_band(c0, c1, self.sampler, timing, sweep, nxt).items():
                sec[k] += v
            if progress:
                progress(c1 - h0, h1 - h0, time.time() - t0)
        sec["wall"] = time.time() - t0
        return dict(sec)

    def layers(self, set=0):
        """(L, H, W) float32 samples of value set `set`, NaN where a pixel is not valid."""
        v = np.empty(self.n_points, np.float32)
        v[self.order] = self.sampler.values(set) if self.band else self.vals[set]
        out = np.full((self.L, self.H * self.W), np.nan, np.float32)
        out[:, self.sel] = v.reshape(self.L, -1)
        return out.reshape(self.L, self.H, self.W)

    def composites(self, set=0):
        """(max, mean) float32 (H, W) of value set `set`, NaN where a pixel is not valid."""
        v = np.empty(self.n_points, np.float32)
        v[self.order] = self.sampler.values(set) if self.band else self.vals[set]
        v = v.reshape(self.L, -1)
        mx = np.full(self.H * self.W, np.nan, np.float32)
        mean = np.full(self.H * self.W, np.nan, np.float32)
        mx[self.sel] = v.max(0)
        mean[self.sel] = v.mean(0, dtype=np.float64)
        return mx.reshape(self.H, self.W), mean.reshape(self.H, self.W)
