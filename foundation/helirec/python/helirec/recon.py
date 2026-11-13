"""High-level API: raw projection folders -> calibrated -> reconstructed xy slices: helirec::Reconstruction
(helirec/reconstruction.hpp), which this module binds; from a shell, helirec-reconstruct.

    from helirec.recon import HelicalRecon
    r = HelicalRecon("/path/to/projections", cache_dir="/path/to/cache", device="cuda")
    slices = r.reconstruct_fractions([0.4, 0.5])          # (n, N, N) float32, attenuation in cm^-1

Slice positions use the reference conventions: a fraction f of the doable view-height span
[lo, hi) maps to view height h = hi - 1 - round_half_even(f * (hi - lo)) (f = 0 bottom, 1 top).

Modes (params["mode"]):
  corrected          (default) attenuation in cm^-1, periodic opposite-view rebalance, ramp padding reflected about
                     the true rotation axis, mending over the merged columns the scans cover, fixed unit-DC Paganin
                     kernel (when a Paganin filter is requested); no projection filters by default.
  legacy_reference   reproduces reconstructions of the reference workflow (Night Rail BM18 / nabu-helical / nabuxx):
                     its conventions and every parameter pinned to LEGACY_REFERENCE. Pinned parameters cannot be
                     changed in this mode. Of the geometry calibration it takes the merged frame alone: one rotation
                     axis and one vector between the scans, with the recorded angles and heights.

The parameters (DEFAULTS) and what each does are described in helirec/reconstruction.hpp. The geometry of every
projection is the geometry calibration's (helirec-calibration geometry: geometry.json, made on first use and kept
in the cache directory, or given: params["geometry"]); the darks, flats, weights and the detector's blur field are
the intensity calibration's (helirec-calibration intensity, cached beside it).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from . import _core
from .sld import sld_config  # noqa: F401  (re-exported: the scintillator model is a parameter)

MODES = ("corrected", "legacy_reference")
DEFAULTS = json.loads(_core.reconstruction_default_parameters())
LEGACY_REFERENCE = json.loads(_core.reconstruction_legacy_parameters())
CALIBRATION_KEYS = tuple(_core.reconstruction_calibration_keys())
# parameters of the calibration the geometry calibration replaced (2026-11-12): named in the error for older configs
REMOVED = ("z_translation_scale", "companion_registration", "stationary_registration", "axis_refinement", "rolling_shutter")
# uint8 display window of the reference workflow (legacy reference units); the same physical window in
# attenuation (cm^-1) is twice as wide
REFERENCE_WINDOW = (-0.03, 0.145)
CONVENTIONS = {m: json.loads(_core.reconstruction_mode_conventions(m)) for m in MODES}


def display_window(mode):
    """default uint8 window for outputs of `mode` (the reference window in that mode's units)"""
    return tuple(_core.reconstruction_display_window(mode))


def _json(v):
    return json.dumps(v, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def _raise(e):
    """a C++ error as the exception Python callers expect of it"""
    msg = str(e)
    if msg.startswith(("unknown parameter", "no filter '", "bad filter reference")) or "removed with the calibration" in msg or "is not TYPE.param" in msg:
        return KeyError(msg)
    return ValueError(msg)


def update_params(params, changes):
    """The parameters changed: the rules of mode legacy_reference (pinned parameters; leaving it starts again from
    the defaults), filter keys (TYPE.param) applied to the chain, unknown and removed keys refused.
    (helirec::update_parameters)"""
    try:
        return json.loads(_core.update_parameters(_json(params), _json(changes)))
    except RuntimeError as e:
        raise _raise(e) from None


def legacy_reference_config():
    """SliceReconstructor config keys of mode legacy_reference (all but geometry / physics / device), for callers
    that build the reconstructor themselves"""
    p = {k: v for k, v in LEGACY_REFERENCE.items() if k not in ("backprojector", "sld")}
    p["filters"] = [dict(f) for f in LEGACY_REFERENCE["filters"]]
    return dict(p, legacy_reference=True, **backprojector_config(LEGACY_REFERENCE["backprojector"]))


def backprojector_config(spec):
    """Backprojector spec (text or dict) -> reconstructor config keys (helirec::backprojector_config): ghbp | fbp(interp=,
    oversample=, filter_name=, filter_par=) | sirt(...) | mlem(...) | gridrec(...) | python(fn=NAME, ...)"""
    try:
        return json.loads(_core.backprojector_config(_json(spec) if isinstance(spec, dict) else str(spec)))
    except RuntimeError as e:
        raise ValueError(str(e)) from None


def angle_subset_config(spec):
    """params["angle_subset"] ("" or "I/N") -> (subset, subsets)"""
    try:
        return tuple(_core.angle_subset_config(str(spec or "")))
    except RuntimeError as e:
        raise ValueError(str(e)) from None


@dataclass
class SliceInfo:
    fraction: float
    view_height: int
    nabu_index: int
    height_mm: float


class HelicalRecon:
    """The reconstruction of the scan pair in `projections_dir` (helirec::Reconstruction). params: changes over
    DEFAULTS; cache_dir: the calibration cache (geometry.json, calibration_arrays.bin, calibration.json)."""

    def __init__(self, projections_dir, cache_dir=None, device="cuda", params=None, log=print):
        self.projections_dir, self.cache_dir, self.device, self.log = projections_dir, cache_dir, device, log
        try:
            self._r = _core.Reconstruction(str(projections_dir), _json(dict(params or {})), str(cache_dir or ""), device, log, log is print)
        except RuntimeError as e:
            raise _raise(e) from None
        self.main, self.companion = self._r.main, self._r.companion
        self._built()

    def _built(self):
        self.cal = self._r.intensity
        self.rec = self._r.reconstructor
        self.info = json.loads(self._r.info())
        self.lo, self.hi = self._r.doable
        self.params = json.loads(self._r.parameters)
        self._arrays = None

    @property
    def arrays(self):
        """the intensity calibration's arrays (calibration.calibrate)"""
        if self._arrays is None:
            self._arrays = self.cal.arrays()
        return self._arrays

    def rot_center(self) -> float:
        """the rotation axis in the merged frame (the frame's own: every projection's axis is in its column offset)"""
        return self._r.rot_center()

    def rolling_shutter(self) -> float:
        """the geometry calibration's row timing, as the time from the first stored row to the last in frame periods
        (0 in mode legacy_reference: one angle per frame)"""
        return self._r.rolling_shutter()

    def exposure_arc(self) -> float:
        """params["exposure_arc"] resolved: a number, or "auto" = the main scan's recorded fraction (0 if unknown)"""
        return self._r.exposure_arc()

    def sharpness_weighting(self) -> str:
        """params["sharpness_weighting"] resolved to "a11:a12:a21:a22:c1:c2" or "" (off); "auto" = the calibration's"""
        try:
            return self._r.sharpness_weighting()
        except RuntimeError as e:
            raise ValueError(str(e)) from None

    def _update(self, kw):
        """self.params changed by kw (update_params)"""
        self.params = update_params(self.params, kw)

    @property
    def legacy(self) -> bool:
        return self.params["mode"] == "legacy_reference"

    def provenance(self) -> dict:
        """Everything that defines the reconstruction: mode and its conventions, resolved parameters, the intensity
        calibration and the merged frame, the geometry calibration, physics and the build."""
        p = json.loads(self._r.provenance())
        p["helirec"] = {**p.get("helirec", {}), **_build_identity()}
        return p

    def set_params(self, **kw):
        """Change reconstruction parameters (rebuilds the reconstructor; calibration is kept). Filter chain:
        filters=[...] or dotted keys, e.g. set_params(**{"paganin.delta_beta": 500}). mode="legacy_reference" pins
        every other parameter (see the module docstring)."""
        try:
            self._r.set_parameters(_json(kw))
        except RuntimeError as e:
            raise _raise(e) from None
        self._built()

    def chains(self, sweep):
        """Override dicts ({"paganin.delta_beta": 500, ...}) -> filter chains based on the current one."""
        from .filters import override
        if self.legacy and any(sweep):
            raise ValueError("mode legacy_reference: the filter chain is pinned (sweeps belong in mode 'corrected')")
        return [override(self.params["filters"], s) for s in sweep]

    # ------------------------------------------------------------------ slice addressing
    def slice_info(self, fraction: float) -> SliceInfo:
        try:
            return SliceInfo(*self._r.slice_info(float(fraction)))
        except RuntimeError as e:
            raise ValueError(str(e)) from None

    def reconstruct_heights(self, h_lo: int, h_hi: int, debug=False, timing=False, roi=None, next=None, uint8=None,
                            float32=True, progress=None, mask=None, region=None, float32_to=None, uint8_to=None, z_to=0):
        """Slices [h_lo, h_hi) -> (n, N, N), or (n, y1 - y0, x1 - x0) with roi = (x0, x1, y0, y1).
        next = (h_lo, h_hi): the chunk the caller will ask for next; its raw data is read while this one computes
        (CUDA).
        uint8 = (lo, hi): also the uint8 slices of that window (0 outside the reconstruction disk when the outer
        circle is clipped), converted on the device -> (float32, uint8), or the uint8 slices alone with float32=False.
        progress: a helirec.ProgressBar advanced by one per slice, its postfix the step in hand.
        mask: the uint8 slices are 0 where it is 0: a uint8 array of the output's shape, or a helirec OutputMask
        (possibly coarser: ReferenceMask.read, the mask of a masked reference volume).
        region = (x0, x1, y0, y1): only these pixels of the slice are computed; the rest of the output is 0. With
        uint8 slices alone and a mask, only the pixels within the bounds of what the mask keeps are computed.
        float32_to / uint8_to: an OmeZarrWriter that takes the float32 / uint8 slices as slices [z_to, z_to + n) of its
        volume, straight from the reconstructor (each slice as the block that was computed: no whole slice is built);
        None is returned in place of that array."""
        if uint8 is not None:   # (lo, hi), or (lo, hi, zero_outside) to say it outright
            uint8 = (float(uint8[0]), float(uint8[1]),
                     bool(uint8[2] if len(uint8) > 2 else self.params["clip_outer_circle"]))
        return self.rec.reconstruct(int(h_lo), int(h_hi), debug, timing, None if roi is None else tuple(map(int, roi)),
                                    None if next is None else (int(next[0]), int(next[1])), uint8, float32, progress, mask,
                                    None if region is None else tuple(map(int, region)), float32_to, uint8_to, int(z_to))

    def reconstruct_fractions(self, fractions):
        out = []
        for f in fractions:
            si = self.slice_info(f)
            out.append(self.reconstruct_heights(si.view_height, si.view_height + 1)[0])
        return np.stack(out)


def _build_identity():
    """what of the build only Python knows: the extension's file and the Python sources' hash (the git revision is
    the C++'s)"""
    out = dict(extension=os.path.abspath(_core.__file__))
    try:
        out["extension_mtime"] = os.path.getmtime(_core.__file__)
        import hashlib
        from pathlib import Path
        digest = hashlib.sha256()
        for path in sorted(Path(__file__).parent.rglob("*.py")) + sorted(Path(__file__).parent.rglob("*.h")):
            digest.update(str(path.relative_to(Path(__file__).parent)).encode())
            digest.update(path.read_bytes())
        out["python_source_sha256"] = digest.hexdigest()
    except OSError:
        pass
    return out
