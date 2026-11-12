"""Scintillator light diffusion models and ESRF BM18 calibration import.

The BM18 profile uses a full-resolution, unit-mass PSF. It retains both frequency
bands directly, rather than approximating them with a coarse digest and a local
high-frequency kernel. Existing scalar specs retain the original binned model.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from . import _core


def _positive_int(value, name, minimum=1):
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number < minimum:
        raise ValueError(f"sld: {name} must be an integer >= {minimum}")
    return int(number)


def _terms(value, name, positive=False):
    if isinstance(value, str):
        value = value.replace(";", ":").split(":")
    elif np.isscalar(value):
        value = [value]
    values = [float(v) for v in value]
    if not values or any(not math.isfinite(v) or (v <= 0 if positive else v < 0) for v in values):
        raise ValueError(f"sld: {name} must contain finite {'positive' if positive else 'non-negative'} values")
    return values


def sld_config(spec):
    """Resolve a scalar spec, BM18 dictionary, or Night Rail diffusion JSON.

    Old: ``fraction=0.1,length=100[,bin=8,stride=10,outside=dark|lit]`` (outside=lit: beyond the detector the
    frame continues by its edge values, for a beam wider than the field; dark is the default).
    BM18: ``profile=bm18,fraction=0.06:0.03,length=40:200`` or ``json=/path/model.json``.
    A dictionary with ``scale_l`` (Night Rail's spelling) also selects BM18.
    BM18 defaults: full resolution (bin=1), every projection (stride=1), illuminated
    borders of 100 detector pixels. ``border_x`` and ``border_y`` override these.
    Nonzero replicas are rejected rather than silently discarded.
    """
    if spec is None or spec == "" or spec == {}:
        return {}
    if isinstance(spec, dict):
        d = dict(spec)
    elif isinstance(spec, os.PathLike) or (isinstance(spec, str) and "=" not in spec):
        d = {"json": os.fspath(spec)}
    else:
        try:
            d = dict((k.strip(), v.strip()) for k, v in (kv.split("=", 1) for kv in str(spec).split(",") if kv.strip()))
        except ValueError as exc:
            raise ValueError("sld: expected key=value entries") from exc
    loaded = False
    if "json" in d:
        path = Path(d.pop("json")).expanduser()
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or "fraction" not in payload or "scale_l" not in payload:
            raise ValueError("sld: Night Rail JSON must contain fraction and scale_l")
        # Only model fields enter reconstruction; fitter ROI/source-image metadata
        # is intentionally not interpreted as processing options.
        names = {"fraction", "scale_l", "replica_shift_x", "replica_shift_y", "replica_factor", "save_binning"}
        d = {**{k: v for k, v in payload.items() if k in names}, **d}
        loaded = True
    known = {"fraction", "length", "scale_l", "bin", "stride", "profile", "border_x", "border_y", "outside",
             "replica_shift_x", "replica_shift_y", "replica_factor", "save_binning"}
    if set(d) - known or "fraction" not in d:
        raise ValueError(f"sld: expected fraction and known model options, got {sorted(d)}")
    if "length" in d and "scale_l" in d:
        raise ValueError("sld: supply length or scale_l, not both")
    fractions = _terms(d["fraction"], "fraction")
    lengths = _terms(d.get("length", d.get("scale_l", 100)), "length", positive=True)
    if len(fractions) != len(lengths):
        raise ValueError("sld: fraction and length lists must have equal lengths")
    inferred = loaded or "scale_l" in d or len(fractions) > 1 or "border_x" in d or "border_y" in d
    profile = d.get("profile", "bm18" if inferred else "legacy")
    if profile not in ("legacy", "bm18"):
        raise ValueError("sld: profile must be legacy or bm18")
    if _positive_int(d.get("save_binning", 1), "save_binning") != 1:
        raise ValueError("sld: calibration was fitted on binned images; convert scale_l to full detector pixels first")
    for name in ("replica_shift_x", "replica_shift_y", "replica_factor"):
        if name in d and (not math.isfinite(float(d[name])) or float(d[name]) != 0):
            raise ValueError("sld: shifted-replica calibrations are not supported")
    bm18 = profile == "bm18"
    outside = str(d.get("outside", "dark"))
    if outside not in ("dark", "lit"):
        raise ValueError("sld: outside must be dark or lit")
    if bm18 and "outside" in d:
        raise ValueError("sld: outside belongs to the binned profile; the BM18 profile takes border_x and border_y")
    binning = _positive_int(d.get("bin", 1 if bm18 else 8), "bin")
    stride = _positive_int(d.get("stride", 1 if bm18 else 10), "stride")
    if bm18:
        if binning != 1:
            raise ValueError("sld: BM18 full-resolution correction requires bin=1; stride may be increased explicitly")
        return dict(sld_bm18=True, sld_fractions=fractions, sld_lengths=lengths, sld_bin=1, sld_stride=stride,
                    sld_border_x=_positive_int(d.get("border_x", 100), "border_x", 0),
                    sld_border_y=_positive_int(d.get("border_y", 100), "border_y", 0))
    if inferred:
        raise ValueError("sld: calibration JSON, multiple tails and borders require profile=bm18")
    if binning > 64 or fractions[0] >= 1 or lengths[0] < 1:
        raise ValueError("sld: legacy profile requires fraction < 1, length >= 1 and bin <= 64")
    if not fractions[0]:
        return {}
    out = dict(sld_fraction=fractions[0], sld_length=lengths[0], sld_bin=binning, sld_stride=stride)
    if outside == "lit":      # (absent otherwise: a resolved configuration without it is the same as before)
        out["sld_lit_outside"] = True
    return out


def correct(frames, spec):
    """Correct dark-subtracted, current-normalized frames (N, rows, columns).

    Apply to projections AND flats before dividing. BM18 returns the full
    deconvolution, clipped at zero (as Nabuxx's flat correction); the legacy
    profile omits a common scale factor that cancels in the flat division.
    The reconstruction's projection correction remains linear/unclipped.
    """
    frames = np.ascontiguousarray(frames, dtype=np.float32)
    return _core.sld_correct_config(frames, sld_config(spec))
