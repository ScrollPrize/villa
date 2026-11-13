"""Scintillator light diffusion models and ESRF BM18 calibration import.

The BM18 profile uses a full-resolution, unit-mass PSF. It retains both frequency
bands directly, rather than approximating them with a coarse digest and a local
high-frequency kernel. Existing scalar specs retain the original binned model.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import _core


def sld_config(spec):
    """Resolve a scintillator light diffusion model to the reconstructor's keys (sld_fraction, ...; {} for none).

    Old: ``fraction=0.1,length=100[,bin=8,stride=10,outside=dark|lit]`` (outside=lit: beyond the detector the
    frame continues by its edge values, for a beam wider than the field; dark is the default).
    BM18: ``profile=bm18,fraction=0.06:0.03,length=40:200`` or ``json=/path/model.json`` (or the path alone).
    A dictionary with ``scale_l`` (Night Rail's spelling) also selects BM18.
    BM18 defaults: full resolution (bin=1), every projection (stride=1), illuminated
    borders of 100 detector pixels. ``border_x`` and ``border_y`` override these.
    Nonzero replicas are rejected rather than silently discarded. (helirec::sld_params)
    """
    if spec is None or spec == "" or spec == {}:
        return {}
    text = json.dumps(spec, default=lambda o: os.fspath(o) if isinstance(o, os.PathLike) else float(o)) if isinstance(spec, dict) else os.fspath(spec) if isinstance(spec, os.PathLike) else str(spec)
    try:
        return json.loads(_core.sld_config(text))
    except RuntimeError as e:
        raise ValueError(str(e)) from None


def correct(frames, spec):
    """Correct dark-subtracted, current-normalized frames (N, rows, columns).

    Apply to projections AND flats before dividing. BM18 returns the full
    deconvolution, clipped at zero (as Nabuxx's flat correction); the legacy
    profile omits a common scale factor that cancels in the flat division.
    The reconstruction's projection correction remains linear/unclipped.
    """
    frames = np.ascontiguousarray(frames, dtype=np.float32)
    return _core.sld_correct_config(frames, sld_config(spec))
