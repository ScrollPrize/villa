"""Projection filter chains (C++ registry: paganin, epr, mtf_paganin, unsharp, gaussian, median, python).

A chain is a list of dicts, applied in order to every gridded stack before (stage "intensity", default) or after
(stage "log") -log:

    [{"type": "paganin", "delta_beta": 1000}, {"type": "unsharp", "sigma": 1.2, "coeff": 4}]

Text form (CLI): "paganin(delta_beta=1000)+unsharp(sigma=1.2,coeff=4)", or JSON.
In place of paganin + unsharp, "mtf_paganin(delta_beta=1000,sigma=1.06)" retrieves the phase and deconvolves the
detector MTF (Gaussian `sigma` in pixels, or a measured table mtf=0.1:0.8/0.2:0.45/...) in one filter: the blur is
replaced by a Gaussian of `target` pixels (default 0.5) with the gain capped at `gain` (default 12); nsr=0.01 selects
the Wiener form instead.
Overrides address a filter by type (first of that type) or type[k] (k-th of that type):
    {"paganin.delta_beta": 500, "unsharp.coeff": 0, "unsharp[1].sigma": 3}
Python filters: register(name, fn); fn(frames, **params) gets (n, H, W) float32 frames (NumPy; on CUDA a host copy,
or with host=0 the device array, for cupy/torch via DLPack) and modifies them in place or returns new ones:
    register("clip", lambda f, lo=0.0: np.maximum(f, lo, out=f))
    [{"type": "python", "fn": "clip", "lo": 0.05, "margin": 0}]
"""
from __future__ import annotations

import copy
import json

from . import _core

_python_filters: dict = {}


def register(name, fn):
    """Make fn usable as {"type": "python", "fn": name, ...}."""
    _python_filters[name] = fn


def parse_chain(text):
    """'paganin(delta_beta=1000)+unsharp(sigma=1.2,coeff=4)' or JSON -> chain; '' or 'none' -> [].
    (vc_tomography::parse_filter_chain)"""
    if isinstance(text, list):
        return copy.deepcopy(text)
    return json.loads(_core.parse_filter_chain(str(text)))


def chain_text(chain):
    return _core.filter_chain_text(json.dumps(chain))


def override(chain, changes):
    """New chain with {"TYPE[.k].param": value} changes ("filters": a whole chain). (vc_tomography::override_filter_chain)"""
    return json.loads(_core.override_filter_chain(json.dumps(chain), json.dumps(changes)))


def is_filter_key(key):
    return _core.is_filter_key(key)


def sweep_sets(items):
    """['paganin.delta_beta=250,500', 'unsharp.coeff=0,4'] -> [{...}, {...}] (lists zipped position by position;
    one-value lists apply to every set; filters=CHAIN1;CHAIN2 sweeps whole chains). (vc_tomography::surface::sweep_sets)"""
    return json.loads(_core.sweep_sets(list(items)))


def set_name(changes):
    """a parameter set's folder name: {"paganin.delta_beta": 500} -> "db500" (vc_tomography::surface::set_name)"""
    return _core.set_name(json.dumps(changes))
