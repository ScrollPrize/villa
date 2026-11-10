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
import re

# default: no projection filters (no Paganin, no unsharp). The reference (Night Rail BM18) chain is REFERENCE_CHAIN.
DEFAULT_CHAIN: list = []
REFERENCE_CHAIN = [{"type": "paganin", "delta_beta": 1000.0}, {"type": "unsharp", "sigma": 1.2, "coeff": 4.0}]

_python_filters: dict = {}


def register(name, fn):
    """Make fn usable as {"type": "python", "fn": name, ...}."""
    _python_filters[name] = fn


def _value(v):
    try:
        f = float(v)
        return int(f) if re.fullmatch(r"[+-]?\d+", v.strip()) else f
    except (TypeError, ValueError):
        return v


def parse_chain(text):
    """'paganin(delta_beta=1000)+unsharp(sigma=1.2,coeff=4)' or JSON -> chain; '' or 'none' -> []."""
    if isinstance(text, list):
        return copy.deepcopy(text)
    text = text.strip()
    if text in ("", "none", "[]"):
        return []
    if text.startswith("["):
        return json.loads(text)
    chain = []
    for item in text.split("+"):
        m = re.fullmatch(r"\s*([\w-]+)\s*(?:\((.*)\))?\s*", item)
        if not m:
            raise ValueError(f"cannot parse filter '{item}'")
        f = {"type": m.group(1)}
        for kv in filter(None, (m.group(2) or "").split(",")):
            k, v = kv.split("=", 1)
            f[k.strip()] = _value(v.strip())
        chain.append(f)
    return chain


def chain_text(chain):
    return "+".join(f["type"] + ("(" + ",".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                                                for k, v in f.items() if k != "type") + ")" if len(f) > 1 else "")
                    for f in chain)


def _locate(chain, ref):
    m = re.fullmatch(r"([\w-]+)(?:\[(\d+)\])?", ref)
    if not m:
        raise KeyError(f"bad filter reference '{ref}'")
    typ, k = m.group(1), int(m.group(2) or 0)
    hits = [f for f in chain if f["type"] == typ]
    if len(hits) <= k:
        raise KeyError(f"no filter '{ref}' in chain {chain_text(chain)}")
    return hits[k]


def override(chain, changes):
    """New chain with {"TYPE[.k].param": value} changes ("filters": a whole chain)."""
    out = copy.deepcopy(chain)
    for key, v in changes.items():
        if key == "filters":
            out = parse_chain(v)
            continue
        ref, _, param = key.rpartition(".")
        if not ref:
            raise KeyError(f"'{key}' is not TYPE.param")
        _locate(out, ref)[param] = v
    return out


def is_filter_key(key):
    return key == "filters" or "." in key


def sweep_sets(items):
    """['paganin.delta_beta=250,500', 'unsharp.coeff=0,4'] -> [{...}, {...}] (lists zipped position by position;
    one-value lists apply to every set)."""
    cols = {}
    for item in items:
        key, vals = item.split("=", 1)
        cols[key] = [_value(v) for v in vals.split(",")] if key != "filters" else vals.split(";")
    n = max(len(v) for v in cols.values())
    for k, v in cols.items():
        if len(v) not in (1, n):
            raise ValueError("--sweep lists must have one length (or one value)")
    return [{k: (v[i] if len(v) > 1 else v[0]) for k, v in cols.items()} for i in range(n)]


def set_name(changes):
    def short(k):
        return (k.replace("mtf_paganin.", "mp_").replace("paganin.delta_beta", "db").replace("unsharp.", "us_")
                .replace(".", "_"))
    return "_".join(f"{short(k)}{v:g}" if isinstance(v, (int, float)) else f"{short(k)}{v}"
                    for k, v in changes.items())
