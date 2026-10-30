"""The label-free split rule: cut winding chains at every step where Lasagna's grad_mag counts more than one
sheet (split rule v2, params digest cd273aed7673; v1, c6d2eecbebc2), and export the pieces. Needs numpy and scipy.
"""

import hashlib
import json
import math
import time
from pathlib import Path


PARAMS = {
    "version": "split-v1",
    "k_s": 0.765,
    "q_own": None, "abs_own": 0.95,
    "reg_r": None, "reg_stat": None, "q_reg": None, "abs_reg": None,
    "cos_min": None,
}


PARAMS_V2 = dict(PARAMS, version="split-v2", abs_own=1.0)


SPLIT_RULES = {"v1": PARAMS, "v2": PARAMS_V2}


CHAIN_RULE_V1 = "0e53e0ae5016"          # the chain rule (version 1) the split rule was tested on


GM_SCALEDOWN = 4


GM_DECODE = 0.25 / 1000.0


SPLIT_FLAG = ("windlabel-flag-split-step", [0.8, 0.3, 1.0],
              "a step the split rule cut (grad_mag says it holds more than one sheet): check the winding count there")


class SplitError(ValueError):
    pass


def _np():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - needs numpy
        raise SplitError("relative_windings_box needs numpy") from exc
    return np


def canonical(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


def digest(o):
    return hashlib.sha256(canonical(o).encode()).hexdigest()[:12]


def steps_of(chains):
    """(chain, j, a_point_index, b_point_index) for every consecutive pair; a has the lower count."""
    out = []
    for c, ch in enumerate(chains):
        k = ch["counts"]
        for j in range(len(k) - 1):
            if abs(k[j + 1] - k[j]) != 1:
                raise ValueError(f"chain {c}: counts {k[j]} and {k[j + 1]} are not one apart")
            out.append((c, j))
    return out


def _qrank(x):
    """Rank share in (0, 1) among finite values; NaN stays NaN."""
    np = _np()
    out = np.full(len(x), np.nan)
    ok = np.flatnonzero(np.isfinite(x))
    if len(ok):
        r = np.argsort(np.argsort(x[ok], kind="stable"), kind="stable")
        out[ok] = (r + 0.5) / len(ok)
    return out


def flags(chains, feats, params):
    """Boolean flag per step (order of steps_of) and the per-step values used."""
    np = _np()
    from scipy.spatial import cKDTree
    st = steps_of(chains)
    key = {(f["chain"], f["j"]): f for f in feats}
    gm = np.array([np.nan if key[s].get("gm_x") is None else key[s]["gm_x"] for s in st], float)
    cos = np.array([np.nan if key[s].get("cos") is None else key[s]["cos"] for s in st], float)
    mid = np.array([0.5 * (np.asarray(chains[c]["points"][j], float) + np.asarray(chains[c]["points"][j + 1], float))
                    for c, j in st]).reshape(-1, 3)
    fin = lambda x: np.where(np.isfinite(x), x, -np.inf)  # noqa: E731
    f = np.zeros(len(st), bool)
    if params.get("q_own") is not None:
        f |= fin(_qrank(gm)) > 1 - params["q_own"]
    if params.get("abs_own") is not None:
        f |= fin(gm) > params["abs_own"]
    reg = np.full(len(st), np.nan)
    if params.get("reg_r") is not None and len(st):
        tree = cKDTree(mid)
        for i, idx in enumerate(tree.query_ball_point(mid, params["reg_r"])):
            v = gm[np.asarray(idx)]
            v = v[np.isfinite(v)]
            if len(v):
                reg[i] = {"mean": np.mean, "p75": lambda x: np.quantile(x, 0.75),
                          "p90": lambda x: np.quantile(x, 0.9)}[params["reg_stat"]](v)
        if params.get("q_reg") is not None:
            f |= fin(_qrank(reg)) > 1 - params["q_reg"]
        if params.get("abs_reg") is not None:
            f |= fin(reg) > params["abs_reg"]
    if params.get("cos_min") is not None:
        f |= np.nan_to_num(cos, nan=0) >= params["cos_min"]
    return st, f, gm, reg


def split_chains(chains, features_or_fields, params=None):
    """Split every chain at its flagged steps. features_or_fields: a list of per-step records
    {chain, j, gm_x, cos} (as written by `features`), or a callable(chains) that returns such a list."""
    params = dict(PARAMS if params is None else params)
    feats = features_or_fields(chains) if callable(features_or_fields) else features_or_fields
    st, f, gm, reg = flags(chains, feats, params)
    cut = {}
    for (c, j), fl in zip(st, f):
        if fl:
            cut.setdefault(c, set()).add(j)
    out = []
    for c, ch in enumerate(chains):
        n = len(ch["points"])
        if n == 0:
            continue
        bounds, lo = [], 0
        for j in sorted(cut.get(c, ())):
            bounds.append((lo, j + 1))
            lo = j + 1
        bounds.append((lo, n))
        k = list(ch["counts"])
        for (s, e) in bounds:
            sub = k[s:e]
            if 0 in sub:
                off = 0
            else:
                off = sub[len(sub) // 2]
            piece = {"points": ch["points"][s:e], "counts": [x - off for x in sub],
                     "normals": ch["normals"][s:e], "steps": list(ch["steps"][s:max(s, e - 1)]),
                     "seed": ch["points"][s + sub.index(off)],
                     "stop": {"in": ch["stop"].get("in") if s == 0 else "split: flagged step",
                              "out": ch["stop"].get("out") if e == n else "split: flagged step"},
                     "split_from": c, "count_offset": off}
            out.append(piece)
    return out, {"flagged_steps": int(f.sum()), "steps": len(st), "pieces": len(out),
                 "chains_in": sum(1 for ch in chains if ch["points"])}


def split_doc(doc, feats, params=None, features_file=None):
    """The split chains file: the input chains file with its chains replaced by the
    pieces and a `split_rule` record (params, digests, counts)."""
    params = dict(PARAMS) if params is None else params
    new, info = split_chains(doc["chains"], feats, params)
    out = dict(doc)
    out["chains"] = new
    out["split_rule"] = {"params": params, "params_digest": digest(params),
                         "features_file": str(features_file), "input_rule_digest": doc.get("rule_digest"), **info,
                         "note": "label-free split of windlabel chains at flagged steps; Model output"}
    out["split_rule"]["output_digest"] = digest([[ch["points"], ch["counts"]] for ch in new])
    return out


def sample_features(chains, inputs, workers=2):
    """gm_x and the cos counter per step (steps_of order) on given _pairs.Inputs: the pair from the
    lower-count crossing to the higher, read in batches of 96 sorted by 192-voxel block, counted with windpair's
    RULE and PARAMS's k_s. cos is None without a cos field."""
    np = _np()
    from . import _pairs as wp
    inp = inputs
    for fld in (inp.m7, inp.cos, inp.gm):
        if fld is not None:
            fld.workers = workers
    st = steps_of(chains)
    A, B = [], []
    for c, j in st:
        p, q = chains[c]["points"][j], chains[c]["points"][j + 1]
        a, b = (p, q) if chains[c]["counts"][j] < chains[c]["counts"][j + 1] else (q, p)
        A.append(a); B.append(b)  # noqa: E702
    A, B = np.asarray(A, float).reshape(-1, 3), np.asarray(B, float).reshape(-1, 3)
    mid = 0.5 * (A + B)
    order = np.lexsort(((mid[:, 0] // 192), (mid[:, 1] // 192), (mid[:, 2] // 192)))
    out = [None] * len(st)
    for lo in range(0, len(order), 96):
        sel = order[lo:lo + 96]
        for i, s in zip(sel, wp.sample_pairs(inp, A[sel], B[sel])):
            cnt = wp.count_from_samples(s, wp.RULE, PARAMS["k_s"])
            gx = cnt.get("gm_x")
            out[int(i)] = {"chain": st[i][0], "j": st[i][1],
                           "gm_x": gx if gx is not None and math.isfinite(gx) else None, "cos": cnt["cos"]}
    return out


def _base_per_query(base_affine):
    np = _np()
    A = np.asarray(base_affine, dtype=np.float64)
    A = A[:3] if A.shape == (4, 4) else A
    sv = np.linalg.svd(A[:, :3], compute_uv=False)
    return float(np.exp(np.log(sv).mean()))


def _is_manifest(source):
    s = str(source)
    return s.endswith(".json")


def _manifest(source):
    """(folder the group zarrs resolve against, manifest dict) of a local lasagna manifest or a
    *.lasagna.json URL."""
    from . import _lasagna as lasagna
    s = str(source)
    if "://" in s:
        base, _, man = lasagna.remote_manifest(s)
        return base.rstrip("/"), man
    p = Path(s)
    if not p.is_file():
        raise SplitError(f"no lasagna manifest at {s}")
    return str(p.parent), json.loads(p.read_text())


def open_grad_mag(source, base_affine=None, scaledown=None, decode=None):
    """(grad_mag Field, record) for query points in the chains' frame. `source` is a lasagna manifest (a local
    .json or a *.lasagna.json URL; its grad_mag group, scaledown and decode, grad_mag_factor over
    grad_mag_encode_scale) or a grad_mag OME-Zarr (local or https; its
    level 0 is the stored level `scaledown`, default 4, decoded by `decode`, default 0.25 / 1000). base_affine
    (3 x 4 or 4 x 4 on x, y, z) maps chain-frame voxels to the lasagna base grid; identity when None."""
    np = _np()
    from . import _lasagna as lasagna, _chains as windlabel, _pairs as wp
    if not str(source or "").strip():
        raise SplitError("no grad_mag source given; this tool reads a published lasagna grad_mag and never "
                         "computes one")
    base = np.eye(3, 4) if base_affine is None else np.asarray(base_affine, dtype=np.float64)
    rec = {"source": str(source)}
    if _is_manifest(source):
        root, man = _manifest(source)
        bad = lasagna.check_manifest_doc(man)
        if bad:
            raise SplitError(f"{source}: {'; '.join(bad)}")
        if float(man.get("source_to_base", 1.0)) != 1.0:
            raise SplitError(f"{source}: source_to_base {man.get('source_to_base')} is not 1")
        g = (man.get("groups") or {}).get("grad_mag")
        if not isinstance(g, dict):
            raise SplitError(f"{source}: no grad_mag group")
        enc = float(man.get("grad_mag_encode_scale", 1000.0))
        fac = man.get("grad_mag_factor")
        if fac is None or not enc > 0 or not float(fac) > 0:
            raise SplitError(f"manifest lacks a usable grad_mag_factor ({fac}) or encode scale ({enc})")
        k = int(g["scaledown"]) if scaledown is None else int(scaledown)
        dec = float(fac) / enc if decode is None else float(decode)
        loc = root.rstrip("/") + "/" + g["zarr"]
        shape = man.get("base_shape_zyx")
    else:
        k = GM_SCALEDOWN if scaledown is None else int(scaledown)
        dec = GM_DECODE if decode is None else float(decode)
        loc, shape = str(source), None
    if k < 0 or not (dec > 0 and math.isfinite(dec)):
        raise SplitError(f"grad_mag scaledown {k} or decode {dec} is not usable")
    try:
        vol = wp.open_volume(loc, cache_chunks=20000)
        meta = vol.levels[0]["meta"]
    except Exception as exc:  # noqa: BLE001 - any failure to open is a refusal naming the source
        raise SplitError(f"cannot open the grad_mag zarr {loc}: {type(exc).__name__}: {exc}") from exc
    got = list(meta["shape"])[-3:]
    if shape:
        want = [-(-int(s) // 2 ** k) for s in shape]
        if any(abs(a - b) > 1 for a, b in zip(got, want)):
            raise SplitError(f"grad_mag: level {k} shape {got} is not ceil({list(shape)} / 2**{k}) = {want}")
    fld = wp.Field.from_volume(vol, 0, affine=windlabel.level_index_affine(base, 2 ** k), scale=dec)
    rec.update({"zarr": loc, "scaledown": k, "decode": dec, "shape": got,
                "base_affine": None if base_affine is None else base.tolist()})
    return fld, rec


def inputs_for(gm_field, base_affine=None):
    """_pairs.Inputs with grad_mag only (the split rule reads gm_x alone): no m7, cos or axis."""
    from . import _pairs as wp
    bpq = 1.0 if base_affine is None else _base_per_query(base_affine)
    return wp.Inputs(m7=None, cos=None, gm=gm_field, axis=None, base_per_query=bpq)


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_split_flags(flags_doc, chains, feats, params=None, affine=None, scale=1.0, max_flags=None):
    """Add one collection to an export flags file: the midpoint of every step the split rule cut, in the same
    frame as the other flags (affine, then scale). Returns the number added."""
    np = _np()
    from . import _chains as wl
    st, f, gm, _ = flags(chains, feats, dict(PARAMS) if params is None else params)
    items = []
    for (c, j), fl, g in zip(st, f, gm):
        if fl:
            a = np.asarray(chains[c]["points"][j], float)
            b = np.asarray(chains[c]["points"][j + 1], float)
            items.append({"p": ((a + b) / 2).tolist(), "chain": c, "j": j,
                          "gm_x": None if not math.isfinite(g) else round(float(g), 4)})
    if max_flags is not None:
        items = items[:max_flags]
    if not items:
        return 0
    tf = wl._affine_fn(affine, scale)
    cols = flags_doc["collections"]
    fmeta = flags_doc.setdefault("relative_windings_box", {}).setdefault("points", {})
    fid = max([int(p) for c in cols.values() for p in c["points"]] or [0])
    nm, color, why = SPLIT_FLAG
    pts = {}
    for c_t, (xyz, it) in enumerate(zip(tf([it["p"] for it in items]), items), 1):
        fid += 1
        pts[str(fid)] = {"p": [round(float(x), 3) for x in xyz], "wind_a": None, "creation_time": c_t}
        fmeta[str(fid)] = dict({k: v for k, v in it.items() if k != "p"}, kind="split")
    cols[str(len(cols) + 1)] = {"name": nm, "points": pts, "color": color,
                                "metadata": {"winding_is_absolute": False, "source": "relative_windings_box split",
                                             "why": why}}
    return len(items)


def export_split(chains_doc, feats, out, params=None, chains_file=None, features_file=None, min_points=2,
                 affine=None, scale=1.0, max_flags=None, extra=None):
    """Split the chains, write split.json, then _chains.export's relative_windings.json, flags.json (with the
    split steps added) and export.json into `out`. Returns the export summary (also written as export.json)."""
    from . import _chains as wl
    t0, c0 = time.time(), time.process_time()
    params = dict(PARAMS) if params is None else params
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sd = split_doc(chains_doc, feats, params, features_file=features_file)
    (out / "split.json").write_text(json.dumps(sd))
    meta = {"chains_file": None if chains_file is None else str(chains_file),
            "rule_digest": chains_doc.get("rule_digest"), "prediction": chains_doc.get("prediction"),
            "split_rule": {k: sd["split_rule"][k] for k in ("params_digest", "flagged_steps", "steps", "pieces",
                                                            "chains_in", "output_digest")}}
    rel, fl, summ = wl.export(sd["chains"], min_points=min_points, affine=affine, scale=scale, meta=meta,
                              max_flags=max_flags)
    n_split = add_split_flags(fl, chains_doc["chains"], feats, params, affine=affine, scale=scale,
                              max_flags=max_flags)
    summ["flags"]["split"] = n_split
    (out / "relative_windings.json").write_text(json.dumps(rel, indent=1))
    (out / "flags.json").write_text(json.dumps(fl, indent=1))
    chk = {"relative_windings.json": wl.check_pcl(rel, "relative"), "flags.json": wl.check_pcl(fl, "flags")}
    summ["checks"] = {k: {"ok": v["ok"], "issues": v["issues"][:5]} for k, v in chk.items()}
    summ["calibration"] = None
    summ["split_rule"] = dict(sd["split_rule"], features_file=None if features_file is None else str(features_file))
    summ["inputs"] = {k: meta[k] for k in ("chains_file", "rule_digest", "prediction")}
    if chains_doc.get("rule_digest", "")[:12] != CHAIN_RULE_V1:
        summ["warning"] = (f"chains grown under rule {str(chains_doc.get('rule_digest'))[:12]}; the split rule was "
                           f"tested on rule {CHAIN_RULE_V1} (chain rule version 1)")
    summ["seconds"] = {"split_export_wall": round(time.time() - t0, 2),
                       "split_export_cpu": round(time.process_time() - c0, 2),
                       "chains_wall": chains_doc.get("seconds"), "chains_cpu": chains_doc.get("cpu_seconds")}
    if extra:
        summ.update(extra)
    summ["files"] = {n: str(out / n) for n in ("relative_windings.json", "flags.json", "export.json", "split.json")}
    summ["ok"] = all(v["ok"] for v in chk.values())
    summ["note"] = "Model output from a surface prediction and lasagna grad_mag; not a reading."
    (out / "export.json").write_text(json.dumps(summ, indent=2))
    return summ


def features_doc(chains_file, chains_doc, source, base_affine=None, scaledown=None, decode=None, workers=2):
    """The features file `features` writes: gm_x (and cos, None here) per step of the chains file."""
    fld, rec = open_grad_mag(source, base_affine, scaledown, decode)
    t0, c0 = time.time(), time.process_time()
    f = sample_features(chains_doc["chains"], inputs_for(fld, base_affine), workers=workers)
    return {"chains_file": str(chains_file), "chains_sha256": _sha_file(chains_file), "k_s": PARAMS["k_s"],
            "grad_mag": rec, "steps": f, "seconds": round(time.time() - t0, 2),
            "cpu_seconds": round(time.process_time() - c0, 2),
            "note": "label-free gm_x per step from lasagna grad_mag; Model output"}
