"""Post-fit CT-ridge correction of a fitted Spiral checkpoint: the three steps.

    estimate(fit, ct, band)    -> per-node ridge offset u* and confidence
    field(fit, offset, conf)   -> per-node displacement field u (windings)
    write(fit, u, output)      -> corrected checkpoint (only the gap logits change)

Node = a cell of the checkpoint's gap-logit grid (z row r, strip column c); the
columns of strip segment k - 1 belong to winding k. The numerical work is done
by the modules ``profiles``, ``estimate``, ``field`` and ``writer`` of this
package, which are kept identical to the reference implementation; this module
adds the checkpoint and CT plumbing: z-slab streaming of the CT, per-slab
results on disk so an interrupted run resumes, a memory plan, and the write.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

import numpy as np

from . import estimate as est
from . import field as fld
from . import gapmap as WB
from . import profiles as prof
from . import tracking as trk
from . import writer as W3

# Settings of the method. They are part of the method, not tuning knobs, so
# the command line does not expose them.
SETTINGS = {
    "tmax": 0.40,  # offset search range, windings
    "win": 0.55,  # template window, windings
    "iters": 3,  # template re-alignments
    "r0": 0.70,  # confidence ramp start ...
    "r1": 0.90,  # ... and end
    "cap": 0.35,  # largest move, windings
    "taper": 150.0,  # cosine taper at the z-band ends, voxels
    "gfloor": 3.0,  # no gap is written below min(old gap, gfloor) voxels
    # ridge tracking (the default method): tracker parameters (tracking.DEF unless overridden here), field
    # smoothing sigma (nodes; 0 = none), the distance taper D (nodes) toward unassigned nodes, and the block
    # size (gap rows, counted from the first extracted row) in which any profile leaving the CT stills every node
    "tracking": {"P": {}, "sigma": 0.0, "D": 3.0, "edge_block_rows": 8},
}
METHODS = ("tracking", "nearest")
TAG = "C"  # profile-module key of the one checkpoint handled here
PROVENANCE_KEY = "post_fit_ridge_correction"
# Write-back fidelity bar, between nodes in the moved region (field.between_node_fidelity): the shift the
# edited model realises must correlate with the asked shift at >= 0.90, and its RMS error must be <= 0.50 of
# the asked shift's RMS. Otherwise no checkpoint is written. The writer leaves a roughly constant ~0.01-winding
# residual, so fields whose moves are mostly under ~0.02 winding fail the ratio.
FIDELITY = {"between_corr_min": 0.90, "between_ratio_max": 0.50}
VERIFY_POINTS = 20000
# Bytes per fine-lattice point of one (winding, slab) work item, for the memory
# plan: about twice the 6.4 kB peak numpy allocation measured with tracemalloc
# on the synthetic test fit (torch's share is not counted there).
WORK_BYTES_PER_POINT = 12_000


@dataclass
class Fit:
    path: Path
    checkpoint: dict
    config: dict
    model: object
    umbilicus: Path
    geo: dict
    dr: float


def load_fit(path, umbilicus=None):
    """Load a checkpoint and rebuild its model on the CPU (the flattener's path)."""
    import torch

    import flatten_spiral_checkpoint as fsc
    from checkpoint_io import load_checkpoint_cpu

    path = Path(path).expanduser().resolve(strict=True)
    checkpoint = load_checkpoint_cpu(str(path))
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint root must be a dictionary")
    config = fsc._checkpoint_config(checkpoint)
    umb = fsc._resolve_umbilicus(path, Path(umbilicus) if umbilicus else None)
    model = fsc._build_model(checkpoint, config, umb, torch.device("cpu"))
    gp = model.gap_expander_params
    idx = gp.winding_first_logit_idx.double().numpy()
    geo = {"idx": idx, "num": np.diff(idx), "Wc": int(idx[-1]), "Z": int(gp.logits.shape[2]),
               "zmin": float(model.flow_min_corner_zyx[0]), "zmax": float(model.flow_max_corner_zyx[0])}
    dr = float(model.get_dr_per_winding().detach())
    return Fit(path, checkpoint, config, model, umb, geo, dr)


def default_windings(fit):
    """The windings the fit reports (the flattener's rule), within the grid."""
    cfg = fit.config
    first = next((v for v in (fit.checkpoint.get("preview_first_winding"), cfg.get("first_winding"),
                              cfg.get("output_first_winding")) if v is not None), 10)
    last = cfg.get("shell_outer_winding_idx")
    if last is None:
        last = int(cfg["model_gap_expander_num_windings"]) - 1
    nw = len(fit.geo["num"])
    return max(1, int(first)), min(nw, int(last))


def rows_for_band(geo, band):
    """Gap rows inside the z band plus one on each side."""
    zmin, zmax, Z = geo["zmin"], geo["zmax"], geo["Z"]

    def rz(z):
        return (z - zmin) / (zmax - zmin) * (Z - 1)

    r0 = max(0, math.ceil(rz(band[0])) - 1)
    r1 = min(Z, math.floor(rz(band[1])) + 2)
    if r1 <= r0:
        raise ValueError(f"z band {band} holds no gap rows of this checkpoint")
    return r0, r1


def log_line(message):
    """Progress line, flushed so it shows up in redirected logs as it happens."""
    print(message, flush=True)


def state_digest(state, checkpoint):
    from checkpoint_io import model_state_sha256

    return model_state_sha256(state, (), int(checkpoint["z_begin"]), int(checkpoint["z_end"]))


# ---------------------------------------------------------------------------
# Profile extraction, streamed per z slab
# ---------------------------------------------------------------------------

def _install(fit, ct):
    """Fill the profile module's globals (inherited by forked workers)."""
    prof.G.clear()
    prof.G.update(tr={TAG: fit.model.get_slice_to_spiral_transform()},
                  dr={TAG: fit.dr}, geo={TAG: fit.geo}, ct=ct)


def _worker_init():
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    threadpool_limits(1)


def _probe_points(fit, ra, rb, k0, k1):
    """CT positions bounding every profile of rows [ra, rb), windings k0..k1:
    integer and half-integer windings k0 - 1 .. k1 + 1 at rows ra - 1 .. rb,
    at angles no more than ~32 voxels apart on the outermost winding."""
    g = fit.geo
    rows = np.arange(ra - 1, rb + 1, dtype=np.float64)
    z = g["zmin"] + rows * (g["zmax"] - g["zmin"]) / (g["Z"] - 1)
    nth = max(64, math.ceil(2 * math.pi * (k1 + 2) * fit.dr / 32.0))
    th = np.linspace(-0.2, 2 * math.pi + 0.2, nth)
    w = np.arange(2 * (k0 - 1), 2 * (k1 + 1) + 1) / 2.0
    Z, T, Wv = np.meshgrid(z, th, w, indexing="ij")
    return prof.inv(TAG, Z.ravel(), T.ravel(), Wv.ravel())


PROBE_MARGIN = 16.0  # voxels around each probe point


def _row_keys(job):
    """CT chunks near the probe points of one gap row (row r alone: rows r .. r)."""
    r, k0, k1 = job
    return prof.G["ct"].keys_near(_probe_points(_PLAN_FIT[0], r + 1, r, k0, k1), PROBE_MARGIN)


_PLAN_FIT = []  # the fit whose rows are being probed (inherited by forked planning workers)


def probe_row_keys(fit, ct, rows, windings, threads=1):
    """{gap row: CT chunk keys near its probe points} for every row in ``rows``.

    A slab of rows [ra, rb) needs the union over rows ra - 1 .. rb, which is
    exactly the chunk set of probing those rows together; each row is probed
    once, in worker processes when ``threads`` > 1."""
    _install(fit, ct)
    _PLAN_FIT[:] = [fit]
    jobs = [(int(r), windings[0], windings[1]) for r in rows]
    if threads <= 1 or len(jobs) < 2:
        keys = [_row_keys(j) for j in jobs]
    else:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(min(threads, len(jobs)), initializer=_worker_init) as pool:
            keys = pool.map(_row_keys, jobs)
    _PLAN_FIT.clear()
    return {j[0]: k for j, k in zip(jobs, keys, strict=True)}


def _slab_keys(row_keys, ra, rb):
    parts = [row_keys[r] for r in range(ra - 1, rb + 1)]
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)


def _memory(fit, ct, rows, windings, slab_rows, threads, row_keys):
    """Memory estimate (bytes) of extracting ``rows`` in slabs of ``slab_rows``."""
    g = fit.geo
    r0, r1 = rows
    k0, k1 = windings
    slabs = [(ra, min(ra + slab_rows, r1)) for ra in range(r0, r1, slab_rows)]
    keys = [_slab_keys(row_keys, ra, rb) for ra, rb in slabs]
    ct_bytes = [len(k) * ct.chunk_bytes for k in keys]  # upper bound: absent (air) chunks cost nothing
    idx = g["idx"]
    c0, c1 = int(idx[k0 - 1]), int(idx[k1])
    widest = int(max(g["num"][k0 - 1:k1]))
    item_points = (3 * slab_rows + 2) * (3 * widest + 2)
    nr = r1 - r0
    model_bytes = sum(t.numel() * t.element_size() for t in fit.model.state_dict().values())
    mem = {
        "model": 2 * model_bytes,
        "ct_slab_max": max(ct_bytes) if ct_bytes else 0,
        "workers": max(1, threads) * item_points * WORK_BYTES_PER_POINT,
        "slab_output": slab_rows * (c1 - c0) * 41 * 4 * 4,
        "estimate": int(nr * widest * 41 * 8 * 8),
        "write": int(nr * g["Wc"] * 8 * 24),
    }
    # interpreter, torch and numpy per process (0.3 GB RSS measured for the whole synthetic test run)
    mem["runtime"] = (1 + (threads if threads > 1 else 0)) * 2**29
    base = mem["model"] + mem["runtime"]
    mem["extraction_peak"] = base + mem["ct_slab_max"] + mem["workers"] + mem["slab_output"]
    mem["peak"] = max(mem["extraction_peak"], base + mem["estimate"], base + mem["write"])
    return slabs, keys, mem


@dataclass
class Plan:
    rows: tuple
    windings: tuple
    slabs: list
    columns: tuple
    keys: list
    memory: dict
    fingerprint: str
    row_keys: dict = dataclass_field(default_factory=dict, repr=False)


def plan_extraction(fit, ct, band, windings, slab_rows, threads):
    """Slabs, the CT chunks each one prefetches and a memory estimate (bytes)."""
    g = fit.geo
    r0, r1 = rows_for_band(g, band)
    k0, k1 = windings
    if not 1 <= k0 <= k1 <= len(g["num"]):
        raise ValueError(f"winding range {windings} outside 1..{len(g['num'])}")
    row_keys = probe_row_keys(fit, ct, range(r0 - 1, r1 + 1), (k0, k1), threads)
    slabs, keys, mem = _memory(fit, ct, (r0, r1), (k0, k1), slab_rows, threads, row_keys)
    idx = g["idx"]
    c0, c1 = int(idx[k0 - 1]), int(idx[k1])
    ident = {"model_state_sha256": state_digest(fit.checkpoint["spiral_and_transform"], fit.checkpoint),
             "ct": ct.location, "ct_shape": ct.shape, "ct_chunks": ct.chunks, "rows": [r0, r1], "windings": [k0, k1],
             "slabs": slabs, "settings": SETTINGS}
    fp = hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()
    return Plan((r0, r1), (k0, k1), slabs, (c0, c1), keys, mem, fp, row_keys)


def fitting_slab_rows(fit, ct, plan, threads, max_memory_gb):
    """Largest ``--slab-rows`` below the plan's whose estimate fits, or None."""
    current = plan.slabs[0][1] - plan.slabs[0][0] if plan.slabs else 1
    for rows in range(current - 1, 0, -1):
        mem = _memory(fit, ct, plan.rows, plan.windings, rows, threads, plan.row_keys)[2]
        if mem["peak"] / 2**30 <= max_memory_gb:
            return rows
    return None


def _slab_name(ra, rb):
    return f"slab_{ra:05d}_{rb:05d}"


def _slab_done(work_dir, ra, rb, fingerprint):
    meta = work_dir / f"{_slab_name(ra, rb)}.json"
    data = work_dir / f"{_slab_name(ra, rb)}.npy"
    nan = work_dir / f"{_slab_name(ra, rb)}.nan.npy"
    rows = work_dir / f"{_slab_name(ra, rb)}.nanrows.npy"
    if not (meta.is_file() and data.is_file() and nan.is_file() and rows.is_file()):
        return False
    try:
        return json.loads(meta.read_text()).get("fingerprint") == fingerprint
    except ValueError:
        return False


def _atomic_json(path, obj):
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def _run_slab(fit, ct, plan, ra, rb, threads):
    k0, k1 = plan.windings
    C0, C1 = plan.columns
    out = np.full((rb - ra, C1 - C0, 41), np.nan, np.float32)
    nan_nodes = np.ones((rb - ra, C1 - C0), bool)
    nan_fine = np.zeros(3 * (rb - ra) + 2, bool)
    items = [(TAG, k, ra, rb) for k in range(k1, k0 - 1, -1)]
    nan_profiles = 0
    cpu = 0.0
    on_demand_before = ct.stats["on_demand_chunks"]

    def take(result):
        nonlocal nan_profiles, cpu
        _tag, _k, _ra, _rb, c0, c1, (stacks, nan_k, nan_fine_k), nnan, dt = result
        out[:, c0 - C0:c1 - C0] = stacks
        nan_nodes[:, c0 - C0:c1 - C0] = nan_k
        nan_fine[:] |= nan_fine_k
        nan_profiles += int(nnan)
        cpu += dt

    on_demand = 0
    if threads <= 1:
        for item in items:
            take(prof.work(item))
        on_demand = ct.stats["on_demand_chunks"] - on_demand_before
    else:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(threads, initializer=_worker_init) as pool:
            for result in pool.imap_unordered(_work_counted, items):
                on_demand += result[1]
                take(result[0])
    return out, nan_nodes, nan_fine, {"nan_profiles": nan_profiles, "work_cpu_s": cpu, "on_demand_chunks": on_demand}


def _work_counted(item):
    before = prof.G["ct"].stats["on_demand_chunks"]
    result = prof.work(item)
    return result, prof.G["ct"].stats["on_demand_chunks"] - before


def extract(fit, ct, plan, work_dir, threads=1, log=None, stop_after_slabs=None):
    """Profile stacks of every node of the plan, one z slab at a time.

    A finished slab is written to ``work_dir`` (data first, then its marker,
    each atomically), so a re-run with the same plan skips it."""
    log = log or log_line
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    plan_file = work_dir / "plan.json"
    if plan_file.is_file():
        old = json.loads(plan_file.read_text()).get("fingerprint")
        if old != plan.fingerprint:
            raise RuntimeError(f"{work_dir} holds the work of a different run "
                               "(checkpoint, CT, band, windings or slabs differ); use another --work-dir")
    else:
        _atomic_json(plan_file, {"fingerprint": plan.fingerprint, "rows": plan.rows, "windings": plan.windings,
                                     "slabs": plan.slabs, "columns": plan.columns})
    _install(fit, ct)
    stats = []
    done_now = 0
    t_start = time.time()
    for i, (ra, rb) in enumerate(plan.slabs):
        name = _slab_name(ra, rb)
        if _slab_done(work_dir, ra, rb, plan.fingerprint):
            stats.append(dict(json.loads((work_dir / f"{name}.json").read_text()), resumed=True))
            continue
        if stop_after_slabs is not None and done_now >= stop_after_slabs:
            raise KeyboardInterrupt(f"stopped before slab {name} (test hook)")
        t0 = time.time()
        ct.retain(plan.keys[i])
        t_load = time.time() - t0
        resident = ct.resident_bytes()
        out, nan_nodes, nan_fine, st = _run_slab(fit, ct, plan, ra, rb, threads)
        for suffix, arr in ((".npy", out), (".nan.npy", nan_nodes), (".nanrows.npy", nan_fine)):
            tmp = work_dir / f".{name}{suffix}.tmp-{os.getpid()}"
            with open(tmp, "wb") as fh:
                np.save(fh, arr)
            os.replace(tmp, work_dir / f"{name}{suffix}")
        st.update(rows=[ra, rb], fingerprint=plan.fingerprint, ct_load_s=t_load, ct_resident_bytes=resident,
                  wall_s=time.time() - t0)
        _atomic_json(work_dir / f"{name}.json", st)
        stats.append(st)
        done_now += 1
        log(f"[ridge] slab rows {ra}-{rb} ({i + 1}/{len(plan.slabs)}): CT load {t_load:.0f}s, "
            f"resident {resident / 2**30:.2f} GiB, NaN profiles {st['nan_profiles']}, "
            f"on-demand chunks {st['on_demand_chunks']}, wall {st['wall_s']:.0f}s, total {time.time() - t_start:.0f}s")
    ct.retain([])
    return stats


def _nan_nodes(work_dir, plan):
    """(rows, C1 - C0) bool: nodes whose box stack or either half of the 3 x 3 lattice is NaN."""
    return np.concatenate([np.load(work_dir / f"{_slab_name(ra, rb)}.nan.npy") for ra, rb in plan.slabs], 0)


def _candidates_job(job):
    work_dir, plan, k, c0, c1 = job
    nr = plan.rows[1] - plan.rows[0]
    cu, cr, u1, c1v, hist = trk.winding_candidates(_sheet_stack(work_dir, plan, c0, c1).reshape(-1, 41), history=True)
    return (k, c0, c1, cu.reshape(nr, c1 - c0, -1), cr.reshape(nr, c1 - c0, -1), u1.reshape(nr, -1), c1v.reshape(nr, -1),
            template_centre(hist))


def template_centre(hist):
    """Final template centre of one winding (windings) from estimate.sheet_template's history: the centre,
    whether its last step fell back to the grid maximum, whether any step did, and whether it sits on the
    edge of the centre window. None when the winding had too few finite profiles for a template."""
    if not hist:
        return None
    z = hist[-1]["final_z0"]
    return {"centre": z, "clipped": bool(hist[-1]["clipped"]), "clipped_any_step": any(h["clipped"] for h in hist),
            "on_window_edge": bool(abs(z) >= est.CENTRE_WINDOW - 1e-9)}


def tracking_inputs(work_dir, plan, idx, threads=1):
    """Per-node ridge candidates (cu, cr: (rows, W, NC)), the plain estimate (u1, c1: (rows, W)) and the
    validity mask (rows, W) of the plan's windings, W = idx[last winding]. A node is invalid if its box stack
    or either half of its 3 x 3 lattice is NaN. In addition the gap rows are grouped in blocks of
    SETTINGS["tracking"]["edge_block_rows"] from the first extracted row, and a block in which any lattice
    profile (including the 1/3-row margin sampled around it) leaves the CT is invalid as a whole."""
    k0, k1 = plan.windings
    nr = plan.rows[1] - plan.rows[0]
    W = int(idx[k1])
    C0 = plan.columns[0]
    cu = np.full((nr, W, trk.NC), np.nan, np.float32)
    cr = np.full((nr, W, trk.NC), np.nan, np.float32)
    u1 = np.zeros((nr, W), np.float32)
    c1 = np.zeros((nr, W), np.float32)
    jobs = [(work_dir, plan, k, int(idx[k - 1]), int(idx[k])) for k in range(k0, k1 + 1)]
    jobs.sort(key=lambda j: -(j[4] - j[3]))
    centres = {}
    if threads <= 1:
        results = map(_candidates_job, jobs)
        pool = None
    else:
        import multiprocessing as mp

        pool = mp.get_context("fork").Pool(threads, initializer=_worker_init)
        results = pool.imap_unordered(_candidates_job, jobs)
    try:
        for k, a, b, cu_k, cr_k, u1_k, c1_k, centre in results:
            centres[k] = centre
            cu[:, a:b] = cu_k
            cr[:, a:b] = cr_k
            u1[:, a:b] = u1_k
            c1[:, a:b] = c1_k
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    nan = np.zeros((nr, W), bool)
    nan[:, C0:W] = _nan_nodes(work_dir, plan)[:, :W - C0]
    r0, r1 = plan.rows
    nan_fine = set()  # NaN fine-row positions, in thirds of a gap row
    for ra, rb in plan.slabs:
        flags = np.load(work_dir / f"{_slab_name(ra, rb)}.nanrows.npy")
        nan_fine.update(int(3 * ra - 2 + i) for i in np.flatnonzero(flags))
    block = int(SETTINGS["tracking"]["edge_block_rows"])
    for a in range(r0, r1, block):
        b = min(a + block, r1)
        if any(3 * a - 2 <= q <= 3 * (b - 1) + 2 for q in nan_fine):
            nan[a - r0:b - r0] = True
    return {"cu": cu, "cr": cr, "u1": u1, "c1": c1, "valid": ~nan, "centres": dict(sorted(centres.items()))}


def _sheet_stack(work_dir, plan, c0, c1):
    """(rows, c1 - c0, 41) float64 profile stack of one winding's columns, from the slab files."""
    C0 = plan.columns[0]
    parts = [np.load(work_dir / f"{_slab_name(ra, rb)}.npy", mmap_mode="r")[:, c0 - C0:c1 - C0] for ra, rb in plan.slabs]
    return np.concatenate([np.asarray(p, np.float64) for p in parts], 0)


# ---------------------------------------------------------------------------
# The three steps
# ---------------------------------------------------------------------------

def memory_limit_gib():
    """Planning limit: 32 GiB or 80% of this machine's memory, whichever is smaller."""
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        total = 40.0
    return min(32.0, 0.8 * total)


def estimate(fit, ct, band, *, windings=None, work_dir, threads=1, slab_rows=4, max_memory_gb=None,
             log=None, stop_after_slabs=None, method="tracking"):
    """Per-node ridge estimates for the band's rows: ridge candidates and a validity mask for
    ``method="tracking"``, or the single best offset u* (windings) and its confidence for "nearest".

    The CT is read in slabs of ``slab_rows`` gap rows, fewer if the memory plan
    for that size is above ``max_memory_gb`` (default: memory_limit_gib()).
    Returns a dict with the method's arrays (rows = the plan's gap rows), ``plan``
    and per-slab ``slabs`` statistics."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    log = log or log_line
    limit = memory_limit_gib() if max_memory_gb is None else max_memory_gb
    windings = tuple(windings) if windings else default_windings(fit)
    plan = plan_extraction(fit, ct, band, windings, slab_rows, threads)
    if plan.memory["peak"] / 2**30 > limit:
        fits = fitting_slab_rows(fit, ct, plan, threads, limit)
        if fits is None:
            raise MemoryError(f"memory estimate {plan.memory['peak'] / 2**30:.1f} GiB is above {limit:.1f} GiB even with "
                              f"1-row slabs on {threads} threads: use fewer --threads or a narrower --windings range")
        plan = plan_extraction(fit, ct, band, windings, fits, threads)
    peak = plan.memory["peak"] / 2**30
    nrows = plan.slabs[0][1] - plan.slabs[0][0]
    log(f"[ridge] rows {plan.rows[0]}-{plan.rows[1]} ({len(plan.slabs)} slabs of {nrows}), windings "
        f"{windings[0]}-{windings[1]}, memory estimate {peak:.1f} GiB "
        f"(CT per slab <= {plan.memory['ct_slab_max'] / 2**30:.1f} GiB; limit {limit:.1f} GiB)")
    work_dir = Path(work_dir)
    slab_stats = extract(fit, ct, plan, work_dir, threads, log, stop_after_slabs)
    out = {"method": method, "plan": plan, "slabs": slab_stats, "windings": windings, "band": tuple(band)}
    if method == "tracking":
        out.update(tracking_inputs(work_dir, plan, fit.geo["idx"], threads))
        return out
    # Per winding: template, then offsets (the reference estimator's loop).
    E = est.Est(tmax=SETTINGS["tmax"], win=SETTINGS["win"])
    idx = fit.geo["idx"]
    nr = plan.rows[1] - plan.rows[0]
    u = np.zeros((nr, fit.geo["Wc"]), np.float32)
    c = np.zeros((nr, fit.geo["Wc"]), np.float32)
    centres = {}
    for k in range(windings[0], windings[1] + 1):
        c0, c1 = int(idx[k - 1]), int(idx[k])
        P = _sheet_stack(work_dir, plan, c0, c1).reshape(-1, 41)
        T, hist = est.sheet_template(E, P, iters=SETTINGS["iters"])
        centres[k] = template_centre(hist)
        if T is None:
            continue
        uu, cc = E.estimate(P, T)
        u[:, c0:c1] = uu.reshape(nr, c1 - c0)
        c[:, c0:c1] = cc.reshape(nr, c1 - c0)
    out.update(offset=u, confidence=c, centres=centres)
    return out


def tracking_field(fit, e, band):
    """Ridge-tracking move field (Z, Wc), float32-rounded as it is stored, plus the tracker's statistics."""
    g = fit.geo
    idx = np.asarray(g["idx"]).astype(int)
    k1 = e["windings"][1]
    r0w, r1w = e["plan"].rows
    Z = int(g["Z"])
    zrow = g["zmin"] + np.arange(Z) * (g["zmax"] - g["zmin"]) / (Z - 1)
    tw = fld.taper_rows(zrow, tuple(band), SETTINGS["taper"])[r0w:r1w]
    t = SETTINGS["tracking"]
    T = trk.track(e["cu"], e["cr"], e["u1"], e["c1"], e["valid"], idx, k1, t["P"])
    F = trk.field(T, t["sigma"], t["D"], tw)
    U = np.zeros((Z, int(g["Wc"])))
    U[r0w:r1w, :F.shape[1]] = F
    U[:, 0] = 0.0
    U = U.astype(np.float32).astype(np.float64)
    st = T["state"]
    names = ["invalid", "unreached", "ambiguous", "nofit", "assigned", "anchor"]
    stats = dict(T["info"], states={n: float((st == i).mean()) for i, n in enumerate(names)},
                 share_moved=float((np.abs(F) > 1e-6).mean()), mean_abs_F=float(np.abs(F).mean()),
                 max_abs_F=float(np.abs(F).max()))
    return U, stats


def field(fit, e, *, band, windings, rows, scale=1.0):
    """Displacement field u (Z, Wc) in windings from the estimate ``e``.

    ``scale`` multiplies the field. A scaled field passes through float32 (as
    in the method's reference runs, where the field is stored as float32);
    scale 1 leaves the field untouched."""
    if e["method"] == "tracking":
        U, stats = tracking_field(fit, e, band)
    else:
        U, stats = fld.field(e["offset"], e["confidence"], fit.geo, rows, tuple(band), tuple(windings),
                             r0=SETTINGS["r0"], r1=SETTINGS["r1"], cap=SETTINGS["cap"], taper=SETTINGS["taper"])
    stats["method"] = e["method"]
    stats["field_scale"] = float(scale)
    if scale != 1.0:
        U = (float(scale) * U.astype(np.float32).astype(np.float64)).astype(np.float32).astype(np.float64)
        r0w, r1w = rows
        stats["scaled_mean_abs_u"] = float(np.abs(U[r0w:r1w]).mean())
    return U, stats


class FidelityError(RuntimeError):
    """The edited model does not realise the written field well enough."""


def fidelity_failures(fid):
    """Reasons the write-back check fails (empty when it passes)."""
    out = []
    if fid["corr"] is None or not fid["corr"] >= FIDELITY["between_corr_min"]:
        out.append(f"correlation between nodes {fid['corr']} < {FIDELITY['between_corr_min']}")
    if fid["ratio"] is None or not fid["ratio"] <= FIDELITY["between_ratio_max"]:
        out.append(f"error ratio between nodes {fid['ratio']} > {FIDELITY['between_ratio_max']}")
    return out


def write(fit, U, output, *, band, windings, extra=None, verify_points=VERIFY_POINTS):
    """Write U into the checkpoint's gap logits and save it to ``output``.

    Only ``gap_expander_params.logits`` can change. Cells whose solved logit
    equals the stored one to 1e-12 keep the stored float32 bits, rows where U
    is zero are never touched, and when nothing changes the model state is
    the input's own tensors. ``model_state_sha256`` is recomputed.

    Before saving, the edited model is checked against the field it was asked
    to realise (FIDELITY); a failing check raises FidelityError and writes
    nothing. A field confined to the z taper (band shorter than twice the
    taper) cannot be checked and is refused the same way."""
    import torch

    ck = fit.checkpoint
    st_in = ck["spiral_and_transform"]
    key = "gap_expander_params.logits"
    stored = st_in[key][0, 0].numpy()
    L0 = stored.astype(np.float64)
    gm = WB.GapMap(fit.model, fit.config)
    idx = fit.geo["idx"]
    rows = np.flatnonzero(np.abs(U).max(1) > 0)
    L1 = L0.copy()
    Ueff = U.copy()
    info = {}
    if len(rows):
        Ln, Ue, info = W3.write_field(L0[rows], gm, idx, gm.dr, U[rows], gfloor=SETTINGS["gfloor"])
        Ueff[rows] = Ue
        L1[rows] = Ln
    L1f = L1.astype(np.float32)
    keep = np.abs(L1 - L0) < 1e-12
    L1f[keep] = stored[keep]
    changed = L1f.view(np.uint32) != stored.view(np.uint32)
    state = dict(st_in)
    if changed.any():
        new = st_in[key].clone()
        new[0, 0] = torch.from_numpy(np.ascontiguousarray(L1f))
        state[key] = new.contiguous()
    digest_in_stored = ck.get("model_state_sha256")
    digest_in = state_digest(st_in, ck)
    digest_out = state_digest(state, ck)
    out = dict(ck)
    out["spiral_and_transform"] = state
    out["model_state_sha256"] = digest_out
    out[PROVENANCE_KEY] = {"input": fit.path.name, "input_model_state_sha256": digest_in_stored,
                           "settings": SETTINGS, **(extra or {})}
    g0 = gm.gaps(L0)
    g1 = gm.gaps(L1f.astype(np.float64))
    rep = {
        "logits_changed": int(changed.sum()),
        "max_abs_logit_change": float(np.abs(L1f.astype(np.float64) - L0).max()),
        "rows_written": [int(rows.min()), int(rows.max())] if len(rows) else [],
        "min_gap_written_rows_before_after": [float(g0[rows][:, 1:].min()), float(g1[rows][:, 1:].min())] if len(rows) else None,
        "writer": info,
        "mean_abs_field_asked": float(np.abs(U).mean()),
        "mean_abs_field_written": float(np.abs(Ueff).mean()),
        "input_model_state_sha256_stored": digest_in_stored,
        "input_model_state_sha256_recomputed": digest_in,
        "input_digest_consistent": digest_in_stored == digest_in,
        "output_model_state_sha256": digest_out,
    }
    if changed.any():
        band_in = (band[0] + SETTINGS["taper"], band[1] - SETTINGS["taper"])
        if band_in[1] <= band_in[0]:
            raise FidelityError(f"z band {band} is shorter than twice the {SETTINGS['taper']:g}-voxel taper: the write "
                                "cannot be checked; no checkpoint written")
        model_new = copy.deepcopy(fit.model)
        model_new.load_state_dict(state)
        rep["fidelity"] = fld.between_node_fidelity(fit.model, model_new, U, Ueff, fit.geo, band_in, tuple(windings),
                                                    np.random.default_rng(0), verify_points)
        # information only: the written field at random exact node positions
        rep["verify_at_nodes"] = fld.verify(fit.model, model_new, Ueff, fit.geo, band_in, np.random.default_rng(1),
                                            verify_points // 2, tuple(windings), at_nodes=True)
        rep["fidelity_failures"] = fidelity_failures(rep["fidelity"])
        if rep["fidelity_failures"]:
            err = FidelityError("write-back check failed (" + "; ".join(rep["fidelity_failures"]) + "); no checkpoint written")
            err.report = rep
            raise err
    output = Path(output)
    tmp = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    torch.save(out, tmp)
    os.replace(tmp, output)
    return rep, Ueff
