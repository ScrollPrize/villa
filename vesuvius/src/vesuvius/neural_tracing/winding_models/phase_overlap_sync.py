"""Topology-free synchronization of overlapping local winding phase fields.

Each cached slab predicts a continuous phase field whose additive gauge is
currently fixed by declaring the fitted seed midpoint to be an integer
crossing.  That is fragile when the fitted spiral is locally wrong.  This
module instead samples every slab at nearby slabs' seed locations, builds a
world-space overlap graph, and robustly solves one continuous gauge correction
per slab.  The fitted seed winding is retained only as a weak gauge prior; no
edge in the graph comes from fitted winding adjacency or spiral topology.

The prepass is deliberately separate from normal reconstruction.  It reads
each selected native phase slab once, writes bounded probe blocks, solves the
compact graph, and caches the resulting offsets in the output scratch
directory.  Reconstruction then rereads the native cache normally and applies
the offsets to both integer-passage decoding and crossing-volume projection.
"""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _phase_sync_taper(value, low, high, width):
    """Raised-cosine reliability inside inclusive scalar bounds."""
    width = float(width)
    if width <= 0:
        return 1.0
    distance = min(float(value) - float(low), float(high) - float(value))
    fraction = float(np.clip((distance + 1.0) / (width + 1.0), 0.0, 1.0))
    return math.sin(fraction * (0.5 * math.pi)) ** 2


def _sample_phase_trilinear(phase, ijk):
    """Sample native ``[column_a, column_b, ray]`` phase at fractional ijk.

    ``ijk`` is expressed in full-resolution slab samples; transverse native
    columns are separated by ``column_stride`` and are converted by the
    caller.  Returns NaN when any interpolation corner is outside the field.
    """
    coordinates = np.asarray(ijk, dtype=np.float64)
    if coordinates.shape != (3,) or not np.isfinite(coordinates).all():
        return float("nan")
    lower = np.floor(coordinates).astype(np.int64)
    upper = lower + 1
    shape = np.asarray(phase.shape, dtype=np.int64)
    if (lower < 0).any() or (upper >= shape).any():
        return float("nan")
    fraction = coordinates - lower
    result = 0.0
    for da in (0, 1):
        wa = fraction[0] if da else 1.0 - fraction[0]
        for db in (0, 1):
            wb = fraction[1] if db else 1.0 - fraction[1]
            for dk in (0, 1):
                wk = fraction[2] if dk else 1.0 - fraction[2]
                result += (
                    wa * wb * wk
                    * float(phase[
                        lower[0] + da, lower[1] + db, lower[2] + dk
                    ])
                )
    return result


def _sample_cached_phase(phase, valid, frame, point_xyz, column_stride):
    """Return ``(phase, density, full-resolution ijk)`` at a world point."""
    slab = np.asarray(frame.to_slab(
        np.asarray(point_xyz, dtype=np.float64)[None]
    )[0], dtype=np.float64)
    native = np.array([
        slab[0] / float(column_stride),
        slab[1] / float(column_stride),
        slab[2],
    ], dtype=np.float64)
    value = _sample_phase_trilinear(phase, native)
    if not math.isfinite(value):
        return float("nan"), float("nan"), slab

    nearest = np.rint(slab).astype(np.int64)
    if ((nearest < 0).any()
            or (nearest >= np.asarray(valid.shape, dtype=np.int64)).any()
            or not bool(valid[tuple(nearest)])):
        return float("nan"), float("nan"), slab

    before = native.copy()
    after = native.copy()
    before[2] -= 1.0
    after[2] += 1.0
    previous = _sample_phase_trilinear(phase, before)
    following = _sample_phase_trilinear(phase, after)
    density = abs(following - previous) / (2.0 * float(frame.spacing))
    return value, density, slab


def _phase_sync_neighbor_table(
    seed_xyz, seed_winding, *, radius, max_neighbors, candidates_factor=4
):
    """Spatial neighbors, stratified only to avoid one dense seed sheet.

    Initial seed winding is used solely to limit how many of the nearest
    candidates can come from one sampling sheet.  It never creates an edge or
    specifies the offset on an edge; all retained edges are subsequently
    validated by world-space slab containment and phase agreement.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(seed_xyz, dtype=np.float64)
    winding = np.asarray(seed_winding)
    count = len(points)
    maximum = max(1, int(max_neighbors))
    query_count = min(count, maximum * max(2, int(candidates_factor)) + 1)
    output = np.full((count, maximum), -1, dtype=np.int32)
    per_sheet = max(2, int(math.ceil(maximum / 6)))
    tree = cKDTree(points)
    # A full 500k x ~100 query result is close to a GiB. Querying in blocks
    # bounds peak memory without changing deterministic neighbor selection.
    block_size = 1 << 15
    for begin in range(0, count, block_size):
        end = min(begin + block_size, count)
        distances, indices = tree.query(
            points[begin:end], k=query_count,
            distance_upper_bound=float(radius), workers=-1)
        if query_count == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        for local, node in enumerate(range(begin, end)):
            used = {}
            selected = []
            for distance, neighbor in zip(distances[local], indices[local]):
                neighbor = int(neighbor)
                if (neighbor == node or neighbor >= count
                        or not math.isfinite(float(distance))):
                    continue
                label = int(winding[neighbor])
                if used.get(label, 0) >= per_sheet:
                    continue
                selected.append(neighbor)
                used[label] = used.get(label, 0) + 1
                if len(selected) == maximum:
                    break
            output[node, :len(selected)] = selected
    return output


_PROBE_CONTEXT = None


def _initialize_probe_worker(context):
    global _PROBE_CONTEXT

    from vesuvius.neural_tracing.winding_models.infer_winding_volume import (
        _NativePhaseCacheReader,
    )

    _PROBE_CONTEXT = {
        **context,
        "seed_xyz": np.load(context["seed_xyz_path"], mmap_mode="r"),
        "seed_winding": np.load(context["seed_winding_path"], mmap_mode="r"),
        "global_index": np.load(context["global_index_path"], mmap_mode="r"),
        "neighbors": np.load(context["neighbors_path"], mmap_mode="r"),
        "reader": _NativePhaseCacheReader(context["phase_cache"]),
    }


def _write_probe_block(task):
    """Read one slab block and write compact directed overlap probes."""
    start, end, output_path = task
    context = _PROBE_CONTEXT
    if context is None:  # pragma: no cover - protects direct misuse
        raise RuntimeError("phase-overlap probe worker was not initialized")

    seeds = context["seed_xyz"]
    seed_winding = context["seed_winding"]
    global_index = context["global_index"]
    neighbors = context["neighbors"]
    reader = context["reader"]
    column_stride = int(context["column_stride"])
    transverse = int(context["transverse_size"])
    ray_length = int(context["ray_length"])
    transverse_margin = float(context["transverse_margin"])
    ray_margin = float(context["ray_margin"])
    taper_width = float(context["taper_width"])
    min_density = float(context["min_density"])

    reference = np.full(end - start, np.nan, dtype=np.float32)
    anchor = np.full(end - start, np.nan, dtype=np.float32)
    edge_u = []
    edge_v = []
    sampled_phase = []
    edge_weight = []

    # The production decoder's anchor after align-corners interpolation is
    # invariant to transverse crop/upsampling and lies at this physical
    # native-grid coordinate.
    native_columns = transverse // column_stride
    decoded_columns = (native_columns - 1) * column_stride + 1
    decoded_center = int(round((decoded_columns - 1) / 2.0))
    decoded_ray_anchor = int(round((ray_length - 1) / 2.0))

    for node in range(start, end):
        cached = reader.read(int(global_index[node]))
        if cached is None:
            continue
        phase, valid, frame = cached
        own_phase, own_density, _ = _sample_cached_phase(
            phase, valid, frame, seeds[node], column_stride)
        reference[node - start] = own_phase
        anchor_value = _sample_phase_trilinear(phase, np.array([
            decoded_center / float(column_stride),
            decoded_center / float(column_stride),
            float(decoded_ray_anchor),
        ]))
        anchor[node - start] = anchor_value
        if not (math.isfinite(own_phase) and math.isfinite(anchor_value)
                and math.isfinite(own_density) and own_density >= min_density):
            continue

        for probe_owner in neighbors[node]:
            probe_owner = int(probe_owner)
            if probe_owner < 0:
                continue
            value, density, slab = _sample_cached_phase(
                phase, valid, frame, seeds[probe_owner], column_stride)
            if not (math.isfinite(value) and math.isfinite(density)
                    and density >= min_density):
                continue
            if not (
                transverse_margin <= slab[0] <= transverse - 1 - transverse_margin
                and transverse_margin <= slab[1] <= transverse - 1 - transverse_margin
                and ray_margin <= slab[2] <= ray_length - 1 - ray_margin
            ):
                continue
            weight = (
                _phase_sync_taper(
                    slab[0], transverse_margin,
                    transverse - 1 - transverse_margin, taper_width)
                * _phase_sync_taper(
                    slab[1], transverse_margin,
                    transverse - 1 - transverse_margin, taper_width)
                * _phase_sync_taper(
                    slab[2], ray_margin,
                    ray_length - 1 - ray_margin, taper_width)
                * min(1.0, density / max(2.0 * min_density, 1e-6))
            )
            if weight <= 0:
                continue
            edge_u.append(probe_owner)
            edge_v.append(node)
            sampled_phase.append(value)
            edge_weight.append(weight)

    np.savez_compressed(
        output_path,
        start=np.int64(start),
        reference=reference,
        anchor=anchor,
        edge_u=np.asarray(edge_u, dtype=np.int32),
        edge_v=np.asarray(edge_v, dtype=np.int32),
        sampled_phase=np.asarray(sampled_phase, dtype=np.float32),
        edge_weight=np.asarray(edge_weight, dtype=np.float32),
        seed_winding=np.asarray(seed_winding[start:end], dtype=np.int16),
    )
    return end - start, len(edge_u)


def _solve_phase_overlap_graph(
    prior, edge_u, edge_v, edge_delta, edge_weight, *, iterations=5,
    huber=0.25, prior_weight=0.02, prior_huber=0.5,
    max_correction=4.0,
):
    """Robust matrix-free IRLS solve of ``offset[v]-offset[u]=delta``."""
    from scipy.sparse.linalg import LinearOperator, cg

    prior = np.asarray(prior, dtype=np.float64)
    edge_u = np.asarray(edge_u, dtype=np.int64)
    edge_v = np.asarray(edge_v, dtype=np.int64)
    edge_delta = np.asarray(edge_delta, dtype=np.float64)
    base_weight = np.asarray(edge_weight, dtype=np.float64)
    nodes = len(prior)
    edges = len(edge_u)
    if not edges:
        return prior.copy(), {
            "edges": 0,
            "supported_nodes": 0,
            "edge_residual_median_abs": None,
            "edge_residual_p95_abs": None,
        }

    finite = (
        np.isfinite(edge_delta) & np.isfinite(base_weight) & (base_weight > 0)
        & (edge_u >= 0) & (edge_u < nodes)
        & (edge_v >= 0) & (edge_v < nodes) & (edge_u != edge_v)
    )
    edge_u, edge_v, edge_delta, base_weight = (
        value[finite] for value in
        (edge_u, edge_v, edge_delta, base_weight))
    edges = len(edge_u)
    if not edges:
        return prior.copy(), {
            "edges": 0,
            "supported_nodes": 0,
            "edge_residual_median_abs": None,
            "edge_residual_p95_abs": None,
        }

    target = edge_delta - (prior[edge_v] - prior[edge_u])
    correction = np.zeros(nodes, dtype=np.float64)
    robust_edge_weight = base_weight.copy()
    robust_prior_weight = np.full(nodes, float(prior_weight), dtype=np.float64)

    for _ in range(max(1, int(iterations))):
        def matvec(values):
            return (
                np.bincount(
                    edge_u,
                    robust_edge_weight * (values[edge_u] - values[edge_v]),
                    minlength=nodes)
                + np.bincount(
                    edge_v,
                    robust_edge_weight * (values[edge_v] - values[edge_u]),
                    minlength=nodes)
                + robust_prior_weight * values
            )

        operator = LinearOperator(
            (nodes, nodes), matvec=matvec, dtype=np.float64)
        diagonal = (
            np.bincount(edge_u, robust_edge_weight, minlength=nodes)
            + np.bincount(edge_v, robust_edge_weight, minlength=nodes)
            + robust_prior_weight)
        preconditioner = LinearOperator(
            (nodes, nodes),
            matvec=lambda values: values / np.maximum(diagonal, 1e-12),
            dtype=np.float64)
        weighted_target = robust_edge_weight * target
        rhs = (
            np.bincount(edge_v, weighted_target, minlength=nodes)
            - np.bincount(edge_u, weighted_target, minlength=nodes))
        correction, info = cg(
            operator, rhs, x0=correction, M=preconditioner,
            rtol=1e-5, atol=1e-8, maxiter=200)
        if info < 0:  # pragma: no cover - scipy input/internal failure
            raise RuntimeError(f"phase overlap graph solve failed: cg={info}")
        np.clip(
            correction, -float(max_correction), float(max_correction),
            out=correction)
        residual = (
            correction[edge_v] - correction[edge_u] - target)
        robust_edge_weight = base_weight * np.minimum(
            1.0, float(huber) / np.maximum(np.abs(residual), 1e-12))
        robust_prior_weight = float(prior_weight) * np.minimum(
            1.0,
            float(prior_huber) / np.maximum(np.abs(correction), 1e-12))

    # Edge equations do not set the additive gauge; the weak robust prior in
    # the augmented solve fixes it independently in each connected component.
    residual = (
        (prior[edge_v] + correction[edge_v])
        - (prior[edge_u] + correction[edge_u]) - edge_delta)
    degree = (
        np.bincount(edge_u, minlength=nodes)
        + np.bincount(edge_v, minlength=nodes))
    absolute = np.abs(residual)
    stats = {
        "edges": int(edges),
        "supported_nodes": int(np.count_nonzero(degree)),
        "edge_residual_median_abs": float(np.median(absolute)),
        "edge_residual_p95_abs": float(np.quantile(absolute, 0.95)),
        "correction_median_abs": float(np.median(np.abs(correction))),
        "correction_p95_abs": float(np.quantile(np.abs(correction), 0.95)),
        "correction_max_abs": float(np.max(np.abs(correction))),
    }
    return prior + correction, stats


def _sync_fingerprint(rays, options):
    import hashlib

    digest = hashlib.sha256()
    for key in ("global_index", "seed_winding", "seed_xyz"):
        digest.update(np.ascontiguousarray(rays[key]).view(np.uint8))
    digest.update(json.dumps(options, sort_keys=True).encode("utf8"))
    return digest.hexdigest()


def build_phase_overlap_offsets(args, rays, model_cfg, scratch):
    """Run or reuse the topology-free overlap synchronization prepass."""
    import multiprocessing as mp
    from tqdm import tqdm

    scratch = Path(scratch)
    sync_dir = scratch / "phase_overlap_sync"
    sync_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "phase_cache": str(Path(args.phase_cache).resolve()),
        "radius": float(args.phase_sync_radius),
        "neighbors": int(args.phase_sync_neighbors),
        "transverse_margin": int(args.phase_sync_transverse_margin),
        "ray_margin": int(args.phase_sync_ray_margin),
        "taper_width": int(args.phase_sync_taper),
        "min_density": float(args.phase_sync_min_density),
        "iterations": int(args.phase_sync_iterations),
        "huber": float(args.phase_sync_huber),
        "prior_weight": float(args.phase_sync_prior_weight),
        "prior_huber": float(args.phase_sync_prior_huber),
        "max_correction": float(args.phase_sync_max_correction),
    }
    fingerprint = _sync_fingerprint(rays, options)
    solution_path = sync_dir / "offsets.npz"
    if solution_path.is_file() and not bool(args.phase_sync_recompute):
        cached = np.load(solution_path, allow_pickle=False)
        if str(cached["fingerprint"].item()) == fingerprint:
            stats = json.loads(str(cached["stats_json"].item()))
            print(
                f"[phase-sync] reusing {len(cached['correction']):,} overlap "
                f"offsets from {solution_path}", flush=True)
            return np.asarray(cached["correction"], dtype=np.float32), stats

    seed_xyz = np.asarray(rays["seed_xyz"], dtype=np.float32)
    seed_winding = np.asarray(rays["seed_winding"], dtype=np.int16)
    global_index = np.asarray(rays["global_index"], dtype=np.int64)
    print(
        f"[phase-sync] finding up to {args.phase_sync_neighbors} spatial "
        f"neighbors within {args.phase_sync_radius:g} voxels for "
        f"{len(seed_xyz):,} slabs", flush=True)
    neighbors = _phase_sync_neighbor_table(
        seed_xyz, seed_winding, radius=args.phase_sync_radius,
        max_neighbors=args.phase_sync_neighbors)

    array_paths = {}
    for name, value in (
        ("seed_xyz", seed_xyz), ("seed_winding", seed_winding),
        ("global_index", global_index), ("neighbors", neighbors),
    ):
        path = sync_dir / f"{name}.npy"
        np.save(path, value)
        array_paths[f"{name}_path"] = str(path)

    for stale in sync_dir.glob("probe_*.npz"):
        stale.unlink()
    block = max(1, int(args.phase_sync_block_size))
    tasks = []
    for start in range(0, len(seed_xyz), block):
        end = min(start + block, len(seed_xyz))
        tasks.append((start, end, str(sync_dir / f"probe_{start:09d}.npz")))
    workers = min(
        int(args.phase_sync_workers or min(16, os.cpu_count() or 1)),
        max(1, len(tasks)))
    context = {
        **array_paths,
        "phase_cache": str(args.phase_cache),
        "column_stride": int(model_cfg.get("column_stride", 4)),
        "transverse_size": int(model_cfg.get("transverse_size", 128)),
        "ray_length": int(model_cfg.get("ray_length", 384)),
        "transverse_margin": int(args.phase_sync_transverse_margin),
        "ray_margin": int(args.phase_sync_ray_margin),
        "taper_width": int(args.phase_sync_taper),
        "min_density": float(args.phase_sync_min_density),
    }
    print(
        f"[phase-sync] sampling overlap probes with {workers} worker(s); "
        "this is one additional read of each selected native phase slab",
        flush=True)
    if workers <= 1:
        _initialize_probe_worker(context)
        results = map(_write_probe_block, tasks)
        for _ in tqdm(
            results, total=len(tasks), desc="phase overlap probes",
            unit="block"):
            pass
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_initialize_probe_worker,
            initargs=(context,), mp_context=mp.get_context("spawn")) as pool:
            for _ in tqdm(
                pool.map(_write_probe_block, tasks, chunksize=1),
                total=len(tasks), desc="phase overlap probes", unit="block"):
                pass

    reference = np.full(len(seed_xyz), np.nan, dtype=np.float32)
    anchor = np.full(len(seed_xyz), np.nan, dtype=np.float32)
    edge_parts = [[], [], [], []]
    for _start, _end, path in tasks:
        values = np.load(path, allow_pickle=False)
        start = int(values["start"])
        end = start + len(values["reference"])
        reference[start:end] = values["reference"]
        anchor[start:end] = values["anchor"]
        for destination, name in zip(
            edge_parts, ("edge_u", "edge_v", "sampled_phase", "edge_weight")
        ):
            destination.append(np.asarray(values[name]))
    edge_u, edge_v, sampled_phase, edge_weight = (
        np.concatenate(parts) if parts else np.empty(0)
        for parts in edge_parts)
    valid_nodes = np.isfinite(reference) & np.isfinite(anchor)
    valid_edges = (
        valid_nodes[edge_u.astype(np.int64)]
        & valid_nodes[edge_v.astype(np.int64)]
        & np.isfinite(sampled_phase) & np.isfinite(edge_weight)
        & (edge_weight > 0))
    edge_u = edge_u[valid_edges].astype(np.int64, copy=False)
    edge_v = edge_v[valid_edges].astype(np.int64, copy=False)
    edge_weight = edge_weight[valid_edges].astype(np.float64, copy=False)
    # Directed probe u->v samples phase_v at seed_u.  Equality of the two
    # global phase fields at that world point gives offset_v-offset_u below.
    edge_delta = (
        reference[edge_u].astype(np.float64)
        - sampled_phase[valid_edges].astype(np.float64))
    prior = seed_winding.astype(np.float64) - anchor.astype(np.float64)
    prior[~valid_nodes] = 0.0
    solved, stats = _solve_phase_overlap_graph(
        prior, edge_u, edge_v, edge_delta, edge_weight,
        iterations=args.phase_sync_iterations, huber=args.phase_sync_huber,
        prior_weight=args.phase_sync_prior_weight,
        prior_huber=args.phase_sync_prior_huber,
        max_correction=args.phase_sync_max_correction)
    correction = solved - prior
    correction[~valid_nodes] = 0.0
    stats.update({
        "mode": "world_overlap_phase_synchronization",
        "nodes": int(len(seed_xyz)),
        "valid_nodes": int(np.count_nonzero(valid_nodes)),
        "options": options,
        "fingerprint": fingerprint,
    })
    np.savez_compressed(
        solution_path, correction=correction.astype(np.float32),
        global_index=global_index, fingerprint=np.asarray(fingerprint),
        stats_json=np.asarray(json.dumps(stats, sort_keys=True)))
    print(
        f"[phase-sync] solved {stats['edges']:,} constraints over "
        f"{stats['supported_nodes']:,}/{len(seed_xyz):,} supported slabs; "
        f"median |edge residual|={stats['edge_residual_median_abs']:.3f}, "
        f"p95={stats['edge_residual_p95_abs']:.3f} windings; "
        f"p95 |correction|={stats['correction_p95_abs']:.3f}, "
        f"max={stats['correction_max_abs']:.3f}", flush=True)
    if stats["supported_nodes"] < 0.9 * len(seed_xyz):
        print(
            "[warning] phase-overlap graph supports fewer than 90% of selected "
            "slabs; unsupported slabs retain legacy seed registration",
            flush=True)
    if stats["correction_max_abs"] >= 0.95 * float(args.phase_sync_max_correction):
        print(
            "[warning] phase synchronization reached its correction safety "
            "limit; inspect output synchronization diagnostics before raising "
            "--phase-sync-max-correction", flush=True)
    return correction.astype(np.float32), stats
