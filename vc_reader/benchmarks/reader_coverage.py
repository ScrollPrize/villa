"""Untimed reader workload placement and exact storage-chunk coverage."""
from __future__ import annotations

import itertools
import numpy as np


def box_chunks(origin, shape, chunks):
    lo = np.asarray(origin, dtype=np.int64) // chunks
    hi = (np.asarray(origin, dtype=np.int64) + shape - 1) // chunks
    return set(itertools.product(*(range(int(a), int(b) + 1) for a, b in zip(lo, hi))))


def sampled_chunks(coords, origin, chunks, volume_shape):
    """Chunks containing nonzero-weight trilinear neighbors of actual samples.

    Coordinates are relative to an integer origin, so integer conversion happens
    before adding the global position. Linear chunk IDs keep temporary storage
    bounded to one integer per sample instead of Python tuples per voxel.
    """
    coords = np.asarray(coords).reshape(3, -1)
    origin = np.asarray(origin, dtype=np.int64)[:, None]
    chunks = np.asarray(chunks, dtype=np.int64)[:, None]
    low = (np.floor(coords).astype(np.int64) + origin) // chunks
    high = (np.ceil(coords).astype(np.int64) + origin) // chunks
    counts = (np.asarray(volume_shape, dtype=np.int64) + chunks[:, 0] - 1) // chunks[:, 0]
    ids = set()
    for choice in itertools.product((0, 1), repeat=3):
        z, y, x = [high[i] if choice[i] else low[i] for i in range(3)]
        ids.update(np.unique((z * counts[1] + y) * counts[2] + x).tolist())
    return {(int(i // (counts[1] * counts[2])), int(i // counts[2] % counts[1]),
             int(i % counts[2])) for i in ids}


def spread_workload(name, templates, reader, args, sampling_box, coordinate_box):
    """Sweep template geometry along Z, adding nearby XY columns as needed.

    Translations are whole storage chunks. This preserves interpolation phases
    and lets us compute exact template coverage once, rather than visiting every
    output voxel again for every placement. Surface coordinate arrays are shared.
    Every enclosing ROI must be nonzero, so accepted samples cannot hit padding.
    """
    chunks = np.asarray(reader.chunks, dtype=np.int64)
    volume_shape = np.asarray(reader.shape, dtype=np.int64)
    if int(np.prod((volume_shape + chunks - 1) // chunks)) < args.min_unique_chunks:
        raise ValueError(f"{name}: input has fewer than {args.min_unique_chunks} storage chunks")
    rng = np.random.default_rng(args.seed + 1000 +
                                ("ml_batch", "plane", "surface", "slab", "rotated_box").index(name))
    plans = []
    for batch in templates:
        planned = []
        for grid in batch:
            if name == "surface":
                origin, shape, coords = coordinate_box(np.moveaxis(grid[2], -1, 0), grid[1])
            elif name in ("plane", "rotated_box"):
                origin, shape, coords = sampling_box(grid)
            else:
                origin, shape, coords = grid[1], grid[0], None
            origin, shape = np.asarray(origin, dtype=np.int64), np.asarray(shape, dtype=np.int64)
            touched = (box_chunks(origin, shape, chunks) if coords is None else
                       sampled_chunks(coords, origin, chunks, volume_shape))
            enclosing = box_chunks(origin, shape, chunks)
            # ceil(-origin / chunks), calculated entirely with integer arithmetic.
            lower = -(origin // chunks)
            upper = (volume_shape - origin - shape) // chunks
            radius = args.xy_radius // chunks[1:]
            lower[1:] = np.maximum(lower[1:], -radius)
            upper[1:] = np.minimum(upper[1:], radius)
            if np.any(lower > upper):
                raise ValueError(f"{name}: no in-bounds placement for a template")
            planned.append(dict(grid=grid, origin=origin, shape=shape, touched=touched,
                                enclosing=enclosing, lower=lower, upper=upper, cursor=0))
        plans.append(planned)

    seen, enclosing_seen = set(), set()
    requests, descriptions, counts, roi_counts = [], [], [], []
    rejected = 0
    attempts = 0
    candidate_requests = 0
    limit = max(1000, args.min_unique_chunks * 20)
    while len(seen) < args.min_unique_chunks or len(requests) < args.crops:
        if attempts >= limit:
            raise ValueError(f"{name}: only {len(seen)} nonzero-region chunks after {attempts} candidates; "
                             "increase --xy-radius, choose another --start/--surface, or use a larger array")
        template_index = candidate_requests % len(plans)
        candidate_requests += 1
        batch, batch_chunks, batch_roi = [], set(), set()
        for plan in plans[template_index]:
            while True:
                if attempts >= limit:
                    raise ValueError(f"{name}: only {len(seen)} nonzero-region chunks after {attempts} candidates; "
                                     "increase --xy-radius or choose another --start/--surface")
                attempts += 1
                lower, upper = plan['lower'], plan['upper']
                nz = int(upper[0] - lower[0] + 1)
                if plan['cursor'] % nz == 0:
                    plan['z_order'] = rng.permutation(np.arange(lower[0], upper[0] + 1))
                    plan['xy'] = (np.clip(np.zeros(2, dtype=np.int64), lower[1:], upper[1:])
                                  if plan['cursor'] == 0 else rng.integers(lower[1:], upper[1:] + 1))
                shift = np.array([plan['z_order'][plan['cursor'] % nz], *plan['xy']], dtype=np.int64)
                plan['cursor'] += 1
                delta = shift * chunks
                # Full enclosing-region check is stronger than checking a coarse
                # mask or a few points, and is excluded from all timings.
                data = reader.read_region(tuple((plan['origin'] + delta).tolist()), tuple(plan['shape'].tolist()))
                nonzero = bool(np.all(data))
                del data
                if nonzero:
                    break
                rejected += 1
                if attempts % 100 == 0:
                    print(f"{name}: {len(seen)}/{args.min_unique_chunks} chunks; "
                          f"{rejected} padding placements rejected", flush=True)
            batch_chunks.update(tuple(int(a + b) for a, b in zip(index, shift)) for index in plan['touched'])
            batch_roi.update(tuple(int(a + b) for a, b in zip(index, shift)) for index in plan['enclosing'])
            grid = plan['grid']
            batch.append((grid[0], (np.asarray(grid[1]) + delta).tolist(), grid[2]))
        if batch_chunks - seen or len(requests) < args.crops:
            requests.append(batch)
            descriptions.append(dict(template=template_index, coordinate_origin=batch[0][1], shape=batch[0][0]))
            counts.append(len(batch_chunks))
            roi_counts.append(len(batch_roi))
            seen.update(batch_chunks)
            enclosing_seen.update(batch_roi)
            if len(requests) % 100 == 0:
                print(f"{name}: planning {len(requests)} requests, {len(seen)}/{args.min_unique_chunks} unique chunks "
                      f"({rejected} padding placements rejected)", flush=True)
    coverage = dict(minimum_unique_chunks=args.min_unique_chunks, unique_chunks=len(seen),
                    enclosing_unique_chunks=len(enclosing_seen), chunks_per_request=counts,
                    enclosing_chunks_per_request=roi_counts, requests=len(requests),
                    candidate_regions=attempts, candidate_requests=candidate_requests, rejected_padding_regions=rejected,
                    nonzero_check="every voxel in each enclosing ROI is nonzero; untimed CPU preflight",
                    definition="union of storage chunks containing requested voxels or nonzero-weight trilinear neighbors",
                    placement="seeded Z sweeps with whole-chunk XY translations within xy_radius; geometry reused")
    print(f"{name}: planned {len(requests)} requests covering {len(seen)} unique chunks", flush=True)
    return requests, coverage, descriptions
