"""Merge per-ray winding crossing observations into one global winding field.

Input is an ``infer_winding_volume.py`` output archive: billions of crossing
points (``points/{xyz,winding,prob}``), the CSR strip layout
(``strips/{offsets,slab}``), and one ray per slab (``rays/*``).  Each point is
a direct sample ``W(x) = w`` of the continuous global winding field that the
per-column monotone phase fields observe locally.  This script reconciles all
of them into one smooth scalar volume by gradient descent on a coarse voxel
grid, and writes the result as a sparse OME-Zarr v2 pyramid (levels 0..5,
uint16 ``round(W * 256)``, 0 = invalid) that VC3D can stream.

Pipeline (every pass persists its product to ``--scratch-dir`` and is skipped
on ``--resume`` when the product already exists):

  prescan       strips -> block plan (strip/point/slab ranges per block)
  patch raster  verified tifxyz sheet patches -> ds4 patch-id grid
  registration  stream points once; per-(slab, patch) winding stats; robust
                IRLS+CG solve of per-slab integer label offsets ("two rays
                hitting the same patch see the same winding")
  scatter       stream points again with corrected labels -> per-voxel
                sufficient statistics at ds8 (value/direction/density) and
                ds4 (value)
  ds8 solve     initialize from the ratio + pull-push fill; Adam on the full
                ds8 grid against data + smoothness + eikonal + monotonicity
  ds4 refine    trilinear upsample; Adam with z-slab gradient accumulation
  write         OME-Zarr v2 levels 0..5; level 2 = solved ds4 field, 3..5 by
                valid-child mean pooling, 1 and 0 by tiled trilinear upsample

Deliberately not used: the spiral fit checkpoint and the umbilicus.  Ray
directions orient the field, crossing spacing along strips sets the eikonal
gradient-magnitude target, and the points set its values.

A single-valued winding field on a connected spiral necessarily contains one
thin helical jump surface per turn (the angular branch cut).  The robust data
term lets that seam localize itself; it is reported, not "fixed".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CHUNK = 128
LEVELS = 6
BAND_ALIGN = 32
TRANSVERSE_HALF = 63.5  # (transverse_size - 1) / 2 for the 128-column slabs
LEVEL_HALF_LIFE = 2.0   # matches infer_winding_volume._VOTE_LEVEL_HALF_LIFE
MAX_LEVEL = 64


def _log(message):
    print(f"[merge {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _write_json_atomic(path, payload):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=1)
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# geometry


@dataclass(frozen=True)
class BandGeometry:
    """Solve band: a z slab of the full volume, 32-aligned so that every
    pyramid level's embedding offset ``z_lo >> level`` is exact."""

    full_shape: tuple  # (Z, Y, X) at full resolution
    z_lo: int
    z_hi: int  # exclusive

    def band_shape(self, ds):
        _, full_y, full_x = self.full_shape
        return (
            (self.z_hi - self.z_lo) // ds,
            -(-full_y // ds),
            -(-full_x // ds),
        )

    def level_shape(self, level):
        return tuple(-(-size // (1 << level)) for size in self.full_shape)

    def origin(self, ds):
        return self.z_lo // ds


def make_band(full_shape, z_range, margin):
    full_z = int(full_shape[0])
    z_lo = max(0, (int(z_range[0]) - int(margin)) // BAND_ALIGN * BAND_ALIGN)
    z_hi = min(
        full_z // BAND_ALIGN * BAND_ALIGN,
        -(-(int(z_range[1]) + int(margin)) // BAND_ALIGN) * BAND_ALIGN,
    )
    if z_hi <= z_lo:
        raise ValueError(f"empty solve band from z_range={z_range}")
    return BandGeometry(tuple(int(s) for s in full_shape), z_lo, z_hi)


def reconstruct_frames(direction_xyz):
    """Vectorized copy of VolumeSlabExtractor.slab_frame's transverse axes.

    The frame is deterministic given only the ray direction, which is what
    makes per-point transverse coordinates recoverable without the (dropped)
    strip-to-column bookkeeping.
    """
    direction = np.asarray(direction_xyz, dtype=np.float64)
    direction = direction / np.linalg.norm(direction, axis=1, keepdims=True)
    reference = np.zeros_like(direction)
    reference[np.arange(len(direction)), np.argmin(np.abs(direction), axis=1)] = 1.0
    axis_a = np.cross(reference, direction)
    axis_a /= np.linalg.norm(axis_a, axis=1, keepdims=True)
    axis_b = np.cross(direction, axis_a)
    return axis_a.astype(np.float32), axis_b.astype(np.float32)


# ---------------------------------------------------------------------------
# input archive + block plan


class InferenceArchive:
    def __init__(self, path):
        import zarr

        self.path = Path(path)
        self.group = zarr.open_group(str(path), mode="r")
        self.attrs = dict(self.group.attrs)
        self.points_xyz = self.group["points/xyz"]
        self.points_winding = self.group["points/winding"]
        self.points_prob = self.group["points/prob"]
        self.strips_offsets = self.group["strips/offsets"]
        self.strips_slab = self.group["strips/slab"]
        self.num_points = int(self.points_xyz.shape[0])
        self.num_strips = int(self.strips_slab.shape[0])
        self.full_shape = tuple(int(s) for s in self.group["winding"].shape)
        self.seed_xyz = np.asarray(self.group["rays/seed_xyz"][:], dtype=np.float32)
        self.direction_xyz = np.asarray(
            self.group["rays/direction_xyz"][:], dtype=np.float32)
        self.seed_winding = np.asarray(
            self.group["rays/seed_winding"][:], dtype=np.int32)
        self.num_slabs = len(self.seed_xyz)
        self.axis_a, self.axis_b = reconstruct_frames(self.direction_xyz)
        self.strip_chunk = int(self.strips_slab.chunks[0])


def build_block_plan(archive, scratch, strip_block_chunks):
    """One full pass over strips/slab: block boundaries + slab ranges.

    Also validates the (relied-upon) invariant that strips are grouped by
    nondecreasing slab index, which makes z-crop block skipping sound.
    """
    plan_path = scratch / "plan.npz"
    if plan_path.exists():
        plan = np.load(plan_path)
        if int(plan["num_strips"]) == archive.num_strips:
            return {key: plan[key] for key in plan.files}
    block_strips = archive.strip_chunk * int(strip_block_chunks)
    starts = np.arange(0, archive.num_strips, block_strips, dtype=np.int64)
    ends = np.minimum(starts + block_strips, archive.num_strips)
    slab_lo = np.empty(len(starts), dtype=np.int64)
    slab_hi = np.empty(len(starts), dtype=np.int64)
    previous_max = -1
    for index, (lo, hi) in enumerate(zip(starts, ends)):
        slab = archive.strips_slab[lo:hi]
        if len(slab) > 1 and np.any(np.diff(slab) < 0):
            raise RuntimeError("strips/slab is not nondecreasing")
        if slab[0] < previous_max:
            raise RuntimeError("strips/slab is not nondecreasing across blocks")
        previous_max = int(slab[-1])
        slab_lo[index] = slab[0]
        slab_hi[index] = slab[-1]
    point_lo = np.array(
        [int(archive.strips_offsets[int(s)]) for s in starts], dtype=np.int64)
    point_hi = np.append(point_lo[1:], archive.num_points)
    plan = {
        "num_strips": np.int64(archive.num_strips),
        "strip_lo": starts, "strip_hi": ends,
        "point_lo": point_lo, "point_hi": point_hi,
        "slab_lo": slab_lo, "slab_hi": slab_hi,
    }
    np.savez(plan_path, **plan)
    return plan


def select_blocks(plan, archive, band, limit_blocks):
    """Blocks whose slabs could produce points inside the band (seed z plus
    the worst-case ray half-length keeps this conservative)."""
    seed_z = archive.seed_xyz[:, 2]
    reach = 384.0  # ray_length; points cannot land further from the seed
    keep = []
    for index in range(len(plan["strip_lo"])):
        slab_range = slice(int(plan["slab_lo"][index]),
                           int(plan["slab_hi"][index]) + 1)
        z = seed_z[slab_range]
        if z.size and (z.max() >= band.z_lo - reach) and (
                z.min() < band.z_hi + reach):
            keep.append(index)
    if limit_blocks:
        keep = keep[: int(limit_blocks)]
    return keep


def stream_blocks(archive, plan, block_indices, *, read_prob, start=0):
    """Yield raw per-block numpy payloads with a one-block prefetch."""

    def read(index):
        lo = int(plan["strip_lo"][index])
        hi = int(plan["strip_hi"][index])
        offsets = np.asarray(archive.strips_offsets[lo:hi + 1], dtype=np.int64)
        slab = np.asarray(archive.strips_slab[lo:hi], dtype=np.int64)
        p0, p1 = int(offsets[0]), int(offsets[-1])
        payload = {
            "index": index,
            "offsets": offsets,
            "slab": slab,
            "xyz": np.asarray(archive.points_xyz[p0:p1], dtype=np.float32),
            "winding": np.asarray(
                archive.points_winding[p0:p1], dtype=np.int32),
        }
        payload["prob"] = (
            np.asarray(archive.points_prob[p0:p1], dtype=np.uint8)
            if read_prob else None)
        return payload

    pending = block_indices[start:]
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        upcoming = pool.submit(read, pending[0])
        for position in range(len(pending)):
            payload = upcoming.result()
            if position + 1 < len(pending):
                upcoming = pool.submit(read, pending[position + 1])
            yield payload


# ---------------------------------------------------------------------------
# GPU join


class RaysGpu:
    def __init__(self, archive, device):
        import torch

        as_t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
        self.seed = as_t(archive.seed_xyz, torch.float32)
        self.direction = as_t(archive.direction_xyz, torch.float32)
        self.axis_a = as_t(archive.axis_a, torch.float32)
        self.axis_b = as_t(archive.axis_b, torch.float32)
        self.seed_winding = as_t(archive.seed_winding, torch.float32)


def raised_cosine(distance, half, width):
    import torch

    if width <= 0:
        return torch.ones_like(distance)
    fraction = ((half - distance) / width).clamp(0.0, 1.0)
    return torch.sin(fraction * (0.5 * math.pi)) ** 2


def join_block(raw, rays, cfg, device):
    """Per-point weights, densities and slab context for one block."""
    import torch

    offsets = torch.as_tensor(raw["offsets"], device=device)
    offsets = offsets - offsets[0]
    lengths = offsets[1:] - offsets[:-1]
    slab_strip = torch.as_tensor(raw["slab"], device=device)
    strip_of_point = torch.repeat_interleave(
        torch.arange(len(lengths), device=device), lengths)
    slab = slab_strip[strip_of_point]
    xyz = torch.as_tensor(raw["xyz"], device=device)
    winding = torch.as_tensor(raw["winding"], device=device).float()

    delta = xyz - rays.seed[slab]
    coord_a = (delta * rays.axis_a[slab]).sum(dim=1)
    coord_b = (delta * rays.axis_b[slab]).sum(dim=1)
    taper = (
        raised_cosine(coord_a.abs(), TRANSVERSE_HALF, cfg.taper_width)
        * raised_cosine(coord_b.abs(), TRANSVERSE_HALF, cfg.taper_width))
    level = (winding - rays.seed_winding[slab]).abs().clamp_(max=MAX_LEVEL)
    omega = taper * torch.pow(
        torch.tensor(0.5, device=device), level / LEVEL_HALF_LIFE)
    if raw["prob"] is not None:
        omega = omega * (
            torch.as_tensor(raw["prob"], device=device).float() / 255.0)

    count = len(xyz)
    same = strip_of_point[1:] == strip_of_point[:-1]
    gap = (xyz[1:] - xyz[:-1]).norm(dim=1)
    unit_step = (winding[1:] - winding[:-1]).abs() == 1.0
    pair_valid = same & unit_step & (gap > 1e-3)
    pair_index = torch.nonzero(pair_valid, as_tuple=False).squeeze(1)
    rho_pair = 1.0 / gap[pair_index]
    rho_sum = torch.zeros(count, device=device)
    rho_count = torch.zeros(count, device=device)
    ones = torch.ones_like(rho_pair)
    for endpoint in (pair_index, pair_index + 1):
        rho_sum.index_add_(0, endpoint, rho_pair)
        rho_count.index_add_(0, endpoint, ones)
    rho = rho_sum / rho_count.clamp(min=1.0)
    return {
        "xyz": xyz, "winding": winding, "slab": slab, "omega": omega,
        "rho": rho, "rho_valid": rho_count > 0,
        "direction": rays.direction[slab],
    }


def voxelize(xyz, band, ds, device):
    """Cell-centered voxel indices at downsample ``ds`` inside the band."""
    import torch

    zyx = xyz[:, [2, 1, 0]]
    cells = torch.floor(zyx / ds).long()
    shape = band.band_shape(ds)
    cells[:, 0] -= band.origin(ds)
    inside = (
        (cells[:, 0] >= 0) & (cells[:, 0] < shape[0])
        & (cells[:, 1] >= 0) & (cells[:, 1] < shape[1])
        & (cells[:, 2] >= 0) & (cells[:, 2] < shape[2]))
    flat = (cells[:, 0] * shape[1] + cells[:, 1]) * shape[2] + cells[:, 2]
    return flat, inside


# ---------------------------------------------------------------------------
# pass R1: patch rasterization


def load_patch_metas(patches_dir, band, scratch):
    cache = scratch / "patch_metas.json"
    if cache.exists():
        entries = json.loads(cache.read_text())
        if entries.get("patches_dir") == str(patches_dir):
            return entries["names"]

    def read_meta(entry):
        try:
            meta = json.loads((entry / "meta.json").read_text())
            bbox = meta.get("bbox")
            return entry.name, bbox
        except (OSError, ValueError, KeyError):
            return entry.name, None

    directories = sorted(
        entry for entry in Path(patches_dir).iterdir() if entry.is_dir())
    names = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for name, bbox in pool.map(read_meta, directories):
            if bbox is None:
                continue
            z_min, z_max = float(bbox[0][2]), float(bbox[1][2])
            if z_max >= band.z_lo and z_min < band.z_hi:
                names.append(name)
    _write_json_atomic(
        cache, {"patches_dir": str(patches_dir), "names": names})
    return names


def _patch_samples(patch_dir, erode, subdiv):
    """World-space surface samples of one eroded tifxyz patch."""
    import tifffile
    from scipy.ndimage import binary_erosion

    grids = [tifffile.imread(patch_dir / f"{axis}.tif") for axis in "xyz"]
    xyz = np.stack(grids, axis=-1).astype(np.float32)
    valid = np.all(xyz != -1, axis=-1)
    if erode > 0:
        valid = binary_erosion(
            valid, structure=np.ones((3, 3), dtype=bool), iterations=erode)
    quad = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    rows, cols = np.nonzero(quad)
    if not len(rows):
        return None, None
    c00 = xyz[rows, cols]
    c01 = xyz[rows, cols + 1]
    c10 = xyz[rows + 1, cols]
    c11 = xyz[rows + 1, cols + 1]
    steps = (np.arange(subdiv, dtype=np.float32) + 0.5) / subdiv
    u, v = np.meshgrid(steps, steps, indexing="ij")
    u = u.reshape(-1)[None, :, None]
    v = v.reshape(-1)[None, :, None]
    samples = (
        c00[:, None] * (1 - u) * (1 - v) + c01[:, None] * (1 - u) * v
        + c10[:, None] * u * (1 - v) + c11[:, None] * u * v
    ).reshape(-1, 3)
    vertices = xyz[valid]
    return samples, vertices


def rasterize_patches(names, patches_dir, band, cfg, device, scratch):
    """Dense ds4 int32 patch-id grid (and per-patch probe vertices)."""
    import torch

    shape = band.band_shape(4)
    grid_path = scratch / "patch_grid_ds4.dat"
    vertex_path = scratch / "patch_vertices.npz"
    stamp_path = scratch / "patch_grid_params.json"
    stamp = {
        "patches_dir": str(patches_dir), "erode": cfg.patch_erode,
        "subdiv": cfg.patch_subdiv, "dilate": cfg.patch_dilate,
        "band": [band.z_lo, band.z_hi], "patches": len(names),
    }
    if grid_path.exists() and vertex_path.exists() and stamp_path.exists():
        if json.loads(stamp_path.read_text()) == stamp:
            grid = torch.from_numpy(
                np.memmap(
                    grid_path, dtype=np.int32, mode="r", shape=shape).copy()
            ).to(device)
            return grid

    grid = torch.full(shape, -1, dtype=torch.int32, device=device)
    flat = grid.view(-1)
    vertex_chunks, vertex_ids = [], []
    buffer_xyz, buffer_id = [], []

    def splat():
        if not buffer_xyz:
            return
        xyz = torch.as_tensor(
            np.concatenate(buffer_xyz), dtype=torch.float32, device=device)
        ids = torch.as_tensor(
            np.concatenate(buffer_id), dtype=torch.int32, device=device)
        buffer_xyz.clear()
        buffer_id.clear()
        cells, inside = voxelize(xyz, band, 4, device)
        flat[cells[inside]] = ids[inside]

    with ThreadPoolExecutor(max_workers=16) as pool:
        jobs = pool.map(
            lambda name: _patch_samples(
                Path(patches_dir) / name, cfg.patch_erode, cfg.patch_subdiv),
            names)
        buffered = 0
        for patch_id, (samples, vertices) in enumerate(jobs):
            if samples is None:
                continue
            buffer_xyz.append(samples)
            buffer_id.append(
                np.full(len(samples), patch_id, dtype=np.int32))
            buffered += len(samples)
            if vertices is not None and len(vertices):
                stride = max(1, len(vertices) // 256)
                kept = vertices[::stride][:256]
                vertex_chunks.append(kept)
                vertex_ids.append(
                    np.full(len(kept), patch_id, dtype=np.int32))
            if buffered >= 8_000_000:
                splat()
                buffered = 0
        splat()

    for _ in range(int(cfg.patch_dilate)):
        filled = grid >= 0
        for axis in range(3):
            for shift in (1, -1):
                neighbor = torch.roll(grid, shifts=shift, dims=axis)
                neighbor_filled = torch.roll(filled, shifts=shift, dims=axis)
                take = (~filled) & neighbor_filled
                grid[take] = neighbor[take]
        del filled

    memmap = np.memmap(grid_path, dtype=np.int32, mode="w+", shape=shape)
    memmap[:] = grid.cpu().numpy()
    memmap.flush()
    np.savez(
        vertex_path,
        xyz=(np.concatenate(vertex_chunks)
             if vertex_chunks else np.zeros((0, 3), np.float32)),
        patch=(np.concatenate(vertex_ids)
               if vertex_ids else np.zeros(0, np.int32)))
    _write_json_atomic(stamp_path, stamp)
    return grid


# ---------------------------------------------------------------------------
# pass R2: registration


def accumulate_registration(archive, plan, blocks, patch_grid, num_patches,
                            band, cfg, device):
    """Per-(slab, patch) weighted winding statistics from one point stream."""
    import torch

    rays = RaysGpu(archive, device)
    flat_grid = patch_grid.view(-1)
    key_chunks, stat_chunks = [], []
    started = time.time()
    for position, raw in enumerate(
            stream_blocks(archive, plan, blocks, read_prob=not cfg.no_prob)):
        joined = join_block(raw, rays, cfg, device)
        cells, inside = voxelize(joined["xyz"], band, 4, device)
        pid = torch.full_like(cells, -1, dtype=torch.int32)
        pid[inside] = flat_grid[cells[inside]]
        selected = pid >= 0
        if not bool(selected.any()):
            continue
        key = (joined["slab"][selected] * num_patches
               + pid[selected].long())
        omega = joined["omega"][selected]
        winding = joined["winding"][selected]
        xyz = joined["xyz"][selected]
        unique, inverse = torch.unique(key, return_inverse=True)
        stats = torch.zeros((len(unique), 6), device=device)
        columns = torch.stack([
            omega, omega * winding, omega * winding * winding,
            omega * xyz[:, 0], omega * xyz[:, 1], omega * xyz[:, 2],
        ], dim=1)
        stats.index_add_(0, inverse, columns)
        key_chunks.append(unique.cpu().numpy())
        stat_chunks.append(stats.double().cpu().numpy())
        if position % 25 == 0:
            elapsed = time.time() - started
            _log(f"registration block {position + 1}/{len(blocks)}"
                 f" ({elapsed:.0f}s)")
    if not key_chunks:
        return np.zeros(0, np.int64), np.zeros((0, 6))
    keys = np.concatenate(key_chunks)
    stats = np.concatenate(stat_chunks)
    order = np.argsort(keys, kind="stable")
    keys, stats = keys[order], stats[order]
    unique_keys, first = np.unique(keys, return_index=True)
    reduced = np.add.reduceat(stats, first, axis=0)
    return unique_keys, reduced


def build_patch_edges(keys, stats, num_patches, cfg):
    """Co-hit constraints: same patch => same winding, gated and localized."""
    omega = stats[:, 0]
    strong = omega >= cfg.patch_min_weight
    keys, stats, omega = keys[strong], stats[strong], omega[strong]
    if not len(keys):
        return (np.zeros(0, np.int64),) * 2 + (np.zeros(0),) * 2
    mean = stats[:, 1] / omega
    variance = np.maximum(stats[:, 2] / omega - mean ** 2, 0.0)
    tight = np.sqrt(variance) <= cfg.patch_var_max
    keys, stats, omega, mean = keys[tight], stats[tight], omega[tight], mean[tight]
    centroid = stats[:, 3:6] / omega[:, None]
    pid = keys % num_patches
    slab = keys // num_patches

    order = np.argsort(pid, kind="stable")
    pid, slab, omega, mean, centroid = (
        pid[order], slab[order], omega[order], mean[order], centroid[order])
    groups, starts = np.unique(pid, return_index=True)
    starts = np.append(starts, len(pid))
    edge_u, edge_v, edge_delta, edge_weight = [], [], [], []
    for group in range(len(groups)):
        lo, hi = starts[group], starts[group + 1]
        size = hi - lo
        if size < 2:
            continue
        if size > cfg.patch_max_group:
            top = np.argsort(omega[lo:hi])[::-1][: cfg.patch_max_group] + lo
        else:
            top = np.arange(lo, hi)
        points = centroid[top]
        distance = np.linalg.norm(
            points[:, None] - points[None, :], axis=-1)
        np.fill_diagonal(distance, np.inf)
        neighbors = min(cfg.patch_edge_k, len(top) - 1)
        nearest = np.argpartition(distance, neighbors - 1, axis=1)[:, :neighbors]
        for row in range(len(top)):
            u = top[row]
            for v in top[nearest[row]]:
                if slab[u] == slab[v]:
                    continue
                edge_u.append(slab[u])
                edge_v.append(slab[v])
                edge_delta.append(mean[u] - mean[v])
                edge_weight.append(
                    min(omega[u], omega[v]) * math.exp(
                        -(distance[row, np.where(top == v)[0][0]]
                          / cfg.patch_centroid_gate) ** 2))
    if not edge_u:
        return (np.zeros(0, np.int64),) * 2 + (np.zeros(0),) * 2
    edge_u = np.asarray(edge_u, np.int64)
    edge_v = np.asarray(edge_v, np.int64)
    edge_delta = np.asarray(edge_delta)
    edge_weight = np.asarray(edge_weight)
    swap = edge_u > edge_v
    edge_u[swap], edge_v[swap] = edge_v[swap], edge_u[swap]
    edge_delta[swap] = -edge_delta[swap]
    packed = edge_u * (edge_v.max() + 1) + edge_v
    unique, inverse = np.unique(packed, return_inverse=True)
    weight_sum = np.bincount(inverse, weights=edge_weight)
    delta_sum = np.bincount(inverse, weights=edge_weight * edge_delta)
    first = np.zeros(len(unique), dtype=np.int64)
    first[inverse[::-1]] = np.arange(len(packed))[::-1]
    return (
        edge_u[first], edge_v[first],
        delta_sum / np.maximum(weight_sum, 1e-12), weight_sum)


def solve_slab_offsets(num_slabs, edge_u, edge_v, edge_delta, edge_weight, *,
                       iterations=5, huber=0.25, prior_weight=0.05,
                       prior_huber=0.5, max_correction=4.0):
    """Robust matrix-free IRLS solve of ``offset[v] - offset[u] = delta``.

    Adapted from phase_overlap_sync._solve_phase_overlap_graph with a zero
    prior: the model's absolute labels are presumed correct on average, and
    the weak prior both fixes the gauge per connected component and keeps
    holonomy-frustrated corrections bounded.
    """
    from scipy.sparse.linalg import LinearOperator, cg

    prior = np.zeros(num_slabs, dtype=np.float64)
    edge_u = np.asarray(edge_u, dtype=np.int64)
    edge_v = np.asarray(edge_v, dtype=np.int64)
    edge_delta = np.asarray(edge_delta, dtype=np.float64)
    base_weight = np.asarray(edge_weight, dtype=np.float64)
    finite = (
        np.isfinite(edge_delta) & np.isfinite(base_weight) & (base_weight > 0)
        & (edge_u >= 0) & (edge_u < num_slabs)
        & (edge_v >= 0) & (edge_v < num_slabs) & (edge_u != edge_v))
    edge_u, edge_v, edge_delta, base_weight = (
        value[finite] for value in (edge_u, edge_v, edge_delta, base_weight))
    if not len(edge_u):
        return prior, {"edges": 0, "supported_nodes": 0}

    correction = np.zeros(num_slabs, dtype=np.float64)
    robust_edge_weight = base_weight.copy()
    robust_prior_weight = np.full(num_slabs, float(prior_weight))
    for _ in range(max(1, int(iterations))):
        def matvec(values):
            return (
                np.bincount(
                    edge_u,
                    robust_edge_weight * (values[edge_u] - values[edge_v]),
                    minlength=num_slabs)
                + np.bincount(
                    edge_v,
                    robust_edge_weight * (values[edge_v] - values[edge_u]),
                    minlength=num_slabs)
                + robust_prior_weight * values)

        operator = LinearOperator(
            (num_slabs, num_slabs), matvec=matvec, dtype=np.float64)
        diagonal = (
            np.bincount(edge_u, robust_edge_weight, minlength=num_slabs)
            + np.bincount(edge_v, robust_edge_weight, minlength=num_slabs)
            + robust_prior_weight)
        preconditioner = LinearOperator(
            (num_slabs, num_slabs),
            matvec=lambda values: values / np.maximum(diagonal, 1e-12),
            dtype=np.float64)
        weighted = robust_edge_weight * edge_delta
        rhs = (
            np.bincount(edge_v, weighted, minlength=num_slabs)
            - np.bincount(edge_u, weighted, minlength=num_slabs))
        correction, info = cg(
            operator, rhs, x0=correction, M=preconditioner,
            rtol=1e-5, atol=1e-8, maxiter=200)
        if info < 0:  # pragma: no cover - scipy internal failure
            raise RuntimeError(f"slab offset solve failed: cg={info}")
        np.clip(correction, -max_correction, max_correction, out=correction)
        residual = correction[edge_v] - correction[edge_u] - edge_delta
        robust_edge_weight = base_weight * np.minimum(
            1.0, huber / np.maximum(np.abs(residual), 1e-12))
        robust_prior_weight = float(prior_weight) * np.minimum(
            1.0, prior_huber / np.maximum(np.abs(correction), 1e-12))

    residual = correction[edge_v] - correction[edge_u] - edge_delta
    degree = (
        np.bincount(edge_u, minlength=num_slabs)
        + np.bincount(edge_v, minlength=num_slabs))
    fractional = np.abs(correction - np.round(correction))
    stats = {
        "edges": int(len(edge_u)),
        "supported_nodes": int(np.count_nonzero(degree)),
        "edge_residual_median_abs": float(np.median(np.abs(residual))),
        "edge_residual_p95_abs": float(np.quantile(np.abs(residual), 0.95)),
        "correction_p95_abs": float(np.quantile(np.abs(correction), 0.95)),
        "correction_max_abs": float(np.max(np.abs(correction))),
        "nonzero_offset_fraction": float(
            np.mean(np.round(correction) != 0)),
        "fractional_gt_0p3_fraction": float(np.mean(fractional > 0.3)),
    }
    return correction, stats


def run_registration(archive, plan, blocks, band, cfg, device, scratch):
    offsets_path = scratch / "patch_offsets.npz"
    fingerprint = hashlib.sha256(json.dumps({
        "patches_dir": str(cfg.patches_dir),
        "num_points": archive.num_points,
        "erode": cfg.patch_erode, "subdiv": cfg.patch_subdiv,
        "dilate": cfg.patch_dilate, "edge_k": cfg.patch_edge_k,
        "gate": cfg.patch_centroid_gate, "var_max": cfg.patch_var_max,
        "min_weight": cfg.patch_min_weight, "no_prob": cfg.no_prob,
        "blocks": [int(b) for b in blocks],
    }, sort_keys=True).encode()).hexdigest()
    if offsets_path.exists():
        cached = np.load(offsets_path, allow_pickle=True)
        if str(cached["fingerprint"]) == fingerprint:
            return (np.asarray(cached["offsets"], np.int32),
                    json.loads(str(cached["stats"])))

    names = load_patch_metas(cfg.patches_dir, band, scratch)
    _log(f"registration: {len(names)} patches overlap the band")
    patch_grid = rasterize_patches(
        names, cfg.patches_dir, band, cfg, device, scratch)
    occupancy = float((patch_grid >= 0).float().mean())
    _log(f"registration: patch grid occupancy {occupancy:.4f}")
    keys, stats = accumulate_registration(
        archive, plan, blocks, patch_grid, len(names), band, cfg, device)
    del patch_grid
    _log(f"registration: {len(keys)} (slab, patch) pairs")
    edge_u, edge_v, edge_delta, edge_weight = build_patch_edges(
        keys, stats, len(names), cfg)
    _log(f"registration: {len(edge_u)} deduplicated edges")
    solution, solve_stats = solve_slab_offsets(
        archive.num_slabs, edge_u, edge_v, edge_delta, edge_weight)
    offsets = np.round(solution).astype(np.int32)
    solve_stats["patches"] = len(names)
    solve_stats["pairs"] = int(len(keys))
    _log(f"registration: {solve_stats}")
    np.savez(
        offsets_path, offsets=offsets, solution=solution,
        stats=json.dumps(solve_stats), fingerprint=fingerprint)
    return offsets, solve_stats


# ---------------------------------------------------------------------------
# pass 1: scatter


DS8_CHANNELS = ("sum_wv", "sum_w", "sum_wvv", "dir_z", "dir_y", "dir_x",
                "rho_wr", "rho_w")
DS4_CHANNELS = ("sum_wv", "sum_w")


class ScatterAccumulator:
    def __init__(self, band, device, winding_center):
        import torch

        self.band = band
        self.winding_center = float(winding_center)
        self.shape8 = band.band_shape(8)
        self.shape4 = band.band_shape(4)
        count8 = int(np.prod(self.shape8))
        count4 = int(np.prod(self.shape4))
        self.ds8 = {name: torch.zeros(count8, device=device)
                    for name in DS8_CHANNELS}
        self.ds4 = {name: torch.zeros(count4, device=device)
                    for name in DS4_CHANNELS}
        self.dropped_points = 0
        self.total_points = 0

    def add(self, joined, slab_offsets, band, device):
        winding = joined["winding"]
        if slab_offsets is not None:
            winding = winding + slab_offsets[joined["slab"]]
        centered = winding - self.winding_center
        omega = joined["omega"]
        self.total_points += len(omega)

        for ds, grids in ((8, self.ds8), (4, self.ds4)):
            flat, inside = voxelize(joined["xyz"], band, ds, device)
            index = flat[inside]
            w = omega[inside]
            wv = w * centered[inside]
            grids["sum_wv"].index_add_(0, index, wv)
            grids["sum_w"].index_add_(0, index, w)
            if ds == 8:
                self.dropped_points += int(len(omega) - int(inside.sum()))
                grids["sum_wvv"].index_add_(0, index, wv * centered[inside])
                direction = joined["direction"][inside]
                grids["dir_z"].index_add_(0, index, w * direction[:, 2])
                grids["dir_y"].index_add_(0, index, w * direction[:, 1])
                grids["dir_x"].index_add_(0, index, w * direction[:, 0])
                rho_mask = inside & joined["rho_valid"]
                flat_rho = flat[rho_mask]
                w_rho = omega[rho_mask]
                grids["rho_wr"].index_add_(
                    0, flat_rho, w_rho * joined["rho"][rho_mask])
                grids["rho_w"].index_add_(0, flat_rho, w_rho)

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        for name, grids, shape in (
                ("ds8", self.ds8, self.shape8), ("ds4", self.ds4, self.shape4)):
            for channel, tensor in grids.items():
                np.save(directory / f"{name}_{channel}.npy",
                        tensor.view(shape).cpu().numpy())

    def load(self, directory, device):
        import torch

        for name, grids in (("ds8", self.ds8), ("ds4", self.ds4)):
            for channel in grids:
                grids[channel] = torch.as_tensor(
                    np.load(directory / f"{name}_{channel}.npy"),
                    device=device).view(-1)


def run_scatter(archive, plan, blocks, band, cfg, device, scratch,
                slab_offsets_np):
    import torch

    stats_dir = scratch / "scatter"
    accumulator = ScatterAccumulator(band, device, cfg.winding_center)
    done_marker = stats_dir / "complete.json"
    if done_marker.exists():
        accumulator.load(stats_dir, device)
        return accumulator

    checkpoint_dirs = [scratch / "scatter_ckpt_a", scratch / "scatter_ckpt_b"]
    start_block = 0
    newest = None
    for candidate in checkpoint_dirs:
        cursor = candidate / "cursor.json"
        if cursor.exists():
            state = json.loads(cursor.read_text())
            if newest is None or state["time"] > newest[0]:
                newest = (state["time"], candidate, state["next_position"])
    if newest is not None:
        _log(f"scatter: resuming from {newest[1].name} at block {newest[2]}")
        accumulator.load(newest[1], device)
        start_block = int(newest[2])

    rays = RaysGpu(archive, device)
    slab_offsets = None
    if slab_offsets_np is not None and np.any(slab_offsets_np):
        slab_offsets = torch.as_tensor(
            slab_offsets_np, dtype=torch.float32, device=device)
    last_checkpoint = time.time()
    slot = 0
    started = time.time()
    for position, raw in enumerate(
            stream_blocks(archive, plan, blocks, read_prob=not cfg.no_prob,
                          start=start_block)):
        joined = join_block(raw, rays, cfg, device)
        accumulator.add(joined, slab_offsets, band, device)
        absolute = start_block + position
        if absolute % 25 == 0:
            elapsed = time.time() - started
            _log(f"scatter block {absolute + 1}/{len(blocks)} ({elapsed:.0f}s,"
                 f" {accumulator.total_points / 1e9:.2f}e9 points)")
        if (time.time() - last_checkpoint) > cfg.checkpoint_minutes * 60:
            target = checkpoint_dirs[slot % 2]
            accumulator.save(target)
            _write_json_atomic(target / "cursor.json", {
                "time": time.time(), "next_position": absolute + 1})
            slot += 1
            last_checkpoint = time.time()

    accumulator.save(stats_dir)
    _write_json_atomic(done_marker, {
        "total_points": accumulator.total_points,
        "dropped_points": accumulator.dropped_points,
    })
    for candidate in checkpoint_dirs:
        shutil.rmtree(candidate, ignore_errors=True)
    return accumulator


# ---------------------------------------------------------------------------
# passes 2/3: solve


@dataclass
class SolveTargets:
    mu: object          # absolute winding mean per voxel (0 where unsupported)
    weight: object      # capped data weight
    support: object     # bool: any observation
    domain: object      # bool: dilated solve region
    rho: object         # gradient-magnitude target, per-voxel units
    dir_unit: object    # [3, ...] unit direction toward winding+1 (zyx order)
    dir_conf: object    # 0..1 direction agreement


def _subsampled_quantile(values, q):
    """torch.quantile rejects inputs over ~16M elements; stride-subsample."""
    import torch

    stride = max(1, len(values) // 4_000_000)
    return torch.quantile(values.float()[::stride], q)


def dilate_mask(mask, iterations):
    import torch
    import torch.nn.functional as F

    result = mask
    for _ in range(int(iterations)):
        result = F.max_pool3d(
            result[None, None].float(), kernel_size=3, stride=1, padding=1
        )[0, 0] > 0
    return result


def pull_push_fill(sum_wv, sum_w, winding_center, eps=1e-6):
    """Smooth global initialization: observed ratios, hole-filled coarse-to-fine."""
    import torch
    import torch.nn.functional as F

    pyramid = [(sum_wv[None, None], sum_w[None, None])]
    while min(pyramid[-1][0].shape[2:]) > 4:
        wv, w = pyramid[-1]
        pyramid.append((
            F.avg_pool3d(wv, 2, stride=2, ceil_mode=True,
                         count_include_pad=False),
            F.avg_pool3d(w, 2, stride=2, ceil_mode=True,
                         count_include_pad=False)))
    wv, w = pyramid[-1]
    total = w.sum()
    global_mean = (wv.sum() / total) if float(total) > eps else wv.sum() * 0.0
    coarse = torch.where(w > eps, wv / w.clamp(min=eps), global_mean)
    for wv, w in reversed(pyramid[:-1]):
        up = F.interpolate(
            coarse, size=wv.shape[2:], mode="trilinear", align_corners=False)
        coarse = torch.where(w > eps, wv / w.clamp(min=eps), up)
    return coarse[0, 0] + winding_center


def make_targets(accumulator, cfg, device):
    import torch

    shape = accumulator.shape8
    view = lambda name: accumulator.ds8[name].view(shape)
    sum_w = view("sum_w")
    sum_wv = view("sum_wv")
    support = sum_w > cfg.support_omega_min
    safe = sum_w.clamp(min=1e-12)
    mu = torch.where(
        support, sum_wv / safe + accumulator.winding_center,
        torch.zeros_like(sum_w))
    positive = sum_w[support]
    cap = (_subsampled_quantile(positive, 0.95)
           if len(positive) else torch.tensor(1.0, device=device))
    weight = torch.where(
        support, sum_w.clamp(max=cap) ** cfg.data_weight_power,
        torch.zeros_like(sum_w))

    direction = torch.stack(
        [view("dir_z"), view("dir_y"), view("dir_x")])
    norm = direction.norm(dim=0)
    dir_conf = torch.where(support, norm / safe, torch.zeros_like(sum_w))
    dir_unit = direction / norm.clamp(min=1e-12)

    rho_w = view("rho_w")
    rho_observed = rho_w > cfg.support_omega_min
    rho_raw = view("rho_wr") / rho_w.clamp(min=1e-12)
    if bool(rho_observed.any()):
        rho_median = _subsampled_quantile(rho_raw[rho_observed], 0.5)
    else:
        rho_median = torch.tensor(1.0 / 16.0, device=device)
    rho = torch.where(rho_observed, rho_raw, rho_median) * 8.0

    domain = dilate_mask(support, cfg.domain_dilate)
    return SolveTargets(
        mu=mu, weight=weight, support=support, domain=domain, rho=rho,
        dir_unit=dir_unit, dir_conf=dir_conf), float(rho_median)


def field_losses(view, tgt_slice, cfg, interior, normalizers):
    """All loss terms for one contiguous z view; each interior voxel's
    stencils are computed exactly once (halo rows belong to neighbors)."""
    import torch
    import torch.nn.functional as F

    padded = F.pad(view[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]
    center = padded[1:-1, 1:-1, 1:-1]
    neighbors = (
        padded[2:, 1:-1, 1:-1] + padded[:-2, 1:-1, 1:-1]
        + padded[1:-1, 2:, 1:-1] + padded[1:-1, :-2, 1:-1]
        + padded[1:-1, 1:-1, 2:] + padded[1:-1, 1:-1, :-2])
    laplacian = neighbors - 6.0 * center
    gradient = torch.stack([
        0.5 * (padded[2:, 1:-1, 1:-1] - padded[:-2, 1:-1, 1:-1]),
        0.5 * (padded[1:-1, 2:, 1:-1] - padded[1:-1, :-2, 1:-1]),
        0.5 * (padded[1:-1, 1:-1, 2:] - padded[1:-1, 1:-1, :-2]),
    ])

    inner = slice(interior[0], interior[1])
    domain = tgt_slice["domain"][inner]
    weight = tgt_slice["weight"][inner]
    value = view[inner]

    residual = value - tgt_slice["mu"][inner]
    absolute = residual.abs()
    delta = cfg.huber_delta
    huber = torch.where(
        absolute <= delta, 0.5 * residual ** 2,
        delta * (absolute - 0.5 * delta))
    data = (weight * huber).sum() / normalizers["weight"]

    lap = laplacian[inner]
    smooth = ((lap ** 2) * domain).sum() / normalizers["domain"]

    grad = gradient[:, inner]
    magnitude = grad.norm(dim=0)
    eikonal = ((((magnitude - tgt_slice["rho"][inner]) ** 2) * domain).sum()
               / normalizers["domain"])

    aligned = (grad * tgt_slice["dir_unit"][:, inner]).sum(dim=0)
    confident = (tgt_slice["dir_conf"][inner] > cfg.dir_conf_min) & domain
    mono = ((F.relu(-aligned) ** 2) * confident).sum() / normalizers["domain"]

    return (cfg.w_data * data + cfg.w_smooth * smooth
            + cfg.w_eikonal * eikonal + cfg.w_mono * mono,
            {"data": float(data.detach()), "smooth": float(smooth.detach()),
             "eikonal": float(eikonal.detach()),
             "mono": float(mono.detach())})


def _patch_probe(scratch, band, device):
    import torch

    stored = np.load(scratch / "patch_vertices.npz")
    if not len(stored["xyz"]):
        return None
    xyz = torch.as_tensor(stored["xyz"], device=device)
    patch = torch.as_tensor(stored["patch"].astype(np.int64), device=device)
    flat, inside = voxelize(xyz, band, 8, device)
    return flat[inside], patch[inside]


def patch_constancy_loss(field_flat, probe, cfg):
    """Pull the field toward a per-patch constant at eroded patch vertices."""
    import torch

    index, patch = probe
    sampled = field_flat[index]
    groups = int(patch.max()) + 1
    total = torch.zeros(groups, device=sampled.device)
    count = torch.zeros(groups, device=sampled.device)
    total.index_add_(0, patch, sampled)
    count.index_add_(0, patch, torch.ones_like(sampled))
    mean = total / count.clamp(min=1.0)
    residual = sampled - mean[patch]
    absolute = residual.abs()
    delta = cfg.huber_delta
    huber = torch.where(
        absolute <= delta, 0.5 * residual ** 2,
        delta * (absolute - 0.5 * delta))
    return huber.mean()


def solve_ds8(accumulator, targets, cfg, device, scratch):
    import torch

    result_path = scratch / "w_ds8.npy"
    if result_path.exists():
        return torch.as_tensor(np.load(result_path), device=device)

    shape = accumulator.shape8
    initial = pull_push_fill(
        accumulator.ds8["sum_wv"].view(shape),
        accumulator.ds8["sum_w"].view(shape),
        accumulator.winding_center)
    field_tensor = initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([field_tensor], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.iters_ds8, eta_min=cfg.lr * 0.03)
    normalizers = {
        "weight": targets.weight.sum().clamp(min=1e-6),
        "domain": targets.domain.float().sum().clamp(min=1.0),
    }
    tgt = {
        "mu": targets.mu, "weight": targets.weight,
        "domain": targets.domain, "rho": targets.rho,
        "dir_unit": targets.dir_unit, "dir_conf": targets.dir_conf,
    }
    probe = _patch_probe(scratch, accumulator.band, device) if (
        cfg.w_patch > 0 and (scratch / "patch_vertices.npz").exists()) else None

    for iteration in range(cfg.iters_ds8):
        optimizer.zero_grad(set_to_none=True)
        loss, parts = field_losses(
            field_tensor, tgt, cfg, (0, shape[0]), normalizers)
        if probe is not None:
            loss = loss + cfg.w_patch * patch_constancy_loss(
                field_tensor.view(-1), probe, cfg)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if iteration % 100 == 0 or iteration == cfg.iters_ds8 - 1:
            with torch.no_grad():
                metrics = solve_metrics(field_tensor, targets)
            _log(f"ds8 iter {iteration}: {parts} {metrics}")
    solved = field_tensor.detach()
    np.save(result_path, solved.cpu().numpy())
    return solved


def solve_metrics(field_tensor, targets):
    import torch

    residual = (field_tensor - targets.mu).abs()[targets.support]
    if not len(residual):
        return {}
    return {
        "res_med": float(residual.median()),
        "res_p95": float(_subsampled_quantile(residual, 0.95)),
    }


def refine_ds4(field8, accumulator, targets8, cfg, device, scratch):
    import torch
    import torch.nn.functional as F

    result_path = scratch / "w_ds4.npy"
    if result_path.exists():
        return torch.as_tensor(np.load(result_path), device=device)

    shape4 = accumulator.shape4
    field_tensor = F.interpolate(
        field8[None, None], size=shape4, mode="trilinear",
        align_corners=False)[0, 0].contiguous().requires_grad_(True)

    sum_w4 = accumulator.ds4["sum_w"].view(shape4)
    sum_wv4 = accumulator.ds4["sum_wv"].view(shape4)
    support4 = sum_w4 > cfg.support_omega_min
    mu4 = torch.where(
        support4, sum_wv4 / sum_w4.clamp(min=1e-12)
        + accumulator.winding_center, torch.zeros_like(sum_w4))
    positive = sum_w4[support4]
    cap = (_subsampled_quantile(positive, 0.95)
           if len(positive) else torch.tensor(1.0, device=device))
    weight4 = torch.where(
        support4, sum_w4.clamp(max=cap) ** cfg.data_weight_power,
        torch.zeros_like(sum_w4))
    del sum_w4, sum_wv4, support4, positive

    def upsample2(tensor, z_lo8, z_hi8):
        piece = tensor[..., z_lo8:z_hi8, :, :]
        for axis in (-3, -2, -1):
            piece = piece.repeat_interleave(2, dim=axis)
        return piece

    if cfg.ds4_optimizer == "adam":
        optimizer = torch.optim.Adam([field_tensor], lr=cfg.lr * 0.3)
    else:
        optimizer = torch.optim.SGD(
            [field_tensor], lr=cfg.lr * 0.3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.iters_ds4, eta_min=cfg.lr * 0.01)

    normalizers = {
        "weight": weight4.sum().clamp(min=1e-6),
        "domain": targets8.domain.float().sum().clamp(min=1.0) * 8.0,
    }
    slab = int(cfg.ds4_z_slab)
    starts = list(range(0, shape4[0], slab))
    for iteration in range(cfg.iters_ds4):
        optimizer.zero_grad(set_to_none=True)
        parts_total = {"data": 0.0, "smooth": 0.0, "eikonal": 0.0, "mono": 0.0}
        for z0 in starts:
            z1 = min(z0 + slab, shape4[0])
            lo = max(z0 - 1, 0)
            hi = min(z1 + 1, shape4[0])
            view = field_tensor[lo:hi]
            z0_8, z1_8 = lo // 2, (hi + 1) // 2
            tgt_slice = {
                "mu": mu4[lo:hi], "weight": weight4[lo:hi],
                "domain": upsample2(
                    targets8.domain, z0_8, z1_8)[lo - z0_8 * 2:, :shape4[1],
                                                 :shape4[2]][:hi - lo],
                "rho": upsample2(
                    targets8.rho, z0_8, z1_8)[lo - z0_8 * 2:, :shape4[1],
                                              :shape4[2]][:hi - lo] * 0.5,
                "dir_unit": upsample2(
                    targets8.dir_unit, z0_8, z1_8)[:, lo - z0_8 * 2:,
                                                   :shape4[1], :shape4[2]]
                                                  [:, :hi - lo],
                "dir_conf": upsample2(
                    targets8.dir_conf, z0_8, z1_8)[lo - z0_8 * 2:, :shape4[1],
                                                   :shape4[2]][:hi - lo],
            }
            loss, parts = field_losses(
                view, tgt_slice, cfg, (z0 - lo, z1 - lo), normalizers)
            loss.backward()
            for key in parts_total:
                parts_total[key] += parts[key]
        optimizer.step()
        scheduler.step()
        if iteration % 25 == 0 or iteration == cfg.iters_ds4 - 1:
            _log(f"ds4 iter {iteration}: {parts_total}")
    solved = field_tensor.detach()
    np.save(result_path, solved.cpu().numpy())
    return solved


# ---------------------------------------------------------------------------
# pass 4: OME-Zarr v2 pyramid writer


class OmeZarrPyramidWriter:
    """Sparse Zarr-v2 OME pyramid, modeled on winding_centerlines.OmeZarrWriter.

    VC3D's openLocalZarrPyramid wants numbered level directories with a v2
    ``.zarray`` each, uint8/uint16 payloads, and 2x scale per level.
    """

    def __init__(self, path, band, source, parameters, zstd_level):
        from numcodecs import Zstd

        self.path = Path(path)
        self.band = band
        self.compressor = Zstd(level=int(zstd_level))
        self.level_shapes = [band.level_shape(level) for level in range(LEVELS)]
        self.path.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(self.path / ".zgroup", {"zarr_format": 2})
        axes = [{"name": axis, "type": "space"} for axis in "zyx"]
        self.attributes = {
            "kind": "winding_field_merge",
            "created": datetime.now(timezone.utc).isoformat(),
            "command_line": shlex.join(sys.argv),
            "source": str(source),
            "encoding": {"scale": 1.0 / 256.0, "offset": 0.0, "invalid": 0},
            "z_band": [band.z_lo, band.z_hi],
            "pyramid": {
                "levels": LEVELS,
                "downsampling_method": "mean",
                "scale_factor_zyx": [2, 2, 2],
            },
            "complete": False,
            "parameters": parameters,
            "multiscales": [{
                "version": "0.4",
                "name": "winding_field",
                "axes": axes,
                "datasets": [{
                    "path": str(level),
                    "coordinateTransformations": [{
                        "type": "scale",
                        "scale": [float(1 << level)] * 3,
                    }],
                } for level in range(LEVELS)],
            }],
        }
        _write_json_atomic(self.path / ".zattrs", self.attributes)
        for level, shape in enumerate(self.level_shapes):
            level_path = self.path / str(level)
            level_path.mkdir(exist_ok=True)
            _write_json_atomic(level_path / ".zarray", {
                "zarr_format": 2,
                "shape": list(shape),
                "chunks": [CHUNK, CHUNK, CHUNK],
                "dtype": "<u2",
                "compressor": {"id": "zstd", "level": int(zstd_level)},
                "fill_value": 0,
                "order": "C",
                "filters": None,
                "dimension_separator": "/",
            })
            _write_json_atomic(
                level_path / ".zattrs",
                {"_ARRAY_DIMENSIONS": ["z", "y", "x"],
                 **({"downsampling_method": "mean"} if level else {})})
        self._pool = ThreadPoolExecutor(max_workers=16)
        self._futures = []

    def _write_chunk(self, level, chunk_index, values):
        destination = (
            self.path / str(level)
            / str(chunk_index[0]) / str(chunk_index[1]) / str(chunk_index[2]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = bytes(self.compressor.encode(
            np.ascontiguousarray(values).tobytes()))
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}")
        with open(temporary, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, destination)

    def write_block(self, level, origin, values):
        """Write a uint16 block at ``origin`` (level voxels), splitting into
        chunks; all-zero chunks are skipped (fill).  Chunks straddling the
        block edge are zero-padded — the band is this array's only content,
        so no read-modify-write is ever needed."""
        shape = self.level_shapes[level]
        for cz in range(origin[0] // CHUNK,
                        -(-(origin[0] + values.shape[0]) // CHUNK)):
            for cy in range(origin[1] // CHUNK,
                            -(-(origin[1] + values.shape[1]) // CHUNK)):
                for cx in range(origin[2] // CHUNK,
                                -(-(origin[2] + values.shape[2]) // CHUNK)):
                    chunk_lo = np.array([cz, cy, cx]) * CHUNK
                    copy_lo = np.maximum(chunk_lo, origin)
                    copy_hi = np.minimum(
                        chunk_lo + CHUNK,
                        np.array(origin) + values.shape)
                    if np.any(copy_hi <= copy_lo):
                        continue
                    source = values[
                        copy_lo[0] - origin[0]:copy_hi[0] - origin[0],
                        copy_lo[1] - origin[1]:copy_hi[1] - origin[1],
                        copy_lo[2] - origin[2]:copy_hi[2] - origin[2]]
                    if not source.any():
                        continue
                    chunk = np.zeros((CHUNK, CHUNK, CHUNK), dtype=np.uint16)
                    chunk[
                        copy_lo[0] - chunk_lo[0]:copy_hi[0] - chunk_lo[0],
                        copy_lo[1] - chunk_lo[1]:copy_hi[1] - chunk_lo[1],
                        copy_lo[2] - chunk_lo[2]:copy_hi[2] - chunk_lo[2],
                    ] = source
                    self._futures.append(self._pool.submit(
                        self._write_chunk, level, (cz, cy, cx), chunk))
        self._drain(limit=256)

    def _drain(self, limit=0):
        while len(self._futures) > limit:
            self._futures.pop(0).result()

    def finalize(self, extra_attributes):
        self._drain()
        self._pool.shutdown()
        self.attributes.update(extra_attributes)
        self.attributes["complete"] = True
        _write_json_atomic(self.path / ".zattrs", self.attributes)


def encode_u16(field_np, mask_np):
    values = np.clip(np.rint(field_np * 256.0), 1, 65535).astype(np.uint16)
    values[~mask_np] = 0
    return values


def masked_mean_pool(field_tensor, mask):
    import torch.nn.functional as F

    weight = mask.float()[None, None]
    pooled_w = F.avg_pool3d(weight, 2, stride=2, ceil_mode=True,
                            count_include_pad=False)
    pooled_v = F.avg_pool3d((field_tensor * mask.float())[None, None], 2,
                            stride=2, ceil_mode=True, count_include_pad=False)
    new_mask = pooled_w[0, 0] > 0
    return (pooled_v[0, 0] / pooled_w[0, 0].clamp(min=1e-12)), new_mask


def write_pyramid(field4, support8, band, writer, cfg, device):
    import torch
    import torch.nn.functional as F

    mask8 = dilate_mask(support8, cfg.mask_dilate)
    mask4 = mask8
    for axis in range(3):
        mask4 = mask4.repeat_interleave(2, dim=axis)
    mask4 = mask4[: field4.shape[0], : field4.shape[1], : field4.shape[2]]
    if bool(mask4.any()):
        valid_min = float(field4[mask4].min())
        if valid_min < 0.5:
            _log(f"WARNING: solved winding min {valid_min:.3f} < 0.5;"
                 " values below 1/256 clamp into the invalid code")

    # levels 2..5 by successive valid-child mean pooling
    current, current_mask = field4, mask4
    for level in range(2, LEVELS):
        writer.write_block(
            level, (band.origin(1 << level), 0, 0),
            encode_u16(current.cpu().numpy(), current_mask.cpu().numpy()))
        _log(f"write: level {level} done")
        if level + 1 < LEVELS:
            current, current_mask = masked_mean_pool(current, current_mask)

    # levels 1 and 0 by tiled trilinear upsampling of the ds4 field
    shape4 = field4.shape
    tiles = [
        (z0, y0, x0)
        for z0 in range(0, shape4[0], CHUNK)
        for y0 in range(0, shape4[1], CHUNK)
        for x0 in range(0, shape4[2], CHUNK)]
    written = 0
    for z0, y0, x0 in tiles:
        z1 = min(z0 + CHUNK, shape4[0])
        y1 = min(y0 + CHUNK, shape4[1])
        x1 = min(x0 + CHUNK, shape4[2])
        tile_mask = mask4[z0:z1, y0:y1, x0:x1]
        if not bool(tile_mask.any()):
            continue
        lo = (max(z0 - 1, 0), max(y0 - 1, 0), max(x0 - 1, 0))
        hi = (min(z1 + 1, shape4[0]), min(y1 + 1, shape4[1]),
              min(x1 + 1, shape4[2]))
        haloed = field4[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        for level, factor in ((1, 2), (0, 4)):
            up = F.interpolate(
                haloed[None, None], scale_factor=factor, mode="trilinear",
                align_corners=False)[0, 0]
            crop = tuple(
                slice((start - halo_start) * factor,
                      (start - halo_start) * factor + (stop - start) * factor)
                for start, stop, halo_start in zip(
                    (z0, y0, x0), (z1, y1, x1), lo))
            block = up[crop[0], crop[1], crop[2]]
            up_mask = tile_mask
            for axis in range(3):
                up_mask = up_mask.repeat_interleave(factor, dim=axis)
            origin = (band.origin(1 << level) + z0 * (4 >> level),
                      y0 * (4 >> level), x0 * (4 >> level))
            writer.write_block(
                level, origin,
                encode_u16(block.cpu().numpy(), up_mask.cpu().numpy()))
        written += 1
        if written % 100 == 0:
            _log(f"write: {written} populated tiles done")
    _log(f"write: levels 0/1 done ({written} tiles)")


# ---------------------------------------------------------------------------
# agreement report


def agreement_report(archive, field4, support8, band, samples, device):
    """Compare round(W) against the input's voted raster on random populated
    full-res chunks with confident votes — a free validation signal."""
    import torch

    winding = archive.group["winding"]
    confidence = archive.group["confidence"]
    rng = np.random.default_rng(0)
    shape4 = field4.shape
    matched = total = 0
    attempts = 0
    while total < 1 and attempts < samples * 20 or attempts < samples:
        attempts += 1
        z0 = int(rng.integers(band.z_lo, band.z_hi - CHUNK))
        y0 = int(rng.integers(0, archive.full_shape[1] - CHUNK))
        x0 = int(rng.integers(0, archive.full_shape[2] - CHUNK))
        voted = winding[z0:z0 + CHUNK, y0:y0 + CHUNK, x0:x0 + CHUNK]
        conf = confidence[z0:z0 + CHUNK, y0:y0 + CHUNK, x0:x0 + CHUNK]
        good = (voted >= 0) & (conf >= 128)
        if not good.any():
            continue
        zz, yy, xx = np.nonzero(good)
        index = lambda a: torch.as_tensor(a, device=device)
        cz = index(np.clip((zz + z0) // 4 - band.origin(4), 0, shape4[0] - 1))
        cy = index(np.clip((yy + y0) // 4, 0, shape4[1] - 1))
        cx = index(np.clip((xx + x0) // 4, 0, shape4[2] - 1))
        merged = field4[cz, cy, cx].round().long().cpu().numpy()
        matched += int((merged == voted[good]).sum())
        total += int(good.sum())
    return {"agreement": (matched / total if total else None),
            "voxels_compared": total}


# ---------------------------------------------------------------------------
# CLI


@dataclass
class Config:
    patches_dir: str
    no_patches: bool = False
    no_prob: bool = False
    margin: int = 140
    taper_width: float = 24.0
    winding_center: float = 68.0
    patch_erode: int = 1
    patch_subdiv: int = 8
    patch_dilate: int = 0
    patch_edge_k: int = 8
    patch_max_group: int = 64
    patch_centroid_gate: float = 500.0
    patch_var_max: float = 0.3
    patch_min_weight: float = 3.0
    iters_ds8: int = 3000
    iters_ds4: int = 800
    lr: float = 3e-2
    w_data: float = 1.0
    w_smooth: float = 0.05
    w_eikonal: float = 0.2
    w_mono: float = 0.5
    w_patch: float = 0.0
    huber_delta: float = 0.25
    data_weight_power: float = 0.5
    dir_conf_min: float = 0.7
    support_omega_min: float = 1e-3
    domain_dilate: int = 6
    mask_dilate: int = 3
    ds4_optimizer: str = "adam"
    ds4_z_slab: int = 64
    checkpoint_minutes: float = 30.0
    zstd_level: int = 5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_zarr")
    parser.add_argument("output_zarr")
    parser.add_argument(
        "--patches-dir",
        default="/home/sean/Desktop/spiral_dataset/verified_patches_graph_patches")
    parser.add_argument("--no-patches", action="store_true")
    parser.add_argument("--scratch-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--margin", type=int, default=140)
    parser.add_argument("--z-crop", type=int, nargs=2, default=None)
    parser.add_argument("--limit-blocks", type=int, default=None)
    parser.add_argument("--strip-block-chunks", type=int, default=8)
    parser.add_argument("--no-prob", action="store_true")
    parser.add_argument("--taper-width", type=float, default=24.0)
    parser.add_argument("--winding-center", type=float, default=68.0)
    parser.add_argument("--patch-erode", type=int, default=1)
    parser.add_argument("--patch-subdiv", type=int, default=8)
    parser.add_argument("--patch-dilate", type=int, default=0)
    parser.add_argument("--patch-edge-k", type=int, default=8)
    parser.add_argument("--patch-centroid-gate", type=float, default=500.0)
    parser.add_argument("--patch-var-max", type=float, default=0.3)
    parser.add_argument("--patch-min-weight", type=float, default=3.0)
    parser.add_argument("--iters-ds8", type=int, default=3000)
    parser.add_argument("--iters-ds4", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--w-data", type=float, default=1.0)
    parser.add_argument("--w-smooth", type=float, default=0.05)
    parser.add_argument("--w-eikonal", type=float, default=0.2)
    parser.add_argument("--w-mono", type=float, default=0.5)
    parser.add_argument("--w-patch", type=float, default=0.0)
    parser.add_argument("--huber-delta", type=float, default=0.25)
    parser.add_argument("--data-weight-power", type=float, default=0.5)
    parser.add_argument("--dir-conf-min", type=float, default=0.7)
    parser.add_argument("--support-omega-min", type=float, default=1e-3)
    parser.add_argument("--domain-dilate", type=int, default=6)
    parser.add_argument("--mask-dilate", type=int, default=3)
    parser.add_argument("--ds4-optimizer", choices=("adam", "sgdm"),
                        default="adam")
    parser.add_argument("--ds4-z-slab", type=int, default=64)
    parser.add_argument("--checkpoint-minutes", type=float, default=30.0)
    parser.add_argument("--zstd-level", type=int, default=5)
    parser.add_argument("--agreement-samples", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = Config(
        patches_dir=args.patches_dir, no_patches=args.no_patches,
        no_prob=args.no_prob, margin=args.margin,
        taper_width=args.taper_width, winding_center=args.winding_center,
        patch_erode=args.patch_erode, patch_subdiv=args.patch_subdiv,
        patch_dilate=args.patch_dilate, patch_edge_k=args.patch_edge_k,
        patch_centroid_gate=args.patch_centroid_gate,
        patch_var_max=args.patch_var_max,
        patch_min_weight=args.patch_min_weight,
        iters_ds8=args.iters_ds8, iters_ds4=args.iters_ds4, lr=args.lr,
        w_data=args.w_data, w_smooth=args.w_smooth,
        w_eikonal=args.w_eikonal, w_mono=args.w_mono, w_patch=args.w_patch,
        huber_delta=args.huber_delta,
        data_weight_power=args.data_weight_power,
        dir_conf_min=args.dir_conf_min,
        support_omega_min=args.support_omega_min,
        domain_dilate=args.domain_dilate, mask_dilate=args.mask_dilate,
        ds4_optimizer=args.ds4_optimizer, ds4_z_slab=args.ds4_z_slab,
        checkpoint_minutes=args.checkpoint_minutes,
        zstd_level=args.zstd_level)

    import torch

    device = torch.device(args.device)
    output = Path(args.output_zarr)
    scratch = Path(args.scratch_dir or output.with_suffix(".scratch"))
    scratch.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for stale in ("plan.npz", "patch_offsets.npz", "w_ds8.npy",
                      "w_ds4.npy"):
            (scratch / stale).unlink(missing_ok=True)
        for stale_dir in ("scatter", "scatter_ckpt_a", "scatter_ckpt_b"):
            shutil.rmtree(scratch / stale_dir, ignore_errors=True)

    archive = InferenceArchive(args.input_zarr)
    z_range = args.z_crop or archive.attrs.get(
        "z_range", [0, archive.full_shape[0]])
    band = make_band(archive.full_shape, z_range, cfg.margin)
    _log(f"band z [{band.z_lo}, {band.z_hi}) of {archive.full_shape};"
         f" ds8 {band.band_shape(8)}, ds4 {band.band_shape(4)}")

    plan = build_block_plan(archive, scratch, args.strip_block_chunks)
    blocks = select_blocks(plan, archive, band, args.limit_blocks)
    _log(f"{len(blocks)}/{len(plan['strip_lo'])} blocks intersect the band")

    report = {"band": [band.z_lo, band.z_hi], "blocks": len(blocks)}

    slab_offsets = None
    if not cfg.no_patches:
        slab_offsets, registration_stats = run_registration(
            archive, plan, blocks, band, cfg, device, scratch)
        report["registration"] = registration_stats

    accumulator = run_scatter(
        archive, plan, blocks, band, cfg, device, scratch, slab_offsets)
    targets, rho_median = make_targets(accumulator, cfg, device)
    support_count = int(targets.support.sum())
    _log(f"scatter: {support_count} supported ds8 voxels,"
         f" median density {rho_median:.4f} windings/voxel")
    report["supported_ds8_voxels"] = support_count
    report["median_density"] = rho_median
    if not support_count:
        raise RuntimeError("no observations landed inside the solve band")

    field8 = solve_ds8(accumulator, targets, cfg, device, scratch)
    # the raw ds8 statistics are folded into `targets`; free them before the
    # memory-tight ds4 refinement
    for channel in list(accumulator.ds8):
        del accumulator.ds8[channel]
    field4 = refine_ds4(field8, accumulator, targets, cfg, device, scratch)
    del field8

    if args.agreement_samples:
        report["agreement"] = agreement_report(
            archive, field4, targets.support, band,
            args.agreement_samples, device)
        _log(f"agreement: {report['agreement']}")

    if output.exists():
        existing_attrs = output / ".zattrs"
        previous_kind = None
        if existing_attrs.exists():
            try:
                previous_kind = json.loads(
                    existing_attrs.read_text()).get("kind")
            except ValueError:
                pass
        if previous_kind != "winding_field_merge":
            raise RuntimeError(
                f"{output} exists and is not a previous merge output;"
                " refusing to overwrite")
        shutil.rmtree(output)
    writer = OmeZarrPyramidWriter(
        output, band, archive.path,
        parameters={key: getattr(cfg, key) for key in vars(cfg)}
        | {"winding_center": cfg.winding_center}, zstd_level=cfg.zstd_level)
    write_pyramid(field4, targets.support, band, writer, cfg, device)
    writer.finalize({"report": report})
    _write_json_atomic(scratch / "merge_report.json", report)
    _log(f"done: {output}")


if __name__ == "__main__":
    main()
