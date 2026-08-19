"""Precomputed winding-inference crossing attachments to verified patches."""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import numpy as np
import torch
from tqdm import tqdm

from point_collection import (
    build_surface_patch_index,
    locate_points_on_patches,
)
from tifxyz import Patch, load_tifxyz


ARTIFACT_TYPE = "winding_inference_patch_assignments"
FORMAT_VERSION = 1
INFERENCE_ARTIFACT_TYPE = "winding_inference_crossings"
INFERENCE_FORMAT_VERSION = 1


def _canonical_digest(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_description(path: Path, root: Path, value: np.ndarray) -> dict:
    return {
        "file": str(path.relative_to(root)),
        "shape": list(value.shape),
        "dtype": np.dtype(value.dtype).str,
        "sha256": _sha256(path),
    }


def _load_array(root: Path, description: Mapping, *, verify: bool) -> np.ndarray:
    path = root / description["file"]
    if verify and _sha256(path) != description["sha256"]:
        raise ValueError(f"assignment array checksum mismatch: {path}")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(value.shape) != list(description["shape"]):
        raise ValueError(f"assignment array shape mismatch: {path}")
    if np.dtype(value.dtype).str != str(description["dtype"]):
        raise ValueError(f"assignment array dtype mismatch: {path}")
    return value


def _read_inference_manifest(root: Path) -> dict:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"winding-inference manifest is missing: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("artifact_type") != INFERENCE_ARTIFACT_TYPE:
        raise ValueError(f"not a winding-inference crossing store: {root}")
    if int(manifest.get("format_version", -1)) != INFERENCE_FORMAT_VERSION:
        raise ValueError(
            "unsupported winding-inference format version: "
            f"{manifest.get('format_version')!r}")
    if manifest.get("coordinate_order") != "zyx":
        raise ValueError("winding-inference coordinates must use zyx order")
    identity = copy.deepcopy(manifest)
    claimed = identity.pop("fingerprint", None)
    identity.pop("elapsed_seconds", None)
    identity.pop("export_workers", None)
    identity.pop("rays_per_task", None)
    for shard in identity.get("shards", []):
        shard.pop("elapsed_seconds", None)
    if claimed != _canonical_digest(identity):
        raise ValueError("winding-inference manifest fingerprint mismatch")
    return manifest


def _load_source_array(root: Path, description: Mapping, *, verify: bool):
    path = root / description["file"]
    if verify and _sha256(path) != description["sha256"]:
        raise ValueError(f"winding-inference array checksum mismatch: {path}")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(value.shape) != list(description["shape"]):
        raise ValueError(f"winding-inference array shape mismatch: {path}")
    if np.dtype(value.dtype).str != str(description["dtype"]):
        raise ValueError(f"winding-inference array dtype mismatch: {path}")
    return value


def _load_verified_patch(entry: Path):
    """Process-pool entry point; keep it top-level for spawn pickling."""
    # The attachment index needs only coordinates, scale, and validity. Avoid
    # decoding/transferring winding.tif and overlap metadata for every patch.
    patch = load_tifxyz(entry, geometry_only=True)
    # Returning Torch tensors makes multiprocessing use its shared-memory
    # resource sharer (one handle per tensor/patch). A plain NumPy payload is
    # more robust for thousands of patches and is reconstructed immediately in
    # the parent.
    return (
        entry.name,
        patch.zyxs.numpy(),
        patch.scale.numpy(),
        patch.uuid,
        patch.erosion_cells_override,
    )


def _patch_from_worker_payload(payload):
    patch_id, zyxs, scale, uuid, erosion_cells_override = payload
    return patch_id, Patch(
        torch.from_numpy(zyxs),
        torch.from_numpy(scale),
        None,
        None,
        uuid,
        erosion_cells_override,
    )


def _load_verified_patches(path: Path, *, workers=0):
    # This directory is commonly on NFS. Do one readdir and let workers open
    # meta.json themselves: entry.is_dir()/meta.is_file() here would issue
    # thousands of serial metadata RPCs before the process pool even exists.
    print(f"listing verified patches in {path}", flush=True)
    entries = [path / name for name in sorted(os.listdir(path))]
    requested_workers = int(workers)
    if requested_workers < 0:
        raise ValueError("patch workers must be non-negative")
    worker_count = min(
        len(entries),
        requested_workers or min(8, os.cpu_count() or 1),
    )
    patches = {}
    failed = 0
    if worker_count <= 1:
        for entry in tqdm(entries, desc="loading verified patches"):
            try:
                patch_id, patch = _patch_from_worker_payload(
                    _load_verified_patch(entry))
                patches[patch_id] = patch
            except Exception as exc:
                failed += 1
                print(f"WARNING: failed to load patch {entry.name}: {exc}")
    else:
        print(
            f"loading {len(entries):,} verified patches with "
            f"{worker_count} worker processes",
            flush=True,
        )
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
                max_workers=worker_count, mp_context=context) as executor:
            futures = {
                executor.submit(_load_verified_patch, entry): entry
                for entry in entries
            }
            for future in tqdm(
                    as_completed(futures), total=len(futures),
                    desc="loading verified patches"):
                entry = futures[future]
                try:
                    patch_id, patch = _patch_from_worker_payload(
                        future.result())
                    patches[patch_id] = patch
                except Exception as exc:
                    failed += 1
                    print(f"WARNING: failed to load patch {entry.name}: {exc}")
    if not patches:
        raise RuntimeError(f"no verified patches found in {path}")
    print(
        f"loaded {len(patches):,} verified patches ({failed:,} failed)",
        flush=True,
    )
    return {patch_id: patches[patch_id] for patch_id in sorted(patches)}


def _publish_directory(temporary: Path, destination: Path, *, force: bool):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        raise FileExistsError(
            f"assignment output already exists: {destination}; pass --force to replace it")
    backup = None
    try:
        if destination.exists():
            backup = destination.with_name(
                f".{destination.name}.replaced-{os.getpid()}")
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(destination, backup)
        os.replace(temporary, destination)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise


def build_winding_patch_assignments(
    winding_inference_path,
    verified_patches_path,
    output_path,
    *,
    tolerance=2.5,
    chunk_size=250_000,
    patch_workers=0,
    force=False,
    verify=True,
):
    """Build a sparse crossing-to-patch attachment artifact."""
    tolerance = float(tolerance)
    chunk_size = int(chunk_size)
    patch_workers = int(patch_workers)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a finite number greater than zero")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if patch_workers < 0:
        raise ValueError("patch_workers must be non-negative")

    source_root = Path(winding_inference_path).resolve()
    patch_root = Path(verified_patches_path).resolve()
    destination = Path(output_path).resolve()
    if destination.exists() and not force:
        raise FileExistsError(
            f"assignment output already exists: {destination}; "
            "pass --force to replace it")
    print(f"reading winding-inference manifest from {source_root}", flush=True)
    source_manifest = _read_inference_manifest(source_root)
    patches = _load_verified_patches(patch_root, workers=patch_workers)
    print(
        f"building native surface index over {len(patches):,} patches",
        flush=True,
    )
    built_index = build_surface_patch_index(patches, tolerance)
    if built_index is None:
        raise RuntimeError(
            "winding patch preprocessing requires the vc.surface_index backend")
    _, surface_ids = built_index

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.building-", dir=destination.parent))
    crossing_base = 0
    total_attached = 0
    output_shards = []
    try:
        for shard_number, source_shard in enumerate(
                tqdm(source_manifest["shards"], desc="attaching inference shards")):
            source_shard_root = source_root / source_shard["name"]
            arrays = source_shard["arrays"]
            origins = _load_source_array(
                source_shard_root, arrays["ray_origin_zyx"], verify=verify)
            steps = _load_source_array(
                source_shard_root, arrays["ray_step_zyx"], verify=verify)
            crossing_t = _load_source_array(
                source_shard_root, arrays["crossing_t"], verify=verify)
            offsets = _load_source_array(
                source_shard_root, arrays["crossing_offsets"], verify=verify)
            if len(offsets) != len(origins) + 1 or int(offsets[-1]) != len(crossing_t):
                raise ValueError(
                    f"winding-inference ray arrays disagree: {source_shard_root}")

            attached_index_parts = []
            attached_patch_parts = []
            attached_ij_parts = []
            attached_distance_parts = []
            with tqdm(
                    total=len(crossing_t),
                    desc=f"crossings in {source_shard['name']}",
                    unit="crossing", unit_scale=True, leave=False) as progress:
                for begin in range(0, len(crossing_t), chunk_size):
                    end = min(begin + chunk_size, len(crossing_t))
                    flat = np.arange(begin, end, dtype=np.int64)
                    ray = np.searchsorted(offsets[1:], flat, side="right")
                    points = (
                        np.asarray(origins[ray], dtype=np.float32)
                        + np.asarray(
                            crossing_t[begin:end], dtype=np.float32)[:, None]
                        * np.asarray(steps[ray], dtype=np.float32)
                    )
                    patch_index, ij, distance, chunk_surface_ids = (
                        locate_points_on_patches(
                            patches, points, tolerance,
                            built_index=built_index,
                            general_hit_policy="largest_area",
                        )
                    )
                    if list(chunk_surface_ids) != list(surface_ids):
                        raise RuntimeError(
                            "surface-index patch ordering changed during build")
                    keep = patch_index >= 0
                    if keep.any():
                        attached_index_parts.append(flat[keep])
                        attached_patch_parts.append(patch_index[keep])
                        attached_ij_parts.append(ij[keep])
                        attached_distance_parts.append(distance[keep])
                    progress.update(end - begin)

            def concatenate(parts, shape, dtype):
                if not parts:
                    return np.empty(shape, dtype=dtype)
                return np.concatenate(parts).astype(dtype, copy=False)

            local_index = concatenate(
                attached_index_parts, (0,), np.int64)
            patch_index = concatenate(
                attached_patch_parts, (0,), np.int32)
            patch_ij = concatenate(
                attached_ij_parts, (0, 2), np.float32)
            distance = concatenate(
                attached_distance_parts, (0,), np.float32)
            shard_name = f"shard-{shard_number:05d}"
            shard_root = temporary / shard_name
            shard_root.mkdir()
            stored_arrays = {}
            for name, value in (
                ("crossing_index", local_index),
                ("patch_index", patch_index),
                ("patch_ij", patch_ij),
                ("distance", distance),
            ):
                array_path = shard_root / f"{name}.npy"
                np.save(array_path, value, allow_pickle=False)
                stored_arrays[name] = _array_description(
                    array_path, shard_root, value)
            output_shards.append({
                "name": shard_name,
                "source_shard": source_shard["name"],
                "source_crossing_base": crossing_base,
                "num_source_rays": len(origins),
                "num_source_crossings": len(crossing_t),
                "num_attached": len(local_index),
                "arrays": stored_arrays,
            })
            crossing_base += len(crossing_t)
            total_attached += len(local_index)

        manifest = {
            "artifact_type": ARTIFACT_TYPE,
            "format_version": FORMAT_VERSION,
            "coordinate_order": "zyx",
            "source_winding_inference_fingerprint": source_manifest["fingerprint"],
            "attachment_tolerance": tolerance,
            "hit_policy": "largest_area",
            "patch_ids": list(surface_ids),
            "num_source_crossings": crossing_base,
            "num_attached": total_attached,
            "shards": output_shards,
        }
        manifest["fingerprint"] = _canonical_digest(manifest)
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        _publish_directory(temporary, destination, force=force)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)

    print(
        f"wrote {destination}: {total_attached:,} / {crossing_base:,} "
        f"crossings attached to {len(surface_ids):,} patches")
    return destination


class PreparedWindingPatchAssignments:
    """Fit-time, per-assignment validated compact relative-winding source."""

    def __init__(
        self,
        *,
        crossing_index,
        patch_index,
        patch_ij,
        level,
        ray,
        source_local,
        patch_ids,
        eligible_ray_ids,
        ray_patch_offsets,
        ray_patch_indices,
        patch_assignment_offsets,
        patch_assignment_rows,
        span_first,
        span_last,
        stats,
        fingerprint,
    ):
        self.crossing_index = crossing_index
        self.patch_index = patch_index
        self.patch_ij = patch_ij
        self.level = level
        self.ray = ray
        self.source_local = source_local
        self.patch_ids = tuple(patch_ids)
        self.eligible_ray_ids = eligible_ray_ids
        self.ray_patch_offsets = ray_patch_offsets
        self.ray_patch_indices = ray_patch_indices
        self.patch_assignment_offsets = patch_assignment_offsets
        self.patch_assignment_rows = patch_assignment_rows
        self.span_first = span_first
        self.span_last = span_last
        self.stats = dict(stats)
        self.fingerprint = fingerprint
        self._topology_node_start = None
        self._topology_offsets = None

    @property
    def num_eligible_rays(self):
        return len(self.eligible_ray_ids)

    def register_theta_topology(self, crossing_map, winding_store):
        lengths = self.span_last - self.span_first + 1
        offsets = np.empty(len(lengths) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        total = int(offsets[-1])
        if total == 0:
            self._topology_node_start = None
            self._topology_offsets = offsets
            return
        flat_parts = [
            np.arange(first, last + 1, dtype=np.int64)
            for first, last in zip(self.span_first, self.span_last)
        ]
        flat_indices = torch.from_numpy(np.concatenate(flat_parts)).to(
            device=winding_store.device)
        start = crossing_map.register_nodes(
            total,
            lambda local, indices=flat_indices, store=winding_store:
                store.materialize_flat(indices[local]),
        )
        edge_keep = np.ones(max(0, total - 1), dtype=bool)
        boundaries = offsets[1:-1] - 1
        edge_keep[boundaries] = False
        edge_left = np.arange(total - 1, dtype=np.int64)[edge_keep] + start
        if len(edge_left):
            crossing_map.register_edges(
                np.stack([edge_left, edge_left + 1], axis=1))
        self._topology_node_start = start
        self._topology_offsets = offsets

    def _rows_for_patch_group(self, group_index):
        begin = self.patch_assignment_offsets[group_index]
        end = self.patch_assignment_offsets[group_index + 1]
        return self.patch_assignment_rows[begin:end]

    def sample_pair_requests(self, cfg):
        """Return the common patch-relative evaluator's request tuples."""
        if self._topology_node_start is None or not self.num_eligible_rays:
            return []
        count = min(
            int(cfg["sample_count_winding_model_patch_relative_rays"]),
            self.num_eligible_rays,
        )
        if count <= 0:
            return []
        selected = np.random.choice(
            self.num_eligible_rays, count, replace=False)
        pairs_per_ray = int(cfg["sample_count_winding_model_patch_pairs_per_ray"])
        min_delta, max_delta = map(
            int, cfg["winding_model_relative_pair_delta"])
        requests = []
        for eligible_index in selected:
            patch_begin = self.ray_patch_offsets[eligible_index]
            patch_end = self.ray_patch_offsets[eligible_index + 1]
            group_indices = np.arange(patch_begin, patch_end, dtype=np.int64)
            if cfg["pcl_rel_winding_adjacent_patches_only"]:
                group_pairs = list(zip(group_indices[:-1], group_indices[1:]))
            else:
                group_pairs = [
                    (group_indices[a], group_indices[b])
                    for a in range(len(group_indices))
                    for b in range(a + 1, len(group_indices))
                ]
            candidates = []
            for group_a, group_b in group_pairs:
                rows_a = self._rows_for_patch_group(group_a)
                rows_b = self._rows_for_patch_group(group_b)
                a_grid = np.repeat(rows_a, len(rows_b))
                b_grid = np.tile(rows_b, len(rows_a))
                separation = (
                    self.source_local[b_grid] - self.source_local[a_grid])
                target = self.level[b_grid] - self.level[a_grid]
                valid = (
                    (separation >= min_delta)
                    & (separation <= max_delta)
                    & (target != 0)
                )
                if valid.any():
                    candidates.append((group_a, group_b, a_grid[valid], b_grid[valid]))
            if not candidates or pairs_per_ray <= 0:
                continue
            chosen = np.random.choice(
                len(candidates), min(pairs_per_ray, len(candidates)),
                replace=False)
            span_first = self.span_first[eligible_index]
            topology_base = (
                self._topology_node_start
                + self._topology_offsets[eligible_index])
            for candidate_index in chosen:
                group_a, group_b, rows_a, rows_b = candidates[candidate_index]
                choice = np.random.randint(len(rows_a))
                row_a, row_b = int(rows_a[choice]), int(rows_b[choice])
                flat_a = int(self.crossing_index[row_a])
                flat_b = int(self.crossing_index[row_b])
                node_a = topology_base + flat_a - span_first
                node_b = topology_base + flat_b - span_first
                chain_nodes = np.arange(node_a, node_b + 1, dtype=np.int64)
                patch_a = self.patch_ids[int(self.patch_index[row_a])]
                patch_b = self.patch_ids[int(self.patch_index[row_b])]
                ij_a = self.patch_ij[row_a]
                ij_b = self.patch_ij[row_b]
                requests.append((
                    (patch_a, int(ij_a[0]), int(ij_a[1])),
                    (patch_b, int(ij_b[0]), int(ij_b[1])),
                    patch_a,
                    patch_b,
                    float(self.level[row_b] - self.level[row_a]),
                    chain_nodes,
                    node_a,
                    node_b,
                ))
        return requests


def _prepare_compact_groups(
    crossing_index,
    patch_index,
    patch_ij,
    level,
    ray,
    source_local,
    patch_ids,
    *,
    stats,
    fingerprint,
):
    if len(crossing_index):
        order = np.argsort(crossing_index, kind="stable")
        crossing_index = crossing_index[order]
        patch_index = patch_index[order]
        patch_ij = patch_ij[order]
        level = level[order]
        ray = ray[order]
        source_local = source_local[order]

        group_order = np.lexsort((level, patch_index, ray))
        sorted_ray = ray[group_order]
        sorted_patch = patch_index[group_order]
        sorted_level = level[group_order]
        starts = np.r_[0, np.flatnonzero(
            (sorted_ray[1:] != sorted_ray[:-1])
            | (sorted_patch[1:] != sorted_patch[:-1])) + 1]
        ends = np.r_[starts[1:], len(group_order)]
        ambiguous_sorted = np.zeros(len(group_order), dtype=bool)
        for begin, end in zip(starts, ends):
            if sorted_level[begin] != sorted_level[end - 1]:
                ambiguous_sorted[begin:end] = True
        ambiguous = np.zeros(len(group_order), dtype=bool)
        ambiguous[group_order] = ambiguous_sorted
        stats["ambiguous_level"] = int(ambiguous.sum())
        keep = ~ambiguous
        crossing_index = crossing_index[keep]
        patch_index = patch_index[keep]
        patch_ij = patch_ij[keep]
        level = level[keep]
        ray = ray[keep]
        source_local = source_local[keep]
    else:
        stats["ambiguous_level"] = 0

    eligible_ray_ids = []
    ray_patch_offsets = [0]
    ray_patch_indices = []
    patch_assignment_offsets = [0]
    patch_assignment_rows = []
    span_first = []
    span_last = []
    if len(crossing_index):
        unique_rays, ray_starts = np.unique(ray, return_index=True)
        ray_ends = np.r_[ray_starts[1:], len(ray)]
        for ray_id, begin, end in zip(unique_rays, ray_starts, ray_ends):
            rows_by_patch = {}
            for row in range(begin, end):
                rows_by_patch.setdefault(int(patch_index[row]), []).append(row)
            if len(rows_by_patch) < 2:
                continue
            eligible_ray_ids.append(ray_id)
            span_first.append(int(crossing_index[begin]))
            span_last.append(int(crossing_index[end - 1]))
            for current_patch, rows in rows_by_patch.items():
                ray_patch_indices.append(current_patch)
                patch_assignment_rows.extend(rows)
                patch_assignment_offsets.append(len(patch_assignment_rows))
            ray_patch_offsets.append(len(ray_patch_indices))

    stats["retained"] = len(crossing_index)
    stats["eligible_rays"] = len(eligible_ray_ids)
    return PreparedWindingPatchAssignments(
        crossing_index=np.asarray(crossing_index, dtype=np.int64),
        patch_index=np.asarray(patch_index, dtype=np.int32),
        patch_ij=np.asarray(patch_ij, dtype=np.float32),
        level=np.asarray(level, dtype=np.int32),
        ray=np.asarray(ray, dtype=np.int64),
        source_local=np.asarray(source_local, dtype=np.int64),
        patch_ids=patch_ids,
        eligible_ray_ids=np.asarray(eligible_ray_ids, dtype=np.int64),
        ray_patch_offsets=np.asarray(ray_patch_offsets, dtype=np.int64),
        ray_patch_indices=np.asarray(ray_patch_indices, dtype=np.int32),
        patch_assignment_offsets=np.asarray(
            patch_assignment_offsets, dtype=np.int64),
        patch_assignment_rows=np.asarray(patch_assignment_rows, dtype=np.int64),
        span_first=np.asarray(span_first, dtype=np.int64),
        span_last=np.asarray(span_last, dtype=np.int64),
        stats=stats,
        fingerprint=fingerprint,
    )


def load_winding_patch_assignments(
    path,
    winding_store,
    patches,
    *,
    verify=True,
    validation_chunk_size=250_000,
):
    """Load an artifact and discard only assignments stale for this fit."""
    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"winding patch assignment manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError(f"not a winding patch assignment store: {root}")
    if int(manifest.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(
            f"unsupported winding patch assignment version: "
            f"{manifest.get('format_version')!r}")
    identity = copy.deepcopy(manifest)
    claimed = identity.pop("fingerprint", None)
    if claimed != _canonical_digest(identity):
        raise ValueError("winding patch assignment manifest fingerprint mismatch")
    if (manifest.get("source_winding_inference_fingerprint")
            != winding_store.fingerprint["fingerprint"]):
        raise ValueError(
            "assignment source fingerprint does not match the loaded "
            "winding-inference store")
    tolerance = float(manifest["attachment_tolerance"])
    artifact_patch_ids = list(manifest["patch_ids"])
    current_patch_ids = list(patches)
    current_patch_index = {patch_id: idx
                           for idx, patch_id in enumerate(current_patch_ids)}

    crossing_parts = []
    artifact_patch_parts = []
    ij_parts = []
    for shard in manifest["shards"]:
        shard_root = root / shard["name"]
        arrays = shard["arrays"]
        local = np.asarray(_load_array(
            shard_root, arrays["crossing_index"], verify=verify))
        artifact_patch = np.asarray(_load_array(
            shard_root, arrays["patch_index"], verify=verify))
        ij = np.asarray(_load_array(
            shard_root, arrays["patch_ij"], verify=verify))
        distance = np.asarray(_load_array(
            shard_root, arrays["distance"], verify=verify))
        count = int(shard["num_attached"])
        if not (len(local) == len(artifact_patch) == len(ij) == len(distance) == count):
            raise ValueError(f"assignment shard arrays disagree: {shard_root}")
        if (len(local) and
                (local.min() < 0
                 or local.max() >= int(shard["num_source_crossings"]))):
            raise ValueError(f"assignment crossing index is out of range: {shard_root}")
        if (len(artifact_patch) and
                (artifact_patch.min() < 0
                 or artifact_patch.max() >= len(artifact_patch_ids))):
            raise ValueError(f"assignment patch index is out of range: {shard_root}")
        crossing_parts.append(
            local.astype(np.int64, copy=False)
            + int(shard["source_crossing_base"]))
        artifact_patch_parts.append(artifact_patch.astype(np.int32, copy=False))
        ij_parts.append(ij.astype(np.float32, copy=False))

    crossing_index = (
        np.concatenate(crossing_parts) if crossing_parts
        else np.empty(0, dtype=np.int64))
    artifact_patch_index = (
        np.concatenate(artifact_patch_parts) if artifact_patch_parts
        else np.empty(0, dtype=np.int32))
    patch_ij = (
        np.concatenate(ij_parts) if ij_parts
        else np.empty((0, 2), dtype=np.float32))
    stats = {
        "stored": len(crossing_index),
        "missing_patch": 0,
        "invalid_ij": 0,
        "geometry_mismatch": 0,
    }
    mapped_patch_index = np.fromiter((
        current_patch_index.get(artifact_patch_ids[int(index)], -1)
        for index in artifact_patch_index
    ), dtype=np.int32, count=len(artifact_patch_index))
    present = mapped_patch_index >= 0
    stats["missing_patch"] = int((~present).sum())
    crossing_index = crossing_index[present]
    mapped_patch_index = mapped_patch_index[present]
    patch_ij = patch_ij[present]

    valid = np.zeros(len(crossing_index), dtype=bool)
    for current_index, patch_id in enumerate(current_patch_ids):
        rows = np.flatnonzero(mapped_patch_index == current_index)
        if not len(rows):
            continue
        patch = patches[patch_id]
        for begin in range(0, len(rows), validation_chunk_size):
            chunk_rows = rows[begin:begin + validation_chunk_size]
            ijs = torch.from_numpy(patch_ij[chunk_rows])
            current_zyx, ij_valid = patch.ij_to_zyx(ijs)
            source_parts = []
            for source_begin in range(0, len(chunk_rows), validation_chunk_size):
                indices = torch.from_numpy(
                    crossing_index[chunk_rows[source_begin:
                                              source_begin + validation_chunk_size]])
                source_parts.append(
                    winding_store.materialize_flat(
                        indices.to(winding_store.device)).detach().cpu())
            source_zyx = torch.cat(source_parts, dim=0)
            close = torch.linalg.vector_norm(
                current_zyx - source_zyx, dim=-1) <= tolerance + 1e-4
            valid[chunk_rows] = (ij_valid & close).numpy()
            stats["invalid_ij"] += int((~ij_valid).sum().item())
            stats["geometry_mismatch"] += int(
                (ij_valid & ~close).sum().item())

    crossing_index = crossing_index[valid]
    mapped_patch_index = mapped_patch_index[valid]
    patch_ij = patch_ij[valid]
    if len(crossing_index):
        flat_gpu = torch.from_numpy(crossing_index).to(winding_store.device)
        ray_gpu = torch.searchsorted(winding_store.offset[1:], flat_gpu,
                                     right=True)
        level = winding_store.crossing_level[flat_gpu].detach().cpu().numpy()
        ray = ray_gpu.detach().cpu().numpy()
        starts = winding_store.offset[ray_gpu].detach().cpu().numpy()
        source_local = crossing_index - starts
    else:
        level = np.empty(0, dtype=np.int32)
        ray = np.empty(0, dtype=np.int64)
        source_local = np.empty(0, dtype=np.int64)

    prepared = _prepare_compact_groups(
        crossing_index,
        mapped_patch_index,
        patch_ij,
        level,
        ray,
        source_local,
        current_patch_ids,
        stats=stats,
        fingerprint={
            "artifact_type": ARTIFACT_TYPE,
            "format_version": FORMAT_VERSION,
            "fingerprint": claimed,
        },
    )
    summary = ", ".join(f"{key}={value:,}" for key, value in stats.items())
    print(f"winding patch assignments: {summary}")
    return prepared
