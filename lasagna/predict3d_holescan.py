#!/usr/bin/env python3
"""Scan predict3d OME-Zarr outputs for page-release zero bands.

The bug fixed by ScrollPrize/villa#1192 could leave output chunk files present
while one channel contained zeroed Z bands.  This scanner therefore measures
actual per-slice nonzero density inside each Y/X chunk column and reports low
or empty Z runs that are bracketed by populated slices in the same column.
"""
from __future__ import annotations

import argparse
import math
import json
import multiprocessing
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from dtypes import numpy_float_hi
from scipy import ndimage
import tensorstore as ts
import zarr


DEFAULT_CHANNELS = ("grad_mag",)
ProgressCallback = Callable[[str, int, int, str], None]


def _default_workers() -> int:
	return os.cpu_count() or 1


@dataclass(frozen=True)
class ScanTarget:
	name: str
	path: Path
	channel_index: int | None = None
	scaledown: int | None = None


@dataclass(frozen=True)
class HoleFinding:
	channel: str
	path: str
	z0: int
	z1: int
	y0: int
	y1: int
	x0: int
	x1: int
	min_density: float
	max_density: float
	before_density: float
	after_density: float
	zero_slices: int
	low_slices: int
	center_xyz: tuple[int, int, int] | None = None


ChunkCoord = tuple[int, int, int]

FACE_Z0 = 0
FACE_Z1 = 1
FACE_Y0 = 2
FACE_Y1 = 3
FACE_X0 = 4
FACE_X1 = 5
FACES = (FACE_Z0, FACE_Z1, FACE_Y0, FACE_Y1, FACE_X0, FACE_X1)
OPPOSITE_FACE = {
	FACE_Z0: FACE_Z1,
	FACE_Z1: FACE_Z0,
	FACE_Y0: FACE_Y1,
	FACE_Y1: FACE_Y0,
	FACE_X0: FACE_X1,
	FACE_X1: FACE_X0,
}
FACE_DELTAS: dict[int, ChunkCoord] = {
	FACE_Z0: (-1, 0, 0),
	FACE_Z1: (1, 0, 0),
	FACE_Y0: (0, -1, 0),
	FACE_Y1: (0, 1, 0),
	FACE_X0: (0, 0, -1),
	FACE_X1: (0, 0, 1),
}

CHUNK_MISSING = "MISSING"
CHUNK_ALL_NONZERO = "ALL_NONZERO"
CHUNK_ALL_ZERO = "ALL_ZERO"
CHUNK_MIXED_NO_EDGE_ZERO = "MIXED_NO_EDGE_ZERO"
CHUNK_MIXED_EDGE_ZERO = "MIXED_EDGE_ZERO"


@dataclass(frozen=True)
class ChunkAnalysis:
	coord: ChunkCoord
	status: str
	component_count: int = 0
	face_labels: tuple[tuple[int, ...], ...] = ((), (), (), (), (), ())
	face_arrays: tuple[np.ndarray | None, ...] = (None, None, None, None, None, None)


@dataclass
class FloodFilter:
	target: ScanTarget
	shape: tuple[int, int, int]
	chunks: tuple[int, int, int]
	grid_shape: tuple[int, int, int]
	analyses: dict[ChunkCoord, ChunkAnalysis]
	reached: set[tuple[ChunkCoord, int]]
	suspicious_masks: dict[ChunkCoord, np.ndarray]


FindingCallback = Callable[[HoleFinding], None]

_WORKER_TARGET: ScanTarget | None = None
_WORKER_SHAPE: tuple[int, int, int] | None = None
_WORKER_CHUNKS: tuple[int, int, int] | None = None
_WORKER_CHANNEL_INDEX: int | None = None
_WORKER_STORE = None


def _is_zarr_array(obj: object) -> bool:
	return hasattr(obj, "shape") and hasattr(obj, "dtype")


def _open_zarr_array(path: Path):
	obj = zarr.open(str(path), mode="r")
	if _is_zarr_array(obj):
		return obj
	keys = sorted(str(k) for k in obj.keys())
	numeric = [k for k in keys if k.isdigit()]
	if numeric:
		return obj[numeric[0]]
	if len(keys) == 1 and _is_zarr_array(obj[keys[0]]):
		return obj[keys[0]]
	raise ValueError(f"{path} is a zarr group; pass a concrete array or numeric OME-Zarr level")


def _load_manifest_targets(
	manifest_path: Path,
	*,
	channels: set[str] | None,
) -> list[ScanTarget]:
	with manifest_path.open("r", encoding="utf-8") as handle:
		raw = json.load(handle)
	groups = raw.get("groups", {})
	if not isinstance(groups, dict):
		raise ValueError(f"{manifest_path} has no groups object")

	targets: list[ScanTarget] = []
	for group_name, group in groups.items():
		if not isinstance(group, dict):
			continue
		group_channels = [str(v) for v in group.get("channels", [])]
		if not group_channels:
			group_channels = [str(group_name)]
		zarr_rel = group.get("zarr")
		if not zarr_rel:
			continue
		for ch_idx, ch_name in enumerate(group_channels):
			if channels is not None and ch_name not in channels and str(group_name) not in channels:
				continue
			targets.append(
				ScanTarget(
					name=ch_name,
					path=(manifest_path.parent / str(zarr_rel)).resolve(),
					channel_index=ch_idx if len(group_channels) > 1 else None,
					scaledown=int(group["scaledown"]) if "scaledown" in group else None,
				)
			)
	return targets


def discover_targets(path: Path, *, channels: Iterable[str] | None = None) -> list[ScanTarget]:
	channel_set = {str(ch) for ch in channels} if channels else None
	if path.name.endswith(".lasagna.json"):
		targets = _load_manifest_targets(path, channels=channel_set)
		if not targets:
			raise ValueError(f"no matching zarr groups found in {path}")
		return targets

	arr = _open_zarr_array(path)
	shape = tuple(int(v) for v in arr.shape)
	if len(shape) == 3:
		name = path.parent.name if path.name.isdigit() else path.name
		return [ScanTarget(name=name, path=path.resolve())]
	if len(shape) == 4:
		attrs = dict(getattr(arr, "attrs", {}) or {})
		params = attrs.get("preprocess_params", {}) if isinstance(attrs.get("preprocess_params", {}), dict) else {}
		names = [str(v) for v in params.get("channels", [])]
		if len(names) != shape[0]:
			names = [f"ch{idx}" for idx in range(shape[0])]
		targets = [
			ScanTarget(name=name, path=path.resolve(), channel_index=idx)
			for idx, name in enumerate(names)
			if channel_set is None or name in channel_set
		]
		if not targets:
			raise ValueError(f"no matching channels found in {path}")
		return targets
	raise ValueError(f"expected 3D or channel-first 4D zarr at {path}, got shape={shape}")


def _target_array(target: ScanTarget):
	arr = _open_zarr_array(target.path)
	shape = tuple(int(v) for v in arr.shape)
	if len(shape) == 3:
		if target.channel_index not in (None, 0):
			raise ValueError(f"{target.path} is 3D but target requests channel {target.channel_index}")
		return arr, shape, None
	if len(shape) == 4:
		ch = 0 if target.channel_index is None else int(target.channel_index)
		if ch < 0 or ch >= shape[0]:
			raise ValueError(f"channel {ch} out of range for {target.path} shape={shape}")
		return arr, shape[1:], ch
	raise ValueError(f"expected 3D or channel-first 4D zarr at {target.path}, got shape={shape}")


def _target_spatial_chunks(arr, channel_index: int | None) -> tuple[int, int, int] | None:
	chunks = tuple(int(v) for v in getattr(arr, "chunks", ()) or ())
	if channel_index is None:
		return chunks[:3] if len(chunks) >= 3 else None
	return chunks[1:4] if len(chunks) >= 4 else None


def _tensorstore_context() -> ts.Context:
	return ts.Context({
		"cache_pool": {"total_bytes_limit": 512 << 20},
		"file_io_concurrency": {"limit": 4},
		"data_copy_concurrency": {"limit": 1},
	})


def _open_tensorstore_array(path: Path):
	return ts.open(
		{
			"driver": "zarr",
			"kvstore": {"driver": "file", "path": str(path)},
		},
		context=_tensorstore_context(),
		open=True,
		read=True,
		recheck_cached_data="open",
	).result()


def _init_tensorstore_worker(
	target: ScanTarget,
	shape: tuple[int, int, int] | None,
	chunks: tuple[int, int, int] | None,
) -> None:
	global _WORKER_TARGET, _WORKER_SHAPE, _WORKER_CHUNKS, _WORKER_CHANNEL_INDEX, _WORKER_STORE
	_arr, target_shape, channel_index = _target_array(target)
	_WORKER_TARGET = target
	_WORKER_SHAPE = tuple(shape) if shape is not None else target_shape
	_WORKER_CHUNKS = tuple(chunks) if chunks is not None else None
	_WORKER_CHANNEL_INDEX = channel_index
	_WORKER_STORE = _open_tensorstore_array(target.path)


def _worker_store():
	if _WORKER_STORE is None:
		raise RuntimeError("TensorStore worker was not initialized")
	return _WORKER_STORE


def _read_ts_region(
	store,
	channel_index: int | None,
	z0: int,
	z1: int,
	y0: int,
	y1: int,
	x0: int,
	x1: int,
) -> np.ndarray:
	if channel_index is None:
		return np.asarray(store[z0:z1, y0:y1, x0:x1].read().result())
	return np.asarray(store[channel_index, z0:z1, y0:y1, x0:x1].read().result())


def _count_nonzero_by_z(
	store,
	channel_index: int | None,
	z0: int,
	z1: int,
	y0: int,
	y1: int,
	x0: int,
	x1: int,
) -> np.ndarray:
	data = _read_ts_region(store, channel_index, z0, z1, y0, y1, x0, x1)
	if data.size == 0:
		return np.zeros((max(0, z1 - z0),), dtype=np.int64)
	return np.count_nonzero(data.reshape((data.shape[0], -1)), axis=1).astype(np.int64)


def _iter_runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
	i = 0
	n = int(mask.shape[0])
	while i < n:
		if not bool(mask[i]):
			i += 1
			continue
		j = i + 1
		while j < n and bool(mask[j]):
			j += 1
		yield i, j
		i = j


def _bracket_density(density: np.ndarray, start: int, end: int, window: int) -> tuple[float, float]:
	before = density[max(0, start - window):start]
	after = density[end:min(int(density.shape[0]), end + window)]
	before_pop = before[before > 0]
	after_pop = after[after > 0]
	before_med = float(np.median(before_pop)) if before_pop.size else 0.0
	after_med = float(np.median(after_pop)) if after_pop.size else 0.0
	return before_med, after_med


def _find_holes_in_counts(
	*,
	target: ScanTarget,
	counts: np.ndarray,
	y0: int,
	y1: int,
	x0: int,
	x1: int,
	bracket_window: int,
	min_neighbor_density: float,
	drop_ratio: float,
	max_low_density: float,
) -> list[HoleFinding]:
	voxels_per_slice = max(1, (y1 - y0) * (x1 - x0))
	density = counts.astype(numpy_float_hi) / float(voxels_per_slice)
	findings: list[HoleFinding] = []
	for start, end in _iter_runs(density <= max_low_density):
		before_med, after_med = _bracket_density(density, start, end, bracket_window)
		neighbor = min(before_med, after_med)
		if neighbor < min_neighbor_density:
			continue
		low_threshold = max(max_low_density, neighbor * drop_ratio)
		expanded_start = start
		expanded_end = end
		while expanded_start > 0 and density[expanded_start - 1] <= low_threshold:
			expanded_start -= 1
		while expanded_end < int(density.shape[0]) and density[expanded_end] <= low_threshold:
			expanded_end += 1
		run_density = density[expanded_start:expanded_end]
		center_xyz = None
		if target.scaledown is not None:
			scale = 2 ** int(target.scaledown)
			center_xyz = (
				int(((x0 + x1) // 2) * scale),
				int(((y0 + y1) // 2) * scale),
				int(((expanded_start + expanded_end) // 2) * scale),
			)
		findings.append(
			HoleFinding(
				channel=target.name,
				path=str(target.path),
				z0=int(expanded_start),
				z1=int(expanded_end),
				y0=int(y0),
				y1=int(y1),
				x0=int(x0),
				x1=int(x1),
				min_density=float(run_density.min()) if run_density.size else 0.0,
				max_density=float(run_density.max()) if run_density.size else 0.0,
				before_density=float(before_med),
				after_density=float(after_med),
				zero_slices=int(np.count_nonzero(counts[expanded_start:expanded_end] == 0)),
				low_slices=int(expanded_end - expanded_start),
				center_xyz=center_xyz,
			)
		)
	return findings


def _scan_block_job(
	target: ScanTarget,
	z_size: int,
	y0: int,
	y1: int,
	x0: int,
	x1: int,
	slab_z: int,
	bracket_window: int,
	min_neighbor_density: float,
	drop_ratio: float,
	max_low_density: float,
) -> list[HoleFinding]:
	_arr, _shape, channel_index = _target_array(target)
	store = _open_tensorstore_array(target.path)
	counts = np.zeros((z_size,), dtype=np.int64)
	for z0 in range(0, z_size, slab_z):
		z1 = min(z_size, z0 + slab_z)
		counts[z0:z1] = _count_nonzero_by_z(store, channel_index, z0, z1, y0, y1, x0, x1)
	return _find_holes_in_counts(
		target=target,
		counts=counts,
		y0=y0,
		y1=y1,
		x0=x0,
		x1=x1,
		bracket_window=bracket_window,
		min_neighbor_density=min_neighbor_density,
		drop_ratio=drop_ratio,
		max_low_density=max_low_density,
	)


def _scan_block_worker(args: tuple[int, int, int, int, int, int, int, float, float, float]) -> list[HoleFinding]:
	if _WORKER_TARGET is None:
		raise RuntimeError("TensorStore worker was not initialized")
	(
		z_size,
		y0,
		y1,
		x0,
		x1,
		slab_z,
		bracket_window,
		min_neighbor_density,
		drop_ratio,
		max_low_density,
	) = args
	counts = np.zeros((z_size,), dtype=np.int64)
	store = _worker_store()
	for z0 in range(0, z_size, slab_z):
		z1 = min(z_size, z0 + slab_z)
		counts[z0:z1] = _count_nonzero_by_z(store, _WORKER_CHANNEL_INDEX, z0, z1, y0, y1, x0, x1)
	return _find_holes_in_counts(
		target=_WORKER_TARGET,
		counts=counts,
		y0=y0,
		y1=y1,
		x0=x0,
		x1=x1,
		bracket_window=bracket_window,
		min_neighbor_density=min_neighbor_density,
		drop_ratio=drop_ratio,
		max_low_density=max_low_density,
	)


def _target_blocks(target: ScanTarget, block_yx: int | None, slab_z: int | None) -> tuple[int, int, list[tuple[int, int, int, int]]]:
	arr, shape, channel_index = _target_array(target)
	Z, Y, X = (int(v) for v in shape)
	spatial_chunks = _target_spatial_chunks(arr, channel_index)
	eff_block_yx = block_yx
	if eff_block_yx is None:
		if spatial_chunks is not None:
			eff_block_yx = max(1, max(int(spatial_chunks[1]), int(spatial_chunks[2])))
		else:
			eff_block_yx = 128
	eff_slab_z = slab_z
	if eff_slab_z is None:
		eff_slab_z = max(1, int(spatial_chunks[0]) if spatial_chunks is not None else 32)

	blocks: list[tuple[int, int, int, int]] = []
	for y0 in range(0, Y, eff_block_yx):
		y1 = min(Y, y0 + eff_block_yx)
		for x0 in range(0, X, eff_block_yx):
			x1 = min(X, x0 + eff_block_yx)
			blocks.append((y0, y1, x0, x1))
	return Z, eff_slab_z, blocks


def _ceil_div(a: int, b: int) -> int:
	return int(math.ceil(float(a) / float(b)))


def _chunk_grid(shape: tuple[int, int, int], chunks: tuple[int, int, int]) -> tuple[int, int, int]:
	return tuple(_ceil_div(int(s), int(c)) for s, c in zip(shape, chunks))  # type: ignore[return-value]


def _chunk_bounds(
	coord: ChunkCoord,
	shape: tuple[int, int, int],
	chunks: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int]:
	cz, cy, cx = coord
	z0 = cz * chunks[0]
	y0 = cy * chunks[1]
	x0 = cx * chunks[2]
	return (
		z0,
		min(shape[0], z0 + chunks[0]),
		y0,
		min(shape[1], y0 + chunks[1]),
		x0,
		min(shape[2], x0 + chunks[2]),
	)


def _read_zarray_metadata(path: Path) -> dict:
	with (path / ".zarray").open("r", encoding="utf-8") as handle:
		return json.load(handle)


def _dimension_separator(path: Path) -> str:
	return str(_read_zarray_metadata(path).get("dimension_separator", "."))


def _chunk_key_parts(target: ScanTarget, arr, coord: ChunkCoord) -> tuple[int, ...]:
	if target.channel_index is None:
		return coord
	chunks = tuple(int(v) for v in getattr(arr, "chunks", ()) or ())
	channel_chunk = max(1, int(chunks[0])) if chunks else 1
	return (int(target.channel_index) // channel_chunk, *coord)


def _chunk_path(target: ScanTarget, arr, coord: ChunkCoord, dimension_separator: str) -> Path:
	parts = [str(v) for v in _chunk_key_parts(target, arr, coord)]
	if dimension_separator == "/":
		return target.path.joinpath(*parts)
	return target.path / ".".join(parts)


def _iter_chunk_coords(grid_shape: tuple[int, int, int]) -> Iterable[ChunkCoord]:
	for cz in range(grid_shape[0]):
		for cy in range(grid_shape[1]):
			for cx in range(grid_shape[2]):
				yield (cz, cy, cx)


def _stored_chunk_coords(target: ScanTarget, arr, shape: tuple[int, int, int], chunks: tuple[int, int, int]) -> set[ChunkCoord]:
	sep = _dimension_separator(target.path)
	grid_shape = _chunk_grid(shape, chunks)
	return {
		coord
		for coord in _iter_chunk_coords(grid_shape)
		if _chunk_path(target, arr, coord, sep).exists()
	}


def _read_chunk_data(
	store,
	channel_index: int | None,
	coord: ChunkCoord,
	shape: tuple[int, int, int],
	chunks: tuple[int, int, int],
) -> np.ndarray:
	z0, z1, y0, y1, x0, x1 = _chunk_bounds(coord, shape, chunks)
	return _read_ts_region(store, channel_index, z0, z1, y0, y1, x0, x1)


def _edge_zero(zero: np.ndarray) -> bool:
	if zero.size == 0:
		return False
	return (
		bool(np.any(zero[0, :, :])) or
		bool(np.any(zero[-1, :, :])) or
		bool(np.any(zero[:, 0, :])) or
		bool(np.any(zero[:, -1, :])) or
		bool(np.any(zero[:, :, 0])) or
		bool(np.any(zero[:, :, -1]))
	)


def _face_view(labels: np.ndarray, face: int) -> np.ndarray:
	if face == FACE_Z0:
		return labels[0, :, :]
	if face == FACE_Z1:
		return labels[-1, :, :]
	if face == FACE_Y0:
		return labels[:, 0, :]
	if face == FACE_Y1:
		return labels[:, -1, :]
	if face == FACE_X0:
		return labels[:, :, 0]
	if face == FACE_X1:
		return labels[:, :, -1]
	raise ValueError(f"invalid face {face}")


def _label_structure_26() -> np.ndarray:
	return np.ones((3, 3, 3), dtype=np.uint8)


def _analyze_chunk_job(
	target: ScanTarget,
	shape: tuple[int, int, int],
	chunks: tuple[int, int, int],
	coord: ChunkCoord,
) -> ChunkAnalysis:
	arr, _shape, channel_index = _target_array(target)
	store = _open_tensorstore_array(target.path)
	data = _read_chunk_data(store, channel_index, coord, shape, chunks)
	zero = data == 0
	if not bool(np.any(zero)):
		return ChunkAnalysis(coord=coord, status=CHUNK_ALL_NONZERO)
	if bool(np.all(zero)):
		return ChunkAnalysis(coord=coord, status=CHUNK_ALL_ZERO)
	if not _edge_zero(zero):
		return ChunkAnalysis(coord=coord, status=CHUNK_MIXED_NO_EDGE_ZERO)

	labels, component_count = ndimage.label(zero, structure=_label_structure_26())
	face_arrays: list[np.ndarray | None] = []
	face_labels: list[tuple[int, ...]] = []
	for face in FACES:
		face_arr = np.asarray(_face_view(labels, face), dtype=np.int32).copy()
		ids = np.unique(face_arr)
		ids = ids[ids > 0]
		face_arrays.append(face_arr)
		face_labels.append(tuple(int(v) for v in ids.tolist()))
	return ChunkAnalysis(
		coord=coord,
		status=CHUNK_MIXED_EDGE_ZERO,
		component_count=int(component_count),
		face_labels=tuple(face_labels),
		face_arrays=tuple(face_arrays),
	)


def _analyze_chunk_worker(coord: ChunkCoord) -> ChunkAnalysis:
	if _WORKER_TARGET is None or _WORKER_SHAPE is None or _WORKER_CHUNKS is None:
		raise RuntimeError("TensorStore worker was not initialized")
	data = _read_chunk_data(_worker_store(), _WORKER_CHANNEL_INDEX, coord, _WORKER_SHAPE, _WORKER_CHUNKS)
	zero = data == 0
	if not bool(np.any(zero)):
		return ChunkAnalysis(coord=coord, status=CHUNK_ALL_NONZERO)
	if bool(np.all(zero)):
		return ChunkAnalysis(coord=coord, status=CHUNK_ALL_ZERO)
	if not _edge_zero(zero):
		return ChunkAnalysis(coord=coord, status=CHUNK_MIXED_NO_EDGE_ZERO)

	labels, component_count = ndimage.label(zero, structure=_label_structure_26())
	face_arrays: list[np.ndarray | None] = []
	face_labels: list[tuple[int, ...]] = []
	for face in FACES:
		face_arr = np.asarray(_face_view(labels, face), dtype=np.int32).copy()
		ids = np.unique(face_arr)
		ids = ids[ids > 0]
		face_arrays.append(face_arr)
		face_labels.append(tuple(int(v) for v in ids.tolist()))
	return ChunkAnalysis(
		coord=coord,
		status=CHUNK_MIXED_EDGE_ZERO,
		component_count=int(component_count),
		face_labels=tuple(face_labels),
		face_arrays=tuple(face_arrays),
	)


def _process_pool_context():
	try:
		return multiprocessing.get_context("spawn")
	except ValueError:
		return None


def _build_flood_filter(
	target: ScanTarget,
	*,
	workers: int,
	progress: ProgressCallback | None = None,
) -> FloodFilter:
	arr, shape, channel_index = _target_array(target)
	spatial_chunks = _target_spatial_chunks(arr, channel_index)
	if spatial_chunks is None:
		spatial_chunks = shape
	chunks = tuple(max(1, int(v)) for v in spatial_chunks)
	grid_shape = _chunk_grid(shape, chunks)
	stored = _stored_chunk_coords(target, arr, shape, chunks)
	total = len(stored)
	analyses: dict[ChunkCoord, ChunkAnalysis] = {
		coord: ChunkAnalysis(coord=coord, status=CHUNK_MISSING)
		for coord in _iter_chunk_coords(grid_shape)
		if coord not in stored
	}

	if workers <= 1 or total <= 1:
		for done, coord in enumerate(sorted(stored), start=1):
			analyses[coord] = _analyze_chunk_job(target, shape, chunks, coord)
			if progress is not None:
				progress(target.name, done, max(1, total), f"flood analyze chunk={coord}")
	else:
		kwargs = {}
		ctx = _process_pool_context()
		if ctx is not None:
			kwargs["mp_context"] = ctx
		kwargs["initializer"] = _init_tensorstore_worker
		kwargs["initargs"] = (target, shape, chunks)
		with ProcessPoolExecutor(max_workers=max(1, int(workers)), **kwargs) as pool:
			futures = {
				pool.submit(_analyze_chunk_worker, coord): coord
				for coord in sorted(stored)
			}
			done = 0
			for future in as_completed(futures):
				analysis = future.result()
				analyses[analysis.coord] = analysis
				done += 1
				if progress is not None:
					progress(target.name, done, max(1, total), f"flood analyze chunk={analysis.coord}")

	reached = _flood_reachable_components(
		analyses,
		grid_shape,
		progress=progress,
		channel=target.name,
	)
	if progress is not None:
		progress(target.name, max(1, total), max(1, total), "flood graph done")
	suspicious_masks = _build_suspicious_masks(
		target,
		shape,
		chunks,
		analyses,
		reached,
		progress=progress,
	)
	return FloodFilter(
		target=target,
		shape=shape,
		chunks=chunks,
		grid_shape=grid_shape,
		analyses=analyses,
		reached=reached,
		suspicious_masks=suspicious_masks,
	)


def _component_nodes(analysis: ChunkAnalysis) -> list[tuple[ChunkCoord, int]]:
	if analysis.status == CHUNK_ALL_ZERO:
		return [(analysis.coord, 0)]
	if analysis.status == CHUNK_MIXED_EDGE_ZERO:
		return [(analysis.coord, comp_id) for comp_id in range(1, analysis.component_count + 1)]
	return []


def _face_component_ids(analysis: ChunkAnalysis, face: int) -> tuple[int, ...]:
	if analysis.status == CHUNK_ALL_ZERO:
		return (0,)
	if analysis.status == CHUNK_MIXED_EDGE_ZERO:
		return analysis.face_labels[face]
	return ()


def _add_edge(
	adj: dict[tuple[ChunkCoord, int], set[tuple[ChunkCoord, int]]],
	a: tuple[ChunkCoord, int],
	b: tuple[ChunkCoord, int],
) -> bool:
	left = adj.setdefault(a, set())
	already_present = b in left
	left.add(b)
	adj.setdefault(b, set()).add(a)
	return not already_present


def _touching_face_pairs(a: np.ndarray, b: np.ndarray) -> set[tuple[int, int]]:
	pairs: set[tuple[int, int]] = set()
	a = np.asarray(a)
	b = np.asarray(b)
	for dy in (-1, 0, 1):
		if dy < 0:
			a_y = slice(0, dy)
			b_y = slice(-dy, None)
		elif dy > 0:
			a_y = slice(dy, None)
			b_y = slice(0, -dy)
		else:
			a_y = slice(None)
			b_y = slice(None)
		for dx in (-1, 0, 1):
			if dx < 0:
				a_x = slice(0, dx)
				b_x = slice(-dx, None)
			elif dx > 0:
				a_x = slice(dx, None)
				b_x = slice(0, -dx)
			else:
				a_x = slice(None)
				b_x = slice(None)
			aa = a[a_y, a_x]
			bb = b[b_y, b_x]
			mask = (aa > 0) & (bb > 0)
			if not bool(np.any(mask)):
				continue
			stacked = np.stack((aa[mask], bb[mask]), axis=1)
			for left, right in np.unique(stacked, axis=0):
				pairs.add((int(left), int(right)))
	return pairs


def _connect_adjacent_chunks(
	adj: dict[tuple[ChunkCoord, int], set[tuple[ChunkCoord, int]]],
	left: ChunkAnalysis,
	right: ChunkAnalysis,
	face: int,
) -> int:
	left_ids = _face_component_ids(left, face)
	right_ids = _face_component_ids(right, OPPOSITE_FACE[face])
	if not left_ids or not right_ids:
		return 0
	added = 0
	if left.status == CHUNK_ALL_ZERO or right.status == CHUNK_ALL_ZERO:
		for left_id in left_ids:
			for right_id in right_ids:
				added += int(_add_edge(adj, (left.coord, left_id), (right.coord, right_id)))
		return added
	left_face = left.face_arrays[face]
	right_face = right.face_arrays[OPPOSITE_FACE[face]]
	if left_face is None or right_face is None:
		return 0
	for left_id, right_id in _touching_face_pairs(left_face, right_face):
		added += int(_add_edge(adj, (left.coord, left_id), (right.coord, right_id)))
	return added


def _flood_reachable_components(
	analyses: dict[ChunkCoord, ChunkAnalysis],
	grid_shape: tuple[int, int, int],
	*,
	progress: ProgressCallback | None = None,
	channel: str = "",
) -> set[tuple[ChunkCoord, int]]:
	adj: dict[tuple[ChunkCoord, int], set[tuple[ChunkCoord, int]]] = {}
	seeds: set[tuple[ChunkCoord, int]] = set()
	total_chunks = max(1, len(analyses))
	missing = ChunkAnalysis((-1, -1, -1), CHUNK_MISSING)
	for done, analysis in enumerate(analyses.values(), start=1):
		for node in _component_nodes(analysis):
			adj.setdefault(node, set())
		if analysis.status == CHUNK_ALL_ZERO:
			seeds.add((analysis.coord, 0))
		cz, cy, cx = analysis.coord
		for face in FACES:
			dz, dy, dx = FACE_DELTAS[face]
			neighbor = (cz + dz, cy + dy, cx + dx)
			outside_volume = not (
				0 <= neighbor[0] < grid_shape[0] and
				0 <= neighbor[1] < grid_shape[1] and
				0 <= neighbor[2] < grid_shape[2]
			)
			if outside_volume or analyses.get(neighbor, missing).status == CHUNK_MISSING:
				for comp_id in _face_component_ids(analysis, face):
					seeds.add((analysis.coord, comp_id))
		if progress is not None:
			progress(channel, done, total_chunks, f"flood graph seed nodes={len(seeds)} components={len(adj)}")

	edge_count = 0
	for done, (coord, analysis) in enumerate(analyses.items(), start=1):
		if analysis.status not in (CHUNK_ALL_ZERO, CHUNK_MIXED_EDGE_ZERO):
			if progress is not None:
				progress(channel, done, total_chunks, "flood graph connect")
			continue
		cz, cy, cx = coord
		for face in (FACE_Z1, FACE_Y1, FACE_X1):
			dz, dy, dx = FACE_DELTAS[face]
			neighbor_coord = (cz + dz, cy + dy, cx + dx)
			neighbor = analyses.get(neighbor_coord)
			if neighbor is None or neighbor.status == CHUNK_MISSING:
				continue
			edge_count += _connect_adjacent_chunks(adj, analysis, neighbor, face)
		if progress is not None:
			progress(channel, done, total_chunks, f"flood graph connect edges={edge_count}")

	reached: set[tuple[ChunkCoord, int]] = set()
	queue = deque(node for node in seeds if node in adj)
	processed = 0
	if progress is not None and not queue:
		progress(channel, 0, max(len(adj), 1), f"flood bfs reached=0 queue=0 seeds={len(seeds)}")
	while queue:
		node = queue.popleft()
		if node in reached:
			continue
		reached.add(node)
		processed += 1
		queue.extend(adj.get(node, ()))
		if progress is not None:
			progress(channel, processed, max(processed + len(queue), len(adj), 1), f"flood bfs reached={len(reached)} queue={len(queue)}")
	if progress is not None:
		progress(channel, max(processed, 1), max(processed, 1), f"flood bfs done reached={len(reached)}")
	return reached


def _build_suspicious_masks(
	target: ScanTarget,
	shape: tuple[int, int, int],
	chunks: tuple[int, int, int],
	analyses: dict[ChunkCoord, ChunkAnalysis],
	reached: set[tuple[ChunkCoord, int]],
	*,
	progress: ProgressCallback | None = None,
) -> dict[ChunkCoord, np.ndarray]:
	_arr, _shape, channel_index = _target_array(target)
	store = _open_tensorstore_array(target.path)
	masks: dict[ChunkCoord, np.ndarray] = {}
	candidates = [
		analysis
		for analysis in analyses.values()
		if analysis.status in (CHUNK_MIXED_NO_EDGE_ZERO, CHUNK_MIXED_EDGE_ZERO)
	]
	total = max(1, len(candidates))
	for done, analysis in enumerate(candidates, start=1):
		data = _read_chunk_data(store, channel_index, analysis.coord, shape, chunks)
		zero = data == 0
		if not bool(np.any(zero)):
			if progress is not None:
				progress(target.name, done, total, "flood mask chunk")
			continue
		if analysis.status == CHUNK_MIXED_NO_EDGE_ZERO:
			masks[analysis.coord] = zero
		else:
			unreached = [
				comp_id
				for comp_id in range(1, analysis.component_count + 1)
				if (analysis.coord, comp_id) not in reached
			]
			if unreached:
				labels, _component_count = ndimage.label(zero, structure=_label_structure_26())
				mask = np.isin(labels, np.asarray(unreached, dtype=labels.dtype))
				if bool(np.any(mask)):
					masks[analysis.coord] = mask
		if progress is not None:
			progress(target.name, done, total, f"flood mask chunk suspicious={len(masks)}")
	if progress is not None:
		progress(target.name, total, total, f"flood mask done suspicious={len(masks)}")
	return masks


def _finding_overlaps_unreached_zero(finding: HoleFinding, flood: FloodFilter) -> bool:
	z0 = max(0, int(finding.z0))
	z1 = min(flood.shape[0], int(finding.z1))
	y0 = max(0, int(finding.y0))
	y1 = min(flood.shape[1], int(finding.y1))
	x0 = max(0, int(finding.x0))
	x1 = min(flood.shape[2], int(finding.x1))
	if z0 >= z1 or y0 >= y1 or x0 >= x1:
		return False
	cz0 = z0 // flood.chunks[0]
	cz1 = (z1 - 1) // flood.chunks[0]
	cy0 = y0 // flood.chunks[1]
	cy1 = (y1 - 1) // flood.chunks[1]
	cx0 = x0 // flood.chunks[2]
	cx1 = (x1 - 1) // flood.chunks[2]
	for cz in range(cz0, cz1 + 1):
		for cy in range(cy0, cy1 + 1):
			for cx in range(cx0, cx1 + 1):
				coord = (cz, cy, cx)
				mask = flood.suspicious_masks.get(coord)
				if mask is None:
					continue
				cb = _chunk_bounds(coord, flood.shape, flood.chunks)
				oz0, oz1 = max(z0, cb[0]), min(z1, cb[1])
				oy0, oy1 = max(y0, cb[2]), min(y1, cb[3])
				ox0, ox1 = max(x0, cb[4]), min(x1, cb[5])
				local = mask[oz0 - cb[0]:oz1 - cb[0], oy0 - cb[2]:oy1 - cb[2], ox0 - cb[4]:ox1 - cb[4]]
				if local.size > 0 and bool(np.any(local)):
					return True
	return False


def _filter_findings_with_flood(
	findings: list[HoleFinding],
	*,
	flood: FloodFilter,
	progress: ProgressCallback | None,
) -> list[HoleFinding]:
	if not findings:
		return findings
	filtered: list[HoleFinding] = []
	total = len(findings)
	for idx, finding in enumerate(findings, start=1):
		if _finding_overlaps_unreached_zero(finding, flood):
			filtered.append(finding)
		if progress is not None:
			progress(flood.target.name, idx, total, "flood filter findings")
	return filtered


def _center_xyz_for_bbox(target: ScanTarget, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> tuple[int, int, int] | None:
	if target.scaledown is None:
		return None
	scale = 2 ** int(target.scaledown)
	return (
		int(((x0 + x1) // 2) * scale),
		int(((y0 + y1) // 2) * scale),
		int(((z0 + z1) // 2) * scale),
	)


def _finding_from_component(
	target: ScanTarget,
	mask: np.ndarray,
	coord: ChunkCoord,
	shape: tuple[int, int, int],
	chunks: tuple[int, int, int],
	local_slices: tuple[slice, slice, slice],
) -> HoleFinding:
	cb = _chunk_bounds(coord, shape, chunks)
	lsz, lsy, lsx = local_slices
	lz0 = 0 if lsz.start is None else int(lsz.start)
	lz1 = int(mask.shape[0]) if lsz.stop is None else int(lsz.stop)
	ly0 = 0 if lsy.start is None else int(lsy.start)
	ly1 = int(mask.shape[1]) if lsy.stop is None else int(lsy.stop)
	lx0 = 0 if lsx.start is None else int(lsx.start)
	lx1 = int(mask.shape[2]) if lsx.stop is None else int(lsx.stop)
	z0, z1 = cb[0] + lz0, cb[0] + lz1
	y0, y1 = cb[2] + ly0, cb[2] + ly1
	x0, x1 = cb[4] + lx0, cb[4] + lx1
	component = mask[lz0:lz1, ly0:ly1, lx0:lx1]
	z_has_zero = np.any(component, axis=(1, 2)) if component.size else np.zeros((0,), dtype=bool)
	return HoleFinding(
		channel=target.name,
		path=str(target.path),
		z0=int(z0),
		z1=int(z1),
		y0=int(y0),
		y1=int(y1),
		x0=int(x0),
		x1=int(x1),
		min_density=0.0,
		max_density=0.0,
		before_density=0.0,
		after_density=0.0,
		zero_slices=int(np.count_nonzero(z_has_zero)),
		low_slices=int(z1 - z0),
		center_xyz=_center_xyz_for_bbox(target, z0, z1, y0, y1, x0, x1),
	)


def _report_findings_from_flood(
	flood: FloodFilter,
	*,
	progress: ProgressCallback | None = None,
	on_finding: FindingCallback | None = None,
) -> list[HoleFinding]:
	findings: list[HoleFinding] = []
	items = sorted(flood.suspicious_masks.items())
	total = max(1, len(items))
	for done, (coord, mask) in enumerate(items, start=1):
		labels, component_count = ndimage.label(mask, structure=_label_structure_26())
		for local_slices in ndimage.find_objects(labels, max_label=int(component_count)):
			if local_slices is None:
				continue
			finding = _finding_from_component(
				flood.target,
				mask,
				coord,
				flood.shape,
				flood.chunks,
				local_slices,
			)
			findings.append(finding)
			if on_finding is not None:
				on_finding(finding)
		if progress is not None:
			progress(flood.target.name, done, total, f"flood report chunks findings={len(findings)}")
	if progress is not None:
		progress(flood.target.name, total, total, f"flood report done findings={len(findings)}")
	findings.sort(key=lambda f: (f.z0, f.z1, f.y0, f.x0))
	return findings


def scan_target(
	target: ScanTarget,
	*,
	block_yx: int | None = None,
	slab_z: int | None = None,
	bracket_window: int = 64,
	min_neighbor_density: float = 0.005,
	drop_ratio: float = 0.01,
	max_low_density: float = 0.0,
	progress: ProgressCallback | None = None,
	on_finding: FindingCallback | None = None,
	progress_offset: int = 0,
	progress_total: int | None = None,
	workers: int = 1,
) -> list[HoleFinding]:
	Z, eff_slab_z, blocks = _target_blocks(target, block_yx, slab_z)
	findings: list[HoleFinding] = []
	target_total = len(blocks)
	overall_total = int(progress_total) if progress_total is not None else target_total
	max_workers = max(1, int(workers))
	flood = _build_flood_filter(target, workers=max_workers, progress=progress)
	if max_workers <= 1 or target_total <= 1:
		for done_blocks, (y0, y1, x0, x1) in enumerate(blocks):
			if progress is not None:
				progress(
					target.name,
					progress_offset + done_blocks,
					overall_total,
					f"y={y0}-{y1 - 1} x={x0}-{x1 - 1}",
				)
			block_findings = _filter_findings_with_flood(
				_scan_block_job(
					target,
					Z,
					y0,
					y1,
					x0,
					x1,
					eff_slab_z,
					bracket_window,
					min_neighbor_density,
					drop_ratio,
					max_low_density,
				),
				flood=flood,
				progress=None,
			)
			findings.extend(block_findings)
			if on_finding is not None:
				for finding in block_findings:
					on_finding(finding)
	else:
		kwargs = {}
		ctx = _process_pool_context()
		if ctx is not None:
			kwargs["mp_context"] = ctx
		kwargs["initializer"] = _init_tensorstore_worker
		kwargs["initargs"] = (target, None, None)
		with ProcessPoolExecutor(max_workers=max_workers, **kwargs) as pool:
			futures = {
				pool.submit(
					_scan_block_worker,
					(
						Z,
						y0,
						y1,
						x0,
						x1,
						eff_slab_z,
						bracket_window,
						min_neighbor_density,
						drop_ratio,
						max_low_density,
					),
				): (y0, y1, x0, x1)
				for y0, y1, x0, x1 in blocks
			}
			done_blocks = 0
			for future in as_completed(futures):
				y0, y1, x0, x1 = futures[future]
				block_findings = _filter_findings_with_flood(
					future.result(),
					flood=flood,
					progress=None,
				)
				findings.extend(block_findings)
				if on_finding is not None:
					for finding in sorted(block_findings, key=lambda f: (f.z0, f.z1, f.y0, f.x0)):
						on_finding(finding)
				done_blocks += 1
				if progress is not None:
					progress(
						target.name,
						progress_offset + done_blocks,
						overall_total,
						f"finished y={y0}-{y1 - 1} x={x0}-{x1 - 1}",
					)
	if progress is not None:
		progress(target.name, progress_offset + target_total, overall_total, "done")
	findings.sort(key=lambda f: (f.z0, f.z1, f.y0, f.x0))
	return findings


def scan_path(
	path: Path,
	*,
	channels: Iterable[str] | None = DEFAULT_CHANNELS,
	block_yx: int | None = None,
	slab_z: int | None = None,
	bracket_window: int = 64,
	min_neighbor_density: float = 0.005,
	drop_ratio: float = 0.01,
	max_low_density: float = 0.0,
	progress: ProgressCallback | None = None,
	on_finding: FindingCallback | None = None,
	workers: int = 1,
) -> list[HoleFinding]:
	findings: list[HoleFinding] = []
	targets = discover_targets(path, channels=channels)
	target_block_counts: list[int] = []
	for target in targets:
		_target_z, _target_slab_z, target_blocks = _target_blocks(target, block_yx, slab_z)
		target_block_counts.append(len(target_blocks))

	progress_total = sum(target_block_counts)
	progress_offset = 0
	for target, target_blocks in zip(targets, target_block_counts):
		findings.extend(
			scan_target(
				target,
				block_yx=block_yx,
				slab_z=slab_z,
				bracket_window=bracket_window,
				min_neighbor_density=min_neighbor_density,
				drop_ratio=drop_ratio,
				max_low_density=max_low_density,
				progress=progress,
				on_finding=on_finding,
				progress_offset=progress_offset,
				progress_total=progress_total,
				workers=workers,
			)
		)
		progress_offset += target_blocks
	findings.sort(key=lambda f: (f.channel, f.path, f.z0, f.z1, f.y0, f.x0))
	return findings


def _make_stderr_progress(*, interval_s: float = 1.0) -> ProgressCallback:
	t0 = time.monotonic()
	last = [0.0]

	def _progress(channel: str, done: int, total: int, detail: str) -> None:
		now = time.monotonic()
		if done < total and now - last[0] < interval_s:
			return
		last[0] = now
		frac = done / max(1, total)
		elapsed = now - t0
		eta = elapsed * (1.0 - frac) / frac if frac > 0.0 else 0.0
		unit = "chunks" if detail.startswith("flood ") else "blocks"
		print(
			f"\r[holescan] {done}/{total} {unit} ({100.0 * frac:5.1f}%) "
			f"channel={channel} {detail} elapsed={elapsed:.0f}s eta={eta:.0f}s",
			file=sys.stderr,
			end="" if done < total else "\n",
			flush=True,
		)

	return _progress


def _format_bytes(n_bytes: int) -> str:
	value = float(n_bytes)
	for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
		if value < 1024.0 or unit == "TiB":
			return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
		value /= 1024.0
	return f"{n_bytes} B"


def _format_gib(n_bytes: int) -> str:
	return f"{float(n_bytes) / (1024.0 ** 3):.3f} GiB"


def _flood_fill_estimate_lines(
	path: Path,
	*,
	channels: Iterable[str] | None,
	block_yx: int | None,
	slab_z: int | None,
) -> list[str]:
	lines: list[str] = []
	total_voxels = 0
	for target in discover_targets(path, channels=channels):
		_arr, shape, _channel_index = _target_array(target)
		Z, Y, X = (int(v) for v in shape)
		if Z <= 0 or Y <= 0 or X <= 0:
			continue
		voxels = Z * Y * X
		total_voxels += voxels
		lines.append(
			f"[holescan] chunk flood-filter channel={target.name} "
			f"shape_zyx=({Z},{Y},{X}) voxels={voxels:,} "
			f"state_uint8={_format_bytes(voxels)} ({_format_gib(voxels)})"
		)
	if total_voxels > 0:
		lines.append(
			f"[holescan] chunk flood-filter total voxels={total_voxels:,} "
			f"dense_state_would_be={_format_bytes(total_voxels)} ({_format_gib(total_voxels)})"
		)
	return lines


def _format_finding(f: HoleFinding) -> str:
	base = (
		f"HOLE channel={f.channel} z={f.z0}-{f.z1 - 1} "
		f"y={f.y0}-{f.y1 - 1} x={f.x0}-{f.x1 - 1} "
		f"density={f.min_density:.6g}-{f.max_density:.6g} "
		f"context={f.before_density:.6g}/{f.after_density:.6g}"
	)
	if f.center_xyz is None:
		return base
	x, y, z = f.center_xyz
	return f"{base} center_xyz=({x},{y},{z})"


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Scan predict3d grad_mag outputs for zeroed Z bands from the "
			"channel-major page-release bug fixed in ScrollPrize/villa#1192."
		),
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("path", type=Path, help=".lasagna.json manifest, zarr array, or OME-Zarr level path.")
	parser.add_argument(
		"--channels",
		default=",".join(DEFAULT_CHANNELS),
		help="Comma-separated channels to scan; use 'all' for every manifest/group channel.",
	)
	parser.add_argument("--block-yx", type=int, default=None, help="Y/X block size. Defaults to zarr chunk Y/X.")
	parser.add_argument("--slab-z", type=int, default=None, help="Z read slab size. Defaults to zarr chunk Z.")
	parser.add_argument("--bracket-window", type=int, default=64, help="Nearby Z slices used to confirm support before and after a run.")
	parser.add_argument("--min-neighbor-density", type=float, default=0.005, help="Minimum nonzero density on both sides of a candidate run.")
	parser.add_argument("--drop-ratio", type=float, default=0.01, help="Expand/report slices below this fraction of neighboring density.")
	parser.add_argument("--max-low-density", type=float, default=0.0, help="Primary low-density threshold. Default flags exactly zero slices.")
	parser.add_argument("--json", action="store_true", help="Write findings as JSON.")
	parser.add_argument("--quiet", action="store_true", help="Disable progress output.")
	parser.add_argument("--workers", type=int, default=_default_workers(), help="Parallel workers. Use 1 for serial/debug scanning.")
	return parser


def main(argv: list[str] | None = None) -> int:
	args = build_parser().parse_args(argv)
	if args.block_yx is not None and args.block_yx <= 0:
		raise ValueError("--block-yx must be positive")
	if args.slab_z is not None and args.slab_z <= 0:
		raise ValueError("--slab-z must be positive")
	if args.bracket_window <= 0:
		raise ValueError("--bracket-window must be positive")
	if args.workers <= 0:
		raise ValueError("--workers must be positive")

	channels = None if str(args.channels).strip().lower() == "all" else [
		ch.strip() for ch in str(args.channels).split(",") if ch.strip()
	]
	if not args.quiet:
		for line in _flood_fill_estimate_lines(
			args.path,
			channels=channels,
			block_yx=args.block_yx,
			slab_z=args.slab_z,
		):
			print(line, file=sys.stderr, flush=True)
	findings = scan_path(
		args.path,
		channels=channels,
		block_yx=args.block_yx,
		slab_z=args.slab_z,
		bracket_window=int(args.bracket_window),
		min_neighbor_density=float(args.min_neighbor_density),
		drop_ratio=float(args.drop_ratio),
		max_low_density=float(args.max_low_density),
		progress=None if args.quiet else _make_stderr_progress(),
		on_finding=None if args.json else lambda f: print(_format_finding(f), flush=True),
		workers=int(args.workers),
	)

	if args.json:
		print(json.dumps([asdict(f) for f in findings], indent=2, sort_keys=True))
	else:
		if findings:
			print(f"VERDICT: FAIL ({len(findings)} candidate region(s))")
		else:
			print("VERDICT: PASS")
	return 1 if findings else 0


if __name__ == "__main__":
	raise SystemExit(main())
