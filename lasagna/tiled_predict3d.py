from __future__ import annotations

import atexit
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import wraps
import json
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path
import pickle
import queue
import shutil
import sys
import tempfile
import threading
import time
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence, runtime_checkable
import uuid

import numpy as np
import numcodecs
import torch
import torch.nn.functional as F
import zarr

try:
	from omezarr_pyramid import (
		_limit_zarr_codec_threads,
		_single_threaded_native_runtime,
		build_normal_omezarr_pyramid,
		build_scalar_omezarr_pyramid,
		set_pyramid_metadata,
	)
except ImportError:
	from lasagna.omezarr_pyramid import (
		_limit_zarr_codec_threads,
		_single_threaded_native_runtime,
		build_normal_omezarr_pyramid,
		build_scalar_omezarr_pyramid,
		set_pyramid_metadata,
	)

try:
	from live_omezarr_cache import LiveOmeZarrCache, selected_level_lock_for_input
except ImportError:
	from lasagna.live_omezarr_cache import LiveOmeZarrCache, selected_level_lock_for_input


ChunkOriginZYX = tuple[int, int, int]
RegionZYX = tuple[int, int, int, int, int, int]
TileOriginZYX = tuple[int, int, int]
ProductTileOutput = Mapping[str, np.ndarray]
DEFAULT_FLUSH_WORKERS = min(64, max(1, os.cpu_count() or 1))
DEFAULT_INPUT_READER = "tensorstore"
DEFAULT_PREFETCH_TILES_PER_GPU = 4
DEFAULT_INPUT_CACHE_BYTES = 4 << 30
DEFAULT_INPUT_IO_THREADS = 16
DEFAULT_INPUT_COPY_THREADS = 4
DEFAULT_ACCUMULATOR_WORKERS = min(32, max(1, os.cpu_count() or 1))
DEFAULT_OME_COMPRESSOR = "blosc-zstd"
OME_COMPRESSOR_CHOICES = (DEFAULT_OME_COMPRESSOR, "none")
_BLOSC_ZSTD_CONFIG = {
	"id": "blosc",
	"cname": "zstd",
	"clevel": 3,
	"shuffle": 1,
	"blocksize": 0,
}


@dataclass(frozen=True)
class _FlushChunkDescriptor:
	origin: ChunkOriginZYX
	weight_dirty: bool
	dirty_products: tuple[str, ...]


@dataclass(frozen=True)
class _FlushGroupDescriptor:
	sd: int
	flush_from: int
	flush_to: int
	chunks: tuple[_FlushChunkDescriptor, ...]
	submitted_at: float
	b: int
	oc: int
	region: RegionZYX
	products: tuple[Any, ...]
	accumulators: tuple[tuple[str, "_MmapBandDescriptor"], ...]
	weight: "_MmapBandDescriptor"


@dataclass
class _FlushBatchState:
	batch_id: int
	plans: tuple[_FlushGroupDescriptor, ...]
	tasks: deque["_FlushProcessTask"]
	inflight: set[int]
	completed_ids: set[int]
	work_s: float
	written_by_sd: dict[int, int]
	task_sd: dict[int, int]

	@property
	def complete(self) -> bool:
		return not self.tasks and not self.inflight


@dataclass
class _ZBandState:
	z: int
	next_z: int
	first_sequence: int
	expected: int
	generated: int = 0
	committed: int = 0

	@property
	def end_sequence(self) -> int:
		return self.first_sequence + self.expected

	def validate_complete(self) -> None:
		if self.generated != self.expected or self.committed != self.expected:
			raise RuntimeError(
				f"incomplete Z band z={self.z}: generated={self.generated} "
				f"committed={self.committed} expected={self.expected}"
			)


class _SlotLedger:
	"""Single-owner lifecycle for shared input/result slots."""

	def __init__(self, count: int):
		self._count = int(count)
		self._owners: dict[tuple[str, int], tuple[int, int, str]] = {}

	def acquire(self, kind: str, slot: int, *, band: int, sequence: int, stage: str) -> None:
		key = (str(kind), int(slot))
		if key in self._owners:
			raise RuntimeError(f"shared {kind} slot {slot} already owned by {self._owners[key]}")
		if not 0 <= int(slot) < self._count:
			raise RuntimeError(f"invalid shared {kind} slot {slot}")
		self._owners[key] = (int(band), int(sequence), str(stage))

	def transition(self, kind: str, slot: int, *, band: int, sequence: int, stage: str) -> None:
		key = (str(kind), int(slot))
		expected = (int(band), int(sequence))
		owner = self._owners.get(key)
		if owner is None or owner[:2] != expected:
			raise RuntimeError(
				f"shared {kind} slot {slot} transition for {expected} conflicts with owner {owner}"
			)
		self._owners[key] = (expected[0], expected[1], str(stage))

	def release(self, kind: str, slot: int, *, band: int, sequence: int) -> None:
		key = (str(kind), int(slot))
		expected = (int(band), int(sequence))
		owner = self._owners.get(key)
		if owner is None or owner[:2] != expected:
			raise RuntimeError(
				f"shared {kind} slot {slot} release for {expected} conflicts with owner {owner}"
			)
		del self._owners[key]

	def validate_all_free(self) -> None:
		if self._owners:
			raise RuntimeError(f"shared slots still owned at Z-band boundary: {self._owners}")

	def snapshot(self) -> dict[tuple[str, int], tuple[int, int, str]]:
		return dict(self._owners)


_FRONTIER_NEEDS_GPU_STATUSES = frozenset(("fetching", "awaiting_read", "reading", "ready"))


def _effective_accumulation_frontier(
	events: Mapping[int, Mapping[str, Any]], frontier_sequence: int,
) -> tuple[int, tuple[int, ...]]:
	"""Return the first canonical event not already resolved as a sparse skip."""
	sequence = int(frontier_sequence)
	skipped: list[int] = []
	while events.get(sequence, {}).get("status") == "done_skip":
		skipped.append(sequence)
		sequence += 1
	return sequence, tuple(skipped)


def _frontier_slot_reservation(
	events: Mapping[int, Mapping[str, Any]], frontier_sequence: int,
) -> int:
	"""Reserve one shared slot pair until the accumulation frontier enters GPU."""
	effective, _skipped = _effective_accumulation_frontier(events, frontier_sequence)
	event = events.get(effective)
	if event is None:
		return 0
	return int(str(event.get("status", "unknown")) in _FRONTIER_NEEDS_GPU_STATUSES)


def _shared_slot_pair_available(
	*, events: Mapping[int, Mapping[str, Any]], frontier_sequence: int,
	sequence: int, free_input_count: int, free_result_count: int,
) -> bool:
	"""Admit later GPU work only while the frontier's slot pair stays free."""
	if int(free_input_count) <= 0 or int(free_result_count) <= 0:
		return False
	effective, _skipped = _effective_accumulation_frontier(events, frontier_sequence)
	reservation = _frontier_slot_reservation(events, frontier_sequence)
	if int(sequence) == effective or not reservation:
		return True
	return int(free_input_count) > reservation and int(free_result_count) > reservation


def _frontier_capacity_deadlock(
	*, events: Mapping[int, Mapping[str, Any]], frontier_sequence: int,
	free_inputs: Sequence[int], free_results: Sequence[int],
	slot_owners: Mapping[tuple[str, int], tuple[int, int, str]],
) -> str | None:
	"""Describe the canonical-frontier/result-slot circular wait, if present."""
	effective, skipped = _effective_accumulation_frontier(events, frontier_sequence)
	frontier = events.get(effective)
	if (
		frontier is None
		or str(frontier.get("status", "unknown")) not in _FRONTIER_NEEDS_GPU_STATUSES
		or free_results
	):
		return None
	result_owners = {
		int(slot): owner for (kind, slot), owner in slot_owners.items() if kind == "result"
	}
	if not result_owners or any(
		int(owner[1]) <= effective or str(owner[2]) not in ("gpu-provisional", "gpu", "gpu-result")
		for owner in result_owners.values()
	):
		return None
	owner_stages: dict[str, int] = {}
	for owner in result_owners.values():
		stage = str(owner[2])
		owner_stages[stage] = owner_stages.get(stage, 0) + 1
	return (
		"canonical-frontier capacity invariant violation: "
		f"raw_frontier={int(frontier_sequence)} effective_frontier={effective} "
		f"skipped_prefix={list(skipped)} status={frontier.get('status')} "
		f"free_inputs={len(free_inputs)} free_results=0 "
		f"result_owner_stages={owner_stages} result_owners={result_owners} "
		f"slots={dict(slot_owners)}"
	)


def _coordinator_wait_stage(
	*, statuses: Mapping[str, int], reads: int, accumulator_tasks: int,
	slot_owners: Mapping[tuple[str, int], tuple[int, int, str]],
) -> str | None:
	"""Identify the producer that can advance a quiescent coordinator state."""
	if int(statuses.get("assigned", 0)) > 0:
		return "gpu"
	if int(accumulator_tasks) > 0:
		return "accumulator"
	if int(reads) > 0 or int(statuses.get("fetching", 0)) > 0:
		return "reader"
	if slot_owners and int(statuses.get("ready", 0)) > 0:
		# Ready input can be blocked only by a slot still owned by GPU/accumulation.
		owner_stages = {owner[2] for owner in slot_owners.values()}
		if "gpu" in owner_stages:
			return "gpu"
		if "gpu-result" in owner_stages or "accumulating" in owner_stages:
			return "accumulator"
	if any(int(count) > 0 for count in statuses.values()):
		raise RuntimeError(
			"shared tiled coordinator is quiescent with unresolved work and no "
			f"producer: statuses={dict(statuses)} reads={int(reads)} "
			f"accumulator_tasks={int(accumulator_tasks)} slots={dict(slot_owners)}"
		)
	return None


@dataclass(frozen=True)
class _MmapBandDescriptor:
	paths: tuple[str, ...]
	shape_zyx: tuple[int, int, int]
	dtype: str = "float32"

	@property
	def ring_depth(self) -> int:
		return int(self.shape_zyx[0])


@dataclass(frozen=True)
class _FlushProcessTask:
	batch_id: int
	task_id: int
	sd: int
	origin: ChunkOriginZYX
	dirty_products: tuple[str, ...]
	weight_dirty: bool
	b: int
	oc: int
	region: RegionZYX
	products: tuple[Any, ...]
	accumulators: tuple[tuple[str, _MmapBandDescriptor], ...]
	weight: _MmapBandDescriptor


@dataclass(frozen=True)
class _AccumulateProcessTask:
	task_id: int
	event_seq: int
	result_slot: int
	result_spec: "_SharedSlotSpec"
	sd: int
	origin: ChunkOriginZYX
	destination: tuple[int, int, int, int, int, int]
	source: tuple[int, int, int, int, int, int]
	dirty_products: tuple[str, ...]
	accumulators: tuple[tuple[str, _MmapBandDescriptor], ...]
	weight: _MmapBandDescriptor


@dataclass(frozen=True)
class _TileReadSpec:
	slices_zyx: tuple[slice, slice, slice] | None
	pad_width_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
	tile_size: int


@dataclass
class _TensorStoreReadTask:
	future: Any | None
	spec: _TileReadSpec
	submitted_at: float
	completed_at: float | None = None


def resolve_inference_devices(
	*,
	device: str | torch.device | None = None,
	devices: str | Sequence[str | torch.device] | None = None,
	cuda_available: bool | None = None,
	cuda_count: int | None = None,
) -> tuple[torch.device, ...]:
	"""Resolve legacy singular or explicit plural inference device selection."""
	if device is not None and devices is not None:
		raise ValueError("pass either --device or --devices, not both")
	available = torch.cuda.is_available() if cuda_available is None else bool(cuda_available)
	count = torch.cuda.device_count() if cuda_count is None else int(cuda_count)
	if devices is None:
		requested = "auto" if device is None else str(device).strip().lower()
		if requested == "auto":
			return (torch.device("cuda:0" if available and count > 0 else "cpu"),)
		resolved = torch.device(requested)
		if resolved.type == "cuda":
			index = 0 if resolved.index is None else int(resolved.index)
			if not available or index < 0 or index >= count:
				raise ValueError(f"CUDA device cuda:{index} is unavailable (visible count={count})")
			resolved = torch.device(f"cuda:{index}")
		return (resolved,)
	if isinstance(devices, str):
		raw = devices.strip().lower()
		if raw == "all":
			if not available or count <= 0:
				raise ValueError("--devices all requires at least one visible CUDA device")
			return tuple(torch.device(f"cuda:{index}") for index in range(count))
		tokens = tuple(part.strip() for part in raw.split(",") if part.strip())
	else:
		tokens = tuple(str(part).strip().lower() for part in devices)
	if not tokens:
		raise ValueError("--devices must be 'all' or a non-empty comma-separated CUDA list")
	resolved_devices = []
	for token in tokens:
		resolved = torch.device(token)
		if resolved.type != "cuda":
			raise ValueError("multi-device inference supports CUDA devices only")
		index = 0 if resolved.index is None else int(resolved.index)
		if not available or index < 0 or index >= count:
			raise ValueError(f"CUDA device cuda:{index} is unavailable (visible count={count})")
		resolved_devices.append(torch.device(f"cuda:{index}"))
	keys = tuple(str(value) for value in resolved_devices)
	if len(set(keys)) != len(keys):
		raise ValueError(f"duplicate inference devices are not allowed: {keys}")
	return tuple(resolved_devices)

PYRAMID_POLICY_NONE = "none"
PYRAMID_POLICY_SCALAR = "scalar"
PYRAMID_POLICY_DIRECTION = "direction"
PYRAMID_POLICY_CUSTOM = "custom"
VALID_PYRAMID_POLICIES = frozenset({
	PYRAMID_POLICY_NONE,
	PYRAMID_POLICY_SCALAR,
	PYRAMID_POLICY_DIRECTION,
	PYRAMID_POLICY_CUSTOM,
})


@dataclass(frozen=True)
class OutputChannelSpec:
	"""One logical output channel within an independently resumable product."""

	name: str
	relative_path: str | None = None

	def __post_init__(self) -> None:
		name = str(self.name).strip()
		if not name:
			raise ValueError("output channel name must be non-empty")
		object.__setattr__(self, "name", name)
		if self.relative_path is not None:
			rel = str(self.relative_path).strip()
			if not rel:
				raise ValueError(f"output channel {name!r} relative_path must be non-empty")
			object.__setattr__(self, "relative_path", rel)


@dataclass(frozen=True)
class OutputProductSpec:
	"""A coherent output product whose channel chunks are resumed as one unit."""

	name: str
	level: int
	scaledown: int
	channels: Sequence[str | OutputChannelSpec]
	chunk_size: int
	dtype: Any = np.uint8
	value_range: tuple[float, float] | None = (0.0, 255.0)
	pyramid_policy: str = PYRAMID_POLICY_NONE
	accumulator_channel_count: int | None = None
	inference_scaledown: int | None = None

	def __post_init__(self) -> None:
		name = str(self.name).strip()
		if not name:
			raise ValueError("output product name must be non-empty")
		level = int(self.level)
		scaledown = int(self.scaledown)
		chunk_size = int(self.chunk_size)
		accumulator_channel_count = (
			None
			if self.accumulator_channel_count is None
			else int(self.accumulator_channel_count)
		)
		inference_scaledown = (
			None if self.inference_scaledown is None else int(self.inference_scaledown)
		)
		if level < 0:
			raise ValueError(f"output product {name!r} level must be >= 0")
		if scaledown <= 0:
			raise ValueError(f"output product {name!r} scaledown must be > 0")
		if chunk_size <= 0:
			raise ValueError(f"output product {name!r} chunk_size must be > 0")
		channels = tuple(
			OutputChannelSpec(ch) if isinstance(ch, str) else ch
			for ch in self.channels
		)
		if not channels:
			raise ValueError(f"output product {name!r} must contain at least one channel")
		if any(not isinstance(ch, OutputChannelSpec) for ch in channels):
			raise TypeError("channels must be strings or OutputChannelSpec values")
		channel_names = [ch.name for ch in channels]
		if len(set(channel_names)) != len(channel_names):
			raise ValueError(f"output product {name!r} channel names must be unique")
		if accumulator_channel_count is not None and accumulator_channel_count <= 0:
			raise ValueError(
				f"output product {name!r} accumulator_channel_count must be > 0"
			)
		if inference_scaledown is not None and inference_scaledown <= 0:
			raise ValueError(
				f"output product {name!r} inference_scaledown must be > 0"
			)
		dtype = np.dtype(self.dtype)
		value_range = self.value_range
		if value_range is not None:
			lo, hi = (float(value_range[0]), float(value_range[1]))
			if hi <= lo:
				raise ValueError(f"output product {name!r} value_range must be increasing")
			value_range = (lo, hi)
		pyramid_policy = str(self.pyramid_policy)
		if pyramid_policy not in VALID_PYRAMID_POLICIES:
			raise ValueError(
				f"output product {name!r} pyramid_policy={pyramid_policy!r} "
				f"must be one of {sorted(VALID_PYRAMID_POLICIES)}"
			)
		object.__setattr__(self, "name", name)
		object.__setattr__(self, "level", level)
		object.__setattr__(self, "scaledown", scaledown)
		object.__setattr__(self, "chunk_size", chunk_size)
		object.__setattr__(self, "channels", channels)
		object.__setattr__(self, "dtype", dtype)
		object.__setattr__(self, "value_range", value_range)
		object.__setattr__(self, "pyramid_policy", pyramid_policy)
		object.__setattr__(
			self,
			"accumulator_channel_count",
			accumulator_channel_count,
		)
		object.__setattr__(self, "inference_scaledown", inference_scaledown)

	@property
	def channel_count(self) -> int:
		return len(self.channels)

	@property
	def raw_channel_count(self) -> int:
		return (
			len(self.channels)
			if self.accumulator_channel_count is None
			else int(self.accumulator_channel_count)
		)

	@property
	def channel_names(self) -> tuple[str, ...]:
		return tuple(ch.name for ch in self.channels)


@runtime_checkable
class ModelAdapter(Protocol):
	"""Product-specific model boundary for shared tiled 3D inference."""

	@property
	def output_products(self) -> tuple[OutputProductSpec, ...]:
		"""Products emitted by this model, with coherent channel grouping."""
		...

	def load_model(self, *, device: torch.device) -> Any:
		"""Load and return the product-specific model object."""
		...

	def run_tile_inference(self, model: Any, tile: torch.Tensor, *, device: torch.device) -> Any:
		"""Run one normalized tile through the model and return raw model output."""
		...

	def product_tensors_from_output(self, raw_output: Any) -> Mapping[str, torch.Tensor]:
		"""Split one raw model output into per-product raw tensors."""
		...

	def finalize_product_slab(
		self,
		product: OutputProductSpec,
		raw_slab: np.ndarray,
	) -> ProductTileOutput:
		"""Convert an averaged raw product slab to persisted channel arrays."""
		...


@runtime_checkable
class OutputAdapter(Protocol):
	"""Product-specific chunk completeness, writing, and metadata boundary."""

	def product_chunk_complete(
		self,
		product: OutputProductSpec,
		*,
		chunk_origin_zyx: ChunkOriginZYX,
	) -> bool:
		"""Return True only when every required channel chunk for product exists."""
		...

	def write_product_chunk(
		self,
		product: OutputProductSpec,
		*,
		chunk_origin_zyx: ChunkOriginZYX,
		data: ProductTileOutput,
	) -> None:
		"""Postprocess and atomically write one complete product chunk."""
		...

	def update_metadata(self, products: Sequence[OutputProductSpec]) -> None:
		"""Create or update product-specific manifests, groups, and pyramid metadata."""
		...


def _round_up_to_multiple(v: int, f: int) -> int:
	f = max(1, int(f))
	return ((max(0, int(v)) + f - 1) // f) * f


def _crop_xyzwhd_bounds(
	*,
	shape_zyx: tuple[int, int, int],
	crop_xyzwhd: tuple[int, int, int, int, int, int] | None,
) -> tuple[int, int, int, int, int, int]:
	zs, ys, xs = (int(v) for v in shape_zyx)
	if crop_xyzwhd is None:
		return 0, zs, 0, ys, 0, xs
	x, y, z, w, h, d = (int(v) for v in crop_xyzwhd)
	x0 = max(0, min(x, xs))
	y0 = max(0, min(y, ys))
	z0 = max(0, min(z, zs))
	x1 = max(x0, min(x + max(0, w), xs))
	y1 = max(y0, min(y + max(0, h), ys))
	z1 = max(z0, min(z + max(0, d), zs))
	return z0, z1, y0, y1, x0, x1


def _ds_size(v: int, f: int) -> int:
	# Match interpolate(scale_factor=1/f) floor behavior.
	return max(1, int(v) // int(f))


def _ds_index(v: int, f: int) -> int:
	return max(0, int(v) // int(f))


def _downscaled_tile_clip(local_pos: int, sd: int, tile_down: int, out_size: int):
	start = int(local_pos) // int(sd)
	dst0 = max(0, start)
	dst1 = min(int(out_size), start + int(tile_down))
	if dst1 <= dst0:
		return 0, 0, 0, 0
	src0 = max(0, -start)
	src1 = src0 + (dst1 - dst0)
	return dst0, dst1, src0, src1


def _build_tile_positions(size: int, tile: int, stride: int) -> list[int]:
	size = int(size)
	tile = int(tile)
	stride = max(1, int(stride))
	if size <= tile:
		return [0]
	positions = list(range(0, size - tile + 1, stride))
	last = size - tile
	if positions[-1] != last:
		positions.append(last)
	return positions


def _canonical_local_tile_positions(
	*,
	volume_size: int,
	crop_start: int,
	crop_padded_size: int,
	tile_size: int,
	stride: int,
	border: int,
	scaledown_multiple: int,
) -> list[int]:
	"""Return global-lattice tile positions in local padded-crop coordinates."""
	full_padded_size = _round_up_to_multiple(
		int(volume_size) + 2 * int(border),
		max(1, int(scaledown_multiple)),
	)
	out: list[int] = []
	for pos in _build_tile_positions(full_padded_size, int(tile_size), int(stride)):
		local_pos = int(pos) - int(crop_start)
		if local_pos < int(crop_padded_size) and local_pos + int(tile_size) > 0:
			out.append(local_pos)
	if not out:
		out.append(0)
	return out


def _canonical_tile_positions_for_output_region(
	*,
	volume_size: int,
	output_start: int,
	output_end: int,
	scaledown: int,
	tile_size: int,
	stride: int,
	border: int,
	scaledown_multiple: int,
) -> list[int]:
	"""Return global padded tile positions that contribute to output interval."""
	sd = max(1, int(scaledown))
	full_padded_size = _round_up_to_multiple(
		int(volume_size) + 2 * int(border),
		max(1, int(scaledown_multiple)),
	)
	tile_down = int(tile_size) // sd
	border_down = int(border) // sd
	region0 = int(output_start) + border_down
	region1 = int(output_end) + border_down
	out: list[int] = []
	for pos in _build_tile_positions(full_padded_size, int(tile_size), int(stride)):
		t0 = int(pos) // sd
		t1 = t0 + tile_down
		if t0 < region1 and t1 > region0:
			out.append(int(pos))
	return out


def _pyrdown3d(t: torch.Tensor, *, factor: int) -> torch.Tensor:
	"""Gaussian pyramid downscale for 3D volume tensors."""
	f = int(factor)
	if f <= 1:
		return t
	if (f & (f - 1)) != 0:
		raise ValueError("downscale factor must be a power of 2 for pyramid scaling")
	k = torch.tensor([1, 4, 6, 4, 1], dtype=t.dtype, device=t.device) / 16.0
	while f > 1:
		C = t.shape[0]
		for dim, pad_arg in enumerate([(0,0,0,0,2,2), (0,0,2,2,0,0), (2,2,0,0,0,0)]):
			shape = [1, 1, 1, 1, 1]
			shape[dim + 2] = 5
			kd = k.view(*shape).expand(C, 1, *shape[2:])
			t = F.conv3d(F.pad(t.unsqueeze(0), pad_arg, mode='reflect'), kd, groups=C)[0]
		t = t[:, ::2, ::2, ::2]
		f //= 2
	return t


_input_meta_cache: dict[str, tuple[tuple[int, ...], str]] = {}


def _get_input_meta(zarr_path: str) -> tuple[tuple[int, ...], str]:
	"""Read chunk sizes and dimension_separator from a zarr array's .zarray."""
	if zarr_path in _input_meta_cache:
		return _input_meta_cache[zarr_path]
	import json as _json
	zarray_file = os.path.join(zarr_path, ".zarray")
	with open(zarray_file) as f:
		meta = _json.load(f)
	chunks = tuple(meta["chunks"])
	sep = meta.get("dimension_separator", ".")
	_input_meta_cache[zarr_path] = (chunks, sep)
	return chunks, sep


def _input_has_chunks(zarr_path: str, z0: int, z1: int, y0: int, y1: int,
					  x0: int, x1: int) -> bool:
	"""Check if any chunk files exist in the zarr array for the given region."""
	chunks, sep = _get_input_meta(zarr_path)
	cz, cy, cx = chunks[0], chunks[min(1, len(chunks)-1)], chunks[min(2, len(chunks)-1)]
	for iz in range(max(0, z0 // cz), (z1 + cz - 1) // cz):
		for iy in range(max(0, y0 // cy), (y1 + cy - 1) // cy):
			for ix in range(max(0, x0 // cx), (x1 + cx - 1) // cx):
				path = _zarr_chunk_path(zarr_path, sep, iz, iy, ix)
				if os.path.isfile(path):
					return True
	return False


def _download_one_path(
	zarr_path: str,
	crop_xyzwhd: tuple[int, int, int, int, int, int] | None,
	download_workers: int = 64,
) -> None:
	"""Download chunks for one zarr path from the S3 source in _download metadata."""
	import sys as _sys

	_lasagna_dir = str(Path(__file__).resolve().parent)
	if _lasagna_dir not in _sys.path:
		_sys.path.insert(0, _lasagna_dir)
	from scripts.download_omezarr import download

	p = Path(str(zarr_path).rstrip("/")).resolve()
	group_root = None
	dl_meta = None
	check = p
	for _ in range(5):
		zattrs_path = check / ".zattrs"
		if zattrs_path.is_file():
			zattrs = json.loads(zattrs_path.read_text(encoding="utf-8"))
			if "_download" in zattrs:
				group_root = check
				dl_meta = zattrs["_download"]
				break
		if check.parent == check:
			break
		check = check.parent

	if group_root is None or dl_meta is None:
		raise ValueError(
			f"no _download metadata found walking up from {zarr_path} - "
			"run download_omezarr.py on this volume first "
			"(it records the S3 source), or pass --no-download to skip"
		)

	scales: list[int] | None = None
	if p.name.isdigit():
		scales = [int(p.name)]

	bbox: tuple[int, int, int, int, int, int] | None = None
	if crop_xyzwhd is not None:
		x, y, z, w, h, d = crop_xyzwhd
		bbox = (x, y, z, x + w, y + h, z + d)

	source_uri = dl_meta["source"]
	anon = dl_meta.get("anon", False)
	region = dl_meta.get("region")

	print(
		f"[predict3d] downloading {source_uri} "
		f"scales={scales or 'all'} dest={group_root} ...",
		flush=True,
	)
	ret = download(
		source=source_uri,
		dest=str(group_root),
		scales=scales,
		bbox_xyzxyz=bbox,
		anon=anon,
		region=region,
		workers=int(download_workers),
	)
	if ret != 0:
		raise RuntimeError(f"download from {source_uri} failed (exit {ret})")


def _auto_download(
	input_path: str,
	crop_xyzwhd: tuple[int, int, int, int, int, int] | None,
	pred_dt_path: str | None = None,
	download_workers: int = 64,
) -> None:
	"""Auto-download input and optional pred-dt data from S3 metadata."""
	if int(download_workers) <= 0:
		raise ValueError("download_workers must be a positive integer")
	_download_one_path(input_path, crop_xyzwhd, int(download_workers))
	if pred_dt_path:
		_download_one_path(pred_dt_path, crop_xyzwhd, int(download_workers))
	print("[predict3d] all downloads complete", flush=True)


def _resolve_base_shape(
	input_path: str,
	base_ref: str | None,
	base_scale: int | None,
) -> tuple[int, int, int] | None:
	"""Resolve base_shape_zyx from --base-ref/--base-scale or OME-Zarr level 0."""
	if base_ref is not None:
		ref = zarr.open(str(base_ref), mode="r")
		if hasattr(ref, "shape"):
			sh = tuple(int(v) for v in ref.shape)
			if len(sh) == 4:
				sh = sh[1:]
			if len(sh) != 3:
				raise ValueError(
					f"--base-ref array must be 3D or 4D (CZYX), got shape={sh}"
				)
		else:
			raise ValueError(f"--base-ref must point to a zarr array, got group: {base_ref}")
		scale = base_scale if base_scale is not None else 0
		factor = 2 ** int(scale)
		return (sh[0] * factor, sh[1] * factor, sh[2] * factor)

	try:
		inp = Path(str(input_path).rstrip("/"))
		group_path = inp.parent if inp.name.isdigit() else inp

		level0_zarray = group_path / "0" / ".zarray"
		if level0_zarray.is_file():
			with level0_zarray.open("r", encoding="utf-8") as handle:
				meta = json.load(handle)
			sh = tuple(int(v) for v in meta["shape"])
			if len(sh) == 3:
				print(f"[predict3d] base shape from level 0 .zarray: {sh}", flush=True)
				return sh

		zattrs_path = group_path / ".zattrs"
		if zattrs_path.is_file():
			with zattrs_path.open("r", encoding="utf-8") as handle:
				zattrs = json.load(handle)
			ms = zattrs.get("multiscales", [])
			if ms:
				grp = zarr.open_group(str(group_path), mode="r")
				if "0" in [str(k) for k in grp.keys()]:
					arr = grp["0"]
					sh = tuple(int(v) for v in arr.shape)
					if len(sh) == 3:
						print(f"[predict3d] base shape from level 0 array: {sh}", flush=True)
						return sh

		grp = zarr.open_group(str(group_path), mode="r")
		level_keys = sorted(int(k) for k in grp.keys() if k.isdigit())
		if level_keys:
			finest_lv = level_keys[0]
			arr = grp[str(finest_lv)]
			sh = tuple(int(v) for v in arr.shape)
			if len(sh) == 3:
				factor = 2 ** finest_lv
				base = (sh[0] * factor, sh[1] * factor, sh[2] * factor)
				print(
					f"[predict3d] WARNING: base shape estimated from level {finest_lv} "
					f"shape={sh} x {factor} -> {base} (may be off by a few voxels)",
					flush=True,
				)
				return base
	except Exception:
		pass
	return None


def _invalidate_pyramid_chunks(omezarr_path: str, data_level: int, n_levels: int,
							   iz: int, iy: int, ix: int) -> None:
	"""Delete coarser pyramid chunks that depend on data chunk (iz, iy, ix)."""
	sep = _omezarr_dim_sep(omezarr_path, data_level)
	for lv in range(data_level + 1, n_levels):
		iz, iy, ix = iz // 2, iy // 2, ix // 2
		level_path = os.path.join(omezarr_path, str(lv))
		path = _zarr_chunk_path(level_path, sep, iz, iy, ix)
		try:
			os.unlink(path)
		except FileNotFoundError:
			pass


def _zarr_chunk_path(level_path: str, sep: str, iz: int, iy: int, ix: int) -> str:
	"""Filesystem path for a zarr chunk within a level directory."""
	if sep == "/":
		return os.path.join(level_path, str(iz), str(iy), str(ix))
	return os.path.join(level_path, f"{iz}{sep}{iy}{sep}{ix}")


def _remove_path_quiet(path: str | Path) -> bool:
	"""Remove a temp file/dir if it exists. Returns True when anything was removed."""
	p = Path(path)
	try:
		if p.is_dir():
			shutil.rmtree(p)
			return True
		if p.exists():
			p.unlink()
			return True
	except FileNotFoundError:
		return False
	except OSError:
		return False
	return False


def _pid_is_running(pid: int) -> bool:
	if pid <= 0:
		return False
	try:
		os.kill(int(pid), 0)
	except ProcessLookupError:
		return False
	except PermissionError:
		return True
	except OSError:
		return False
	return True


def _predict3d_temp_pid(name: str) -> int | None:
	"""Best-effort pid extraction from predict3d temp artifact names."""
	if name.startswith(".tmp."):
		marker = ".ome.zarr."
		pos = name.find(marker)
		if pos >= 0:
			tail = name[pos + len(marker):].split(".")
			if len(tail) >= 3 and tail[1].isdigit():
				return int(tail[1])
	if name.startswith(".predict3d_pid"):
		rest = name[len(".predict3d_pid"):]
		pid_txt = rest.split("_", 1)[0]
		if pid_txt.isdigit():
			return int(pid_txt)
	return None


def _cleanup_predict3d_temp_files(
	out_dir: str | Path,
	prefix: str = "",
	*,
	remove_current_process: bool = False,
) -> int:
	"""Remove stale predict3d temp files/dirs in one predict3d output directory.

	All predict3d temp artifacts in the output directory are considered, not only
	the current output prefix. Pid-bearing temp paths owned by a live process are
	left alone so concurrent runs are not damaged; normal finish may remove this
	process's own leftovers by passing ``remove_current_process=True``.
	"""
	root = Path(out_dir)
	if not root.is_dir():
		return 0
	_ = prefix  # kept for old tests/callers; cleanup is directory-wide by design.
	removed = 0
	for child in root.iterdir():
		name = child.name
		is_tmp_chunk = name.startswith(".tmp.") and ".ome.zarr." in name
		is_tmp_acc = name.startswith(".predict3d_")
		if is_tmp_chunk or is_tmp_acc:
			pid = _predict3d_temp_pid(name)
			if (
				pid is not None
				and _pid_is_running(pid)
				and not (remove_current_process and pid == os.getpid())
			):
				continue
			removed += int(_remove_path_quiet(child))
	return removed


def _atomic_zarr_write(omezarr_path: str, level: int,
					   z0: int, y0: int, x0: int,
					   z1: int, y1: int, x1: int,
					   data: np.ndarray, chunk_size: int,
					   n_levels: int = 0) -> None:
	"""Write data to a temp zarr level, then atomically rename chunks into the real output.
	If n_levels > 0, also invalidates coarser pyramid chunks that depend on the written data."""
	sep = _omezarr_dim_sep(omezarr_path, level)
	level_path = os.path.join(omezarr_path, str(level))
	out_dir = os.path.dirname(omezarr_path)
	zarr_name = os.path.basename(omezarr_path)
	tmp_path = os.path.join(
		out_dir,
		f".tmp.{zarr_name}.{level}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}",
	)

	try:
		os.makedirs(tmp_path, exist_ok=True)
		zarray_src = os.path.join(level_path, ".zarray")
		zarray_dst = os.path.join(tmp_path, ".zarray")
		if not os.path.isfile(zarray_dst) and os.path.isfile(zarray_src):
			shutil.copy2(zarray_src, zarray_dst)

		tmp_arr = zarr.open(tmp_path, mode="r+")
		tmp_arr[z0:z1, y0:y1, x0:x1] = data

		for cz in range(z0, z1, chunk_size):
			for cy in range(y0, y1, chunk_size):
				for cx in range(x0, x1, chunk_size):
					iz, iy, ix = cz // chunk_size, cy // chunk_size, cx // chunk_size
					src = _zarr_chunk_path(tmp_path, sep, iz, iy, ix)
					dst = _zarr_chunk_path(level_path, sep, iz, iy, ix)
					if os.path.isfile(src):
						os.makedirs(os.path.dirname(dst), exist_ok=True)
						if n_levels > 0:
							_invalidate_pyramid_chunks(omezarr_path, level, n_levels, iz, iy, ix)
						os.replace(src, dst)
	finally:
		_remove_path_quiet(tmp_path)


def _omezarr_dim_sep(omezarr_path: str, level: int) -> str:
	"""Read dimension_separator from .zarray metadata. Defaults to '.'."""
	import json as _json
	zarray_path = os.path.join(omezarr_path, str(level), ".zarray")
	try:
		with open(zarray_path) as f:
			return _json.load(f).get("dimension_separator", ".")
	except Exception:
		return "."


_dim_sep_cache: dict[tuple[str, int], str] = {}


def _omezarr_chunk_exists(omezarr_path: str, level: int, z: int, y: int, x: int, chunk_size: int) -> bool:
	"""Check if an OME-Zarr chunk file exists on disk."""
	key = (omezarr_path, level)
	if key not in _dim_sep_cache:
		_dim_sep_cache[key] = _omezarr_dim_sep(omezarr_path, level)
	sep = _dim_sep_cache[key]
	iz, iy, ix = z // chunk_size, y // chunk_size, x // chunk_size
	if sep == "/":
		chunk_path = os.path.join(omezarr_path, str(level), str(iz), str(iy), str(ix))
	else:
		chunk_path = os.path.join(omezarr_path, str(level), f"{iz}{sep}{iy}{sep}{ix}")
	return os.path.isfile(chunk_path)


def _omezarr_chunk_group_complete(
	paths: tuple[str, ...],
	level: int,
	z: int,
	y: int,
	x: int,
	chunk_size: int,
) -> bool:
	"""A product chunk is complete only when every required channel chunk exists."""
	return all(_omezarr_chunk_exists(path, level, z, y, x, chunk_size) for path in paths)


class OmeZarrOutputAdapter:
	"""Generic OME-Zarr chunk completeness/write adapter for predict3d products."""

	def __init__(self, *, products: Sequence[OutputProductSpec], n_levels: int) -> None:
		self.products = tuple(products)
		self.n_levels = int(n_levels)
		self._products_by_name = {product.name: product for product in self.products}

	def product_by_name(self, name: str) -> OutputProductSpec:
		return self._products_by_name[name]

	@staticmethod
	def channel_path(channel: OutputChannelSpec) -> str:
		if channel.relative_path is None:
			raise ValueError(f"output channel {channel.name!r} has no OME-Zarr path")
		return str(channel.relative_path)

	def product_chunk_complete(
		self,
		product: OutputProductSpec,
		*,
		chunk_origin_zyx: ChunkOriginZYX,
	) -> bool:
		z, y, x = (int(v) for v in chunk_origin_zyx)
		if product.channel_count == 1:
			path = self.channel_path(product.channels[0])
			return _omezarr_chunk_exists(path, product.level, z, y, x, product.chunk_size)
		paths = tuple(self.channel_path(channel) for channel in product.channels)
		return _omezarr_chunk_group_complete(paths, product.level, z, y, x, product.chunk_size)

	def channel_chunk_exists(
		self,
		product: OutputProductSpec,
		channel_name: str,
		*,
		chunk_origin_zyx: ChunkOriginZYX,
	) -> bool:
		z, y, x = (int(v) for v in chunk_origin_zyx)
		for channel in product.channels:
			if channel.name == channel_name:
				return _omezarr_chunk_exists(
					self.channel_path(channel),
					product.level,
					z,
					y,
					x,
					product.chunk_size,
				)
		raise KeyError(channel_name)

	def write_product_chunk(
		self,
		product: OutputProductSpec,
		*,
		chunk_origin_zyx: ChunkOriginZYX,
		data: ProductTileOutput,
	) -> None:
		z, y, x = (int(v) for v in chunk_origin_zyx)
		for channel in product.channels:
			if channel.name not in data:
				continue
			block = np.ascontiguousarray(data[channel.name])
			if block.ndim != 3:
				raise ValueError(
					f"output channel {channel.name!r} chunk must be 3D, got shape={block.shape}"
				)
			wz, wy, wx = (int(v) for v in block.shape)
			if wz <= 0 or wy <= 0 or wx <= 0:
				continue
			_atomic_zarr_write(
				self.channel_path(channel),
				product.level,
				z,
				y,
				x,
				z + wz,
				y + wy,
				x + wx,
				block.astype(product.dtype, copy=False),
				product.chunk_size,
				self.n_levels,
			)

	def update_metadata(self, products: Sequence[OutputProductSpec]) -> None:
		_ = products


def _format_eta(seconds: float) -> str:
	seconds = max(0.0, float(seconds))
	return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _eta_from_processed_rate(time_sum: float, processed: int, remaining: int) -> float | None:
	remaining = max(0, int(remaining))
	processed = int(processed)
	if remaining == 0:
		return 0.0 if processed > 0 else None
	if processed <= 0:
		return None
	return max(0.0, float(time_sum) / float(processed) * float(remaining))


def _predict3d_overall_eta(progress: dict | None) -> str:
	if progress is None:
		return ""
	eta = 0.0
	have_rate = False
	tile_eta = _eta_from_processed_rate(
		float(progress.get("tile_time_sum", 0.0)),
		int(progress.get("tiles_processed", 0)),
		int(progress.get(
			"tiles_remaining_est",
			max(0, int(progress.get("tiles_total", 0)) - int(progress.get("tiles_done", 0))),
		)),
	)
	if tile_eta is not None:
		eta += tile_eta
		have_rate = True
	edt_eta = _eta_from_processed_rate(
		float(progress.get("edt_time_sum", 0.0)),
		int(progress.get("edt_processed", 0)),
		int(progress.get(
			"edt_remaining_est",
			max(0, int(progress.get("edt_total_est", 0)) - int(progress.get("edt_done", 0))),
		)),
	)
	if edt_eta is not None:
		eta += edt_eta
		have_rate = True
	if not have_rate:
		return ""
	return f" | overall eta {_format_eta(eta)}"


def _predict3d_finalized_status(progress: dict | None) -> str:
	if progress is None or "finalized_base_z" not in progress:
		return ""
	final_z = int(progress.get("finalized_base_z", 0))
	total_z = int(progress.get("finalized_base_z_total", 0))
	cos_z = int(progress.get("finalized_cos_base_z", final_z))
	other_z = int(progress.get("finalized_other_base_z", final_z))
	if total_z <= 0:
		return f" final_z={final_z}"
	if cos_z != other_z:
		return f" final_z={final_z}/{total_z} (cos={cos_z} other={other_z})"
	return f" final_z={final_z}/{total_z}"


def _predict3d_progress_line(progress: dict) -> str:
	total = max(1, int(progress.get("tiles_total", 0)))
	done = int(progress.get("tiles_done", 0))
	processed = int(progress.get("tiles_processed", 0))
	tile_time_sum = float(progress.get("tile_time_sum", 0.0))
	tile_eta = _eta_from_processed_rate(
		tile_time_sum,
		processed,
		int(progress.get("tiles_remaining_est", max(0, total - done))),
	)
	if tile_eta is None:
		eta_text = "--:--"
	else:
		eta_text = _format_eta(tile_eta)
	avg = ""
	if processed > 0:
		avg = f" avg={1000.0 * tile_time_sum / processed:.0f}ms/tile"
	bar_w = 30
	fill = int(round(done / total * bar_w))
	fill = max(0, min(bar_w, fill))
	bar = "#" * fill + "-" * (bar_w - fill)
	live_cache = progress.get("live_cache")
	cache_status = ""
	if isinstance(live_cache, Mapping):
		resident = int(live_cache.get("resident_bytes", 0)) / float(1 << 40)
		target = int(live_cache.get("target_bytes", 0)) / float(1 << 40)
		safe_plane = int(live_cache.get("safe_plane_exclusive", 0))
		downloaded = int(live_cache.get("downloaded_chunks", 0))
		missing = int(live_cache.get("missing_chunks", live_cache.get("missing_requests", 0)))
		inflight = int(live_cache.get("inflight_downloads", 0))
		evicted = int(live_cache.get("evicted_planes", 0))
		rate = float(live_cache.get("download_rate_bytes_s", 0.0)) / float(1 << 20)
		over = "+" if live_cache.get("over_target") else ""
		cache_status = (
			f" cache={resident:.2f}/{target:.2f}TiB{over} dl={downloaded} "
			f"miss={missing} inflight={inflight} rate={rate:.1f}MiB/s "
			f"safe_zchunk={safe_plane} evicted={evicted}"
		)
	return (
		f"[predict3d] [{bar}] {done}/{total} tiles "
		f"({100.0 * done / total:.1f}%) "
		f"eta {eta_text}"
		f"{avg}"
		f"{_predict3d_overall_eta(progress)}"
		f"{_predict3d_finalized_status(progress)}"
		f"{cache_status}"
	)


class _Predict3dProgressEmitter:
	"""Render identical live progress through a terminal or manager pipe."""

	def __init__(
		self, *, refresh_seconds: float = 1.0, history_seconds: float = 60.0,
		clock: Any | None = None,
	) -> None:
		self._refresh_seconds = float(refresh_seconds)
		self._history_seconds = float(history_seconds)
		self._clock = clock or time.monotonic
		now = float(self._clock())
		self._last_refresh = now
		self._last_history = now
		self._last_done = 0
		self._line_open = False

	def emit(self, status: str, *, done: int, total: int) -> None:
		now = float(self._clock())
		changed = int(done) != self._last_done
		self._last_done = int(done)
		if not changed:
			return
		if int(done) >= int(total) or now - self._last_history >= self._history_seconds:
			print(("\r" if self._line_open else "") + status, flush=True)
			self._line_open = False
			self._last_refresh = now
			self._last_history = now
		elif now - self._last_refresh >= self._refresh_seconds:
			print(f"\r{status}  ", end="", flush=True)
			self._line_open = True
			self._last_refresh = now

	def finish_line(self) -> None:
		if self._line_open:
			print(flush=True)
			self._line_open = False


def _iter_chunk_origins_for_region(
	z0: int,
	z1: int,
	y0: int,
	y1: int,
	x0: int,
	x1: int,
	chunk_size: int,
	shape_zyx: tuple[int, int, int],
):
	"""Yield global chunk origins intersecting a half-open region."""
	zs, ys, xs = (int(v) for v in shape_zyx)
	z0 = max(0, min(int(z0), zs))
	y0 = max(0, min(int(y0), ys))
	x0 = max(0, min(int(x0), xs))
	z1 = max(z0, min(int(z1), zs))
	y1 = max(y0, min(int(y1), ys))
	x1 = max(x0, min(int(x1), xs))
	if z1 <= z0 or y1 <= y0 or x1 <= x0:
		return
	cs = int(chunk_size)
	for z in range((z0 // cs) * cs, ((z1 + cs - 1) // cs) * cs, cs):
		for y in range((y0 // cs) * cs, ((y1 + cs - 1) // cs) * cs, cs):
			for x in range((x0 // cs) * cs, ((x1 + cs - 1) // cs) * cs, cs):
				yield z, y, x


def _omezarr_level_shape(
	base_shape: tuple[int, int, int], level: int,
) -> tuple[int, int, int]:
	"""Shape at a given pyramid level (halving with ceil, like OME-Zarr)."""
	z, y, x = (int(v) for v in base_shape)
	for _ in range(max(0, int(level))):
		z = max(1, (z + 1) // 2)
		y = max(1, (y + 1) // 2)
		x = max(1, (x + 1) // 2)
	return z, y, x


def omezarr_compressor_config(name: str) -> dict[str, Any] | None:
	"""Return the exact Zarr-v2 compressor metadata for a public CLI choice."""
	value = str(name).strip().lower()
	if value == DEFAULT_OME_COMPRESSOR:
		return dict(_BLOSC_ZSTD_CONFIG)
	if value == "none":
		return None
	raise ValueError(
		f"unsupported OME-Zarr compressor {name!r}; expected one of "
		f"{', '.join(OME_COMPRESSOR_CHOICES)}"
	)


def _omezarr_compressor(name: str) -> numcodecs.abc.Codec | None:
	config = omezarr_compressor_config(name)
	return numcodecs.get_codec(config) if config is not None else None


def _read_level_compressor_config(path: str | Path, level: int) -> dict[str, Any] | None:
	with (Path(path) / str(level) / ".zarray").open(encoding="utf-8") as handle:
		metadata = json.load(handle)
	return metadata.get("compressor")


def _create_omezarr(
	path: str,
	base_shape_zyx: tuple[int, int, int],
	first_level: int,
	n_levels: int,
	chunk: int,
	channel_name: str,
	compressor: str = DEFAULT_OME_COMPRESSOR,
) -> zarr.Group:
	"""Create an OME-Zarr group with pyramid level arrays."""
	codec = _omezarr_compressor(compressor)
	try:
		g = zarr.open_group(str(path), mode="w", zarr_format=2)
	except TypeError:
		g = zarr.open_group(str(path), mode="w")
	datasets = []
	for lv in range(first_level, n_levels):
		sh = _omezarr_level_shape(base_shape_zyx, lv)
		chunks = (min(sh[0], chunk), min(sh[1], chunk), min(sh[2], chunk))
		try:
			g.create_array(
				str(lv), shape=sh,
				chunks=chunks,
				dtype=np.uint8, fill_value=0, overwrite=True,
				compressor=codec,
				chunk_key_encoding={"name": "v2", "separator": "/"},
			)
		except (AttributeError, TypeError):
			try:
				g.create_dataset(
					str(lv), shape=sh,
					chunks=chunks,
					dtype=np.uint8, fill_value=0, overwrite=True,
					compressor=codec,
					dimension_separator="/",
				)
			except TypeError:
				g.create_dataset(
					str(lv), shape=sh,
					chunks=chunks,
					dtype=np.uint8, fill_value=0, overwrite=True,
					compressor=codec,
				)
		datasets.append({
			"path": str(lv),
			"coordinateTransformations": [{"type": "scale", "scale": [float(2 ** lv)] * 3}],
		})
	g.attrs["multiscales"] = [{
		"version": "0.4",
		"name": channel_name,
		"axes": [
			{"name": "z", "type": "space", "unit": "pixel"},
			{"name": "y", "type": "space", "unit": "pixel"},
			{"name": "x", "type": "space", "unit": "pixel"},
		],
		"datasets": datasets,
	}]
	set_pyramid_metadata(g, method="mean_pool2x")
	return g


def _open_or_create_omezarr(
	path: str,
	base_shape_zyx: tuple[int, int, int],
	first_level: int,
	n_levels: int,
	chunk: int,
	channel_name: str,
	compressor: str = DEFAULT_OME_COMPRESSOR,
) -> zarr.Group:
	"""Open existing OME-Zarr group or create a new one."""
	if os.path.exists(path):
		try:
			g = zarr.open_group(str(path), mode="r+")
			expected = _omezarr_level_shape(base_shape_zyx, first_level)
			arr = g[str(first_level)]
			if tuple(int(v) for v in arr.shape) == expected:
				import json as _json
				zarray_path = os.path.join(path, str(first_level), ".zarray")
				if os.path.isfile(zarray_path):
					with open(zarray_path) as f:
						meta = _json.load(f)
					zfmt = meta.get("zarr_format", None)
					if zfmt != 2:
						raise ValueError(
							f"{path} level {first_level} has zarr_format={zfmt}, expected 2. "
							"Delete and re-create the output."
						)
				requested = omezarr_compressor_config(compressor)
				actual_by_level = {
					int(level): _read_level_compressor_config(path, int(level))
					for level in range(int(first_level), int(n_levels))
					if (Path(path) / str(level) / ".zarray").is_file()
				}
				mismatched = {
					level: actual for level, actual in actual_by_level.items()
					if actual != requested
				}
				if mismatched:
					print(
						f"[predict3d] WARNING: preserving existing compressor(s) for "
						f"{os.path.basename(path)} despite requested {requested}: "
						f"{mismatched}",
						flush=True,
					)
				print(f"[predict3d] reusing existing {os.path.basename(path)} "
					  f"(level {first_level} shape={expected})", flush=True)
				return g
		except (KeyError, ValueError):
			raise
		except Exception:
			pass
		print(f"[predict3d] {path} shape mismatch, recreating", flush=True)
	print(f"[predict3d] creating new {os.path.basename(path)} "
		  f"(levels {first_level}-{n_levels-1})", flush=True)
	return _create_omezarr(
		path, base_shape_zyx, first_level, n_levels, chunk, channel_name, compressor,
	)


def create_product_omezarr_groups(
	*,
	products: Sequence[OutputProductSpec],
	base_shape_zyx: tuple[int, int, int],
	n_levels: int,
	ome_chunk: int,
	ome_compressor: str = DEFAULT_OME_COMPRESSOR,
) -> dict[str, zarr.Group]:
	"""Create or open per-channel OME-Zarr groups for product specs."""
	groups: dict[str, zarr.Group] = {}
	for product in products:
		for channel in product.channels:
			path = OmeZarrOutputAdapter.channel_path(channel)
			groups[channel.name] = _open_or_create_omezarr(
				path,
				base_shape_zyx,
				int(product.level),
				int(n_levels),
				int(ome_chunk),
				channel.name,
				ome_compressor,
			)
	return groups


def _manifest_zarr_path(manifest_path: Path, path: str | Path, level: int) -> str:
	root_abs = manifest_path.parent.resolve()
	path_abs = Path(path).resolve()
	try:
		rel = path_abs.relative_to(root_abs).as_posix()
	except ValueError:
		rel = path_abs.as_posix()
	return f"{rel}/{int(level)}"


def write_lasagna_product_manifest(
	*,
	output_path: str | Path,
	products: Sequence[OutputProductSpec],
	base_shape_zyx: tuple[int, int, int],
	crop_xyzwhd_base: tuple[int, int, int, int, int, int] | None = None,
	source_to_base: float = 1.0,
	grad_mag_factor: float | None = None,
	provenance_json: str | None = None,
) -> None:
	"""Write a Lasagna manifest whose groups come directly from product specs."""
	try:
		from lasagna_volume import ChannelGroup, LasagnaVolume
	except ImportError:  # pragma: no cover - package import mode.
		from lasagna.lasagna_volume import ChannelGroup, LasagnaVolume

	manifest_path = Path(output_path)
	preexisting = manifest_path.exists()
	if preexisting:
		vol = LasagnaVolume.load(manifest_path)
		vol.base_shape_zyx = tuple(int(v) for v in base_shape_zyx)
	else:
		vol = LasagnaVolume(
			path=manifest_path.resolve(),
			source_to_base=float(source_to_base),
			base_shape_zyx=tuple(int(v) for v in base_shape_zyx),
		)
	if grad_mag_factor is not None:
		vol.grad_mag_factor = float(grad_mag_factor)
	if provenance_json is not None:
		vol.provenance_json = str(provenance_json)
	if crop_xyzwhd_base is not None:
		vol.add_crop(tuple(int(v) for v in crop_xyzwhd_base))
	groups: dict[str, ChannelGroup] = {}
	for product in products:
		for channel in product.channels:
			groups[channel.name] = ChannelGroup(
				zarr_path=_manifest_zarr_path(
					manifest_path,
					OmeZarrOutputAdapter.channel_path(channel),
					int(product.level),
				),
				scaledown=int(product.level),
				channels=[channel.name],
			)
	vol.groups = groups
	vol.save(
		backup_existing=preexisting,
		backup_suffix=time.strftime("%Y%m%d_%H%M%S"),
	)


def _build_omezarr_pyramid(
	omezarr_path: str,
	data_level: int,
	n_levels: int,
	chunk: int,
	workers: int = 0,
	crop_zyx: tuple[int, int, int, int, int, int] | None = None,
	label: str = "",
	zero_overrides: bool = False,
	scan_existing_source_chunks: bool = False,
) -> None:
	"""Build coarser scalar pyramid levels by chunked 2x pooling."""
	build_scalar_omezarr_pyramid(
		omezarr_path,
		data_level,
		n_levels,
		chunk,
		workers=workers,
		crop_zyx=crop_zyx,
		label=label,
		zero_overrides=zero_overrides,
		scan_existing_source_chunks=scan_existing_source_chunks,
	)


def _product_channel_kind(channel_name: str) -> str:
	name = str(channel_name)
	for kind in ("presence", "grad_mag", "pred_dt", "cos", "nx", "ny"):
		if name == kind or name.endswith(f"_{kind}"):
			return kind
	return name


def build_product_omezarr_pyramids(
	*,
	products: Sequence[OutputProductSpec],
	n_levels: int,
	ome_chunk: int,
	crop_zyx: tuple[int, int, int, int, int, int] | None = None,
	crop_zyx_by_product: Mapping[str, tuple[int, int, int, int, int, int]] | None = None,
	workers: int = 0,
) -> None:
	"""Build standard scalar and normal-pair pyramids for product channels."""
	for product in products:
		if product.pyramid_policy == PYRAMID_POLICY_NONE:
			continue
		product_crop = (
			crop_zyx_by_product.get(product.name)
			if crop_zyx_by_product is not None and product.name in crop_zyx_by_product
			else crop_zyx
		)
		paths_by_kind: dict[str, str] = {
			_product_channel_kind(channel.name): OmeZarrOutputAdapter.channel_path(channel)
			for channel in product.channels
		}
		for channel in product.channels:
			kind = _product_channel_kind(channel.name)
			if kind in {"nx", "ny"}:
				continue
			_build_omezarr_pyramid(
				OmeZarrOutputAdapter.channel_path(channel),
				int(product.level),
				int(n_levels),
				int(ome_chunk),
				workers=int(workers),
				crop_zyx=product_crop,
				label=channel.name,
				zero_overrides=(kind == "grad_mag"),
				scan_existing_source_chunks=True,
			)
		if "nx" in paths_by_kind and "ny" in paths_by_kind:
			build_normal_omezarr_pyramid(
				paths_by_kind["nx"],
				paths_by_kind["ny"],
				int(product.level),
				int(n_levels),
				int(ome_chunk),
				workers=int(workers),
				crop_zyx=product_crop,
				label=product.name,
				scan_existing_source_chunks=True,
			)


def _find_resume_z(omezarr_path: str, level: int) -> int:
	"""Find the highest z-index with non-zero data in an OME-Zarr level."""
	if not os.path.exists(omezarr_path):
		return 0
	try:
		g = zarr.open_group(str(omezarr_path), mode="r")
		arr = g[str(level)]
		z_total = int(arr.shape[0])
		if z_total == 0:
			return 0
		lo, hi = 0, z_total
		mid_z = z_total // 2
		sample = np.asarray(arr[mid_z])
		if not np.any(sample != 0):
			sample = np.asarray(arr[0])
			if not np.any(sample != 0):
				return 0
			hi = mid_z
		while lo < hi - 1:
			mid = (lo + hi) // 2
			sample = np.asarray(arr[mid])
			if np.any(sample != 0):
				lo = mid
			else:
				hi = mid
		return lo + 1
	except Exception:
		return 0


@dataclass(frozen=True)
class _CircularZLayout:
	"""Pure logical layout for a fixed-depth circular Z accumulator."""

	ring_depth: int
	y_size: int
	x_size: int

	@property
	def plane_size(self) -> int:
		return int(self.y_size) * int(self.x_size)

	def split(self, z0: int, z1: int) -> tuple[tuple[int, int, int], ...]:
		"""Return ``(physical_z, logical_z, length)`` pieces for ``[z0,z1)``."""
		z0, z1 = int(z0), int(z1)
		if z1 < z0 or z1 - z0 > self.ring_depth:
			raise ValueError(
				f"logical z range [{z0},{z1}) does not fit ring depth {self.ring_depth}"
			)
		pieces: list[tuple[int, int, int]] = []
		logical = z0
		while logical < z1:
			physical = logical % self.ring_depth
			length = min(z1 - logical, self.ring_depth - physical)
			pieces.append((physical, logical, length))
			logical += length
		return tuple(pieces)


def _plan_circular_z_depth(
	*,
	z_positions: Sequence[int],
	tile_size: int,
	scaledown: int,
	z_size: int,
	chunk_size: int,
	output_begin: int,
	output_end: int,
	retain_one_flush: bool = True,
) -> int:
	"""Compute capacity for one frozen async flush plus the following Z row."""
	sd = max(1, int(scaledown))
	oc = max(1, int(chunk_size))
	ts_out = int(tile_size) // sd
	# Runtime keeps the chunk-aligned flush frontier separate from the physical
	# ring origin.  In particular, ``flushed`` begins at output_begin while the
	# ring still owns the prefix from logical plane zero until a later frontier
	# actually advances and calls discard_before().
	origin = 0
	submitted = int(output_begin)
	pending_to: int | None = None
	max_live = 1
	positions = tuple(int(v) for v in z_positions)
	for index, tz in enumerate(positions):
		az0, az1, _, _ = _downscaled_tile_clip(tz, sd, ts_out, int(z_size))
		max_live = max(max_live, az1 - origin)
		next_tz = positions[index + 1] if index + 1 < len(positions) else int(z_size) * sd
		complete = next_tz // sd
		if complete >= int(output_end):
			flush_to = int(output_end)
		else:
			flush_to = int(output_begin) + (
				max(0, complete - int(output_begin)) // oc
			) * oc
		if flush_to > submitted:
			# Runtime joins and releases the previous interval only at the next
			# advancing frontier.  The row above was therefore accumulated while
			# that previous interval still occupied the ring.
			if pending_to is not None:
				origin = pending_to
			submitted = flush_to
			if retain_one_flush:
				pending_to = flush_to
			else:
				origin = flush_to
				pending_to = None
	return min(int(z_size), max(1, int(max_live)))


class _CircularZBand:
	"""Fixed-depth mmap ring addressed by monotonically increasing logical Z."""

	def __init__(
		self,
		*,
		name: str,
		channel_count: int,
		z_size: int,
		y_size: int,
		x_size: int,
		tmp_dir: str | None,
		prefix: str,
		ring_depth: int | None = None,
		dtype: str | np.dtype = np.float32,
	) -> None:
		self.name = str(name)
		self.channel_count = int(channel_count)
		self.z_size = int(z_size)
		self.y_size = int(y_size)
		self.x_size = int(x_size)
		self.ring_depth = min(
			self.z_size,
			max(1, int(self.z_size if ring_depth is None else ring_depth)),
		)
		self.layout = _CircularZLayout(self.ring_depth, self.y_size, self.x_size)
		self.tmp_dir = tmp_dir
		self.prefix = str(prefix)
		self.dtype = np.dtype(dtype)
		if self.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
			raise ValueError(f"circular accumulator dtype must be float16 or float32, got {self.dtype}")
		self.origin_z = 0
		self._generation = np.full(self.ring_depth, -1, dtype=np.int64)
		self._arrays = [self._new_array(ch) for ch in range(self.channel_count)]

	def _new_array(self, ch: int) -> np.memmap:
		fd, path = tempfile.mkstemp(
			prefix=f".predict3d_pid{os.getpid()}_{self.prefix}{self.name}_ch{ch}_",
			suffix=".tmp",
			dir=self.tmp_dir if self.tmp_dir else None,
		)
		try:
			logical_bytes = (
				max(0, self.ring_depth)
				* max(0, self.y_size)
				* max(0, self.x_size)
				* self.dtype.itemsize
			)
			os.ftruncate(fd, logical_bytes)
		except Exception:
			os.close(fd)
			_remove_path_quiet(path)
			raise
		os.close(fd)
		mm = np.memmap(
			path,
			dtype=self.dtype,
			mode="r+",
			shape=(self.ring_depth, self.y_size, self.x_size),
		)
		mm._lasagna_tmp_path = path
		atexit.register(lambda p=path: os.path.exists(p) and os.unlink(p))
		return mm

	@property
	def end_z(self) -> int:
		return self.z_size

	def mmap_descriptor(self) -> _MmapBandDescriptor:
		paths = tuple(
			str(Path(str(getattr(arr, "_lasagna_tmp_path"))).resolve())
			for arr in self._arrays
		)
		return _MmapBandDescriptor(
			paths=paths,
			shape_zyx=(self.ring_depth, self.y_size, self.x_size),
			dtype=self.dtype.name,
		)

	def validate_frozen(self, z0: int, z1: int) -> None:
		for logical_z in range(int(z0), int(z1)):
			generation = int(self._generation[logical_z % self.ring_depth])
			if generation not in (-1, logical_z):
				raise ValueError(
					f"{self.name} cannot freeze logical z={logical_z}; slot contains z={generation}"
				)

	def ensure(self, z0: int, z1: int) -> None:
		z0 = int(z0)
		z1 = int(z1)
		if z1 <= z0:
			return
		if z0 < self.origin_z:
			raise ValueError(
				f"{self.name} rolling band cannot revisit z={z0}; "
				f"current origin is {self.origin_z}"
			)
		if z1 > self.z_size:
			raise ValueError(
				f"{self.name} rolling band cannot extend to z={z1}; "
				f"logical size is {self.z_size}"
			)
		if z1 - self.origin_z > self.ring_depth:
			raise ValueError(
				f"{self.name} ring overwrite: write end {z1}, origin {self.origin_z}, "
				f"depth {self.ring_depth}"
			)
		for logical_z in range(z0, z1):
			physical_z = logical_z % self.ring_depth
			generation = int(self._generation[physical_z])
			if generation not in (-1, logical_z):
				raise ValueError(
					f"{self.name} would overwrite unflushed z={generation} with z={logical_z}"
				)
			self._generation[physical_z] = logical_z

	def add(
		self,
		ch: int,
		z0: int,
		z1: int,
		y0: int,
		y1: int,
		x0: int,
		x1: int,
		data: np.ndarray,
	) -> None:
		if z1 <= z0 or y1 <= y0 or x1 <= x0:
			return
		self.ensure(z0, z1)
		data_offset = 0
		for physical_z, _logical_z, length in self.layout.split(z0, z1):
			_add_accumulator_view(
				self._arrays[int(ch)][physical_z:physical_z + length, y0:y1, x0:x1],
				data[data_offset:data_offset + length],
			)
			data_offset += length

	def read(
		self, ch: int, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int,
	) -> np.ndarray:
		if z1 <= z0:
			raise ValueError(f"{self.name} rolling band has no data for z=[{z0},{z1})")
		if z0 < self.origin_z or z1 > self.end_z:
			raise ValueError(
				f"{self.name} rolling band missing z=[{z0},{z1}); "
				f"available=[{self.origin_z},{self.end_z})"
			)
		out = np.zeros((int(z1) - int(z0), int(y1) - int(y0), int(x1) - int(x0)), dtype=np.float32)
		for offset, logical_z in enumerate(range(int(z0), int(z1))):
			physical_z = logical_z % self.ring_depth
			generation = int(self._generation[physical_z])
			if generation == logical_z:
				out[offset] = self._arrays[int(ch)][physical_z, y0:y1, x0:x1]
			elif generation != -1:
				raise ValueError(
					f"{self.name} stale logical z={logical_z}; slot contains z={generation}"
				)
		return out

	def view(self, ch: int, z0: int, z1: int) -> np.ndarray:
		"""Compatibility read; returned data is bounded to the requested Z range."""
		return self.read(ch, z0, z1, 0, self.y_size, 0, self.x_size)

	def clear(self, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> None:
		for physical_z, _logical_z, length in self.layout.split(z0, z1):
			for arr in self._arrays:
				arr[physical_z:physical_z + length, y0:y1, x0:x1] = 0.0

	def discard_before(self, z_new: int) -> None:
		z_new = int(z_new)
		if z_new <= self.origin_z:
			return
		z_release = min(z_new, self.z_size)
		for logical_z in range(self.origin_z, z_release):
			physical_z = logical_z % self.ring_depth
			if int(self._generation[physical_z]) == logical_z:
				self._generation[physical_z] = -1
		self.origin_z = z_release

	def _cleanup_array(self, arr: np.ndarray) -> None:
		path = getattr(arr, "_lasagna_tmp_path", None)
		mmap_obj = getattr(arr, "_mmap", None)
		if mmap_obj is not None:
			try:
				mmap_obj.close()
			except Exception:
				pass
		if path:
			_remove_path_quiet(path)

	def cleanup(self) -> None:
		for arr in self._arrays:
			self._cleanup_array(arr)
		self._arrays = []
		self.origin_z = self.z_size


def _tile_read_spec(
	volume_shape: tuple[int, int, int],
	crop_offset: tuple[int, int, int],
	tz: int, ty: int, tx: int,
	tile_size: int,
	border: int,
) -> _TileReadSpec:
	Zv, Yv, Xv = volume_shape
	oz, oy, ox = crop_offset

	src_z0 = tz + oz - border
	src_y0 = ty + oy - border
	src_x0 = tx + ox - border

	src_z1 = src_z0 + tile_size
	src_y1 = src_y0 + tile_size
	src_x1 = src_x0 + tile_size

	rz0 = max(0, src_z0)
	ry0 = max(0, src_y0)
	rx0 = max(0, src_x0)
	rz1 = min(Zv, src_z1)
	ry1 = min(Yv, src_y1)
	rx1 = min(Xv, src_x1)

	pad_before = (rz0 - src_z0, ry0 - src_y0, rx0 - src_x0)
	pad_after = (src_z1 - rz1, src_y1 - ry1, src_x1 - rx1)
	return _TileReadSpec(
		slices_zyx=(
			None if rz1 <= rz0 or ry1 <= ry0 or rx1 <= rx0
			else (slice(rz0, rz1), slice(ry0, ry1), slice(rx0, rx1))
		),
		pad_width_zyx=(
			(pad_before[0], pad_after[0]),
			(pad_before[1], pad_after[1]),
			(pad_before[2], pad_after[2]),
		),
		tile_size=int(tile_size),
	)


def _finalize_tile_read(raw: Any | None, spec: _TileReadSpec) -> np.ndarray:
	if spec.slices_zyx is None:
		# Preserve the historical fully-outside dtype behavior.
		return np.zeros((spec.tile_size, spec.tile_size, spec.tile_size), dtype=np.uint8)
	chunk = np.asarray(raw)
	if any(value > 0 for pair in spec.pad_width_zyx for value in pair):
		chunk = np.pad(chunk, spec.pad_width_zyx, mode="reflect")
	return chunk


def _read_tile_zarr(
	zarr_arr,
	volume_shape: tuple[int, int, int],
	crop_offset: tuple[int, int, int],
	tz: int, ty: int, tx: int,
	tile_size: int | None,
	border: int,
) -> np.ndarray:
	"""Read a single tile from zarr, using reflect-padding only at volume boundaries."""
	spec = _tile_read_spec(
		volume_shape, crop_offset, tz, ty, tx, int(tile_size), border,
	)
	raw = None if spec.slices_zyx is None else zarr_arr[spec.slices_zyx]
	return _finalize_tile_read(raw, spec)


def _input_tile_to_device(tile_np: np.ndarray, device: torch.device) -> torch.Tensor:
	"""Transfer compact integer input and reproduce historical FP32 normalization."""
	if tile_np.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
		raise TypeError(f"predict3d input must be uint8 or uint16, got {tile_np.dtype}")
	if device.type != "cuda":
		if tile_np.dtype == np.uint16:
			prepared = (tile_np // 257).astype(np.uint8).astype(np.float32) / 255.0
		else:
			prepared = tile_np.astype(np.float32) / 255.0
		return torch.from_numpy(prepared).unsqueeze(0).unsqueeze(0).to(device)
	compact = torch.from_numpy(tile_np).unsqueeze(0).unsqueeze(0).to(device)
	if compact.dtype == torch.uint16:
		return torch.div(compact.to(torch.int32), 257, rounding_mode="floor").to(torch.float32).div_(255.0)
	return compact.to(torch.float32).div_(255.0)


class _TensorStoreTileReader:
	"""Bounded asynchronous local Zarr-v2 bounding-box reader."""

	def __init__(
		self, path: str, *, cache_bytes: int, file_io_threads: int,
		data_copy_threads: int,
	) -> None:
		level_path = Path(str(path).rstrip("/"))
		if not (level_path / ".zarray").is_file():
			raise ValueError(
				"TensorStore inference input must be a local Zarr-v2 array path "
				f"containing .zarray, got: {path}"
			)
		try:
			from tensorstore_omezarr import TensorStoreConfig, open_tensorstore, tensorstore_context
		except ImportError:
			from lasagna.tensorstore_omezarr import TensorStoreConfig, open_tensorstore, tensorstore_context
		cfg = TensorStoreConfig(
			cache_pool_bytes=int(cache_bytes), file_io_threads=int(file_io_threads),
			data_copy_threads=int(data_copy_threads),
		)
		self.context = tensorstore_context(cfg)
		self.store = open_tensorstore(level_path, self.context, read=True, write=False)

	def submit(
		self, *, volume_shape: tuple[int, int, int], crop_offset: tuple[int, int, int],
		coord: TileOriginZYX, tile_size: int, border: int, profile: bool = False,
	) -> _TensorStoreReadTask:
		spec = _tile_read_spec(volume_shape, crop_offset, *coord, tile_size, border)
		future = None if spec.slices_zyx is None else self.store[spec.slices_zyx].read()
		task = _TensorStoreReadTask(future=future, spec=spec, submitted_at=time.perf_counter())
		if profile:
			if future is None:
				task.completed_at = task.submitted_at
			else:
				future.add_done_callback(lambda _future: setattr(task, "completed_at", time.perf_counter()))
		return task

	@staticmethod
	def done(task: _TensorStoreReadTask) -> bool:
		return task.future is None or bool(task.future.done())

	@staticmethod
	def result(task: _TensorStoreReadTask) -> tuple[np.ndarray, float]:
		raw = None if task.future is None else task.future.result()
		return _finalize_tile_read(raw, task.spec), time.perf_counter() - task.submitted_at

	@staticmethod
	def cancel(tasks: Iterable[_TensorStoreReadTask]) -> None:
		pending = tuple(task for task in tasks if task.future is not None)
		for task in pending:
			task.future.cancel()
		for task in pending:
			try:
				task.future.result()
			except BaseException:
				pass


@dataclass(frozen=True)
class _SharedArrayLayout:
	name: str
	offset: int
	shape: tuple[int, ...]
	dtype: str

	@property
	def nbytes(self) -> int:
		return int(np.prod(self.shape, dtype=np.int64)) * np.dtype(self.dtype).itemsize


@dataclass(frozen=True)
class _SharedSlotSpec:
	shm_name: str
	nbytes: int
	layouts: tuple[_SharedArrayLayout, ...]


def _packed_layouts(entries: Iterable[tuple[str, tuple[int, ...], Any]]) -> tuple[tuple[_SharedArrayLayout, ...], int]:
	layouts = []
	offset = 0
	for name, shape, dtype in entries:
		offset = ((offset + 63) // 64) * 64
		layout = _SharedArrayLayout(str(name), offset, tuple(int(v) for v in shape), np.dtype(dtype).str)
		layouts.append(layout)
		offset += layout.nbytes
	return tuple(layouts), max(1, ((offset + 63) // 64) * 64)


def _slot_array(shm: shared_memory.SharedMemory, layout: _SharedArrayLayout) -> np.ndarray:
	return np.ndarray(layout.shape, dtype=np.dtype(layout.dtype), buffer=shm.buf, offset=layout.offset)


def _attach_shared_memory(name: str) -> shared_memory.SharedMemory:
	try:
		return shared_memory.SharedMemory(name=name, create=False, track=False)
	except TypeError:  # Python < 3.13
		return shared_memory.SharedMemory(name=name, create=False)


def _create_shared_slots(count: int, size: int) -> list[shared_memory.SharedMemory]:
	created = []
	try:
		for _ in range(int(count)):
			created.append(shared_memory.SharedMemory(create=True, size=int(size)))
		return created
	except BaseException:
		for shm in created:
			shm.close()
			shm.unlink()
		raise


def _limit_native_worker_threads():
	for name in (
		"OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
		"NUMEXPR_NUM_THREADS", "BLOSC_NTHREADS",
	):
		os.environ[name] = "1"
	_limit_zarr_codec_threads()
	try:
		torch.set_num_threads(1)
		torch.set_num_interop_threads(1)
	except RuntimeError:
		pass
	try:
		from threadpoolctl import threadpool_limits
		return threadpool_limits(limits=1)
	except ImportError:
		return None


def _read_mmap_band_chunk(
	descriptor: _MmapBandDescriptor,
	channel: int,
	z0: int, z1: int, y0: int, y1: int, x0: int, x1: int,
	cache: dict[str, np.memmap],
) -> np.ndarray:
	path = descriptor.paths[int(channel)]
	arr = cache.get(path)
	if arr is None:
		arr = np.memmap(
			path, dtype=np.dtype(descriptor.dtype), mode="r",
			shape=descriptor.shape_zyx,
		)
		cache[path] = arr
	out = np.zeros((z1-z0, y1-y0, x1-x0), dtype=np.float32)
	offset = 0
	logical = int(z0)
	while logical < int(z1):
		physical = logical % descriptor.ring_depth
		length = min(int(z1) - logical, descriptor.ring_depth - physical)
		out[offset:offset+length] = arr[physical:physical+length, y0:y1, x0:x1]
		offset += length
		logical += length
	return out


_ACCUMULATOR_NATIVE = None


def _accumulator_native_module():
	global _ACCUMULATOR_NATIVE
	if _ACCUMULATOR_NATIVE is None:
		try:
			import accumulator_add as module
		except ImportError:
			module = False
		_ACCUMULATOR_NATIVE = module
	return _ACCUMULATOR_NATIVE


def _add_accumulator_view(destination: np.ndarray, source: np.ndarray) -> str:
	"""Add float32 source into an f16/f32 destination, using native code when installed."""
	module = _accumulator_native_module()
	if module:
		module.add_inplace(destination, source)
		return str(module.backend())
	# NumPy is the portable no-extension fallback.  For float16 destinations it
	# has the required per-add rounding semantics, albeit without the fast kernel.
	np.add(destination, source, out=destination, casting="unsafe")
	return "numpy"


def _add_mmap_band_chunk(
	descriptor: _MmapBandDescriptor,
	channel: int,
	z0: int, z1: int, y0: int, y1: int, x0: int, x1: int,
	source: np.ndarray,
	cache: dict[str, np.memmap],
) -> str:
	path = descriptor.paths[int(channel)]
	arr = cache.get(path)
	if arr is None:
		arr = np.memmap(path, dtype=np.dtype(descriptor.dtype), mode="r+", shape=descriptor.shape_zyx)
		cache[path] = arr
	offset = 0
	logical = int(z0)
	backend = "numpy"
	while logical < int(z1):
		physical = logical % descriptor.ring_depth
		length = min(int(z1) - logical, descriptor.ring_depth - physical)
		backend = _add_accumulator_view(
			arr[physical:physical + length, y0:y1, x0:x1],
			source[offset:offset + length],
		)
		offset += length
		logical += length
	return backend


def _stable_accumulator_owner(sd: int, origin: ChunkOriginZYX, worker_count: int) -> int:
	"""Stable spatial ownership; unlike hash(), this is invariant across processes."""
	z, y, x = (int(value) for value in origin)
	value = int(sd) * 73856093 + z * 19349663 + y * 83492791 + x * 2654435761
	return int(value % int(worker_count))


def _accumulate_process_main(worker_index: int, task_queue, result_queue) -> None:
	_native_limits = _limit_native_worker_threads()
	shms: dict[str, shared_memory.SharedMemory] = {}
	mmaps: dict[str, np.memmap] = {}
	try:
		while True:
			task = task_queue.get()
			if task is None:
				break
			started = time.perf_counter()
			try:
				shm = shms.get(task.result_spec.shm_name)
				if shm is None:
					shm = _attach_shared_memory(task.result_spec.shm_name)
					shms[task.result_spec.shm_name] = shm
				layouts = {layout.name: layout for layout in task.result_spec.layouts}
				lz0, lz1, ly0, ly1, lx0, lx1 = task.destination
				pz0, pz1, py0, py1, px0, px1 = task.source
				weight = _slot_array(shm, layouts[f"weight:{task.sd}"])[pz0:pz1, py0:py1, px0:px1]
				backend = _add_mmap_band_chunk(
					task.weight, 0, lz0, lz1, ly0, ly1, lx0, lx1, weight, mmaps,
				)
				accumulators = dict(task.accumulators)
				for name in task.dirty_products:
					product = _slot_array(shm, layouts[f"product:{name}"])
					for channel in range(int(product.shape[0])):
						backend = _add_mmap_band_chunk(
							accumulators[name], channel,
							lz0, lz1, ly0, ly1, lx0, lx1,
							product[channel, pz0:pz1, py0:py1, px0:px1], mmaps,
						)
				result_queue.put((
					"result", task.task_id, task.event_seq, worker_index, backend,
					time.perf_counter() - started,
				))
			except BaseException as exc:
				result_queue.put(("error", task.task_id, task.event_seq, worker_index, type(exc).__name__, str(exc)))
				break
	finally:
		for arr in mmaps.values():
			mmap_obj = getattr(arr, "_mmap", None)
			if mmap_obj is not None:
				mmap_obj.close()
		for shm in shms.values():
			shm.close()


def _execute_flush_process_task(
	task: _FlushProcessTask,
	model_adapter: Any,
	output_adapter: Any,
	cache: dict[str, np.memmap],
) -> tuple[int, int, int, float]:
	started = time.perf_counter()
	oz0, oy0, ox0, oz1, oy1, ox1 = task.region
	gz, gy, gx = task.origin
	lz0, ly0, lx0 = task.b + gz - oz0, task.b + gy - oy0, task.b + gx - ox0
	lz1 = min(lz0 + task.oc, task.b + oz1 - oz0)
	ly1 = min(ly0 + task.oc, task.b + oy1 - oy0)
	lx1 = min(lx0 + task.oc, task.b + ox1 - ox0)
	written = 0
	if task.weight_dirty:
		denom = _read_mmap_band_chunk(task.weight, 0, lz0, lz1, ly0, ly1, lx0, lx1, cache)
		if np.any(denom > 0.0):
			np.maximum(denom, 1.0e-7, out=denom)
			products = {product.name: product for product in task.products}
			accumulators = dict(task.accumulators)
			for name in task.dirty_products:
				product = products[name]
				raw = np.stack([
					_read_mmap_band_chunk(
						accumulators[name], ch, lz0, lz1, ly0, ly1, lx0, lx1, cache,
					)
					for ch in range(product.raw_channel_count)
				])
				raw /= denom[None]
				finalizer = getattr(model_adapter, "finalize_product_slab", None)
				if finalizer is None:
					persisted = {
						channel.name: np.clip(raw[i] * 255.0, 0.0, 255.0).astype(product.dtype)
						for i, channel in enumerate(product.channels)
					}
				else:
					persisted = finalizer(product, raw)
				output_adapter.write_product_chunk(
					product, chunk_origin_zyx=task.origin,
					data={channel.name: persisted[channel.name] for channel in product.channels},
				)
				written += 1
				del raw, persisted
		del denom
	return task.batch_id, task.task_id, written, time.perf_counter() - started


def _flush_process_main(
	worker_index: int,
	model_adapter: Any,
	output_adapter: Any,
	task_queue,
	result_queue,
) -> None:
	_native_limits = _limit_native_worker_threads()
	cache: dict[str, np.memmap] = {}
	try:
		while True:
			task = task_queue.get()
			if task is None:
				break
			try:
				batch_id, task_id, written, elapsed = _execute_flush_process_task(
					task, model_adapter, output_adapter, cache,
				)
				result_queue.put(("result", batch_id, task_id, worker_index, written, elapsed))
			except BaseException as exc:
				result_queue.put((
					"error", task.batch_id, task.task_id, worker_index,
					type(exc).__name__, str(exc),
				))
				break
	finally:
		for arr in cache.values():
			mmap_obj = getattr(arr, "_mmap", None)
			if mmap_obj is not None:
				mmap_obj.close()


def _multi_gpu_worker_main(
	worker_index: int,
	device_text: str,
	model_adapter: ModelAdapter,
	model_state: Mapping[str, torch.Tensor] | None,
	tile_size: int,
	blend_ramp: np.ndarray,
	product_scales: Mapping[str, int],
	input_specs: tuple[_SharedSlotSpec, ...],
	result_specs: tuple[_SharedSlotSpec, ...],
	work_queue,
	result_queue,
	profile_pipeline: bool = False,
) -> None:
	"""Persistent spawned GPU worker; queues carry descriptors only."""
	_native_limits = _limit_native_worker_threads()
	device = torch.device(device_text)
	if device.type == "cuda":
		torch.cuda.set_device(device)
	inputs = [_attach_shared_memory(spec.shm_name) for spec in input_specs]
	results = [_attach_shared_memory(spec.shm_name) for spec in result_specs]
	try:
		model = model_adapter.load_model(device=device)
		if model_state is not None:
			prepare_state_load = getattr(model_adapter, "prepare_model_for_state_load", None)
			if prepare_state_load is not None:
				prepare_state_load(model, model_state, device=device)
			model.load_state_dict(model_state, strict=True)
		model.eval()
		ramp_t = torch.as_tensor(blend_ramp, dtype=torch.float32, device=device)
		w_full = ramp_t[:, None, None] * ramp_t[None, :, None] * ramp_t[None, None, :]
		weights_by_scale = {
			sd: (_pyrdown3d(w_full.unsqueeze(0), factor=sd).squeeze(0) if sd > 1 else w_full)
			for sd in sorted(set(int(value) for value in product_scales.values()))
		}
		while True:
			task = work_queue.get()
			if task is None:
				break
			seq, coord, input_slot, result_slot, needed_names = task
			started = time.perf_counter()
			try:
				profile = {} if profile_pipeline else None
				input_layout = input_specs[input_slot].layouts[0]
				tile_np = _slot_array(inputs[input_slot], input_layout)
				cuda_events = {}
				if profile is not None and device.type == "cuda":
					for name in ("h2d", "cuda_convert", "adapter_preprocess", "model_inference", "output"):
						cuda_events[name] = (
							torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True),
						)
				if device.type == "cuda":
					source = torch.from_numpy(tile_np).unsqueeze(0).unsqueeze(0)
					if cuda_events:
						cuda_events["h2d"][0].record()
					stage_started = time.perf_counter()
					compact = source.to(device)
					if profile is not None:
						profile["compact_h2d_wall_s"] = time.perf_counter() - stage_started
						cuda_events["h2d"][1].record()
					# Pageable shared-memory H2D is synchronous, so this slot is now reusable.
					result_queue.put(("input_released", seq, input_slot, worker_index))
					if cuda_events:
						cuda_events["cuda_convert"][0].record()
					stage_started = time.perf_counter()
					if compact.dtype == torch.uint16:
						tile = torch.div(compact.to(torch.int32), 257, rounding_mode="floor").to(torch.float32).div_(255.0)
					elif compact.dtype == torch.uint8:
						tile = compact.to(torch.float32).div_(255.0)
					else:
						raise TypeError(f"predict3d input must be uint8 or uint16, got {tile_np.dtype}")
					if profile is not None:
						profile["cuda_convert_submit_wall_s"] = time.perf_counter() - stage_started
						cuda_events["cuda_convert"][1].record()
				else:
					stage_started = time.perf_counter()
					tile = _input_tile_to_device(tile_np, device)
					if profile is not None:
						profile["cpu_convert_wall_s"] = time.perf_counter() - stage_started
					result_queue.put(("input_released", seq, input_slot, worker_index))
				preprocess = getattr(model_adapter, "preprocess_tile", None)
				if cuda_events:
					cuda_events["adapter_preprocess"][0].record()
				stage_started = time.perf_counter()
				if preprocess is not None:
					tile = preprocess(tile, torch.ones_like(tile, dtype=torch.bool))
				if profile is not None:
					profile["adapter_preprocess_wall_s"] = time.perf_counter() - stage_started
					if cuda_events:
						cuda_events["adapter_preprocess"][1].record()
						cuda_events["model_inference"][0].record()
				stage_started = time.perf_counter()
				with torch.inference_mode():
					raw_output = model_adapter.run_tile_inference(model, tile, device=device)
				if profile is not None:
					profile["model_inference_submit_wall_s"] = time.perf_counter() - stage_started
					if cuda_events:
						cuda_events["model_inference"][1].record()
						cuda_events["output"][0].record()
				stage_started = time.perf_counter()
				tensors = model_adapter.product_tensors_from_output(raw_output)
				layouts = {layout.name: layout for layout in result_specs[result_slot].layouts}
				needed_scales = sorted({int(product_scales[name]) for name in needed_names})
				for sd in needed_scales:
					weight = weights_by_scale[sd]
					_slot_array(results[result_slot], layouts[f"weight:{sd}"])[:] = weight.detach().cpu().numpy()
				for name in needed_names:
					sd = int(product_scales[name])
					weighted = tensors[name][0] * w_full
					if sd > 1:
						weighted = _pyrdown3d(weighted, factor=sd)
					_slot_array(results[result_slot], layouts[f"product:{name}"])[:] = weighted.detach().cpu().numpy()
				if profile is not None:
					profile["output_and_d2h_wall_s"] = time.perf_counter() - stage_started
					if cuda_events:
						cuda_events["output"][1].record()
				sync_started = time.perf_counter()
				if device.type == "cuda":
					torch.cuda.synchronize(device)
				finished = time.perf_counter()
				if profile is not None:
					profile["terminal_sync_wall_s"] = finished - sync_started
					profile["worker_total_wall_s"] = finished - started
					for name, (event_start, event_end) in cuda_events.items():
						profile[f"{name}_cuda_s"] = event_start.elapsed_time(event_end) / 1000.0
					result_queue.put((
						"result", seq, coord, result_slot, tuple(needed_names), worker_index,
						finished - started, profile, finished,
					))
				else:
					result_queue.put((
						"result", seq, coord, result_slot, tuple(needed_names), worker_index,
						finished - started,
					))
			except BaseException as exc:
				result_queue.put(("error", seq, coord, worker_index, type(exc).__name__, str(exc)))
				break
	finally:
		for shm in inputs + results:
			shm.close()


def _iter_canonical_tile_events(
	z_positions: Sequence[int], y_positions: Sequence[int], x_positions: Sequence[int], Zp: int,
) -> Iterator[tuple[int, int, int, bool, int]]:
	"""Yield coordinates lazily plus the next safe row-flush frontier."""
	for iz, tz in enumerate(z_positions):
		next_tz = int(z_positions[iz + 1]) if iz + 1 < len(z_positions) else int(Zp)
		for iy, ty in enumerate(y_positions):
			for ix, tx in enumerate(x_positions):
				yield int(tz), int(ty), int(tx), iy == len(y_positions) - 1 and ix == len(x_positions) - 1, next_tz


def _lock_selected_level_reader(function):
	@wraps(function)
	def wrapped(*args, **kwargs):
		if kwargs.get("live_cache") is not None:
			return function(*args, **kwargs)
		input_path = kwargs.get("input_zarr_path")
		lock = (
			selected_level_lock_for_input(input_path, exclusive=False)
			if input_path is not None else None
		)
		if lock is None:
			return function(*args, **kwargs)
		with lock:
			return function(*args, **kwargs)
	return wrapped


@_lock_selected_level_reader
def run_tiled_inference_3d(
	model,
	zarr_arr,
	*,
	crop_slices: RegionZYX,
	device: torch.device,
	model_adapter: ModelAdapter,
	output_adapter: OutputAdapter,
	products: Sequence[OutputProductSpec],
	output_regions_zyx: Mapping[str, RegionZYX],
	full_output_shapes_zyx: Mapping[str, tuple[int, int, int]],
	input_zarr_path: str | None = None,
	output_scaledown_base: Mapping[str, int] | int | None = None,
	tile_size: int = 256,
	overlap: int = 64,
	border: int = 16,
	tmp_dir: str | None = None,
	progress: dict | None = None,
	temp_prefix: str = "",
	devices: Sequence[torch.device] | None = None,
	model_state: Mapping[str, torch.Tensor] | None = None,
	prefetch_workers: int = 0,
	slots_per_gpu: int = 2,
	flush_workers: int = 0,
	input_reader: str = "python-zarr",
	prefetch_tiles_per_gpu: int = DEFAULT_PREFETCH_TILES_PER_GPU,
	input_cache_bytes: int = DEFAULT_INPUT_CACHE_BYTES,
	input_io_threads: int = DEFAULT_INPUT_IO_THREADS,
	input_copy_threads: int = DEFAULT_INPUT_COPY_THREADS,
	profile_pipeline: bool = False,
	product_accumulator_dtype: str | np.dtype = "float16",
	accumulator_workers: int = 0,
	live_cache: LiveOmeZarrCache | None = None,
) -> None:
	"""Authoritative one-pass, multi-scale tiled inference engine."""
	products = tuple(products)
	if not products:
		raise ValueError("at least one neural output product is required")
	volume_shape = tuple(int(v) for v in zarr_arr.shape)
	z0, z1, y0, y1, x0, x1 = (int(v) for v in crop_slices)
	nz, ny, nx = z1 - z0, y1 - y0, x1 - x0
	pad = max(0, int(border))
	stride = max(1, int(tile_size) - int(overlap))
	resolved_devices = tuple(devices or (device,))
	parallel = len(resolved_devices) > 1
	if parallel and model is not None:
		raise ValueError("parallel tiled inference loads models in workers; pass model=None")
	if not parallel and model is None:
		raise ValueError("single-device tiled inference requires a loaded model")
	if int(slots_per_gpu) <= 0:
		raise ValueError("slots_per_gpu must be > 0")
	if int(prefetch_workers) < 0:
		raise ValueError("prefetch_workers must be >= 0")
	if int(flush_workers) < 0:
		raise ValueError("flush_workers must be >= 0")
	if int(accumulator_workers) < 0:
		raise ValueError("accumulator_workers must be >= 0")
	input_reader = str(input_reader).strip().lower()
	if input_reader not in {"tensorstore", "python-zarr"}:
		raise ValueError("input_reader must be 'tensorstore' or 'python-zarr'")
	if int(prefetch_tiles_per_gpu) <= 0:
		raise ValueError("prefetch_tiles_per_gpu must be > 0")
	if int(input_cache_bytes) < 0:
		raise ValueError("input_cache_bytes must be >= 0")
	if int(input_io_threads) <= 0:
		raise ValueError("input_io_threads must be > 0")
	if int(input_copy_threads) <= 0:
		raise ValueError("input_copy_threads must be > 0")
	product_accumulator_dtype = np.dtype(product_accumulator_dtype)
	if product_accumulator_dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
		raise ValueError(
			"product_accumulator_dtype must be float16 or float32, "
			f"got {product_accumulator_dtype}"
		)
	if input_reader == "tensorstore" and input_zarr_path is None:
		raise ValueError("TensorStore input requires input_zarr_path")
	if live_cache is not None:
		if input_zarr_path is None:
			raise ValueError("live cache requires input_zarr_path")
		if Path(str(input_zarr_path).rstrip("/")).resolve() != live_cache.source.level_path.resolve():
			raise ValueError("live cache selected level does not match input_zarr_path")
		if (z0, z1, y0, y1, x0, x1) != (
			0, volume_shape[0], 0, volume_shape[1], 0, volume_shape[2],
		):
			raise ValueError("live cache supports only full, non-cropped inference")
	scales = {
		p.name: max(1, int(p.inference_scaledown or 1))
		for p in products
	}
	if isinstance(output_scaledown_base, Mapping):
		input_scales = set()
		for product in products:
			base_sd = int(output_scaledown_base[product.name])
			if base_sd != int(product.scaledown):
				raise ValueError(
					f"product {product.name!r} base scaledown mismatch: "
					f"spec={product.scaledown}, runner={base_sd}"
				)
			if base_sd % scales[product.name]:
				raise ValueError(
					f"product {product.name!r} base scaledown {base_sd} is not divisible "
					f"by inference_scaledown {scales[product.name]}"
				)
			input_scales.add(base_sd // scales[product.name])
		if len(input_scales) != 1:
			raise ValueError(f"products disagree on selected input scale: {sorted(input_scales)}")
	sd_max = max(scales.values())
	for sd in sorted(set(scales.values())):
		for label, value in (("tile_size", tile_size), ("stride", stride), ("border", pad)):
			if int(value) % sd:
				raise ValueError(f"{label}={value} must be divisible by inference_scaledown={sd}")
	Zp = _round_up_to_multiple(nz + 2 * pad, sd_max)
	Yp = _round_up_to_multiple(ny + 2 * pad, sd_max)
	Xp = _round_up_to_multiple(nx + 2 * pad, sd_max)
	z_positions = _canonical_local_tile_positions(
		volume_size=volume_shape[0], crop_start=z0, crop_padded_size=Zp,
		tile_size=tile_size, stride=stride, border=pad, scaledown_multiple=sd_max,
	)
	y_positions = _canonical_local_tile_positions(
		volume_size=volume_shape[1], crop_start=y0, crop_padded_size=Yp,
		tile_size=tile_size, stride=stride, border=pad, scaledown_multiple=sd_max,
	)
	x_positions = _canonical_local_tile_positions(
		volume_size=volume_shape[2], crop_start=x0, crop_padded_size=Xp,
		tile_size=tile_size, stride=stride, border=pad, scaledown_multiple=sd_max,
	)

	def _blend_ramp(length: int) -> np.ndarray:
		ramp = np.zeros(length, dtype=np.float32)
		core_start, core_end = min(pad, length), max(min(pad, length), length - pad)
		core_len = core_end - core_start
		if core_len <= 0:
			return ramp
		core = np.ones(core_len, dtype=np.float32)
		ov = min(max(0, int(overlap) - 2 * pad), core_len // 2)
		if ov:
			edges = np.linspace(0.0, 1.0, ov + 1, dtype=np.float32)[1:]
			core[:ov], core[-ov:] = edges, edges[::-1]
		ramp[core_start:core_end] = core
		return ramp

	r = _blend_ramp(int(tile_size))
	w_full = None
	w_by_scale = None
	if not parallel:
		w_full = torch.from_numpy(r[:, None, None] * r[None, :, None] * r[None, None, :]).to(device)
		w_by_scale = {
			sd: (_pyrdown3d(w_full.unsqueeze(0), factor=sd).squeeze(0).cpu().numpy()
				 if sd > 1 else w_full.cpu().numpy())
			for sd in sorted(set(scales.values()))
		}
	groups: dict[int, dict[str, Any]] = {}
	for sd in sorted(set(scales.values())):
		group_products = tuple(p for p in products if scales[p.name] == sd)
		chunk_sizes = {int(p.chunk_size) for p in group_products}
		if len(chunk_sizes) != 1:
			raise ValueError(f"products at inference scale {sd} must share a chunk size")
		oc = chunk_sizes.pop()
		Zo, Yo, Xo = Zp // sd, Yp // sd, Xp // sd
		regions = {p.name: tuple(int(v) for v in output_regions_zyx[p.name]) for p in group_products}
		# Products sharing a denominator must describe the same output region.
		if len(set(regions.values())) != 1:
			raise ValueError(f"products at inference scale {sd} must share output_region_zyx")
		region = next(iter(regions.values()))
		full_shapes = {
			p.name: tuple(int(v) for v in full_output_shapes_zyx[p.name])
			for p in group_products
		}
		if len(set(full_shapes.values())) != 1:
			raise ValueError(f"products at inference scale {sd} must share full_output_shapes_zyx")
		full_shape = next(iter(full_shapes.values()))
		oz0, oy0, ox0, oz1, oy1, ox1 = region
		b = pad // sd
		depth = _plan_circular_z_depth(
			z_positions=z_positions, tile_size=tile_size, scaledown=sd,
			z_size=Zo, chunk_size=oc, output_begin=b, output_end=b + (oz1 - oz0),
			retain_one_flush=int(flush_workers) > 0,
		)
		acc = {
			p.name: _CircularZBand(
				name=f"acc_{p.name}", channel_count=p.raw_channel_count,
				z_size=Zo, y_size=Yo, x_size=Xo, tmp_dir=tmp_dir,
				prefix=temp_prefix, ring_depth=depth, dtype=product_accumulator_dtype,
			)
			for p in group_products
		}
		weight = _CircularZBand(
			name=f"weight_sd{sd}", channel_count=1, z_size=Zo, y_size=Yo, x_size=Xo,
			tmp_dir=tmp_dir, prefix=temp_prefix, ring_depth=depth, dtype=np.float32,
		)
		product_bytes = sum(p.raw_channel_count for p in group_products) * depth * Yo * Xo * product_accumulator_dtype.itemsize
		weight_bytes = depth * Yo * Xo * np.dtype(np.float32).itemsize
		bytes_total = product_bytes + weight_bytes
		print(
			f"[predict3d] ring sd={sd} zyx=({depth},{Yo},{Xo}) logical_z={Zo} "
			f"products={len(group_products)} product_dtype={product_accumulator_dtype.name} "
			f"weight_dtype=float32 product_backing={product_bytes / 1024**3:.2f}GiB "
			f"weight_backing={weight_bytes / 1024**3:.2f}GiB backing={bytes_total / 1024**3:.2f}GiB",
			flush=True,
		)
		groups[sd] = dict(
			products=group_products, oc=oc, shape=(Zo, Yo, Xo),
			full_shape=full_shape, region=region,
			b=b, depth=depth, acc=acc, weight=weight,
			submitted=b, completed=b, released=0,
			activity={}, support_cache={}, unsupported_origins=set(),
			resume_origins=set(), touched_bytes=0, cleared_bytes=0,
		)

	if input_zarr_path is not None:
		input_dir = str(Path(str(input_zarr_path).rstrip("/")).resolve())
	else:
		store_path = getattr(getattr(zarr_arr, "store", None), "path", None)
		input_dir = str(Path(str(store_path or ".")).resolve())

	def _chunk_supported(sd: int, g: dict[str, Any], origin: ChunkOriginZYX) -> bool:
		"""Whether the output chunk's own selected-input footprint has storage.

		The footprint deliberately excludes the model and Gaussian halos.  This
		keeps absent masked output chunks absent instead of materializing them
		because a neighbouring source chunk happened to feed the same model tile.
		"""
		cache = g["support_cache"]
		if origin not in cache:
			# ``origin`` is in the global output lattice, whereas ``g['shape']``
			# describes only this crop's padded rolling accumulator.  Clip against
			# the full output volume before mapping the footprint to selected-input
			# coordinates.
			shape = g["full_shape"]
			ends = tuple(min(int(shape[i]), int(origin[i]) + int(g["oc"])) for i in range(3))
			bounds = (
				max(0, int(origin[0]) * sd), min(volume_shape[0], int(ends[0]) * sd),
				max(0, int(origin[1]) * sd), min(volume_shape[1], int(ends[1]) * sd),
				max(0, int(origin[2]) * sd), min(volume_shape[2], int(ends[2]) * sd),
			)
			cache[origin] = all(bounds[i] < bounds[i + 1] for i in (0, 2, 4)) and (
				live_cache.region_has_remote_chunks(bounds)
				if live_cache is not None else _input_has_chunks(input_dir, *bounds)
			)
		return bool(cache[origin])

	def _needed_chunks(sd: int, g: dict[str, Any], reg: RegionZYX) -> dict[ChunkOriginZYX, tuple[OutputProductSpec, ...]]:
		shape = g["full_shape"]
		needed: dict[ChunkOriginZYX, tuple[OutputProductSpec, ...]] = {}
		for origin in _iter_chunk_origins_for_region(*reg, g["oc"], shape):
			missing = tuple(
				p for p in g["products"]
				if not output_adapter.product_chunk_complete(p, chunk_origin_zyx=origin)
			)
			if not missing:
				g["resume_origins"].add(origin)
			elif _chunk_supported(sd, g, origin):
				needed[origin] = missing
			else:
				g["unsupported_origins"].add(origin)
		return needed

	def _tile_input_bounds(tz: int, ty: int, tx: int) -> RegionZYX:
		return (
			max(0, tz + z0 - pad), min(volume_shape[0], tz + z0 - pad + tile_size),
			max(0, ty + y0 - pad), min(volume_shape[1], ty + y0 - pad + tile_size),
			max(0, tx + x0 - pad), min(volume_shape[2], tx + x0 - pad + tile_size),
		)

	def _tile_may_need_output(tz: int, ty: int, tx: int) -> bool:
		"""Check durable output resume state without consulting input cache state."""
		for sd, g in groups.items():
			Zo, Yo, Xo = g["shape"]
			ts = tile_size // sd
			clips = [
				_downscaled_tile_clip(value, sd, ts, size)
				for value, size in zip((tz, ty, tx), (Zo, Yo, Xo))
			]
			oz0, oy0, ox0, oz1, oy1, ox1 = g["region"]
			b = g["b"]
			reg = (
				max(oz0, oz0 + clips[0][0] - b), min(oz1, oz0 + clips[0][1] - b),
				max(oy0, oy0 + clips[1][0] - b), min(oy1, oy0 + clips[1][1] - b),
				max(ox0, ox0 + clips[2][0] - b), min(ox1, ox0 + clips[2][1] - b),
			)
			for origin in _iter_chunk_origins_for_region(*reg, g["oc"], g["full_shape"]):
				if any(
					not output_adapter.product_chunk_complete(product, chunk_origin_zyx=origin)
					for product in g["products"]
				):
					return True
		return False

	def _tile_work(tz: int, ty: int, tx: int) -> tuple[str, ...]:
		bounds = _tile_input_bounds(tz, ty, tx)
		has_source = (
			live_cache.region_has_remote_chunks(bounds)
			if live_cache is not None else _input_has_chunks(input_dir, *bounds)
		)
		if not has_source:
			return ()
		needed_names: set[str] = set()
		for sd, g in groups.items():
			Zo, Yo, Xo = g["shape"]
			ts = tile_size // sd
			clips = [_downscaled_tile_clip(v, sd, ts, size) for v, size in zip((tz, ty, tx), (Zo, Yo, Xo))]
			oz0, oy0, ox0, oz1, oy1, ox1 = g["region"]
			b = g["b"]
			reg = (
				max(oz0, oz0 + clips[0][0] - b), min(oz1, oz0 + clips[0][1] - b),
				max(oy0, oy0 + clips[1][0] - b), min(oy1, oy0 + clips[1][1] - b),
				max(ox0, ox0 + clips[2][0] - b), min(ox1, ox0 + clips[2][1] - b),
			)
			for missing in _needed_chunks(sd, g, reg).values():
				needed_names.update(product.name for product in missing)
		return tuple(sorted(needed_names))

	def _accumulate_group(
		sd: int, g: dict[str, Any], product_np: Mapping[str, np.ndarray],
		weight_np: np.ndarray, tz: int, ty: int, tx: int,
	) -> None:
		Zo, Yo, Xo = g["shape"]
		ts = tile_size // sd
		clips = [_downscaled_tile_clip(v, sd, ts, size) for v, size in zip((tz, ty, tx), (Zo, Yo, Xo))]
		(az0, az1, sz0, sz1), (ay0, ay1, sy0, sy1), (ax0, ax1, sx0, sx1) = clips
		if az1 <= az0 or ay1 <= ay0 or ax1 <= ax0:
			return
		oz0, oy0, ox0, oz1, oy1, ox1 = g["region"]
		b, oc = g["b"], g["oc"]
		reg = (
			max(oz0, oz0 + az0 - b), min(oz1, oz0 + az1 - b),
			max(oy0, oy0 + ay0 - b), min(oy1, oy0 + ay1 - b),
			max(ox0, ox0 + ax0 - b), min(ox1, ox0 + ax1 - b),
		)
		incomplete_by_chunk = _needed_chunks(sd, g, reg)
		if not incomplete_by_chunk:
			return
		for origin, missing in incomplete_by_chunk.items():
			cz, cy, cx = origin
			gz0, gy0, gx0 = max(reg[0], cz), max(reg[2], cy), max(reg[4], cx)
			gz1, gy1, gx1 = min(reg[1], cz + oc), min(reg[3], cy + oc), min(reg[5], cx + oc)
			lz0, ly0, lx0 = b + gz0 - oz0, b + gy0 - oy0, b + gx0 - ox0
			lz1, ly1, lx1 = b + gz1 - oz0, b + gy1 - oy0, b + gx1 - ox0
			pz0, py0, px0 = sz0 + lz0 - az0, sy0 + ly0 - ay0, sx0 + lx0 - ax0
			pz1, py1, px1 = pz0 + lz1 - lz0, py0 + ly1 - ly0, px0 + lx1 - lx0
			weight_part = weight_np[pz0:pz1, py0:py1, px0:px1]
			if not np.any(weight_part > 0.0):
				continue
			g["weight"].add(0, lz0, lz1, ly0, ly1, lx0, lx1, weight_part)
			voxels = (lz1 - lz0) * (ly1 - ly0) * (lx1 - lx0)
			g["touched_bytes"] += voxels * 4
			activity = g["activity"].setdefault(
				origin, {"weight_dirty": False, "dirty_products": set()}
			)
			activity["weight_dirty"] = True
			for product in missing:
				arr = product_np[product.name]
				for ch in range(product.raw_channel_count):
					g["acc"][product.name].add(ch, lz0, lz1, ly0, ly1, lx0, lx1, arr[ch, pz0:pz1, py0:py1, px0:px1])
				g["touched_bytes"] += product.raw_channel_count * voxels * g["acc"][product.name].dtype.itemsize
				activity["dirty_products"].add(product.name)

	def _plan_accumulate_tasks(
		sd: int, g: dict[str, Any], result_slot: int, result_spec: _SharedSlotSpec,
		result_shm: shared_memory.SharedMemory, tz: int, ty: int, tx: int,
	) -> list[tuple[_AccumulateProcessTask, dict[str, Any]]]:
		"""Plan disjoint chunk-owner work while retaining coordinator ring metadata."""
		layouts = {layout.name: layout for layout in result_spec.layouts}
		weight_np = _slot_array(result_shm, layouts[f"weight:{sd}"])
		Zo, Yo, Xo = g["shape"]
		ts = tile_size // sd
		clips = [_downscaled_tile_clip(v, sd, ts, size) for v, size in zip((tz, ty, tx), (Zo, Yo, Xo))]
		(az0, az1, sz0, sz1), (ay0, ay1, sy0, sy1), (ax0, ax1, sx0, sx1) = clips
		if az1 <= az0 or ay1 <= ay0 or ax1 <= ax0:
			return []
		oz0, oy0, ox0, oz1, oy1, ox1 = g["region"]
		b, oc = g["b"], g["oc"]
		reg = (
			max(oz0, oz0 + az0 - b), min(oz1, oz0 + az1 - b),
			max(oy0, oy0 + ay0 - b), min(oy1, oy0 + ay1 - b),
			max(ox0, ox0 + ax0 - b), min(ox1, ox0 + ax1 - b),
		)
		planned = []
		for origin, missing in _needed_chunks(sd, g, reg).items():
			cz, cy, cx = origin
			gz0, gy0, gx0 = max(reg[0], cz), max(reg[2], cy), max(reg[4], cx)
			gz1, gy1, gx1 = min(reg[1], cz + oc), min(reg[3], cy + oc), min(reg[5], cx + oc)
			lz0, ly0, lx0 = b + gz0 - oz0, b + gy0 - oy0, b + gx0 - ox0
			lz1, ly1, lx1 = b + gz1 - oz0, b + gy1 - oy0, b + gx1 - ox0
			pz0, py0, px0 = sz0 + lz0 - az0, sy0 + ly0 - ay0, sx0 + lx0 - ax0
			pz1, py1, px1 = pz0 + lz1 - lz0, py0 + ly1 - ly0, px0 + lx1 - lx0
			if not np.any(weight_np[pz0:pz1, py0:py1, px0:px1] > 0.0):
				continue
			# Only the coordinator mutates generation/frontier metadata.
			g["weight"].ensure(lz0, lz1)
			for product in missing:
				g["acc"][product.name].ensure(lz0, lz1)
			products_here = tuple(product.name for product in missing)
			task = _AccumulateProcessTask(
				task_id=-1, event_seq=-1, result_slot=int(result_slot), result_spec=result_spec,
				sd=int(sd), origin=origin,
				destination=(lz0, lz1, ly0, ly1, lx0, lx1),
				source=(pz0, pz1, py0, py1, px0, px1),
				dirty_products=products_here,
				accumulators=tuple((name, g["acc"][name].mmap_descriptor()) for name in products_here),
				weight=g["weight"].mmap_descriptor(),
			)
			voxels = (lz1-lz0) * (ly1-ly0) * (lx1-lx0)
			metadata = {"group": g, "origin": origin, "products": products_here, "voxels": voxels}
			planned.append((task, metadata))
		return planned

	def _flush_target(sd: int, g: dict[str, Any], complete_padded: int) -> int:
		complete = int(complete_padded) // sd
		b, oc = g["b"], g["oc"]
		oz0, _oy0, _ox0, oz1, _oy1, _ox1 = g["region"]
		end = b + oz1 - oz0
		return end if complete >= end else b + (max(0, complete - b) // oc) * oc

	def _prepare_flush_group(sd: int, g: dict[str, Any], flush_to: int) -> _FlushGroupDescriptor:
		b, oc = g["b"], g["oc"]
		oz0, oy0, ox0, oz1, oy1, ox1 = g["region"]
		flush_from = max(int(g["submitted"]), b)
		eligible = sorted(
			origin for origin in g["activity"]
			if flush_from <= b + int(origin[0]) - oz0 < flush_to
		)
		chunks = []
		for origin in eligible:
			activity = g["activity"].pop(origin)
			chunk = _FlushChunkDescriptor(
				origin=origin,
				weight_dirty=bool(activity["weight_dirty"]),
				dirty_products=tuple(sorted(str(name) for name in activity["dirty_products"])),
			)
			gz, gy, gx = origin
			lz0, ly0, lx0 = b + gz - oz0, b + gy - oy0, b + gx - ox0
			lz1 = min(lz0 + oc, b + oz1 - oz0)
			if chunk.weight_dirty:
				g["weight"].validate_frozen(lz0, lz1)
			for name in chunk.dirty_products:
				g["acc"][name].validate_frozen(lz0, lz1)
			chunks.append(chunk)
		g["submitted"] = int(flush_to)
		print(
			f"[predict3d] flush sd={sd} z=[{oz0 + flush_from - b},{oz0 + flush_to - b}) "
			f"dirty_chunks={len(eligible)}",
			flush=True,
		)
		return _FlushGroupDescriptor(
			sd=int(sd), flush_from=int(flush_from), flush_to=int(flush_to),
			chunks=tuple(chunks), submitted_at=time.perf_counter(), b=int(b),
			oc=int(oc), region=g["region"], products=tuple(g["products"]),
			accumulators=tuple(
				(name, band.mmap_descriptor()) for name, band in sorted(g["acc"].items())
			),
			weight=g["weight"].mmap_descriptor(),
		)

	flush_ctx = mp.get_context("spawn")
	flush_task_queue = None
	flush_result_queue = None
	flush_processes: list[Any] = []
	pending_flush: _FlushBatchState | None = None
	flush_batch_id = 0
	flush_work_total = 0.0
	flush_wait_total = 0.0
	flush_chunks_total = 0
	flush_started_at: float | None = None

	def _start_flush_workers() -> None:
		nonlocal flush_task_queue, flush_result_queue, flush_processes
		if int(flush_workers) <= 0 or flush_processes:
			return
		try:
			pickle.dumps((model_adapter, output_adapter))
		except BaseException as exc:
			raise TypeError(
				"process-parallel flush requires spawn-picklable model/output adapters; "
				"use flush_workers=0 for the synchronous baseline"
			) from exc
		window = max(1, 2 * int(flush_workers))
		flush_task_queue = flush_ctx.Queue(maxsize=window)
		flush_result_queue = flush_ctx.Queue(maxsize=window)
		started = []
		try:
			with _single_threaded_native_runtime():
				for index in range(int(flush_workers)):
					process = flush_ctx.Process(
						target=_flush_process_main,
					args=(
						index, model_adapter, output_adapter, flush_task_queue,
						flush_result_queue,
					),
						name=f"predict3d-flush-{index}", daemon=True,
					)
					process.start()
					started.append(process)
		except BaseException:
			for process in started:
				if process.is_alive():
					process.terminate()
				process.join(timeout=5.0)
			flush_task_queue.close()
			flush_result_queue.close()
			raise
		flush_processes = started
		print(
			f"[predict3d] flush processes={len(flush_processes)} ipc_window={window} "
			"native_threads_per_worker=1 blosc_threads_per_worker=1", flush=True,
		)

	def _check_flush_workers() -> None:
		if int(flush_workers) <= 0:
			return
		for index, process in enumerate(flush_processes):
			if not process.is_alive() and process.exitcode is not None:
				raise RuntimeError(
					f"flush worker {index} exited unexpectedly with code {process.exitcode}"
				)

	def _make_flush_tasks(
		plans: tuple[_FlushGroupDescriptor, ...], batch_id: int,
	) -> deque[_FlushProcessTask]:
		tasks: deque[_FlushProcessTask] = deque()
		for plan in plans:
			for chunk in plan.chunks:
				tasks.append(_FlushProcessTask(
					batch_id=batch_id, task_id=len(tasks), sd=plan.sd,
					origin=chunk.origin, dirty_products=chunk.dirty_products,
					weight_dirty=chunk.weight_dirty, b=plan.b, oc=plan.oc,
					region=plan.region, products=plan.products,
					accumulators=plan.accumulators, weight=plan.weight,
				))
		return tasks

	def _handle_flush_message(message: tuple[Any, ...]) -> None:
		if pending_flush is None:
			raise RuntimeError(f"flush result arrived without a pending batch: {message}")
		if int(message[1]) != int(pending_flush.batch_id):
			raise RuntimeError(f"unexpected flush batch result: {message}")
		task_id = int(message[2])
		if task_id not in pending_flush.inflight:
			raise RuntimeError(f"unexpected or duplicate flush task result: {message}")
		pending_flush.inflight.remove(task_id)
		if message[0] == "error":
			raise RuntimeError(
				f"flush worker {message[3]} failed task {task_id}: {message[4]}: {message[5]}"
			)
		if message[0] != "result":
			raise RuntimeError(f"unknown flush worker message: {message}")
		pending_flush.completed_ids.add(task_id)
		pending_flush.written_by_sd[pending_flush.task_sd[task_id]] += int(message[4])
		pending_flush.work_s += float(message[5])

	def _pump_pending_flush(*, block: bool) -> bool:
		if pending_flush is None:
			return True
		if int(flush_workers) <= 0:
			return True
		made_progress = False
		window = max(1, 2 * int(flush_workers))
		while pending_flush.tasks and len(pending_flush.inflight) < window:
			task = pending_flush.tasks[0]
			try:
				flush_task_queue.put_nowait(task)
			except queue.Full:
				break
			pending_flush.tasks.popleft()
			pending_flush.inflight.add(task.task_id)
			made_progress = True
		while True:
			try:
				message = flush_result_queue.get_nowait()
			except queue.Empty:
				break
			made_progress = True
			_handle_flush_message(message)
		if block and not made_progress and not pending_flush.complete:
			try:
				message = flush_result_queue.get(timeout=0.05)
			except queue.Empty:
				_check_flush_workers()
			else:
				_handle_flush_message(message)
		complete = pending_flush.complete
		if complete:
			_finalize_pending_flush()
		return complete

	def _shutdown_flush_workers(*, terminate: bool) -> None:
		nonlocal flush_processes, flush_task_queue, flush_result_queue
		if not flush_processes:
			return
		if terminate:
			for process in flush_processes:
				if process.is_alive():
					process.terminate()
		else:
			for _process in flush_processes:
				flush_task_queue.put(None)
		for process in flush_processes:
			process.join(timeout=10.0)
			if process.is_alive():
				process.terminate()
				process.join(timeout=5.0)
		if terminate:
			flush_task_queue.cancel_join_thread()
			flush_result_queue.cancel_join_thread()
		flush_task_queue.close()
		flush_result_queue.close()
		if not terminate:
			flush_task_queue.join_thread()
			flush_result_queue.join_thread()
		flush_processes = []
		flush_task_queue = None
		flush_result_queue = None

	def _finalize_pending_flush() -> None:
		nonlocal pending_flush, flush_work_total, flush_wait_total, flush_chunks_total
		if pending_flush is None:
			return
		if not pending_flush.complete:
			raise RuntimeError(
				f"cannot finalize incomplete flush batch {pending_flush.batch_id}: "
				f"queued={len(pending_flush.tasks)} inflight={len(pending_flush.inflight)}"
			)
		flush_work_total += float(pending_flush.work_s)
		flush_chunks_total += len(pending_flush.completed_ids)
		completed_batch = pending_flush
		pending_flush = None
		for plan in completed_batch.plans:
			sd, g = plan.sd, groups[plan.sd]
			b, oc = g["b"], g["oc"]
			oz0, oy0, ox0, oz1, oy1, ox1 = g["region"]
			cleared_bytes = 0
			by_name = {p.name: p for p in g["products"]}
			for chunk in plan.chunks:
				origin = chunk.origin
				gz, gy, gx = origin
				lz0, ly0, lx0 = b + gz - oz0, b + gy - oy0, b + gx - ox0
				lz1 = min(lz0 + oc, b + oz1 - oz0)
				ly1 = min(ly0 + oc, b + oy1 - oy0)
				lx1 = min(lx0 + oc, b + ox1 - ox0)
				voxels = (lz1-lz0) * (ly1-ly0) * (lx1-lx0)
				for name in chunk.dirty_products:
					g["acc"][name].clear(lz0, lz1, ly0, ly1, lx0, lx1)
					cleared_bytes += by_name[name].raw_channel_count * voxels * g["acc"][name].dtype.itemsize
				if chunk.weight_dirty:
					g["weight"].clear(lz0, lz1, ly0, ly1, lx0, lx1)
					cleared_bytes += voxels * 4
			g["cleared_bytes"] += cleared_bytes
			for acc in g["acc"].values():
				acc.discard_before(plan.flush_to)
			g["weight"].discard_before(plan.flush_to)
			g["released"] = g["completed"] = int(plan.flush_to)
			if progress is not None:
				if isinstance(output_scaledown_base, Mapping):
					base_sd = min(int(output_scaledown_base[p.name]) for p in g["products"])
				elif output_scaledown_base is None:
					base_sd = sd
				else:
					base_sd = int(output_scaledown_base)
				progress[f"finalized_base_z_sd{sd}"] = (oz0 + plan.flush_to - b) * base_sd
				progress["finalized_base_z"] = min(
					int(progress.get(f"finalized_base_z_sd{group_sd}", progress.get("finalized_base_z", 0)))
					for group_sd in groups
				)
			print(
				f"[predict3d] flush complete sd={sd} z={oz0 + plan.flush_to - b} "
				f"dirty_chunks={len(plan.chunks)} "
				f"products_written={completed_batch.written_by_sd[sd]} "
				f"unsupported={len(g['unsupported_origins'])} resume={len(g['resume_origins'])} "
				f"touched={g['touched_bytes'] / 1024**2:.2f}MiB "
				f"cleared={g['cleared_bytes'] / 1024**2:.2f}MiB "
				f"elapsed={time.perf_counter()-plan.submitted_at:.2f}s",
				flush=True,
			)

	def _complete_pending_flush() -> None:
		nonlocal pending_flush, flush_wait_total
		if pending_flush is None:
			return
		wait_started = time.perf_counter()
		if int(flush_workers) > 0:
			while pending_flush is not None:
				_pump_pending_flush(block=True)
		else:
			cache: dict[str, np.memmap] = {}
			try:
				while pending_flush.tasks:
					task = pending_flush.tasks.popleft()
					_batch, task_id, written, elapsed = _execute_flush_process_task(
						task, model_adapter, output_adapter, cache,
					)
					pending_flush.completed_ids.add(task_id)
					pending_flush.written_by_sd[task.sd] += int(written)
					pending_flush.work_s += float(elapsed)
			finally:
				for arr in cache.values():
					mmap_obj = getattr(arr, "_mmap", None)
					if mmap_obj is not None:
						mmap_obj.close()
			_finalize_pending_flush()
		flush_wait_total += time.perf_counter() - wait_started

	def _advance_flushes(complete_padded: int) -> None:
		nonlocal pending_flush, flush_batch_id, flush_started_at
		targets = {
			sd: _flush_target(sd, g, complete_padded)
			for sd, g in groups.items()
		}
		if not any(targets[sd] > int(groups[sd]["submitted"]) for sd in groups):
			return
		_complete_pending_flush()
		plans = tuple(
			_prepare_flush_group(sd, groups[sd], targets[sd])
			for sd in sorted(groups)
			if targets[sd] > int(groups[sd]["submitted"])
		)
		flush_batch_id += 1
		if flush_started_at is None:
			flush_started_at = time.perf_counter()
		tasks = _make_flush_tasks(plans, flush_batch_id)
		pending_flush = _FlushBatchState(
			batch_id=flush_batch_id, plans=plans, tasks=tasks,
			inflight=set(), completed_ids=set(), work_s=0.0,
			written_by_sd={sd: 0 for sd in groups},
			task_sd={task.task_id: task.sd for task in tasks},
		)
		if int(flush_workers) <= 0:
			_complete_pending_flush()
		else:
			_pump_pending_flush(block=False)

	total = len(z_positions) * len(y_positions) * len(x_positions)
	done = processed = skipped = 0
	tile_time = 0.0
	t0 = time.time()
	if progress is not None:
		progress.update(tiles_total=total, tiles_done=0, tiles_processed=0, tiles_skipped=0, tile_time_sum=0.0, tiles_remaining_est=total)
		if live_cache is not None:
			progress["live_cache"] = live_cache.snapshot()
	print(_predict3d_progress_line(progress) if progress is not None else f"[predict3d] 0/{total} tiles", flush=True)
	progress_emitter = _Predict3dProgressEmitter()
	try:
		def _record_commit(*, was_processed: bool, elapsed: float, row_end: bool, next_tz: int) -> None:
			nonlocal done, processed, skipped, tile_time
			done += 1
			if was_processed:
				processed += 1
				# Parallel ETA is based on observed committed throughput, not the
				# sum of overlapping per-GPU durations (which would overestimate by
				# approximately the worker count).
				tile_time = time.time() - t0 if parallel else tile_time + float(elapsed)
			else:
				skipped += 1
			if progress is not None:
				progress.update(tiles_done=done, tiles_processed=processed, tiles_skipped=skipped, tile_time_sum=tile_time, tiles_remaining_est=total-done)
				if live_cache is not None:
					progress["live_cache"] = live_cache.snapshot()
			status = _predict3d_progress_line(progress) if progress is not None else f"[predict3d] {done}/{total} tiles"
			progress_emitter.emit(status, done=done, total=total)
			if row_end:
				progress_emitter.finish_line()
				_advance_flushes(next_tz)
				if live_cache is not None:
					safe_input_z = min(
						volume_shape[0], max(0, int(next_tz) + int(z0) - int(pad)),
					)
					live_cache.advance_safe_boundary(safe_input_z)
					if progress is not None:
						progress["live_cache"] = live_cache.snapshot()

		if not parallel:
			_start_flush_workers()
			assert w_full is not None and w_by_scale is not None
			serial_reader = None
			serial_pending: deque[dict[str, Any]] = deque()
			serial_source = iter(_iter_canonical_tile_events(z_positions, y_positions, x_positions, Zp))
			serial_source_done = False
			serial_read_sum = 0.0
			serial_read_bytes = 0
			serial_ready_highwater = 0
			if input_reader == "tensorstore":
				serial_reader = _TensorStoreTileReader(
					str(input_zarr_path), cache_bytes=int(input_cache_bytes),
					file_io_threads=int(input_io_threads), data_copy_threads=int(input_copy_threads),
				)
				print(
					f"[predict3d] input reader=tensorstore prefetch_window={int(prefetch_tiles_per_gpu)} "
					f"cache={int(input_cache_bytes)/1024**3:.2f}GiB io_threads={int(input_io_threads)} "
					f"copy_threads={int(input_copy_threads)}", flush=True,
				)

			def _next_serial_event() -> dict[str, Any] | None:
				nonlocal serial_source_done, serial_ready_highwater
				if live_cache is not None:
					while not serial_source_done and len(serial_pending) < live_cache.lookahead_tiles:
						try:
							tz, ty, tx, row_end, next_tz = next(serial_source)
						except StopIteration:
							serial_source_done = True
							break
						event = dict(
							coord=(tz, ty, tx), row_end=row_end, next_tz=next_tz,
						)
						if not _tile_may_need_output(tz, ty, tx):
							event.update(status="done_skip", needed=())
						else:
							event.update(
								status="fetching",
								cache_task=live_cache.request_region(_tile_input_bounds(tz, ty, tx)),
							)
						serial_pending.append(event)

					def finish_materialization(event: dict[str, Any], *, block: bool) -> bool:
						if event["status"] != "fetching":
							return True
						task = event["cache_task"]
						if not block and not task.done():
							return False
						task.result()
						tz, ty, tx = event["coord"]
						event["needed"] = _tile_work(tz, ty, tx)
						event["status"] = "awaiting_read" if event["needed"] else "done_skip"
						return True

					for pending_event in serial_pending:
						finish_materialization(pending_event, block=False)
					active_reads = sum(
						1 for pending_event in serial_pending if pending_event.get("status") == "reading"
					)
					for pending_event in serial_pending:
						if active_reads >= int(prefetch_tiles_per_gpu):
							break
						if pending_event.get("status") != "awaiting_read":
							continue
						if serial_reader is None:
							pending_event["status"] = "ready"
							continue
						tz, ty, tx = pending_event["coord"]
						pending_event["read_task"] = serial_reader.submit(
							volume_shape=volume_shape, crop_offset=(z0, y0, x0), coord=(tz, ty, tx),
							tile_size=tile_size, border=pad,
						)
						pending_event["status"] = "reading"
						active_reads += 1
					if not serial_pending:
						return None
					head = serial_pending[0]
					if head["status"] == "fetching":
						finish_materialization(head, block=True)
					if head["status"] == "awaiting_read":
						if serial_reader is None:
							head["status"] = "ready"
						else:
							tz, ty, tx = head["coord"]
							head["read_task"] = serial_reader.submit(
								volume_shape=volume_shape, crop_offset=(z0, y0, x0), coord=(tz, ty, tx),
								tile_size=tile_size, border=pad,
							)
							head["status"] = "reading"
					serial_ready_highwater = max(
						serial_ready_highwater,
						sum(
							1 for pending_event in serial_pending
							if pending_event.get("status") == "reading"
							and serial_reader is not None
							and serial_reader.done(pending_event["read_task"])
						),
					)
					return serial_pending.popleft()
				if serial_reader is None:
					try:
						tz, ty, tx, row_end, next_tz = next(serial_source)
					except StopIteration:
						return None
					needed = _tile_work(tz, ty, tx)
					tile_np = None if not needed else _read_tile_zarr(
						zarr_arr, volume_shape, (z0, y0, x0), tz, ty, tx, tile_size, pad,
					)
					return dict(coord=(tz, ty, tx), row_end=row_end, next_tz=next_tz, needed=needed, tile=tile_np)
				while not serial_source_done and len(serial_pending) < int(prefetch_tiles_per_gpu):
					try:
						tz, ty, tx, row_end, next_tz = next(serial_source)
					except StopIteration:
						serial_source_done = True
						break
					needed = _tile_work(tz, ty, tx)
					event = dict(coord=(tz, ty, tx), row_end=row_end, next_tz=next_tz, needed=needed)
					if needed:
						event["read_task"] = serial_reader.submit(
							volume_shape=volume_shape, crop_offset=(z0, y0, x0), coord=(tz, ty, tx),
							tile_size=tile_size, border=pad,
						)
					serial_pending.append(event)
				serial_ready_highwater = max(
					serial_ready_highwater,
					sum(1 for event in serial_pending if event.get("read_task") is not None and serial_reader.done(event["read_task"])),
				)
				return serial_pending.popleft() if serial_pending else None

			try:
				while True:
					event = _next_serial_event()
					if event is None:
						break
					tz, ty, tx = event["coord"]
					row_end, next_tz, needed_names = event["row_end"], event["next_tz"], event["needed"]
					_pump_pending_flush(block=False)
					if not needed_names:
						_record_commit(was_processed=False, elapsed=0.0, row_end=row_end, next_tz=next_tz)
						continue
					tile_t0 = time.time()
					if serial_reader is None:
						if live_cache is not None:
							tile_np = _read_tile_zarr(
								zarr_arr, volume_shape, (z0, y0, x0), tz, ty, tx, tile_size, pad,
							)
						else:
							tile_np = event["tile"]
					else:
						tile_np, read_elapsed = serial_reader.result(event["read_task"])
						serial_read_sum += float(read_elapsed)
						serial_read_bytes += int(tile_np.nbytes)
					tile = _input_tile_to_device(tile_np, device)
					preprocess = getattr(model_adapter, "preprocess_tile", None)
					if preprocess is not None:
						tile = preprocess(tile, torch.ones_like(tile, dtype=torch.bool))
					with torch.inference_mode():
						raw_output = model_adapter.run_tile_inference(model, tile, device=device)
					diagnostic = raw_output.get("output") if isinstance(raw_output, dict) else raw_output
					if isinstance(diagnostic, torch.Tensor):
						nan_count = int(torch.isnan(diagnostic).sum().item())
						if nan_count or processed == 0:
							print(
								f"[predict3d] tile pos=({tz},{ty},{tx}) input=({float(tile.min()):.4f},{float(tile.max()):.4f}) "
								f"raw=({float(diagnostic.min()):.4f},{float(diagnostic.max()):.4f}) "
								f"nan={nan_count}/{diagnostic.numel()} dtype={diagnostic.dtype}", flush=True,
							)
					tensors = model_adapter.product_tensors_from_output(raw_output)
					prepared_products = {}
					for name in needed_names:
						sd = scales[name]
						weighted = tensors[name][0] * w_full
						if sd > 1:
							weighted = _pyrdown3d(weighted, factor=sd)
						prepared_products[name] = weighted.detach().cpu().numpy()
					for sd, group in groups.items():
						_accumulate_group(sd, group, prepared_products, w_by_scale[sd], tz, ty, tx)
					_record_commit(was_processed=True, elapsed=time.time() - tile_t0, row_end=row_end, next_tz=next_tz)
			finally:
				if serial_reader is not None:
					serial_reader.cancel(
						event["read_task"] for event in serial_pending if event.get("read_task") is not None
					)
				for pending_event in serial_pending:
					cache_task = pending_event.get("cache_task")
					if cache_task is not None:
						cache_task.cancel()
			if serial_reader is not None:
				print(
					f"[predict3d] input stats backend=tensorstore reads={processed} "
					f"bytes={serial_read_bytes} read_sum={serial_read_sum:.2f}s "
					f"ready_highwater={serial_ready_highwater}", flush=True,
				)
		else:
			_slot_count = len(resolved_devices) * int(slots_per_gpu)
			_prefetch_window = len(resolved_devices) * int(prefetch_tiles_per_gpu)
			_materialization_window = (
				live_cache.lookahead_tiles if live_cache is not None else _prefetch_window
			)
			_input_layouts, input_bytes = _packed_layouts((("input", (tile_size, tile_size, tile_size), np.dtype(zarr_arr.dtype)),))
			_result_entries = []
			for product in products:
				sd = scales[product.name]
				ts = tile_size // sd
				_result_entries.append((f"product:{product.name}", (product.raw_channel_count, ts, ts, ts), np.float32))
			for sd in sorted(set(scales.values())):
				ts = tile_size // sd
				_result_entries.append((f"weight:{sd}", (ts, ts, ts), np.float32))
			_result_layouts, result_bytes = _packed_layouts(_result_entries)
			input_shms = _create_shared_slots(_slot_count, input_bytes)
			try:
				result_shms = _create_shared_slots(_slot_count, result_bytes)
			except BaseException:
				for shm in input_shms:
					shm.close()
					shm.unlink()
				raise
			input_specs = tuple(_SharedSlotSpec(shm.name, input_bytes, _input_layouts) for shm in input_shms)
			result_specs = tuple(_SharedSlotSpec(shm.name, result_bytes, _result_layouts) for shm in result_shms)
			print(
				f"[predict3d] multi-device devices={','.join(str(v) for v in resolved_devices)} slots={_slot_count} "
				f"prefetch_window={_prefetch_window} materialization_window={_materialization_window} "
				f"input_shared={_slot_count * input_bytes / 1024**3:.2f}GiB "
				f"result_shared={_slot_count * result_bytes / 1024**3:.2f}GiB",
				flush=True,
			)
			ctx = mp.get_context("spawn")
			result_queue = ctx.Queue(maxsize=max(4, _slot_count * 3))
			work_queues = [ctx.Queue(maxsize=int(slots_per_gpu)) for _ in resolved_devices]
			accum_result_queue = None
			accum_queues: list[Any] = []
			accum_processes: list[Any] = []
			worker_state = None
			if model_state is not None:
				worker_state = {}
				for key, value in model_state.items():
					shared_value = value.detach().cpu().contiguous()
					shared_value.share_memory_()
					worker_state[key] = shared_value
			workers = [
				ctx.Process(
					target=_multi_gpu_worker_main,
					args=(
						index, str(worker_device), model_adapter, worker_state, tile_size, r,
						scales, input_specs, result_specs, work_queues[index], result_queue,
						bool(profile_pipeline),
					),
					name=f"predict3d-{worker_device}", daemon=True,
				)
				for index, worker_device in enumerate(resolved_devices)
			]
			started_workers = []
			try:
				with _single_threaded_native_runtime():
					for worker in workers:
						worker.start()
						started_workers.append(worker)
			except BaseException:
				for worker in started_workers:
					if worker.is_alive():
						worker.terminate()
					worker.join(timeout=5.0)
				for work_queue in work_queues:
					work_queue.close()
					work_queue.join_thread()
				result_queue.close()
				result_queue.join_thread()
				for shm in input_shms + result_shms:
					shm.close()
					shm.unlink()
				raise
			_start_flush_workers()
			if int(accumulator_workers) > 0:
				accum_result_queue = ctx.Queue(maxsize=max(4, _slot_count * 4))
				accum_queues = [ctx.Queue(maxsize=2) for _ in range(int(accumulator_workers))]
				try:
					with _single_threaded_native_runtime():
						for index, task_queue in enumerate(accum_queues):
							process = ctx.Process(
								target=_accumulate_process_main,
								args=(index, task_queue, accum_result_queue),
								name=f"predict3d-accumulate-{index}", daemon=True,
							)
							process.start()
							accum_processes.append(process)
				except BaseException:
					for process in accum_processes:
						if process.is_alive():
							process.terminate()
						process.join(timeout=5.0)
					raise
				module = _accumulator_native_module()
				backend = str(module.backend()) if module else "numpy"
				print(
					f"[predict3d] accumulator processes={len(accum_processes)} "
					f"queue_depth=2 native_threads_per_worker=1 backend={backend}", flush=True,
				)
			reader_count = int(prefetch_workers) if int(prefetch_workers) > 0 else min(_slot_count, max(1, len(resolved_devices) * 2))
			executor = None
			tensorstore_reader = None
			try:
				if input_reader == "tensorstore":
					tensorstore_reader = _TensorStoreTileReader(
						str(input_zarr_path), cache_bytes=int(input_cache_bytes),
						file_io_threads=int(input_io_threads), data_copy_threads=int(input_copy_threads),
					)
				else:
					executor = ThreadPoolExecutor(max_workers=reader_count, thread_name_prefix="predict3d-read")
			except BaseException:
				for task_queue in accum_queues:
					try:
						task_queue.put_nowait(None)
					except queue.Full:
						pass
				for process in accum_processes:
					process.join(timeout=5.0)
					if process.is_alive():
						process.terminate()
						process.join(timeout=5.0)
				for work_queue in work_queues:
					try:
						work_queue.put_nowait(None)
					except queue.Full:
						pass
				for worker in workers:
					worker.join(timeout=5.0)
					if worker.is_alive():
						worker.terminate()
						worker.join(timeout=5.0)
				for shm in input_shms + result_shms:
					shm.close()
					shm.unlink()
				raise
			print(
				f"[predict3d] input reader={input_reader} prefetch_window={_prefetch_window} "
				f"cache={int(input_cache_bytes)/1024**3:.2f}GiB io_threads={int(input_io_threads)} "
				f"copy_threads={int(input_copy_threads)}"
				+ (f" fallback_threads={reader_count}" if input_reader == "python-zarr" else ""), flush=True,
			)
			free_inputs = list(range(_slot_count - 1, -1, -1))
			free_results = list(range(_slot_count - 1, -1, -1))
			slot_ledger = _SlotLedger(_slot_count)
			gpu_outstanding: list[dict[int, dict[str, Any]]] = [dict() for _ in workers]
			accum_outstanding: list[dict[int, int]] = [dict() for _ in accum_processes]
			events: dict[int, dict[str, Any]] = {}
			futures: dict[Any, int] = {}
			band_index = -1
			active_band: _ZBandState | None = None
			band_iter: Iterator[tuple[int, int]] = iter(())
			band_source_done = True
			source_done = False
			cache_descriptor_source = (
				iter(_iter_canonical_tile_events(z_positions, y_positions, x_positions, Zp))
				if live_cache is not None else None
			)
			cache_descriptor_done = live_cache is None
			cache_backlog: deque[tuple[int, dict[str, Any]]] = deque()
			active_event_window = max(_slot_count, _prefetch_window)
			next_sequence = next_commit = next_accum_dispatch = 0
			next_worker = 0
			gpu_counts = [0 for _ in workers]
			gpu_time_sums = [0.0 for _ in workers]
			commit_time = 0.0
			accum_task_id = 0
			accum_pending: dict[int, dict[str, Any]] = {}
			accum_work_sum = 0.0
			accum_queue_wait = 0.0
			accum_started_at: float | None = None
			accum_backends: set[str] = set()
			read_time = 0.0
			read_bytes = 0
			read_submitted = 0
			read_completed = 0
			read_live_highwater = 0
			read_ready_highwater = 0
			input_copy_time = 0.0
			input_starve_time = 0.0
			starve_started: float | None = None
			pipeline_profile = None
			if profile_pipeline:
				pipeline_profile = {
					"read_service_sum_s": 0.0, "read_service_max_s": 0.0,
					"read_collect_lag_sum_s": 0.0, "read_collect_lag_max_s": 0.0,
					"read_first_submitted_at": None, "read_last_completed_at": None,
					"ready_to_assign_sum_s": 0.0, "ready_to_assign_max_s": 0.0,
					"result_receive_lag_sum_s": 0.0, "result_receive_lag_max_s": 0.0,
					"worker_stage_sums": [dict() for _ in workers],
					"worker_stage_max": [dict() for _ in workers],
				}

			def _read_coord(coord: TileOriginZYX) -> tuple[np.ndarray, float]:
				started = time.perf_counter()
				arr = _read_tile_zarr(zarr_arr, volume_shape, (z0, y0, x0), *coord, tile_size, pad)
				return arr, time.perf_counter() - started

			def _submit_read(seq: int, event: dict[str, Any]) -> None:
				"""Submit one materialized tile to the unchanged bounded reader stage."""
				nonlocal read_submitted
				tz, ty, tx = event["coord"]
				event["status"] = "reading"
				if tensorstore_reader is not None:
					task = tensorstore_reader.submit(
						volume_shape=volume_shape, crop_offset=(z0, y0, x0),
						coord=(tz, ty, tx), tile_size=tile_size, border=pad,
						profile=bool(profile_pipeline),
					)
					event["read_task"] = task
					futures[id(task)] = int(seq)
				else:
					event["read_submitted_at"] = time.perf_counter()
					future = executor.submit(_read_coord, (tz, ty, tx))
					if profile_pipeline:
						future.add_done_callback(
							lambda _future, current=event: current.__setitem__(
								"read_completed_at", time.perf_counter(),
							)
						)
					event["read_task"] = future
					futures[future] = int(seq)
				read_submitted += 1

			def _handle_gpu_message(message: tuple[Any, ...]) -> None:
				if not isinstance(message, tuple) or not message:
					raise RuntimeError(f"GPU result queue received malformed message: {message!r}")
				kind = message[0]
				expected_lengths = {"input_released": 4, "result": (7, 9), "error": 6}
				expected = expected_lengths.get(kind)
				if expected is None or (
					isinstance(expected, int) and len(message) != expected
				) or (
					isinstance(expected, tuple) and len(message) not in expected
				):
					raise RuntimeError(
						f"GPU result queue received malformed {kind!r} message: {message!r}"
					)
				source_worker = int(message[3] if kind in ("input_released", "error") else message[5])
				if not 0 <= source_worker < len(gpu_outstanding):
					raise RuntimeError(f"GPU result names invalid worker {source_worker}: {message}")
				seq = int(message[1])
				owner = gpu_outstanding[source_worker].get(seq)
				if owner is None:
					raise RuntimeError(
						f"GPU worker {source_worker} returned foreign or duplicate sequence {seq}; "
						f"outstanding={sorted(gpu_outstanding[source_worker])}"
					)
				if kind == "input_released":
					slot = int(message[2])
					if owner["input_released"] or slot != int(owner["input_slot"]):
						raise RuntimeError(
							f"GPU worker {source_worker} returned invalid input slot {slot} for "
							f"sequence {seq}: owner={owner}"
						)
					event = events.get(seq)
					if event is None:
						raise RuntimeError(f"input slot {slot} released for unknown event {seq}")
					slot_ledger.release("input", slot, band=int(event["band"]), sequence=seq)
					free_inputs.append(slot)
					owner["input_released"] = True
				elif kind == "result":
					_, seq, _coord, _slot, _names, worker_index, elapsed, *profile_tail = message
					seq = int(seq)
					if not owner["input_released"] or int(_slot) != int(owner["result_slot"]):
						raise RuntimeError(
							f"GPU worker {source_worker} returned invalid result for sequence {seq}: "
							f"slot={_slot} owner={owner}"
						)
					gpu_outstanding[source_worker].pop(seq)
					events[int(seq)]["status"] = "done_result"
					slot_ledger.transition(
						"result", int(_slot), band=int(events[int(seq)]["band"]),
						sequence=int(seq), stage="gpu-result",
					)
					events[int(seq)]["gpu_elapsed"] = float(elapsed)
					gpu_counts[int(worker_index)] += 1
					gpu_time_sums[int(worker_index)] += float(elapsed)
					if pipeline_profile is not None:
						stage_values, worker_finished_at = profile_tail
						received_at = time.perf_counter()
						lag = max(0.0, received_at - float(worker_finished_at))
						pipeline_profile["result_receive_lag_sum_s"] += lag
						pipeline_profile["result_receive_lag_max_s"] = max(
							pipeline_profile["result_receive_lag_max_s"], lag,
						)
						sums = pipeline_profile["worker_stage_sums"][int(worker_index)]
						maxima = pipeline_profile["worker_stage_max"][int(worker_index)]
						for name, value in stage_values.items():
							sums[name] = sums.get(name, 0.0) + float(value)
							maxima[name] = max(maxima.get(name, 0.0), float(value))
				elif kind == "error":
					_, seq, coord, worker_index, kind, detail = message
					gpu_outstanding[source_worker].pop(int(seq))
					raise RuntimeError(
						f"multi-GPU worker {worker_index} failed at tile {coord} "
						f"seq={seq}: {kind}: {detail}"
					)
				else:
					raise RuntimeError(f"unknown multi-GPU worker message: {message}")

			def _handle_accum_message(message: tuple[Any, ...]) -> None:
				nonlocal accum_work_sum
				if not isinstance(message, tuple) or len(message) != 6 or message[0] not in ("result", "error"):
					raise RuntimeError(f"accumulator result queue received malformed message: {message!r}")
				task_id = int(message[1])
				source_worker = int(message[3])
				if not 0 <= source_worker < len(accum_outstanding):
					raise RuntimeError(f"accumulator result names invalid worker {source_worker}: {message}")
				event_seq = accum_outstanding[source_worker].pop(task_id, None)
				if event_seq is None or int(event_seq) != int(message[2]):
					raise RuntimeError(
						f"accumulator worker {source_worker} returned foreign or duplicate task "
						f"{task_id}: event={message[2]} outstanding={accum_outstanding[source_worker]}"
					)
				metadata = accum_pending.pop(task_id, None)
				if metadata is None:
					raise RuntimeError(f"unexpected or duplicate accumulator result: {message}")
				if message[0] == "error":
					raise RuntimeError(
						f"accumulator worker {message[3]} failed task {task_id}: {message[4]}: {message[5]}"
					)
				if message[0] != "result":
					raise RuntimeError(f"unknown accumulator worker message: {message}")
				accum_backends.add(str(message[4]))
				accum_work_sum += float(message[5])
				g = metadata["group"]
				origin = metadata["origin"]
				activity = g["activity"].setdefault(origin, {"weight_dirty": False, "dirty_products": set()})
				activity["weight_dirty"] = True
				activity["dirty_products"].update(metadata["products"])
				voxels = int(metadata["voxels"])
				g["touched_bytes"] += voxels * 4
				for name in metadata["products"]:
					g["touched_bytes"] += g["acc"][name].channel_count * voxels * g["acc"][name].dtype.itemsize
				event = events[int(message[2])]
				event["accum_remaining"] -= 1
				if event["accum_remaining"] == 0:
					result_slot = int(event["result_slot"])
					slot_ledger.release(
						"result", result_slot, band=int(event["band"]), sequence=int(message[2]),
					)
					free_results.append(result_slot)
					event["status"] = "done_accum"

			def _pump_gpu_results(*, timeout: float = 0.0) -> bool:
				progressed = False
				while True:
					try:
						message = (
							result_queue.get(timeout=float(timeout))
							if not progressed and float(timeout) > 0.0
							else result_queue.get_nowait()
						)
					except queue.Empty:
						break
					_handle_gpu_message(message)
					progressed = True
				return progressed

			def _pump_accumulator_results(*, timeout: float = 0.0) -> bool:
				if accum_result_queue is None:
					return False
				progressed = False
				while True:
					try:
						message = (
							accum_result_queue.get(timeout=float(timeout))
							if not progressed and float(timeout) > 0.0
							else accum_result_queue.get_nowait()
						)
					except queue.Empty:
						break
					_handle_accum_message(message)
					progressed = True
				return progressed

			def _prepare_accumulator_event(seq: int, event: dict[str, Any]) -> None:
				nonlocal accum_started_at
				result_slot = int(event["result_slot"])
				plans = []
				for sd, group in groups.items():
					plans.extend(_plan_accumulate_tasks(
						sd, group, result_slot, result_specs[result_slot], result_shms[result_slot], *event["coord"],
					))
				if not plans:
					slot_ledger.release(
						"result", result_slot, band=int(event["band"]), sequence=int(seq),
					)
					free_results.append(result_slot)
					event["status"] = "done_accum"
					return
				event["accum_remaining"] = len(plans)
				event["status"] = "accumulating"
				event["accum_to_submit"] = deque(plans)
				slot_ledger.transition(
					"result", result_slot, band=int(event["band"]), sequence=int(seq),
					stage="accumulating",
				)
				if accum_started_at is None:
					accum_started_at = time.perf_counter()

			def _pump_accumulator_submissions(seq: int, event: dict[str, Any]) -> bool:
				nonlocal accum_task_id, accum_queue_wait
				pending = event.get("accum_to_submit")
				if not pending:
					return False
				progressed = False
				while pending:
					task_template, metadata = pending[0]
					task = replace(task_template, task_id=accum_task_id, event_seq=int(seq))
					owner = _stable_accumulator_owner(task.sd, task.origin, len(accum_queues))
					if len(accum_outstanding[owner]) >= 2:
						break
					wait_started = time.perf_counter()
					try:
						accum_queues[owner].put_nowait(task)
					except queue.Full:
						break
					accum_queue_wait += time.perf_counter() - wait_started
					pending.popleft()
					accum_pending[accum_task_id] = metadata
					accum_outstanding[owner][accum_task_id] = int(seq)
					accum_task_id += 1
					progressed = True
				if not pending:
					event.pop("accum_to_submit", None)
				return progressed

			def _validate_worker_ownership() -> None:
				gpu_sequences: dict[int, int] = {}
				for worker_index, outstanding in enumerate(gpu_outstanding):
					for seq, owner in outstanding.items():
						if seq in gpu_sequences:
							raise RuntimeError(
								f"GPU sequence {seq} is owned by workers "
								f"{gpu_sequences[seq]} and {worker_index}"
							)
						gpu_sequences[seq] = worker_index
						event = events.get(seq)
						if event is None or event.get("status") != "assigned":
							raise RuntimeError(
								f"GPU worker {worker_index} owns sequence {seq} without an "
								f"assigned event: {event}"
							)
						if int(event.get("worker", -1)) != worker_index:
							raise RuntimeError(
								f"GPU sequence {seq} event claims worker {event.get('worker')} "
								f"but worker {worker_index} owns it"
							)
						if (
							int(event.get("input_slot", -1)) != int(owner["input_slot"])
							or int(event.get("result_slot", -1)) != int(owner["result_slot"])
						):
							raise RuntimeError(
								f"GPU sequence {seq} slot ownership mismatch: event={event} owner={owner}"
							)
				assigned_sequences = {
					int(seq) for seq, event in events.items() if event.get("status") == "assigned"
				}
				if assigned_sequences != set(gpu_sequences):
					raise RuntimeError(
						f"GPU ownership mismatch: assigned={sorted(assigned_sequences)} "
						f"outstanding={sorted(gpu_sequences)}"
					)

				accumulator_tasks: dict[int, int] = {}
				for worker_index, outstanding in enumerate(accum_outstanding):
					for task_id, event_seq in outstanding.items():
						if task_id in accumulator_tasks:
							raise RuntimeError(
								f"accumulator task {task_id} is owned by multiple workers"
							)
						accumulator_tasks[task_id] = int(event_seq)
				pending_task_ids = {int(task_id) for task_id in accum_pending}
				if pending_task_ids != set(accumulator_tasks):
					raise RuntimeError(
						f"accumulator ownership mismatch: pending={sorted(pending_task_ids)} "
						f"outstanding={accumulator_tasks}"
					)

			try:
				while not source_done or events:
					_pump_pending_flush(block=False)
					made_progress = _pump_accumulator_results()
					if band_source_done and not events:
						if active_band is not None:
							active_band.validate_complete()
							if futures or accum_pending or any(gpu_outstanding) or any(accum_outstanding):
								raise RuntimeError(
									f"Z band z={active_band.z} completed with outstanding "
									f"reads={len(futures)} gpu={gpu_outstanding} "
									f"accumulator_tasks={accum_outstanding}"
								)
							slot_ledger.validate_all_free()
						band_index += 1
						if band_index >= len(z_positions):
							source_done = True
							break
						tz = int(z_positions[band_index])
						next_tz = int(z_positions[band_index + 1]) if band_index + 1 < len(z_positions) else int(Zp)
						active_band = _ZBandState(
							z=tz, next_z=next_tz, first_sequence=next_commit,
							expected=len(y_positions) * len(x_positions),
						)
						band_iter = iter((int(ty), int(tx)) for ty in y_positions for tx in x_positions)
						band_source_done = False
						made_progress = True
					assert active_band is not None
					if live_cache is not None:
						while not cache_descriptor_done and len(cache_backlog) < _materialization_window:
							try:
								tz, ty, tx, row_end, next_tz = next(cache_descriptor_source)
							except StopIteration:
								cache_descriptor_done = True
								break
							event = dict(
								coord=(tz, ty, tx), row_end=row_end, next_tz=next_tz, band=int(tz),
							)
							if not _tile_may_need_output(tz, ty, tx):
								event.update(status="done_skip", needed=())
							else:
								event.update(
									status="fetching",
									cache_task=live_cache.request_region(_tile_input_bounds(tz, ty, tx)),
								)
							cache_backlog.append((next_sequence, event))
							next_sequence += 1
							made_progress = True
						while len(events) < active_event_window and cache_backlog:
							seq, event = cache_backlog[0]
							if int(event["band"]) != active_band.z:
								break
							cache_backlog.popleft()
							events[seq] = event
							active_band.generated += 1
							made_progress = True
						band_source_done = active_band.generated == active_band.expected
					else:
						while not band_source_done and len(events) < active_event_window:
							try:
								ty, tx = next(band_iter)
							except StopIteration:
								band_source_done = True
								break
							tz = active_band.z
							row_end = active_band.generated + 1 == active_band.expected
							event = dict(
								coord=(tz, ty, tx), row_end=row_end, next_tz=active_band.next_z,
								band=active_band.z,
							)
							needed_names = _tile_work(tz, ty, tx)
							event["needed"] = needed_names
							event["status"] = "reading" if needed_names else "done_skip"
							events[next_sequence] = event
							if needed_names:
								# Preserve the pre-live-fetch path: discovering a normal local
								# tile submits its read atomically in the same scheduler pass.
								_submit_read(next_sequence, event)
							next_sequence += 1
							active_band.generated += 1
							made_progress = True
					for event in events.values():
						if event.get("status") != "fetching":
							continue
						cache_task = event["cache_task"]
						if not cache_task.done():
							continue
						cache_task.result()
						tz, ty, tx = event["coord"]
						event["needed"] = _tile_work(tz, ty, tx)
						event["status"] = "awaiting_read" if event["needed"] else "done_skip"
						made_progress = True
					for seq, event in events.items():
						if len(futures) >= _prefetch_window:
							break
						if event.get("status") != "awaiting_read":
							continue
						_submit_read(seq, event)
						made_progress = True
					read_live_highwater = max(read_live_highwater, len(futures))
					for future_key, seq in list(futures.items()):
						event = events[seq]
						task = event["read_task"]
						is_done = tensorstore_reader.done(task) if tensorstore_reader is not None else task.done()
						if not is_done:
							continue
						del futures[future_key]
						if tensorstore_reader is not None:
							arr, elapsed = tensorstore_reader.result(task)
							submitted_at = task.submitted_at
							completed_at = task.completed_at
						else:
							arr, elapsed = task.result()
							submitted_at = event.get("read_submitted_at")
							completed_at = event.get("read_completed_at")
						collected_at = time.perf_counter()
						if pipeline_profile is not None and submitted_at is not None:
							completed_at = collected_at if completed_at is None else float(completed_at)
							service = max(0.0, completed_at - float(submitted_at))
							lag = max(0.0, collected_at - completed_at)
							pipeline_profile["read_service_sum_s"] += service
							pipeline_profile["read_service_max_s"] = max(pipeline_profile["read_service_max_s"], service)
							pipeline_profile["read_collect_lag_sum_s"] += lag
							pipeline_profile["read_collect_lag_max_s"] = max(pipeline_profile["read_collect_lag_max_s"], lag)
							first = pipeline_profile["read_first_submitted_at"]
							last = pipeline_profile["read_last_completed_at"]
							pipeline_profile["read_first_submitted_at"] = float(submitted_at) if first is None else min(first, float(submitted_at))
							pipeline_profile["read_last_completed_at"] = completed_at if last is None else max(last, completed_at)
							event["read_completed_at"] = completed_at
						read_time += float(elapsed)
						read_bytes += int(arr.nbytes)
						read_completed += 1
						event["tile"] = arr
						event["status"] = "ready"
						made_progress = True
					read_ready_highwater = max(
						read_ready_highwater, sum(1 for event in events.values() if event["status"] == "ready")
					)
					for seq, event in events.items():
						if event["status"] != "ready":
							continue
						if not _shared_slot_pair_available(
							events=events, frontier_sequence=next_accum_dispatch,
							sequence=seq, free_input_count=len(free_inputs),
							free_result_count=len(free_results),
						):
							continue
						worker_index = None
						for attempt in range(len(workers)):
							candidate = (next_worker + attempt) % len(workers)
							if len(gpu_outstanding[candidate]) < int(slots_per_gpu):
								worker_index = candidate
								break
						if worker_index is None:
							break
						input_slot = free_inputs.pop()
						result_slot = free_results.pop()
						slot_ledger.acquire(
							"input", input_slot, band=int(event["band"]), sequence=int(seq), stage="gpu-provisional",
						)
						slot_ledger.acquire(
							"result", result_slot, band=int(event["band"]), sequence=int(seq), stage="gpu-provisional",
						)
						if pipeline_profile is not None and event.get("read_completed_at") is not None:
							ready_wait = max(0.0, time.perf_counter() - float(event["read_completed_at"]))
							pipeline_profile["ready_to_assign_sum_s"] += ready_wait
							pipeline_profile["ready_to_assign_max_s"] = max(pipeline_profile["ready_to_assign_max_s"], ready_wait)
						copy_started = time.perf_counter()
						try:
							np.copyto(
								_slot_array(input_shms[input_slot], _input_layouts[0]),
								event["tile"], casting="no",
							)
							work_queues[worker_index].put_nowait((
								seq, event["coord"], input_slot, result_slot, event["needed"],
							))
						except queue.Full:
							slot_ledger.release(
								"input", input_slot, band=int(event["band"]), sequence=int(seq),
							)
							slot_ledger.release(
								"result", result_slot, band=int(event["band"]), sequence=int(seq),
							)
							free_inputs.append(input_slot)
							free_results.append(result_slot)
							continue
						input_copy_time += time.perf_counter() - copy_started
						slot_ledger.transition(
							"input", input_slot, band=int(event["band"]), sequence=int(seq), stage="gpu",
						)
						slot_ledger.transition(
							"result", result_slot, band=int(event["band"]), sequence=int(seq), stage="gpu",
						)
						gpu_outstanding[worker_index][int(seq)] = {
							"input_slot": int(input_slot), "result_slot": int(result_slot),
							"input_released": False,
						}
						event.pop("tile", None)
						event["input_slot"] = input_slot
						event["result_slot"] = result_slot
						event["status"] = "assigned"
						event["worker"] = worker_index
						next_worker = (worker_index + 1) % len(workers)
						made_progress = True
					made_progress = _pump_gpu_results() or made_progress
					_pump_accumulator_results()
					# Prepare and submit accumulator work canonically. Queue pressure
					# yields to the other stages instead of monopolizing the coordinator.
					while next_accum_dispatch in events:
						event = events[next_accum_dispatch]
						if event["status"] == "done_skip":
							next_accum_dispatch += 1
							made_progress = True
							continue
						if event["status"] not in ("done_result", "accumulating"):
							break
						if int(accumulator_workers) > 0:
							if event["status"] == "done_result":
								_prepare_accumulator_event(next_accum_dispatch, event)
							if event["status"] == "accumulating":
								submitted = _pump_accumulator_submissions(next_accum_dispatch, event)
								made_progress = submitted or made_progress
								if event.get("accum_to_submit"):
									break
						else:
							layouts = {layout.name: layout for layout in result_specs[event["result_slot"]].layouts}
							result_shm = result_shms[event["result_slot"]]
							prepared = {name: _slot_array(result_shm, layouts[f"product:{name}"]) for name in event["needed"]}
							for sd, group in groups.items():
								weight = _slot_array(result_shm, layouts[f"weight:{sd}"])
								_accumulate_group(sd, group, prepared, weight, *event["coord"])
							result_slot = int(event["result_slot"])
							slot_ledger.release(
								"result", result_slot, band=int(event["band"]),
								sequence=int(next_accum_dispatch),
							)
							free_results.append(result_slot)
							event["status"] = "done_accum"
						next_accum_dispatch += 1
						made_progress = True
					commit_started = time.perf_counter()
					while next_commit in events and events[next_commit]["status"] in ("done_skip", "done_accum"):
						event = events.pop(next_commit)
						if event["status"] == "done_skip":
							_record_commit(was_processed=False, elapsed=0.0, row_end=event["row_end"], next_tz=event["next_tz"])
						else:
							_record_commit(was_processed=True, elapsed=event["gpu_elapsed"], row_end=event["row_end"], next_tz=event["next_tz"])
						if active_band is None or int(event["band"]) != active_band.z:
							raise RuntimeError(
								f"event {next_commit} committed outside active Z band: "
								f"event_band={event['band']} active={None if active_band is None else active_band.z}"
							)
						active_band.committed += 1
						next_commit += 1
						made_progress = True
					commit_time += time.perf_counter() - commit_started
					can_accept_input = bool(free_inputs and free_results) and any(
						len(outstanding) < int(slots_per_gpu) for outstanding in gpu_outstanding
					)
					has_ready_input = any(event["status"] == "ready" for event in events.values())
					has_pending_input = any(
						event["status"] in ("fetching", "awaiting_read", "reading")
						for event in events.values()
					)
					if can_accept_input and not has_ready_input and has_pending_input:
						if starve_started is None:
							starve_started = time.perf_counter()
					elif starve_started is not None:
						input_starve_time += time.perf_counter() - starve_started
						starve_started = None
					_validate_worker_ownership()
					for index, worker in enumerate(workers):
						if not worker.is_alive() and worker.exitcode is not None:
							raise RuntimeError(
								f"multi-GPU worker {index} exited unexpectedly with code "
								f"{worker.exitcode}; outstanding={gpu_outstanding[index]}"
							)
					for index, process in enumerate(accum_processes):
						if not process.is_alive() and process.exitcode is not None:
							raise RuntimeError(
								f"accumulator worker {index} exited unexpectedly with code "
								f"{process.exitcode}; outstanding={accum_outstanding[index]}"
							)
					if not made_progress:
						statuses: dict[str, int] = {}
						for event in events.values():
							status = str(event.get("status", "unknown"))
							statuses[status] = statuses.get(status, 0) + 1
						capacity_deadlock = _frontier_capacity_deadlock(
							events=events, frontier_sequence=next_accum_dispatch,
							free_inputs=free_inputs, free_results=free_results,
							slot_owners=slot_ledger.snapshot(),
						)
						if capacity_deadlock is not None:
							raise RuntimeError(capacity_deadlock)
						try:
							wait_stage = _coordinator_wait_stage(
								statuses=statuses, reads=len(futures),
								accumulator_tasks=len(accum_pending),
								slot_owners=slot_ledger.snapshot(),
							)
						except RuntimeError as exc:
							flush_state = None if pending_flush is None else {
								"batch": pending_flush.batch_id,
								"queued": len(pending_flush.tasks),
								"inflight": len(pending_flush.inflight),
								"completed": len(pending_flush.completed_ids),
							}
							worker_state = [
								(index, worker.pid, worker.is_alive(), worker.exitcode)
								for index, worker in enumerate(workers)
							]
							raise RuntimeError(
								f"{exc}; band={active_band} next_commit={next_commit} "
								f"next_accum={next_accum_dispatch} flush={flush_state} "
								f"workers={worker_state}"
							) from exc
						if wait_stage == "gpu":
							_pump_gpu_results(timeout=0.05)
						elif wait_stage == "accumulator":
							_pump_accumulator_results(timeout=0.05)
						elif wait_stage == "reader":
							time.sleep(0.01)
				if starve_started is not None:
					input_starve_time += time.perf_counter() - starve_started
				accum_wall = 0.0 if accum_started_at is None else time.perf_counter() - accum_started_at
				accum_rate = 0.0 if accum_wall <= 0.0 else accum_task_id / accum_wall
				print(
					f"[predict3d] accumulator stats workers={len(accum_processes)} tasks={accum_task_id} "
					f"work_sum={accum_work_sum:.2f}s queue_wait={accum_queue_wait:.2f}s "
					f"wall={accum_wall:.2f}s rate={accum_rate:.2f}tasks/s "
					f"backends={','.join(sorted(accum_backends)) or ('native-sync' if not accum_processes else 'none')}",
					flush=True,
				)
				print(
					f"[predict3d] multi-device stats input_backend={input_reader} "
					f"reads={read_completed}/{read_submitted} read_bytes={read_bytes} read_sum={read_time:.1f}s "
					f"ready_highwater={read_ready_highwater} live_highwater={read_live_highwater} "
					f"copy_sum={input_copy_time:.1f}s input_starve={input_starve_time:.1f}s "
					f"commit_sum={commit_time:.1f}s "
					+ " ".join(
						f"gpu{index}_tiles={count} gpu{index}_sum={gpu_time_sums[index]:.1f}s"
						for index, count in enumerate(gpu_counts)
					), flush=True,
				)
				if pipeline_profile is not None:
					first = pipeline_profile["read_first_submitted_at"]
					last = pipeline_profile["read_last_completed_at"]
					read_span = 0.0 if first is None or last is None else max(0.0, last - first)
					concurrency = pipeline_profile["read_service_sum_s"] / read_span if read_span > 0 else 0.0
					throughput_gib_s = read_bytes / float(1 << 30) / read_span if read_span > 0 else 0.0
					count = max(1, read_completed)
					print(
						f"[predict3d:profile] loader reads={read_completed} bytes={read_bytes} "
						f"service_sum={pipeline_profile['read_service_sum_s']:.3f}s "
						f"active_span={read_span:.3f}s throughput={throughput_gib_s:.3f}GiB/s "
						f"effective_request_concurrency={concurrency:.2f} "
						f"service_mean={pipeline_profile['read_service_sum_s']/count:.4f}s "
						f"service_max={pipeline_profile['read_service_max_s']:.4f}s "
						f"collect_lag_mean={pipeline_profile['read_collect_lag_sum_s']/count:.4f}s "
						f"collect_lag_max={pipeline_profile['read_collect_lag_max_s']:.4f}s "
						f"ready_wait_mean={pipeline_profile['ready_to_assign_sum_s']/count:.4f}s "
						f"ready_wait_max={pipeline_profile['ready_to_assign_max_s']:.4f}s "
						f"live_highwater={read_live_highwater} ready_highwater={read_ready_highwater}",
						flush=True,
					)
					for worker_index, sums in enumerate(pipeline_profile["worker_stage_sums"]):
						worker_count = max(1, gpu_counts[worker_index])
						stages = " ".join(
							f"{name}={value:.3f}s(mean={value/worker_count:.4f},max={pipeline_profile['worker_stage_max'][worker_index][name]:.4f})"
							for name, value in sorted(sums.items())
						)
						print(f"[predict3d:profile] worker={worker_index} tiles={gpu_counts[worker_index]} {stages}", flush=True)
					print(
						f"[predict3d:profile] coordinator copy_sum={input_copy_time:.3f}s "
						f"input_idle={input_starve_time:.3f}s commit_sum={commit_time:.3f}s "
						f"result_receive_lag_sum={pipeline_profile['result_receive_lag_sum_s']:.3f}s "
						f"result_receive_lag_max={pipeline_profile['result_receive_lag_max_s']:.4f}s",
						flush=True,
					)
			finally:
				for event in events.values():
					cache_task = event.get("cache_task")
					if cache_task is not None:
						cache_task.cancel()
				for _seq, event in cache_backlog:
					cache_task = event.get("cache_task")
					if cache_task is not None:
						cache_task.cancel()
				if tensorstore_reader is not None:
					tensorstore_reader.cancel(
						event["read_task"] for event in events.values()
						if event.get("status") == "reading" and event.get("read_task") is not None
					)
				if executor is not None:
					executor.shutdown(wait=True, cancel_futures=True)
				for task_queue in accum_queues:
					try:
						task_queue.put_nowait(None)
					except queue.Full:
						pass
				for process in accum_processes:
					process.join(timeout=10.0)
					if process.is_alive():
						process.terminate()
						process.join(timeout=5.0)
				for task_queue in accum_queues:
					task_queue.close()
					task_queue.join_thread()
				if accum_result_queue is not None:
					accum_result_queue.close()
					accum_result_queue.join_thread()
				for work_queue in work_queues:
					try:
						work_queue.put_nowait(None)
					except queue.Full:
						pass
				for worker in workers:
					worker.join(timeout=5.0)
					if worker.is_alive():
						worker.terminate()
						worker.join(timeout=5.0)
				for work_queue in work_queues:
					work_queue.close()
					work_queue.join_thread()
				result_queue.close()
				result_queue.join_thread()
				for shm in input_shms + result_shms:
					shm.close()
					shm.unlink()
		_complete_pending_flush()
		_shutdown_flush_workers(terminate=False)
		progress_emitter.finish_line()
		flush_wall = 0.0 if flush_started_at is None else time.perf_counter() - flush_started_at
		flush_rate = 0.0 if flush_wall <= 0.0 else flush_chunks_total / flush_wall
		print(
			f"[predict3d] flush stats workers={int(flush_workers)} chunks={flush_chunks_total} "
			f"work_sum={flush_work_total:.2f}s wait={flush_wait_total:.2f}s "
			f"wall={flush_wall:.2f}s rate={flush_rate:.2f}chunks/s",
			flush=True,
		)
		print(f"[predict3d] inference done in {time.time()-t0:.1f}s ({processed} processed, {skipped} skipped)", flush=True)
	finally:
		active_error = sys.exc_info()[1]
		flush_cleanup_error: BaseException | None = None
		try:
			_shutdown_flush_workers(terminate=active_error is not None or pending_flush is not None)
		except BaseException as exc:
			if flush_cleanup_error is None:
				flush_cleanup_error = exc
		for g in groups.values():
			for acc in g["acc"].values():
				acc.cleanup()
			g["weight"].cleanup()
		if flush_cleanup_error is not None:
			if active_error is not None:
				active_error.add_note(f"secondary asynchronous flush shutdown error: {flush_cleanup_error!r}")
			else:
				raise flush_cleanup_error
