from __future__ import annotations

import atexit
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path
import queue
import shutil
import sys
import tempfile
import threading
import time
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence, runtime_checkable
import uuid

import numpy as np
import torch
import torch.nn.functional as F
import zarr

try:
	from omezarr_pyramid import (
		build_normal_omezarr_pyramid,
		build_scalar_omezarr_pyramid,
		set_pyramid_metadata,
	)
except ImportError:
	from lasagna.omezarr_pyramid import (
		build_normal_omezarr_pyramid,
		build_scalar_omezarr_pyramid,
		set_pyramid_metadata,
	)


ChunkOriginZYX = tuple[int, int, int]
RegionZYX = tuple[int, int, int, int, int, int]
TileOriginZYX = tuple[int, int, int]
ProductTileOutput = Mapping[str, np.ndarray]


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
	accumulators: tuple[tuple[str, Any], ...]
	weight: Any


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
	return (
		f"[predict3d] [{bar}] {done}/{total} tiles "
		f"({100.0 * done / total:.1f}%) "
		f"eta {eta_text}"
		f"{avg}"
		f"{_predict3d_overall_eta(progress)}"
		f"{_predict3d_finalized_status(progress)}"
	)


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


def _create_omezarr(
	path: str,
	base_shape_zyx: tuple[int, int, int],
	first_level: int,
	n_levels: int,
	chunk: int,
	channel_name: str,
) -> zarr.Group:
	"""Create an OME-Zarr group with pyramid level arrays."""
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
				chunk_key_encoding={"name": "v2", "separator": "/"},
			)
		except (AttributeError, TypeError):
			try:
				g.create_dataset(
					str(lv), shape=sh,
					chunks=chunks,
					dtype=np.uint8, fill_value=0, overwrite=True,
					dimension_separator="/",
				)
			except TypeError:
				g.create_dataset(
					str(lv), shape=sh,
					chunks=chunks,
					dtype=np.uint8, fill_value=0, overwrite=True,
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
	return _create_omezarr(path, base_shape_zyx, first_level, n_levels, chunk, channel_name)


def create_product_omezarr_groups(
	*,
	products: Sequence[OutputProductSpec],
	base_shape_zyx: tuple[int, int, int],
	n_levels: int,
	ome_chunk: int,
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
			pending_to = flush_to
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
				* np.dtype(np.float32).itemsize
			)
			os.ftruncate(fd, logical_bytes)
		except Exception:
			os.close(fd)
			_remove_path_quiet(path)
			raise
		os.close(fd)
		mm = np.memmap(
			path,
			dtype=np.float32,
			mode="r+",
			shape=(self.ring_depth, self.y_size, self.x_size),
		)
		mm._lasagna_tmp_path = path
		atexit.register(lambda p=path: os.path.exists(p) and os.unlink(p))
		return mm

	@property
	def end_z(self) -> int:
		return self.z_size

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
			self._arrays[int(ch)][physical_z:physical_z + length, y0:y1, x0:x1] += data[
				data_offset:data_offset + length
			]
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


def _read_tile_zarr(
	zarr_arr,
	volume_shape: tuple[int, int, int],
	crop_offset: tuple[int, int, int],
	tz: int, ty: int, tx: int,
	tile_size: int | None,
	border: int,
) -> np.ndarray:
	"""Read a single tile from zarr, using reflect-padding only at volume boundaries."""
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

	if rz1 <= rz0 or ry1 <= ry0 or rx1 <= rx0:
		return np.zeros((tile_size, tile_size, tile_size), dtype=np.uint8)

	chunk = np.asarray(zarr_arr[rz0:rz1, ry0:ry1, rx0:rx1])

	pad_before = (rz0 - src_z0, ry0 - src_y0, rx0 - src_x0)
	pad_after = (src_z1 - rz1, src_y1 - ry1, src_x1 - rx1)
	needs_pad = any(p > 0 for p in pad_before + pad_after)
	if needs_pad:
		chunk = np.pad(
			chunk,
			[(pad_before[0], pad_after[0]),
			 (pad_before[1], pad_after[1]),
			 (pad_before[2], pad_after[2])],
			mode="reflect",
		)
	return chunk


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
	for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
		os.environ[name] = "1"
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
				input_layout = input_specs[input_slot].layouts[0]
				tile_np = _slot_array(inputs[input_slot], input_layout)
				if tile_np.dtype == np.uint16:
					prepared = (tile_np // 257).astype(np.uint8).astype(np.float32) / 255.0
				else:
					prepared = tile_np.astype(np.float32) / 255.0
				tile = torch.from_numpy(prepared).unsqueeze(0).unsqueeze(0).to(device)
				# The synchronous copy above makes the input slot reusable before model work.
				result_queue.put(("input_released", seq, input_slot, worker_index))
				preprocess = getattr(model_adapter, "preprocess_tile", None)
				if preprocess is not None:
					tile = preprocess(tile, torch.ones_like(tile, dtype=torch.bool))
				with torch.inference_mode():
					raw_output = model_adapter.run_tile_inference(model, tile, device=device)
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
				if device.type == "cuda":
					torch.cuda.synchronize(device)
				result_queue.put(("result", seq, coord, result_slot, tuple(needed_names), worker_index, time.perf_counter() - started))
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
		)
		acc = {
			p.name: _CircularZBand(
				name=f"acc_{p.name}", channel_count=p.raw_channel_count,
				z_size=Zo, y_size=Yo, x_size=Xo, tmp_dir=tmp_dir,
				prefix=temp_prefix, ring_depth=depth,
			)
			for p in group_products
		}
		weight = _CircularZBand(
			name=f"weight_sd{sd}", channel_count=1, z_size=Zo, y_size=Yo, x_size=Xo,
			tmp_dir=tmp_dir, prefix=temp_prefix, ring_depth=depth,
		)
		bytes_total = (1 + sum(p.raw_channel_count for p in group_products)) * depth * Yo * Xo * 4
		print(
			f"[predict3d] ring sd={sd} zyx=({depth},{Yo},{Xo}) logical_z={Zo} "
			f"products={len(group_products)} backing={bytes_total / 1024**3:.2f}GiB",
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
			cache[origin] = all(bounds[i] < bounds[i + 1] for i in (0, 2, 4)) and _input_has_chunks(input_dir, *bounds)
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

	def _tile_work(tz: int, ty: int, tx: int) -> tuple[str, ...]:
		bounds = (
			max(0, tz + z0 - pad), min(volume_shape[0], tz + z0 - pad + tile_size),
			max(0, ty + y0 - pad), min(volume_shape[1], ty + y0 - pad + tile_size),
			max(0, tx + x0 - pad), min(volume_shape[2], tx + x0 - pad + tile_size),
		)
		if not _input_has_chunks(input_dir, *bounds):
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
				g["touched_bytes"] += product.raw_channel_count * voxels * 4
				activity["dirty_products"].add(product.name)

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
			chunks.append(_FlushChunkDescriptor(
				origin=origin,
				weight_dirty=bool(activity["weight_dirty"]),
				dirty_products=tuple(sorted(str(name) for name in activity["dirty_products"])),
			))
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
			accumulators=tuple(sorted(g["acc"].items())), weight=g["weight"],
		)

	def _run_flush_batch(plans: tuple[_FlushGroupDescriptor, ...]) -> tuple[dict[str, Any], ...]:
		limits = None
		try:
			from threadpoolctl import threadpool_limits
			limits = threadpool_limits(limits=1)
			limits.__enter__()
		except ImportError:
			limits = None
		started = time.perf_counter()
		results = []
		try:
			for plan in plans:
				sd, b, oc = plan.sd, plan.b, plan.oc
				oz0, oy0, ox0, oz1, oy1, ox1 = plan.region
				by_name = {p.name: p for p in plan.products}
				accumulators = dict(plan.accumulators)
				written = 0
				for index, chunk in enumerate(plan.chunks, 1):
					origin = chunk.origin
					gz, gy, gx = origin
					lz0, ly0, lx0 = b + gz - oz0, b + gy - oy0, b + gx - ox0
					lz1 = min(lz0 + oc, b + oz1 - oz0)
					ly1 = min(ly0 + oc, b + oy1 - oy0)
					lx1 = min(lx0 + oc, b + ox1 - ox0)
					if chunk.weight_dirty:
						denom = plan.weight.read(0, lz0, lz1, ly0, ly1, lx0, lx1)
						if np.any(denom > 0.0):
							np.maximum(denom, 1.0e-7, out=denom)
							for name in chunk.dirty_products:
								product = by_name[name]
								raw = np.stack([
									accumulators[name].read(ch, lz0, lz1, ly0, ly1, lx0, lx1)
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
									product, chunk_origin_zyx=origin,
									data={c.name: persisted[c.name] for c in product.channels},
								)
								written += 1
								del raw, persisted
						del denom
					if index == len(plan.chunks) or index % 64 == 0:
						print(
							f"[predict3d] flush sd={sd} chunks={index}/{len(plan.chunks)} "
							f"written={written}", flush=True,
						)
				results.append({"plan": plan, "written": written})
			work_s = time.perf_counter() - started
			return tuple({**result, "batch_work_s": work_s} for result in results)
		finally:
			if limits is not None:
				limits.__exit__(None, None, None)

	flush_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="predict3d-flush")
	pending_flush: Future | None = None
	flush_work_total = 0.0
	flush_wait_total = 0.0

	def _complete_pending_flush() -> None:
		nonlocal pending_flush, flush_work_total, flush_wait_total
		if pending_flush is None:
			return
		wait_started = time.perf_counter()
		try:
			results = pending_flush.result()
		except KeyboardInterrupt:
			raise
		except BaseException:
			pending_flush = None
			raise
		flush_wait_total += time.perf_counter() - wait_started
		pending_flush = None
		if results:
			flush_work_total += float(results[0]["batch_work_s"])
		for result in results:
			plan = result["plan"]
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
					cleared_bytes += by_name[name].raw_channel_count * voxels * 4
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
				f"dirty_chunks={len(plan.chunks)} products_written={result['written']} "
				f"unsupported={len(g['unsupported_origins'])} resume={len(g['resume_origins'])} "
				f"touched={g['touched_bytes'] / 1024**2:.2f}MiB "
				f"cleared={g['cleared_bytes'] / 1024**2:.2f}MiB "
				f"elapsed={time.perf_counter()-plan.submitted_at:.2f}s",
				flush=True,
			)

	def _advance_flushes(complete_padded: int) -> None:
		nonlocal pending_flush
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
		pending_flush = flush_executor.submit(_run_flush_batch, plans)

	total = len(z_positions) * len(y_positions) * len(x_positions)
	done = processed = skipped = 0
	tile_time = 0.0
	t0 = time.time()
	if progress is not None:
		progress.update(tiles_total=total, tiles_done=0, tiles_processed=0, tiles_skipped=0, tile_time_sum=0.0, tiles_remaining_est=total)
	print(_predict3d_progress_line(progress) if progress is not None else f"[predict3d] 0/{total} tiles", flush=True)
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
			status = _predict3d_progress_line(progress) if progress is not None else f"[predict3d] {done}/{total} tiles"
			if was_processed and sys.stdout.isatty():
				print(f"\r{status}  ", end="", flush=True)
			elif done == total or (row_end and done % max(1, len(y_positions) * len(x_positions)) == 0):
				print(status, flush=True)
			if row_end:
				_advance_flushes(next_tz)

		if not parallel:
			assert w_full is not None and w_by_scale is not None
			for tz, ty, tx, row_end, next_tz in _iter_canonical_tile_events(z_positions, y_positions, x_positions, Zp):
				needed_names = _tile_work(tz, ty, tx)
				if not needed_names:
					_record_commit(was_processed=False, elapsed=0.0, row_end=row_end, next_tz=next_tz)
					continue
				tile_t0 = time.time()
				tile_np = _read_tile_zarr(zarr_arr, volume_shape, (z0, y0, x0), tz, ty, tx, tile_size, pad)
				if tile_np.dtype == np.uint16:
					tile_np = (tile_np // 257).astype(np.uint8)
				tile = torch.from_numpy(tile_np.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
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
		else:
			_slot_count = len(resolved_devices) * int(slots_per_gpu)
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
				f"input_shared={_slot_count * input_bytes / 1024**3:.2f}GiB result_shared={_slot_count * result_bytes / 1024**3:.2f}GiB",
				flush=True,
			)
			ctx = mp.get_context("spawn")
			result_queue = ctx.Queue(maxsize=max(4, _slot_count * 3))
			work_queues = [ctx.Queue(maxsize=int(slots_per_gpu)) for _ in resolved_devices]
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
					args=(index, str(worker_device), model_adapter, worker_state, tile_size, r, scales, input_specs, result_specs, work_queues[index], result_queue),
					name=f"predict3d-{worker_device}", daemon=True,
				)
				for index, worker_device in enumerate(resolved_devices)
			]
			started_workers = []
			try:
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
				result_queue.close()
				for shm in input_shms + result_shms:
					shm.close()
					shm.unlink()
				raise
			reader_count = int(prefetch_workers) if int(prefetch_workers) > 0 else min(_slot_count, max(1, len(resolved_devices) * 2))
			executor = ThreadPoolExecutor(max_workers=reader_count, thread_name_prefix="predict3d-read")
			free_inputs = list(range(_slot_count - 1, -1, -1))
			free_results = list(range(_slot_count - 1, -1, -1))
			events: dict[int, dict[str, Any]] = {}
			futures: dict[Future, int] = {}
			event_iter = iter(_iter_canonical_tile_events(z_positions, y_positions, x_positions, Zp))
			source_done = False
			next_sequence = next_commit = 0
			next_worker = 0
			gpu_counts = [0 for _ in workers]
			gpu_time_sums = [0.0 for _ in workers]
			commit_time = 0.0
			read_time = 0.0

			def _read_slot(slot_index: int, coord: TileOriginZYX) -> float:
				started = time.perf_counter()
				arr = _read_tile_zarr(zarr_arr, volume_shape, (z0, y0, x0), *coord, tile_size, pad)
				np.copyto(_slot_array(input_shms[slot_index], _input_layouts[0]), arr, casting="no")
				return time.perf_counter() - started

			try:
				while not source_done or events:
					made_progress = False
					while not source_done and len(events) < _slot_count:
						try:
							tz, ty, tx, row_end, next_tz = next(event_iter)
						except StopIteration:
							source_done = True
							break
						needed_names = _tile_work(tz, ty, tx)
						event = dict(coord=(tz, ty, tx), row_end=row_end, next_tz=next_tz, needed=needed_names)
						if not needed_names:
							event["status"] = "done_skip"
						elif free_results:
							event["result_slot"] = free_results.pop()
							event["status"] = "waiting_read"
						else:
							break
						events[next_sequence] = event
						next_sequence += 1
						made_progress = True
					for seq, event in events.items():
						if event["status"] != "waiting_read" or not free_inputs:
							continue
						input_slot = free_inputs.pop()
						event["input_slot"] = input_slot
						future = executor.submit(_read_slot, input_slot, event["coord"])
						event["status"] = "reading"
						futures[future] = seq
						made_progress = True
					for future, seq in list(futures.items()):
						if not future.done():
							continue
						del futures[future]
						read_time += float(future.result())
						events[seq]["status"] = "ready"
						made_progress = True
					for seq, event in events.items():
						if event["status"] != "ready":
							continue
						placed = False
						for attempt in range(len(workers)):
							worker_index = (next_worker + attempt) % len(workers)
							try:
								work_queues[worker_index].put_nowait((seq, event["coord"], event["input_slot"], event["result_slot"], event["needed"]))
							except queue.Full:
								continue
							event["status"] = "assigned"
							event["worker"] = worker_index
							next_worker = (worker_index + 1) % len(workers)
							placed = True
							made_progress = True
							break
						if not placed:
							break
					while True:
						try:
							message = result_queue.get_nowait()
						except queue.Empty:
							break
						made_progress = True
						if message[0] == "input_released":
							free_inputs.append(int(message[2]))
						elif message[0] == "result":
							_, seq, _coord, _slot, _names, worker_index, elapsed = message
							events[int(seq)]["status"] = "done_result"
							events[int(seq)]["gpu_elapsed"] = float(elapsed)
							gpu_counts[int(worker_index)] += 1
							gpu_time_sums[int(worker_index)] += float(elapsed)
						elif message[0] == "error":
							_, seq, coord, worker_index, kind, detail = message
							raise RuntimeError(f"multi-GPU worker {worker_index} failed at tile {coord} seq={seq}: {kind}: {detail}")
					commit_started = time.perf_counter()
					while next_commit in events and events[next_commit]["status"] in ("done_skip", "done_result"):
						event = events.pop(next_commit)
						if event["status"] == "done_skip":
							_record_commit(was_processed=False, elapsed=0.0, row_end=event["row_end"], next_tz=event["next_tz"])
						else:
							layouts = {layout.name: layout for layout in result_specs[event["result_slot"]].layouts}
							result_shm = result_shms[event["result_slot"]]
							prepared = {name: _slot_array(result_shm, layouts[f"product:{name}"]) for name in event["needed"]}
							for sd, group in groups.items():
								weight = _slot_array(result_shm, layouts[f"weight:{sd}"])
								_accumulate_group(sd, group, prepared, weight, *event["coord"])
							free_results.append(event["result_slot"])
							_record_commit(was_processed=True, elapsed=event["gpu_elapsed"], row_end=event["row_end"], next_tz=event["next_tz"])
						next_commit += 1
						made_progress = True
					commit_time += time.perf_counter() - commit_started
					for index, worker in enumerate(workers):
						if not worker.is_alive() and worker.exitcode is not None:
							raise RuntimeError(f"multi-GPU worker {index} exited unexpectedly with code {worker.exitcode}")
					if not made_progress:
						try:
							message = result_queue.get(timeout=0.05)
							# Requeue one descriptor for the normal drain path.
							result_queue.put(message)
						except queue.Empty:
							pass
				print(
					f"[predict3d] multi-device stats read_sum={read_time:.1f}s commit_sum={commit_time:.1f}s "
					+ " ".join(
						f"gpu{index}_tiles={count} gpu{index}_sum={gpu_time_sums[index]:.1f}s"
						for index, count in enumerate(gpu_counts)
					), flush=True,
				)
			finally:
				executor.shutdown(wait=True, cancel_futures=True)
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
				result_queue.close()
				for shm in input_shms + result_shms:
					shm.close()
					shm.unlink()
		_complete_pending_flush()
		flush_executor.shutdown(wait=True, cancel_futures=False)
		if progress is not None:
			print(_predict3d_progress_line(progress), flush=True)
		print(
			f"[predict3d] flush stats work={flush_work_total:.2f}s wait={flush_wait_total:.2f}s",
			flush=True,
		)
		print(f"[predict3d] inference done in {time.time()-t0:.1f}s ({processed} processed, {skipped} skipped)", flush=True)
	finally:
		active_error = sys.exc_info()[1]
		flush_cleanup_error: BaseException | None = None
		while pending_flush is not None:
			try:
				_complete_pending_flush()
			except KeyboardInterrupt as exc:
				# A Python thread cannot be cancelled while it reads the mmap. Keep
				# waiting before cleanup even when shutdown is interrupted again.
				if flush_cleanup_error is None:
					flush_cleanup_error = exc
			except BaseException as exc:
				pending_flush = None
				if flush_cleanup_error is None:
					flush_cleanup_error = exc
		try:
			flush_executor.shutdown(wait=True, cancel_futures=False)
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
