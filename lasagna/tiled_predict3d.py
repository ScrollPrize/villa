from __future__ import annotations

import atexit
import ctypes
import ctypes.util
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable
import uuid

import numpy as np
import torch
import torch.nn.functional as F
import zarr

try:
	from omezarr_pyramid import build_scalar_omezarr_pyramid, set_pyramid_metadata
except ImportError:
	from lasagna.omezarr_pyramid import build_scalar_omezarr_pyramid, set_pyramid_metadata


ChunkOriginZYX = tuple[int, int, int]
RegionZYX = tuple[int, int, int, int, int, int]
TileOriginZYX = tuple[int, int, int]
ProductTileOutput = Mapping[str, np.ndarray]

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

	def __post_init__(self) -> None:
		name = str(self.name).strip()
		if not name:
			raise ValueError("output product name must be non-empty")
		level = int(self.level)
		scaledown = int(self.scaledown)
		chunk_size = int(self.chunk_size)
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

	@property
	def channel_count(self) -> int:
		return len(self.channels)

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

	def accumulate_tile_output(
		self,
		raw_output: Any,
		*,
		tile_origin_zyx: TileOriginZYX,
		tile_weight: torch.Tensor | np.ndarray,
		accumulators: Mapping[str, Any],
	) -> None:
		"""Convert raw tile output into logical product accumulators."""
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
	)
	if ret != 0:
		raise RuntimeError(f"download from {source_uri} failed (exit {ret})")


def _auto_download(
	input_path: str,
	crop_xyzwhd: tuple[int, int, int, int, int, int] | None,
	pred_dt_path: str | None = None,
) -> None:
	"""Auto-download input and optional pred-dt data from S3 metadata."""
	_download_one_path(input_path, crop_xyzwhd)
	if pred_dt_path:
		_download_one_path(pred_dt_path, crop_xyzwhd)
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


def _rolling_band_has_range(band: object, z0: int, z1: int) -> bool:
	"""Return True when a rolling band currently contains z=[z0,z1)."""
	origin = getattr(band, "origin_z", None)
	if origin is None:
		return False
	end_z = getattr(band, "end_z", 0)
	return int(z0) >= int(origin) and int(z1) <= int(end_z)


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


_libc = None


def _get_libc():
	global _libc
	if _libc is None:
		_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
	return _libc


def _release_memmap_pages(arr: np.ndarray, z0: int, z1: int) -> None:
	"""Release memmap pages for z-slice range [z0, z1)."""
	if z1 <= z0 or not hasattr(arr, 'ctypes'):
		return
	page = 4096
	aligned_offset = 0
	aligned_length = 0
	try:
		libc = _get_libc()
		if arr.ndim == 3:
			bytes_per_z = int(np.prod(arr.shape[1:])) * arr.itemsize
			offset = z0 * bytes_per_z
		elif arr.ndim == 4:
			for ch in range(arr.shape[0]):
				_release_memmap_pages(arr[ch], z0, z1)
			return
		else:
			return
		length = (z1 - z0) * bytes_per_z
		aligned_offset = ((offset + page - 1) // page) * page
		aligned_end = ((offset + length) // page) * page
		aligned_length = aligned_end - aligned_offset
		if aligned_length <= 0:
			return
		addr = ctypes.c_void_p(arr.ctypes.data + aligned_offset)
		MADV_DONTNEED = 4
		libc.madvise(addr, ctypes.c_size_t(aligned_length), ctypes.c_int(MADV_DONTNEED))
	except Exception:
		pass
	try:
		path = getattr(arr, "_lasagna_tmp_path", None)
		if path and aligned_length > 0 and os.path.exists(path):
			fd = os.open(path, os.O_RDWR)
			try:
				libc = _get_libc()
				FALLOC_FL_KEEP_SIZE = 0x01
				FALLOC_FL_PUNCH_HOLE = 0x02
				ret = libc.fallocate(
					ctypes.c_int(fd),
					ctypes.c_int(FALLOC_FL_KEEP_SIZE | FALLOC_FL_PUNCH_HOLE),
					ctypes.c_longlong(aligned_offset),
					ctypes.c_longlong(aligned_length),
				)
				if ret != 0:
					err = ctypes.get_errno()
					if err not in (0, 38, 45, 95):
						print(f"[predict3d] warning: hole punch failed for {path}: errno={err}", flush=True)
			finally:
				os.close(fd)
	except Exception:
		pass


class _RollingZBand:
	"""Per-channel sparse mmap-backed z band with fixed logical coordinates."""

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
	) -> None:
		self.name = str(name)
		self.channel_count = int(channel_count)
		self.z_size = int(z_size)
		self.y_size = int(y_size)
		self.x_size = int(x_size)
		self.tmp_dir = tmp_dir
		self.prefix = str(prefix)
		self.origin_z = 0
		self._arrays = [self._new_array(ch) for ch in range(self.channel_count)]

	def _new_array(self, ch: int) -> np.memmap:
		fd, path = tempfile.mkstemp(
			prefix=f".predict3d_pid{os.getpid()}_{self.prefix}{self.name}_ch{ch}_",
			suffix=".tmp",
			dir=self.tmp_dir if self.tmp_dir else None,
		)
		try:
			logical_bytes = (
				max(0, self.z_size)
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
			shape=(self.z_size, self.y_size, self.x_size),
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
		self._arrays[int(ch)][int(z0):int(z1), y0:y1, x0:x1] += data

	def view(self, ch: int, z0: int, z1: int) -> np.ndarray:
		if z1 <= z0:
			raise ValueError(f"{self.name} rolling band has no data for z=[{z0},{z1})")
		if z0 < self.origin_z or z1 > self.end_z:
			raise ValueError(
				f"{self.name} rolling band missing z=[{z0},{z1}); "
				f"available=[{self.origin_z},{self.end_z})"
			)
		return self._arrays[int(ch)][int(z0):int(z1)]

	def discard_before(self, z_new: int) -> None:
		z_new = int(z_new)
		if z_new <= self.origin_z:
			return
		z_release = min(z_new, self.z_size)
		for arr in self._arrays:
			_release_memmap_pages(arr, self.origin_z, z_release)
		self.origin_z = z_release

	def _cleanup_array(self, arr: np.ndarray) -> None:
		path = getattr(arr, "_lasagna_tmp_path", None)
		try:
			arr.flush()
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


def _infer_tiled_products_3d(
	model,
	zarr_arr,
	*,
	crop_slices: tuple[int, int, int, int, int, int],
	device: torch.device,
	model_adapter: ModelAdapter,
	output_adapter: OutputAdapter,
	products: Sequence[OutputProductSpec],
	output_region_zyx: tuple[int, int, int, int, int, int],
	full_output_shape_zyx: tuple[int, int, int],
	input_zarr_path: str | None = None,
	output_scaledown_base: int | None = None,
	tile_size: int = 256,
	overlap: int = 64,
	border: int = 16,
	scaledown: int = 1,
	tmp_dir: str | None = None,
	progress: dict | None = None,
	temp_prefix: str = "",
) -> None:
	"""Run tiled 3D inference for coherent products stored at one output scale."""
	z0, z1, y0, y1, x0, x1 = crop_slices
	nz, ny, nx = z1 - z0, y1 - y0, x1 - x0
	volume_shape = tuple(int(v) for v in zarr_arr.shape)
	products = tuple(products)
	if not products:
		raise ValueError("at least one output product is required")

	sd = max(1, int(scaledown))
	stride = max(1, int(tile_size) - int(overlap))
	for name, val in [("tile_size", tile_size), ("stride", stride), ("border", border)]:
		if int(val) % sd != 0:
			raise ValueError(f"{name}={val} must be divisible by scaledown={sd}")

	pad0 = max(0, int(border))
	Zp = _round_up_to_multiple(nz + 2 * pad0, sd)
	Yp = _round_up_to_multiple(ny + 2 * pad0, sd)
	Xp = _round_up_to_multiple(nx + 2 * pad0, sd)
	Zo, Yo, Xo = Zp // sd, Yp // sd, Xp // sd

	out_oz0, out_oy0, out_ox0, out_oz1, out_oy1, out_ox1 = (
		int(v) for v in output_region_zyx
	)
	out_wz = out_oz1 - out_oz0
	out_wy = out_oy1 - out_oy0
	out_wx = out_ox1 - out_ox0
	if out_wz <= 0 or out_wy <= 0 or out_wx <= 0:
		raise ValueError(f"empty output region: {output_region_zyx}")
	full_out_shape = tuple(int(v) for v in full_output_shape_zyx)
	oc = int(products[0].chunk_size)
	if any(int(product.chunk_size) != oc for product in products):
		raise ValueError("single-scale tiled product inference requires one chunk size")

	ov_eff = max(0, int(overlap) - 2 * pad0)

	def _blend_ramp(length, ov, b):
		ramp = np.zeros(length, dtype=np.float32)
		if length <= 0:
			return ramp
		core_start = min(b, length)
		core_end = max(core_start, length - b)
		core_len = core_end - core_start
		if core_len <= 0:
			return ramp
		core = np.ones(core_len, dtype=np.float32)
		if ov > 0:
			ov_core = min(ov, core_len // 2)
			if ov_core > 0:
				edges = np.linspace(0.0, 1.0, ov_core + 1, dtype=np.float32)[1:]
				core[:ov_core] = edges
				core[-ov_core:] = edges[::-1]
		ramp[core_start:core_end] = core
		return ramp

	rz_full = _blend_ramp(int(tile_size), ov_eff, pad0)
	ry_full = _blend_ramp(int(tile_size), ov_eff, pad0)
	rx_full = _blend_ramp(int(tile_size), ov_eff, pad0)
	w_full = torch.from_numpy(
		rz_full[:, None, None] * ry_full[None, :, None] * rx_full[None, None, :]
	).to(device)
	w_out = (
		_pyrdown3d(w_full.unsqueeze(0), factor=sd).squeeze(0).cpu().numpy()
		if sd > 1
		else w_full.cpu().numpy()
	)

	z_positions = _canonical_local_tile_positions(
		volume_size=volume_shape[0],
		crop_start=z0,
		crop_padded_size=Zp,
		tile_size=int(tile_size),
		stride=stride,
		border=pad0,
		scaledown_multiple=sd,
	)
	y_positions = _canonical_local_tile_positions(
		volume_size=volume_shape[1],
		crop_start=y0,
		crop_padded_size=Yp,
		tile_size=int(tile_size),
		stride=stride,
		border=pad0,
		scaledown_multiple=sd,
	)
	x_positions = _canonical_local_tile_positions(
		volume_size=volume_shape[2],
		crop_start=x0,
		crop_padded_size=Xp,
		tile_size=int(tile_size),
		stride=stride,
		border=pad0,
		scaledown_multiple=sd,
	)

	accumulators: dict[str, _RollingZBand] = {}
	wsums: dict[str, _RollingZBand] = {}
	for product in products:
		accumulators[product.name] = _RollingZBand(
			name=f"acc_{product.name}",
			channel_count=int(product.channel_count),
			z_size=Zo,
			y_size=Yo,
			x_size=Xo,
			tmp_dir=tmp_dir,
			prefix=temp_prefix,
		)
		wsums[product.name] = _RollingZBand(
			name=f"wsum_{product.name}",
			channel_count=1,
			z_size=Zo,
			y_size=Yo,
			x_size=Xo,
			tmp_dir=tmp_dir,
			prefix=temp_prefix,
		)
	print(
		f"[predict3d] rolling product accumulators: products={len(products)} "
		f"zyx=({Zo},{Yo},{Xo}) sd={sd}",
		flush=True,
	)

	crop_offset = (z0, y0, x0)
	if input_zarr_path is not None:
		input_zarr_dir = str(Path(str(input_zarr_path).rstrip("/")).resolve())
	else:
		store_path = getattr(getattr(zarr_arr, "store", None), "path", None)
		input_zarr_dir = str(Path(str(store_path or ".")).resolve())
	b = pad0 // sd
	out_end = b + out_wz
	prev_flush = 0
	progress_scaledown = max(1, int(output_scaledown_base if output_scaledown_base is not None else sd))
	tiles_per_zrow = len(y_positions) * len(x_positions)
	total_tiles = len(z_positions) * tiles_per_zrow
	done = 0
	processed_tiles = 0
	tile_time_sum = 0.0
	t0 = time.time()
	if progress is not None:
		progress["tiles_total"] = total_tiles
		progress["tiles_done"] = 0
		progress["tiles_skipped"] = 0
		progress["tiles_processed"] = 0
		progress["tile_time_sum"] = 0.0
		progress["tiles_remaining_est"] = total_tiles

	def _output_chunk_has_input_support(cz: int, cy: int, cx: int) -> bool:
		z_end = min(full_out_shape[0], int(cz) + oc)
		y_end = min(full_out_shape[1], int(cy) + oc)
		x_end = min(full_out_shape[2], int(cx) + oc)
		z_pos = _canonical_tile_positions_for_output_region(
			volume_size=volume_shape[0],
			output_start=int(cz),
			output_end=z_end,
			scaledown=sd,
			tile_size=int(tile_size),
			stride=stride,
			border=pad0,
			scaledown_multiple=sd,
		)
		y_pos = _canonical_tile_positions_for_output_region(
			volume_size=volume_shape[1],
			output_start=int(cy),
			output_end=y_end,
			scaledown=sd,
			tile_size=int(tile_size),
			stride=stride,
			border=pad0,
			scaledown_multiple=sd,
		)
		x_pos = _canonical_tile_positions_for_output_region(
			volume_size=volume_shape[2],
			output_start=int(cx),
			output_end=x_end,
			scaledown=sd,
			tile_size=int(tile_size),
			stride=stride,
			border=pad0,
			scaledown_multiple=sd,
		)
		for pz in z_pos:
			src_z0 = max(0, int(pz) - pad0)
			src_z1 = min(volume_shape[0], int(pz) - pad0 + int(tile_size))
			for py in y_pos:
				src_y0 = max(0, int(py) - pad0)
				src_y1 = min(volume_shape[1], int(py) - pad0 + int(tile_size))
				for px in x_pos:
					src_x0 = max(0, int(px) - pad0)
					src_x1 = min(volume_shape[2], int(px) - pad0 + int(tile_size))
					if _input_has_chunks(input_zarr_dir, src_z0, src_z1, src_y0, src_y1, src_x0, src_x1):
						return True
		return False

	def _is_tile_done(tz: int, ty: int, tx: int) -> bool:
		in_z0 = max(0, int(tz) + z0 - pad0)
		in_z1 = min(volume_shape[0], int(tz) + z0 - pad0 + int(tile_size))
		in_y0 = max(0, int(ty) + y0 - pad0)
		in_y1 = min(volume_shape[1], int(ty) + y0 - pad0 + int(tile_size))
		in_x0 = max(0, int(tx) + x0 - pad0)
		in_x1 = min(volume_shape[2], int(tx) + x0 - pad0 + int(tile_size))
		if not _input_has_chunks(input_zarr_dir, in_z0, in_z1, in_y0, in_y1, in_x0, in_x1):
			return True
		ts_out = int(tile_size) // sd
		az0, az1, _, _ = _downscaled_tile_clip(tz, sd, ts_out, Zo)
		ay0, ay1, _, _ = _downscaled_tile_clip(ty, sd, ts_out, Yo)
		ax0, ax1, _, _ = _downscaled_tile_clip(tx, sd, ts_out, Xo)
		rz0 = max(out_oz0, out_oz0 + az0 - b)
		rz1 = min(out_oz1, out_oz0 + az1 - b)
		ry0 = max(out_oy0, out_oy0 + ay0 - b)
		ry1 = min(out_oy1, out_oy0 + ay1 - b)
		rx0 = max(out_ox0, out_ox0 + ax0 - b)
		rx1 = min(out_ox1, out_ox0 + ax1 - b)
		for chunk_origin in _iter_chunk_origins_for_region(
			rz0, rz1, ry0, ry1, rx0, rx1, oc, full_out_shape,
		):
			for product in products:
				if not output_adapter.product_chunk_complete(
					product,
					chunk_origin_zyx=chunk_origin,
				):
					return False
		return True

	def _flush(complete_z_padded: int) -> None:
		nonlocal prev_flush
		complete_z = int(complete_z_padded) // sd
		flush_from = max(prev_flush, b)
		if complete_z >= out_end:
			flush_to = out_end
		else:
			complete_out_z = complete_z - b
			flush_to = b + (complete_out_z // oc) * oc
		if flush_to <= flush_from:
			return
		out_zs = flush_from - b
		out_ze = flush_to - b
		if out_ze > out_zs:
			for product in products:
				acc = accumulators[product.name]
				wsum = wsums[product.name]
				have_acc = _rolling_band_has_range(acc, flush_from, flush_to)
				if not have_acc:
					continue
				acc_bands = [
					acc.view(ch, flush_from, flush_to)
					for ch in range(int(product.channel_count))
				]
				ws_band = wsum.view(0, flush_from, flush_to)
				for acc_band in acc_bands:
					acc_band /= np.maximum(ws_band, 1.0e-7)
				slab = None
				oz = out_oz0 + out_zs
				local_from = 0
				local_to = flush_to - flush_from
				y_base = pad0 // sd
				x_base = pad0 // sd
				for dz in range(0, out_ze - out_zs, oc):
					for dy in range(0, out_wy, oc):
						for dx in range(0, out_wx, oc):
							cz = oz + dz
							cy = out_oy0 + dy
							cx = out_ox0 + dx
							chunk_origin = (cz, cy, cx)
							if output_adapter.product_chunk_complete(
								product,
								chunk_origin_zyx=chunk_origin,
							):
								continue
							if not _output_chunk_has_input_support(cz, cy, cx):
								continue
							if slab is None:
								slab = np.ascontiguousarray(np.stack([
									acc_band[
										local_from:local_to,
										y_base:y_base + out_wy,
										x_base:x_base + out_wx,
									]
									for acc_band in acc_bands
								], axis=0))
								slab = np.clip(slab * 255.0, 0.0, 255.0).astype(np.uint8)
							cze = min(out_ze - out_zs, dz + oc)
							cye = min(out_wy, dy + oc)
							cxe = min(out_wx, dx + oc)
							chunk_data = {
								channel.name: slab[channel_index, dz:cze, dy:cye, dx:cxe]
								for channel_index, channel in enumerate(product.channels)
							}
							output_adapter.write_product_chunk(
								product,
								chunk_origin_zyx=chunk_origin,
								data=chunk_data,
							)
		for product in products:
			accumulators[product.name].discard_before(flush_to)
			wsums[product.name].discard_before(flush_to)
		prev_flush = max(prev_flush, flush_to)
		if progress is not None:
			progress["finalized_base_z"] = max(
				int(progress.get("finalized_base_z", 0)),
				int((out_oz0 + max(0, flush_to - b)) * progress_scaledown),
			)

	try:
		for i_tz, tz in enumerate(z_positions):
			for ty in y_positions:
				for tx in x_positions:
					if _is_tile_done(tz, ty, tx):
						done += 1
						if progress is not None:
							progress["tiles_done"] = done
							progress["tiles_skipped"] = int(progress.get("tiles_skipped", 0)) + 1
							progress["tiles_remaining_est"] = max(0, total_tiles - done)
						continue

					tile_t0 = time.time()
					tile_np = _read_tile_zarr(
						zarr_arr,
						volume_shape,
						crop_offset,
						tz,
						ty,
						tx,
						int(tile_size),
						pad0,
					)
					if tile_np.dtype == np.uint16:
						tile_np = (tile_np // 257).astype(np.uint8)
					tile_f = tile_np.astype(np.float32) / 255.0
					tile_t = torch.from_numpy(tile_f).unsqueeze(0).unsqueeze(0).to(device)
					valid_t = torch.ones_like(tile_t, dtype=torch.bool, device=device)
					preprocess_tile = getattr(model_adapter, "preprocess_tile", None)
					if preprocess_tile is not None:
						tile_t = preprocess_tile(tile_t, valid_t)
					with torch.inference_mode():
						raw_output = model_adapter.run_tile_inference(
							model,
							tile_t,
							device=device,
						)
					product_tensors = model_adapter.product_tensors_from_output(raw_output)
					for product in products:
						tensor = product_tensors[product.name][0] * w_full
						if sd > 1:
							tensor = _pyrdown3d(tensor, factor=sd)
						product_np = tensor.detach().cpu().numpy()
						ts_out = int(tile_size) // sd
						az0, az1, sz0, sz1 = _downscaled_tile_clip(tz, sd, ts_out, Zo)
						ay0, ay1, sy0, sy1 = _downscaled_tile_clip(ty, sd, ts_out, Yo)
						ax0, ax1, sx0, sx1 = _downscaled_tile_clip(tx, sd, ts_out, Xo)
						if az1 > az0 and ay1 > ay0 and ax1 > ax0:
							acc = accumulators[product.name]
							wsum = wsums[product.name]
							for ch in range(int(product.channel_count)):
								acc.add(
									ch,
									az0,
									az1,
									ay0,
									ay1,
									ax0,
									ax1,
									product_np[ch, sz0:sz1, sy0:sy1, sx0:sx1],
								)
							wsum.add(
								0,
								az0,
								az1,
								ay0,
								ay1,
								ax0,
								ax1,
								w_out[sz0:sz1, sy0:sy1, sx0:sx1],
							)

					tile_elapsed = time.time() - tile_t0
					tile_time_sum += tile_elapsed
					processed_tiles += 1
					done += 1
					if progress is not None:
						progress["tiles_done"] = done
						progress["tiles_processed"] = processed_tiles
						progress["tile_time_sum"] = tile_time_sum
						progress["tiles_remaining_est"] = max(0, total_tiles - done)
						status = _predict3d_progress_line(progress)
					else:
						per = tile_time_sum / max(1, processed_tiles)
						eta = max(0.0, per * (total_tiles - done))
						bar_w = 30
						fill = max(0, min(bar_w, int(round(done / max(1, total_tiles) * bar_w))))
						bar = "#" * fill + "-" * (bar_w - fill)
						status = (
							f"[predict3d] [{bar}] {done}/{total_tiles} tiles "
							f"({100.0 * done / max(1, total_tiles):.1f}%) "
							f"eta {_format_eta(eta)} avg={1000.0 * per:.0f}ms/tile"
						)
					print(f"\r{status}  ", end="", flush=True)

			next_tz = z_positions[i_tz + 1] if i_tz + 1 < len(z_positions) else Zp
			_flush(next_tz)
		print("", flush=True)
		print(
			f"[predict3d] product inference done in {time.time() - t0:.1f}s "
			f"({processed_tiles} processed, {done - processed_tiles} skipped)",
			flush=True,
		)
	finally:
		for acc in accumulators.values():
			acc.cleanup()
		for wsum in wsums.values():
			wsum.cleanup()


def _infer_tiled_3d(
	model,
	zarr_arr,
	*,
	crop_slices: tuple[int, int, int, int, int, int],
	device: torch.device,
	tile_size: int = 256,
	overlap: int = 64,
	border: int = 16,
	out_channels: int = 8,
	cos_scaledown: int = 2,
	other_scaledown: int = 4,
	tmp_dir: str | None = None,
	output_sigmoid: bool = True,
	on_z_complete=None,
	skip_z_positions: int = 0,
	progress: dict | None = None,
	is_tile_done=None,
	temp_prefix: str = "",
	model_adapter: ModelAdapter | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
	"""Run 3D UNet inference with dual-resolution accumulators."""
	z0, z1, y0, y1, x0, x1 = crop_slices
	nz, ny, nx = z1 - z0, y1 - y0, x1 - x0
	volume_shape = tuple(int(v) for v in zarr_arr.shape)

	sd_fine = max(1, int(cos_scaledown))
	sd_coarse = max(1, int(other_scaledown))
	stride = max(1, tile_size - overlap)

	for sd_label, sd_val in [("cos_scaledown", sd_fine), ("other_scaledown", sd_coarse)]:
		if sd_val > 1:
			for name, val in [("tile_size", tile_size), ("stride", stride), ("border", border)]:
				if val % sd_val != 0:
					raise ValueError(f"{name}={val} must be divisible by {sd_label}={sd_val}")

	pad0 = max(0, int(border))
	Zp = nz + 2 * pad0
	Yp = ny + 2 * pad0
	Xp = nx + 2 * pad0

	sd_max = max(sd_fine, sd_coarse)
	if sd_max > 1:
		Zp = _round_up_to_multiple(Zp, sd_max)
		Yp = _round_up_to_multiple(Yp, sd_max)
		Xp = _round_up_to_multiple(Xp, sd_max)

	Zo_f, Yo_f, Xo_f = Zp // sd_fine, Yp // sd_fine, Xp // sd_fine
	Zo_c, Yo_c, Xo_c = Zp // sd_coarse, Yp // sd_coarse, Xp // sd_coarse

	ov_eff = max(0, overlap - 2 * border)

	z_positions = _canonical_local_tile_positions(
		volume_size=volume_shape[0], crop_start=z0, crop_padded_size=Zp,
		tile_size=tile_size, stride=stride, border=pad0, scaledown_multiple=sd_max,
	)
	y_positions = _canonical_local_tile_positions(
		volume_size=volume_shape[1], crop_start=y0, crop_padded_size=Yp,
		tile_size=tile_size, stride=stride, border=pad0, scaledown_multiple=sd_max,
	)
	x_positions = _canonical_local_tile_positions(
		volume_size=volume_shape[2], crop_start=x0, crop_padded_size=Xp,
		tile_size=tile_size, stride=stride, border=pad0, scaledown_multiple=sd_max,
	)

	def _blend_ramp(length, ov, b):
		ramp = np.zeros(length, dtype=np.float32)
		if length <= 0:
			return ramp
		core_start = min(b, length)
		core_end = max(core_start, length - b)
		core_len = core_end - core_start
		if core_len <= 0:
			return ramp
		core = np.ones(core_len, dtype=np.float32)
		if ov > 0:
			ov_core = min(ov, core_len // 2)
			if ov_core > 0:
				edges = np.linspace(0.0, 1.0, ov_core + 1, dtype=np.float32)[1:]
				core[:ov_core] = edges
				core[-ov_core:] = edges[::-1]
		ramp[core_start:core_end] = core
		return ramp

	rz_full = _blend_ramp(tile_size, ov_eff, border)
	ry_full = _blend_ramp(tile_size, ov_eff, border)
	rx_full = _blend_ramp(tile_size, ov_eff, border)
	w_full = torch.from_numpy(
		rz_full[:, None, None] * ry_full[None, :, None] * rx_full[None, None, :]
	).to(device)

	w_fine = (_pyrdown3d(w_full.unsqueeze(0), factor=sd_fine).squeeze(0).cpu().numpy()
			  if sd_fine > 1 else w_full.cpu().numpy())
	w_coarse = (_pyrdown3d(w_full.unsqueeze(0), factor=sd_coarse).squeeze(0).cpu().numpy()
				if sd_coarse > 1 else w_full.cpu().numpy())

	streaming = on_z_complete is not None

	def _make_memmap(suffix, shape):
		fd, p = tempfile.mkstemp(
			prefix=f".predict3d_pid{os.getpid()}_{temp_prefix}{suffix}_",
			suffix=".tmp",
			dir=tmp_dir if tmp_dir else None,
		)
		os.close(fd)
		mm = np.memmap(p, dtype=np.float32, mode="w+", shape=shape)
		mm._lasagna_tmp_path = p
		atexit.register(lambda path=p: os.path.exists(path) and os.unlink(path))
		return mm

	n_other = out_channels - 1
	if streaming:
		acc_fine = _RollingZBand(
			name="acc_fine", channel_count=1, z_size=Zo_f, y_size=Yo_f, x_size=Xo_f,
			tmp_dir=tmp_dir, prefix=temp_prefix,
		)
		wsum_fine = _RollingZBand(
			name="wsum_fine", channel_count=1, z_size=Zo_f, y_size=Yo_f, x_size=Xo_f,
			tmp_dir=tmp_dir, prefix=temp_prefix,
		)
		acc_coarse = _RollingZBand(
			name="acc_coarse", channel_count=n_other, z_size=Zo_c, y_size=Yo_c, x_size=Xo_c,
			tmp_dir=tmp_dir, prefix=temp_prefix,
		)
		wsum_coarse = _RollingZBand(
			name="wsum_coarse", channel_count=1, z_size=Zo_c, y_size=Yo_c, x_size=Xo_c,
			tmp_dir=tmp_dir, prefix=temp_prefix,
		)
		print(
			f"[predict3d] rolling accumulators: fine channels=1 zyx=({Zo_f},{Yo_f},{Xo_f}) sd={sd_fine}; "
			f"coarse channels={n_other} zyx=({Zo_c},{Yo_c},{Xo_c}) sd={sd_coarse}",
			flush=True,
		)
	else:
		acc_fine = _make_memmap("acc_fine", (1, Zo_f, Yo_f, Xo_f))
		wsum_fine = _make_memmap("wsum_fine", (1, Zo_f, Yo_f, Xo_f))
		acc_coarse = _make_memmap("acc_coarse", (n_other, Zo_c, Yo_c, Xo_c))
		wsum_coarse = _make_memmap("wsum_coarse", (1, Zo_c, Yo_c, Xo_c))

		fine_bytes = (np.prod(acc_fine.shape) + np.prod(wsum_fine.shape)) * 4
		coarse_bytes = (np.prod(acc_coarse.shape) + np.prod(wsum_coarse.shape)) * 4
		print(
			f"[predict3d] accumulators: fine ({1},{Zo_f},{Yo_f},{Xo_f}) sd={sd_fine} "
			f"({fine_bytes / (1024**3):.2f} GiB) + "
			f"coarse ({n_other},{Zo_c},{Yo_c},{Xo_c}) sd={sd_coarse} "
			f"({coarse_bytes / (1024**3):.2f} GiB)",
			flush=True,
		)

	tiles_per_zrow = len(y_positions) * len(x_positions)
	total_tiles = len(z_positions) * tiles_per_zrow
	skipped_tiles = skip_z_positions * tiles_per_zrow
	done = skipped_tiles
	processed_tiles = 0
	t0 = time.time()
	_tile_time_sum = 0.0
	crop_offset = (z0, y0, x0)
	if progress is not None:
		progress["tiles_total"] = total_tiles
		progress["tiles_done"] = done
		progress["tiles_skipped"] = skipped_tiles
		progress["tiles_processed"] = 0
		progress["tile_time_sum"] = 0.0
		progress["tiles_remaining_est"] = max(0, total_tiles - done)

	for i_tz, tz in enumerate(z_positions):
		if i_tz < skip_z_positions:
			continue

		for ty in y_positions:
			for tx in x_positions:
				if is_tile_done is not None and is_tile_done(tz, ty, tx):
					done += 1
					if progress is not None:
						progress["tiles_done"] = done
						progress["tiles_skipped"] = int(progress.get("tiles_skipped", 0)) + 1
						progress["tiles_remaining_est"] = max(0, total_tiles - done)
					continue

				_tile_t0 = time.time()
				tile_np = _read_tile_zarr(
					zarr_arr, volume_shape, crop_offset,
					tz, ty, tx, tile_size, border,
				)
				if tile_np.dtype == np.uint16:
					tile_np = (tile_np // 257).astype(np.uint8)

				tile_f = tile_np.astype(np.float32) / 255.0
				tile_t = torch.from_numpy(tile_f).unsqueeze(0).unsqueeze(0).to(device)

				with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
					if model_adapter is None:
						pred = model(tile_t)
					else:
						pred = model_adapter.run_tile_inference(model, tile_t, device=device)
				if isinstance(pred, dict):
					pred = pred["output"]

				raw_nan = torch.isnan(pred).sum().item()
				if raw_nan > 0 or done == skipped_tiles:
					print(flush=True)
					print(
						f"  tile {done}/{total_tiles} "
						f"pos=({tz},{ty},{tx}) "
						f"input: min={tile_f.min():.4f} max={tile_f.max():.4f} "
						f"raw_out: min={pred.min().item():.4f} max={pred.max().item():.4f} "
						f"nan={raw_nan}/{pred.numel()} "
						f"dtype={pred.dtype}",
						flush=True,
					)

				if output_sigmoid:
					pred = torch.sigmoid(pred.float())
				else:
					pred = pred.float().clamp(0.0, 1.0)

				pred_cos = pred[0, 0:1] * w_full
				pred_other = pred[0, 1:] * w_full

				if sd_fine > 1:
					pred_cos = _pyrdown3d(pred_cos, factor=sd_fine)
				cos_np = pred_cos.cpu().numpy()
				ts_f = tile_size // sd_fine
				azl_f, azr_f, szl_f, szr_f = _downscaled_tile_clip(tz, sd_fine, ts_f, Zo_f)
				ayl_f, ayr_f, syl_f, syr_f = _downscaled_tile_clip(ty, sd_fine, ts_f, Yo_f)
				axl_f, axr_f, sxl_f, sxr_f = _downscaled_tile_clip(tx, sd_fine, ts_f, Xo_f)
				if azr_f > azl_f and ayr_f > ayl_f and axr_f > axl_f:
					if streaming:
						acc_fine.add(0, azl_f, azr_f, ayl_f, ayr_f, axl_f, axr_f,
									 cos_np[0, szl_f:szr_f, syl_f:syr_f, sxl_f:sxr_f])
						wsum_fine.add(0, azl_f, azr_f, ayl_f, ayr_f, axl_f, axr_f,
									  w_fine[szl_f:szr_f, syl_f:syr_f, sxl_f:sxr_f])
					else:
						acc_fine[:, azl_f:azr_f, ayl_f:ayr_f, axl_f:axr_f] += cos_np[:, szl_f:szr_f, syl_f:syr_f, sxl_f:sxr_f]
						wsum_fine[0, azl_f:azr_f, ayl_f:ayr_f, axl_f:axr_f] += w_fine[szl_f:szr_f, syl_f:syr_f, sxl_f:sxr_f]

				if sd_coarse > 1:
					pred_other = _pyrdown3d(pred_other, factor=sd_coarse)
				other_np = pred_other.cpu().numpy()
				ts_c = tile_size // sd_coarse
				azl_c, azr_c, szl_c, szr_c = _downscaled_tile_clip(tz, sd_coarse, ts_c, Zo_c)
				ayl_c, ayr_c, syl_c, syr_c = _downscaled_tile_clip(ty, sd_coarse, ts_c, Yo_c)
				axl_c, axr_c, sxl_c, sxr_c = _downscaled_tile_clip(tx, sd_coarse, ts_c, Xo_c)
				if azr_c > azl_c and ayr_c > ayl_c and axr_c > axl_c:
					if streaming:
						for ch in range(n_other):
							acc_coarse.add(ch, azl_c, azr_c, ayl_c, ayr_c, axl_c, axr_c,
										   other_np[ch, szl_c:szr_c, syl_c:syr_c, sxl_c:sxr_c])
						wsum_coarse.add(0, azl_c, azr_c, ayl_c, ayr_c, axl_c, axr_c,
										w_coarse[szl_c:szr_c, syl_c:syr_c, sxl_c:sxr_c])
					else:
						acc_coarse[:, azl_c:azr_c, ayl_c:ayr_c, axl_c:axr_c] += other_np[:, szl_c:szr_c, syl_c:syr_c, sxl_c:sxr_c]
						wsum_coarse[0, azl_c:azr_c, ayl_c:ayr_c, axl_c:axr_c] += w_coarse[szl_c:szr_c, syl_c:syr_c, sxl_c:sxr_c]

				tile_elapsed = time.time() - _tile_t0
				_tile_time_sum += tile_elapsed
				processed_tiles += 1
				done += 1
				if progress is not None:
					progress["tiles_done"] = done
					progress["tiles_processed"] = processed_tiles
					progress["tile_time_sum"] = _tile_time_sum
					progress["tiles_remaining_est"] = max(0, total_tiles - done)
				if progress is None:
					per = _tile_time_sum / max(1, processed_tiles)
					remaining = total_tiles - done
					eta = max(0.0, per * remaining)
					bar_w = 30
					fill = int(round(done / max(1, total_tiles) * bar_w))
					fill = max(0, min(bar_w, fill))
					bar = "#" * fill + "-" * (bar_w - fill)
					status = (
						f"[predict3d] [{bar}] {done}/{total_tiles} tiles "
						f"({100.0 * done / max(1, total_tiles):.1f}%) "
						f"eta {_format_eta(eta)} "
						f"avg={1000.0 * per:.0f}ms/tile"
					)
				else:
					status = _predict3d_progress_line(progress)
				print(f"\r{status}  ", end="", flush=True)

		if on_z_complete is not None:
			next_tz = z_positions[i_tz + 1] if i_tz + 1 < len(z_positions) else Zp
			on_z_complete(acc_fine, wsum_fine, acc_coarse, wsum_coarse, next_tz, pad0)

	print("", flush=True)
	print(
		f"[predict3d] inference done in {time.time() - t0:.1f}s "
		f"({processed_tiles} processed, {done - processed_tiles} skipped)",
		flush=True,
	)

	if on_z_complete is not None:
		acc_fine.cleanup()
		wsum_fine.cleanup()
		acc_coarse.cleanup()
		wsum_coarse.cleanup()
		del acc_fine, wsum_fine, acc_coarse, wsum_coarse
		return None

	acc_fine /= np.maximum(wsum_fine, 1e-7)
	acc_coarse /= np.maximum(wsum_coarse, 1e-7)
	del wsum_fine, wsum_coarse

	b_f = pad0 // sd_fine
	b_c = pad0 // sd_coarse
	nz_f, ny_f, nx_f = nz // sd_fine, ny // sd_fine, nx // sd_fine
	nz_c, ny_c, nx_c = nz // sd_coarse, ny // sd_coarse, nx // sd_coarse

	result_fine = acc_fine[:, b_f:b_f + nz_f, b_f:b_f + ny_f, b_f:b_f + nx_f]
	result_coarse = acc_coarse[:, b_c:b_c + nz_c, b_c:b_c + ny_c, b_c:b_c + nx_c]
	return result_fine, result_coarse
