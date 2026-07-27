from __future__ import annotations

import json
import multiprocessing
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import zarr


def shape_div2(shape: tuple[int, int, int], n: int) -> tuple[int, int, int]:
	z, y, x = (int(v) for v in shape)
	for _ in range(max(0, int(n))):
		z = max(1, (z + 1) // 2)
		y = max(1, (y + 1) // 2)
		x = max(1, (x + 1) // 2)
	return z, y, x


def print_progress(*, prefix: str, done: int, total: int, t0: float, suffix: str = "") -> None:
	d = max(0, int(done))
	t = max(1, int(total))
	elapsed = max(1e-6, float(time.time() - t0))
	per = elapsed / float(max(1, d)) if d > 0 else 0.0
	eta = max(0.0, per * float(max(0, t - d))) if d > 0 else 0.0
	eta_m = int(eta // 60.0)
	eta_s = int(eta % 60.0)
	bar_w = 30
	fill = int(round((float(d) / float(t)) * float(bar_w)))
	bar = "#" * max(0, min(bar_w, fill)) + "-" * max(0, bar_w - max(0, min(bar_w, fill)))
	print(
		f"\r{prefix} [{bar}] {d}/{t} ({(100.0 * d / float(t)):.1f}%) eta {eta_m:02d}:{eta_s:02d}{suffix}",
		end="",
		flush=True,
	)


def zarr_chunk_path(level_path: str | Path, sep: str, iz: int, iy: int, ix: int) -> Path:
	level_path = Path(level_path)
	if sep == "/":
		return level_path / str(iz) / str(iy) / str(ix)
	return level_path / f"{iz}{sep}{iy}{sep}{ix}"


def omezarr_dim_sep(omezarr_path: str | Path, level: int) -> str:
	zarray_path = Path(omezarr_path) / str(level) / ".zarray"
	try:
		with zarray_path.open() as f:
			return json.load(f).get("dimension_separator", ".")
	except Exception:
		return "."


_dim_sep_cache: dict[tuple[str, int], str] = {}


def omezarr_chunk_exists(
	omezarr_path: str | Path,
	level: int,
	z: int,
	y: int,
	x: int,
	chunk_size: int | tuple[int, int, int],
) -> bool:
	key = (str(omezarr_path), int(level))
	if key not in _dim_sep_cache:
		_dim_sep_cache[key] = omezarr_dim_sep(omezarr_path, level)
	sep = _dim_sep_cache[key]
	cz, cy, cx = _normalize_chunk_zyx(chunk_size)
	iz, iy, ix = int(z) // cz, int(y) // cy, int(x) // cx
	return zarr_chunk_path(Path(omezarr_path) / str(level), sep, iz, iy, ix).is_file()


def omezarr_region_has_chunks(
	omezarr_path: str | Path,
	level: int,
	z0: int,
	z1: int,
	y0: int,
	y1: int,
	x0: int,
	x1: int,
	chunk_size: int | tuple[int, int, int],
) -> bool:
	cz, cy, cx = _normalize_chunk_zyx(chunk_size)
	for z in range((max(0, int(z0)) // cz) * cz, max(0, int(z1) + cz - 1), cz):
		if z >= int(z1):
			break
		for y in range((max(0, int(y0)) // cy) * cy, max(0, int(y1) + cy - 1), cy):
			if y >= int(y1):
				break
			for x in range((max(0, int(x0)) // cx) * cx, max(0, int(x1) + cx - 1), cx):
				if x >= int(x1):
					break
				if omezarr_chunk_exists(omezarr_path, level, z, y, x, chunk_size):
					return True
	return False


def set_pyramid_metadata(group: zarr.Group, *, method: str) -> None:
	group.attrs["lasagna_pyramid_downsample"] = method


def _normalize_chunk_zyx(chunk: int | tuple[int, int, int]) -> tuple[int, int, int]:
	if isinstance(chunk, tuple):
		cz, cy, cx = (int(v) for v in chunk)
	else:
		cz = cy = cx = int(chunk)
	if cz <= 0 or cy <= 0 or cx <= 0:
		raise ValueError(f"invalid chunk size: {(cz, cy, cx)}")
	return cz, cy, cx


def _level_chunks_zyx(group: zarr.Group, level: int) -> tuple[int, int, int]:
	return tuple(int(v) for v in group[str(level)].chunks[-3:])


def _mean_pool2x_u8(slab: np.ndarray, *, zero_overrides: bool = False) -> np.ndarray:
	if slab.ndim != 3:
		raise ValueError(f"expected 3D slab, got shape={slab.shape}")
	z, y, x = (int(v) for v in slab.shape)
	out_shape = ((z + 1) // 2, (y + 1) // 2, (x + 1) // 2)
	acc = np.zeros(out_shape, dtype=np.float32)
	cnt = np.zeros(out_shape, dtype=np.float32)
	zero = np.zeros(out_shape, dtype=bool) if zero_overrides else None
	for dz in (0, 1):
		for dy in (0, 1):
			for dx in (0, 1):
				part = slab[dz::2, dy::2, dx::2]
				if part.size == 0:
					continue
				sz, sy, sx = part.shape
				acc[:sz, :sy, :sx] += part.astype(np.float32, copy=False)
				cnt[:sz, :sy, :sx] += 1.0
				if zero is not None:
					zero[:sz, :sy, :sx] |= part == 0
	out = np.rint(acc / np.maximum(cnt, 1.0)).clip(0.0, 255.0).astype(np.uint8)
	if zero is not None:
		out[zero] = 0
	return out


def _mean_pool2x_f32(slab: np.ndarray) -> np.ndarray:
	if slab.ndim != 3:
		raise ValueError(f"expected 3D slab, got shape={slab.shape}")
	z, y, x = (int(v) for v in slab.shape)
	out_shape = ((z + 1) // 2, (y + 1) // 2, (x + 1) // 2)
	acc = np.zeros(out_shape, dtype=np.float32)
	cnt = np.zeros(out_shape, dtype=np.float32)
	for dz in (0, 1):
		for dy in (0, 1):
			for dx in (0, 1):
				part = slab[dz::2, dy::2, dx::2]
				if part.size == 0:
					continue
				sz, sy, sx = part.shape
				acc[:sz, :sy, :sx] += part.astype(np.float32, copy=False)
				cnt[:sz, :sy, :sx] += 1.0
	return acc / np.maximum(cnt, 1.0)


def _decode_normals(nx_u8: np.ndarray, ny_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	nx = (nx_u8.astype(np.float32, copy=False) - 128.0) / 127.0
	ny = (ny_u8.astype(np.float32, copy=False) - 128.0) / 127.0
	nz = np.sqrt(np.maximum(0.0, 1.0 - nx * nx - ny * ny)).astype(np.float32, copy=False)
	return nx, ny, nz


def _moment_pool2x_normals(nx_u8: np.ndarray, ny_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	if nx_u8.shape != ny_u8.shape:
		raise ValueError(f"normal channel shape mismatch: nx={nx_u8.shape} ny={ny_u8.shape}")
	nx, ny, nz = _decode_normals(nx_u8, ny_u8)
	xx = _mean_pool2x_f32(nx * nx)
	xy = _mean_pool2x_f32(nx * ny)
	xz = _mean_pool2x_f32(nx * nz)
	yy = _mean_pool2x_f32(ny * ny)
	yz = _mean_pool2x_f32(ny * nz)
	zz = _mean_pool2x_f32(nz * nz)

	mat = np.empty(xx.shape + (3, 3), dtype=np.float32)
	mat[..., 0, 0] = xx
	mat[..., 0, 1] = xy
	mat[..., 0, 2] = xz
	mat[..., 1, 0] = xy
	mat[..., 1, 1] = yy
	mat[..., 1, 2] = yz
	mat[..., 2, 0] = xz
	mat[..., 2, 1] = yz
	mat[..., 2, 2] = zz
	vals, vecs = np.linalg.eigh(mat)
	v = np.take_along_axis(vecs, np.argmax(vals, axis=-1)[..., None, None], axis=-1)[..., 0]
	flip = np.where(v[..., 2:3] < 0.0, -1.0, 1.0).astype(np.float32)
	v = v * flip
	norm = np.sqrt(np.maximum(np.sum(v * v, axis=-1, keepdims=True), 1e-12))
	v = v / norm
	nx_out = np.rint(v[..., 0] * 127.0 + 128.0).clip(0.0, 255.0).astype(np.uint8)
	ny_out = np.rint(v[..., 1] * 127.0 + 128.0).clip(0.0, 255.0).astype(np.uint8)
	return nx_out, ny_out


def clear_level_chunks(level_path: str | Path) -> None:
	level_path = Path(level_path)
	if not level_path.is_dir():
		return
	for child in level_path.iterdir():
		if child.name in {".zarray", ".zattrs", ".zgroup", "zarr.json"}:
			continue
		if child.is_dir():
			shutil.rmtree(child)
		else:
			child.unlink()


def clear_coarser_levels(omezarr_path: str | Path, data_level: int, n_levels: int) -> None:
	for lv in range(int(data_level) + 1, int(n_levels)):
		clear_level_chunks(Path(omezarr_path) / str(lv))


def _write_level_block(
	*,
	omezarr_path: str,
	level: int,
	z0: int,
	y0: int,
	x0: int,
	data: np.ndarray,
) -> None:
	g = zarr.open_group(str(omezarr_path), mode="r+")
	dst = g[str(level)]
	z1 = min(int(dst.shape[0]), int(z0) + int(data.shape[0]))
	y1 = min(int(dst.shape[1]), int(y0) + int(data.shape[1]))
	x1 = min(int(dst.shape[2]), int(x0) + int(data.shape[2]))
	wz, wy, wx = z1 - int(z0), y1 - int(y0), x1 - int(x0)
	if wz > 0 and wy > 0 and wx > 0:
		dst[int(z0):z1, int(y0):y1, int(x0):x1] = data[:wz, :wy, :wx]


def downsample_scalar_chunk_worker(args_tuple) -> None:
	if len(args_tuple) == 9:
		(out_path_str, src_level, dst_level, z0, z1, y0, y1, x0, x1) = args_tuple
		zero_overrides = False
		skip_existing = False
		require_source_chunks = False
		dst_chunk_zyx = None
		src_chunk_zyx = None
	elif len(args_tuple) == 10:
		(out_path_str, src_level, dst_level, z0, z1, y0, y1, x0, x1, zero_overrides) = args_tuple
		skip_existing = False
		require_source_chunks = False
		dst_chunk_zyx = None
		src_chunk_zyx = None
	else:
		(
			out_path_str, src_level, dst_level, z0, z1, y0, y1, x0, x1,
			zero_overrides, skip_existing, require_source_chunks, dst_chunk_zyx, src_chunk_zyx,
		) = args_tuple
	if skip_existing and omezarr_chunk_exists(
		out_path_str, dst_level, int(z0) // 2, int(y0) // 2, int(x0) // 2, dst_chunk_zyx,
	):
		return "skipped_existing"
	if require_source_chunks and not omezarr_region_has_chunks(
		out_path_str, src_level, z0, z1, y0, y1, x0, x1, src_chunk_zyx,
	):
		return "skipped_empty_source"
	g = zarr.open_group(str(out_path_str), mode="r+")
	src = g[str(src_level)]
	slab = np.asarray(src[z0:z1, y0:y1, x0:x1], dtype=np.uint8)
	if slab.size == 0:
		return "skipped_empty_slice"
	down = _mean_pool2x_u8(slab, zero_overrides=bool(zero_overrides))
	_write_level_block(
		omezarr_path=str(out_path_str),
		level=int(dst_level),
		z0=int(z0) // 2,
		y0=int(y0) // 2,
		x0=int(x0) // 2,
		data=down,
	)
	return "written"


def downsample_normal_pair_chunk_worker(args_tuple) -> None:
	if len(args_tuple) == 10:
		(nx_path_str, ny_path_str, src_level, dst_level, z0, z1, y0, y1, x0, x1) = args_tuple
		skip_existing = False
		require_source_chunks = False
		dst_chunk_zyx = None
		nx_src_chunk_zyx = None
		ny_src_chunk_zyx = None
	else:
		(
			nx_path_str, ny_path_str, src_level, dst_level, z0, z1, y0, y1, x0, x1,
			skip_existing, require_source_chunks, dst_chunk_zyx, nx_src_chunk_zyx, ny_src_chunk_zyx,
		) = args_tuple
	dst_z, dst_y, dst_x = int(z0) // 2, int(y0) // 2, int(x0) // 2
	if (
		skip_existing
		and omezarr_chunk_exists(nx_path_str, dst_level, dst_z, dst_y, dst_x, dst_chunk_zyx)
		and omezarr_chunk_exists(ny_path_str, dst_level, dst_z, dst_y, dst_x, dst_chunk_zyx)
	):
		return "skipped_existing"
	if require_source_chunks:
		nx_has = omezarr_region_has_chunks(
			nx_path_str, src_level, z0, z1, y0, y1, x0, x1, nx_src_chunk_zyx,
		)
		ny_has = omezarr_region_has_chunks(
			ny_path_str, src_level, z0, z1, y0, y1, x0, x1, ny_src_chunk_zyx,
		)
		if not (nx_has and ny_has):
			return "skipped_empty_source"
	nx_g = zarr.open_group(str(nx_path_str), mode="r+")
	ny_g = zarr.open_group(str(ny_path_str), mode="r+")
	nx_slab = np.asarray(nx_g[str(src_level)][z0:z1, y0:y1, x0:x1], dtype=np.uint8)
	ny_slab = np.asarray(ny_g[str(src_level)][z0:z1, y0:y1, x0:x1], dtype=np.uint8)
	if nx_slab.size == 0:
		return "skipped_empty_slice"
	nx_down, ny_down = _moment_pool2x_normals(nx_slab, ny_slab)
	dz0, dy0, dx0 = int(z0) // 2, int(y0) // 2, int(x0) // 2
	_write_level_block(omezarr_path=str(nx_path_str), level=int(dst_level), z0=dz0, y0=dy0, x0=dx0, data=nx_down)
	_write_level_block(omezarr_path=str(ny_path_str), level=int(dst_level), z0=dz0, y0=dy0, x0=dx0, data=ny_down)
	return "written"


def _downsample_candidate_bounds(
	*,
	src_shape: tuple[int, int, int],
	chunk_zyx: tuple[int, int, int],
	crop_zyx: tuple[int, int, int, int, int, int] | None,
) -> tuple[int, int, int, int, int, int, int, int, int]:
	cz2, cy2, cx2 = (2 * v for v in chunk_zyx)
	if crop_zyx is not None:
		sz0, sy0, sx0, sz1, sy1, sx1 = (int(v) for v in crop_zyx)
	else:
		sz0 = sy0 = sx0 = 0
		sz1, sy1, sx1 = src_shape
	sz0 = max(0, min(src_shape[0], (sz0 // cz2) * cz2))
	sy0 = max(0, min(src_shape[1], (sy0 // cy2) * cy2))
	sx0 = max(0, min(src_shape[2], (sx0 // cx2) * cx2))
	sz1 = max(sz0, min(src_shape[0], ((sz1 + cz2 - 1) // cz2) * cz2))
	sy1 = max(sy0, min(src_shape[1], ((sy1 + cy2 - 1) // cy2) * cy2))
	sx1 = max(sx0, min(src_shape[2], ((sx1 + cx2 - 1) // cx2) * cx2))
	nz = (sz1 - sz0 + cz2 - 1) // cz2 if sz1 > sz0 else 0
	ny = (sy1 - sy0 + cy2 - 1) // cy2 if sy1 > sy0 else 0
	nx = (sx1 - sx0 + cx2 - 1) // cx2 if sx1 > sx0 else 0
	return sz0, sy0, sx0, sz1, sy1, sx1, nz, ny, nx


def _iter_downsample_regions(
	*,
	src_shape: tuple[int, int, int],
	chunk_zyx: tuple[int, int, int],
	crop_zyx: tuple[int, int, int, int, int, int] | None,
):
	sz0, sy0, sx0, sz1, sy1, sx1, _nz, _ny, _nx = _downsample_candidate_bounds(
		src_shape=src_shape,
		chunk_zyx=chunk_zyx,
		crop_zyx=crop_zyx,
	)
	cz2, cy2, cx2 = (2 * v for v in chunk_zyx)
	for z0 in range(sz0, sz1, cz2):
		z1 = min(sz1, z0 + cz2)
		for y0 in range(sy0, sy1, cy2):
			y1 = min(sy1, y0 + cy2)
			for x0 in range(sx0, sx1, cx2):
				x1 = min(sx1, x0 + cx2)
				yield z0, z1, y0, y1, x0, x1


def _make_downsample_work(
	*,
	omezarr_path: str | Path,
	src_level: int,
	dst_level: int,
	chunk: int | tuple[int, int, int] | None,
	crop_zyx: tuple[int, int, int, int, int, int] | None,
	skip_existing: bool,
	zero_overrides: bool = False,
	require_source_chunks: bool = False,
) -> tuple[list[tuple], int]:
	g = zarr.open_group(str(omezarr_path), mode="r+")
	src_shape = tuple(int(v) for v in g[str(src_level)].shape)
	src_chunk_zyx = _level_chunks_zyx(g, src_level)
	chunk_zyx = _level_chunks_zyx(g, dst_level) if chunk is None else _normalize_chunk_zyx(chunk)

	work: list[tuple] = []
	skipped = 0
	for z0, z1, y0, y1, x0, x1 in _iter_downsample_regions(
		src_shape=src_shape,
		chunk_zyx=chunk_zyx,
		crop_zyx=crop_zyx,
	):
		if skip_existing and omezarr_chunk_exists(omezarr_path, dst_level, z0 // 2, y0 // 2, x0 // 2, chunk_zyx):
			skipped += 1
			continue
		if require_source_chunks and not omezarr_region_has_chunks(
			omezarr_path, src_level, z0, z1, y0, y1, x0, x1, src_chunk_zyx,
		):
			skipped += 1
			continue
		work.append((str(omezarr_path), int(src_level), int(dst_level), z0, z1, y0, y1, x0, x1, bool(zero_overrides)))
	return work, skipped


def _make_scalar_downsample_stream(
	*,
	omezarr_path: str | Path,
	src_level: int,
	dst_level: int,
	chunk: int | tuple[int, int, int] | None,
	crop_zyx: tuple[int, int, int, int, int, int] | None,
	skip_existing: bool,
	zero_overrides: bool = False,
	require_source_chunks: bool = False,
):
	g = zarr.open_group(str(omezarr_path), mode="r+")
	src_shape = tuple(int(v) for v in g[str(src_level)].shape)
	src_chunk_zyx = _level_chunks_zyx(g, src_level)
	chunk_zyx = _level_chunks_zyx(g, dst_level) if chunk is None else _normalize_chunk_zyx(chunk)
	*_bounds, nz, ny, nx = _downsample_candidate_bounds(
		src_shape=src_shape,
		chunk_zyx=chunk_zyx,
		crop_zyx=crop_zyx,
	)
	total = int(nz) * int(ny) * int(nx)

	def _iter():
		for z0, z1, y0, y1, x0, x1 in _iter_downsample_regions(
			src_shape=src_shape,
			chunk_zyx=chunk_zyx,
			crop_zyx=crop_zyx,
		):
			yield (
				str(omezarr_path), int(src_level), int(dst_level), z0, z1, y0, y1, x0, x1,
				bool(zero_overrides), bool(skip_existing), bool(require_source_chunks),
				chunk_zyx, src_chunk_zyx,
			)

	return _iter(), total


def _make_normal_downsample_stream(
	*,
	nx_omezarr_path: str | Path,
	ny_omezarr_path: str | Path,
	src_level: int,
	dst_level: int,
	chunk: int | tuple[int, int, int] | None,
	crop_zyx: tuple[int, int, int, int, int, int] | None,
	skip_existing: bool,
	require_source_chunks: bool = False,
):
	nx_g = zarr.open_group(str(nx_omezarr_path), mode="r+")
	ny_g = zarr.open_group(str(ny_omezarr_path), mode="r+")
	src_shape = tuple(int(v) for v in nx_g[str(src_level)].shape)
	chunk_zyx = _level_chunks_zyx(nx_g, dst_level) if chunk is None else _normalize_chunk_zyx(chunk)
	nx_src_chunk_zyx = _level_chunks_zyx(nx_g, src_level)
	ny_src_chunk_zyx = _level_chunks_zyx(ny_g, src_level)
	*_bounds, nz, ny, nx = _downsample_candidate_bounds(
		src_shape=src_shape,
		chunk_zyx=chunk_zyx,
		crop_zyx=crop_zyx,
	)
	total = int(nz) * int(ny) * int(nx)

	def _iter():
		for z0, z1, y0, y1, x0, x1 in _iter_downsample_regions(
			src_shape=src_shape,
			chunk_zyx=chunk_zyx,
			crop_zyx=crop_zyx,
		):
			yield (
				str(nx_omezarr_path), str(ny_omezarr_path), int(src_level), int(dst_level),
				z0, z1, y0, y1, x0, x1,
				bool(skip_existing), bool(require_source_chunks), chunk_zyx, nx_src_chunk_zyx, ny_src_chunk_zyx,
			)

	return _iter(), total


def _scaled_crop_for_source_level(
	crop_zyx: tuple[int, int, int, int, int, int] | None,
	levels_above_data: int,
) -> tuple[int, int, int, int, int, int] | None:
	if crop_zyx is None:
		return None
	z0, y0, x0, z1, y1, x1 = (int(v) for v in crop_zyx)
	scale = 2 ** max(0, int(levels_above_data))
	return (
		z0 // scale,
		y0 // scale,
		x0 // scale,
		(z1 + scale - 1) // scale,
		(y1 + scale - 1) // scale,
		(x1 + scale - 1) // scale,
	)


def _pyramid_status_suffix(counts: dict[str, int]) -> str:
	parts = []
	for key, label in (
		("written", "write"),
		("skipped_existing", "skip_existing"),
		("skipped_empty_source", "skip_empty"),
		("skipped_empty_slice", "skip_empty_slice"),
	):
		value = int(counts.get(key, 0))
		if value:
			parts.append(f"{label}={value}")
	return " " + " ".join(parts) if parts else ""


def _run_pool(work, worker, *, workers: int, tag: str, total: int | None = None) -> None:
	n_work = int(total) if total is not None else len(work)
	if n_work == 0:
		return
	t0 = time.time()
	done_count = [0]
	status_counts: dict[str, int] = {}
	lock = threading.Lock()
	stop = threading.Event()

	def _prog() -> None:
		while not stop.is_set():
			with lock:
				d = done_count[0]
				suffix = _pyramid_status_suffix(status_counts)
			print_progress(prefix=tag, done=d, total=n_work, t0=t0, suffix=suffix)
			stop.wait(0.5)

	prog_thread = threading.Thread(target=_prog, daemon=True)
	prog_thread.start()
	if int(workers) <= 1:
		for status in map(worker, work):
			with lock:
				done_count[0] += 1
				if status:
					status_counts[str(status)] = status_counts.get(str(status), 0) + 1
	else:
		with multiprocessing.Pool(processes=min(max(1, int(workers)), n_work)) as pool:
			for status in pool.imap_unordered(worker, work):
				with lock:
					done_count[0] += 1
					if status:
						status_counts[str(status)] = status_counts.get(str(status), 0) + 1
	stop.set()
	prog_thread.join(timeout=2)
	print_progress(prefix=tag, done=n_work, total=n_work, t0=t0, suffix=_pyramid_status_suffix(status_counts))
	print("", flush=True)


def build_scalar_omezarr_pyramid(
	omezarr_path: str | Path,
	data_level: int,
	n_levels: int,
	chunk: int | tuple[int, int, int] | None = None,
	*,
	workers: int = 0,
	crop_zyx: tuple[int, int, int, int, int, int] | None = None,
	label: str = "",
	force: bool = False,
	zero_overrides: bool = False,
	scan_existing_source_chunks: bool = False,
) -> None:
	if workers <= 0:
		workers = max(1, multiprocessing.cpu_count())
	g = zarr.open_group(str(omezarr_path), mode="r+")
	if force:
		clear_coarser_levels(omezarr_path, data_level, n_levels)
	for lv in range(int(data_level) + 1, int(n_levels)):
		src_lv = lv - 1
		src_crop = None if scan_existing_source_chunks else _scaled_crop_for_source_level(crop_zyx, src_lv - int(data_level))
		work, total = _make_scalar_downsample_stream(
			omezarr_path=omezarr_path,
			src_level=src_lv,
			dst_level=lv,
			chunk=chunk,
			crop_zyx=src_crop,
			skip_existing=not force,
			zero_overrides=zero_overrides,
			require_source_chunks=scan_existing_source_chunks,
		)
		tag = f"[pyramid {label} L{lv}]" if label else f"[pyramid L{lv}]"
		_run_pool(work, downsample_scalar_chunk_worker, workers=workers, tag=tag, total=total)
	set_pyramid_metadata(g, method="mean_pool2x_zero_overrides" if zero_overrides else "mean_pool2x")


def build_normal_omezarr_pyramid(
	nx_omezarr_path: str | Path,
	ny_omezarr_path: str | Path,
	data_level: int,
	n_levels: int,
	chunk: int | tuple[int, int, int] | None = None,
	*,
	workers: int = 0,
	crop_zyx: tuple[int, int, int, int, int, int] | None = None,
	label: str = "normal",
	force: bool = False,
	scan_existing_source_chunks: bool = False,
) -> None:
	if workers <= 0:
		workers = max(1, multiprocessing.cpu_count())
	nx_g = zarr.open_group(str(nx_omezarr_path), mode="r+")
	ny_g = zarr.open_group(str(ny_omezarr_path), mode="r+")
	if force:
		clear_coarser_levels(nx_omezarr_path, data_level, n_levels)
		clear_coarser_levels(ny_omezarr_path, data_level, n_levels)
	for lv in range(int(data_level) + 1, int(n_levels)):
		src_lv = lv - 1
		src_crop = None if scan_existing_source_chunks else _scaled_crop_for_source_level(crop_zyx, src_lv - int(data_level))
		work, total = _make_normal_downsample_stream(
			nx_omezarr_path=nx_omezarr_path,
			ny_omezarr_path=ny_omezarr_path,
			src_level=src_lv,
			dst_level=lv,
			chunk=chunk,
			crop_zyx=src_crop,
			skip_existing=not force,
			require_source_chunks=scan_existing_source_chunks,
		)
		tag = f"[pyramid {label} L{lv}]"
		_run_pool(work, downsample_normal_pair_chunk_worker, workers=workers, tag=tag, total=total)
	set_pyramid_metadata(nx_g, method="normal_second_moment_mean_pool2x")
	set_pyramid_metadata(ny_g, method="normal_second_moment_mean_pool2x")
