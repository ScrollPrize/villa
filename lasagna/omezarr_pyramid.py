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


def print_progress(*, prefix: str, done: int, total: int, t0: float) -> None:
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
		f"\r{prefix} [{bar}] {d}/{t} ({(100.0 * d / float(t)):.1f}%) eta {eta_m:02d}:{eta_s:02d}",
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
	else:
		(out_path_str, src_level, dst_level, z0, z1, y0, y1, x0, x1, zero_overrides) = args_tuple
	g = zarr.open_group(str(out_path_str), mode="r+")
	src = g[str(src_level)]
	slab = np.asarray(src[z0:z1, y0:y1, x0:x1], dtype=np.uint8)
	if slab.size == 0:
		return
	down = _mean_pool2x_u8(slab, zero_overrides=bool(zero_overrides))
	_write_level_block(
		omezarr_path=str(out_path_str),
		level=int(dst_level),
		z0=int(z0) // 2,
		y0=int(y0) // 2,
		x0=int(x0) // 2,
		data=down,
	)


def downsample_normal_pair_chunk_worker(args_tuple) -> None:
	(nx_path_str, ny_path_str, src_level, dst_level, z0, z1, y0, y1, x0, x1) = args_tuple
	nx_g = zarr.open_group(str(nx_path_str), mode="r+")
	ny_g = zarr.open_group(str(ny_path_str), mode="r+")
	nx_slab = np.asarray(nx_g[str(src_level)][z0:z1, y0:y1, x0:x1], dtype=np.uint8)
	ny_slab = np.asarray(ny_g[str(src_level)][z0:z1, y0:y1, x0:x1], dtype=np.uint8)
	if nx_slab.size == 0:
		return
	nx_down, ny_down = _moment_pool2x_normals(nx_slab, ny_slab)
	dz0, dy0, dx0 = int(z0) // 2, int(y0) // 2, int(x0) // 2
	_write_level_block(omezarr_path=str(nx_path_str), level=int(dst_level), z0=dz0, y0=dy0, x0=dx0, data=nx_down)
	_write_level_block(omezarr_path=str(ny_path_str), level=int(dst_level), z0=dz0, y0=dy0, x0=dx0, data=ny_down)


def _make_downsample_work(
	*,
	omezarr_path: str | Path,
	src_level: int,
	dst_level: int,
	chunk: int | tuple[int, int, int] | None,
	crop_zyx: tuple[int, int, int, int, int, int] | None,
	skip_existing: bool,
	zero_overrides: bool = False,
) -> tuple[list[tuple], int]:
	g = zarr.open_group(str(omezarr_path), mode="r+")
	src_shape = tuple(int(v) for v in g[str(src_level)].shape)
	chunk_zyx = _level_chunks_zyx(g, dst_level) if chunk is None else _normalize_chunk_zyx(chunk)
	cz2, cy2, cx2 = (2 * v for v in chunk_zyx)
	if crop_zyx is not None:
		cz0_base, cy0_base, cx0_base, cz1_base, cy1_base, cx1_base = (int(v) for v in crop_zyx)
		# crop_zyx is always expressed in data-level coordinates by callers.
		# Source level is data_level + k, so callers pre-scale it through level_offset.
		sz0, sy0, sx0, sz1, sy1, sx1 = cz0_base, cy0_base, cx0_base, cz1_base, cy1_base, cx1_base
	else:
		sz0 = sy0 = sx0 = 0
		sz1, sy1, sx1 = src_shape
	sz0 = max(0, min(src_shape[0], (sz0 // cz2) * cz2))
	sy0 = max(0, min(src_shape[1], (sy0 // cy2) * cy2))
	sx0 = max(0, min(src_shape[2], (sx0 // cx2) * cx2))
	sz1 = max(sz0, min(src_shape[0], ((sz1 + cz2 - 1) // cz2) * cz2))
	sy1 = max(sy0, min(src_shape[1], ((sy1 + cy2 - 1) // cy2) * cy2))
	sx1 = max(sx0, min(src_shape[2], ((sx1 + cx2 - 1) // cx2) * cx2))

	work: list[tuple] = []
	skipped = 0
	for z0 in range(sz0, sz1, cz2):
		z1 = min(sz1, z0 + cz2)
		for y0 in range(sy0, sy1, cy2):
			y1 = min(sy1, y0 + cy2)
			for x0 in range(sx0, sx1, cx2):
				x1 = min(sx1, x0 + cx2)
				if skip_existing and omezarr_chunk_exists(omezarr_path, dst_level, z0 // 2, y0 // 2, x0 // 2, chunk_zyx):
					skipped += 1
					continue
				work.append((str(omezarr_path), int(src_level), int(dst_level), z0, z1, y0, y1, x0, x1, bool(zero_overrides)))
	return work, skipped


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


def _run_pool(work: list[tuple], worker, *, workers: int, tag: str) -> None:
	n_work = len(work)
	if n_work == 0:
		return
	t0 = time.time()
	done_count = [0]
	lock = threading.Lock()
	stop = threading.Event()

	def _prog() -> None:
		while not stop.is_set():
			with lock:
				d = done_count[0]
			print_progress(prefix=tag, done=d, total=n_work, t0=t0)
			stop.wait(0.5)

	prog_thread = threading.Thread(target=_prog, daemon=True)
	prog_thread.start()
	with multiprocessing.Pool(processes=min(max(1, int(workers)), n_work)) as pool:
		for _ in pool.imap_unordered(worker, work):
			with lock:
				done_count[0] += 1
	stop.set()
	prog_thread.join(timeout=2)
	print_progress(prefix=tag, done=n_work, total=n_work, t0=t0)
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
) -> None:
	if workers <= 0:
		workers = max(1, multiprocessing.cpu_count())
	g = zarr.open_group(str(omezarr_path), mode="r+")
	if force:
		clear_coarser_levels(omezarr_path, data_level, n_levels)
	for lv in range(int(data_level) + 1, int(n_levels)):
		src_lv = lv - 1
		src_crop = _scaled_crop_for_source_level(crop_zyx, src_lv - int(data_level))
		work, skipped = _make_downsample_work(
			omezarr_path=omezarr_path,
			src_level=src_lv,
			dst_level=lv,
			chunk=chunk,
			crop_zyx=src_crop,
			skip_existing=not force,
			zero_overrides=zero_overrides,
		)
		tag = f"[pyramid {label} L{lv}]" if label else f"[pyramid L{lv}]"
		if skipped > 0:
			print(f"{tag} skipped {skipped} existing chunks", flush=True)
		_run_pool(work, downsample_scalar_chunk_worker, workers=workers, tag=tag)
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
		src_crop = _scaled_crop_for_source_level(crop_zyx, src_lv - int(data_level))
		work_base, skipped = _make_downsample_work(
			omezarr_path=nx_omezarr_path,
			src_level=src_lv,
			dst_level=lv,
			chunk=chunk,
			crop_zyx=src_crop,
			skip_existing=not force,
		)
		work = [
			(str(nx_omezarr_path), str(ny_omezarr_path), src_lv, lv, z0, z1, y0, y1, x0, x1)
			for _path, _src, _dst, z0, z1, y0, y1, x0, x1, *_rest in work_base
		]
		tag = f"[pyramid {label} L{lv}]"
		if skipped > 0:
			print(f"{tag} skipped {skipped} existing chunks", flush=True)
		_run_pool(work, downsample_normal_pair_chunk_worker, workers=workers, tag=tag)
	set_pyramid_metadata(nx_g, method="normal_second_moment_mean_pool2x")
	set_pyramid_metadata(ny_g, method="normal_second_moment_mean_pool2x")
