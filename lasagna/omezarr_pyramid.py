from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
import multiprocessing
import os
import threading
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from vc_reader.reader import ZarrWriter, create_group, open_zarr, write_pyramid_level

from vesuvius.neural_tracing.normals import decode_normals as _decode_normals

_NATIVE_THREAD_ENV = (
	"OPENBLAS_NUM_THREADS",
	"OMP_NUM_THREADS",
	"MKL_NUM_THREADS",
	"BLIS_NUM_THREADS",
	"VECLIB_MAXIMUM_THREADS",
	"NUMEXPR_NUM_THREADS",
	"BLOSC_NTHREADS",
)


def _set_native_thread_env_one() -> None:
	for name in _NATIVE_THREAD_ENV:
		os.environ[name] = "1"


@contextmanager
def _single_threaded_native_runtime():
	"""Temporarily constrain parent and newly spawned native runtimes."""
	missing = object()
	previous = {name: os.environ.get(name, missing) for name in _NATIVE_THREAD_ENV}
	_set_native_thread_env_one()
	try:
		with threadpool_limits(limits=1):
			yield
	finally:
		for name, value in previous.items():
			if value is missing:
				os.environ.pop(name, None)
			else:
				os.environ[name] = value


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


def set_pyramid_metadata(group, *, method: str) -> None:
	group.attrs["lasagna_pyramid_downsample"] = method


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


def _moment_pool2x_normals(nx_u8: np.ndarray, ny_u8: np.ndarray, *, validity: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
	if nx_u8.shape != ny_u8.shape:
		raise ValueError(f"normal channel shape mismatch: nx={nx_u8.shape} ny={ny_u8.shape}")
	nx, ny, nz = _decode_normals(nx_u8, ny_u8)
	if validity is not None:
		if validity.shape != nx.shape:
			raise ValueError("normal validity shape mismatch")
		valid = validity != 0
		nx, ny, nz = (v * valid for v in (nx, ny, nz))
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
	"""Remove a level's chunks, keeping its metadata and attributes."""
	level_path = Path(level_path)
	if not level_path.is_dir():
		return
	level = ZarrWriter.open(level_path)
	attrs = dict(level.attrs)
	cleared = ZarrWriter.create_like(level_path, level, overwrite=True)
	if attrs:
		cleared.attrs.update(attrs)


def clear_coarser_levels(omezarr_path: str | Path, data_level: int, n_levels: int) -> None:
	for lv in range(int(data_level) + 1, int(n_levels)):
		clear_level_chunks(Path(omezarr_path) / str(lv))


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


def _crop_chunk_range(
	dst_shape: tuple[int, int, int],
	dst_chunks: tuple[int, int, int],
	crop_zyx: tuple[int, int, int, int, int, int] | None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
	"""The output chunks ``[begin, end)`` under a crop given in source-level voxels."""
	grid = tuple((n + c - 1) // c for n, c in zip(dst_shape, dst_chunks))
	if crop_zyx is None:
		return (0, 0, 0), grid
	begin = tuple(max(0, int(a)) // (2 * c) for a, c in zip(crop_zyx[:3], dst_chunks))
	end = tuple(min(n, (int(b) + 2 * c - 1) // (2 * c)) for b, c, n in zip(crop_zyx[3:], dst_chunks, grid))
	return begin, end


def _stored_output_chunks(source_path: str | Path, dst_chunks) -> set[tuple[int, int, int]]:
	"""The output chunks over any chunk stored in the source level."""
	src_chunks = open_zarr(source_path).chunks
	return {tuple(i * s // 2 // d for i, s, d in zip(index, src_chunks, dst_chunks)) for index in ZarrWriter.open(source_path).stored_chunks()}


def _pool_chunks(
	source_paths,
	dst_shape,
	dst_chunks,
	crop_zyx,
	require_source_chunks: bool,
) -> list[tuple[int, int, int]]:
	"""Output chunks to consider, in chunk order. With ``require_source_chunks``, every source must store a chunk under each."""
	begin, end = _crop_chunk_range(dst_shape, dst_chunks, crop_zyx)
	if require_source_chunks:
		stored = None
		for path in source_paths:
			found = _stored_output_chunks(path, dst_chunks)
			stored = found if stored is None else stored & found
		return sorted(c for c in stored if all(b <= i < e for b, i, e in zip(begin, c, end)))
	return [
		(z, y, x)
		for z in range(begin[0], end[0])
		for y in range(begin[1], end[1])
		for x in range(begin[2], end[2])
	]


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


_PROCESS_WORKER = None


def _init_process_worker(worker) -> None:
	global _PROCESS_WORKER
	_PROCESS_WORKER = worker


def _call_process_worker(item):
	return _PROCESS_WORKER(item)


def _run_pool(work, worker, *, workers: int, tag: str, total: int | None = None, processes: bool = False) -> None:
	"""``worker`` over ``work``, on threads, or with ``processes`` on spawned processes for work that keeps the GIL
	(``worker`` then is picklable).
	"""
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

	prog_thread = threading.Thread(
		target=_prog, daemon=True, name=f"pyramid-progress:{tag}"
	)
	prog_thread.start()
	complete = False
	pool_size = min(max(1, int(workers)), n_work)
	print(f"\n{tag} workers={pool_size}", flush=True)
	try:
		with _single_threaded_native_runtime():
			if processes and pool_size > 1:
				pool = ProcessPoolExecutor(
					max_workers=pool_size, mp_context=multiprocessing.get_context("spawn"),
					initializer=_init_process_worker, initargs=(worker,),
				)
				statuses = lambda: pool.map(_call_process_worker, work)
			else:
				pool = ThreadPoolExecutor(max_workers=pool_size)
				statuses = lambda: pool.map(worker, work)
			with pool:
				for status in statuses():
					with lock:
						done_count[0] += 1
						if status:
							status_counts[str(status)] = status_counts.get(str(status), 0) + 1
		complete = True
	finally:
		stop.set()
		prog_thread.join(timeout=2)
		if complete:
			print_progress(prefix=tag, done=n_work, total=n_work, t0=t0, suffix=_pyramid_status_suffix(status_counts))
		print("", flush=True)


def invalidate_coarser_chunks(level_writers: dict[int, ZarrWriter], level: int, index: tuple[int, int, int]) -> None:
	"""Coarser chunks made from this chunk are out of date once it changes: an interrupted build recomputes them."""
	for lv in sorted(lv for lv in level_writers if lv > level):
		index = tuple(i // 2 for i in index)
		level_writers[lv].remove_chunk(index)


class _PoolChunk:
	"""Writes one output chunk of each root's level from the level below. Picklable."""

	def __init__(self, roots, src_level: int, dst_level: int, n_levels: int, reduce, skip_existing: bool):
		self.sources = [open_zarr(Path(root) / str(src_level), threads=1) for root in roots]
		self.targets = [ZarrWriter.open(Path(root) / str(dst_level), skip_empty=True) for root in roots]
		self.coarser = [
			{lv: ZarrWriter.open(Path(root) / str(lv)) for lv in range(dst_level + 1, n_levels) if (Path(root) / str(lv)).is_dir()}
			for root in roots
		]
		self.dst_level = dst_level
		self.reduce = reduce
		self.skip_existing = skip_existing

	def __call__(self, index) -> str:
		if self.skip_existing and all(target.chunk_exists(index) for target in self.targets):
			return "skipped_existing"
		chunks, shape = self.targets[0].chunks, self.targets[0].shape
		start = [i * c * 2 for i, c in zip(index, chunks)]
		size = [min(c * 2, n - a) for c, n, a in zip(chunks, self.sources[0].shape, start)]
		if min(size) <= 0:
			return "skipped_empty_slice"
		blocks = self.reduce(*(source.read_region(start, size) for source in self.sources))
		origin = [i * c for i, c in zip(index, chunks)]
		for target, levels, block in zip(self.targets, self.coarser, blocks):
			invalidate_coarser_chunks(levels, self.dst_level, index)
			region = tuple(slice(a, min(n, a + b)) for a, n, b in zip(origin, shape, block.shape))
			target[region] = block[tuple(slice(0, r.stop - r.start) for r in region)]
		return "written"


def _pool_level(
	roots,
	src_level: int,
	dst_level: int,
	n_levels: int,
	reduce,
	*,
	crop_zyx,
	skip_existing: bool,
	require_source_chunks: bool,
	workers: int,
	tag: str,
	processes: bool = False,
) -> None:
	"""Write ``dst_level`` of each root from the level below, chunk by chunk, with an application reduction:
	``reduce(*source_blocks)`` returns one block per root.
	"""
	worker = _PoolChunk(roots, src_level, dst_level, n_levels, reduce, skip_existing)
	work = _pool_chunks(
		[Path(root) / str(src_level) for root in roots], worker.targets[0].shape, worker.targets[0].chunks,
		crop_zyx, require_source_chunks,
	)
	_run_pool(work, worker, workers=workers, tag=tag, processes=processes)


def _reduce_zero_overrides(slab: np.ndarray) -> tuple[np.ndarray]:
	return (_mean_pool2x_u8(np.asarray(slab, dtype=np.uint8), zero_overrides=True),)


def _reduce_normals(nx: np.ndarray, ny: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	return _moment_pool2x_normals(np.asarray(nx, dtype=np.uint8), np.asarray(ny, dtype=np.uint8))


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
	"""Coarser levels as 2x means. Plain means are rewritten over the crop (or the whole grid with
	``scan_existing_source_chunks``); ``zero_overrides`` keeps existing chunks unless ``force``.
	``chunk`` is unused: each level's own chunks decide.
	"""
	root = Path(omezarr_path)
	if workers <= 0:
		workers = max(1, multiprocessing.cpu_count())
	if force:
		clear_coarser_levels(root, data_level, n_levels)
	for lv in range(int(data_level) + 1, int(n_levels)):
		src_lv = lv - 1
		src_crop = None if scan_existing_source_chunks else _scaled_crop_for_source_level(crop_zyx, src_lv - int(data_level))
		tag = f"[pyramid {label} L{lv}]" if label else f"[pyramid L{lv}]"
		if zero_overrides:
			_pool_level(
				(root,), src_lv, lv, int(n_levels),
				_reduce_zero_overrides,
				crop_zyx=src_crop,
				skip_existing=not force,
				require_source_chunks=scan_existing_source_chunks,
				workers=workers,
				tag=tag,
			)
		else:
			print(f"{tag} workers={workers}", flush=True)
			level = open_zarr(root / str(lv))
			write_pyramid_level(root, lv, threads=workers, chunk_range=_crop_chunk_range(level.shape, level.chunks, src_crop))
	set_pyramid_metadata(create_group(root), method="mean_pool2x_zero_overrides" if zero_overrides else "mean_pool2x")


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
	"""Coarser levels of a normal pair by the second-moment reduction. Existing chunks stay unless ``force``.
	``chunk`` is unused: each level's own chunks decide.
	"""
	roots = (Path(nx_omezarr_path), Path(ny_omezarr_path))
	if workers <= 0:
		workers = max(1, multiprocessing.cpu_count())
	if force:
		for root in roots:
			clear_coarser_levels(root, data_level, n_levels)

	for lv in range(int(data_level) + 1, int(n_levels)):
		src_lv = lv - 1
		_pool_level(
			roots, src_lv, lv, int(n_levels), _reduce_normals,
			crop_zyx=None if scan_existing_source_chunks else _scaled_crop_for_source_level(crop_zyx, src_lv - int(data_level)),
			skip_existing=not force,
			require_source_chunks=scan_existing_source_chunks,
			workers=workers,
			tag=f"[pyramid {label} L{lv}]",
			processes=True,  # numpy's eigh keeps the GIL
		)
	for root in roots:
		set_pyramid_metadata(create_group(root), method="normal_second_moment_mean_pool2x")
