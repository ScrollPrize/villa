from __future__ import annotations

import json
import multiprocessing
import os
import queue
import resource
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorstore as ts

try:
	from omezarr_pyramid import (
		_mean_pool2x_u8,
		_moment_pool2x_normals,
		clear_coarser_levels,
		print_progress,
		shape_div2,
	)
except ImportError:
	from .omezarr_pyramid import (
		_mean_pool2x_u8,
		_moment_pool2x_normals,
		clear_coarser_levels,
		print_progress,
		shape_div2,
	)


@dataclass(frozen=True)
class TensorStoreConfig:
	cache_pool_bytes: int = 4 << 30
	file_io_threads: int = 2
	data_copy_threads: int = 1
	workers: int = 0
	debug_chunks: bool = False

	def worker_count(self) -> int:
		if self.workers > 0:
			return int(self.workers)
		return max(1, multiprocessing.cpu_count())


@dataclass(frozen=True)
class LevelMeta:
	shape: tuple[int, int, int]
	chunks: tuple[int, int, int]
	dtype: np.dtype
	fill_value: int | float
	compressor_config: dict | None


def _rss_mib() -> float | None:
	try:
		usage = resource.getrusage(resource.RUSAGE_SELF)
	except Exception:
		return None
	rss = float(usage.ru_maxrss)
	if rss <= 0:
		return None
	if os.uname().sysname == "Darwin":
		return rss / float(1 << 20)
	return rss / 1024.0


def _debug_log(prefix: str, message: str) -> None:
	rss = _rss_mib()
	rss_text = "rss=unknown" if rss is None else f"rss={rss:.1f}MiB"
	print(f"{prefix} {message} {rss_text}", flush=True)


def numeric_levels(omezarr_path: str | Path) -> list[int]:
	root = Path(omezarr_path)
	return sorted(
		int(p.name)
		for p in root.iterdir()
		if p.is_dir() and p.name.isdigit() and (p / ".zarray").is_file()
	)


def infer_n_levels(omezarr_path: str | Path) -> int:
	levels = numeric_levels(omezarr_path)
	if not levels:
		raise ValueError(f"no numeric OME-Zarr levels found in {omezarr_path}")
	return max(levels) + 1


def level_meta(level_path: str | Path) -> LevelMeta:
	with (Path(level_path) / ".zarray").open() as f:
		meta = json.load(f)
	shape = tuple(int(v) for v in meta["shape"][-3:])
	chunks = tuple(int(v) for v in meta["chunks"][-3:])
	dtype = np.dtype(meta["dtype"])
	fill_value = meta.get("fill_value", 0)
	compressor_cfg = meta.get("compressor")
	return LevelMeta(
		shape=(shape[0], shape[1], shape[2]),
		chunks=(chunks[0], chunks[1], chunks[2]),
		dtype=dtype,
		fill_value=fill_value,
		compressor_config=compressor_cfg,
	)


def normalize_chunk(chunk_size: int | tuple[int, int, int]) -> tuple[int, int, int]:
	if isinstance(chunk_size, tuple):
		chunk = tuple(int(v) for v in chunk_size)
	else:
		c = int(chunk_size)
		chunk = (c, c, c)
	if len(chunk) != 3 or min(chunk) <= 0:
		raise ValueError(f"invalid chunk size: {chunk}")
	return chunk


def tensorstore_context(cfg: TensorStoreConfig) -> ts.Context:
	return ts.Context({
		"cache_pool": {"total_bytes_limit": int(cfg.cache_pool_bytes)},
		"file_io_concurrency": {"limit": int(cfg.file_io_threads)},
		"data_copy_concurrency": {"limit": int(cfg.data_copy_threads)},
	})


def open_tensorstore(path: str | Path, ctx: ts.Context, *, read: bool, write: bool):
	return ts.open(
		{
			"driver": "zarr",
			"kvstore": {"driver": "file", "path": str(path)},
		},
		context=ctx,
		open=True,
		read=read,
		write=write,
		recheck_cached_data="open",
	).result()


def _read_array(store, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
	return np.asarray(store[z0:z1, y0:y1, x0:x1].read().result())


def _debug_read_array(prefix: str, store, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
	_debug_log(prefix, f"READ start source[{z0}:{z1}, {y0}:{y1}, {x0}:{x1}]")
	t0 = time.time()
	future = store[z0:z1, y0:y1, x0:x1].read()
	_debug_log(prefix, "READ future created")
	data = future.result()
	_debug_log(prefix, f"READ result returned in {time.time() - t0:.3f}s")
	t1 = time.time()
	arr = np.asarray(data)
	_debug_log(
		prefix,
		f"np.asarray done in {time.time() - t1:.3f}s "
		f"shape={arr.shape} dtype={arr.dtype} nbytes={arr.nbytes / float(1 << 20):.1f}MiB",
	)
	return arr


def _write_array(store, z0: int, y0: int, x0: int, data: np.ndarray, *, fill_value=0) -> None:
	if data.size == 0:
		return
	if fill_value == 0 and not np.any(data):
		return
	z1 = int(z0) + int(data.shape[0])
	y1 = int(y0) + int(data.shape[1])
	x1 = int(x0) + int(data.shape[2])
	store[int(z0):z1, int(y0):y1, int(x0):x1].write(np.ascontiguousarray(data)).result()


def _write_array_by_chunks(
	store,
	*,
	z0: int,
	y0: int,
	x0: int,
	data: np.ndarray,
	chunk: tuple[int, int, int],
	fill_value=0,
) -> int:
	if data.ndim != 3:
		raise ValueError(f"expected 3D data, got shape={data.shape}")
	cz, cy, cx = chunk
	written = 0
	for lz0 in range(0, int(data.shape[0]), cz):
		lz1 = min(int(data.shape[0]), lz0 + cz)
		for ly0 in range(0, int(data.shape[1]), cy):
			ly1 = min(int(data.shape[1]), ly0 + cy)
			for lx0 in range(0, int(data.shape[2]), cx):
				lx1 = min(int(data.shape[2]), lx0 + cx)
				_write_array(
					store,
					int(z0) + lz0,
					int(y0) + ly0,
					int(x0) + lx0,
					data[lz0:lz1, ly0:ly1, lx0:lx1],
					fill_value=fill_value,
				)
				written += 1
	return written


def _iter_chunk_jobs_for_z_range(
	shape: tuple[int, int, int],
	chunk: tuple[int, int, int],
	z0_limit: int,
	z1_limit: int,
):
	z, y, x = shape
	cz, cy, cx = chunk
	for z0 in range(int(z0_limit), int(z1_limit), cz):
		z1 = min(z, z0 + cz)
		for y0 in range(0, y, cy):
			y1 = min(y, y0 + cy)
			for x0 in range(0, x, cx):
				x1 = min(x, x0 + cx)
				yield (z0, z1, y0, y1, x0, x1)


def _chunk_jobs_for_z_range(
	shape: tuple[int, int, int],
	chunk: tuple[int, int, int],
	z0_limit: int,
	z1_limit: int,
) -> list[tuple[int, int, int, int, int, int]]:
	return list(_iter_chunk_jobs_for_z_range(shape, chunk, z0_limit, z1_limit))


def _first_chunk_job(
	shape: tuple[int, int, int],
	chunk: tuple[int, int, int],
	z0_limit: int = 0,
) -> tuple[int, int, int, int, int, int] | None:
	z, y, x = shape
	if z <= 0 or y <= 0 or x <= 0:
		return None
	cz, cy, cx = chunk
	z0 = int(z0_limit)
	if z0 >= z:
		return None
	return (z0, min(z, z0 + cz), 0, min(y, cy), 0, min(x, cx))


def _chunk_count(shape: tuple[int, int, int], chunk: tuple[int, int, int]) -> int:
	z, y, x = shape
	cz, cy, cx = chunk
	return int(((z + cz - 1) // cz) * ((y + cy - 1) // cy) * ((x + cx - 1) // cx))


def _z_shards(
	shape: tuple[int, int, int],
	chunk: tuple[int, int, int],
	workers: int,
) -> list[dict]:
	z, y, x = shape
	cz, cy, cx = chunk
	z_starts = list(range(0, z, cz))
	if not z_starts:
		return []
	n_shards = min(max(1, int(workers)), len(z_starts))
	shards: list[dict] = []
	y_chunks = (y + cy - 1) // cy
	x_chunks = (x + cx - 1) // cx
	for si in range(n_shards):
		i0 = (si * len(z_starts)) // n_shards
		i1 = ((si + 1) * len(z_starts)) // n_shards
		if i0 >= i1:
			continue
		z0 = z_starts[i0]
		z1 = z if i1 >= len(z_starts) else z_starts[i1]
		shards.append({
			"z0": int(z0),
			"z1": int(z1),
			"jobs": int((i1 - i0) * y_chunks * x_chunks),
		})
	return shards


def _cfg_for_worker(cfg_values: dict) -> TensorStoreConfig:
	return TensorStoreConfig(
		cache_pool_bytes=int(cfg_values["cache_pool_bytes"]),
		file_io_threads=int(cfg_values["file_io_threads"]),
		data_copy_threads=int(cfg_values["data_copy_threads"]),
		workers=1,
	)


def _progress_emit(progress_q, n: int) -> None:
	if n > 0:
		progress_q.put(("progress", int(n)))


def _process_copy_shard(payload: dict, cfg_values: dict, progress_q) -> None:
	cfg = _cfg_for_worker(cfg_values)
	ctx = tensorstore_context(cfg)
	src_store = open_tensorstore(payload["src_path"], ctx, read=True, write=False)
	dst_store = open_tensorstore(payload["dst_path"], ctx, read=False, write=True)
	meta = level_meta(payload["dst_path"])
	pending = 0
	for job in _iter_chunk_jobs_for_z_range(meta.shape, meta.chunks, payload["z0"], payload["z1"]):
		z0, z1, y0, y1, x0, x1 = job
		data = _read_array(src_store, z0, z1, y0, y1, x0, x1).astype(np.uint8, copy=False)
		_write_array(dst_store, z0, y0, x0, data, fill_value=meta.fill_value)
		pending += 1
		if pending >= 16:
			_progress_emit(progress_q, pending)
			pending = 0
	_progress_emit(progress_q, pending)


def _process_scalar_scale_shard(payload: dict, cfg_values: dict, progress_q) -> None:
	cfg = _cfg_for_worker(cfg_values)
	ctx = tensorstore_context(cfg)
	src_store = open_tensorstore(payload["src_path"], ctx, read=True, write=False)
	dst_store = open_tensorstore(payload["dst_path"], ctx, read=False, write=True)
	src_meta = level_meta(payload["src_path"])
	dst_meta = level_meta(payload["dst_path"])
	zero_overrides = bool(payload["zero_overrides"])
	pending = 0
	for job in _iter_chunk_jobs_for_z_range(dst_meta.shape, dst_meta.chunks, payload["z0"], payload["z1"]):
		z0, z1, y0, y1, x0, x1 = job
		sz0, sy0, sx0 = z0 * 2, y0 * 2, x0 * 2
		sz1 = min(src_meta.shape[0], z1 * 2)
		sy1 = min(src_meta.shape[1], y1 * 2)
		sx1 = min(src_meta.shape[2], x1 * 2)
		slab = _read_array(src_store, sz0, sz1, sy0, sy1, sx0, sx1).astype(np.uint8, copy=False)
		down = _mean_pool2x_u8(slab, zero_overrides=zero_overrides)
		_write_array(dst_store, z0, y0, x0, down, fill_value=dst_meta.fill_value)
		pending += 1
		if pending >= 16:
			_progress_emit(progress_q, pending)
			pending = 0
	_progress_emit(progress_q, pending)


def _process_scalar_scale_source_chunk_shard(payload: dict, cfg_values: dict, progress_q) -> None:
	cfg = _cfg_for_worker(cfg_values)
	ctx = tensorstore_context(cfg)
	src_store = open_tensorstore(payload["src_path"], ctx, read=True, write=False)
	dst_store = open_tensorstore(payload["dst_path"], ctx, read=False, write=True)
	src_meta = level_meta(payload["src_path"])
	dst_meta = level_meta(payload["dst_path"])
	zero_overrides = bool(payload["zero_overrides"])
	job_chunk = tuple(int(v) for v in payload["job_chunk"])
	pending = 0
	for job in _iter_chunk_jobs_for_z_range(src_meta.shape, job_chunk, payload["z0"], payload["z1"]):
		z0, z1, y0, y1, x0, x1 = job
		if (
			z0 % (2 * dst_meta.chunks[0]) != 0
			or y0 % (2 * dst_meta.chunks[1]) != 0
			or x0 % (2 * dst_meta.chunks[2]) != 0
		):
			raise ValueError(
				f"source job start is not destination-chunk aligned: "
				f"job={(z0, y0, x0)} dst_chunks={dst_meta.chunks}"
			)
		slab = _read_array(src_store, z0, z1, y0, y1, x0, x1).astype(np.uint8, copy=False)
		down = _mean_pool2x_u8(slab, zero_overrides=zero_overrides)
		written = _write_array_by_chunks(
			dst_store,
			z0=z0 // 2,
			y0=y0 // 2,
			x0=x0 // 2,
			data=down,
			chunk=dst_meta.chunks,
			fill_value=dst_meta.fill_value,
		)
		pending += written
		if pending >= 1:
			_progress_emit(progress_q, pending)
			pending = 0
	_progress_emit(progress_q, pending)


def _process_normal_scale_shard(payload: dict, cfg_values: dict, progress_q) -> None:
	cfg = _cfg_for_worker(cfg_values)
	ctx = tensorstore_context(cfg)
	nx_src = open_tensorstore(payload["nx_src_path"], ctx, read=True, write=False)
	ny_src = open_tensorstore(payload["ny_src_path"], ctx, read=True, write=False)
	nx_dst = open_tensorstore(payload["nx_dst_path"], ctx, read=False, write=True)
	ny_dst = open_tensorstore(payload["ny_dst_path"], ctx, read=False, write=True)
	src_meta = level_meta(payload["nx_src_path"])
	dst_meta = level_meta(payload["nx_dst_path"])
	pending = 0
	for job in _iter_chunk_jobs_for_z_range(dst_meta.shape, dst_meta.chunks, payload["z0"], payload["z1"]):
		z0, z1, y0, y1, x0, x1 = job
		sz0, sy0, sx0 = z0 * 2, y0 * 2, x0 * 2
		sz1 = min(src_meta.shape[0], z1 * 2)
		sy1 = min(src_meta.shape[1], y1 * 2)
		sx1 = min(src_meta.shape[2], x1 * 2)
		nx_slab = _read_array(nx_src, sz0, sz1, sy0, sy1, sx0, sx1).astype(np.uint8, copy=False)
		ny_slab = _read_array(ny_src, sz0, sz1, sy0, sy1, sx0, sx1).astype(np.uint8, copy=False)
		nx_down, ny_down = _moment_pool2x_normals(nx_slab, ny_slab)
		_write_array(nx_dst, z0, y0, x0, nx_down, fill_value=dst_meta.fill_value)
		_write_array(ny_dst, z0, y0, x0, ny_down, fill_value=dst_meta.fill_value)
		pending += 1
		if pending >= 16:
			_progress_emit(progress_q, pending)
			pending = 0
	_progress_emit(progress_q, pending)


def _process_entry(kind: str, payload: dict, cfg_values: dict, progress_q) -> None:
	try:
		if kind == "copy":
			_process_copy_shard(payload, cfg_values, progress_q)
		elif kind == "scalar_scale":
			_process_scalar_scale_shard(payload, cfg_values, progress_q)
		elif kind == "scalar_scale_source_chunk":
			_process_scalar_scale_source_chunk_shard(payload, cfg_values, progress_q)
		elif kind == "normal_scale":
			_process_normal_scale_shard(payload, cfg_values, progress_q)
		else:
			raise ValueError(f"unknown shard kind: {kind}")
	except BaseException as exc:
		progress_q.put(("error", f"{type(exc).__name__}: {exc}"))
		raise


def _debug_process_one_chunk(
	*,
	kind: str,
	base_payload: dict,
	shape: tuple[int, int, int],
	chunk: tuple[int, int, int],
	cfg: TensorStoreConfig,
	tag: str,
) -> None:
	job = _first_chunk_job(shape, chunk, 0)
	if job is None:
		print(f"{tag} debug: no jobs", flush=True)
		return
	print(
		f"{tag} debug: processing one chunk without writing\n"
		f"  kind={kind} shape={shape} job_chunk={chunk} job={job}",
		flush=True,
	)
	if kind not in {"scalar_scale_source_chunk", "scalar_scale"}:
		raise ValueError(f"--debug-chunks currently supports scalar scale jobs, got kind={kind}")

	_debug_log(
		f"{tag} debug:",
		f"using TensorStore config cache={cfg.cache_pool_bytes / float(1 << 20):.1f}MiB "
		f"file_io_threads={cfg.file_io_threads} data_copy_threads={cfg.data_copy_threads}",
	)
	_debug_log(f"{tag} debug:", "creating TensorStore context")
	ctx = tensorstore_context(cfg)
	_debug_log(f"{tag} debug:", "opening source TensorStore")
	src_store = open_tensorstore(base_payload["src_path"], ctx, read=True, write=False)
	_debug_log(f"{tag} debug:", "source TensorStore opened")
	src_meta = level_meta(base_payload["src_path"])
	dst_meta = level_meta(base_payload["dst_path"])
	_debug_log(
		f"{tag} debug:",
		f"metadata loaded src_shape={src_meta.shape} src_chunks={src_meta.chunks} "
		f"dst_shape={dst_meta.shape} dst_chunks={dst_meta.chunks}",
	)
	zero_overrides = bool(base_payload["zero_overrides"])

	if kind == "scalar_scale_source_chunk":
		z0, z1, y0, y1, x0, x1 = job
		dst_z0, dst_y0, dst_x0 = z0 // 2, y0 // 2, x0 // 2
	else:
		z0, z1, y0, y1, x0, x1 = job
		dst_z0, dst_y0, dst_x0 = z0, y0, x0
		z0, y0, x0 = z0 * 2, y0 * 2, x0 * 2
		z1 = min(src_meta.shape[0], z1 * 2)
		y1 = min(src_meta.shape[1], y1 * 2)
		x1 = min(src_meta.shape[2], x1 * 2)

	t0 = time.time()
	slab_raw = _debug_read_array(f"{tag} debug:", src_store, z0, z1, y0, y1, x0, x1)
	_debug_log(f"{tag} debug:", "astype uint8 start")
	slab = slab_raw.astype(np.uint8, copy=False)
	read_s = time.time() - t0
	_debug_log(f"{tag} debug:", "astype uint8 done")
	t1 = time.time()
	_debug_log(f"{tag} debug:", "mean_pool2x start")
	down = _mean_pool2x_u8(slab, zero_overrides=zero_overrides)
	pool_s = time.time() - t1
	_debug_log(f"{tag} debug:", "mean_pool2x done")
	dst_chunks = _chunk_count(down.shape, dst_meta.chunks)
	print(
		f"{tag} debug: read source[{z0}:{z1}, {y0}:{y1}, {x0}:{x1}] "
		f"shape={slab.shape} dtype={slab.dtype} nbytes={slab.nbytes / float(1 << 20):.1f}MiB "
		f"time={read_s:.3f}s\n"
		f"  pooled shape={down.shape} dtype={down.dtype} "
		f"nbytes={down.nbytes / float(1 << 20):.1f}MiB time={pool_s:.3f}s "
		f"zero_overrides={zero_overrides}\n"
		f"  would write dst[{dst_z0}:{dst_z0 + down.shape[0]}, "
		f"{dst_y0}:{dst_y0 + down.shape[1]}, {dst_x0}:{dst_x0 + down.shape[2]}] "
		f"as {dst_chunks} chunks of {dst_meta.chunks}; skipped all writes",
		flush=True,
	)


def _run_process_shards(
	*,
	kind: str,
	base_payload: dict,
	shape: tuple[int, int, int],
	chunk: tuple[int, int, int],
	cfg: TensorStoreConfig,
	tag: str,
	progress_total: int | None = None,
) -> None:
	shards = _z_shards(shape, chunk, cfg.worker_count())
	total = sum(int(s["jobs"]) for s in shards) if progress_total is None else int(progress_total)
	if total == 0:
		return

	n_procs = len(shards)
	per_proc_cache = max(64 << 20, int(cfg.cache_pool_bytes) // max(1, n_procs))
	print(
		f"{tag} jobs={total} workers={n_procs} chunk={chunk} "
		f"cache_per_worker={per_proc_cache / float(1 << 20):.1f}MiB "
		f"file_io_threads={int(cfg.file_io_threads)} data_copy_threads={int(cfg.data_copy_threads)}",
		flush=True,
	)
	if cfg.debug_chunks:
		_debug_process_one_chunk(
			kind=kind,
			base_payload=base_payload,
			shape=shape,
			chunk=chunk,
			cfg=cfg,
			tag=tag,
		)
		raise SystemExit(0)
	cfg_values = {
		"cache_pool_bytes": per_proc_cache,
		"file_io_threads": int(cfg.file_io_threads),
		"data_copy_threads": int(cfg.data_copy_threads),
	}
	mp_ctx = multiprocessing.get_context("spawn")
	progress_q = mp_ctx.Queue()
	procs: list[multiprocessing.Process] = []
	for shard in shards:
		payload = dict(base_payload)
		payload.update(shard)
		proc = mp_ctx.Process(target=_process_entry, args=(kind, payload, cfg_values, progress_q))
		proc.start()
		procs.append(proc)

	t0 = time.time()
	done = 0
	last_print_done = -1
	try:
		while procs:
			try:
				msg, value = progress_q.get(timeout=0.5)
			except queue.Empty:
				if done < total and done != last_print_done:
					print_progress(prefix=tag, done=done, total=total, t0=t0)
					last_print_done = done
			else:
				if msg == "progress":
					done += int(value)
					if done < total:
						print_progress(prefix=tag, done=done, total=total, t0=t0)
						last_print_done = done
				elif msg == "error":
					raise RuntimeError(str(value))
			alive: list[multiprocessing.Process] = []
			for proc in procs:
				if proc.is_alive():
					alive.append(proc)
				else:
					proc.join()
					if proc.exitcode != 0:
						raise RuntimeError(f"worker process failed with exit code {proc.exitcode}")
			procs = alive
		while True:
			try:
				msg, value = progress_q.get_nowait()
			except queue.Empty:
				break
			if msg == "progress":
				done += int(value)
			elif msg == "error":
				raise RuntimeError(str(value))
	finally:
		for proc in procs:
			if proc.is_alive():
				proc.terminate()
		for proc in procs:
			proc.join(timeout=5)
	print_progress(prefix=tag, done=total, total=total, t0=t0)
	print("", flush=True)


def _aligned_scalar_scale_job_chunk(
	*,
	src_chunk: tuple[int, int, int],
	dst_chunk: tuple[int, int, int],
) -> tuple[int, int, int]:
	"""Return the aligned source-level read chunk for scalar 2x downsampling."""
	out: list[int] = []
	for axis, sc, dc in zip("zyx", src_chunk, dst_chunk):
		sc = int(sc)
		dc = int(dc)
		if sc <= 0 or dc <= 0:
			raise ValueError(f"invalid {axis} chunk sizes: src={sc} dst={dc}")
		dst_span_in_src = 2 * dc
		job = max(sc, dst_span_in_src)
		if job % sc != 0:
			raise ValueError(
				f"{axis} chunks are not aligned: read_chunk={job} "
				f"src_chunk={sc} dst_chunk_in_src={dst_span_in_src}"
			)
		if job % dst_span_in_src != 0:
			raise ValueError(
				f"{axis} chunks are not target-aligned: read_chunk={job} "
				f"src_chunk={sc} dst_chunk_in_src={dst_span_in_src}"
			)
		out.append(job)
	return (out[0], out[1], out[2])


def _read_json(path: Path, default):
	if not path.is_file():
		return default
	with path.open() as f:
		return json.load(f)


def _write_json(path: Path, data) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
	with tmp.open("w") as f:
		json.dump(data, f, indent=2)
		f.write("\n")
	tmp.replace(path)


def _read_attrs(root: Path) -> dict:
	return dict(_read_json(root / ".zattrs", {}))


def _write_attrs(root: Path, attrs: dict) -> None:
	_write_json(root / ".zattrs", attrs)


def _copy_attrs(src: Path, dst: Path) -> dict:
	attrs = _read_attrs(src)
	_write_attrs(dst, attrs)
	return attrs


def _json_fill_value(value):
	if isinstance(value, np.generic):
		return value.item()
	return value


def _create_zarr_group(root: Path) -> None:
	root.mkdir(parents=True, exist_ok=True)
	_write_json(root / ".zgroup", {"zarr_format": 2})


def _create_zarr_array(
	level_path: Path,
	*,
	shape: tuple[int, int, int],
	chunks: tuple[int, int, int],
	dtype: np.dtype,
	fill_value,
	compressor_config: dict | None,
) -> None:
	level_path.mkdir(parents=True, exist_ok=True)
	meta = {
		"zarr_format": 2,
		"shape": [int(v) for v in shape],
		"chunks": [int(v) for v in chunks],
		"dtype": np.dtype(dtype).str,
		"compressor": compressor_config,
		"fill_value": _json_fill_value(fill_value),
		"order": "C",
		"filters": None,
		"dimension_separator": "/",
	}
	_write_json(level_path / ".zarray", meta)
	_write_json(level_path / ".zattrs", {})


def _update_multiscales(
	root: Path,
	*,
	data_level: int,
	n_levels: int,
	source_attrs: dict | None,
	name: str,
) -> None:
	axes = [
		{"name": "z", "type": "space", "unit": "pixel"},
		{"name": "y", "type": "space", "unit": "pixel"},
		{"name": "x", "type": "space", "unit": "pixel"},
	]
	if source_attrs:
		ms = source_attrs.get("multiscales")
		if isinstance(ms, list) and ms:
			axes = ms[0].get("axes", axes)
			name = str(ms[0].get("name", name))
	datasets = []
	for lv in range(int(data_level), int(n_levels)):
		datasets.append({
			"path": str(lv),
			"coordinateTransformations": [{
				"type": "scale",
				"scale": [float(2 ** lv)] * 3,
			}],
		})
	attrs = _read_attrs(root)
	attrs["multiscales"] = [{
		"version": "0.4",
		"name": name,
		"axes": axes,
		"datasets": datasets,
	}]
	_write_attrs(root, attrs)


def _set_downsample_method(root: str | Path, method: str) -> None:
	root = Path(root)
	attrs = _read_attrs(root)
	attrs["lasagna_pyramid_downsample"] = method
	_write_attrs(root, attrs)


def create_omezarr_like(
	*,
	src_root: str | Path,
	dst_root: str | Path,
	data_level: int,
	n_levels: int,
	chunk: tuple[int, int, int],
	name: str,
	overwrite: bool,
) -> None:
	src_root = Path(src_root)
	dst_root = Path(dst_root)
	if dst_root.exists():
		if not overwrite:
			raise FileExistsError(f"output already exists: {dst_root}")
		shutil.rmtree(dst_root)
	dst_root.parent.mkdir(parents=True, exist_ok=True)

	src_meta = level_meta(src_root / str(data_level))
	if src_meta.dtype != np.dtype("uint8"):
		raise ValueError(f"only uint8 OME-Zarr arrays are supported, got dtype={src_meta.dtype}")
	_create_zarr_group(dst_root)
	for lv in range(int(data_level), int(n_levels)):
		level_shape = shape_div2(src_meta.shape, lv - int(data_level))
		level_chunk = tuple(min(level_shape[i], chunk[i]) for i in range(3))
		_create_zarr_array(
			dst_root / str(lv),
			shape=level_shape,
			chunks=level_chunk,
			dtype=src_meta.dtype,
			fill_value=src_meta.fill_value,
			compressor_config=src_meta.compressor_config,
		)
	src_attrs = _copy_attrs(src_root, dst_root)
	_update_multiscales(
		dst_root,
		data_level=data_level,
		n_levels=n_levels,
		source_attrs=src_attrs,
		name=name,
	)


def copy_data_level(
	*,
	src_root: str | Path,
	dst_root: str | Path,
	data_level: int,
	cfg: TensorStoreConfig,
	label: str,
) -> None:
	dst_path = Path(dst_root) / str(data_level)
	meta = level_meta(dst_path)
	_run_process_shards(
		kind="copy",
		base_payload={
			"src_path": str(Path(src_root) / str(data_level)),
			"dst_path": str(dst_path),
		},
		shape=meta.shape,
		chunk=meta.chunks,
		cfg=cfg,
		tag=f"[rechunk {label} L{data_level}]",
	)


def rebuild_scalar_scales(
	*,
	root: str | Path,
	data_level: int,
	n_levels: int,
	cfg: TensorStoreConfig,
	label: str,
	zero_overrides: bool,
	clear_existing: bool,
) -> None:
	root = Path(root)
	if clear_existing and cfg.debug_chunks:
		print("[debug-chunks] skipping clear_coarser_levels", flush=True)
	elif clear_existing:
		clear_coarser_levels(root, data_level, n_levels)
	for lv in range(int(data_level) + 1, int(n_levels)):
		src_path = root / str(lv - 1)
		dst_path = root / str(lv)
		if cfg.debug_chunks:
			_debug_log(
				f"[debug-chunks scale {label} L{lv}]",
				f"reading metadata src={src_path} dst={dst_path}",
			)
		src_meta = level_meta(src_path)
		dst_meta = level_meta(dst_path)
		if cfg.debug_chunks:
			_debug_log(
				f"[debug-chunks scale {label} L{lv}]",
				f"metadata src_shape={src_meta.shape} src_chunks={src_meta.chunks} "
				f"dst_shape={dst_meta.shape} dst_chunks={dst_meta.chunks} "
				f"src_dtype={src_meta.dtype} dst_dtype={dst_meta.dtype}",
			)
		expected_dst_shape = shape_div2(src_meta.shape, 1)
		if dst_meta.shape != expected_dst_shape:
			raise ValueError(
				f"scalar scale shape mismatch at L{lv}: "
				f"src_shape={src_meta.shape} expected_dst_shape={expected_dst_shape} "
				f"dst_shape={dst_meta.shape}"
			)
		source_job_chunk = _aligned_scalar_scale_job_chunk(
			src_chunk=src_meta.chunks,
			dst_chunk=dst_meta.chunks,
		)
		_run_process_shards(
			kind="scalar_scale_source_chunk",
			base_payload={
				"src_path": str(src_path),
				"dst_path": str(dst_path),
				"zero_overrides": bool(zero_overrides),
				"job_chunk": source_job_chunk,
			},
			shape=src_meta.shape,
			chunk=source_job_chunk,
			cfg=cfg,
			tag=f"[scale {label} L{lv} aligned]",
			progress_total=_chunk_count(dst_meta.shape, dst_meta.chunks),
		)
	_set_downsample_method(root, "mean_pool2x_zero_overrides" if zero_overrides else "mean_pool2x")


def rebuild_normal_scales(
	*,
	nx_root: str | Path,
	ny_root: str | Path,
	data_level: int,
	n_levels: int,
	cfg: TensorStoreConfig,
	clear_existing: bool,
) -> None:
	nx_root = Path(nx_root)
	ny_root = Path(ny_root)
	if clear_existing and cfg.debug_chunks:
		print("[debug-chunks] skipping clear_coarser_levels", flush=True)
	elif clear_existing:
		clear_coarser_levels(nx_root, data_level, n_levels)
		clear_coarser_levels(ny_root, data_level, n_levels)
	for lv in range(int(data_level) + 1, int(n_levels)):
		nx_src_path = nx_root / str(lv - 1)
		ny_src_path = ny_root / str(lv - 1)
		nx_dst_path = nx_root / str(lv)
		ny_dst_path = ny_root / str(lv)
		src_meta = level_meta(nx_src_path)
		dst_meta = level_meta(nx_dst_path)
		if level_meta(ny_src_path).shape != src_meta.shape:
			raise ValueError(f"normal source shape mismatch at L{lv - 1}")
		if level_meta(ny_dst_path).shape != dst_meta.shape:
			raise ValueError(f"normal destination shape mismatch at L{lv}")
		_run_process_shards(
			kind="normal_scale",
			base_payload={
				"nx_src_path": str(nx_src_path),
				"ny_src_path": str(ny_src_path),
				"nx_dst_path": str(nx_dst_path),
				"ny_dst_path": str(ny_dst_path),
			},
			shape=dst_meta.shape,
			chunk=dst_meta.chunks,
			cfg=cfg,
			tag=f"[scale normal L{lv}]",
		)
	_set_downsample_method(nx_root, "normal_second_moment_mean_pool2x")
	_set_downsample_method(ny_root, "normal_second_moment_mean_pool2x")


def rechunk_scalar_out_of_place(
	*,
	src_root: str | Path,
	dst_root: str | Path,
	chunk: tuple[int, int, int],
	data_level: int,
	n_levels: int | None,
	cfg: TensorStoreConfig,
	zero_overrides: bool,
	overwrite: bool,
	label: str,
) -> None:
	if n_levels is None:
		n_levels = infer_n_levels(src_root)
	create_omezarr_like(
		src_root=src_root,
		dst_root=dst_root,
		data_level=data_level,
		n_levels=n_levels,
		chunk=chunk,
		name=label,
		overwrite=overwrite,
	)
	copy_data_level(src_root=src_root, dst_root=dst_root, data_level=data_level, cfg=cfg, label=label)
	rebuild_scalar_scales(
		root=dst_root,
		data_level=data_level,
		n_levels=n_levels,
		cfg=cfg,
		label=label,
		zero_overrides=zero_overrides,
		clear_existing=False,
	)


def rechunk_normal_pair_out_of_place(
	*,
	nx_src: str | Path,
	ny_src: str | Path,
	nx_dst: str | Path,
	ny_dst: str | Path,
	chunk: tuple[int, int, int],
	data_level: int,
	n_levels: int | None,
	cfg: TensorStoreConfig,
	overwrite: bool,
) -> None:
	if n_levels is None:
		n_levels = max(infer_n_levels(nx_src), infer_n_levels(ny_src))
	if level_meta(Path(nx_src) / str(data_level)).shape != level_meta(Path(ny_src) / str(data_level)).shape:
		raise ValueError("normal data-level shape mismatch")
	create_omezarr_like(
		src_root=nx_src,
		dst_root=nx_dst,
		data_level=data_level,
		n_levels=n_levels,
		chunk=chunk,
		name="nx",
		overwrite=overwrite,
	)
	create_omezarr_like(
		src_root=ny_src,
		dst_root=ny_dst,
		data_level=data_level,
		n_levels=n_levels,
		chunk=chunk,
		name="ny",
		overwrite=overwrite,
	)
	copy_data_level(src_root=nx_src, dst_root=nx_dst, data_level=data_level, cfg=cfg, label="nx")
	copy_data_level(src_root=ny_src, dst_root=ny_dst, data_level=data_level, cfg=cfg, label="ny")
	rebuild_normal_scales(
		nx_root=nx_dst,
		ny_root=ny_dst,
		data_level=data_level,
		n_levels=n_levels,
		cfg=cfg,
		clear_existing=False,
	)


def rebuild_scalar_in_place(
	*,
	root: str | Path,
	data_level: int,
	n_levels: int | None,
	cfg: TensorStoreConfig,
	zero_overrides: bool,
	label: str,
) -> None:
	if n_levels is None:
		if cfg.debug_chunks:
			_debug_log(f"[debug-chunks rebuild {label}]", f"inferring numeric levels from {root}")
		n_levels = infer_n_levels(root)
		if cfg.debug_chunks:
			_debug_log(f"[debug-chunks rebuild {label}]", f"inferred n_levels={n_levels}")
	rebuild_scalar_scales(
		root=root,
		data_level=data_level,
		n_levels=n_levels,
		cfg=cfg,
		label=label,
		zero_overrides=zero_overrides,
		clear_existing=True,
	)


def rebuild_normal_pair_in_place(
	*,
	nx_root: str | Path,
	ny_root: str | Path,
	data_level: int,
	n_levels: int | None,
	cfg: TensorStoreConfig,
) -> None:
	if n_levels is None:
		n_levels = max(infer_n_levels(nx_root), infer_n_levels(ny_root))
	rebuild_normal_scales(
		nx_root=nx_root,
		ny_root=ny_root,
		data_level=data_level,
		n_levels=n_levels,
		cfg=cfg,
		clear_existing=True,
	)


def debug_rechunk_scalar_one_chunk(
	*,
	src_root: str | Path,
	chunk: tuple[int, int, int],
	data_level: int,
	cfg: TensorStoreConfig,
	label: str,
) -> None:
	src_path = Path(src_root) / str(data_level)
	_debug_log(f"[debug-chunks rechunk {label} L{data_level}]", f"reading source metadata {src_path}")
	src_meta = level_meta(src_path)
	_debug_log(
		f"[debug-chunks rechunk {label} L{data_level}]",
		f"metadata loaded src_shape={src_meta.shape} src_chunks={src_meta.chunks} dtype={src_meta.dtype}",
	)
	read_chunk = tuple(min(src_meta.shape[i], int(chunk[i])) for i in range(3))
	job = _first_chunk_job(src_meta.shape, read_chunk, 0)
	if job is None:
		print(f"[debug-chunks rechunk {label} L{data_level}] no jobs", flush=True)
		return
	prefix = f"[debug-chunks rechunk {label} L{data_level}]"
	print(
		f"{prefix} processing one chunk without writing\n"
		f"  src_path={src_path}\n"
		f"  src_shape={src_meta.shape} src_chunks={src_meta.chunks} dtype={src_meta.dtype} "
		f"compressor={src_meta.compressor_config}\n"
		f"  requested_output_chunk={chunk} read_chunk={read_chunk} job={job}\n"
		f"  cache={cfg.cache_pool_bytes / float(1 << 20):.1f}MiB "
		f"file_io_threads={cfg.file_io_threads} data_copy_threads={cfg.data_copy_threads}",
		flush=True,
	)
	_debug_log(prefix, "creating TensorStore context")
	ctx = tensorstore_context(cfg)
	_debug_log(prefix, "opening source TensorStore")
	src_store = open_tensorstore(src_path, ctx, read=True, write=False)
	_debug_log(prefix, "source TensorStore opened")
	z0, z1, y0, y1, x0, x1 = job
	t0 = time.time()
	data_raw = _debug_read_array(prefix, src_store, z0, z1, y0, y1, x0, x1)
	_debug_log(prefix, "astype uint8 start")
	data = data_raw.astype(np.uint8, copy=False)
	read_s = time.time() - t0
	_debug_log(prefix, "astype uint8 done")
	print(
		f"{prefix} read "
		f"source[{z0}:{z1}, {y0}:{y1}, {x0}:{x1}] "
		f"shape={data.shape} dtype={data.dtype} "
		f"nbytes={data.nbytes / float(1 << 20):.1f}MiB time={read_s:.3f}s; "
		f"skipped all writes",
		flush=True,
	)


def _swap_dir(tmp: Path, target: Path) -> None:
	backup = target.with_name(f"{target.name}.bak-{os.getpid()}")
	if backup.exists():
		raise FileExistsError(f"backup path already exists: {backup}")
	if not tmp.exists():
		raise FileNotFoundError(f"temporary output does not exist: {tmp}")
	target.rename(backup)
	try:
		tmp.rename(target)
	except Exception:
		backup.rename(target)
		raise
	shutil.rmtree(backup)


def rechunk_scalar_in_place(
	*,
	root: str | Path,
	chunk: tuple[int, int, int],
	data_level: int,
	n_levels: int | None,
	cfg: TensorStoreConfig,
	zero_overrides: bool,
	label: str,
) -> None:
	root = Path(root)
	tmp = root.with_name(f"{root.name}.tmp-rechunk-{os.getpid()}")
	if tmp.exists():
		shutil.rmtree(tmp)
	rechunk_scalar_out_of_place(
		src_root=root,
		dst_root=tmp,
		chunk=chunk,
		data_level=data_level,
		n_levels=n_levels,
		cfg=cfg,
		zero_overrides=zero_overrides,
		overwrite=True,
		label=label,
	)
	_swap_dir(tmp, root)


def rechunk_normal_pair_in_place(
	*,
	nx_root: str | Path,
	ny_root: str | Path,
	chunk: tuple[int, int, int],
	data_level: int,
	n_levels: int | None,
	cfg: TensorStoreConfig,
) -> None:
	nx_root = Path(nx_root)
	ny_root = Path(ny_root)
	nx_tmp = nx_root.with_name(f"{nx_root.name}.tmp-rechunk-{os.getpid()}")
	ny_tmp = ny_root.with_name(f"{ny_root.name}.tmp-rechunk-{os.getpid()}")
	for tmp in (nx_tmp, ny_tmp):
		if tmp.exists():
			shutil.rmtree(tmp)
	rechunk_normal_pair_out_of_place(
		nx_src=nx_root,
		ny_src=ny_root,
		nx_dst=nx_tmp,
		ny_dst=ny_tmp,
		chunk=chunk,
		data_level=data_level,
		n_levels=n_levels,
		cfg=cfg,
		overwrite=True,
	)
	_swap_dir(nx_tmp, nx_root)
	_swap_dir(ny_tmp, ny_root)
