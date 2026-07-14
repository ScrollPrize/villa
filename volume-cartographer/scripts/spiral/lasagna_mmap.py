"""Persistent disk-backed Lasagna fields with bounded parallel sparse gathers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time

import numpy as np


CACHE_SCHEMA_VERSION = 1


def _source_fingerprint(path, array):
    root = Path(path).resolve()
    metadata = []
    for name in (".zarray", "zarr.json", ".zattrs", ".zgroup"):
        for candidate in (root / name, root / str(getattr(array, "path", "")) / name):
            if candidate.is_file():
                stat = candidate.stat()
                metadata.append([str(candidate.relative_to(root)), stat.st_size, stat.st_mtime_ns])
    stat = root.stat()
    return {
        "path": str(root), "shape": list(array.shape), "chunks": list(array.chunks),
        "dtype": str(array.dtype), "root_mtime_ns": stat.st_mtime_ns,
        "metadata": sorted(metadata),
    }


def _cache_key(manifest):
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()[:24]


def _probe_coordinates(shape):
    z, y, x = shape
    return [[0, 0, 0], [z - 1, y - 1, x - 1], [z // 2, y // 2, x // 2]]


def _completed_cache_valid(destination, existing, requested, source_arrays, z_lo):
    try:
        if not existing.get("complete") or any(existing.get(k) != v for k, v in requested.items()):
            return False
        normals = np.load(destination / "normals.npy", mmap_mode="r")
        grad = np.load(destination / "grad_mag.npy", mmap_mode="r")
        shape = tuple(requested["mmap_shape_zyx"])
        if normals.shape != (*shape, 2) or grad.shape != shape:
            return False
        probes = existing.get("probes", [])
        if len(probes) != len(_probe_coordinates(shape)):
            return False
        nx, ny, gm = source_arrays
        for probe in probes:
            local = tuple(probe["local_zyx"])
            source = (local[0] + z_lo, local[1], local[2])
            cache_values = [int(normals[local][0]), int(normals[local][1]), int(grad[local])]
            expected = [int(value) for value in probe["values_nx_ny_grad"]]
            if cache_values != expected:
                return False
            current = [int(array[source]) if array is not None else 0 for array in (nx, ny, gm)]
            if current != expected:
                return False
        return True
    except Exception:
        return False


class LasagnaMmapStore:
    def __init__(self, directory, manifest, *, workers=None):
        self.directory = Path(directory)
        self.manifest = manifest
        shape = tuple(manifest["mmap_shape_zyx"])
        self.normals = np.load(self.directory / "normals.npy", mmap_mode="r")
        self.grad_mag = np.load(self.directory / "grad_mag.npy", mmap_mode="r")
        if self.normals.shape != (*shape, 2) or self.normals.dtype != np.uint8:
            raise ValueError("Invalid normals mmap shape/dtype")
        if self.grad_mag.shape != shape or self.grad_mag.dtype != np.uint8:
            raise ValueError("Invalid gradient mmap shape/dtype")
        cpu_count = os.cpu_count() or 1
        self.worker_count = max(1, min(int(workers or min(16, cpu_count)), cpu_count, 32))
        self._pool = ThreadPoolExecutor(max_workers=self.worker_count, thread_name_prefix="lasagna-gather")
        self._staging_lock = threading.Lock()
        self._normal_staging = None
        self._grad_staging = None
        self.last_timings = {}

    @property
    def shape(self):
        return tuple(self.manifest["mmap_shape_zyx"])

    @property
    def z_origin(self):
        return int(self.manifest["mmap_z_origin"])

    @property
    def lasagna_scale(self):
        return int(self.manifest["coordinate_scale"])

    def close(self):
        self._pool.shutdown(wait=True, cancel_futures=True)

    def _parallel_gather(self, array, indices, channels=None):
        indices = np.ascontiguousarray(indices, dtype=np.int64).reshape(-1, 3)
        width = 1 if channels is None else channels
        output = np.empty((len(indices), width), dtype=np.uint8) if width > 1 else np.empty(len(indices), dtype=np.uint8)
        if not len(indices):
            return output
        batch = max(1, (len(indices) + self.worker_count - 1) // self.worker_count)

        def gather(bounds):
            lo, hi = bounds
            part = indices[lo:hi]
            values = array[part[:, 0], part[:, 1], part[:, 2]]
            output[lo:hi] = values

        list(self._pool.map(gather, [(lo, min(lo + batch, len(indices))) for lo in range(0, len(indices), batch)]))
        return output

    def gather_pair(self, normal_zyx, grad_zyx, device):
        """Gather both iteration requests through one persistent bounded pool."""
        import torch
        start = time.perf_counter()
        # The synchronization is explicit and attributed here.
        normal_cpu = normal_zyx.detach().to("cpu", non_blocking=False).numpy()
        grad_cpu = grad_zyx.detach().to("cpu", non_blocking=False).numpy()
        after_indices = time.perf_counter()
        normal_cpu = np.ascontiguousarray(normal_cpu, dtype=np.int64).reshape(-1, 3)
        grad_cpu = np.ascontiguousarray(grad_cpu, dtype=np.int64).reshape(-1, 3)
        normal_values = np.empty((len(normal_cpu), 2), dtype=np.uint8)
        grad_values = np.empty(len(grad_cpu), dtype=np.uint8)
        requests = []
        total = len(normal_cpu) + len(grad_cpu)
        batch = max(1, (total + self.worker_count - 1) // self.worker_count)

        def gather(array, indices, output, lo, hi):
            part = indices[lo:hi]
            output[lo:hi] = array[part[:, 0], part[:, 1], part[:, 2]]

        for array, indices, output in ((self.normals, normal_cpu, normal_values),
                                       (self.grad_mag, grad_cpu, grad_values)):
            for lo in range(0, len(indices), batch):
                requests.append(self._pool.submit(gather, array, indices, output, lo, min(lo + batch, len(indices))))
        for request in requests:
            request.result()
        after_gather = time.perf_counter()
        with self._staging_lock:
            pinned = torch.cuda.is_available()
            if self._normal_staging is None or self._normal_staging.shape != normal_values.shape:
                self._normal_staging = torch.empty(normal_values.shape, dtype=torch.uint8, pin_memory=pinned)
            if self._grad_staging is None or self._grad_staging.shape != grad_values.shape:
                self._grad_staging = torch.empty(grad_values.shape, dtype=torch.uint8, pin_memory=pinned)
            self._normal_staging.numpy()[...] = normal_values
            self._grad_staging.numpy()[...] = grad_values
            normal_gpu = self._normal_staging.to(device=device, non_blocking=True)
            grad_gpu = self._grad_staging.to(device=device, non_blocking=True)
        after_copy = time.perf_counter()
        self.last_timings = {
            "index_gpu_to_cpu_seconds": after_indices - start,
            "gather_seconds": after_gather - after_indices,
            "staging_and_async_copy_seconds": after_copy - after_gather,
        }
        return normal_gpu, grad_gpu

def prepare_lasagna_mmap(*, nx_array, ny_array, grad_mag_array, source_paths,
                         group, z_lo, z_hi, lasagna_scale, cache_directory,
                         slab_depth=32, workers=None, progress=None):
    from lasagna_data import _read_zarr_zslab_chunked

    reference = nx_array if nx_array is not None else grad_mag_array
    requested = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "sources": {name: _source_fingerprint(path, array) for name, path, array in (
            ("normal_x", source_paths.get("normal_x"), nx_array),
            ("normal_y", source_paths.get("normal_y"), ny_array),
            ("gradient_magnitude", source_paths.get("gradient_magnitude"), grad_mag_array),
        ) if array is not None},
        "zarr_group": str(group), "full_resolution_z_range": [z_lo * lasagna_scale, z_hi * lasagna_scale],
        "mmap_z_origin": int(z_lo), "mmap_shape_zyx": [int(z_hi - z_lo), int(reference.shape[1]), int(reference.shape[2])],
        "coordinate_scale": int(lasagna_scale), "normal_channel_order": ["NX", "NY"],
        "normal_layout": "ZYXC", "grad_mag_layout": "ZYX", "dtype": "uint8", "byte_order": "not-applicable",
    }
    key = _cache_key(requested)
    root = Path(cache_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"lasagna-{key}"
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if _completed_cache_valid(destination, existing, requested,
                                  (nx_array, ny_array, grad_mag_array), z_lo):
            return LasagnaMmapStore(destination, existing, workers=workers)

    shape = tuple(requested["mmap_shape_zyx"])
    raw_size = int(np.prod(shape, dtype=np.int64)) * 3
    free = shutil.disk_usage(root).free
    if free < int(raw_size * 1.05):
        raise RuntimeError(f"Insufficient free space for Lasagna mmap cache: need {raw_size}, have {free}")

    lock_path = root / f"lasagna-{key}.lock"
    lock_stream = lock_path.open("a+b")
    try:
        if os.name == "posix":
            import fcntl
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text())
            if _completed_cache_valid(destination, existing, requested,
                                      (nx_array, ny_array, grad_mag_array), z_lo):
                return LasagnaMmapStore(destination, existing, workers=workers)
        temporary = Path(tempfile.mkdtemp(prefix=f".lasagna-{key}.", dir=root))
        started = time.time()
        try:
            normals = np.lib.format.open_memmap(temporary / "normals.npy", mode="w+", dtype=np.uint8, shape=(*shape, 2))
            grad = np.lib.format.open_memmap(temporary / "grad_mag.npy", mode="w+", dtype=np.uint8, shape=shape)
            for channel, array in enumerate((nx_array, ny_array)):
                if array is None:
                    normals[..., channel] = 0
                    continue
                for source_lo in range(z_lo, z_hi, slab_depth):
                    source_hi = min(source_lo + slab_depth, z_hi)
                    normals[source_lo-z_lo:source_hi-z_lo, ..., channel] = _read_zarr_zslab_chunked(array, source_lo, source_hi, max_workers=workers or 16)
                    if progress: progress("normals", channel, source_hi-z_lo, z_hi-z_lo)
            if grad_mag_array is None:
                grad[...] = 0
            else:
                for source_lo in range(z_lo, z_hi, slab_depth):
                    source_hi = min(source_lo + slab_depth, z_hi)
                    grad[source_lo-z_lo:source_hi-z_lo] = _read_zarr_zslab_chunked(grad_mag_array, source_lo, source_hi, max_workers=workers or 16)
                    if progress: progress("grad_mag", 0, source_hi-z_lo, z_hi-z_lo)
            normals.flush(); grad.flush()
            del normals, grad
            for data_name in ("normals.npy", "grad_mag.npy"):
                with (temporary / data_name).open("rb") as data_stream:
                    os.fsync(data_stream.fileno())
            normal_read = np.load(temporary / "normals.npy", mmap_mode="r")
            grad_read = np.load(temporary / "grad_mag.npy", mmap_mode="r")
            probes = [{"local_zyx": coordinate,
                       "values_nx_ny_grad": [int(normal_read[tuple(coordinate)][0]),
                                              int(normal_read[tuple(coordinate)][1]),
                                              int(grad_read[tuple(coordinate)])]}
                      for coordinate in _probe_coordinates(shape)]
            del normal_read, grad_read
            complete = {**requested, "complete": True, "created_unix_seconds": time.time(),
                        "preparation_seconds": time.time() - started,
                        "raw_file_bytes": sum((temporary / name).stat().st_size for name in ("normals.npy", "grad_mag.npy")),
                        "probes": probes,
                        "slab_depth": slab_depth, "workers": workers or min(16, os.cpu_count() or 1)}
            with (temporary / "manifest.json").open("w", encoding="utf-8") as stream:
                json.dump(complete, stream, indent=2, sort_keys=True); stream.flush(); os.fsync(stream.fileno())
            # Never unlink an mmap cache which may still be held by a resident
            # session.  Retire it atomically and let a later cache-maintenance
            # operation remove it once no process can be using it.
            if destination.exists():
                retired = root / f".{destination.name}.retired-{time.time_ns()}"
                os.replace(destination, retired)
            os.replace(temporary, destination)
            temporary = None
            try:
                directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        finally:
            if temporary is not None: shutil.rmtree(temporary, ignore_errors=True)
        return LasagnaMmapStore(destination, complete, workers=workers)
    finally:
        lock_stream.close()
