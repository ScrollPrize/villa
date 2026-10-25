"""Resumable whole-volume chunk tracing of completed local predictions.

Python owns metadata, admission and progress only. Native code continuously
loads, constructs, filters and atomically publishes independent chunk outputs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from . import fiber_trace as ft
from .fiber_evaluation import CHUNK_KEYS, _keys, _settings


def sparse_keys(path, channel, spacing, base_shape_zyx, core_base):
    """Map stored Zarr-v2 presence chunks to deduplicated XYZ tracing cores.

    Payloads are not read. Missing chunks mean declared fill in a *completed*
    sparse volume, not a partially downloaded/inferred volume. Temporary files
    and other channel chunks cannot create jobs.
    """
    path = Path(path)
    metadata = json.loads((path / ".zarray").read_text())
    shape, chunks = metadata["shape"], metadata["chunks"]
    if metadata.get("zarr_format") != 2 or len(shape) not in (3, 4) or len(chunks) != len(shape):
        raise ValueError("enumeration requires Zarr v2 ZYX or CZYX storage")
    if any(type(v) is not int or v <= 0 for v in shape + chunks):
        raise ValueError("invalid Zarr shape/chunks")
    if metadata.get("fill_value", 0) not in (0, None):
        raise ValueError("sparse presence enumeration requires zero fill")
    if not math.isfinite(spacing) or spacing <= 0 or not math.isfinite(core_base) or core_base <= 0:
        raise ValueError("spacing and core size must be positive")
    if len(base_shape_zyx) != 3 or any(type(v) is not int or v <= 0 for v in base_shape_zyx):
        raise ValueError("base shape must be three positive integers")
    if (len(shape) == 3 and channel != 0) or (len(shape) == 4 and not 0 <= channel < shape[0]):
        raise ValueError("presence channel outside storage")
    separator = metadata.get("dimension_separator", ".")
    if separator not in (".", "/"):
        raise ValueError("unsupported Zarr dimension separator")
    if (path / ".native_pool.json").exists():
        recipe = json.loads((path / ".native_pool.json").read_text())
        if (recipe.get("revision"), recipe.get("factor"), recipe.get("mode"), recipe.get("channel")) != (
                1, 2, "top2_spatial_comp", "presence"):
            raise ValueError("unsupported native pooling inventory recipe")
        root = Path(recipe["source"])
        source = Path(recipe["channels"]["presence"]) if "channels" in recipe else (
            Path("fiber_presence.ome.zarr") / recipe.get("source_level", "1"))
        source = source if source.is_absolute() else root / source
        if source.resolve() == path.resolve():
            raise ValueError("recursive native pooling inventory")
        return sparse_keys(source, 0, spacing / 2, base_shape_zyx, core_base)
    keys = set()
    # Walk only numeric paths; never descend into unrelated metadata/scratch.
    def chunks_on_disk(root, prefix=()):
        with os.scandir(root) as entries:
            for entry in entries:
                if separator == ".":
                    parts = entry.name.split(".")
                    if entry.is_file() and len(parts) == len(shape) and all(p.isdecimal() for p in parts):
                        yield tuple(map(int, parts))
                elif entry.name.isdecimal():
                    key = prefix + (int(entry.name),)
                    if len(key) == len(shape) and entry.is_file():
                        yield key
                    elif len(key) < len(shape) and entry.is_dir():
                        yield from chunks_on_disk(entry.path, key)
    for index in chunks_on_disk(path):
        if any(i * c >= n for i, c, n in zip(index, chunks, shape)):
            raise ValueError(f"stored chunk outside array shape: {index}")
        if len(shape) == 4 and index[0] != channel // chunks[0]:
            continue
        ranges = []
        for i, c, n, extent in zip(index[-3:], chunks[-3:], shape[-3:], base_shape_zyx):
            lo, hi = i * c * spacing, min((i + 1) * c * spacing, n * spacing, extent)
            if lo >= hi:
                ranges = []
                break
            ranges.append(range(math.floor(lo / core_base), math.ceil(hi / core_base)))
        if ranges:
            keys.update((x, y, z) for z, y, x in itertools.product(*ranges))
    return sorted(keys, key=lambda key: (key[2], key[1], key[0]))


@contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    # Persistent lock inode, not a done marker. Removing it would permit two
    # owners of different inodes. flock releases automatically even on a crash.
    with (output / ".writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another writer owns {output}") from exc
        yield


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    _keys(cfg, ("prediction_manifest", "normal_manifest", "source_identity", "scaledown_power",
                "enable_width", "cache_bytes", "trace", "chunks", "keys_xyz", "loader"), "volume config")
    if not isinstance(cfg.get("source_identity"), str) or not cfg["source_identity"]:
        raise ValueError("source_identity must identify immutable predictions AND normals")
    for name in ("prediction_manifest", "normal_manifest"):
        if name in cfg:
            value = Path(cfg[name])
            cfg[name] = str(value if value.is_absolute() else path.parent / value)
    library = cfg.get("chunks", {}).get("gpu_library")
    if library and not Path(library).is_absolute():
        cfg["chunks"]["gpu_library"] = str(path.parent / library)
    return cfg


def open_inputs(cfg):
    budget = cfg.get("cache_bytes", 512 << 20)
    field = ft.open_prediction_field(cfg["prediction_manifest"], cache_bytes=budget,
        scaledown_power=cfg.get("scaledown_power", 0), enable_width=cfg.get("enable_width", True))
    if cfg.get("normal_manifest"):
        normals = ft.open_normal_sampler(cfg["normal_manifest"], field.trace_to_base_scale, cache_bytes=budget)
    else:
        normals = ft.open_fiber_normal_sampler(field, cache_bytes=budget)
    trace = dict(parallel_threads=1, step_voxels=16 / field.trace_to_base_scale,
                 trace_to_base_scale=field.trace_to_base_scale)
    trace.update(cfg.get("trace", {}))
    trace = ft.TraceConfig.from_dict(trace)
    chunks = _settings(ft.ChunkConfig, cfg.get("chunks", {}), CHUNK_KEYS)
    if chunks.gpu_library and chunks.gpu_devices == 0:
        # Cache validation requires a valid backend configuration, but does not
        # initialize CUDA. Actual construction replaces this with the grant.
        chunks.gpu_devices = 1
    if chunks.conflict_normal_base <= 0 or chunks.conflict_tangent_base <= 0:
        raise ValueError("whole-volume workflow requires explicit positive conflict ellipse radii")
    if trace.parallel_threads != 1:
        raise ValueError("parallel_threads must be 1; shared queue owns CPU parallelism")
    return field, normals, trace, chunks


def configure_loader(cfg, workers):
    loader = cfg.get("loader", {})
    _keys(loader, ("resource_scaled", "backend", "queued_chunk_window"), "loader")
    scaled = loader.get("resource_scaled", True)
    if type(scaled) is not bool:
        raise ValueError("loader.resource_scaled must be boolean")
    # Explicit legacy control for matched benchmarks / already initialized hosts.
    if not scaled:
        if loader.get("backend", "direct") != "direct" or loader.get("queued_chunk_window", 0):
            raise ValueError("queued loader configuration requires resource_scaled=true")
        return {"resource_scaled": False}
    backend=loader.get("backend", "direct")
    window=loader.get("queued_chunk_window", 0)
    if backend not in ("direct", "queued") or type(window) is not int or window < 0:
        raise ValueError("invalid loader backend or queued_chunk_window")
    if backend=="direct" and window:
        raise ValueError("queued_chunk_window requires the queued backend")
    if backend == "queued" and cfg.get("chunks", {}).get("worker_cache_bytes", 0):
        raise ValueError("queued loader and worker-local partitions are separate backends")
    args=(workers,cfg.get("cache_bytes",512 << 20))
    return json.loads(ft.configure_chunk_loader(*args) if backend=="direct" and not window
                      else ft.configure_chunk_loader(*args,backend,window))


def job_keys(cfg, field, chunks):
    storage, channel, spacing, shape = ft.prediction_presence_storage(field)
    if "keys_xyz" not in cfg:
        return sparse_keys(storage, channel, spacing, shape, chunks.core_base)
    keys = cfg["keys_xyz"]
    if any(len(k) != 3 or any(type(v) is not int or v < 0 or v * chunks.core_base >= shape[2-a]
                              for a, v in enumerate(k)) for k in keys):
        raise ValueError("keys_xyz must be nonnegative integer cores inside base shape")
    if len({tuple(k) for k in keys}) != len(keys):
        raise ValueError("duplicate keys_xyz")
    return keys


def missing_outputs(output, keys):
    return sum(not any((output / ("_".join(map(str, k)) + suffix)).is_file()
                       for suffix in (".msgpack.zst", ".json")) for k in keys)


def _inspect_outputs(cfg, output):
    field, normals, trace, chunks = open_inputs(cfg)
    keys = job_keys(cfg, field, chunks)
    missing = missing_outputs(output, keys)
    return len(keys), missing, bool(chunks.gpu_library and missing and not chunks.cache_only)


def workflow_command():
    launcher = os.environ.get("VC_CHUNK_VOLUME_LAUNCHER")
    return ([sys.executable, launcher] if launcher else [sys.executable, "-m", "vc.fiber_chunk_volume"])


def inspect_outputs(config, output):
    # Metadata opening creates native cache pools. Keep those out of the
    # supervisor so execution can size its pools from the eventual grant.
    # Use the same launcher/build as execution, not an unrelated installed vc.
    completed = subprocess.run(workflow_command() + [str(config.resolve()), str(output), "--inspect-worker"],
                               check=True, text=True, stdout=subprocess.PIPE)
    return tuple(json.loads(completed.stdout.splitlines()[-1]))


def execute(cfg, output, workers, contexts, *, progress=None):
    grant = json.loads(os.environ.get("VILLA_GPU_MANAGER_GRANT", "null"))
    if grant is not None:
        workers = min(workers, int(grant["cpu_cores"]))
        if workers < 1:
            raise RuntimeError("manager grant contains no CPU workers")
    loader = configure_loader(cfg, workers)
    field, normals, trace, chunks = open_inputs(cfg)
    keys = job_keys(cfg, field, chunks)
    if chunks.gpu_library and missing_outputs(output, keys) and not chunks.cache_only:
        if not grant or not grant.get("gpus"):
            raise RuntimeError("GPU construction requires a GPU-manager grant")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if set(visible) != set(grant["gpus"]):
            raise RuntimeError("visible devices differ from manager grant")
        chunks.gpu_devices = len(visible)
        memory_mib = min(int(v) for v in grant["gpus"].values()) // (1024**2)
        if memory_mib < 2048 or workers < 1:
            raise RuntimeError("grant requires at least one CPU and 2 GiB per GPU")
        chunks.gpu_batch_memory_mib = min(chunks.gpu_batch_memory_mib, memory_mib - 1024)
        # Leave 1 GiB for context/resident fields. Workspace changes batch size,
        # never numerical settings, seed selection or tracing costs.
    control = ft.EvaluationControl()
    began = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(ft.prepare_chunk_volume, field, normals, trace, chunks,
            str(output), cfg["source_identity"], keys, control, workers, contexts)
        try:
            while True:
                try:
                    future.result(timeout=5)
                    break
                except TimeoutError:
                    if progress:
                        progress(control.status(), time.monotonic() - began)
        except BaseException:
            control.cancel("volume command interrupted or failed")
            raise  # Executor joins all native work before releasing resources.
    result = dict(control.status(), wall_seconds=time.monotonic()-began,
                  workers=workers, contexts_per_worker=contexts, loader=loader)
    if hasattr(ft,"chunk_loader_stats"):
        result["loader_stats"]=json.loads(ft.chunk_loader_stats())
    if progress:
        progress(result, result["wall_seconds"])
    return result


def print_progress(status, elapsed):
    done, total = status["completed"], status["total"]
    rate = done / elapsed if elapsed else 0
    eta = (total-done) / rate if rate else None
    print(f"[chunk-volume] {done}/{total} built={status.get('built', 0)} "
          f"reused={status.get('reused', 0)} annotated={status.get('annotated', 0)} "
          f"elapsed={elapsed:.1f}s rate={rate:.2f} chunks/s "
          f"eta={'unknown' if eta is None else f'{eta/60:.1f}min'}", flush=True)


def eligible_gpu_uuids(devices, inventory):
    """Let the manager choose devices unless the caller explicitly restricts them."""
    managed = inventory["gpus"]
    if not managed:
        raise ValueError("GPU manager inventory contains no GPUs")
    if devices is None:
        return sorted(managed)
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                                   text=True, timeout=10)
    available = {int(row.split(",")[0]): row.split(",")[1].strip() for row in rows.splitlines()}
    try:
        indices = [int(x) for x in devices.split(",")]
    except ValueError as exc:
        raise ValueError("devices must be comma-separated GPU indices") from exc
    if len(set(indices)) != len(indices) or any(x not in available for x in indices):
        raise ValueError("devices must be unique existing GPU indices")
    selected = [available[x] for x in indices]
    if any(uuid not in managed for uuid in selected):
        raise ValueError("selected devices must belong to the GPU manager inventory")
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--contexts-per-worker", type=int, default=8)
    parser.add_argument("--devices", help="optional GPU-index restriction; default: manager chooses from its full inventory")
    parser.add_argument("--vram-gib", type=float, default=8)
    parser.add_argument("--priority", type=int, default=100)
    parser.add_argument("--plan", action="store_true", help="enumerate only; no tracing or GPU request")
    parser.add_argument("--managed-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--inspect-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.contexts_per_worker < 1 or not math.isfinite(args.vram_gib) or args.vram_gib <= 0:
        parser.error("positive resource budgets required")
    cfg = load_config(args.config)
    output = args.output.resolve()
    if args.inspect_worker:
        print(json.dumps(_inspect_outputs(cfg, output)), flush=True)
        return 0
    if args.managed_worker:
        if not os.environ.get("VILLA_GPU_MANAGER_GRANT"):
            parser.error("managed worker requires inherited manager grant")
        with output_lock(output):
            result = execute(cfg, output, args.workers, args.contexts_per_worker, progress=print_progress)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    if args.plan:
        field, normals, trace, chunks = open_inputs(cfg)
        keys = job_keys(cfg, field, chunks)
        missing = missing_outputs(output, keys)
        print(json.dumps(dict(cores=len(keys), missing=missing, existing=len(keys)-missing,
                              existing_validation="performed on run, not plan")), flush=True)
        return 0
    with output_lock(output):
        count, missing, needs_gpu = inspect_outputs(args.config, output)
        print(f"[chunk-volume] {count} cores; {missing} missing outputs; "
              f"{count-missing} existing outputs to validate", flush=True)
        if not needs_gpu:
            result = execute(cfg, output, args.workers, args.contexts_per_worker, progress=print_progress)
            print(json.dumps(result, sort_keys=True), flush=True)
            return 0
    # Do not hold a parent-only lock while a child writes: the managed worker
    # reacquires it and re-enumerates after admission. Its lock lives as long as
    # the actual writer, including supervisor failure/watchdog cleanup.
    from villa_gpu_manager.bootstrap import ensure_manager
    from villa_gpu_manager.exclusive import run_exclusive
    socket, inventory = ensure_manager()
    try:
        gpu_uuids = eligible_gpu_uuids(args.devices, inventory)
    except ValueError as exc:
        parser.error(str(exc))
    command = workflow_command() + [str(args.config.resolve()), str(output),
               "--workers", str(args.workers), "--contexts-per-worker", str(args.contexts_per_worker),
               "--managed-worker"]
    return run_exclusive(command, socket=socket, gpu_uuids=gpu_uuids,
        cpu_cores=args.workers, vram_bytes_per_gpu=int(args.vram_gib * 1024**3), exclusive=False,
        accept_partial=True, priority=args.priority, label="fiber-chunk-volume", stop_event=threading.Event())


if __name__ == "__main__":
    raise SystemExit(main())
