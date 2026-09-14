"""Standalone native beam replay coordinated with on-demand GPU inference.

Only control descriptors cross the localhost socket. Published predictions use
the ordinary Lasagna/OME-Zarr format and the existing model/tiled finalizers.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import itertools
import importlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import select
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import urlopen

import numpy as np
import torch

from lasagna import tiled_predict3d as tiled
from lasagna import demand_predict3d as demand
from lasagna.inference_provenance import atomic_write, json_digest, sha256_file
from vesuvius.neural_tracing.fiber_trace_2d.sampling import Vc3dCoordinateSampler
from vesuvius.neural_tracing.fiber_trace_3d import inference_adapter
from vesuvius.neural_tracing.fiber_trace_3d.inference_cache import resolve_prediction_cache
from vesuvius.neural_tracing.fiber_trace_3d.infer import _checkpoint_config_for_inference, _resolve_inference_precision


def location(base, value):
    if "://" in value:
        parsed = urlsplit(value)
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError("use an unsigned source locator and ambient credentials; URLs are persisted as source identity")
        return value.rstrip("/")
    return str((base / value).resolve())


def inference_cache_root(cache_root, namespace=None):
    """Optional resumable prediction-only namespace; source fetch caches stay shared."""
    root = Path(cache_root) / "live_inference"
    if namespace is None:
        return root
    if not isinstance(namespace, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", namespace) is None:
        raise ValueError("inference_cache_namespace must be a nonempty safe single component (max64 characters)")
    destination = root / "_namespaces" / namespace
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("inference cache namespace escapes live_inference through a symlink")
    if destination.is_symlink():
        raise ValueError("inference cache namespace must not be a symlink")
    # Model/source identity and DemandStore's writer lock govern safe reuse.
    # Timing runs requiring fresh inference must choose a new namespace.
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def freeze_checkpoint(path, cache_root):
    digest = sha256_file(path)
    destination = cache_root / "live_inference" / "_models" / digest / "snapshot.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256_file(destination) != digest:
            raise ValueError("cached frozen checkpoint is corrupt")
        return destination, digest
    descriptor, name = tempfile.mkstemp(prefix=".snapshot-", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as target, Path(path).open("rb") as source:
            shutil.copyfileobj(source, target, length=8 << 20)
            target.flush()
            os.fsync(target.fileno())
        if sha256_file(name) != digest:
            raise ValueError("checkpoint changed while freezing; use an immutable completed snapshot")
        os.replace(name, destination)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return destination, digest


def reference_chunks(points_base, factor, shape, chunk):
    """Exact segment crossings of interpolation-support chunk boundaries.

    Both a chunk boundary and its preceding voxel can change the corner-chunk
    set. Visit each interval's midpoint and endpoints, not just annotations.
    """
    points = np.asarray(points_base, dtype=np.float64) / factor
    result = set()
    for left, right in zip(points[:-1], points[1:]):
        delta = right - left
        breaks = {0.0, 1.0}
        for xyz in range(3):
            if delta[xyz] == 0:
                continue
            lo, hi = sorted((left[xyz], right[xyz]))
            # Clip traversal ranges to the actual array, not out-of-volume
            # annotation extents. Shape is ZYX; points are XYZ.
            n = shape[2 - xyz]
            for boundary in (0, n - 1):
                t = (boundary - left[xyz]) / delta[xyz]
                if 0 < t < 1:
                    breaks.add(float(t))
            for k in range(max(0, math.floor(lo / chunk)), min(math.ceil(hi / chunk) + 2, math.ceil(n / chunk) + 1)):
                for boundary in (k * chunk, k * chunk - 1):
                    t = (boundary - left[xyz]) / delta[xyz]
                    if 0 < t < 1:
                        breaks.add(float(t))
        ordered = sorted(breaks)
        samples = ordered + [(a + b) * 0.5 for a, b in zip(ordered[:-1], ordered[1:])]
        for t in samples:
            p = (left + t * delta)[::-1]
            if np.any(p < 0) or np.any(p > np.asarray(shape) - 1):
                continue
            floor = np.floor(p).astype(np.int64)
            for corner in itertools.product((0, 1), repeat=3):
                index = np.minimum(floor + corner, np.asarray(shape) - 1) // chunk
                result.add(tuple(int(v) for v in index))
    return result


def input_root_attributes(volume):
    """Read OME metadata without calling the local-only native root_attrs API.

    Native remote_url is the canonical HTTP group root used for chunk reads.
    This requests metadata only; all voxel loading remains native.
    """
    remote = getattr(volume, "remote_url", "")
    if not remote:
        return volume.root_attrs
    with urlopen(str(remote).rstrip("/") + "/.zattrs", timeout=30) as response:
        attrs = json.load(response)
    if not isinstance(attrs, dict):
        raise ValueError("input OME root attributes must be an object")
    return attrs


def validate_input_lattice(attrs, level):
    """Only zero-offset, isotropic OME ZYX pyramids are supported initially.

    Geometry is expressed in level-zero voxel coordinates. Physical units may
    differ across sources, but the relative sampling transform must be exact.
    Reject unsupported affine metadata rather than silently ignoring it.
    """
    if level < 0 or level > 30:
        raise ValueError("invalid input level")
    multiscales = attrs.get("multiscales", [])
    if len(multiscales) != 1:
        raise ValueError("input requires one explicit OME multiscale lattice")
    metadata = multiscales[0]
    if [a if isinstance(a, str) else a.get("name") for a in metadata.get("axes", [])] != ["z", "y", "x"]:
        raise ValueError("input must have exactly ZYX spatial axes")

    def scale(transforms):
        result = np.ones(3, dtype=np.float64)
        for transform in transforms:
            kind = transform.get("type")
            if kind == "scale":
                value = np.asarray(transform["scale"], dtype=np.float64)
                if value.shape != (3,) or not np.isfinite(value).all() or np.any(value <= 0):
                    raise ValueError("invalid input scale transform")
                result *= value
            elif kind == "translation":
                value = np.asarray(transform["translation"], dtype=np.float64)
                if value.shape != (3,) or np.any(value != 0):
                    raise ValueError("translated input lattices are not supported")
            else:
                raise ValueError("unsupported input coordinate transform")
        return result

    shared = scale(metadata.get("coordinateTransformations", []))
    datasets = {str(d["path"]).strip("/"): d for d in metadata.get("datasets", [])}
    if "0" not in datasets or str(level) not in datasets:
        raise ValueError("input metadata must bind level 0 and the selected numeric level")
    base_scale = shared * scale(datasets["0"].get("coordinateTransformations", []))
    selected_scale = shared * scale(datasets[str(level)].get("coordinateTransformations", []))
    if not np.allclose(base_scale, base_scale[0], rtol=1e-8, atol=0):
        raise ValueError("anisotropic input base voxels are not supported")
    if not np.allclose(selected_scale / base_scale, 1 << level, rtol=1e-8, atol=0):
        raise ValueError("input transform is not the expected isotropic power-of-two lattice")


def implementation_identity():
    modules = ["lasagna.tiled_predict3d", "lasagna.demand_predict3d", "lasagna.normal_encoding",
               "vesuvius.neural_tracing.fiber_trace_3d.inference_adapter",
               "vesuvius.neural_tracing.fiber_trace_3d.infer",
               "vesuvius.neural_tracing.fiber_trace_3d.model",
               "vesuvius.neural_tracing.fiber_trace_3d.direction",
               "vesuvius.neural_tracing.fiber_trace_3d.prediction",
               "vesuvius.neural_tracing.fiber_trace.model"]
    files = {name: sha256_file(importlib.import_module(name).__file__) for name in modules}
    # Include shared network components, not only the adapter's caller.
    nets = Path(importlib.import_module("vesuvius.neural_tracing.nets").__file__).parent
    files.update({"nets/" + str(p.relative_to(nets)): sha256_file(p) for p in sorted(nets.rglob("*.py"))})
    files["runner"] = sha256_file(__file__)
    files["runtime"] = {"torch": str(torch.__version__), "numpy": np.__version__, "cuda": torch.version.cuda}
    return files


def native_reader_options(settings):
    """Identical native-cache budgets for live evaluation and its profiler."""
    return dict(cache_budget_bytes=int(settings.get("input_cache_bytes", 4 << 30)),
                io_threads=int(settings.get("input_io_threads", 16)),
                decode_threads=int(settings.get("input_decode_threads", 8)), sample_threads=1)


class NativeTileReader:
    """Read exact-level integer bboxes directly from the shared native cache."""
    def __init__(self, sampler, geometry, dtype, *, profile_samples=None):
        self.sampler, self.geometry, self.dtype = sampler, geometry, np.dtype(dtype)
        self.profile_samples = profile_samples

    def __call__(self, tile):
        start = tuple(int(v) - self.geometry.border for v in tile)
        shape = (self.geometry.tile,) * 3
        with demand.stage_timing(self.profile_samples, "reader.native_block"):
            raw = self.sampler.volume.read_zyx(start, shape, self.sampler.level)
        if raw.dtype != self.dtype or raw.shape != shape or not raw.flags.c_contiguous:
            raise ValueError("native bbox reader returned incompatible shape/dtype/layout")
        return raw


def create_demand_store(cache, source_identity, model_identity, geometry, effective, snapshot, level, settings):
    """Shared numerical/output setup for live tracing and isolated profiling."""
    power = int(math.log2(geometry.scale))
    adapter = inference_adapter.FiberTrace3DPredictAdapter(effective, checkpoint=snapshot,
        level=level + power, scaledown=(1 << level) * geometry.scale, inference_scaledown=geometry.scale,
        chunk_size=geometry.chunk, product_prefix="fiber", zarr_path_prefix=cache / "fiber",
        emit_embeddings=bool(settings.get("emit_embeddings", False)))
    # Construction restores checkpoint config; reapply the resolved precision.
    adapter.config["training"] = {**adapter.config.get("training", {}),
        "mixed_precision": effective["training"]["mixed_precision"]}
    products = adapter.output_products
    output = tiled.OmeZarrOutputAdapter(products=products, n_levels=level + power + 1)
    def prepare_output():
        tiled.create_product_omezarr_groups(products=products, base_shape_zyx=source_identity["base_shape"],
            n_levels=level + power + 1, ome_chunk=geometry.chunk, ome_compressor=settings.get("compressor", "blosc-zstd"))
        if not (cache / "inference.lasagna.json").exists():
            tiled.write_lasagna_product_manifest(output_path=cache / "inference.lasagna.json", products=products,
                base_shape_zyx=source_identity["base_shape"], provenance_json="inference.json")
        if not (cache / "inference.json").exists():
            atomic_write(cache / "inference.json", {"source": source_identity, "model": model_identity,
                "coverage": "partial; readiness requires complete published sibling chunks; live scheduling state is RAM-only",
                "coarser_pyramids": "not generated"})
    return demand.DemandStore(cache, {"source": source_identity, "model": model_identity},
        geometry, adapter, output, accumulator_dtype=settings.get("accumulator_dtype", "float16"),
        accumulator_layout=settings.get("accumulator_layout", "flat"),
        accumulator_prefault=settings.get("accumulator_prefault", False),
        prepare_output=prepare_output)


def run(config_path, checkpoint, devices, native_binary, model_config=None, *, render=False,
        manager_socket=None, manager_priority=None, unmanaged=False):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    manager = config.get("gpu_manager", {})
    if not unmanaged and (manager.get("enabled", False) or manager_socket is not None):
        from .managed_evaluation import launch_managed
        command = [sys.executable, "-m", __name__, "--config", str(config_path),
                   "--checkpoint", str(checkpoint), "--native-binary", str(native_binary)]
        # __name__ is __main__ under python -m; use the importable module name.
        command[2] = "vesuvius.neural_tracing.fiber_trace_3d.live_evaluate"
        if model_config is not None:
            command.extend(["--model-config", str(model_config)])
        if render:
            command.append("--render")
        return launch_managed(command, devices=devices, settings=manager,
                              socket=manager_socket, priority=manager_priority)
    base = config_path.parent
    if config.get("version") != 1 or not config.get("sources"):
        raise ValueError("evaluation config requires version1 and explicit sources")
    binary = Path(shutil.which(str(native_binary)) or native_binary).resolve()
    annotation_hashes = {location(base, path): sha256_file(Path(location(base, path)))
                         for source in config["sources"] for path in source["fibers"]}
    references = json.loads(subprocess.check_output(
        [str(binary), "replay-references", "--config", str(config_path)], text=True))
    if any(sha256_file(Path(path)) != digest for path, digest in annotation_hashes.items()):
        raise ValueError("annotation changed while selecting evaluation references")
    resolver = binary.with_name("vc_cache_root")
    cache_root = Path(subprocess.check_output([str(resolver)], text=True).strip())
    prediction_cache_root = inference_cache_root(cache_root, config.get("inference_cache_namespace"))
    snapshot, checkpoint_digest = freeze_checkpoint(Path(checkpoint), cache_root)
    effective, _ = _checkpoint_config_for_inference(snapshot, model_config)
    settings = dict(config.get("inference", {}))
    trace_args = config.get("trace_args", [])
    if "--embedding-weight" in trace_args:
        weight = float(trace_args[trace_args.index("--embedding-weight") + 1])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("embedding weight must be finite and nonnegative")
        if weight > 0:
            settings["emit_embeddings"] = True
    device_list = tuple(torch.device(f"cuda:{int(v)}") for v in devices.split(","))
    if not device_list or len(set(device_list)) != len(device_list):
        raise ValueError("select a nonempty distinct comma-separated CUDA device list")
    precision, _ = _resolve_inference_precision(settings.get("precision", "auto"), checkpoint=snapshot, devices=device_list)
    effective = dict(effective)
    effective["training"] = {**effective.get("training", {}), "mixed_precision": precision}
    tile, overlap, border = (int(settings.get(k, default)) for k, default in
                             (("tile_size", 192), ("overlap", 64), ("border", 16)))
    power = int(settings.get("scaledown_power", 2))
    if not 0 <= power <= 8:
        raise ValueError("invalid inference scaledown power")
    scale, chunk = 1 << power, int(settings.get("chunk_size", 32))
    dtype = settings.get("accumulator_dtype", "float16")
    max_batch_size = settings.get("max_batch_size", 1)
    if type(max_batch_size) is not int or max_batch_size < 1:
        raise ValueError("inference.max_batch_size must be a positive integer")
    model_identity = {"schema": 1, "checkpoint_sha256": checkpoint_digest, "config": effective,
                      "precision": precision, "tile": tile, "overlap": overlap, "border": border,
                      "scale": scale, "chunk": chunk, "accumulator_dtype": dtype,
                      "compressor": settings.get("compressor", "blosc-zstd"),
                      "max_batch_size": max_batch_size,
                      "embedding_channels": "signed-float32-v2" if settings.get("emit_embeddings", False) else None}
    evaluation = dict(config)
    evaluation["reference_plan"] = references
    evaluation["annotation_sha256"] = annotation_hashes
    if not isinstance(config.get("render", False), bool):
        raise ValueError("render must be a boolean")
    evaluation["render"] = render or config.get("render", False)
    evaluation["output"] = location(base, config["output"])
    if Path(evaluation["output"]).exists():
        raise ValueError("evaluation output already exists")
    evaluation["sources"] = []
    source_to_store, stores, readers, channel_indices, backgrounds = {}, {}, {}, {}, {}
    started = time.monotonic()
    native = None
    connection = None
    pool = None
    with ExitStack() as cleanup:
        for source_index, source in enumerate(config["sources"]):
            source_id = source["id"]
            if source_id in source_to_store:
                raise ValueError("duplicate source ID")
            input_path = location(base, source["input_volume"])
            level = int(source.get("input_level", 0))
            if not 0 <= level <= 30:
                raise ValueError("invalid input level")
            sampler = Vc3dCoordinateSampler(input_path, level=level, cache_root=str(cache_root),
                **native_reader_options(settings))
            volume_shape = tuple(int(v) for v in sampler.volume.shape_at(level))
            base_shape = tuple(int(v) for v in sampler.volume.shape_at(0))
            root_attrs = input_root_attributes(sampler.volume)
            validate_input_lattice(root_attrs, level)
            if volume_shape != tuple((n + (1 << level) - 1) // (1 << level) for n in base_shape):
                raise ValueError("selected input level is not the expected isotropic power-of-two lattice")
            geometry = demand.DemandGeometry(volume_shape, tile, overlap, border, scale, chunk)
            source_identity = {"locator": input_path, "level": level, "shape": volume_shape,
                "base_shape": base_shape, "metadata": root_attrs,
                "revision": source.get("source_revision"), "base_voxel_size_um": source["base_voxel_size_um"],
                "coordinate_space": source.get("coordinate_space")}
            source_key = json_digest(source_identity)
            source_to_store[source_id] = source_key
            cache, stored_model_identity = resolve_prediction_cache(
                prediction_cache_root / source_key, source_identity, model_identity, geometry, settings)
            print(f"live-eval cache source={source_id} existing={(cache / 'cache_identity.json').is_file()} path={cache}", flush=True)
            if source_key not in stores:
                store = cleanup.enter_context(create_demand_store(cache, source_identity, stored_model_identity,
                    geometry, effective, snapshot, level, settings))
                products = store.products
                stores[source_key] = store
                readers[source_key] = NativeTileReader(sampler, geometry, sampler.volume.dtype)
                channel_indices[source_key] = {str((Path(c.relative_path) / str(p.level)).resolve()): geometry
                    for p in products for c in p.channels}
                backgrounds[source_key] = set()
            entry = dict(source)
            entry["fiber_manifest"] = str(cache / "inference.lasagna.json")
            entry["normal_manifest"] = location(base, source["normal_manifest"])
            entry["inference_scaledown_power"] = power
            entry["input_volume"] = input_path
            if "ct_volume" in source:
                entry["ct_volume"] = location(base, source["ct_volume"])
            elif evaluation["render"]:
                # The input pyramid's level zero is the base coordinate frame.
                # Native Volume/cache supports the same remote source as loading.
                entry["ct_volume"] = input_path
                entry["ct_base_volume"] = True
            entry["fibers"] = [location(base, f) for f in source["fibers"]]
            evaluation["sources"].append(entry)
            for fiber in references["sources"][source_index]["fibers"]:
                for run in fiber["runs"]:
                    backgrounds[source_key].update(reference_chunks(run["points_base_xyz"],
                        (1 << level) * scale, geometry.output_shape, chunk))
        if any(sha256_file(Path(path)) != digest for path, digest in annotation_hashes.items()):
            raise ValueError("annotation changed before native evaluation launch")
        return coordinate_native(evaluation, binary, stores, source_to_store, channel_indices,
            backgrounds, readers, device_list, settings, started=started)


def coordinate_native(evaluation, binary, stores, source_to_store, channel_indices,
                      backgrounds, readers, device_list, settings, *, started=None, allow_inference=True):
    """Shared authenticated control loop for live inference and cached-only replay."""
    started = time.monotonic() if started is None else started
    max_batch_size = settings.get("max_batch_size", 1)
    if type(max_batch_size) is not int or max_batch_size < 1:
        raise ValueError("inference.max_batch_size must be a positive integer")
    producer_implementation = implementation_identity()  # Run provenance only, never a cache key.
    native = connection = pool = gpu_lease = None
    adaptive = settings.get("demand_gpu_manager")
    with ExitStack() as cleanup, tempfile.TemporaryDirectory(prefix="fiber-live-control-") as control:
        listener = cleanup.enter_context(socket.socket())
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(120)
        token = secrets.token_hex(32)
        evaluation["live_endpoint"] = {"port": listener.getsockname()[1], "token": token,
                                       "timeout_seconds": int(settings.get("request_timeout_seconds", 600))}
        if adaptive is not None:
            evaluation["live_endpoint"]["heartbeat_wait"] = True
        runtime_config = Path(control) / "evaluation.json"
        atomic_write(runtime_config, evaluation)
        log = cleanup.enter_context((Path(control) / "native.log").open("w+"))
        native = subprocess.Popen([str(binary), "replay-set", "--config", str(runtime_config)], stdout=log, stderr=subprocess.STDOUT)
        try:
            # Observe startup failure instead of waiting the whole accept timeout.
            listener.settimeout(0.1)
            deadline = time.monotonic() + 120
            while connection is None:
                if native.poll() not in (None, 0):
                    raise RuntimeError("native evaluator failed during source validation (see native.log)")
                if time.monotonic() > deadline:
                    raise TimeoutError("native evaluator did not connect")
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    pass
            connection.settimeout(1)
            pending = b""
            authenticated = False
            waiters = {}
            last_report = time.monotonic()
            queue_log = None
            last_queue_report = 0.0
            last_metrics_report, metrics_cases = 0.0, -1
            last_heartbeat = 0.0
            while not authenticated or native.poll() is None or waiters or (pool is not None and pool.busy):
                if evaluation.get("render") and time.monotonic() - last_metrics_report >= 30:
                    root = Path(evaluation["output"])
                    count = sum(1 for _ in (root / "cases").glob("*/trace.json"))
                    image_count = sum(1 for _ in (root / "cases").glob("*/rendered*/strips_*.jpg"))
                    if count and (count, image_count) != metrics_cases:
                        from vesuvius.neural_tracing.fiber_trace_3d.visualization_report import write_trace_progress_report
                        write_trace_progress_report(root, root.parent)
                        metrics_cases = (count, image_count)
                    last_metrics_report = time.monotonic()
                if not allow_inference and time.monotonic() - started > settings.get("cached_trace_timeout_seconds", 300):
                    raise TimeoutError("cached trace profiling exceeded its bounded duration")
                if select.select([connection], [], [], 0.005)[0]:
                    received = connection.recv(65536)
                    if not received:
                        if not authenticated:
                            raise RuntimeError("native evaluator disconnected before authentication")
                        if waiters:
                            raise RuntimeError("native evaluator disconnected with outstanding requests")
                        break
                    pending += received
                    if len(pending) > (16 << 20):
                        raise ValueError("oversized native demand message")
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        message = json.loads(line)
                        if message.get("version") != 1:
                            raise ValueError("unsupported native protocol")
                        if not authenticated:
                            if message.get("type") != "hello" or not secrets.compare_digest(message.get("token", ""), token):
                                raise ValueError("invalid native control authentication")
                            authenticated = True
                            # Hello is sent only after native source/annotation
                            # validation. Prefetch is required even for a case
                            # that terminates without issuing a sample demand.
                            for background_key, indices in backgrounds.items():
                                for index in sorted(indices):
                                    stores[background_key].request(index, background=True)
                            continue
                        if message.get("type") != "demand" or message["request_id"] in waiters:
                            raise ValueError("invalid/duplicate native request")
                        key = source_to_store[message["source"]]
                        targets = set()
                        for item in message["chunks"]:
                            array = str(Path(item["array"]).resolve())
                            if array not in channel_indices[key]:
                                raise ValueError("native request escapes its inference cache")
                            origin = stores[key].request(item["index_zyx"], remaining=float(message["remaining_base_voxels"]))
                            targets.add((key, origin))
                        waiters[message["request_id"]] = targets
                queued = any(s.pending_tile_key() is not None for s in stores.values())
                if pool is None and queued:
                    if not allow_inference:
                        raise RuntimeError("cached-only replay cannot schedule inference")
                    pool = demand.DemandWorkerPool(stores, readers, device_list,
                        slots_per_gpu=int(settings.get("slots_per_gpu", 2)), input_workers=int(settings.get("input_workers", 8)),
                        input_slots=settings.get("input_slots"),
                        result_slots=settings.get("result_slots"),
                        max_batch_size=max_batch_size,
                        accumulator_workers=settings.get("accumulator_workers"),
                        flush_workers=int(settings.get("flush_workers", 4)),
                        deferred_gpu=adaptive is not None)
                    if adaptive is not None:
                        from .demand_gpu import DemandGpuLease
                        gpu_lease = DemandGpuLease(pool, adaptive)
                    queue_log = cleanup.enter_context((Path(evaluation["output"]) / "inference_queues.jsonl").open("x"))
                    print(f"live-eval buffers={pool.stats['shared_slot_bytes'] / (1 << 30):.2f}GiB "
                          f"input_slots={pool.stats['input_slots']} readers={pool.stats['input_workers']} "
                          f"result_slots={pool.stats['result_slots']} max_batch_size={max_batch_size}", flush=True)
                if pool is not None:
                    if gpu_lease is not None:
                        gpu_lease.tick()
                    pool.pump()
                    if time.monotonic() - last_queue_report >= 1:
                        row = {"elapsed_seconds": time.monotonic() - started, "phase": "tracing",
                               "trace_requests_waiting": len(waiters), **pool.queue_snapshot()}
                        queue_log.write(json.dumps(row) + "\n")
                        print("queues " + demand.format_queue_snapshot(row) + f" trace_waiters={len(waiters)}", flush=True)
                        last_queue_report = time.monotonic()
                if adaptive is not None and authenticated and native.poll() is None and time.monotonic() - last_heartbeat >= 1:
                    connection.sendall(b'{"version":1,"type":"heartbeat"}\n')
                    last_heartbeat = time.monotonic()
                for request_id, targets in list(waiters.items()):
                    if all(stores[key].ready(origin) for key, origin in targets):
                        connection.sendall((json.dumps({"version": 1, "type": "ready", "request_id": request_id}) + "\n").encode())
                        del waiters[request_id]
                if time.monotonic() - last_report > 5:
                    print(f"live-eval elapsed={time.monotonic()-started:.1f}s waiting={len(waiters)} "
                          f"tiles={sum(s.stats['inferred_tiles'] for s in stores.values())} "
                          f"published={sum(s.stats['published_chunks'] for s in stores.values())}", flush=True)
                    last_report = time.monotonic()
            if native.wait(timeout=10) != 0:
                raise RuntimeError("native evaluation failed (see native.log)")
            # Reference-intersection work is explicitly drained even if traces
            # finished before lower-priority prefetch did.
            if pool is not None:
                while pool.busy:
                    if gpu_lease is not None:
                        gpu_lease.tick()
                    pool.pump()
                    if time.monotonic() - last_queue_report >= 1:
                        queue_log.write(json.dumps({"elapsed_seconds": time.monotonic() - started,
                            "phase": "draining", "trace_requests_waiting": 0, **pool.queue_snapshot()}) + "\n")
                        last_queue_report = time.monotonic()
                    time.sleep(0.005)
                queue_log.write(json.dumps({"elapsed_seconds": time.monotonic() - started,
                    "phase": "complete", "trace_requests_waiting": 0, **pool.queue_snapshot()}) + "\n")
            from .replay_metrics import correct_summary
            summary_path = Path(evaluation["output"]) / "summary.json"
            atomic_write(summary_path, correct_summary(json.loads(summary_path.read_text())))
            atomic_write(Path(evaluation["output"]) / "inference_summary.json", {
                "producer_implementation": producer_implementation,
                "wall_seconds": time.monotonic() - started, "worker_pool": pool.stats if pool else None,
                "sources": {key: store.stats for key, store in stores.items()},
                "cache_roots": {key: str(store.root) for key, store in stores.items()},
                "reference_prefetch_chunks": {key: len(v) for key, v in backgrounds.items()}})
        except BaseException as error:
            destination = Path(evaluation["output"])
            destination.mkdir(parents=True, exist_ok=True)
            atomic_write(destination / "inference_incomplete.json", {"error": str(error),
                "producer_implementation": producer_implementation,
                "sources": {key: store.stats for key, store in stores.items()}})
            if connection is not None:
                try:
                    connection.sendall((json.dumps({"version": 1, "type": "error", "message": str(error)}) + "\n").encode())
                except OSError:
                    pass
            if native.poll() is None:
                try:
                    native.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    native.terminate()
                    try:
                        native.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        native.kill()
                        native.wait()
            raise
        finally:
            try:
                if gpu_lease is not None:
                    gpu_lease.close()
            finally:
                try:
                    if pool is not None:
                        pool.close()
                finally:
                    if connection is not None:
                        connection.close()
            log.flush()
            log.seek(0)
            text = log.read()
            destination = Path(evaluation["output"])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "native.log").write_text(text)
            print(text, end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", help="legacy checkpoint configuration only")
    parser.add_argument("--devices", required=True, help="explicit CUDA indices, e.g. 0,1; no CPU production inference")
    parser.add_argument("--native-binary", default="vc_fiber_trace_metric")
    parser.add_argument("--render", action="store_true", help="render reference/traced strips directly after each completed case")
    parser.add_argument("--manager-socket", help="request fixed exclusive GPUs from this manager")
    parser.add_argument("--manager-priority", type=int, help="explicit exclusive-job priority (default100)")
    parser.add_argument("--unmanaged", action="store_true", help="explicitly bypass configured GPU management")
    args = parser.parse_args()
    return run(args.config, args.checkpoint, args.devices, args.native_binary, args.model_config,
               render=args.render, manager_socket=args.manager_socket,
               manager_priority=args.manager_priority, unmanaged=args.unmanaged)


if __name__ == "__main__":
    raise SystemExit(main())
