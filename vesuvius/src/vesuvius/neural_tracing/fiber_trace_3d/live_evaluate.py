"""Standalone native beam replay coordinated with on-demand GPU inference.

Only control descriptors cross the localhost socket. Published predictions use
the ordinary Lasagna/OME-Zarr format and the existing model/tiled finalizers.
"""
from __future__ import annotations

import argparse
from collections import deque
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
import selectors
import shutil
import socket
import subprocess
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


def finalize_native_summary(summary):
    """Placement diagnostics have no trace/restart partitions to correct."""
    if summary.get('backend') == 'failure_search':
        if (not summary.get('completed') or
                summary['evaluated_cases'] + summary['cancelled_cases'] != summary['cases']):
            raise ValueError('incomplete failure-search summary')
        return summary
    if summary.get('backend') == 'anchors':
        if not summary.get('completed') or len(summary['anchor_cases']) != summary['cases']:
            raise ValueError('incomplete anchor summary')
        return summary
    from .replay_metrics import correct_summary
    return correct_summary(summary)


def bind_fiberlet_identity(entry, source_identity, model_identity, manifest):
    """Canonical graph-cache binding shared by fresh and historical replay."""
    entry['fiberlet_source_identity'] = {
        'source_volume': source_identity, 'fiber_prediction': model_identity}
    entry['fiberlet_cache_root'] = str(Path(manifest).parent / 'fiberlets')


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


from .reference_chunk_geometry import reference_chunks, reference_chunk_job


def tube_chunks(points_base, margin_base, factor, shape, chunk):
    """Output chunks within ``margin_base`` base voxels of a reference polyline.

    ``reference_chunks`` returns only the chunks the line itself crosses, which
    is what a trace that follows the reference reads. A trace that steps
    differently reads anywhere the evaluator lets it drift before restarting,
    so a parameter search prefetches the whole acceptance tube once and then
    replays every variant without inference. Membership is exact per sample:
    a chunk is included when the distance from a sample point to its axis
    aligned box is at most the margin. Samples are the polyline vertices plus
    midpoints subdivided until spacing is below a quarter chunk, so no chunk
    within the margin of the line can be skipped between samples by more
    than that spacing; the margin should include that slack.
    """
    if not (margin_base >= 0) or not math.isfinite(margin_base):
        raise ValueError("tube margin must be finite and non-negative")
    points = np.asarray(points_base, dtype=np.float64) / factor
    radius = float(margin_base) / factor
    if len(points) == 0:
        return set()
    if len(points) == 1:
        samples = points
    else:
        pieces = []
        spacing = chunk / 4.0
        for left, right in zip(points[:-1], points[1:]):
            n = max(1, int(math.ceil(float(np.linalg.norm(right - left)) / spacing)))
            pieces.append(left + (right - left) * (np.arange(n, dtype=np.float64) / n)[:, None])
        pieces.append(points[-1:])
        samples = np.concatenate(pieces)
    samples = samples[:, ::-1]  # XYZ points to ZYX lattice order
    extent = np.asarray(shape, dtype=np.float64)
    # Chunks entirely outside the array cannot exist; clip candidate windows.
    n_chunks = np.ceil(extent / chunk).astype(np.int64)
    result = set()
    lo_all = np.maximum(np.floor((samples - radius) / chunk).astype(np.int64), 0)
    hi_all = np.minimum(np.floor((samples + radius) / chunk).astype(np.int64), n_chunks - 1)
    for sample, lo, hi in zip(samples, lo_all, hi_all):
        if np.any(hi < lo):
            continue
        grids = np.meshgrid(*(np.arange(a, b + 1) for a, b in zip(lo, hi)), indexing="ij")
        index = np.stack([g.ravel() for g in grids], axis=1)
        box_lo = index * chunk
        box_hi = np.minimum(box_lo + chunk, extent) - 1
        nearest = np.clip(sample, box_lo, box_hi)
        inside = np.sum((nearest - sample) ** 2, axis=1) <= radius * radius
        result.update(tuple(int(v) for v in row) for row in index[inside])
    return result


def _complete_inspection_gallery(directory):
    """Match the native resume validator without trusting partial galleries."""
    directory = Path(directory)
    try:
        metadata = json.loads((directory / "render.json").read_text())
        pages = metadata.get("pages")
        return (metadata.get("format") == "vc_fiber_inspection" and
                type(pages) is int and pages > 0 and
                all((directory / f"strips_{page}.jpg").is_file()
                    for page in range(pages)))
    except (OSError, ValueError, TypeError):
        return False


def resume_visualization_selection(output, source_ids, limit):
    """Return the native round-robin selection and its saved center paths."""
    if type(limit) is not int or limit < 1:
        raise ValueError("resume_render_limit must be a positive integer")
    groups = {source_id: [] for source_id in source_ids}
    cases_root = Path(output) / "cases"
    case_dirs = [path for path in cases_root.iterdir()
                 if path.is_dir() and path.name.isdigit()]
    for case_dir in sorted(case_dirs, key=lambda p: int(p.name)):
        artifact_path = case_dir / "trace.json"
        if not artifact_path.is_file() or _complete_inspection_gallery(case_dir / "rendered"):
            continue
        artifact = json.loads(artifact_path.read_text())
        provenance = artifact.get("provenance", {})
        case_index = provenance.get("case_index")
        source_id = provenance.get("source_id")
        points = artifact.get("reference_points_base_xyz")
        if case_index != int(case_dir.name) or source_id not in groups or not isinstance(points, list):
            raise ValueError(f"invalid visualization-resume trace artifact: {artifact_path}")
        groups[source_id].append((case_index, points))
    selected = []
    for rank in itertools.count():
        added = False
        for source_id in source_ids:
            if rank < len(groups[source_id]):
                selected.append(groups[source_id][rank])
                added = True
                if len(selected) == limit:
                    return selected
        if not added:
            return selected


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
    adapter_options = {}
    if settings.get("embedding_bank") is not None:
        adapter_options["embedding_bank"] = settings["embedding_bank"]
    adapter_class = inference_adapter.FiberTrace3DPredictAdapter
    if settings.get("tta"):
        from .tta import TtaPredictAdapter
        adapter_class = TtaPredictAdapter
    adapter = adapter_class(effective, checkpoint=snapshot,
        level=level + power, scaledown=(1 << level) * geometry.scale, inference_scaledown=geometry.scale,
        chunk_size=geometry.chunk, product_prefix="fiber", zarr_path_prefix=cache / "fiber",
        emit_embeddings=bool(settings.get("emit_embeddings", False)),
        **adapter_options)
    # Construction restores checkpoint config; reapply the resolved precision.
    adapter.config["training"] = {**adapter.config.get("training", {}),
        "mixed_precision": effective["training"]["mixed_precision"]}
    if 'allow_tf32' in settings:
        adapter.config['inference'] = {**adapter.config.get('inference', {}),
                                       'allow_tf32': settings['allow_tf32']}
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


def resolve_inference_model(cache_root, checkpoint, model_config, settings, device_list,
                            *, validate_devices=True):
    """Resolve numerical identity once for either channel role."""
    snapshot, checkpoint_digest = freeze_checkpoint(Path(checkpoint), cache_root)
    effective, _ = _checkpoint_config_for_inference(snapshot, model_config)
    settings = dict(settings)
    if settings.get('tta'):
        if settings.get('allow_tf32', False) is not False:
            raise ValueError('TTA requires allow_tf32=False')
        settings['allow_tf32'] = False
    precision, _ = _resolve_inference_precision(
        settings.get("precision", "auto"), checkpoint=snapshot,
        devices=device_list, validate_devices=validate_devices)
    effective = dict(effective)
    effective["training"] = {**effective.get("training", {}), "mixed_precision": precision}
    if 'allow_tf32' in settings:
        if type(settings['allow_tf32']) is not bool:
            raise ValueError('inference.allow_tf32 must be boolean')
        effective['inference'] = {**effective.get('inference', {}), 'allow_tf32': settings['allow_tf32']}
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
    if settings.get("tta"):
        from .tta import SCHEMA, VARIANTS
        if settings['tta'] not in VARIANTS or precision != 'off' or max_batch_size != 1:
            raise ValueError('TTA requires valid variant, precision off and max_batch_size1')
        model_identity['tta'] = SCHEMA  # Selection is a view, not a new prediction.
    return dict(snapshot=snapshot, effective=effective, settings=settings,
                identity=model_identity, tile=tile, overlap=overlap, border=border,
                scale=scale, chunk=chunk, power=power)


def demand_gpu_manager_settings(config, devices, *, manager_socket=None,
                                manager_priority=None, unmanaged=False):
    """Return runtime-only adaptive GPU settings, or ``None`` when unmanaged.

    These settings deliberately stay outside inference/model identity.  The
    demand coordinator consumes them only after a missing tile has finished
    loading, so cached tracing never registers with or acquires from the GPU
    manager.
    """
    manager = dict(config.get("gpu_manager", {}))
    if unmanaged or not (manager.get("enabled", False) or manager_socket is not None):
        return None
    manager["devices"] = devices
    if manager_socket is not None:
        manager["socket"] = manager_socket
    if manager_priority is not None:
        manager["priority"] = manager_priority
    budget = manager.get("vram_bytes_per_gpu")
    if type(budget) is not int or budget <= (2 << 30):
        raise ValueError(
            "demand-managed live inference requires gpu_manager."
            "vram_bytes_per_gpu > 2GiB")
    return manager


def runtime_inference_settings(settings, adaptive_manager):
    """Keep manager lifecycle controls runtime-only and honor unmanaged mode."""
    runtime = dict(settings)
    runtime.pop("demand_gpu_manager", None)
    if adaptive_manager is not None:
        runtime["demand_gpu_manager"] = dict(adaptive_manager)
    return runtime


def _reference_worker_session():
    if hasattr(os, "setsid"):
        os.setsid()


def run(config_path, checkpoint, devices, native_binary, model_config=None, *, render=False,
        manager_socket=None, manager_priority=None, unmanaged=False, runtime_overrides=None,
        worker_factory=None, cancel_event=None, on_process_start=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    runtime_overrides = dict(runtime_overrides or {})
    unknown_runtime = set(runtime_overrides) - {"resume_visualization", "resume_render_limit"}
    if unknown_runtime:
        raise ValueError(f"unknown evaluation runtime overrides: {sorted(unknown_runtime)}")
    config.update(runtime_overrides)
    adaptive_manager = demand_gpu_manager_settings(config, devices,
        manager_socket=manager_socket, manager_priority=manager_priority,
        unmanaged=unmanaged)
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
    settings = dict(config.get("inference", {}))
    trace_args = config.get("trace_args", [])
    mixed = config.get("embedding_inference")
    if worker_factory is not None and mixed is not None:
        raise ValueError("resident training trace cannot use a different embedding checkpoint")
    if settings.get('tta') and mixed is not None:
        raise ValueError('TTA pilot does not support mixed embedding inference')
    if mixed is not None and (not isinstance(mixed, dict) or not mixed.get("checkpoint")):
        raise ValueError("embedding_inference requires a checkpoint")
    if "--embedding-weight" in trace_args:
        weight = float(trace_args[trace_args.index("--embedding-weight") + 1])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("embedding weight must be finite and nonnegative")
        if weight > 0 and mixed is None:
            settings["emit_embeddings"] = True
    device_list = tuple(torch.device(f"cuda:{int(v)}") for v in devices.split(","))
    if not device_list or len(set(device_list)) != len(device_list):
        raise ValueError("select a nonempty distinct comma-separated CUDA device list")
    primary = resolve_inference_model(cache_root, checkpoint, model_config, settings, device_list,
        validate_devices=adaptive_manager is None and worker_factory is None)
    models = [(primary, prediction_cache_root)]
    if mixed is not None:
        unknown = set(mixed) - {"checkpoint", "checkpoint_sha256", "model_config", "inference", "inference_cache_namespace"}
        if unknown:
            raise ValueError(f"unknown embedding_inference options: {sorted(unknown)}")
        embedding_settings = {**settings, **mixed.get("inference", {}), "emit_embeddings": True}
        alternate = resolve_inference_model(cache_root, location(base, mixed["checkpoint"]),
            location(base, mixed["model_config"]) if mixed.get("model_config") else None,
            embedding_settings, device_list, validate_devices=adaptive_manager is None)
        if mixed.get("checkpoint_sha256") and mixed["checkpoint_sha256"] != alternate["identity"]["checkpoint_sha256"]:
            raise ValueError("embedding checkpoint SHA256 mismatch")
        if alternate["scale"] != primary["scale"]:
            raise ValueError("mixed embeddings must use the same output lattice/scaledown")
        models.append((alternate, inference_cache_root(cache_root,
            mixed.get("inference_cache_namespace", config.get("inference_cache_namespace")))))
    power = primary["power"]
    evaluation = dict(config)
    evaluation["reference_plan"] = references
    evaluation["annotation_sha256"] = annotation_hashes
    if not isinstance(config.get("render", False), bool):
        raise ValueError("render must be a boolean")
    evaluation["render"] = render or config.get("render", False)
    evaluation["output"] = location(base, config["output"])
    if Path(evaluation["output"]).exists() and not config.get("resume_visualization", False):
        raise ValueError("evaluation output already exists")
    if config.get("resume_visualization", False) and not Path(evaluation["output"]).is_dir():
        raise ValueError("visualization resume requires an existing evaluation output")
    evaluation["sources"] = []
    source_to_store, stores, readers, channel_indices, backgrounds = {}, {}, {}, {}, {}
    resume_points = {}
    if config.get("resume_visualization", False):
        selected = resume_visualization_selection(evaluation["output"],
            [source["id"] for source in config["sources"]],
            int(config.get("resume_render_limit", 24)))
        for case_index, points in selected:
            artifact = json.loads((Path(evaluation["output"]) / "cases" /
                str(case_index) / "trace.json").read_text())
            resume_points.setdefault(artifact["provenance"]["source_id"], []).append(points)
        evaluation["resume_render_cases"] = [case_index for case_index, _ in selected]
    pool_groups = {}
    started = time.monotonic()
    native = None
    connection = None
    pool = None
    with ExitStack() as cleanup:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        prefetch_workers = min(32, max(1, int(config.get('workers', 32))))
        prefetch_pool = cleanup.enter_context(ProcessPoolExecutor(
            max_workers=prefetch_workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=_reference_worker_session if on_process_start is not None else None))
        registered = set()
        def submit_reference(*args):
            job = prefetch_pool.submit(reference_chunk_job, *args)
            if on_process_start is not None:
                pids = set(prefetch_pool._processes) - registered
                on_process_start(sorted(pids))
                registered.update(pids)
            return job
        prefetch_jobs = {}
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
            source_identity = {"locator": input_path, "level": level, "shape": volume_shape,
                "base_shape": base_shape, "metadata": root_attrs,
                "revision": source.get("source_revision"), "base_voxel_size_um": source["base_voxel_size_um"],
                "coordinate_space": source.get("coordinate_space")}
            source_key = json_digest(source_identity)
            role_keys, manifests = [], []
            for model, model_cache_root in models:
                geometry = demand.DemandGeometry(volume_shape, model["tile"], model["overlap"],
                    model["border"], model["scale"], model["chunk"])
                cache, stored_model_identity = resolve_prediction_cache(
                    model_cache_root / source_key, source_identity, model["identity"], geometry, model["settings"])
                key = source_key if mixed is None else str(cache.resolve())
                role_keys.append(key)
                manifests.append(cache / "inference.lasagna.json")
                print(f"live-eval cache source={source_id} existing={(cache / 'cache_identity.json').is_file()} path={cache}", flush=True)
                if key not in stores:
                    store = cleanup.enter_context(create_demand_store(cache, source_identity, stored_model_identity,
                        geometry, model["effective"], model["snapshot"], level, model["settings"]))
                    stores[key] = store
                    readers[key] = NativeTileReader(sampler, geometry, sampler.volume.dtype)
                    channel_indices[key] = {str((Path(c.relative_path) / str(p.level)).resolve()): geometry
                        for p in store.products for c in p.channels}
                    backgrounds[key] = set()
                    group_key = json_digest(model["identity"])
                    group = pool_groups.setdefault(group_key, {"keys": [], "settings": model["settings"]})
                    group["keys"].append(key)
                if config.get("resume_visualization", False):
                    for points in resume_points.get(source_id, ()):
                        job = submit_reference((points,
                            (1 << level) * model["scale"], geometry.output_shape, model["chunk"]))
                        prefetch_jobs[job] = key
                else:
                    for fiber in references["sources"][source_index]["fibers"]:
                        for run in fiber["runs"]:
                            job = submit_reference((run["points_base_xyz"],
                                (1 << level) * model["scale"], geometry.output_shape, model["chunk"]))
                            prefetch_jobs[job] = key
            source_to_store[source_id] = role_keys if mixed is not None else role_keys[0]
            entry = dict(source)
            # Scientific producer identity, independent of manifests' runtime locations.
            graph_identity = models[0][0]["identity"]
            if settings.get('tta'):
                graph_identity = {**graph_identity, 'tta_selected_product': settings['tta']}
            bind_fiberlet_identity(entry, source_identity, graph_identity, manifests[0])
            entry["fiber_manifest"] = str(manifests[0])
            if settings.get('tta'):
                from .tta import selected_manifest
                entry['fiber_manifest'] = str(selected_manifest(manifests[0], settings['tta']))
            if mixed is not None:
                from .mixed_inference import write_mixed_manifest
                entry["fiber_manifest"] = str(write_mixed_manifest(
                    cache_root / "live_inference" / "_composites", *manifests))
                entry["inference_roles"] = {"presence_nx_ny": str(manifests[0]),
                                            "embedding": str(manifests[1])}
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
        for done, job in enumerate(as_completed(prefetch_jobs), 1):
            backgrounds[prefetch_jobs[job]].update(job.result())
            if done % 100 == 0 or done == len(prefetch_jobs):
                print(f'reference-prefetch preparation={done}/{len(prefetch_jobs)} workers={prefetch_workers}', flush=True)
        prefetch_pool.shutdown(wait=True)
        if any(sha256_file(Path(path)) != digest for path, digest in annotation_hashes.items()):
            raise ValueError("annotation changed before native evaluation launch")
        runtime_settings = runtime_inference_settings(settings, adaptive_manager)
        return coordinate_native(evaluation, binary, stores, source_to_store, channel_indices,
            backgrounds, readers, device_list, runtime_settings, started=started,
            pool_groups=pool_groups if mixed is not None else None,
            worker_factory=worker_factory, cancel_event=cancel_event, on_process_start=on_process_start)


def create_managed_demand_pool(stores, readers, device_list, settings, pool_groups=None,
                               worker_factory=None, on_process_start=None):
    """Shared worker/lease setup for tracing and explicit inspection fills."""
    from .mixed_inference import ModelWorkerPools
    from .demand_gpu import DemandGpuLease
    adaptive = settings.get("demand_gpu_manager")
    factory = demand.DemandWorkerPool if pool_groups is None else (
        lambda *args, **kwargs: ModelWorkerPools(*args, groups=pool_groups, **kwargs))
    # Chunk-major arena evaluations need enough owner-local hot records to
    # avoid spilling every tile contribution. Older configs predate these
    # knobs, so retain the proven full-evaluation defaults unless either
    # override is explicit.
    accumulator_workers = settings.get("accumulator_workers")
    accumulator_hot_slots = settings.get("accumulator_hot_slots")
    if accumulator_workers is None and accumulator_hot_slots is None:
        accumulator_workers = 128
        accumulator_hot_slots = 163840
    pool = factory(stores, readers, device_list,
        slots_per_gpu=int(settings.get("slots_per_gpu", 2)),
        input_workers=int(settings.get("input_workers", 8)),
        input_slots=settings.get("input_slots"), result_slots=settings.get("result_slots"),
        max_batch_size=settings.get("max_batch_size", 1),
        accumulator_workers=accumulator_workers,
        accumulator_hot_slots=accumulator_hot_slots,
        accumulator_spill_segment_slots=int(settings.get("accumulator_spill_segment_slots", 256)),
        flush_workers=int(settings.get("flush_workers", 4)),
        profile_pipeline=bool(settings.get("profile_pipeline", False)),
        deferred_gpu=adaptive is not None,
        **({"worker_factory": worker_factory} if worker_factory is not None else {}),
        **({"on_process_start": on_process_start} if on_process_start is not None else {}))
    try:
        lease = DemandGpuLease(pool, adaptive) if adaptive is not None else None
    except BaseException:
        pool.close()
        raise
    return pool, lease


def coordinate_native(evaluation, binary, stores, source_to_store, channel_indices,
                      backgrounds, readers, device_list, settings, *, started=None, allow_inference=True,
                      pool_groups=None, cancel_event=None, inference_session=None,
                      worker_factory=None, on_process_start=None):
    """Shared authenticated control loop for live inference and cached-only replay."""
    started = time.monotonic() if started is None else started
    max_batch_size = settings.get("max_batch_size", 1)
    if type(max_batch_size) is not int or max_batch_size < 1:
        raise ValueError("inference.max_batch_size must be a positive integer")
    producer_implementation = implementation_identity()  # Run provenance only, never a cache key.
    native = connection = None
    pool = inference_session.pool if inference_session is not None else None
    gpu_lease = inference_session.gpu_lease if inference_session is not None else None
    reused_pool = pool is not None
    succeeded = False
    initial_stats = {key: dict(store.stats) for key, store in stores.items()}
    if pool is not None and pool.busy:
        raise RuntimeError('borrowed inference session must be drained before replay')
    adaptive = settings.get("demand_gpu_manager")
    def check_cancelled():
        if cancel_event is not None and cancel_event.is_set():
            from concurrent.futures import CancelledError
            raise CancelledError('trace extraction cancelled')
    check_cancelled()
    # Reference prefetch is normally requested in full at hello. An optional
    # cap feeds it in bounded waves instead: every requested chunk holds
    # accumulator records until it publishes, so requesting a whole reference
    # tube at once exhausts the hot arena and stalls accumulation with the GPUs
    # idle. The cap bounds requested-but-unpublished background chunks.
    background_cap = settings.get("background_chunk_cap")
    if background_cap is None:
        background_cap = float("inf")
    elif type(background_cap) is not int or background_cap < 1:
        raise ValueError("inference.background_chunk_cap must be a positive integer")
    background_queues = {}
    background_inflight = {key: deque() for key in stores}

    def background_remaining():
        return any(background_queues.values())

    def top_up_backgrounds():
        inflight = 0
        for key, origins in background_inflight.items():
            while origins and stores[key].ready(origins[0]):
                origins.popleft()
            inflight += len(origins)
        for key, queue in background_queues.items():
            while queue and inflight < background_cap:
                origin = stores[key].request(queue.popleft(), background=True)
                if not stores[key].ready(origin):
                    background_inflight[key].append(origin)
                    inflight += 1
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
        # Runtime-only environment for the native evaluator (for example an
        # alternative allocator via LD_PRELOAD); it never reaches the config
        # the native process reads and is not part of any cache identity.
        native_environment = evaluation.get("native_environment", {})
        if not isinstance(native_environment, dict) or any(
                not isinstance(k, str) or not isinstance(v, str) for k, v in native_environment.items()):
            raise ValueError("native_environment must map strings to strings")
        native = subprocess.Popen([str(binary), "replay-set", "--config", str(runtime_config)],
                                  stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=on_process_start is not None,
                                  env={**os.environ, **native_environment} if native_environment else None)
        try:
            if on_process_start is not None:
                on_process_start([native.pid])
            # Observe startup failure instead of waiting the whole accept timeout.
            listener.settimeout(0.1)
            deadline = time.monotonic() + 120
            while connection is None:
                check_cancelled()
                if native.poll() not in (None, 0):
                    raise RuntimeError("native evaluator failed during source validation (see native.log)")
                if time.monotonic() > deadline:
                    raise TimeoutError("native evaluator did not connect")
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    pass
            connection.settimeout(1)
            # Parallel model jobs can exceed select()'s FD_SETSIZE even with a
            # single watched socket. epoll/kqueue have no such descriptor limit.
            selector = cleanup.enter_context(selectors.DefaultSelector())
            selector.register(connection, selectors.EVENT_READ)
            pending = b""
            authenticated = False
            waiters = {}
            last_report = time.monotonic()
            queue_log = None
            def open_queue_log():
                nonlocal queue_log
                name = (f"inference_queues.resume.{os.getpid()}.{time.time_ns()}.jsonl"
                        if evaluation.get("resume_visualization") else "inference_queues.jsonl")
                queue_log = cleanup.enter_context((Path(evaluation["output"]) / name).open("x"))
                print(f"live-eval shared_buffers={pool.stats['shared_slot_bytes'] / (1 << 30):.2f}GiB "
                      f"hot_capacity={pool.stats.get('accumulator_hot_capacity_bytes', 0) / (1 << 30):.2f}GiB "
                      f"input_slots={pool.stats['input_slots']} readers={pool.stats['input_workers']} "
                      f"result_slots={pool.stats['result_slots']} max_batch_size={max_batch_size} "
                      f"reused_pool={reused_pool}", flush=True)
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
                check_cancelled()
                if not allow_inference and time.monotonic() - started > settings.get("cached_trace_timeout_seconds", 300):
                    raise TimeoutError("cached trace profiling exceeded its bounded duration")
                if selector.select(0.005):
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
                            background_queues = {key: deque(sorted(indices)) for key, indices in backgrounds.items()}
                            if math.isinf(background_cap):
                                demand.prepare_background_requests(
                                    stores, background_queues,
                                    workers=int(settings.get("input_workers", 128)),
                                    check_cancelled=check_cancelled)
                            top_up_backgrounds()
                            continue
                        if message.get("type") != "demand" or message["request_id"] in waiters:
                            raise ValueError("invalid/duplicate native request")
                        targets = set()
                        from .mixed_inference import request_store_key
                        for item in message["chunks"]:
                            array = str(Path(item["array"]).resolve())
                            key = request_store_key(source_to_store[message["source"]], array, channel_indices)
                            origin = stores[key].request(item["index_zyx"], remaining=float(message["remaining_base_voxels"]))
                            targets.add((key, origin))
                        waiters[message["request_id"]] = targets
                if authenticated:
                    top_up_backgrounds()
                queued = any(s.pending_tile_key() is not None for s in stores.values())
                if pool is None and queued:
                    if not allow_inference:
                        raise RuntimeError("cached-only replay cannot schedule inference")
                    pool, gpu_lease = create_managed_demand_pool(
                        stores, readers, device_list, settings, pool_groups,
                        worker_factory=worker_factory, on_process_start=on_process_start)
                    if inference_session is not None:
                        inference_session.pool = pool
                        inference_session.gpu_lease = gpu_lease
                    open_queue_log()
                if pool is not None:
                    # Native connects before exclusively creating its output
                    # directory. A first demand confirms that setup completed;
                    # an idle retained pool must not create outputs ahead of it.
                    if queue_log is None and waiters:
                        open_queue_log()
                    if gpu_lease is not None:
                        gpu_lease.tick()
                    pool.pump()
                    if queue_log is not None and time.monotonic() - last_queue_report >= 1:
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
            if pool is None and background_remaining():
                raise RuntimeError("reference prefetch remains queued without an inference pool")
            if pool is not None:
                if queue_log is None:
                    open_queue_log()
                while pool.busy or background_remaining():
                    check_cancelled()
                    top_up_backgrounds()
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
            summary_path = Path(evaluation["output"]) / "summary.json"
            if pool is not None and settings.get("profile_pipeline", False) and pool_groups is None:
                atomic_write(Path(evaluation["output"]) / "pipeline_profile.json", {
                    "scope": "session cumulative" if inference_session is not None else "batch",
                    "worker_samples": pool.profile_samples,
                    "stage_samples": pool.stage_samples,
                    "accumulator_seconds": pool.accumulator_samples,
                    "accumulator_queue_plus_ack_seconds": pool.accumulator_queue_samples,
                    "note": "Overlapping stage times; worker samples are per tile (batch times divided by batch size)."})
            atomic_write(summary_path, finalize_native_summary(json.loads(summary_path.read_text())))
            atomic_write(Path(evaluation["output"]) / "inference_summary.json", {
                "producer_implementation": producer_implementation,
                "wall_seconds": time.monotonic() - started, "worker_pool": pool.stats if pool else None,
                "worker_pool_scope": "session cumulative" if inference_session is not None else "batch",
                "reused_worker_pool": reused_pool,
                "sources": {key: {k: v-initial_stats[key].get(k,0) for k,v in store.stats.items()}
                            for key, store in stores.items()},
                "cache_roots": {key: str(store.root) for key, store in stores.items()},
                "reference_prefetch_chunks": {key: len(v) for key, v in backgrounds.items()}})
            succeeded = True
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
                if inference_session is not None and not succeeded:
                    inference_session.close()
                elif inference_session is None and gpu_lease is not None:
                    gpu_lease.close()
            finally:
                try:
                    if inference_session is None and pool is not None:
                        pool.close()
                finally:
                    if connection is not None:
                        connection.close()
            log.flush()
            log.seek(0)
            text = log.read()
            destination = Path(evaluation["output"])
            destination.mkdir(parents=True, exist_ok=True)
            name = (f"native.resume.{os.getpid()}.{time.time_ns()}.log"
                    if evaluation.get("resume_visualization") else "native.log")
            (destination / name).write_text(text)
            print(text, end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", help="legacy checkpoint configuration only")
    parser.add_argument("--devices", required=True, help="explicit CUDA indices, e.g. 0,1; no CPU production inference")
    parser.add_argument("--native-binary", default="vc_fiber_trace_metric")
    parser.add_argument("--render", action="store_true", help="render reference/traced strips directly after each completed case")
    parser.add_argument("--manager-socket", help="request GPUs on demand from this manager")
    parser.add_argument("--manager-priority", type=int, help="explicit demand-inference priority (default100)")
    parser.add_argument("--unmanaged", action="store_true", help="explicitly bypass configured GPU management")
    args = parser.parse_args()
    return run(args.config, args.checkpoint, args.devices, args.native_binary, args.model_config,
               render=args.render, manager_socket=args.manager_socket,
               manager_priority=args.manager_priority, unmanaged=args.unmanaged)


if __name__ == "__main__":
    raise SystemExit(main())
