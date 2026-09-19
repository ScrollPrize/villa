"""Short, manager-isolated readiness comparison using immutable published chunks."""
from copy import deepcopy
from collections import OrderedDict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

import numpy as np

from lasagna import tiled_predict3d as tiled
from lasagna.inference_provenance import atomic_write, json_digest, sha256_file
from vesuvius.neural_tracing.fiber_trace_2d.fiber_json import load_fiber_file
from .live_evaluate import coordinate_native


_trial_cancel_event = None
_trial_sessions = OrderedDict()
_trial_session_limit = 1


def reference_backgrounds(evaluation, stores, mapping):
    """Build low-priority centerline chunk sets for a bound reference plan.

    ``evaluation["reference_prefetch_margin_base_voxels"]`` widens the set to
    the whole tube within that many base voxels of each reference, so that a
    tracer-parameter search can prefill everything a variant may read.
    """
    from lasagna.demand_predict3d import DemandGeometry
    from .live_evaluate import reference_chunks, tube_chunks
    backgrounds = {key: set() for key in stores}
    plan = evaluation.get("reference_plan")
    if not plan:
        return backgrounds
    margin = evaluation.get("reference_prefetch_margin_base_voxels")
    if margin is not None and (isinstance(margin, bool) or not isinstance(margin, (int, float)) or margin <= 0):
        raise ValueError("reference_prefetch_margin_base_voxels must be a positive number")
    cache_geometry = {}
    for source in plan["sources"]:
        key = mapping[source["source_id"]]
        if key not in cache_geometry:
            persisted = json.loads((Path(stores[key].root) / "cache_identity.json").read_text())
            geometry = DemandGeometry(**persisted["geometry"])
            factor = (1 << int(persisted["identity"]["source"]["level"])) * geometry.scale
            cache_geometry[key] = geometry, factor
        geometry, factor = cache_geometry[key]
        for fiber in source["fibers"]:
            for run in fiber["runs"]:
                backgrounds[key].update(reference_chunks(
                    run["points_base_xyz"], factor, geometry.output_shape, geometry.chunk))
                if margin is not None:
                    backgrounds[key].update(tube_chunks(
                        run["points_base_xyz"], float(margin), factor, geometry.output_shape, geometry.chunk))
    return backgrounds


def initialize_trial_worker(cancel_event, session_limit=1, vram_budget=None):
    """Spawned CPU coordinator; GPU inference still uses the shared worker pool."""
    global _trial_cancel_event, _trial_session_limit
    _trial_cancel_event = cancel_event
    _trial_session_limit = session_limit
    if vram_budget is not None:
        os.environ['FIBER_INFERENCE_VRAM_BUDGET_BYTES'] = str(vram_budget)
    from multiprocessing.util import Finalize
    Finalize(None, close_trial_worker, exitpriority=10)


def close_trial_worker():
    while _trial_sessions:
        _, session = _trial_sessions.popitem(last=False)
        session.close()


def execute_trial_replay(evaluation, binary, checkpoint=None, devices='0', source_catalog=None):
    """Process-pool entrypoint, keeping Python setup outside the main GIL.

    Each call retains existing cache identity and shared native/inference logic.
    Results stay in the evaluation directory, not duplicated over IPC.
    """
    if _trial_cancel_event is None:
        raise RuntimeError('trial worker must be initialized with cancellation state')
    sources = source_catalog or evaluation['sources']
    caches = {str(Path(s['fiber_manifest']).resolve().parent) for s in sources}
    bindings = {s['id']:str(Path(s['fiber_manifest']).resolve()) for s in sources}
    key = json_digest(dict(bindings=bindings,settings=evaluation['inference'],devices=devices))
    session = _trial_sessions.get(key)
    if session is not None and session.closed:
        del _trial_sessions[key]
        session = None
    if session is None and checkpoint is None:
        run_cached_config(evaluation, binary, cancel_event=_trial_cancel_event)
        return
    if session is None:
        # Idle retained sessions still own cache writer locks. Co-location alone
        # is not sufficient: evict all overlapping owners before opening another.
        for old_key, old in list(_trial_sessions.items()):
            if caches.intersection(old.caches):
                old.close()
                del _trial_sessions[old_key]
        while len(_trial_sessions)>=_trial_session_limit:
            _, old = _trial_sessions.popitem(last=False)
            old.close()
        catalog_evaluation = dict(evaluation,sources=sources)
        session = CachedReplaySession(catalog_evaluation,binary,checkpoint,
            devices=devices,cancel_event=_trial_cancel_event)
        session.caches = caches
        _trial_sessions[key] = session
        print(f'short-benchmark inference-session opened caches={len(caches)} retained={len(_trial_sessions)}',flush=True)
    else:
        _trial_sessions.move_to_end(key)
        print(f'short-benchmark inference-session reused caches={len(caches)}',flush=True)
    # Historical measured batches reserve ~50GiB/GPU. Keep CPU/mmap sessions
    # resident, but only one model's CUDA workers per lane at a time.
    for other in _trial_sessions.values():
        if other is not session:
            other.pause_gpu()
    try:
        session.run(evaluation)
    except BaseException:
        session.close()
        _trial_sessions.pop(key,None)
        raise


def require_single_model_profile(evaluation):
    if evaluation.get("embedding_inference") or any(s.get("inference_roles") for s in evaluation["sources"]):
        raise ValueError("mixed-model caches must use live_evaluate with the original embedding_inference configuration; "
                         "single-model cached profiling/extension is not supported")


@contextmanager
def _cached_replay_resources(evaluation, binary, checkpoint, *, gpu_manager=None, devices='0', cancel_event=None):
    """Continue an existing inference cache through shared demand publication.

    Unlike a fresh experiment namespace, this explicitly validates and locks
    the existing cache. The numerical adapter/store/control loop are unchanged.
    Caller must own its GPU grant, or supply a demand-scoped manager request.
    """
    require_single_model_profile(evaluation)
    from contextlib import ExitStack
    import subprocess
    import torch
    from lasagna.demand_predict3d import DemandGeometry
    from .live_evaluate import (Vc3dCoordinateSampler, NativeTileReader,
        create_demand_store, native_reader_options,
        input_root_attributes, validate_input_lattice)
    from .inference_cache import validate_cached_model
    binary = Path(binary)
    cache_root = subprocess.check_output([str(binary.with_name("vc_cache_root"))], text=True).strip()
    settings = dict(evaluation["inference"])
    settings.pop("demand_gpu_manager", None)
    if gpu_manager is not None:
        settings["demand_gpu_manager"] = gpu_manager
    stores, readers, mapping, channels = {}, {}, {}, {}
    checkpoint_digest = sha256_file(checkpoint)
    with ExitStack() as cleanup:
        for source in evaluation["sources"]:
            if cancel_event is not None and cancel_event.is_set():
                from concurrent.futures import CancelledError
                raise CancelledError('trace extraction cancelled')
            cache = Path(source["fiber_manifest"]).parent
            key = str(cache)
            if key in stores:
                mapping[source["id"]] = key
                continue
            persisted = json.loads((cache / "cache_identity.json").read_text())
            identity = persisted["identity"]
            model, original = identity["model"], identity["source"]
            settings = validate_cached_model(model, checkpoint_digest, settings)
            geometry = DemandGeometry(**persisted["geometry"])
            sampler = Vc3dCoordinateSampler(original["locator"], level=original["level"],
                cache_root=cache_root, **native_reader_options(settings))
            attrs = input_root_attributes(sampler.volume)
            validate_input_lattice(attrs, original["level"])
            if (list(sampler.volume.shape_at(original["level"])) != original["shape"] or
                list(sampler.volume.shape_at(0)) != original["base_shape"] or attrs != original["metadata"]):
                raise ValueError("source metadata changed since cached inference")
            stores[key] = cleanup.enter_context(create_demand_store(cache, original, model, geometry,
                model["config"], checkpoint, original["level"], settings))
            readers[key] = NativeTileReader(sampler, geometry, sampler.volume.dtype)
            mapping[source["id"]] = key
            channels[key] = {str((Path(c.relative_path)/str(p.level)).resolve()): geometry
                            for p in stores[key].products for c in p.channels}
        yield (stores,mapping,channels,readers,
               tuple(torch.device(f"cuda:{int(d)}") for d in str(devices).split(',')),settings)


class CachedReplaySession:
    """One validated producer's readers, mmap stores and GPU workers, reused."""
    def __init__(self,evaluation,binary,checkpoint,**kwargs):
        self.binary = Path(binary)
        self.cancel_event = kwargs.get('cancel_event')
        self.pool = self.gpu_lease = None
        self.gpu_paused = False
        self.closed = False
        self.owner = _cached_replay_resources(evaluation,binary,checkpoint,**kwargs)
        self.resources = self.owner.__enter__()
        self.source_bindings = {s['id']:str(Path(s['fiber_manifest']).resolve()) for s in evaluation['sources']}

    def run(self,evaluation):
        if self.closed:
            raise RuntimeError('inference session already closed')
        for source in evaluation['sources']:
            if self.source_bindings.get(source['id']) != str(Path(source['fiber_manifest']).resolve()):
                raise ValueError('trial sources differ from retained inference session')
        stores,mapping,channels,readers,devices,settings = self.resources
        if self.gpu_paused and self.pool is not None:
            self.pool.start_gpu(devices)
            self.pool.authorize_gpu()
            self.gpu_paused = False
        return coordinate_native(evaluation,self.binary,stores,mapping,channels,
            reference_backgrounds(evaluation, stores, mapping),readers,devices,settings,
            cancel_event=self.cancel_event,inference_session=self)

    def pause_gpu(self):
        if self.pool is not None and not self.gpu_paused:
            if self.pool.busy:
                raise RuntimeError('cannot switch model before inference drained')
            if self.gpu_lease is not None:
                raise RuntimeError('retained model switching requires outer GPU ownership')
            self.pool.stop_gpu()
            self.gpu_paused = True

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.gpu_lease is not None:
                self.gpu_lease.close()
        finally:
            try:
                if self.pool is not None:
                    self.pool.close()
            finally:
                self.pool = self.gpu_lease = None
                self.owner.__exit__(None,None,None)

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()


def extend_cached_replay(evaluation,binary,checkpoint,**kwargs):
    """One-shot convenience wrapper around the same reusable session."""
    with CachedReplaySession(evaluation,binary,checkpoint,**kwargs) as session:
        return session.run(evaluation)


def prefix_points(points, length):
    points = np.asarray(points, dtype=np.float64)
    if length <= 0:
        raise ValueError("prefix length must be positive")
    kept, remaining = [points[0]], float(length)
    for a, b in zip(points, points[1:]):
        distance = float(np.linalg.norm(b - a))
        if distance == 0:
            continue
        if distance >= remaining:
            kept.append(a + (b - a) * (remaining / distance))
            break
        kept.append(b)
        remaining -= distance
    if len(kept) < 2:
        raise ValueError("degenerate reference")
    return np.asarray(kept).tolist()


class PublishedStore:
    """Read-only readiness adapter; actual codec validation remains native.

    Files come from a completed, immutable evaluation. Reuse the production
    all-channel completeness helper; no missing sibling can be treated as fill.
    """
    def __init__(self, manifest):
        manifest = Path(manifest).resolve()
        self.root = manifest.parent
        document = json.loads(manifest.read_text())
        if "mixed_inference" in document:
            raise ValueError("open each underlying ordinary cache separately, not the mixed manifest")
        groups = document["groups"]
        self.channels, self.paths = {}, []
        self.level = self.chunk = self.shape = None
        for group in groups.values():
            level_path = (self.root / group["zarr"]).resolve()
            if not level_path.is_relative_to(self.root):
                raise ValueError("published channel escapes cache")
            meta = json.loads((level_path / ".zarray").read_text())
            level, chunk, shape = int(level_path.name), tuple(meta["chunks"]), tuple(meta["shape"])
            if len(shape) != 3 or len(set(chunk)) != 1 or np.dtype(meta["dtype"]) not in (np.dtype("u1"), np.dtype("float32")):
                raise ValueError("cached profile requires 3D cubic uint8 or float32 chunks")
            if self.level is not None and (level, chunk[0], shape) != (self.level, self.chunk, self.shape):
                raise ValueError("published channels disagree on their lattice")
            self.level, self.chunk, self.shape = level, chunk[0], shape
            self.channels[str(level_path)] = None
            self.paths.append(str(level_path.parent))
        if not self.paths:
            raise ValueError("empty published cache")
        self.stats = {"inferred_tiles": 0, "published_chunks": 0, "warm_chunks": 0}
        self.priorities, self.tiles = {}, {}
        self._ready = set()

    def request(self, index, *, remaining=0, background=False):
        if len(index) != 3 or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in index):
            raise ValueError("invalid cached chunk index")
        origin = tuple(v * self.chunk for v in index)
        if any(o >= n for o, n in zip(origin, self.shape)):
            raise ValueError("cached chunk outside volume")
        if origin not in self._ready:
            if not tiled._omezarr_chunk_group_complete(tuple(self.paths), self.level, *origin, self.chunk):
                raise RuntimeError(f"cached-only inference miss: {self.root} chunk={index}; inference is disabled")
            self._ready.add(origin)
            self.stats["warm_chunks"] += 1
        return origin

    def ready(self, origin):
        return origin in self._ready

    def pending_tile_key(self):
        # Cached-only replay never admits new inference work.
        return None


def published_bindings(sources):
    """Shared immutable, all-channel readiness bindings for cached replay."""
    stores, mapping, channels = {}, {}, {}
    for source in sources:
        key = str(Path(source["fiber_manifest"]).resolve().parent)
        if key not in stores:
            stores[key] = PublishedStore(source["fiber_manifest"])
            channels[key] = stores[key].channels
        mapping[source["id"]] = key
    return stores, mapping, channels


def run_cached_config(evaluation, binary, *, timeout=3600, cancel_event=None):
    """Replay explicit references without inference; missing chunks fail loudly."""
    stores, mapping, channels = published_bindings(evaluation["sources"])
    coordinate_native(evaluation, Path(binary), stores, mapping, channels,
        {key: set() for key in stores}, {}, (),
        {"cached_trace_timeout_seconds": timeout}, allow_inference=False,
        **({'cancel_event': cancel_event} if cancel_event is not None else {}))
    return json.loads((Path(evaluation["output"]) / "summary.json").read_text())


def run_full(workload):
    """Full original-fiber replay, immutable published cache, no GPU or rendering."""
    import subprocess
    previous = Path(workload["cached_evaluation"])
    summary = json.loads((previous / "summary.json").read_text())
    require_single_model_profile(summary["config"])
    inventory = json.loads((previous / "inference_summary.json").read_text())
    if not summary.get("completed") or any((previous / p).exists() for p in ("incomplete.json", "inference_incomplete.json")):
        raise ValueError("cached evaluation requires completed input")
    bindings_path = previous.parent / "full_bindings.json"
    if bindings_path.exists():
        bindings = json.loads(bindings_path.read_text())
        if bindings["checkpoint_sha256"] != workload["checkpoint_sha256"]:
            raise ValueError("completed bindings checkpoint mismatch")
        for source in bindings["sources"]:
            for fiber in source["fibers"]:
                if sha256_file(fiber["path"]) != fiber["sha256"]:
                    raise ValueError("annotation changed since completed evaluation")
    output = Path(workload["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    evaluation = deepcopy(summary["config"])
    evaluation.pop("live_endpoint", None)
    evaluation.update(output=str(output / "full"), render=False)
    binary = Path(workload["native_binary"])
    stores, mapping, channels = {}, {}, {}
    hashes = {}
    for source in evaluation["sources"]:
        if bindings_path.exists():
            recorded = next(b for b in bindings["sources"] if b["collection"] == source["id"])
            if sha256_file(source["normal_manifest"]) != recorded["normal_manifest_sha256"]:
                raise ValueError("normal manifest changed since completed evaluation")
        root = Path(source["fiber_manifest"]).resolve().parent
        if str(root) not in inventory["cache_roots"].values():
            raise ValueError("cache not recorded by completed inference")
        model = json.loads((root / "inference.json").read_text())["model"]
        if model["checkpoint_sha256"] != workload["checkpoint_sha256"]:
            raise ValueError("cached checkpoint mismatch")
        for path in [*source["fibers"], source["normal_manifest"]]:
            hashes[path] = sha256_file(path)
    # Export using the very same native selection implementation as replay.
    config_path = output / "evaluation.json"
    atomic_write(config_path, evaluation)
    evaluation["reference_plan"] = json.loads(subprocess.check_output(
        [str(binary), "replay-references", "--config", str(config_path)], text=True))
    if any(sha256_file(path) != digest for path, digest in hashes.items()):
        raise ValueError("input changed during reference planning")
    atomic_write(config_path, evaluation)
    atomic_write(output / "provenance.json", {"previous_evaluation": str(previous),
        "checkpoint_sha256": workload["checkpoint_sha256"], "input_sha256": hashes,
        "allow_inference": False, "render": False})
    result = run_cached_config(evaluation, binary)
    expected_cases = 2 * sum(len(fiber["runs"]) for source in evaluation["reference_plan"]["sources"] for fiber in source["fibers"])
    if result["cases"] != expected_cases:
        raise ValueError("cached replay coverage differs from selected reference plan")
    report = write_metric_comparison(result, output / "metrics.json")
    print(json.dumps(report["after"], indent=2), flush=True)
    return report


def write_metric_comparison(result, path):
    """Shared reporting-only comparison; preserves raw artifacts, hides old counts."""
    from .replay_metrics import complete_report
    report = complete_report(result)
    before = complete_report(result, episode_distance=0)
    # Publish old distance metrics for comparison, not old event counts.
    for group in [before["global_result"], *before["scrolls"].values(), *before["collections"].values()]:
        for key in ("failures", "reliability_segments", "zero_length_segments"):
            group.pop(key, None)
    comparison = {"before_distance_metrics": before, "after": report}
    atomic_write(path, comparison)
    return comparison


def run(workload):
    settings = workload["profile"]
    previous = Path(settings["completed_evaluation"])
    summary = json.loads((previous / "summary.json").read_text())
    require_single_model_profile(summary["config"])
    inference = json.loads((previous / "inference_summary.json").read_text())
    if not summary.get("completed") or any((previous / p).exists() for p in ("incomplete.json", "inference_incomplete.json")):
        raise ValueError("cached profiling requires a completed evaluation")
    output = Path(settings["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "references").mkdir()
    base = deepcopy(summary["config"])
    original_case = {path: 2 * i for i, path in enumerate(
        path for source in base["sources"] for path in source["fibers"])}
    base.pop("live_endpoint", None)
    base["render"] = False
    base["profile_tracing"] = True
    whole_count = int(settings.get("shortest_whole_fibers", 0))
    allowed = None
    if whole_count:
        ranked = []
        for source in base["sources"]:
            for path in source["fibers"]:
                fiber, = load_fiber_file(path)
                length = float(np.linalg.norm(np.diff(fiber.line_points_xyz, axis=0), axis=1).sum())
                ranked.append((length, path))
        allowed = {path for _, path in sorted(ranked)[:whole_count]}
    selected, audit = [], []
    for source in base["sources"]:
        root = Path(source["fiber_manifest"]).resolve().parent
        if str(root) not in inference["cache_roots"].values():
            raise ValueError("source cache is not in the completed inference inventory")
        provenance = json.loads((root / "inference.json").read_text())
        if provenance["model"]["checkpoint_sha256"] != workload["checkpoint_sha256"]:
            raise ValueError("cached snapshot mismatch")
        paths = []
        for path in source["fibers"]:
            if allowed is not None and path not in allowed:
                continue
            original = Path(path)
            if allowed is not None:
                # Keep original JSON precision and metadata. The Python fiber
                # loader stores float32 geometry; reserializing it would alter
                # the reference used by the native double-precision loader.
                paths.append(str(original))
                audit.append({"original": str(original), "sha256": sha256_file(original),
                              "reference": str(original), "reference_sha256": sha256_file(original)})
                continue
            fiber, = load_fiber_file(original)
            points = (np.asarray(fiber.line_points_xyz).tolist() if allowed is not None else
                      prefix_points(fiber.line_points_xyz, settings.get("prefix_base_voxels", 128)))
            target = output / "references" / f"{len(audit):03d}.json"
            document = {"type": "vc3d_fiber", "version": 1, "line_points": points,
                        "control_points": [points[0], points[-1]],
                        "vc_open_data_coordinate_space": source["coordinate_space"]}
            atomic_write(target, document)
            paths.append(str(target))
            audit.append({"original": str(original), "sha256": sha256_file(original),
                          "reference": str(target), "reference_sha256": sha256_file(target)})
        if paths:
            selected.append({**source, "fibers": paths})
    base["sources"] = selected
    atomic_write(output / "cohort.json", {"completed_evaluation": str(previous), "references": audit,
                 "prefix_base_voxels": None if allowed is not None else settings.get("prefix_base_voxels", 128),
                 "shortest_whole_fibers": whole_count})
    original_geometry = None
    if allowed is not None:
        original_geometry = {}
        for i, reference in enumerate(audit):
            for reverse in (0, 1):
                old_index = original_case[reference["original"]] + reverse
                old = json.loads((previous / "cases" / str(old_index) / "trace.json").read_text())
                original_geometry[str(2 * i + reverse)] = json_digest(old["replay"])
    rows, expected = [], None
    for iteration, entries in enumerate(settings.get("readiness_entries_order", [0, 4096, 4096, 0])):
        evaluation = deepcopy(base)
        destination = output / f"run_{iteration}_entries_{entries}"
        evaluation.update(output=str(destination), readiness_cache_entries=int(entries))
        stores, mapping, channels, backgrounds = {}, {}, {}, {}
        for source in selected:
            key = str(Path(source["fiber_manifest"]).resolve().parent)
            if key not in stores:
                stores[key] = PublishedStore(source["fiber_manifest"])
                channels[key] = stores[key].channels
                backgrounds[key] = set()  # only actual trace demand, no inference/prefetch benchmark
            mapping[source["id"]] = key
        print(f"cached-trace run={iteration} readiness_entries={entries} cases={2 * len(audit)}", flush=True)
        before, started = os.times(), time.monotonic()
        coordinate_native(evaluation, Path(workload["native_binary"]), stores, mapping, channels,
            backgrounds, {}, (), {"cached_trace_timeout_seconds": settings.get("run_timeout_seconds", 300)},
            allow_inference=False)
        after = os.times()
        wall_seconds = time.monotonic() - started
        result = json.loads((destination / "summary.json").read_text())
        if not result.get("completed") or result["cases"] != 2 * len(audit):
            raise RuntimeError("short cached replay did not complete")
        geometry, profiles = {}, []
        for path in sorted((destination / "cases").glob("*/trace.json")):
            geometry[path.parent.name] = json_digest(json.loads(path.read_text())["replay"])
            profiles.append(json.loads((path.parent / "profile.json").read_text()))
        if set(geometry) != {str(i) for i in range(2 * len(audit))}:
            raise RuntimeError("missing or unexpected case artifacts")
        original_matches = None if original_geometry is None else sum(
            geometry[key] == value for key, value in original_geometry.items())
        if original_geometry is not None and geometry != original_geometry and not settings.get("allow_original_replay_difference", False):
            raise RuntimeError("cached replay differs from original completed evaluation")
        if expected is None:
            expected = geometry
        if geometry != expected:
            raise RuntimeError("readiness variant changed replay geometry/failures")
        metrics = {key: sum(p["readiness"][key] for p in profiles) for key in profiles[0]["readiness"]}
        row = {"iteration": iteration, "readiness_entries": entries, "wall_seconds": wall_seconds,
               "native_wall_seconds": result["wall_seconds"],
               "original_replay_matches": original_matches,
               "failures": result["failures"],
               "native_cpu_seconds": after.children_user + after.children_system - before.children_user - before.children_system,
               "case_seconds_p50": float(np.median([p["elapsed_seconds"] for p in profiles])),
               "case_seconds_p95": float(np.percentile([p["elapsed_seconds"] for p in profiles], 95)),
               "readiness": metrics, "replay_hash": json_digest(geometry), "inferred_tiles": 0,
               "trace": {key: sum(p["trace"][key] for p in profiles) for key in profiles[0]["trace"]}}
        rows.append(row)
        atomic_write(output / "results.json", {"runs": rows, "completed": len(rows) == len(settings.get("readiness_entries_order", [0, 4096, 4096, 0]))})
        print("cached-trace result=" + json.dumps(row), flush=True)
    return rows
