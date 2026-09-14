"""Short, manager-isolated readiness comparison using immutable published chunks."""
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import numpy as np

from lasagna import tiled_predict3d as tiled
from lasagna.inference_provenance import atomic_write, json_digest, sha256_file
from vesuvius.neural_tracing.fiber_trace_2d.fiber_json import load_fiber_file
from .live_evaluate import coordinate_native


def extend_cached_replay(evaluation, binary, checkpoint):
    """Continue an existing inference cache through shared demand publication.

    Unlike a fresh experiment namespace, this explicitly validates and locks
    the existing cache. The numerical adapter/store/control loop are unchanged.
    Caller must own its GPU grant before entering.
    """
    from contextlib import ExitStack
    import subprocess
    import torch
    from lasagna.demand_predict3d import DemandGeometry
    from .live_evaluate import (Vc3dCoordinateSampler, NativeTileReader,
        create_demand_store, native_reader_options, implementation_identity,
        input_root_attributes, validate_input_lattice)
    binary = Path(binary)
    cache_root = subprocess.check_output([str(binary.with_name("vc_cache_root"))], text=True).strip()
    settings = evaluation["inference"]
    stores, readers, mapping, channels = {}, {}, {}, {}
    with ExitStack() as cleanup:
        for source in evaluation["sources"]:
            cache = Path(source["fiber_manifest"]).parent
            persisted = json.loads((cache / "cache_identity.json").read_text())
            identity = persisted["identity"]
            model, original = identity["model"], identity["source"]
            if model["implementation"] != implementation_identity() or model["checkpoint_sha256"] != sha256_file(checkpoint):
                raise ValueError("cannot extend cache with different checkpoint/implementation")
            geometry = DemandGeometry(**persisted["geometry"])
            sampler = Vc3dCoordinateSampler(original["locator"], level=original["level"],
                cache_root=cache_root, **native_reader_options(settings))
            attrs = input_root_attributes(sampler.volume)
            validate_input_lattice(attrs, original["level"])
            if (list(sampler.volume.shape_at(original["level"])) != original["shape"] or
                list(sampler.volume.shape_at(0)) != original["base_shape"] or attrs != original["metadata"]):
                raise ValueError("source metadata changed since cached inference")
            key = str(cache)
            stores[key] = cleanup.enter_context(create_demand_store(cache, original, model, geometry,
                model["config"], checkpoint, original["level"], settings))
            readers[key] = NativeTileReader(sampler, geometry, sampler.volume.dtype)
            mapping[source["id"]] = key
            channels[key] = {str((Path(c.relative_path)/str(p.level)).resolve()): geometry
                            for p in stores[key].products for c in p.channels}
        return coordinate_native(evaluation, binary, stores, mapping, channels,
            {k:set() for k in stores}, readers, (torch.device("cuda:0"),), settings)


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
        groups = json.loads(manifest.read_text())["groups"]
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


def run_full(workload):
    """Full original-fiber replay, immutable published cache, no GPU or rendering."""
    import subprocess
    previous = Path(workload["cached_evaluation"])
    summary = json.loads((previous / "summary.json").read_text())
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
        key = str(root)
        if key not in stores:
            stores[key] = PublishedStore(source["fiber_manifest"])
            channels[key] = stores[key].channels
        mapping[source["id"]] = key
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
    coordinate_native(evaluation, binary, stores, mapping, channels,
        {key: set() for key in stores}, {}, (),
        {"cached_trace_timeout_seconds": 3600}, allow_inference=False)
    result = json.loads((output / "full/summary.json").read_text())
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
