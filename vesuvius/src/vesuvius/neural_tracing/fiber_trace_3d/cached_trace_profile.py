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
            if len(shape) != 3 or len(set(chunk)) != 1 or np.dtype(meta["dtype"]) != np.dtype("u1"):
                raise ValueError("cached profile requires ordinary 3D cubic uint8 chunks")
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
