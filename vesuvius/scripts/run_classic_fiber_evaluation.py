#!/usr/bin/env python3
"""Prepare/audit and run the locked September 2026 classic-checkpoint cohort.

Run with the existing las environment. No training, installs, volume scans or
bulk prefetch. The actual inference/tracer implementations remain shared.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "vesuvius/src"), str(REPO), str(REPO / "gpu_manager/src")]

DEFAULT_CONFIG = REPO / "vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_collection.json"


def load_workload(path):
    config = json.loads(Path(path).read_text())
    if config.get("version") != 1 or not config.get("sources"):
        raise ValueError("workload requires version1 and explicit source bindings")
    ids = [s["id"] for s in config["sources"]]
    if len(ids) != len(set(ids)) or config["pilot_source_id"] not in ids:
        raise ValueError("duplicate source IDs or unknown pilot source")
    return config


def validate_locked_membership(split_root, collection):
    """Reject changed test membership/content, including replaced symlinks."""
    from lasagna.inference_provenance import sha256_file
    manifest = json.loads((split_root / "manifest.json").read_text())
    expected = {row["path"]: row["sha256"] for row in manifest["files"]
                if row["source"] == collection and row["split"] == "test"}
    actual = {str(p.relative_to(split_root / "test")): p
              for p in (split_root / "test" / collection).glob("*.json")}
    if actual.keys() != expected.keys():
        raise ValueError(f"locked test membership changed: {collection}")
    for name, path in actual.items():
        if sha256_file(path) != expected[name]:
            raise ValueError(f"locked annotation hash mismatch: {name}")


def select_test_files(files, names=None):
    """Select explicit filenames after the complete collection is validated."""
    if names is None:
        return files
    if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
        raise ValueError("test_fibers must be a nonempty list of filenames")
    if len(set(names)) != len(names):
        raise ValueError("duplicate test_fibers selection")
    if not set(names).issubset({p.name for p in files}):
        raise ValueError("test_fibers contains unknown or non-filename entries")
    return [p for p in files if p.name in names]


def resolve_normal_binding(manager, binding, predictions):
    """Resolve the configured source and exact normal choice, without fallbacks."""
    from lasagna.manager.fiberlets import _completed_dependency, _published_dependency
    from lasagna.manager.catalog import resolve_lasagna_prediction
    sample, scan = binding["sample_id"], binding["volume_id"]
    selection = binding["normal"]
    if selection["kind"] == "catalogue":
        record = resolve_lasagna_prediction(predictions, selection["selector"], sample_id=sample, volume_id=scan)
        return _published_dependency(manager, record)
    if selection["kind"] != "manager-run":
        raise ValueError("unknown normal binding kind")
    normal = _completed_dependency(manager, selection["selector"], expected_kind="lasagna")
    if (normal["source"].get("sample_id"), normal["source"].get("volume_id")) != (sample, scan):
        raise ValueError("approved normal run identifies a different source")
    return {**normal, "base_shape_zyx": normal["manifest"]["base_shape_zyx"],
            "manifest_url": str(normal["manifest_path"])}


def prepare(workload, *, pilot_only=False):
    import numpy as np
    from lasagna.inference_provenance import atomic_write, json_digest, sha256_file
    from lasagna.manager.catalog import CatalogCache, index_volumes, index_lasagna_predictions
    from lasagna.manager.config import load_config
    from vesuvius.neural_tracing.fiber_trace_2d.fiber_json import load_fiber_file
    from vesuvius.neural_tracing.fiber_trace_2d.sampling import Vc3dCoordinateSampler
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import input_root_attributes, validate_input_lattice

    CHECKPOINT, SPLIT, ROOT, BINARY = (Path(workload[k]) for k in ("checkpoint", "split_root", "output_root", "native_binary"))
    CHECKPOINT_SHA = workload["checkpoint_sha256"]
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("classic checkpoint digest differs from prior inference provenance")
    raw = Path(workload["catalog_path"]).read_bytes()
    raw = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    catalogue_sha = hashlib.sha256(raw).hexdigest()
    cache = CatalogCache(json.loads(raw), {"sha256": catalogue_sha})
    volumes, predictions = index_volumes(cache), index_lasagna_predictions(cache)
    cache_root = subprocess.check_output([str(BINARY.with_name("vc_cache_root"))], text=True).strip()
    manager = replace(load_config(), cache_dir=cache_root)
    audit = {"checkpoint": str(CHECKPOINT), "checkpoint_sha256": CHECKPOINT_SHA,
             "catalogue_sha256": catalogue_sha, "split_sha256": sha256_file(SPLIT / "manifest.json"),
             "scope": "pilot" if pilot_only else "full",
             "workload_sha256": json_digest(workload), "sources": []}
    sources, resolved = [], {}
    for binding in workload["sources"]:
        collection, sample, scan, count = (binding[k] for k in ("id", "sample_id", "volume_id", "expected_test_count"))
        if pilot_only and collection != workload["pilot_source_id"]:
            continue
        validate_locked_membership(SPLIT, collection)
        print(f"preflight {collection}: {sample}/{scan}", flush=True)
        matches = [v for v in volumes if (v.sample_id, v.volume_id) == (sample, scan)]
        if len(matches) != 1 or not matches[0].s3_url:
            raise ValueError(f"ambiguous/missing CT: {collection}")
        volume = matches[0]
        if binding["input_volume"].rstrip("/") != volume.s3_url.rstrip("/"):
            raise ValueError("configured CT URL does not match catalogue source identity")
        input_level = int(binding["input_level"])
        coordinate_space = f"{sample}/{scan}@L0"
        resolve_key = (sample, scan, input_level, json.dumps(binding["normal"], sort_keys=True))
        if resolve_key not in resolved:
            normal = resolve_normal_binding(manager, binding, predictions)
            sampler = Vc3dCoordinateSampler(volume.s3_url.rstrip("/"), level=input_level,
                                           cache_root=cache_root, sample_threads=1)
            validate_input_lattice(input_root_attributes(sampler.volume), input_level)
            shape = tuple(int(v) for v in sampler.volume.shape_at(0))
            factor = 1 << input_level
            shape1 = tuple(int(v) for v in sampler.volume.shape_at(input_level))
            if shape1 != tuple((v + factor - 1) // factor for v in shape):
                raise ValueError("CT selected-level extent mismatch")
            if any(abs(a-b) > 1 for a,b in zip(shape, normal["base_shape_zyx"])):
                raise ValueError("normal/base CT extents disagree")
            resolved[resolve_key] = normal, shape
        normal, shape = resolved[resolve_key]
        files = sorted((SPLIT / "test" / collection).glob("*.json"))
        if len(files) != count:
            raise ValueError(f"locked test count changed for {collection}")
        files = select_test_files(files, binding.get("test_fibers"))
        fibers = []
        for path in files:
            document = json.loads(path.read_text())
            if document.get("vc_open_data_coordinate_space", coordinate_space) != coordinate_space:
                raise ValueError(f"coordinate identity mismatch: {path}")
            if document.get("vc_open_data_source_coordinate_level", 0) != 0 or document.get("vc_open_data_source_coordinate_scale_factor", 1) != 1:
                raise ValueError(f"non-base annotation coordinates: {path}")
            if abs(document.get("vc_open_data_source_original_resolution", volume.pixel_size_um) - volume.pixel_size_um) > 1e-6:
                raise ValueError(f"physical spacing mismatch: {path}")
            fiber, = load_fiber_file(path)
            points = np.asarray(fiber.line_points_xyz, dtype=np.float64)
            if not np.isfinite(points).all() or np.any(points < 0) or np.any(points > np.array(shape[::-1])-1):
                raise ValueError(f"nonfinite/out-of-bounds reference: {path}")
            length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            if length <= 0:
                raise ValueError(f"degenerate reference: {path}")
            fibers.append({"path": str(path), "sha256": sha256_file(path), "length_base_voxels": length,
                           "points": len(points), "min_xyz": points.min(axis=0).tolist(), "max_xyz": points.max(axis=0).tolist(),
                           "embedded_coordinate_space": document.get("vc_open_data_coordinate_space")})
        sources.append({"id": collection, "coordinate_space": coordinate_space,
                        "input_volume": binding["input_volume"].rstrip("/"), "input_level": input_level,
                        "normal_manifest": str(normal["manifest_path"]),
                        "base_voxel_size_um": volume.pixel_size_um, "fibers": [str(f) for f in files]})
        audit["sources"].append({"collection": collection, "sample": sample, "scan": scan,
                                 "base_shape_zyx": shape, "normal_manifest_url": normal["manifest_url"],
                                 "normal_manifest_sha256": normal["manifest_sha256"], "fibers": fibers})
    config = {**workload["evaluation"], "sources": sources, "output": str(ROOT / "full")}
    # Shortest unambiguous WHOLE held-out fiber; no shortened/cropped reference.
    choices = [(f["length_base_voxels"], s["collection"], f["path"]) for s in audit["sources"]
               if s["collection"] == workload["pilot_source_id"] for f in s["fibers"]]
    _, selected, path = min(choices)
    pilot = {**config, "output": str(ROOT / "pilot"),
             "sources": [{**next(s for s in sources if s["id"] == selected), "fibers": [path]}]}
    audit["pilot"] = {"collection": selected, "fiber": path, "directions": 2}
    audit["config_sha256"] = json_digest(pilot if pilot_only else config)
    ROOT.mkdir(parents=True, exist_ok=True)
    prepared = [("pilot_bindings.json", audit), ("pilot.json", pilot)] if pilot_only else [
        ("full_bindings.json", audit), ("full.json", config)]
    for name, value in prepared:
        value = json.loads(json.dumps(value))
        target = ROOT / name
        if target.exists() and json.loads(target.read_text()) != value:
            raise ValueError(f"refusing to overwrite changed prepared evidence: {target}")
        if not target.exists():
            # These files are reproducible from the locked inputs. Atomic
            # replacement matters; waiting for unrelated multi-terabyte mmap
            # writeback on the evaluation filesystem does not.
            atomic_write(target, value, durable=False)
    count = sum(len(s["fibers"]) for s in sources)
    print(f"Prepared {'one whole-fiber pilot/two directions' if pilot_only else f'{count} fibers/{2*count} directions'}. Pilot: {selected}/{Path(path).name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["benchmark-review", "benchmark-full", "pair-connectivity", "short-failures", "integration-report", "integration-fit", "integration-audit", "fixed-path-report", "fixed-path-extract", "fixed-path-fit", "freeze-snapshot", "prepare", "prepare-pilot", "validate-pilot", "validate-full", "pilot", "full", "cached-full", "report-metrics", "profile", "observe", "report-progress", "reference-plan"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--devices", help="runtime device selection (default from workload)")
    parser.add_argument("--manager-socket")
    parser.add_argument("--manager-priority", type=int)
    parser.add_argument("--unmanaged", action="store_true", help="explicit bypass of configured manager")
    parser.add_argument("--pid", type=int, help="existing coordinator PID for read-only observe")
    parser.add_argument("--seconds", type=float, default=20, help="read-only observe duration,1–60s")
    parser.add_argument("--extraction-time-limit-seconds", type=float,
                        help="checkpoint and stop short-failures scheduling after this duration (cleanup follows)")
    args = parser.parse_args()
    workload = (json.loads(args.config.read_text()) if args.mode in ('short-failures', 'benchmark-full', 'benchmark-review', 'pair-connectivity') else load_workload(args.config))
    if args.mode == 'pair-connectivity':
        from vesuvius.neural_tracing.fiber_trace_3d.fiber_pair_connectivity import run_pair_connectivity
        run_pair_connectivity(workload)
        return 0
    if args.mode == 'benchmark-review':
        from vesuvius.neural_tracing.fiber_trace_3d.short_benchmark_evaluate import write_render_review_index
        root = Path(workload['output_root'])
        result = json.loads((root / 'full/summary.json').read_text())
        if not result['completed']:
            raise ValueError('cannot index incomplete benchmark')
        rows = write_render_review_index(root, result, json.loads(Path(workload['benchmark_manifest']).read_text()))
        print(f'Indexed {len(rows)} rendered pieces: {root / "README.md"}')
        return 0
    if args.mode == 'benchmark-full':
        devices = args.devices or workload['devices']
        manager = None if args.unmanaged else dict(workload['gpu_manager'], devices=devices)
        if manager is not None:
            if args.manager_socket is not None:
                manager['socket'] = args.manager_socket
            if args.manager_priority is not None:
                manager['priority'] = args.manager_priority
        from vesuvius.neural_tracing.fiber_trace_3d.short_benchmark_evaluate import run
        run(workload, devices, gpu_manager=manager)
        return 0
    if args.mode == "short-failures":
        import os
        fill_missing = workload['short_failures'].get('fill_missing', True)
        devices = args.devices or workload.get('devices', '0')
        if fill_missing and not args.unmanaged:
            from vesuvius.neural_tracing.fiber_trace_3d.managed_evaluation import launch_managed
            manager = dict(workload.get('evaluation', {}).get('gpu_manager', {}))
            manager.update(workload['short_failures'].get('gpu_manager', {}))
            manager.update(exclusive=True, label='short-failure-extraction')
            command = [sys.executable, str(Path(__file__).resolve()), args.mode,
                       '--config', str(args.config.resolve())]
            if args.extraction_time_limit_seconds is not None:
                command.extend(['--extraction-time-limit-seconds',str(args.extraction_time_limit_seconds)])
            return launch_managed(command,
                                  devices=devices, settings=manager, socket=args.manager_socket,
                                  priority=100 if args.manager_priority is None else args.manager_priority)
        if not fill_missing:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from vesuvius.neural_tracing.fiber_trace_3d.short_failure_benchmark import run
        run(workload, devices=devices, time_limit_seconds=args.extraction_time_limit_seconds)
        return 0
    if args.mode == "integration-report":
        from vesuvius.neural_tracing.fiber_trace_3d.embedding_integration_report import report
        report(workload['fixed_path'])
        return 0
    if args.mode == "fixed-path-report":
        from vesuvius.neural_tracing.fiber_trace_3d.embedding_fixed_path import report
        report(workload['fixed_path'])
        return 0
    if args.mode == "fixed-path-extract":
        from vesuvius.neural_tracing.fiber_trace_3d.embedding_fixed_path import extract
        extract(workload['fixed_path'])
        return 0
    if args.mode == "freeze-snapshot":
        from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import freeze_checkpoint
        from lasagna.inference_provenance import atomic_write
        import torch
        snapshot, digest = freeze_checkpoint(Path(workload["checkpoint"]), Path(workload["output_root"]) / "frozen")
        saved = torch.load(snapshot, map_location="cpu", weights_only=False)
        metadata = {"checkpoint": str(snapshot), "checkpoint_sha256": digest,
                    "source": workload["checkpoint"],
                    "metadata": {k: v for k, v in saved.items() if isinstance(v, (str, int, float, bool, type(None)))},
                    "config": saved.get("config", {})}
        atomic_write(Path(workload["output_root"]) / "snapshot.json", metadata)
        print(json.dumps(metadata, indent=2, default=str))
        return 0
    if args.mode == "report-metrics":
        from vesuvius.neural_tracing.fiber_trace_3d.cached_trace_profile import write_metric_comparison
        source = Path(workload["cached_evaluation"]) / "summary.json"
        destination = Path(workload["output_root"]) / "saved_trace_metrics.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        report = write_metric_comparison(json.loads(source.read_text()), destination)
        print(json.dumps(report["after"], indent=2))
        return 0
    if args.mode == "cached-full":
        # Read-only published-cache evaluator: no manager GPU lease, no CUDA.
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from vesuvius.neural_tracing.fiber_trace_3d.cached_trace_profile import run_full
        run_full(workload)
        return 0
    if args.mode == "report-progress":
        from vesuvius.neural_tracing.fiber_trace_3d.visualization_report import write_trace_progress_report
        root = Path(workload["output_root"])
        report = write_trace_progress_report(root / "full", root)
        print(json.dumps({"paired_fibers": report["paired_fibers"],
            "arithmetic_mean_per_fiber_mm": report["arithmetic_mean_per_fiber_mm"],
            "short_below_average": report["short_below_average"][:5]}, indent=2))
        return 0
    if args.mode == "observe":
        if args.pid is None:
            parser.error("observe requires --pid")
        from vesuvius.neural_tracing.fiber_trace_3d.live_process_probe import observe
        observe(args.pid, workload["output_root"], seconds=args.seconds)
        return 0
    devices = args.devices or (workload["pilot_devices"] if args.mode == "pilot" else workload["devices"])
    manager = workload["evaluation"].get("gpu_manager", {})
    if args.mode in ("pilot", "full", "profile", "fixed-path-fit", "integration-fit", "integration-audit") and not args.unmanaged and (
            manager.get("enabled", False) or args.manager_socket is not None):
        from vesuvius.neural_tracing.fiber_trace_3d.managed_evaluation import launch_managed
        return launch_managed([sys.executable, str(Path(__file__).resolve()), args.mode,
                               "--config", str(args.config.resolve())],
                              devices=devices, settings=manager, socket=args.manager_socket,
                              priority=args.manager_priority)
    if args.mode in ('integration-fit','integration-audit'):
        from vesuvius.neural_tracing.fiber_trace_3d.embedding_integration import calibrate, audit
        (calibrate if args.mode=='integration-fit' else audit)(workload['fixed_path'])
        return 0
    if args.mode == "fixed-path-fit":
        from vesuvius.neural_tracing.fiber_trace_3d.embedding_fixed_path import fit
        fit(workload['fixed_path'])
        return 0
    if args.mode == "profile":
        if workload.get("profile", {}).get("kind") == "cached-trace":
            for source in workload["sources"]:
                validate_locked_membership(Path(workload["split_root"]), source["id"])
            from vesuvius.neural_tracing.fiber_trace_3d.cached_trace_profile import run
            run(workload)
            return 0
        from vesuvius.neural_tracing.fiber_trace_3d.live_inference_profile import profile_workload
        profile_workload(workload, devices_override=devices)
        return
    CHECKPOINT, SPLIT, ROOT, BINARY = (Path(workload[k]) for k in ("checkpoint", "split_root", "output_root", "native_binary"))
    validate_only = args.mode.startswith("validate-")
    reference_plan_only = args.mode == "reference-plan"
    if reference_plan_only:
        args.mode = "full"
    if validate_only:
        args.mode = args.mode.removeprefix("validate-")
    if args.mode in ("prepare", "prepare-pilot"):
        prepare(workload, pilot_only=args.mode == "prepare-pilot")
        return
    from lasagna.inference_provenance import json_digest, sha256_file
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import run
    audit = json.loads((ROOT / f"{args.mode}_bindings.json").read_text())
    if audit["workload_sha256"] != json_digest(workload):
        raise ValueError("workload mapping/settings changed after preparation")
    config = json.loads((ROOT / f"{args.mode}.json").read_text())
    if json_digest(config) != audit["config_sha256"]:
        raise ValueError("prepared config changed after audit")
    if sha256_file(CHECKPOINT) != audit["checkpoint_sha256"] or sha256_file(SPLIT / "manifest.json") != audit["split_sha256"]:
        raise ValueError("checkpoint or split changed after preparation")
    for source in audit["sources"]:
        validate_locked_membership(SPLIT, source["collection"])
        for fiber in source["fibers"]:
            if sha256_file(fiber["path"]) != fiber["sha256"]:
                raise ValueError(f"annotation changed: {fiber['path']}")
    expected_normals = {s["collection"]: s["normal_manifest_sha256"] for s in audit["sources"]}
    for source in config["sources"]:
        if sha256_file(source["normal_manifest"]) != expected_normals[source["id"]]:
            raise ValueError(f"normal manifest changed: {source['id']}")
    if reference_plan_only:
        from lasagna.inference_provenance import atomic_write
        plan = json.loads(subprocess.check_output([str(BINARY), "replay-references", "--config", str(ROOT / "full.json")], text=True))
        plan["annotation_sha256"] = {f["path"]: f["sha256"] for s in audit["sources"] for f in s["fibers"]}
        atomic_write(ROOT / "reference_plan.json", plan)
        print(json.dumps({"policy": plan["policy"], "sources": [{"source": s["source_id"],
            "selected_fibers": len(s["fibers"]), "eligible_runs": sum(len(f["runs"]) for f in s["fibers"]),
            "excluded_fibers": sum(not f["runs"] for f in s["fibers"]),
            "original_voxels": sum(f["original_length_base_voxels"] for f in s["fibers"]),
            "eligible_voxels": sum(f["eligible_length_base_voxels"] for f in s["fibers"])} for s in plan["sources"]]}, indent=2))
        return 0
    if args.mode == "full":
        pilot_root = Path(workload.get("pilot_result", ROOT / "pilot"))
        if any((pilot_root / name).exists() for name in ("incomplete.json", "inference_incomplete.json")):
            raise ValueError("pilot is incomplete")
        pilot_summary = json.loads((pilot_root / "summary.json").read_text())
        if not pilot_summary.get("completed") or pilot_summary.get("cases", 0) < 2 or pilot_summary["cases"] % 2:
            raise ValueError("successful bidirectional pilot required")
        json.loads((pilot_root / "inference_summary.json").read_text())
    if validate_only:
        print("Checkpoint, locked split membership/content, config and normal manifests verified.", flush=True)
        return
    result = run(ROOT / f"{args.mode}.json", CHECKPOINT, devices, BINARY, unmanaged=True)
    if config.get("render") and result in (None, 0):
        from vesuvius.neural_tracing.fiber_trace_3d.visualization_report import write_visualization_report
        write_visualization_report(ROOT / args.mode, ROOT,
            provenance={"checkpoint": str(CHECKPOINT), "checkpoint_sha256": audit["checkpoint_sha256"],
                        "workload": str(args.config.resolve()), "native_binary": str(BINARY)})
    return result


if __name__ == "__main__":
    raise SystemExit(main())
