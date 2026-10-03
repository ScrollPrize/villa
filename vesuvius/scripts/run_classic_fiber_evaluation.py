#!/usr/bin/env python3
"""Prepare/audit and run the locked September 2026 classic-checkpoint cohort.

Run with the existing las environment. No training, installs, volume scans or
bulk prefetch. The actual inference/tracer implementations remain shared.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "vesuvius/src"), str(REPO), str(REPO / "gpu_manager/src")]

DEFAULT_CONFIG = REPO / "vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_collection.json"
FIXED_GPU_GRANT_MODES = frozenset((
    "profile", "fixed-path-fit", "integration-fit", "integration-audit"))


def live_run_manager_options(args):
    """Forward CLI manager intent exactly to demand-scoped live evaluation."""
    return {"manager_socket": args.manager_socket,
            "manager_priority": args.manager_priority,
            "unmanaged": args.unmanaged}


def load_workload(path):
    config = json.loads(Path(path).read_text())
    if config.get("version") != 1 or not config.get("sources"):
        raise ValueError("workload requires version1 and explicit source bindings")
    ids = [s["id"] for s in config["sources"]]
    if len(ids) != len(set(ids)) or config["pilot_source_id"] not in ids:
        raise ValueError("duplicate source IDs or unknown pilot source")
    return config


from vesuvius.neural_tracing.fiber_trace_3d.evaluation_workload import (
    validate_locked_membership, select_test_files, resolve_normal_binding, prepare,
    audit_prepared,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["crop-benchmark", "contrast-prepare", "benchmark-remove-case", "benchmark-review", "benchmark-render", "benchmark-full", "pair-connectivity", "short-failures", "failure-union", "integration-report", "integration-fit", "integration-audit", "fixed-path-report", "fixed-path-extract", "fixed-path-fit", "freeze-snapshot", "prepare", "prepare-pilot", "validate-pilot", "validate-full", "pilot", "full", "training-full", "resume-visualization", "cached-full", "report-metrics", "profile", "observe", "report-progress", "reference-plan"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--export-manifests-only", action="store_true",
                        help="failure-union: export Lasagna manifests from completed output")
    parser.add_argument("--devices", help="runtime device selection (default from workload)")
    parser.add_argument("--manager-socket")
    parser.add_argument("--manager-priority", type=int)
    parser.add_argument("--unmanaged", action="store_true", help="explicit bypass of configured manager")
    parser.add_argument("--pid", type=int, help="existing coordinator PID for read-only observe")
    parser.add_argument("--seconds", type=float, default=20, help="read-only observe duration,1–60s")
    parser.add_argument("--extraction-time-limit-seconds", type=float,
                        help="checkpoint and stop short-failures scheduling after this duration (cleanup follows)")
    args = parser.parse_args()
    if args.mode == 'crop-benchmark':
        from vesuvius.neural_tracing.fiber_trace_3d.crop_benchmark import run
        run(json.loads(args.config.read_text()))
        return 0
    workload = (json.loads(args.config.read_text()) if args.mode in ('contrast-prepare', 'benchmark-remove-case', 'short-failures', 'failure-union', 'benchmark-full', 'benchmark-render', 'benchmark-review', 'pair-connectivity') else load_workload(args.config))
    if args.mode == 'contrast-prepare':
        from vesuvius.neural_tracing.fiber_trace_3d.contrast_choice_benchmark import package_contrast_choice
        print(package_contrast_choice(workload))
        return 0
    if args.mode == 'benchmark-remove-case':
        from vesuvius.neural_tracing.fiber_trace_3d.contrast_choice_benchmark import remove_review_case
        print(remove_review_case(workload))
        return 0
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
    if args.mode == 'benchmark-render':
        devices = args.devices or workload['devices']
        manager = None if args.unmanaged else dict(workload['gpu_manager'], devices=devices)
        if manager is not None:
            if args.manager_socket is not None:
                manager['socket'] = args.manager_socket
            if args.manager_priority is not None:
                manager['priority'] = args.manager_priority
        from vesuvius.neural_tracing.fiber_trace_3d.short_benchmark_evaluate import render_existing
        render_existing(workload, devices, gpu_manager=manager)
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
    if args.mode == "failure-union":
        if args.devices:
            workload['failure_union']['devices'] = args.devices
            if workload['failure_union'].get('fill_missing_inference'):
                workload['failure_union']['fill_missing_inference']['devices'] = args.devices
        if args.export_manifests_only:
            from vesuvius.neural_tracing.fiber_trace_3d.short_failure_benchmark import export_union_lasagna_manifests
            export_union_lasagna_manifests(workload['output_root'])
            return 0
        from vesuvius.neural_tracing.fiber_trace_3d.short_failure_benchmark import run_union
        run_union(workload)
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
        if workload.get("beam_sweep"):
            # The sweep blanks CUDA itself unless it was configured for
            # manager-backed demand inference.
            from vesuvius.neural_tracing.fiber_trace_3d.beam_parameter_sweep import run_sweep
            run_sweep(workload)
            return 0
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
    if args.mode in FIXED_GPU_GRANT_MODES and not args.unmanaged and (
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
                validate_locked_membership(Path(workload["split_root"]), source["id"], workload.get("split", "test"))
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
    prepared_mode = "full" if args.mode in ("resume-visualization", "training-full") else args.mode
    from vesuvius.neural_tracing.fiber_trace_3d.evaluation_workload import audit_prepared
    config, audit = audit_prepared(workload, prepared_mode)
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
    runtime_overrides = ({"resume_visualization": True, "resume_render_limit": 96}
                         if args.mode == "resume-visualization" else None)
    if args.mode == "training-full" and not args.unmanaged:
        raise ValueError("training-full must inherit an existing managed training allocation")
    result = run(ROOT / f"{prepared_mode}.json", CHECKPOINT, devices, BINARY,
                 runtime_overrides=runtime_overrides, **live_run_manager_options(args))
    if config.get("render") and result in (None, 0):
        from vesuvius.neural_tracing.fiber_trace_3d.visualization_report import write_visualization_report
        if args.mode == "resume-visualization":
            status = json.loads((ROOT / "full/visualization_resume.json").read_text())
            if status["remaining_cases"]:
                print(json.dumps({"rendered_cases": status["rendered_cases"],
                    "remaining_cases": len(status["remaining_cases"]),
                    "next_cases": status["remaining_cases"][:96]}, indent=2), flush=True)
                return result
        write_visualization_report(ROOT / prepared_mode, ROOT,
            provenance={"checkpoint": str(CHECKPOINT), "checkpoint_sha256": audit["checkpoint_sha256"],
                        "workload": str(args.config.resolve()), "native_binary": str(BINARY)},
            allow_incomplete=args.mode == "resume-visualization")
        if args.mode == "resume-visualization":
            for marker in (ROOT / "full/incomplete.json", ROOT / "full/inference_incomplete.json"):
                marker.unlink(missing_ok=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
