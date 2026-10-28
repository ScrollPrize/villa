"""No-argument, allowlisted test/benchmark launcher. Never installs dependencies."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "local/fiber_tests.json"
SUITES = {
    "stream-volume-build": ("local/test_stream_volume.py::test_build",),
    "stream-volume-policy": ("local/test_stream_volume.py::test_policy",),
    "stream-volume-native": ("local/test_stream_volume.py::test_native",),
    "stream-volume-pilot": ("local/test_stream_volume.py::test_pilot",),
    "stream-volume-resume": ("local/test_stream_volume.py::test_resume",),
    "stream-volume-comparison": ("local/test_stream_volume.py::test_comparison",),
    "stream-volume-full": ("local/test_stream_volume.py::test_full",),
    "stream-volume-restart": ("local/test_stream_volume.py::test_restart",),
    "stream-volume-stop": ("local/test_stream_volume.py::test_stop",),
    "stream-volume-status": ("local/test_stream_volume.py::test_status",),
    "stream-volume-diagnose-cpu": ("local/test_stream_volume.py::test_diagnose_cpu",),
    "retained-sampling-diagnosis": ("local/test_retained_sampling_diagnosis.py::test_build", "local/test_retained_sampling_diagnosis.py::test_run", "local/test_retained_sampling_diagnosis.py::test_report"),
    "unified-sampling-diagnosis": ("local/test_unified_sampling_diagnosis.py::test_build", "local/test_unified_sampling_diagnosis.py::test_run", "local/test_unified_sampling_diagnosis.py::test_report"),
    "sampling-inpass-profile": ("local/test_sampling_inpass_profile.py::test_build", "local/test_sampling_inpass_profile.py::test_run", "local/test_sampling_inpass_profile.py::test_report"),
    "direct-separate-gates-build": ("local/test_direct_separate_gates.py::test_build", "local/test_direct_separate_gates.py::test_regressions"),
    "direct-separate-gates": ("local/test_direct_separate_gates.py::test_run", "local/test_direct_separate_gates.py::test_report"),
    "direct-separate-gates-lower": ("local/test_direct_separate_lower.py::test_run", "local/test_direct_separate_lower.py::test_report"),
    "direct-joint-hard-gates": ("local/test_direct_joint_hard_gates.py::test_run", "local/test_direct_joint_hard_gates.py::test_report"),
    "direct-joint-hard-fine": ("local/test_direct_joint_hard_fine.py::test_run", "local/test_direct_joint_hard_fine.py::test_report"),
    "direct-joint-hard-relaxed": ("local/test_direct_joint_hard_relaxed.py::test_run", "local/test_direct_joint_hard_relaxed.py::test_report"),
    "direct-reverse-delayed": ("local/test_direct_reverse_delayed.py::test_run", "local/test_direct_reverse_delayed.py::test_report"),
    "direct-reverse-tail": ("local/test_direct_reverse_tail.py::test_run", "local/test_direct_reverse_tail.py::test_report"),
    "direct-reverse-tail-report": ("local/test_direct_reverse_tail.py::test_report",),
    "direct-reverse-single": ("local/test_direct_reverse_single.py::test_run", "local/test_direct_reverse_single.py::test_report"),
    "direct-reverse-single-report": ("local/test_direct_reverse_single.py::test_report",),
    "direct-sampling-profile": ("local/test_direct_sampling_profile.py::test_run", "local/test_direct_sampling_profile.py::test_report"),
    "direct-sampling-profile-report": ("local/test_direct_sampling_profile.py::test_report",),
    "sampling-overhead": ("local/test_sampling_overhead.py::test_build", "local/test_sampling_overhead.py::test_run", "local/test_sampling_overhead.py::test_report"),
    "direct-sampling-window": ("local/test_direct_sampling_window.py::test_build", "local/test_direct_sampling_window.py::test_run", "local/test_direct_sampling_window.py::test_report"),
    "direct-sampling-window-full": ("local/test_direct_sampling_window.py::test_full", "local/test_direct_sampling_window.py::test_report"),
    "direct-sampling-window-checks": ("local/test_direct_sampling_window.py::test_build",),
    "direct-sampling-profile-finish": ("local/test_direct_sampling_profile.py::test_finish", "local/test_direct_sampling_profile.py::test_report"),
    "native-chunk-build": ("local/test_native_chunk_benchmark.py::test_build",),
    "native-chunk-pilot": ("local/test_native_chunk_benchmark.py::test_pilot",),
    "native-chunk-full": ("local/test_native_chunk_benchmark.py::test_full",),
    "native-chunk-report": ("local/test_native_chunk_benchmark.py::test_report",),
    "native-chunk-failures": ("local/test_native_chunk_failures.py",),
    "native-chunk-width-audit": ("local/test_chunk_width_audit.py",),
    "direct-batch-before": ("local/test_direct_batch.py::test_stop_old", "local/test_direct_batch.py::test_before"),
    "direct-batch-stop-reverse": ("local/test_direct_batch.py::test_stop_reverse",),
    "direct-batch-build": ("local/test_direct_batch.py::test_build",),
    "direct-batch-tests": ("local/test_direct_batch.py::test_regressions",),
    "direct-batch-after": ("local/test_direct_batch.py::test_after",),
    "native-volume-direct": ("local/test_native_volume_direct.py::test_run",),
    "native-direct-width-sweep": ("local/test_native_direct_width_sweep.py",),
    "native-direct-width-lower": ("local/test_native_direct_width_sweep.py::test_lower_cutoffs", "local/test_native_direct_width_sweep.py::test_report"),
    "native-volume-direct-report": ("local/test_native_volume_direct.py::test_report",),
    "extraction-direct-audit": ("local/test_extraction_direct_audit.py",),
    "extraction-presence-pair": ("local/test_extraction_direct_audit.py::test_presence_pair",),
    "extraction-direct-report": ("local/test_extraction_direct_audit.py::test_report",),
    "stitched-reference": ("local/test_stitched_reference.py::test_run",),
    "stitched-reference-audit": ("local/test_stitched_reference.py::test_audit",),
    "stitched-presence-audit": ("local/test_stitched_reference.py::test_presence_audit",),
    "stitched-reference-report": ("local/test_stitched_reference.py::test_report",),
    "fusion-route-pilot": ("local/test_fusion_routes.py::test_pilot",),
    "fusion-route-full": ("local/test_fusion_routes.py::test_full",),
    "fusion-route-report": ("local/test_fusion_routes.py::test_report",),
    "fusion-route-resume": ("local/test_fusion_routes.py::test_resume",),
    "reduced-route-lengths": ("local/test_raw_path_lengths.py",),
    "reduced-route-pilot": ("local/test_reduced_routes.py::test_pilot",),
    "reduced-route-full": ("local/test_reduced_routes.py::test_full",),
    "reduced-route-resume": ("local/test_reduced_routes.py::test_resume",),
    "reduced-route-report": ("local/test_reduced_routes.py::test_report",),
    "unambiguous-route-pilot": ("local/test_unambiguous_routes.py::test_pilot",),
    "unambiguous-route-full": ("local/test_unambiguous_routes.py::test_full",),
    "unambiguous-route-report": ("local/test_unambiguous_routes.py::test_report",),
    "owned-route-pilot": ("local/test_owned_route_export.py::test_pilot",),
    "owned-route-full": ("local/test_owned_route_export.py::test_full",),
    "owned-route-report": ("local/test_owned_route_export.py::test_report",),
    "paris4-repair-build": ("local/test_paris4_repair.py::test_build",),
    "paris4-repair-tests": ("local/test_paris4_repair.py::test_regressions",),
    "paris4-repair-pilot": ("local/test_paris4_repair.py::test_device_regressions", "local/test_paris4_repair.py::test_boundary_pilot"),
    "paris4-repair-launch": ("local/test_paris4_repair.py::test_launch",),
    "paris4-repair-status": ("local/test_paris4_repair.py::test_status",),
    "paris4-repair-report": ("local/test_paris4_repair.py::test_report",),
    "paris4-route-lengths": ("local/test_paris4_repair.py::test_lengths",),
    "full-route-build": ("local/test_full_route_export.py::test_build",),
    "full-route-tests": ("local/test_full_route_export.py::test_regressions",),
    "full-route-pilot": ("local/test_full_route_export.py::test_pilot",),
    "full-route-report": ("local/test_full_route_export.py::test_report",),
    "trace-fast-decoder": ("local/test_trace_fast_decoder.py",),
    "trace-fast-decoder-report": ("local/test_trace_fast_decoder.py::test_report",),
    "reader-multichannel": ("local/test_reader_multichannel.py",),
    "reader-multichannel-report": ("local/test_reader_multichannel_report.py",),
    "reader-layout": ("local/test_reader_layout.py",),
    "reader-layout-report": ("local/test_reader_layout_report.py::test_report",),
    "reader-layout-pool-report": ("local/test_reader_layout_report.py::test_pool_report",),
    "reader-layout-pool": ("local/test_reader_layout_pool.py",),
    "reader-scaling-diagnosis": ("local/test_reader_scaling_diagnosis.py",),
    "reader-decode-reuse": ("local/test_reader_decode_reuse.py",),
    "reader-decode-reuse-report": ("local/test_reader_decode_reuse_report.py",),
    "reader-decode-reuse-tests": ("local/test_reader_decode_reuse.py::test_regressions",),
    "reader-scaling-report": ("local/test_reader_scaling_report.py",),
    "reader-roi-report": ("local/test_reader_roi_report.py",),
    "reader-roi-after": ("local/test_reader_roi_after.py",),
    "reader-roi-single-after": ("local/test_reader_roi_single_after.py",),
    "reader-wake-capacity": ("local/test_reader_wake_capacity.py",),
    "reader-roi-multi-before": ("local/test_reader_roi_multi.py",),
    "reader-report": ("local/test_reader_report.py",),
    "reader-regressions": ("local/test_reader_final.py::test_regressions",),
    "reader-isolated": ("local/test_isolated_chunk_reader.py",),
    "reader-isolated-after": ("local/test_isolated_chunk_reader_after.py",),
    "reader-final": ("local/test_reader_final.py",),
    "reader-fixed-pool": ("local/test_reader_fixed_pool.py",),
    "reader-capacity": ("local/test_reader_capacity.py",),
    "reader-roi": ("local/test_reader_roi.py",),
    "runner-tests": ("scripts/test_fiber_test_runner.py",),
    "manager-tests": ("gpu_manager/tests/test_exclusive.py",),
    "loader-build": (
        "local/test_queued_wake_work.py::test_configure",
        "local/test_queued_wake_work.py::test_build",
    ),
    "loader-tests": (
        "local/test_queued_wake_work.py::test_queue",
        "local/test_queued_wake_work.py::test_regressions",
    ),
    "loader-scaling": ("local/test_queued_wake_scaling.py",),
    "loader-report": ("local/test_queued_scaling_report.py",),
}


def load_suites(path):
    if not path.exists():
        return ["runner-tests"]
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or set(config) != {"suites"}:
        raise ValueError('Expected only {"suites": ["named-suite", ...]}')
    suites = config["suites"]
    if (not isinstance(suites, list) or not suites or
            any(not isinstance(s, str) or s not in SUITES for s in suites)):
        raise ValueError("Choose nonempty suites from: " + ", ".join(SUITES))
    if len(set(suites)) != len(suites):
        raise ValueError("Duplicate suites are not allowed")
    return suites


def run(suites, root=ROOT):
    # Preflight all selections before starting any suite.
    for suite in suites:
        for node in SUITES[suite]:
            path = root / node.split("::", 1)[0]
            if not path.is_file():
                raise ValueError(f"Missing harness for {suite}: {path}")
    env = dict(os.environ, AGENTS_AGENT_MODE="1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = os.pathsep.join((str(root / "gpu_manager/src"),
                                        str(root), env.get("PYTHONPATH", "")))
    # Do not inherit pytest CLI injection; GPU benchmarks manage their own grants.
    env.pop("PYTEST_ADDOPTS", None)
    for suite in suites:
        print(f"[fiber-tests] {suite}: {' '.join(SUITES[suite])}", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "-s", "-q", *SUITES[suite]],
            cwd=root, env=env,
        )
        if result.returncode:
            return result.returncode if result.returncode > 0 else 128 - result.returncode
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        print("No arguments accepted; edit local/fiber_tests.json", file=sys.stderr)
        return 2
    try:
        suites = load_suites(CONFIG)
        print(f"[fiber-tests] config={CONFIG}; suites={','.join(suites)}", flush=True)
        return run(suites)
    except (ValueError, OSError) as exc:
        print(f"[fiber-tests] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
