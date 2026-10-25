"""No-argument, allowlisted test/benchmark launcher. Never installs dependencies."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "local/fiber_tests.json"
SUITES = {
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
