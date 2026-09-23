"""Native CLI checks: independent ranking must never run reference fitting BP."""
import json
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
BINARY = REPO / "volume-cartographer/build/fiber-evaluation/bin/vc_fiber_trace_chunk"
CONFIG = REPO / "vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_independent_reference_1024_20260923.json"


@pytest.mark.parametrize("extra", [
    ["--reference-conditioning-diagnostic"],
    ["--reference-prune-offenders"],
    ["--winding-solver", "ceres"],
])
def test_rejects_reference_feedback_or_other_solver(extra):
    if not BINARY.is_file():
        pytest.skip("workspace native evaluator not built")
    args = json.loads(CONFIG.read_text())["commands"][0]["args"]
    completed = subprocess.run(
        [str(BINARY), *args, *extra], text=True, capture_output=True, timeout=15,
    )
    assert completed.returncode != 0
    assert "--reference-independent requires" in completed.stdout + completed.stderr
