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


@pytest.mark.parametrize("limit", ["0", "-1", "1.5"])
def test_rejects_invalid_winding_limit(limit):
    if not BINARY.is_file():
        pytest.skip("workspace native evaluator not built")
    args = json.loads(CONFIG.read_text())["commands"][0]["args"]
    completed = subprocess.run(
        [str(BINARY), *args, "--winding-message-iterations", limit],
        text=True, capture_output=True, timeout=15,
    )
    assert completed.returncode != 0
    output = completed.stdout + completed.stderr
    assert "winding-message-iterations" in output
    assert "unknown option" not in output


def test_winding_override_parses_independently_of_shared_bp_limit():
    if not BINARY.is_file():
        pytest.skip("workspace native evaluator not built")
    args = json.loads(CONFIG.read_text())["commands"][0]["args"]
    # Force a later semantic error rather than load a real benchmark. This
    # establishes that both independently supplied limits were accepted.
    completed = subprocess.run(
        [str(BINARY), *args, "--bp-message-iterations", "500",
         "--winding-message-iterations", "1000",
         "--reference-conditioning-diagnostic"],
        text=True, capture_output=True, timeout=15,
    )
    assert completed.returncode != 0
    assert "--reference-independent requires" in completed.stdout + completed.stderr


def test_winding_override_rejects_non_bp_mode():
    if not BINARY.is_file():
        pytest.skip("workspace native evaluator not built")
    completed = subprocess.run(
        [str(BINARY), "trace", "-", "--winding-message-iterations", "500"],
        text=True, capture_output=True, timeout=15,
    )
    assert completed.returncode != 0
    assert "requires direction-ablation joint-grid mixed BP" in completed.stdout + completed.stderr


@pytest.mark.parametrize("extra", [["--winding-solver", "ceres"], ["--bp-inference", "min-sum"]])
def test_winding_override_rejects_solver_that_does_not_use_it(extra):
    if not BINARY.is_file():
        pytest.skip("workspace native evaluator not built")
    args = json.loads(CONFIG.read_text())["commands"][0]["args"]
    completed = subprocess.run(
        [str(BINARY), *args, "--winding-message-iterations", "1000", *extra],
        text=True, capture_output=True, timeout=15,
    )
    assert completed.returncode != 0
    assert "requires direction-ablation joint-grid mixed BP" in completed.stdout + completed.stderr


@pytest.mark.parametrize('value', ['0', '-1', 'nan'])
def test_measurement_disable_rejects_invalid_cost(value):
    if not BINARY.is_file():
        pytest.skip('workspace native evaluator not built')
    completed = subprocess.run([str(BINARY), 'trace', '-', '--winding-measurement-disable-cost', value],
                               text=True, capture_output=True, timeout=15)
    assert completed.returncode != 0
    assert 'unknown option' not in completed.stdout + completed.stderr


def test_piece_cuts_reject_wrong_mode():
    if not BINARY.is_file():
        pytest.skip('workspace native evaluator not built')
    completed = subprocess.run([str(BINARY), 'trace', '-', '--piece-cut-hints', '/nonexistent.json'],
                               text=True, capture_output=True, timeout=15)
    assert completed.returncode != 0
    assert 'Piece cut hints require direction-ablation' in completed.stdout + completed.stderr


def test_local_parallel_ablation_requires_local_events():
    if not BINARY.is_file():
        pytest.skip('workspace native evaluator not built')
    completed = subprocess.run([str(BINARY), 'direction-ablation', '-', '--local-no-parallel'],
                               text=True, capture_output=True, timeout=15)
    assert completed.returncode != 0
    assert '--local-no-parallel requires' in completed.stdout + completed.stderr


def test_local_events_reject_old_piece_multipliers():
    if not BINARY.is_file():
        pytest.skip('workspace native evaluator not built')
    completed = subprocess.run([str(BINARY), 'direction-ablation', '-', '--local-constraint-events',
                                '--winding-defect-multipliers', '/nonexistent.json'],
                               text=True, capture_output=True, timeout=15)
    assert completed.returncode != 0
    assert 'cannot use legacy piece IDs' in completed.stdout + completed.stderr


@pytest.mark.parametrize('value', ['-1', 'nan'])
def test_continuation_cut_rejects_invalid_cost(value):
    if not BINARY.is_file():
        pytest.skip('workspace native evaluator not built')
    completed = subprocess.run([str(BINARY), 'direction-ablation', '-', '--winding-continuity-cut-cost', value],
                               text=True, capture_output=True, timeout=15)
    assert completed.returncode != 0
    output = completed.stdout + completed.stderr
    assert ('Continuity cut cost must be finite and nonnegative' in output or
            '--winding-continuity-cut-cost requires a finite number' in output)


def test_continuation_cut_rejects_non_joint_mode():
    if not BINARY.is_file():
        pytest.skip('workspace native evaluator not built')
    completed = subprocess.run([str(BINARY), 'direction-ablation', '-', '--winding-continuity-cut-cost', '1'],
                               text=True, capture_output=True, timeout=15)
    assert completed.returncode != 0
    assert 'Continuity cut cost requires direction-ablation joint-grid mixed BP' in completed.stdout + completed.stderr
