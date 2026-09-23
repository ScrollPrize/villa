"""Reverse checking must be explicitly enabled on dense crop beam tracing."""
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
BINARY = REPO / "volume-cartographer/build/fiber-evaluation/bin/vc_fiber_trace_chunk"


@pytest.mark.parametrize("args", [
    ["trace", "-", "--reverse-consistency"],
    ["trace", "-", "--trace-tracer", "imported", "--reverse-consistency"],
    ["trace", "-", "--trace-tracer", "beam", "--no-reverse-consistency", "--reverse-backoff", "40"],
    ["reference-replay-benchmark", "-", "--reverse-consistency"],
    ["visualize", "-", "--reverse-consistency"],
])
def test_reverse_requires_explicit_dense_mode(args):
    if not BINARY.is_file():
        pytest.skip("workspace native evaluator not built")
    result = subprocess.run([str(BINARY), *args], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "reverse options require" in result.stdout + result.stderr


@pytest.mark.parametrize("option", ["--reverse-normal", "--reverse-tangential", "--reverse-backoff"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_reverse_rejects_invalid_parameters(option, value):
    if not BINARY.is_file():
        pytest.skip("workspace native evaluator not built")
    result = subprocess.run(
        [str(BINARY), "trace", "-", "--trace-tracer", "beam",
         "--reverse-consistency", option, value],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode != 0
    assert "finite" in result.stdout + result.stderr


@pytest.mark.parametrize("flags", [
    ["--reverse-consistency", "--no-reverse-consistency"],
    ["--no-reverse-consistency", "--reverse-consistency"],
])
def test_conflicting_reverse_flags(flags):
    if not BINARY.is_file():
        pytest.skip("workspace native evaluator not built")
    result = subprocess.run(
        [str(BINARY), "trace", "-", "--trace-tracer", "beam", *flags],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode != 0
    assert "conflicting reverse consistency" in result.stdout + result.stderr
