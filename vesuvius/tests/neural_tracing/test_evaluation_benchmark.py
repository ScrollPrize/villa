"""CPU-only argument checks for the bounded GPU evaluation driver."""
import pytest

from vesuvius.neural_tracing.fiber_trace_3d.evaluation_benchmark import parser


def test_defaults_are_bounded_and_all_gpu():
    args = parser().parse_args(["--reference", "config.json", "--output", "/tmp/report"])
    assert (args.devices, args.batch_size, args.samples, args.passes, args.timeout) == (
        "all", 16, 128, 2, 300)
    assert args.backend == "dataloader"


@pytest.mark.parametrize("backend", ["dataloader", "pipeline", "sync"])
def test_backend_selection(backend):
    args = parser().parse_args(["--evaluation-only", "--reference", "config.json",
                               "--output", "/tmp/report", "--backend", backend])
    assert args.evaluation_only and args.backend == backend


def test_unknown_backend_rejected():
    with pytest.raises(SystemExit):
        parser().parse_args(["--reference", "c", "--output", "o", "--backend", "other"])
