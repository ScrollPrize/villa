"""Trace with either fiber follower checkpoint."""
from vesuvius.neural_tracing.tracers.fibers.tracing.inference import main as infer
from vesuvius.neural_tracing.tracers.models.checkpoint import load_checkpoint
from vesuvius.neural_tracing.tracers.fibers.tracing.trace import FiberTracer


def main(argv=None):
    return infer(argv, checkpoint_loader=load_checkpoint, tracer_class=FiberTracer)


if __name__ == '__main__':
    main()
