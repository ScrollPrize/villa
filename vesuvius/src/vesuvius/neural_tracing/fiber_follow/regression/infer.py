"""Trace with a direct follower checkpoint."""
from vesuvius.neural_tracing.fiber_follow.shared.infer import main as infer
from .train import load_checkpoint
from .data import DirectTracer


def main(argv=None):
    return infer(argv, checkpoint_loader=load_checkpoint, tracer_class=DirectTracer)


if __name__ == '__main__':
    main()
