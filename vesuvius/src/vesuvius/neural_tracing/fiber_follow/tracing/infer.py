"""Trace with a direct follower checkpoint."""
from vesuvius.neural_tracing.fiber_follow.tracing.inference import main as infer
from vesuvius.neural_tracing.fiber_follow.train.train import load_checkpoint
from vesuvius.neural_tracing.fiber_follow.data.observations import DirectTracer


def main(argv=None):
    return infer(argv, checkpoint_loader=load_checkpoint, tracer_class=DirectTracer)


if __name__ == '__main__':
    main()
