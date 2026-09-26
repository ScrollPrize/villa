"""Collect original-fiber replay with the direct model and both crop exclusions."""
from vesuvius.neural_tracing.fiber_follow.collect import main as collect
from vesuvius.neural_tracing.fiber_follow.direct.train import load_checkpoint
from vesuvius.neural_tracing.fiber_follow.direct.data import DirectTracer


def main(argv=None):
    return collect(argv, checkpoint_loader=load_checkpoint, tracer_class=DirectTracer)


if __name__ == '__main__':
    main()
