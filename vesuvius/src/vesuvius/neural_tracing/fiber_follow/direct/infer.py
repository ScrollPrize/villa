"""Trace and export VC3D JSON using a direct_curve_v1 checkpoint."""
from vesuvius.neural_tracing.fiber_follow.infer import main as infer
from vesuvius.neural_tracing.fiber_follow.direct.train import load_checkpoint
from vesuvius.neural_tracing.fiber_follow.direct.data import DirectTracer


def main(argv=None):
    return infer(argv, checkpoint_loader=load_checkpoint, tracer_class=DirectTracer)


if __name__ == '__main__':
    main()
