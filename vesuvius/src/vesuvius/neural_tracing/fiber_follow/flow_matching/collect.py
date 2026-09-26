"""Collect original-fiber replay with the flow-matching model."""
from vesuvius.neural_tracing.fiber_follow.shared.collect import main as collect
from vesuvius.neural_tracing.fiber_follow.flow_matching.train import load_checkpoint


def main(argv=None):
    return collect(argv, checkpoint_loader=load_checkpoint)


if __name__ == '__main__':
    main()
