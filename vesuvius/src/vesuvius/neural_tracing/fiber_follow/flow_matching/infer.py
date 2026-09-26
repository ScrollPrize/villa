"""Trace with a flow-matching checkpoint."""
from vesuvius.neural_tracing.fiber_follow.shared.infer import main as infer
from vesuvius.neural_tracing.fiber_follow.flow_matching.train import load_checkpoint


def main(argv=None):
    return infer(argv, checkpoint_loader=load_checkpoint)


if __name__ == '__main__':
    main()
