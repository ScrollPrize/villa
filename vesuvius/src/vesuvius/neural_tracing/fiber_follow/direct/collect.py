"""Collect original-fiber replay with the direct model and both crop exclusions."""
from vesuvius.neural_tracing.fiber_follow.collect import main as collect
from vesuvius.neural_tracing.fiber_follow.direct.train import load_checkpoint
from vesuvius.neural_tracing.fiber_follow.direct.data import DirectTracer


def configure(args, ck):
    if 'judge_architecture' not in ck:
        return {}
    from .judge_options import load_judge
    return dict(load_judge(ck, args.device), judge_explore_calls=args.explore_calls)


def main(argv=None):
    return collect(argv, checkpoint_loader=load_checkpoint, tracer_class=DirectTracer, configure_tracer=configure)


if __name__ == '__main__':
    main()
