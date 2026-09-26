"""Trace with optional CT judge enforcement or a locked calibrated selection."""
from ..infer import main as infer
from .train import load_checkpoint
from .data import DirectTracer
from .judge_options import load_judge
from .judge_policy import JudgePolicyConfig


def parser(ap):
    ap.add_argument('--judge', action='store_true', help='Explicitly opt into development judge enforcement')
    ap.add_argument('--forecast-only', action='store_true')
    ap.add_argument('--selection', help='Locked joint calibration selection')
    ap.add_argument('--judge-accept', type=float, default=None)
    ap.add_argument('--judge-alarm', type=float, default=None)


def configure(args, ck):
    selection = None
    if args.selection:
        from .judge_evaluation import load_selection
        selection = load_selection(args.selection, args.checkpoint)
        if args.judge_accept is not None or args.judge_alarm is not None:
            raise ValueError('Threshold overrides are forbidden with a locked selection')
        args.confidence = selection['threshold']
        args.n_commit = selection['n_commit']
        args.sampling_seed = selection['sampling_seed']
        args.max_len = selection['max_len']
    if args.forecast_only:
        return {}
    if selection is None and not args.judge:
        return {}
    if selection is not None and selection['opt_in'] and not args.judge:
        raise ValueError('This calibration requires explicit --judge opt-in')
    options = load_judge(ck, args.device)
    if selection:
        options['judge_policy'] = JudgePolicyConfig(**selection['judge_policy'])
        if options['judge_slices'].open().identity != selection['judge_source_sha256']:
            raise ValueError('Selected native source changed')
    else:
        policy = options['judge_policy'].to_dict()
        if args.judge_accept is not None:
            policy['accept'] = args.judge_accept
        if args.judge_alarm is not None:
            policy['alarm'] = args.judge_alarm
        options['judge_policy'] = JudgePolicyConfig(**policy)
    return options


def main(argv=None):
    return infer(argv, checkpoint_loader=load_checkpoint, tracer_class=DirectTracer,
                 configure_parser=parser, configure_tracer=configure)


if __name__ == '__main__':
    main()
