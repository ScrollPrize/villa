"""Collect original-fiber replay with the direct model and the enlarged crop exclusion."""
from vesuvius.neural_tracing.fiber_follow.shared.collect import main as collect
from vesuvius.neural_tracing.fiber_follow.regression.train import load_checkpoint
from vesuvius.neural_tracing.fiber_follow.regression.data import DirectTracer


def failure_banks(args, checkpoint, fibers, band, spec):
    from pathlib import Path
    from .neighbor_bank import NeighborBank
    from .bank_geometry import BankSwitchDetector
    options = checkpoint.get('training_options', {})
    paths = args.failure_bank or [options.get(k) for k in ('negative_bank', 'near_negative_bank')]
    paths = list(dict.fromkeys(str(Path(p).resolve()) for p in paths if p))
    banks = [NeighborBank(p, fibers, band, grid_scale=spec.grid_scale) for p in paths]
    for bank in banks:
        bank.validate_volume(spec)
    return BankSwitchDetector(banks, args.bank_switch_tolerance, args.bank_own_tolerance) if banks else None


def main(argv=None):
    return collect(argv, checkpoint_loader=load_checkpoint, tracer_class=DirectTracer, bank_loader=failure_banks)


if __name__ == '__main__':
    main()
