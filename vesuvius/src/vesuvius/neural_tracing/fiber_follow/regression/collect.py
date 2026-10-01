"""Collect original-fiber replay with the direct model and the enlarged crop exclusion."""
from vesuvius.neural_tracing.fiber_follow.shared.collect import main as collect
from vesuvius.neural_tracing.fiber_follow.regression.train import load_checkpoint
from vesuvius.neural_tracing.fiber_follow.regression.data import DirectTracer


def failure_banks(args, checkpoint, fibers, band, spec):
    from pathlib import Path
    from .neighbor_bank import NeighborBank
    from .bank_geometry import BankSwitchDetector
    if args.dataset_name and hasattr(fibers,'metadata'):
        from .datasets import AFVBank
        return BankSwitchDetector([AFVBank(fibers)],args.bank_switch_tolerance,args.bank_own_tolerance)
    options = checkpoint.get('training_options', {})
    paths = args.failure_bank or [options.get(k) for k in ('negative_bank', 'near_negative_bank')]
    paths = list(dict.fromkeys(str(Path(p).resolve()) for p in paths if p))
    if args.dataset_name:
        from .datasets import HoldoutFilteredBank,load_primary_dataset
        from ..shared.data import ZBand
        document = checkpoint['dataset_config']
        _,_,heldout,_ = load_primary_dataset(document,spec)
        source = next(s for s in document['sources'] if s['name']==args.dataset_name)
        bank_band = ZBand(*(v/spec.grid_scale for v in source['val_z']))
        banks = [HoldoutFilteredBank(p,fibers,bank_band,grid_scale=spec.grid_scale,heldout=heldout) for p in paths]
    else:
        banks = [NeighborBank(p, fibers, band, grid_scale=spec.grid_scale) for p in paths]
    for bank in banks:
        bank.validate_volume(spec)
    return BankSwitchDetector(banks, args.bank_switch_tolerance, args.bank_own_tolerance) if banks else None


def load_dataset(args, checkpoint):
    import hashlib
    from pathlib import Path
    from ..shared.afv import AFVFibers
    from ..shared.data import ZBand
    from ..shared.volume import FiberVolumeSpec
    document = checkpoint.get('dataset_config')
    if not document:
        raise ValueError('Checkpoint has no mixed dataset configuration')
    source = next((s for s in document['sources'] if s['name'] == args.dataset_name), None)
    if source is None:
        raise ValueError('Unknown dataset source')
    if source['kind'] == 'paris4':
        from .datasets import load_primary_dataset
        spec = FiberVolumeSpec(**checkpoint['vol_spec'])
        _,fibers,_,_ = load_primary_dataset(document,spec)
        return spec,fibers,None
    digest = hashlib.sha256()
    with Path(source['path']).open('rb') as stream:
        for block in iter(lambda: stream.read(8<<20), b''):
            digest.update(block)
    if digest.hexdigest() != source['sha256']:
        raise ValueError('AFV source changed since training')
    scale = source['grid_scale']
    fibers = AFVFibers(source['path'],scale,validation=source['validation'],sha256=digest.hexdigest())
    from .datasets import ct_source_spec
    from ..shared.ct_normalization import volume_key
    spec = ct_source_spec(source, document['cache_dir'])
    spec.ct_normalization = checkpoint['ct_normalization']['volumes'][volume_key(spec)]
    return spec,fibers,None


def main(argv=None):
    return collect(argv, checkpoint_loader=load_checkpoint, tracer_class=DirectTracer,
                   bank_loader=failure_banks,dataset_loader=load_dataset)


if __name__ == '__main__':
    main()
