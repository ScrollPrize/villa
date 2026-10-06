"""Collect original-fiber replay with either fiber follower and the enlarged crop exclusion."""
from vesuvius.neural_tracing.fiber_follow.tracing.collection import main as collect
from vesuvius.neural_tracing.fiber_follow.train.train import load_checkpoint
from vesuvius.neural_tracing.fiber_follow.data.observations import FiberTracer


def switch_detector(args, checkpoint, fibers, band, spec):
    """Switch detection against neighboring annotations; AFV sources only (Paris 4 has no neighbor paths)."""
    if not (args.dataset_name and hasattr(fibers, 'metadata')):
        return None
    from vesuvius.neural_tracing.fiber_follow.data.afv_neighbors import AFVBank
    from vesuvius.neural_tracing.fiber_follow.data.bank_geometry import BankSwitchDetector
    return BankSwitchDetector([AFVBank(fibers)], args.bank_switch_tolerance, args.bank_own_tolerance)


def load_dataset(args, checkpoint):
    from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolumeSpec
    document = checkpoint.get('dataset_config')
    if not document:
        raise ValueError('Checkpoint has no mixed dataset configuration')
    source = next((s for s in document['sources'] if s['name'] == args.dataset_name), None)
    if source is None:
        raise ValueError('Unknown dataset source')
    if source['kind'] == 'paris4':
        from vesuvius.neural_tracing.fiber_follow.data.datasets import load_primary_dataset
        spec = FiberVolumeSpec.from_dict(checkpoint['vol_spec'])
        _,fibers,_,_ = load_primary_dataset(document,spec)
        return spec,fibers,None
    from vesuvius.neural_tracing.fiber_follow.data.datasets import open_afv_source
    fibers, _, spec, _ = open_afv_source(source, document['cache_dir'], checkpoint['ct_normalization'])
    return spec,fibers,None


def main(argv=None):
    return collect(argv, checkpoint_loader=load_checkpoint, tracer_class=FiberTracer,
                   bank_loader=switch_detector,dataset_loader=load_dataset)


if __name__ == '__main__':
    main()
