"""Evaluate a direct follower with the shared protocol on the dataset-config held-out sources.

Paris 4 uses its frozen monitor/calibration/final seeds; each AFV source uses its own
seeded validation manifest (the same seeds the trainer writes as ``validation_<name>.json``).
AFV holdouts carry a neighbor detector over their own catalog; Paris 4 bank shards cover
training parents only, so their held-out identity coverage is reported as absent.
"""
from pathlib import Path

from vesuvius.neural_tracing.fiber_follow.shared.evaluation import main as protocol
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolume
from .data import DirectTracer
from .train import load_checkpoint

DATASET_CONFIG = Path(__file__).parents[1]/'configs'/'mixed_ct_datasets_paris50.json'


def dataset_sources(args, config=DATASET_CONFIG):
    from .datasets import AFVBank, load_primary_dataset, open_afv_source, read_dataset_config, validation_manifest
    from .bank_geometry import BankSwitchDetector
    _, _, _, paris_spec, checkpoint = load_checkpoint(args.checkpoint, 'cpu')
    document, _ = read_dataset_config(config)
    normalization = checkpoint['ct_normalization']
    sources = []
    for source in document['sources']:
        if args.sources and source['name'] not in args.sources:
            continue
        if source['kind'] == 'paris4':
            from ..shared.ct_normalization import prepare_normalization
            spec = paris_spec  # the checkpoint's own volume, bound to its recorded CT normalization
            prepare_normalization(Path(args.out).parent if args.command == 'run' else args.out, [spec], known=normalization)
            _, _, validation, manifest = load_primary_dataset(document, spec)
            detector = None
        else:
            _, validation, spec, _ = open_afv_source(source, document['cache_dir'], normalization)
            manifest = validation_manifest(validation, spec, source['validation']['seed'])
            detector = BankSwitchDetector([AFVBank(validation)])
        sources.append(dict(name=source['name'], fibers=validation, manifest=manifest, detector=detector,
                            volume=FiberVolume(spec)))
    return sources


def main(argv=None):
    return protocol(argv, checkpoint_loader=load_checkpoint, tracer_class=DirectTracer, source_loader=dataset_sources)


if __name__ == '__main__':
    main()
