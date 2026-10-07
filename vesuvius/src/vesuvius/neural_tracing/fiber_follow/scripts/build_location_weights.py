"""Build the fresh-location weight caches (data/location_weights.py) for every source of a dataset configuration.
Usage: python scripts/build_location_weights.py DATASET_CONFIG [--sources NAME ...] [--out DIR] [--workers 16] [--force]
The trainer reads them from output/location_weights (or the run option location_weights_dir)."""
import argparse
from pathlib import Path


def main(argv=None):
    from vesuvius.neural_tracing.fiber_follow.data import location_weights
    from vesuvius.neural_tracing.fiber_follow.data.afv import AFVFibers
    from vesuvius.neural_tracing.fiber_follow.data.data import load_fibers
    from vesuvius.neural_tracing.fiber_follow.data.datasets import read_dataset_config
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dataset_config')
    ap.add_argument('--sources', nargs='+')
    ap.add_argument('--out', default=None)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)
    document, _ = read_dataset_config(args.dataset_config)
    for source in document['sources']:
        if args.sources and source['name'] not in args.sources:
            continue
        out = location_weights.cache_path(source, args.out)
        if out.exists() and not args.force:
            print(f"{source['name']}: {out} exists", flush=True)
            continue
        if source['kind'] == 'paris4':
            fibers = load_fibers(source['fibers'], grid_scale=source['grid_scale'])
        else:
            collection = AFVFibers(source['path'], source['grid_scale'], split='all', sha256=source.get('sha256'))
            fibers = [collection.geometry(i) for i in range(len(collection))]
        location_weights.build(source, Path(args.dataset_config).resolve(), fibers, out, args.workers)


if __name__ == '__main__':
    main()
