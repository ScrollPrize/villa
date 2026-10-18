"""Read explicit catalog/user targets in a catalog volume's selected grid."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--sample', required=True)
    parser.add_argument('--volume', required=True)
    parser.add_argument('--level', required=True, type=int)
    parser.add_argument('--start', required=True, type=int, nargs=3)
    parser.add_argument('--size', required=True, type=int, nargs=3)
    parser.add_argument('--target', action='append', default=[], metavar='NAME=SOURCE_JSON')
    parser.add_argument('--catalog-target', action='append', default=[], metavar='NAME=SOURCE_ID')
    parser.add_argument('--cache-dir', action='append')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--cache-writes', action='store_true')
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--native-dir', type=Path)
    args = parser.parse_args()
    if args.native_dir:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
        from bench_pipeline import load_extensions
        load_extensions(args.native_dir)
    from vc_reader import Catalog, Source
    from vc_reader.sampling import open_sample
    catalog = Catalog.from_file(args.metadata)
    reference = catalog.sample(args.sample).volume(args.volume).level(args.level)
    targets = {}
    for selection in args.catalog_target:
        name, identifier = selection.split('=', 1)
        if name in targets:
            raise ValueError('duplicate target name: '+name)
        targets[name] = catalog.sources.get(identifier)
    for selection in args.target:
        name, path = selection.split('=', 1)
        if name in targets:
            raise ValueError('duplicate target name: '+name)
        targets[name] = Source.from_dict(json.loads(Path(path).read_text()))
    with open_sample(reference, targets, backend=args.backend, cache_dir=args.cache_dir,
                     cache_writes=args.cache_writes, offline=args.offline) as reader:
        result = reader.read(args.start, args.size)
        print(json.dumps(dict(data={k: dict(shape=v.shape, dtype=str(v.dtype)) for k, v in result.data.items()},
                              valid={k: dict(shape=v.shape, dtype=str(v.dtype)) for k, v in result.valid.items()},
                              masks={k: list(v) for k, v in result.masks.items()}, metadata=result.metadata), indent=2))


if __name__ == '__main__':
    main()
