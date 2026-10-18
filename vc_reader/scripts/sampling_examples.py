"""Read explicit catalog/user targets in a catalog volume's selected grid."""
import argparse
import json
from pathlib import Path
import sys


def index(value):
    """Parse one spatial integer or unit-step slice without evaluating code."""
    try:
        if ':' not in value:
            return int(value)
        parts = value.split(':')
        if len(parts) not in (2, 3):
            raise ValueError
        result = slice(*(int(part) if part else None for part in parts))
        if result.step not in (None, 1):
            raise ValueError
        return result
    except ValueError:
        raise argparse.ArgumentTypeError('expected an integer or unit-step START:STOP slice') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--sample', required=True)
    parser.add_argument('--volume', required=True)
    parser.add_argument('--level', required=True, type=int)
    parser.add_argument('--slice', required=True, type=index, nargs=3, metavar=('Z', 'Y', 'X'),
                        help='ZYX indices/slices, e.g. 100:132 200:328 300:428')
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
    reference = catalog.samples[args.sample].volumes[args.volume].level(args.level)
    targets = {}
    for selection in args.catalog_target:
        name, identifier = selection.split('=', 1)
        if name in targets:
            raise ValueError('duplicate target name: '+name)
        targets[name] = catalog.samples[args.sample].artifacts[identifier].source
    for selection in args.target:
        name, path = selection.split('=', 1)
        if name in targets:
            raise ValueError('duplicate target name: '+name)
        targets[name] = Source.from_dict(json.loads(Path(path).read_text()))
    with open_sample(reference, targets, backend=args.backend, cache_dir=args.cache_dir,
                     cache_writes=args.cache_writes, offline=args.offline) as reader:
        result = reader[tuple(args.slice)]
        print(json.dumps(dict(data={k: dict(shape=v.shape, dtype=str(v.dtype)) for k, v in result.data.items()},
                              valid={k: dict(shape=v.shape, dtype=str(v.dtype)) for k, v in result.valid.items()},
                              masks={k: list(v) for k, v in result.masks.items()}, metadata=result.metadata), indent=2))


if __name__ == '__main__':
    main()
