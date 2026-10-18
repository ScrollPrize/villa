"""Discover and verify the PHerc0009B CT/ink pair and companion masks."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys


def discover(catalog):
    from vc_reader import ScrollSource
    scroll = ScrollSource('PHerc0009B', '20250820154339', catalog=catalog)
    label = scroll.segments['20250919125754'].labels.ink.one(label_set='20260918')
    return label, label.renders.zarr.one()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', help='local catalog snapshot; defaults to published metadata')
    parser.add_argument('--cache-dir', default='/tmp/vc-reader-paired-cache')
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--native-dir', type=Path)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.native_dir:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
        from bench_pipeline import load_extensions
        load_extensions(args.native_dir)
    import numpy as np
    from vc_reader import load
    from vc_reader.reader import ZarrGroup
    catalog = load(args.metadata) if args.metadata else load(cache_dir=args.cache_dir)
    label, render = discover(catalog)
    options = dict(cache_dir=args.cache_dir, offline=args.offline, memory_cache=64 << 20, threads=1)
    y, x = slice(18576, 18704), slice(7984, 8112)
    with label.open_paired(render=render, layers=slice(40, 65), backend=args.backend, **options) as reader:
        patch = reader[y, x]
        data = {k: v.numpy() if hasattr(v, 'numpy') else v for k, v in patch.data.items()}
        masks = {k: v.numpy() if hasattr(v, 'numpy') else v for k, v in patch.masks['label'].items()}
        # Independently opened arrays: no pairing or sampling helper involved.
        with ExitStack() as stack:
            image_group = stack.enter_context(ZarrGroup.open(render.source.storage.location, **options))
            label_group = stack.enter_context(ZarrGroup.open(label.source.storage.location, **options))
            np.testing.assert_array_equal(data['image'], image_group.level(0)[40:65, y, x][None])
            np.testing.assert_array_equal(data['label'], label_group.level(0)[y, x][None])
            for companion in label.source.companions:
                group = stack.enter_context(ZarrGroup.open(companion.source.storage.location, **options))
                np.testing.assert_array_equal(masks[companion.name], group.level(0)[y, x][None])
        assert np.count_nonzero(data['label']) == 9926
        print(json.dumps(dict(scroll=label.scroll.id, volume=label.volume.id, segment=label.segment.id,
                              label_set=label.label_set, image_shape=data['image'].shape, label_shape=data['label'].shape,
                              label_nonzero=int(np.count_nonzero(data['label'])), masks=list(masks), stats=reader.stats()), indent=2))


if __name__ == '__main__':
    main()
