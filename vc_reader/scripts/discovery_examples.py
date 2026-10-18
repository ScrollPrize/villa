"""Browse scroll relationships and attach an explicit local source without I/O."""
import argparse
import json
from vc_reader import Affine, Catalog, ScrollSource, Source, Storage, VolumeRegistration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--sample', default='PHercParis4')
    parser.add_argument('--volume', required=True)
    args = parser.parse_args()
    catalog = Catalog.from_file(args.metadata)
    scroll = ScrollSource(args.sample, args.volume, catalog=catalog)
    crop = Source('annotation', Storage('/data/annotation.zarr', 'zarr'), role='label', axes=('z', 'y', 'x'),
                  registration=VolumeRegistration(scroll.id, scroll.volume.id,
                      affine=Affine.from_rows(((1, 0, 0, 120), (0, 1, 0, 80), (0, 0, 1, 40)))))
    view = scroll.with_source(crop, kind='label', volume=scroll.volume)
    print('Labels:', [s.id for s in view.labels])
    print('Representations:', [(s.id, s.unresolved_fields) for s in view.representations])
    print('8 um candidates:', [(m.volume.id, m.level, m.verified) for m in view.volumes.at_resolution(8, rel_tol=.02)])
    restored = ScrollSource.from_dict(json.loads(json.dumps(view.to_dict())))
    assert restored.labels[crop.id].source == view.labels[crop.id].source
    print('Overlay descriptor round trip passed')


if __name__ == '__main__':
    main()
