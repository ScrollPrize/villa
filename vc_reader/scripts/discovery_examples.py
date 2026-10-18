"""Metadata-only catalog/user-source example; the supplied crop is not opened."""
import argparse
import json

from vc_reader import Affine, Catalog, Source, SourceCollection, Storage, VolumeRegistration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True, help='Local catalog snapshot')
    parser.add_argument('--sample', default='PHercParis4')
    parser.add_argument('--volume', required=True)
    args = parser.parse_args()
    catalog = Catalog.from_file(args.metadata)
    volume = catalog.sample(args.sample).volume(args.volume)
    crop = Source(
        'my-annotation', storage=Storage('/data/annotation.zarr', 'zarr'),
        role='label', axes=('z', 'y', 'x'),
        registration=VolumeRegistration(volume.sample.id, volume.id, affine=Affine.from_rows([
            [1, 0, 0, 120], [0, 1, 0, 80], [0, 0, 1, 40],
        ])), provenance={'description': 'Explicit crop at level-0 XYZ origin (120, 80, 40)'},
    )
    collection = SourceCollection(catalog, [crop])
    labels = collection.labels(volume=volume.level(2))
    print('Label sources:', len(labels), '(including user:my-annotation)')
    print('Registered volumes:', [v.id for v in collection.registered_volumes(volume=volume)])
    for source in collection.representations(volume=volume.level(2), mappable=False):
        print('Representation:', source.artifact_kind, source.storage.selection)
    # Both published and supplied descriptors use the same serialization contract.
    for source in [crop, *[s for s in labels if s.id.startswith('catalog:')][:1]]:
        restored = Source.from_dict(json.loads(json.dumps(source.to_dict())))
        assert restored == source
        print('Round trip:', source.id, '; unresolved:', source.unresolved_fields)


if __name__ == '__main__':
    main()
