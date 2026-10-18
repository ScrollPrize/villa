"""Public discovery/descriptor contracts; no storage or optional dependencies."""
import json
import pickle
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import subprocess
import sys
import unittest

from vc_reader import (
    Affine, Catalog, Companion, Encoding, Plane, Selection, Source,
    Storage, Surface, SurfaceRegistration, ValueFrame,
    VolumeRegistration,
)


def catalog():
    def artifact(kind, identifier=None, **parameters):
        return {"type": kind, "id": identifier, "parameters": parameters}

    old = artifact('ink-labels', 'old', target_volume='a', label_set='old')
    new = artifact('ink-labels', 'new', target_volume='a', label_set='new')
    new['properties'] = {'supersedes': 'old', 'registration': 'native', 'arrays': ['supervision']}
    mapped = artifact('ink-labels', 'mapped', target_volume='b', label_set='new')
    absent = artifact('ink-labels', 'absent', target_volume='unpublished', label_set='remote')
    return Catalog({'samples': {'s': {
        'sample': {'id': 's', 'properties': {'volume_transforms': [{
            'from_volume_id': 'unpublished', 'transforms': [
                {'to_volume_id': 'a', 'matrix': Affine.scale(2).matrix}]}]}},
        'volumes': {
            'a': {'id': 'a', 'properties': {'superseded_by': {'volume_id': 'b'}}, 'data': [
                artifact('ome-zarr', 'image'), artifact('surface-prediction-zarr', 'prediction', level=2),
                artifact('unknown-bundle'), artifact('unknown-bundle')]},
            'b': {'id': 'b'},
        },
        'segments': {'seg': {'id': 'seg', 'original_volume_id': 'a', 'data': [old, new, mapped, absent]}},
    }}})


class SourceTests(unittest.TestCase):
    def test_descriptor_round_trips_and_immutability(self):
        provenance = {'origin': {'notes': ['original']}}
        registration = VolumeRegistration('s', 'a', affine=Affine.scale(4), declared_grid=Selection(level=0))
        crop = Source('crop', Storage('/tmp/crop.zarr', 'zarr'), role='label', axes=('z', 'y', 'x'),
                      registration=registration, provenance=provenance,
                      encoding=Encoding('color', (((255, 0, 0), 2**32),), 'uint64'))
        plane = VolumeRegistration('s', 'a', plane=Plane((10, 20, 30), (2, 0, 0), (0, 3, 0), (0, 0, 4)))
        surface = Surface('surface', Storage('/tmp/mesh', 'tifxyz'), 's', 'a', axes=('y', 'x'),
                          xyz_to_volume=Affine.scale(4), pixel_to_uv=((2, 0, 0), (0, 3, 0)))
        mask = Source('mask', Storage('/tmp/mask.tif', 'tiff'), role='label', axes=('y', 'x'))
        examples = [crop, replace(crop, registration=VolumeRegistration('s', 'a', affine=Affine.identity(),
                        mode='storage', storage_frame='physical', storage_units='micrometer', volume_spacing_um=(2., 2., 3.))),
                    replace(crop, registration=plane, axes=('y', 'x')),
                    replace(crop, storage=Storage(format='stack', locations=('a.tif', 'b.tif'))),
                    replace(crop, registration=SurfaceRegistration(surface, ((1, 0, 0), (0, 1, 0))),
                            companions=(Companion('supervision', mask, same_grid=True),)),
                    replace(crop, role='coordinate', encoding=None, component_order=('z', 'y', 'x'),
                            value_frame=ValueFrame(kind='external', name='instrument', units='um'))]
        provenance['origin']['notes'].append('later')
        self.assertEqual(crop.provenance['origin']['notes'], ('original',))
        bound = replace(crop, storage=Storage('/tmp/pyramid', 'zarr', selection=Selection(level=2)),
                        registration=VolumeRegistration('s', 'a', affine=Affine.scale(4)))
        self.assertEqual(bound.with_selection(level=0).registration.declared_grid, Selection(level=2))
        for source in examples:
            with self.subTest(id=source.id, registration=type(source.registration).__name__):
                self.assertEqual(Source.from_dict(json.loads(json.dumps(source.to_dict()))), source)
                self.assertEqual(pickle.loads(pickle.dumps(source)), source)
                with self.assertRaises(FrozenInstanceError):
                    source.role = 'scalar'

    def test_malformed_declarations(self):
        invalid = [lambda: Selection(level=1, array_path='0'),
                   lambda: Source('bad', axes=('x', 'x')),
                   lambda: Source('bad', provenance={'value': float('nan')}),
                   lambda: VolumeRegistration('s', 'a', affine=Affine.scale(0)),
                   lambda: Plane((0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 0, 1)),
                   lambda: Encoding('integer', ((1, 256),), 'uint8'),
                   lambda: Source.from_dict({'schema_version': 2, 'source': {}})]
        for make in invalid:
            with self.assertRaises(ValueError):
                make()

    def test_discovery_is_stdlib_only_and_offline(self):
        root = Path(__file__).resolve().parents[1]
        code = f"""
import sys
sys.path[:0] = [{str(root / 'src')!r}, {str(root / 'tests')!r}]
from unittest.mock import patch
from test_sources import catalog
with patch('urllib.request.urlopen', side_effect=AssertionError('network request')):
    c = catalog()
    list(c.samples['s'].artifacts)
    c.samples['s'].on_volume('a').labels.ink()
    c.samples['s'].volumes.at_resolution(8)
assert not any(n in sys.modules for n in ('numpy', 'torch', 'PIL', 'tifffile', 'vc_reader._native'))
"""
        subprocess.run([sys.executable, '-S', '-c', code], check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
