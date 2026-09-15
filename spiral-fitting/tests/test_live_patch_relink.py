"""The point-to-patch linker; revision integration is in test_revisioned_geometry."""

from geometry_fixtures import _point

import unittest
from unittest import mock
import torch
import point_collection
from point_collection import link_unattached_points_to_patches


class _FakePatch:
    """Duck-typed patch: a sphere of radius 0 around ``centre`` with an area."""

    def __init__(self, centre, area):
        self.centre = torch.as_tensor(centre, dtype=torch.float32)
        self.area = area

    def project(self, zyx):
        distance = torch.linalg.norm(zyx - self.centre)
        return torch.tensor([0.5, 0.5]), distance


class LinkUnattachedPointsTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(
            point_collection, 'can_use_surface_index_backend', return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_only_unattached_points_are_offered_and_only_to_new_patches(self):
        old = _FakePatch([0.0, 0.0, 0.0], area=1.0)
        new = _FakePatch([0.0, 0.0, 50.0], area=1.0)
        collection = {
            'id': 1, 'name': 'regular',
            'points': {
                0: _point(0, 1, [0.0, 0.0, 0.0], attached_to='old'),
                # Sits on the old patch too, but was left unattached: it must
                # not be re-offered to 'old', which is not a new patch.
                1: _point(1, 1, [0.0, 0.0, 1.0]),
                2: _point(2, 1, [0.0, 0.0, 50.0]),
                3: _point(3, 1, [0.0, 0.0, 500.0]),
            },
        }

        gained = link_unattached_points_to_patches(
            {1: collection}, {'old': old, 'new': new}, {'new': new},
            tolerance=2.0, general_hit_policy='largest_area')

        self.assertEqual(gained, {1: 1})
        points = collection['points']
        self.assertEqual(points[0]['on_patch']['id'], 'old')
        self.assertNotIn('on_patch', points[1])
        self.assertEqual(points[2]['on_patch']['id'], 'new')
        self.assertNotIn('on_patch', points[3])

    def test_between_patch_collections_only_attach_to_their_named_pair(self):
        a = _FakePatch([0.0, 0.0, 0.0], area=1.0)
        b = _FakePatch([0.0, 0.0, 10.0], area=1.0)
        other = _FakePatch([0.0, 0.0, 20.0], area=100.0)
        between = {
            'id': 7, 'name': 'between_patches__a__b',
            'points': {
                0: _point(0, 7, [0.0, 0.0, 10.0]),
                1: _point(1, 7, [0.0, 0.0, 20.0]),
            },
        }
        all_patches = {'a': a, 'b': b, 'other': other}

        gained = link_unattached_points_to_patches(
            {7: between}, all_patches, {'b': b, 'other': other}, tolerance=2.0)

        self.assertEqual(gained, {7: 1})
        self.assertEqual(between['points'][0]['on_patch']['id'], 'b')
        self.assertNotIn('on_patch', between['points'][1])
