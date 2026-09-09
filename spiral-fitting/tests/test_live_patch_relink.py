"""Patches appended to a running session re-offer themselves to resident points.

Covers the linker helper in isolation and the incorporation path end to end:
regular catalog entries that gain attachments are re-derived into cross-patch
views, fibers re-materialize, and nothing changes when no point is claimed.
"""
import math
import unittest
from unittest import mock

import numpy as np
import torch

from config import Config, FitConfig
import fit_spiral
from fit_spiral import FitContext, _UnattachedPclStripList
import point_collection
from point_collection import link_unattached_points_to_patches
from tifxyz import Patch


class _FakePatch:
    """Duck-typed patch: a sphere of radius 0 around ``centre`` with an area."""

    def __init__(self, centre, area):
        self.centre = torch.as_tensor(centre, dtype=torch.float32)
        self.area = area

    def project(self, zyx):
        distance = torch.linalg.norm(zyx - self.centre)
        return torch.tensor([0.5, 0.5]), distance


def _point(point_id, cid, zyx, attached_to=None):
    z, y, x = zyx
    point = {
        'id': point_id, 'collectionId': cid, 'p': [x, y, z],
        'zyx': np.asarray(zyx, dtype=np.float32),
        'winding_annotation': float('nan'),
    }
    if attached_to is not None:
        point['on_patch'] = {'id': attached_to, 'distance': 0.0, 'ij': [0, 0]}
    return point


def _flat_patch(z, y0, x0, size=5, spacing=10.0):
    grid = torch.zeros((size, size, 3), dtype=torch.float32)
    for i in range(size):
        for j in range(size):
            grid[i, j] = torch.tensor([z, y0 + i * spacing, x0 + j * spacing])
    return Patch(grid, torch.ones(3), None, None)


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

    def test_no_new_patches_is_a_no_op(self):
        collection = {'id': 1, 'name': 'r', 'points': {0: _point(0, 1, [0, 0, 0])}}
        self.assertEqual(
            link_unattached_points_to_patches({1: collection}, {}, {}), {})
        self.assertNotIn('on_patch', collection['points'][0])


class LivePatchRelinkTests(unittest.TestCase):
    Z_BEGIN, Z_END = 0, 200

    def setUp(self):
        for target, value in (
                (point_collection, 'can_use_surface_index_backend'),
                (fit_spiral, 'erode_patch_valid_region'),
                (fit_spiral, 'prepare_patch_dt_target_samples')):
            patcher = mock.patch.object(target, value, return_value=(
                False if value == 'can_use_surface_index_backend' else True))
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ('get_rng_state_all', 'set_rng_state_all'):
            patcher = mock.patch.object(torch.cuda, name, return_value=[])
            patcher.start()
            self.addCleanup(patcher.stop)

    def _context(self, regular_catalog, fiber_catalog=None):
        context = FitContext.__new__(FitContext)
        context.config = FitConfig(Config({
            'z_begin': self.Z_BEGIN, 'z_end': self.Z_END,
            'pcl_unattached_pcl_min_point_spacing': 0.0,
        }).as_dict())
        context.verified_patches = {}
        context.verified_patches_list = []
        context.patch_atlas = mock.Mock()
        context._prepare_patch_sampling_cache = mock.Mock()
        context._patch_sampling_probabilities = mock.Mock(
            return_value=np.ones(1, dtype=np.float32))
        context.regular_pcl_catalog = regular_catalog
        context.fiber_catalog = fiber_catalog or {}
        context.next_id = 100
        context.cross_patch_pcls = []
        context.unattached_pcl_strips = _UnattachedPclStripList()
        context.unattached_strip_sampling_groups = []
        context.resolved_links = []
        context.link_components = []
        context.link_distance_tolerance = 2.5
        context.dt_target_cache_manager = mock.Mock()
        context._rebuild_pcl_sampling_strata = mock.Mock()
        context._build_theta_crossing_map = mock.Mock(return_value=[])
        context._trusted_geometry_from_active_inputs = mock.Mock(
            return_value=torch.empty((0, 3)))
        context._vertical_fiber_radial_offset = mock.Mock(return_value=0.0)
        context.run_dt_resume_iteration = None
        return context

    def _add_patch(self, context, patch_id, patch):
        with mock.patch.object(fit_spiral, 'load_tifxyz', return_value=patch):
            return context._incorporate_prevalidated_interactive_inputs(
                [{'kind': 'patch', 'id': patch_id, 'path': f'/inputs/{patch_id}'}],
                {'influence_enabled': False})

    @staticmethod
    def _regular_pcl(cid, zyxs):
        return {
            'id': 3, 'name': 'drawn', 'source_file': '/inputs/drawn.json',
            'sampling_group': '/inputs/drawn.json',
            'metadata': {
                'winding_is_absolute': False, 'input_role': 'legacy',
                'resident_collection_id': cid,
            },
            'points': {i: _point(i, cid, zyx) for i, zyx in enumerate(zyxs)},
        }

    def _resident_strip(self, cid, pcl):
        return {
            'id': cid, 'name': pcl['name'], 'source_file': pcl['source_file'],
            'zyxs': np.stack([p['zyx'] for p in pcl['points'].values()]),
            'windings': np.zeros(len(pcl['points']), dtype=np.float32),
            'radial_offsets': np.zeros(len(pcl['points']), dtype=np.float32),
            'link_points': {}, 'logical_input_kind': None,
            'logical_input_id': None, 'logical_input_revision': None,
        }

    def test_added_patch_claims_resident_regular_points(self):
        cid = 5
        # Three points on the patch surface to come, one far away.
        pcl = self._regular_pcl(cid, [
            [50.0, 10.0, 10.0], [50.0, 20.0, 20.0], [50.0, 30.0, 30.0],
            [50.0, 900.0, 900.0]])
        context = self._context({cid: pcl})
        old_strip = self._resident_strip(cid, pcl)
        context.unattached_pcl_strips.append(old_strip)
        context.unattached_strip_sampling_groups.append(pcl['sampling_group'])

        self._add_patch(context, 'p1', _flat_patch(50.0, 0.0, 0.0))

        catalog_points = context.regular_pcl_catalog[cid]['points']
        self.assertEqual(
            [catalog_points[i].get('on_patch', {}).get('id') for i in range(4)],
            ['p1', 'p1', 'p1', None])
        # The pristine catalog is never trimmed or normalised.
        self.assertTrue(all(math.isnan(p['winding_annotation'])
                            for p in catalog_points.values()))

        self.assertEqual(len(context.cross_patch_pcls), 1)
        cross = context.cross_patch_pcls[0]
        self.assertIsNot(cross, context.regular_pcl_catalog[cid])
        self.assertEqual(cross['metadata']['resident_collection_id'], cid)
        self.assertEqual(list(cross['points_by_patch']), ['p1'])
        self.assertEqual(len(cross['points_by_patch']['p1']), 3)
        self.assertIn('chain', cross)

        # The stale strip is replaced by one derived from the relinked copy.
        self.assertFalse(any(s is old_strip for s in context.unattached_pcl_strips))
        self.assertEqual([s['id'] for s in context.unattached_pcl_strips], [cid])
        self.assertEqual(context.unattached_pcl_strips[0]['zyxs'].shape, (4, 3))
        self.assertEqual(context.unattached_strip_sampling_groups,
                         [pcl['sampling_group']])
        self.assertIsNone(context.unattached_pcl_strips.flat)
        context._rebuild_pcl_sampling_strata.assert_called()
        self.assertIn('p1', context.verified_patches)

    def test_unrelated_patch_leaves_resident_views_untouched(self):
        cid = 5
        pcl = self._regular_pcl(cid, [[50.0, 10.0, 10.0], [50.0, 20.0, 20.0]])
        context = self._context({cid: pcl})
        old_strip = self._resident_strip(cid, pcl)
        context.unattached_pcl_strips.append(old_strip)
        context.unattached_strip_sampling_groups.append(pcl['sampling_group'])

        self._add_patch(context, 'far', _flat_patch(150.0, 700.0, 700.0))

        self.assertIs(context.unattached_pcl_strips[0], old_strip)
        self.assertEqual(context.cross_patch_pcls, [])
        self.assertNotIn('on_patch', context.regular_pcl_catalog[cid]['points'][0])
        context._rebuild_pcl_sampling_strata.assert_not_called()

    def test_second_patch_claims_points_the_first_left_over(self):
        cid = 5
        pcl = self._regular_pcl(cid, [
            [50.0, 10.0, 10.0], [50.0, 20.0, 20.0],
            [50.0, 510.0, 510.0], [50.0, 520.0, 520.0]])
        context = self._context({cid: pcl})
        context.unattached_pcl_strips.append(self._resident_strip(cid, pcl))
        context.unattached_strip_sampling_groups.append(pcl['sampling_group'])

        self._add_patch(context, 'p1', _flat_patch(50.0, 0.0, 0.0))
        first_cross = context.cross_patch_pcls[0]
        self._add_patch(context, 'p2', _flat_patch(50.0, 500.0, 500.0))

        self.assertEqual(len(context.cross_patch_pcls), 1)
        self.assertIsNot(context.cross_patch_pcls[0], first_cross)
        self.assertEqual(
            {pid: len(pts) for pid, pts in
             context.cross_patch_pcls[0]['points_by_patch'].items()},
            {'p1': 2, 'p2': 2})
        # Fully attached now: no unattached strip remains for this pcl.
        self.assertEqual(list(context.unattached_pcl_strips), [])

    def test_added_patch_claims_resident_fiber_points(self):
        fiber = {
            'id': 30, 'name': 'fiber a', 'file_basename': 'a.json',
            'sampling_group': 'fibers', 'source_file': '/inputs/a.json',
            'metadata': {
                'logical_input_id': 'a', 'logical_input_kind': 'fiber',
                'winding_is_absolute': False, 'input_role': 'fiber',
            },
            'points': {
                i: _point(i, 30, zyx) for i, zyx in enumerate(
                    [[50.0, 10.0, 10.0], [50.0, 20.0, 20.0], [50.0, 900.0, 900.0]])
            },
            'kept_orig_indices': np.asarray([0, 1, 2]),
            'control_line_indices': np.asarray([0, 1, 2]),
            'branches': [],
        }
        context = self._context({}, {'a': fiber})
        old_strip = {
            'id': 30, 'logical_input_kind': 'fiber', 'logical_input_id': 'a',
            'zyxs': np.zeros((3, 3), dtype=np.float32),
            'windings': np.zeros(3, dtype=np.float32),
        }
        context.unattached_pcl_strips.append(old_strip)
        context.unattached_strip_sampling_groups.append('fibers')

        self._add_patch(context, 'p1', _flat_patch(50.0, 0.0, 0.0))

        self.assertEqual(len(context.cross_patch_pcls), 1)
        cross = context.cross_patch_pcls[0]
        self.assertEqual(cross['metadata']['logical_input_id'], 'a')
        self.assertEqual(len(cross['points_by_patch']['p1']), 2)
        self.assertFalse(any(s is old_strip for s in context.unattached_pcl_strips))
        self.assertEqual([s['id'] for s in context.unattached_pcl_strips], [30])
        self.assertIs(context.fiber_catalog['a'], fiber)


if __name__ == '__main__':
    unittest.main()
