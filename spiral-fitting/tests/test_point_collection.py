import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import torch

import point_collection
from point_collection import (
    PatchLinkOptions,
    SIDE_BEHIND,
    SIDE_FRONT,
    link_points_to_patches,
    link_unattached_points_to_patches,
    load_point_collection,
    umbilicus_inward_direction,
)
from tifxyz import Patch


class LoadPointCollectionTests(unittest.TestCase):
    def test_loads_valid_collection(self):
        payload = {
            "vc_pointcollections_json_version": "1",
            "collections": {
                "0": {
                    "name": "col-a",
                    "points": {
                        "0": {"p": [1.0, 2.0, 3.0], "wind_a": 1.5},
                        "1": {"p": [4.0, 5.0, 6.0]},
                    },
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "abs_winding.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            loaded = load_point_collection(path)

        self.assertIsNotNone(loaded)
        self.assertEqual(list(loaded.keys()), [0])
        collection = loaded[0]
        self.assertEqual(collection["name"], "col-a")
        self.assertEqual(collection["points"][0]["p"], [1.0, 2.0, 3.0])
        self.assertEqual(collection["points"][0]["winding_annotation"], 1.5)
        # Absent winding annotations become NaN, not None.
        self.assertNotEqual(
            collection["points"][1]["winding_annotation"],
            collection["points"][1]["winding_annotation"],
        )

    def test_missing_file_warns_and_skips(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "drawn_control_points.json")
            output = io.StringIO()
            with redirect_stdout(output):
                loaded = load_point_collection(path)

        self.assertIsNone(loaded)
        message = output.getvalue()
        self.assertIn("not found", message)
        self.assertIn(path, message)
        self.assertNotIn("Error", message)

    def test_malformed_file_still_reports_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "abs_winding.json")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write("{not valid json")
            output = io.StringIO()
            with redirect_stdout(output):
                loaded = load_point_collection(path)

        self.assertIsNone(loaded)
        self.assertIn("Error loading point collection", output.getvalue())

    def test_unsupported_version_still_reports_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "abs_winding.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump({"vc_pointcollections_json_version": "2"}, stream)
            output = io.StringIO()
            with redirect_stdout(output):
                loaded = load_point_collection(path)

        self.assertIsNone(loaded)
        self.assertIn("Unsupported JSON version", output.getvalue())


def _grid_patch(rows, cols, at, scale=1.0):
    """Patch whose vertex (i, j) is ``at(i, j)`` (zyx)."""
    grid = torch.zeros((rows, cols, 3), dtype=torch.float32)
    for i in range(rows):
        for j in range(cols):
            grid[i, j] = torch.tensor(at(i, j), dtype=torch.float32)
    return Patch(grid, torch.tensor([scale, scale]), None, None)


def _z_plane(z, y0, x0, size=5, spacing=10.0, scale=1.0):
    return _grid_patch(size, size, lambda i, j: (z, y0 + i * spacing, x0 + j * spacing), scale)


def _y_plane(y, z0, x0, size=5, spacing=10.0):
    return _grid_patch(size, size, lambda i, j: (z0 + i * spacing, y, x0 + j * spacing))


def _collection(cid, zyxs, name='c', metadata=None):
    return {
        'id': cid, 'name': name, 'metadata': metadata or {},
        'points': {
            i: {'id': i, 'collectionId': cid, 'p': [x, y, z],
                'zyx': np.asarray([z, y, x], dtype=np.float32),
                'winding_annotation': float('nan')}
            for i, (z, y, x) in enumerate(zyxs)
        },
    }


def _attached(collection):
    return [collection['points'][i].get('on_patch', {}).get('id')
            for i in sorted(collection['points'])]


class _BackendCases:
    """Runs each linking test through the brute-force projection path and,
    when vc_spiral is installed, the surface-index path."""

    def backends(self):
        yield 'brute_force', mock.patch.object(
            point_collection, 'can_use_surface_index_backend', return_value=False)
        if point_collection._load_surface_index_backend() is not None:
            yield 'surface_index', mock.patch.object(
                point_collection, 'can_use_surface_index_backend', return_value=True)

    def for_each_backend(self, body):
        for name, patcher in self.backends():
            with self.subTest(backend=name), patcher:
                body()


class WindowLinkingTests(unittest.TestCase, _BackendCases):
    TOL = 2.0

    def _scene(self):
        # Wide patch A (16 unit quads, area 16) 0.5 below three points in a
        # row; small patch B (one quad, scale 0.05 => area 400) 0.2 below the
        # middle point only.
        a = _z_plane(50.5, -20.0, -20.0)
        b = _grid_patch(2, 2, lambda i, j: (50.2, -5.0 + 10.0 * i, -5.0 + 10.0 * j), scale=0.05)
        points = _collection(1, [(50.0, 0.0, -10.0), (50.0, 0.0, 0.0), (50.0, 0.0, 10.0)])
        return {'a': a, 'b': b}, points

    def _link(self, options):
        patches, pcl = self._scene()
        link_points_to_patches(
            patches, {1: pcl}, tolerance=self.TOL, surface_index_tolerance=self.TOL,
            general_hit_policy='largest_area', options=options)
        return pcl

    def test_single_point_keeps_the_largest_area_choice(self):
        def body():
            pcl = self._link(PatchLinkOptions(window_points=1))
            self.assertEqual(_attached(pcl), ['a', 'b', 'a'])
            self.assertAlmostEqual(pcl['points'][1]['on_patch']['distance'], 0.2, places=5)
        self.for_each_backend(body)

    def test_min_hit_gate_drops_the_patch_only_one_point_touches(self):
        def body():
            # B is hit by the middle point alone, so with two hits required it
            # is ineligible and A (hit by the whole window) wins even though B
            # is nearer and larger; the recorded distance is the point's own.
            pcl = self._link(PatchLinkOptions(window_points=3, window_min_points=2))
            self.assertEqual(_attached(pcl), ['a', 'a', 'a'])
            self.assertAlmostEqual(pcl['points'][1]['on_patch']['distance'], 0.5, places=5)
            # A window without a requirement changes nothing.
            pcl = self._link(PatchLinkOptions(window_points=3, window_min_points=1))
            self.assertEqual(_attached(pcl), ['a', 'b', 'a'])
        self.for_each_backend(body)

    def test_gate_is_clipped_at_the_collection_ends(self):
        def body():
            # Requiring the full window of three: the end points only have two
            # window members, so they need both; the middle point needs all
            # three. A satisfies every requirement, B none.
            pcl = self._link(PatchLinkOptions(window_points=3, window_min_points=3))
            self.assertEqual(_attached(pcl), ['a', 'a', 'a'])
        self.for_each_backend(body)

    def test_even_window_rounds_up_and_window_needs_own_hit(self):
        self.assertEqual(PatchLinkOptions(window_points=2).window_half, 1)
        self.assertEqual(PatchLinkOptions(window_points=4).window_half, 2)
        with self.assertRaises(ValueError):
            PatchLinkOptions(window_points=0)
        with self.assertRaises(ValueError):
            PatchLinkOptions(window_points=3, window_min_points=4)
        with self.assertRaises(ValueError):
            PatchLinkOptions(window_points=3, window_min_points=0)

        def body():
            patches, pcl = self._scene()
            # A far point never attaches, however good its neighbours' hits.
            pcl['points'][3] = {
                'id': 3, 'collectionId': 1, 'p': [20.0, 0.0, 500.0],
                'zyx': np.asarray([500.0, 0.0, 20.0], dtype=np.float32),
                'winding_annotation': float('nan')}
            link_points_to_patches(
                patches, {1: pcl}, tolerance=self.TOL, surface_index_tolerance=self.TOL,
                general_hit_policy='largest_area',
                options=PatchLinkOptions(window_points=3, window_min_points=2))
            self.assertEqual(_attached(pcl), ['a', 'a', 'a', None])
        self.for_each_backend(body)

    def test_relink_counts_attached_neighbours_as_window_members(self):
        def body():
            patches, pcl = self._scene()
            # The outer points are already on A; only the middle one is
            # offered to the (new) patches. Its attached neighbours still count
            # toward the two hits required, so A is eligible and B is not.
            for i in (0, 2):
                pcl['points'][i]['on_patch'] = {'id': 'a', 'distance': 0.5, 'ij': [0, 0]}
            gained = link_unattached_points_to_patches(
                {1: pcl}, patches, patches, tolerance=self.TOL,
                surface_index_tolerance=self.TOL, general_hit_policy='largest_area',
                options=PatchLinkOptions(window_points=3, window_min_points=2))
            self.assertEqual(gained, {1: 1})
            self.assertEqual(_attached(pcl), ['a', 'a', 'a'])
        self.for_each_backend(body)


class SideRuleLinkingTests(unittest.TestCase, _BackendCases):
    TOL = 2.0

    def _scene(self):
        # Two y-plane patches straddling points at y = 21: 'front' at y = 20
        # (inward of the points, the umbilicus being at y = 0) and 'behind' at
        # y = 22. Same area, same distance: without a rule patch order wins.
        front = _y_plane(20.0, 40.0, -20.0)
        behind = _y_plane(22.0, 40.0, -20.0)
        pcl = _collection(7, [(60.0, 21.0, 0.0), (61.0, 21.0, 1.0), (62.0, 21.0, 2.0)])
        return {'front': front, 'behind': behind}, pcl

    def _link(self, rule, margin=0.0):
        patches, pcl = self._scene()
        inward = umbilicus_inward_direction(lambda zs: np.zeros((len(zs), 2)))
        options = PatchLinkOptions(
            side_rules={7: rule} if rule else {}, inward_direction=inward,
            side_margin=margin)
        link_points_to_patches(
            patches, {7: pcl}, tolerance=self.TOL, surface_index_tolerance=self.TOL,
            general_hit_policy='largest_area', options=options)
        return pcl

    def test_umbilicus_inward_direction_points_at_the_axis(self):
        inward = umbilicus_inward_direction(lambda zs: np.tile([5.0, 5.0], (len(zs), 1)))
        direction = inward(np.asarray([[0.0, 15.0, 5.0], [3.0, 5.0, 5.0]]))
        np.testing.assert_allclose(direction[0], [0.0, -1.0, 0.0])
        np.testing.assert_allclose(direction[1], [0.0, 0.0, 0.0])

    def test_front_rule_rejects_the_patch_behind(self):
        def body():
            pcl = self._link(SIDE_FRONT)
            self.assertEqual(_attached(pcl), ['front'] * 3)
        self.for_each_backend(body)

    def test_behind_rule_rejects_the_patch_in_front(self):
        def body():
            pcl = self._link(SIDE_BEHIND)
            self.assertEqual(_attached(pcl), ['behind'] * 3)
        self.for_each_backend(body)

    def test_margin_admits_a_patch_just_across_the_point(self):
        def body():
            # Both patches sit 1.0 across; a margin of 1.5 accepts both, so
            # the plain policy (equal area and distance -> patch order) decides.
            pcl = self._link(SIDE_BEHIND, margin=1.5)
            self.assertEqual(_attached(pcl), ['front'] * 3)
        self.for_each_backend(body)

    def test_unruled_collections_are_untouched(self):
        def body():
            pcl = self._link(None)
            self.assertEqual(_attached(pcl), ['front'] * 3)
        self.for_each_backend(body)

    def test_rules_require_a_direction(self):
        with self.assertRaises(ValueError):
            PatchLinkOptions(side_rules={1: SIDE_FRONT})
        with self.assertRaises(ValueError):
            PatchLinkOptions(side_rules={1: 'sideways'}, inward_direction=lambda z: z)


class PatchFilterLinkingTests(unittest.TestCase, _BackendCases):
    """Reviewed placements: allowed / rejected patch ids per collection."""
    TOL = 2.0

    def _link(self, **filters):
        # The window scene: largest-area A under all three points, B under
        # the middle one only.
        patches, pcl = WindowLinkingTests._scene(self)
        other = _collection(2, [(50.0, 0.0, 0.0)])
        link_points_to_patches(
            patches, {1: pcl, 2: other}, tolerance=self.TOL, surface_index_tolerance=self.TOL,
            general_hit_policy='largest_area', options=PatchLinkOptions(**filters))
        return pcl, other

    def test_no_filter_keeps_the_native_choice(self):
        def body():
            pcl, other = self._link()
            self.assertEqual(_attached(pcl), ['a', 'b', 'a'])
            self.assertEqual(_attached(other), ['b'])
        self.for_each_backend(body)

    def test_rejected_patch_is_never_linked(self):
        def body():
            pcl, other = self._link(rejected_patches={1: ['b']})
            self.assertEqual(_attached(pcl), ['a', 'a', 'a'])
            self.assertEqual(_attached(other), ['b'])  # other collections untouched
        self.for_each_backend(body)

    def test_allowed_set_restricts_the_collection(self):
        def body():
            pcl, _ = self._link(allowed_patches={1: ['b']})
            self.assertEqual(_attached(pcl), [None, 'b', None])
            pcl, _ = self._link(allowed_patches={1: []})
            self.assertEqual(_attached(pcl), [None, None, None])
        self.for_each_backend(body)

    def test_filter_applies_before_the_window_gate(self):
        def body():
            # With A rejected only B remains; one window member hits it, so
            # the two-of-three gate drops it too.
            pcl, _ = self._link(rejected_patches={1: ['a']}, window_points=3,
                                window_min_points=2)
            self.assertEqual(_attached(pcl), [None, None, None])
        self.for_each_backend(body)


if __name__ == "__main__":
    unittest.main()
