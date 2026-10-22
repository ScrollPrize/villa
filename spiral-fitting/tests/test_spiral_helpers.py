import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

from config import Config, FitConfig
from spiral_helpers import (
    _DENSE_WEIGHT_KEYS_NEEDING_OUTER_WINDING_IDX,
    _resolve_shell_outer_winding_idx,
    _structurally_disabled_dense_weight_keys,
    load_fiber_point_collection,
    resolve_outer_winding_idx_and_notes,
)
from fit_spiral import (
    FitContext,
    _UnattachedPclStripList,
    get_dt_loss_eligibility,
    get_unattached_pcl_dt_start,
    get_run_dt_resume_iteration,
    materialize_fiber_fit_inputs,
)
from tifxyz import load_tifxyz


class RunDtLossScheduleTests(unittest.TestCase):
    def test_ten_thousand_iterations_at_25_percent(self):
        resume = get_run_dt_resume_iteration(400, 10_000, 0.25)
        self.assertEqual(resume, 7_900)
        self.assertEqual(resume - 400, 7_500)
        self.assertEqual(10_400 - resume, 2_500)

    def test_small_runs_use_a_ceiling_sized_eligible_suffix(self):
        for iterations, fraction, eligible in (
                (1, 0.25, 1), (2, 0.25, 1), (3, 0.25, 1),
                (5, 0.25, 2), (5, 0.0, 0), (5, 1.0, 5)):
            resume = get_run_dt_resume_iteration(10, iterations, fraction)
            self.assertEqual(10 + iterations - resume, eligible)

    def test_schedule_composes_with_strict_family_start_thresholds(self):
        cfg = Config({
            'loss_start_patch_dt': 10,
            'loss_start_track_dt': 20,
            'loss_start_unattached_pcl_dt': 25,
        }).as_dict()
        resume = 15
        self.assertEqual(get_dt_loss_eligibility(cfg, 14, resume), {
            'verified_patch': False,
            'track': False,
            'unattached_pcl': False,
        })
        # Existing loss starts remain strict `iteration > start` checks.
        at_starts = get_dt_loss_eligibility(cfg, 20, resume)
        self.assertTrue(at_starts['verified_patch'])
        self.assertFalse(at_starts['unattached_pcl'])
        self.assertFalse(at_starts['track'])
        self.assertFalse(get_dt_loss_eligibility(cfg, 25, resume)['unattached_pcl'])
        self.assertTrue(get_dt_loss_eligibility(cfg, 26, resume)['unattached_pcl'])
        after_all = get_dt_loss_eligibility(cfg, 31, resume)
        self.assertTrue(all(after_all.values()))

    def test_unattached_pcl_dt_start_follows_patch_start_when_unset(self):
        cfg = Config({'loss_start_patch_dt': 10}).as_dict()
        self.assertIsNone(cfg['loss_start_unattached_pcl_dt'])
        self.assertEqual(get_unattached_pcl_dt_start(cfg), 10)
        self.assertFalse(get_dt_loss_eligibility(cfg, 10)['unattached_pcl'])
        self.assertTrue(get_dt_loss_eligibility(cfg, 11)['unattached_pcl'])
        decoupled = Config({'loss_start_patch_dt': 10,
                            'loss_start_unattached_pcl_dt': 0}).as_dict()
        self.assertEqual(get_unattached_pcl_dt_start(decoupled), 0)
        self.assertTrue(get_dt_loss_eligibility(decoupled, 1)['unattached_pcl'])
        self.assertFalse(get_dt_loss_eligibility(decoupled, 1)['verified_patch'])


class FiberPointCollectionTests(unittest.TestCase):
    def _write_fiber(self, directory, data):
        path = Path(directory) / "fiber.json"
        path.write_text(json.dumps(data))
        return path

    def test_loads_line_points_trimmed_to_the_control_point_span(self):
        # The tracer extends line_points past the first and last control point;
        # those dangling ends must not become constraints.
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_fiber(temporary, {
                "control_points": [[4, 8, 12], [20, 24, 28]],
                "line_points": [
                    [0, 0, 0],            # dangling start: dropped
                    [4, 8, 12],           # first control point
                    [12, 16, 20],         # dense in-between point: kept
                    [20, 24, 28],         # last control point
                    [100, 100, 100],      # dangling end: dropped
                ],
            })

            collection = load_fiber_point_collection(
                path, collection_id=7, min_point_spacing=0)

            points = [point["p"] for point in collection["points"].values()]
            np.testing.assert_array_equal(
                points, [[1, 2, 3], [3, 4, 5], [5, 6, 7]])
            np.testing.assert_array_equal(
                collection["control_line_indices"], [0, 2])

    def test_falls_back_to_control_points_without_a_dense_polyline(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_fiber(temporary, {
                "control_points": [[4, 8, 12], [20, 24, 28]],
                "line_points": [[400, 800, 1200]],
            })

            collection = load_fiber_point_collection(
                path, collection_id=7, min_point_spacing=0)

            points = [point["p"] for point in collection["points"].values()]
            np.testing.assert_array_equal(points, [[1, 2, 3], [5, 6, 7]])

    def test_ignores_adjacent_links_unsupported_by_spiral_fitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_fiber(temporary, {
                "control_points": [[4, 8, 12], [20, 24, 28]],
                "line_points": [[4, 8, 12], [20, 24, 28]],
                "adjacent_branches": [{
                    "branch_file": "peer.json",
                    "control_point_index": 0,
                    "branch_control_point_index": 1,
                }],
            })

            collection = load_fiber_point_collection(
                path, collection_id=7, min_point_spacing=0)

            self.assertEqual(collection["branches"], [])

    def _tagged_fiber_docs(self, tag):
        # A v4 fiber along x (line points every 4 units, control points at
        # 0, 12, 28, 36) whose span 12 -> 28 carries `tag`, plus a peer fiber
        # linked reciprocally to its interior control point at 28 (which
        # greedy decimation at spacing 16 would otherwise discard).
        from test_bracket_fiber_links import _segment
        common = {
            "type": "vc3d_fiber", "version": 4, "generation": 3,
            "optimization_mode": "native_fiber_trace3d",
            "hv_classification": {"automatic_tag": "H", "automatic_certainty": 1.0, "manual_tag": ""},
            "sequence": 1, "started_at": "20260101T000000000", "tags": [], "username": "test",
        }
        fiber = dict(common, filename="fiber.json", control_points=[
            {"position": [0, 0, 0], "segment_to_next": _segment()},
            {"position": [12, 0, 0], "segment_to_next": dict(_segment(), tags=[tag] if tag else [])},
            {"position": [28, 0, 0], "segment_to_next": _segment()},
            {"position": [36, 0, 0]},
        ], line_points=[[x, 0, 0] for x in range(0, 40, 4)], branches=[
            {"branch_file": "peer.json", "control_point_index": 2, "branch_control_point_index": 0}])
        peer = dict(common, filename="peer.json", control_points=[
            {"position": [28, 0, 0], "segment_to_next": _segment()},
            {"position": [28, 10, 0]},
        ], line_points=[[28, 0, 0], [28, 5, 0], [28, 10, 0]], branches=[
            {"branch_file": "fiber.json", "control_point_index": 0, "branch_control_point_index": 2}])
        return fiber, peer

    def _write_tagged_pair(self, directory, tag, **load):
        fiber, peer = self._tagged_fiber_docs(tag)
        paths = {}
        for name, doc in (("fiber.json", fiber), ("peer.json", peer)):
            paths[name] = Path(directory) / name
            paths[name].write_text(json.dumps(doc))
        return (load_fiber_point_collection(paths["fiber.json"], collection_id=7, coordinate_scale=1.0, **load),
                load_fiber_point_collection(paths["peer.json"], collection_id=8, coordinate_scale=1.0, **load))

    def test_flags_the_interior_line_points_of_gap_and_damaged_spans(self):
        # Format v4 span tags: a `gap` or `damaged` span's line points are the
        # tracer's bridge over a hole, not sheet. They stay in the polyline
        # (the chain unwrap needs the geometry) but are flagged fit_excluded;
        # the span's two control points are ordinary points.
        from spiral_helpers import resolve_fiber_links
        for tag in ("gap", "damaged", None):
            with self.subTest(tag=tag), tempfile.TemporaryDirectory() as temporary:
                fiber, peer = self._write_tagged_pair(temporary, tag, min_point_spacing=0)

                points = [point["p"][0] for point in fiber["points"].values()]
                self.assertEqual(points, list(range(0, 40, 4)))
                excluded = [point["fit_excluded"] for point in fiber["points"].values()]
                expected = [x in (16, 20, 24) for x in points] if tag else [False] * 10
                self.assertEqual(excluded, expected)
                np.testing.assert_array_equal(fiber["control_line_indices"], [0, 3, 7, 9])
                self.assertEqual(fiber["metadata"]["fiber_excluded_spans"], 1 if tag else 0)
                self.assertEqual(fiber["metadata"]["fiber_excluded_span_points"], 3 if tag else 0)
                self.assertFalse(any(point["fit_excluded"] for point in peer["points"].values()))

    def test_flags_survive_coarse_decimation_and_links_resolve_exactly(self):
        # Greedy decimation at spacing 16 keeps x = 0 and 16 (inside the
        # span, so flagged); the link endpoint 28 is only 12 further and
        # survives by force, after which 36 is too close. The reciprocal peer
        # link resolves to that exact point.
        from spiral_helpers import resolve_fiber_links
        with tempfile.TemporaryDirectory() as temporary:
            fiber, peer = self._write_tagged_pair(temporary, "gap", min_point_spacing=16)

            points = [point["p"][0] for point in fiber["points"].values()]
            self.assertEqual(points, [0, 16, 28])
            np.testing.assert_array_equal(fiber["kept_orig_indices"], [0, 4, 7])
            self.assertEqual([point["fit_excluded"] for point in fiber["points"].values()],
                             [False, True, False])
            links = resolve_fiber_links({7: fiber, 8: peer})
            self.assertEqual(len(links), 1)
            endpoint = {links[0]["a_coll"]: links[0]["a_point"], links[0]["b_coll"]: links[0]["b_point"]}
            self.assertEqual(endpoint, {7: 2, 8: 0})
            self.assertEqual(fiber["points"][2]["p"], [28, 0, 0])
            self.assertFalse(fiber["points"][2]["fit_excluded"])

    def test_materialized_strips_carry_the_fit_mask_through_decimation(self):
        # The catalog's fit_excluded flags become the strip's fit_mask, in the
        # strip's decimated point order, on every (deterministic) rebuild.
        with tempfile.TemporaryDirectory() as temporary:
            fiber, peer = self._write_tagged_pair(temporary, "damaged", min_point_spacing=0)
            catalog = {"fiber": fiber, "peer": peer}
            for spacing in (0, 16):
                with self.subTest(spacing=spacing):
                    views = [materialize_fiber_fit_inputs(
                        catalog, {}, z_begin=-10, z_end=10, z_margin=0,
                        min_point_spacing=spacing) for _ in range(2)]
                    strips = {strip["id"]: strip for strip in views[0][1]}
                    strip = strips[7]
                    xs = strip["zyxs"][:, 2].tolist()
                    if spacing == 0:
                        self.assertEqual(xs, list(range(0, 40, 4)))
                        self.assertEqual(strip["link_points"], {7: 7})
                    else:
                        # 28 survives only as the force-kept link endpoint
                        # (greedy spacing would pick 32); 36 as the last point.
                        self.assertEqual(xs, [0, 16, 28, 36])
                        self.assertEqual(strip["link_points"], {7: 2})
                    np.testing.assert_array_equal(
                        strip["fit_mask"], [x not in (16, 20, 24) for x in xs])
                    self.assertTrue(strips[8]["fit_mask"].all())
                    again = {strip["id"]: strip for strip in views[1][1]}[7]
                    np.testing.assert_array_equal(again["fit_mask"], strip["fit_mask"])
                    self.assertEqual(len(views[0][3]), 1)

    def test_excluded_points_do_not_make_an_attached_fiber_a_strip(self):
        # Every eligible point is attached to a patch: without the tag the
        # fiber yields no unattached strip, and an excluded point inside it
        # must not change that (it would hand the attached endpoints strip
        # constraints they did not have).
        for excluded in (False, True):
            with self.subTest(excluded=excluded):
                member = self._linked_member(10, "a", [], 0)
                member["points"][2] = {
                    "id": 2, "collectionId": 10, "p": [2.0, 0.0, 0.0],
                    "zyx": np.asarray([0.0, 0.0, 2.0], dtype=np.float32),
                    "winding_annotation": float("nan"), "fit_excluded": True}
                member["points"] = {0: member["points"][0], 1: member["points"][2], 2: member["points"][1]}
                for point_id, point in member["points"].items():
                    point["id"] = point_id
                if not excluded:
                    member["points"][1]["fit_excluded"] = False
                    member["points"][1]["on_patch"] = {"id": "patch-a"}
                cross, strips, _, _, _ = self._materialize({"a": member})
                self.assertEqual(len(cross), 1)
                self.assertEqual(strips, [])

    def test_outer_winding_evidence_skips_excluded_points(self):
        # An excluded excursion to winding 20 on a strip whose eligible
        # points sit at winding 10 must not widen the winding range or the
        # strip's extent (which seed the shell outer-winding inference).
        from spiral_helpers import compute_winding_range_and_input_extents
        from fit_spiral import _build_strip_flat_bundle

        class Identity:
            def __call__(self, zyxs):
                return zyxs

        dr = torch.tensor(12.0)
        radii = np.array([10.0, 10.0, 10.0, 20.0]) * 12.0
        zyxs = np.stack([np.zeros(4), np.zeros(4), radii], axis=1).astype(np.float32)
        cfg = Config().as_dict()
        results = {}
        for name, fit_mask in (("plain", None), ("masked", np.array([True, True, True, False]))):
            strip = {"zyxs": zyxs, "windings": np.zeros(4, np.float32), "fit_mask": fit_mask}
            flat = _build_strip_flat_bundle([(zyxs, strip["windings"], None, None, fit_mask)], torch.device("cpu"))
            results[name] = compute_winding_range_and_input_extents(
                Identity(), dr, [], [strip], cfg, -1, 1, lambda _s, _d: flat)
        self.assertEqual(results["plain"][2][0][1], 20)
        self.assertEqual(results["masked"][2][0][1], 10)
        self.assertEqual(results["plain"][0][1] - results["masked"][0][1], 10)

    def test_hv_fallback_ignores_an_excluded_excursion(self):
        # Eligible points on a vertical line; an excluded excursion dominates
        # the path length. The geometric fallback (no usable VC3D tag) must
        # read the eligible geometry: V, hence the vertical radial offset on
        # the strip and the front-side link rule for the collection.
        from spiral_helpers import classify_fiber_hv, fiber_collection_hv_tag
        zyxs = np.asarray([[0, 0, 100], [10, 0, 100], [15, 0, 1000], [20, 0, 100], [30, 0, 100]],
                          dtype=np.float32)
        mask = np.asarray([True, True, False, True, True])
        kwargs = dict(min_z_fraction=0.8, min_auto_certainty=0.5)
        self.assertEqual(classify_fiber_hv({}, zyxs, **kwargs), "H")
        self.assertEqual(classify_fiber_hv({}, zyxs, fit_mask=mask, **kwargs), "V")
        self.assertEqual(classify_fiber_hv({"manual_tag": "H"}, zyxs, fit_mask=mask, **kwargs), "H")
        member = self._linked_member(10, "a", [], 0, attached=False)
        member["points"] = {
            k: {"id": k, "collectionId": 10, "p": [float(z[2]), 0.0, float(z[0])],
                "zyx": np.asarray(z, dtype=np.float32), "winding_annotation": float("nan"),
                "fit_excluded": not bool(m)}
            for k, (z, m) in enumerate(zip(zyxs, mask))}
        member["metadata"]["hv_classification"] = {}
        self.assertEqual(fiber_collection_hv_tag(member, **kwargs), "V")
        _, strips, _, _, _ = materialize_fiber_fit_inputs(
            {"a": member}, {}, z_begin=-10, z_end=100, z_margin=0,
            min_point_spacing=0, vertical_radial_offset=4.0)
        self.assertEqual(strips[0]["hv_tag"], "V")
        np.testing.assert_array_equal(strips[0]["radial_offsets"], [4.0] * 5)
        np.testing.assert_array_equal(strips[0]["fit_mask"], mask)

    def test_overlapping_span_projections_never_exclude_a_control_point(self):
        # Control points whose nearest line points are out of order
        # ([0, 7, 3, 9]) with both the 0->28 and the 12->36 span tagged: the
        # interiors overlap, but the control points at 12 and 28 stay
        # ordinary points.
        from test_bracket_fiber_links import _segment
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_fiber(temporary, {
                "type": "vc3d_fiber", "version": 4, "generation": 3,
                "optimization_mode": "native_fiber_trace3d",
                "hv_classification": {"automatic_tag": "H", "automatic_certainty": 1.0, "manual_tag": ""},
                "sequence": 1, "started_at": "20260101T000000000", "tags": [], "username": "test",
                "filename": "fiber.json",
                "control_points": [
                    {"position": [0, 0, 0], "segment_to_next": dict(_segment(), tags=["gap"])},
                    {"position": [28, 0, 0], "segment_to_next": _segment()},
                    {"position": [12, 0, 0], "segment_to_next": dict(_segment(), tags=["damaged"])},
                    {"position": [36, 0, 0]},
                ],
                "line_points": [[x, 0, 0] for x in range(0, 40, 4)],
            })
            collection = load_fiber_point_collection(
                path, collection_id=7, coordinate_scale=1.0, min_point_spacing=0)
            np.testing.assert_array_equal(collection["control_line_indices"], [0, 7, 3, 9])
            excluded = [x for point in collection["points"].values()
                        for x in [point["p"][0]] if point["fit_excluded"]]
            self.assertEqual(excluded, [4, 8, 16, 20, 24, 32])
            self.assertEqual(collection["metadata"]["fiber_excluded_spans"], 2)
            self.assertEqual(collection["metadata"]["fiber_excluded_span_points"], 6)

    def test_segmented_median_skips_excluded_points(self):
        from spiral_helpers import _segmented_median_per_strip
        lengths = torch.tensor([4, 3, 2])
        starts = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
        values = torch.tensor([5.0, 9.0, 1.0, 9.5,   # kept: 5, 1 -> median 1
                               2.0, 8.0, 3.0,        # kept: 2, 8, 3 -> median 3
                               7.0, 6.0])            # nothing kept
        fit_mask = torch.tensor([True, False, True, False,
                                 True, True, True,
                                 False, False])
        ctx = {'normalised_radii': values, 'strip_id': torch.repeat_interleave(torch.arange(3), lengths),
               'starts': starts, 'lengths': lengths, 'S': 3, 'device': torch.device('cpu'),
               'fit_mask': fit_mask}
        medians = _segmented_median_per_strip(ctx)
        self.assertEqual(medians[:2].tolist(), [1.0, 3.0])
        ctx.pop('fit_mask')
        self.assertEqual(_segmented_median_per_strip(ctx).tolist(), [5.0, 3.0, 6.0])

    def test_segmented_median_skips_excluded_points_in_the_two_pass_branch(self):
        # FIT_SPIRAL_MAX_PRECISION_FLOAT=32 selects the value-sort-then-strip-
        # sort branch (MPS); it must agree with the composite-key branch.
        import devices
        from spiral_helpers import _segmented_median_per_strip
        lengths = torch.tensor([4, 3, 2])
        values = torch.tensor([5.0, 9.0, 1.0, 9.5, 2.0, 8.0, 3.0, 7.0, 6.0])
        fit_mask = torch.tensor([True, False, True, False, True, True, True, False, False])
        ctx = {'normalised_radii': values, 'strip_id': torch.repeat_interleave(torch.arange(3), lengths),
               'starts': torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)]),
               'lengths': lengths, 'S': 3, 'device': torch.device('cpu'), 'fit_mask': fit_mask}
        with mock.patch.dict(os.environ, {'FIT_SPIRAL_MAX_PRECISION_FLOAT': '32'}):
            devices.max_precision_bits.cache_clear()
            try:
                self.assertEqual(devices.float_hi('cpu'), torch.float32)
                self.assertEqual(_segmented_median_per_strip(ctx)[:2].tolist(), [1.0, 3.0])
            finally:
                devices.max_precision_bits.cache_clear()
        devices.max_precision_bits.cache_clear()

    def test_segmented_median_tolerates_empty_strips(self):
        # Intervening and trailing zero-length strips (the satisfaction API
        # supports empty collections) must not index past the values.
        from spiral_helpers import _segmented_median_per_strip
        lengths = torch.tensor([3, 0, 2, 0])
        starts = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
        values = torch.tensor([2.0, 9.0, 4.0, 7.0, 1.0])
        ctx = {'normalised_radii': values, 'strip_id': torch.repeat_interleave(torch.arange(4), lengths),
               'starts': starts, 'lengths': lengths, 'S': 4, 'device': torch.device('cpu')}
        medians = _segmented_median_per_strip(ctx)
        self.assertEqual(medians[0].item(), 4.0)
        self.assertEqual(medians[2].item(), 1.0)
        ctx['fit_mask'] = torch.tensor([True, False, True, False, False])
        medians = _segmented_median_per_strip(ctx)
        self.assertEqual(medians[0].item(), 2.0)

    def test_trusted_geometry_leaves_excluded_strip_points_out(self):
        from fit_spiral import _strip_trusted_point_mask
        context = FitContext.__new__(FitContext)
        context.config = FitConfig(Config({"z_begin": 0, "z_end": 10}).as_dict())
        context.verified_patches_list = []
        zyxs = np.asarray([[1, 0, 0], [2, 0, 0], [3, 0, 0], [20, 0, 0]], dtype=np.float32)
        plain = {'zyxs': zyxs}
        masked = {'zyxs': zyxs, 'fit_mask': np.asarray([True, False, True, True])}
        np.testing.assert_array_equal(
            _strip_trusted_point_mask(plain, 0, 10), [True, True, True, False])
        np.testing.assert_array_equal(
            _strip_trusted_point_mask(masked, 0, 10), [True, False, True, False])
        context.unattached_pcl_strips = _UnattachedPclStripList([masked, plain])
        trusted = context._trusted_geometry_from_active_inputs()
        self.assertEqual(trusted[:, 0].tolist(), [1.0, 3.0, 1.0, 2.0, 3.0])

    def test_declared_coordinate_domain_overrides_legacy_scale(self):
        for shape, expected_scale in (([101, 201, 301], 1),
                                      ([100, 200, 300], 1),
                                      ([202, 402, 602], .5),
                                      ([403, 803, 1203], .25),
                                      ([405, 805, 1205], .25),
                                      ([51, 101, 151], 2)):
            with self.subTest(shape=shape), tempfile.TemporaryDirectory() as directory:
                path = self._write_fiber(directory, {
                    "control_points": [[4, 8, 12], [20, 24, 28]],
                    "line_points": [],
                    "coordinate_base_shape_zyx": shape,
                })
                pcl = load_fiber_point_collection(
                    path, 7, min_point_spacing=0,
                    base_shape_zyx=[101, 201, 301])
                np.testing.assert_array_equal(
                    [point['p'] for point in pcl['points'].values()],
                    np.array([[4, 8, 12], [20, 24, 28]]) * expected_scale)

    def test_declared_coordinate_domain_requires_compatible_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_fiber(directory, {
                "control_points": [[4, 8, 12], [20, 24, 28]],
                "coordinate_base_shape_zyx": [100, 200, 300],
            })
            for shape in (None, [100, 100, 300], [0, 200, 300]):
                with self.subTest(shape=shape), self.assertRaises(ValueError):
                    load_fiber_point_collection(path, 7, base_shape_zyx=shape)

    def test_skips_fibers_without_control_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_fiber(temporary, {
                "line_points": [[4, 8, 12]],
            })

            collection = load_fiber_point_collection(path, collection_id=7)

            self.assertIsNone(collection)

    @staticmethod
    def _linked_member(collection_id, logical_id, targets, x_offset,
                       attached=True):
        points = {
            point_id: {
                "id": point_id,
                "collectionId": collection_id,
                "p": [x_offset + point_id, 0.0, 0.0],
                "zyx": np.asarray([0.0, 0.0, x_offset + point_id],
                              dtype=np.float32),
                "winding_annotation": float("nan"),
                **({"on_patch": {"id": f"patch-{logical_id}"}}
                   if attached else {}),
            }
            for point_id in range(2)
        }
        return {
            "id": collection_id,
            "file_basename": f"{logical_id}.json",
            "sampling_group": "fibers",
            "metadata": {
                "logical_input_id": logical_id,
                "logical_input_kind": "fiber",
                "winding_is_absolute": False,
            },
            "points": points,
            "kept_orig_indices": np.asarray([0, 1]),
            "control_line_indices": np.asarray([0, 1]),
            "branches": [
                {
                    "local_index": local_index,
                    "branch_file": f"{target_id}.json",
                    "branch_index": target_index,
                    "pending": False,
                }
                for local_index, target_id, target_index in targets
            ],
        }

    def _materialize(self, catalog, spacing=0):
        patches = {
            f"patch-{logical_id}": object() for logical_id in catalog
        }
        return materialize_fiber_fit_inputs(
            catalog, patches, z_begin=-10, z_end=10, z_margin=0,
            min_point_spacing=spacing)

    def test_catalog_materializes_branching_and_loop_closing_graphs(self):
        catalog = {
            "a": self._linked_member(
                10, "a", [(0, "b", 0), (1, "c", 0)], 0),
            "b": self._linked_member(
                11, "b", [(0, "a", 0), (1, "c", 1)], 10),
            "c": self._linked_member(
                12, "c", [(0, "a", 1), (1, "b", 1)], 20),
        }

        cross, strips, groups, links, components = self._materialize(catalog)

        self.assertEqual(len(cross), 1)
        self.assertEqual(cross[0]["metadata"]["logical_input_ids"],
                         ["a", "b", "c"])
        self.assertEqual(len(links), 3)
        self.assertEqual(len(components), 1)
        self.assertEqual(len(strips), 3)
        self.assertEqual(groups, ["fibers"] * 3)
        self.assertEqual(len(cross[0]["chain"].extra_edges), 1)

    def test_revision_replaces_one_catalog_value_and_rebuilds_all_views(self):
        a = self._linked_member(10, "a", [(0, "b", 0)], 0)
        b = self._linked_member(
            11, "b", [(0, "a", 0), (1, "c", 0)], 10)
        c = self._linked_member(12, "c", [(0, "b", 1)], 20)
        catalog = {"a": a, "b": b, "c": c}
        self._materialize(catalog)

        revised_a = self._linked_member(10, "a", [], 100)
        candidate = dict(catalog)
        candidate["a"] = revised_a
        cross, strips, _, links, _ = self._materialize(candidate, spacing=1000)

        self.assertEqual(revised_a["id"], a["id"])
        self.assertEqual(list(candidate), ["a", "b", "c"])
        self.assertEqual(len(cross), 1)
        self.assertEqual(cross[0]["metadata"]["logical_input_ids"],
                         ["a", "b", "c"])
        merged_points = list(cross[0]["points"].values())
        self.assertTrue(any(point is revised_a["points"][0]
                            for point in merged_points))
        self.assertFalse(any(point is a["points"][0]
                             for point in merged_points))
        self.assertTrue(any(point is b["points"][0]
                            for point in merged_points))
        self.assertTrue(any(point is c["points"][0]
                            for point in merged_points))
        # B's authoritative reciprocal record keeps A-B active, and junction
        # points survive even though the strip spacing exceeds its full length.
        self.assertEqual(len(links), 2)
        strips_by_id = {strip["id"]: strip for strip in strips}
        self.assertIn(0, strips_by_id[10]["link_points"])
        self.assertIn(0, strips_by_id[11]["link_points"])
        self.assertEqual(sum(
            strip["logical_input_id"] == "a" for strip in strips), 1)

    def test_batched_reciprocal_removal_separates_component(self):
        catalog = {
            "a": self._linked_member(10, "a", [], 0),
            "b": self._linked_member(11, "b", [], 10),
        }

        cross, strips, _, links, components = self._materialize(catalog)

        self.assertEqual(links, [])
        self.assertEqual(components, [])
        self.assertEqual(len(cross), 2)
        self.assertEqual(strips, [])

    def test_new_fiber_links_to_resident_and_materialization_is_deterministic(self):
        resident = self._linked_member(10, "a", [], 0)
        added = self._linked_member(11, "b", [(0, "a", 0)], 10)
        catalog = {"a": resident, "b": added}

        first = self._materialize(catalog)
        second = self._materialize(catalog)

        self.assertEqual(len(first[3]), 1)
        self.assertEqual(first[3], second[3])
        self.assertEqual(first[4], second[4])
        self.assertEqual(
            [pcl["metadata"] for pcl in first[0]],
            [pcl["metadata"] for pcl in second[0]])
        for left, right in zip(first[1], second[1]):
            self.assertEqual(left["id"], right["id"])
            self.assertEqual(left["link_points"], right["link_points"])
            np.testing.assert_array_equal(left["zyxs"], right["zyxs"])

    def _editable_context(self, cross_patch, strips, groups):
        context = FitContext.__new__(FitContext)
        context.config = FitConfig(Config({
            "z_begin": 0, "z_end": 200,
        }).as_dict())
        context.fiber_catalog = {}
        context.next_id = 30
        context.verified_patches = {}
        context.verified_patches_list = []
        context.cross_patch_pcls = list(cross_patch)
        context.unattached_pcl_strips = _UnattachedPclStripList(list(strips))
        context.unattached_strip_sampling_groups = list(groups)
        context.resolved_links = []
        context.link_components = []
        context.link_distance_tolerance = 2.5
        context.dt_target_cache_manager = mock.Mock()
        context._rebuild_pcl_sampling_strata = mock.Mock()
        context._build_theta_crossing_map = mock.Mock(return_value=[])
        context._trusted_geometry_from_active_inputs = mock.Mock(
            return_value=torch.empty((0, 3)))
        context.run_dt_resume_iteration = None
        return context


class TifxyzMetadataTests(unittest.TestCase):
    def _write_patch(self, root, metadata):
        (root / "meta.json").write_text(json.dumps(metadata))
        values = np.ones((2, 2), dtype=np.float32)
        for coordinate in "zyx":
            Image.fromarray(values).save(root / f"{coordinate}.tif")

    def test_patch_can_override_configured_erosion_with_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_patch(root, {
                "format": "tifxyz",
                "scale": [1.0, 1.0],
                "spiral_patch_erode_cells": 0,
            })

            patch = load_tifxyz(root)

            self.assertEqual(patch.erosion_cells(7), 0)

    def test_ordinary_patch_uses_configured_erosion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_patch(root, {"format": "tifxyz", "scale": [1.0, 1.0]})

            patch = load_tifxyz(root)

            self.assertEqual(patch.erosion_cells(7), 7)

    def test_z_prefilter_skips_x_and_y_decode_for_out_of_roi_patch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "meta.json").write_text(json.dumps({
                "format": "tifxyz", "scale": [1.0, 1.0],
            }))
            Image.fromarray(np.full((3, 4), 20, dtype=np.float32)).save(
                root / "z.tif")
            # x.tif and y.tif deliberately do not exist: returning None proves
            # the expensive coordinate planes were never opened.
            self.assertIsNone(load_tifxyz(root, z_range=(100, 200)))

    def test_z_prefilter_respects_mask_and_half_open_interval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "meta.json").write_text(json.dumps({
                "format": "tifxyz", "scale": [1.0, 1.0],
            }))
            z = np.full((2, 2), 100, dtype=np.float32)
            z[0, 0] = 150
            Image.fromarray(z).save(root / "z.tif")
            mask = np.ones((2, 2), dtype=np.uint8)
            mask[0, 0] = 0
            Image.fromarray(mask).save(root / "mask.tif")
            self.assertIsNone(load_tifxyz(root, z_range=(150, 151)))
            # z == end is excluded.
            self.assertIsNone(load_tifxyz(root, z_range=(99, 100)))


class ShellOuterWindingIdxResolutionTests(unittest.TestCase):
    # The index used to be resolved only inside the shell-loss branch, which
    # silently zeroed the dense lasagna losses, the symmetric Dirichlet
    # regulariser and the phase bundle on shell-less runs (#1220).

    def _weights(self, value):
        return {key: value
                for key in _DENSE_WEIGHT_KEYS_NEEDING_OUTER_WINDING_IDX}

    def test_configured_index_is_coerced_and_returned(self):
        cfg = {'shell_outer_winding_idx': 130}
        self.assertEqual(_resolve_shell_outer_winding_idx(cfg), 130)

    def test_float_config_is_coerced_to_int(self):
        cfg = {'shell_outer_winding_idx': 130.0}
        resolved = _resolve_shell_outer_winding_idx(cfg)
        self.assertEqual(resolved, 130)
        self.assertIsInstance(resolved, int)

    def test_unset_index_stays_none(self):
        cfg = {'shell_outer_winding_idx': None}
        self.assertIsNone(_resolve_shell_outer_winding_idx(cfg))

    def test_no_weight_is_reported_when_the_index_resolves(self):
        self.assertEqual(
            _structurally_disabled_dense_weight_keys(self._weights(1.0), 130),
            ())

    def test_every_dense_weight_is_reported_when_unresolved(self):
        # Locks the real blast radius: every sampler bounded by the index
        # must be listed here, so adding one without registering it fails.
        self.assertEqual(
            _structurally_disabled_dense_weight_keys(self._weights(1.0), None),
            (
                'loss_weight_dense_normals',
                'loss_weight_dense_spacing',
                'loss_weight_dense_spacing_density',
                'loss_weight_min_spacing',
                'loss_weight_sym_dirichlet',
            ))

    def test_zero_weights_are_not_reported(self):
        # A deliberately loss-free run must not warn (a shell-less fit with
        # every dense weight at zero is a valid use-case, not a defect).
        self.assertEqual(
            _structurally_disabled_dense_weight_keys(self._weights(0.0), None),
            ())

    def test_only_the_nonzero_weights_are_reported(self):
        cfg = self._weights(0.0)
        cfg['loss_weight_min_spacing'] = 1.0
        cfg['loss_weight_sym_dirichlet'] = 2.0
        self.assertEqual(
            _structurally_disabled_dense_weight_keys(cfg, None),
            ('loss_weight_min_spacing', 'loss_weight_sym_dirichlet'))

    def test_degenerate_indices_are_rejected_with_a_clear_error(self):
        # sample_spiral_surface_frame draws windings from [1, idx); 0 and 1
        # used to crash multinomial at the first step with an opaque error.
        for bad in (0, 1, -3, 'x', '130.5'):
            with self.assertRaises(ValueError):
                _resolve_shell_outer_winding_idx(
                    {'shell_outer_winding_idx': bad})


class ResolveOuterWindingIdxWiringTests(unittest.TestCase):
    # These lock the wiring decision that used to live inline in
    # fit_spiral.main (the actual #1220 bug): without shell losses the
    # configured index must survive, inference must not run, and the
    # gap-expander control must fire shell or not.

    def _cfg(self, idx, gap=200):
        return {'shell_outer_winding_idx': idx,
                'model_gap_expander_num_windings': gap,
                'model_gap_expander_capacity_windings': gap}

    def test_configured_index_survives_a_shell_less_run(self):
        idx, notes = resolve_outer_winding_idx_and_notes(
            self._cfg(130), shell_active=False,
            infer_outer_winding_idx=self.fail)
        self.assertEqual(idx, 130)
        self.assertTrue(any('no outer-shell losses' in n for n in notes))

    def test_unset_index_stays_none_without_a_shell(self):
        idx, notes = resolve_outer_winding_idx_and_notes(
            self._cfg(None), shell_active=False,
            infer_outer_winding_idx=self.fail)
        self.assertIsNone(idx)
        self.assertEqual(notes, [])

    def test_inference_runs_only_with_a_shell_and_no_config(self):
        idx, notes = resolve_outer_winding_idx_and_notes(
            self._cfg(None), shell_active=True,
            infer_outer_winding_idx=lambda: 42)
        self.assertEqual(idx, 42)
        self.assertTrue(any('inferred' in n for n in notes))

    def test_shell_run_keeps_the_configured_index(self):
        idx, notes = resolve_outer_winding_idx_and_notes(
            self._cfg(130), shell_active=True,
            infer_outer_winding_idx=self.fail)
        self.assertEqual(idx, 130)
        self.assertTrue(any('using configured' in n for n in notes))

    def test_gap_expander_control_also_runs_without_a_shell(self):
        with self.assertRaisesRegex(
                ValueError,
                'model_gap_expander_capacity_windings >= 133'):
            resolve_outer_winding_idx_and_notes(
                self._cfg(130, gap=130), shell_active=False,
                infer_outer_winding_idx=self.fail)

    def test_active_outer_winding_can_change_within_fixed_capacity(self):
        cfg = self._cfg(130, gap=144)
        idx, notes = resolve_outer_winding_idx_and_notes(
            cfg, shell_active=False, infer_outer_winding_idx=self.fail)
        self.assertEqual(idx, 130)
        self.assertFalse(any('capacity' in note for note in notes))


if __name__ == "__main__":
    unittest.main()
