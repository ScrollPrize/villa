import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from config import Config
from spiral_helpers import (
    _resolve_shell_outer_winding_idx,
    load_fiber_point_collection,
    resolve_outer_winding_idx_and_notes,
)
from fit_spiral import get_dt_loss_eligibility, materialize_fiber_fit_inputs
from tifxyz import load_tifxyz


class RunDtLossScheduleTests(unittest.TestCase):
    def test_schedule_composes_with_strict_family_start_thresholds(self):
        cfg = Config({
            'loss_start_patch_dt': 10,
            'loss_start_track_dt': 20,
            'loss_start_unverified_patch_dt': 30,
            'loss_start_unattached_pcl_dt': 25,
        }).as_dict()
        resume = 15
        self.assertEqual(get_dt_loss_eligibility(cfg, 14, resume), {
            'verified_patch': False,
            'unverified_patch': False,
            'track': False,
            'unattached_pcl': False,
        })
        # Existing loss starts remain strict `iteration > start` checks.
        at_starts = get_dt_loss_eligibility(cfg, 20, resume)
        self.assertTrue(at_starts['verified_patch'])
        self.assertFalse(at_starts['unattached_pcl'])
        self.assertFalse(at_starts['track'])
        self.assertFalse(at_starts['unverified_patch'])
        self.assertFalse(get_dt_loss_eligibility(cfg, 25, resume)['unattached_pcl'])
        self.assertTrue(get_dt_loss_eligibility(cfg, 26, resume)['unattached_pcl'])
        after_all = get_dt_loss_eligibility(cfg, 31, resume)
        self.assertTrue(all(after_all.values()))


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
            self._cfg(130, gap=144), shell_active=False,
            infer_outer_winding_idx=self.fail)
        self.assertEqual(idx, 130)
        self.assertTrue(any('no outer-shell losses' in n for n in notes))

    def test_inference_runs_only_with_a_shell_and_no_config(self):
        idx, notes = resolve_outer_winding_idx_and_notes(
            self._cfg(None), shell_active=True,
            infer_outer_winding_idx=lambda: 42)
        self.assertEqual(idx, 42)
        self.assertTrue(any('inferred' in n for n in notes))
