import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from config import Config
from spiral_helpers import (
    _DENSE_WEIGHT_KEYS_NEEDING_OUTER_WINDING_IDX,
    _resolve_shell_outer_winding_idx,
    _structurally_disabled_dense_weight_keys,
    resolve_outer_winding_idx_and_notes,
)
from fit_spiral import (
    get_dt_loss_eligibility,
    get_unattached_pcl_dt_start,
    get_run_dt_resume_iteration,
)
from pcl_views import fiber_training_views, prepare_fiber_catalog
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


class FiberLinkViewTests(unittest.TestCase):
    """Fiber training views over store-resolved same-winding branch links."""

    @staticmethod
    def _fiber(collection_id, x_offset, attached=True):
        points = {
            point_id: {
                "id": point_id,
                "collectionId": collection_id,
                "p": [x_offset + point_id, 0.0, 0.0],
                "zyx": np.asarray([0.0, 0.0, x_offset + point_id],
                                  dtype=np.float32),
                "winding_annotation": float("nan"),
                **({"attachments": [{"id": f"patch-{collection_id}",
                                     "ij": [0.0, float(point_id)]}]}
                   if attached else {}),
            }
            for point_id in range(2)
        }
        return {"id": collection_id, "metadata": {}, "points": points}

    @staticmethod
    def _link(a_coll, a_point, b_coll, b_point):
        return {"a_coll": a_coll, "a_point": a_point,
                "b_coll": b_coll, "b_point": b_point, "pending": False}

    def _views(self, catalog, links, spacing=0):
        fibers = prepare_fiber_catalog(catalog)
        patches = {f"patch-{cid}": object() for cid in fibers}
        return fiber_training_views(
            fibers, links, patches, z_begin=-10, z_end=10, z_margin=0,
            min_point_spacing=spacing)

    def _catalog(self):
        return {"a": self._fiber(10, 0), "b": self._fiber(11, 10),
                "c": self._fiber(12, 20)}

    def test_branching_and_loop_closing_links_merge_one_component(self):
        links = [self._link(10, 0, 11, 0), self._link(10, 1, 12, 0),
                 self._link(11, 1, 12, 1)]

        cross, strips, groups, components = self._views(self._catalog(), links)

        self.assertEqual(len(cross), 1)
        self.assertEqual(cross[0]["metadata"]["logical_input_ids"],
                         ["a", "b", "c"])
        self.assertEqual(len(components), 1)
        self.assertEqual(len(strips), 3)
        self.assertEqual(groups, ["fibers"] * 3)
        self.assertEqual(len(cross[0]["chain"].extra_edges), 1)

    def test_link_endpoints_survive_strip_decimation(self):
        links = [self._link(10, 0, 11, 0), self._link(11, 1, 12, 0)]

        _, strips, _, _ = self._views(self._catalog(), links, spacing=1000)

        strips_by_id = {strip["id"]: strip for strip in strips}
        self.assertIn(0, strips_by_id[10]["link_points"])
        self.assertEqual(set(strips_by_id[11]["link_points"]), {0, 1})
        self.assertIn(0, strips_by_id[12]["link_points"])

    def test_unlinked_fibers_stay_separate_and_keep_their_strips(self):
        cross, strips, _, components = self._views(self._catalog(), [])

        self.assertEqual(components, [])
        self.assertEqual(len(cross), 3)
        # Every fiber keeps its strip, attached or not.
        self.assertEqual(sorted(strip["id"] for strip in strips), [10, 11, 12])

    def test_views_are_deterministic(self):
        links = [self._link(11, 0, 10, 0)]

        first = self._views(self._catalog(), links)
        second = self._views(self._catalog(), links)

        self.assertEqual(first[3], second[3])
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

    def test_ordinary_patch_uses_configured_erosion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_patch(root, {"format": "tifxyz", "scale": [1.0, 1.0]})

            patch = load_tifxyz(root)

            self.assertEqual(patch.erosion_cells(7), 7)


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
