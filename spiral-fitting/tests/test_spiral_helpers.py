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
    get_run_dt_resume_iteration,
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
        }).as_dict()
        resume = 15
        self.assertEqual(get_dt_loss_eligibility(cfg, 14, resume), {
            'verified_patch': False,
            'track': False,
        })
        # Existing loss starts remain strict `iteration > start` checks.
        at_starts = get_dt_loss_eligibility(cfg, 20, resume)
        self.assertTrue(at_starts['verified_patch'])
        self.assertFalse(at_starts['track'])
        after_all = get_dt_loss_eligibility(cfg, 31, resume)
        self.assertTrue(all(after_all.values()))


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
