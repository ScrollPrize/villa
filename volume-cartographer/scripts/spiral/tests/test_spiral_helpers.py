import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from spiral_helpers import (
    _DENSE_WEIGHT_KEYS_NEEDING_OUTER_WINDING_IDX,
    _resolve_shell_outer_winding_idx,
    _structurally_disabled_dense_weight_keys,
    load_fiber_point_collection,
    resolve_outer_winding_idx_and_notes,
)
from tifxyz import load_tifxyz


class FiberPointCollectionTests(unittest.TestCase):
    def _write_fiber(self, directory, data):
        path = Path(directory) / "fiber.json"
        path.write_text(json.dumps(data))
        return path

    def test_loads_control_points_instead_of_line_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_fiber(temporary, {
                "control_points": [[4, 8, 12], [20, 24, 28]],
                "line_points": [[400, 800, 1200]],
            })

            collection = load_fiber_point_collection(
                path, collection_id=7, min_point_spacing=0)

            points = [point["p"] for point in collection["points"].values()]
            np.testing.assert_array_equal(points, [[1, 2, 3], [5, 6, 7]])

    def test_does_not_fall_back_to_line_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_fiber(temporary, {
                "line_points": [[4, 8, 12]],
            })

            collection = load_fiber_point_collection(path, collection_id=7)

            self.assertIsNone(collection)


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
                'loss_weight_dense_spacing_count',
                'loss_weight_dense_spacing_density',
                'loss_weight_dense_attachment',
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
                'gap_expander_num_windings': gap}

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
        idx, notes = resolve_outer_winding_idx_and_notes(
            self._cfg(130, gap=130), shell_active=False,
            infer_outer_winding_idx=self.fail)
        self.assertEqual(idx, 130)
        self.assertTrue(any('gap_expander_num_windings >= 133' in n
                            for n in notes))


if __name__ == "__main__":
    unittest.main()
