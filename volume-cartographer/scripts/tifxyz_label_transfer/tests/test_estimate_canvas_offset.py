from __future__ import annotations

import unittest

import numpy as np
from scipy.ndimage import gaussian_filter

from tifxyz_label_transfer.core import Surface
from tifxyz_label_transfer.estimate_canvas_offset import (
    _fit_shift_field,
    estimate_canvas_offset,
    measure_render_shift,
)


def smooth_texture(height: int, width: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = gaussian_filter(rng.normal(size=(height, width)), 3.0)
    scaled = (noise - noise.min()) / max(np.ptp(noise), 1e-9)
    return (scaled * 200.0 + 20.0).astype(np.uint8)


def plane_surface(height: int, width: int) -> Surface:
    rows, cols = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    return Surface(
        x=cols.copy(),
        y=rows.copy(),
        z=np.full((height, width), 10.0, dtype=np.float32),
    )


class MeasureRenderShiftTests(unittest.TestCase):
    def test_measures_circular_shift_with_documented_convention(self) -> None:
        reference = smooth_texture(128, 128)
        moving = np.roll(reference, (5, -7), axis=(0, 1))

        result = measure_render_shift(reference, moving, tile_size=128)

        # moving(u) = reference(u - k) for k = (5, -7); the documented
        # convention reference(u + s) = moving(u) therefore gives s = -k.
        self.assertAlmostEqual(result["shift_yx"][0], -5.0, delta=0.25)
        self.assertAlmostEqual(result["shift_yx"][1], 7.0, delta=0.25)
        self.assertGreaterEqual(result["tiles_used"], 1)

    def test_search_window_excludes_distant_peaks(self) -> None:
        reference = smooth_texture(128, 128)
        moving = np.roll(reference, (0, -20), axis=(0, 1))

        unbounded = measure_render_shift(reference, moving, tile_size=128)
        self.assertAlmostEqual(unbounded["shift_yx"][1], 20.0, delta=0.25)

        # When the only real peak lies outside the admissible window, the
        # measurement must fail loudly instead of reporting an in-window
        # noise peak as a shift.
        with self.assertRaises(ValueError):
            measure_render_shift(
                reference, moving, tile_size=128, max_shift_px=8.0
            )

    def test_rejects_renders_without_textured_overlap(self) -> None:
        blank = np.zeros((128, 128), dtype=np.uint8)
        with self.assertRaises(ValueError):
            measure_render_shift(blank, blank, tile_size=64)

    def test_shift_field_reports_spatial_drift_and_rejects_outlier(self) -> None:
        centers = np.asarray(
            [
                [64, 64],
                [64, 192],
                [192, 64],
                [192, 192],
                [128, 128],
                [100, 150],
            ],
            dtype=np.float64,
        )
        shifts = np.column_stack(
            (
                2.0 + 0.02 * (centers[:, 0] - 128),
                -3.0 + 0.01 * (centers[:, 1] - 128),
            )
        )
        shifts[-1] = [30.0, -20.0]
        field = _fit_shift_field(
            centers,
            shifts,
            np.ones(centers.shape[0]),
            (256, 256),
        )

        np.testing.assert_allclose(
            field["center_shift_yx"], [2.0, -3.0], atol=0.2
        )
        self.assertGreater(field["max_corner_drift_px"], 2.0)
        self.assertGreaterEqual(field["outlier_tiles"], 1)


class EstimatorTests(unittest.TestCase):
    def test_recovers_known_canvas_offset(self) -> None:
        height, width = 96, 128
        texture = smooth_texture(height, width)
        # Identical geometry in both TIFXYZ, but the source render's raster
        # is offset: source pixel (i, j) depicts canvas (i + 2, j - 3).
        source = plane_surface(height, width)
        target = plane_surface(height, width)
        source_render = np.roll(texture, (-2, 3), axis=(0, 1))
        target_render = texture

        result = estimate_canvas_offset(
            source,
            target,
            source_render,
            target_render,
            max_iterations=3,
            tolerance_px=0.35,
            measure_tile_px=64,
            min_coverage=0.5,
            max_tiles=16,
        )

        self.assertTrue(result["converged"])
        np.testing.assert_allclose(
            result["offset_yx_render_px"], [2.0, -3.0], atol=0.35
        )
        self.assertLessEqual(result["residual_render_px"], 0.35)
        # The uncorrected projection must have shown the injected offset.
        first = result["iterations"][0]["measured_shift_yx_render_px"]
        self.assertAlmostEqual(first[0], 2.0, delta=0.35)
        self.assertAlmostEqual(first[1], -3.0, delta=0.35)

    def test_aligned_renders_converge_immediately(self) -> None:
        height, width = 96, 128
        texture = smooth_texture(height, width)
        surface = plane_surface(height, width)

        result = estimate_canvas_offset(
            surface,
            surface,
            texture,
            texture,
            max_iterations=2,
            tolerance_px=0.35,
            measure_tile_px=64,
            min_coverage=0.5,
        )

        self.assertTrue(result["converged"])
        self.assertEqual(len(result["iterations"]), 1)
        np.testing.assert_allclose(
            result["offset_yx_render_px"], [0.0, 0.0], atol=0.35
        )


if __name__ == "__main__":
    unittest.main()
