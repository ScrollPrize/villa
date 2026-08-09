from __future__ import annotations

import unittest

import numpy as np

from tifxyz_label_transfer import core
from tifxyz_label_transfer.core import Surface, transfer_array
from tifxyz_label_transfer.planar import transfer_array_planar


def plane(height: int, width: int) -> Surface:
    rows, cols = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    return Surface(
        x=cols,
        y=rows,
        z=np.full((height, width), 10.0, dtype=np.float32),
    )


class PlanarTransferTests(unittest.TestCase):
    def test_planar_method_is_separate_from_geometry_core(self) -> None:
        self.assertFalse(hasattr(core, "transfer_array_planar"))

    def test_planar_matches_geometry_on_identical_planes(self) -> None:
        source = plane(16, 16)
        target = plane(16, 16)
        rng = np.random.default_rng(3)
        label = rng.integers(0, 255, (16, 16), dtype=np.uint8)

        geometry_output, geometry_valid, _, _ = transfer_array(
            source, target, label
        )
        planar_output, planar_valid, matrix, report = transfer_array_planar(
            source, target, label
        )

        np.testing.assert_array_equal(planar_output, geometry_output)
        np.testing.assert_array_equal(planar_valid, geometry_valid)
        self.assertLess(report["residual_label_px"]["max"], 1e-6)
        np.testing.assert_allclose(
            matrix[:, :2], np.eye(2), atol=1e-9
        )

    def test_planar_recovers_transposed_parameterization(self) -> None:
        rows, cols = np.meshgrid(
            np.arange(20, dtype=np.float32),
            np.arange(20, dtype=np.float32),
            indexing="ij",
        )
        source = plane(20, 20)
        target = Surface(
            x=rows,
            y=cols,
            z=np.full((20, 20), 10.0, dtype=np.float32),
        )
        rng = np.random.default_rng(4)
        label = rng.integers(0, 255, (20, 20), dtype=np.uint8)

        output, valid, _, report = transfer_array_planar(
            source, target, label
        )

        np.testing.assert_array_equal(output, label.T)
        self.assertEqual(int(valid.min()), 255)
        self.assertLess(report["residual_label_px"]["max"], 1e-6)

    def test_planar_fills_geometry_rejection_holes(self) -> None:
        source = plane(24, 24)
        target = plane(24, 24)
        # A band of target vertices lifted beyond max_distance: the
        # per-pixel transfer must reject them, the planar warp must not.
        target.z[8:12, :] += 5.0
        rng = np.random.default_rng(5)
        label = rng.integers(1, 255, (24, 24), dtype=np.uint8)

        _, geometry_valid, _, _ = transfer_array(source, target, label)
        planar_output, planar_valid, _, report = transfer_array_planar(
            source, target, label
        )

        self.assertGreater(int((geometry_valid == 0).sum()), 0)
        self.assertEqual(int(planar_valid.min()), 255)
        np.testing.assert_array_equal(planar_output, label)
        self.assertLess(report["residual_label_px"]["max"], 1e-6)
