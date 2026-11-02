from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np

from vesuvius.tifxyz_label_transfer.core import Surface, transfer_array
from vesuvius.tifxyz_label_transfer.native import resolve_rasterizer


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


def wavy_surface(height: int, width: int, seed: int) -> Surface:
    rows, cols = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    rng = np.random.default_rng(seed)
    valid = rng.random((height, width)) > 0.04
    valid[[0, -1], :] = True
    valid[:, [0, -1]] = True
    return Surface(
        x=cols + 0.15 * np.sin(rows / 2.3),
        y=rows + 0.12 * np.cos(cols / 3.1),
        z=(
            10.0
            + 0.3 * np.sin(cols / 2.7)
            + 0.2 * np.cos(rows / 3.3)
            + rng.normal(0.0, 0.01, rows.shape)
        ),
        valid=valid,
    )


class NativeDifferentialTests(unittest.TestCase):
    def assert_differential(
        self,
        source: Surface,
        target: Surface,
        label: np.ndarray,
        *,
        output_shape: tuple[int, int],
        additional_label: np.ndarray | None = None,
        source_validity: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        def run(kind: str, cache: Path):
            output = np.full(output_shape, 37, dtype=label.dtype)
            valid = np.zeros(output_shape, dtype=np.uint8)
            distance = np.full(output_shape, np.inf, dtype=np.float32)
            additional_output = (
                None
                if additional_label is None
                else np.full(output_shape, 91, dtype=additional_label.dtype)
            )
            result = transfer_array(
                source,
                target,
                label,
                output_shape=output_shape,
                output=output,
                valid_output=valid,
                distance_output=distance,
                additional_source_labels=(
                    None
                    if additional_label is None
                    else [additional_label]
                ),
                additional_outputs=(
                    None
                    if additional_output is None
                    else [additional_output]
                ),
                source_validity=source_validity,
                uv_cache=cache,
                rasterizer=kind,
                **kwargs,
            )
            return (
                result[0].copy(),
                result[1].copy(),
                result[2].copy(),
                result[3].as_dict(),
                None
                if additional_output is None
                else additional_output.copy(),
            )

        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "uv.npz"
            expected = run("python", cache)
            actual = run("native", cache)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_allclose(actual[2], expected[2], atol=2e-5, rtol=2e-4)
        for key in actual[3]:
            if isinstance(actual[3][key], float):
                self.assertAlmostEqual(actual[3][key], expected[3][key], delta=2e-5)
            else:
                self.assertEqual(actual[3][key], expected[3][key], key)
        if additional_label is not None:
            np.testing.assert_array_equal(actual[4], expected[4])

    def test_affine_offset_dtype_and_edge_tiles_are_exact(self) -> None:
        source = wavy_surface(13, 17, seed=3)
        matrix = np.asarray(
            [
                [1.02, 0.01, 0.0, 30.0],
                [-0.02, 0.98, 0.0, -12.0],
                [0.0, 0.0, 1.01, 4.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        points = np.column_stack(
            (source.x.ravel(), source.y.ravel(), source.z.ravel())
        )
        transformed = points @ matrix[:3, :3].T + matrix[:3, 3]
        target = Surface(
            x=transformed[:, 0].reshape(source.shape),
            y=transformed[:, 1].reshape(source.shape),
            z=transformed[:, 2].reshape(source.shape),
            valid=source.valid,
        )
        label = (
            np.arange(19, dtype=np.uint16)[:, None] * 101
            + np.arange(23, dtype=np.uint16)[None, :]
        )
        additional = np.asarray(label * 7 + 3, dtype=np.uint16)
        source_validity = np.full(label.shape, 255, dtype=np.uint8)
        source_validity[3:5, 6:9] = 128
        source_validity[0, :4] = 0
        self.assert_differential(
            source,
            target,
            label,
            output_shape=(35, 41),
            additional_label=additional,
            source_validity=source_validity,
            affine=matrix,
            max_distance=0.2,
            label_offset_yx=(0.375, -0.625),
            tile_size=16,
            workers=3,
            fill_value=37,
        )

    def test_fold_rejection_and_seam_provenance_are_exact(self) -> None:
        rows = np.broadcast_to(
            np.arange(2, dtype=np.float32)[:, None], (2, 12)
        ).copy()
        folded_x = np.concatenate(
            (
                np.arange(6, dtype=np.float32),
                np.arange(5, -1, -1, dtype=np.float32),
            )
        )
        folded_z = np.concatenate(
            (
                np.full(6, 10.0, dtype=np.float32),
                np.full(6, 10.6, dtype=np.float32),
            )
        )
        source = Surface(
            x=np.broadcast_to(folded_x, (2, 12)).copy(),
            y=rows,
            z=np.broadcast_to(folded_z, (2, 12)).copy(),
        )
        target = Surface(
            x=np.ones((2, 2), dtype=np.float32),
            y=np.broadcast_to(
                np.arange(2, dtype=np.float32)[:, None], (2, 2)
            ).copy(),
            z=np.broadcast_to(
                np.asarray([10.0, 10.6], dtype=np.float32), (2, 2)
            ).copy(),
        )
        label = np.broadcast_to(
            np.arange(12, dtype=np.uint8), source.shape
        ).copy()
        validity = np.full(label.shape, 255, dtype=np.uint8)
        validity[:, 0] = 0
        validity[:, 10] = 128
        self.assert_differential(
            source,
            target,
            label,
            output_shape=(2, 37),
            source_validity=validity,
            max_distance=0.2,
            nearest_vertices=4,
            tile_size=16,
            workers=4,
            fill_seams=True,
        )

    def test_content_anchored_seam_gate_is_exact(self) -> None:
        # The content anchor changes only which distance field feeds the
        # seam gate; the native kernel must apply it identically, including
        # blocking the whole disagreeing right side.
        source = plane(8, 24)
        target = plane(8, 24)
        target.z[:, 16:] += 5.0
        label = np.zeros((8, 24), dtype=np.uint8)
        label[:, :3] = 7
        self.assert_differential(
            source,
            target,
            label,
            output_shape=(8, 24),
            max_distance=0.2,
            nearest_vertices=4,
            tile_size=16,
            workers=2,
            fill_seams=True,
            max_seam_distance=6.0,
            seam_anchor="content",
        )

    def test_extreme_canvas_offset_is_safely_rejected(self) -> None:
        source = plane(5, 6)
        label = np.arange(30, dtype=np.uint8).reshape(source.shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.assert_differential(
                source,
                source,
                label,
                output_shape=(11, 13),
                max_distance=0.1,
                label_offset_yx=(1e300, -1e300),
                tile_size=7,
                workers=2,
            )

    def test_randomized_surface_matrix_is_exact(self) -> None:
        for seed in range(8):
            with self.subTest(seed=seed):
                source = wavy_surface(9 + seed % 3, 11 + seed % 4, seed)
                rng = np.random.default_rng(seed + 100)
                target = Surface(
                    x=source.x + rng.normal(0.0, 0.015, source.shape),
                    y=source.y + rng.normal(0.0, 0.015, source.shape),
                    z=source.z + rng.normal(0.0, 0.025, source.shape),
                    valid=source.valid,
                )
                label_shape = (15 + seed % 4, 18 + seed % 5)
                wide_label = rng.integers(
                    0,
                    2**16,
                    (label_shape[0], label_shape[1] * 2),
                    dtype=np.uint16,
                )
                label = wide_label[:, ::2]
                wide_validity = rng.choice(
                    np.asarray([0, 128, 255], dtype=np.uint8),
                    size=(label_shape[0], label_shape[1] * 2),
                    p=(0.05, 0.1, 0.85),
                )
                source_validity = wide_validity[:, ::2]
                self.assert_differential(
                    source,
                    target,
                    label,
                    output_shape=(21 + seed, 27 + 2 * seed),
                    source_validity=source_validity,
                    max_distance=0.35,
                    label_offset_yx=(0.125 * seed, -0.2 * seed),
                    nearest_vertices=4,
                    tile_size=7 + seed,
                    workers=1 + seed % 4,
                    fill_seams=bool(seed % 2),
                    max_seam_distance=1.0 + 0.75 * seed,
                    fill_value=37,
                )
