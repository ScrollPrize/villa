from __future__ import annotations

import numpy as np
import pytest

from vesuvius.neural_tracing.autoreg_mesh.serialization import (
    deserialize_continuation_grid,
    deserialize_full_grid,
    serialize_split_conditioning_example,
)


def _make_surface(rows: int, cols: int) -> np.ndarray:
    row_axis = np.arange(rows, dtype=np.float32)[:, None]
    col_axis = np.arange(cols, dtype=np.float32)[None, :]
    row_grid = np.broadcast_to(row_axis, (rows, cols))
    col_grid = np.broadcast_to(col_axis, (rows, cols))
    z = 2.0 + (row_grid * 10.0) + col_grid
    y = 4.0 + row_grid
    x = 8.0 + col_grid
    return np.stack([z, y, x], axis=-1).astype(np.float32)


@pytest.mark.parametrize(
    ("direction", "cond", "masked", "frontier_band_width", "expected_target"),
    [
        (
            "left",
            _make_surface(3, 3)[:, :2],
            _make_surface(3, 3)[:, 2:],
            1,
            _make_surface(3, 3)[:, 2:].reshape(-1, 3),
        ),
        (
            "right",
            _make_surface(3, 4)[:, 2:],
            _make_surface(3, 4)[:, :2],
            1,
            np.concatenate([
                _make_surface(3, 4)[:, 1].reshape(-1, 3),
                _make_surface(3, 4)[:, 0].reshape(-1, 3),
            ], axis=0),
        ),
        (
            "up",
            _make_surface(4, 3)[:2, :],
            _make_surface(4, 3)[2:, :],
            1,
            _make_surface(4, 3)[2:, :].reshape(-1, 3),
        ),
        (
            "down",
            _make_surface(4, 3)[2:, :],
            _make_surface(4, 3)[:2, :],
            1,
            np.concatenate([
                _make_surface(4, 3)[1, :].reshape(-1, 3),
                _make_surface(4, 3)[0, :].reshape(-1, 3),
            ], axis=0),
        ),
    ],
)
def test_serialize_split_conditioning_example_orders_frontier_outward(
    direction: str,
    cond: np.ndarray,
    masked: np.ndarray,
    frontier_band_width: int,
    expected_target: np.ndarray,
) -> None:
    example = serialize_split_conditioning_example(
        cond_zyxs_local=cond,
        masked_zyxs_local=masked,
        direction=direction,
        volume_shape=(16, 16, 16),
        patch_size=(8, 8, 8),
        offset_num_bins=(4, 4, 4),
        frontier_band_width=frontier_band_width,
    )

    np.testing.assert_allclose(example["target_xyz"], expected_target)
    np.testing.assert_array_equal(example["target_stop"][:-1], np.zeros_like(example["target_stop"][:-1]))
    assert float(example["target_stop"][-1]) == pytest.approx(1.0)
    assert tuple(example["target_grid_shape"]) == tuple(masked.shape[:2])

    reconstructed = deserialize_continuation_grid(
        example["target_xyz"],
        direction=direction,
        grid_shape=example["target_grid_shape"],
    )
    np.testing.assert_allclose(reconstructed, masked)


def test_frontier_prompt_band_is_extracted_without_default_downsampling() -> None:
    full = _make_surface(3, 5)
    cond = full[:, :4]
    masked = full[:, 4:]
    example = serialize_split_conditioning_example(
        cond_zyxs_local=cond,
        masked_zyxs_local=masked,
        direction="left",
        volume_shape=(16, 16, 16),
        patch_size=(8, 8, 8),
        offset_num_bins=(4, 4, 4),
        frontier_band_width=2,
    )

    expected_prompt = cond[:, -2:, :]
    np.testing.assert_allclose(example["prompt_grid_local"], expected_prompt)
    assert tuple(example["prompt_meta"]["prompt_grid_shape"]) == (3, 2)
    assert tuple(example["target_grid_shape"]) == tuple(masked.shape[:2])

    full_grid = deserialize_full_grid(
        example["prompt_grid_local"],
        example["target_grid_local"],
        direction="left",
    )
    np.testing.assert_allclose(full_grid, np.concatenate([expected_prompt, masked], axis=1))
