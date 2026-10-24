"""Low positive Gaussian weights remain covered pixels, not missing data."""
import numpy as np
import pytest

from vesuvius.ink_detection.inference import infer, infer_full3d_tifxyz as native


@pytest.mark.parametrize("value,encoded", [(0., 0), (.5, 127), (1., 255)])
def test_flat_gaussian_single_patch_preserves_edges_and_mask(value, encoded):
    weights = infer.compute_importance_map_2d(patch_size=(16, 16), mode="gaussian").numpy()
    weights[5:7, 5:7] = 0
    with np.errstate(divide="raise", invalid="raise"):
        actual = next(infer.iter_probability_tiles(value * weights, weights, (16, 16)))
    expected = np.full((16, 16), encoded, dtype=np.uint8)
    expected[5:7, 5:7] = 0
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("value,encoded", [(0., 0), (.5, 128), (1., 255)])
def test_native_gaussian_single_patch_preserves_edges_and_mask(value, encoded):
    weights = native.create_importance_map((16, 16, 16), mode="gaussian")
    weights[5:7, 5:7, 5:7] = 0
    output = np.zeros(weights.shape, dtype=np.uint8)
    chunk = native.ChunkId(0, 0, 0)
    accumulator = native.ChunkAccumulator3D(
        output=output, target_chunks={chunk}, contribution_counts={chunk: 1},
        chunk_shape_zyx=weights.shape,
    )
    with np.errstate(divide="raise", invalid="raise"):
        accumulator.add_patch(
            patch_start_zyx=(0, 0, 0), probabilities=np.full(weights.shape, value, dtype=np.float32),
            weights=weights,
        )
    expected = np.full(weights.shape, encoded, dtype=np.uint8)
    expected[5:7, 5:7, 5:7] = 0
    np.testing.assert_array_equal(output, expected)


def test_low_weight_overlap_normalizes_weighted_average():
    shape = (16, 16, 16)
    weights = native.create_importance_map(shape, mode="gaussian")
    output = np.zeros(shape, dtype=np.uint8)
    chunk = native.ChunkId(0, 0, 0)
    accumulator = native.ChunkAccumulator3D(
        output=output, target_chunks={chunk}, contribution_counts={chunk: 2},
        chunk_shape_zyx=shape,
    )
    for value in (.25, .75):
        accumulator.add_patch(
            patch_start_zyx=(0, 0, 0), probabilities=np.full(shape, value, dtype=np.float32),
            weights=weights,
        )
    np.testing.assert_array_equal(output, np.full(shape, 128, dtype=np.uint8))
