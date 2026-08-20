import math
from types import SimpleNamespace

import numpy as np
import torch

from satisfaction_metrics import (
    PATCH_EVALUATION_CHUNK_SIZE,
    _ListPatchAtlas,
    evaluate_patch_satisfaction_packed,
    get_patch_satisfied_areas,
    metrics_config,
)
from spiral_sampling import load_spiral_sampling


class CountingIdentityTransform:
    def __init__(self):
        self.forward_calls = 0
        self.inverse_calls = 0

    def __call__(self, points):
        self.forward_calls += 1
        return points

    def inv(self, points):
        self.inverse_calls += 1
        return points


def spiral_point(theta, winding, dr=10.0, z=0.0):
    radius = (winding + theta / (2 * math.pi)) * dr
    return torch.tensor(
        [z, math.sin(theta) * radius, math.cos(theta) * radius],
        dtype=torch.float32,
    )


def patch_from_centres(centres, mask=None):
    rows, columns = len(centres), len(centres[0])
    # This construction preserves the requested centres exactly while being
    # sufficient for satisfaction, whose geometry sample is the corner mean.
    vertices = torch.zeros((rows + 1, columns + 1, 3), dtype=torch.float32)
    for row in range(rows):
        for column in range(columns):
            vertices[row + 1, column + 1] = (
                4 * centres[row][column]
                - vertices[row, column]
                - vertices[row, column + 1]
                - vertices[row + 1, column]
            )
    valid = (torch.ones((rows, columns), dtype=torch.bool)
             if mask is None else torch.as_tensor(mask, dtype=torch.bool))
    return SimpleNamespace(
        zyxs=vertices,
        valid_quad_mask=valid,
        area=float(valid.sum()),
    )


def assert_legacy_equal(packed, legacy):
    for index in (0, 1, 2, 4):
        torch.testing.assert_close(packed[index], legacy[index])
    for index in (3, 5):
        assert len(packed[index]) == len(legacy[index])
        for actual, expected in zip(packed[index], legacy[index]):
            torch.testing.assert_close(actual, expected)


def test_native_layout_handles_ragged_boundaries_and_empty_roi():
    native = load_spiral_sampling()
    assert native is not None and hasattr(native, 'PatchSatisfactionAtlas')
    masks = [
        np.ones((2, 3), dtype=bool),
        np.array([[1, 0, 1], [1, 1, 0]], dtype=bool),
        np.ones((1, 1), dtype=bool),
    ]
    zs = [
        np.zeros((3, 4), dtype=np.float32),
        np.zeros((3, 4), dtype=np.float32),
        np.full((2, 2), 10, dtype=np.float32),
    ]
    atlas = native.PatchSatisfactionAtlas(masks, zs, -1, 1)
    layout = atlas.packed_layout()
    np.testing.assert_array_equal(layout['patch_offsets'], [0, 6, 10, 10])
    np.testing.assert_array_equal(layout['full_valid_counts'], [6, 4, 1])
    np.testing.assert_array_equal(layout['quad_shapes'], [[2, 3], [2, 3], [1, 1]])
    # Solid 2x3: every cell touches the outside. Ragged: every selected cell
    # touches either the outside or a hole.
    assert np.asarray(layout['boundary_flags']).all()
    corners = np.asarray(layout['corner_vertex_ids'])
    assert corners.shape == (10, 4)
    assert len(np.unique(corners[:6])) == 12


def test_packed_matches_reference_for_crossing_multiwrap_and_disconnected():
    centres = [
        [spiral_point(2 * math.pi - .1, 3), spiral_point(.1, 4),
         spiral_point(.2, 4), spiral_point(.3, 4)],
        [spiral_point(2 * math.pi - .2, 5), spiral_point(.2, 6),
         spiral_point(.4, 8), spiral_point(.5, 8)],
        [spiral_point(2 * math.pi - .3, 7), spiral_point(.3, 8),
         spiral_point(.6, 8), spiral_point(.7, 8)],
    ]
    patches = [
        patch_from_centres(centres),
        patch_from_centres(
            centres,
            [[True, True, False, True],
             [True, True, False, True],
             [False, False, False, True]],
        ),
    ]
    transform = CountingIdentityTransform()
    packed = get_patch_satisfied_areas(
        transform, torch.tensor(10.0), patches, -1, 1)
    # A full config mapping deliberately selects the retained reference path.
    legacy = get_patch_satisfied_areas(
        CountingIdentityTransform(), torch.tensor(10.0), patches, -1, 1,
        metrics_overrides=dict(metrics_config))
    assert_legacy_equal(packed, legacy)


def test_strict_and_splicing_share_transform_residuals():
    patch = patch_from_centres([[spiral_point(.2, 3.47)]])
    patches = [patch]
    transform = CountingIdentityTransform()
    evaluation = evaluate_patch_satisfaction_packed(
        transform, torch.tensor(10.0), patches,
        _ListPatchAtlas(patches, torch.device('cpu')),
        -1, 1, include_splicing=True)
    assert transform.forward_calls == 1
    assert transform.inverse_calls == 1
    assert not evaluation.profiles['strict'].satisfied_patches.item()
    assert evaluation.profiles['splicing'].satisfied_patches.item()


def test_90k_one_quad_patches_use_point_chunks_not_patch_calls():
    patch = patch_from_centres([[spiral_point(.2, 3)]])
    patch_count = 90_000
    patches = [patch] * patch_count
    transform = CountingIdentityTransform()
    evaluation = evaluate_patch_satisfaction_packed(
        transform, torch.tensor(10.0), patches,
        _ListPatchAtlas(patches, torch.device('cpu')),
        -1, 1, include_splicing=True)
    expected = math.ceil(patch_count / PATCH_EVALUATION_CHUNK_SIZE)
    assert evaluation.forward_batches == expected
    assert evaluation.inverse_batches == expected
    assert transform.forward_calls == expected
    assert transform.inverse_calls == expected
    assert evaluation.profiles.keys() == {'strict', 'splicing'}
