import importlib.util

import numpy as np
import pytest


if importlib.util.find_spec("vc") is None:
    pytest.skip("vc is required to import triplet wrap inference", allow_module_level=True)

from vesuvius.neural_tracing.inference import infer_rowcol_triplet_wraps as triplet_wraps


def test_triplet_slot_assignment_follows_local_chart_side_when_slots_swap():
    world = np.zeros((2, 3), dtype=np.float32)
    uv = np.asarray([[0, 0], [0, 1]], dtype=np.int32)
    normals = np.zeros((1, 2, 3), dtype=np.float32)
    normals[0, 0] = [1.0, 0.0, 0.0]
    normals[0, 1] = [0.0, 1.0, 0.0]
    normals_valid = np.ones((1, 2), dtype=bool)

    slot_a_disp = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
        ],
        dtype=np.float32,
    )
    slot_b_disp = np.asarray(
        [
            [-2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )

    front, front_valid, back, back_valid, assigned = triplet_wraps._assign_triplet_slots_to_chart_sides(
        world=world,
        uv_rc=uv,
        slot_a_disp=slot_a_disp,
        slot_a_valid=np.ones(2, dtype=bool),
        slot_b_disp=slot_b_disp,
        slot_b_valid=np.ones(2, dtype=bool),
        normals=normals,
        normals_valid=normals_valid,
    )

    np.testing.assert_allclose(front, [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    np.testing.assert_allclose(back, [[-2.0, 0.0, 0.0], [0.0, -2.0, 0.0]])
    assert front_valid.tolist() == [True, True]
    assert back_valid.tolist() == [True, True]
    assert assigned == 2


def test_displacement_sampling_can_average_dense_local_yx_kernel():
    disp = np.zeros((3, 3, 3, 3), dtype=np.float32)
    disp[0, 1, :, :] = 1.0
    disp[0, 1, 1, 1] = 91.0
    coords = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32)

    legacy_disp, legacy_valid = triplet_wraps._sample_trilinear_displacement_stack(
        disp,
        coords,
        sample_radius=0.0,
    )
    robust_disp, robust_valid = triplet_wraps._sample_trilinear_displacement_stack(
        disp,
        coords,
        sample_radius=1.0,
        sample_spacing=1.0,
        sample_reduce="mean",
    )

    np.testing.assert_allclose(legacy_disp, [[91.0, 0.0, 0.0]])
    assert legacy_valid.tolist() == [True]
    np.testing.assert_allclose(robust_disp, [[11.0, 0.0, 0.0]])
    assert robust_valid.tolist() == [True]
