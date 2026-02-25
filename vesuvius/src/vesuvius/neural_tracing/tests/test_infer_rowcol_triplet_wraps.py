from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vesuvius.neural_tracing.inference.infer_rowcol_triplet_wraps as infer_module


def test_build_triplet_direction_priors_for_crop_cond_mask():
    crop_size = (2, 2, 2)
    cond_vox = np.zeros(crop_size, dtype=np.float32)
    cond_vox[0, 0, 0] = 1.0
    normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    priors = infer_module._build_triplet_direction_priors_for_crop(
        crop_size=crop_size,
        cond_vox=cond_vox,
        global_unit_normal=normal,
        mask_mode="cond",
    )

    assert priors.shape == (6, *crop_size)
    assert np.allclose(priors[0:3, 0, 0, 0], normal)
    assert np.allclose(priors[3:6, 0, 0, 0], -normal)
    assert np.allclose(priors[:, 1, 1, 1], 0.0)


def test_build_triplet_direction_priors_for_crop_full_mask():
    crop_size = (1, 1, 1)
    cond_vox = np.zeros(crop_size, dtype=np.float32)
    normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    priors = infer_module._build_triplet_direction_priors_for_crop(
        crop_size=crop_size,
        cond_vox=cond_vox,
        global_unit_normal=normal,
        mask_mode="full",
    )

    assert np.allclose(priors[0:3, 0, 0, 0], normal)
    assert np.allclose(priors[3:6, 0, 0, 0], -normal)


def test_estimate_global_unit_normal_raises_when_no_valid_normals():
    normals = np.zeros((2, 2, 3), dtype=np.float32)
    valid = np.zeros((2, 2), dtype=bool)
    with pytest.raises(RuntimeError, match="No valid surface normals"):
        infer_module._estimate_global_unit_normal(normals, valid)


def test_run_triplet_inference_requires_direction_conditioned_channels():
    args = SimpleNamespace(
        batch_size=1,
        crop_input_workers=1,
        device="cpu",
        tta=False,
        verbose=False,
        dp_radius=0,
        dp_reject_frac=0.0,
        dp_reject_min_keep=1,
    )
    with pytest.raises(RuntimeError, match="in_channels=8"):
        infer_module._run_triplet_inference(
            args=args,
            model_state={"expected_in_channels": 2, "model_config": {}},
            records=[],
            crop_size=(1, 1, 1),
            world_points=np.zeros((1, 3), dtype=np.float32),
            uv_points=np.zeros((1, 2), dtype=np.int32),
            volume_arr=np.zeros((1, 1, 1), dtype=np.float32),
            shape_hw=(1, 1),
            dense_subsample_stride=1,
            input_normals=np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32),
            input_normals_valid=np.array([[True]], dtype=bool),
        )


def test_run_triplet_inference_uses_fixed_branch_mapping(monkeypatch):
    args = SimpleNamespace(
        batch_size=1,
        crop_input_workers=1,
        device="cpu",
        tta=False,
        verbose=False,
        dp_radius=0,
        dp_reject_frac=0.0,
        dp_reject_min_keep=1,
    )

    item = {
        "bbox_id": 0,
        "min_corner": np.array([0, 0, 0], dtype=np.int32),
        "uv": np.array([[0, 0]], dtype=np.int32),
        "world": np.array([[10.0, 20.0, 30.0]], dtype=np.float32),
        "local": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "cond_vox": np.ones((1, 1, 1), dtype=np.float32),
        "volume": np.zeros((1, 1, 1), dtype=np.float32),
    }

    def fake_gather_batch_items(**kwargs):
        return [item]

    def fake_predict_displacement(_args, _model_state, _model_inputs, use_tta=None, profiler=None):
        out = np.zeros((1, 6, 1, 1, 1), dtype=np.float32)
        out[0, 0:3, 0, 0, 0] = np.array([1.0, 2.0, 3.0], dtype=np.float32)   # slot A
        out[0, 3:6, 0, 0, 0] = np.array([-1.0, -2.0, -3.0], dtype=np.float32)  # slot B
        return torch.from_numpy(out)

    def fake_sample_trilinear(disp, local):
        vec = np.array([disp[0, 0, 0, 0], disp[1, 0, 0, 0], disp[2, 0, 0, 0]], dtype=np.float32)
        sampled = np.repeat(vec[None, :], local.shape[0], axis=0)
        return sampled, np.ones((local.shape[0],), dtype=bool)

    def fake_robustify(uv_rc, world_zyx, **kwargs):
        return np.asarray(world_zyx, dtype=np.float32)

    def fake_project_sparse(pred_grid, pred_valid):
        return pred_grid, pred_valid, {}

    monkeypatch.setattr(infer_module, "_gather_batch_items", fake_gather_batch_items)
    monkeypatch.setattr(infer_module, "predict_displacement", fake_predict_displacement)
    monkeypatch.setattr(infer_module, "_sample_trilinear_displacement_stack", fake_sample_trilinear)
    monkeypatch.setattr(infer_module, "_robustify_samples_with_dense_projection", fake_robustify)
    monkeypatch.setattr(infer_module, "_project_sparse_to_input_lattice", fake_project_sparse)

    out = infer_module._run_triplet_inference(
        args=args,
        model_state={
            "expected_in_channels": 8,
            "model_config": {"triplet_direction_prior_mask": "cond"},
        },
        records=[{"bbox_id": 0, "bbox": (0, 0, 0, 0, 0, 0)}],
        crop_size=(1, 1, 1),
        world_points=np.array([[10.0, 20.0, 30.0]], dtype=np.float32),
        uv_points=np.array([[0, 0]], dtype=np.int32),
        volume_arr=np.zeros((2, 2, 2), dtype=np.float32),
        shape_hw=(1, 1),
        dense_subsample_stride=1,
        input_normals=np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32),
        input_normals_valid=np.array([[True]], dtype=bool),
    )

    assert bool(out["front_valid"][0, 0])
    assert bool(out["back_valid"][0, 0])
    assert np.allclose(out["front_grid"][0, 0], np.array([11.0, 22.0, 33.0], dtype=np.float32))
    assert np.allclose(out["back_grid"][0, 0], np.array([9.0, 18.0, 27.0], dtype=np.float32))
    assert out["orientation_mode"] == "global_direction_prior_fixed"
    assert out["triplet_slot_to_output"] == {"A": "front", "B": "back"}
