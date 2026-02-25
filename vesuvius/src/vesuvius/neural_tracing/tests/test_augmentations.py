import numpy as np
import torch
from scipy import ndimage

from dataset_rowcol_cond_test_setup import (
    build_minimal_split_dataset,
    build_minimal_triplet_dataset,
    set_all_rng_seeds,
)


def _sample_with_split_geometric_change(dataset, max_seed: int = 256):
    for seed in range(max_seed):
        set_all_rng_seeds(seed)
        sample = dataset[0]
        pre = dataset._last_split_masks_pre_aug
        post = dataset._last_split_target_inputs
        pre_surface = torch.maximum(pre["cond_gt"], pre["masked_seg"])
        post_surface = torch.maximum(post["cond_seg_gt"], post["masked_seg"])
        if not torch.equal(pre_surface, post_surface):
            return sample, pre, post
    raise RuntimeError("Could not find split seed that changes geometry")


def _sample_with_triplet_geometric_change(dataset, max_seed: int = 256):
    for seed in range(max_seed):
        set_all_rng_seeds(seed)
        sample = dataset[0]
        pre = dataset._last_triplet_masks_pre_aug
        post = dataset._last_triplet_target_inputs
        pre_stack = torch.stack([pre["cond_gt"], pre["behind_seg"], pre["front_seg"]], dim=0)
        post_stack = torch.stack([post["cond_seg_gt"], post["behind_seg"], post["front_seg"]], dim=0)
        if not torch.equal(pre_stack, post_stack):
            return sample, pre, post
    raise RuntimeError("Could not find triplet seed that changes geometry")


def _oracle_dense_displacement(surface_mask: np.ndarray):
    surface = np.asarray(surface_mask, dtype=bool)
    if surface.ndim != 3:
        raise ValueError(f"surface_mask must be 3D, got shape {tuple(surface.shape)}")
    if not surface.any():
        raise ValueError("surface_mask has no foreground voxels")

    distances, nearest_idx = ndimage.distance_transform_edt(
        ~surface,
        return_distances=True,
        return_indices=True,
    )
    expected = nearest_idx.astype(np.float32, copy=False)
    z_axis = np.arange(surface.shape[0], dtype=np.float32)[:, None, None]
    y_axis = np.arange(surface.shape[1], dtype=np.float32)[None, :, None]
    x_axis = np.arange(surface.shape[2], dtype=np.float32)[None, None, :]
    expected[0] -= z_axis
    expected[1] -= y_axis
    expected[2] -= x_axis
    return expected, distances.astype(np.float32, copy=False), nearest_idx.astype(np.float32, copy=False)


def _assert_displacement_matches_surface_oracle(pred_disp: np.ndarray, surface_mask: np.ndarray, *, label: str):
    surface = np.asarray(surface_mask, dtype=bool)
    pred = np.asarray(pred_disp, dtype=np.float32)
    if pred.shape != (3, *surface.shape):
        raise AssertionError(
            f"{label}: displacement shape must be (3, D, H, W), got {tuple(pred.shape)} "
            f"for surface shape {tuple(surface.shape)}"
        )
    if not np.isfinite(pred).all():
        raise AssertionError(f"{label}: displacement contains non-finite values")

    expected_disp, expected_dist, nearest_idx = _oracle_dense_displacement(surface)
    np.testing.assert_allclose(
        pred,
        expected_disp,
        atol=1e-5,
        rtol=0.0,
        err_msg=f"{label}: displacement components (dz,dy,dx) do not match oracle",
    )

    zz, yy, xx = np.indices(surface.shape, dtype=np.float32)
    endpoint = np.stack([zz + pred[0], yy + pred[1], xx + pred[2]], axis=0)
    np.testing.assert_allclose(
        endpoint,
        nearest_idx,
        atol=1e-4,
        rtol=0.0,
        err_msg=f"{label}: displacement endpoints do not land on oracle nearest-surface indices",
    )

    endpoint_int = np.rint(endpoint).astype(np.int64, copy=False)
    z_ok = (endpoint_int[0] >= 0) & (endpoint_int[0] < surface.shape[0])
    y_ok = (endpoint_int[1] >= 0) & (endpoint_int[1] < surface.shape[1])
    x_ok = (endpoint_int[2] >= 0) & (endpoint_int[2] < surface.shape[2])
    if not bool((z_ok & y_ok & x_ok).all()):
        raise AssertionError(f"{label}: some displacement endpoints are out of bounds")
    if not bool(surface[endpoint_int[0], endpoint_int[1], endpoint_int[2]].all()):
        raise AssertionError(f"{label}: some displacement endpoints are not on the surface mask")

    pred_dist = np.linalg.norm(pred, axis=0)
    np.testing.assert_allclose(
        pred_dist,
        expected_dist,
        atol=1e-4,
        rtol=1e-4,
        err_msg=f"{label}: displacement magnitude does not match nearest-surface distance",
    )

    surface_pred = pred[:, surface]
    if surface_pred.size and not np.allclose(surface_pred, 0.0, atol=1e-6):
        raise AssertionError(f"{label}: surface voxels must have zero displacement")

    for axis, axis_name in enumerate(("dz", "dy", "dx")):
        axis_abs = np.abs(expected_disp[axis])
        flat_idx = int(np.argmax(axis_abs))
        max_val = float(axis_abs.reshape(-1)[flat_idx])
        if max_val < 0.5:
            continue
        zyx = np.unravel_index(flat_idx, surface.shape)
        pred_comp = float(pred[(axis,) + zyx])
        exp_comp = float(expected_disp[(axis,) + zyx])
        if not np.isclose(pred_comp, exp_comp, atol=1e-5, rtol=0.0):
            raise AssertionError(
                f"{label}: axis sentinel mismatch for {axis_name} at voxel {zyx}: "
                f"pred={pred_comp} expected={exp_comp}"
            )


def test_split_dense_displacement_orientation_matches_augmented_surface():
    dataset = build_minimal_split_dataset()
    sample, _, post = _sample_with_split_geometric_change(dataset)

    cond_aug = post["cond_seg_gt"]
    masked_aug = post["masked_seg"]
    surface_aug = torch.maximum(cond_aug, masked_aug).detach().cpu().numpy() > 0.5

    torch.testing.assert_close(sample["masked_seg"], masked_aug, atol=0, rtol=0)
    _assert_displacement_matches_surface_oracle(
        sample["dense_gt_displacement"].detach().cpu().numpy(),
        surface_aug,
        label="split",
    )

    dense_weight = sample["dense_loss_weight"]
    assert dense_weight.shape == (1, *cond_aug.shape)
    assert bool(torch.isfinite(dense_weight).all())
    assert float(dense_weight.sum()) > 0.0


def test_triplet_dense_displacement_orientation_matches_augmented_surfaces():
    dataset = build_minimal_triplet_dataset()
    sample, _, post = _sample_with_triplet_geometric_change(dataset)

    behind_aug = post["behind_seg"].detach().cpu().numpy() > 0.5
    front_aug = post["front_seg"].detach().cpu().numpy() > 0.5
    branch_surfaces = [behind_aug, front_aug]

    channel_order = [int(x) for x in sample["triplet_channel_order"].tolist()]
    assert sorted(channel_order) == [0, 1]

    dense_disp = sample["dense_gt_displacement"].detach().cpu().numpy()
    branch0_disp = dense_disp[0:3]
    branch1_disp = dense_disp[3:6]
    _assert_displacement_matches_surface_oracle(
        branch0_disp,
        branch_surfaces[channel_order[0]],
        label=f"triplet_branch0_order{channel_order[0]}",
    )
    _assert_displacement_matches_surface_oracle(
        branch1_disp,
        branch_surfaces[channel_order[1]],
        label=f"triplet_branch1_order{channel_order[1]}",
    )

    dense_weight = sample["dense_loss_weight"]
    assert dense_weight.shape == (1, *post["cond_seg_gt"].shape)
    assert bool(torch.isfinite(dense_weight).all())
    assert float(dense_weight.sum()) > 0.0
