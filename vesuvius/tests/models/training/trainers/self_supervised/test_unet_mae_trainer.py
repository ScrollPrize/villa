import torch
from types import SimpleNamespace

from vesuvius.models.training.trainers.self_supervised.train_unet_mae import TrainUNetMAE


def _make_mgr():
    return SimpleNamespace(
        gpu_ids=None,
        use_ddp=False,
        targets={},
        tr_configs={},
        train_patch_size=[8, 8],
        min_labeled_ratio=0.0,
        min_bbox_percent=0.0,
        skip_patch_validation=False,
        allow_unlabeled_data=False,
        normalization_scheme="zscore",
        intensity_properties={},
        slice_sampling_enabled=False,
        slice_sample_planes=[],
        slice_plane_weights={},
        slice_plane_patch_sizes={},
        slice_random_rotation_planes={},
        slice_random_tilt_planes={},
        slice_label_interpolation={},
        slice_save_plane_masks=False,
        slice_plane_mask_mode="plane",
        cache_valid_patches=False,
        dataset_config={},
        skip_intensity_sampling=True,
        data_path=".",
        in_channels=1,
    )


class _IdentityModel(torch.nn.Module):
    def forward(self, x):
        return {"mae": x.clone()}


def test_mae_trainer_mask_shape_and_values():
    trainer = TrainUNetMAE(mgr=_make_mgr(), verbose=False)
    inputs = torch.zeros((2, 1, 8, 8))
    mask = trainer._make_mask(inputs)

    assert mask.shape == (2, 1, 8, 8)
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_mae_trainer_compute_loss():
    trainer = TrainUNetMAE(mgr=_make_mgr(), verbose=False)
    loss_dict = trainer._build_loss()
    assert "mae" in loss_dict

    inputs = torch.randn((1, 1, 8, 8))
    mask = trainer._make_mask(inputs)
    trainer._current_mask = mask

    outputs = {"mae": inputs.clone() * 0.0}
    targets = {"mae": inputs}

    loss, task_losses = trainer._compute_train_loss(outputs, targets, loss_dict)
    assert loss.item() >= 0
    assert "mae" in task_losses


def test_mae_trainer_get_model_outputs_sets_mask():
    trainer = TrainUNetMAE(mgr=_make_mgr(), verbose=False)
    model = _IdentityModel()

    data_dict = {"image": torch.ones((1, 1, 8, 8))}
    inputs, targets, outputs = trainer._get_model_outputs(model, data_dict)

    assert "mae" in outputs
    assert trainer._current_mask is not None
    assert inputs.shape == targets["mae"].shape
