import numpy as np
import pytest
import torch
from types import SimpleNamespace

from vesuvius.models.training.trainers.auxiliary import (
    AuxiliaryTrainer,
    DistanceTransformTrainer,
    SurfaceNormalsTrainer,
    StructureTensorTrainer,
    InplaneDirectionTrainer,
    NearestComponentTrainer,
)


def _make_mgr(aux_target_name: str, task_type: str, extra: dict | None = None):
    targets = {
        "ink": {"auxiliary_task": False},
        aux_target_name: {
            "auxiliary_task": True,
            "task_type": task_type,
            "source_target": "ink",
        },
    }
    if extra:
        targets[aux_target_name].update(extra)

    return SimpleNamespace(
        gpu_ids=None,
        use_ddp=False,
        targets=targets,
    )


def _base_sample(shape):
    image = torch.zeros((1, *shape), dtype=torch.float32)
    label = torch.zeros((1, *shape), dtype=torch.float32)
    if len(shape) == 2:
        label[:, shape[0] // 4 : shape[0] // 2, shape[1] // 4 : shape[1] // 2] = 1.0
    else:
        label[:, shape[0] // 4 : shape[0] // 2, shape[1] // 4 : shape[1] // 2, shape[2] // 4 : shape[2] // 2] = 1.0
    return {"image": image, "ink": label}


def test_distance_transform_trainer_injects_aux():
    mgr = _make_mgr("ink_dt", "distance_transform", {"distance_type": "inside"})
    trainer = DistanceTransformTrainer(mgr=mgr, verbose=False)

    sample = _base_sample((8, 8))
    augmented = trainer._prepare_sample(sample, is_training=True)

    assert "ink_dt" in augmented
    aux = augmented["ink_dt"]
    assert isinstance(aux, torch.Tensor)
    assert aux.shape == (1, 8, 8)
    assert "ink_dt" in augmented.get("regression_keys", [])


def test_surface_normals_trainer_injects_aux():
    mgr = _make_mgr("ink_normals", "surface_normals", {})
    trainer = SurfaceNormalsTrainer(mgr=mgr, verbose=False)

    sample = _base_sample((8, 8))
    augmented = trainer._prepare_sample(sample, is_training=True)

    aux = augmented["ink_normals"]
    assert aux.shape[0] == 2
    assert np.isfinite(aux.numpy()).all()


def test_structure_tensor_trainer_injects_aux():
    mgr = _make_mgr("ink_tensor", "structure_tensor", {})
    trainer = StructureTensorTrainer(mgr=mgr, verbose=False)

    sample = _base_sample((8, 8))
    augmented = trainer._prepare_sample(sample, is_training=True)

    aux = augmented["ink_tensor"]
    assert aux.ndim == 3
    assert aux.shape[0] in (1, 2, 3)


def test_inplane_direction_trainer_injects_aux():
    mgr = _make_mgr("ink_direction", "inplane_direction", {})
    trainer = InplaneDirectionTrainer(mgr=mgr, verbose=False)

    sample = _base_sample((8, 8))
    augmented = trainer._prepare_sample(sample, is_training=True)

    aux = augmented["ink_direction"]
    assert aux.shape[0] == 2
    assert torch.isfinite(aux).all()


def test_nearest_component_trainer_injects_aux():
    mgr = _make_mgr("ink_nearest", "nearest_component", {})
    trainer = NearestComponentTrainer(mgr=mgr, verbose=False)

    sample = _base_sample((8, 8))
    augmented = trainer._prepare_sample(sample, is_training=True)

    aux = augmented["ink_nearest"]
    assert aux.shape[0] == 3  # 2 direction + 1 distance in 2D
    assert torch.isfinite(aux).all()


def test_aux_trainer_loss_value_passes_source_predictions():
    class _SourceAwareLoss(torch.nn.Module):
        def forward(self, pred, target, *, source_pred=None):
            assert source_pred is not None
            return torch.mean(torch.abs(pred - target)) + torch.mean(source_pred)

    mgr = _make_mgr("ink_dt", "distance_transform", {})
    trainer = DistanceTransformTrainer(mgr=mgr, verbose=False)

    loss_fn = _SourceAwareLoss()
    pred = torch.ones(2, 2)
    target = torch.zeros(2, 2)
    outputs = {"ink": torch.ones(2, 2)}
    targets_dict = {"ink_dt": target}

    value = trainer._compute_loss_value(
        loss_fn,
        pred,
        target,
        target_name="ink_dt",
        targets_dict=targets_dict,
        outputs=outputs,
    )
    assert torch.isclose(value, torch.tensor(2.0))


def _make_mixed_mgr():
    mgr = _make_mgr("ink_dt", "distance_transform")
    mgr.targets["ink_normals"] = {
        "auxiliary_task": True,
        "task_type": "surface_normals",
        "source_target": "ink",
    }
    return mgr


def test_auxiliary_trainer_generates_each_configured_task_type():
    trainer = AuxiliaryTrainer(mgr=_make_mixed_mgr(), verbose=False)
    sample = _base_sample((8, 8, 8))

    augmented = trainer._prepare_sample(sample, is_training=True)

    expected_dt = DistanceTransformTrainer(
        mgr=_make_mgr("ink_dt", "distance_transform"), verbose=False
    )._prepare_sample(sample, is_training=True)["ink_dt"]
    expected_normals = SurfaceNormalsTrainer(
        mgr=_make_mgr("ink_normals", "surface_normals"), verbose=False
    )._prepare_sample(sample, is_training=True)["ink_normals"]
    assert torch.equal(augmented["ink_dt"], expected_dt)
    assert torch.equal(augmented["ink_normals"], expected_normals)
    assert set(augmented["regression_keys"]) == {"ink_dt", "ink_normals"}


def test_auxiliary_trainer_prepare_batch_stacks_each_task():
    trainer = AuxiliaryTrainer(mgr=_make_mixed_mgr(), verbose=False)
    sample = _base_sample((8, 8, 8))
    batch = {key: torch.stack([value, value]) for key, value in sample.items()}

    prepared = trainer._prepare_batch(batch, is_training=True)

    assert prepared["ink_dt"].shape == (2, 1, 8, 8, 8)
    assert prepared["ink_normals"].shape == (2, 3, 8, 8, 8)
    trainer._check_auxiliary_targets()


def test_auxiliary_trainer_rejects_unknown_task_type():
    with pytest.raises(ValueError, match="ink_unknown"):
        AuxiliaryTrainer(mgr=_make_mgr("ink_unknown", "unknown_task"), verbose=False)
