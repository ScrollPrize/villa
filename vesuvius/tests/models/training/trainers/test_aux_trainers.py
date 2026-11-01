import numpy as np
import pytest
import torch
from contextlib import nullcontext
from types import SimpleNamespace

from vesuvius.models.training.trainers.auxiliary import (
    DistanceTransformTrainer,
    SurfaceNormalsTrainer,
    StructureTensorTrainer,
    InplaneDirectionTrainer,
    NearestComponentTrainer,
    AuxiliaryTrainer,
)
from vesuvius.models.training.cli import (
    _build_base_trainer,
    _configured_auxiliary_target_names,
)
from vesuvius.models.training.train import BaseTrainer


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


def test_unified_auxiliary_trainer_injects_multiple_task_types():
    mgr = _make_mgr("ink_dt", "distance_transform", {"distance_type": "inside"})
    mgr.targets["ink_normals"] = {
        "auxiliary_task": True,
        "task_type": "surface_normals",
        "source_target": "ink",
    }
    trainer = AuxiliaryTrainer(mgr=mgr, verbose=False)
    samples = [_base_sample((8, 8)), _base_sample((8, 8))]
    batch = {
        "image": torch.stack([sample["image"] for sample in samples]),
        "ink": torch.stack([sample["ink"] for sample in samples]),
    }

    augmented = trainer._prepare_batch(batch, is_training=True)

    assert augmented["ink_dt"].shape == (2, 1, 8, 8)
    assert augmented["ink_normals"].shape == (2, 2, 8, 8)
    assert torch.equal(augmented["ink"], batch["ink"])
    assert augmented["image"].data_ptr() == batch["image"].data_ptr()
    assert augmented["ink"].data_ptr() == batch["ink"].data_ptr()
    assert augmented["regression_keys"] == ["ink_dt", "ink_normals"]
    assert torch.isfinite(augmented["ink_dt"]).all()
    assert torch.isfinite(augmented["ink_normals"]).all()


def test_unified_auxiliary_trainer_rejects_unknown_task_type():
    mgr = _make_mgr("unknown", "not_a_task")

    with pytest.raises(ValueError, match="Unsupported auxiliary task types: not_a_task"):
        AuxiliaryTrainer(mgr=mgr, verbose=False)


def test_unified_auxiliary_trainer_rejects_missing_source_target():
    mgr = _make_mgr("ink_dt", "distance_transform", {"distance_type": "inside"})
    trainer = AuxiliaryTrainer(mgr=mgr, verbose=False)
    batch = {"image": torch.zeros((1, 1, 8, 8))}

    with pytest.raises(ValueError, match="every sample must provide"):
        trainer._prepare_batch(batch, is_training=True)


def test_base_trainer_selection_activates_configured_auxiliary_targets():
    mgr = _make_mgr("ink_dt", "distance_transform", {"distance_type": "inside"})

    assert _configured_auxiliary_target_names(mgr) == ["ink_dt"]
    assert isinstance(_build_base_trainer(mgr, verbose=False), AuxiliaryTrainer)


def test_base_trainer_selection_is_unchanged_without_auxiliary_targets():
    mgr = SimpleNamespace(
        gpu_ids=None,
        use_ddp=False,
        targets={"ink": {"auxiliary_task": False}},
    )

    assert _configured_auxiliary_target_names(mgr) == []
    assert type(_build_base_trainer(mgr, verbose=False)) is BaseTrainer


def test_base_trainer_selection_rejects_on_device_auxiliary_augmentation():
    mgr = _make_mgr("ink_dt", "distance_transform", {"distance_type": "inside"})
    mgr.augment_on_device = True

    with pytest.raises(ValueError, match="augment_on_device=true"):
        _build_base_trainer(mgr, verbose=False)


@pytest.mark.parametrize("verbose", [False, True])
def test_auxiliary_training_step_with_batch_metadata(verbose, capsys):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Conv3d(1, 2, kernel_size=1)
            torch.nn.init.zeros_(self.decoder.weight)
            torch.nn.init.zeros_(self.decoder.bias)

        def forward(self, image):
            output = self.decoder(image)
            return {"ink": output[:, :1], "ink_dt": output[:, 1:]}

    mgr = _make_mgr("ink_dt", "distance_transform", {"distance_type": "inside"})
    trainer = _build_base_trainer(mgr, verbose=verbose)
    sample = _base_sample((8, 8, 8))
    batch = {key: value.unsqueeze(0) for key, value in sample.items()}
    batch["patch_info"] = {
        "volume_name": ["scroll1"],
        "coordinates": {"start": (0, 0, 0), "valid": True},
        "optional": None,
        "empty": torch.empty(0),
    }
    model = Model().to(trainer.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    losses = {name: [(torch.nn.MSELoss(), 1.0)] for name in mgr.targets}

    result = trainer._train_step(
        model, batch, losses, False, nullcontext(), 0, 0, verbose=verbose,
        scaler=torch.amp.GradScaler(trainer.device.type, enabled=False),
        optimizer=optimizer, num_iters=1,
    )

    assert result[-1] is True
    assert result[1]["ink_dt"] > 0
    assert torch.all(model.decoder.bias.detach() > 0)
    output = capsys.readouterr().out
    if verbose:
        assert "regression_keys: ['ink_dt']" in output
        assert "volume_name: ['scroll1']" in output
        assert "start: (0, 0, 0)" in output
        assert "valid: True" in output
        assert "optional: None" in output
        assert "empty: torch.float32, torch.Size([0]), empty" in output
    else:
        assert "Items from the first batch" not in output


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
