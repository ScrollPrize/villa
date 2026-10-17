from __future__ import annotations

import importlib.util
import math
import os
import site
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace


if os.environ.get("VESUVIUS_FALLBACK_SITE_PACKAGES"):
    site.addsitedir(os.environ["VESUVIUS_FALLBACK_SITE_PACKAGES"])

import pytest
import torch
import torch.nn.functional as F


VILLA_ROOT = Path(__file__).parents[3]
TARGET = (
    VILLA_ROOT
    / "src/vesuvius/models/training/trainers/semi_supervised/"
    "train_uncertainty_aware_mean_teacher.py"
)
RAMPS = (
    VILLA_ROOT
    / "src/vesuvius/models/training/trainers/semi_supervised/ramps.py"
)


class _BaseTrainer:
    def __init__(self, mgr=None, verbose=True):
        self.mgr = mgr
        self.verbose = verbose
        self.device = torch.device("cpu")

    def _compute_train_loss(self, outputs, targets, loss_fns):
        return torch.tensor(0.0), {}


class _ConstantTeacher(torch.nn.Module):
    def __init__(self, logit):
        super().__init__()
        self.logit = float(logit)

    def forward(self, inputs):
        return {
            "surface": torch.full(
                (inputs.shape[0], 1, *inputs.shape[2:]),
                self.logit,
                dtype=inputs.dtype,
                device=inputs.device,
            )
        }


def _load_target():
    replacements = {}

    def install(name, module):
        replacements[name] = sys.modules.get(name)
        sys.modules[name] = module

    train_module = ModuleType("vesuvius.models.training.train")
    train_module.BaseTrainer = _BaseTrainer
    install(train_module.__name__, train_module)

    ramps_name = "vesuvius.models.training.trainers.semi_supervised.ramps"
    ramps_spec = importlib.util.spec_from_file_location(ramps_name, RAMPS)
    assert ramps_spec is not None and ramps_spec.loader is not None
    ramps = importlib.util.module_from_spec(ramps_spec)
    install(ramps_name, ramps)
    ramps_spec.loader.exec_module(ramps)

    package = ModuleType("vesuvius.models.training.trainers.semi_supervised")
    package.__path__ = [str(TARGET.parent)]
    package.ramps = ramps
    install(package.__name__, package)

    sampler = ModuleType(
        "vesuvius.models.training.trainers.semi_supervised."
        "two_stream_batch_sampler"
    )
    sampler.TwoStreamBatchSampler = object
    install(sampler.__name__, sampler)

    name = "uamt_binary_probability_target"
    spec = importlib.util.spec_from_file_location(name, TARGET)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    install(name, module)
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        for module_name, previous in reversed(list(replacements.items())):
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous


@pytest.fixture(scope="module")
def target():
    return _load_target()


def _manager(num_classes=1, aggregation="average"):
    return SimpleNamespace(
        train_batch_size=2,
        out_channels=num_classes,
        max_steps_per_epoch=1,
        max_epoch=1,
        uncertainty_T=2,
        consistency_weight=1.0,
        consistency_rampup=0.0,
        noise_scale=0.0,
        ensemble_model_paths=[],
        ensemble_aggregation=aggregation,
    )


def test_single_channel_logits_use_sigmoid(target):
    logits = torch.tensor([[[2.0, -2.0]]])

    assert target.logits_to_probabilities(logits) == pytest.approx(
        torch.sigmoid(logits)
    )


@pytest.mark.parametrize("num_classes", [2, 3, 4])
def test_multiclass_logits_still_use_softmax(target, num_classes):
    logits = torch.linspace(-1.0, 1.0, num_classes).reshape(1, num_classes, 1)

    assert target.logits_to_probabilities(logits) == pytest.approx(
        F.softmax(logits, dim=1)
    )


def test_single_channel_probability_mse_and_gradient(target):
    logits = torch.tensor([[[-2.0]]], requires_grad=True)
    teacher = torch.sigmoid(torch.tensor([[[2.0]]]))
    actual = target.prob_mse_loss(logits, teacher).mean()
    actual.backward()

    reference_logits = torch.tensor([[[-2.0]]], requires_grad=True)
    reference = ((torch.sigmoid(reference_logits) - teacher) ** 2).mean()
    reference.backward()

    assert actual.detach().item() == pytest.approx(reference.detach().item())
    assert logits.grad == pytest.approx(reference_logits.grad)
    assert torch.linalg.vector_norm(logits.grad) > 0


def test_single_channel_uncertainty_is_binary_entropy(target):
    trainer = target.TrainUncertaintyAwareMeanTeacher(_manager(), verbose=False)
    trainer.ema_model = _ConstantTeacher(2.0)
    inputs = torch.zeros((1, 1, 2))

    uncertainty, probability = trainer._compute_uncertainty(
        inputs, nullcontext()
    )
    p = torch.sigmoid(torch.tensor(2.0))
    expected_entropy = -(p * torch.log(p) + (1.0 - p) * torch.log1p(-p))

    assert probability == pytest.approx(torch.full_like(probability, p))
    assert uncertainty == pytest.approx(
        torch.full_like(uncertainty, expected_entropy)
    )


def test_binary_max_confidence_uses_distance_from_half_per_voxel(target):
    trainer = target.TrainUncertaintyAwareMeanTeacher(
        _manager(aggregation="max_confidence"), verbose=False
    )
    candidates = torch.tensor(
        [
            [[[[0.1, 0.7]]]],
            [[[[0.6, 0.2]]]],
        ]
    )

    actual = trainer._aggregate_max_confidence(candidates)

    assert actual.flatten().tolist() == pytest.approx([0.1, 0.2])


def test_categorical_max_confidence_is_unchanged(target):
    trainer = target.TrainUncertaintyAwareMeanTeacher(_manager(3), verbose=False)
    candidates = torch.tensor(
        [
            [[[0.8], [0.1], [0.1]]],
            [[[0.2], [0.3], [0.5]]],
        ]
    )

    actual = trainer._aggregate_max_confidence(candidates)

    assert actual.flatten().tolist() == pytest.approx([0.8, 0.1, 0.1])


def test_end_to_end_binary_consistency_is_nonzero(target):
    trainer = target.TrainUncertaintyAwareMeanTeacher(_manager(), verbose=False)
    trainer.ema_model = _ConstantTeacher(2.0)
    trainer.global_step = 1
    student = torch.cat(
        [torch.zeros((1, 1, 2)), torch.full((1, 1, 2), -2.0)], dim=0
    ).requires_grad_(True)
    outputs = {
        "_inputs": torch.zeros((2, 1, 2)),
        "surface": student,
    }
    targets = {
        "surface": torch.zeros((2, 1, 2)),
        "is_unlabeled": torch.tensor([False, True]),
    }

    total, parts = trainer._compute_train_loss(outputs, targets, {})
    total.backward()
    expected = (
        torch.sigmoid(torch.tensor(-2.0))
        - torch.sigmoid(torch.tensor(2.0))
    ) ** 2

    assert parts["consistency"] == pytest.approx(expected)
    assert torch.isfinite(student.grad).all()
    assert torch.linalg.vector_norm(student.grad[1:]) > 0
