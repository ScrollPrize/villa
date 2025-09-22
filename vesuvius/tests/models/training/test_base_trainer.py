import torch
from types import SimpleNamespace

from vesuvius.models.training.train import BaseTrainer


def _make_mgr():
    return SimpleNamespace(
        gpu_ids=None,
        use_ddp=False,
        targets={"ink": {"weight": 1.0}},
        model_name="test",
    )


class _DummyTrainer(BaseTrainer):
    def __init__(self):
        super().__init__(mgr=_make_mgr(), verbose=False)


def test_base_trainer_prepare_hooks_noop():
    trainer = _DummyTrainer()
    sample = {"image": torch.zeros(1)}
    assert trainer._prepare_sample(sample, is_training=True) is sample
    assert trainer._prepare_batch(sample, is_training=True) is sample


def test_base_trainer_loss_helpers():
    trainer = _DummyTrainer()
    trainer.mgr.targets = {"ink": {"weight": 1.0}}

    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([0.0, 2.0])
    loss_fn = torch.nn.MSELoss()

    value = trainer._compute_loss_value(
        loss_fn,
        pred,
        target,
        target_name="ink",
        targets_dict={"ink": target},
        outputs={"ink": pred},
    )
    assert torch.isclose(value, torch.tensor(0.5))


def test_base_trainer_should_include_target_in_loss():
    trainer = _DummyTrainer()
    assert trainer._should_include_target_in_loss("ink") is True
    assert trainer._should_include_target_in_loss("ink_skel") is False
    assert trainer._should_include_target_in_loss("is_unlabeled") is False
