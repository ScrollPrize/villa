import builtins
import torch
import pytest
from types import SimpleNamespace

from vesuvius.models.training.optimizers import create_optimizer
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


def test_create_optimizer_adamw_honors_eps_override():
    model = torch.nn.Linear(4, 2)
    optimizer = create_optimizer(
        {
            "name": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 3e-5,
            "eps": 1e-4,
        },
        model,
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["eps"] == 1e-4


def test_create_optimizer_muon_groups_params_and_honors_kwargs():
    class _TinyMuonModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 4)
            self.linear = torch.nn.Linear(4, 3)
            self.norm = torch.nn.LayerNorm(3)
            self.extra_bias = torch.nn.Parameter(torch.zeros(3))

    model = _TinyMuonModel()
    optimizer = create_optimizer(
        {
            "name": "muon",
            "learning_rate": 0.02,
            "momentum": 0.9,
            "weight_decay": 0.01,
            "weight_decouple": True,
            "nesterov": False,
            "ns_steps": 7,
            "use_adjusted_lr": True,
            "adamw_lr": 3e-4,
            "adamw_betas": (0.8, 0.9),
            "adamw_wd": 0.02,
            "adamw_eps": 1e-9,
        },
        model,
    )

    assert optimizer.__class__.__name__ == "Muon"
    muon_groups = [group for group in optimizer.param_groups if group.get("use_muon") is True]
    aux_groups = [group for group in optimizer.param_groups if group.get("use_muon") is False]
    assert len(muon_groups) == 1
    assert len(aux_groups) == 1

    muon_param_ids = {id(param) for param in muon_groups[0]["params"]}
    aux_param_ids = {id(param) for param in aux_groups[0]["params"]}
    assert id(model.linear.weight) in muon_param_ids
    assert id(model.embed.weight) in aux_param_ids
    assert id(model.linear.bias) in aux_param_ids
    assert id(model.norm.weight) in aux_param_ids
    assert id(model.extra_bias) in aux_param_ids
    assert muon_groups[0]["momentum"] == pytest.approx(0.9)
    assert muon_groups[0]["ns_steps"] == 7
    assert muon_groups[0]["nesterov"] is False
    assert muon_groups[0]["use_adjusted_lr"] is True
    assert aux_groups[0]["lr"] == pytest.approx(3e-4)
    assert aux_groups[0]["betas"] == (0.8, 0.9)
    assert aux_groups[0]["weight_decay"] == pytest.approx(0.02)
    assert aux_groups[0]["eps"] == pytest.approx(1e-9)


def test_create_optimizer_muon_missing_dependency_raises_clear_error(monkeypatch: pytest.MonkeyPatch):
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pytorch_optimizer":
            raise ModuleNotFoundError("No module named 'pytorch_optimizer'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    model = torch.nn.Linear(4, 2)
    with pytest.raises(ImportError, match="optimizer.name='muon'"):
        create_optimizer({"name": "muon", "learning_rate": 0.02}, model)
