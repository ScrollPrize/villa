"""Extracted handoff regressions and real-data snapshot/resume validation."""
import random
from collections import OrderedDict
from copy import deepcopy
from dataclasses import FrozenInstanceError
import os

import numpy as np
import torch
import pytest

from vesuvius.neural_tracing.fiber_trace_3d.training_state import (
    TrainingBoundary, capture_handoff, config_fingerprint, cpu_clone,
    model_optimizer_payload, replica_signature, restore_handoff, restore_model_optimizer,
)
from tests.neural_tracing.test_sample_pipeline import _real_paris4_loader
from tests.neural_tracing.test_effective_batch_accumulation import _pointwise_fiber_model
from vesuvius.neural_tracing.fiber_trace_3d.targets import materialize_targets
from vesuvius.neural_tracing.fiber_trace_3d.train import compute_losses, _save_snapshot, _load_snapshot


def test_handoff_preserves_optimizer_rng_and_logical_progress():
    torch.manual_seed(23)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    source = torch.randn(4, 3)
    model(source).square().mean().backward()
    optimizer.step()
    progress = {"step": 9, "sample_frontier": 1152, "best_metric": 0.95,
                "test": {"step": 9, "next_logical_group": 2}, "last_logged_step": 9}
    state = capture_handoff(model=model, optimizer=optimizer, progress=progress)
    signature = replica_signature(state)
    draws = (random.random(), np.random.rand(), torch.rand(3))
    restored = torch.nn.Linear(3, 2)
    new_optimizer = torch.optim.AdamW(restored.parameters(), lr=1)
    restored_progress = restore_handoff(state, model=restored, optimizer=new_optimizer)
    assert restored_progress == progress
    assert new_optimizer.param_groups[0]["lr"] == 0.003
    assert random.random() == draws[0]
    assert np.random.rand() == draws[1]
    torch.testing.assert_close(torch.rand(3), draws[2], rtol=0, atol=0)
    for current_model, current_optimizer in ((model, optimizer), (restored, new_optimizer)):
        current_optimizer.zero_grad()
        current_model(source).square().mean().backward()
        current_optimizer.step()
    for actual, expected in zip(restored.parameters(), model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The saved CPU payload must not alias live model/optimizer storage.
    assert replica_signature(state) == signature
    new_state = capture_handoff(model=restored, optimizer=new_optimizer, progress=progress)
    assert replica_signature(new_state) != signature


def test_replica_signature_includes_optimizer_not_rank_rng():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    state = capture_handoff(model=model, optimizer=optimizer, progress={"step": 0})
    original = replica_signature(state)
    state["rng"]["torch"] = torch.zeros_like(state["rng"]["torch"])
    assert replica_signature(state) == original
    state["optimizer"]["param_groups"][0]["lr"] *= 2
    assert replica_signature(state) != original


def _real_training(tmp_path, device):
    loader = _real_paris4_loader(tmp_path)
    batch = materialize_targets(loader.load_batch(0, sample_mode="flat").to(device), loader.config)
    model = _pointwise_fiber_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.125)
    return batch, model, optimizer


def _step(model, optimizer, batch, *, dtype=None, scaler=None):
    optimizer.zero_grad(set_to_none=True)
    device = batch.volume.device
    # All three host RNGs and, on CUDA, that device's RNG influence the update.
    jitter = .001 * (random.random() + float(np.random.rand()) + float(torch.rand(())))
    jitter += .001 * torch.rand((), device=device)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
        output = model(batch.volume + jitter)
        loss = compute_losses(output, batch, direction_weight=2, presence_weight=3)["total"]
    assert torch.isfinite(loss)
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return loss.detach()


def _rng_draws(device):
    return random.random(), np.random.rand(), torch.rand(3), torch.rand(3, device=device)


def _assert_draws_equal(actual, expected):
    assert actual[:2] == expected[:2]
    for a, b in zip(actual[2:], expected[2:], strict=True):
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)


def _real_handoff(tmp_path, device, dtype=None, scaled=False, restore_device=None):
    random.seed(43)
    np.random.seed(43)
    torch.manual_seed(43)
    batch, model, optimizer = _real_training(tmp_path, device)
    scaler = torch.amp.GradScaler(device.type, enabled=scaled, init_scale=4., growth_interval=1)
    for _ in range(2):
        _step(model, optimizer, batch, dtype=dtype, scaler=scaler)
    progress = {"step": 2, "sample_frontier": 256, "last_logged_step": 2,
                "test": {"step": 2, "next_logical_group": 1}, "best_metric": .95}
    gpu = device if device.type == "cuda" else None
    payload = capture_handoff(model=model, optimizer=optimizer, progress=progress,
                              grad_scaler=scaler, cuda_device=gpu)
    signature = replica_signature(payload)
    expected_draws = _rng_draws(device)
    expected_loss = _step(model, optimizer, batch, dtype=dtype, scaler=scaler)
    expected_state = cpu_clone(model_optimizer_payload(model, optimizer, scaler))
    restored_device = device if restore_device is None else restore_device
    restored_model = _pointwise_fiber_model(restored_device)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1.)
    restored_scaler = torch.amp.GradScaler(restored_device.type, enabled=scaled,
                                          init_scale=1., growth_interval=100)
    actual_progress = restore_handoff(payload, model=restored_model, optimizer=restored_optimizer,
        grad_scaler=restored_scaler,
        cuda_device=restored_device if restored_device.type == "cuda" else None)
    assert actual_progress == progress
    actual_progress["test"]["next_logical_group"] += 1
    assert payload["progress"] == progress
    _assert_draws_equal(_rng_draws(restored_device), expected_draws)
    actual_loss = _step(restored_model, restored_optimizer, batch.to(restored_device),
                        dtype=dtype, scaler=restored_scaler)
    torch.testing.assert_close(actual_loss.cpu(), expected_loss.cpu(), rtol=0, atol=0)
    assert replica_signature(cpu_clone(model_optimizer_payload(
        restored_model, restored_optimizer, restored_scaler))) == replica_signature(expected_state)
    assert replica_signature(payload) == signature, "handoff must not alias live optimizer/model storage"


@pytest.mark.parametrize("dtype,scaled", [(None, False), (torch.bfloat16, False), (None, True)])
def test_real_paris4_handoff_continuation_is_exact(tmp_path, dtype, scaled):
    _real_handoff(tmp_path, torch.device("cpu"), dtype=dtype, scaled=scaled)


@pytest.mark.skipif(os.environ.get("FIBER_ASYNC_CUDA_TEST") != "1", reason="requires manager GPU reservation")
@pytest.mark.parametrize("dtype,scaled", [(None, False), (torch.bfloat16, False), (torch.float16, True)])
@pytest.mark.parametrize("new_device", [0, 1])
def test_cuda_real_paris4_handoff_restores_optimizer_rng_and_scaler(tmp_path, dtype, scaled, new_device):
    assert os.environ.get("FIBER_COOPERATIVE_TEST_GPUS"), "manager reservation required"
    assert torch.cuda.device_count() >= 2
    _real_handoff(tmp_path, torch.device("cuda:0"), dtype=dtype, scaled=scaled,
                   restore_device=torch.device("cuda", new_device))


@pytest.mark.parametrize("scaled", [False, True])
def test_real_snapshot_schema_and_continuation_are_unchanged(tmp_path, scaled):
    batch, model, optimizer = _real_training(tmp_path, torch.device("cpu"))
    scaler = torch.amp.GradScaler("cpu", enabled=scaled, init_scale=4., growth_interval=1)
    _step(model, optimizer, batch, scaler=scaler)
    path = tmp_path / "current.pt"
    config = {"training": {"learning_rate": .003, "run_path": tmp_path}}
    _save_snapshot(path, model=torch.nn.DataParallel(model), optimizer=optimizer,
        grad_scaler=scaler, step=7, config=config, metric=.25, metric_name="test_loss")
    saved = torch.load(path, weights_only=True)
    expected_keys = {"model", "optimizer", "step", "config", "metric", "metric_name"}
    if scaled:
        expected_keys.add("grad_scaler")
    assert set(saved) == expected_keys
    assert not any(name.startswith("module.") for name in saved["model"])
    assert saved["config"]["training"]["run_path"] == str(tmp_path)
    assert saved["step"] == 7 and saved["metric"] == .25 and saved["metric_name"] == "test_loss"
    # Disk resume intentionally retains its existing no-RNG-restoration policy.
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    expected_loss = _step(model, optimizer, batch, scaler=scaler)
    resumed = _pointwise_fiber_model("cpu")
    opt = torch.optim.AdamW(resumed.parameters(), lr=1.)
    amp = torch.amp.GradScaler("cpu", enabled=scaled, init_scale=1.)
    before_load = torch.get_rng_state().clone()
    assert _load_snapshot(path, model=resumed, optimizer=opt, grad_scaler=amp) == 7
    assert torch.equal(torch.get_rng_state(), before_load)
    assert opt.param_groups[0]["lr"] == .003
    random.setstate(rng[0])
    np.random.set_state(rng[1])
    torch.set_rng_state(rng[2])
    actual_loss = _step(resumed, opt, batch, scaler=amp)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    assert replica_signature(cpu_clone(model_optimizer_payload(resumed, opt, amp))) == replica_signature(
        cpu_clone(model_optimizer_payload(model, optimizer, scaler)))


def test_raw_weights_and_explicit_optimizer_overrides_keep_existing_contract(tmp_path):
    batch, model, optimizer = _real_training(tmp_path, torch.device("cpu"))
    _step(model, optimizer, batch)
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    resumed = _pointwise_fiber_model("cpu")
    opt = torch.optim.AdamW(resumed.parameters(), lr=.75)
    assert _load_snapshot(path, model=resumed, optimizer=opt,
        optimizer_hparams={"lr": .1}) == 0
    assert opt.param_groups[0]["lr"] == .75, "no optimizer payload means no override"
    for p, q in zip(model.parameters(), resumed.parameters(), strict=True):
        assert torch.equal(p, q)
    _save_snapshot(path, model=model, optimizer=optimizer, step=9, config={}, metric=None)
    saved = torch.load(path, weights_only=True)
    assert _load_snapshot(path, model=resumed, optimizer=opt,
        optimizer_hparams={"lr": .01, "weight_decay": .5}) == 9
    assert opt.param_groups[0]["lr"] == .01 and opt.param_groups[0]["weight_decay"] == .5
    assert replica_signature(dict(opt.state_dict()["state"])) == replica_signature(saved["optimizer"]["state"])
    malformed = {"model": {"unexpected.weight": torch.ones(1)}}
    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        restore_model_optimizer(malformed, model=resumed)


def test_cpu_state_helpers_never_probe_gpu_runtime(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("CPU handoff must not probe or initialize CUDA")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    for name in ("is_available", "get_rng_state", "set_rng_state", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, fail)
    state = capture_handoff(model=model, optimizer=optimizer, progress={"step": 0})
    assert restore_handoff(state, model=model, optimizer=optimizer) == {"step": 0}


def test_cpu_clone_keeps_numpy_containers_and_state_dict_metadata_owned():
    model = torch.nn.BatchNorm1d(3)
    source = {"weights": model.state_dict(), "list": [np.arange(4), torch.arange(4)],
              "tuple": (np.float32(2), b"bytes", None)}
    cloned = cpu_clone(source)
    assert isinstance(cloned["weights"], OrderedDict)
    assert cloned["weights"]._metadata == source["weights"]._metadata
    cloned["weights"]._metadata[""]["version"] = -1
    assert source["weights"]._metadata[""]["version"] != -1
    cloned["list"][0][:] = -1
    cloned["list"][1].zero_()
    assert np.array_equal(source["list"][0], np.arange(4))
    assert torch.equal(source["list"][1], torch.arange(4))
    assert cloned["tuple"] == source["tuple"]
    with pytest.raises(TypeError, match="unsupported handoff"):
        cpu_clone(object())


def test_fingerprint_ignores_only_cooperative_config_and_does_not_mutate_input():
    raw = {"datasets": [{"volume": "fixture"}],
           "training": {"learning_rate": .01, "cooperative": {"priority": 1}}}
    original = deepcopy(raw)
    changed = deepcopy(raw)
    changed["training"]["cooperative"] = {"priority": 100, "max_gpus": 1}
    assert config_fingerprint(raw) == config_fingerprint(changed)
    assert raw == original
    changed["training"]["learning_rate"] = .001
    assert config_fingerprint(raw) != config_fingerprint(changed)
    assert config_fingerprint(dict(reversed(list(raw.items())))) == config_fingerprint(raw)
    with pytest.raises(ValueError):
        config_fingerprint({"training": {"learning_rate": float("nan")}})


def test_boundary_signature_and_invalid_handoff_contracts():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    state = capture_handoff(model=model, optimizer=optimizer, progress={"step": 2})
    boundary = TrainingBoundary(2, False, lambda: state, phase="test")
    assert boundary.capture() is state and boundary.phase == "test"
    with pytest.raises(FrozenInstanceError):
        boundary.step = 3
    signature = replica_signature(state)
    changed = cpu_clone(state)
    changed["progress"]["step"] += 1
    assert replica_signature(changed) != signature
    with pytest.raises(ValueError, match="dense CPU"):
        replica_signature({"model": torch.ones(2, device="meta")})
    with pytest.raises(TypeError, match="unsupported replica"):
        replica_signature({"unsupported": object()})
    for missing in ("rng", "progress"):
        incomplete = {k: v for k, v in state.items() if k != missing}
        with pytest.raises(ValueError, match="incomplete in-RAM"):
            restore_handoff(incomplete, model=model, optimizer=optimizer)
    with pytest.raises(ValueError, match="no CUDA RNG"):
        restore_handoff(state, model=model, optimizer=optimizer, cuda_device=torch.device("cuda:0"))
    state["rng"]["cuda"] = torch.ones(1, dtype=torch.uint8)
    with pytest.raises(ValueError, match="explicit worker device"):
        restore_handoff(state, model=model, optimizer=optimizer)
