import random
from dataclasses import replace

import numpy as np
import torch
import pytest

from vesuvius.neural_tracing.fiber_trace_3d.training_state import (
    capture_handoff, replica_signature, restore_handoff,
)


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


@pytest.mark.parametrize("channels", [7, 14])
@pytest.mark.parametrize("physical_size", [1, 2])
def test_test_rows_keep_logical_normalization_with_physical_microbatches(channels, physical_size):
    from tests.neural_tracing.test_fiber_trace_3d import _loader_with_mapping
    from vesuvius.neural_tracing.fiber_trace_3d import train

    loader = _loader_with_mapping({"patch_shape_zyx": [8, 8, 8], "batch_size": 3,
                                  "augment_enabled": False})
    torch.manual_seed(81)
    model = torch.nn.Conv3d(1, channels, 1)
    options = dict(device=torch.device("cpu"), start_sample_index=0,
                   sample_count=5, direction_weight=1.0, presence_weight=1.0)
    if channels != 7:
        # Target-only normalization intentionally disabled these modes before
        # the EMA work; assert the explicit guard rather than old support.
        with pytest.raises(ValueError, match="temporarily unsupported"):
            train._evaluate_dense_loss_rows(model, loader, **options)
        return
    expected = train._evaluate_dense_loss_rows(model, loader, **options)
    actual = train._evaluate_dense_loss_rows(model, loader, microbatch_size=physical_size, **options)
    assert [index for index, _ in actual] == [0, 1]
    for (_, row), (_, reference) in zip(actual, expected, strict=True):
        assert row == pytest.approx(reference, rel=2e-5, abs=2e-6)


@pytest.mark.parametrize("interrupt_test", [False, True])
def test_shared_training_loop_resume_preserves_updates_with_changed_microbatch(monkeypatch, tmp_path, interrupt_test):
    from tests.neural_tracing.test_fiber_trace_3d import _loader_with_mapping, _load_materialized_batch
    from vesuvius.neural_tracing.fiber_trace_3d import train

    loader = _loader_with_mapping({"patch_shape_zyx": [8, 8, 8], "batch_size": 2, "augment_enabled": False})
    config = {"training": {"device": "cpu", "bounded_gradient_accumulation": True,
        "ddp_broadcast_buffers": False, "sync_batchnorm": False,
        "optimizer_global_batch_size": 5, "max_steps": 4, "initialization_seed": 17,
        "run_path": str(tmp_path), "run_datestr": "test", "tensorboard_enabled": False,
        "test_interval": 100, "checkpoint_interval": 100, "sample_vis_interval": 0,
        "sample_pipeline_workers": 2, "sample_pipeline_slots": 8,
        "sample_pipeline_initial_in_flight": 4}}
    monkeypatch.setattr(train, "_load_raw_config", lambda _: config)
    monkeypatch.setattr(train, "load_config", lambda _: loader.config)
    monkeypatch.setattr(train, "build_fiber_trace_3d_model", lambda _: torch.nn.Conv3d(1, 7, 1))
    if interrupt_test:
        config["test_datasets"] = [{"name": "synthetic"}]
        config["training"].update(test_interval=2, checkpoint_interval=2,
            kept_snapshot_interval=0, test_control_points=5, test_sample_vis_interval=0,
            test_trace2cp_enabled=False)
        monkeypatch.setattr(train, "config_from_mapping", lambda *a, **k: replace(loader.config, batch_size=3))
        monkeypatch.setattr(train, "_make_batch_dataloader", lambda *a, **k: None)
        monkeypatch.setattr(train, "_save_snapshot", lambda *a, **k: None)

    def make_loader(cfg):
        object.__setattr__(cfg, "_array_records", loader.config._array_records)
        return type(loader)(cfg)

    monkeypatch.setattr(train, "FiberTrace3DLoader", make_loader)

    class CpuPipeline:
        def __init__(self, **kwargs):
            self.mapping = kwargs["index_map"]
            self.pattern = kwargs["batch_sizes"]
            self.index = self.ordinal = 0
            self.loader = make_loader(replace(loader.config, batch_size=kwargs["batch_size"]))

        def __iter__(self):
            return self

        def __next__(self):
            take = self.pattern[self.ordinal % len(self.pattern)]
            batch = _load_materialized_batch(self.loader, self.mapping(self.index))
            self.index += take
            self.ordinal += 1
            return train._slice_batch(batch, 0, take)

        def release_current(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(train, "FiberTrace3DTrainingBatchPipeline", CpuPipeline)
    reference = train.training_units(tmp_path / "unused.json")
    try:
        for boundary in reference:
            if boundary.done:
                expected = boundary.capture()
    finally:
        reference.close()
    interrupted = train.training_units(tmp_path / "unused.json",
                                      physical_microbatch_sizes=(2,) if interrupt_test else None)
    try:
        for boundary in interrupted:
            if boundary.step == 2:
                assert boundary.phase == ("test" if interrupt_test else "update")
                state = boundary.capture()
                break
    finally:
        interrupted.close()
    assert state["progress"]["step"] == 2
    resumed = train.training_units(tmp_path / "unused.json", handoff=state, physical_microbatch_sizes=(1,))
    try:
        steps = []
        for boundary in resumed:
            steps.append(boundary.step)
            if boundary.done:
                actual = boundary.capture()
    finally:
        resumed.close()
    if interrupt_test:
        assert state["progress"]["pending_test"]["step"] == 2
        assert len(state["progress"]["pending_test"]["rows"]) == 1
        assert steps == [2, 2, 3, 4, 4, 4]
    else:
        assert steps == [2, 3, 4]
    assert actual["progress"].pop("best_metric") == pytest.approx(expected["progress"].pop("best_metric"), rel=2e-5)
    assert actual["progress"] == expected["progress"]
    for name in expected["model"]:
        torch.testing.assert_close(actual["model"][name], expected["model"][name], rtol=1e-5, atol=1e-6)
