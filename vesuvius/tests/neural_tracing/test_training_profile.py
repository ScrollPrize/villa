import json
import pytest
import torch

from vesuvius.neural_tracing.fiber_trace_3d.training_profile import (
    TrainingProfile, compile_training_model, loader_profile_enabled,
    replay_batch_enabled, replay_cpu_batch_enabled,
)


def test_loader_profile_requires_explicit_timing_output():
    assert not loader_profile_enabled({})
    assert loader_profile_enabled({"profile_loader": True, "profile_enabled": True})
    with pytest.raises(ValueError):
        loader_profile_enabled({"profile_loader": "false"})
    with pytest.raises(ValueError):
        loader_profile_enabled({"profile_loader": True})


def test_replay_batch_requires_explicit_timing_output():
    assert not replay_batch_enabled({})
    assert replay_batch_enabled({"profile_replay_batch": True, "profile_enabled": True})
    with pytest.raises(ValueError):
        replay_batch_enabled({"profile_replay_batch": "true", "profile_enabled": True})
    with pytest.raises(ValueError):
        replay_batch_enabled({"profile_replay_batch": True})


def test_replay_cpu_batch_requires_explicit_timing_output():
    assert not replay_cpu_batch_enabled({})
    assert replay_cpu_batch_enabled(
        {"profile_replay_cpu_batch": True, "profile_enabled": True}
    )
    with pytest.raises(ValueError):
        replay_cpu_batch_enabled(
            {"profile_replay_cpu_batch": "true", "profile_enabled": True}
        )
    with pytest.raises(ValueError):
        replay_cpu_batch_enabled({"profile_replay_cpu_batch": True})


def test_disabled_creates_nothing(tmp_path):
    profile = TrainingProfile({}, tmp_path, 0, torch.device("cpu"))
    profile.begin(1)
    with profile.stage("forward"):
        pass
    profile.end()
    profile.close()
    assert list(tmp_path.iterdir()) == []


def test_bounded_cpu_trace(tmp_path):
    profile = TrainingProfile({"profile_enabled": True, "profile_trace_start": 2,
                               "profile_trace_steps": 1}, tmp_path, 1, torch.device("cpu"))
    for step in range(1, 4):
        profile.begin(step)
        with profile.stage("forward"):
            torch.ones(2).sum()
        profile.end()
    profile.close()
    rows = [json.loads(line) for line in (tmp_path / "timings_rank1.jsonl").read_text().splitlines()]
    assert [r["trace_active"] for r in rows] == [False, True, False]
    assert all(r["forward_wall_ms"] >= 0 for r in rows)
    assert (tmp_path / "trace_rank1.json").exists()


def test_compile_default_and_checkpoint_identity(monkeypatch):
    model = torch.nn.Linear(2, 2)
    assert compile_training_model(model, "off") is model
    def compile_stub(wrapped, *, mode):
        assert wrapped is model and mode == "default"
        return object()
    monkeypatch.setattr(torch, "compile", compile_stub)
    compiled = compile_training_model(model, "default")
    assert compiled is not model
    assert list(model.state_dict()) == ["weight", "bias"]
    with pytest.raises(ValueError):
        compile_training_model(model, "typo")


def test_compiled_wrapper_preserves_model_semantics():
    from vesuvius.neural_tracing.fiber_trace_3d.train import _unwrap_model
    model = torch.nn.Linear(2, 2)
    model.conditioned_decoder_enabled = True
    compiled = torch.compile(model, backend="eager")
    assert _unwrap_model(compiled) is model
    assert _unwrap_model(compiled).conditioned_decoder_enabled


def test_summary_keeps_loader_stalls_excludes_warmup_and_test(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.profile_experiment import summarize_experiment
    run = tmp_path / "s1a_test"
    run.mkdir()
    keys = ("iteration_wall_ms", "wait_ms", "to_device_ms", "target_ms", "fw_bw_step_ms",
            "forward_loss_wall_ms", "backward_wall_ms", "optimizer_wall_ms", "scalar_reduce_sync_wall_ms")
    rows = []
    for step, ms in [(26, 9000), (31, 500), (32, 8500), (100, 10000)]:
        rows.append(dict(step=step, trace_active=False, **{key: ms for key in keys}))
    (run / "timings_rank0.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    result = summarize_experiment(tmp_path)
    stats = result["ranks"]["timings_rank0"]["steady"]["iteration_wall_ms"]
    assert stats["count"] == 2
    assert stats["mean"] == 4500
