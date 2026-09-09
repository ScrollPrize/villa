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


def test_summary_reports_per_step_ddp_critical_path(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.profile_experiment import summarize_experiment

    run = tmp_path / "s1a_test"
    run.mkdir()
    keys = (
        "iteration_wall_ms", "wait_ms", "to_device_ms",
        "value_augmentation_cuda_ms", "target_ms", "fw_bw_step_ms",
    )
    for rank, values in enumerate(((10.0, 40.0), (20.0, 30.0))):
        rows = [
            dict(
                step=step,
                trace_active=False,
                **{key: value for key in keys},
            )
            for step, value in zip((31, 32), values, strict=True)
        ]
        (run / f"timings_rank{rank}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows)
        )

    result = summarize_experiment(tmp_path)

    stats = result["critical_path"]["iteration_wall_ms"]
    assert stats["count"] == 2
    assert stats["mean"] == 30.0
    assert stats["min"] == 20.0
    assert stats["max"] == 40.0


def test_lr_range_analysis_uses_rank_zero_global_metrics(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.profile_experiment import analyze_lr_range

    run = tmp_path / "s1a_test"
    run.mkdir()
    rank_zero = [
        {
            "step": index + 1,
            "lr_range_index": index,
            "learning_rate": 10.0 ** (-3.0 + index / 10.0),
            "loss_total": 1.0 if index < 21 else 5.0,
            "loss_direction": 0.1,
            "loss_presence": 0.9,
            "grad_norm_l2": float(index + 1),
        }
        for index in range(30)
    ]
    (run / "timings_rank0.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rank_zero)
    )
    (run / "timings_rank1.jsonl").write_text(
        json.dumps({"lr_range_index": 0, "loss_total": 999.0})
    )

    result = analyze_lr_range(tmp_path, beta=0.0, divergence_factor=4.0)

    assert result["range_rows"] == 30
    assert result["usable_rows"] == 21
    assert result["divergence_lr"] == pytest.approx(
        rank_zero[21]["learning_rate"]
    )
    assert result["grad_norm_l2"]["count"] == 21
    assert result["grad_norm_l2"]["median"] == pytest.approx(11.0)
    assert (tmp_path / "lr_range_summary.json").exists()


def test_tuning_suite_helpers_derive_candidates_and_parse_test_rows(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.profile_experiment import (
        _extract_test_metrics,
        _round_threshold_up,
        _tuning_lr_candidates,
    )

    analysis = {"minimum_loss_lr": 0.081, "divergence_lr": 1.5}
    assert _tuning_lr_candidates(analysis, baseline_lr=0.01) == pytest.approx(
        [0.01, 0.027, 0.081, 0.15]
    )
    assert _round_threshold_up(2.7802) == pytest.approx(2.8)
    (tmp_path / "run.log").write_text(
        "noise\n"
        "test step=0 loss_total=2.5 loss_direction=0.1 loss_presence=1.5 "
        "angle_mean_deg=60 branch0=1.000\n"
        "test step=200 loss_total=1.25 loss_direction=0.05 loss_presence=0.75 "
        "angle_mean_deg=30 branch0=1.000\n"
    )
    rows = _extract_test_metrics(tmp_path)
    assert [row["step"] for row in rows] == [0, 200]
    assert rows[-1]["loss_total"] == pytest.approx(1.25)


def test_tuning_report_keeps_choice_manual(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.profile_experiment import (
        _write_tuning_suite_report,
    )

    report = {
        "range_experiment": "/range",
        "range_analysis": {
            "minimum_loss_lr": 0.08,
            "divergence_lr": 1.5,
        },
        "derived_clip_threshold": 2.8,
        "fixed_trials": [
            {
                "learning_rate": 0.01,
                "clip_max_norm": None,
                "experiment": "/fixed",
                "final_test": {"step": 200, "loss_total": 1.25},
            }
        ],
    }
    _write_tuning_suite_report(tmp_path, report)
    markdown = (tmp_path / "tuning_report.md").read_text()
    assert "No final setting is selected automatically" in markdown
    assert "| 0.01 | disabled | 200 | 1.250000 | `/fixed` |" in markdown
    assert (tmp_path / "tuning_report.json").exists()
