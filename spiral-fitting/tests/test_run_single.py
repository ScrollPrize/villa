import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from runners import run_single


def test_volume_cartographer_root_matches_top_level_project_layout():
    assert run_single.VC_ROOT == run_single.SPIRAL_DIR.parent / "volume-cartographer"
    assert run_single.DEFAULT_VC_RENDER_BIN == (
        run_single.VC_ROOT / "build" / "bin" / "vc_render_tifxyz"
    )


def _write_json(path: Path, value) -> Path:
    path.write_text(json.dumps(value))
    return path


def test_default_run_config_is_loaded_automatically():
    overrides, project, entity = run_single.load_run_config(None)

    assert overrides == {}
    assert project == "spiral_fitting"
    assert entity == "vesuvius-challenge"


def test_user_config_overlays_default_wandb_and_fit_values(tmp_path, monkeypatch):
    default = _write_json(tmp_path / "default.json", {
        "wandb_project": "default-project",
        "wandb_entity": "default-entity",
        "optimizer_learning_rate": 0.25,
        "optimizer_random_seed": 11,
    })
    user = _write_json(tmp_path / "user.json", {
        "wandb_project": "user-project",
        "wandb_entity": "user-entity",
        "optimizer_learning_rate": 0.5,
    })
    monkeypatch.setattr(run_single, "DEFAULT_RUN_CONFIG", default)

    overrides, project, entity = run_single.load_run_config(user)

    assert overrides == {
        "optimizer_learning_rate": 0.5,
        "optimizer_random_seed": 11,
    }
    assert project == "user-project"
    assert entity == "user-entity"


@pytest.mark.parametrize("contents", ["{", "null", "[]", '"config"'])
def test_user_config_must_be_a_json_object(tmp_path, contents):
    config = tmp_path / "config.json"
    config.write_text(contents)

    with pytest.raises(ValueError, match="run config"):
        run_single.load_run_config(config)


@pytest.mark.parametrize("contents", ["{", "null", "[]"])
def test_default_config_must_be_a_json_object(tmp_path, monkeypatch, contents):
    default = tmp_path / "default.json"
    default.write_text(contents)
    monkeypatch.setattr(run_single, "DEFAULT_RUN_CONFIG", default)

    with pytest.raises(ValueError, match="default run config"):
        run_single.load_run_config(None)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("wandb_project", None),
        ("wandb_project", 12),
        ("wandb_project", ""),
        ("wandb_entity", []),
        ("wandb_entity", "   "),
    ],
)
def test_wandb_values_must_be_nonempty_strings(tmp_path, key, value):
    config = _write_json(tmp_path / "config.json", {key: value})

    with pytest.raises(ValueError, match=key):
        run_single.load_run_config(config)


@pytest.mark.parametrize(
    "fit_override",
    [
        {"not_a_fit_setting": 1},
        {"optimizer_learning_rate": "fast"},
        {"optimizer_learning_rate": -1},
    ],
)
def test_fit_overrides_are_validated(tmp_path, fit_override):
    config = _write_json(tmp_path / "config.json", fit_override)

    with pytest.raises(ValueError):
        run_single.load_run_config(config)


def test_fit_environment_enables_configured_wandb_by_default(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_PROJECT", "inherited-project")
    monkeypatch.setenv("WANDB_ENTITY", "inherited-entity")
    monkeypatch.setenv("WANDB_API_KEY", "inherited-credential")

    env = run_single.fit_environment(
        {},
        tmp_path,
        None,
        wandb_project="configured-project",
        wandb_entity="configured-entity",
        wandb_enabled=True,
    )

    assert env["WANDB_MODE"] == "online"
    assert env["WANDB_PROJECT"] == "configured-project"
    assert env["WANDB_ENTITY"] == "configured-entity"
    assert env["WANDB_API_KEY"] == "inherited-credential"


def test_fit_environment_can_disable_wandb(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "online")

    env = run_single.fit_environment(
        {},
        tmp_path,
        None,
        wandb_project="configured-project",
        wandb_entity="configured-entity",
        wandb_enabled=False,
    )

    assert env["WANDB_MODE"] == "disabled"


def test_parser_accepts_config_and_no_wandb(tmp_path):
    args = run_single.build_parser().parse_args([
        "--dataset", str(tmp_path / "dataset"),
        "--ink-volume", str(tmp_path / "ink"),
        "--config", str(tmp_path / "config.json"),
        "--no-wandb",
    ])

    assert args.config == tmp_path / "config.json"
    assert args.no_wandb is True


@pytest.mark.parametrize(
    ("text", "expected"),
    [("0", (0,)), ("0,1,2,3", (0, 1, 2, 3)), (" 3, 1 ", (3, 1))],
)
def test_parse_gpu_ids(text, expected):
    assert run_single.parse_gpu_ids(text) == expected


@pytest.mark.parametrize(
    "text", ["", " ", ",", "0,", ",0", "0,,1", "-1", "1.5", "gpu0"])
def test_parse_gpu_ids_rejects_malformed_values(text):
    with pytest.raises(Exception):
        run_single.parse_gpu_ids(text)


@pytest.mark.parametrize("text", ["0,0", "01,1", "2, 2"])
def test_parse_gpu_ids_rejects_duplicates(text):
    with pytest.raises(Exception, match="duplicate"):
        run_single.parse_gpu_ids(text)


def test_parser_no_longer_accepts_overrides(tmp_path):
    with pytest.raises(SystemExit):
        run_single.build_parser().parse_args([
            "--dataset", str(tmp_path / "dataset"),
            "--ink-volume", str(tmp_path / "ink"),
            "--overrides", str(tmp_path / "config.json"),
        ])


@pytest.mark.parametrize(
    ("text", "expected"),
    [("0", [0]), ("1,2,3", [1, 2, 3]), (" 4,  8 ,15 ", [4, 8, 15])],
)
def test_parse_seeds(text, expected):
    assert run_single.parse_seeds(text) == expected


@pytest.mark.parametrize("text", ["", " ", ",", "1,", ",1", "1,,2", "-1", "1.0", "x"])
def test_parse_seeds_rejects_malformed_values(text):
    with pytest.raises(Exception, match="non-negative integers"):
        run_single.parse_seeds(text)


@pytest.mark.parametrize("text", ["1,1", "01,1", "2, 2"])
def test_parse_seeds_rejects_duplicates(text):
    with pytest.raises(Exception, match="duplicate seed"):
        run_single.parse_seeds(text)


@pytest.mark.parametrize("value", ["batch", "batch-1", "A.b_c-9"])
def test_run_id_accepts_path_safe_wandb_ids(value):
    assert run_single.run_id(value) == value


@pytest.mark.parametrize("value", ["", ".hidden", "bad/id", "bad id", "bad:tag"])
def test_run_id_rejects_unsafe_values(value):
    with pytest.raises(Exception):
        run_single.run_id(value)


def _runner_args(tmp_path, *extra):
    return run_single.build_parser().parse_args([
        "--dataset", str(tmp_path / "dataset"),
        "--ink-volume", str(tmp_path / "ink"),
        "--output", str(tmp_path / "output"),
        *extra,
    ])


def _fake_pipeline_subprocess(calls, *, fail_seed=None):
    def fake_run(command, *, check, env):
        calls.append((command, env))
        script = next(Path(part).name for part in command if part.endswith(".py"))
        if script == "fit_spiral.py":
            output = Path(env["FIT_SPIRAL_OUT_DIR"])
            seed = json.loads(env["FIT_SPIRAL_CONFIG_OVERRIDES"])[
                "optimizer_random_seed"]
            if seed == fail_seed:
                raise subprocess.CalledProcessError(1, command)
            fitted = output / "generated" / "meshes" / "fitted-result"
            fitted.mkdir(parents=True)
            Path(env["FIT_SPIRAL_METRICS_HISTORY"]).write_text(
                json.dumps({"iteration": 0, "metrics": {"loss": seed + 1}}) + "\n")
        elif script == "get_ink_metrics.py":
            fitted = Path(command[2]).parent
            seed = int(next(part.name[5:] for part in fitted.parents
                            if part.name.startswith("seed-")))
            metric_dir = fitted / "ink_metric"
            metric_dir.mkdir()
            _write_json(metric_dir / "metrics.json", {
                "summary": {"score": seed * 2, "path": "/not/numeric"}
            })
        return SimpleNamespace(returncode=0)
    return fake_run


def _pipeline_script(command):
    return next(Path(part).name for part in command if part.endswith(".py"))


def test_omitted_gpus_preserves_direct_fit_and_inherited_visibility(
    tmp_path, monkeypatch
):
    args = _runner_args(tmp_path, "--seeds", "1", "--no-wandb")
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,6")
    monkeypatch.setattr(
        run_single.subprocess, "run", _fake_pipeline_subprocess(calls))

    run_single.run(args)

    fit_cmd, fit_env = calls[0]
    assert fit_cmd[:2] == [
        run_single.sys.executable,
        str(run_single.SPIRAL_DIR / "fit_spiral.py"),
    ]
    assert all(env["CUDA_VISIBLE_DEVICES"] == "5,6" for _command, env in calls)
    assert fit_env["CUDA_VISIBLE_DEVICES"] == "5,6"


def test_one_gpu_pins_entire_pipeline_without_distributed_launch(
    tmp_path, monkeypatch
):
    args = _runner_args(
        tmp_path, "--seeds", "1", "--no-wandb", "--gpus", "3")
    calls = []
    monkeypatch.setattr(
        run_single.subprocess, "run", _fake_pipeline_subprocess(calls))

    run_single.run(args)

    assert calls[0][0][:2] == [
        run_single.sys.executable, str(run_single.SPIRAL_DIR / "fit_spiral.py")]
    assert all(env["CUDA_VISIBLE_DEVICES"] == "3" for _command, env in calls)


def test_multiple_gpus_launch_distributed_fit_and_pin_entire_pipeline(
    tmp_path, monkeypatch
):
    args = _runner_args(
        tmp_path, "--seeds", "1,2", "--no-wandb", "--gpus", "0,2,3")
    calls = []
    monkeypatch.setattr(
        run_single.subprocess, "run", _fake_pipeline_subprocess(calls))

    run_single.run(args)

    expected_prefix = [
        run_single.sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=3",
        str(run_single.SPIRAL_DIR / "fit_spiral.py"),
    ]
    fit_commands = [command for command, _env in calls
                    if _pipeline_script(command) == "fit_spiral.py"]
    assert len(fit_commands) == 2
    assert all(command[:6] == expected_prefix for command in fit_commands)
    assert all(env["CUDA_VISIBLE_DEVICES"] == "0,2,3" for _command, env in calls)


def test_seeded_run_is_sequential_and_overrides_config_seed(tmp_path, monkeypatch):
    config = _write_json(tmp_path / "config.json", {"optimizer_random_seed": 99})
    args = _runner_args(
        tmp_path, "--seeds", "3,1", "--run-id", "batch",
        "--config", str(config), "--no-wandb")
    calls = []
    monkeypatch.setattr(run_single.subprocess, "run", _fake_pipeline_subprocess(calls))

    run_single.run(args)

    assert [_pipeline_script(command) for command, _env in calls] == [
        "fit_spiral.py", "render_ink.py", "get_ink_metrics.py",
        "fit_spiral.py", "render_ink.py", "get_ink_metrics.py",
    ]
    fit_envs = [env for command, env in calls
                if _pipeline_script(command) == "fit_spiral.py"]
    assert [json.loads(env["FIT_SPIRAL_CONFIG_OVERRIDES"])["optimizer_random_seed"]
            for env in fit_envs] == [3, 1]
    assert [env["FIT_SPIRAL_OUT_DIR"] for env in fit_envs] == [
        str((tmp_path / "output" / "seed-3").resolve()),
        str((tmp_path / "output" / "seed-1").resolve()),
    ]
    assert [(env["WANDB_RUN_ID"], env["WANDB_NAME"], env["WANDB_RUN_GROUP"])
            for env in fit_envs] == [
        ("batch_seed_3", "batch_seed_3", "batch"),
        ("batch_seed_1", "batch_seed_1", "batch"),
    ]
    aggregate = json.loads(
        (tmp_path / "output" / "aggregate_metrics.json").read_text())
    assert aggregate["run_id"] == "batch"
    assert aggregate["seeds"] == [3, 1]
    assert aggregate["training"][0]["metrics"]["loss"] == {
        "mean": 3.0, "stddev": 1.0, "count": 2}
    assert aggregate["final"]["score"] == {
        "mean": 4.0, "stddev": 2.0, "count": 2}


def test_seeded_run_fails_fast_without_aggregate(tmp_path, monkeypatch):
    args = _runner_args(
        tmp_path, "--seeds", "1,2,3", "--run-id", "batch", "--no-wandb")
    calls = []
    monkeypatch.setattr(
        run_single.subprocess, "run", _fake_pipeline_subprocess(calls, fail_seed=2))

    with pytest.raises(subprocess.CalledProcessError):
        run_single.run(args)

    fit_seeds = [json.loads(env["FIT_SPIRAL_CONFIG_OVERRIDES"])["optimizer_random_seed"]
                 for command, env in calls
                 if _pipeline_script(command) == "fit_spiral.py"]
    assert fit_seeds == [1, 2]
    assert not (tmp_path / "output" / "aggregate_metrics.json").exists()


def test_one_seed_has_no_aggregate_and_logs_final(tmp_path, monkeypatch):
    args = _runner_args(tmp_path, "--seeds", "7", "--run-id", "batch")
    calls = []
    logged = []
    monkeypatch.setattr(run_single.subprocess, "run", _fake_pipeline_subprocess(calls))
    monkeypatch.setattr(
        run_single, "log_seed_final_metrics",
        lambda summary, **kwargs: logged.append((summary, kwargs)))

    run_single.run(args)

    assert logged[0][0]["score"] == 14
    assert logged[0][1]["seed_run_id"] == "batch_seed_7"
    assert not (tmp_path / "output" / "aggregate_metrics.json").exists()


def test_no_wandb_suppresses_all_runner_uploads(tmp_path, monkeypatch):
    args = _runner_args(
        tmp_path, "--seeds", "1,2", "--run-id", "batch", "--no-wandb")
    monkeypatch.setattr(run_single.subprocess, "run", _fake_pipeline_subprocess([]))
    monkeypatch.setattr(
        run_single, "log_seed_final_metrics",
        lambda *args, **kwargs: pytest.fail("seed upload was not suppressed"))
    monkeypatch.setattr(
        run_single, "log_aggregate_metrics",
        lambda *args, **kwargs: pytest.fail("aggregate upload was not suppressed"))

    run_single.run(args)

    assert (tmp_path / "output" / "aggregate_metrics.json").exists()


def test_generated_run_id_is_used_for_seed_runs(tmp_path, monkeypatch):
    args = _runner_args(tmp_path, "--seeds", "5", "--no-wandb")
    calls = []
    monkeypatch.setattr(run_single.subprocess, "run", _fake_pipeline_subprocess(calls))
    monkeypatch.setattr(run_single.uuid, "uuid4", lambda: SimpleNamespace(hex="abc12345more"))

    run_single.run(args)

    fit_env = calls[0][1]
    assert fit_env["WANDB_RUN_ID"] == "abc12345_seed_5"
    assert fit_env["WANDB_RUN_GROUP"] == "abc12345"


def test_aggregate_metrics_aligns_steps_and_excludes_non_numeric_values():
    histories = [
        [{"iteration": 0, "metrics": {"loss": 1, "partial": 4, "flag": True}},
         {"iteration": 200, "metrics": {"loss": 5}}],
        [{"iteration": 0, "metrics": {"loss": 3, "label": "x"}},
         {"iteration": 400, "metrics": {"loss": 9}}],
    ]
    final_summaries = [
        {"score": 2, "only_one": 8, "path": "/tmp", "flag": False},
        {"score": 6, "items": [1, 2]},
    ]

    training, final = run_single.aggregate_metrics(histories, final_summaries)

    assert [record["iteration"] for record in training] == [0, 200, 400]
    assert training[0]["metrics"]["loss"] == {
        "mean": 2.0, "stddev": 1.0, "count": 2}
    assert training[0]["metrics"]["partial"]["count"] == 1
    assert "flag" not in training[0]["metrics"]
    assert final["score"] == {"mean": 4.0, "stddev": 2.0, "count": 2}
    assert final["only_one"]["count"] == 1
    assert set(final) == {"score", "only_one"}


def test_aggregate_wandb_logs_only_complete_means(monkeypatch):
    fake_run = SimpleNamespace(log_calls=[])
    fake_run.log = lambda payload, **kwargs: fake_run.log_calls.append((payload, kwargs))
    fake_run.finish = lambda: None
    init_calls = []
    monkeypatch.setattr(
        run_single, "_wandb_init",
        lambda **kwargs: init_calls.append(kwargs) or fake_run)

    run_single.log_aggregate_metrics(
        [{"iteration": 200, "metrics": {
            "loss": {"mean": 2.0, "stddev": 1.0, "count": 2},
            "partial": {"mean": 4.0, "stddev": 0.0, "count": 1},
        }}],
        {"score": {"mean": 6.0, "stddev": 2.0, "count": 2}},
        seed_count=2, project="project", entity="entity",
        aggregate_run_id="batch_aggregate", group="batch")

    assert init_calls[0]["run_id"] == "batch_aggregate"
    assert init_calls[0]["name"] == "batch_aggregate"
    assert init_calls[0]["group"] == "batch"
    assert fake_run.log_calls == [
        ({"loss": 2.0}, {"step": 200}),
        ({"final/score": 6.0}, {}),
    ]
