import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import zipfile

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


def test_parser_accepts_config_and_wandb_options(tmp_path):
    args = run_single.build_parser().parse_args([
        "--dataset", str(tmp_path / "dataset"),
        "--ink-volume", str(tmp_path / "ink"),
        "--config", str(tmp_path / "config.json"),
        "--wandb-group", "experiment-1",
        "--overwrite",
        "--no-wandb",
    ])

    assert args.config == tmp_path / "config.json"
    assert args.wandb_group == "experiment-1"
    assert args.overwrite is True
    assert args.no_wandb is True


def test_overwrite_output_removes_only_the_selected_directory(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "stale.txt").write_text("stale")
    sibling = tmp_path / "keep.txt"
    sibling.write_text("keep")

    run_single.overwrite_output(output)

    assert not output.exists()
    assert sibling.read_text() == "keep"


def test_overwrite_output_refuses_to_delete_a_directory_containing_an_input(
    tmp_path,
):
    output = tmp_path / "output"
    dataset = output / "dataset"
    dataset.mkdir(parents=True)

    with pytest.raises(ValueError, match="containing an input"):
        run_single.overwrite_output(output, protected_paths=(dataset,))

    assert dataset.is_dir()


def test_overwrite_and_resume_are_mutually_exclusive(tmp_path):
    args = _runner_args(tmp_path, "--overwrite", "--resume")
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")

    with pytest.raises(ValueError, match="cannot be combined"):
        run_single.run(args)

    assert marker.read_text() == "keep"


def test_fit_environment_sets_an_explicit_wandb_group(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_RUN_GROUP", raising=False)

    ungrouped = run_single.fit_environment(
        {}, tmp_path, None, wandb_project="project", wandb_entity="entity",
        wandb_enabled=True)
    grouped = run_single.fit_environment(
        {}, tmp_path, None, wandb_project="project", wandb_entity="entity",
        wandb_enabled=True, wandb_group="explicit-group")

    assert "WANDB_RUN_GROUP" not in ungrouped
    assert grouped["WANDB_RUN_GROUP"] == "explicit-group"


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


def test_render_artifact_requires_at_least_one_strip_image(tmp_path):
    ink = tmp_path / "ink"
    ink.mkdir()

    with pytest.raises(RuntimeError, match="without ink strip images"):
        run_single._require_ink_output(ink)

    (ink / "renderer.log").write_text("not an image")
    with pytest.raises(RuntimeError, match="without ink strip images"):
        run_single._require_ink_output(ink)

    (ink / "w001-002_flat.000.jpg").touch()
    run_single._require_ink_output(ink)


def test_completed_render_state_rejects_an_empty_ink_directory(tmp_path):
    fitted = tmp_path / "fitted"
    fitted.mkdir()
    ink = tmp_path / "ink"
    ink.mkdir()
    state = {"runs": {"single": {
        "fit": {"status": "complete", "fitted_output": "fitted"},
        "render": {"status": "complete", "ink_output": "ink"},
        "metrics": {"status": "pending"},
    }}}

    with pytest.raises(RuntimeError, match="render artifact is invalid"):
        run_single._validate_completed_stages(tmp_path, state)


def test_interrupted_fit_resumes_checkpoint_in_original_run_directory(
    tmp_path, monkeypatch
):
    args = _runner_args(
        tmp_path, "--seeds", "1", "--run-id", "batch", "--resume",
        "--no-wandb")
    fit_environments = []

    def fake_run(command, *, check, env):
        script = next(Path(part).name for part in command if part.endswith(".py"))
        if script == "fit_spiral.py":
            fit_environments.append(env)
            output = Path(env["FIT_SPIRAL_OUT_DIR"])
            run_dir = output / "original-dated-run"
            run_dir.mkdir(parents=True, exist_ok=True)
            history = Path(env["FIT_SPIRAL_METRICS_HISTORY"])
            if len(fit_environments) == 1:
                history.write_text(
                    json.dumps({"iteration": 0, "metrics": {"loss": 2}}) + "\n")
                with zipfile.ZipFile(
                    run_dir / "checkpoint_fitted.ckpt", "w"
                ) as archive:
                    archive.writestr("archive/data.pkl", b"checkpoint")
                raise KeyboardInterrupt
            fitted = run_dir / "meshes" / "fitted-result"
            fitted.mkdir(parents=True)
            with history.open("a") as stream:
                stream.write(
                    json.dumps({"iteration": 1000, "metrics": {"loss": 1}})
                    + "\n")
        elif script == "render_ink.py":
            ink = Path(command[2]) / "ink"
            ink.mkdir()
            (ink / "w001-002_flat.000.jpg").touch()
        elif script == "get_ink_metrics.py":
            fitted = Path(command[2]).parent
            metric_dir = fitted / "ink_metric"
            metric_dir.mkdir()
            _write_json(metric_dir / "metrics.json", {"summary": {"score": 3}})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_single.subprocess, "run", fake_run)

    with pytest.raises(KeyboardInterrupt):
        run_single.run(args)
    state_path = tmp_path / "output" / run_single.STATE_FILENAME
    assert json.loads(state_path.read_text())["runs"]["1"]["fit"][
        "status"] == "interrupted"

    run_single.run(args)

    resumed = fit_environments[1]
    expected_run_dir = (
        tmp_path / "output" / "seed-1" / "original-dated-run").resolve()
    assert resumed["FIT_SPIRAL_RESUME_PATH"] == str(
        expected_run_dir / "checkpoint_fitted.ckpt")
    assert resumed["FIT_SPIRAL_RUN_DIR"] == str(expected_run_dir)
    assert resumed["FIT_SPIRAL_WANDB_RESUME"] == "1"
    stages = json.loads(state_path.read_text())["runs"]["1"]
    assert {name: stage["status"] for name, stage in stages.items()} == {
        "fit": "complete", "render": "complete", "metrics": "complete"}


def test_interrupted_fit_rejects_a_corrupt_checkpoint(tmp_path):
    output = tmp_path / "seed-1"
    run_dir = output / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint_fitted.ckpt").write_bytes(b"truncated")

    with pytest.raises(RuntimeError, match="incomplete or corrupt"):
        run_single._recover_interrupted_fit(output)


def _write_pending_resume_state(tmp_path, *, gpus="0"):
    args = _runner_args(
        tmp_path, "--seeds", "1", "--run-id", "batch", "--resume",
        "--no-wandb", "--gpus", gpus)
    invocation = run_single._resume_invocation(
        args, {}, "project", "entity")
    output = args.output.resolve()
    output.mkdir()
    state = run_single._new_resume_state(invocation)
    state_path = output / run_single.STATE_FILENAME
    run_single._atomic_write_json(state_path, state)
    return args, invocation, state, state_path


def test_resume_can_change_gpu_count_while_fits_are_pending(tmp_path):
    _args, _invocation, _state, state_path = _write_pending_resume_state(
        tmp_path, gpus="0")
    resumed_args = _runner_args(
        tmp_path, "--seeds", "1", "--run-id", "batch", "--resume",
        "--no-wandb", "--gpus", "0,1")
    resumed_invocation = run_single._resume_invocation(
        resumed_args, {}, "project", "entity")

    state, _ = run_single._load_or_create_state(
        resumed_args.output.resolve(), resumed_invocation)

    assert state["gpu_count"] == 2
    assert state["invocation"]["gpu_count"] == 2
    assert json.loads(state_path.read_text())["gpu_count"] == 2


@pytest.mark.parametrize("fit_status", ["running", "interrupted"])
def test_resume_cannot_change_gpu_count_during_interrupted_fit(
    tmp_path, fit_status
):
    args, _invocation, state, state_path = _write_pending_resume_state(
        tmp_path, gpus="0")
    state["runs"]["1"]["fit"]["status"] = fit_status
    run_single._atomic_write_json(state_path, state)
    resumed_args = _runner_args(
        tmp_path, "--seeds", "1", "--run-id", "batch", "--resume",
        "--no-wandb", "--gpus", "0,1")
    resumed_invocation = run_single._resume_invocation(
        resumed_args, {}, "project", "entity")

    with pytest.raises(RuntimeError, match="cannot change GPU count"):
        run_single._load_or_create_state(
            args.output.resolve(), resumed_invocation)


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
        "--config", str(config), "--wandb-group", "experiment", "--no-wandb")
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
        ("batch_seed_3", "batch_seed_3", "experiment"),
        ("batch_seed_1", "batch_seed_1", "experiment"),
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


def test_generated_run_id_does_not_implicitly_group_seed_runs(
    tmp_path, monkeypatch
):
    args = _runner_args(tmp_path, "--seeds", "5", "--no-wandb")
    calls = []
    monkeypatch.delenv("WANDB_RUN_GROUP", raising=False)
    monkeypatch.setattr(run_single.subprocess, "run", _fake_pipeline_subprocess(calls))
    monkeypatch.setattr(run_single.uuid, "uuid4", lambda: SimpleNamespace(hex="abc12345more"))

    run_single.run(args)

    fit_env = calls[0][1]
    assert fit_env["WANDB_RUN_ID"] == "abc12345_seed_5"
    assert "WANDB_RUN_GROUP" not in fit_env


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


def test_seed_final_wandb_retries_until_training_run_is_released(monkeypatch):
    fake_run = SimpleNamespace(log_calls=[], finish_calls=0)
    fake_run.log = lambda payload: fake_run.log_calls.append(payload)

    def finish():
        fake_run.finish_calls += 1

    fake_run.finish = finish
    attempts = []

    def fake_init(**kwargs):
        attempts.append(kwargs)
        if len(attempts) < 3:
            raise RuntimeError(f"run ID {kwargs['run_id']} is in use")
        return fake_run

    sleeps = []
    monkeypatch.setattr(run_single, "_wandb_init", fake_init)
    monkeypatch.setattr(run_single.time, "sleep", sleeps.append)

    uploaded = run_single.log_seed_final_metrics(
        {"score": 4, "path": "/not/numeric"}, project="project",
        entity="entity", seed_run_id="batch_seed_1")

    assert uploaded is True
    assert len(attempts) == 3
    assert sleeps == list(run_single._WANDB_IN_USE_RETRY_DELAYS[:2])
    assert fake_run.log_calls == [{"final/score": 4}]
    assert fake_run.finish_calls == 1


def test_seed_final_wandb_failure_does_not_fail_pipeline(monkeypatch, capsys):
    monkeypatch.setattr(
        run_single, "_wandb_init",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("network down")))

    uploaded = run_single.log_seed_final_metrics(
        {"score": 4}, project="project", entity="entity",
        seed_run_id="batch_seed_1")

    assert uploaded is False
    assert "WARNING: could not upload final W&B metrics" in capsys.readouterr().err


def test_successful_seed_final_upload_is_recorded_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        run_single, "log_seed_final_metrics",
        lambda summary, **kwargs: calls.append((summary, kwargs)) or True)
    state = {"runs": {"1": {"metrics": {"status": "complete"}}}}
    state_path = tmp_path / "state.json"

    for _ in range(2):
        run_single._log_seed_final_metrics_once(
            {"score": 4}, metrics_stage=state["runs"]["1"]["metrics"],
            state=state, state_path=state_path, project="project",
            entity="entity", seed_run_id="batch_seed_1")

    assert len(calls) == 1
    assert state["runs"]["1"]["metrics"]["wandb_final_logged"] is True
    assert json.loads(state_path.read_text()) == state
