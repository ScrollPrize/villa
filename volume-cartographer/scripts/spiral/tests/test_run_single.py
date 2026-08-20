import json
from pathlib import Path

import pytest

from runners import run_single


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


def test_parser_no_longer_accepts_overrides(tmp_path):
    with pytest.raises(SystemExit):
        run_single.build_parser().parse_args([
            "--dataset", str(tmp_path / "dataset"),
            "--ink-volume", str(tmp_path / "ink"),
            "--overrides", str(tmp_path / "config.json"),
        ])
