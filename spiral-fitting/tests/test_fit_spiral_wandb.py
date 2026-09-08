from types import SimpleNamespace

import fit_spiral


def test_batch_wandb_init_does_not_require_a_group():
    kwargs = fit_spiral._wandb_init_kwargs(
        {"optimizer_random_seed": 1},
        "online",
        {
            "FIT_SPIRAL_BATCH_RUN": "1",
            "WANDB_RUN_ID": "batch_seed_1",
            "WANDB_NAME": "batch_seed_1",
            "WANDB_PROJECT": "project",
            "WANDB_ENTITY": "entity",
        },
    )

    assert kwargs["id"] == "batch_seed_1"
    assert kwargs["name"] == "batch_seed_1"
    assert "group" not in kwargs


def test_batch_wandb_init_uses_an_explicit_group():
    kwargs = fit_spiral._wandb_init_kwargs(
        {},
        "online",
        {
            "FIT_SPIRAL_BATCH_RUN": "1",
            "WANDB_RUN_ID": "batch_seed_1",
            "WANDB_NAME": "batch_seed_1",
            "WANDB_RUN_GROUP": "experiment",
        },
    )

    assert kwargs["group"] == "experiment"


def test_resumed_batch_wandb_run_allows_existing_remote_run():
    kwargs = fit_spiral._wandb_init_kwargs(
        {},
        "online",
        {
            "FIT_SPIRAL_BATCH_RUN": "1",
            "FIT_SPIRAL_WANDB_RESUME": "1",
            "WANDB_RUN_ID": "batch_seed_1",
            "WANDB_NAME": "batch_seed_1",
        },
    )

    assert kwargs["resume"] == "allow"


def test_resumed_fit_reuses_exact_run_directory(tmp_path):
    run_dir = tmp_path / "2026-08-21_original-run"
    context = SimpleNamespace(run_dir=str(run_dir))

    resolved = fit_spiral.FitContext.resolve_output_path(context)

    assert resolved == str(run_dir)
    assert run_dir.is_dir()


def test_finish_wandb_run_releases_active_run(monkeypatch):
    calls = []
    monkeypatch.setattr(
        fit_spiral, "wandb",
        SimpleNamespace(run=object(), finish=lambda: calls.append("finish")))

    assert fit_spiral._finish_wandb_run() is True
    assert calls == ["finish"]


def test_finish_wandb_failure_is_nonfatal(monkeypatch, capsys):
    def fail():
        raise RuntimeError("service stopped")

    monkeypatch.setattr(
        fit_spiral, "wandb", SimpleNamespace(run=object(), finish=fail))

    assert fit_spiral._finish_wandb_run() is False
    assert "WARNING: could not finish W&B run cleanly" in capsys.readouterr().err
