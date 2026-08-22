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
