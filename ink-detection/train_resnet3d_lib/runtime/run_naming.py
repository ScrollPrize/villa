from train_resnet3d_lib.config import normalize_cv_fold


def build_default_run_slug(*, objective, sampler, loss_mode, lr, weight_decay, cv_fold=None):
    lr_tag = f"{float(lr):.2e}"
    wd_tag = f"{float(weight_decay):.2e}"
    run_slug = f"{objective}_{sampler}_{loss_mode}_lr={lr_tag}_wd={wd_tag}"
    normalized_cv_fold = normalize_cv_fold(cv_fold)
    if normalized_cv_fold is not None:
        run_slug = f"{run_slug}_fold={normalized_cv_fold}"
    return run_slug


def build_default_run_slug_from_metadata(merged_config):
    training_cfg = merged_config["training"]
    hp_train_cfg = merged_config["training_hyperparameters"]["training"]
    return build_default_run_slug(
        objective=str(training_cfg["objective"]).lower(),
        sampler=str(training_cfg["sampler"]).lower(),
        loss_mode=str(training_cfg["loss_mode"]).lower(),
        lr=float(hp_train_cfg["lr"]),
        weight_decay=float(hp_train_cfg["weight_decay"]),
        cv_fold=training_cfg.get("cv_fold"),
    )
