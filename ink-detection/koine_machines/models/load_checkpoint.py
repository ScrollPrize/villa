import os

import torch


def _load_model_state_with_ddp_compat(model, model_state):
    try:
        model.load_state_dict(model_state)
        return
    except RuntimeError as exc:
        if not isinstance(model_state, dict):
            raise exc

        stripped_state = {
            (key[len("module."):] if str(key).startswith("module.") else key): value
            for key, value in model_state.items()
        }
        if stripped_state != model_state:
            try:
                model.load_state_dict(stripped_state)
                return
            except RuntimeError:
                pass

        prefixed_state = {
            (key if str(key).startswith("module.") else f"module.{key}"): value
            for key, value in model_state.items()
        }
        if prefixed_state != model_state:
            model.load_state_dict(prefixed_state)
            return

        raise exc


def resolve_training_checkpoint_path(ckpt_path, config_path=None):
    if ckpt_path in (None, ""):
        return None

    ckpt_path = os.path.expanduser(str(ckpt_path))
    if config_path and not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(config_path)), ckpt_path)
    return ckpt_path


def load_training_checkpoint(ckpt_path):
    ckpt_path = resolve_training_checkpoint_path(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu")


def load_training_checkpoint_from_config(config, config_path):
    checkpoint_path = resolve_training_checkpoint_path(config.get("checkpoint"), config_path)
    checkpoint = None
    weights_only = bool(config.get("weights_only", False))
    if checkpoint_path:
        checkpoint = load_training_checkpoint(checkpoint_path)
    return checkpoint_path, checkpoint, weights_only


def restore_training_state(
    model,
    optimizer,
    lr_scheduler,
    checkpoint,
    ckpt_path,
    load_weights_only=False,
    ema_model=None,
):
    model_state = checkpoint.get("model")
    if model_state is None:
        raise ValueError(f"Checkpoint '{ckpt_path}' is missing 'model'")
    _load_model_state_with_ddp_compat(model, model_state)

    if load_weights_only:
        return 0, 0

    if "optimizer" not in checkpoint:
        raise KeyError(f"Checkpoint missing optimizer state: {ckpt_path}")
    if "lr_scheduler" not in checkpoint:
        raise KeyError(f"Checkpoint missing lr_scheduler state: {ckpt_path}")
    if "step" not in checkpoint:
        raise KeyError(f"Checkpoint missing step: {ckpt_path}")

    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    optimizer_step = 0
    if ema_model is not None:
        ema_model_state = checkpoint.get("ema_model")
        if ema_model_state is not None:
            ema_model.load_state_dict(ema_model_state)
            optimizer_step = int(checkpoint.get("ema_optimizer_step", 0))

    return int(checkpoint["step"]) + 1, optimizer_step
