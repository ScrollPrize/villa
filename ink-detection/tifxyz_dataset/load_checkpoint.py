import os

import torch


def load_training_checkpoint(ckpt_path):
    ckpt_path = os.path.expanduser(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu")


def restore_training_state(model, optimizer, lr_scheduler, checkpoint, ckpt_path, load_weights_only=False):
    model.load_state_dict(checkpoint["model"])

    if load_weights_only:
        return 0

    if "optimizer" not in checkpoint:
        raise KeyError(f"Checkpoint missing optimizer state: {ckpt_path}")
    if "lr_scheduler" not in checkpoint:
        raise KeyError(f"Checkpoint missing lr_scheduler state: {ckpt_path}")
    if "step" not in checkpoint:
        raise KeyError(f"Checkpoint missing step: {ckpt_path}")

    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    return int(checkpoint["step"]) + 1
