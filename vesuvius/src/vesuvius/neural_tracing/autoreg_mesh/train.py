from __future__ import annotations

import json
import random
from pathlib import Path

import click
import numpy as np
import torch
from torch.utils.data import DataLoader

from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.neural_tracing.autoreg_mesh.config import load_autoreg_mesh_config, validate_autoreg_mesh_config
from vesuvius.neural_tracing.autoreg_mesh.dataset import AutoregMeshDataset, autoreg_mesh_collate
from vesuvius.neural_tracing.autoreg_mesh.losses import compute_autoreg_mesh_losses
from vesuvius.neural_tracing.autoreg_mesh.model import AutoregMeshModel


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def run_autoreg_mesh_training(
    config: dict,
    *,
    dataset=None,
    model: AutoregMeshModel | None = None,
    device: str | torch.device | None = None,
    max_steps: int | None = None,
) -> dict:
    cfg = validate_autoreg_mesh_config(config)
    _seed_everything(int(cfg["seed"]))

    if dataset is None:
        dataset = AutoregMeshDataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        collate_fn=autoreg_mesh_collate,
    )
    if model is None:
        model = AutoregMeshModel(cfg)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.train()

    optimizer = create_optimizer(dict(cfg["optimizer"]), model)
    scheduler = None
    scheduler_name = str(cfg.get("scheduler", "constant")).lower()
    if scheduler_name != "constant":
        from vesuvius.models.training.lr_schedulers import get_scheduler

        scheduler = get_scheduler(
            scheduler_type=scheduler_name,
            optimizer=optimizer,
            initial_lr=float(cfg["optimizer"]["learning_rate"]),
            max_steps=int(max_steps or cfg["num_steps"]),
            **dict(cfg.get("scheduler_kwargs") or {}),
        )

    num_steps = int(max_steps or cfg["num_steps"])
    history = []
    iterator = iter(dataloader)

    for step_idx in range(num_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        if "prompt_tokens" in batch:
            batch["prompt_tokens"] = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch["prompt_tokens"].items()
            }

        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss_dict = compute_autoreg_mesh_losses(
            outputs,
            batch,
            offset_num_bins=tuple(int(v) for v in cfg["offset_num_bins"]),
            occupancy_loss_weight=float(cfg.get("occupancy_loss_weight", 0.0)),
        )
        loss = loss_dict["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Encountered non-finite training loss at step {step_idx}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg["grad_clip"]))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        history.append({key: float(value.detach().cpu().item()) for key, value in loss_dict.items()})

    return {
        "model": model,
        "optimizer": optimizer,
        "history": history,
        "final_metrics": history[-1] if history else {},
    }


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def train(config_path: str) -> None:
    cfg = load_autoreg_mesh_config(Path(config_path))
    result = run_autoreg_mesh_training(cfg)
    print(json.dumps(result["final_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    train()
