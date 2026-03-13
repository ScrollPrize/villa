#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import DatasetConfig, InkCropDataset, build_train_augmentations
from model import InkPatchEmbedder, create_frozen_dino_backbone, normalize_for_backbone


DEFAULT_CONFIG: dict[str, Any] = {
    "output_dir": Path("embedding_runs/latest"),
    "backbone_name": "vit_small_patch14_dinov2.lvd142m",
    "backbone_checkpoint": None,
    "crop_size": 224,
    "downsample_factor": 1,
    "embedding_dim": 96,
    "hidden_dim": 256,
    "dropout": 0.1,
    "epochs": 50,
    "train_samples": 4096,
    "test_samples": 1024,
    "batch_size": 64,
    "num_workers": 4,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "test_fraction": 0.15,
    "min_foreground_fraction": 0.04,
    "foreground_threshold": 0.2,
    "max_crop_attempts": 24,
    "sim_coeff": 25.0,
    "std_coeff": 25.0,
    "cov_coeff": 1.0,
    "log_every": 20,
    "media_every": 200,
    "seed": 1337,
    "device": None,
    "amp": True,
    "wandb_project": "ink-embedding",
    "wandb_entity": None,
    "wandb_run_name": None,
    "wandb_tags": "",
    "disable_wandb": False,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def make_grid_image(images: torch.Tensor, max_images: int = 8) -> np.ndarray:
    images = images[:max_images].detach().cpu().clamp(0.0, 1.0)
    if images.ndim != 4:
        raise ValueError(f"Expected BCHW tensor for logging, got shape {tuple(images.shape)!r}")
    tiles = []
    for image in images:
        tile = (image[0].numpy() * 255.0).round().astype(np.uint8)
        tiles.append(tile)
    return np.concatenate(tiles, axis=1)


def safe_json_config(config: dict[str, Any]) -> dict[str, Any]:
    def _convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        return value

    return _convert(config)


def _require_path(value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise click.ClickException(f"Config field '{field_name}' must be a non-empty path string")
    return Path(value)


def _optional_path(value: Any, field_name: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise click.ClickException(f"Config field '{field_name}' must be a path string or null")
    return Path(value)


def _require_int(value: Any, field_name: str, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise click.ClickException(f"Config field '{field_name}' must be an integer")
    if min_value is not None and value < min_value:
        raise click.ClickException(f"Config field '{field_name}' must be >= {min_value}")
    return value


def _require_float(
    value: Any,
    field_name: str,
    min_value: float | None = None,
    max_value: float | None = None,
    open_min: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise click.ClickException(f"Config field '{field_name}' must be a number")
    converted = float(value)
    if min_value is not None:
        if open_min and converted <= min_value:
            raise click.ClickException(f"Config field '{field_name}' must be > {min_value}")
        if not open_min and converted < min_value:
            raise click.ClickException(f"Config field '{field_name}' must be >= {min_value}")
    if max_value is not None and converted > max_value:
        raise click.ClickException(f"Config field '{field_name}' must be <= {max_value}")
    return converted


def _optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise click.ClickException(f"Config field '{field_name}' must be a string or null")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise click.ClickException(f"Config field '{field_name}' must be a boolean")
    return value


def load_config(config_path: Path) -> dict[str, Any]:
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Failed to parse JSON config {config_path}: {exc}") from exc

    if not isinstance(raw_config, dict):
        raise click.ClickException(f"Config file {config_path} must contain a top-level JSON object")

    unknown_fields = sorted(set(raw_config) - (set(DEFAULT_CONFIG) | {"image_dir"}))
    if unknown_fields:
        raise click.ClickException(f"Unknown config field(s): {', '.join(unknown_fields)}")

    merged = {**DEFAULT_CONFIG, **raw_config}

    image_dir = _require_path(merged.get("image_dir"), "image_dir")
    output_dir = _require_path(merged["output_dir"], "output_dir")
    backbone_checkpoint = _optional_path(merged["backbone_checkpoint"], "backbone_checkpoint")
    if not image_dir.exists() or not image_dir.is_dir():
        raise click.ClickException(f"Configured image_dir does not exist or is not a directory: {image_dir}")
    if backbone_checkpoint is not None and (not backbone_checkpoint.exists() or not backbone_checkpoint.is_file()):
        raise click.ClickException(f"Configured backbone_checkpoint does not exist or is not a file: {backbone_checkpoint}")

    wandb_tags_value = merged["wandb_tags"]
    if isinstance(wandb_tags_value, list):
        if any(not isinstance(tag, str) for tag in wandb_tags_value):
            raise click.ClickException("Config field 'wandb_tags' must be a string or a list of strings")
        wandb_tags = ",".join(tag for tag in wandb_tags_value if tag)
    elif isinstance(wandb_tags_value, str):
        wandb_tags = wandb_tags_value
    else:
        raise click.ClickException("Config field 'wandb_tags' must be a string or a list of strings")

    return {
        "image_dir": image_dir,
        "output_dir": output_dir,
        "backbone_name": _optional_str(merged["backbone_name"], "backbone_name") or DEFAULT_CONFIG["backbone_name"],
        "backbone_checkpoint": backbone_checkpoint,
        "crop_size": _require_int(merged["crop_size"], "crop_size", min_value=32),
        "downsample_factor": _require_int(merged["downsample_factor"], "downsample_factor", min_value=1),
        "embedding_dim": _require_int(merged["embedding_dim"], "embedding_dim", min_value=8),
        "hidden_dim": _require_int(merged["hidden_dim"], "hidden_dim", min_value=16),
        "dropout": _require_float(merged["dropout"], "dropout", min_value=0.0, max_value=1.0),
        "epochs": _require_int(merged["epochs"], "epochs", min_value=1),
        "train_samples": _require_int(merged["train_samples"], "train_samples", min_value=1),
        "test_samples": _require_int(merged["test_samples"], "test_samples", min_value=1),
        "batch_size": _require_int(merged["batch_size"], "batch_size", min_value=2),
        "num_workers": _require_int(merged["num_workers"], "num_workers", min_value=0),
        "learning_rate": _require_float(merged["learning_rate"], "learning_rate", min_value=0.0, open_min=True),
        "weight_decay": _require_float(merged["weight_decay"], "weight_decay", min_value=0.0),
        "grad_clip": _require_float(merged["grad_clip"], "grad_clip", min_value=0.0),
        "test_fraction": _require_float(merged["test_fraction"], "test_fraction", min_value=0.05, max_value=0.5),
        "min_foreground_fraction": _require_float(
            merged["min_foreground_fraction"], "min_foreground_fraction", min_value=0.0, max_value=1.0
        ),
        "foreground_threshold": _require_float(
            merged["foreground_threshold"], "foreground_threshold", min_value=0.0, max_value=1.0
        ),
        "max_crop_attempts": _require_int(merged["max_crop_attempts"], "max_crop_attempts", min_value=1),
        "sim_coeff": _require_float(merged["sim_coeff"], "sim_coeff", min_value=0.0),
        "std_coeff": _require_float(merged["std_coeff"], "std_coeff", min_value=0.0),
        "cov_coeff": _require_float(merged["cov_coeff"], "cov_coeff", min_value=0.0),
        "log_every": _require_int(merged["log_every"], "log_every", min_value=1),
        "media_every": _require_int(merged["media_every"], "media_every", min_value=1),
        "seed": _require_int(merged["seed"], "seed"),
        "device": _optional_str(merged["device"], "device"),
        "amp": _require_bool(merged["amp"], "amp"),
        "wandb_project": _optional_str(merged["wandb_project"], "wandb_project") or DEFAULT_CONFIG["wandb_project"],
        "wandb_entity": _optional_str(merged["wandb_entity"], "wandb_entity"),
        "wandb_run_name": _optional_str(merged["wandb_run_name"], "wandb_run_name"),
        "wandb_tags": wandb_tags,
        "disable_wandb": _require_bool(merged["disable_wandb"], "disable_wandb"),
    }


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError(f"Expected square matrix, got {tuple(x.shape)!r}")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_coeff: float,
    std_coeff: float,
    cov_coeff: float,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    repr_loss = F.mse_loss(z1, z2)

    centered_z1 = z1 - z1.mean(dim=0)
    centered_z2 = z2 - z2.mean(dim=0)

    std_z1 = torch.sqrt(centered_z1.var(dim=0, unbiased=False) + eps)
    std_z2 = torch.sqrt(centered_z2.var(dim=0, unbiased=False) + eps)
    std_loss = 0.5 * (F.relu(1.0 - std_z1).mean() + F.relu(1.0 - std_z2).mean())

    cov_z1 = (centered_z1.T @ centered_z1) / max(1, z1.shape[0] - 1)
    cov_z2 = (centered_z2.T @ centered_z2) / max(1, z2.shape[0] - 1)
    cov_loss = off_diagonal(cov_z1).pow(2).mean() + off_diagonal(cov_z2).pow(2).mean()

    total = sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss
    diagnostics = {
        "repr_loss": repr_loss.detach(),
        "std_loss": std_loss.detach(),
        "cov_loss": cov_loss.detach(),
        "std_mean": 0.5 * (std_z1.mean().detach() + std_z2.mean().detach()),
        "std_min": torch.minimum(std_z1.min().detach(), std_z2.min().detach()),
        "embed_norm": 0.5 * (z1.norm(dim=1).mean().detach() + z2.norm(dim=1).mean().detach()),
        "pair_cosine": F.cosine_similarity(z1, z2, dim=1).mean().detach(),
    }
    return total, diagnostics


def compute_grad_norm(parameters: Any) -> float:
    total_sq = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = float(parameter.grad.detach().norm(2).item())
        total_sq += grad_norm * grad_norm
    return math.sqrt(total_sq)


def evaluate(
    model: InkPatchEmbedder,
    dataloader: DataLoader,
    device: torch.device,
    sim_coeff: float,
    std_coeff: float,
    cov_coeff: float,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    model.eval()
    metrics = {
        "loss": 0.0,
        "repr_loss": 0.0,
        "std_loss": 0.0,
        "cov_loss": 0.0,
        "std_mean": 0.0,
        "std_min": 0.0,
        "embed_norm": 0.0,
        "pair_cosine": 0.0,
        "foreground_fraction": 0.0,
    }
    count = 0

    autocast_kwargs = {"device_type": device.type, "enabled": amp_dtype is not None}
    if amp_dtype is not None and device.type == "cuda":
        autocast_kwargs["dtype"] = amp_dtype

    with torch.no_grad():
        for batch in dataloader:
            view1 = batch["view1"].to(device, non_blocking=True)
            view2 = batch["view2"].to(device, non_blocking=True)
            foreground_fraction = batch["foreground_fraction"].to(device, non_blocking=True)

            with torch.autocast(**autocast_kwargs):
                z1, _ = model(normalize_for_backbone(view1))
                z2, _ = model(normalize_for_backbone(view2))
                loss, diagnostics = vicreg_loss(z1.float(), z2.float(), sim_coeff, std_coeff, cov_coeff)

            metrics["loss"] += float(loss.item())
            metrics["foreground_fraction"] += float(foreground_fraction.mean().item())
            for key in ("repr_loss", "std_loss", "cov_loss", "std_mean", "std_min", "embed_norm", "pair_cosine"):
                metrics[key] += float(diagnostics[key].item())
            count += 1

    if count == 0:
        raise ValueError("Evaluation dataloader produced no batches")
    return {key: value / count for key, value in metrics.items()}


def save_checkpoint(
    output_dir: Path,
    model: InkPatchEmbedder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    best_val_loss: float,
    config: dict[str, Any],
    name: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / name
    state = {
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
        "model": model.head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": safe_json_config(config),
    }
    torch.save(state, checkpoint_path)
    return checkpoint_path


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(config_path: Path) -> None:
    loaded_config = load_config(config_path)
    image_dir = loaded_config["image_dir"]
    output_dir = loaded_config["output_dir"]
    backbone_name = loaded_config["backbone_name"]
    backbone_checkpoint = loaded_config["backbone_checkpoint"]
    crop_size = loaded_config["crop_size"]
    downsample_factor = loaded_config["downsample_factor"]
    embedding_dim = loaded_config["embedding_dim"]
    hidden_dim = loaded_config["hidden_dim"]
    dropout = loaded_config["dropout"]
    epochs = loaded_config["epochs"]
    train_samples = loaded_config["train_samples"]
    test_samples = loaded_config["test_samples"]
    batch_size = loaded_config["batch_size"]
    num_workers = loaded_config["num_workers"]
    learning_rate = loaded_config["learning_rate"]
    weight_decay = loaded_config["weight_decay"]
    grad_clip = loaded_config["grad_clip"]
    test_fraction = loaded_config["test_fraction"]
    min_foreground_fraction = loaded_config["min_foreground_fraction"]
    foreground_threshold = loaded_config["foreground_threshold"]
    max_crop_attempts = loaded_config["max_crop_attempts"]
    sim_coeff = loaded_config["sim_coeff"]
    std_coeff = loaded_config["std_coeff"]
    cov_coeff = loaded_config["cov_coeff"]
    log_every = loaded_config["log_every"]
    media_every = loaded_config["media_every"]
    seed = loaded_config["seed"]
    device = loaded_config["device"]
    amp = loaded_config["amp"]
    wandb_project = loaded_config["wandb_project"]
    wandb_entity = loaded_config["wandb_entity"]
    wandb_run_name = loaded_config["wandb_run_name"]
    wandb_tags = loaded_config["wandb_tags"]
    disable_wandb = loaded_config["disable_wandb"]

    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_image_dir = image_dir.resolve()

    resolved_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if amp and resolved_device.type == "cuda" else None

    train_cfg = DatasetConfig(
        image_dir=resolved_image_dir,
        split="train",
        seed=seed,
        crop_size=crop_size,
        downsample_factor=downsample_factor,
        samples_per_epoch=train_samples,
        min_foreground_fraction=min_foreground_fraction,
        max_crop_attempts=max_crop_attempts,
        test_fraction=test_fraction,
        foreground_threshold=foreground_threshold,
        cache_images=True,
    )
    test_cfg = DatasetConfig(
        image_dir=resolved_image_dir,
        split="test",
        seed=seed,
        crop_size=crop_size,
        downsample_factor=downsample_factor,
        samples_per_epoch=test_samples,
        min_foreground_fraction=min_foreground_fraction,
        max_crop_attempts=max_crop_attempts,
        test_fraction=test_fraction,
        foreground_threshold=foreground_threshold,
        cache_images=True,
    )

    train_dataset = InkCropDataset(train_cfg, build_train_augmentations(crop_size))
    test_dataset = InkCropDataset(test_cfg, build_train_augmentations(crop_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=resolved_device.type == "cuda",
        drop_last=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=resolved_device.type == "cuda",
        drop_last=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
    )

    backbone, backbone_dim = create_frozen_dino_backbone(backbone_name, backbone_checkpoint, crop_size, resolved_device)
    model = InkPatchEmbedder(backbone, backbone_dim, embedding_dim, hidden_dim, dropout).to(resolved_device)

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs * steps_per_epoch))

    config = {
        "image_dir": resolved_image_dir,
        "output_dir": output_dir.resolve(),
        "backbone_name": backbone_name,
        "backbone_checkpoint": backbone_checkpoint,
        "crop_size": crop_size,
        "downsample_factor": downsample_factor,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "epochs": epochs,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "test_fraction": test_fraction,
        "min_foreground_fraction": min_foreground_fraction,
        "foreground_threshold": foreground_threshold,
        "max_crop_attempts": max_crop_attempts,
        "sim_coeff": sim_coeff,
        "std_coeff": std_coeff,
        "cov_coeff": cov_coeff,
        "seed": seed,
        "device": str(resolved_device),
        "amp": amp,
        "train_split_files": [str(path) for path in train_dataset.paths],
        "test_split_files": [str(path) for path in test_dataset.paths],
        "train_dataset": asdict(train_cfg),
        "test_dataset": asdict(test_cfg),
        "backbone_dim": backbone_dim,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(safe_json_config(config), f, indent=2)

    run = None
    if not disable_wandb:
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            tags=[tag for tag in wandb_tags.split(",") if tag],
            config=safe_json_config(config),
        )

    best_val_loss = float("inf")
    global_step = 0
    progress = tqdm(range(epochs), dynamic_ncols=True)

    autocast_kwargs = {"device_type": resolved_device.type, "enabled": amp_dtype is not None}
    if amp_dtype is not None and resolved_device.type == "cuda":
        autocast_kwargs["dtype"] = amp_dtype

    for epoch in progress:
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            view1 = batch["view1"].to(resolved_device, non_blocking=True)
            view2 = batch["view2"].to(resolved_device, non_blocking=True)
            base = batch["base"].to(resolved_device, non_blocking=True)
            foreground_fraction = batch["foreground_fraction"].to(resolved_device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(**autocast_kwargs):
                z1, f1 = model(normalize_for_backbone(view1))
                z2, f2 = model(normalize_for_backbone(view2))
                loss, diagnostics = vicreg_loss(z1.float(), z2.float(), sim_coeff, std_coeff, cov_coeff)

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=grad_clip)
            grad_norm = compute_grad_norm(model.head.parameters())
            optimizer.step()
            scheduler.step()

            if global_step % log_every == 0:
                log_data = {
                    "train/loss": float(loss.item()),
                    "train/repr_loss": float(diagnostics["repr_loss"].item()),
                    "train/std_loss": float(diagnostics["std_loss"].item()),
                    "train/cov_loss": float(diagnostics["cov_loss"].item()),
                    "train/std_mean": float(diagnostics["std_mean"].item()),
                    "train/std_min": float(diagnostics["std_min"].item()),
                    "train/embed_norm": float(diagnostics["embed_norm"].item()),
                    "train/pair_cosine": float(diagnostics["pair_cosine"].item()),
                    "train/backbone_norm": float(0.5 * (f1.norm(dim=1).mean().item() + f2.norm(dim=1).mean().item())),
                    "train/foreground_fraction": float(foreground_fraction.mean().item()),
                    "train/grad_norm": grad_norm,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "step": global_step,
                }
                if run is not None:
                    wandb.log(log_data, step=global_step)

            if run is not None and global_step % media_every == 0:
                wandb.log(
                    {
                        "media/base": wandb.Image(make_grid_image(base)),
                        "media/view1": wandb.Image(make_grid_image(view1)),
                        "media/view2": wandb.Image(make_grid_image(view2)),
                    },
                    step=global_step,
                )

            progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "repr": f"{diagnostics['repr_loss'].item():.4f}",
                    "std": f"{diagnostics['std_loss'].item():.4f}",
                    "cov": f"{diagnostics['cov_loss'].item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                },
                refresh=False,
            )
            global_step += 1

        val_metrics = evaluate(model, test_loader, resolved_device, sim_coeff, std_coeff, cov_coeff, amp_dtype)
        if run is not None:
            wandb.log({f"val/{key}": value for key, value in val_metrics.items()} | {"epoch": epoch, "step": global_step}, step=global_step)

        save_checkpoint(output_dir, model, optimizer, epoch, global_step, best_val_loss, config, "last.pt")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(output_dir, model, optimizer, epoch, global_step, best_val_loss, config, "best.pt")

    if run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
