from __future__ import annotations

import importlib
import json
import random
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.neural_tracing.autoreg_mesh.config import load_autoreg_mesh_config, validate_autoreg_mesh_config
from vesuvius.neural_tracing.autoreg_mesh.dataset import AutoregMeshDataset, autoreg_mesh_collate
from vesuvius.neural_tracing.autoreg_mesh.infer import infer_autoreg_mesh
from vesuvius.neural_tracing.autoreg_mesh.losses import compute_autoreg_mesh_losses
from vesuvius.neural_tracing.autoreg_mesh.model import AutoregMeshModel
from vesuvius.neural_tracing.autoreg_mesh.serialization import deserialize_continuation_grid


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        elif key == "prompt_tokens":
            moved[key] = {
                inner_key: inner_value.to(device) if torch.is_tensor(inner_value) else inner_value
                for inner_key, inner_value in value.items()
            }
        else:
            moved[key] = value
    return moved


def _next_batch(iterator, dataloader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        batch = next(iterator)
    return batch, iterator


def _split_dataset(dataset: Dataset, *, seed: int, val_fraction: float) -> tuple[Dataset, Dataset | None]:
    total = len(dataset)
    if total <= 0:
        raise ValueError("autoreg_mesh training requires a non-empty dataset")
    if total < 2 or float(val_fraction) <= 0.0:
        return dataset, None

    num_val = int(round(total * float(val_fraction)))
    num_val = max(1, min(num_val, total - 1))
    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(total)
    val_indices = indices[:num_val].tolist()
    train_indices = indices[num_val:].tolist()
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def _make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=autoreg_mesh_collate,
        persistent_workers=bool(num_workers > 0),
        generator=generator,
    )


def _maybe_import_wandb(cfg: dict):
    if not cfg.get("wandb_project"):
        return None
    try:
        return importlib.import_module("wandb")
    except ImportError as exc:
        raise ImportError(
            "wandb_project is configured for autoreg_mesh training, but the 'wandb' package is not installed."
        ) from exc


def _load_checkpoint_payload(path: str | Path | None):
    if path is None:
        return None
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _resolve_wandb_run_id(cfg: dict, ckpt_payload: dict | None) -> str | None:
    run_id = cfg.get("wandb_run_id")
    if run_id is not None:
        return str(run_id)
    if not bool(cfg.get("wandb_resume", False)) or ckpt_payload is None:
        return None
    run_id = ckpt_payload.get("wandb_run_id")
    if run_id is None:
        ckpt_config = ckpt_payload.get("config", {})
        if isinstance(ckpt_config, dict):
            run_id = ckpt_config.get("wandb_run_id")
    return None if run_id is None else str(run_id)


def _make_checkpoint_payload(
    *,
    model: AutoregMeshModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict,
    step: int,
) -> dict:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "step": int(step),
        "wandb_run_id": config.get("wandb_run_id"),
    }
    if scheduler is not None:
        payload["lr_scheduler"] = scheduler.state_dict()
    return payload


def _save_checkpoint(
    *,
    out_dir: Path,
    filename: str,
    model: AutoregMeshModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict,
    step: int,
) -> Path:
    path = out_dir / filename
    torch.save(
        _make_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            step=step,
        ),
        path,
    )
    return path


def _loss_dict_to_metrics(loss_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    metrics = {}
    for key, value in loss_dict.items():
        if torch.is_tensor(value):
            metrics[key] = float(value.detach().cpu().item())
        else:
            metrics[key] = float(value)
    return metrics


def _mean_metric_dict(metric_dicts: list[dict[str, float]], *, prefix: str) -> dict[str, float]:
    if not metric_dicts:
        return {}
    sums: dict[str, float] = {}
    for metrics in metric_dicts:
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value)
    return {f"{prefix}{key}": value / float(len(metric_dicts)) for key, value in sums.items()}


def _as_numpy_grid(grid) -> np.ndarray:
    if torch.is_tensor(grid):
        return grid.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(grid, dtype=np.float32)


def _draw_line_2d(canvas: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr

    while True:
        if 0 <= r0 < canvas.shape[0] and 0 <= c0 < canvas.shape[1]:
            canvas[r0, c0] = 1.0
        if r0 == r1 and c0 == c1:
            break
        err2 = 2 * err
        if err2 > -dr:
            err -= dr
            c0 += sc
        if err2 < dc:
            err += dc
            r0 += sr


def _render_surface_projection(
    grid_local: np.ndarray,
    *,
    axes: tuple[int, int],
    panel_shape: tuple[int, int],
) -> np.ndarray:
    grid = np.asarray(grid_local, dtype=np.float32)
    panel = np.zeros(panel_shape, dtype=np.float32)
    valid = np.isfinite(grid).all(axis=-1)

    def _project(point: np.ndarray) -> tuple[int, int]:
        row = int(np.clip(np.rint(point[axes[0]]), 0, panel_shape[0] - 1))
        col = int(np.clip(np.rint(point[axes[1]]), 0, panel_shape[1] - 1))
        return row, col

    rows, cols = grid.shape[:2]
    for row_idx in range(rows):
        for col_idx in range(cols):
            if not valid[row_idx, col_idx]:
                continue
            r0, c0 = _project(grid[row_idx, col_idx])
            panel[r0, c0] = 1.0
            if col_idx + 1 < cols and valid[row_idx, col_idx + 1]:
                r1, c1 = _project(grid[row_idx, col_idx + 1])
                _draw_line_2d(panel, r0, c0, r1, c1)
            if row_idx + 1 < rows and valid[row_idx + 1, col_idx]:
                r1, c1 = _project(grid[row_idx + 1, col_idx])
                _draw_line_2d(panel, r0, c0, r1, c1)
    return panel


def _voxelize_grid_projection_panels(grid_local: np.ndarray, crop_shape: tuple[int, int, int]) -> list[tuple[str, np.ndarray]]:
    return [
        ("ZY", _render_surface_projection(grid_local, axes=(0, 1), panel_shape=(crop_shape[0], crop_shape[1]))),
        ("ZX", _render_surface_projection(grid_local, axes=(0, 2), panel_shape=(crop_shape[0], crop_shape[2]))),
        ("YX", _render_surface_projection(grid_local, axes=(1, 2), panel_shape=(crop_shape[1], crop_shape[2]))),
    ]


def _panel_to_rgb(panel: np.ndarray, *, color: tuple[int, int, int]) -> np.ndarray:
    clipped = np.clip(panel, 0.0, 1.0)
    image = np.zeros((*clipped.shape, 3), dtype=np.uint8)
    for channel, value in enumerate(color):
        image[..., channel] = (clipped * float(value)).astype(np.uint8)
    return image


def _pad_panel_height(panel: np.ndarray, *, height: int) -> np.ndarray:
    if int(panel.shape[0]) >= int(height):
        return panel
    pad_rows = int(height) - int(panel.shape[0])
    return np.pad(panel, ((0, pad_rows), (0, 0)), mode="constant")


def _add_header(canvas: np.ndarray, *, title: str, labels: list[str], background: tuple[int, int, int]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    header_height = 22
    header = np.zeros((header_height, canvas.shape[1], 3), dtype=np.uint8)
    header[..., 0] = background[0]
    header[..., 1] = background[1]
    header[..., 2] = background[2]
    image = Image.fromarray(header)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((6, 5), title, fill=(255, 255, 255), font=font)

    panel_width = canvas.shape[1] // len(labels)
    for idx, label in enumerate(labels):
        x = idx * panel_width + max(6, panel_width // 2 - 12)
        draw.text((x, 5), label, fill=(230, 230, 230), font=font)
    return np.concatenate([np.asarray(image, dtype=np.uint8), canvas], axis=0)


def _make_labeled_triptych(
    *,
    grid_local: np.ndarray,
    crop_shape: tuple[int, int, int],
    title: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    projection_panels = _voxelize_grid_projection_panels(grid_local, crop_shape)
    target_height = max(int(panel.shape[0]) for _, panel in projection_panels)
    rgb_panels = [
        _panel_to_rgb(_pad_panel_height(panel, height=target_height), color=color)
        for _, panel in projection_panels
    ]
    separator = np.full((target_height, 3, 3), 28, dtype=np.uint8)
    body = np.concatenate(
        [rgb_panels[0], separator, rgb_panels[1], separator.copy(), rgb_panels[2]],
        axis=1,
    )
    return _add_header(
        body,
        title=title,
        labels=[name for name, _ in projection_panels],
        background=tuple(max(12, int(value * 0.18)) for value in color),
    )


def _make_projection_canvas(
    *,
    prompt_grid_local: np.ndarray,
    target_grid_local: np.ndarray,
    pred_grid_local: np.ndarray,
    crop_shape: tuple[int, int, int],
) -> np.ndarray:
    prompt_panel = _make_labeled_triptych(
        grid_local=prompt_grid_local,
        crop_shape=crop_shape,
        title="Prompt",
        color=(90, 180, 255),
    )
    target_panel = _make_labeled_triptych(
        grid_local=target_grid_local,
        crop_shape=crop_shape,
        title="Target",
        color=(110, 235, 110),
    )
    pred_panel = _make_labeled_triptych(
        grid_local=pred_grid_local,
        crop_shape=crop_shape,
        title="Prediction",
        color=(255, 190, 90),
    )
    target_height = max(int(prompt_panel.shape[0]), int(target_panel.shape[0]), int(pred_panel.shape[0]))
    prompt_panel = _pad_panel_height(prompt_panel, height=target_height)
    target_panel = _pad_panel_height(target_panel, height=target_height)
    pred_panel = _pad_panel_height(pred_panel, height=target_height)
    separator = np.full((target_height, 8, 3), 24, dtype=np.uint8)
    return np.concatenate([prompt_panel, separator, target_panel, separator.copy(), pred_panel], axis=1)


def _make_teacher_forced_prediction_canvas(batch: dict, outputs: dict, *, sample_idx: int = 0) -> np.ndarray:
    count = int(batch["target_lengths"][sample_idx].item())
    grid_shape = tuple(int(v) for v in batch["target_grid_shape"][sample_idx].tolist())
    direction = str(batch["direction"][sample_idx])
    pred_xyz = outputs["pred_xyz"][sample_idx, :count].detach().cpu().numpy()
    pred_grid_local = deserialize_continuation_grid(pred_xyz, direction=direction, grid_shape=grid_shape)
    prompt_grid_local = _as_numpy_grid(batch["prompt_grid_local"][sample_idx])
    target_grid_local = _as_numpy_grid(batch["target_grid_local"][sample_idx])
    crop_shape = tuple(int(v) for v in batch["volume"][sample_idx].shape[-3:])
    return _make_projection_canvas(
        prompt_grid_local=prompt_grid_local,
        target_grid_local=target_grid_local,
        pred_grid_local=pred_grid_local,
        crop_shape=crop_shape,
    )


def _make_inference_prediction_canvas(raw_sample: dict, inference_result: dict) -> np.ndarray:
    prompt_grid_local = _as_numpy_grid(raw_sample["prompt_grid_local"])
    target_grid_local = _as_numpy_grid(raw_sample["target_grid_local"])
    pred_grid_local = np.asarray(inference_result["continuation_grid_local"], dtype=np.float32)
    crop_shape = tuple(int(v) for v in raw_sample["volume"].shape[-3:])
    return _make_projection_canvas(
        prompt_grid_local=prompt_grid_local,
        target_grid_local=target_grid_local,
        pred_grid_local=pred_grid_local,
        crop_shape=crop_shape,
    )


@torch.no_grad()
def _evaluate_validation(
    *,
    model: AutoregMeshModel,
    dataloader: DataLoader,
    iterator,
    cfg: dict,
    device: torch.device,
) -> tuple[dict[str, float], Any]:
    model.eval()
    metric_dicts: list[dict[str, float]] = []
    for _ in range(int(cfg["val_batches_per_log"])):
        raw_batch, iterator = _next_batch(iterator, dataloader)
        batch = _move_batch_to_device(raw_batch, device)
        outputs = model(batch)
        loss_dict = compute_autoreg_mesh_losses(
            outputs,
            batch,
            offset_num_bins=tuple(int(v) for v in cfg["offset_num_bins"]),
            occupancy_loss_weight=float(cfg.get("occupancy_loss_weight", 0.0)),
        )
        metric_dicts.append(_loss_dict_to_metrics(loss_dict))
    model.train()
    return _mean_metric_dict(metric_dicts, prefix="val_"), iterator


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

    out_dir = Path(cfg["out_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if dataset is None:
        dataset = AutoregMeshDataset(cfg)
    train_dataset, val_dataset = _split_dataset(
        dataset,
        seed=int(cfg["seed"]),
        val_fraction=float(cfg.get("val_fraction", 0.0)),
    )
    train_dataloader = _make_dataloader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        shuffle=True,
        seed=int(cfg["seed"]),
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = _make_dataloader(
            val_dataset,
            batch_size=int(cfg["batch_size"]),
            num_workers=int(cfg["val_num_workers"]),
            shuffle=False,
            seed=int(cfg["seed"]) + 1,
        )

    if model is None:
        model = AutoregMeshModel(cfg)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.train()

    optimizer = create_optimizer(dict(cfg["optimizer"]), model)
    scheduler = None
    scheduler_name = str(cfg.get("scheduler", "constant")).lower()
    total_steps = int(max_steps or cfg["num_steps"])
    if scheduler_name != "constant":
        from vesuvius.models.training.lr_schedulers import get_scheduler

        scheduler = get_scheduler(
            scheduler_type=scheduler_name,
            optimizer=optimizer,
            initial_lr=float(cfg["optimizer"]["learning_rate"]),
            max_steps=total_steps,
            **dict(cfg.get("scheduler_kwargs") or {}),
        )

    preloaded_ckpt = _load_checkpoint_payload(cfg.get("load_ckpt"))
    resolved_wandb_run_id = _resolve_wandb_run_id(cfg, preloaded_ckpt)
    if resolved_wandb_run_id is not None:
        cfg["wandb_run_id"] = resolved_wandb_run_id

    start_step = 0
    if preloaded_ckpt is not None:
        model.load_state_dict(preloaded_ckpt["model"])
        if not bool(cfg.get("load_weights_only", False)):
            start_step = int(preloaded_ckpt.get("step", 0))
            if "optimizer" in preloaded_ckpt:
                optimizer.load_state_dict(preloaded_ckpt["optimizer"])
            if scheduler is not None and "lr_scheduler" in preloaded_ckpt:
                scheduler.load_state_dict(preloaded_ckpt["lr_scheduler"])

    wandb = _maybe_import_wandb(cfg)
    wandb_run = None
    saved_checkpoints: list[str] = []
    final_checkpoint_path = None
    history: list[dict[str, float]] = []

    try:
        if wandb is not None:
            wandb_kwargs = {
                "project": cfg["wandb_project"],
                "config": cfg,
            }
            if cfg.get("wandb_entity") is not None:
                wandb_kwargs["entity"] = cfg["wandb_entity"]
            if cfg.get("wandb_run_name") is not None:
                wandb_kwargs["name"] = cfg["wandb_run_name"]
            if bool(cfg.get("wandb_resume", False)):
                wandb_kwargs["resume"] = cfg.get("wandb_resume_mode", "allow")
                if cfg.get("wandb_run_id") is not None:
                    wandb_kwargs["id"] = cfg["wandb_run_id"]
            wandb_run = wandb.init(**wandb_kwargs)
            active_run = getattr(wandb, "run", None) or wandb_run
            active_run_id = getattr(active_run, "id", None)
            if active_run_id is not None:
                cfg["wandb_run_id"] = str(active_run_id)

        if bool(cfg.get("ckpt_at_step_zero", False)) and start_step == 0:
            ckpt_path = _save_checkpoint(
                out_dir=out_dir,
                filename="ckpt_000000.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=cfg,
                step=0,
            )
            saved_checkpoints.append(str(ckpt_path))

        train_iterator = iter(train_dataloader)
        val_iterator = iter(val_dataloader) if val_dataloader is not None else None
        global_step = int(start_step)
        progress_bar = tqdm(total=max(0, total_steps - global_step), desc="autoreg_mesh", leave=False)

        while global_step < total_steps:
            raw_batch, train_iterator = _next_batch(train_iterator, train_dataloader)
            batch = _move_batch_to_device(raw_batch, device)

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
                raise RuntimeError(f"Encountered non-finite training loss at step {global_step}: {loss.item()}")

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg["grad_clip"]))
            grad_norm_value = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
            skipped_step = 0.0
            if np.isfinite(grad_norm_value):
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            else:
                skipped_step = 1.0
            global_step += 1

            metrics = _loss_dict_to_metrics(loss_dict)
            metrics["current_lr"] = float(optimizer.param_groups[0]["lr"])
            metrics["grad_norm"] = grad_norm_value
            metrics["step"] = float(global_step)
            if skipped_step > 0.0:
                metrics["skipped_step_nonfinite_grad"] = skipped_step

            should_run_validation = (
                val_dataloader is not None and
                global_step % int(cfg["log_frequency"]) == 0
            )
            if should_run_validation:
                val_metrics, val_iterator = _evaluate_validation(
                    model=model,
                    dataloader=val_dataloader,
                    iterator=val_iterator,
                    cfg=cfg,
                    device=device,
                )
                metrics.update(val_metrics)

            wandb_payload = dict(metrics)
            should_log_images = (
                wandb is not None and
                bool(cfg.get("wandb_log_images", True)) and
                global_step % int(cfg["wandb_image_frequency"]) == 0
            )
            if should_log_images:
                wandb_payload["train_example"] = wandb.Image(
                    _make_teacher_forced_prediction_canvas(batch, outputs, sample_idx=0),
                    caption=f"step={global_step} train teacher-forced",
                )
                if val_dataset is not None and len(val_dataset) > 0:
                    model.eval()
                    raw_val_sample = val_dataset[0]
                    val_infer = infer_autoreg_mesh(model, raw_val_sample, greedy=True)
                    model.train()
                    wandb_payload["val_example"] = wandb.Image(
                        _make_inference_prediction_canvas(raw_val_sample, val_infer),
                        caption=f"step={global_step} val autoregressive",
                    )

            history.append(dict(metrics))
            if wandb is not None:
                wandb.log(wandb_payload, step=global_step)

            progress_bar.set_postfix({"loss": f"{metrics['loss']:.4f}"})
            progress_bar.update(1)

            if global_step % int(cfg["ckpt_frequency"]) == 0:
                ckpt_path = _save_checkpoint(
                    out_dir=out_dir,
                    filename=f"ckpt_{global_step:06}.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=cfg,
                    step=global_step,
                )
                saved_checkpoints.append(str(ckpt_path))

        progress_bar.close()

        if bool(cfg.get("save_final_checkpoint", True)):
            final_ckpt = _save_checkpoint(
                out_dir=out_dir,
                filename="final.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=cfg,
                step=global_step,
            )
            final_checkpoint_path = str(final_ckpt)
            saved_checkpoints.append(final_checkpoint_path)

        return {
            "model": model,
            "optimizer": optimizer,
            "history": history,
            "final_metrics": history[-1] if history else {},
            "start_step": start_step,
            "wandb_run_id": cfg.get("wandb_run_id"),
            "checkpoint_paths": saved_checkpoints,
            "final_checkpoint_path": final_checkpoint_path,
            "out_dir": str(out_dir),
        }
    finally:
        if wandb is not None:
            wandb.finish()


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def train(config_path: str) -> None:
    cfg = load_autoreg_mesh_config(Path(config_path))
    result = run_autoreg_mesh_training(cfg)
    print(json.dumps(result["final_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    train()
