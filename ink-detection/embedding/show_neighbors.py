#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import click
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import DatasetConfig, IndexedInkCropDataset, build_eval_augmentations
from model import PCAEmbedder, InkPatchEmbedder, create_frozen_dino_backbone, normalize_for_backbone


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def resolve_checkpoint_config(checkpoint_path: Path) -> tuple[Path, dict[str, Any]]:
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if isinstance(config, dict):
        return checkpoint_path, config

    config_path = checkpoint_path.parent / "config.json"
    if not config_path.exists():
        raise ValueError(f"Could not find config in checkpoint or at {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return checkpoint_path, json.load(handle)


def load_adapted_model(checkpoint_path: Path, device: torch.device) -> tuple[InkPatchEmbedder, dict[str, Any]]:
    checkpoint_path, config = resolve_checkpoint_config(checkpoint_path)
    backbone_name = str(config["backbone_name"])
    backbone_checkpoint = config.get("backbone_checkpoint")
    crop_size = int(config["crop_size"])
    embedding_dim = int(config["embedding_dim"])
    checkpoint_method = str(config.get("adaptation_method", "trained"))
    hidden_dim = int(config.get("hidden_dim", embedding_dim))
    dropout = float(config.get("dropout", 0.0 if checkpoint_method == "pca" else 0.1))

    backbone_ckpt_path = Path(backbone_checkpoint) if backbone_checkpoint else None
    backbone, backbone_dim = create_frozen_dino_backbone(backbone_name, backbone_ckpt_path, crop_size, device)
    if checkpoint_method == "pca":
        head = PCAEmbedder(backbone_dim, embedding_dim)
        model = InkPatchEmbedder(backbone, backbone_dim, embedding_dim, embedding_dim, 0.0, head=head).to(device)
    else:
        model = InkPatchEmbedder(backbone, backbone_dim, embedding_dim, hidden_dim, dropout).to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    head_state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(head_state, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a saved embedding head")
    missing, unexpected = model.head.load_state_dict(head_state, strict=True)
    if missing or unexpected:
        raise ValueError(f"Unexpected head state mismatch: missing={missing}, unexpected={unexpected}")

    model.eval()
    return model, config

def make_plain_config(
    image_dir: Path,
    backbone_name: str,
    backbone_checkpoint: Path | None,
    crop_size: int,
    downsample_factor: int,
    test_fraction: float,
    min_foreground_fraction: float,
    foreground_threshold: float,
    max_crop_attempts: int,
    seed: int | None,
) -> dict[str, Any]:
    return {
        "image_dir": str(image_dir.resolve()),
        "backbone_name": backbone_name,
        "backbone_checkpoint": str(backbone_checkpoint.resolve()) if backbone_checkpoint is not None else None,
        "crop_size": crop_size,
        "downsample_factor": downsample_factor,
        "test_fraction": test_fraction,
        "min_foreground_fraction": min_foreground_fraction,
        "foreground_threshold": foreground_threshold,
        "max_crop_attempts": max_crop_attempts,
        "seed": 1337 if seed is None else seed,
        "adaptation_method": "plain",
    }


def build_plain_model(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[InkPatchEmbedder, dict[str, Any]]:
    backbone_checkpoint = config.get("backbone_checkpoint")
    backbone_ckpt_path = Path(backbone_checkpoint) if backbone_checkpoint else None
    backbone, backbone_dim = create_frozen_dino_backbone(
        str(config["backbone_name"]),
        backbone_ckpt_path,
        int(config["crop_size"]),
        device,
    )
    model = InkPatchEmbedder(
        backbone=backbone,
        backbone_dim=backbone_dim,
        embedding_dim=backbone_dim,
        hidden_dim=backbone_dim,
        dropout=0.0,
        head=nn.Identity(),
    ).to(device)
    model.eval()
    config = {
        **config,
        "embedding_dim": backbone_dim,
        "hidden_dim": backbone_dim,
        "dropout": 0.0,
    }
    return model, config


def build_dataset(image_dir: Path, split: str, samples: int, config: dict[str, Any], seed: int | None) -> IndexedInkCropDataset:
    dataset_seed = int(config["seed"] if seed is None else seed)
    dataset_cfg = DatasetConfig(
        image_dir=image_dir.resolve(),
        split=split,
        seed=dataset_seed,
        crop_size=int(config["crop_size"]),
        downsample_factor=int(config["downsample_factor"]),
        samples_per_epoch=samples,
        min_foreground_fraction=float(config["min_foreground_fraction"]),
        max_crop_attempts=int(config["max_crop_attempts"]),
        test_fraction=float(config["test_fraction"]),
        foreground_threshold=float(config["foreground_threshold"]),
        cache_images=True,
    )
    return IndexedInkCropDataset(dataset_cfg, build_eval_augmentations(dataset_cfg.crop_size))


def embed_dataset(
    model: InkPatchEmbedder,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    embeddings_batches: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    for batch in tqdm(dataloader, desc="Embedding crops", dynamic_ncols=True):
        images = batch["image"].to(device, non_blocking=True)
        with torch.no_grad():
            embeddings, _ = model(normalize_for_backbone(images))
        embeddings_batches.append(F.normalize(embeddings.float().cpu(), dim=1))

        base = batch["base"].mul(255.0).round().to(torch.uint8).cpu()
        batch_size = images.shape[0]
        for idx in range(batch_size):
            records.append(
                {
                    "sample_index": int(batch["sample_index"][idx].item()),
                    "image_index": int(batch["image_index"][idx].item()),
                    "top": int(batch["top"][idx].item()),
                    "left": int(batch["left"][idx].item()),
                    "foreground_fraction": float(batch["foreground_fraction"][idx].item()),
                    "path": batch["path"][idx],
                    "crop": base[idx, 0].numpy(),
                }
            )
    if not embeddings_batches:
        raise ValueError("Dataloader produced no crops")
    return torch.cat(embeddings_batches, dim=0), records


def search_neighbors(
    embeddings: torch.Tensor,
    query_indices: torch.Tensor,
    image_indices: torch.Tensor,
    neighbors_k: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if neighbors_k < 1:
        raise ValueError(f"neighbors_k must be >= 1, got {neighbors_k}")

    database = embeddings
    top_scores: list[torch.Tensor] = []
    top_indices: list[torch.Tensor] = []
    device = database.device
    total = database.shape[0]
    effective_k = min(neighbors_k, total)
    if effective_k < 1:
        raise ValueError("Need at least 2 embeddings to find neighbors")
    database_image_indices = image_indices.to(device)

    for start in tqdm(range(0, query_indices.numel(), chunk_size), desc="Searching neighbors", dynamic_ncols=True):
        query_chunk = query_indices[start : start + chunk_size]
        query_embeddings = database.index_select(0, query_chunk.to(device))
        scores = query_embeddings @ database.T
        row_ids = torch.arange(query_chunk.shape[0], device=device)
        scores[row_ids, query_chunk.to(device)] = -math.inf
        query_image_indices = database_image_indices.index_select(0, query_chunk.to(device))
        same_image_mask = query_image_indices.unsqueeze(1).eq(database_image_indices.unsqueeze(0))
        scores[same_image_mask] = -math.inf
        chunk_scores, chunk_indices = torch.topk(scores, k=effective_k, dim=1)
        top_scores.append(chunk_scores.cpu())
        top_indices.append(chunk_indices.cpu())

    return torch.cat(top_scores, dim=0), torch.cat(top_indices, dim=0)


def render_query_panel(
    query_record: dict[str, Any],
    neighbor_records: list[dict[str, Any]],
    neighbor_scores: list[float],
    tile_size: int,
) -> np.ndarray:
    columns = 1 + len(neighbor_records)
    header_h = 44
    border_px = 6
    width = columns * tile_size + max(columns - 1, 0) * border_px
    canvas = np.full((tile_size + header_h, width), 18, dtype=np.uint8)
    border_color = 235

    def _place(tile: np.ndarray, x0: int, title: str, subtitle: str) -> None:
        resized = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        canvas[header_h:, x0 : x0 + tile_size] = resized
        cv2.putText(canvas, title, (x0 + 6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 235, 1, cv2.LINE_AA)
        cv2.putText(canvas, subtitle, (x0 + 6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.40, 195, 1, cv2.LINE_AA)

    def _x0(column: int) -> int:
        return column * (tile_size + border_px)

    for column in range(1, columns):
        x0 = _x0(column) - border_px
        canvas[:, x0 : x0 + border_px] = border_color

    query_name = Path(query_record["path"]).stem
    _place(query_record["crop"], _x0(0), "query", f"{query_name} @ {query_record['top']},{query_record['left']}")
    for rank, (record, score) in enumerate(zip(neighbor_records, neighbor_scores, strict=True), start=1):
        name = Path(record["path"]).stem
        subtitle = f"{score:.4f} {name} @ {record['top']},{record['left']}"
        _place(record["crop"], _x0(rank), f"nn {rank}", subtitle[:48])
    return canvas


def save_cache(cache_path: Path, embeddings: torch.Tensor, records: list[dict[str, Any]], config: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_records = [{**record, "crop": torch.from_numpy(record["crop"])} for record in records]
    torch.save({"embeddings": embeddings, "records": serializable_records, "config": config}, cache_path)


def load_cache(cache_path: Path) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    payload = torch.load(cache_path, map_location="cpu")
    embeddings = payload["embeddings"]
    records = [{**record, "crop": record["crop"].numpy()} for record in payload["records"]]
    return embeddings, records, payload["config"]


@click.command()
@click.argument("image_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("checkpoint_path", required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("embedding_neighbors/latest"))
@click.option("--adaptation-method", type=click.Choice(["plain", "trained", "pca"]), default="pca", show_default=True)
@click.option("--split", type=click.Choice(["train", "test"]), default="test", show_default=True)
@click.option("--samples", type=click.IntRange(min=2), default=2048, show_default=True)
@click.option("--query-count", type=click.IntRange(min=1), default=32, show_default=True)
@click.option("--neighbors-k", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--batch-size", type=click.IntRange(min=1), default=64, show_default=True)
@click.option("--num-workers", type=click.IntRange(min=0), default=4, show_default=True)
@click.option("--seed", type=int, default=None)
@click.option("--device", type=str, default=None)
@click.option("--search-device", type=click.Choice(["cpu", "cuda"]), default=None)
@click.option("--query-chunk-size", type=click.IntRange(min=1), default=256, show_default=True)
@click.option("--tile-size", type=click.IntRange(min=32), default=192, show_default=True)
@click.option("--cache-path", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--backbone-name", default="vit_small_patch14_dinov2.lvd142m", show_default=True)
@click.option("--backbone-checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--crop-size", type=click.IntRange(min=32), default=224, show_default=True)
@click.option("--downsample-factor", type=click.IntRange(min=1), default=2, show_default=True)
@click.option("--test-fraction", type=click.FloatRange(min=0.05, max=0.5), default=0.15, show_default=True)
@click.option("--min-foreground-fraction", type=click.FloatRange(min=0.0, max=1.0), default=0.04, show_default=True)
@click.option("--foreground-threshold", type=click.FloatRange(min=0.0, max=1.0), default=0.2, show_default=True)
@click.option("--max-crop-attempts", type=click.IntRange(min=1), default=24, show_default=True)
def main(
    image_dir: Path,
    checkpoint_path: Path | None,
    output_dir: Path,
    adaptation_method: str,
    split: str,
    samples: int,
    query_count: int,
    neighbors_k: int,
    batch_size: int,
    num_workers: int,
    seed: int | None,
    device: str | None,
    search_device: str | None,
    query_chunk_size: int,
    tile_size: int,
    cache_path: Path | None,
    backbone_name: str,
    backbone_checkpoint: Path | None,
    crop_size: int,
    downsample_factor: int,
    test_fraction: float,
    min_foreground_fraction: float,
    foreground_threshold: float,
    max_crop_attempts: int,
) -> None:
    resolved_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(1337 if seed is None else seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    if adaptation_method in {"trained", "pca"} and checkpoint_path is None:
        raise ValueError(f"checkpoint_path is required when --adaptation-method={adaptation_method}")

    if cache_path is not None and cache_path.exists():
        embeddings, records, config = load_cache(cache_path)
    else:
        if adaptation_method == "plain":
            config = make_plain_config(
                image_dir=image_dir,
                backbone_name=backbone_name,
                backbone_checkpoint=backbone_checkpoint,
                crop_size=crop_size,
                downsample_factor=downsample_factor,
                test_fraction=test_fraction,
                min_foreground_fraction=min_foreground_fraction,
                foreground_threshold=foreground_threshold,
                max_crop_attempts=max_crop_attempts,
                seed=seed,
            )
            model, config = build_plain_model(config, resolved_device)
        else:
            model, config = load_adapted_model(checkpoint_path, resolved_device)
            checkpoint_method = str(config.get("adaptation_method", "trained"))
            if checkpoint_method != adaptation_method:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} uses adaptation_method={checkpoint_method!r}, "
                    f"expected {adaptation_method!r}"
                )
        dataset = build_dataset(image_dir, split, samples, config, seed)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=resolved_device.type == "cuda",
            drop_last=False,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
        )
        embeddings, records = embed_dataset(model, dataloader, resolved_device)
        if cache_path is not None:
            save_cache(cache_path, embeddings, records, config)

    if len(records) != embeddings.shape[0]:
        raise ValueError("Embedding count does not match record count")
    if embeddings.shape[0] < 2:
        raise ValueError("Need at least 2 embedded crops to search neighbors")

    query_count = min(query_count, embeddings.shape[0])
    selection_generator = torch.Generator().manual_seed(1337 if seed is None else seed)
    query_indices = torch.randperm(embeddings.shape[0], generator=selection_generator)[:query_count]
    image_indices = torch.tensor([int(record["image_index"]) for record in records], dtype=torch.long)
    requested_search_device = search_device or ("cuda" if resolved_device.type == "cuda" else "cpu")
    if requested_search_device == "cuda" and not torch.cuda.is_available():
        requested_search_device = "cpu"
    search_embeddings = embeddings.to(requested_search_device)
    scores, neighbor_indices = search_neighbors(
        search_embeddings,
        query_indices,
        image_indices,
        neighbors_k,
        query_chunk_size,
    )

    manifest = {
        "image_dir": str(image_dir.resolve()),
        "adaptation_method": adaptation_method,
        "split": split,
        "samples": int(embeddings.shape[0]),
        "query_count": int(query_count),
        "neighbors_k": int(neighbors_k),
        "search_device": requested_search_device,
        "query_chunk_size": query_chunk_size,
        "model_config": config,
    }
    if checkpoint_path is not None:
        manifest["checkpoint_path"] = str(checkpoint_path.resolve())

    rendered_paths: list[str] = []
    for row, query_index in enumerate(query_indices.tolist()):
        query_record = records[query_index]
        ranked_pairs = [
            (index, score)
            for index, score in zip(neighbor_indices[row].tolist(), scores[row].tolist(), strict=True)
            if math.isfinite(score)
        ]
        nn_indices = [index for index, _ in ranked_pairs]
        nn_scores = [score for _, score in ranked_pairs]
        nn_records = [records[index] for index in nn_indices]
        panel = render_query_panel(query_record, nn_records, nn_scores, tile_size=tile_size)
        output_path = output_dir / f"query_{row:03d}_sample_{query_record['sample_index']:05d}.png"
        cv2.imwrite(str(output_path), panel)
        rendered_paths.append(str(output_path))

    manifest["rendered_queries"] = rendered_paths
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
