from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from embedding.dataset import DatasetConfig, IndexedInkCropDataset
from embedding.model import InkPatchEmbedder, normalize_for_backbone
from train_resnet3d_lib.embedding_loss import (
    _crop_to_valid_bbox,
    build_embedding_model_for_similarity,
    resolve_embedding_runtime_config,
)


class _IdentityAugmentation:
    def __call__(self, *, image: np.ndarray, **_: Any) -> dict[str, np.ndarray]:
        return {"image": image}


@dataclass(frozen=True)
class EmbeddingNeighborRuntimeConfig:
    config_path: Path | None
    checkpoint_path: Path | None
    dataset_split: str
    dataset_samples: int
    dataset_seed: int | None
    dataset_batch_size: int
    dataset_num_workers: int
    neighbors_k: int
    query_crop_size: int | None
    query_stride: int | None
    query_downsample_factor: int
    query_batch_size: int
    search_chunk_size: int


def resolve_sliding_starts(length: int, crop_size: int, stride: int) -> list[int]:
    length = int(length)
    crop_size = int(crop_size)
    stride = int(stride)
    if length <= 0 or crop_size <= 0 or stride <= 0:
        raise ValueError(
            f"Expected positive length/crop_size/stride, got {length}, {crop_size}, {stride}"
        )
    if length <= crop_size:
        return [0]
    starts = list(range(0, length - crop_size + 1, stride))
    last_start = length - crop_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def extract_sliding_embedding_patches(
    image: torch.Tensor,
    *,
    crop_size: int,
    stride: int,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape={tuple(image.shape)!r}")
    crop_size = int(crop_size)
    stride = int(stride)
    starts_y = resolve_sliding_starts(int(image.shape[0]), crop_size, stride)
    starts_x = resolve_sliding_starts(int(image.shape[1]), crop_size, stride)
    patches: list[torch.Tensor] = []
    coords: list[tuple[int, int]] = []
    for top in starts_y:
        for left in starts_x:
            patch = image[top : top + crop_size, left : left + crop_size]
            if patch.shape != (crop_size, crop_size):
                padded = image.new_zeros((crop_size, crop_size))
                padded[: patch.shape[0], : patch.shape[1]] = patch
                patch = padded
            patches.append(patch)
            coords.append((int(top), int(left)))
    return torch.stack(patches, dim=0).unsqueeze(1), coords


def downsample_crops_after_extraction(
    patches: torch.Tensor,
    *,
    downsample_factor: int,
    target_size: int,
) -> torch.Tensor:
    if patches.ndim != 4:
        raise ValueError(f"Expected patches with shape (N,1,H,W), got {tuple(patches.shape)!r}")
    downsample_factor = int(downsample_factor)
    target_size = int(target_size)
    if downsample_factor < 1:
        raise ValueError(f"downsample_factor must be >= 1, got {downsample_factor}")
    if target_size < 1:
        raise ValueError(f"target_size must be >= 1, got {target_size}")

    if downsample_factor == 1 and int(patches.shape[-1]) == target_size and int(patches.shape[-2]) == target_size:
        return patches

    if downsample_factor > 1:
        expected_size = max(1, int(round(float(patches.shape[-1]) / float(downsample_factor))))
        if expected_size != target_size:
            raise ValueError(
                "query crop size and downsample factor must match embedding input size: "
                f"crop={int(patches.shape[-1])}, factor={downsample_factor}, "
                f"expected target={expected_size}, actual target={target_size}"
            )

    return F.interpolate(
        patches,
        size=(target_size, target_size),
        mode="area",
    )


def search_embedding_neighbors(
    *,
    database_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    neighbors_k: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if database_embeddings.ndim != 2 or query_embeddings.ndim != 2:
        raise ValueError(
            "database_embeddings and query_embeddings must both be rank-2 tensors, "
            f"got {tuple(database_embeddings.shape)!r} and {tuple(query_embeddings.shape)!r}"
        )
    if database_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"{database_embeddings.shape[1]} vs {query_embeddings.shape[1]}"
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if neighbors_k < 1:
        raise ValueError(f"neighbors_k must be >= 1, got {neighbors_k}")
    if int(database_embeddings.shape[0]) < 1:
        raise ValueError("Need at least one database embedding")
    if int(query_embeddings.shape[0]) < 1:
        raise ValueError("Need at least one query embedding")

    effective_k = min(int(neighbors_k), int(database_embeddings.shape[0]))
    top_scores: list[torch.Tensor] = []
    top_indices: list[torch.Tensor] = []
    for start in range(0, int(query_embeddings.shape[0]), int(chunk_size)):
        query_chunk = query_embeddings[start : start + int(chunk_size)]
        scores = query_chunk @ database_embeddings.T
        chunk_scores, chunk_indices = torch.topk(scores, k=effective_k, dim=1)
        top_scores.append(chunk_scores.cpu())
        top_indices.append(chunk_indices.cpu())
    return torch.cat(top_scores, dim=0), torch.cat(top_indices, dim=0)


def summarize_neighbor_scores(neighbor_scores: torch.Tensor) -> dict[str, float]:
    if neighbor_scores.ndim != 2 or neighbor_scores.numel() == 0:
        raise ValueError(f"Expected non-empty rank-2 neighbor_scores, got {tuple(neighbor_scores.shape)!r}")
    per_crop_topk_cosine = neighbor_scores.mean(dim=1)
    per_crop_top1_cosine = neighbor_scores[:, 0]
    per_crop_topk_distance = 1.0 - per_crop_topk_cosine
    per_crop_top1_distance = 1.0 - per_crop_top1_cosine
    return {
        "top1_cosine_mean": float(per_crop_top1_cosine.mean().item()),
        "top1_distance_mean": float(per_crop_top1_distance.mean().item()),
        "topk_cosine_mean": float(per_crop_topk_cosine.mean().item()),
        "topk_distance_mean": float(per_crop_topk_distance.mean().item()),
        "topk_distance_std": float(per_crop_topk_distance.std(unbiased=False).item()),
        "num_crops": float(neighbor_scores.shape[0]),
        "neighbors_k": float(neighbor_scores.shape[1]),
    }


def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def _embed_batches(
    *,
    model: InkPatchEmbedder,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    embeddings: list[torch.Tensor] = []
    for start in range(0, int(images.shape[0]), int(batch_size)):
        batch = images[start : start + int(batch_size)].to(device, non_blocking=True)
        with torch.no_grad():
            batch_embeddings, _ = model(normalize_for_backbone(batch))
        embeddings.append(F.normalize(batch_embeddings.float().cpu(), dim=1))
    return torch.cat(embeddings, dim=0)


class EmbeddingNeighborEvaluator:
    def __init__(
        self,
        *,
        model: InkPatchEmbedder,
        model_device: torch.device,
        input_crop_size: int,
        database_embeddings: torch.Tensor,
        runtime_config: EmbeddingNeighborRuntimeConfig,
    ) -> None:
        if database_embeddings.ndim != 2:
            raise ValueError(
                "database_embeddings must be a rank-2 tensor of precomputed embeddings "
                "compatible with the model output space, "
                f"got {tuple(database_embeddings.shape)!r}"
            )
        self.model = model
        self.model_device = model_device
        self.input_crop_size = int(input_crop_size)
        self.database_embeddings = F.normalize(database_embeddings.float().cpu(), dim=1)
        self.runtime_config = runtime_config

    @classmethod
    def from_runtime_config(
        cls,
        *,
        runtime_config: EmbeddingNeighborRuntimeConfig,
        model_device: torch.device,
    ) -> "EmbeddingNeighborEvaluator":
        resolved_cfg, resolved_ckpt, embedding_config = resolve_embedding_runtime_config(
            config_path=runtime_config.config_path,
            checkpoint_path=runtime_config.checkpoint_path,
        )
        model, model_runtime_config = build_embedding_model_for_similarity(
            config=embedding_config,
            checkpoint_path=resolved_ckpt,
            device=model_device,
        )
        dataset_cfg = DatasetConfig(
            image_dir=Path(model_runtime_config["image_dir"]).expanduser().resolve(),
            split=str(runtime_config.dataset_split),
            seed=int(
                model_runtime_config["seed"]
                if runtime_config.dataset_seed is None
                else runtime_config.dataset_seed
            ),
            crop_size=int(model_runtime_config["crop_size"]),
            downsample_factor=int(model_runtime_config["downsample_factor"]),
            samples_per_epoch=int(runtime_config.dataset_samples),
            min_foreground_fraction=float(model_runtime_config["min_foreground_fraction"]),
            max_crop_attempts=int(model_runtime_config["max_crop_attempts"]),
            test_fraction=float(model_runtime_config["test_fraction"]),
            foreground_threshold=float(model_runtime_config["foreground_threshold"]),
            cache_images=True,
        )
        dataset = IndexedInkCropDataset(dataset_cfg, _IdentityAugmentation())
        dataloader = DataLoader(
            dataset,
            batch_size=int(runtime_config.dataset_batch_size),
            shuffle=False,
            num_workers=int(runtime_config.dataset_num_workers),
            pin_memory=model_device.type == "cuda",
            drop_last=False,
            persistent_workers=int(runtime_config.dataset_num_workers) > 0,
            worker_init_fn=_worker_init_fn,
        )
        embedding_batches: list[torch.Tensor] = []
        for batch in tqdm(dataloader, 'preparing dataset embeddings'):
            base = batch["base"]
            embedding_batches.append(
                _embed_batches(
                    model=model,
                    images=base,
                    device=model_device,
                    batch_size=int(runtime_config.dataset_batch_size),
                )
            )
        if not embedding_batches:
            raise ValueError("Embedding neighbor dataset produced no crops")
        database_embeddings = torch.cat(embedding_batches, dim=0)
        return cls(
            model=model,
            model_device=model_device,
            input_crop_size=int(model_runtime_config["crop_size"]),
            database_embeddings=database_embeddings,
            runtime_config=EmbeddingNeighborRuntimeConfig(
                config_path=resolved_cfg,
                checkpoint_path=resolved_ckpt,
                dataset_split=runtime_config.dataset_split,
                dataset_samples=runtime_config.dataset_samples,
                dataset_seed=runtime_config.dataset_seed,
                dataset_batch_size=runtime_config.dataset_batch_size,
                dataset_num_workers=runtime_config.dataset_num_workers,
                neighbors_k=runtime_config.neighbors_k,
                query_crop_size=runtime_config.query_crop_size,
                query_stride=runtime_config.query_stride,
                query_downsample_factor=runtime_config.query_downsample_factor,
                query_batch_size=runtime_config.query_batch_size,
                search_chunk_size=runtime_config.search_chunk_size,
            ),
        )

    def score_prediction(
        self,
        *,
        prediction: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, float]:
        cropped_prediction = _crop_to_valid_bbox(
            prediction.float() * valid_mask.to(dtype=prediction.dtype),
            valid_mask.bool(),
        )

        query_crop_size = (
            int(self.runtime_config.query_crop_size)
            if self.runtime_config.query_crop_size is not None
            else int(self.input_crop_size * self.runtime_config.query_downsample_factor)
        )
        query_stride = (
            int(self.runtime_config.query_stride)
            if self.runtime_config.query_stride is not None
            else max(1, query_crop_size // 2)
        )
        query_patches, _ = extract_sliding_embedding_patches(
            cropped_prediction,
            crop_size=query_crop_size,
            stride=query_stride,
        )
        query_patches = downsample_crops_after_extraction(
            query_patches,
            downsample_factor=int(self.runtime_config.query_downsample_factor),
            target_size=int(self.input_crop_size),
        )
        query_embeddings = _embed_batches(
            model=self.model,
            images=query_patches,
            device=self.model_device,
            batch_size=int(self.runtime_config.query_batch_size),
        )
        neighbor_scores, _ = search_embedding_neighbors(
            database_embeddings=self.database_embeddings,
            query_embeddings=query_embeddings,
            neighbors_k=int(self.runtime_config.neighbors_k),
            chunk_size=int(self.runtime_config.search_chunk_size),
        )
        return summarize_neighbor_scores(neighbor_scores)


_EVALUATOR_CACHE: dict[tuple[Any, ...], EmbeddingNeighborEvaluator] = {}


def _resolve_eval_runtime_config_from_cfg(cfg: Any) -> EmbeddingNeighborRuntimeConfig | None:
    enabled = bool(getattr(cfg, "eval_embedding_nn_metric", False))
    if not enabled:
        return None

    config_path_value = getattr(cfg, "eval_embedding_nn_model_config_path", None)
    if config_path_value is None:
        config_path_value = getattr(cfg, "stitch_embedding_model_config_path", None)
    checkpoint_path_value = getattr(cfg, "eval_embedding_nn_model_checkpoint_path", None)
    if checkpoint_path_value is None:
        checkpoint_path_value = getattr(cfg, "stitch_embedding_model_checkpoint_path", None)
    if config_path_value is None and checkpoint_path_value is None:
        raise ValueError(
            "Embedding NN metric requires either eval_embedding_nn_model_config_path / "
            "eval_embedding_nn_model_checkpoint_path or the stitch embedding equivalents"
        )

    return EmbeddingNeighborRuntimeConfig(
        config_path=Path(config_path_value).expanduser().resolve() if config_path_value else None,
        checkpoint_path=Path(checkpoint_path_value).expanduser().resolve() if checkpoint_path_value else None,
        dataset_split=str(getattr(cfg, "eval_embedding_nn_dataset_split", "train")),
        dataset_samples=int(getattr(cfg, "eval_embedding_nn_dataset_samples", 2048)),
        dataset_seed=(
            None
            if getattr(cfg, "eval_embedding_nn_dataset_seed", None) is None
            else int(getattr(cfg, "eval_embedding_nn_dataset_seed"))
        ),
        dataset_batch_size=int(getattr(cfg, "eval_embedding_nn_dataset_batch_size", 64)),
        dataset_num_workers=int(getattr(cfg, "eval_embedding_nn_dataset_num_workers", 4)),
        neighbors_k=int(getattr(cfg, "eval_embedding_nn_neighbors_k", 4)),
        query_crop_size=(
            None
            if getattr(cfg, "eval_embedding_nn_crop_size", None) is None
            else int(getattr(cfg, "eval_embedding_nn_crop_size"))
        ),
        query_stride=(
            None
            if getattr(cfg, "eval_embedding_nn_stride", None) is None
            else int(getattr(cfg, "eval_embedding_nn_stride"))
        ),
        query_downsample_factor=int(
            getattr(
                cfg,
                "eval_embedding_nn_downsample_factor",
                getattr(cfg, "stitch_embedding_downsample_factor", 1),
            )
        ),
        query_batch_size=int(getattr(cfg, "eval_embedding_nn_query_batch_size", 64)),
        search_chunk_size=int(getattr(cfg, "eval_embedding_nn_search_chunk_size", 256)),
    )


def get_embedding_neighbor_evaluator(
    *,
    cfg: Any,
    model_device: torch.device,
) -> EmbeddingNeighborEvaluator | None:
    runtime_config = _resolve_eval_runtime_config_from_cfg(cfg)
    if runtime_config is None:
        return None

    if runtime_config.dataset_samples < 1:
        raise ValueError("eval_embedding_nn_dataset_samples must be >= 1")
    if runtime_config.dataset_batch_size < 1:
        raise ValueError("eval_embedding_nn_dataset_batch_size must be >= 1")
    if runtime_config.dataset_num_workers < 0:
        raise ValueError("eval_embedding_nn_dataset_num_workers must be >= 0")
    if runtime_config.neighbors_k < 1:
        raise ValueError("eval_embedding_nn_neighbors_k must be >= 1")
    if runtime_config.query_downsample_factor < 1:
        raise ValueError("eval_embedding_nn_downsample_factor must be >= 1")
    if runtime_config.query_batch_size < 1:
        raise ValueError("eval_embedding_nn_query_batch_size must be >= 1")
    if runtime_config.search_chunk_size < 1:
        raise ValueError("eval_embedding_nn_search_chunk_size must be >= 1")

    cache_key = (
        runtime_config,
        str(model_device),
    )
    evaluator = _EVALUATOR_CACHE.get(cache_key)
    if evaluator is None:
        evaluator = EmbeddingNeighborEvaluator.from_runtime_config(
            runtime_config=runtime_config,
            model_device=model_device,
        )
        _EVALUATOR_CACHE[cache_key] = evaluator
    return evaluator
