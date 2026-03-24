from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import click
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embedding.dataset import load_grayscale_image
from metrics.embedding_neighbors import (
    EmbeddingNeighborEvaluator,
    EmbeddingNeighborRuntimeConfig,
)
from train_resnet3d_lib.embedding_loss import resolve_embedding_runtime_config


DEFAULT_COMPARISON_ROOT = Path("/mnt/bcache/projects/vesuvius-scrolls/comparison-website-final")
DEFAULT_EMBEDDING_CHECKPOINT = REPO_ROOT / "embedding" / "embedding_runs" / "latest" / "best.pt"


@dataclass(frozen=True)
class PreferenceRow:
    row_id: int
    pair_id: str
    fold: str
    sample: str
    left_image: str
    right_image: str
    preference: str


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_preference_rows(
    *,
    db_path: Path,
    fold: str | None,
    sample: str | None,
    limit: int | None,
) -> list[PreferenceRow]:
    query = """
        SELECT
            id,
            pair_id,
            fold,
            sample,
            left_image,
            right_image,
            preference
        FROM preference_logs
    """
    clauses: list[str] = []
    params: list[object] = []
    if fold is not None:
        clauses.append("fold = ?")
        params.append(fold)
    if sample is not None:
        clauses.append("sample = ?")
        params.append(sample)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id ASC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(int(limit))

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    return [
        PreferenceRow(
            row_id=int(row[0]),
            pair_id=str(row[1]),
            fold=str(row[2]),
            sample=str(row[3]),
            left_image=str(row[4]),
            right_image=str(row[5]),
            preference=str(row[6]),
        )
        for row in rows
    ]


def _score_image(
    *,
    evaluator: EmbeddingNeighborEvaluator,
    image_path: Path,
    score_cache: dict[tuple[Path, float | None], dict[str, float]],
    threshold: float | None,
) -> dict[str, float]:
    cache_key = (image_path, threshold)
    cached = score_cache.get(cache_key)
    if cached is not None:
        return cached

    image = load_grayscale_image(image_path, downsample_factor=1)
    if threshold is not None:
        image = (image >= float(threshold)).astype(image.dtype, copy=False)
    prediction = torch.from_numpy(image)
    valid_mask = torch.ones_like(prediction, dtype=torch.bool)
    scores = evaluator.score_prediction(
        prediction=prediction,
        valid_mask=valid_mask,
    )
    score_cache[cache_key] = scores
    return scores


def _format_pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def _write_original_dino_config(
    *,
    embedding_config_path: Path | None,
    embedding_checkpoint_path: Path | None,
    temp_dir: Path,
) -> Path:
    _, _, base_config = resolve_embedding_runtime_config(
        config_path=embedding_config_path,
        checkpoint_path=embedding_checkpoint_path,
    )
    config = dict(base_config)
    config["adaptation_method"] = "plain"
    config["backbone_checkpoint"] = None
    config["freeze_backbone"] = True
    config["dropout"] = 0.0

    output_path = temp_dir / "original_dino_config.json"
    output_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


@click.command()
@click.option(
    "--comparison-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_COMPARISON_ROOT,
    show_default=True,
)
@click.option(
    "--db-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
)
@click.option(
    "--images-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
)
@click.option("--fold", type=str, default=None)
@click.option("--sample", type=str, default=None)
@click.option("--limit", type=click.IntRange(min=1), default=None)
@click.option("--embedding-config-path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option(
    "--embedding-checkpoint-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_EMBEDDING_CHECKPOINT,
    show_default=True,
)
@click.option(
    "--embedding-original-dino/--no-embedding-original-dino",
    default=False,
    show_default=True,
    help="Ignore the learned embedding head and use the original HF/timm DINO backbone implied by the embedding config.",
)
@click.option("--dataset-split", type=click.Choice(["train", "test"]), default="train", show_default=True)
@click.option("--dataset-samples", type=click.IntRange(min=1), default=2048, show_default=True)
@click.option("--dataset-seed", type=int, default=None)
@click.option("--dataset-batch-size", type=click.IntRange(min=1), default=64, show_default=True)
@click.option("--dataset-num-workers", type=click.IntRange(min=0), default=4, show_default=True)
@click.option("--neighbors-k", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--query-crop-size", type=click.IntRange(min=1), default=None)
@click.option("--query-stride", type=click.IntRange(min=1), default=None)
@click.option("--downsample-factor", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--threshold", type=click.FloatRange(min=0.0, max=1.0), default=None)
@click.option("--query-batch-size", type=click.IntRange(min=1), default=64, show_default=True)
@click.option("--search-chunk-size", type=click.IntRange(min=1), default=256, show_default=True)
@click.option("--device", type=str, default=None)
@click.option("--json-output", type=click.Path(dir_okay=False, path_type=Path), default=None)
def main(
    comparison_root: Path,
    db_path: Path | None,
    images_dir: Path | None,
    fold: str | None,
    sample: str | None,
    limit: int | None,
    embedding_config_path: Path | None,
    embedding_checkpoint_path: Path | None,
    embedding_original_dino: bool,
    dataset_split: str,
    dataset_samples: int,
    dataset_seed: int | None,
    dataset_batch_size: int,
    dataset_num_workers: int,
    neighbors_k: int,
    query_crop_size: int | None,
    query_stride: int | None,
    downsample_factor: int,
    threshold: float | None,
    query_batch_size: int,
    search_chunk_size: int,
    device: str | None,
    json_output: Path | None,
) -> None:
    resolved_db_path = db_path if db_path is not None else comparison_root / "data" / "preferences.db"
    resolved_images_dir = images_dir if images_dir is not None else comparison_root / "images"
    if not resolved_db_path.exists():
        raise click.ClickException(f"Preference DB not found: {resolved_db_path}")
    if not resolved_images_dir.exists():
        raise click.ClickException(f"Images directory not found: {resolved_images_dir}")

    rows = _load_preference_rows(
        db_path=resolved_db_path,
        fold=fold,
        sample=sample,
        limit=limit,
    )
    if not rows:
        raise click.ClickException("No preference rows matched the requested filters")

    model_device = _resolve_device(device)
    resolved_embedding_config_path = (
        embedding_config_path.expanduser().resolve() if embedding_config_path is not None else None
    )
    resolved_embedding_checkpoint_path = (
        embedding_checkpoint_path.expanduser().resolve()
        if embedding_checkpoint_path is not None
        else None
    )
    with tempfile.TemporaryDirectory(prefix="embedding-original-dino-") as temp_dir_str:
        if embedding_original_dino:
            if resolved_embedding_checkpoint_path is not None:
                click.echo(
                    (
                        "Warning: --embedding-checkpoint-path is used only to resolve embedding config; "
                        "model weights come from the original pretrained DINO backbone."
                    ),
                    err=True,
                )
            resolved_embedding_config_path = _write_original_dino_config(
                embedding_config_path=resolved_embedding_config_path,
                embedding_checkpoint_path=resolved_embedding_checkpoint_path,
                temp_dir=Path(temp_dir_str),
            )
            resolved_embedding_checkpoint_path = None

        runtime_config = EmbeddingNeighborRuntimeConfig(
            config_path=resolved_embedding_config_path,
            checkpoint_path=resolved_embedding_checkpoint_path,
            dataset_split=dataset_split,
            dataset_samples=dataset_samples,
            dataset_seed=dataset_seed,
            dataset_batch_size=dataset_batch_size,
            dataset_num_workers=dataset_num_workers,
            neighbors_k=neighbors_k,
            query_crop_size=query_crop_size,
            query_stride=query_stride,
            query_downsample_factor=downsample_factor,
            query_batch_size=query_batch_size,
            search_chunk_size=search_chunk_size,
        )
        evaluator = EmbeddingNeighborEvaluator.from_runtime_config(
            runtime_config=runtime_config,
            model_device=model_device,
        )

        # Collect votes and image paths per pair (one pass, no scoring yet)
        per_pair_votes: dict[str, list[str]] = defaultdict(list)
        per_pair_images: dict[str, tuple[Path, Path]] = {}
        for row in rows:
            left_path = (resolved_images_dir / row.left_image).resolve()
            right_path = (resolved_images_dir / row.right_image).resolve()
            if not left_path.exists():
                raise click.ClickException(f"Missing left image for pair {row.pair_id}: {left_path}")
            if not right_path.exists():
                raise click.ClickException(f"Missing right image for pair {row.pair_id}: {right_path}")
            per_pair_votes[row.pair_id].append(row.preference)
            per_pair_images[row.pair_id] = (left_path, right_path)

        # Score each unique pair once
        score_cache: dict[tuple[Path, float | None], dict[str, float]] = {}
        per_pair_predictions: dict[str, str] = {}
        for pair_id, (left_path, right_path) in tqdm(per_pair_images.items(), desc="Scoring pairs", unit="pair"):
            left_scores = _score_image(
                evaluator=evaluator,
                image_path=left_path,
                score_cache=score_cache,
                threshold=threshold,
            )
            right_scores = _score_image(
                evaluator=evaluator,
                image_path=right_path,
                score_cache=score_cache,
                threshold=threshold,
            )
            left_distance = float(left_scores["topk_distance_mean"])
            right_distance = float(right_scores["topk_distance_mean"])
            if left_distance < right_distance:
                per_pair_predictions[pair_id] = "left"
            elif right_distance < left_distance:
                per_pair_predictions[pair_id] = "right"
            else:
                per_pair_predictions[pair_id] = "tie"

        # Compute row-level metrics by joining predictions back to individual ratings
        row_correct = 0
        row_incorrect = 0
        row_ties = 0
        for row in rows:
            predicted_preference = per_pair_predictions[row.pair_id]
            if predicted_preference == "tie":
                row_ties += 1
            elif predicted_preference == row.preference:
                row_correct += 1
            else:
                row_incorrect += 1

        pair_majority_correct = 0
        pair_majority_incorrect = 0
        pair_majority_ties = 0
        pair_prediction_ties = 0
        for pair_id, votes in per_pair_votes.items():
            predicted_preference = per_pair_predictions[pair_id]
            if predicted_preference == "tie":
                pair_prediction_ties += 1
                continue

            counts = Counter(votes)
            left_votes = int(counts.get("left", 0))
            right_votes = int(counts.get("right", 0))
            if left_votes == right_votes:
                pair_majority_ties += 1
                continue
            majority_preference = "left" if left_votes > right_votes else "right"
            if predicted_preference == majority_preference:
                pair_majority_correct += 1
            else:
                pair_majority_incorrect += 1

        summary = {
            "db_path": str(resolved_db_path),
            "images_dir": str(resolved_images_dir),
            "filters": {
                "fold": fold,
                "sample": sample,
                "limit": limit,
            },
            "embedding_runtime": {
                "config_path": str(runtime_config.config_path) if runtime_config.config_path is not None else None,
                "checkpoint_path": (
                    str(runtime_config.checkpoint_path) if runtime_config.checkpoint_path is not None else None
                ),
                "original_dino": embedding_original_dino,
                "dataset_split": runtime_config.dataset_split,
                "dataset_samples": runtime_config.dataset_samples,
                "dataset_seed": runtime_config.dataset_seed,
                "dataset_batch_size": runtime_config.dataset_batch_size,
                "dataset_num_workers": runtime_config.dataset_num_workers,
                "neighbors_k": runtime_config.neighbors_k,
                "query_crop_size": runtime_config.query_crop_size,
                "query_stride": runtime_config.query_stride,
                "downsample_factor": runtime_config.query_downsample_factor,
                "threshold": threshold,
                "query_batch_size": runtime_config.query_batch_size,
                "search_chunk_size": runtime_config.search_chunk_size,
                "device": str(model_device),
            },
            "row_metrics": {
                "total": len(rows),
                "correct": row_correct,
                "incorrect": row_incorrect,
                "ties": row_ties,
                "accuracy_percent": _format_pct(row_correct, len(rows)),
                "decisive_accuracy_percent": _format_pct(row_correct, row_correct + row_incorrect),
            },
            "pair_majority_metrics": {
                "total_pairs": len(per_pair_votes),
                "correct": pair_majority_correct,
                "incorrect": pair_majority_incorrect,
                "human_majority_ties": pair_majority_ties,
                "metric_prediction_ties": pair_prediction_ties,
                "accuracy_percent": _format_pct(pair_majority_correct, len(per_pair_votes)),
                "decisive_accuracy_percent": _format_pct(
                    pair_majority_correct,
                    pair_majority_correct + pair_majority_incorrect,
                ),
            },
            "unique_images_scored": len(score_cache),
        }

        click.echo(json.dumps(summary, indent=2, sort_keys=True))
        if json_output is not None:
            json_output.parent.mkdir(parents=True, exist_ok=True)
            json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
