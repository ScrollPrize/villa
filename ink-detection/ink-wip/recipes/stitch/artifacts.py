from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ink.core.run_fs import to_plain
from ink.recipes.stitch.store import downsample_preview_for_media, write_preview_png


def resolve_media_downsample(runtime) -> int:
    wandb_cfg = getattr(runtime, "wandb", None)
    return max(1, int(getattr(wandb_cfg, "media_downsample", 1)))


def stitch_source_downsample(stitch_runtime, stitch_train=None) -> int:
    for candidate in (getattr(stitch_train, "data", None), getattr(stitch_runtime, "data", None)):
        layout = getattr(candidate, "layout", None)
        if layout is not None:
            return int(getattr(layout, "downsample", 1))
    return 1


def write_segment_viz_artifacts(
    *,
    root_dir=None,
    split_name: str,
    epoch: int,
    segment_viz,
    source_downsample: int,
    media_downsample: int,
) -> dict[str, dict[str, object]]:
    if not segment_viz:
        return {}

    epoch_dir = None
    if root_dir is not None:
        epoch_dir = Path(root_dir) / str(split_name) / f"epoch_{int(epoch):04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

    logged_images: dict[str, dict[str, object]] = {}
    for segment_id, payload in dict(segment_viz).items():
        segment_key = str(segment_id).replace("/", "__")
        preview_u8 = downsample_preview_for_media(
            np.asarray(payload["img_u8"], dtype=np.uint8),
            source_downsample=int(source_downsample),
            media_downsample=int(media_downsample),
        )
        logged_images[f"{split_name}/{segment_key}"] = {
            "image": preview_u8,
            "caption": f"{segment_id} ({split_name} ds={int(source_downsample)})",
        }

        if epoch_dir is None:
            continue
        base_path = epoch_dir / segment_key
        write_preview_png(out_path=base_path.with_suffix(".png"), image_u8=preview_u8)
        meta_payload = {}
        if "meta" in payload:
            meta_payload["meta"] = to_plain(payload["meta"])
        if "loss_components" in payload:
            meta_payload["loss_components"] = to_plain(payload["loss_components"])
        if meta_payload:
            base_path.with_suffix(".json").write_text(
                json.dumps(meta_payload, indent=2, sort_keys=False) + "\n",
                encoding="utf-8",
            )
    return logged_images


def export_store_preview_artifacts(
    *,
    store,
    media_downsample: int,
    split_name: str = "stitch_eval",
) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    if store is None:
        return {}, {}
    if not callable(getattr(store, "segment_ids", None)):
        return {}, {}
    if not callable(getattr(store, "full_segment_prob_preview_u8", None)):
        return {}, {}
    if not callable(getattr(store, "write_full_segment_preview_png", None)):
        return {}, {}

    preview_paths: dict[str, str] = {}
    logged_images: dict[str, dict[str, object]] = {}
    source_downsample = int(getattr(store, "downsample", 1))
    for segment_id in tuple(store.segment_ids()):
        segment_id = str(segment_id)
        segment_key = segment_id.replace("/", "__")
        preview_u8 = store.full_segment_prob_preview_u8(
            segment_id=segment_id,
            media_downsample=int(media_downsample),
        )
        preview_paths[segment_id] = store.write_full_segment_preview_png(
            segment_id=segment_id,
            media_downsample=int(media_downsample),
            image_u8=preview_u8,
        )
        logged_images[f"{split_name}/{segment_key}"] = {
            "image": preview_u8,
            "caption": f"{segment_id} (val ds={source_downsample})",
        }
    return preview_paths, logged_images


def export_store_artifacts(
    *,
    store,
    media_downsample: int,
    split_name: str = "stitch_eval",
) -> tuple[str | None, dict[str, str] | None, dict[str, str] | None, dict[str, dict[str, object]]]:
    if store is None or getattr(store, "root_dir", None) is None:
        return None, None, None, {}
    if not callable(getattr(store, "segment_ids", None)):
        return str(store.root_dir), None, None, {}
    if not callable(getattr(store, "write_full_segment_probs", None)):
        return str(store.root_dir), None, None, {}

    store_root_dir = str(store.root_dir)
    segment_ids = tuple(store.segment_ids())
    if not segment_ids:
        return store_root_dir, None, None, {}

    segment_prob_paths = {
        str(segment_id): store.write_full_segment_probs(segment_id=str(segment_id))
        for segment_id in segment_ids
    }
    segment_preview_paths, logged_images = export_store_preview_artifacts(
        store=store,
        media_downsample=int(media_downsample),
        split_name=split_name,
    )
    return store_root_dir, segment_prob_paths, segment_preview_paths or None, logged_images
