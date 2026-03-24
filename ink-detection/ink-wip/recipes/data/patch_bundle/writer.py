from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import yaml
import zarr

from ink.recipes.data.zarr_io import resolve_segment_volume
from ink.recipes.data.zarr_data import (
    ZarrDataContext,
    ZarrPatchDataRecipe,
    build_zarr_split_samples,
    load_raw_zarr_patch,
)

_PROGRESS_LOG = logging.getLogger("ink.progress")
_PATCH_BUNDLE_SCHEMA_VERSION = 2


def _log(message: str) -> None:
    _PROGRESS_LOG.info(str(message))


def _bundle_manifest_path(bundle_root: str | Path, *, split: str) -> Path:
    return Path(bundle_root).expanduser().resolve() / str(split) / "manifest.yaml"


def _bundle_patches_root(bundle_root: str | Path, *, split: str) -> Path:
    return Path(bundle_root).expanduser().resolve() / str(split) / "patches.zarr"


def _bundle_exists(bundle_root: str | Path) -> bool:
    root = Path(bundle_root).expanduser().resolve()
    for split in ("train", "valid"):
        if not _bundle_manifest_path(root, split=split).exists():
            return False
        if not _bundle_patches_root(root, split=split).exists():
            return False
    return True


def _bundle_matches_source(
    *,
    bundle_root: str | Path,
    recipe: ZarrPatchDataRecipe,
) -> bool:
    if not _bundle_exists(bundle_root):
        return False
    split_segment_ids = recipe.split_segment_ids()
    for split_name in ("train", "valid"):
        context = ZarrDataContext.from_recipe_split(recipe, split_name=split_name)
        manifest = load_patch_bundle_manifest(bundle_root, split=split_name)
        if str(manifest.get("recipe_family")) != "patch":
            return False
        expected_fingerprint = _source_fingerprint(
            layout=context.layout,
            context=context,
            recipe=recipe,
            split_segment_ids=split_segment_ids[split_name],
        )
        if str(manifest.get("source_fingerprint")) != expected_fingerprint:
            return False
    return True


def load_patch_bundle_manifest(bundle_root: str | Path, *, split: str) -> dict[str, Any]:
    path = _bundle_manifest_path(bundle_root, split=split)
    if not path.exists():
        raise FileNotFoundError(f"Could not resolve patch bundle manifest at {str(path)!r}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"patch bundle manifest must decode to a dict, got {type(loaded).__name__}")
    return dict(loaded)


def _normalization_manifest(recipe) -> dict[str, Any]:
    return {
        "type": type(recipe).__name__,
    }


def _source_fingerprint(*, layout, context: ZarrDataContext, recipe, split_segment_ids: tuple[str, ...]) -> str:
    payload = {
        "schema_version": int(_PATCH_BUNDLE_SCHEMA_VERSION),
        "dataset_root": str(recipe.dataset_root),
        "segments": {
            str(segment_id): layout.segment_source_metadata_fingerprint(
                str(segment_id),
                label_suffix=str(context.label_suffix),
                mask_suffix=str(context.mask_suffix),
                mask_names=context.mask_names_for_segment(segment_id),
            )
            for segment_id in split_segment_ids
        },
        "extraction": {
            "in_channels": int(recipe.in_channels),
            "patch_size": int(recipe.patch_size),
            "tile_size": int(recipe.patch_size if recipe.tile_size is None else recipe.tile_size),
            "stride": int(recipe.patch_size if recipe.stride is None else recipe.stride),
            "label_suffix": str(context.label_suffix),
            "mask_suffix": str(context.mask_suffix),
            "mask_name": str(context.mask_name),
            "train_segment_ids": sorted(str(segment_id) for segment_id in context.train_segment_ids),
            "segments": {
                str(segment_id): dict(recipe.segments[str(segment_id)])
                for segment_id in split_segment_ids
            },
        },
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _segment_specs_for_manifest(*, context: ZarrDataContext, split_segment_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    segment_specs: list[dict[str, Any]] = []
    for segment_id in split_segment_ids:
        volume = resolve_segment_volume(
            layout=context.layout,
            segments=context.segments,
            segment_id=str(segment_id),
            in_channels=int(context.in_channels),
            volume_cache=context.volume_cache,
        )
        mask_names = context.mask_names_for_segment(segment_id)
        label_mask_store = context.label_mask_store_cache.get(
            (str(segment_id), str(context.label_suffix), str(context.mask_suffix), mask_names)
        )
        bbox_rows = None
        if label_mask_store is not None and getattr(label_mask_store, "bbox_rows", None):
            bbox_rows = [
                [int(value) for value in row]
                for row in tuple(getattr(label_mask_store, "bbox_rows") or ())
            ]
        segment_specs.append(
            {
                "segment_id": str(segment_id),
                "shape": [int(value) for value in volume.image_shape_hw],
                "bbox": bbox_rows,
            }
        )
    return segment_specs


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)


def _create_array(root, name: str, *, shape: tuple[int, ...], dtype) -> Any:
    chunks = [1]
    chunks.extend(max(1, min(int(dim), 256)) for dim in shape[1:])
    return root.create_dataset(
        name,
        shape=tuple(int(v) for v in shape),
        chunks=tuple(chunks),
        dtype=dtype,
        overwrite=True,
    )


@dataclass(frozen=True)
class PatchBundleWriter:
    source: ZarrPatchDataRecipe

    def ensure(self, *, out_root: str | Path) -> dict[str, str]:
        out_root = Path(out_root).expanduser().resolve()
        if _bundle_matches_source(bundle_root=out_root, recipe=self.source):
            _log(f"[bundle] reuse out={out_root}")
            return {
                "train": str(out_root / "train"),
                "valid": str(out_root / "valid"),
            }
        if _bundle_exists(out_root):
            _log(f"[bundle] rebuild stale out={out_root}")
        return self.write(out_root=out_root)

    def write(self, *, out_root: str | Path) -> dict[str, str]:
        recipe = self.source
        if not isinstance(recipe, ZarrPatchDataRecipe):
            raise TypeError("PatchBundleWriter source must be ZarrPatchDataRecipe")

        patch_size = int(recipe.patch_size)
        tile_size = patch_size if recipe.tile_size is None else int(recipe.tile_size)
        stride = patch_size if recipe.stride is None else int(recipe.stride)
        split_segment_ids = recipe.split_segment_ids()
        all_segment_ids = tuple(
            str(segment_id)
            for segment_ids in split_segment_ids.values()
            for segment_id in segment_ids
        )
        layout = ZarrDataContext.from_recipe_split(recipe, split_name="train").layout
        shared_volume_cache = {}
        shared_label_mask_store_cache = {}
        contexts = {
            split_name: ZarrDataContext.from_recipe_split(
                recipe,
                split_name=split_name,
                layout=layout,
                volume_cache=shared_volume_cache,
                label_mask_store_cache=shared_label_mask_store_cache,
            )
            for split_name in ("train", "valid")
        }
        samples_by_split = {
            split_name: build_zarr_split_samples(
                contexts[split_name],
                segment_ids=split_segment_ids[split_name],
                patch_size=patch_size,
                tile_size=tile_size,
                stride=stride,
                split_name=split_name,
                build_workers=max(0, int(recipe.num_workers)),
                group_segment_ids=all_segment_ids,
            )
            for split_name in ("train", "valid")
        }
        out_root = Path(out_root).expanduser().resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        recipe_family = "patch"
        train_group_counts = (recipe.extras or {}).get("group_counts")
        if train_group_counts is not None:
            train_group_counts = [int(value) for value in train_group_counts]
            if not train_group_counts:
                train_group_counts = None

        written = {}
        for split_name, samples in samples_by_split.items():
            context = contexts[split_name]
            split_dir = out_root / str(split_name)
            patches_root = _bundle_patches_root(out_root, split=split_name)
            manifest_path = _bundle_manifest_path(out_root, split=split_name)
            split_dir.mkdir(parents=True, exist_ok=True)
            _log(f"[bundle] writing split={split_name} samples={len(samples)} out={split_dir}")

            store = zarr.open_group(str(patches_root), mode="w")
            x_arr = _create_array(
                store,
                "x",
                shape=(len(samples), patch_size, patch_size, int(context.in_channels)),
                dtype="u1",
            )
            y_arr = _create_array(store, "y", shape=(len(samples), patch_size, patch_size, 1), dtype="u1")
            valid_arr = _create_array(store, "valid_mask", shape=(len(samples), patch_size, patch_size, 1), dtype="u1")
            xyxy_arr = _create_array(store, "xyxy", shape=(len(samples), 4), dtype="i8")
            segment_index_arr = _create_array(store, "segment_index", shape=(len(samples),), dtype="i4")
            group_idx_arr = _create_array(store, "group_idx", shape=(len(samples),), dtype="i8")

            ordered_segment_ids = tuple(str(segment_id) for segment_id in split_segment_ids[split_name])
            segment_index_by_id = {segment_id: idx for idx, segment_id in enumerate(ordered_segment_ids)}
            for idx, sample in enumerate(samples):
                image, label, valid_mask, xyxy, segment_id = load_raw_zarr_patch(
                    context,
                    sample=sample,
                    include_valid_mask=True,
                )
                x_arr[idx] = image
                y_arr[idx] = label
                valid_arr[idx] = valid_mask
                xyxy_arr[idx] = xyxy
                segment_index_arr[idx] = int(segment_index_by_id[segment_id])
                group_idx_arr[idx] = int(sample[3])

            manifest = {
                "schema_version": int(_PATCH_BUNDLE_SCHEMA_VERSION),
                "recipe_family": recipe_family,
                "split": str(split_name),
                "source_fingerprint": _source_fingerprint(
                    layout=context.layout,
                    context=context,
                    recipe=recipe,
                    split_segment_ids=split_segment_ids[split_name],
                ),
                "dataset_root": str(recipe.dataset_root),
                "segment_ids": [str(segment_id) for segment_id in ordered_segment_ids],
                "segment_specs": _segment_specs_for_manifest(
                    context=context,
                    split_segment_ids=ordered_segment_ids,
                ),
                "counts": {
                    "samples": int(len(samples)),
                    "segments": int(len(ordered_segment_ids)),
                },
                "extraction": {
                    "in_channels": int(context.in_channels),
                    "patch_size": patch_size,
                    "tile_size": tile_size,
                    "stride": stride,
                    "label_suffix": str(context.label_suffix),
                    "mask_suffix": str(context.mask_suffix),
                    "mask_name": str(context.mask_name),
                    "train_segment_ids": sorted(str(segment_id) for segment_id in context.train_segment_ids),
                },
                "normalization": _normalization_manifest(recipe.normalization),
                "normalization_stats": dict((recipe.extras or {}).get("normalization_stats") or {}),
                "group_counts": None if split_name != "train" else train_group_counts,
                "arrays": {
                    "x": {"shape": list(x_arr.shape), "dtype": str(x_arr.dtype)},
                    "y": {"shape": list(y_arr.shape), "dtype": str(y_arr.dtype)},
                    "valid_mask": {"shape": list(valid_arr.shape), "dtype": str(valid_arr.dtype)},
                    "xyxy": {"shape": list(xyxy_arr.shape), "dtype": str(xyxy_arr.dtype)},
                    "segment_index": {"shape": list(segment_index_arr.shape), "dtype": str(segment_index_arr.dtype)},
                    "group_idx": {"shape": list(group_idx_arr.shape), "dtype": str(group_idx_arr.dtype)},
                },
            }
            _write_manifest(manifest_path, manifest)
            written[split_name] = str(split_dir)
        return written


__all__ = [
    "PatchBundleWriter",
    "load_patch_bundle_manifest",
]
