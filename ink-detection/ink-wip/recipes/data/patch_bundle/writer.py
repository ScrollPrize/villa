from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import yaml
import zarr

from ink.recipes.data.grouped_zarr_data import GroupedZarrPatchDataRecipe
from ink.recipes.data.zarr_data import ZarrPatchDataRecipe, _load_raw_patch_sample

_PROGRESS_LOG = logging.getLogger("ink.progress")
_PATCH_BUNDLE_SCHEMA_VERSION = 1


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


def _source_fingerprint(*, layout, recipe, split_segment_ids: tuple[str, ...]) -> str:
    payload = {
        "schema_version": int(_PATCH_BUNDLE_SCHEMA_VERSION),
        "dataset_root": str(recipe.dataset_root),
        "segments": {
            str(segment_id): layout.segment_source_fingerprint(
                str(segment_id),
                label_suffix=str(recipe.label_suffix),
                mask_suffix=str(recipe.mask_suffix),
            )
            for segment_id in split_segment_ids
        },
        "extraction": {
            "in_channels": int(recipe.in_channels),
            "patch_size": int(recipe.patch_size),
            "tile_size": int(recipe.patch_size if recipe.tile_size is None else recipe.tile_size),
            "stride": int(recipe.patch_size if recipe.stride is None else recipe.stride),
            "label_suffix": str(recipe.label_suffix),
            "mask_suffix": str(recipe.mask_suffix),
            "segments": {
                str(segment_id): dict(recipe.segments[str(segment_id)])
                for segment_id in split_segment_ids
            },
        },
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


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
    source: ZarrPatchDataRecipe | GroupedZarrPatchDataRecipe

    def ensure(self, *, out_root: str | Path) -> dict[str, str]:
        out_root = Path(out_root).expanduser().resolve()
        if _bundle_exists(out_root):
            _log(f"[bundle] reuse out={out_root}")
            return {
                "train": str(out_root / "train"),
                "valid": str(out_root / "valid"),
            }
        return self.write(out_root=out_root)

    def write(self, *, out_root: str | Path) -> dict[str, str]:
        recipe = self.source
        if not isinstance(recipe, (ZarrPatchDataRecipe, GroupedZarrPatchDataRecipe)):
            raise TypeError("PatchBundleWriter source must be ZarrPatchDataRecipe or GroupedZarrPatchDataRecipe")

        layout = recipe._build_layout()
        segments = recipe.segments
        patch_size = int(recipe.patch_size)
        in_channels = int(recipe.in_channels)
        tile_size = patch_size if recipe.tile_size is None else int(recipe.tile_size)
        stride = patch_size if recipe.stride is None else int(recipe.stride)
        volume_cache = {}
        mask_cache = {}
        samples_by_split = recipe._build_samples_by_split(
            layout=layout,
            segments=segments,
            in_channels=in_channels,
            patch_size=patch_size,
            tile_size=tile_size,
            stride=stride,
            volume_cache=volume_cache,
            mask_cache=mask_cache,
        )
        split_segment_ids = recipe._split_segment_ids()
        out_root = Path(out_root).expanduser().resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        recipe_family = "grouped_patch" if isinstance(recipe, GroupedZarrPatchDataRecipe) else "patch"
        train_group_counts = (
            recipe._build_group_counts(samples_by_split=samples_by_split)
            if isinstance(recipe, GroupedZarrPatchDataRecipe)
            else None
        )

        written = {}
        for split_name, samples in samples_by_split.items():
            split_dir = out_root / str(split_name)
            patches_root = _bundle_patches_root(out_root, split=split_name)
            manifest_path = _bundle_manifest_path(out_root, split=split_name)
            split_dir.mkdir(parents=True, exist_ok=True)
            _log(f"[bundle] writing split={split_name} samples={len(samples)} out={split_dir}")

            store = zarr.open_group(str(patches_root), mode="w")
            x_arr = _create_array(store, "x", shape=(len(samples), patch_size, patch_size, in_channels), dtype="u1")
            y_arr = _create_array(store, "y", shape=(len(samples), patch_size, patch_size, 1), dtype="u1")
            valid_arr = _create_array(store, "valid_mask", shape=(len(samples), patch_size, patch_size, 1), dtype="u1")
            xyxy_arr = _create_array(store, "xyxy", shape=(len(samples), 4), dtype="i8")
            segment_index_arr = _create_array(store, "segment_index", shape=(len(samples),), dtype="i4")
            group_idx_arr = None
            if recipe_family == "grouped_patch":
                group_idx_arr = _create_array(store, "group_idx", shape=(len(samples),), dtype="i8")

            ordered_segment_ids = tuple(str(segment_id) for segment_id in split_segment_ids[split_name])
            segment_index_by_id = {segment_id: idx for idx, segment_id in enumerate(ordered_segment_ids)}
            for idx, sample in enumerate(samples):
                image, label, valid_mask, xyxy, segment_id = _load_raw_patch_sample(
                    layout=layout,
                    segments=segments,
                    sample=sample,
                    in_channels=in_channels,
                    label_suffix=recipe.label_suffix,
                    mask_suffix=recipe.mask_suffix,
                    volume_cache=volume_cache,
                    mask_cache=mask_cache,
                )
                x_arr[idx] = image
                y_arr[idx] = label
                valid_arr[idx] = valid_mask
                xyxy_arr[idx] = xyxy
                segment_index_arr[idx] = int(segment_index_by_id[segment_id])
                if group_idx_arr is not None:
                    group_idx_arr[idx] = int(sample[4])

            manifest = {
                "schema_version": int(_PATCH_BUNDLE_SCHEMA_VERSION),
                "recipe_family": recipe_family,
                "split": str(split_name),
                "source_fingerprint": _source_fingerprint(
                    layout=layout,
                    recipe=recipe,
                    split_segment_ids=split_segment_ids[split_name],
                ),
                "dataset_root": str(recipe.dataset_root),
                "segment_ids": [str(segment_id) for segment_id in ordered_segment_ids],
                "counts": {
                    "samples": int(len(samples)),
                    "segments": int(len(ordered_segment_ids)),
                },
                "extraction": {
                    "in_channels": in_channels,
                    "patch_size": patch_size,
                    "tile_size": tile_size,
                    "stride": stride,
                    "label_suffix": str(recipe.label_suffix),
                    "mask_suffix": str(recipe.mask_suffix),
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
                },
            }
            if group_idx_arr is not None:
                manifest["arrays"]["group_idx"] = {
                    "shape": list(group_idx_arr.shape),
                    "dtype": str(group_idx_arr.dtype),
                }
            _write_manifest(manifest_path, manifest)
            written[split_name] = str(split_dir)
        return written


__all__ = [
    "PatchBundleWriter",
    "load_patch_bundle_manifest",
]
