from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import yaml
import zarr

from ink.recipes.data.layout import NestedZarrLayout
from ink.recipes.data.masks import SUPERVISION_MASK_NAME, VALIDATION_MASK_NAME
from ink.recipes.data.zarr_data import ZarrPatchDataRecipe
from ink.recipes.data.zarr_io import (
    read_label_region,
    read_optional_supervision_mask_for_shape,
    resolve_segment_volume,
)

_PROGRESS_LOG = logging.getLogger("ink.progress")
_PATCH_BUNDLE_SCHEMA_VERSION = 3


def _log(message: str) -> None:
    _PROGRESS_LOG.info(str(message))


def _segment_dir(bundle_root: str | Path, *, group_name: str, segment_id: str) -> Path:
    return Path(bundle_root).expanduser().resolve() / str(group_name) / str(segment_id)


def _segment_manifest_path(bundle_root: str | Path, *, group_name: str, segment_id: str) -> Path:
    return _segment_dir(bundle_root, group_name=group_name, segment_id=segment_id) / "manifest.yaml"


def _bundle_patch_index_cache_dir(bundle_root: str | Path) -> Path:
    return Path(bundle_root).expanduser().resolve() / ".patch_index_cache"


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)


def _source_segment_config(recipe: ZarrPatchDataRecipe, segment_id: str) -> dict[str, Any]:
    raw_config = recipe.segments[str(segment_id)]
    return dict(raw_config or {})


def _ordered_segment_ids(recipe: ZarrPatchDataRecipe) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *(str(segment_id) for segment_id in recipe.train_segment_ids),
                *(str(segment_id) for segment_id in recipe.val_segment_ids),
            )
        )
    )


def _source_segment_fingerprint(
    *,
    layout: NestedZarrLayout,
    recipe: ZarrPatchDataRecipe,
    segment_id: str,
    mask_names,
) -> str:
    payload = {
        "schema_version": int(_PATCH_BUNDLE_SCHEMA_VERSION),
        "source_dataset_root": str(recipe.dataset_root),
        "segment_id": str(segment_id),
        "source_segment_config": _source_segment_config(recipe, segment_id),
        "in_channels": int(recipe.in_channels),
        "label_suffix": str(recipe.label_suffix),
        "mask_suffix": str(recipe.mask_suffix),
        "mask_names": [str(mask_name) for mask_name in tuple(mask_names or ())],
        "source_metadata": layout.segment_source_metadata_fingerprint(
            str(segment_id),
            label_suffix=str(recipe.label_suffix),
            mask_suffix=str(recipe.mask_suffix),
            mask_names=mask_names,
        ),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _segment_meta_by_id(
    recipe: ZarrPatchDataRecipe,
    *,
    layout: NestedZarrLayout,
    segment_ids,
) -> dict[str, dict[str, Any]]:
    segment_meta: dict[str, dict[str, Any]] = {}
    for segment_id in segment_ids:
        segment_id = str(segment_id)
        available_mask_names = layout.resolve_existing_artifact_names(
            segment_id,
            artifact_names=(SUPERVISION_MASK_NAME, VALIDATION_MASK_NAME),
            suffix=str(recipe.mask_suffix),
        )
        if not available_mask_names:
            raise FileNotFoundError(
                f"{segment_id}: expected at least one of "
                f"{SUPERVISION_MASK_NAME!r} or {VALIDATION_MASK_NAME!r}"
            )
        segment_meta[segment_id] = {
            "group_name": str(layout.resolve_group_name(segment_id)),
            "available_mask_names": available_mask_names,
            "source_fingerprint": _source_segment_fingerprint(
                layout=layout,
                recipe=recipe,
                segment_id=segment_id,
                mask_names=available_mask_names,
            ),
        }
    return segment_meta


def _chunk_hw(shape_hw: tuple[int, int]) -> tuple[int, int]:
    image_h, image_w = (int(shape_hw[0]), int(shape_hw[1]))
    return max(1, min(image_h, 256)), max(1, min(image_w, 256))


def _create_zarr_array(path: Path, *, shape: tuple[int, ...], chunks: tuple[int, ...], dtype) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    return zarr.open(
        str(path),
        mode="w",
        shape=tuple(int(value) for value in shape),
        chunks=tuple(int(value) for value in chunks),
        dtype=dtype,
        fill_value=0,
    )


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    mask_bool = np.asarray(mask) > 0
    if not bool(mask_bool.any()):
        return None
    ys, xs = np.nonzero(mask_bool)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _iter_chunk_windows(shape_hw: tuple[int, int], *, chunk_hw: tuple[int, int], bbox) -> tuple[tuple[int, int, int, int], ...]:
    image_h, image_w = (int(shape_hw[0]), int(shape_hw[1]))
    if bbox is None:
        return ()
    chunk_h, chunk_w = (int(chunk_hw[0]), int(chunk_hw[1]))
    y0, y1, x0, x1 = (int(value) for value in bbox)
    start_y = max(0, (y0 // chunk_h) * chunk_h)
    stop_y = min(image_h, ((y1 + chunk_h - 1) // chunk_h) * chunk_h)
    start_x = max(0, (x0 // chunk_w) * chunk_w)
    stop_x = min(image_w, ((x1 + chunk_w - 1) // chunk_w) * chunk_w)
    windows: list[tuple[int, int, int, int]] = []
    for win_y0 in range(start_y, stop_y, chunk_h):
        win_y1 = min(image_h, win_y0 + chunk_h)
        for win_x0 in range(start_x, stop_x, chunk_w):
            win_x1 = min(image_w, win_x0 + chunk_w)
            windows.append((win_y0, win_y1, win_x0, win_x1))
    return tuple(windows)


def _load_segment_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return dict(loaded) if isinstance(loaded, dict) else {}


def _expected_segment_files(
    segment_dir: Path,
    *,
    segment_id: str,
    available_mask_names,
) -> tuple[Path, ...]:
    return (
        segment_dir / f"{segment_id}.zarr",
        segment_dir / f"{segment_id}_inklabels.zarr",
        *(
            segment_dir / f"{segment_id}_{mask_name}.zarr"
            for mask_name in tuple(available_mask_names or ())
        ),
        segment_dir / "manifest.yaml",
    )


def _existing_bundle_segment_dirs(bundle_root: Path) -> tuple[Path, ...]:
    if not bundle_root.exists():
        return ()
    segment_dirs: list[Path] = []
    for group_dir in sorted(bundle_root.iterdir(), key=lambda path: path.name):
        if not group_dir.is_dir() or group_dir.name.startswith("."):
            continue
        for segment_dir in sorted(group_dir.iterdir(), key=lambda path: path.name):
            if not segment_dir.is_dir():
                continue
            segment_dirs.append(segment_dir)
    return tuple(segment_dirs)


@dataclass(frozen=True)
class PatchBundleWriter:
    source: ZarrPatchDataRecipe

    def ensure(self, *, out_root: str | Path) -> str:
        recipe = self.source
        if not isinstance(recipe, ZarrPatchDataRecipe):
            raise TypeError("PatchBundleWriter source must be ZarrPatchDataRecipe")

        out_root = Path(out_root).expanduser().resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        layout = NestedZarrLayout(recipe.dataset_root)
        segment_ids = _ordered_segment_ids(recipe)
        segment_meta = _segment_meta_by_id(recipe, layout=layout, segment_ids=segment_ids)
        desired_dirs: dict[str, Path] = {}
        volume_cache = {}

        reused = 0
        written = 0
        for segment_id in segment_ids:
            current_segment_meta = segment_meta[str(segment_id)]
            group_name = str(current_segment_meta["group_name"])
            available_mask_names = tuple(str(name) for name in tuple(current_segment_meta["available_mask_names"]))
            source_fingerprint = str(current_segment_meta["source_fingerprint"])
            segment_dir = _segment_dir(out_root, group_name=group_name, segment_id=segment_id)
            desired_dirs[str(segment_id)] = segment_dir
            if self._segment_matches_source(
                out_root=out_root,
                segment_id=str(segment_id),
                group_name=group_name,
                available_mask_names=available_mask_names,
                source_fingerprint=source_fingerprint,
            ):
                reused += 1
                _log(f"[bundle] reuse segment={segment_id} out={segment_dir}")
                continue
            self._write_segment(
                out_root=out_root,
                layout=layout,
                segment_id=str(segment_id),
                group_name=group_name,
                available_mask_names=available_mask_names,
                source_fingerprint=source_fingerprint,
                volume_cache=volume_cache,
            )
            written += 1
            _log(f"[bundle] wrote segment={segment_id} out={segment_dir}")

        self._remove_stale_segments(out_root=out_root, desired_dirs=desired_dirs)
        if written > 0:
            patch_index_cache_dir = _bundle_patch_index_cache_dir(out_root)
            if patch_index_cache_dir.exists():
                shutil.rmtree(patch_index_cache_dir)
        _log(f"[bundle] ready out={out_root} reused={reused} written={written}")
        return str(out_root)

    def write(self, *, out_root: str | Path) -> str:
        out_root = Path(out_root).expanduser().resolve()
        if out_root.exists():
            for stale_segment_dir in _existing_bundle_segment_dirs(out_root):
                shutil.rmtree(stale_segment_dir)
            patch_index_cache_dir = _bundle_patch_index_cache_dir(out_root)
            if patch_index_cache_dir.exists():
                shutil.rmtree(patch_index_cache_dir)
        return self.ensure(out_root=out_root)

    def _segment_matches_source(
        self,
        *,
        out_root: Path,
        segment_id: str,
        group_name: str,
        available_mask_names,
        source_fingerprint: str,
    ) -> bool:
        segment_dir = _segment_dir(out_root, group_name=group_name, segment_id=segment_id)
        if not all(
            path.exists()
            for path in _expected_segment_files(
                segment_dir,
                segment_id=segment_id,
                available_mask_names=available_mask_names,
            )
        ):
            return False
        manifest = _load_segment_manifest(_segment_manifest_path(out_root, group_name=group_name, segment_id=segment_id))
        return (
            int(manifest.get("schema_version", -1)) == int(_PATCH_BUNDLE_SCHEMA_VERSION)
            and str(manifest.get("recipe_family")) == "masked_zarr_segment"
            and str(manifest.get("segment_id")) == str(segment_id)
            and str(manifest.get("group_name")) == str(group_name)
            and tuple(str(name) for name in tuple(manifest.get("available_mask_names") or ())) == tuple(available_mask_names)
            and str(manifest.get("source_fingerprint")) == str(source_fingerprint)
        )

    def _write_segment(
        self,
        *,
        out_root: Path,
        layout: NestedZarrLayout,
        segment_id: str,
        group_name: str,
        available_mask_names,
        source_fingerprint: str,
        volume_cache,
    ) -> None:
        recipe = self.source
        segment_dir = _segment_dir(out_root, group_name=group_name, segment_id=segment_id)
        if segment_dir.exists():
            shutil.rmtree(segment_dir)
        segment_dir.mkdir(parents=True, exist_ok=True)

        volume = resolve_segment_volume(
            layout=layout,
            segments=recipe.segments,
            segment_id=str(segment_id),
            in_channels=int(recipe.in_channels),
            volume_cache=volume_cache,
        )
        image_shape_hw = tuple(int(value) for value in volume.image_shape_hw)
        full_bbox = (0, int(image_shape_hw[0]), 0, int(image_shape_hw[1]))
        label = np.asarray(
            read_label_region(
                layout,
                str(segment_id),
                image_shape_hw,
                full_bbox,
                label_suffix=str(recipe.label_suffix),
            ),
            dtype=np.uint8,
        )
        supervision_mask = read_optional_supervision_mask_for_shape(
            layout,
            str(segment_id),
            image_shape_hw,
            mask_suffix=str(recipe.mask_suffix),
            mask_names=(SUPERVISION_MASK_NAME,),
        )
        validation_mask = read_optional_supervision_mask_for_shape(
            layout,
            str(segment_id),
            image_shape_hw,
            mask_suffix=str(recipe.mask_suffix),
            mask_names=(VALIDATION_MASK_NAME,),
        )
        if supervision_mask is None and validation_mask is None:
            raise FileNotFoundError(
                f"{segment_id}: expected at least one of "
                f"{SUPERVISION_MASK_NAME!r} or {VALIDATION_MASK_NAME!r}"
            )
        if supervision_mask is not None:
            supervision_mask = np.asarray(supervision_mask, dtype=np.uint8)
        if validation_mask is not None:
            validation_mask = np.asarray(validation_mask, dtype=np.uint8)
        bundle_region = np.zeros(image_shape_hw, dtype=np.uint8)
        if supervision_mask is not None:
            bundle_region = np.maximum(bundle_region, supervision_mask)
        if validation_mask is not None:
            bundle_region = np.maximum(bundle_region, validation_mask)
        chunk_h, chunk_w = _chunk_hw(image_shape_hw)

        volume_arr = _create_zarr_array(
            segment_dir / f"{segment_id}.zarr",
            shape=(int(recipe.in_channels), int(image_shape_hw[0]), int(image_shape_hw[1])),
            chunks=(int(recipe.in_channels), int(chunk_h), int(chunk_w)),
            dtype=np.uint8,
        )
        label_arr = _create_zarr_array(
            segment_dir / f"{segment_id}_inklabels.zarr",
            shape=(int(image_shape_hw[0]), int(image_shape_hw[1])),
            chunks=(int(chunk_h), int(chunk_w)),
            dtype=np.uint8,
        )
        supervision_arr = None
        if supervision_mask is not None:
            supervision_arr = _create_zarr_array(
                segment_dir / f"{segment_id}_supervision_mask.zarr",
                shape=(int(image_shape_hw[0]), int(image_shape_hw[1])),
                chunks=(int(chunk_h), int(chunk_w)),
                dtype=np.uint8,
            )
        validation_arr = None
        if validation_mask is not None:
            validation_arr = _create_zarr_array(
                segment_dir / f"{segment_id}_validation_mask.zarr",
                shape=(int(image_shape_hw[0]), int(image_shape_hw[1])),
                chunks=(int(chunk_h), int(chunk_w)),
                dtype=np.uint8,
            )

        for y0, y1, x0, x1 in _iter_chunk_windows(image_shape_hw, chunk_hw=(chunk_h, chunk_w), bbox=_mask_bbox(bundle_region)):
            region_mask = np.asarray(bundle_region[y0:y1, x0:x1] > 0, dtype=np.uint8)
            if not bool(region_mask.any()):
                continue
            image_patch = np.asarray(volume.read_patch(y0, y1, x0, x1), dtype=np.uint8)
            image_patch *= region_mask[..., None]
            volume_arr[:, y0:y1, x0:x1] = np.transpose(image_patch, (2, 0, 1))
            label_arr[y0:y1, x0:x1] = np.asarray(label[y0:y1, x0:x1], dtype=np.uint8) * region_mask
            if supervision_arr is not None and supervision_mask is not None:
                supervision_arr[y0:y1, x0:x1] = np.asarray(supervision_mask[y0:y1, x0:x1], dtype=np.uint8)
            if validation_arr is not None and validation_mask is not None:
                validation_arr[y0:y1, x0:x1] = np.asarray(validation_mask[y0:y1, x0:x1], dtype=np.uint8)

        _write_manifest(
            segment_dir / "manifest.yaml",
            {
                "schema_version": int(_PATCH_BUNDLE_SCHEMA_VERSION),
                "recipe_family": "masked_zarr_segment",
                "segment_id": str(segment_id),
                "group_name": str(group_name),
                "available_mask_names": [str(mask_name) for mask_name in tuple(available_mask_names or ())],
                "source_dataset_root": str(recipe.dataset_root),
                "source_fingerprint": str(source_fingerprint),
                "source_segment_config": _source_segment_config(recipe, segment_id),
                "source_in_channels": int(recipe.in_channels),
                "source_label_suffix": str(recipe.label_suffix),
                "source_mask_suffix": str(recipe.mask_suffix),
                "shape": [int(image_shape_hw[0]), int(image_shape_hw[1])],
            },
        )

    def _remove_stale_segments(self, *, out_root: Path, desired_dirs: dict[str, Path]) -> None:
        for segment_dir in _existing_bundle_segment_dirs(out_root):
            segment_id = str(segment_dir.name)
            desired_dir = desired_dirs.get(segment_id)
            if desired_dir is not None and desired_dir == segment_dir:
                continue
            shutil.rmtree(segment_dir)


__all__ = [
    "PatchBundleWriter",
]
