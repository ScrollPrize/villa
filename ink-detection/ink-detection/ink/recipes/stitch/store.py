from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import zarr

from ink.recipes.stitch.ops import (
    gaussian_weights,
    normalize_xyxy_rows,
    resolve_buffer_crop,
    stitch_logit_map,
    stitch_prob_map,
)


def segment_store_key(segment_id: str) -> str:
    return str(segment_id).replace("/", "__")


def _downsample_shape(shape: tuple[int, int], *, downsample: int) -> tuple[int, int]:
    h, w = [int(v) for v in shape]
    ds = max(1, int(downsample))
    return ((h + ds - 1) // ds, (w + ds - 1) // ds)


def _chunk_shape(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = [max(1, int(v)) for v in shape]
    return (min(h, 256), min(w, 256))


def _chunk_bounds(
    *,
    full_shape: tuple[int, int],
    chunk_shape: tuple[int, int],
    chunk_y: int,
    chunk_x: int,
) -> tuple[int, int, int, int]:
    full_h, full_w = [int(v) for v in full_shape]
    chunk_h, chunk_w = [max(1, int(v)) for v in chunk_shape]
    y0 = int(chunk_y) * chunk_h
    x0 = int(chunk_x) * chunk_w
    y1 = min(int(full_h), y0 + chunk_h)
    x1 = min(int(full_w), x0 + chunk_w)
    return y0, y1, x0, x1


def _normalize_rois(
    rois: Any,
    *,
    full_shape: tuple[int, int],
) -> tuple[tuple[int, int, int, int], ...]:
    full_h, full_w = [int(v) for v in full_shape]
    if rois is None:
        return ((0, full_h, 0, full_w),)

    normalized = []
    for item in tuple(rois):
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            raise ValueError("segment_rois entries must have 4 values")
        y0, y1, x0, x1 = [int(v) for v in item]
        y0 = max(0, min(y0, full_h))
        y1 = max(0, min(y1, full_h))
        x0 = max(0, min(x0, full_w))
        x1 = max(0, min(x1, full_w))
        if y1 > y0 and x1 > x0:
            normalized.append((y0, y1, x0, x1))
    if not normalized:
        return ((0, full_h, 0, full_w),)
    return tuple(normalized)


def _open_sparse_array(path: Path, shape: tuple[int, int]) -> Any:
    return zarr.open(
        str(path),
        mode="w",
        shape=tuple(int(v) for v in shape),
        chunks=_chunk_shape(shape),
        dtype=np.float32,
        fill_value=0.0,
        write_empty_chunks=False,
    )


def _clamp_bbox(
    bbox: tuple[int, int, int, int] | None,
    *,
    full_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    full_h, full_w = [int(v) for v in full_shape]
    if bbox is None:
        return (0, full_h, 0, full_w)
    y0, y1, x0, x1 = [int(v) for v in bbox]
    return (
        max(0, min(y0, full_h)),
        max(0, min(y1, full_h)),
        max(0, min(x0, full_w)),
        max(0, min(x1, full_w)),
    )


def _segment_array_paths(root_dir: Path, segment_id: str) -> tuple[Path, Path]:
    key = segment_store_key(segment_id)
    return (
        root_dir / f"{key}__logit_sum.zarr",
        root_dir / f"{key}__weight_sum.zarr",
    )


def _prepare_patch_logits(patch_logits: torch.Tensor, *, target_h: int, target_w: int) -> torch.Tensor:
    if patch_logits.shape[-2:] == (int(target_h), int(target_w)):
        return patch_logits
    return F.interpolate(
        patch_logits,
        size=(int(target_h), int(target_w)),
        mode="bilinear",
        align_corners=False,
    )


def _is_full_segment_roi(
    roi_bbox: tuple[int, int, int, int],
    *,
    full_shape: tuple[int, int],
) -> bool:
    full_h, full_w = [int(v) for v in full_shape]
    return tuple(int(v) for v in roi_bbox) == (0, full_h, 0, full_w)


def _build_chunk_validity(
    rois: tuple[tuple[int, int, int, int], ...],
    *,
    full_shape: tuple[int, int],
    chunk_shape: tuple[int, int],
) -> dict[tuple[int, int], tuple[tuple[int, int, int, int], ...] | None] | None:
    if len(rois) == 1 and _is_full_segment_roi(rois[0], full_shape=full_shape):
        return None

    chunk_h, chunk_w = [max(1, int(v)) for v in chunk_shape]
    chunk_regions: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for y0, y1, x0, x1 in rois:
        chunk_y0 = int(y0) // chunk_h
        chunk_y1 = max(int(y0), int(y1) - 1) // chunk_h
        chunk_x0 = int(x0) // chunk_w
        chunk_x1 = max(int(x0), int(x1) - 1) // chunk_w
        for chunk_y in range(chunk_y0, chunk_y1 + 1):
            for chunk_x in range(chunk_x0, chunk_x1 + 1):
                chunk_top, chunk_bottom, chunk_left, chunk_right = _chunk_bounds(
                    full_shape=full_shape,
                    chunk_shape=chunk_shape,
                    chunk_y=int(chunk_y),
                    chunk_x=int(chunk_x),
                )
                local_y0 = max(int(y0), chunk_top) - chunk_top
                local_y1 = min(int(y1), chunk_bottom) - chunk_top
                local_x0 = max(int(x0), chunk_left) - chunk_left
                local_x1 = min(int(x1), chunk_right) - chunk_left
                if local_y1 <= local_y0 or local_x1 <= local_x0:
                    continue
                chunk_regions.setdefault((int(chunk_y), int(chunk_x)), []).append(
                    (int(local_y0), int(local_y1), int(local_x0), int(local_x1))
                )

    chunk_validity: dict[tuple[int, int], tuple[tuple[int, int, int, int], ...] | None] = {}
    for chunk_key, regions in chunk_regions.items():
        chunk_y, chunk_x = chunk_key
        chunk_top, chunk_bottom, chunk_left, chunk_right = _chunk_bounds(
            full_shape=full_shape,
            chunk_shape=chunk_shape,
            chunk_y=int(chunk_y),
            chunk_x=int(chunk_x),
        )
        chunk_height = int(chunk_bottom - chunk_top)
        chunk_width = int(chunk_right - chunk_left)
        unique_regions = tuple(dict.fromkeys(regions))
        if len(unique_regions) == 1 and unique_regions[0] == (0, chunk_height, 0, chunk_width):
            chunk_validity[(int(chunk_y), int(chunk_x))] = None
            continue
        chunk_validity[(int(chunk_y), int(chunk_x))] = unique_regions
    return chunk_validity


@dataclass
class _SegmentLayout:
    shape: tuple[int, int]
    chunk_shape: tuple[int, int]
    chunk_validity: dict[tuple[int, int], tuple[tuple[int, int, int, int], ...] | None] | None = None


@dataclass
class _ChunkCacheEntry:
    y0: int
    y1: int
    x0: int
    x1: int
    sum_chunk: np.ndarray
    weight_chunk: np.ndarray
    dirty: bool = False


@dataclass
class ZarrStitchStore:
    root_dir: str | Path | None = None
    downsample: int = 1
    gaussian_sigma_scale: float = 1.0 / 8.0
    gaussian_min_weight: float = 1e-6
    chunk_cache_max_chunks: int = 512

    _segment_layouts: dict[str, _SegmentLayout] = field(default_factory=dict, init=False, repr=False)
    _sum_arrays: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _weight_arrays: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _gaussian_cache: dict[tuple[int, int], np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _chunk_cache: OrderedDict[tuple[str, int, int], _ChunkCacheEntry] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )

    def build(
        self,
        *,
        segment_shapes: dict[str, tuple[int, int]],
        downsample: int | None = None,
        segment_rois: dict[str, tuple[tuple[int, int, int, int], ...]] | None = None,
    ) -> ZarrStitchStore:
        built = ZarrStitchStore(
            root_dir=self.root_dir,
            downsample=int(self.downsample if downsample is None else downsample),
            gaussian_sigma_scale=float(self.gaussian_sigma_scale),
            gaussian_min_weight=float(self.gaussian_min_weight),
            chunk_cache_max_chunks=int(self.chunk_cache_max_chunks),
        )
        built._configure_segments(segment_shapes, segment_rois=segment_rois)
        if built.root_dir is not None:
            built.reset()
        return built

    def _configure_segments(
        self,
        segment_shapes: dict[str, tuple[int, int]],
        *,
        segment_rois: dict[str, tuple[tuple[int, int, int, int], ...]] | None = None,
    ) -> None:
        rois_lookup = {
            str(segment_id): rois
            for segment_id, rois in dict(segment_rois or {}).items()
        }
        self._segment_layouts = {}
        for segment_id, shape in dict(segment_shapes).items():
            segment_id = str(segment_id)
            ds_shape = _downsample_shape(tuple(int(v) for v in shape), downsample=int(self.downsample))
            chunk_shape = _chunk_shape(ds_shape)
            rois = _normalize_rois(
                rois_lookup.get(segment_id),
                full_shape=ds_shape,
            )
            self._segment_layouts[segment_id] = _SegmentLayout(
                shape=ds_shape,
                chunk_shape=chunk_shape,
                chunk_validity=_build_chunk_validity(
                    rois,
                    full_shape=ds_shape,
                    chunk_shape=chunk_shape,
                ),
            )
        self._sum_arrays = {}
        self._weight_arrays = {}
        self._chunk_cache = OrderedDict()

    def _resolved_root_dir(self) -> Path:
        root_dir = self.root_dir
        if root_dir is None:
            raise ValueError("stitch store root_dir must be set before stitched inference begins")
        return Path(root_dir).resolve()

    def _require_configured_segments(self) -> None:
        if self._segment_layouts:
            return
        raise ValueError("stitch store must be built with segment shapes before use")

    def _open_root_arrays(self, root_dir: Path) -> None:
        for segment_id, layout in self._segment_layouts.items():
            self._sum_arrays[segment_id], self._weight_arrays[segment_id] = self._open_segment_arrays(
                root_dir,
                segment_id=segment_id,
                shape=layout.shape,
            )

    def reset(self) -> None:
        self._require_configured_segments()
        root_dir = self._resolved_root_dir()
        root_dir.mkdir(parents=True, exist_ok=True)
        self._sum_arrays = {}
        self._weight_arrays = {}
        self._chunk_cache = OrderedDict()
        self._open_root_arrays(root_dir)

    def _open_segment_arrays(self, root_dir: Path, *, segment_id: str, shape: tuple[int, int]) -> tuple[Any, Any]:
        sum_path, weight_path = _segment_array_paths(root_dir, segment_id)
        return _open_sparse_array(sum_path, shape), _open_sparse_array(weight_path, shape)

    def _segment_arrays(self, segment_id: str) -> tuple[Any, Any]:
        segment_id = str(segment_id)
        if not self._sum_arrays or not self._weight_arrays:
            raise RuntimeError("stitch store arrays are not initialized; set root_dir and call reset() before use")
        sum_arr = self._sum_arrays.get(segment_id)
        weight_arr = self._weight_arrays.get(segment_id)
        if sum_arr is None or weight_arr is None:
            raise KeyError(f"unknown stitched segment id {segment_id!r}")
        return sum_arr, weight_arr

    def _segment_layout(self, segment_id: str) -> _SegmentLayout:
        layout = self._segment_layouts.get(str(segment_id))
        if layout is None:
            raise KeyError(f"unknown stitched segment id {segment_id!r}")
        return layout

    def segment_ds_shape(self, segment_id: str) -> tuple[int, int]:
        return tuple(int(v) for v in self._segment_layout(segment_id).shape)

    def segment_ids(self) -> tuple[str, ...]:
        return tuple(str(segment_id) for segment_id in self._segment_layouts)

    def _write_chunk_entry(self, *, segment_id: str, entry: _ChunkCacheEntry) -> None:
        if not entry.dirty:
            return
        sum_arr, weight_arr = self._segment_arrays(segment_id)
        sum_arr[entry.y0:entry.y1, entry.x0:entry.x1] = entry.sum_chunk
        weight_arr[entry.y0:entry.y1, entry.x0:entry.x1] = entry.weight_chunk
        entry.dirty = False

    def flush(self) -> None:
        if not self._chunk_cache:
            return
        for key, entry in list(self._chunk_cache.items()):
            segment_id, _chunk_y, _chunk_x = key
            self._write_chunk_entry(segment_id=str(segment_id), entry=entry)
        self._chunk_cache = OrderedDict()

    def _segment_chunk_shape(self, segment_id: str) -> tuple[int, int]:
        return tuple(int(v) for v in self._segment_layout(segment_id).chunk_shape)

    def _evict_chunk_if_needed(self) -> None:
        max_chunks = max(1, int(self.chunk_cache_max_chunks))
        if len(self._chunk_cache) < max_chunks:
            return
        key, entry = self._chunk_cache.popitem(last=False)
        segment_id, _chunk_y, _chunk_x = key
        self._write_chunk_entry(segment_id=str(segment_id), entry=entry)

    def _chunk_entry(self, *, segment_id: str, chunk_y: int, chunk_x: int) -> _ChunkCacheEntry:
        key = (str(segment_id), int(chunk_y), int(chunk_x))
        entry = self._chunk_cache.get(key)
        if entry is not None:
            self._chunk_cache.move_to_end(key)
            return entry

        self._evict_chunk_if_needed()
        sum_arr, weight_arr = self._segment_arrays(segment_id)
        layout = self._segment_layout(segment_id)
        full_h, full_w = layout.shape
        chunk_h, chunk_w = layout.chunk_shape
        y0, y1, x0, x1 = _chunk_bounds(
            full_shape=(int(full_h), int(full_w)),
            chunk_shape=(int(chunk_h), int(chunk_w)),
            chunk_y=int(chunk_y),
            chunk_x=int(chunk_x),
        )
        entry = _ChunkCacheEntry(
            y0=y0,
            y1=y1,
            x0=x0,
            x1=x1,
            sum_chunk=np.array(sum_arr[y0:y1, x0:x1], dtype=np.float32, copy=True),
            weight_chunk=np.array(weight_arr[y0:y1, x0:x1], dtype=np.float32, copy=True),
        )
        self._chunk_cache[key] = entry
        return entry

    def _accumulate_entry_slices(
        self,
        *,
        entry: _ChunkCacheEntry,
        local_y0: int,
        local_y1: int,
        local_x0: int,
        local_x1: int,
        weighted_patch: np.ndarray,
        weight_patch: np.ndarray,
    ) -> None:
        if local_y1 <= local_y0 or local_x1 <= local_x0:
            return
        entry.sum_chunk[int(local_y0):int(local_y1), int(local_x0):int(local_x1)] += weighted_patch
        entry.weight_chunk[int(local_y0):int(local_y1), int(local_x0):int(local_x1)] += weight_patch
        entry.dirty = True

    def _accumulate_patch(
        self,
        *,
        segment_id: str,
        crop: dict[str, int],
        patch_crop_np: np.ndarray,
        weight_crop: np.ndarray,
    ) -> None:
        segment_id = str(segment_id)
        layout = self._segment_layout(segment_id)
        crop_y0 = int(crop["y1"])
        crop_y1 = int(crop["y2"])
        crop_x0 = int(crop["x1"])
        crop_x1 = int(crop["x2"])
        chunk_h, chunk_w = layout.chunk_shape
        chunk_y0 = int(crop_y0) // chunk_h
        chunk_y1 = max(int(crop_y0), int(crop_y1) - 1) // chunk_h
        chunk_x0 = int(crop_x0) // chunk_w
        chunk_x1 = max(int(crop_x0), int(crop_x1) - 1) // chunk_w
        chunk_validity = layout.chunk_validity
        full_segment = chunk_validity is None

        for chunk_y in range(chunk_y0, chunk_y1 + 1):
            for chunk_x in range(chunk_x0, chunk_x1 + 1):
                chunk_key = (int(chunk_y), int(chunk_x))
                if full_segment:
                    local_regions = None
                else:
                    assert chunk_validity is not None
                    if chunk_key not in chunk_validity:
                        continue
                    local_regions = chunk_validity[chunk_key]
                chunk_is_full = local_regions is None

                entry = self._chunk_entry(
                    segment_id=segment_id,
                    chunk_y=int(chunk_y),
                    chunk_x=int(chunk_x),
                )
                inter_y0 = max(int(crop_y0), int(entry.y0))
                inter_y1 = min(int(crop_y1), int(entry.y1))
                inter_x0 = max(int(crop_x0), int(entry.x0))
                inter_x1 = min(int(crop_x1), int(entry.x1))
                if inter_y1 <= inter_y0 or inter_x1 <= inter_x0:
                    continue

                patch_y0 = int(inter_y0 - int(crop_y0))
                patch_y1 = int(patch_y0 + (inter_y1 - inter_y0))
                patch_x0 = int(inter_x0 - int(crop_x0))
                patch_x1 = int(patch_x0 + (inter_x1 - inter_x0))
                chunk_y0_local = int(inter_y0 - int(entry.y0))
                chunk_y1_local = int(chunk_y0_local + (inter_y1 - inter_y0))
                chunk_x0_local = int(inter_x0 - int(entry.x0))
                chunk_x1_local = int(chunk_x0_local + (inter_x1 - inter_x0))

                chunk_weight = weight_crop[patch_y0:patch_y1, patch_x0:patch_x1]
                chunk_weighted = patch_crop_np[patch_y0:patch_y1, patch_x0:patch_x1] * chunk_weight
                if chunk_is_full:
                    self._accumulate_entry_slices(
                        entry=entry,
                        local_y0=int(chunk_y0_local),
                        local_y1=int(chunk_y1_local),
                        local_x0=int(chunk_x0_local),
                        local_x1=int(chunk_x1_local),
                        weighted_patch=chunk_weighted,
                        weight_patch=chunk_weight,
                    )
                    continue

                for region_y0, region_y1, region_x0, region_x1 in local_regions:
                    valid_y0 = max(int(chunk_y0_local), int(region_y0))
                    valid_y1 = min(int(chunk_y1_local), int(region_y1))
                    valid_x0 = max(int(chunk_x0_local), int(region_x0))
                    valid_x1 = min(int(chunk_x1_local), int(region_x1))
                    if valid_y1 <= valid_y0 or valid_x1 <= valid_x0:
                        continue

                    patch_local_y0 = int(valid_y0 - int(chunk_y0_local))
                    patch_local_y1 = int(patch_local_y0 + (valid_y1 - valid_y0))
                    patch_local_x0 = int(valid_x0 - int(chunk_x0_local))
                    patch_local_x1 = int(patch_local_x0 + (valid_x1 - valid_x0))
                    self._accumulate_entry_slices(
                        entry=entry,
                        local_y0=int(valid_y0),
                        local_y1=int(valid_y1),
                        local_x0=int(valid_x0),
                        local_x1=int(valid_x1),
                        weighted_patch=chunk_weighted[patch_local_y0:patch_local_y1, patch_local_x0:patch_local_x1],
                        weight_patch=chunk_weight[patch_local_y0:patch_local_y1, patch_local_x0:patch_local_x1],
                    )

    def add_batch(self, *, logits: torch.Tensor, xyxys: Any, segment_ids: tuple[str, ...] | list[str]) -> None:
        if logits.ndim != 4 or int(logits.shape[1]) != 1:
            raise ValueError(f"stitch logits must have shape (N,1,H,W), got {tuple(logits.shape)}")
        xyxy_rows = normalize_xyxy_rows(xyxys)
        segment_ids = [str(segment_id) for segment_id in segment_ids]
        if int(logits.shape[0]) != len(xyxy_rows) or len(segment_ids) != len(xyxy_rows):
            raise ValueError(
                "stitched batch size mismatch between logits, xyxys, and segment ids: "
                f"{int(logits.shape[0])} vs {len(xyxy_rows)} vs {len(segment_ids)}"
            )

        logits_cpu = logits.detach().to(device="cpu", dtype=torch.float32)
        grouped_crops: dict[tuple[str, int, int], list[tuple[int, dict[str, int]]]] = {}
        for batch_idx, segment_id in enumerate(segment_ids):
            crop = resolve_buffer_crop(
                xyxy=xyxy_rows[batch_idx],
                downsample=int(self.downsample),
                offset=(0, 0),
                buffer_shape=self.segment_ds_shape(segment_id),
            )
            if crop is None:
                continue
            group_key = (
                str(segment_id),
                int(crop["target_h"]),
                int(crop["target_w"]),
            )
            grouped_crops.setdefault(group_key, []).append((int(batch_idx), crop))

        for (segment_id, target_h, target_w), items in grouped_crops.items():
            batch_indices = [int(batch_idx) for batch_idx, _crop in items]
            patch_logits = logits_cpu[batch_indices]
            patch_logits = _prepare_patch_logits(
                patch_logits,
                target_h=int(target_h),
                target_w=int(target_w),
            )
            patch_logits_np = patch_logits.squeeze(1).numpy()
            weights = gaussian_weights(
                self._gaussian_cache,
                h=int(target_h),
                w=int(target_w),
                sigma_scale=float(self.gaussian_sigma_scale),
                min_weight=float(self.gaussian_min_weight),
            )
            for local_idx, (_, crop) in enumerate(items):
                patch_crop = patch_logits_np[
                    int(local_idx),
                    int(crop["py0"]):int(crop["py1"]),
                    int(crop["px0"]):int(crop["px1"]),
                ]
                weight_crop = weights[
                    int(crop["py0"]):int(crop["py1"]),
                    int(crop["px0"]):int(crop["px1"]),
                ]
                self._accumulate_patch(
                    segment_id=str(segment_id),
                    crop=crop,
                    patch_crop_np=patch_crop,
                    weight_crop=weight_crop,
                )

    def _read_region_arrays(
        self,
        *,
        segment_id: str,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        segment_id = str(segment_id)
        self.flush()
        sum_arr, weight_arr = self._segment_arrays(segment_id)
        y0, y1, x0, x1 = _clamp_bbox(
            bbox,
            full_shape=self.segment_ds_shape(segment_id),
        )
        if y1 <= y0 or x1 <= x0:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
        return (
            np.asarray(sum_arr[y0:y1, x0:x1], dtype=np.float32),
            np.asarray(weight_arr[y0:y1, x0:x1], dtype=np.float32),
        )

    def read_region_logits_and_coverage(
        self,
        *,
        segment_id: str,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        stitched_sum, stitched_weight = self._read_region_arrays(
            segment_id=segment_id,
            bbox=bbox,
        )
        return stitch_logit_map(stitched_sum, stitched_weight)

    def read_region_probs_and_coverage(
        self,
        *,
        segment_id: str,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        stitched_sum, stitched_weight = self._read_region_arrays(
            segment_id=segment_id,
            bbox=bbox,
        )
        return stitch_prob_map(stitched_sum, stitched_weight)

    def write_full_segment_probs(
        self,
        *,
        segment_id: str,
        out_path: str | Path | None = None,
        probs: np.ndarray | None = None,
    ) -> str:
        segment_id = str(segment_id)
        if out_path is None:
            out_path = Path(self.root_dir).resolve() / f"{segment_store_key(segment_id)}__prob.zarr"
        out_path = Path(out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if probs is None:
            probs, _coverage = self.read_region_probs_and_coverage(segment_id=segment_id)
        else:
            probs = np.asarray(probs, dtype=np.float32)
        prob_arr = zarr.open(
            str(out_path),
            mode="w",
            shape=tuple(probs.shape),
            chunks=_chunk_shape(tuple(probs.shape)),
            dtype=np.float32,
            fill_value=0.0,
            write_empty_chunks=False,
        )
        prob_arr[:] = probs
        return str(out_path)
