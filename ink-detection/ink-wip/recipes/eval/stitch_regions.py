from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import uuid

import numpy as np
import torch

from ink.core.types import ModelOutputBatch
from ink.recipes.data.masks import SUPERVISION_MASK_NAME, resolve_segment_mask_names
from ink.recipes.data.zarr_io import (
    read_label_and_supervision_mask_region,
    read_label_region,
    read_supervision_mask_region,
)
from ink.recipes.eval.stitch_components import (
    ComponentEvalItem,
    DetectedComponentRegion,
    detect_component_regions,
)
from ink.recipes.eval.stitch_prepared import StitchEvalArtifactKey, StitchEvalArtifactStore
from ink.recipes.metrics.stitching import StitchMetricBatch
from ink.recipes.stitch.store import segment_store_key


def expand_component_bbox(
    bbox: tuple[int, int, int, int],
    *,
    pad: int = 1,
    segment_ds_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    y0, y1, x0, x1 = [int(v) for v in bbox]
    pad_i = max(0, int(pad))
    max_y, max_x = [int(v) for v in segment_ds_shape]
    return (
        max(0, y0 - pad_i),
        min(max_y, y1 + pad_i),
        max(0, x0 - pad_i),
        min(max_x, x1 + pad_i),
    )


def project_component_mask(
    region: DetectedComponentRegion,
    *,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    y0, y1, x0, x1 = [int(v) for v in bbox]
    out = np.zeros((int(y1 - y0), int(x1 - x0)), dtype=bool)

    region_y0, region_y1, region_x0, region_x1 = [int(v) for v in region.bbox]
    inter_y0 = max(y0, region_y0)
    inter_y1 = min(y1, region_y1)
    inter_x0 = max(x0, region_x0)
    inter_x1 = min(x1, region_x1)
    if inter_y1 <= inter_y0 or inter_x1 <= inter_x0:
        return out

    out_y0 = int(inter_y0 - y0)
    out_y1 = int(out_y0 + (inter_y1 - inter_y0))
    out_x0 = int(inter_x0 - x0)
    out_x1 = int(out_x0 + (inter_x1 - inter_x0))

    region_mask_y0 = int(inter_y0 - region_y0)
    region_mask_y1 = int(region_mask_y0 + (inter_y1 - inter_y0))
    region_mask_x0 = int(inter_x0 - region_x0)
    region_mask_x1 = int(region_mask_x0 + (inter_x1 - inter_x0))

    out[out_y0:out_y1, out_x0:out_x1] = np.asarray(
        region.mask[region_mask_y0:region_mask_y1, region_mask_x0:region_mask_x1],
        dtype=bool,
    )
    return out


def _save_detected_components(path: Path, components: tuple[DetectedComponentRegion, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bbox_rows = np.asarray([region.bbox for region in components], dtype=np.int32)
    arrays = {"bbox_rows": bbox_rows}
    for idx, region in enumerate(components):
        arrays[f"mask_{idx:05d}"] = np.asarray(region.mask, dtype=np.uint8)
    tmp_path = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _load_detected_components(path: Path) -> tuple[DetectedComponentRegion, ...]:
    with np.load(path, allow_pickle=False) as loaded:
        bbox_rows = np.asarray(loaded["bbox_rows"], dtype=np.int32)
        components: list[DetectedComponentRegion] = []
        for idx, bbox_row in enumerate(bbox_rows):
            components.append(
                DetectedComponentRegion(
                    bbox=tuple(int(value) for value in bbox_row.tolist()),
                    mask=np.asarray(loaded[f"mask_{idx:05d}"], dtype=bool),
                )
            )
    return tuple(components)


@dataclass
class StitchEvalRegionReader:
    layout: object
    label_suffix: str = ""
    mask_suffix: str = ""
    train_segment_ids: frozenset[str] = field(default_factory=frozenset)
    mask_name: str = SUPERVISION_MASK_NAME
    cache_root: str | Path | None = None
    segment_shapes: dict[str, tuple[int, int]] = field(default_factory=dict, repr=False)
    _bbox_label_cache: dict[tuple[str, tuple[int, int, int, int]], np.ndarray] = field(
        default_factory=dict,
        repr=False,
    )
    _bbox_supervision_cache: dict[tuple[str, tuple[int, int, int, int]], np.ndarray] = field(
        default_factory=dict,
        repr=False,
    )
    _bbox_mask_cache: dict[tuple[str, tuple[int, int, int, int]], tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict,
        repr=False,
    )
    _source_fingerprint_cache: dict[str, str] = field(default_factory=dict, repr=False)
    _detected_component_cache: dict[tuple[str, int], tuple[DetectedComponentRegion, ...]] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.label_suffix = str(self.label_suffix)
        self.mask_suffix = str(self.mask_suffix)
        self.train_segment_ids = frozenset(str(segment_id) for segment_id in self.train_segment_ids)
        self.mask_name = str(self.mask_name).strip()
        if not self.mask_name:
            raise ValueError("mask_name must be a non-empty string")
        if self.cache_root is None:
            return
        root = Path(self.cache_root).expanduser().resolve() / "components"
        root.mkdir(parents=True, exist_ok=True)
        self.cache_root = root

    def _mask_names_for_segment(self, segment_id: str) -> tuple[str, ...]:
        return resolve_segment_mask_names(
            split_name="valid",
            segment_id=segment_id,
            train_segment_ids=self.train_segment_ids,
            default_mask_name=self.mask_name,
        )

    def _segment_portable_source_fingerprint(self, segment_id: str) -> str:
        segment_id = str(segment_id)
        cached = self._source_fingerprint_cache.get(segment_id)
        if cached is not None:
            return cached

        layout_fingerprint = getattr(self.layout, "label_mask_metadata_fingerprint", None)
        fingerprint_kwargs = {
            "label_suffix": self.label_suffix,
            "mask_suffix": self.mask_suffix,
            "mask_names": self._mask_names_for_segment(segment_id),
        }
        if not callable(layout_fingerprint):
            layout_fingerprint = getattr(self.layout, "label_mask_fingerprint", None)
        if not callable(layout_fingerprint):
            raise TypeError("StitchEvalRegionReader layout must provide label_mask_fingerprint(segment_id)")
        fingerprint = str(layout_fingerprint(segment_id, **fingerprint_kwargs))

        self._source_fingerprint_cache[segment_id] = fingerprint
        return fingerprint

    def _cache_key(self, *, segment_id: str, bbox: tuple[int, int, int, int]) -> tuple[str, tuple[int, int, int, int]]:
        return (str(segment_id), tuple(int(v) for v in bbox))

    def _segment_full_shape(self, segment_id: str) -> tuple[int, int]:
        full_shape = self.segment_shapes.get(str(segment_id))
        if full_shape is None:
            raise KeyError(f"missing stitched segment shape for {segment_id!r}")
        return (int(full_shape[0]), int(full_shape[1]))

    def _read_metric_label_and_supervision(
        self,
        *,
        segment_id: str,
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        full_shape = self._segment_full_shape(segment_id)
        raw_labels, raw_supervision = read_label_and_supervision_mask_region(
            self.layout,
            str(segment_id),
            full_shape,
            bbox,
            label_suffix=self.label_suffix,
            mask_suffix=self.mask_suffix,
            mask_names=self._mask_names_for_segment(segment_id),
        )
        return (
            np.asarray(raw_labels, dtype=bool),
            np.asarray(raw_supervision, dtype=bool),
        )

    def _component_cache_path(self, *, segment_id: str, connectivity: int) -> Path | None:
        if self.cache_root is None:
            return None
        filename = f"components_{self._segment_portable_source_fingerprint(segment_id)}_c{int(connectivity)}.npz"
        return Path(self.cache_root) / segment_store_key(segment_id) / filename

    def detected_segment_components(
        self,
        *,
        segment_id: str,
        store,
        connectivity: int,
    ) -> tuple[DetectedComponentRegion, ...]:
        segment_id = str(segment_id)
        cache_key = (segment_id, int(connectivity))
        cached = self._detected_component_cache.get(cache_key)
        if cached is not None:
            return cached

        cache_path = self._component_cache_path(segment_id=segment_id, connectivity=int(connectivity))
        if cache_path is not None and cache_path.exists():
            try:
                detected = _load_detected_components(cache_path)
                self._detected_component_cache[cache_key] = detected
                return detected
            except Exception:
                pass

        segment_ds_shape = store.segment_ds_shape(segment_id)
        full_segment_bbox = (0, int(segment_ds_shape[0]), 0, int(segment_ds_shape[1]))
        supervision = self.read_supervision(segment_id=segment_id, bbox=full_segment_bbox, cache=False)

        components: list[DetectedComponentRegion] = []
        supervision_rois = detect_component_regions(
            supervision,
            connectivity=int(connectivity),
        )
        for roi_region in supervision_rois:
            y0, y1, x0, x1 = [int(v) for v in roi_region.bbox]
            roi_labels = self.read_label(
                segment_id=segment_id,
                bbox=roi_region.bbox,
                cache=False,
            )
            component_source = np.asarray(roi_labels, dtype=bool) & np.asarray(roi_region.mask, dtype=bool)
            components.extend(
                detect_component_regions(
                    component_source,
                    connectivity=int(connectivity),
                    offset=(y0, x0),
                )
            )

        detected = tuple(components)
        if cache_path is not None:
            _save_detected_components(cache_path, detected)
        self._detected_component_cache[cache_key] = detected
        return detected

    def read_label_and_supervision(
        self,
        *,
        segment_id: str,
        bbox: tuple[int, int, int, int],
        cache: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        cache_key = self._cache_key(segment_id=str(segment_id), bbox=bbox)
        if cache:
            cached = self._bbox_mask_cache.get(cache_key)
            if cached is not None:
                return cached

        out = self._read_metric_label_and_supervision(
            segment_id=segment_id,
            bbox=bbox,
        )
        if cache:
            self._bbox_mask_cache[cache_key] = out
            self._bbox_label_cache[cache_key] = out[0]
            self._bbox_supervision_cache[cache_key] = out[1]
        return out

    def read_label(
        self,
        *,
        segment_id: str,
        bbox: tuple[int, int, int, int],
        cache: bool = True,
    ) -> np.ndarray:
        cache_key = self._cache_key(segment_id=str(segment_id), bbox=bbox)
        if cache:
            cached = self._bbox_label_cache.get(cache_key)
            if cached is not None:
                return cached
            pair_cached = self._bbox_mask_cache.get(cache_key)
            if pair_cached is not None:
                return pair_cached[0]

        full_shape = self._segment_full_shape(segment_id)
        labels = np.asarray(
            read_label_region(
                self.layout,
                str(segment_id),
                full_shape,
                bbox,
                label_suffix=self.label_suffix,
            ),
            dtype=bool,
        )
        if cache:
            self._bbox_label_cache[cache_key] = labels
        return labels

    def read_supervision(
        self,
        *,
        segment_id: str,
        bbox: tuple[int, int, int, int],
        cache: bool = True,
    ) -> np.ndarray:
        cache_key = self._cache_key(segment_id=str(segment_id), bbox=bbox)
        if cache:
            cached = self._bbox_supervision_cache.get(cache_key)
            if cached is not None:
                return cached
            pair_cached = self._bbox_mask_cache.get(cache_key)
            if pair_cached is not None:
                return pair_cached[1]

        full_shape = self._segment_full_shape(segment_id)
        supervision = np.asarray(
            read_supervision_mask_region(
                self.layout,
                str(segment_id),
                full_shape,
                bbox,
                mask_suffix=self.mask_suffix,
                mask_names=self._mask_names_for_segment(segment_id),
            ),
            dtype=bool,
        )
        if cache:
            self._bbox_supervision_cache[cache_key] = supervision
        return supervision

    def read_component_arrays(
        self,
        *,
        segment_id: str,
        bbox: tuple[int, int, int, int],
        region: DetectedComponentRegion,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Stitched metrics score the selected GT component only. Neighboring GT
        # inside a padded metric crop remains masked out unless it belongs to
        # the selected component.
        labels, supervision = self.read_label_and_supervision(segment_id=segment_id, bbox=bbox)
        labels = np.asarray(labels, dtype=bool)
        supervision = np.asarray(supervision, dtype=bool)
        component_mask = project_component_mask(region, bbox=bbox)
        valid_region = supervision & (np.logical_not(labels) | component_mask)
        return labels, supervision, np.asarray(component_mask, dtype=bool), np.asarray(valid_region, dtype=bool)

    def eval_batch_from_arrays(
        self,
        *,
        store,
        segment_id: str,
        bbox: tuple[int, int, int, int],
        labels: np.ndarray,
        valid_region: np.ndarray,
    ) -> ModelOutputBatch:
        stitched_logits, coverage = store.read_region_logits_and_coverage(segment_id=segment_id, bbox=bbox)
        valid_mask = np.asarray(valid_region, dtype=bool) & np.asarray(coverage, dtype=bool)
        return ModelOutputBatch(
            logits=torch.as_tensor(stitched_logits, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            segment_ids=(segment_id,),
            patch_xyxy=None,
            targets=torch.as_tensor(np.asarray(labels, dtype=np.float32)).unsqueeze(0).unsqueeze(0),
            valid_mask=torch.as_tensor(valid_mask, dtype=torch.bool).unsqueeze(0).unsqueeze(0),
        )

    def stitch_metric_batch(
        self,
        *,
        store,
        item: ComponentEvalItem,
        component_batch: ModelOutputBatch,
        connectivity: int,
        component_pad: int,
        prepared_store: StitchEvalArtifactStore,
    ) -> StitchMetricBatch:
        # All stitched metrics score the same padded crop. component_pad is
        # therefore a stitched-metric crop hyperparameter for every metric,
        # not just the ones that need prepared context artifacts.
        segment_ds_shape = store.segment_ds_shape(item.segment_id)
        bbox = tuple(int(v) for v in item.region.bbox)
        metric_bbox = expand_component_bbox(
            bbox,
            pad=int(component_pad),
            segment_ds_shape=segment_ds_shape,
        )
        labels, supervision, component_mask, valid_region = self.read_component_arrays(
            segment_id=item.segment_id,
            bbox=metric_bbox,
            region=item.region,
        )
        if metric_bbox == bbox:
            metric_batch = component_batch
        else:
            metric_batch = self.eval_batch_from_arrays(
                store=store,
                segment_id=item.segment_id,
                bbox=metric_bbox,
                labels=labels,
                valid_region=valid_region,
            )
        prepared = prepared_store.get_prepared(
            key=StitchEvalArtifactKey(
                segment_id=str(item.segment_id),
                source_fingerprint=self._segment_portable_source_fingerprint(item.segment_id),
                metric_bbox=metric_bbox,
                component_bbox=bbox,
                connectivity=int(connectivity),
                component_pad=int(component_pad),
            ),
            labels=labels,
            supervision=supervision,
            component_mask=component_mask,
        )
        return StitchMetricBatch.from_model_output_batch(
            metric_batch,
            connectivity=int(connectivity),
            prepared=prepared,
        )
