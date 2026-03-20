from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ink.core.types import ModelOutputBatch
from ink.recipes.data.zarr_io import read_label_and_supervision_mask_region
from ink.recipes.eval.stitch_components import ComponentEvalItem, DetectedComponentRegion
from ink.recipes.eval.stitch_prepared import StitchEvalArtifactKey, StitchEvalArtifactStore
from ink.recipes.metrics.stitching import StitchMetricBatch


def fullres_bbox_from_ds_bbox(
    bbox: tuple[int, int, int, int],
    *,
    downsample: int,
    full_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    y0, y1, x0, x1 = [int(v) for v in bbox]
    ds = max(1, int(downsample))
    full_h, full_w = [int(v) for v in full_shape]
    return (
        max(0, min(y0 * ds, full_h)),
        max(0, min(y1 * ds, full_h)),
        max(0, min(x0 * ds, full_w)),
        max(0, min(x1 * ds, full_w)),
    )


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


def downsample_binary_mask_any(
    mask: np.ndarray,
    *,
    downsample: int,
    out_shape: tuple[int, int],
) -> np.ndarray:
    out_h, out_w = [int(v) for v in out_shape]
    ds = max(1, int(downsample))
    mask_bool = np.asarray(mask) > 0
    if ds == 1 and tuple(mask_bool.shape) == (out_h, out_w):
        return mask_bool

    target_h = out_h * ds
    target_w = out_w * ds
    mask_bool = mask_bool[: min(int(mask_bool.shape[0]), target_h), : min(int(mask_bool.shape[1]), target_w)]
    pad_h = max(0, target_h - int(mask_bool.shape[0]))
    pad_w = max(0, target_w - int(mask_bool.shape[1]))
    if pad_h or pad_w:
        mask_bool = np.pad(mask_bool, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
    return mask_bool.reshape(out_h, ds, out_w, ds).any(axis=(1, 3))


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


@dataclass
class StitchEvalRegionReader:
    layout: object
    downsample: int = 1
    segment_shapes: dict[str, tuple[int, int]] = field(default_factory=dict, repr=False)
    _bbox_mask_cache: dict[tuple[str, tuple[int, int, int, int], int], tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict,
        repr=False,
    )
    _source_fingerprint_cache: dict[str, str] = field(default_factory=dict, repr=False)

    def _segment_portable_source_fingerprint(self, segment_id: str) -> str:
        segment_id = str(segment_id)
        cached = self._source_fingerprint_cache.get(segment_id)
        if cached is not None:
            return cached

        layout_fingerprint = getattr(self.layout, "label_mask_fingerprint", None)
        if not callable(layout_fingerprint):
            raise TypeError("StitchEvalRegionReader layout must provide label_mask_fingerprint(segment_id)")
        fingerprint = str(layout_fingerprint(segment_id))

        self._source_fingerprint_cache[segment_id] = fingerprint
        return fingerprint

    def read_label_and_supervision(
        self,
        *,
        segment_id: str,
        bbox: tuple[int, int, int, int],
        cache: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        cache_key = (str(segment_id), tuple(int(v) for v in bbox), int(self.downsample))
        if cache:
            cached = self._bbox_mask_cache.get(cache_key)
            if cached is not None:
                return cached

        full_shape = self.segment_shapes.get(str(segment_id))
        if full_shape is None:
            raise KeyError(f"missing stitched segment shape for {segment_id!r}")

        raw_bbox = fullres_bbox_from_ds_bbox(
            bbox,
            downsample=int(self.downsample),
            full_shape=full_shape,
        )
        raw_labels, raw_supervision = read_label_and_supervision_mask_region(
            self.layout,
            str(segment_id),
            full_shape,
            raw_bbox,
        )
        out_shape = (int(bbox[1] - bbox[0]), int(bbox[3] - bbox[2]))
        labels = downsample_binary_mask_any(
            raw_labels,
            downsample=int(self.downsample),
            out_shape=out_shape,
        )
        supervision = downsample_binary_mask_any(
            raw_supervision,
            downsample=int(self.downsample),
            out_shape=out_shape,
        )
        out = (np.asarray(labels, dtype=bool), np.asarray(supervision, dtype=bool))
        if cache:
            self._bbox_mask_cache[cache_key] = out
        return out

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
                downsample=int(self.downsample),
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
