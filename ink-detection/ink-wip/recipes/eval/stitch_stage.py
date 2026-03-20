from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch

from ink.core.device import move_batch_to_device
from ink.core.types import Batch, DataBundle, EvalReport
from ink.recipes.data.layout import NestedZarrLayout
from ink.recipes.data.zarr_io import read_label_and_supervision_mask_for_shape
from ink.recipes.metrics import MetricBatch, merge_metric_reports
from ink.recipes.stitch import StitchRuntime
from ink.recipes.stitch.data import coerce_component_specs
from ink.recipes.stitch.store import ZarrStitchStore


def _resolve_layout_from_bundle(data: DataBundle) -> NestedZarrLayout:
    dataset = getattr(getattr(data, "val_loader", None), "dataset", None)
    layout = getattr(dataset, "layout", None)
    if not isinstance(layout, NestedZarrLayout):
        raise ValueError("StitchEval requires val_loader.dataset.layout as NestedZarrLayout")
    return layout


def _segment_shapes_from_stitch(stitch: StitchRuntime) -> dict[str, tuple[int, int]]:
    shapes = {
        str(spec.segment_id): tuple(int(v) for v in spec.shape)
        for spec in stitch.data.eval.segments
    }
    if not shapes:
        raise ValueError("StitchEval requires stitch.eval.segments with segment shapes")
    return shapes


def _component_bbox(
    component,
    *,
    downsample: int,
    segment_ds_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if component.bbox is None:
        return None
    y0, y1, x0, x1 = [int(v) for v in component.bbox]
    ds = max(1, int(downsample))
    y0 = y0 // ds
    y1 = (y1 + ds - 1) // ds
    x0 = x0 // ds
    x1 = (x1 + ds - 1) // ds
    max_y, max_x = [int(v) for v in segment_ds_shape]
    y0 = max(0, min(y0, max_y))
    y1 = max(0, min(y1, max_y))
    x0 = max(0, min(x0, max_x))
    x1 = max(0, min(x1, max_x))
    if y1 <= y0 or x1 <= x0:
        return None
    return (y0, y1, x0, x1)


def _downsample_binary_mask_any(
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


def _component_mean_reports(component_reports: list[tuple[str, EvalReport]]) -> EvalReport:
    summary_values: dict[str, list[float]] = defaultdict(list)
    by_segment_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for segment_id, report in component_reports:
        for key, value in report.summary.items():
            metric_key = str(key)
            metric_value = float(value)
            summary_values[metric_key].append(metric_value)
            by_segment_values[str(segment_id)][metric_key].append(metric_value)

    return EvalReport(
        summary={
            key: float(sum(values) / len(values))
            for key, values in summary_values.items()
            if values
        },
        by_segment={
            segment_id: {
                key: float(sum(values) / len(values))
                for key, values in metric_map.items()
                if values
            }
            for segment_id, metric_map in by_segment_values.items()
        },
    )


@dataclass(frozen=True, kw_only=True)
class StitchEval:
    metrics: tuple[Any, ...] = ()
    store: ZarrStitchStore = field(default_factory=ZarrStitchStore)
    components: tuple[Any, ...] = ()
    aggregation: str | None = None

    layout: NestedZarrLayout | None = field(default=None, repr=False)
    downsample: int = 1
    segment_shapes: dict[str, tuple[int, int]] = field(default_factory=dict, repr=False)
    _component_specs: tuple[Any, ...] = field(default_factory=tuple, repr=False)
    _label_mask_cache: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "components", tuple(self.components))
        aggregation = self.aggregation
        if aggregation is None:
            aggregation = "component_mean" if self.components else "pooled"
        aggregation = str(aggregation).strip().lower()
        if aggregation not in {"pooled", "component_mean"}:
            raise ValueError(
                "StitchEval aggregation must be 'pooled' or 'component_mean', "
                f"got {self.aggregation!r}"
            )
        object.__setattr__(self, "aggregation", aggregation)

    def build(self, *, data, runtime=None, stitch=None, logger=None) -> StitchEval:
        if not isinstance(stitch, StitchRuntime):
            raise ValueError("StitchEval requires experiment.stitch as StitchRuntime")

        layout = _resolve_layout_from_bundle(data)
        segment_shapes = _segment_shapes_from_stitch(stitch)
        downsample = int(stitch.data.layout.downsample)

        component_specs = tuple(coerce_component_specs(self.components)) if self.components else ()

        metrics = tuple(
            metric.build(
                data=data,
                runtime=runtime,
                stitch=stitch,
                logger=logger,
            )
            for metric in self.metrics
        )

        store = self.store.build(segment_shapes=segment_shapes, downsample=downsample)

        return replace(
            self,
            metrics=metrics,
            store=store,
            layout=layout,
            downsample=downsample,
            segment_shapes=segment_shapes,
            _component_specs=component_specs,
            _label_mask_cache={},
        )

    def begin_epoch(self) -> None:
        self.store.reset()

    def observe_batch(self, batch: MetricBatch) -> None:
        if batch.patch_xyxy is None:
            raise ValueError("StitchEval requires batch.patch_xyxy for stitched accumulation")
        self.store.add_batch(
            logits=batch.logits,
            xyxys=batch.patch_xyxy,
            segment_ids=tuple(batch.segment_ids),
        )

    def _iter_component_regions(self):
        if self._component_specs:
            for component in self._component_specs:
                segment_id, _component_idx = component.component_key
                segment_ds_shape = self.store.segment_ds_shape(segment_id)
                bbox = _component_bbox(
                    component,
                    downsample=int(self.downsample),
                    segment_ds_shape=segment_ds_shape,
                )
                if bbox is None:
                    yield str(segment_id), (0, int(segment_ds_shape[0]), 0, int(segment_ds_shape[1]))
                else:
                    yield str(segment_id), bbox
            return

        for segment_id in self.segment_shapes:
            segment_ds_shape = self.store.segment_ds_shape(segment_id)
            yield str(segment_id), (0, int(segment_ds_shape[0]), 0, int(segment_ds_shape[1]))

    def _segment_label_and_mask(self, segment_id: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._label_mask_cache.get(segment_id)
        if cached is not None:
            return cached
        shape = self.segment_shapes.get(segment_id)
        if shape is None:
            raise KeyError(f"missing stitched segment shape for {segment_id!r}")
        labels, supervision_mask = read_label_and_supervision_mask_for_shape(
            self.layout,
            segment_id,
            shape,
        )
        self._label_mask_cache[segment_id] = (labels, supervision_mask)
        return labels, supervision_mask

    def _region_metric_batch(
        self,
        *,
        segment_id: str,
        bbox: tuple[int, int, int, int],
    ) -> MetricBatch:
        stitched_logits, coverage = self.store.read_region_logits_and_coverage(segment_id=segment_id, bbox=bbox)
        labels, supervision_mask = self._segment_label_and_mask(segment_id)
        segment_ds_shape = self.store.segment_ds_shape(segment_id)
        labels_ds = _downsample_binary_mask_any(
            labels,
            downsample=int(self.downsample),
            out_shape=segment_ds_shape,
        )
        supervision_ds = _downsample_binary_mask_any(
            supervision_mask,
            downsample=int(self.downsample),
            out_shape=segment_ds_shape,
        )
        y0, y1, x0, x1 = [int(v) for v in bbox]

        target_region = labels_ds[y0:y1, x0:x1].astype(np.float32)
        supervision_region = supervision_ds[y0:y1, x0:x1]
        valid_region = supervision_region & coverage

        return MetricBatch(
            logits=torch.as_tensor(stitched_logits, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            targets=torch.as_tensor(target_region, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            valid_mask=torch.as_tensor(valid_region, dtype=torch.bool).unsqueeze(0).unsqueeze(0),
            segment_ids=(segment_id,),
        )

    def finalize_epoch(self) -> EvalReport:
        if not self.metrics:
            return EvalReport(summary={})

        if str(self.aggregation) == "component_mean":
            component_reports: list[tuple[str, EvalReport]] = []
            for segment_id, bbox in self._iter_component_regions():
                metric_batch = self._region_metric_batch(segment_id=segment_id, bbox=bbox)
                shared = {}
                reports = []
                for metric in self.metrics:
                    state = metric.empty_state()
                    state = metric.update(state, metric_batch, shared=shared)
                    reports.append(metric.finalize(state))
                component_reports.append((segment_id, merge_metric_reports(reports)))
            return _component_mean_reports(component_reports)

        states = [metric.empty_state() for metric in self.metrics]
        for segment_id, bbox in self._iter_component_regions():
            metric_batch = self._region_metric_batch(segment_id=segment_id, bbox=bbox)
            shared = {}
            for idx, metric in enumerate(self.metrics):
                states[idx] = metric.update(states[idx], metric_batch, shared=shared)

        return merge_metric_reports(
            [metric.finalize(state) for metric, state in zip(self.metrics, states)]
        )

    def evaluate(self, model, val_loader, *, device=None) -> EvalReport:
        if not callable(model):
            raise TypeError("evaluation model must be callable")

        if device is not None and hasattr(model, "to"):
            model.to(device)

        self.begin_epoch()

        was_training = bool(getattr(model, "training", False))
        if callable(getattr(model, "eval", None)):
            model.eval()

        try:
            with torch.inference_mode():
                for batch in val_loader:
                    if not isinstance(batch, Batch):
                        raise TypeError("validation batch must be Batch")
                    batch = move_batch_to_device(batch, device=device)
                    logits = model(batch.x)
                    self.observe_batch(
                        MetricBatch(
                            logits=logits,
                            targets=batch.y if batch.y is not None else torch.zeros_like(logits),
                            valid_mask=batch.meta.valid_mask,
                            group_idx=batch.meta.group_idx,
                            segment_ids=tuple(batch.meta.segment_ids),
                            patch_xyxy=batch.meta.patch_xyxy,
                        )
                    )
        finally:
            if was_training and callable(getattr(model, "train", None)):
                model.train()

        return self.finalize_epoch()
