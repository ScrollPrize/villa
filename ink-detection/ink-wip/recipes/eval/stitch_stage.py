from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ink.core.types import EvalReport
from ink.recipes.data.layout import NestedZarrLayout
from ink.recipes.eval.stitch_components import StitchComponentCatalog
from ink.recipes.eval.stitch_prepared import StitchEvalArtifactStore
from ink.recipes.eval.stitch_regions import StitchEvalRegionReader
from ink.recipes.data.masks import SUPERVISION_MASK_NAME
from ink.recipes.metrics import merge_metric_reports
from ink.recipes.stitch import StitchInference, StitchRuntime


@dataclass(kw_only=True)
class StitchEval:
    metrics: tuple[Any, ...] = ()
    prepared_cache_root: str | Path | None = None
    components: tuple[Any, ...] = ()
    component_connectivity: int = 2
    component_pad: int = 1

    _component_catalog: StitchComponentCatalog | None = field(default=None, repr=False)
    _region_reader: StitchEvalRegionReader | None = field(default=None, repr=False)
    _inference: StitchInference | None = field(default=None, repr=False)
    _prepared_store: StitchEvalArtifactStore | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.metrics = tuple(self.metrics)
        self.components = tuple(self.components)
        connectivity = int(self.component_connectivity)
        if connectivity not in {1, 2}:
            raise ValueError(
                "StitchEval component_connectivity must be 1 (4-neighborhood) or 2 (8-neighborhood), "
                f"got {self.component_connectivity!r}"
            )
        self.component_connectivity = connectivity
        component_pad = int(self.component_pad)
        if component_pad < 0:
            raise ValueError(f"StitchEval component_pad must be >= 0, got {self.component_pad!r}")
        self.component_pad = component_pad

    def build(self, *, data, runtime=None, logger=None, inference: StitchInference | None = None) -> StitchEval:
        if not self.metrics:
            raise ValueError("StitchEval requires at least one metric")
        if inference is None:
            raise ValueError("StitchEval requires bound StitchInference from ValidationEvaluator.build(...)")
        if not isinstance(inference, StitchInference):
            raise TypeError("StitchEval inference must be StitchInference")
        stitch_runtime = inference.stitch_runtime
        if not isinstance(stitch_runtime, StitchRuntime):
            raise ValueError("StitchEval requires bound StitchInference with stitch_runtime")
        layout_info = stitch_runtime.eval_segment_layout()
        component_catalog = StitchComponentCatalog.build(
            raw_components=self.components,
            segment_ids=layout_info.segment_shapes.keys(),
            connectivity=int(self.component_connectivity),
        )
        dataset = getattr(getattr(data, "eval_loader", None), "dataset", None)
        layout = getattr(dataset, "layout", None)
        if not isinstance(layout, NestedZarrLayout):
            raise ValueError("stitch evaluation requires eval_loader.dataset.layout as NestedZarrLayout")
        metrics = tuple(
            metric.build(
                data=data,
                runtime=runtime,
                stitch=stitch_runtime,
                logger=logger,
            )
            if callable(getattr(metric, "build", None))
            else metric
            for metric in self.metrics
        )
        bound_inference = inference
        if not bound_inference.segment_shapes:
            raise ValueError("StitchEval requires already-bound StitchInference from ValidationEvaluator.build(...)")
        region_reader = StitchEvalRegionReader(
            layout=layout,
            label_suffix=str(getattr(dataset, "label_suffix", "")),
            mask_suffix=str(getattr(dataset, "mask_suffix", "")),
            train_segment_ids=frozenset(str(segment_id) for segment_id in getattr(dataset, "train_segment_ids", ())),
            mask_name=str(getattr(dataset, "mask_name", SUPERVISION_MASK_NAME)),
            cache_root=self.prepared_cache_root,
            segment_shapes=dict(bound_inference.segment_shapes),
            _bbox_label_cache={},
            _bbox_supervision_cache={},
            _bbox_mask_cache={},
            _source_fingerprint_cache={},
            _detected_component_cache={},
        )
        prepared_store = StitchEvalArtifactStore(cache_root=self.prepared_cache_root)

        return replace(
            self,
            metrics=metrics,
            _component_catalog=component_catalog,
            _region_reader=region_reader,
            _inference=bound_inference,
            _prepared_store=prepared_store,
        )

    def finalize_epoch(self) -> EvalReport:
        inference, component_catalog, region_reader, prepared_store = self._require_epoch_state()
        global_states = [metric.empty_state() for metric in self.metrics]
        by_segment_values: dict[str, dict[str, float]] = {}
        items = component_catalog.iter_items(
            store=inference.store,
            read_bbox_label_and_supervision=region_reader.read_label_and_supervision,
            detected_segment_components=region_reader.detected_segment_components,
        )
        for item in items:
            stitch_batch = self._metric_batch_for_item(
                inference,
                region_reader,
                prepared_store,
                item=item,
            )
            by_segment_values[str(item.report_key)] = self._observe_metric_batch(
                stitch_batch,
                global_states=global_states,
            )
        summary_report = merge_metric_reports(
            [metric.finalize(state) for metric, state in zip(self.metrics, global_states)]
        )
        return EvalReport(
            summary={str(key): float(value) for key, value in summary_report.summary.items()},
            by_segment=by_segment_values,
        )

    def _require_epoch_state(self) -> tuple[StitchInference, StitchComponentCatalog, StitchEvalRegionReader, StitchEvalArtifactStore]:
        if not self.metrics:
            raise ValueError("StitchEval requires at least one metric")
        inference = self._inference
        if inference is None:
            raise ValueError("StitchEval requires a bound StitchInference")
        component_catalog = self._component_catalog
        if component_catalog is None:
            raise ValueError("StitchEval requires build(...) before finalizing stitched metrics")
        region_reader = self._region_reader
        if region_reader is None:
            raise ValueError("StitchEval requires build(...) before reading stitched metric regions")
        prepared_store = self._prepared_store
        if prepared_store is None:
            raise ValueError("StitchEval requires build(...) before preparing stitched GT artifacts")
        return inference, component_catalog, region_reader, prepared_store

    def _metric_batch_for_item(
        self,
        inference: StitchInference,
        region_reader: StitchEvalRegionReader,
        prepared_store: StitchEvalArtifactStore,
        *,
        item,
    ):
        component_labels, _, _, component_valid_region = region_reader.read_component_arrays(
            segment_id=item.segment_id,
            bbox=item.region.bbox,
            region=item.region,
        )
        component_batch = region_reader.eval_batch_from_arrays(
            store=inference.store,
            segment_id=item.segment_id,
            bbox=item.region.bbox,
            labels=component_labels,
            valid_region=component_valid_region,
        )
        return region_reader.stitch_metric_batch(
            store=inference.store,
            item=item,
            component_batch=component_batch,
            connectivity=int(self.component_connectivity),
            component_pad=int(self.component_pad),
            prepared_store=prepared_store,
        )

    def _observe_metric_batch(self, stitch_batch, *, global_states: list[Any]) -> dict[str, float]:
        reports = []
        for idx, metric in enumerate(self.metrics):
            state = metric.empty_state()
            state = metric.update(state, stitch_batch)
            reports.append(metric.finalize(state))
            global_states[idx] = metric.update(global_states[idx], stitch_batch)
        report = merge_metric_reports(reports)
        return {str(key): float(value) for key, value in report.summary.items()}
