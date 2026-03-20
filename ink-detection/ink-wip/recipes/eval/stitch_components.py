from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ink.recipes.components import label_components
from ink.recipes.stitch.data import coerce_component_specs


@dataclass(frozen=True)
class DetectedComponentRegion:
    bbox: tuple[int, int, int, int]
    mask: np.ndarray


@dataclass(frozen=True)
class ComponentEvalItem:
    report_key: str
    segment_id: str
    region: DetectedComponentRegion


def component_report_key(component_key: tuple[str, int]) -> str:
    segment_id, component_idx = component_key
    return f"{str(segment_id)}#component_{int(component_idx)}"


def component_bbox(
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


def detect_component_regions(
    mask: np.ndarray,
    *,
    connectivity: int,
    offset: tuple[int, int] = (0, 0),
) -> tuple[DetectedComponentRegion, ...]:
    component_labels, n_components = label_components(mask, connectivity=connectivity)
    off_y, off_x = [int(v) for v in offset]
    components: list[DetectedComponentRegion] = []
    for component_label in range(1, int(n_components) + 1):
        local_mask = np.asarray(component_labels == component_label, dtype=bool)
        if not bool(local_mask.any()):
            continue
        ys, xs = np.where(local_mask)
        local_y0 = int(ys.min())
        local_y1 = int(ys.max()) + 1
        local_x0 = int(xs.min())
        local_x1 = int(xs.max()) + 1
        components.append(
            DetectedComponentRegion(
                bbox=(
                    int(off_y + local_y0),
                    int(off_y + local_y1),
                    int(off_x + local_x0),
                    int(off_x + local_x1),
                ),
                mask=np.asarray(local_mask[local_y0:local_y1, local_x0:local_x1], dtype=bool),
            )
        )
    return tuple(components)


@dataclass
class StitchComponentCatalog:
    explicit_specs: tuple[Any, ...] = ()
    segment_ids: tuple[str, ...] = ()
    connectivity: int = 2
    downsample: int = 1
    _detected_components: dict[str, tuple[DetectedComponentRegion, ...]] = field(default_factory=dict, repr=False)

    @classmethod
    def build(
        cls,
        *,
        raw_components,
        segment_ids,
        downsample: int,
        connectivity: int,
    ) -> StitchComponentCatalog:
        component_specs = tuple(coerce_component_specs(raw_components)) if raw_components else ()
        for component in component_specs:
            if component.bbox is None:
                raise ValueError(
                    "StitchEval explicit components require bbox for every component spec; "
                    f"missing bbox for component_key={component.component_key!r}"
                )
        return cls(
            explicit_specs=component_specs,
            segment_ids=tuple(str(segment_id) for segment_id in segment_ids),
            connectivity=int(connectivity),
            downsample=int(downsample),
            _detected_components={},
        )

    def iter_items(
        self,
        *,
        store,
        read_bbox_label_and_supervision: Callable[..., tuple[np.ndarray, np.ndarray]],
    ) -> Iterator[ComponentEvalItem]:
        if self.explicit_specs:
            for component in self.explicit_specs:
                yield self._explicit_item(
                    component=component,
                    store=store,
                    read_bbox_label_and_supervision=read_bbox_label_and_supervision,
                )
            return

        for segment_id in self.segment_ids:
            for component_idx, region in enumerate(
                self.detected_segment_components(
                    segment_id=segment_id,
                    store=store,
                    read_bbox_label_and_supervision=read_bbox_label_and_supervision,
                )
            ):
                yield ComponentEvalItem(
                    report_key=component_report_key((str(segment_id), int(component_idx))),
                    segment_id=str(segment_id),
                    region=region,
                )

    def _explicit_item(
        self,
        *,
        component,
        store,
        read_bbox_label_and_supervision: Callable[..., tuple[np.ndarray, np.ndarray]],
    ) -> ComponentEvalItem:
        segment_id, _component_idx = component.component_key
        segment_ds_shape = store.segment_ds_shape(segment_id)
        bbox = component_bbox(
            component,
            downsample=int(self.downsample),
            segment_ds_shape=segment_ds_shape,
        )
        if bbox is None:
            raise ValueError(
                "StitchEval explicit component bbox becomes empty after downsampling/clamping; "
                f"component_key={component.component_key!r} bbox={component.bbox!r}"
            )
        labels, supervision = read_bbox_label_and_supervision(
            segment_id=str(segment_id),
            bbox=bbox,
        )
        regions = detect_component_regions(
            np.asarray(labels, dtype=bool) & np.asarray(supervision, dtype=bool),
            connectivity=int(self.connectivity),
            offset=(int(bbox[0]), int(bbox[2])),
        )
        if len(regions) != 1:
            raise ValueError(
                "StitchEval explicit component bbox must isolate exactly one supervised GT component; "
                f"component_key={component.component_key!r} bbox={bbox!r} found={len(regions)}"
            )
        return ComponentEvalItem(
            report_key=component_report_key(component.component_key),
            segment_id=str(segment_id),
            region=regions[0],
        )

    def detected_segment_components(
        self,
        *,
        segment_id: str,
        store,
        read_bbox_label_and_supervision: Callable[..., tuple[np.ndarray, np.ndarray]],
    ) -> tuple[DetectedComponentRegion, ...]:
        segment_id = str(segment_id)
        cached = self._detected_components.get(segment_id)
        if cached is not None:
            return cached

        segment_ds_shape = store.segment_ds_shape(segment_id)
        full_segment_bbox = (0, int(segment_ds_shape[0]), 0, int(segment_ds_shape[1]))
        labels, supervision = read_bbox_label_and_supervision(
            segment_id=segment_id,
            bbox=full_segment_bbox,
            cache=False,
        )

        components: list[DetectedComponentRegion] = []
        supervision_rois = detect_component_regions(
            supervision,
            connectivity=int(self.connectivity),
        )
        for roi_region in supervision_rois:
            y0, y1, x0, x1 = [int(v) for v in roi_region.bbox]
            component_source = np.asarray(labels[y0:y1, x0:x1], dtype=bool) & np.asarray(
                supervision[y0:y1, x0:x1],
                dtype=bool,
            )
            components.extend(
                detect_component_regions(
                    component_source,
                    connectivity=int(self.connectivity),
                    offset=(y0, x0),
                )
            )

        detected = tuple(components)
        self._detected_components[segment_id] = detected
        return detected
