from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from ink.recipes.components import label_components
from ink.recipes.stitch.config import coerce_component_specs


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

    @classmethod
    def build(
        cls,
        *,
        raw_components,
        segment_ids,
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
        )

    def iter_items(
        self,
        *,
        store,
        read_bbox_label_and_supervision: Callable[..., tuple[np.ndarray, np.ndarray]],
        detected_segment_components: Callable[..., tuple[DetectedComponentRegion, ...]] | None = None,
    ) -> Iterator[ComponentEvalItem]:
        if self.explicit_specs:
            for component in self.explicit_specs:
                yield self._explicit_item(
                    component=component,
                    store=store,
                    read_bbox_label_and_supervision=read_bbox_label_and_supervision,
                )
            return

        if detected_segment_components is None:
            raise TypeError("implicit stitched component discovery requires detected_segment_components(...)")
        for segment_id in self.segment_ids:
            for component_idx, region in enumerate(
                detected_segment_components(
                    segment_id=segment_id,
                    store=store,
                    connectivity=int(self.connectivity),
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
        if component.bbox is None:
            raise ValueError(
                "StitchEval explicit components require bbox for every component spec; "
                f"missing bbox for component_key={component.component_key!r}"
            )
        y0, y1, x0, x1 = [int(v) for v in component.bbox]
        max_y, max_x = [int(v) for v in segment_ds_shape]
        bbox = (
            max(0, min(y0, max_y)),
            max(0, min(y1, max_y)),
            max(0, min(x0, max_x)),
            max(0, min(x1, max_x)),
        )
        if bbox[1] <= bbox[0] or bbox[3] <= bbox[2]:
            raise ValueError(
                "StitchEval explicit component bbox becomes empty after clamping; "
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
