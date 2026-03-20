from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

from ink.recipes.stitch.data import StitchData, normalize_component_key
from ink.recipes.stitch.ops import allocate_segment_buffers, build_segment_roi_meta


def _noop_log(*_args, **_kwargs) -> None:
    return None


def _null_precision_context():
    return nullcontext()


_UNSET = object()


def _log(owner, message) -> None:
    logger = getattr(owner, "log", None)
    if callable(logger):
        logger(message)


def _deep_merge_dicts(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


def _coerce_component_dataset_map(raw_datasets) -> dict[tuple[str, int], Any]:
    if raw_datasets is None:
        return {}
    return {
        normalize_component_key(component_key): dataset
        for component_key, dataset in dict(raw_datasets).items()
    }


@dataclass
class _StitchState:
    data: StitchData
    roi_meta_by_split: dict[str, dict[str, dict]] = field(init=False)
    roi_buffers_by_split: dict[str, dict[str, list]] = field(init=False)
    train_component_meta: dict[tuple[str, int], dict] = field(init=False)
    _gaussian_cache: dict = field(default_factory=dict)
    _torch_gaussian_cache: dict = field(default_factory=dict)
    _boundary_dist_maps_cpu: dict = field(default_factory=dict)
    _boundary_dist_maps_torch: dict = field(default_factory=dict)
    _gaussian_sigma_scale: float = 1.0 / 8.0
    _gaussian_min_weight: float = 1e-6

    def __post_init__(self) -> None:
        assert isinstance(self.data, StitchData)
        self.rebuild()

    @property
    def enabled(self) -> bool:
        return bool(
            self.roi_buffers_by_split["eval"] or self.roi_meta_by_split["train"] or self.train_component_meta
        )

    def clear_boundary_caches(self) -> None:
        self._boundary_dist_maps_cpu.clear()
        self._boundary_dist_maps_torch.clear()

    def rebuild(self) -> None:
        self.roi_meta_by_split = {"eval": {}, "train": {}}
        self.roi_buffers_by_split = {"eval": {}}
        self.train_component_meta = {}
        self.clear_boundary_caches()

        layout = self.data.layout
        for spec in self.data.eval.segments:
            self._register_segment("eval", spec.segment_id, spec.shape, spec.bbox, layout.downsample)
        for spec in self.data.train.segments:
            self._register_segment("train", spec.segment_id, spec.shape, spec.bbox, layout.downsample)
        for spec in self.data.train.components:
            self.train_component_meta[spec.component_key] = build_segment_roi_meta(
                spec.shape,
                spec.bbox,
                layout.downsample,
                use_roi=True,
            )

    def _register_segment(self, split: str, segment_id: str, shape, bbox, downsample: int) -> None:
        split_name = str(split)
        sid = str(segment_id)
        self.roi_meta_by_split[split_name][sid] = build_segment_roi_meta(
            shape,
            bbox,
            downsample,
            use_roi=self.data.layout.use_roi,
        )
        if split_name == "eval":
            self.roi_buffers_by_split["eval"][sid] = allocate_segment_buffers(self.roi_meta_by_split["eval"][sid])

    def reset_split_buffers(self, split: str) -> None:
        buffers_by_segment = self.roi_buffers_by_split.get(str(split))
        if not buffers_by_segment:
            return
        for roi_buffers in buffers_by_segment.values():
            for pred_buf, count_buf, _offset in roi_buffers:
                pred_buf.fill(0)
                count_buf.fill(0)


@dataclass
class StitchExecutionContext:
    precision_context_factory: object = _null_precision_context
    sanity_checking: bool = False
    is_global_zero: bool = True
    distributed_world_size: int = 1
    distributed_reduce_sum: object = None

    def __post_init__(self) -> None:
        self.sanity_checking = bool(self.sanity_checking)
        self.is_global_zero = bool(self.is_global_zero)
        self.distributed_world_size = int(self.distributed_world_size)
        if self.distributed_world_size < 1:
            raise ValueError(
                f"stitch execution distributed_world_size must be >= 1, got {self.distributed_world_size!r}"
            )

    def forward_context(self):
        factory = self.precision_context_factory
        if callable(factory):
            context = factory()
            if context is not None:
                return context
        return nullcontext()

    def reduce_sum(self, tensor):
        reducer = self.distributed_reduce_sum
        if not callable(reducer):
            raise RuntimeError(
                "distributed stitch reduction requested but "
                "execution.distributed_reduce_sum is unavailable"
            )
        return reducer(tensor)
