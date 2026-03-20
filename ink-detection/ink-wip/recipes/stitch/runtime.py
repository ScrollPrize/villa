from __future__ import annotations

from dataclasses import dataclass, field

from ink.recipes.stitch.data import StitchData, normalize_component_key
from ink.recipes.stitch.ops import allocate_segment_buffers, build_segment_roi_meta
from ink.recipes.stitch.terms import compute_stitched_loss_components
from ink.recipes.stitch.zarr_prep import prepare_stitch_data_for_bundle


def _noop_log(*_args, **_kwargs) -> None:
    return None


def _log(owner, message) -> None:
    logger = getattr(owner, "log", None)
    if callable(logger):
        logger(message)


@dataclass
class StitchRuntimeState:
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

    def clear_boundary_caches(self) -> None:
        self._boundary_dist_maps_cpu.clear()
        self._boundary_dist_maps_torch.clear()

    def rebuild(self) -> None:
        self.roi_meta_by_split = {"eval": {}, "train": {}, "log_only": {}}
        self.roi_buffers_by_split = {"eval": {}}
        self.train_component_meta = {}
        self.clear_boundary_caches()

        self._register_split_segments("eval", self.data.eval.segments)
        self._register_split_segments("train", self.data.train.segments)
        self._register_split_segments("log_only", self.data.log_only.segments)
        self._register_train_components()

    def eval_segment_meta(self, segment_id: str) -> dict | None:
        return self.roi_meta_by_split["eval"].get(str(segment_id))

    def train_segment_meta(self, segment_id: str) -> dict | None:
        return self.roi_meta_by_split["train"].get(str(segment_id))

    def component_meta_for(self, component_key) -> dict | None:
        return self.train_component_meta.get(normalize_component_key(component_key))

    def log_only_segment_meta(self, segment_id: str) -> dict | None:
        return self.roi_meta_by_split["log_only"].get(str(segment_id))

    def _register_split_segments(self, split: str, segment_specs) -> None:
        downsample = int(self.data.layout.downsample)
        for spec in segment_specs:
            self._register_segment(split, spec.segment_id, spec.shape, spec.bbox, downsample)

    def _register_train_components(self) -> None:
        downsample = int(self.data.layout.downsample)
        for spec in self.data.train.components:
            self.train_component_meta[spec.component_key] = build_segment_roi_meta(
                spec.shape,
                spec.bbox,
                downsample,
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


def _segment_rois_from_meta(meta: dict) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (
            int(roi["offset"][0]),
            int(roi["offset"][0] + roi["buffer_shape"][0]),
            int(roi["offset"][1]),
            int(roi["offset"][1] + roi["buffer_shape"][1]),
        )
        for roi in meta.get("rois", ())
    )


@dataclass(frozen=True)
class SegmentLayout:
    segment_shapes: dict[str, tuple[int, int]]
    segment_rois: dict[str, tuple[tuple[int, int, int, int], ...]]
    downsample: int


@dataclass
class StitchRuntime:
    data: StitchData
    state: StitchRuntimeState
    train: "TrainStitchRuntime"
    log: object = _noop_log

    def segment_layout(self) -> SegmentLayout:
        segment_shapes: dict[str, tuple[int, int]] = {}
        segment_rois: dict[str, tuple[tuple[int, int, int, int], ...]] = {}

        for spec in self.data.eval.segments:
            segment_id = str(spec.segment_id)
            segment_shapes[segment_id] = tuple(int(v) for v in spec.shape)
            meta = self.state.eval_segment_meta(segment_id)
            if meta is None:
                raise KeyError(f"stitch state is missing eval ROI metadata for segment_id={segment_id!r}")
            segment_rois[segment_id] = _segment_rois_from_meta(meta)

        if not segment_shapes:
            raise ValueError("stitch runtime requires stitch.eval.segments with segment shapes")

        return SegmentLayout(
            segment_shapes=segment_shapes,
            segment_rois=segment_rois,
            downsample=int(self.data.layout.downsample),
        )

    @classmethod
    def _from_config(
        cls,
        stitch_data: StitchData | dict | None = None,
        *,
        logger=None,
        patch_loss=None,
    ) -> "StitchRuntime":
        from ink.recipes.stitch.train_runtime import TrainStitchRuntime

        data = StitchData.from_config(stitch_data or {})
        state = StitchRuntimeState(data)
        log = logger or _noop_log
        train = TrainStitchRuntime(
            data=data,
            state=state,
            patch_loss=patch_loss,
            patch_loss_weight=float(data.train.loss.patch_loss_weight),
            gradient_checkpointing=bool(data.train.loss.gradient_checkpointing),
            save_on_cpu=bool(data.train.loss.save_on_cpu),
            log=log,
        )
        return cls(
            data=data,
            state=state,
            train=train,
            log=log,
        )


@dataclass(frozen=True)
class StitchRuntimeRecipe:
    config: StitchData | dict | None = None

    def build(self, bundle, *, runtime=None, logger=None, patch_loss=None) -> StitchRuntime:
        stitch_runtime = StitchRuntime._from_config(
            prepare_stitch_data_for_bundle(bundle, config=self.config),
            logger=logger,
            patch_loss=patch_loss,
        )
        precision_context = getattr(runtime, "precision_context", None)
        if callable(precision_context):
            stitch_runtime.train.precision_context = precision_context
        return stitch_runtime


__all__ = [
    "SegmentLayout",
    "StitchRuntime",
    "StitchRuntimeRecipe",
    "StitchRuntimeState",
    "compute_stitched_loss_components",
]
