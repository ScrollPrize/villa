from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


_STITCH_CONFIG_KEYS = {
    "downsample",
    "use_roi",
    "eval",
    "log_only",
    "train",
}
_STITCH_TRAIN_CONFIG_KEYS = {
    "segment_ids",
    "segments",
    "components",
    "viz",
    "loss",
}
_STITCH_EVAL_CONFIG_KEYS = {
    "segment_ids",
    "segments",
}
_STITCH_TRAIN_VIZ_CONFIG_KEYS = {
    "enabled",
    "every_n_epochs",
    "loss_components",
}
_STITCH_TRAIN_LOSS_CONFIG_KEYS = {
    "patch_batch_size",
    "valid_batch_size",
    "patch_loss_weight",
    "gradient_checkpointing",
    "save_on_cpu",
    "terms",
}
_STITCH_LOG_ONLY_CONFIG_KEYS = {
    "segments",
    "segment_ids",
    "every_n_epochs",
}


def _coerce_mapping(raw_cfg, path: str) -> dict[str, Any]:
    if raw_cfg is None:
        return {}
    if not isinstance(raw_cfg, Mapping):
        raise TypeError(f"{path} must be a mapping, got {raw_cfg!r}")
    return dict(raw_cfg)


def _raise_on_unknown_keys(path: str, cfg: Mapping[str, Any], allowed_keys: set[str]) -> None:
    unsupported_keys = sorted(set(cfg.keys()).difference(allowed_keys))
    if not unsupported_keys:
        return
    unsupported_path_list = ", ".join(f"{path}.{key}" for key in unsupported_keys)
    supported_path_list = ", ".join(f"{path}.{key}" for key in sorted(allowed_keys))
    raise ValueError(
        f"unsupported stitch config keys: {unsupported_path_list}. "
        f"Supported keys: {supported_path_list}."
    )


def normalize_component_key(component_key):
    if not isinstance(component_key, (list, tuple)) or len(component_key) != 2:
        raise ValueError(f"component_key must be a pair (segment_id, component_idx), got {component_key!r}")
    return str(component_key[0]), int(component_key[1])


def _coerce_shape(shape) -> tuple[int, int]:
    if not isinstance(shape, Sequence) or len(shape) < 2:
        raise ValueError(f"stitch shape must be a sequence with at least two values, got {shape!r}")
    return int(shape[0]), int(shape[1])


def _ensure_unique(name: str, values) -> None:
    seen = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {name}: {value!r}")
        seen.add(value)


def _coerce_loss_component_names(raw_names) -> tuple[str, ...]:
    if raw_names is None:
        return ()
    if isinstance(raw_names, (str, bytes, bytearray)):
        return (str(raw_names),)
    if not isinstance(raw_names, Sequence):
        raise TypeError("stitch train viz loss_components must be a list or tuple of names")
    return tuple(str(name) for name in raw_names)


def _coerce_stitch_terms(raw_terms) -> tuple[object, ...]:
    if raw_terms is None:
        return ()
    if isinstance(raw_terms, tuple):
        return raw_terms
    if isinstance(raw_terms, list):
        return tuple(raw_terms)
    return (raw_terms,)


def _coerce_segment_ids(raw_segment_ids) -> list[str]:
    if raw_segment_ids is None:
        return []
    if isinstance(raw_segment_ids, (str, bytes, bytearray)):
        return [str(raw_segment_ids)]
    if not isinstance(raw_segment_ids, Sequence):
        raise TypeError("stitch segment_ids must be a list or tuple of segment ids")
    return [str(segment_id) for segment_id in raw_segment_ids]


def _validated_stitch_config(
    stitch_cfg,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    stitch_cfg = _coerce_mapping(stitch_cfg, "stitch")
    if not stitch_cfg:
        return {}, {}, {}, {}, {}, {}

    _raise_on_unknown_keys("stitch", stitch_cfg, _STITCH_CONFIG_KEYS)

    train_cfg = _coerce_mapping(stitch_cfg.get("train"), "stitch.train")
    eval_cfg = _coerce_mapping(stitch_cfg.get("eval"), "stitch.eval")
    log_only_cfg = _coerce_mapping(stitch_cfg.get("log_only"), "stitch.log_only")
    _raise_on_unknown_keys("stitch.train", train_cfg, _STITCH_TRAIN_CONFIG_KEYS)
    _raise_on_unknown_keys("stitch.eval", eval_cfg, _STITCH_EVAL_CONFIG_KEYS)
    _raise_on_unknown_keys("stitch.log_only", log_only_cfg, _STITCH_LOG_ONLY_CONFIG_KEYS)

    train_viz_cfg = _coerce_mapping(train_cfg.get("viz"), "stitch.train.viz")
    train_loss_cfg = _coerce_mapping(train_cfg.get("loss"), "stitch.train.loss")
    _raise_on_unknown_keys("stitch.train.viz", train_viz_cfg, _STITCH_TRAIN_VIZ_CONFIG_KEYS)
    _raise_on_unknown_keys("stitch.train.loss", train_loss_cfg, _STITCH_TRAIN_LOSS_CONFIG_KEYS)
    return stitch_cfg, eval_cfg, train_cfg, train_viz_cfg, train_loss_cfg, log_only_cfg


@dataclass
class StitchSegmentSpec:
    segment_id: str
    shape: tuple[int, int]
    bbox: Any = None

    def __post_init__(self) -> None:
        self.segment_id = str(self.segment_id)
        self.shape = _coerce_shape(self.shape)


@dataclass
class StitchComponentSpec:
    component_key: tuple[str, int]
    shape: tuple[int, int]
    bbox: Any = None

    def __post_init__(self) -> None:
        self.component_key = normalize_component_key(self.component_key)
        self.shape = _coerce_shape(self.shape)


def coerce_segment_specs(raw_specs) -> list[StitchSegmentSpec]:
    if raw_specs is None:
        return []
    if isinstance(raw_specs, Mapping):
        raise TypeError("stitch segments must be a list of structured segment specs")
    if not isinstance(raw_specs, Sequence) or isinstance(raw_specs, (str, bytes, bytearray)):
        raise TypeError("stitch segments must be a list of structured segment specs")

    out = []
    for item in raw_specs:
        if isinstance(item, StitchSegmentSpec):
            out.append(item)
            continue
        if not isinstance(item, Mapping):
            raise TypeError(f"unsupported stitch segment spec entry: {item!r}")
        if "segment_id" not in item:
            raise ValueError("stitch segment spec mapping requires segment_id")
        if "shape" not in item:
            raise ValueError("stitch segment spec mapping requires shape")
        out.append(
            StitchSegmentSpec(
                segment_id=item["segment_id"],
                shape=item["shape"],
                bbox=item.get("bbox"),
            )
        )

    _ensure_unique("stitch segment id", [spec.segment_id for spec in out])
    return out


def coerce_component_specs(raw_specs) -> list[StitchComponentSpec]:
    if raw_specs is None:
        return []
    if isinstance(raw_specs, Mapping):
        raise TypeError("stitch components must be a list of structured component specs")
    if not isinstance(raw_specs, Sequence) or isinstance(raw_specs, (str, bytes, bytearray)):
        raise TypeError("stitch components must be a list of structured component specs")

    out = []
    for item in raw_specs:
        if isinstance(item, StitchComponentSpec):
            out.append(item)
            continue
        if not isinstance(item, Mapping):
            raise TypeError(f"unsupported stitch component spec entry: {item!r}")
        if "component_key" not in item:
            raise ValueError("stitch component spec mapping requires component_key")
        if "shape" not in item:
            raise ValueError("stitch component spec mapping requires shape")
        out.append(
            StitchComponentSpec(
                component_key=item["component_key"],
                shape=item["shape"],
                bbox=item.get("bbox"),
            )
        )

    _ensure_unique("train component key", [spec.component_key for spec in out])
    return out


@dataclass
class StitchLayout:
    downsample: int = 1
    use_roi: bool = False

    def __post_init__(self) -> None:
        self.downsample = int(self.downsample)
        self.use_roi = bool(self.use_roi)


@dataclass
class EvalStitchConfig:
    segment_ids: list[str] = field(default_factory=list)
    segments: list[StitchSegmentSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.segment_ids = _coerce_segment_ids(self.segment_ids)
        self.segments = coerce_segment_specs(self.segments)


@dataclass
class LogOnlyStitchConfig:
    segments: list[StitchSegmentSpec] = field(default_factory=list)
    segment_ids: list[str] = field(default_factory=list)
    every_n_epochs: int = 1

    def __post_init__(self) -> None:
        self.segments = coerce_segment_specs(self.segments)
        self.segment_ids = _coerce_segment_ids(self.segment_ids)
        self.every_n_epochs = int(self.every_n_epochs)
        if not self.segment_ids and self.segments:
            self.segment_ids = [spec.segment_id for spec in self.segments]


@dataclass
class TrainStitchVizConfig:
    enabled: bool = False
    every_n_epochs: int = 1
    loss_components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.every_n_epochs = int(self.every_n_epochs)
        self.loss_components = _coerce_loss_component_names(self.loss_components)


@dataclass
class TrainStitchLossConfig:
    patch_batch_size: int = 1
    valid_batch_size: int = 1
    patch_loss_weight: float = 1.0
    gradient_checkpointing: bool = False
    save_on_cpu: bool = False
    terms: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        self.patch_batch_size = int(self.patch_batch_size)
        self.valid_batch_size = int(self.valid_batch_size)
        self.patch_loss_weight = float(self.patch_loss_weight)
        self.gradient_checkpointing = bool(self.gradient_checkpointing)
        self.save_on_cpu = bool(self.save_on_cpu)
        self.terms = _coerce_stitch_terms(self.terms)
        if self.patch_loss_weight < 0.0:
            raise ValueError(f"stitch patch_loss_weight must be >= 0, got {self.patch_loss_weight!r}")


@dataclass
class TrainStitchConfig:
    segment_ids: list[str] = field(default_factory=list)
    segments: list[StitchSegmentSpec] = field(default_factory=list)
    components: list[StitchComponentSpec] = field(default_factory=list)
    viz: TrainStitchVizConfig = field(default_factory=TrainStitchVizConfig)
    loss: TrainStitchLossConfig = field(default_factory=TrainStitchLossConfig)

    def __post_init__(self) -> None:
        self.segment_ids = _coerce_segment_ids(self.segment_ids)
        self.segments = coerce_segment_specs(self.segments)
        self.components = coerce_component_specs(self.components)
        if not isinstance(self.viz, TrainStitchVizConfig):
            self.viz = TrainStitchVizConfig(**dict(self.viz or {}))
        if not isinstance(self.loss, TrainStitchLossConfig):
            self.loss = TrainStitchLossConfig(**dict(self.loss or {}))

    @property
    def component_keys(self) -> list[tuple[str, int]]:
        return [spec.component_key for spec in self.components]


def _build_layout_config(stitch_cfg: Mapping[str, Any], *, base: StitchLayout | None = None) -> StitchLayout:
    return StitchLayout(
        downsample=int(stitch_cfg.get("downsample", 1 if base is None else base.downsample)),
        use_roi=bool(stitch_cfg.get("use_roi", False if base is None else base.use_roi)),
    )


def _build_eval_stitch_config(eval_cfg: Mapping[str, Any], *, base: EvalStitchConfig | None = None) -> EvalStitchConfig:
    return EvalStitchConfig(
        segment_ids=eval_cfg.get("segment_ids", [] if base is None else base.segment_ids),
        segments=eval_cfg.get("segments", [] if base is None else base.segments),
    )


def _build_train_viz_config(
    train_viz_cfg: Mapping[str, Any],
    *,
    base: TrainStitchVizConfig | None = None,
) -> TrainStitchVizConfig:
    return TrainStitchVizConfig(
        enabled=bool(train_viz_cfg.get("enabled", False if base is None else base.enabled)),
        every_n_epochs=int(train_viz_cfg.get("every_n_epochs", 1 if base is None else base.every_n_epochs)),
        loss_components=train_viz_cfg.get("loss_components", () if base is None else base.loss_components),
    )


def _build_train_loss_config(
    train_loss_cfg: Mapping[str, Any],
    *,
    base: TrainStitchLossConfig | None = None,
) -> TrainStitchLossConfig:
    return TrainStitchLossConfig(
        patch_batch_size=int(train_loss_cfg.get("patch_batch_size", 1 if base is None else base.patch_batch_size)),
        valid_batch_size=int(train_loss_cfg.get("valid_batch_size", 1 if base is None else base.valid_batch_size)),
        patch_loss_weight=float(train_loss_cfg.get("patch_loss_weight", 1.0 if base is None else base.patch_loss_weight)),
        gradient_checkpointing=bool(
            train_loss_cfg.get("gradient_checkpointing", False if base is None else base.gradient_checkpointing)
        ),
        save_on_cpu=bool(train_loss_cfg.get("save_on_cpu", False if base is None else base.save_on_cpu)),
        terms=train_loss_cfg.get("terms", () if base is None else base.terms),
    )


def _build_train_stitch_config(
    train_cfg: Mapping[str, Any],
    train_viz_cfg: Mapping[str, Any],
    train_loss_cfg: Mapping[str, Any],
    *,
    base: TrainStitchConfig | None = None,
) -> TrainStitchConfig:
    return TrainStitchConfig(
        segment_ids=train_cfg.get("segment_ids", [] if base is None else base.segment_ids),
        segments=train_cfg.get("segments", [] if base is None else base.segments),
        components=train_cfg.get("components", [] if base is None else base.components),
        viz=_build_train_viz_config(train_viz_cfg, base=None if base is None else base.viz),
        loss=_build_train_loss_config(train_loss_cfg, base=None if base is None else base.loss),
    )


def _build_log_only_stitch_config(
    log_only_cfg: Mapping[str, Any],
    *,
    base: LogOnlyStitchConfig | None = None,
) -> LogOnlyStitchConfig:
    return LogOnlyStitchConfig(
        segments=log_only_cfg.get("segments", [] if base is None else base.segments),
        segment_ids=log_only_cfg.get("segment_ids", [] if base is None else base.segment_ids),
        every_n_epochs=int(log_only_cfg.get("every_n_epochs", 1 if base is None else base.every_n_epochs)),
    )


@dataclass
class StitchData:
    layout: StitchLayout = field(default_factory=StitchLayout)
    eval: EvalStitchConfig = field(default_factory=EvalStitchConfig)
    train: TrainStitchConfig = field(default_factory=TrainStitchConfig)
    log_only: LogOnlyStitchConfig = field(default_factory=LogOnlyStitchConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.layout, StitchLayout):
            self.layout = StitchLayout(**dict(self.layout or {}))
        if not isinstance(self.eval, EvalStitchConfig):
            self.eval = EvalStitchConfig(**dict(self.eval or {}))
        if not isinstance(self.train, TrainStitchConfig):
            self.train = TrainStitchConfig(**dict(self.train or {}))
        if not isinstance(self.log_only, LogOnlyStitchConfig):
            self.log_only = LogOnlyStitchConfig(**dict(self.log_only or {}))

    @classmethod
    def from_config(cls, stitch_cfg) -> StitchData:
        if isinstance(stitch_cfg, StitchData):
            return stitch_cfg

        stitch_cfg, eval_cfg, train_cfg, train_viz_cfg, train_loss_cfg, log_only_cfg = _validated_stitch_config(
            stitch_cfg
        )
        if not stitch_cfg:
            return cls()

        return cls(
            layout=_build_layout_config(stitch_cfg),
            eval=_build_eval_stitch_config(eval_cfg),
            train=_build_train_stitch_config(train_cfg, train_viz_cfg, train_loss_cfg),
            log_only=_build_log_only_stitch_config(log_only_cfg),
        )


def merge_stitch_data(base: StitchData, overlay) -> StitchData:
    base = StitchData.from_config(base)
    if overlay is None:
        return base
    if isinstance(overlay, StitchData):
        return overlay

    overlay_cfg, eval_cfg, train_cfg, train_viz_cfg, train_loss_cfg, log_only_cfg = _validated_stitch_config(overlay)
    if not overlay_cfg:
        return base

    return StitchData(
        layout=_build_layout_config(overlay_cfg, base=base.layout),
        eval=_build_eval_stitch_config(eval_cfg, base=base.eval),
        train=_build_train_stitch_config(train_cfg, train_viz_cfg, train_loss_cfg, base=base.train),
        log_only=_build_log_only_stitch_config(log_only_cfg, base=base.log_only),
    )
