from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ink.core.types import DataBundle


_STITCH_CONFIG_KEYS = {
    "downsample",
    "use_roi",
    "eval",
    "train",
}
_STITCH_TRAIN_CONFIG_KEYS = {
    "segments",
    "components",
    "borders",
    "viz",
    "loss",
}
_STITCH_EVAL_CONFIG_KEYS = {
    "segments",
    "loader_to_segment",
    "metrics",
    "borders",
}
_STITCH_TRAIN_VIZ_CONFIG_KEYS = {
    "enabled",
    "every_n_epochs",
    "segment_ids",
    "metrics",
}
_STITCH_TRAIN_LOSS_CONFIG_KEYS = {
    "patch_batch_size",
    "valid_batch_size",
    "patch_loss_weight",
    "gradient_checkpointing",
    "save_on_cpu",
    "terms",
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


def _coerce_metric_names(raw_names) -> tuple[str, ...]:
    if raw_names is None:
        return ()
    if isinstance(raw_names, (str, bytes, bytearray)):
        return (str(raw_names),)
    if not isinstance(raw_names, Sequence):
        raise TypeError("stitch metrics must be a list or tuple of metric names")
    return tuple(str(name) for name in raw_names)


def _coerce_stitch_terms(raw_terms) -> tuple[object, ...]:
    if raw_terms is None:
        return ()
    if isinstance(raw_terms, tuple):
        return raw_terms
    if isinstance(raw_terms, list):
        return tuple(raw_terms)
    return (raw_terms,)


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
    borders_by_split: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {"train": {}, "eval": {}}
    )

    def __post_init__(self) -> None:
        self.downsample = int(self.downsample)
        self.use_roi = bool(self.use_roi)
        self.borders_by_split = {
            "train": dict((self.borders_by_split or {}).get("train", {})),
            "eval": dict((self.borders_by_split or {}).get("eval", {})),
        }


@dataclass
class EvalStitchConfig:
    segments: list[StitchSegmentSpec] = field(default_factory=list)
    loader_to_segment: dict[int, str] = field(default_factory=dict)
    metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.segments = coerce_segment_specs(self.segments)
        self.loader_to_segment = {
            int(loader_idx): str(segment_id)
            for loader_idx, segment_id in dict(self.loader_to_segment or {}).items()
        }
        self.metrics = _coerce_metric_names(self.metrics)


@dataclass
class TrainStitchVizConfig:
    enabled: bool = False
    every_n_epochs: int = 1
    segment_ids: list[str] = field(default_factory=list)
    metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.every_n_epochs = int(self.every_n_epochs)
        self.segment_ids = [str(segment_id) for segment_id in (self.segment_ids or [])]
        self.metrics = _coerce_metric_names(self.metrics)


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
    segments: list[StitchSegmentSpec] = field(default_factory=list)
    components: list[StitchComponentSpec] = field(default_factory=list)
    viz: TrainStitchVizConfig = field(default_factory=TrainStitchVizConfig)
    loss: TrainStitchLossConfig = field(default_factory=TrainStitchLossConfig)

    def __post_init__(self) -> None:
        self.segments = coerce_segment_specs(self.segments)
        self.components = coerce_component_specs(self.components)
        if not isinstance(self.viz, TrainStitchVizConfig):
            self.viz = TrainStitchVizConfig(**dict(self.viz or {}))
        if not isinstance(self.loss, TrainStitchLossConfig):
            self.loss = TrainStitchLossConfig(**dict(self.loss or {}))

    @property
    def component_keys(self) -> list[tuple[str, int]]:
        return [spec.component_key for spec in self.components]


@dataclass
class StitchData:
    layout: StitchLayout = field(default_factory=StitchLayout)
    eval: EvalStitchConfig = field(default_factory=EvalStitchConfig)
    train: TrainStitchConfig = field(default_factory=TrainStitchConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.layout, StitchLayout):
            self.layout = StitchLayout(**dict(self.layout or {}))
        if not isinstance(self.eval, EvalStitchConfig):
            self.eval = EvalStitchConfig(**dict(self.eval or {}))
        if not isinstance(self.train, TrainStitchConfig):
            self.train = TrainStitchConfig(**dict(self.train or {}))

    @property
    def enabled(self) -> bool:
        return bool(self.eval.segments or self.train.segments or self.train.components)

    @classmethod
    def from_bundle(cls, bundle: DataBundle) -> StitchData:
        extras = dict(bundle.extras or {})
        return cls.from_config(extras.get("stitch") or {})

    @classmethod
    def from_config(cls, stitch_cfg) -> StitchData:
        if isinstance(stitch_cfg, StitchData):
            return stitch_cfg

        stitch_cfg = _coerce_mapping(stitch_cfg, "stitch")
        if not stitch_cfg:
            return cls()

        _raise_on_unknown_keys("stitch", stitch_cfg, _STITCH_CONFIG_KEYS)

        train_cfg = _coerce_mapping(stitch_cfg.get("train"), "stitch.train")
        eval_cfg = _coerce_mapping(stitch_cfg.get("eval"), "stitch.eval")
        _raise_on_unknown_keys("stitch.train", train_cfg, _STITCH_TRAIN_CONFIG_KEYS)
        _raise_on_unknown_keys("stitch.eval", eval_cfg, _STITCH_EVAL_CONFIG_KEYS)

        train_viz_cfg = _coerce_mapping(train_cfg.get("viz"), "stitch.train.viz")
        train_loss_cfg = _coerce_mapping(train_cfg.get("loss"), "stitch.train.loss")
        _raise_on_unknown_keys("stitch.train.viz", train_viz_cfg, _STITCH_TRAIN_VIZ_CONFIG_KEYS)
        _raise_on_unknown_keys("stitch.train.loss", train_loss_cfg, _STITCH_TRAIN_LOSS_CONFIG_KEYS)

        return cls(
            layout=StitchLayout(
                downsample=int(stitch_cfg.get("downsample", 1)),
                use_roi=bool(stitch_cfg.get("use_roi", False)),
                borders_by_split={
                    "train": dict(train_cfg.get("borders") or {}),
                    "eval": dict(eval_cfg.get("borders") or {}),
                },
            ),
            eval=EvalStitchConfig(
                segments=coerce_segment_specs(eval_cfg.get("segments")),
                loader_to_segment={
                    int(loader_idx): str(segment_id)
                    for loader_idx, segment_id in dict(eval_cfg.get("loader_to_segment") or {}).items()
                },
                metrics=eval_cfg.get("metrics"),
            ),
            train=TrainStitchConfig(
                segments=coerce_segment_specs(train_cfg.get("segments")),
                components=coerce_component_specs(train_cfg.get("components")),
                viz=TrainStitchVizConfig(
                    enabled=bool(train_viz_cfg.get("enabled", False)),
                    every_n_epochs=int(train_viz_cfg.get("every_n_epochs", 1)),
                    segment_ids=train_viz_cfg.get("segment_ids") or [],
                    metrics=train_viz_cfg.get("metrics"),
                ),
                loss=TrainStitchLossConfig(
                    patch_batch_size=int(train_loss_cfg.get("patch_batch_size", 1)),
                    valid_batch_size=int(train_loss_cfg.get("valid_batch_size", 1)),
                    patch_loss_weight=float(train_loss_cfg.get("patch_loss_weight", 1.0)),
                    gradient_checkpointing=bool(train_loss_cfg.get("gradient_checkpointing", False)),
                    save_on_cpu=bool(train_loss_cfg.get("save_on_cpu", False)),
                    terms=train_loss_cfg.get("terms"),
                ),
            ),
        )


def _segment_spec_to_config(spec: StitchSegmentSpec) -> dict[str, Any]:
    out = {
        "segment_id": spec.segment_id,
        "shape": tuple(spec.shape),
    }
    if spec.bbox is not None:
        out["bbox"] = spec.bbox
    return out


def _component_spec_to_config(spec: StitchComponentSpec) -> dict[str, Any]:
    out = {
        "component_key": tuple(spec.component_key),
        "shape": tuple(spec.shape),
    }
    if spec.bbox is not None:
        out["bbox"] = spec.bbox
    return out


def stitch_data_to_config(stitch_data: StitchData | dict | None) -> dict[str, Any]:
    data = StitchData.from_config(stitch_data or {})
    return {
        "downsample": int(data.layout.downsample),
        "use_roi": bool(data.layout.use_roi),
        "eval": {
            "segments": [_segment_spec_to_config(spec) for spec in data.eval.segments],
            "loader_to_segment": dict(data.eval.loader_to_segment),
            "metrics": list(data.eval.metrics),
            "borders": dict(data.layout.borders_by_split["eval"]),
        },
        "train": {
            "segments": [_segment_spec_to_config(spec) for spec in data.train.segments],
            "components": [_component_spec_to_config(spec) for spec in data.train.components],
            "borders": dict(data.layout.borders_by_split["train"]),
            "viz": {
                "enabled": bool(data.train.viz.enabled),
                "every_n_epochs": int(data.train.viz.every_n_epochs),
                "segment_ids": list(data.train.viz.segment_ids),
                "metrics": list(data.train.viz.metrics),
            },
            "loss": {
                "patch_batch_size": int(data.train.loss.patch_batch_size),
                "valid_batch_size": int(data.train.loss.valid_batch_size),
                "patch_loss_weight": float(data.train.loss.patch_loss_weight),
                "gradient_checkpointing": bool(data.train.loss.gradient_checkpointing),
                "save_on_cpu": bool(data.train.loss.save_on_cpu),
                "terms": tuple(data.train.loss.terms),
            },
        },
    }
