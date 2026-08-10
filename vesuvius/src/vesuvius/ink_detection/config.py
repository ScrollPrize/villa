"""Typed data, augmentation, normalization, and sampling configuration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping


InkMode = Literal["flat", "full_3d", "full_3d_single_wrap"]
PatchDiscoveryMode = Literal["labeled", "unlabeled"]
PatchFindingType = Literal["default", "subtiling"]
SamplingStrategy = Literal[
    "uniform", "scroll_segment_balanced", "fixed_scroll_prior_stratified"
]


class _FrozenMapping(Mapping):
    """Small ordered immutable mapping that remains safe across spawn/pickle."""

    __slots__ = ("_items",)

    def __init__(self, values: Mapping | None = None) -> None:
        object.__setattr__(
            self,
            "_items",
            tuple(() if values is None else values.items()),
        )

    def __setattr__(self, _name, _value) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, _name) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __getitem__(self, key):
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self._items) == dict(other.items())

    def __repr__(self) -> str:
        return repr(dict(self._items))

    def __reduce__(self):
        return type(self), (dict(self._items),)


def _tuple3(value: Any, *, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three ZYX dimensions")
    result = tuple(int(item) for item in value)
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} dimensions must all be positive, got {result!r}")
    return result


def _string_mapping(value: Any, *, name: str) -> Mapping[str, str]:
    if value is None:
        return _FrozenMapping()
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return _FrozenMapping({str(key): str(item) for key, item in value.items()})


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenMapping(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, _FrozenMapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return deepcopy(value)


@dataclass(frozen=True)
class NormalizationConfig:
    """One of the eight image normalization modes and its constants."""

    mode: str = "robust_mad"
    percentile_lower: float = 1.0
    percentile_upper: float = 99.0
    clip_min: float | None = None
    clip_max: float | None = None
    divisor: float = 255.0
    mean: float | None = None
    std: float | None = None

    @classmethod
    def from_value(cls, value: Any) -> "NormalizationConfig":
        aliases = {
            "robust": "robust_mad",
            "mad": "robust_mad",
            "robust_percentile": "robust_percentile_span",
            "percentile_span": "robust_percentile_span",
            "min_max": "minmax",
            "percentile_min_max": "percentile_minmax",
            "clipped_minmax": "percentile_minmax",
            "clipped_min_max": "percentile_minmax",
            "divide_255": "divide",
            "identity": "none",
        }
        if value is None:
            authored: Mapping[str, Any] = {}
        elif isinstance(value, str):
            authored = {"mode": value}
        elif isinstance(value, Mapping):
            authored = value
        else:
            raise TypeError(
                "image_normalization must be a string, object, or null, "
                f"got {type(value).__name__}"
            )
        raw_mode = str(authored.get("mode", "robust_mad")).strip().lower()
        mode = aliases.get(raw_mode, raw_mode)
        allowed = {
            "robust_mad",
            "robust_percentile_span",
            "minmax",
            "percentile_minmax",
            "clip_divide",
            "clip_zscore",
            "divide",
            "none",
        }
        if mode not in allowed:
            raise ValueError(
                f"Unsupported image_normalization mode {raw_mode!r}; "
                f"allowed: {', '.join(sorted(allowed))}"
            )
        lower = 1.0
        upper = 99.0
        if mode in {
            "robust_mad",
            "robust_percentile_span",
            "percentile_minmax",
        }:
            lower = float(authored.get("percentile_lower", 1.0))
            upper = float(authored.get("percentile_upper", 99.0))
            if not 0.0 <= lower < upper <= 100.0:
                raise ValueError(
                    "image_normalization percentiles must satisfy "
                    f"0 <= lower < upper <= 100, got {lower!r}, {upper!r}"
                )
        clip_min = authored.get("clip_min")
        clip_max = authored.get("clip_max")
        if mode == "clip_divide":
            clip_min = 0.0 if clip_min is None else float(clip_min)
            clip_max = 200.0 if clip_max is None else float(clip_max)
        elif mode == "clip_zscore":
            missing = [
                key
                for key in ("clip_min", "clip_max", "mean", "std")
                if key not in authored
            ]
            if missing:
                raise ValueError(
                    "clip_zscore requires " + ", ".join(missing)
                )
        result = cls(
            mode=mode,
            percentile_lower=lower,
            percentile_upper=upper,
            clip_min=None if clip_min is None else float(clip_min),
            clip_max=None if clip_max is None else float(clip_max),
            divisor=float(authored.get("divisor", 255.0)),
            mean=None if authored.get("mean") is None else float(authored["mean"]),
            std=None if authored.get("std") is None else float(authored["std"]),
        )
        if result.mode in {"divide", "clip_divide"} and result.divisor <= 0.0:
            raise ValueError(f"{result.mode} requires divisor > 0, got {result.divisor!r}")
        if result.mode in {"clip_divide", "clip_zscore"} and not (
            result.clip_min is not None
            and result.clip_max is not None
            and result.clip_min < result.clip_max
        ):
            raise ValueError(
                f"{result.mode} requires clip_min < clip_max, "
                f"got {result.clip_min!r} and {result.clip_max!r}"
            )
        if result.mode == "clip_zscore" and not (
            result.std is not None and result.std > 0.0
        ):
            raise ValueError(f"clip_zscore requires std > 0, got {result.std!r}")
        return result


@dataclass(frozen=True)
class FlatZWindowJitterConfig:
    enabled: bool = False
    window_depth: int | None = None
    max_offset: int = 0
    probability: float = 1.0
    padding: str = "forbidden"

    @classmethod
    def from_mapping(cls, value: Any) -> "FlatZWindowJitterConfig":
        authored = {} if value is None else value
        if not isinstance(authored, Mapping):
            raise TypeError("flat_z_window_jitter must be an object or null")
        result = cls(
            enabled=bool(authored.get("enabled", False)),
            window_depth=(
                None
                if authored.get("window_depth") is None
                else int(authored["window_depth"])
            ),
            max_offset=int(authored.get("max_offset", 0)),
            probability=float(authored.get("probability", 1.0)),
            padding=str(authored.get("padding", "forbidden")).strip().lower(),
        )
        if result.max_offset < 0:
            raise ValueError("flat_z_window_jitter.max_offset must be >= 0")
        if not 0.0 <= result.probability <= 1.0:
            raise ValueError("flat_z_window_jitter.probability must be in [0, 1]")
        if result.padding != "forbidden":
            raise ValueError("flat_z_window_jitter.padding must be 'forbidden'")
        return result


@dataclass(frozen=True)
class PatchFindingConfig:
    overlap: float
    min_labeled_coverage: float
    kind: PatchFindingType = "default"
    unlabeled_min_data_coverage: float = 0.15
    scan_scale: int | None = None
    tile_size: int | None = None
    stride: int | None = None
    filter_empty_tile: bool = False

    @classmethod
    def from_mapping(cls, authored: Mapping[str, Any]) -> "PatchFindingConfig":
        kind = str(authored.get("patch_finding_type", "default")).strip().lower()
        if kind not in {"default", "subtiling"}:
            raise ValueError(
                f"patch_finding_type must be 'default' or 'subtiling', got {kind!r}"
            )
        result = cls(
            kind=kind,
            overlap=float(authored["patch_overlap"]),
            min_labeled_coverage=float(authored["patch_min_labeled_coverage"]),
            unlabeled_min_data_coverage=float(
                authored.get("unlabeled_patch_min_data_coverage", 0.15)
            ),
            scan_scale=(
                None
                if authored.get("patch_finding_scale") is None
                else int(authored["patch_finding_scale"])
            ),
            tile_size=(
                None
                if authored.get("patch_finding_tile_size") is None
                else int(authored["patch_finding_tile_size"])
            ),
            stride=(
                None
                if authored.get("patch_finding_stride") is None
                else int(authored["patch_finding_stride"])
            ),
            filter_empty_tile=bool(
                authored.get("patch_finding_filter_empty_tile", False)
            ),
        )
        if result.kind == "subtiling" and not result.filter_empty_tile:
            raise ValueError(
                "patch_finding_type='subtiling' does not support "
                "patch_finding_filter_empty_tile=false; set "
                "patch_finding_filter_empty_tile=true"
            )
        if result.kind == "subtiling":
            patch_size = authored.get("patch_size")
            patch_y = (
                int(patch_size[1])
                if isinstance(patch_size, (list, tuple)) and len(patch_size) >= 2
                else 0
            )
            default_stride = int(patch_y * result.overlap)
            effective_stride = default_stride if result.stride is None else result.stride
            if effective_stride <= 0:
                raise ValueError("patch_finding_stride must be positive for subtiling")
        return result


@dataclass(frozen=True)
class AugmentationConfig:
    preset: Literal["default", "spatial_only", "spatial_intensity_no_clip", "none"] = "default"
    rotation_axes: tuple[int, ...] | None = None
    disabled: bool = False

    @classmethod
    def from_mapping(cls, authored: Mapping[str, Any]) -> "AugmentationConfig":
        preset = str(authored.get("augmentation_preset", "default")).strip().lower()
        allowed = {"default", "spatial_only", "spatial_intensity_no_clip", "none"}
        if preset not in allowed:
            raise ValueError(
                f"augmentation_preset must be one of {sorted(allowed)!r}, got {preset!r}"
            )
        axes = authored.get("augmentation_rotation_axes")
        return cls(
            preset=preset,
            rotation_axes=None if axes is None else tuple(int(axis) for axis in axes),
            disabled=bool(authored.get("disable_augmentations", False)),
        )


@dataclass(frozen=True)
class Full3DConfig:
    projection_half_thickness: float = 1.0
    label_projection_half_thickness: float | None = None
    background_projection_half_thickness: float | None = None
    support_grid_max_distance: float | None = 64.0

    @classmethod
    def from_mapping(cls, authored: Mapping[str, Any]) -> "Full3DConfig":
        full = authored.get("full_3d") or {}
        pooling = authored.get("normal_pooling") or {}
        if not isinstance(full, Mapping) or not isinstance(pooling, Mapping):
            raise TypeError("full_3d and normal_pooling must be objects or null")
        default = float(full.get("projection_half_thickness", 1.0))
        label = float(full.get("label_projection_half_thickness", default))
        background = float(
            full.get(
                "background_projection_half_thickness",
                full.get("supervision_projection_half_thickness", default),
            )
        )
        if label < 0.0 or background < 0.0:
            raise ValueError("full_3d projection half-thickness values must be >= 0")
        max_distance = pooling.get("support_grid_max_distance", 64.0)
        return cls(
            projection_half_thickness=default,
            label_projection_half_thickness=label,
            background_projection_half_thickness=background,
            support_grid_max_distance=(
                None if max_distance is None else float(max_distance)
            ),
        )


@dataclass(frozen=True)
class DatasetSource:
    segments_path: Path
    volume_scale: int
    volume_path: str | Path | None = None
    segment_names: tuple[str, ...] = ()
    surface_volume_path: str | Path | None = None
    surface_volume_paths: Mapping[str, str | Path] = field(
        default_factory=_FrozenMapping
    )
    sampling_scroll: str = ""
    sampling_physical_segment_keys: Mapping[str, str] = field(
        default_factory=_FrozenMapping
    )
    sampling_representation_keys: Mapping[str, str] = field(
        default_factory=_FrozenMapping
    )

    @classmethod
    def from_mapping(cls, value: Any, *, index: int) -> "DatasetSource":
        if not isinstance(value, Mapping):
            raise TypeError(f"datasets[{index}] must be an object")
        if "segments_path" not in value or "volume_scale" not in value:
            raise ValueError(
                f"datasets[{index}] requires segments_path and volume_scale"
            )
        paths = value.get("surface_volume_paths") or {}
        if not isinstance(paths, Mapping):
            raise TypeError(f"datasets[{index}].surface_volume_paths must be an object")
        return cls(
            segments_path=Path(str(value["segments_path"])),
            volume_scale=int(value["volume_scale"]),
            volume_path=(
                None if value.get("volume_path") in (None, "") else str(value["volume_path"])
            ),
            segment_names=tuple(
                str(item)
                for item in (value.get("segments") or value.get("segment_names") or ())
            ),
            surface_volume_path=(
                None
                if value.get("surface_volume_path") in (None, "")
                else str(value["surface_volume_path"])
            ),
            surface_volume_paths=_FrozenMapping(
                {str(key): str(path) for key, path in paths.items()}
            ),
            sampling_scroll=str(value.get("sampling_scroll", "")).strip(),
            sampling_physical_segment_keys=_string_mapping(
                value.get("sampling_physical_segment_keys"),
                name=f"datasets[{index}].sampling_physical_segment_keys",
            ),
            sampling_representation_keys=_string_mapping(
                value.get("sampling_representation_keys"),
                name=f"datasets[{index}].sampling_representation_keys",
            ),
        )


@dataclass(frozen=True)
class SamplingConfig:
    strategy: SamplingStrategy = "uniform"
    seed: int = 0
    fixed_batch_quotas: Mapping[str, int] = field(
        default_factory=_FrozenMapping
    )

    @classmethod
    def from_mapping(cls, authored: Mapping[str, Any]) -> "SamplingConfig":
        strategy = str(authored.get("sampling_strategy", "uniform")).strip().lower()
        allowed = {
            "uniform",
            "scroll_segment_balanced",
            "fixed_scroll_prior_stratified",
        }
        if strategy not in allowed:
            raise ValueError(
                f"sampling_strategy must be one of {sorted(allowed)!r}, got {strategy!r}"
            )
        fixed = authored.get("fixed_scroll_prior") or {}
        if not isinstance(fixed, Mapping):
            raise TypeError("fixed_scroll_prior must be an object or null")
        quotas = fixed.get("target_batch_counts") or {}
        if not isinstance(quotas, Mapping):
            raise TypeError("fixed_scroll_prior.target_batch_counts must be an object")
        seed = int(authored.get("seed", 0))
        if strategy == "fixed_scroll_prior_stratified" and int(
            fixed.get("seed", -1)
        ) != seed:
            raise ValueError(
                "fixed_scroll_prior.seed must match the training seed: "
                f"{fixed.get('seed')!r} vs {seed!r}"
            )
        return cls(
            strategy=strategy,
            seed=seed,
            fixed_batch_quotas=_FrozenMapping(
                {str(key): int(value) for key, value in quotas.items()}
            ),
        )


@dataclass(frozen=True)
class InkDataConfig:
    """Resolved data contract consumed by datasets and sampling policies."""

    mode: InkMode
    patch_size: tuple[int, int, int]
    datasets: tuple[DatasetSource, ...]
    unlabeled_datasets: tuple[DatasetSource, ...]
    discovery_mode: PatchDiscoveryMode
    patch_finding: PatchFindingConfig
    normalization: NormalizationConfig
    jitter: FlatZWindowJitterConfig
    augmentation: AugmentationConfig
    full_3d: Full3DConfig
    sampling: SamplingConfig
    label_version: str | None
    volume_auth_json: Path | None
    volume_cache_dir: Path | None
    volume_cache_max_gb: float | None
    patch_cache_filename: Path | None
    out_dir: Path
    dataloader_workers: int
    seed: int

    @classmethod
    def from_mapping(cls, authored: Mapping[str, Any]) -> "InkDataConfig":
        if not isinstance(authored, Mapping):
            raise TypeError("ink data config must be an object")
        mode = str(authored.get("mode", "flat")).strip().lower()
        if mode not in {"flat", "full_3d", "full_3d_single_wrap"}:
            raise ValueError(
                "mode must be one of 'flat', 'full_3d', or "
                f"'full_3d_single_wrap', got {mode!r}"
            )
        discovery = str(authored.get("patch_discovery_mode", "labeled")).strip().lower()
        if discovery not in {"labeled", "unlabeled"}:
            raise ValueError(
                f"patch_discovery_mode must be 'labeled' or 'unlabeled', got {discovery!r}"
            )
        patch_size = _tuple3(authored["patch_size"], name="patch_size")
        jitter = FlatZWindowJitterConfig.from_mapping(
            authored.get("flat_z_window_jitter")
        )
        if jitter.window_depth is not None:
            if jitter.window_depth <= 0 or jitter.window_depth > patch_size[0]:
                raise ValueError(
                    "flat_z_window_jitter.window_depth must be in "
                    f"[1, {patch_size[0]}]"
                )
            if (patch_size[0] - jitter.window_depth) % 2:
                raise ValueError(
                    "flat_z_window_jitter requires a symmetric canonical crop"
                )
        datasets_value = authored.get("datasets") or ()
        unlabeled_value = authored.get("unlabeled_datasets") or ()
        if not isinstance(datasets_value, (list, tuple)) or not isinstance(
            unlabeled_value, (list, tuple)
        ):
            raise TypeError("datasets and unlabeled_datasets must be arrays")
        datasets = tuple(
            DatasetSource.from_mapping(value, index=index)
            for index, value in enumerate(datasets_value)
        )
        unlabeled = tuple(
            DatasetSource.from_mapping(value, index=index)
            for index, value in enumerate(unlabeled_value)
        )
        selected = unlabeled if discovery == "unlabeled" else datasets
        if not selected:
            raise ValueError(
                "unlabeled_datasets is empty"
                if discovery == "unlabeled"
                else "datasets is empty"
            )
        return cls(
            mode=mode,
            patch_size=patch_size,
            datasets=datasets,
            unlabeled_datasets=unlabeled,
            discovery_mode=discovery,
            patch_finding=PatchFindingConfig.from_mapping(authored),
            normalization=NormalizationConfig.from_value(
                authored.get("image_normalization")
            ),
            jitter=jitter,
            augmentation=AugmentationConfig.from_mapping(authored),
            full_3d=Full3DConfig.from_mapping(authored),
            sampling=SamplingConfig.from_mapping(authored),
            label_version=(
                None
                if authored.get("label_version") in (None, "")
                else str(authored["label_version"]).strip()
            ),
            volume_auth_json=(
                None
                if authored.get("volume_auth_json") in (None, "")
                else Path(str(authored["volume_auth_json"]))
            ),
            volume_cache_dir=(
                None
                if authored.get("volume_cache_dir") in (None, "")
                else Path(str(authored["volume_cache_dir"]))
            ),
            volume_cache_max_gb=(
                None
                if authored.get("volume_cache_max_gb") is None
                else float(authored["volume_cache_max_gb"])
            ),
            patch_cache_filename=(
                None
                if authored.get("patch_cache_filename") in (None, "")
                else Path(str(authored["patch_cache_filename"]))
            ),
            out_dir=Path(str(authored.get("out_dir", "."))),
            dataloader_workers=int(authored.get("dataloader_workers", 8)),
            seed=int(authored.get("seed", 0)),
        )

    @property
    def active_datasets(self) -> tuple[DatasetSource, ...]:
        return (
            self.unlabeled_datasets
            if self.discovery_mode == "unlabeled"
            else self.datasets
        )


@dataclass(frozen=True)
class TargetConfig:
    """One output head and its checkpointed activation/projection contract."""

    name: Literal["ink"]
    out_channels: int
    activation: Literal["none"]
    z_projection_mode: str
    _settings: Mapping[str, Any] = field(repr=False)

    @classmethod
    def from_mapping(
        cls,
        name: str,
        authored: Any,
        *,
        model_config: Mapping[str, Any],
    ) -> "TargetConfig":
        if name != "ink":
            raise ValueError(f"Unsupported target {name!r}; expected 'ink'")
        if not isinstance(authored, Mapping):
            raise TypeError("targets.ink must be an object")
        out_channels = int(authored["out_channels"])
        if out_channels != 1:
            raise ValueError(
                f"targets.ink.out_channels must be 1, got {out_channels!r}"
            )
        activation = str(authored["activation"]).strip().lower()
        if activation != "none":
            raise ValueError(
                "targets.ink.activation must be 'none', "
                f"got {activation!r}"
            )

        projection = authored.get("z_projection")
        if projection is not None and not isinstance(projection, Mapping):
            raise TypeError("targets.ink.z_projection must be an object")
        if isinstance(projection, Mapping) and "mode" in projection:
            projection_mode = str(projection["mode"]).strip().lower()
        elif "z_projection_mode" in authored:
            projection_mode = str(authored["z_projection_mode"]).strip().lower()
        else:
            projection_mode = str(
                model_config.get("z_projection_mode", "none")
            ).strip().lower()
        if projection_mode in {"", "off", "false", "0"}:
            projection_mode = "none"
        allowed = {"none", "max", "mean", "logsumexp", "learned_mlp"}
        if projection_mode not in allowed:
            raise ValueError(
                "Unsupported targets.ink z_projection mode "
                f"{projection_mode!r}; allowed: {', '.join(sorted(allowed))}"
            )
        return cls(
            name="ink",
            out_channels=out_channels,
            activation="none",
            z_projection_mode=projection_mode,
            _settings=_FrozenMapping(
                {
                    str(key): _freeze_json(value)
                    for key, value in authored.items()
                }
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the independent target mapping used by the model builder."""

        return _thaw_json(self._settings)


@dataclass(frozen=True)
class InkModelConfig:
    """Resolved model family and the values consumed by NetworkFromConfig."""

    model_type: str
    crop_size: tuple[int, int, int]
    batch_size: int
    in_channels: int
    model_name: str
    autoconfigure: bool
    enable_deep_supervision: bool
    model_config: Mapping[str, Any]
    stem_channels: int
    input_pad_depth_to: int | None

    def model_settings_mapping(self) -> dict[str, Any]:
        """Return independent model settings for NetworkFromConfig."""

        return _thaw_json(self.model_config)

    @classmethod
    def from_mapping(cls, authored: Mapping[str, Any]) -> "InkModelConfig":
        model_type = str(authored["model_type"]).strip().lower()
        allowed = {
            "vesuvius_unet",
            "unet",
            "vesuvius_unet_2p5d",
            "unet_2p5d",
            "vesuvius_unet_3d_stem_2d",
            "unet_3d_stem_2d",
        }
        if model_type == "resnet3d" or model_type.startswith("resnet3d-"):
            raise ValueError(f"Unsupported model_type {model_type!r}: resnet3d")
        if model_type not in allowed:
            raise ValueError(
                f"Unsupported model_type {model_type!r}; "
                f"allowed: {', '.join(sorted(allowed))}"
            )
        raw_model_config = authored.get("model_config") or {}
        if not isinstance(raw_model_config, Mapping):
            raise TypeError("model_config must be an object or null")
        architecture_type = str(
            raw_model_config.get("architecture_type", "unet")
        ).strip().lower()
        if architecture_type.startswith("mednext"):
            raise ValueError(
                "model_config.architecture_type selects MedNeXt, which is not "
                "supported by the ink checkpoint contract"
            )
        if raw_model_config.get("guide_backbone"):
            raise ValueError(
                "model_config.guide_backbone selects guided model construction, "
                "which is not supported by the ink checkpoint contract"
            )
        upsample_mode = str(
            raw_model_config.get("upsample_mode", "transpconv")
        ).strip().lower()
        if upsample_mode in {"pixelshuffle", "trilinear"}:
            raise ValueError(
                f"model_config.upsample_mode={upsample_mode!r} selects a decoder "
                "path that is not supported by the ink checkpoint contract"
            )
        if raw_model_config.get("target_z_projection"):
            raise ValueError(
                "model_config.target_z_projection selects projection behavior "
                "that is not supported by the ink checkpoint contract"
            )
        if (
            raw_model_config.get("pretrained_backbone")
            and raw_model_config.get("pretrained_backbone_config_path")
        ):
            raise ValueError(
                "model_config.pretrained_backbone_config_path selects backbone "
                "configuration behavior that is not supported by the ink "
                "checkpoint contract"
            )
        model_config = _FrozenMapping(
            {str(key): _freeze_json(value) for key, value in raw_model_config.items()}
        )
        crop_value = authored.get("crop_size", authored["patch_size"])
        crop_size = _tuple3(crop_value, name="crop_size")
        input_pad_depth_to = raw_model_config.get("input_pad_depth_to")
        if input_pad_depth_to is not None:
            input_pad_depth_to = int(input_pad_depth_to)
            if input_pad_depth_to <= 0:
                raise ValueError(
                    "model_config.input_pad_depth_to must be positive, "
                    f"got {input_pad_depth_to!r}"
                )
        stem_channels = int(raw_model_config.get("stem_channels", 16))
        if stem_channels <= 0:
            raise ValueError(
                f"model_config.stem_channels must be positive, got {stem_channels!r}"
            )
        return cls(
            model_type=model_type,
            crop_size=crop_size,
            batch_size=int(authored.get("batch_size", 1)),
            in_channels=int(authored["in_channels"]),
            model_name=str(authored.get("model_name", "ink_det")),
            autoconfigure=bool(
                raw_model_config.get(
                    "autoconfigure", authored.get("autoconfigure", True)
                )
            ),
            enable_deep_supervision=bool(
                authored.get("enable_deep_supervision", False)
            ),
            model_config=model_config,
            stem_channels=stem_channels,
            input_pad_depth_to=input_pad_depth_to,
        )


@dataclass(frozen=True)
class LossTermConfig:
    """One weighted loss term in authored evaluation order."""

    name: Literal["LabelSmoothedDCAndBCELoss"]
    metric_name: str
    weight: float
    weight_dice: float
    weight_ce: float
    dice_label_smoothing: float
    bce_label_smoothing: float
    bce_kwargs: Mapping[str, Any]


@dataclass(frozen=True)
class LossConfig:
    """The ordered active terms used by the ink segmentation objective."""

    terms: tuple[LossTermConfig, ...]

    @classmethod
    def from_mapping(cls, authored: Mapping[str, Any]) -> "LossConfig":
        raw_loss = authored.get("loss") or {}
        if not isinstance(raw_loss, Mapping):
            raise TypeError("loss must be an object or null")
        raw_terms = raw_loss.get("terms")
        if raw_terms is None:
            term_values: list[Mapping[str, Any]] = [
                {
                    "name": "LabelSmoothedDCAndBCELoss",
                    "metric_name": "base",
                    "weight": 1.0,
                    "weight_dice": float(raw_loss.get("dice_weight", 0.25)),
                    "weight_ce": float(raw_loss.get("ce_weight", 1.0)),
                    "dice_label_smoothing": float(
                        raw_loss.get(
                            "dice_label_smoothing",
                            authored.get("dice_label_smoothing", 0.0),
                        )
                    ),
                    "bce_label_smoothing": float(
                        raw_loss.get(
                            "bce_label_smoothing",
                            authored.get("bce_label_smoothing", 0.0),
                        )
                    ),
                }
            ]
        else:
            if not isinstance(raw_terms, list) or not raw_terms:
                raise ValueError("loss.terms must be a non-empty list when provided")
            term_values = []
            for index, value in enumerate(raw_terms):
                if not isinstance(value, Mapping):
                    raise TypeError(f"loss.terms[{index}] must be an object")
                term_values.append(value)

        terms = []
        for index, value in enumerate(term_values):
            if "name" not in value or not value["name"]:
                raise ValueError(
                    f"loss term at index {index} is missing required key 'name'"
                )
            name = str(value["name"])
            if name == "BettiMatchingLoss":
                raise ValueError(
                    "Unsupported loss term 'BettiMatchingLoss': Betti matching"
                )
            if name != "LabelSmoothedDCAndBCELoss":
                raise ValueError(
                    f"Unsupported loss term {name!r}; supported: "
                    "LabelSmoothedDCAndBCELoss"
                )
            weight = float(value.get("weight", 1.0))
            if weight == 0.0:
                continue
            bce_kwargs = value.get("bce_kwargs") or {}
            if not isinstance(bce_kwargs, Mapping):
                raise TypeError(f"loss.terms[{index}].bce_kwargs must be an object")
            terms.append(
                LossTermConfig(
                    name="LabelSmoothedDCAndBCELoss",
                    metric_name=str(value.get("metric_name") or name),
                    weight=weight,
                    weight_dice=float(
                        value.get("weight_dice", value.get("dice_weight", 1.0))
                    ),
                    weight_ce=float(
                        value.get("weight_ce", value.get("ce_weight", 1.0))
                    ),
                    dice_label_smoothing=float(
                        value.get(
                            "dice_label_smoothing",
                            authored.get("dice_label_smoothing", 0.0),
                        )
                    ),
                    bce_label_smoothing=float(
                        value.get(
                            "bce_label_smoothing",
                            authored.get("bce_label_smoothing", 0.0),
                        )
                    ),
                    bce_kwargs=_FrozenMapping(
                        {
                            str(key): _freeze_json(item)
                            for key, item in bce_kwargs.items()
                        }
                    ),
                )
            )
        if not terms:
            raise ValueError(
                "All configured loss terms have zero weight; "
                "at least one active term is required"
            )
        return cls(terms=tuple(terms))


@dataclass(frozen=True)
class CheckpointConfig:
    """Training checkpoint selection authored by a JSON configuration."""

    path: Path | None
    weights_only: bool

    @classmethod
    def from_mapping(cls, authored: Mapping[str, Any]) -> "CheckpointConfig":
        value = authored.get("checkpoint")
        return cls(
            path=None if value in (None, "") else Path(str(value)),
            weights_only=bool(authored.get("weights_only", False)),
        )


def _canonical_model_mapping(authored: Mapping[str, Any]) -> dict[str, Any]:
    canonical = deepcopy(dict(authored))
    model_type = str(canonical["model_type"]).strip().lower()
    if model_type != "dinov2":
        return canonical
    raw_model_config = canonical.get("model_config")
    if raw_model_config is None:
        model_config: dict[str, Any] = {}
        canonical["model_config"] = model_config
    elif isinstance(raw_model_config, Mapping):
        model_config = dict(raw_model_config)
        canonical["model_config"] = model_config
    else:
        raise TypeError("model_config must be an object or null")
    for key in ("pretrained_backbone", "pretrained_decoder_type"):
        if key in canonical:
            model_config.setdefault(key, canonical[key])
    if not model_config.get("pretrained_backbone"):
        raise ValueError(
            "model_type='dinov2' requires model_config.pretrained_backbone "
            "or a top-level pretrained_backbone entry"
        )
    canonical["model_type"] = "vesuvius_unet"
    return canonical


@dataclass(frozen=True)
class InkConfig:
    """Canonical checkpoint mapping with typed data, model, target, and loss views."""

    data: InkDataConfig
    model: InkModelConfig
    targets: Mapping[str, TargetConfig]
    loss: LossConfig
    checkpoint: CheckpointConfig
    _canonical: Mapping[str, Any] = field(repr=False)

    @classmethod
    def from_mapping(cls, authored: Mapping[str, Any]) -> "InkConfig":
        if not isinstance(authored, Mapping):
            raise TypeError("ink config must be an object")
        canonical = _canonical_model_mapping(authored)
        data = InkDataConfig.from_mapping(canonical)
        model = InkModelConfig.from_mapping(canonical)
        raw_targets = canonical.get("targets")
        if not isinstance(raw_targets, Mapping) or not raw_targets:
            raise ValueError("targets must be a non-empty object containing 'ink'")
        unknown_targets = [str(name) for name in raw_targets if str(name) != "ink"]
        if unknown_targets:
            raise ValueError(
                f"Unsupported target {unknown_targets[0]!r}; expected 'ink'"
            )
        if "ink" not in raw_targets:
            raise ValueError("targets must contain 'ink'")
        targets = _FrozenMapping(
            {
                "ink": TargetConfig.from_mapping(
                    "ink",
                    raw_targets["ink"],
                    model_config=canonical.get("model_config") or {},
                )
            }
        )
        return cls(
            data=data,
            model=model,
            targets=targets,
            loss=LossConfig.from_mapping(canonical),
            checkpoint=CheckpointConfig.from_mapping(canonical),
            _canonical=_FrozenMapping(
                {
                    str(key): _freeze_json(value)
                    for key, value in canonical.items()
                }
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return an independent JSON-shaped copy in authored key order."""

        return _thaw_json(self._canonical)
