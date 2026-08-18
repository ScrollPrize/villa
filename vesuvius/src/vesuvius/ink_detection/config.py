"""Typed ink data, model, and training configuration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Literal, Mapping


InkMode = Literal["flat", "full_3d", "full_3d_single_wrap"]
PatchDiscoveryMode = Literal["labeled", "unlabeled"]
PatchFindingType = Literal["default", "subtiling"]
SamplingStrategy = Literal[
    "uniform", "scroll_segment_balanced", "fixed_scroll_prior_stratified"
]


@dataclass(frozen=True)
class DinoGuidedLabelConfig:
    """Frozen inputs for the v1/v2 DINO-guided label recipe."""

    kind: Literal["dino_guided"]
    unet_ckpt: Path
    dino_ckpt: Path
    ref_embedding: Path
    dino_stride: int
    dino_minibatch: int
    dino_blend_sigma: float
    threshold: float
    input_mask_threshold: float | None


@dataclass(frozen=True)
class SelfDistillLabelConfig:
    """Frozen inputs for the v3 teacher-student label recipe."""

    kind: Literal["self_distill"]
    primary_ckpt: Path
    ensemble_ckpt: Path
    primary_threshold: float
    ensemble_threshold: float
    mean_hi: float
    std_lo: float
    tta: bool
    input_mask_threshold: float


DynamicLabelConfig = DinoGuidedLabelConfig | SelfDistillLabelConfig


@dataclass(frozen=True)
class ExtraPatchesConfig:
    """Optional v3 coordinate-patch sampling controls."""

    enabled: bool = False
    coords_xyz: tuple[tuple[int, int, int], ...] = ()
    jitter: int = 1024
    fraction: float = 0.25


def _required_path(authored: Mapping[str, Any], key: str, *, parent: str) -> Path:
    value = authored.get(key)
    if value in (None, ""):
        raise ValueError(f"{parent}.{key} is required")
    return Path(str(value))


def _dynamic_label_config(
    authored: Mapping[str, Any], *, mode: str, patch_size: tuple[int, int, int]
) -> DynamicLabelConfig | None:
    value = authored.get("dynamic_label") or {}
    if not isinstance(value, Mapping):
        raise TypeError("dynamic_label must be an object or null")
    if not bool(value.get("enabled", False)):
        return None
    if mode != "full_3d":
        raise ValueError("dynamic_label is supported only in mode='full_3d'")
    if patch_size != (256, 256, 256):
        raise ValueError(
            "dynamic_label requires patch_size [256, 256, 256], got "
            f"{list(patch_size)!r}"
        )
    kind = str(value.get("kind", "dino_guided")).strip().lower()
    if kind == "dino_guided":
        stride = int(value.get("dino_stride", 64))
        minibatch = int(value.get("dino_minibatch", 8))
        sigma = float(value.get("dino_blend_sigma", 4.0))
        threshold = float(value.get("threshold", 0.5))
        mask_threshold = value.get("input_mask_threshold")
        if stride <= 0 or minibatch <= 0 or sigma <= 0.0:
            raise ValueError(
                "dynamic_label DINO stride, minibatch, and blend sigma must be positive"
            )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("dynamic_label.threshold must be in [0,1]")
        return DinoGuidedLabelConfig(
            kind="dino_guided",
            unet_ckpt=_required_path(value, "unet_ckpt", parent="dynamic_label"),
            dino_ckpt=_required_path(value, "dino_ckpt", parent="dynamic_label"),
            ref_embedding=_required_path(
                value, "ref_embedding", parent="dynamic_label"
            ),
            dino_stride=stride,
            dino_minibatch=minibatch,
            dino_blend_sigma=sigma,
            threshold=threshold,
            input_mask_threshold=(
                None if mask_threshold is None else float(mask_threshold)
            ),
        )
    if kind == "self_distill":
        primary_threshold = float(value["primary_threshold"])
        ensemble_threshold = float(value["ensemble_threshold"])
        if not 0.0 <= primary_threshold <= 1.0:
            raise ValueError("dynamic_label.primary_threshold must be in [0,1]")
        if not 0.0 <= ensemble_threshold <= 1.0:
            raise ValueError("dynamic_label.ensemble_threshold must be in [0,1]")
        return SelfDistillLabelConfig(
            kind="self_distill",
            primary_ckpt=_required_path(
                value, "primary_ckpt", parent="dynamic_label"
            ),
            ensemble_ckpt=_required_path(
                value, "ensemble_ckpt", parent="dynamic_label"
            ),
            primary_threshold=primary_threshold,
            ensemble_threshold=ensemble_threshold,
            mean_hi=float(value["mean_hi"]),
            std_lo=float(value["std_lo"]),
            tta=bool(value.get("tta", True)),
            input_mask_threshold=float(value.get("input_mask_threshold", 50.0)),
        )
    raise ValueError(
        "dynamic_label.kind must be 'dino_guided' or 'self_distill', "
        f"got {kind!r}"
    )


def _extra_patches_config(
    authored: Mapping[str, Any], *, dynamic_label: DynamicLabelConfig | None
) -> ExtraPatchesConfig:
    value = authored.get("extra_patches") or {}
    if not isinstance(value, Mapping):
        raise TypeError("extra_patches must be an object or null")
    if not bool(value.get("enabled", False)):
        return ExtraPatchesConfig()
    if not isinstance(dynamic_label, SelfDistillLabelConfig):
        raise ValueError("extra_patches requires dynamic_label.kind='self_distill'")
    coords_value = value.get("coords_xyz") or ()
    if not isinstance(coords_value, (list, tuple)):
        raise TypeError("extra_patches.coords_xyz must be an array")
    coords: list[tuple[int, int, int]] = []
    for index, coord in enumerate(coords_value):
        if not isinstance(coord, (list, tuple)) or len(coord) != 3:
            raise ValueError(
                f"extra_patches.coords_xyz[{index}] must be one XYZ triple"
            )
        coords.append(tuple(int(item) for item in coord))
    if not coords:
        raise ValueError("extra_patches.coords_xyz cannot be empty")
    jitter = int(value.get("jitter", 1024))
    fraction = float(value.get("fraction", 0.25))
    if jitter < 0:
        raise ValueError("extra_patches.jitter must be nonnegative")
    if not 0.0 < fraction < 1.0:
        raise ValueError("extra_patches.fraction must be in (0,1)")
    return ExtraPatchesConfig(
        enabled=True,
        coords_xyz=tuple(coords),
        jitter=jitter,
        fraction=fraction,
    )


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
    """One of the seven image normalization modes and its constants."""

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
            "minmax",
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
        mean = authored.get("mean")
        std = authored.get("std")
        if mode == "clip_zscore":
            missing = [
                key
                for key in ("clip_min", "clip_max", "mean", "std")
                if key not in authored
            ]
            if missing:
                raise ValueError(
                    "clip_zscore requires " + ", ".join(missing)
                )
            clip_min = float(clip_min)
            clip_max = float(clip_max)
            mean = float(mean)
            std = float(std)
            if not clip_min < clip_max:
                raise ValueError(
                    "clip_zscore requires clip_min < clip_max, "
                    f"got {clip_min!r} and {clip_max!r}"
                )
            if std <= 0.0:
                raise ValueError(f"clip_zscore requires std > 0, got {std!r}")
        result = cls(
            mode=mode,
            percentile_lower=lower,
            percentile_upper=upper,
            clip_min=None if clip_min is None else float(clip_min),
            clip_max=None if clip_max is None else float(clip_max),
            divisor=float(authored.get("divisor", 255.0)),
            mean=None if mean is None else float(mean),
            std=None if std is None else float(std),
        )
        if result.mode == "divide" and result.divisor <= 0.0:
            raise ValueError(f"{result.mode} requires divisor > 0, got {result.divisor!r}")
        return result


@dataclass(frozen=True)
class FlatZWindowJitterConfig:
    enabled: bool = False
    window_depth: int | None = None
    max_offset: int = 0
    probability: float = 1.0

    @classmethod
    def from_mapping(cls, value: Any) -> "FlatZWindowJitterConfig":
        authored = {} if value is None else value
        if not isinstance(authored, Mapping):
            raise TypeError("flat_z_window_jitter must be an object or null")
        padding = str(authored.get("padding", "forbidden")).strip().lower()
        result = cls(
            enabled=bool(authored.get("enabled", False)),
            window_depth=(
                None
                if authored.get("window_depth") is None
                else int(authored["window_depth"])
            ),
            max_offset=int(authored.get("max_offset", 0)),
            probability=float(authored.get("probability", 1.0)),
        )
        if result.max_offset < 0:
            raise ValueError("flat_z_window_jitter.max_offset must be >= 0")
        if not 0.0 <= result.probability <= 1.0:
            raise ValueError("flat_z_window_jitter.probability must be in [0, 1]")
        if padding != "forbidden":
            raise ValueError("flat_z_window_jitter.padding must be 'forbidden'")
        return result


@dataclass(frozen=True)
class PatchFindingConfig:
    overlap: float
    min_labeled_coverage: float
    kind: PatchFindingType = "default"
    scan_scale: int | None = None
    tile_size: int | None = None
    stride: int | None = None
    filter_empty_tile: bool = False

    @classmethod
    def from_mapping(cls, authored: Mapping[str, Any]) -> "PatchFindingConfig":
        if "unlabeled_patch_min_data_coverage" in authored:
            raise ValueError(
                "unlabeled discovery threshold is fixed at 0.25; "
                "unlabeled_patch_min_data_coverage is not honored"
            )
        kind = str(authored.get("patch_finding_type", "default")).strip().lower()
        if kind not in {"default", "subtiling"}:
            raise ValueError(
                f"patch_finding_type must be 'default' or 'subtiling', got {kind!r}"
            )
        result = cls(
            kind=kind,
            overlap=float(authored["patch_overlap"]),
            min_labeled_coverage=float(authored["patch_min_labeled_coverage"]),
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
            # A nonpositive stride would make subtiling discovery stop advancing.
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
    label_projection_half_thickness: float | None = None
    background_projection_half_thickness: float | None = None
    support_grid_max_distance: float | None = 64.0
    label_dilation_distance: float = 0.0
    supervision_dilation_distance: float = 0.0

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
            label_projection_half_thickness=label,
            background_projection_half_thickness=background,
            support_grid_max_distance=(
                None if max_distance is None else float(max_distance)
            ),
            label_dilation_distance=float(
                full.get("label_dilation_distance", 0.0)
            ),
            supervision_dilation_distance=float(
                full.get("supervision_dilation_distance", 0.0)
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
            # TrainingConfig requires out_dir instead of defaulting it.
            out_dir=Path(str(authored.get("out_dir", "."))),
            # TrainingConfig defaults dataloader_workers to 0.
            dataloader_workers=int(authored.get("dataloader_workers", 8)),
            # TrainingConfig requires seed instead of defaulting it.
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
    """One validated output head's frozen authored builder mapping."""

    _settings: Mapping[str, Any] = field(repr=False)

    @classmethod
    def from_mapping(
        cls,
        authored: Any,
        *,
        model_config: Mapping[str, Any],
    ) -> "TargetConfig":
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
        authored_modes = []
        if isinstance(projection, Mapping) and "mode" in projection:
            authored_modes.append(str(projection["mode"]).strip().lower())
        if "z_projection_mode" in authored:
            authored_modes.append(
                str(authored["z_projection_mode"]).strip().lower()
            )
        if "z_projection_mode" in model_config:
            authored_modes.append(
                str(model_config["z_projection_mode"]).strip().lower()
            )
        allowed = {
            "",
            "0",
            "false",
            "learned_mlp",
            "logsumexp",
            "max",
            "mean",
            "none",
            "off",
        }
        invalid_mode = next(
            (mode for mode in authored_modes if mode not in allowed), None
        )
        if invalid_mode is not None:
            raise ValueError(
                "Unsupported targets.ink z_projection mode "
                f"{invalid_mode!r}; allowed: {', '.join(sorted(allowed))}"
            )
        return cls(
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
    pretrained_backbone: str | None
    freeze_encoder: bool
    spacing: tuple[float, float, float]
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
        raw_pretrained_backbone = raw_model_config.get("pretrained_backbone")
        pretrained_backbone = (
            None if not raw_pretrained_backbone else str(raw_pretrained_backbone)
        )
        freeze_encoder = bool(
            pretrained_backbone
            and (
                authored.get("freeze_encoder", False)
                or raw_model_config.get("freeze_encoder", False)
            )
        )
        raw_spacing = raw_model_config.get("spacing", (1.0, 1.0, 1.0))
        if not isinstance(raw_spacing, (list, tuple)) or len(raw_spacing) != 3:
            raise ValueError(
                "model_config.spacing must contain exactly three axes, "
                f"got {raw_spacing!r}"
            )
        spacing = (
            float(raw_spacing[0]),
            float(raw_spacing[1]),
            float(raw_spacing[2]),
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
            pretrained_backbone=pretrained_backbone,
            freeze_encoder=freeze_encoder,
            spacing=spacing,
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


def _normalize_model_config(canonical: dict[str, Any]) -> dict[str, Any]:
    raw_model_config = canonical.get("model_config")
    if raw_model_config is None:
        model_config: dict[str, Any] = {}
    elif isinstance(raw_model_config, Mapping):
        model_config = dict(raw_model_config)
    else:
        raise TypeError("model_config must be an object or null")
    canonical["model_config"] = model_config
    return model_config


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
        return cls._from_canonical_mapping(deepcopy(dict(authored)))

    @classmethod
    def _from_canonical_mapping(
        cls, canonical: Mapping[str, Any]
    ) -> "InkConfig":
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
        targets = _FrozenMapping(
            {
                "ink": TargetConfig.from_mapping(
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


@dataclass(frozen=True)
class EmaConfig:
    """Resolved EMA settings that are persisted as one complete mapping."""

    enabled: bool
    decay: float
    start_step: int
    update_every_steps: int
    validate: bool
    save_in_checkpoint: bool

    @classmethod
    def from_mapping(cls, authored: Mapping[str, Any]) -> "EmaConfig":
        value = authored.get("ema") or {}
        if not isinstance(value, Mapping):
            raise TypeError("ema must be an object or null")
        enabled = bool(value.get("enabled", False))
        return cls(
            enabled=enabled,
            decay=float(value.get("decay", 0.999)),
            start_step=int(value.get("start_step", 0)),
            update_every_steps=int(value.get("update_every_steps", 1)),
            validate=bool(value.get("validate", enabled)),
            save_in_checkpoint=bool(value.get("save_in_checkpoint", enabled)),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "decay": self.decay,
            "start_step": self.start_step,
            "update_every_steps": self.update_every_steps,
            "validate": self.validate,
            "save_in_checkpoint": self.save_in_checkpoint,
        }


@dataclass(frozen=True)
class BenchmarkConfig:
    """Local inline-benchmark controls; absent defaults are never checkpointed."""

    enabled: bool
    warmup_steps: int
    output_path: Path | None

    @classmethod
    def from_mapping(
        cls, authored: Mapping[str, Any], *, num_iterations: int
    ) -> "BenchmarkConfig":
        value = authored.get("benchmark") or {}
        if not isinstance(value, Mapping):
            raise TypeError("benchmark must be an object or null")
        enabled = bool(value.get("enabled", False))
        warmup_steps = int(value.get("warmup_steps", 0))
        if enabled and not 0 <= warmup_steps < num_iterations:
            raise ValueError(
                "benchmark warmup_steps must be in [0, num_iterations)"
            )
        output_path = value.get("output_path")
        return cls(
            enabled=enabled,
            warmup_steps=warmup_steps,
            output_path=(
                None if output_path in (None, "") else Path(str(output_path))
            ),
        )


@dataclass(frozen=True)
class OptimizerConfig:
    """Local optimizer constructor view delegated to vesuvius's factory."""

    name: str
    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    momentum: float
    nesterov: bool
    encoder_lr_mult: float

    @classmethod
    def from_mapping(cls, authored: Mapping[str, Any]) -> "OptimizerConfig":
        raw_betas = authored.get("optimizer_betas", (0.9, 0.999))
        if not isinstance(raw_betas, (list, tuple)) or len(raw_betas) != 2:
            raise ValueError("optimizer_betas must contain exactly two values")
        return cls(
            name=str(authored.get("optimizer", "sgd")),
            learning_rate=float(authored.get("learning_rate", 0.01)),
            weight_decay=float(authored.get("weight_decay", 3e-5)),
            betas=(float(raw_betas[0]), float(raw_betas[1])),
            momentum=float(authored.get("optimizer_momentum", 0.99)),
            nesterov=bool(authored.get("optimizer_nesterov", True)),
            encoder_lr_mult=float(authored.get("encoder_lr_mult", 1.0)),
        )


@dataclass(frozen=True)
class SchedulerConfig:
    """Resolved scheduler settings with stable defaults and key locations."""

    name: Literal[
        "diffusers_cosine_warmup", "cosine_annealing", "one_cycle"
    ]
    t_max: int
    eta_min: float
    max_lr: float | None
    total_steps: int
    pct_start: float
    final_div_factor: float
    warmup_steps: int

    @classmethod
    def from_mapping(
        cls, authored: Mapping[str, Any], *, max_steps: int
    ) -> "SchedulerConfig":
        value = authored.get("scheduler") or {}
        if not isinstance(value, Mapping):
            raise TypeError("scheduler must be an object or null")
        name = str(value.get("name", "diffusers_cosine_warmup")).strip().lower()
        allowed = {
            "diffusers_cosine_warmup",
            "cosine_annealing",
            "one_cycle",
        }
        if name not in allowed:
            raise ValueError(
                f"Unsupported scheduler.name={name!r}; expected "
                "'diffusers_cosine_warmup', 'cosine_annealing', or 'one_cycle'"
            )
        max_lr = None
        t_max = max_steps
        eta_min = 0.0
        total_steps = max_steps
        pct_start = 0.3
        final_div_factor = 1e4
        warmup_steps = 1000
        if name == "cosine_annealing":
            t_max = int(value.get("t_max", max_steps))
            eta_min = float(value.get("eta_min", 0.0))
        elif name == "one_cycle":
            max_lr = float(value["max_lr"])
            total_steps = int(value.get("total_steps", max_steps))
            pct_start = float(value.get("pct_start", 0.3))
            final_div_factor = float(value.get("final_div_factor", 1e4))
        else:
            warmup_steps = int(authored.get("warmup_steps", 1000))
        return cls(
            name=name,
            t_max=t_max,
            eta_min=eta_min,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            final_div_factor=final_div_factor,
            warmup_steps=warmup_steps,
        )


def resolve_training_mapping(authored: Mapping[str, Any]) -> dict[str, Any]:
    """Apply ordered training mutations before creating frozen views."""

    if not isinstance(authored, Mapping):
        raise TypeError("ink training config must be an object")
    raw_targets = authored.get("targets")
    if (
        not isinstance(raw_targets, Mapping)
        or not raw_targets
        or "ink" not in raw_targets
    ):
        raise ValueError("targets must be a non-empty object containing 'ink'")
    for name, target in raw_targets.items():
        if not isinstance(target, Mapping):
            raise TypeError(f"targets.{name} must be an object")
    canonical = deepcopy(dict(authored))

    ema = EmaConfig.from_mapping(canonical)
    canonical["ema"] = ema.to_mapping()

    mode = str(canonical.get("mode", "flat")).strip().lower()
    native_mode = mode in {"full_3d", "full_3d_single_wrap"}
    if native_mode:
        model_config = _normalize_model_config(canonical)
        model_config["z_projection_mode"] = "none"
        targets = {
            str(name): dict(value)
            for name, value in canonical["targets"].items()
        }
        canonical["targets"] = targets
        for target in targets.values():
            target["z_projection_mode"] = "none"
            if isinstance(target.get("z_projection"), Mapping):
                projection = dict(target["z_projection"])
                projection["mode"] = "none"
                target["z_projection"] = projection
        canonical["in_channels"] = (
            2 if mode == "full_3d_single_wrap" else 1
        )

    requested_stitch_factor = int(canonical.get("stitch_factor", 1))
    use_stitched_forward = bool(
        canonical.get(
            "use_stitched_forward", requested_stitch_factor > 1
        )
    )
    if native_mode:
        use_stitched_forward = False
    stitch_factor = 1
    if use_stitched_forward:
        if requested_stitch_factor <= 0:
            raise ValueError("stitch_factor must be positive")
        stitch_factor = requested_stitch_factor
    model_crop_size = list(_tuple3(canonical["patch_size"], name="patch_size"))
    canonical["crop_size"] = model_crop_size
    canonical["patch_size"] = list(model_crop_size)
    canonical["stitch_factor"] = stitch_factor
    canonical["use_stitched_forward"] = use_stitched_forward
    canonical["stitched_gradient_checkpointing"] = bool(
        canonical.get("stitched_gradient_checkpointing", True)
    )
    # The reference overwrites these silently; contradicting authored values
    # fail fast here so TargetConfig's guard is not dead on the training path.
    ink_target = canonical["targets"]["ink"]
    authored_channels = ink_target.get("out_channels")
    if authored_channels is not None and int(authored_channels) != 1:
        raise ValueError(
            f"targets.ink.out_channels must be 1, got {authored_channels!r}"
        )
    authored_activation = ink_target.get("activation")
    if (
        authored_activation is not None
        and str(authored_activation).strip().lower() != "none"
    ):
        raise ValueError(
            f"targets.ink.activation must be 'none', got {authored_activation!r}"
        )
    ink_target["out_channels"] = 1
    ink_target["activation"] = "none"
    return canonical


@dataclass(frozen=True)
class TrainingConfig:
    """Frozen local training views derived from one resolved InkConfig."""

    ink: InkConfig
    num_iterations: int
    out_dir: Path
    seed: int
    wandb_run_id: str | None
    wandb_resume: bool
    batch_size: int
    model_crop_size: tuple[int, int, int]
    loader_patch_size: tuple[int, int, int]
    use_stitched_forward: bool
    stitched_gradient_checkpointing: bool
    ema: EmaConfig
    benchmark: BenchmarkConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    grad_acc_steps: int
    grad_clip: float | None
    max_steps: int
    val_every: int
    save_every: int
    log_every: int
    val_preview_batches: int
    val_steps: int
    best_checkpoint_metric: Literal[
        "val_loss", "val_balanced_accuracy"
    ] | None
    mixed_precision: str
    ddp_find_unused_parameters: bool
    ddp_broadcast_buffers: bool
    dataloader_workers: int
    pin_memory: bool | None
    prefetch_factor: int
    sampling_audit_every: int
    verify_finite_gradients_steps: int
    max_amp_overflow_events: int
    dynamic_label: DynamicLabelConfig | None
    extra_patches: ExtraPatchesConfig
    force_full_supervision: bool
    save_iterations: tuple[int, ...]

    @classmethod
    def from_mapping(cls, canonical: Mapping[str, Any]) -> "TrainingConfig":
        ink = InkConfig._from_canonical_mapping(canonical)
        num_iterations = int(canonical["num_iterations"])
        seed = int(canonical["seed"])
        batch_size = int(canonical["batch_size"])
        grad_acc_steps = int(canonical.get("grad_acc_steps", 1))
        max_steps = int(
            canonical.get(
                "max_steps", math.ceil(num_iterations / grad_acc_steps)
            )
        )
        val_every = int(canonical.get("val_every", 500))
        save_every = int(canonical.get("save_every", val_every))
        best_metric = canonical.get("best_checkpoint_metric")
        if best_metric not in (None, "val_loss", "val_balanced_accuracy"):
            raise ValueError(
                "best_checkpoint_metric must be one of "
                "{null, 'val_loss', 'val_balanced_accuracy'}"
            )
        sampling_audit_every = int(
            canonical.get("sampling_audit_every", save_every)
        )
        if sampling_audit_every <= 0:
            raise ValueError("sampling_audit_every must be positive")
        verify_finite = int(canonical.get("verify_finite_gradients_steps", 0))
        if verify_finite < 0:
            raise ValueError("verify_finite_gradients_steps cannot be negative")
        max_overflows = int(canonical.get("max_amp_overflow_events", 0))
        if max_overflows < 0:
            raise ValueError("max_amp_overflow_events cannot be negative")
        model_crop_size = ink.model.crop_size
        stitch_factor = int(canonical["stitch_factor"])
        loader_patch_size = (
            model_crop_size[0],
            model_crop_size[1] * stitch_factor,
            model_crop_size[2] * stitch_factor,
        )
        pin_memory = (
            None
            if "pin_memory" not in canonical
            else bool(canonical["pin_memory"])
        )
        if "wandb_project" in canonical and "wandb_entity" not in canonical:
            raise ValueError("wandb_project requires wandb_entity")
        raw_wandb_run_id = canonical.get("wandb_run_id")
        dynamic_label = _dynamic_label_config(
            canonical,
            mode=ink.data.mode,
            patch_size=ink.data.patch_size,
        )
        extra_patches = _extra_patches_config(
            canonical, dynamic_label=dynamic_label
        )
        if extra_patches.enabled and ink.data.sampling.strategy != "uniform":
            raise ValueError(
                "extra_patches requires sampling_strategy='uniform'"
            )
        force_full_supervision = bool(
            canonical.get("force_full_supervision", False)
        )
        if force_full_supervision and not isinstance(
            dynamic_label, SelfDistillLabelConfig
        ):
            raise ValueError(
                "force_full_supervision requires dynamic_label.kind='self_distill'"
            )
        save_iterations_value = canonical.get("save_iterations") or ()
        if not isinstance(save_iterations_value, (list, tuple)):
            raise TypeError("save_iterations must be an array")
        save_iterations = tuple(sorted({int(item) for item in save_iterations_value}))
        if any(item <= 0 or item > num_iterations for item in save_iterations):
            raise ValueError(
                "save_iterations values must be completed-iteration counts in "
                f"[1, {num_iterations}]"
            )
        return cls(
            ink=ink,
            num_iterations=num_iterations,
            # InkDataConfig defaults out_dir to the current directory.
            out_dir=Path(str(canonical["out_dir"])),
            # InkDataConfig defaults seed to 0.
            seed=seed,
            wandb_run_id=(
                None if not raw_wandb_run_id else str(raw_wandb_run_id)
            ),
            wandb_resume=bool(canonical.get("wandb_resume", False)),
            batch_size=batch_size,
            model_crop_size=model_crop_size,
            loader_patch_size=loader_patch_size,
            use_stitched_forward=bool(canonical["use_stitched_forward"]),
            stitched_gradient_checkpointing=bool(
                canonical["stitched_gradient_checkpointing"]
            ),
            ema=EmaConfig.from_mapping(canonical),
            benchmark=BenchmarkConfig.from_mapping(
                canonical, num_iterations=num_iterations
            ),
            optimizer=OptimizerConfig.from_mapping(canonical),
            scheduler=SchedulerConfig.from_mapping(
                canonical, max_steps=max_steps
            ),
            grad_acc_steps=grad_acc_steps,
            grad_clip=(
                None
                if canonical.get("grad_clip") is None
                else float(canonical["grad_clip"])
            ),
            max_steps=max_steps,
            val_every=val_every,
            save_every=save_every,
            log_every=int(canonical.get("log_every", 50)),
            val_preview_batches=int(canonical.get("val_preview_batches", 3)),
            val_steps=int(canonical.get("val_steps", 10)),
            best_checkpoint_metric=best_metric,
            mixed_precision=str(canonical.get("mixed_precision", "fp16")),
            ddp_find_unused_parameters=bool(
                canonical.get("ddp_find_unused_parameters", False)
            ),
            ddp_broadcast_buffers=bool(
                canonical.get("ddp_broadcast_buffers", False)
            ),
            # InkDataConfig defaults dataloader_workers to 8.
            dataloader_workers=int(canonical.get("dataloader_workers", 0)),
            pin_memory=pin_memory,
            prefetch_factor=int(canonical.get("prefetch_factor", 2)),
            sampling_audit_every=sampling_audit_every,
            verify_finite_gradients_steps=verify_finite,
            max_amp_overflow_events=max_overflows,
            dynamic_label=dynamic_label,
            extra_patches=extra_patches,
            force_full_supervision=force_full_supervision,
            save_iterations=save_iterations,
        )

    @property
    def surface_mask_channel(self) -> bool:
        return self.ink.data.mode == "full_3d_single_wrap"

    def to_mapping(self) -> dict[str, Any]:
        """Return the resolved training mapping persisted in checkpoints."""

        return self.ink.to_mapping()
