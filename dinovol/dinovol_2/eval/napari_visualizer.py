from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from skimage.filters import threshold_otsu
from skimage.morphology import ball
from skimage.morphology.binary import binary_dilation

from dinovol_2.dataset.normalization import NORMALIZATION_SCHEMES, get_normalization
from dinovol_2.dataset.ssl_zarr_dataset import open_zarr_handle
from dinovol_2.model.model import DinoVitStudentTeacher, _upgrade_weight_norm_state_dict_keys

DEFAULT_NORMALIZATION_SCHEME = "robust"


@dataclass
class LoadedBackbone:
    checkpoint_path: Path
    backbone: torch.nn.Module
    source_branch: str
    device: torch.device
    patch_size: tuple[int, int, int]
    input_channels: int
    normalization_scheme: str
    intensity_properties: dict[str, Any] | None
    embedding_type: str
    global_input_size: tuple[int, int, int]


@dataclass
class EmbeddingCache:
    checkpoint_path: Path
    image_layer_name: str
    source_shape: tuple[int, int, int]
    padded_shape: tuple[int, int, int]
    patch_size: tuple[int, int, int]
    normalized_patch_embeddings: np.ndarray
    source_bbox: tuple[int, int, int, int, int, int]
    bbox_layer_name: str | None


@dataclass(frozen=True)
class OmeZarrScale:
    index: int
    path: str
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    spatial_shape: tuple[int, int, int]
    spatial_scale: tuple[float, float, float]
    spatial_translate: tuple[float, float, float]
    channel_axis: int | None


@dataclass(frozen=True)
class OmeZarrSpec:
    path: str
    scales: tuple[OmeZarrScale, ...]


@dataclass(frozen=True)
class SpatialCrop:
    start_zyx: tuple[int, int, int]
    stop_zyx: tuple[int, int, int]
    bbox_layer_name: str | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(stop) - int(start) for start, stop in zip(self.start_zyx, self.stop_zyx))

    @property
    def bounds(self) -> tuple[int, int, int, int, int, int]:
        return (*self.start_zyx, *self.stop_zyx)


def _normalize_path_string(path: str | Path) -> str:
    path_str = str(path).strip()
    if "://" in path_str:
        return path_str
    return str(Path(path_str).expanduser().resolve())


def _default_ome_axes(ndim: int) -> tuple[str, ...]:
    if ndim == 3:
        return ("z", "y", "x")
    if ndim == 4:
        return ("c", "z", "y", "x")
    raise ValueError(f"unsupported OME-Zarr dimensionality {ndim}; expected 3D or 4D arrays")


def _normalize_ome_axes(raw_axes: Any, *, ndim: int) -> tuple[str, ...]:
    if raw_axes is None:
        return _default_ome_axes(ndim)

    axes: list[str] = []
    for axis in raw_axes:
        if isinstance(axis, dict):
            axis_name = str(axis.get("name", "")).strip().lower()
        else:
            axis_name = str(axis).strip().lower()
        if not axis_name:
            raise ValueError(f"invalid OME-Zarr axes metadata: {raw_axes!r}")
        axes.append(axis_name)

    normalized = tuple(axes)
    if len(normalized) != ndim:
        raise ValueError(f"OME-Zarr axes {normalized} do not match array dimensionality {ndim}")
    return normalized


def _resolve_ome_channel_axis(axes: tuple[str, ...]) -> int | None:
    allowed_axes = {"c", "z", "y", "x"}
    unsupported_axes = sorted(set(axes) - allowed_axes)
    if unsupported_axes:
        raise ValueError(
            "unsupported OME-Zarr axes "
            f"{unsupported_axes}; expected only channel/spatial axes from {{'c', 'z', 'y', 'x'}}"
        )
    for required_axis in ("z", "y", "x"):
        if required_axis not in axes:
            raise ValueError(f"OME-Zarr axes {axes} must include {required_axis!r}")
    return axes.index("c") if "c" in axes else None


def _combine_coordinate_transforms(
    coordinate_transformations: Any,
    *,
    axis_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    scale = np.ones(axis_count, dtype=np.float64)
    translation = np.zeros(axis_count, dtype=np.float64)
    if not isinstance(coordinate_transformations, list):
        return scale, translation

    for transformation in coordinate_transformations:
        if not isinstance(transformation, dict):
            continue
        transformation_type = str(transformation.get("type", "")).strip().lower()
        if transformation_type == "scale":
            values = transformation.get("scale")
            if isinstance(values, (list, tuple)) and len(values) == axis_count:
                scale *= np.asarray(values, dtype=np.float64)
        elif transformation_type == "translation":
            values = transformation.get("translation")
            if isinstance(values, (list, tuple)) and len(values) == axis_count:
                translation += np.asarray(values, dtype=np.float64)
    return scale, translation


def load_ome_zarr_spec(path: str | Path) -> OmeZarrSpec:
    path_str = _normalize_path_string(path)
    root = zarr.open_group(path_str, mode="r")
    multiscales = root.attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise ValueError("OME-Zarr root is missing multiscales metadata")

    primary_multiscale = multiscales[0]
    if not isinstance(primary_multiscale, dict):
        raise ValueError("OME-Zarr multiscales metadata is invalid")

    dataset_entries = primary_multiscale.get("datasets")
    if not isinstance(dataset_entries, list) or not dataset_entries:
        raise ValueError("OME-Zarr multiscales metadata does not contain any datasets")

    first_dataset_entry = dataset_entries[0]
    first_dataset_path = (
        str(first_dataset_entry.get("path", 0))
        if isinstance(first_dataset_entry, dict)
        else "0"
    )
    first_dataset_shape = tuple(int(v) for v in root[first_dataset_path].shape)
    multiscale_axes = primary_multiscale.get("axes")
    first_axes = _normalize_ome_axes(multiscale_axes, ndim=len(first_dataset_shape))
    multiscale_scale, multiscale_translate = _combine_coordinate_transforms(
        primary_multiscale.get("coordinateTransformations"),
        axis_count=len(first_axes),
    )

    resolved_scales: list[OmeZarrScale] = []
    for dataset_index, dataset_entry in enumerate(dataset_entries):
        if not isinstance(dataset_entry, dict):
            raise ValueError(f"OME-Zarr dataset entry {dataset_index} is invalid")

        dataset_path = str(dataset_entry.get("path", dataset_index))
        node = root[dataset_path]
        if not hasattr(node, "shape"):
            raise ValueError(f"OME-Zarr path {dataset_path!r} is not an array dataset")

        shape = tuple(int(v) for v in node.shape)
        axes = _normalize_ome_axes(multiscale_axes, ndim=len(shape))
        channel_axis = _resolve_ome_channel_axis(axes)

        dataset_scale, dataset_translate = _combine_coordinate_transforms(
            dataset_entry.get("coordinateTransformations"),
            axis_count=len(axes),
        )
        total_scale = multiscale_scale * dataset_scale
        total_translate = multiscale_translate + dataset_translate

        spatial_indices = tuple(axes.index(axis_name) for axis_name in ("z", "y", "x"))
        resolved_scales.append(
            OmeZarrScale(
                index=dataset_index,
                path=dataset_path,
                axes=axes,
                shape=shape,
                spatial_shape=tuple(int(shape[index]) for index in spatial_indices),
                spatial_scale=tuple(float(total_scale[index]) for index in spatial_indices),
                spatial_translate=tuple(float(total_translate[index]) for index in spatial_indices),
                channel_axis=channel_axis,
            )
        )

    return OmeZarrSpec(path=path_str, scales=tuple(resolved_scales))


def _reorder_ome_zarr_array(array: np.ndarray, *, axes: tuple[str, ...]) -> np.ndarray:
    channel_axis = _resolve_ome_channel_axis(axes)
    target_axes = ("c", "z", "y", "x") if channel_axis is not None else ("z", "y", "x")
    permutation = tuple(axes.index(axis_name) for axis_name in target_axes)
    if permutation == tuple(range(len(target_axes))):
        return np.asarray(array)
    return np.transpose(np.asarray(array), axes=permutation)


def load_ome_zarr_array(path: str | Path, scale: OmeZarrScale) -> np.ndarray:
    handle = open_zarr_handle(_normalize_path_string(path), scale.path)
    try:
        array = np.asarray(handle.array)
    finally:
        handle.close()
    return _reorder_ome_zarr_array(array, axes=scale.axes)


def ome_zarr_layer_transform(scale: OmeZarrScale) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if scale.channel_axis is None:
        return scale.spatial_scale, scale.spatial_translate
    return (1.0, *scale.spatial_scale), (0.0, *scale.spatial_translate)


def _as_3tuple(value: Any, *, name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        result = (int(value), int(value), int(value))
    else:
        result = tuple(int(v) for v in value)
    if len(result) != 3:
        raise ValueError(f"{name} must contain 3 integers, got {result}")
    return result


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _extract_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint does not contain a saved config dictionary")

    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint config does not contain a model configuration")
    return model_config


def _extract_dataset_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        return {}
    dataset_config = config.get("dataset")
    return dataset_config if isinstance(dataset_config, dict) else {}


def _extract_backbone_state_dict(
    checkpoint: dict[str, Any],
    *,
    preferred_branch: str = "teacher",
) -> tuple[dict[str, Any], str]:
    branch_order = [preferred_branch]
    if preferred_branch != "teacher":
        branch_order.append("teacher")
    if preferred_branch != "student":
        branch_order.append("student")

    for branch_name in branch_order:
        branch_state = checkpoint.get(branch_name)
        if not isinstance(branch_state, dict):
            continue
        backbone_state = {
            key.replace("backbone.", "", 1): value
            for key, value in branch_state.items()
            if key.startswith("backbone.")
        }
        if backbone_state:
            return _upgrade_weight_norm_state_dict_keys(backbone_state), branch_name

    raise ValueError("checkpoint does not contain teacher.backbone or student.backbone weights")


def load_backbone_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | None = None,
    preferred_branch: str = "teacher",
) -> LoadedBackbone:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = _extract_model_config(checkpoint)
    dataset_config = _extract_dataset_config(checkpoint)
    backbone_state_dict, source_branch = _extract_backbone_state_dict(
        checkpoint,
        preferred_branch=preferred_branch,
    )
    backbone = DinoVitStudentTeacher._build_backbone(model_config)
    backbone.load_state_dict(backbone_state_dict, strict=True)

    selected_device = device or _select_device()
    backbone = backbone.to(selected_device).eval()

    normalization_scheme = str(
        dataset_config.get("normalization_scheme", DEFAULT_NORMALIZATION_SCHEME)
    ).strip().lower()
    if normalization_scheme not in NORMALIZATION_SCHEMES:
        normalization_scheme = DEFAULT_NORMALIZATION_SCHEME

    return LoadedBackbone(
        checkpoint_path=checkpoint_path,
        backbone=backbone,
        source_branch=source_branch,
        device=selected_device,
        patch_size=_as_3tuple(model_config.get("patch_size", (16, 16, 16)), name="patch_size"),
        input_channels=int(model_config.get("input_channels", 1)),
        normalization_scheme=normalization_scheme,
        intensity_properties=dataset_config.get("intensity_properties"),
        embedding_type=str(model_config.get("embedding_type", "default")).strip().lower(),
        global_input_size=_as_3tuple(model_config.get("global_crops_size", (16, 16, 16)), name="global_crops_size"),
    )


def _configure_backbone_for_spatial_shape(backbone: torch.nn.Module, spatial_shape: tuple[int, int, int]) -> None:
    if getattr(backbone, "embedding_type", "default") != "deeper":
        return

    spatial_shape = tuple(int(v) for v in spatial_shape)
    backbone.global_crops_size = spatial_shape
    backbone.local_crops_size = spatial_shape
    backbone.global_input_size = spatial_shape
    backbone.local_input_size = spatial_shape
    backbone.global_ref_feat_shape = tuple(
        int(size) // int(patch) for size, patch in zip(spatial_shape, backbone.patch_size)
    )
    backbone.local_ref_feat_shape = backbone.global_ref_feat_shape


def prepare_volume_array(image: np.ndarray, *, input_channels: int) -> np.ndarray:
    volume = np.asarray(image)
    if volume.ndim == 3:
        if input_channels != 1:
            raise ValueError(
                f"checkpoint expects {input_channels} input channels but the image layer is single-channel"
            )
        volume = volume[np.newaxis, ...]
    elif volume.ndim == 4 and volume.shape[0] == input_channels:
        volume = volume
    elif volume.ndim == 4 and volume.shape[-1] == input_channels:
        volume = np.moveaxis(volume, -1, 0)
    else:
        raise ValueError(
            "image layer must be either 3D `(z, y, x)` or 4D with channel axis matching the checkpoint input channels"
        )

    if volume.ndim != 4:
        raise ValueError(f"expected a `(c, z, y, x)` array after conversion, got shape {volume.shape}")
    return volume.astype(np.float32, copy=False)


def infer_image_spatial_shape(
    image: np.ndarray,
    *,
    input_channels: int,
) -> tuple[int, int, int]:
    image_shape = tuple(int(v) for v in np.asarray(image).shape)
    if len(image_shape) == 3:
        if input_channels != 1:
            raise ValueError(
                f"checkpoint expects {input_channels} input channels but the image layer is single-channel"
            )
        return image_shape
    if len(image_shape) == 4 and image_shape[0] == input_channels:
        return image_shape[-3:]
    if len(image_shape) == 4 and image_shape[-1] == input_channels:
        return image_shape[:3]
    raise ValueError(
        "image layer must be either 3D `(z, y, x)` or 4D with channel axis matching the checkpoint input channels"
    )


def crop_image_to_spatial_bbox(
    image: np.ndarray,
    spatial_bbox: tuple[int, int, int, int, int, int],
    *,
    input_channels: int,
) -> np.ndarray:
    z0, y0, x0, z1, y1, x1 = (int(v) for v in spatial_bbox)
    spatial_slices = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
    volume = np.asarray(image)
    if volume.ndim == 3:
        if input_channels != 1:
            raise ValueError(
                f"checkpoint expects {input_channels} input channels but the image layer is single-channel"
            )
        return volume[spatial_slices]
    if volume.ndim == 4 and volume.shape[0] == input_channels:
        return volume[(slice(None), *spatial_slices)]
    if volume.ndim == 4 and volume.shape[-1] == input_channels:
        return volume[(*spatial_slices, slice(None))]
    raise ValueError(
        "image layer must be either 3D `(z, y, x)` or 4D with channel axis matching the checkpoint input channels"
    )


def cropped_spatial_translate(
    image_layer: Any,
    *,
    crop_start_zyx: tuple[int, int, int],
) -> tuple[float, float, float]:
    spatial_scale = np.asarray(image_layer.scale, dtype=np.float64)[-3:]
    spatial_translate = np.asarray(image_layer.translate, dtype=np.float64)[-3:]
    return tuple(
        float(translation + start * scale)
        for translation, start, scale in zip(spatial_translate, crop_start_zyx, spatial_scale)
    )


def point_within_spatial_bbox(
    point_zyx: np.ndarray | tuple[float, float, float],
    *,
    spatial_bbox: tuple[int, int, int, int, int, int],
) -> bool:
    point = np.asarray(point_zyx, dtype=np.float64)
    lower = np.asarray(spatial_bbox[:3], dtype=np.float64)
    upper = np.asarray(spatial_bbox[3:], dtype=np.float64)
    return bool(np.all(point >= lower) and np.all(point < upper))


def normalize_volume(
    volume: np.ndarray,
    *,
    normalization_scheme: str,
    intensity_properties: dict[str, Any] | None = None,
) -> np.ndarray:
    normalizer = get_normalization(normalization_scheme, intensityproperties=intensity_properties)
    if normalizer is None:
        return volume.astype(np.float32, copy=False)

    normalized = np.empty_like(volume, dtype=np.float32)
    for channel_index in range(volume.shape[0]):
        normalized[channel_index] = normalizer.run(volume[channel_index])
    return normalized


def pad_volume_to_patch_size(
    volume: np.ndarray,
    patch_size: tuple[int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    spatial_shape = tuple(int(v) for v in volume.shape[-3:])
    padding = []
    padded_shape = []
    for size, patch in zip(spatial_shape, patch_size):
        remainder = size % patch
        pad_after = 0 if remainder == 0 else patch - remainder
        padding.append((0, pad_after))
        padded_shape.append(size + pad_after)
    padded = np.pad(volume, ((0, 0), *padding), mode="edge")
    return padded, tuple(padded_shape)


def _patch_grid_shape_for_spatial_shape(
    spatial_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> tuple[int, int, int]:
    return tuple(int(size) // int(patch) for size, patch in zip(spatial_shape, patch_size))


def _resolve_window_spatial_shape(
    *,
    padded_shape: tuple[int, int, int],
    loaded_backbone: LoadedBackbone,
    window_size: tuple[int, int, int] | None,
) -> tuple[int, int, int] | None:
    if loaded_backbone.embedding_type != "default":
        return None

    if window_size is None:
        candidate = tuple(
            min(int(full), int(limit))
            for full, limit in zip(padded_shape, loaded_backbone.global_input_size)
        )
        if candidate == padded_shape:
            return None
        window_size = candidate

    resolved = tuple(int(size) for size in window_size)
    if len(resolved) != 3:
        raise ValueError(f"window_size must contain 3 integers, got {resolved}")
    if any(size <= 0 for size in resolved):
        raise ValueError(f"window_size must be positive, got {resolved}")
    if any(size % patch != 0 for size, patch in zip(resolved, loaded_backbone.patch_size)):
        raise ValueError(
            f"window_size must be divisible by patch_size {loaded_backbone.patch_size}, got {resolved}"
        )
    return tuple(min(full, size) for full, size in zip(padded_shape, resolved))


def _normalize_patch_overlap(
    overlap_patches: int | tuple[int, int, int],
    *,
    window_patch_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    if isinstance(overlap_patches, int):
        overlap = (int(overlap_patches),) * 3
    else:
        overlap = tuple(int(v) for v in overlap_patches)
    if len(overlap) != 3:
        raise ValueError(f"window_overlap_patches must contain 3 integers, got {overlap}")
    if any(v < 0 for v in overlap):
        raise ValueError(f"window_overlap_patches must be non-negative, got {overlap}")
    for axis_overlap, axis_window in zip(overlap, window_patch_shape):
        if axis_overlap >= axis_window:
            raise ValueError(
                "window_overlap_patches must be smaller than the window patch shape, "
                f"got overlap={overlap} and window_patch_shape={window_patch_shape}"
            )
    return overlap


def _compute_window_starts(axis_size: int, window_size: int, overlap: int) -> list[int]:
    if window_size >= axis_size:
        return [0]
    step = window_size - overlap
    if step <= 0:
        raise ValueError(
            f"window overlap must be smaller than the window size, got window_size={window_size} and overlap={overlap}"
        )
    starts = list(range(0, axis_size - window_size + 1, step))
    last_start = axis_size - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _axis_window_weights(length: int, overlap: int) -> np.ndarray:
    weights = np.ones(length, dtype=np.float32)
    if overlap <= 0 or length <= 1:
        return weights

    ramp = np.arange(length, dtype=np.float32)
    edge_distance = np.minimum(ramp + 1.0, (length - ramp).astype(np.float32))
    return np.minimum(edge_distance / float(overlap + 1), 1.0).astype(np.float32, copy=False)


def _window_weight_grid(
    window_patch_shape: tuple[int, int, int],
    overlap_patches: tuple[int, int, int],
) -> np.ndarray:
    weights = _axis_window_weights(window_patch_shape[0], overlap_patches[0])[:, None, None]
    weights = weights * _axis_window_weights(window_patch_shape[1], overlap_patches[1])[None, :, None]
    weights = weights * _axis_window_weights(window_patch_shape[2], overlap_patches[2])[None, None, :]
    return weights.astype(np.float32, copy=False)


def _reshape_and_normalize_patch_tokens(
    patch_tokens: torch.Tensor,
    *,
    patch_grid_shape: tuple[int, int, int],
) -> np.ndarray:
    patch_grid = patch_tokens.reshape(*patch_grid_shape, patch_tokens.shape[-1])
    patch_grid = F.normalize(patch_grid, dim=-1)
    return patch_grid.detach().cpu().to(torch.float32).numpy()


def _compute_patch_embedding_grid_full_volume(
    padded: np.ndarray,
    *,
    padded_shape: tuple[int, int, int],
    loaded_backbone: LoadedBackbone,
) -> np.ndarray:
    _configure_backbone_for_spatial_shape(loaded_backbone.backbone, padded_shape)

    tensor = torch.from_numpy(padded).unsqueeze(0).to(loaded_backbone.device)
    outputs = loaded_backbone.backbone.forward_features(tensor, masks=None, view_kind="global")
    patch_tokens = outputs["x_norm_patchtokens"][0]
    return _reshape_and_normalize_patch_tokens(
        patch_tokens,
        patch_grid_shape=_patch_grid_shape_for_spatial_shape(padded_shape, loaded_backbone.patch_size),
    )


def _compute_patch_embedding_grid_windowed(
    padded: np.ndarray,
    *,
    padded_shape: tuple[int, int, int],
    loaded_backbone: LoadedBackbone,
    window_spatial_shape: tuple[int, int, int],
    window_overlap_patches: int | tuple[int, int, int],
) -> np.ndarray:
    full_patch_shape = _patch_grid_shape_for_spatial_shape(padded_shape, loaded_backbone.patch_size)
    window_patch_shape = _patch_grid_shape_for_spatial_shape(window_spatial_shape, loaded_backbone.patch_size)
    overlap_patch_shape = _normalize_patch_overlap(
        window_overlap_patches,
        window_patch_shape=window_patch_shape,
    )

    axis_starts = [
        _compute_window_starts(full, window, overlap)
        for full, window, overlap in zip(full_patch_shape, window_patch_shape, overlap_patch_shape)
    ]
    weight_grid = _window_weight_grid(window_patch_shape, overlap_patch_shape)

    embedding_sum: np.ndarray | None = None
    weight_sum = np.zeros(full_patch_shape, dtype=np.float32)

    for patch_starts in product(*axis_starts):
        patch_stops = tuple(start + size for start, size in zip(patch_starts, window_patch_shape))
        voxel_slices = tuple(
            slice(start * patch, stop * patch)
            for start, stop, patch in zip(patch_starts, patch_stops, loaded_backbone.patch_size)
        )
        tile = padded[(slice(None), *voxel_slices)]
        tensor = torch.from_numpy(tile).unsqueeze(0).to(loaded_backbone.device)
        outputs = loaded_backbone.backbone.forward_features(tensor, masks=None, view_kind="global")
        patch_tokens = outputs["x_norm_patchtokens"][0]
        tile_grid = _reshape_and_normalize_patch_tokens(
            patch_tokens,
            patch_grid_shape=window_patch_shape,
        )

        if embedding_sum is None:
            embedding_sum = np.zeros((*full_patch_shape, tile_grid.shape[-1]), dtype=np.float32)

        destination = tuple(slice(start, stop) for start, stop in zip(patch_starts, patch_stops))
        embedding_sum[destination] += tile_grid * weight_grid[..., None]
        weight_sum[destination] += weight_grid

    if embedding_sum is None:
        raise RuntimeError("windowed patch embedding inference produced no tiles")

    patch_grid = embedding_sum / np.maximum(weight_sum[..., None], np.finfo(np.float32).eps)
    patch_grid = F.normalize(torch.from_numpy(patch_grid), dim=-1)
    return patch_grid.numpy().astype(np.float32, copy=False)


@torch.inference_mode()
def compute_patch_embedding_grid(
    volume: np.ndarray,
    loaded_backbone: LoadedBackbone,
    *,
    window_size: tuple[int, int, int] | None = None,
    window_overlap_patches: int | tuple[int, int, int] = 1,
) -> tuple[np.ndarray, tuple[int, int, int], tuple[int, int, int]]:
    prepared = prepare_volume_array(volume, input_channels=loaded_backbone.input_channels)
    normalized = normalize_volume(
        prepared,
        normalization_scheme=loaded_backbone.normalization_scheme,
        intensity_properties=loaded_backbone.intensity_properties,
    )
    padded, padded_shape = pad_volume_to_patch_size(normalized, loaded_backbone.patch_size)
    window_spatial_shape = _resolve_window_spatial_shape(
        padded_shape=padded_shape,
        loaded_backbone=loaded_backbone,
        window_size=window_size,
    )

    if window_spatial_shape is None or window_spatial_shape == padded_shape:
        patch_grid = _compute_patch_embedding_grid_full_volume(
            padded,
            padded_shape=padded_shape,
            loaded_backbone=loaded_backbone,
        )
    else:
        patch_grid = _compute_patch_embedding_grid_windowed(
            padded,
            padded_shape=padded_shape,
            loaded_backbone=loaded_backbone,
            window_spatial_shape=window_spatial_shape,
            window_overlap_patches=window_overlap_patches,
        )
    return (
        patch_grid,
        tuple(int(v) for v in prepared.shape[-3:]),
        padded_shape,
    )


def point_to_patch_index(
    point_zyx: np.ndarray | tuple[float, float, float],
    *,
    source_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    patch_grid_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    point = np.asarray(point_zyx, dtype=np.float64)
    if point.shape != (3,):
        raise ValueError(f"point must contain exactly 3 spatial coordinates, got shape {point.shape}")

    voxel_index = np.floor(point).astype(np.int64)
    voxel_index = np.clip(voxel_index, 0, np.asarray(source_shape, dtype=np.int64) - 1)
    patch_index = voxel_index // np.asarray(patch_size, dtype=np.int64)
    patch_index = np.clip(patch_index, 0, np.asarray(patch_grid_shape, dtype=np.int64) - 1)
    return tuple(int(v) for v in patch_index)


def cosine_similarity_patch_grid(
    normalized_patch_embeddings: np.ndarray,
    reference_patch_index: tuple[int, int, int],
) -> np.ndarray:
    reference_embedding = normalized_patch_embeddings[reference_patch_index]
    similarity = np.tensordot(normalized_patch_embeddings, reference_embedding, axes=([-1], [0]))
    return similarity.astype(np.float32, copy=False)


def upsample_patch_grid_to_volume(
    patch_similarity: np.ndarray,
    *,
    patch_size: tuple[int, int, int],
    output_shape: tuple[int, int, int],
) -> np.ndarray:
    dense = patch_similarity
    for axis, repeat_factor in enumerate(patch_size):
        dense = np.repeat(dense, repeat_factor, axis=axis)
    trailing_slices = (slice(None),) * max(dense.ndim - 3, 0)
    slices = tuple(slice(0, int(size)) for size in output_shape) + trailing_slices
    return dense[slices].astype(np.float32, copy=False)


def compute_otsu_foreground_mask(
    image: np.ndarray,
    *,
    input_channels: int,
    dilation_radius: int = 0,
) -> np.ndarray:
    prepared = prepare_volume_array(image, input_channels=input_channels)
    intensity_volume = prepared.mean(axis=0)
    finite_mask = np.isfinite(intensity_volume)
    if not finite_mask.any():
        raise ValueError("cannot compute an Otsu foreground mask from an all-NaN volume")

    threshold = float(threshold_otsu(intensity_volume[finite_mask]))
    foreground_mask = np.zeros_like(intensity_volume, dtype=bool)
    foreground_mask[finite_mask] = intensity_volume[finite_mask] > threshold
    if not foreground_mask.any():
        foreground_mask[finite_mask] = intensity_volume[finite_mask] >= threshold

    dilation_radius = int(dilation_radius)
    if dilation_radius < 0:
        raise ValueError(f"dilation_radius must be non-negative, got {dilation_radius}")
    if dilation_radius > 0:
        foreground_mask = binary_dilation(foreground_mask, footprint=ball(dilation_radius))
    return foreground_mask


def foreground_mask_to_patch_mask(
    foreground_mask: np.ndarray,
    *,
    patch_size: tuple[int, int, int],
    padded_shape: tuple[int, int, int],
) -> np.ndarray:
    mask = np.asarray(foreground_mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"foreground_mask must be 3D, got shape {mask.shape}")

    if any(int(padded) < int(size) for padded, size in zip(padded_shape, mask.shape)):
        raise ValueError(
            f"padded_shape {padded_shape} must be greater than or equal to the mask shape {mask.shape}"
        )
    if any(int(padded) % int(patch) != 0 for padded, patch in zip(padded_shape, patch_size)):
        raise ValueError(f"padded_shape {padded_shape} must be divisible by patch_size {patch_size}")

    padding = tuple((0, int(padded) - int(size)) for size, padded in zip(mask.shape, padded_shape))
    padded_mask = np.pad(mask, padding, mode="constant", constant_values=False)
    patch_d, patch_h, patch_w = (int(v) for v in patch_size)
    reshaped = padded_mask.reshape(
        padded_shape[0] // patch_d,
        patch_d,
        padded_shape[1] // patch_h,
        patch_h,
        padded_shape[2] // patch_w,
        patch_w,
    )
    return reshaped.any(axis=(1, 3, 5))


def project_patch_embeddings_to_pca_rgb(
    normalized_patch_embeddings: np.ndarray,
    *,
    patch_mask: np.ndarray | None = None,
) -> np.ndarray:
    embeddings = np.asarray(normalized_patch_embeddings, dtype=np.float32)
    if embeddings.ndim != 4:
        raise ValueError(
            "normalized_patch_embeddings must have shape `(patch_z, patch_y, patch_x, channels)`"
        )

    flat_embeddings = embeddings.reshape(-1, embeddings.shape[-1]).astype(np.float64, copy=False)
    if patch_mask is None:
        flat_mask = np.ones(flat_embeddings.shape[0], dtype=bool)
    else:
        mask = np.asarray(patch_mask, dtype=bool)
        if mask.shape != embeddings.shape[:3]:
            raise ValueError(f"patch_mask shape {mask.shape} must match {embeddings.shape[:3]}")
        flat_mask = mask.reshape(-1)
        if not flat_mask.any():
            raise ValueError("patch_mask excludes all patch embeddings")

    selected_embeddings = flat_embeddings[flat_mask]
    mean_embedding = selected_embeddings.mean(axis=0, keepdims=True)
    centered_selected = selected_embeddings - mean_embedding
    centered_all = flat_embeddings - mean_embedding

    component_count = min(3, centered_all.shape[1], max(centered_selected.shape[0] - 1, 1))
    projected = np.zeros((flat_embeddings.shape[0], 3), dtype=np.float32)

    if component_count > 0 and np.any(centered_selected):
        covariance = centered_selected.T @ centered_selected
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1][:component_count]
        components = eigenvectors[:, order]
        projected[:, :component_count] = (centered_all @ components).astype(np.float32, copy=False)

    rgb = projected.reshape(*embeddings.shape[:3], 3)
    active_values = rgb.reshape(-1, 3)[flat_mask]
    for channel_index in range(3):
        channel_values = active_values[:, channel_index]
        minimum = float(channel_values.min())
        maximum = float(channel_values.max())
        if maximum > minimum:
            rgb[..., channel_index] = (rgb[..., channel_index] - minimum) / (maximum - minimum)
        else:
            rgb[..., channel_index] = 0.0

    if patch_mask is not None:
        rgb[~patch_mask] = 0.0
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


try:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QCheckBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QVBoxLayout,
        QWidget,
        QComboBox,
    )

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False


if _QT_AVAILABLE:
    class CosineSimilarityWidget(QWidget):
        def __init__(self, viewer: Any) -> None:
            super().__init__()
            self.viewer = viewer
            self.loaded_backbone: LoadedBackbone | None = None
            self.embedding_cache: EmbeddingCache | None = None
            self.ome_zarr_spec: OmeZarrSpec | None = None

            self.setWindowTitle("DINO Cosine Similarity")
            self.zarr_path_edit = QLineEdit()
            self.zarr_scale_combo = QComboBox()
            self.checkpoint_path_edit = QLineEdit()
            self.normalization_combo = QComboBox()
            for scheme in NORMALIZATION_SCHEMES:
                self.normalization_combo.addItem(scheme)

            self.image_layer_combo = QComboBox()
            self.points_layer_combo = QComboBox()
            self.otsu_mask_checkbox = QCheckBox()
            self.otsu_mask_checkbox.setChecked(False)
            self.mask_dilation_spinbox = QSpinBox()
            self.mask_dilation_spinbox.setRange(0, 1024)
            self.mask_dilation_spinbox.setValue(0)
            self.status_label = QLabel(
                "Select a checkpoint, then open an OME-Zarr or choose an existing image layer."
            )

            browse_zarr_button = QPushButton("Browse Zarr")
            browse_zarr_button.clicked.connect(self._browse_zarr_directory)

            load_scales_button = QPushButton("Load Scales")
            load_scales_button.clicked.connect(self._load_zarr_scales)

            open_zarr_button = QPushButton("Open Zarr")
            open_zarr_button.clicked.connect(self._open_zarr)

            browse_checkpoint_button = QPushButton("Browse")
            browse_checkpoint_button.clicked.connect(self._browse_checkpoint)

            refresh_button = QPushButton("Refresh Layers")
            refresh_button.clicked.connect(self.refresh_layer_choices)

            cache_button = QPushButton("Cache Embeddings")
            cache_button.clicked.connect(self._cache_embeddings)

            pca_button = QPushButton("Show Feature PCA")
            pca_button.clicked.connect(self._show_feature_pca)

            selected_button = QPushButton("Similarity For Selected Points")
            selected_button.clicked.connect(self._create_layers_for_selected_points)

            all_button = QPushButton("Similarity For All Points")
            all_button.clicked.connect(self._create_layers_for_all_points)

            zarr_path_row = QHBoxLayout()
            zarr_path_row.addWidget(self.zarr_path_edit)
            zarr_path_row.addWidget(browse_zarr_button)

            zarr_scale_row = QHBoxLayout()
            zarr_scale_row.addWidget(self.zarr_scale_combo)
            zarr_scale_row.addWidget(load_scales_button)
            zarr_scale_row.addWidget(open_zarr_button)

            checkpoint_row = QHBoxLayout()
            checkpoint_row.addWidget(self.checkpoint_path_edit)
            checkpoint_row.addWidget(browse_checkpoint_button)

            form_layout = QFormLayout()
            form_layout.addRow("OME-Zarr", zarr_path_row)
            form_layout.addRow("Zarr Scale", zarr_scale_row)
            form_layout.addRow("Checkpoint", checkpoint_row)
            form_layout.addRow("Normalization", self.normalization_combo)
            form_layout.addRow("Image Layer", self.image_layer_combo)
            form_layout.addRow("Points Layer", self.points_layer_combo)
            form_layout.addRow("Otsu Foreground Mask", self.otsu_mask_checkbox)
            form_layout.addRow("Mask Dilation", self.mask_dilation_spinbox)

            button_layout = QHBoxLayout()
            button_layout.addWidget(refresh_button)
            button_layout.addWidget(cache_button)
            button_layout.addWidget(pca_button)
            button_layout.addWidget(selected_button)
            button_layout.addWidget(all_button)

            content_widget = QWidget()
            content_layout = QVBoxLayout()
            content_layout.addLayout(form_layout)
            content_layout.addLayout(button_layout)
            content_layout.addWidget(self.status_label)
            content_widget.setLayout(content_layout)

            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll_area.setWidget(content_widget)

            layout = QVBoxLayout()
            layout.addWidget(scroll_area)
            self.setLayout(layout)

            self.normalization_combo.currentTextChanged.connect(self._invalidate_cache)
            self.image_layer_combo.currentTextChanged.connect(self._invalidate_cache)
            self.zarr_scale_combo.currentIndexChanged.connect(self._invalidate_cache)

            self.refresh_layer_choices()

        def _set_status(self, message: str) -> None:
            self.status_label.setText(message)

        def _invalidate_cache(self, *_args: Any) -> None:
            self.embedding_cache = None

        def _browse_zarr_directory(self) -> None:
            directory = QFileDialog.getExistingDirectory(
                self,
                "Select OME-Zarr directory",
                str(Path.cwd()),
            )
            if not directory:
                return
            self.zarr_path_edit.setText(directory)
            self._load_zarr_scales()

        def _browse_checkpoint(self) -> None:
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Select checkpoint",
                str(Path.cwd()),
                "PyTorch checkpoints (*.pt *.pth *.bin);;All files (*)",
            )
            if not filename:
                return
            self.checkpoint_path_edit.setText(filename)
            self.loaded_backbone = None
            self._invalidate_cache()
            self._set_status("Checkpoint updated.")

        def _load_zarr_scales(self) -> None:
            path_text = self.zarr_path_edit.text().strip()
            current_index = self.zarr_scale_combo.currentIndex()
            self.zarr_scale_combo.clear()
            self.ome_zarr_spec = None
            if not path_text:
                return

            spec = load_ome_zarr_spec(path_text)
            self.ome_zarr_spec = spec
            for scale in spec.scales:
                label = (
                    f"{scale.index}: {scale.path} | shape={scale.spatial_shape} | "
                    f"voxel={tuple(round(v, 6) for v in scale.spatial_scale)}"
                )
                self.zarr_scale_combo.addItem(label)

            if spec.scales:
                self.zarr_scale_combo.setCurrentIndex(max(0, min(current_index, len(spec.scales) - 1)))
            self._set_status(f"Loaded {len(spec.scales)} OME-Zarr scale(s) from {spec.path}.")

        def _selected_zarr_scale(self) -> OmeZarrScale:
            if self.ome_zarr_spec is None:
                self._load_zarr_scales()
            if self.ome_zarr_spec is None or not self.ome_zarr_spec.scales:
                raise ValueError("load an OME-Zarr before opening a scale")

            scale_index = self.zarr_scale_combo.currentIndex()
            if scale_index < 0:
                scale_index = 0
            return self.ome_zarr_spec.scales[scale_index]

        def _open_zarr(self) -> None:
            selected_scale = self._selected_zarr_scale()
            if self.ome_zarr_spec is None:
                raise RuntimeError("OME-Zarr metadata was not loaded")

            self._set_status(
                f"Opening {Path(self.ome_zarr_spec.path.rstrip('/')).name or self.ome_zarr_spec.path} "
                f"scale {selected_scale.index}..."
            )
            image_data = load_ome_zarr_array(self.ome_zarr_spec.path, selected_scale)
            layer_scale, layer_translate = ome_zarr_layer_transform(selected_scale)
            layer_basename = Path(self.ome_zarr_spec.path.rstrip("/")).name or self.ome_zarr_spec.path.rstrip("/").split("/")[-1]
            layer_name = f"{layer_basename}_s{selected_scale.index}"
            bbox_layer_name = f"{layer_name}_bbox"
            metadata = {
                "ome_zarr_path": self.ome_zarr_spec.path,
                "ome_zarr_scale_index": int(selected_scale.index),
                "ome_zarr_scale_path": selected_scale.path,
                "ome_zarr_axes": list(selected_scale.axes),
                "bbox_layer_name": bbox_layer_name,
            }

            if layer_name in self.viewer.layers:
                image_layer = self.viewer.layers[layer_name]
                image_layer.data = image_data
                image_layer.scale = layer_scale
                image_layer.translate = layer_translate
                image_layer.metadata = metadata
            else:
                image_layer = self.viewer.add_image(
                    image_data,
                    name=layer_name,
                    scale=layer_scale,
                    translate=layer_translate,
                    metadata=metadata,
                )

            bbox_layer = self._ensure_bbox_layer_for_image(image_layer)
            self._invalidate_cache()
            self.refresh_layer_choices()
            self._restore_combo_text(self.image_layer_combo, image_layer.name)
            self._set_status(
                f"Opened {layer_name}. Draw a rectangle in {bbox_layer.name}; embeddings use that YX bbox "
                f"across the full Z span of scale {selected_scale.index}."
            )

        def refresh_layer_choices(self) -> None:
            from napari.layers import Image, Points

            current_image = self.image_layer_combo.currentText()
            current_points = self.points_layer_combo.currentText()

            self.image_layer_combo.clear()
            self.points_layer_combo.clear()

            for layer in self.viewer.layers:
                if isinstance(layer, Image):
                    self.image_layer_combo.addItem(layer.name)
                if isinstance(layer, Points):
                    self.points_layer_combo.addItem(layer.name)

            self._restore_combo_text(self.image_layer_combo, current_image)
            self._restore_combo_text(self.points_layer_combo, current_points)

        @staticmethod
        def _restore_combo_text(combo: QComboBox, text: str) -> None:
            if not text:
                return
            index = combo.findText(text)
            if index >= 0:
                combo.setCurrentIndex(index)

        def _selected_image_layer(self) -> Any:
            name = self.image_layer_combo.currentText()
            if not name:
                raise ValueError("select an image layer")
            return self.viewer.layers[name]

        def _selected_points_layer(self) -> Any:
            name = self.points_layer_combo.currentText()
            if not name:
                raise ValueError("select a points layer")
            return self.viewer.layers[name]

        def _load_backbone(self) -> LoadedBackbone:
            checkpoint_text = self.checkpoint_path_edit.text().strip()
            if not checkpoint_text:
                raise ValueError("select a checkpoint file")

            checkpoint_path = Path(checkpoint_text).expanduser().resolve()
            if self.loaded_backbone is not None and self.loaded_backbone.checkpoint_path == checkpoint_path:
                self.loaded_backbone.normalization_scheme = self.normalization_combo.currentText()
                return self.loaded_backbone

            self._set_status("Loading checkpoint...")
            loaded_backbone = load_backbone_from_checkpoint(checkpoint_path)
            combo_index = self.normalization_combo.findText(loaded_backbone.normalization_scheme)
            if combo_index >= 0:
                self.normalization_combo.setCurrentIndex(combo_index)
            loaded_backbone.normalization_scheme = self.normalization_combo.currentText()
            self.loaded_backbone = loaded_backbone
            self._invalidate_cache()
            self._set_status(
                f"Loaded {loaded_backbone.source_branch} backbone from {checkpoint_path.name} on "
                f"{loaded_backbone.device.type} with patch size {loaded_backbone.patch_size}."
            )
            return loaded_backbone

        def _bbox_layer_for_image(self, image_layer: Any) -> Any | None:
            from napari.layers import Shapes

            bbox_layer_name = str((getattr(image_layer, "metadata", {}) or {}).get("bbox_layer_name", "")).strip()
            if not bbox_layer_name:
                return None
            if bbox_layer_name not in self.viewer.layers:
                raise ValueError(
                    f"bbox layer {bbox_layer_name!r} is missing; reopen the OME-Zarr to recreate it"
                )
            bbox_layer = self.viewer.layers[bbox_layer_name]
            if not isinstance(bbox_layer, Shapes):
                raise ValueError(f"bbox layer {bbox_layer_name!r} is not a Shapes layer")
            return bbox_layer

        def _ensure_bbox_layer_for_image(self, image_layer: Any) -> Any:
            from napari.layers import Shapes

            metadata = dict(getattr(image_layer, "metadata", {}) or {})
            bbox_layer_name = str(metadata.get("bbox_layer_name", f"{image_layer.name}_bbox")).strip()
            metadata["bbox_layer_name"] = bbox_layer_name
            image_layer.metadata = metadata

            image_scale = tuple(float(v) for v in np.asarray(image_layer.scale, dtype=np.float64))
            image_translate = tuple(float(v) for v in np.asarray(image_layer.translate, dtype=np.float64))
            bbox_metadata = {
                "target_image_layer": image_layer.name,
                "ome_zarr_path": metadata.get("ome_zarr_path"),
                "ome_zarr_scale_index": metadata.get("ome_zarr_scale_index"),
            }

            if bbox_layer_name in self.viewer.layers:
                bbox_layer = self.viewer.layers[bbox_layer_name]
                if not isinstance(bbox_layer, Shapes):
                    raise ValueError(f"existing layer {bbox_layer_name!r} is not a Shapes layer")
                bbox_layer.scale = image_scale
                bbox_layer.translate = image_translate
                bbox_layer.metadata = bbox_metadata
                return bbox_layer

            return self.viewer.add_shapes(
                ndim=np.asarray(image_layer.data).ndim,
                name=bbox_layer_name,
                scale=image_scale,
                translate=image_translate,
                edge_color="yellow",
                face_color=[0.0, 0.0, 0.0, 0.0],
                edge_width=2.0,
                metadata=bbox_metadata,
            )

        def _primary_rectangle_index(self, bbox_layer: Any) -> int:
            shape_types = [str(shape_type).strip().lower() for shape_type in getattr(bbox_layer, "shape_type", [])]
            rectangle_indices = [index for index, shape_type in enumerate(shape_types) if shape_type == "rectangle"]
            if not rectangle_indices:
                raise ValueError(f"draw a rectangle in {bbox_layer.name} to define the active bbox")

            selected_rectangles = [
                int(index)
                for index in sorted(int(index) for index in getattr(bbox_layer, "selected_data", set()))
                if index in rectangle_indices
            ]
            if len(selected_rectangles) == 1:
                return selected_rectangles[0]
            if len(rectangle_indices) == 1:
                return rectangle_indices[0]
            raise ValueError(f"select exactly one rectangle in {bbox_layer.name}")

        def _spatial_crop_for_image(
            self,
            *,
            image_layer: Any,
            loaded_backbone: LoadedBackbone,
        ) -> SpatialCrop:
            source_shape = infer_image_spatial_shape(
                np.asarray(image_layer.data),
                input_channels=loaded_backbone.input_channels,
            )
            bbox_layer = self._bbox_layer_for_image(image_layer)
            if bbox_layer is None:
                return SpatialCrop((0, 0, 0), source_shape, None)

            rectangle_index = self._primary_rectangle_index(bbox_layer)
            rectangle_vertices = np.asarray(bbox_layer.data[rectangle_index], dtype=np.float64)
            world_vertices = np.asarray(
                [bbox_layer.data_to_world(vertex) for vertex in rectangle_vertices],
                dtype=np.float64,
            )
            image_vertices = np.asarray(
                [image_layer.world_to_data(vertex) for vertex in world_vertices],
                dtype=np.float64,
            )
            spatial_vertices = image_vertices[:, -3:]

            y0 = max(0, int(np.floor(np.min(spatial_vertices[:, 1]))))
            x0 = max(0, int(np.floor(np.min(spatial_vertices[:, 2]))))
            y1 = min(source_shape[1], int(np.ceil(np.max(spatial_vertices[:, 1]))))
            x1 = min(source_shape[2], int(np.ceil(np.max(spatial_vertices[:, 2]))))
            if y1 <= y0 or x1 <= x0:
                raise ValueError(f"bbox rectangle in {bbox_layer.name} collapses after scaling; redraw it")

            return SpatialCrop(
                start_zyx=(0, y0, x0),
                stop_zyx=(source_shape[0], y1, x1),
                bbox_layer_name=bbox_layer.name,
            )

        def _cache_embeddings(self) -> None:
            loaded_backbone = self._load_backbone()
            image_layer = self._selected_image_layer()
            spatial_crop = self._spatial_crop_for_image(image_layer=image_layer, loaded_backbone=loaded_backbone)
            self._set_status(f"Computing patch embeddings for {image_layer.name} within bbox {spatial_crop.bounds}...")

            cropped_volume = crop_image_to_spatial_bbox(
                np.asarray(image_layer.data),
                spatial_crop.bounds,
                input_channels=loaded_backbone.input_channels,
            )
            patch_embeddings, source_shape, padded_shape = compute_patch_embedding_grid(
                cropped_volume,
                loaded_backbone,
            )
            self.embedding_cache = EmbeddingCache(
                checkpoint_path=loaded_backbone.checkpoint_path,
                image_layer_name=image_layer.name,
                source_shape=source_shape,
                padded_shape=padded_shape,
                patch_size=loaded_backbone.patch_size,
                normalized_patch_embeddings=patch_embeddings,
                source_bbox=spatial_crop.bounds,
                bbox_layer_name=spatial_crop.bbox_layer_name,
            )
            patch_grid_shape = patch_embeddings.shape[:3]
            self._set_status(
                f"Cached embeddings for {image_layer.name}: patch grid {patch_grid_shape}, "
                f"crop shape {source_shape}, crop bbox {spatial_crop.bounds}."
            )

        def _ensure_embedding_cache(self) -> tuple[LoadedBackbone, EmbeddingCache]:
            loaded_backbone = self._load_backbone()
            image_layer = self._selected_image_layer()
            spatial_crop = self._spatial_crop_for_image(image_layer=image_layer, loaded_backbone=loaded_backbone)
            if (
                self.embedding_cache is None
                or self.embedding_cache.checkpoint_path != loaded_backbone.checkpoint_path
                or self.embedding_cache.image_layer_name != image_layer.name
                or self.embedding_cache.source_bbox != spatial_crop.bounds
                or self.embedding_cache.bbox_layer_name != spatial_crop.bbox_layer_name
            ):
                self._cache_embeddings()
            if self.embedding_cache is None:
                raise RuntimeError("embedding cache was not created")
            return loaded_backbone, self.embedding_cache

        def _point_image_coordinates(self, image_layer: Any, points_layer: Any, point_index: int) -> np.ndarray:
            point = np.asarray(points_layer.data[point_index], dtype=np.float64)
            world = np.asarray(points_layer.data_to_world(point), dtype=np.float64)
            image_coords = np.asarray(image_layer.world_to_data(world), dtype=np.float64)
            return image_coords[-3:]

        def _selected_point_indices(self, points_layer: Any) -> list[int]:
            selected = sorted(int(index) for index in points_layer.selected_data)
            if selected:
                return selected
            if len(points_layer.data) == 1:
                return [0]
            raise ValueError("select at least one point in the points layer")

        def _replace_or_add_similarity_layer(
            self,
            *,
            layer_name: str,
            similarity_volume: np.ndarray,
            image_layer: Any,
            spatial_bbox: tuple[int, int, int, int, int, int],
            metadata: dict[str, Any],
        ) -> None:
            spatial_scale = tuple(float(v) for v in np.asarray(image_layer.scale, dtype=np.float64)[-3:])
            spatial_translate = cropped_spatial_translate(
                image_layer,
                crop_start_zyx=tuple(int(v) for v in spatial_bbox[:3]),
            )
            if layer_name in self.viewer.layers:
                layer = self.viewer.layers[layer_name]
                layer.data = similarity_volume
                layer.scale = spatial_scale
                layer.translate = spatial_translate
                layer.metadata = metadata
                layer.contrast_limits = (-1.0, 1.0)
                return

            self.viewer.add_image(
                similarity_volume,
                name=layer_name,
                scale=spatial_scale,
                translate=spatial_translate,
                colormap="turbo",
                opacity=0.6,
                blending="additive",
                contrast_limits=(-1.0, 1.0),
                metadata=metadata,
            )

        def _replace_or_add_pca_layer(
            self,
            *,
            layer_name: str,
            pca_volume: np.ndarray,
            image_layer: Any,
            spatial_bbox: tuple[int, int, int, int, int, int],
            metadata: dict[str, Any],
        ) -> None:
            spatial_scale = tuple(float(v) for v in np.asarray(image_layer.scale, dtype=np.float64)[-3:])
            spatial_translate = cropped_spatial_translate(
                image_layer,
                crop_start_zyx=tuple(int(v) for v in spatial_bbox[:3]),
            )
            if layer_name in self.viewer.layers:
                layer = self.viewer.layers[layer_name]
                layer.data = pca_volume
                layer.scale = spatial_scale
                layer.translate = spatial_translate
                layer.metadata = metadata
                layer.opacity = 0.85
                layer.blending = "translucent"
                return

            self.viewer.add_image(
                pca_volume,
                name=layer_name,
                scale=spatial_scale,
                translate=spatial_translate,
                opacity=0.85,
                blending="translucent",
                metadata=metadata,
                rgb=True,
            )

        def _foreground_mask_for_pca(
            self,
            *,
            image_layer: Any,
            loaded_backbone: LoadedBackbone,
            embedding_cache: EmbeddingCache,
        ) -> tuple[np.ndarray | None, np.ndarray | None]:
            if not self.otsu_mask_checkbox.isChecked():
                return None, None

            dilation_radius = int(self.mask_dilation_spinbox.value())
            cropped_image = crop_image_to_spatial_bbox(
                np.asarray(image_layer.data),
                embedding_cache.source_bbox,
                input_channels=loaded_backbone.input_channels,
            )
            foreground_mask = compute_otsu_foreground_mask(
                cropped_image,
                input_channels=loaded_backbone.input_channels,
                dilation_radius=dilation_radius,
            )
            patch_mask = foreground_mask_to_patch_mask(
                foreground_mask,
                patch_size=embedding_cache.patch_size,
                padded_shape=embedding_cache.padded_shape,
            )
            return foreground_mask, patch_mask

        def _create_similarity_layers(self, point_indices: list[int]) -> None:
            _, embedding_cache = self._ensure_embedding_cache()
            image_layer = self._selected_image_layer()
            points_layer = self._selected_points_layer()
            patch_grid_shape = embedding_cache.normalized_patch_embeddings.shape[:3]
            crop_origin = np.asarray(embedding_cache.source_bbox[:3], dtype=np.float64)

            created_count = 0
            skipped_points: list[int] = []
            for point_index in point_indices:
                image_coords = self._point_image_coordinates(image_layer, points_layer, point_index)
                if not point_within_spatial_bbox(image_coords, spatial_bbox=embedding_cache.source_bbox):
                    skipped_points.append(int(point_index))
                    continue

                local_image_coords = image_coords - crop_origin
                reference_patch_index = point_to_patch_index(
                    local_image_coords,
                    source_shape=embedding_cache.source_shape,
                    patch_size=embedding_cache.patch_size,
                    patch_grid_shape=patch_grid_shape,
                )
                patch_similarity = cosine_similarity_patch_grid(
                    embedding_cache.normalized_patch_embeddings,
                    reference_patch_index,
                )
                dense_similarity = upsample_patch_grid_to_volume(
                    patch_similarity,
                    patch_size=embedding_cache.patch_size,
                    output_shape=embedding_cache.source_shape,
                )
                layer_name = f"cosine_{points_layer.name}_{point_index:03d}"
                metadata = {
                    "checkpoint_path": str(embedding_cache.checkpoint_path),
                    "reference_point_index": int(point_index),
                    "reference_point_zyx": image_coords.tolist(),
                    "reference_point_local_zyx": local_image_coords.tolist(),
                    "reference_patch_index": list(reference_patch_index),
                    "source_bbox_zyxzyx": list(embedding_cache.source_bbox),
                    "bbox_layer_name": embedding_cache.bbox_layer_name,
                }
                self._replace_or_add_similarity_layer(
                    layer_name=layer_name,
                    similarity_volume=dense_similarity,
                    image_layer=image_layer,
                    spatial_bbox=embedding_cache.source_bbox,
                    metadata=metadata,
                )
                created_count += 1

            if created_count == 0:
                raise ValueError("no requested points fall inside the active bbox")

            status = f"Created {created_count} cosine similarity layer(s) within bbox {embedding_cache.source_bbox}."
            if skipped_points:
                status += f" Skipped points outside bbox: {skipped_points}."
            self._set_status(status)

        def _show_feature_pca(self) -> None:
            loaded_backbone, embedding_cache = self._ensure_embedding_cache()
            image_layer = self._selected_image_layer()
            self._set_status(f"Computing PCA visualization for {image_layer.name} within bbox...")

            foreground_mask, patch_mask = self._foreground_mask_for_pca(
                image_layer=image_layer,
                loaded_backbone=loaded_backbone,
                embedding_cache=embedding_cache,
            )
            pca_patch_grid = project_patch_embeddings_to_pca_rgb(
                embedding_cache.normalized_patch_embeddings,
                patch_mask=patch_mask,
            )
            pca_volume = upsample_patch_grid_to_volume(
                pca_patch_grid,
                patch_size=embedding_cache.patch_size,
                output_shape=embedding_cache.source_shape,
            )
            if foreground_mask is not None:
                pca_volume[~foreground_mask] = 0.0

            layer_name = f"feature_pca_{image_layer.name}"
            metadata = {
                "checkpoint_path": str(embedding_cache.checkpoint_path),
                "mask_method": "otsu" if foreground_mask is not None else "none",
                "mask_dilation_radius": int(self.mask_dilation_spinbox.value()) if foreground_mask is not None else 0,
                "source_bbox_zyxzyx": list(embedding_cache.source_bbox),
                "bbox_layer_name": embedding_cache.bbox_layer_name,
            }
            self._replace_or_add_pca_layer(
                layer_name=layer_name,
                pca_volume=pca_volume,
                image_layer=image_layer,
                spatial_bbox=embedding_cache.source_bbox,
                metadata=metadata,
            )

            mask_suffix = (
                f" with Otsu mask (dilation={int(self.mask_dilation_spinbox.value())})"
                if foreground_mask is not None
                else ""
            )
            self._set_status(
                f"Created PCA feature layer for {image_layer.name} within bbox {embedding_cache.source_bbox}"
                f"{mask_suffix}."
            )

        def _create_layers_for_selected_points(self) -> None:
            points_layer = self._selected_points_layer()
            point_indices = self._selected_point_indices(points_layer)
            self._set_status(f"Creating similarity layers for {len(point_indices)} selected point(s)...")
            self._create_similarity_layers(point_indices)

        def _create_layers_for_all_points(self) -> None:
            points_layer = self._selected_points_layer()
            point_indices = list(range(len(points_layer.data)))
            if not point_indices:
                raise ValueError("points layer does not contain any points")
            self._set_status(f"Creating similarity layers for all {len(point_indices)} point(s)...")
            self._create_similarity_layers(point_indices)
else:
    CosineSimilarityWidget = None


def add_cosine_similarity_widget(viewer: Any) -> Any:
    if not _QT_AVAILABLE or CosineSimilarityWidget is None:
        raise ImportError("qtpy is required to create the cosine similarity widget")
    widget = CosineSimilarityWidget(viewer)
    viewer.window.add_dock_widget(widget, area="bottom", name="DINO Cosine Similarity")
    return widget


def main() -> None:
    try:
        import napari
    except ImportError as exc:
        raise SystemExit("napari is required to run this viewer script") from exc

    viewer = napari.Viewer()
    add_cosine_similarity_widget(viewer)
    napari.run()


if __name__ == "__main__":
    main()
