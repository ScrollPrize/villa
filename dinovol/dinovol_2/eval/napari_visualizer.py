from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from skimage.filters import threshold_otsu
from skimage.morphology import ball
from skimage.morphology.binary import binary_dilation

from dinovol_2.dataset.normalization import NORMALIZATION_SCHEMES, get_normalization
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
    from qtpy.QtWidgets import (
        QCheckBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
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

            self.setWindowTitle("DINO Cosine Similarity")
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
            self.status_label = QLabel("Select a checkpoint, image layer, and points layer.")

            browse_button = QPushButton("Browse")
            browse_button.clicked.connect(self._browse_checkpoint)

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

            checkpoint_row = QHBoxLayout()
            checkpoint_row.addWidget(self.checkpoint_path_edit)
            checkpoint_row.addWidget(browse_button)

            form_layout = QFormLayout()
            form_layout.addRow("Checkpoint", checkpoint_row)
            form_layout.addRow("Normalization", self.normalization_combo)
            form_layout.addRow("Image Layer", self.image_layer_combo)
            form_layout.addRow("Points Layer", self.points_layer_combo)
            form_layout.addRow("Otsu Foreground Mask", self.otsu_mask_checkbox)
            form_layout.addRow("Mask Dilation", self.mask_dilation_spinbox)

            button_layout = QVBoxLayout()
            button_layout.addWidget(refresh_button)
            button_layout.addWidget(cache_button)
            button_layout.addWidget(pca_button)
            button_layout.addWidget(selected_button)
            button_layout.addWidget(all_button)

            layout = QVBoxLayout()
            layout.addLayout(form_layout)
            layout.addLayout(button_layout)
            layout.addWidget(self.status_label)
            self.setLayout(layout)

            self.refresh_layer_choices()

        def _set_status(self, message: str) -> None:
            self.status_label.setText(message)

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
            self.embedding_cache = None
            self._set_status("Checkpoint updated.")

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
                return self.loaded_backbone

            self._set_status("Loading checkpoint...")
            loaded_backbone = load_backbone_from_checkpoint(checkpoint_path)
            combo_index = self.normalization_combo.findText(loaded_backbone.normalization_scheme)
            if combo_index >= 0:
                self.normalization_combo.setCurrentIndex(combo_index)
            self.loaded_backbone = loaded_backbone
            self.embedding_cache = None
            self._set_status(
                f"Loaded {loaded_backbone.source_branch} backbone from {checkpoint_path.name} on "
                f"{loaded_backbone.device.type} with patch size {loaded_backbone.patch_size}."
            )
            return loaded_backbone

        def _cache_embeddings(self) -> None:
            loaded_backbone = self._load_backbone()
            image_layer = self._selected_image_layer()
            self._set_status(f"Computing patch embeddings for {image_layer.name}...")

            loaded_backbone.normalization_scheme = self.normalization_combo.currentText()
            patch_embeddings, source_shape, padded_shape = compute_patch_embedding_grid(
                np.asarray(image_layer.data),
                loaded_backbone,
            )
            self.embedding_cache = EmbeddingCache(
                checkpoint_path=loaded_backbone.checkpoint_path,
                image_layer_name=image_layer.name,
                source_shape=source_shape,
                padded_shape=padded_shape,
                patch_size=loaded_backbone.patch_size,
                normalized_patch_embeddings=patch_embeddings,
            )
            patch_grid_shape = patch_embeddings.shape[:3]
            self._set_status(
                f"Cached embeddings for {image_layer.name}: patch grid {patch_grid_shape}, source shape {source_shape}."
            )

        def _ensure_embedding_cache(self) -> tuple[LoadedBackbone, EmbeddingCache]:
            loaded_backbone = self._load_backbone()
            image_layer = self._selected_image_layer()
            if (
                self.embedding_cache is None
                or self.embedding_cache.checkpoint_path != loaded_backbone.checkpoint_path
                or self.embedding_cache.image_layer_name != image_layer.name
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
            metadata: dict[str, Any],
        ) -> None:
            spatial_scale = tuple(float(v) for v in np.asarray(image_layer.scale, dtype=np.float64)[-3:])
            spatial_translate = tuple(float(v) for v in np.asarray(image_layer.translate, dtype=np.float64)[-3:])
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
            metadata: dict[str, Any],
        ) -> None:
            spatial_scale = tuple(float(v) for v in np.asarray(image_layer.scale, dtype=np.float64)[-3:])
            spatial_translate = tuple(float(v) for v in np.asarray(image_layer.translate, dtype=np.float64)[-3:])
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
            foreground_mask = compute_otsu_foreground_mask(
                np.asarray(image_layer.data),
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

            for point_index in point_indices:
                image_coords = self._point_image_coordinates(image_layer, points_layer, point_index)
                reference_patch_index = point_to_patch_index(
                    image_coords,
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
                    "reference_patch_index": list(reference_patch_index),
                }
                self._replace_or_add_similarity_layer(
                    layer_name=layer_name,
                    similarity_volume=dense_similarity,
                    image_layer=image_layer,
                    metadata=metadata,
                )

            self._set_status(f"Created {len(point_indices)} cosine similarity layer(s).")

        def _show_feature_pca(self) -> None:
            loaded_backbone, embedding_cache = self._ensure_embedding_cache()
            image_layer = self._selected_image_layer()
            self._set_status(f"Computing PCA visualization for {image_layer.name}...")

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
            }
            self._replace_or_add_pca_layer(
                layer_name=layer_name,
                pca_volume=pca_volume,
                image_layer=image_layer,
                metadata=metadata,
            )

            mask_suffix = (
                f" with Otsu mask (dilation={int(self.mask_dilation_spinbox.value())})"
                if foreground_mask is not None
                else ""
            )
            self._set_status(f"Created PCA feature layer for {image_layer.name}{mask_suffix}.")

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
    viewer.window.add_dock_widget(widget, area="right", name="DINO Cosine Similarity")
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
