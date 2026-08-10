"""Ink sample assembly for flat, full_3d, and full_3d_single_wrap modes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from vesuvius.ink_detection.data.augmentations import (
    build_augmentations,
    maybe_translate_crop_bbox,
)
from vesuvius.ink_detection.config import InkDataConfig
from vesuvius.ink_detection.data.geometry import (
    SURFACE_MASK_MAX_DISTANCE_LEVEL0_VOXELS,
    compute_native_crop_bbox,
    filter_support_components,
    native_tifxyz_pyramid_params,
    native_volume_downsample_factor,
    project_labels_and_supervision,
    project_surface_distance,
    read_tifxyz_on_flat_grid,
    select_flat_pixels_via_stored_resolution,
)
from vesuvius.ink_detection.data.normalization import (
    exclude_validation_voxels,
    normalize_image,
)
from vesuvius.ink_detection.data.patch_cache import (
    load_patch_cache,
    patch_cache_path,
    save_patch_cache,
)
from vesuvius.ink_detection.data.patch_finding_default import (
    find_segment_patches as find_default_patches,
    find_segment_unlabeled_patches,
)
from vesuvius.ink_detection.data.patch_finding_subtiling import (
    find_segment_patches as find_subtiling_patches,
)
from vesuvius.ink_detection.data.segment import gather_segments
from vesuvius.ink_detection.types import Patch, Segment
from vesuvius.ink_detection.volume_io import open_volume, read_bbox_with_padding


def flat_z_window_bbox(
    bbox_zyx: tuple[int, int, int, int, int, int],
    *,
    config: InkDataConfig,
    do_augmentations: bool,
    is_validation: bool,
    rng=random,
) -> tuple[tuple[int, int, int, int, int, int], int]:
    """Select the canonical or randomly shifted real-layer flat Z window."""
    z0, y0, x0, z1, y1, x1 = bbox_zyx
    jitter = config.jitter
    base_depth = z1 - z0
    window_depth = base_depth if jitter.window_depth is None else jitter.window_depth
    if window_depth <= 0 or window_depth > base_depth:
        raise ValueError(
            f"flat_z_window_jitter.window_depth must be in [1, {base_depth}], got {window_depth}"
        )
    reduction = base_depth - window_depth
    if reduction % 2:
        raise ValueError(
            "flat_z_window_jitter requires a symmetric canonical crop; "
            f"base depth {base_depth} minus window depth {window_depth} must be even"
        )
    trim = reduction // 2
    canonical = z0 + trim, y0, x0, z1 - trim, y1, x1
    if (
        not jitter.enabled
        or jitter.max_offset == 0
        or not do_augmentations
        or is_validation
        or rng.random() >= jitter.probability
    ):
        return canonical, 0
    offset = rng.randint(-jitter.max_offset, jitter.max_offset)
    z0, y0, x0, z1, y1, x1 = canonical
    return (z0 + offset, y0, x0, z1 + offset, y1, x1), offset


def _require_real_z(volume, bbox_zyx, *, name: str) -> None:
    z0, _, _, z1, _, _ = bbox_zyx
    shape = tuple(int(value) for value in volume.shape)
    if z0 < 0 or z1 > shape[0]:
        raise ValueError(
            f"{name} bbox {bbox_zyx!r} exceeds array shape {shape!r}; "
            "real-layer Z jitter forbids Z padding"
        )


def _read_flat_surface(volume, *, y0: int, y1: int, x0: int, x1: int):
    surface = int(volume.shape[0] // 2)
    patch, _ = read_bbox_with_padding(
        volume, (surface, y0, x0, surface + 1, y1, x1), fill_value=0
    )
    return patch[0]


class InkDataset(Dataset):
    """Assemble samples using resolved data settings and an optional patch list."""

    def __init__(
        self,
        config: InkDataConfig,
        *,
        do_augmentations: bool = True,
        patches: list[Patch] | None = None,
        segments: list[Segment] | None = None,
    ) -> None:
        if not isinstance(config, InkDataConfig):
            raise TypeError("InkDataset requires a resolved InkDataConfig")
        self.config = config
        self.patch_size = config.patch_size
        self.mode = config.mode
        self.do_augmentations = bool(do_augmentations)
        self.input_channels = 2 if self.mode == "full_3d_single_wrap" else 1
        self._zarr_cache: dict[tuple, object] = {}
        self._tifxyz_cache: dict[str, object] = {}
        self._stored_resolution_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.augmentations = None
        if self.do_augmentations and not config.augmentation.disabled:
            self.augmentations = build_augmentations(
                config.augmentation.preset,
                config.patch_size,
                rotation_axes=config.augmentation.rotation_axes,
            )

        if patches is None:
            self.segments = list(gather_segments(config))
            self.patches = self._discover_or_load_patches()
        else:
            self.patches = list(patches)
            if segments is not None:
                self.segments = list(segments)
            else:
                self.segments = []
                seen_segment_keys = set()
                for patch in self.patches:
                    key = patch.segment.cache_key
                    if key not in seen_segment_keys:
                        seen_segment_keys.add(key)
                        self.segments.append(patch.segment)
        self.training_patches = [
            patch for patch in self.patches if not patch.is_validation
        ]
        self.validation_patches = [patch for patch in self.patches if patch.is_validation]
        self.unlabeled_patches = [patch for patch in self.patches if patch.is_unlabeled]
        self._segments_by_volume: dict[tuple[int, str, int], list[Segment]] = {}
        for segment in self.segments:
            key = segment.dataset_idx, str(segment.image_volume), segment.scale
            self._segments_by_volume.setdefault(key, []).append(segment)

    def _open(self, path: str | Path, resolution: int):
        key = (
            str(path),
            int(resolution),
            None if self.config.volume_auth_json is None else str(self.config.volume_auth_json),
            None if self.config.volume_cache_dir is None else str(self.config.volume_cache_dir),
            self.config.volume_cache_max_gb,
        )
        volume = self._zarr_cache.get(key)
        if volume is None:
            volume = open_volume(
                path,
                resolution,
                self.config.volume_auth_json,
                cache_dir=self.config.volume_cache_dir,
                cache_max_gb=self.config.volume_cache_max_gb,
            )
            self._zarr_cache[key] = volume
        return volume

    def _discover_or_load_patches(self) -> list[Patch]:
        path = patch_cache_path(self.config)
        if path.exists():
            cached = load_patch_cache(
                path, config=self.config, segments=self.segments
            )
            if cached is not None:
                return cached

        def process(segment: Segment):
            try:
                if self.config.discovery_mode == "unlabeled":
                    return find_segment_unlabeled_patches(segment, self._open)
                if self.config.patch_finding.kind == "subtiling":
                    return find_subtiling_patches(segment, self._open)
                return find_default_patches(segment, self._open)
            except Exception as exc:
                raise RuntimeError(
                    "Failed finding patches for "
                    f"dataset_idx={segment.dataset_idx!r}, "
                    f"segment={segment.segment_relpath!r}, "
                    f"image_volume={segment.image_volume!r}, "
                    f"volume_scale={segment.scale!r}, "
                    f"supervision_mask={segment.supervision_mask!r}, "
                    f"inklabels={segment.inklabels!r}, "
                    f"validation_mask={segment.validation_mask!r}"
                ) from exc

        training: list[Patch] = []
        validation: list[Patch] = []
        with ThreadPoolExecutor(
            max_workers=self.config.dataloader_workers
        ) as executor:
            for segment_training, segment_validation in executor.map(
                process, self.segments
            ):
                training.extend(segment_training)
                validation.extend(segment_validation)
        patches = training + validation
        save_patch_cache(path, patches)
        return patches

    def _tifxyz(self, segment: Segment):
        key = str(segment.segment_dir)
        value = self._tifxyz_cache.get(key)
        if value is None:
            import vesuvius.tifxyz as tifxyz

            value = tifxyz.read_tifxyz(segment.segment_dir)
            value.use_full_resolution()
            self._tifxyz_cache[key] = value
        return value

    def _coarse_positions(self, segment: Segment, patch_tifxyz):
        key = str(segment.segment_dir)
        value = self._stored_resolution_cache.get(key)
        if value is None:
            positions = np.asarray(
                patch_tifxyz.get_zyxs(stored_resolution=True), dtype=np.float32
            )
            valid = np.isfinite(positions).all(axis=-1)
            valid &= (positions >= 0).all(axis=-1)
            value = positions, valid
            self._stored_resolution_cache[key] = value
        return value

    def __len__(self) -> int:
        return len(self.patches)

    def _normal_thicknesses(self, resolution: int) -> tuple[float, float]:
        factor = native_volume_downsample_factor(resolution)
        label = self.config.full_3d.label_projection_half_thickness
        background = self.config.full_3d.background_projection_half_thickness
        return float(label) / factor, float(background) / factor

    def _flat_sample(self, patch: Patch) -> dict[str, torch.Tensor]:
        bbox, _ = flat_z_window_bbox(
            patch.bbox,
            config=self.config,
            do_augmentations=self.do_augmentations,
            is_validation=patch.is_validation,
        )
        image_volume = self._open(patch.image_volume, patch.segment.scale)
        if self.config.jitter.enabled:
            _require_real_z(image_volume, bbox, name="image")
        image, valid_slices = read_bbox_with_padding(image_volume, bbox, fill_value=0)
        if patch.is_unlabeled:
            labels = np.zeros(self.patch_size, dtype=np.uint8)
            supervision = np.zeros(self.patch_size, dtype=np.uint8)
        else:
            supervision_volume = self._open(patch.supervision_mask, patch.segment.scale)
            labels_volume = self._open(patch.inklabels, patch.segment.scale)
            if self.config.jitter.enabled:
                _require_real_z(supervision_volume, bbox, name="supervision")
                _require_real_z(labels_volume, bbox, name="inklabels")
            supervision, _ = read_bbox_with_padding(
                supervision_volume, bbox, fill_value=0
            )
            if not patch.is_validation and patch.segment.validation_mask is not None:
                validation_volume = self._open(
                    patch.segment.validation_mask, patch.segment.scale
                )
                if self.config.jitter.enabled:
                    _require_real_z(validation_volume, bbox, name="validation")
                validation, _ = read_bbox_with_padding(
                    validation_volume, bbox, fill_value=0
                )
                supervision = exclude_validation_voxels(
                    supervision, validation, is_validation_patch=False
                )
            labels, _ = read_bbox_with_padding(labels_volume, bbox, fill_value=0)
        return self._tensor_sample(
            patch, image, valid_slices, labels, supervision, surface_mask=None
        )

    def _native_sample(self, patch: Patch) -> dict[str, torch.Tensor]:
        z0, y0, x0, z1, y1, x1 = patch.bbox
        image_volume = self._open(patch.image_volume, patch.segment.scale)
        supervision_volume = self._open(patch.supervision_mask, patch.segment.scale)
        labels_volume = self._open(patch.inklabels, patch.segment.scale)
        validation_volume = (
            None
            if patch.is_validation or patch.segment.validation_mask is None
            else self._open(patch.segment.validation_mask, patch.segment.scale)
        )
        patch_tifxyz = self._tifxyz(patch.segment)
        coarse_positions, coarse_valid = self._coarse_positions(
            patch.segment, patch_tifxyz
        )
        stride, coordinate_scale, coarse_pad = native_tifxyz_pyramid_params(
            patch.segment.scale
        )
        patch_positions, patch_valid = read_tifxyz_on_flat_grid(
            patch_tifxyz,
            y0=y0,
            y1=y1,
            x0=x0,
            x1=x1,
            flat_grid_stride=stride,
            native_coordinate_scale=coordinate_scale,
        )
        crop_bbox = compute_native_crop_bbox(
            patch_positions, patch_valid, self.patch_size
        )
        patch_supervision = _read_flat_surface(
            supervision_volume, y0=y0, y1=y1, x0=x0, x1=x1
        )
        if validation_volume is not None:
            patch_validation = _read_flat_surface(
                validation_volume, y0=y0, y1=y1, x0=x0, x1=x1
            )
            patch_supervision = exclude_validation_voxels(
                patch_supervision, patch_validation
            )
        if self.do_augmentations:
            crop_bbox = maybe_translate_crop_bbox(
                crop_bbox, patch_positions, patch_valid, patch_supervision
            )
        (
            support_bbox,
            support_positions,
            support_valid,
            _,
            _,
            _,
        ) = select_flat_pixels_via_stored_resolution(
            patch_tifxyz,
            crop_bbox,
            coarse_native_pad=coarse_pad,
            coarse_positions_zyx=coarse_positions,
            coarse_valid=coarse_valid,
            native_coordinate_scale=coordinate_scale,
            flat_grid_stride=stride,
            return_halo=True,
        )
        support_y0, support_y1, support_x0, support_x1 = support_bbox
        support_supervision = _read_flat_surface(
            supervision_volume,
            y0=support_y0,
            y1=support_y1,
            x0=support_x0,
            x1=support_x1,
        )
        if validation_volume is not None:
            support_validation = _read_flat_surface(
                validation_volume,
                y0=support_y0,
                y1=support_y1,
                x0=support_x0,
                x1=support_x1,
            )
            support_supervision = exclude_validation_voxels(
                support_supervision, support_validation
            )
        support_labels = _read_flat_surface(
            labels_volume,
            y0=support_y0,
            y1=support_y1,
            x0=support_x0,
            x1=support_x1,
        )
        (
            support_bbox,
            support_positions,
            support_valid,
            support_labels,
            support_supervision,
        ) = filter_support_components(
            support_bbox_yx=support_bbox,
            positions_zyx=support_positions,
            valid_mask=support_valid,
            inklabels_flat=support_labels,
            supervision_flat=support_supervision,
            crop_bbox_zyx=crop_bbox,
            patch_bbox_zyx=patch.bbox,
            max_supervision_grid_distance=self.config.full_3d.support_grid_max_distance,
        )
        support_shape = tuple(int(value) for value in support_valid.shape)
        support_limits = self.patch_size[1] * 4, self.patch_size[2] * 4
        if support_shape[0] > support_limits[0] or support_shape[1] > support_limits[1]:
            raise ValueError(
                f"Oversized native support grid {support_shape!r} exceeded "
                f"side limits {support_limits!r}"
            )
        support_y0, support_y1, support_x0, support_x1 = support_bbox
        nx, ny, nz = patch_tifxyz.get_normals(
            support_y0 * stride,
            support_y1 * stride,
            support_x0 * stride,
            support_x1 * stride,
        )
        normals = np.stack([nz, ny, nx], axis=-1)[::stride, ::stride].astype(
            np.float32, copy=False
        )
        image, valid_slices = read_bbox_with_padding(
            image_volume, crop_bbox, fill_value=0
        )
        label_thickness, background_thickness = self._normal_thicknesses(
            patch.segment.scale
        )
        labels, supervision = project_labels_and_supervision(
            positions_zyx=support_positions,
            valid_mask=support_valid,
            inklabels_flat=support_labels,
            supervision_flat=support_supervision,
            crop_bbox_zyx=crop_bbox,
            normals_zyx=normals,
            label_half_thickness=label_thickness,
            background_half_thickness=background_thickness,
        )
        if self.mode == "full_3d":
            labels, supervision = self._merge_intersecting(
                patch, crop_bbox, labels, supervision
            )
        surface_mask = None
        if self.mode == "full_3d_single_wrap":
            surface_mask = project_surface_distance(
                support_positions,
                support_valid,
                crop_bbox,
                max_distance_voxels=(
                    SURFACE_MASK_MAX_DISTANCE_LEVEL0_VOXELS / float(stride)
                ),
            )
        return self._tensor_sample(
            patch, image, valid_slices, labels, supervision, surface_mask
        )

    def _merge_intersecting(
        self,
        patch: Patch,
        crop_bbox,
        labels: np.ndarray,
        supervision: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        key = patch.segment.dataset_idx, str(patch.image_volume), patch.segment.scale
        for segment in self._segments_by_volume.get(key, ()):
            if segment.segment_relpath == patch.segment.segment_relpath:
                continue
            if segment.inklabels is None or segment.supervision_mask is None:
                continue
            patch_tifxyz = self._tifxyz(segment)
            coarse_positions, coarse_valid = self._coarse_positions(
                segment, patch_tifxyz
            )
            stride, coordinate_scale, coarse_pad = native_tifxyz_pyramid_params(
                segment.scale
            )
            selection = select_flat_pixels_via_stored_resolution(
                patch_tifxyz,
                crop_bbox,
                coarse_native_pad=coarse_pad,
                coarse_positions_zyx=coarse_positions,
                coarse_valid=coarse_valid,
                native_coordinate_scale=coordinate_scale,
                flat_grid_stride=stride,
                required=False,
            )
            if selection is None:
                continue
            support_bbox, positions, valid = selection
            y0, y1, x0, x1 = support_bbox
            active_supervision_path = (
                segment.validation_mask
                if patch.is_validation and segment.validation_mask is not None
                else segment.supervision_mask
            )
            other_supervision = _read_flat_surface(
                self._open(active_supervision_path, segment.scale),
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
            )
            if not patch.is_validation and segment.validation_mask is not None:
                other_validation = _read_flat_surface(
                    self._open(segment.validation_mask, segment.scale),
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                )
                other_supervision = exclude_validation_voxels(
                    other_supervision, other_validation
                )
            valid &= np.asarray(other_supervision) > 0
            if not np.any(valid):
                continue
            other_labels = _read_flat_surface(
                self._open(segment.inklabels, segment.scale),
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
            )
            nx, ny, nz = patch_tifxyz.get_normals(
                y0 * stride, y1 * stride, x0 * stride, x1 * stride
            )
            normals = np.stack([nz, ny, nx], axis=-1)[::stride, ::stride]
            label_thickness, background_thickness = self._normal_thicknesses(
                segment.scale
            )
            other_labels, other_supervision = project_labels_and_supervision(
                positions_zyx=positions,
                valid_mask=valid,
                inklabels_flat=other_labels,
                supervision_flat=other_supervision,
                crop_bbox_zyx=crop_bbox,
                normals_zyx=normals,
                label_half_thickness=label_thickness,
                background_half_thickness=background_thickness,
            )
            np.maximum(labels, other_labels, out=labels)
            np.maximum(supervision, other_supervision, out=supervision)
        return labels, supervision

    def _tensor_sample(
        self,
        patch: Patch,
        image: np.ndarray,
        image_valid_slices,
        labels: np.ndarray,
        supervision: np.ndarray,
        surface_mask: np.ndarray | None,
    ) -> dict[str, torch.Tensor]:
        image = image.astype(np.float32, copy=False)
        if image_valid_slices is not None:
            image[image_valid_slices] = normalize_image(
                image[image_valid_slices], self.config.normalization
            )
        for name, value in (
            ("image", image),
            ("inklabels", labels),
            ("supervision_mask", supervision),
        ):
            if tuple(value.shape) != self.patch_size:
                raise AssertionError(
                    f"{name} crop shape {tuple(value.shape)} does not match "
                    f"requested patch size {self.patch_size}"
                )
        data = {
            "image": torch.from_numpy(image).float().unsqueeze(0),
            "inklabels": torch.from_numpy(np.asarray(labels)).float().unsqueeze(0),
            "supervision_mask": torch.from_numpy(np.asarray(supervision)).float().unsqueeze(0),
        }
        if surface_mask is not None:
            data["surface_mask"] = torch.from_numpy(surface_mask).float().unsqueeze(0)
        if self.augmentations is not None:
            augmentation_data = data
            if self.mode == "full_3d_single_wrap" and "surface_mask" in data:
                augmentation_data = dict(data)
                augmentation_data["regression_keys"] = ["surface_mask"]
            data = self.augmentations(**augmentation_data)
        data["is_unlabeled"] = torch.tensor(patch.is_unlabeled, dtype=torch.bool)
        return data

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        requested_index = int(index)
        current_index = requested_index
        while True:
            patch = self.patches[current_index]
            if self.mode == "flat":
                return self._flat_sample(patch)
            try:
                return self._native_sample(patch)
            except ValueError as exc:
                resample_error = str(exc) == "No valid tifxyz points found for patch" or str(
                    exc
                ).startswith("Oversized native support grid")
                if not resample_error or len(self.patches) <= 1:
                    raise
                rng = random.Random(self.config.seed + current_index * 7919)
                replacement = current_index
                while replacement == current_index:
                    replacement = rng.randrange(len(self.patches))
                warnings.warn(
                    "Native patch could not produce an admissible crop for "
                    f"requested idx {requested_index}, patch idx {current_index}: "
                    f"{exc}; resampling idx {replacement}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                current_index = replacement
