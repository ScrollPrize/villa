from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from vesuvius.utils.utils import pad_or_crop_2d, pad_or_crop_3d

from .orchestrator import DatasetOrchestrator
from .slicers.chunk import ChunkPatch


@dataclass(frozen=True)
class _CoarsePatchMeta:
    start_indices_fine: Tuple[int, ...]
    window_size_fine: Tuple[int, ...]


class MultiSpacingDataset(DatasetOrchestrator):
    """Dataset orchestrator that optionally materializes a coarse-resolution patch per sample."""

    def __init__(
        self,
        mgr,
        *,
        adapter: str,
        adapter_kwargs: Optional[Dict[str, object]] = None,
        is_training: bool = True,
        logger=None,
        mesh_config: Optional[Dict[str, object]] = None,
    ) -> None:
        self._dual_spacing_enabled = bool(getattr(mgr, "dual_spacing_enabled", False))
        # Stash raw values before BaseDataset init mutates patch size / spacing state.
        self._requested_fine_spacing = getattr(mgr, "fine_spacing_um", None)
        self._requested_coarse_spacing = getattr(mgr, "coarse_spacing_um", None)
        self._requested_coarse_patch_size = getattr(mgr, "coarse_patch_size", None)
        self._dual_resample_mode = str(
            getattr(mgr, "dual_spacing_resample_mode", "trilinear")
        ).lower()
        super().__init__(
            mgr=mgr,
            adapter=adapter,
            adapter_kwargs=adapter_kwargs,
            is_training=is_training,
            logger=logger,
            mesh_config=mesh_config,
        )
        self._configure_spacing_state()

    # --------------------------------------------------------------------- #
    # BaseDataset overrides
    # --------------------------------------------------------------------- #

    def _build_chunk_data_dict(self, chunk_patch: ChunkPatch, chunk_result):
        data_dict = super()._build_chunk_data_dict(chunk_patch, chunk_result)
        if not self._dual_spacing_enabled:
            return data_dict

        coarse_tensor, meta = self._compute_coarse_patch(chunk_patch, chunk_result)
        if coarse_tensor is None:
            return data_dict

        data_dict['image_coarse'] = coarse_tensor

        # Inject metadata so downstream consumers can trace provenance.
        patch_info = dict(data_dict.get('patch_info', {}))
        patch_info['coarse_spacing_um'] = list(self._coarse_spacing.tolist())
        patch_info['coarse_patch_size'] = list(self._coarse_patch_size.tolist())
        patch_info['coarse_window_start_fine'] = [int(v) for v in meta.start_indices_fine]
        patch_info['coarse_window_size_fine'] = [int(v) for v in meta.window_size_fine]
        data_dict['patch_info'] = patch_info
        return data_dict

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _configure_spacing_state(self) -> None:
        dims = len(self.patch_size)
        self._fine_spacing = self._normalize_spacing(
            self._requested_fine_spacing, dims, default_value=1.0
        )
        self._coarse_spacing = self._normalize_spacing(
            self._requested_coarse_spacing, dims, default_value=None
        )
        if not self._dual_spacing_enabled:
            # No additional state needed when feature disabled.
            self._coarse_patch_size = np.zeros(dims, dtype=int)
            return

        if self._coarse_spacing is None:
            raise ValueError(
                "dual_spacing_enabled is true but coarse_spacing_um was not provided."
            )

        requested_size = self._normalize_patch_size(self._requested_coarse_patch_size, dims)
        if requested_size is not None:
            self._coarse_patch_size = requested_size
        else:
            # Fallback: derive coarse patch voxels that roughly match the fine patch FoV.
            self._coarse_patch_size = np.maximum(
                1,
                np.round(
                    np.asarray(self.patch_size, dtype=float)
                    * self._fine_spacing
                    / self._coarse_spacing
                ).astype(int),
            )

        if np.any(self._coarse_patch_size <= 0):
            raise ValueError("coarse_patch_size must be positive along every axis.")

    def _normalize_spacing(
        self,
        value: Optional[Sequence[float]],
        dims: int,
        *,
        default_value: Optional[float],
    ) -> Optional[np.ndarray]:
        if value is None:
            if default_value is None:
                return None
            return np.full(dims, float(default_value), dtype=float)
        if isinstance(value, (int, float)):
            return np.full(dims, float(value), dtype=float)
        seq = list(value)
        if not seq:
            if default_value is None:
                return None
            return np.full(dims, float(default_value), dtype=float)
        if len(seq) == 1:
            return np.full(dims, float(seq[0]), dtype=float)
        if len(seq) != dims:
            raise ValueError(
                f"Spacing specification must match dimensionality ({dims}); received {seq}"
            )

        return np.asarray([float(val) for val in seq], dtype=float)

    def _normalize_patch_size(
        self, size: Optional[Sequence[int]], dims: int
    ) -> Optional[np.ndarray]:
        if size is None:
            return None
        seq = list(size)
        if not seq:
            return None
        if len(seq) == 1:
            return np.full(dims, int(seq[0]), dtype=int)
        if len(seq) != dims:
            raise ValueError(
                f"coarse_patch_size must contain {dims} entries (received {seq})"
            )
        return np.asarray([int(max(1, val)) for val in seq], dtype=int)

    def _compute_coarse_patch(
        self,
        chunk_patch: ChunkPatch,
        chunk_result,
    ) -> Tuple[Optional[torch.Tensor], Optional[_CoarsePatchMeta]]:
        dims = chunk_result.image.ndim - 1  # subtract channel dim
        if dims not in (2, 3):
            raise ValueError(f"Unsupported spatial dimensionality for dual spacing: {dims}D")

        coarse_patch_size = self._coarse_patch_size[:dims]
        coarse_spacing = self._coarse_spacing[:dims]
        fine_spacing = self._fine_spacing[:dims]

        # Convert to numpy arrays for vectorised operations.
        coarse_patch_size = np.asarray(coarse_patch_size, dtype=float)
        coarse_spacing = np.asarray(coarse_spacing, dtype=float)
        fine_spacing = np.asarray(fine_spacing, dtype=float)

        window_size_fine = np.ceil(
            coarse_patch_size * coarse_spacing / fine_spacing
        ).astype(int)
        window_size_fine = np.maximum(window_size_fine, 1)

        fine_patch_size = np.asarray(self.patch_size[:dims], dtype=float)
        fine_start = np.asarray(chunk_patch.position[:dims], dtype=float)
        fine_center = fine_start + fine_patch_size / 2.0

        desired_start = np.floor(fine_center - window_size_fine / 2.0)

        coarse_window, start_indices = self._read_fine_window(
            chunk_patch.volume_index,
            desired_start.astype(int),
            window_size_fine.astype(int),
            dims=dims,
        )
        if coarse_window is None:
            return None, None

        if self.normalizer is not None:
            coarse_window = self.normalizer.run(coarse_window)
        else:
            coarse_window = coarse_window.astype(np.float32, copy=False)

        # Add channel dimension: (C, ...) with C=1
        coarse_window = np.expand_dims(coarse_window, axis=0)

        coarse_resampled = self._resample_to_coarse(
            coarse_window,
            target_size=tuple(int(v) for v in coarse_patch_size.astype(int)),
            dims=dims,
        )
        coarse_tensor = torch.from_numpy(coarse_resampled.astype(np.float32, copy=False))
        meta = _CoarsePatchMeta(
            start_indices_fine=tuple(int(v) for v in start_indices),
            window_size_fine=tuple(int(v) for v in window_size_fine.tolist()),
        )
        return coarse_tensor, meta

    def _read_fine_window(
        self,
        volume_index: int,
        start_indices: np.ndarray,
        window_size: np.ndarray,
        *,
        dims: int,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        volume = self.chunk_slicer._get_volume(volume_index)  # type: ignore[attr-defined]
        spatial_shape = np.asarray(volume.image.spatial_shape[:dims], dtype=int)

        start = start_indices.astype(int)
        size = window_size.astype(int)

        pad_before = np.zeros(dims, dtype=int)
        pad_after = np.zeros(dims, dtype=int)

        actual_start = start.copy()
        actual_end = start + size
        for axis in range(dims):
            if actual_start[axis] < 0:
                pad_before[axis] = -actual_start[axis]
                actual_start[axis] = 0
            if actual_end[axis] > spatial_shape[axis]:
                pad_after[axis] = actual_end[axis] - spatial_shape[axis]
                actual_end[axis] = spatial_shape[axis]

        actual_size = actual_end - actual_start
        if np.any(actual_size <= 0):
            # The requested window lies completely outside the volume.
            padded = np.zeros(tuple(size.tolist()), dtype=np.float32)
            return padded, start

        region = volume.image.read_window(
            tuple(int(v) for v in actual_start.tolist()),
            tuple(int(v) for v in actual_size.tolist()),
        )
        region = np.asarray(region)
        pad_width = [(int(pad_before[i]), int(pad_after[i])) for i in range(dims)]
        if any(pw != (0, 0) for pw in pad_width):
            region = np.pad(region, pad_width, mode='constant', constant_values=0)

        # Ensure we return the requested shape even if numerical drift occurred.
        desired_shape = tuple(size.tolist())
        if dims == 3:
            region = pad_or_crop_3d(region, desired_shape)
        elif dims == 2:
            region = pad_or_crop_2d(region, desired_shape)
        else:
            raise ValueError(f"Unsupported dimensionality {dims}")

        return region.astype(np.float32, copy=False), start

    def _resample_to_coarse(
        self,
        coarse_window: np.ndarray,
        *,
        target_size: Tuple[int, ...],
        dims: int,
    ) -> np.ndarray:
        if dims == 2:
            mode, align_corners = self._resolve_interpolation_mode_2d()
        else:
            mode, align_corners = self._resolve_interpolation_mode_3d()

        tensor = torch.from_numpy(coarse_window).unsqueeze(0)  # (N=1, C, ...)
        resized = F.interpolate(
            tensor,
            size=target_size,
            mode=mode,
            align_corners=align_corners,
        )
        return resized.squeeze(0).numpy()

    def _resolve_interpolation_mode_2d(self) -> Tuple[str, Optional[bool]]:
        mode = self._dual_resample_mode
        if mode in {"nearest", "bilinear", "bicubic", "area"}:
            mapping = mode
        elif mode in {"bspline", "trilinear"}:
            mapping = "bilinear"
        elif mode in {"avg"}:
            mapping = "area"
        else:
            mapping = "bilinear"
        align_corners = False if mapping in {"bilinear", "bicubic"} else None
        return mapping, align_corners

    def _resolve_interpolation_mode_3d(self) -> Tuple[str, Optional[bool]]:
        mode = self._dual_resample_mode
        if mode in {"nearest"}:
            mapping = "nearest"
            align_corners = None
        else:
            # Treat bilinear/bspline/avg/etc. as trilinear
            mapping = "trilinear"
            align_corners = False
        return mapping, align_corners
