from __future__ import annotations

import glob
import heapq
import hashlib
import itertools
import json
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace, field
from .embedding import (EmbeddingConfig, EmbeddingGeometry, build_eligibility,
                        InvalidNormalSampleError)
from .arc_sampling import FiberLocation, ScrollArcSampler
from .training_spans import training_fiber_spans
from .refinement import RefinementConfig, RefinementObservation, active_store, arclength, interpolate, export_metadata, resample
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vesuvius.data.affine import read_transform_json
from vesuvius.neural_tracing.datasets.common import open_zarr
from vesuvius.neural_tracing.fiber_trace.dataset import (
    _load_lasagna_volume,
    _omezarr_level_shape,
    _resolve_config_relative_path,
    _validate_shape,
    _volume_shape_zyx,
)
from vesuvius.neural_tracing.fiber_trace_2d.fiber_json import (
    Vc3dFiber,
    load_fiber_file,
)
from vesuvius.neural_tracing.fiber_trace_2d.loader import (
    _download_prefetch_request,
    _empty_marker_path,
    _existing_data_path,
)
from vesuvius.neural_tracing.fiber_trace_2d.loader_support import ZarrChunkRequest
from vesuvius.neural_tracing.fiber_trace_2d.sampling import (
    CoordinateSampler,
    NumpyZarrCoordinateSampler,
    make_coordinate_sampler,
)
from vesuvius.neural_tracing.fiber_trace_2d.strip_geometry import (
    control_point_line_index,
)

_REMOTE_PREFIXES = ("http://", "https://", "s3://")
_TARGET_MODE_CP_ONLY = 0
_TARGET_MODE_DENSE_LINE = 1
DEFAULT_VOLUME_CACHE_MEMORY_MIB = 512.0
_FULL_VALID_MASK_CACHE: dict[tuple[int, ...], np.ndarray] = {}


@dataclass(frozen=True)
class FiberTrace3DConfig:
    datasets: tuple[dict[str, Any], ...]
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    refinement_split: str = "train"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    batch_size: int = 1
    patch_shape_zyx: tuple[int, int, int] = (192, 192, 192)
    seed: int = 1
    cp_margin_voxels: int = 4
    presence_radius_voxels: float = 2.0
    presence_target_mode: str = "binary"
    presence_gaussian_width_ratio: float = 4.0
    presence_gaussian_truncate_sigma: float = 3.0
    presence_sideband_weight: float = 0.0
    presence_sideband_half_width_base: float = 64.0
    presence_sideband_half_height_base: float = 16.0
    presence_negative_edge_margin_voxels: int | None = None
    image_normalization: str = "zscore"
    augment_enabled: bool = True
    augment_shift_zyx: tuple[float, float, float] = (0.0, 0.0, 0.0)
    augment_rotation_degrees: float = 0.0
    augment_scale_min: float = 1.0
    augment_scale_max: float = 1.0
    augment_flip_probability: float = 0.0
    augment_brightness: float = 0.0
    augment_contrast_min: float = 1.0
    augment_contrast_max: float = 1.0
    augment_gamma_min: float = 1.0
    augment_gamma_max: float = 1.0
    augment_noise_std: float = 0.0
    augment_blur_sigma: float = 0.0
    augment_smooth_displacement_mode: str = "none"
    augment_smooth_displacement_amplitude_zyx: tuple[float, float, float] = (0.0, 0.0, 0.0)
    augment_smooth_displacement_control_spacing_zyx: tuple[float, float, float] = (16.0, 16.0, 16.0)
    augment_smooth_displacement_probability: float = 0.0
    augment_anisotropic_blur_probability: float = 0.0
    augment_anisotropic_blur_sigma_along: float = 0.0
    augment_anisotropic_blur_sigma_across: float = 0.0
    augment_anisotropic_blur_orientation: str = "fiber"
    augment_anisotropic_blur_roll_degrees: float = 0.0
    round_source_to_chunk_boundaries: bool = True
    geometry_tile_depth: int = 0
    geometry_backend: str = "auto"
    value_augmentation_backend: str = "auto"
    prefetch_workers: int = 16
    prefetch_sampler_workers: int = 2
    volume_cache_dir: str | None = None
    volume_cache_memory_mib: float | None = DEFAULT_VOLUME_CACHE_MEMORY_MIB
    volume_io_threads: int | None = None
    volume_decode_threads: int | None = None
    volume_sample_threads: int | None = None
    volume_chunk_locality_fast_path: bool = True
    volume_collect_locality_stats: bool = False
    volume_cache_offline: bool = False
    volume_cache_retry_seconds: float = 0.0
    config_dir: Path | None = None
    sampling_strategy: str = "control_points"

    @property
    def volume_cache_memory_bytes(self) -> int | None:
        if self.volume_cache_memory_mib is None:
            return None
        return int(float(self.volume_cache_memory_mib) * 1024.0 * 1024.0)


@dataclass(frozen=True)
class _Record:
    fiber: Vc3dFiber
    volume: Any
    sampler: CoordinateSampler
    volume_path: str
    volume_scale: int
    volume_spacing_base: float
    base_shape_zyx: tuple[int, int, int]
    fiber_identity: str
    dataset_config: dict[str, Any]
    label_breaks: tuple[int, ...] = ()


@dataclass(frozen=True)
class _SmoothDisplacement3DParams:
    mode: str
    amplitude_zyx: np.ndarray
    control_spacing_zyx: np.ndarray
    control_min_zyx: np.ndarray
    controls: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class _Augment3DParams:
    stream_index: int
    cp_local_zyx: np.ndarray
    cp_volume_zyx: np.ndarray
    source_to_output_zyx: np.ndarray
    output_to_source_zyx: np.ndarray
    brightness: float
    contrast: float
    gamma: float
    noise_std: float
    blur_sigma: float
    smooth: _SmoothDisplacement3DParams | None
    anisotropic_blur_sigma_along: float
    anisotropic_blur_sigma_across: float
    anisotropic_blur_axis_zyx: np.ndarray | None
    value_seed_index: int | None = None


@dataclass(frozen=True)
class _DirectGaussianKernel3D:
    dense_zyx: np.ndarray


@dataclass(frozen=True)
class _BlurKernels3D:
    isotropic_1d: np.ndarray | None
    directional_zyx: _DirectGaussianKernel3D | None


@dataclass(frozen=True)
class _GeometryMaps3D:
    backward_source_zyx: torch.Tensor | None
    params: _Augment3DParams
    backward_source_base_zyx: np.ndarray | None = None
    valid_base: np.ndarray | None = None


@dataclass
class _PrefetchCounters3D:
    samples_done: int = 0
    samples_skipped: int = 0
    unique_chunks_seen: int = 0
    queued_for_download: int = 0
    download_done: int = 0
    cache_hits: int = 0
    known_missing: int = 0
    downloaded: int = 0
    newly_missing: int = 0
    download_errors: int = 0
    bytes_downloaded: int = 0
    valid_voxels: int = 0
    first_error: str = ""
    first_sample_skip: str = ""
    queued_download_futures: int = 0
    max_exclusive_sample_index: int = 0


@dataclass(frozen=True)
class FiberTrace3DSample:
    stream_index: int
    data_index: int
    record_index: int
    fiber_path: str
    control_point_index: int
    volume: torch.Tensor
    valid_mask: torch.Tensor | None
    cp_local_zyx: torch.Tensor
    crop_origin_zyx: torch.Tensor
    target_mode: int
    target_segment_starts_zyx: torch.Tensor
    target_segment_ends_zyx: torch.Tensor
    target_segment_bbox_lo_zyx: torch.Tensor
    target_segment_bbox_hi_zyx: torch.Tensor
    target_tangent_zyx: torch.Tensor
    target_segment_normals_zyx: torch.Tensor | None = None
    volume_spacing_base: float = 1.0
    presence_target_mode: str = "binary"
    presence_sideband_weight: float = 0.0
    profile_timings_ms: dict[str, float] | None = None
    embedding_geometry: EmbeddingGeometry | None = None
    refinement_geometry: RefinementObservation | None = None


@dataclass(frozen=True)
class FiberTrace3DRawSample:
    """Value-unaugmented sample plus deterministic parameters for stage two."""

    sample: FiberTrace3DSample
    augmentation: _Augment3DParams
    load_wall_ms: float
    load_cpu_ms: float
    blur_kernels: _BlurKernels3D | None = None


@dataclass(frozen=True)
class _SharedFiberTrace3DPayload:
    slot: int
    generation: int
    value: FiberTrace3DSample | FiberTrace3DRawSample


class FiberTrace3DSharedTensorSlots:
    """Bounded shared storage for large per-sample volume tensors."""

    def __init__(
        self,
        *,
        start_index: int,
        slot_count: int,
        patch_shape_zyx: tuple[int, int, int],
        stream_index_map: Any = None,
        volume_dtype: torch.dtype = torch.float32,
        embedding_stride: int = 0,
    ) -> None:
        if int(slot_count) <= 0:
            raise ValueError("slot_count must be positive")
        if type(embedding_stride) is not int or embedding_stride < 0:
            raise ValueError("embedding_stride must be a nonnegative integer")
        self.start_index = int(start_index)
        self.slot_count = int(slot_count)
        self.stream_index_map = stream_index_map
        shape = (self.slot_count, 1, *(int(v) for v in patch_shape_zyx))
        self.volumes = torch.empty(shape, dtype=volume_dtype).share_memory_()
        self.embedding_eligibility = None
        if embedding_stride:
            embedding_shape = (self.slot_count, *((int(v) + embedding_stride - 1) // embedding_stride for v in patch_shape_zyx))
            self.embedding_eligibility = torch.empty(embedding_shape, dtype=torch.bool).share_memory_()
        self.generations = torch.full(
            (self.slot_count,), -1, dtype=torch.int64
        ).share_memory_()

    def _location(self, index: int) -> tuple[int, int]:
        offset = int(index) - self.start_index
        if offset < 0:
            raise RuntimeError(f"shared-slot index precedes start: {index}")
        return offset % self.slot_count, offset // self.slot_count

    def encode(
        self,
        index: int,
        value: FiberTrace3DSample | FiberTrace3DRawSample,
        *,
        final: bool,
    ) -> _SharedFiberTrace3DPayload:
        del final
        sample = value.sample if isinstance(value, FiberTrace3DRawSample) else value
        slot, generation = self._location(index)
        expected_stream_index = (
            int(index)
            if self.stream_index_map is None
            else int(self.stream_index_map(int(index)))
        )
        if int(sample.stream_index) != expected_stream_index:
            raise RuntimeError("shared-slot sample/index mismatch")
        self.volumes[slot].copy_(sample.volume)
        embedding_geometry = sample.embedding_geometry
        if embedding_geometry is not None:
            if self.embedding_eligibility is None:
                raise ValueError("embedding samples require an embedding-enabled shared-slot bank")
            if embedding_geometry.eligible is None or embedding_geometry.eligible.shape != self.embedding_eligibility[slot].shape:
                raise ValueError("embedding eligibility does not match the shared-slot shape")
            self.embedding_eligibility[slot].copy_(embedding_geometry.eligible)
            embedding_geometry = replace(embedding_geometry, eligible=None)
        self.generations[slot] = generation
        tensor_metadata = {
            name: np.asarray(getattr(sample, name).detach().cpu().numpy())
            for name in (
                "cp_local_zyx",
                "crop_origin_zyx",
                "target_segment_starts_zyx",
                "target_segment_ends_zyx",
                "target_segment_bbox_lo_zyx",
                "target_segment_bbox_hi_zyx",
                "target_tangent_zyx",
            )
        }
        if sample.target_segment_normals_zyx is not None:
            tensor_metadata["target_segment_normals_zyx"] = np.asarray(
                sample.target_segment_normals_zyx.detach().cpu().numpy()
            )
        metadata_sample = replace(
            sample,
            embedding_geometry=embedding_geometry,
            volume=None,
            valid_mask=None,
            **tensor_metadata,
        )
        metadata_value = (
            replace(value, sample=metadata_sample)
            if isinstance(value, FiberTrace3DRawSample)
            else metadata_sample
        )
        return _SharedFiberTrace3DPayload(slot, generation, metadata_value)

    def decode(
        self,
        index: int,
        payload: _SharedFiberTrace3DPayload,
        *,
        final: bool,
    ) -> FiberTrace3DSample | FiberTrace3DRawSample:
        del final
        if not isinstance(payload, _SharedFiberTrace3DPayload):
            raise RuntimeError("shared-slot transport received an invalid descriptor")
        slot, generation = self._location(index)
        if payload.slot != slot or payload.generation != generation:
            raise RuntimeError("shared-slot descriptor location mismatch")
        observed = int(self.generations[slot].item())
        if observed != generation:
            raise RuntimeError(
                f"shared-slot generation mismatch for index {index}: "
                f"expected {generation}, observed {observed}"
            )
        value = payload.value
        sample = value.sample if isinstance(value, FiberTrace3DRawSample) else value
        if sample.embedding_geometry is not None and self.embedding_eligibility is None:
            raise ValueError("embedding descriptor requires an embedding-enabled shared-slot bank")
        tensor_metadata = {
            name: torch.from_numpy(np.asarray(getattr(sample, name)))
            for name in (
                "cp_local_zyx",
                "crop_origin_zyx",
                "target_segment_starts_zyx",
                "target_segment_ends_zyx",
                "target_segment_bbox_lo_zyx",
                "target_segment_bbox_hi_zyx",
                "target_tangent_zyx",
            )
        }
        if sample.target_segment_normals_zyx is not None:
            tensor_metadata["target_segment_normals_zyx"] = torch.from_numpy(
                np.asarray(sample.target_segment_normals_zyx)
            )
        restored = replace(
            sample,
            embedding_geometry=None if sample.embedding_geometry is None else replace(
                sample.embedding_geometry, eligible=self.embedding_eligibility[slot]),
            volume=self.volumes[slot],
            valid_mask=None,
            **tensor_metadata,
        )
        return (
            replace(value, sample=restored)
            if isinstance(value, FiberTrace3DRawSample)
            else restored
        )

    def batch_tensors(self, indices: list[int], *, allow_wrap: bool = False) -> torch.Tensor:
        if not indices:
            raise ValueError("shared-slot batch must not be empty")
        locations = [self._location(index) for index in indices]
        slots = [slot for slot, _generation in locations]
        expected = list(range(slots[0], slots[0] + len(slots)))
        wrapped = allow_wrap and len(slots) <= self.slot_count and slots == [i % self.slot_count for i in expected]
        if slots != expected and not wrapped:
            raise RuntimeError("shared-slot batch crosses or is not contiguous")
        for index, (slot, generation) in zip(indices, locations, strict=True):
            observed = int(self.generations[slot].item())
            if observed != generation:
                raise RuntimeError(
                    f"shared-slot batch generation mismatch for index {index}: "
                    f"expected {generation}, observed {observed}"
                )
        end = slots[0] + len(slots)
        if end > self.slot_count:
            # Only partial/variable batches can straddle the ring. The common
            # aligned path stays zero-copy; never read another sample's slot.
            return torch.cat((self.volumes[slots[0]:], self.volumes[:end - self.slot_count]), dim=0)
        return self.volumes[slots[0] : end]


@dataclass(frozen=True)
class FiberTrace3DBatch:
    volume: torch.Tensor
    valid_mask: torch.Tensor | None
    cp_local_zyx: torch.Tensor
    crop_origin_zyx: torch.Tensor
    stream_indices: torch.Tensor
    data_indices: torch.Tensor
    record_indices: torch.Tensor
    control_point_indices: torch.Tensor
    fiber_paths: tuple[str, ...]
    target_modes: torch.Tensor
    target_segment_offsets: torch.Tensor
    target_segment_counts: torch.Tensor
    target_segment_starts_zyx: torch.Tensor
    target_segment_ends_zyx: torch.Tensor
    target_segment_bbox_lo_zyx: torch.Tensor
    target_segment_bbox_hi_zyx: torch.Tensor
    target_tangent_zyx: torch.Tensor
    target_segment_normals_zyx: torch.Tensor | None = None
    volume_spacing_base: torch.Tensor | None = None
    presence_target_mode: str = "binary"
    presence_sideband_weight: float = 0.0
    direction_target: torch.Tensor | None = None
    direction_weight: torch.Tensor | None = None
    direction_mask: torch.Tensor | None = None
    direction_indices_bzyx: torch.Tensor | None = None
    direction_target_sparse: torch.Tensor | None = None
    direction_weight_sparse: torch.Tensor | None = None
    direction_tangent_sparse_zyx: torch.Tensor | None = None
    presence_target: torch.Tensor | None = None
    presence_sideband_mask: torch.Tensor | None = None
    presence_mask: torch.Tensor | None = None
    deferred_augmentation_params: tuple[_Augment3DParams, ...] | None = None
    deferred_blur_kernels: tuple[_BlurKernels3D | None, ...] | None = None
    profile_timings_ms: dict[str, float] | None = None
    embedding_geometry: tuple[EmbeddingGeometry | None, ...] | None = None
    refinement_geometry: tuple[RefinementObservation | None, ...] | None = None

    # Local effective-batch target counts, prepared before model forward.
    normalization_counts: torch.Tensor | None = None

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "FiberTrace3DBatch":
        return FiberTrace3DBatch(
            presence_target_mode=self.presence_target_mode,
            presence_sideband_weight=self.presence_sideband_weight,
            refinement_geometry=self.refinement_geometry,
            normalization_counts=None if self.normalization_counts is None else self.normalization_counts.to(device, non_blocking=non_blocking),
            embedding_geometry=None if self.embedding_geometry is None else tuple(
                None if item is None else replace(item, eligible=item.eligible.to(device, non_blocking=non_blocking))
                for item in self.embedding_geometry),
            volume=self.volume.to(device, non_blocking=non_blocking),
            valid_mask=None
            if self.valid_mask is None
            else self.valid_mask.to(device, non_blocking=non_blocking),
            cp_local_zyx=self.cp_local_zyx.to(device, non_blocking=non_blocking),
            crop_origin_zyx=self.crop_origin_zyx.to(device, non_blocking=non_blocking),
            stream_indices=self.stream_indices.to(device, non_blocking=non_blocking),
            data_indices=self.data_indices.to(device, non_blocking=non_blocking),
            record_indices=self.record_indices.to(device, non_blocking=non_blocking),
            control_point_indices=self.control_point_indices.to(
                device, non_blocking=non_blocking
            ),
            fiber_paths=self.fiber_paths,
            target_modes=self.target_modes,
            target_segment_offsets=self.target_segment_offsets,
            target_segment_counts=self.target_segment_counts,
            target_segment_starts_zyx=self.target_segment_starts_zyx.to(
                device, non_blocking=non_blocking
            ),
            target_segment_ends_zyx=self.target_segment_ends_zyx.to(
                device, non_blocking=non_blocking
            ),
            target_segment_bbox_lo_zyx=self.target_segment_bbox_lo_zyx,
            target_segment_bbox_hi_zyx=self.target_segment_bbox_hi_zyx,
            target_tangent_zyx=self.target_tangent_zyx.to(
                device, non_blocking=non_blocking
            ),
            target_segment_normals_zyx=None
            if self.target_segment_normals_zyx is None
            else self.target_segment_normals_zyx.to(device, non_blocking=non_blocking),
            volume_spacing_base=None
            if self.volume_spacing_base is None
            else self.volume_spacing_base.to(device, non_blocking=non_blocking),
            direction_target=None
            if self.direction_target is None
            else self.direction_target.to(device, non_blocking=non_blocking),
            direction_weight=None
            if self.direction_weight is None
            else self.direction_weight.to(device, non_blocking=non_blocking),
            direction_mask=None
            if self.direction_mask is None
            else self.direction_mask.to(device, non_blocking=non_blocking),
            direction_indices_bzyx=None
            if self.direction_indices_bzyx is None
            else self.direction_indices_bzyx.to(device, non_blocking=non_blocking),
            direction_target_sparse=None
            if self.direction_target_sparse is None
            else self.direction_target_sparse.to(device, non_blocking=non_blocking),
            direction_weight_sparse=None
            if self.direction_weight_sparse is None
            else self.direction_weight_sparse.to(device, non_blocking=non_blocking),
            direction_tangent_sparse_zyx=None
            if self.direction_tangent_sparse_zyx is None
            else self.direction_tangent_sparse_zyx.to(
                device, non_blocking=non_blocking
            ),
            presence_target=None
            if self.presence_target is None
            else self.presence_target.to(device, non_blocking=non_blocking),
            presence_sideband_mask=None
            if self.presence_sideband_mask is None
            else self.presence_sideband_mask.to(device, non_blocking=non_blocking),
            presence_mask=None
            if self.presence_mask is None
            else self.presence_mask.to(device, non_blocking=non_blocking),
            deferred_augmentation_params=self.deferred_augmentation_params,
            deferred_blur_kernels=self.deferred_blur_kernels,
            profile_timings_ms=self.profile_timings_ms,
        )

    def pin_memory(self) -> "FiberTrace3DBatch":
        """Copy CUDA-bound CPU tensors into allocator-managed pinned storage."""

        def pin(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None or value.device.type != "cpu":
                return value
            return value.pin_memory()

        return replace(
            self,
            normalization_counts=pin(self.normalization_counts),
            embedding_geometry=None if self.embedding_geometry is None else tuple(
                None if item is None else replace(item, eligible=pin(item.eligible))
                for item in self.embedding_geometry),
            volume=pin(self.volume),
            valid_mask=pin(self.valid_mask),
            cp_local_zyx=pin(self.cp_local_zyx),
            crop_origin_zyx=pin(self.crop_origin_zyx),
            stream_indices=pin(self.stream_indices),
            data_indices=pin(self.data_indices),
            record_indices=pin(self.record_indices),
            control_point_indices=pin(self.control_point_indices),
            target_segment_starts_zyx=pin(self.target_segment_starts_zyx),
            target_segment_ends_zyx=pin(self.target_segment_ends_zyx),
            target_tangent_zyx=pin(self.target_tangent_zyx),
            target_segment_normals_zyx=pin(self.target_segment_normals_zyx),
            volume_spacing_base=pin(self.volume_spacing_base),
            direction_target=pin(self.direction_target),
            direction_weight=pin(self.direction_weight),
            direction_mask=pin(self.direction_mask),
            direction_indices_bzyx=pin(self.direction_indices_bzyx),
            direction_target_sparse=pin(self.direction_target_sparse),
            direction_weight_sparse=pin(self.direction_weight_sparse),
            direction_tangent_sparse_zyx=pin(self.direction_tangent_sparse_zyx),
            presence_target=pin(self.presence_target),
            presence_sideband_mask=pin(self.presence_sideband_mask),
            presence_mask=pin(self.presence_mask),
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Keep allocator-owned CUDA tensors alive through work on ``stream``."""

        for value in vars(self).values():
            if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                value.record_stream(stream)
        for item in self.embedding_geometry or ():
            if item is not None and item.eligible.device.type == "cuda":
                item.eligible.record_stream(stream)


def collate_fiber_trace_3d_samples(
    samples: list[FiberTrace3DSample],
    *,
    volume_batch: torch.Tensor | None = None,
    deferred_augmentation_params: tuple[_Augment3DParams, ...] | None = None,
    deferred_blur_kernels: tuple[_BlurKernels3D | None, ...] | None = None,
    profile: bool = False,
    total_start: float | None = None,
    cpu_total_start: float | None = None,
    wall_start_ns: int | None = None,
) -> FiberTrace3DBatch:
    """Collate ordered samples, optionally retaining zero-copy shared tensor views."""

    if not samples:
        raise ValueError("cannot collate an empty Fiber3D sample list")
    stack_start = time.perf_counter()
    total_start = stack_start if total_start is None else float(total_start)
    cpu_total_start = time.process_time() if cpu_total_start is None else float(cpu_total_start)
    wall_start_ns = time.time_ns() if wall_start_ns is None else int(wall_start_ns)
    timings: dict[str, float] | None = None
    if len({sample.presence_target_mode for sample in samples}) != 1:
        raise ValueError("cannot collate mixed presence target modes")
    if len({sample.presence_sideband_weight for sample in samples}) != 1:
        raise ValueError("cannot collate mixed presence sideband weights")
    if profile:
        timings = {}
        keys = set().union(*(set(sample.profile_timings_ms or {}) for sample in samples))
        for key in sorted(keys):
            timings[key] = float(
                sum(float((sample.profile_timings_ms or {}).get(key, 0.0)) for sample in samples)
            )
    segment_offsets: list[int] = []
    segment_counts: list[int] = []
    offset = 0
    for sample in samples:
        count = int(sample.target_segment_starts_zyx.shape[0])
        segment_offsets.append(offset)
        segment_counts.append(count)
        offset += count
    if offset > 0:
        segment_starts = torch.cat([sample.target_segment_starts_zyx for sample in samples])
        segment_ends = torch.cat([sample.target_segment_ends_zyx for sample in samples])
        segment_bbox_lo = torch.cat([sample.target_segment_bbox_lo_zyx for sample in samples])
        segment_bbox_hi = torch.cat([sample.target_segment_bbox_hi_zyx for sample in samples])
        segment_normals = (
            torch.cat([sample.target_segment_normals_zyx for sample in samples])
            if all(sample.target_segment_normals_zyx is not None for sample in samples)
            else None
        )
    else:
        device = samples[0].volume.device
        segment_starts = torch.zeros((0, 3), dtype=torch.float32, device=device)
        segment_ends = torch.zeros((0, 3), dtype=torch.float32, device=device)
        segment_bbox_lo = torch.zeros((0, 3), dtype=torch.long)
        segment_bbox_hi = torch.zeros((0, 3), dtype=torch.long)
        segment_normals = None
    if volume_batch is None:
        volume_batch = torch.stack([sample.volume for sample in samples])
    if int(volume_batch.shape[0]) != len(samples):
        raise ValueError("prebuilt batch tensors do not match sample count")
    if deferred_blur_kernels is not None and len(deferred_blur_kernels) != len(samples):
        raise ValueError("deferred blur-kernel count does not match sample count")
    if deferred_augmentation_params is not None and len(deferred_augmentation_params) != len(samples):
        raise ValueError("deferred augmentation-parameter count does not match sample count")
    return FiberTrace3DBatch(
        presence_target_mode=samples[0].presence_target_mode,
        presence_sideband_weight=samples[0].presence_sideband_weight,
        refinement_geometry=tuple(sample.refinement_geometry for sample in samples),
        embedding_geometry=tuple(sample.embedding_geometry for sample in samples),
        volume=volume_batch,
        valid_mask=None,
        cp_local_zyx=torch.stack([sample.cp_local_zyx for sample in samples]),
        crop_origin_zyx=torch.stack([sample.crop_origin_zyx for sample in samples]),
        stream_indices=torch.as_tensor(
            [sample.stream_index for sample in samples],
            dtype=torch.long,
            device=samples[0].volume.device,
        ),
        data_indices=torch.as_tensor(
            [sample.data_index for sample in samples], dtype=torch.long,
            device=samples[0].volume.device,
        ),
        record_indices=torch.as_tensor(
            [sample.record_index for sample in samples], dtype=torch.long,
            device=samples[0].volume.device,
        ),
        control_point_indices=torch.as_tensor(
            [sample.control_point_index for sample in samples], dtype=torch.long,
            device=samples[0].volume.device,
        ),
        fiber_paths=tuple(sample.fiber_path for sample in samples),
        target_modes=torch.as_tensor([sample.target_mode for sample in samples], dtype=torch.long),
        target_segment_offsets=torch.as_tensor(segment_offsets, dtype=torch.long),
        target_segment_counts=torch.as_tensor(segment_counts, dtype=torch.long),
        volume_spacing_base=torch.as_tensor(
            [sample.volume_spacing_base for sample in samples], dtype=torch.float32
        ),
        target_segment_starts_zyx=segment_starts,
        target_segment_ends_zyx=segment_ends,
        target_segment_bbox_lo_zyx=segment_bbox_lo,
        target_segment_bbox_hi_zyx=segment_bbox_hi,
        target_tangent_zyx=torch.stack([sample.target_tangent_zyx for sample in samples]),
        target_segment_normals_zyx=segment_normals,
        deferred_augmentation_params=deferred_augmentation_params,
        deferred_blur_kernels=deferred_blur_kernels,
        profile_timings_ms={
            **(timings or {}),
            "batch_stack_ms": (time.perf_counter() - stack_start) * 1000.0,
            "batch_total_ms": (time.perf_counter() - total_start) * 1000.0,
            "batch_cpu_ms": (time.process_time() - cpu_total_start) * 1000.0,
            "worker_pid": float(os.getpid()),
            "worker_batch_start_ns": float(wall_start_ns),
            "worker_batch_end_ns": float(time.time_ns()),
        } if profile else None,
    )


def _as_zyx3(value: Any, *, key: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        size = int(value)
        return size, size, size
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key} must be an int or length-3 sequence, got {value!r}")
    result = tuple(int(v) for v in value)
    if any(v <= 0 for v in result):
        raise ValueError(f"{key} values must be positive, got {value!r}")
    return result


def _as_float_zyx3(value: Any, *, key: str) -> tuple[float, float, float]:
    if isinstance(value, (int, float)):
        raw = (float(value),) * 3
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        raw = tuple(float(v) for v in value)
    else:
        raise ValueError(f"{key} must be a number or length-3 sequence, got {value!r}")
    if any((not math.isfinite(v)) or v < 0.0 for v in raw):
        raise ValueError(f"{key} values must be finite and >= 0, got {value!r}")
    return raw


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "little", signed=False)


def _stable_digest(*parts: Any, digest_size: int = 24) -> str:
    digest = hashlib.blake2b(digest_size=digest_size)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _round_json_array(value: np.ndarray, *, decimals: int = 6) -> str:
    return json.dumps(
        np.round(np.asarray(value, dtype=np.float64), decimals).tolist(),
        separators=(",", ":"),
    )


def _resolve_path(path: str | Path, config_dir: Path | None) -> str:
    path_s = str(path)
    if path_s.startswith(_REMOTE_PREFIXES):
        return path_s
    path_obj = Path(path_s).expanduser()
    if path_obj.is_absolute():
        return str(path_obj)
    if config_dir is not None:
        return str((config_dir / path_obj).resolve())
    return str((Path.cwd() / path_obj).resolve())


def _parse_worker_count(value: Any, *, key: str) -> int:
    count = int(value)
    if count <= 0:
        raise ValueError(f"{key} must be > 0")
    return count


def _parse_volume_cache_memory_mib(value: Any) -> float:
    if value is None:
        return float(DEFAULT_VOLUME_CACHE_MEMORY_MIB)
    cache_memory_mib = float(value)
    if not math.isfinite(cache_memory_mib) or cache_memory_mib <= 0.0:
        raise ValueError("volume_cache_memory_mib must be positive when provided")
    return cache_memory_mib


def load_config(path: str | Path) -> FiberTrace3DConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    datasets = raw.get("datasets")
    if raw.get("_array_records") is not None:
        datasets = raw.get("datasets", [{"array_records": True}])
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("config must contain a non-empty 'datasets' list")
    cache_memory_mib = _parse_volume_cache_memory_mib(raw.get("volume_cache_memory_mib"))
    smooth_mode = str(raw.get("augment_smooth_displacement_mode", "none")).lower()
    if smooth_mode not in {"none", "1d", "2d", "3d"}:
        raise ValueError("augment_smooth_displacement_mode must be one of none, 1d, 2d, 3d")
    smooth_spacing = _as_float_zyx3(
        raw.get("augment_smooth_displacement_control_spacing_zyx", [16.0, 16.0, 16.0]),
        key="augment_smooth_displacement_control_spacing_zyx",
    )
    if any(v <= 0.0 for v in smooth_spacing):
        raise ValueError("augment_smooth_displacement_control_spacing_zyx values must be > 0")
    smooth_probability = float(raw.get("augment_smooth_displacement_probability", 0.0))
    if not 0.0 <= smooth_probability <= 1.0:
        raise ValueError("augment_smooth_displacement_probability must be in [0,1]")
    anisotropic_probability = float(raw.get("augment_anisotropic_blur_probability", 0.0))
    if not 0.0 <= anisotropic_probability <= 1.0:
        raise ValueError("augment_anisotropic_blur_probability must be in [0,1]")
    anisotropic_orientation = str(raw.get("augment_anisotropic_blur_orientation", "fiber")).lower()
    if anisotropic_orientation not in {"fiber", "random", "axis", "z", "y", "x"}:
        raise ValueError(
            "augment_anisotropic_blur_orientation must be fiber, random, axis, z, y, or x"
        )
    presence_target_mode = str(raw.get("presence_target_mode", "binary")).lower()
    if presence_target_mode not in {"binary", "anisotropic_gaussian"}:
        raise ValueError("presence_target_mode must be binary or anisotropic_gaussian")
    for unsupported in (
        "augment_shear_x",
        "augment_shear_y",
        "augment_shear_z",
        "augment_ringing",
        "augment_ringing_strength",
    ):
        if unsupported in raw and float(raw[unsupported]) != 0.0:
            raise ValueError(
                f"{unsupported} is not supported for 3D CP training yet; "
                "3D skew/ringing semantics must be specified explicitly"
            )
    return FiberTrace3DConfig(
       datasets=tuple(dict(entry) for entry in datasets),
        sampling_strategy=str(raw.get("sampling_strategy", "control_points")),
        embedding=EmbeddingConfig.from_mapping(raw),
        refinement=RefinementConfig.parse(raw),
        refinement_split=raw.get("_refinement_split", "train"),
        batch_size=int(raw.get("batch_size", 1)),
        patch_shape_zyx=_as_zyx3(
            raw.get("patch_shape_zyx", raw.get("crop_size", [192, 192, 192])),
            key="patch_shape_zyx",
        ),
        seed=int(raw.get("seed", 1)),
        cp_margin_voxels=int(raw.get("cp_margin_voxels", raw.get("control_point_margin_voxels", 4))),
        presence_radius_voxels=float(raw.get("presence_radius_voxels", 2.0)),
        presence_target_mode=presence_target_mode,
        presence_gaussian_width_ratio=float(raw.get("presence_gaussian_width_ratio", 4.0)),
        presence_gaussian_truncate_sigma=float(raw.get("presence_gaussian_truncate_sigma", 3.0)),
        presence_sideband_weight=float(raw.get("presence_sideband_weight", 0.0)),
        presence_sideband_half_width_base=float(
            raw.get("presence_sideband_half_width_base", 64.0)
        ),
        presence_sideband_half_height_base=float(
            raw.get("presence_sideband_half_height_base", 16.0)
        ),
        presence_negative_edge_margin_voxels=(
            None
            if raw.get("presence_negative_edge_margin_voxels") is None
            else int(raw.get("presence_negative_edge_margin_voxels"))
        ),
        image_normalization=str(raw.get("image_normalization", "zscore")),
        augment_enabled=bool(raw.get("augment_enabled", True)),
        augment_shift_zyx=_as_float_zyx3(
            raw.get("augment_shift_zyx", raw.get("augment_shift_xyz", [0.0, 0.0, 0.0])),
            key="augment_shift_zyx",
        ),
        augment_rotation_degrees=float(raw.get("augment_rotation_degrees", 0.0)),
        augment_scale_min=float(raw.get("augment_scale_min", 1.0)),
        augment_scale_max=float(raw.get("augment_scale_max", 1.0)),
        augment_flip_probability=float(raw.get("augment_flip_probability", 0.0)),
        augment_brightness=float(raw.get("augment_brightness", 0.0)),
        augment_contrast_min=float(raw.get("augment_contrast_min", 1.0)),
        augment_contrast_max=float(raw.get("augment_contrast_max", 1.0)),
        augment_gamma_min=float(raw.get("augment_gamma_min", 1.0)),
        augment_gamma_max=float(raw.get("augment_gamma_max", 1.0)),
        augment_noise_std=float(raw.get("augment_noise_std", 0.0)),
        augment_blur_sigma=float(raw.get("augment_blur_sigma", 0.0)),
        augment_smooth_displacement_mode=smooth_mode,
        augment_smooth_displacement_amplitude_zyx=_as_float_zyx3(
            raw.get("augment_smooth_displacement_amplitude_zyx", [0.0, 0.0, 0.0]),
            key="augment_smooth_displacement_amplitude_zyx",
        ),
        augment_smooth_displacement_control_spacing_zyx=smooth_spacing,
        augment_smooth_displacement_probability=smooth_probability,
        augment_anisotropic_blur_probability=anisotropic_probability,
        augment_anisotropic_blur_sigma_along=float(
            raw.get("augment_anisotropic_blur_sigma_along", 0.0)
        ),
        augment_anisotropic_blur_sigma_across=float(
            raw.get("augment_anisotropic_blur_sigma_across", 0.0)
        ),
        augment_anisotropic_blur_orientation=anisotropic_orientation,
        augment_anisotropic_blur_roll_degrees=float(
            raw.get("augment_anisotropic_blur_roll_degrees", 0.0)
        ),
        round_source_to_chunk_boundaries=bool(raw.get("round_source_to_chunk_boundaries", True)),
        geometry_tile_depth=int(raw.get("geometry_tile_depth", 0)),
        geometry_backend=str(raw.get("geometry_backend", "auto")).lower(),
        value_augmentation_backend=str(
            raw.get("value_augmentation_backend", "auto")
        ).lower(),
        prefetch_workers=_parse_worker_count(raw.get("prefetch_workers", 16), key="prefetch_workers"),
        prefetch_sampler_workers=_parse_worker_count(
            raw.get("prefetch_sampler_workers", 2),
            key="prefetch_sampler_workers",
        ),
        volume_cache_dir=None if raw.get("volume_cache_dir") is None else str(raw.get("volume_cache_dir")),
        volume_cache_memory_mib=cache_memory_mib,
        volume_io_threads=(
            None if raw.get("volume_io_threads") is None else int(raw.get("volume_io_threads"))
        ),
        volume_decode_threads=(
            None
            if raw.get("volume_decode_threads") is None
            else int(raw.get("volume_decode_threads"))
        ),
        volume_sample_threads=(
            None
            if raw.get("volume_sample_threads") is None
            else int(raw.get("volume_sample_threads"))
        ),
        volume_chunk_locality_fast_path=bool(
            raw.get("volume_chunk_locality_fast_path", True)
        ),
        volume_collect_locality_stats=bool(
            raw.get("volume_collect_locality_stats", False)
        ),
        volume_cache_offline=bool(raw.get("volume_cache_offline", False)),
        volume_cache_retry_seconds=float(raw.get("volume_cache_retry_seconds", 0.0)),
        config_dir=config_path.parent,
    )


def _common_zarr_config(config: FiberTrace3DConfig) -> dict[str, Any]:
    return {
        "volume_cache_dir": config.volume_cache_dir,
        "volume_cache_memory_mib": config.volume_cache_memory_mib,
        "volume_io_threads": config.volume_io_threads,
        "volume_decode_threads": config.volume_decode_threads,
        "volume_sample_threads": config.volume_sample_threads,
        "volume_chunk_locality_fast_path": config.volume_chunk_locality_fast_path,
        "volume_collect_locality_stats": config.volume_collect_locality_stats,
        "volume_cache_offline": config.volume_cache_offline,
        "volume_cache_retry_seconds": config.volume_cache_retry_seconds,
        "_config_dir": str(config.config_dir) if config.config_dir is not None else None,
    }


def _open_dataset_volume(dataset_config: dict[str, Any], config: FiberTrace3DConfig) -> Any:
    volume_path = dataset_config.get("base_volume_path", dataset_config.get("volume_path"))
    if volume_path is None:
        raise ValueError("dataset entry must define base_volume_path")
    scale = int(dataset_config.get("base_volume_scale", dataset_config.get("volume_scale", 0)))
    return open_zarr(
        _resolve_path(volume_path, config.config_dir),
        scale=scale,
        auth_json_path=dataset_config.get("base_volume_auth_json", dataset_config.get("volume_auth_json")),
        config=_common_zarr_config(config),
    )


def _validate_dataset_manifest(
    dataset_config: dict[str, Any],
    config: FiberTrace3DConfig,
    *,
    volume: Any,
    volume_path: str,
) -> tuple[int, int, int]:
    manifest_path = dataset_config.get("lasagna_manifest_path")
    common_config = _common_zarr_config(config)
    volume_scale = int(dataset_config.get("base_volume_scale", dataset_config.get("volume_scale", 0)))
    if not manifest_path:
        if dataset_config.get("_array_records") or volume_scale == 0:
            base_shape_zyx = _volume_shape_zyx(volume, label="base volume selected level")
        else:
            base0 = open_zarr(
                volume_path,
                scale=0,
                auth_json_path=dataset_config.get("base_volume_auth_json", dataset_config.get("volume_auth_json")),
                config=common_config,
            )
            base_shape_zyx = _volume_shape_zyx(base0, label="base volume level 0")
        _validate_shape(
            _volume_shape_zyx(volume, label="base volume selected level"),
            _omezarr_level_shape(tuple(base_shape_zyx), volume_scale),
            label="base volume selected level",
            context=f"base_shape_zyx={base_shape_zyx} base_volume_scale={volume_scale}",
        )
        return tuple(int(v) for v in base_shape_zyx)

    resolved_manifest = _resolve_config_relative_path(manifest_path, common_config)
    lasagna_volume = _load_lasagna_volume(resolved_manifest, common_config)
    if lasagna_volume.base_shape_zyx is None:
        raise ValueError("Lasagna manifest must provide base_shape_zyx")
    base0 = open_zarr(
        volume_path,
        scale=0,
        auth_json_path=dataset_config.get("base_volume_auth_json", dataset_config.get("volume_auth_json")),
        config=common_config,
    )
    _validate_shape(
        _volume_shape_zyx(base0, label="base volume level 0"),
        tuple(lasagna_volume.base_shape_zyx),
        label="base volume level 0",
        context=f"Lasagna manifest {resolved_manifest}",
    )
    _validate_shape(
        _volume_shape_zyx(volume, label="base volume selected level"),
        _omezarr_level_shape(tuple(lasagna_volume.base_shape_zyx), volume_scale),
        label="base volume selected level",
        context=f"base_shape_zyx={lasagna_volume.base_shape_zyx} base_volume_scale={volume_scale}",
    )
    return tuple(int(v) for v in lasagna_volume.base_shape_zyx)


def _resolve_fiber_paths(dataset_config: dict[str, Any], config: FiberTrace3DConfig) -> list[Path]:
    paths: list[Path] = []
    raw_paths = dataset_config.get("fiber_paths")
    if raw_paths is not None:
        if isinstance(raw_paths, (str, Path)):
            raw_paths = [raw_paths]
        for raw in raw_paths:
            paths.append(Path(_resolve_path(raw, config.config_dir)))
    raw_glob = dataset_config.get("fiber_glob")
    if raw_glob:
        resolved = _resolve_path(raw_glob, config.config_dir)
        paths.extend(Path(path) for path in sorted(glob.glob(resolved)))
    if not paths:
        raise ValueError("dataset entry must define fiber_paths or fiber_glob")
    return paths


def _homogeneous_matrix_from_config(raw_matrix: Any, *, key: str) -> np.ndarray:
    matrix = np.asarray(raw_matrix, dtype=np.float64)
    if matrix.shape == (3, 4):
        matrix = np.vstack([matrix, [0.0, 0.0, 0.0, 1.0]])
    if matrix.shape != (4, 4):
        raise ValueError(f"{key} must be a 3x4 or 4x4 XYZ affine matrix")
    if not bool(np.isfinite(matrix).all()):
        raise ValueError(f"{key} contains non-finite values")
    if not np.allclose(matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)):
        raise ValueError(f"{key} must have homogeneous last row [0, 0, 0, 1]")
    return matrix


def _dataset_fiber_transform(
    dataset_config: dict[str, Any],
    config: FiberTrace3DConfig,
) -> tuple[np.ndarray | None, str]:
    json_sources = [
        (key, dataset_config[key])
        for key in ("fiber_transform_json", "fiber_transform_json_path")
        if dataset_config.get(key) is not None
    ]
    inline_sources = [
        (key, dataset_config[key])
        for key in ("fiber_transform", "transform")
        if dataset_config.get(key) is not None
    ]
    if len(json_sources) + len(inline_sources) == 0:
        return None, "none"
    if len(json_sources) + len(inline_sources) > 1:
        raise ValueError("set only one fiber transform source")
    invert = bool(
        dataset_config.get(
            "fiber_transform_invert",
            dataset_config.get("transform_invert", False),
        )
    )
    if json_sources:
        key, raw_path = json_sources[0]
        resolved = _resolve_path(raw_path, config.config_dir)
        matrix = np.asarray(read_transform_json(resolved).matrix_xyz, dtype=np.float64)
        source_label = f"{key}:{resolved}"
    else:
        key, raw_matrix = inline_sources[0]
        matrix = _homogeneous_matrix_from_config(raw_matrix, key=key)
        source_label = key
    if invert:
        matrix = np.linalg.inv(matrix)
    identity = _stable_digest(
        "fiber_transform",
        source_label,
        f"invert={invert}",
        _round_json_array(matrix, decimals=9),
    )
    return matrix, identity


def _first_out_of_bounds_control_point(
    fiber: Vc3dFiber,
    base_shape_zyx: tuple[int, int, int],
) -> tuple[int, np.ndarray] | None:
    cps = np.asarray(fiber.control_points_zyx, dtype=np.float64)
    shape = np.asarray(base_shape_zyx, dtype=np.float64)
    valid = np.isfinite(cps).all(axis=1)
    valid &= np.all(cps >= 0.0, axis=1)
    valid &= np.all(cps <= (shape - 1.0), axis=1)
    bad = np.flatnonzero(~valid)
    if bad.size == 0:
        return None
    index = int(bad[0])
    return index, cps[index]


def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / norm
    z, y, x = axis
    skew = np.asarray(
        [[0.0, -x, y], [x, 0.0, -z], [-y, z, 0.0]],
        dtype=np.float64,
    )
    ident = np.eye(3, dtype=np.float64)
    return ident + math.sin(angle_rad) * skew + (1.0 - math.cos(angle_rad)) * (skew @ skew)


def _zyx_grid(
    shape_zyx: tuple[int, int, int],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    z = torch.arange(int(shape_zyx[0]), dtype=dtype, device=device)
    y = torch.arange(int(shape_zyx[1]), dtype=dtype, device=device)
    x = torch.arange(int(shape_zyx[2]), dtype=dtype, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack([zz, yy, xx], dim=-1)


def _sample_control_1d(
    control: torch.Tensor,
    coord: torch.Tensor,
    *,
    origin: float,
    spacing: float,
    batch_index=None,
) -> torch.Tensor:
    if isinstance(control, np.ndarray):
        if batch_index is not None:
            from scipy.ndimage import map_coordinates
            return map_coordinates(control, np.stack((batch_index, (coord-origin)/spacing)), order=1, mode="nearest", prefilter=False)
        return np.interp((coord - origin) / spacing, np.arange(len(control)), control)
    if control.ndim != 1:
        raise ValueError("1D smooth displacement control must be one-dimensional")
    if control.numel() == 1:
        return torch.full_like(coord, float(control[0]))
    u = (coord - float(origin)) / float(spacing)
    u = torch.clamp(u, 0.0, float(control.numel() - 1))
    lo = torch.floor(u).to(dtype=torch.long)
    hi = torch.clamp(lo + 1, max=int(control.numel() - 1))
    frac = (u - lo.to(dtype=u.dtype)).to(dtype=control.dtype)
    return control[lo] * (1.0 - frac) + control[hi] * frac


def _sample_control_2d(
    control: torch.Tensor,
    coord_a: torch.Tensor,
    coord_b: torch.Tensor,
    *,
    origin_a: float,
    origin_b: float,
    spacing_a: float,
    spacing_b: float,
    batch_index=None,
) -> torch.Tensor:
    if isinstance(control, np.ndarray):
        from scipy.ndimage import map_coordinates
        coords = np.stack(((coord_a-origin_a)/spacing_a, (coord_b-origin_b)/spacing_b))
        if batch_index is not None:
            coords = np.concatenate((np.asarray(batch_index)[None], coords))
        return map_coordinates(control.astype(np.float64, copy=False), coords, order=1, mode="nearest", prefilter=False)
    if control.ndim != 2:
        raise ValueError("2D smooth displacement control must be two-dimensional")
    height, width = int(control.shape[0]), int(control.shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("2D smooth displacement control must not be empty")
    ua = torch.clamp((coord_a - float(origin_a)) / float(spacing_a), 0.0, float(max(height - 1, 0)))
    ub = torch.clamp((coord_b - float(origin_b)) / float(spacing_b), 0.0, float(max(width - 1, 0)))
    a0 = torch.floor(ua).to(dtype=torch.long)
    b0 = torch.floor(ub).to(dtype=torch.long)
    a1 = torch.clamp(a0 + 1, max=max(height - 1, 0))
    b1 = torch.clamp(b0 + 1, max=max(width - 1, 0))
    wa = (ua - a0.to(dtype=ua.dtype)).to(dtype=control.dtype)
    wb = (ub - b0.to(dtype=ub.dtype)).to(dtype=control.dtype)
    c00 = control[a0, b0]
    c01 = control[a0, b1]
    c10 = control[a1, b0]
    c11 = control[a1, b1]
    c0 = c00 * (1.0 - wb) + c01 * wb
    c1 = c10 * (1.0 - wb) + c11 * wb
    return c0 * (1.0 - wa) + c1 * wa


def _smooth_control_tensors(
    smooth: _SmoothDisplacement3DParams | None,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    if smooth is None:
        return ()
    return tuple(torch.as_tensor(control, dtype=torch.float32, device=device) for control in smooth.controls)


def _smooth_forward_rel_zyx(
    rel_zyx: torch.Tensor,
    smooth: _SmoothDisplacement3DParams | None,
    controls: tuple[torch.Tensor, ...] | None = None,
    batch_index=None,
) -> torch.Tensor:
    if smooth is None or smooth.mode == "none":
        return rel_zyx
    array_backend = isinstance(rel_zyx, np.ndarray)
    tensors = (smooth.controls if array_backend else _smooth_control_tensors(smooth, device=rel_zyx.device)) if controls is None else controls
    out = rel_zyx.copy() if array_backend else rel_zyx.clone()
    origin = np.asarray(smooth.control_min_zyx, dtype=np.float64)
    spacing = np.asarray(smooth.control_spacing_zyx, dtype=np.float64)
    if smooth.mode == "1d":
        out[..., 0] = out[..., 0] + _sample_control_1d(
            tensors[0],
            out[..., 2],
            origin=float(origin[2]),
            spacing=float(spacing[2]),
            batch_index=batch_index,
        )
    elif smooth.mode == "2d":
        out[..., 0] = out[..., 0] + _sample_control_2d(
            tensors[0],
            out[..., 1],
            out[..., 2],
            origin_a=float(origin[1]),
            origin_b=float(origin[2]),
            spacing_a=float(spacing[1]),
            spacing_b=float(spacing[2]),
            batch_index=batch_index,
        )
    elif smooth.mode == "3d":
        out[..., 0] = out[..., 0] + _sample_control_2d(
            tensors[0],
            out[..., 1],
            out[..., 2],
            origin_a=float(origin[1]),
            origin_b=float(origin[2]),
            spacing_a=float(spacing[1]),
            spacing_b=float(spacing[2]),
            batch_index=batch_index,
        )
        out[..., 1] = out[..., 1] + _sample_control_2d(
            tensors[1],
            out[..., 0],
            out[..., 2],
            origin_a=float(origin[0]),
            origin_b=float(origin[2]),
            spacing_a=float(spacing[0]),
            spacing_b=float(spacing[2]),
            batch_index=batch_index,
        )
        out[..., 2] = out[..., 2] + _sample_control_2d(
            tensors[2],
            out[..., 0],
            out[..., 1],
            origin_a=float(origin[0]),
            origin_b=float(origin[1]),
            spacing_a=float(spacing[0]),
            spacing_b=float(spacing[1]),
            batch_index=batch_index,
        )
    else:
        raise ValueError(f"unsupported smooth displacement mode {smooth.mode!r}")
    return out


def _smooth_backward_rel_zyx(
    rel_zyx: torch.Tensor,
    smooth: _SmoothDisplacement3DParams | None,
    controls: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    if smooth is None or smooth.mode == "none":
        return rel_zyx
    tensors = _smooth_control_tensors(smooth, device=rel_zyx.device) if controls is None else controls
    out = rel_zyx.clone()
    origin = np.asarray(smooth.control_min_zyx, dtype=np.float64)
    spacing = np.asarray(smooth.control_spacing_zyx, dtype=np.float64)
    if smooth.mode == "1d":
        out[..., 0] = out[..., 0] - _sample_control_1d(
            tensors[0],
            out[..., 2],
            origin=float(origin[2]),
            spacing=float(spacing[2]),
        )
    elif smooth.mode == "2d":
        out[..., 0] = out[..., 0] - _sample_control_2d(
            tensors[0],
            out[..., 1],
            out[..., 2],
            origin_a=float(origin[1]),
            origin_b=float(origin[2]),
            spacing_a=float(spacing[1]),
            spacing_b=float(spacing[2]),
        )
    elif smooth.mode == "3d":
        out[..., 2] = out[..., 2] - _sample_control_2d(
            tensors[2],
            out[..., 0],
            out[..., 1],
            origin_a=float(origin[0]),
            origin_b=float(origin[1]),
            spacing_a=float(spacing[0]),
            spacing_b=float(spacing[1]),
        )
        out[..., 1] = out[..., 1] - _sample_control_2d(
            tensors[1],
            out[..., 0],
            out[..., 2],
            origin_a=float(origin[0]),
            origin_b=float(origin[2]),
            spacing_a=float(spacing[0]),
            spacing_b=float(spacing[2]),
        )
        out[..., 0] = out[..., 0] - _sample_control_2d(
            tensors[0],
            out[..., 1],
            out[..., 2],
            origin_a=float(origin[1]),
            origin_b=float(origin[2]),
            spacing_a=float(spacing[1]),
            spacing_b=float(spacing[2]),
        )
    else:
        raise ValueError(f"unsupported smooth displacement mode {smooth.mode!r}")
    return out


def _clip_segment_to_aabb(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    lo: np.ndarray,
    hi: np.ndarray,
    eps: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray] | None:
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    direction = p1 - p0
    if (not np.isfinite(p0).all()) or (not np.isfinite(p1).all()):
        return None
    t_min = 0.0
    t_max = 1.0
    for axis in range(3):
        d = float(direction[axis])
        if abs(d) <= eps:
            if float(p0[axis]) < float(lo[axis]) or float(p0[axis]) > float(hi[axis]):
                return None
            continue
        t0 = (float(lo[axis]) - float(p0[axis])) / d
        t1 = (float(hi[axis]) - float(p0[axis])) / d
        if t0 > t1:
            t0, t1 = t1, t0
        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_min > t_max:
            return None
    if t_max - t_min <= eps:
        return None
    return p0 + t_min * direction, p0 + t_max * direction


def _uses_dense_fiber_supervision(record: _Record) -> bool:
    if record.dataset_config.get("dense_supervision", False):
        return True
    source_format = str(record.fiber.metadata.get("source_format", "")).lower()
    if source_format == "nml":
        return True
    if record.fiber.path is not None and str(record.fiber.path).lower().endswith(".nml"):
        return True
    return False


def _read_raw_block(array: Any, start_zyx: np.ndarray, end_zyx: np.ndarray) -> np.ndarray:
    start = np.asarray(start_zyx, dtype=np.int64)
    end = np.asarray(end_zyx, dtype=np.int64)
    shape = tuple(int(v) for v in getattr(array, "shape", ()))
    if len(shape) != 3:
        raise ValueError(f"3D loader expects a 3D ZYX base volume, got shape={shape}")
    block_shape = tuple(int(v) for v in np.maximum(end - start, 0))
    sample = np.asarray(array[0:1, 0:1, 0:1])
    out = np.zeros(block_shape, dtype=sample.dtype)
    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, np.asarray(shape, dtype=np.int64))
    if not bool(np.all(src_end > src_start)):
        return out
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)
    out[
        int(dst_start[0]) : int(dst_end[0]),
        int(dst_start[1]) : int(dst_end[1]),
        int(dst_start[2]) : int(dst_end[2]),
    ] = array[
        int(src_start[0]) : int(src_end[0]),
        int(src_start[1]) : int(src_end[1]),
        int(src_start[2]) : int(src_end[2]),
    ]
    return out


def _chunks_zyx(array: Any) -> tuple[int, int, int] | None:
    chunks = getattr(array, "chunks", None)
    if chunks is None:
        return None
    chunks_tuple = tuple(int(v) for v in chunks)
    if len(chunks_tuple) == 3:
        return chunks_tuple
    if len(chunks_tuple) == 4:
        return chunks_tuple[1:]
    return None


def _round_bbox_to_chunks(
    array: Any,
    start_zyx: np.ndarray,
    end_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    chunks = _chunks_zyx(array)
    if chunks is None:
        return start_zyx.astype(np.int64), end_zyx.astype(np.int64)
    chunk = np.asarray(chunks, dtype=np.int64)
    start = (np.asarray(start_zyx, dtype=np.int64) // chunk) * chunk
    end = ((np.asarray(end_zyx, dtype=np.int64) + chunk - 1) // chunk) * chunk
    return start.astype(np.int64), end.astype(np.int64)


def _normalize_image(volume: torch.Tensor, valid_mask: torch.Tensor, mode: str) -> torch.Tensor:
    mode_l = str(mode).lower()
    if mode_l in {"none", "raw", "identity"}:
        return volume
    valid = valid_mask.to(dtype=torch.bool)
    if not bool(valid.any()):
        return volume
    values = volume[valid]
    if mode_l == "zscore":
        mean = values.mean()
        std = values.std(unbiased=False).clamp_min(1.0e-6)
        return (volume - mean) / std
    if mode_l == "minmax":
        lo = values.min()
        hi = values.max()
        return (volume - lo) / (hi - lo).clamp_min(1.0e-6)
    raise ValueError(f"unsupported image_normalization {mode!r}")

def _gaussian_kernel1d(sigma: float, *, device: torch.device) -> torch.Tensor:
    """Retained shared 1D kernel helper for trace-to-control-point smoothing."""
    sigma_f = float(sigma)
    if sigma_f <= 0.0:
        return torch.ones((1,), dtype=torch.float32, device=device)
    radius = max(1, int(math.ceil(3.0 * sigma_f)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-0.5 * (coords / sigma_f) ** 2)
    return kernel / kernel.sum().clamp_min(1.0e-12)


def _perpendicular_basis_zyx(axis_zyx: np.ndarray) -> np.ndarray:
    """Return an orthonormal ZYX basis with the requested principal axis first."""
    axis = np.asarray(axis_zyx, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("anisotropic blur axis must be finite and nonzero")
    along = axis / norm
    reference = (
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(along[0])) > 0.9
        else np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    )
    across_a = np.cross(along, reference)
    across_a /= max(float(np.linalg.norm(across_a)), 1.0e-12)
    across_b = np.cross(along, across_a)
    across_b /= max(float(np.linalg.norm(across_b)), 1.0e-12)
    return np.stack([along, across_a, across_b], axis=1)


def _gaussian_kernel_1d(sigma: float) -> np.ndarray | None:
    """Build one normalized, truncated 1D Gaussian in FP64, then cast to FP32."""
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("isotropic blur sigma must be finite and nonnegative")
    if sigma == 0.0:
        return None
    radius = int(math.ceil(3.0 * sigma))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (coordinates / sigma) ** 2)
    weights /= float(weights.sum())
    return np.ascontiguousarray(weights, dtype=np.float32)


def _directional_gaussian_kernel_3d(
    params: _Augment3DParams,
) -> _DirectGaussianKernel3D | None:
    """Build one normalized directional-only discrete 3D Gaussian."""
    if params.anisotropic_blur_axis_zyx is None:
        return None
    basis = _perpendicular_basis_zyx(params.anisotropic_blur_axis_zyx)
    sigmas = np.asarray(
        [
            float(params.anisotropic_blur_sigma_along),
            float(params.anisotropic_blur_sigma_across),
            float(params.anisotropic_blur_sigma_across),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(sigmas).all() or bool((sigmas < 0.0).any()):
        raise ValueError("blur sigmas must be finite and nonnegative")
    if bool((sigmas <= 0.0).all()):
        return None

    principal_radii = np.ceil(3.0 * sigmas).astype(np.int64)
    world_radii = np.ceil(np.abs(basis) @ principal_radii).astype(np.int64)
    axes = [
        np.arange(-int(radius), int(radius) + 1, dtype=np.float64)
        for radius in world_radii
    ]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    offsets = np.stack([zz, yy, xx], axis=-1)
    principal = offsets @ basis
    inside = np.ones(offsets.shape[:-1], dtype=bool)
    exponent = np.zeros(offsets.shape[:-1], dtype=np.float64)
    for component in range(3):
        radius = int(principal_radii[component])
        coordinate = principal[..., component]
        inside &= np.abs(coordinate) <= float(radius) + 1.0e-7
        sigma = float(sigmas[component])
        if sigma > 0.0:
            exponent += (coordinate / sigma) ** 2
        else:
            inside &= np.abs(coordinate) <= 1.0e-7
    weights = np.where(inside, np.exp(-0.5 * exponent), 0.0)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("direct Gaussian kernel has no finite support")
    dense = np.ascontiguousarray(weights / total, dtype=np.float32)
    return _DirectGaussianKernel3D(dense)


def _build_blur_kernels_3d(params: _Augment3DParams) -> _BlurKernels3D:
    return _BlurKernels3D(
        isotropic_1d=_gaussian_kernel_1d(params.blur_sigma),
        directional_zyx=_directional_gaussian_kernel_3d(params),
    )


def _apply_isotropic_kernels_3d(
    volume: torch.Tensor,
    kernels_1d: tuple[np.ndarray | None, ...],
) -> torch.Tensor:
    """Apply independently weighted isotropic kernels as three grouped 1D passes."""
    if volume.ndim != 5 or int(volume.shape[1]) != 1:
        raise ValueError("isotropic Gaussian blur expects Bx1xZxYxX volumes")
    batch = int(volume.shape[0])
    if len(kernels_1d) != batch:
        raise ValueError("isotropic Gaussian kernel count does not match batch size")
    active = [kernel for kernel in kernels_1d if kernel is not None]
    if not active:
        return volume
    lengths = [int(kernel.size) for kernel in active]
    if any(length <= 0 or length % 2 == 0 for length in lengths):
        raise ValueError("isotropic Gaussian kernels must have positive odd lengths")
    max_length = max(lengths)
    weights_1d = torch.zeros(
        (batch, max_length),
        dtype=volume.dtype,
        device=volume.device,
    )
    center = max_length // 2
    for sample_index, kernel in enumerate(kernels_1d):
        if kernel is None:
            weights_1d[sample_index, center] = 1.0
            continue
        tensor = torch.as_tensor(
            np.asarray(kernel, dtype=np.float32),
            dtype=volume.dtype,
            device=volume.device,
        )
        start = (max_length - int(tensor.numel())) // 2
        weights_1d[sample_index, start : start + int(tensor.numel())] = tensor
    result = volume.transpose(0, 1)
    radius = max_length // 2
    for axis in range(3):
        shape = [batch, 1, 1, 1, 1]
        shape[axis + 2] = max_length
        weights = weights_1d.reshape(shape)
        padding = [0, 0, 0, 0, 0, 0]
        pair = 2 * (2 - axis)
        padding[pair : pair + 2] = [radius, radius]
        result = F.conv3d(
            F.pad(result, tuple(padding), mode="replicate"),
            weights,
            groups=batch,
        )
    return result.transpose(0, 1)


def _directional_kernel_groups(
    kernels_zyx: tuple[_DirectGaussianKernel3D | None, ...],
    *,
    max_padding_ratio: float = 1.25,
) -> list[list[int]]:
    """Deterministically form small rectangular-convolution groups."""
    shapes = {
        index: tuple(int(value) for value in kernel.dense_zyx.shape)
        for index, kernel in enumerate(kernels_zyx)
        if kernel is not None
    }
    if any(any(value <= 0 or value % 2 == 0 for value in shape) for shape in shapes.values()):
        raise ValueError("directional Gaussian kernels must have positive odd ZYX shapes")
    order = sorted(shapes, key=lambda index: (-math.prod(shapes[index]), index))
    groups: list[list[int]] = []
    for sample_index in order:
        best: tuple[float, int] | None = None
        for group_index, group in enumerate(groups):
            members = [*group, sample_index]
            common = tuple(max(shapes[index][axis] for index in members) for axis in range(3))
            ratio = len(members) * math.prod(common) / sum(
                math.prod(shapes[index]) for index in members
            )
            candidate = (ratio, group_index)
            if ratio <= max_padding_ratio and (best is None or candidate < best):
                best = candidate
        if best is None:
            groups.append([sample_index])
        else:
            groups[best[1]].append(sample_index)
    return groups


def _apply_directional_kernels_3d(
    volume: torch.Tensor,
    kernels_zyx: tuple[_DirectGaussianKernel3D | None, ...],
    *,
    max_padding_ratio: float = 1.25,
) -> torch.Tensor:
    """Apply greedily grouped rectangular directional kernels."""
    if volume.ndim != 5 or int(volume.shape[1]) != 1:
        raise ValueError("directional Gaussian blur expects Bx1xZxYxX volumes")
    if len(kernels_zyx) != int(volume.shape[0]):
        raise ValueError("directional Gaussian kernel count does not match batch size")
    groups = _directional_kernel_groups(
        kernels_zyx,
        max_padding_ratio=max_padding_ratio,
    )
    if not groups:
        return volume
    result = volume.clone()
    for group in groups:
        indices = torch.as_tensor(group, dtype=torch.long, device=volume.device)
        selected = volume.index_select(0, indices)
        kernels = [kernels_zyx[index] for index in group]
        assert all(kernel is not None for kernel in kernels)
        max_shape = tuple(
            max(int(kernel.dense_zyx.shape[axis]) for kernel in kernels if kernel is not None)
            for axis in range(3)
        )
        weights = torch.zeros(
            (len(group), 1, *max_shape),
            dtype=volume.dtype,
            device=volume.device,
        )
        for local_index, kernel in enumerate(kernels):
            assert kernel is not None
            tensor = torch.as_tensor(kernel.dense_zyx, dtype=volume.dtype, device=volume.device)
            starts = tuple((max_shape[axis] - int(tensor.shape[axis])) // 2 for axis in range(3))
            slices = tuple(
                slice(starts[axis], starts[axis] + int(tensor.shape[axis]))
                for axis in range(3)
            )
            weights[(local_index, 0, *slices)] = tensor
        rz, ry, rx = (value // 2 for value in max_shape)
        filtered = F.conv3d(
            F.pad(selected.transpose(0, 1), (rx, rx, ry, ry, rz, rz), mode="replicate"),
            weights,
            groups=len(group),
        ).transpose(0, 1)
        result.index_copy_(0, indices, filtered)
    return result


def _apply_blur_kernels_3d(
    volume: torch.Tensor,
    kernels: tuple[_BlurKernels3D | None, ...],
    *,
    max_directional_padding_ratio: float = 1.25,
) -> torch.Tensor:
    if len(kernels) != int(volume.shape[0]):
        raise ValueError("blur-kernel descriptor count does not match batch size")
    isotropic = tuple(None if item is None else item.isotropic_1d for item in kernels)
    directional = tuple(None if item is None else item.directional_zyx for item in kernels)
    return _apply_directional_kernels_3d(
        _apply_isotropic_kernels_3d(volume, isotropic),
        directional,
        max_padding_ratio=max_directional_padding_ratio,
    )


def _native_value_transform(
    volume: torch.Tensor,
    *,
    normalization: str,
    contrast: float,
    brightness: float,
    gamma: float,
    backend: str,
) -> tuple[torch.Tensor | None, bool]:
    eligible = (
        volume.device.type == "cpu"
        and volume.dtype == torch.float32
        and volume.is_contiguous()
        and not volume.requires_grad
    )
    if not eligible:
        if backend == "native":
            raise RuntimeError(
                "value_augmentation_backend='native' requires contiguous, "
                "gradient-free contiguous float32 CPU values"
            )
        return None, False
    if backend not in {"auto", "native"}:
        return None, False
    try:
        from vc.volume import fiber3d_value_transform
    except ImportError as exc:
        if backend == "native":
            raise RuntimeError(
                "value_augmentation_backend='native' requires a rebuilt "
                "volume-cartographer binding with fiber3d_value_transform"
            ) from exc
        return None, False
    # Invalid source coordinates were zero-filled during sampling. The binding
    # still accepts its legacy mask argument, so each worker reuses one
    # immutable all-valid array rather than transporting source validity.
    shape = tuple(int(value) for value in volume.shape)
    full_patch = _FULL_VALID_MASK_CACHE.get(shape)
    if full_patch is None:
        full_patch = np.ones(shape, dtype=bool)
        _FULL_VALID_MASK_CACHE[shape] = full_patch
    native_result = fiber3d_value_transform(
        volume.detach().numpy(),
        full_patch,
        str(normalization),
        float(contrast),
        float(brightness),
        float(gamma),
    )
    return torch.from_numpy(np.asarray(native_result, dtype=np.float32)), True


def _normalize_complete_image(volume: torch.Tensor, mode: str) -> torch.Tensor:
    mode_l = str(mode).lower()
    if mode_l in {"none", "raw", "identity"}:
        return volume
    if mode_l == "zscore":
        return (volume - volume.mean()) / volume.std(unbiased=False).clamp_min(1.0e-6)
    if mode_l == "minmax":
        lo = volume.min()
        hi = volume.max()
        return (volume - lo) / (hi - lo).clamp_min(1.0e-6)
    raise ValueError(f"unsupported image_normalization {mode!r}")


def _apply_value_augmentation_prefix_3d(
    config: FiberTrace3DConfig,
    volume: torch.Tensor,
    params: _Augment3DParams,
    *,
    profile_timings_ms: dict[str, float] | None = None,
) -> torch.Tensor:
    """Apply complete-patch normalization and the prefix through seeded noise."""
    def record(name: str, started: float) -> None:
        if profile_timings_ms is not None:
            profile_timings_ms[name] = (time.perf_counter() - started) * 1000.0

    volume = volume.float()
    backend = str(config.value_augmentation_backend)
    contrast = float(params.contrast) if config.augment_enabled else 1.0
    brightness = float(params.brightness) if config.augment_enabled else 0.0
    gamma = float(params.gamma) if config.augment_enabled else 1.0
    started = time.perf_counter()
    native_out, value_transform_native_used = _native_value_transform(
        volume,
        normalization=config.image_normalization,
        contrast=contrast,
        brightness=brightness,
        gamma=gamma,
        backend=backend,
    )
    if native_out is not None:
        out = native_out
        record("value_native_transform_ms", started)
    else:
        out = _normalize_complete_image(volume, config.image_normalization)
        record("value_normalize_ms", started)
        started = time.perf_counter()
        if contrast != 1.0:
            center = out.mean()
            out = (out - center) * contrast + center
        record("value_contrast_ms", started)
        started = time.perf_counter()
        if brightness != 0.0:
            out = out + brightness
        record("value_brightness_ms", started)
        started = time.perf_counter()
        if gamma != 1.0:
            lo = out.min()
            hi = out.max()
            scaled = ((out - lo) / (hi - lo).clamp_min(1.0e-6)).clamp(0.0, 1.0)
            out = torch.pow(scaled, gamma) * (hi - lo) + lo
        record("value_gamma_ms", started)
    if profile_timings_ms is not None:
        profile_timings_ms["value_native_transform_used"] = float(
            value_transform_native_used
        )
    if not config.augment_enabled:
        return out
    started = time.perf_counter()
    if params.noise_std > 0.0:
        gen = torch.Generator(device=out.device)
        gen.manual_seed(_stable_seed(config.seed, "value_noise", params.stream_index if params.value_seed_index is None else params.value_seed_index) % (2**63 - 1))
        out = out + torch.randn(out.shape, generator=gen, device=out.device) * float(params.noise_std)
    record("value_noise_ms", started)
    return out


def _apply_value_augmentation_suffix_3d(
    out: torch.Tensor,
    params: _Augment3DParams,
    *,
    blur_kernels: _BlurKernels3D | None = None,
    profile_timings_ms: dict[str, float] | None = None,
) -> torch.Tensor:
    """Apply the direct discrete Gaussian blur; no final validity mask exists."""
    started = time.perf_counter()
    kernels = _build_blur_kernels_3d(params) if blur_kernels is None else blur_kernels
    result = _apply_blur_kernels_3d(
        out.view(1, 1, *out.shape),
        (kernels,),
    )[0, 0]
    if profile_timings_ms is not None:
        profile_timings_ms["value_direct_blur_ms"] = (
            time.perf_counter() - started
        ) * 1000.0
    return result


def apply_value_augmentation_3d(
    config: FiberTrace3DConfig,
    volume: torch.Tensor,
    params: _Augment3DParams,
    *,
    profile_timings_ms: dict[str, float] | None = None,
) -> torch.Tensor:
    """Apply the canonical mask-free value-augmentation chain."""
    out = _apply_value_augmentation_prefix_3d(
        config,
        volume,
        params,
        profile_timings_ms=profile_timings_ms,
    )
    if not config.augment_enabled:
        return out
    return _apply_value_augmentation_suffix_3d(
        out,
        params,
        profile_timings_ms=profile_timings_ms,
    )


def apply_value_augmentation_suffix_batch_3d(
    volume: torch.Tensor,
    kernels: tuple[_BlurKernels3D | None, ...],
) -> torch.Tensor:
    """Apply already-constructed sequential blur kernels to a device batch."""
    return _apply_blur_kernels_3d(volume, kernels)


def apply_value_augmentation_batch_3d(
    config: FiberTrace3DConfig,
    volume: torch.Tensor,
    params: tuple[_Augment3DParams, ...],
    kernels: tuple[_BlurKernels3D | None, ...],
) -> torch.Tensor:
    """Apply the complete post-sampling value chain to one device batch."""
    if volume.ndim != 5 or int(volume.shape[1]) != 1:
        raise ValueError("batched value augmentation expects Bx1xZxYxX volumes")
    volume = volume.float()
    batch = int(volume.shape[0])
    if len(params) != batch or len(kernels) != batch:
        raise ValueError("batched value augmentation descriptor count mismatch")
    reduce_dims = (1, 2, 3, 4)
    mode = str(config.image_normalization).lower()
    zscore_mean: torch.Tensor | None = None
    zscore_std: torch.Tensor | None = None
    source_lo: torch.Tensor | None = None
    source_hi: torch.Tensor | None = None
    if mode in {"none", "raw", "identity"}:
        out = volume
    elif mode == "zscore":
        variance, zscore_mean = torch.var_mean(
            volume,
            dim=reduce_dims,
            correction=0,
            keepdim=True,
        )
        zscore_std = variance.sqrt().clamp_min_(1.0e-6)
        out = volume.sub(zscore_mean).div_(zscore_std)
    elif mode == "minmax":
        lo = volume.amin(dim=reduce_dims, keepdim=True)
        hi = volume.amax(dim=reduce_dims, keepdim=True)
        out = (volume - lo) / (hi - lo).clamp_min(1.0e-6)
    else:
        raise ValueError(f"unsupported image_normalization {config.image_normalization!r}")

    if config.augment_enabled:
        shape = (batch, 1, 1, 1, 1)
        needs_contrast = any(item.contrast != 1.0 for item in params)
        needs_brightness = any(item.brightness != 0.0 for item in params)
        needs_gamma = any(item.gamma != 1.0 for item in params)
        if needs_gamma and mode == "zscore":
            source_lo = volume.amin(dim=reduce_dims, keepdim=True)
            source_hi = volume.amax(dim=reduce_dims, keepdim=True)
        if needs_contrast:
            contrast = torch.as_tensor(
                [item.contrast for item in params], dtype=out.dtype, device=out.device
            ).view(shape)
            center = (
                torch.zeros_like(zscore_mean)
                if mode == "zscore" and zscore_mean is not None
                else out.mean(dim=reduce_dims, keepdim=True)
            )
            out = (out - center) * contrast + center
        else:
            contrast = torch.ones(shape, dtype=out.dtype, device=out.device)
        if needs_brightness:
            brightness = torch.as_tensor(
                [item.brightness for item in params], dtype=out.dtype, device=out.device
            ).view(shape)
            out = out + brightness
        else:
            brightness = torch.zeros(shape, dtype=out.dtype, device=out.device)
        if needs_gamma:
            gamma = torch.as_tensor(
                [item.gamma for item in params], dtype=out.dtype, device=out.device
            ).view(shape)
            if (
                mode == "zscore"
                and source_lo is not None
                and source_hi is not None
                and zscore_mean is not None
                and zscore_std is not None
            ):
                endpoint_a = (source_lo - zscore_mean) / zscore_std * contrast + brightness
                endpoint_b = (source_hi - zscore_mean) / zscore_std * contrast + brightness
                lo = torch.minimum(endpoint_a, endpoint_b)
                hi = torch.maximum(endpoint_a, endpoint_b)
            else:
                lo = out.amin(dim=reduce_dims, keepdim=True)
                hi = out.amax(dim=reduce_dims, keepdim=True)
            span = (hi - lo).clamp_min(1.0e-6)
            scaled = ((out - lo) / span).clamp(0.0, 1.0)
            out = torch.pow(scaled, gamma) * span + lo

        if any(item.noise_std > 0.0 for item in params):
            noise_parts: list[torch.Tensor] = []
            for item in params:
                if item.noise_std <= 0.0:
                    noise_parts.append(torch.zeros_like(out[0]))
                    continue
                generator = torch.Generator(device=out.device)
                generator.manual_seed(
                    _stable_seed(config.seed, "value_noise", item.stream_index if item.value_seed_index is None else item.value_seed_index) % (2**63 - 1)
                )
                noise_parts.append(
                    torch.randn(
                        out[0].shape,
                        generator=generator,
                        device=out.device,
                        dtype=out.dtype,
                    ) * float(item.noise_std)
                )
            out = out + torch.stack(noise_parts, dim=0)
    return _apply_blur_kernels_3d(out, kernels) if config.augment_enabled else out


def prepare_fiber_trace_3d_cuda_suffix(
    config: FiberTrace3DConfig,
    raw: FiberTrace3DRawSample,
    *,
    profile: bool = False,
    mode: str = "blur",
) -> FiberTrace3DRawSample:
    """Prepare descriptors for blur-only or complete CUDA value augmentation."""
    if mode not in {"blur", "all"}:
        raise ValueError("CUDA value augmentation mode must be 'blur' or 'all'")
    started = time.perf_counter()
    cpu_started = time.process_time()
    sample = raw.sample
    timings = dict(sample.profile_timings_ms or {}) if profile else None
    volume = sample.volume
    if mode == "blur":
        volume = _apply_value_augmentation_prefix_3d(
            config,
            sample.volume[0],
            raw.augmentation,
            profile_timings_ms=timings,
        ).view_as(sample.volume)
    kernel_started = time.perf_counter()
    kernels = _build_blur_kernels_3d(raw.augmentation) if config.augment_enabled else None
    if profile:
        assert timings is not None
        timings["value_kernel_build_ms"] = (time.perf_counter() - kernel_started) * 1000.0
        timings["value_cpu_prefix_ms"] = (time.perf_counter() - started) * 1000.0
        timings["value_cpu_prefix_cpu_ms"] = (
            time.process_time() - cpu_started
        ) * 1000.0
    return replace(
        raw,
        sample=replace(sample, volume=volume, profile_timings_ms=timings),
        blur_kernels=kernels,
    )


def augment_fiber_trace_3d_raw_sample(
    config: FiberTrace3DConfig,
    raw: FiberTrace3DRawSample,
    *,
    profile: bool = False,
) -> FiberTrace3DSample:
    """Finish a raw sample using the same stage-two path in every caller."""
    started = time.perf_counter()
    cpu_started = time.process_time()
    sample = raw.sample
    timings = dict(sample.profile_timings_ms or {}) if profile else None
    volume = apply_value_augmentation_3d(
        config,
        sample.volume[0],
        raw.augmentation,
        profile_timings_ms=timings,
    ).view_as(sample.volume)
    augment_ms = (time.perf_counter() - started) * 1000.0
    augment_cpu_ms = (time.process_time() - cpu_started) * 1000.0
    if profile:
        assert timings is not None
        timings["value_augmentation_ms"] = augment_ms
        timings["value_augmentation_cpu_ms"] = augment_cpu_ms
        timings["sample_total_ms"] = float(raw.load_wall_ms) + augment_ms
        timings["sample_cpu_ms"] = float(raw.load_cpu_ms) + augment_cpu_ms
    return replace(sample, volume=volume, profile_timings_ms=timings)


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds + 0.5), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _bounded_data_sample_index(sample_index: int, sample_index_limit: int | None) -> int:
    limit = 0 if sample_index_limit is None else int(sample_index_limit)
    if limit <= 0:
        return int(sample_index)
    return int(sample_index) % limit


class FiberTrace3DLoader:
    def __init__(self, config: FiberTrace3DConfig) -> None:
        if config.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if config.cp_margin_voxels < 0:
            raise ValueError("cp_margin_voxels must be >= 0")
        if min(config.patch_shape_zyx) <= 2 * int(config.cp_margin_voxels):
            raise ValueError(
                "cp_margin_voxels must leave at least one possible CP voxel "
                f"inside patch_shape_zyx={config.patch_shape_zyx}"
            )
        if config.presence_radius_voxels <= 0.0:
            raise ValueError("presence_radius_voxels must be > 0")
        if config.presence_target_mode not in {"binary", "anisotropic_gaussian"}:
            raise ValueError("unsupported presence_target_mode")
        if not math.isfinite(config.presence_gaussian_width_ratio) or config.presence_gaussian_width_ratio < 1.0:
            raise ValueError("presence_gaussian_width_ratio must be finite and >= 1")
        if not math.isfinite(config.presence_gaussian_truncate_sigma) or config.presence_gaussian_truncate_sigma <= 0.0:
            raise ValueError("presence_gaussian_truncate_sigma must be positive finite")
        if not math.isfinite(config.presence_sideband_weight) or config.presence_sideband_weight < 0.0:
            raise ValueError("presence_sideband_weight must be finite and >= 0")
        if config.presence_sideband_weight > 0.0:
            if config.presence_target_mode != "anisotropic_gaussian":
                raise ValueError("presence sidebands require anisotropic_gaussian targets")
            if not math.isfinite(config.presence_sideband_half_width_base) or config.presence_sideband_half_width_base <= 0.0:
                raise ValueError("presence_sideband_half_width_base must be positive finite")
            if not math.isfinite(config.presence_sideband_half_height_base) or config.presence_sideband_half_height_base <= 0.0:
                raise ValueError("presence_sideband_half_height_base must be positive finite")
        if config.augment_scale_min <= 0.0 or config.augment_scale_max <= 0.0:
            raise ValueError("augment_scale_min/max must be > 0")
        if config.augment_scale_min > config.augment_scale_max:
            raise ValueError("augment_scale_min must be <= augment_scale_max")
        if not (0.0 <= config.augment_flip_probability <= 1.0):
            raise ValueError("augment_flip_probability must be in [0,1]")
        if config.augment_smooth_displacement_mode not in {"none", "1d", "2d", "3d"}:
            raise ValueError("unsupported augment_smooth_displacement_mode")
        if any(v < 0.0 for v in config.augment_smooth_displacement_amplitude_zyx):
            raise ValueError("augment_smooth_displacement_amplitude_zyx values must be >= 0")
        if any(v <= 0.0 for v in config.augment_smooth_displacement_control_spacing_zyx):
            raise ValueError("augment_smooth_displacement_control_spacing_zyx values must be > 0")
        if int(config.geometry_tile_depth) < 0:
            raise ValueError("geometry_tile_depth must be >= 0")
        if str(config.geometry_backend) not in {"auto", "native", "eager"}:
            raise ValueError("geometry_backend must be auto, native, or eager")
        if str(config.value_augmentation_backend) not in {
            "auto",
            "native",
            "torch",
        }:
            raise ValueError(
                "value_augmentation_backend must be auto, native, or torch"
            )
        if not (0.0 <= config.augment_smooth_displacement_probability <= 1.0):
            raise ValueError("augment_smooth_displacement_probability must be in [0,1]")
        if not (0.0 <= config.augment_anisotropic_blur_probability <= 1.0):
            raise ValueError("augment_anisotropic_blur_probability must be in [0,1]")
        if config.augment_anisotropic_blur_sigma_along < 0.0:
            raise ValueError("augment_anisotropic_blur_sigma_along must be >= 0")
        if config.augment_anisotropic_blur_sigma_across < 0.0:
            raise ValueError("augment_anisotropic_blur_sigma_across must be >= 0")
        if config.augment_anisotropic_blur_orientation not in {"fiber", "random", "axis", "z", "y", "x"}:
            raise ValueError("unsupported augment_anisotropic_blur_orientation")
        self.config = config
        if config.embedding.enabled and any(not d.get("lasagna_manifest_path") for d in config.datasets):
            raise ValueError("embedding training requires lasagna_manifest_path for every dataset")
        if config.presence_target_mode == "anisotropic_gaussian":
            if any(not d.get("lasagna_manifest_path") for d in config.datasets):
                raise ValueError("anisotropic Gaussian presence targets require lasagna_manifest_path for every dataset")
            if any(not d.get("dense_supervision", False) for d in config.datasets):
                raise ValueError("anisotropic Gaussian presence targets require dense_supervision for every dataset")
        if config.refinement.orientation_enabled and config.presence_target_mode != "anisotropic_gaussian":
            raise ValueError("orientation refinement requires anisotropic_gaussian presence targets")
        if config.refinement.orientation_enabled and config.presence_gaussian_width_ratio <= 1.0:
            raise ValueError("orientation refinement requires Gaussian width ratio > 1")
        self._embedding_normal_records = {}
        self._missing_normal_samples = 0
        self._replacement_pools = {}
        self.records = self._load_records()
        self._refinement_store = None
        if config.refinement.enabled and (config.refinement.update_enabled or config.refinement_split == "train"):
            if any(not d.get("lasagna_manifest_path") for d in config.datasets):
                raise ValueError("refinement requires lasagna_manifest_path for every dataset")
            if config.sampling_strategy != "scroll_balanced_arclength":
                raise ValueError("refinement requires scroll_balanced_arclength and dense supervision")
            self._refinement_store = active_store()
            self._refinement_keys = []
            for record in self.records:
                key = hashlib.sha256(f"{config.refinement_split}:{record.volume_path}:{record.fiber_identity}".encode()).hexdigest()
                self._refinement_keys.append(key)
                self._refinement_store.register(key, record.fiber.line_points_zyx.astype(np.float64),
                    {"source": str(record.fiber.path), "split": config.refinement_split,
                     "scroll_id": record.dataset_config.get("scroll_id"),
                     "base_volume_path": record.volume_path, "base_shape_zyx": record.base_shape_zyx,
                     "source_identity": record.fiber_identity,
                     "normal_manifest": str(_resolve_path(record.dataset_config["lasagna_manifest_path"], config.config_dir)),
                     "control_span": record.fiber.metadata.get("training_control_span"),
                     "export": export_metadata(record.fiber)}, config.refinement)
            self._refinement_store.validate_split(self._refinement_keys, config.refinement_split)
        if config.sampling_strategy not in {"control_points", "scroll_balanced_arclength"}:
            raise ValueError("sampling_strategy must be control_points or scroll_balanced_arclength")
        for record in self.records:
            if "dense_supervision" in record.dataset_config and type(record.dataset_config["dense_supervision"]) is not bool:
                raise ValueError("dataset dense_supervision must be a boolean")
        self._arc_sampler = None
        if config.sampling_strategy == "scroll_balanced_arclength":
            if any(not _uses_dense_fiber_supervision(r) for r in self.records):
                raise ValueError("scroll_balanced_arclength requires dense supervision for every dataset")
            self._arc_sampler = ScrollArcSampler(self.records)
        if config.embedding.paired_patches and self._arc_sampler is None:
            raise ValueError("paired patches require scroll_balanced_arclength sampling")
        self._flat_offsets: list[int] = []
        flat_total = 0
        for record in self.records:
            self._flat_offsets.append(flat_total)
            flat_total += int(record.fiber.control_points_xyz.shape[0])
        self._flat_sample_count = int(flat_total)
        if self._flat_sample_count <= 0:
            raise ValueError("no control points found in configured fibers")
        self._random_pass_cache: dict[int, np.ndarray] = {}

    @property
    def sample_count(self) -> int:
        return int(self._flat_sample_count) * (2 if self.config.embedding.paired_patches else 1)

    def _load_records(self) -> list[_Record]:
        array_records = getattr(self.config, "_array_records", None)
        del array_records
        records: list[_Record] = []
        raw_array_records = None
        # Tests can attach _array_records after load_config by using from_mapping.
        if hasattr(self.config, "__dict__"):
            raw_array_records = getattr(self.config, "_array_records", None)
        if raw_array_records is not None:
            for index, raw in enumerate(raw_array_records):
                record = dict(raw)
                volume = np.asarray(record["volume"])
                if volume.ndim != 3:
                    raise ValueError("_array_records volume must be 3D")
                fiber = record["fiber"]
                if not isinstance(fiber, Vc3dFiber):
                    fibers = load_fiber_file(record["fiber_path"])
                    if len(fibers) != 1:
                        raise ValueError("_array_records fiber_path must resolve one fiber")
                    fiber = fibers[0]
                records.append(
                    _Record(
                        fiber=fiber,
                        volume=volume,
                        sampler=NumpyZarrCoordinateSampler(volume, level_spacing_base=1.0),
                        volume_path=str(record.get("volume_path", f"array:{index}")),
                        volume_scale=0,
                        volume_spacing_base=1.0,
                        base_shape_zyx=tuple(int(v) for v in volume.shape),
                        fiber_identity=_stable_digest("array", index, id(volume)),
                        dataset_config=record,
                    )
                )
            return records

        for dataset_config in self.config.datasets:
            dataset_record_start = len(records)
            source_fibers = 0
            ineligible_fibers = 0
            volume_path_raw = dataset_config.get("base_volume_path", dataset_config.get("volume_path"))
            if volume_path_raw is None:
                raise ValueError("dataset entry must define base_volume_path")
            volume_path = _resolve_path(volume_path_raw, self.config.config_dir)
            volume_scale = int(dataset_config.get("base_volume_scale", dataset_config.get("volume_scale", 0)))
            volume_spacing_base = float(1 << volume_scale)
            volume = _open_dataset_volume(dataset_config, self.config)
            sampler = make_coordinate_sampler(
                volume_path=volume_path,
                array=volume,
                level=volume_scale,
                level_spacing_base=volume_spacing_base,
                cache_root=self.config.volume_cache_dir,
                cache_budget_bytes=self.config.volume_cache_memory_bytes,
                io_threads=self.config.volume_io_threads,
                decode_threads=self.config.volume_decode_threads,
                sample_threads=self.config.volume_sample_threads,
                chunk_locality_fast_path=self.config.volume_chunk_locality_fast_path,
                collect_locality_stats=self.config.volume_collect_locality_stats,
            )
            base_shape_zyx = _validate_dataset_manifest(
                dataset_config,
                self.config,
                volume=volume,
                volume_path=volume_path,
            )
            transform_xyz, transform_identity = _dataset_fiber_transform(
                dataset_config,
                self.config,
            )
            for fiber_path in _resolve_fiber_paths(dataset_config, self.config):
                fibers = load_fiber_file(
                    fiber_path,
                    transform_xyz=transform_xyz,
                    transform_identity=transform_identity,
                )
                eligible_fibers = []
                for source_fiber_index, fiber in enumerate(fibers):
                    parts = training_fiber_spans(fiber)
                    source_fibers += 1
                    ineligible_fibers += not bool(parts)
                    eligible_fibers.extend((source_fiber_index, part) for part in parts)
                for source_fiber_index, fiber in eligible_fibers:
                    bad_control = _first_out_of_bounds_control_point(fiber, base_shape_zyx)
                    if bad_control is not None:
                        bad_index, bad_zyx = bad_control
                        print(
                            "fiber_trace_3d: skipping fiber with out-of-volume control point "
                            f"fiber_path='{fiber_path}' source_fiber_index={source_fiber_index} "
                            f"control_point_index={bad_index} "
                            f"control_point_zyx=({bad_zyx[0]:.3f}, {bad_zyx[1]:.3f}, {bad_zyx[2]:.3f}) "
                            f"base_shape_zyx={base_shape_zyx}",
                            flush=True,
                        )
                        continue
                    fiber_identity = _stable_digest(
                        "fiber3d",
                        str(fiber_path),
                        source_fiber_index,
                        fiber.metadata.get("training_control_span"),
                        transform_identity,
                        _round_json_array(np.asarray(fiber.line_points_xyz, dtype=np.float64)),
                    )
                    records.append(
                        _Record(
                            fiber=fiber,
                            volume=volume,
                            sampler=sampler,
                            volume_path=volume_path,
                            volume_scale=volume_scale,
                            volume_spacing_base=volume_spacing_base,
                            base_shape_zyx=base_shape_zyx,
                            fiber_identity=fiber_identity,
                            dataset_config=dict(dataset_config),
                        )
                    )
            retained = len(records) - dataset_record_start
            print(f"fiber_trace_3d training spans: scroll={dataset_config.get('scroll_id', '?')} "
                  f"sources={source_fibers} eligible_runs={retained} "
                  f"ineligible_sources={ineligible_fibers} modes=cspline,trace tails=excluded",
                  flush=True)
            if not retained:
                raise ValueError(f"dataset {dataset_config.get('fiber_glob', dataset_config)} "
                                 "has no eligible training spans (requires cspline/trace "
                                 "between control points, or explicit NML edges)")
        if not records:
            raise ValueError("no fiber records loaded")
        return records

    def _descriptor_for_flat_index(self, flat_index: int) -> tuple[_Record, int, int]:
        flat = int(flat_index)
        if flat < 0 or flat >= self._flat_sample_count:
            raise IndexError(f"flat sample index {flat} out of range")
        record_index = int(np.searchsorted(self._flat_offsets, flat, side="right") - 1)
        record_offset = int(self._flat_offsets[record_index])
        cp_index = flat - record_offset
        return self.records[record_index], record_index, int(cp_index)

    def _random_flat_index(self, sample_index: int) -> int:
        sample_count = int(self._flat_sample_count)
        position = int(sample_index)
        pass_index = math.floor(position / sample_count)
        offset = position % sample_count
        if pass_index not in self._random_pass_cache:
            rng = np.random.default_rng(_stable_seed(self.config.seed, "sample_order", pass_index))
            self._random_pass_cache[pass_index] = rng.permutation(sample_count).astype(np.int64)
        return int(self._random_pass_cache[pass_index][offset])

    def descriptor_for_sample_index(
        self,
        sample_index: int,
        *,
        sample_mode: str = "random",
    ) -> tuple[_Record, int, int | FiberLocation]:
        if self.config.embedding.paired_patches:
            original = self._descriptor_for_unpaired_index(sample_index // 2, sample_mode=sample_mode)
            if sample_index % 2:
                return self._arc_sampler.sample_within(self._rng(sample_index // 2, "pair_partner"), [original[1]])
            return original
        return self._descriptor_for_unpaired_index(sample_index, sample_mode=sample_mode)

    def _descriptor_for_unpaired_index(self, sample_index, *, sample_mode):
        if sample_mode == "random":
            if self._arc_sampler is not None:
                return self._arc_sampler.sample(self._rng(sample_index, "scroll_arclength"))
            return self._descriptor_for_flat_index(self._random_flat_index(sample_index))
        if sample_mode == "flat":
            return self._descriptor_for_flat_index(int(sample_index) % int(self._flat_sample_count))
        raise ValueError("sample_mode must be 'random' or 'flat'")

    def _rng(self, sample_index: int, purpose: str) -> np.random.Generator:
        return np.random.default_rng(_stable_seed(self.config.seed, purpose, int(sample_index)))

    def _control_point_volume_zyx(self, record: _Record, cp_index: int | FiberLocation) -> np.ndarray:
        if isinstance(cp_index, FiberLocation):
            return np.asarray(cp_index.xyz[::-1], dtype=np.float64) / float(record.volume_spacing_base)
        cp_zyx_base = np.asarray(record.fiber.control_points_zyx[int(cp_index)], dtype=np.float64)
        return cp_zyx_base / float(record.volume_spacing_base)

    def _line_tangent_volume_zyx(self, record: _Record, cp_index: int | FiberLocation) -> np.ndarray:
        line_points = np.asarray(record.fiber.line_points_zyx, dtype=np.float64) / float(
            record.volume_spacing_base
        )
        if line_points.shape[0] < 2:
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        if isinstance(cp_index, FiberLocation):
            prev_index, next_index = cp_index.segment_index, cp_index.segment_index + 1
        else:
            line_index = control_point_line_index(record.fiber, int(cp_index))
            prev_index = max(0, int(line_index) - 1)
            next_index = min(int(line_points.shape[0]) - 1, int(line_index) + 1)
        tangent = line_points[next_index] - line_points[prev_index]
        norm = float(np.linalg.norm(tangent))
        if not math.isfinite(norm) or norm <= 1.0e-12:
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        return tangent / norm

    def _smooth_control_radius_zyx(self) -> np.ndarray:
        patch = np.asarray(self.config.patch_shape_zyx, dtype=np.float64)
        min_scale = max(float(self.config.augment_scale_min), 1.0e-6)
        shift = np.asarray(self.config.augment_shift_zyx, dtype=np.float64)
        amplitude = np.asarray(
            self.config.augment_smooth_displacement_amplitude_zyx,
            dtype=np.float64,
        )
        radius = (patch - 1.0) / min_scale + shift + amplitude + 8.0
        return np.maximum(radius, np.asarray([8.0, 8.0, 8.0], dtype=np.float64))

    def _make_smooth_params(
        self,
        sample_index: int,
        rng: np.random.Generator,
    ) -> _SmoothDisplacement3DParams | None:
        mode = str(self.config.augment_smooth_displacement_mode).lower()
        amplitude = np.asarray(
            self.config.augment_smooth_displacement_amplitude_zyx,
            dtype=np.float64,
        )
        if (
            (not self.config.augment_enabled)
            or mode == "none"
            or float(np.max(amplitude)) <= 0.0
            or float(self.config.augment_smooth_displacement_probability) <= 0.0
            or float(rng.random()) > float(self.config.augment_smooth_displacement_probability)
        ):
            return None
        spacing = np.asarray(
            self.config.augment_smooth_displacement_control_spacing_zyx,
            dtype=np.float64,
        )
        radius = self._smooth_control_radius_zyx()
        control_min = -radius
        counts = np.maximum(2, np.ceil((2.0 * radius) / spacing).astype(np.int64) + 1)
        control_rng = np.random.default_rng(
            _stable_seed(self.config.seed, "smooth3d", sample_index)
        )

        def control_1d(axis: int, component: int) -> np.ndarray:
            values = control_rng.uniform(
                -float(amplitude[component]),
                float(amplitude[component]),
                size=(int(counts[axis]),),
            )
            return values.astype(np.float32)

        def control_2d(axis_a: int, axis_b: int, component: int) -> np.ndarray:
            values = control_rng.uniform(
                -float(amplitude[component]),
                float(amplitude[component]),
                size=(int(counts[axis_a]), int(counts[axis_b])),
            )
            return values.astype(np.float32)

        if mode == "1d":
            controls = (control_1d(2, 0),)
        elif mode == "2d":
            controls = (control_2d(1, 2, 0),)
        elif mode == "3d":
            controls = (
                control_2d(1, 2, 0),
                control_2d(0, 2, 1),
                control_2d(0, 1, 2),
            )
        else:
            raise ValueError(f"unsupported augment_smooth_displacement_mode {mode!r}")
        return _SmoothDisplacement3DParams(
            mode=mode,
            amplitude_zyx=amplitude.astype(np.float64),
            control_spacing_zyx=spacing.astype(np.float64),
            control_min_zyx=control_min.astype(np.float64),
            controls=tuple(controls),
        )

    def _anisotropic_axis_zyx(
        self,
        record: _Record,
        cp_index: int | FiberLocation,
        source_to_output_zyx: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        if (
            (not self.config.augment_enabled)
            or float(self.config.augment_anisotropic_blur_probability) <= 0.0
            or (
                float(self.config.augment_anisotropic_blur_sigma_along) <= 0.0
                and float(self.config.augment_anisotropic_blur_sigma_across) <= 0.0
            )
            or float(rng.random()) > float(self.config.augment_anisotropic_blur_probability)
        ):
            return None
        orientation = str(self.config.augment_anisotropic_blur_orientation).lower()
        if orientation in {"axis", "z"}:
            axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        elif orientation == "y":
            axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        elif orientation == "x":
            axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        elif orientation == "random":
            axis = rng.normal(size=3).astype(np.float64)
        else:
            tangent = self._line_tangent_volume_zyx(record, cp_index)
            axis = tangent @ np.asarray(source_to_output_zyx, dtype=np.float64).T
        norm = float(np.linalg.norm(axis))
        if not math.isfinite(norm) or norm <= 1.0e-12:
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        axis = axis / norm
        roll = math.radians(float(self.config.augment_anisotropic_blur_roll_degrees))
        if roll != 0.0 and orientation == "fiber":
            # Roll is represented as a deterministic small axis-angle perturbation
            # around a stable perpendicular vector so the principal blur axis stays
            # fiber-aligned up to the configured roll angle.
            perp = np.cross(axis, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
            if float(np.linalg.norm(perp)) <= 1.0e-12:
                perp = np.cross(axis, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
            rot = _axis_angle_rotation(perp, float(rng.uniform(-roll, roll)))
            axis = axis @ rot.T
            axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
        return axis.astype(np.float64)

    def _sample_augment_params(
        self,
        record: _Record,
        cp_index: int | FiberLocation,
        sample_index: int,
        value_index: int | None = None,
    ) -> _Augment3DParams:
        patch = np.asarray(self.config.patch_shape_zyx, dtype=np.int64)
        margin = int(self.config.cp_margin_voxels)
        center = (patch.astype(np.float64) - 1.0) * 0.5
        cp_volume = self._control_point_volume_zyx(record, cp_index)
        rng = self._rng(sample_index, "augment3d")

        if self.config.augment_enabled:
            shift = np.asarray(self.config.augment_shift_zyx, dtype=np.float64)
            low = np.maximum(np.floor(center - shift), margin)
            high = np.minimum(np.ceil(center + shift), patch - margin - 1)
            cp_local = np.asarray(
                [
                    rng.integers(int(lo), int(hi) + 1)
                    if int(hi) >= int(lo)
                    else int(round(center_axis))
                    for lo, hi, center_axis in zip(low, high, center, strict=True)
                ],
                dtype=np.float64,
            )
            scale = float(rng.uniform(self.config.augment_scale_min, self.config.augment_scale_max))
            flip = np.ones(3, dtype=np.float64)
            if self.config.augment_flip_probability > 0.0:
                flip[rng.random(3) < self.config.augment_flip_probability] = -1.0
            max_angle = math.radians(float(self.config.augment_rotation_degrees))
            if max_angle > 0.0:
                axis = rng.normal(size=3)
                angle = float(rng.uniform(-max_angle, max_angle))
                rotation = _axis_angle_rotation(axis, angle)
            else:
                rotation = np.eye(3, dtype=np.float64)
            source_to_output = rotation @ np.diag(flip * scale)
            geometry_rng = rng
            if value_index is not None:
                rng = self._rng(value_index, "paired_value")
            brightness = (
                float(rng.uniform(-self.config.augment_brightness, self.config.augment_brightness))
                if self.config.augment_brightness > 0.0
                else 0.0
            )
            contrast = float(rng.uniform(self.config.augment_contrast_min, self.config.augment_contrast_max))
            gamma = float(rng.uniform(self.config.augment_gamma_min, self.config.augment_gamma_max))
            noise_std = float(self.config.augment_noise_std)
            blur_sigma = float(rng.uniform(0.0, self.config.augment_blur_sigma))
            directional_sigma_along = float(
                rng.uniform(0.0, self.config.augment_anisotropic_blur_sigma_along)
            )
            directional_sigma_across = float(
                rng.uniform(0.0, self.config.augment_anisotropic_blur_sigma_across)
            )
            if value_index is not None:
                rng = geometry_rng
        else:
            cp_local = center
            source_to_output = np.eye(3, dtype=np.float64)
            brightness = 0.0
            contrast = 1.0
            gamma = 1.0
            noise_std = 0.0
            blur_sigma = 0.0
            directional_sigma_along = 0.0
            directional_sigma_across = 0.0
        smooth = self._make_smooth_params(sample_index, rng)
        anisotropic_axis = self._anisotropic_axis_zyx(
            record,
            cp_index,
            source_to_output,
            rng if value_index is None else self._rng(value_index, "paired_blur_axis"),
        )

        return _Augment3DParams(
            stream_index=int(sample_index),
            cp_local_zyx=cp_local.astype(np.float64),
            cp_volume_zyx=cp_volume.astype(np.float64),
            source_to_output_zyx=source_to_output.astype(np.float64),
            output_to_source_zyx=np.linalg.inv(source_to_output).astype(np.float64),
            brightness=brightness,
            contrast=contrast,
            gamma=gamma,
            noise_std=noise_std,
            blur_sigma=blur_sigma,
            smooth=smooth,
            anisotropic_blur_sigma_along=directional_sigma_along
            if anisotropic_axis is not None
            else 0.0,
            anisotropic_blur_sigma_across=directional_sigma_across
            if anisotropic_axis is not None
            else 0.0,
            anisotropic_blur_axis_zyx=anisotropic_axis,
            value_seed_index=value_index,
        )

    def _output_points_to_source_volume_np(
        self,
        output_points_zyx: np.ndarray,
        params: _Augment3DParams,
    ) -> np.ndarray:
        device = torch.device("cpu")
        points = torch.as_tensor(output_points_zyx, dtype=torch.float32, device=device)
        cp_local = torch.as_tensor(params.cp_local_zyx, dtype=torch.float32, device=device)
        cp_volume = torch.as_tensor(params.cp_volume_zyx, dtype=torch.float32, device=device)
        inv_matrix = torch.as_tensor(params.output_to_source_zyx, dtype=torch.float32, device=device)
        rel_output = points - cp_local.view(1, 3)
        rel_aug = torch.einsum("...i,ji->...j", rel_output, inv_matrix)
        rel_source = _smooth_backward_rel_zyx(rel_aug, params.smooth)
        return (cp_volume.view(1, 3) + rel_source).detach().cpu().numpy().astype(np.float64)

    def _source_bbox_for_maps(
        self,
        record: _Record,
        params: _Augment3DParams,
    ) -> tuple[np.ndarray, np.ndarray]:
        patch = np.asarray(self.config.patch_shape_zyx, dtype=np.float64)
        corners = np.asarray(
            list(itertools.product(*[(0.0, float(size - 1.0)) for size in patch])),
            dtype=np.float64,
        )
        rel_output = corners - params.cp_local_zyx.reshape(1, 3)
        del rel_output
        coords = self._output_points_to_source_volume_np(corners, params)
        amplitude = (
            np.asarray(params.smooth.amplitude_zyx, dtype=np.float64)
            if params.smooth is not None
            else np.zeros((3,), dtype=np.float64)
        )
        start = np.floor(np.min(coords, axis=0) - 2.0 - amplitude).astype(np.int64)
        end = np.ceil(np.max(coords, axis=0) + 3.0 + amplitude).astype(np.int64)
        del record
        return start, end

    def _actual_source_bbox(
        self,
        record: _Record,
        params: _Augment3DParams,
    ) -> tuple[np.ndarray, np.ndarray]:
        start, end = self._source_bbox_for_maps(record, params)
        if self.config.round_source_to_chunk_boundaries:
            start, end = _round_bbox_to_chunks(record.volume, start, end)
        return start, end

    def _prefetch_source_bbox(
        self,
        record: _Record,
        cp_index: int | FiberLocation,
    ) -> tuple[np.ndarray, np.ndarray]:
        cp = self._control_point_volume_zyx(record, cp_index)
        patch = np.asarray(self.config.patch_shape_zyx, dtype=np.float64)
        if self.config.augment_enabled:
            min_scale = max(float(self.config.augment_scale_min), 1.0e-6)
            shift = np.asarray(self.config.augment_shift_zyx, dtype=np.float64)
            if (
                self.config.augment_smooth_displacement_mode != "none"
                and float(self.config.augment_smooth_displacement_probability) > 0.0
            ):
                smooth_amp = np.asarray(
                    self.config.augment_smooth_displacement_amplitude_zyx,
                    dtype=np.float64,
                )
            else:
                smooth_amp = np.zeros((3,), dtype=np.float64)
        else:
            min_scale = 1.0
            shift = np.zeros((3,), dtype=np.float64)
            smooth_amp = np.zeros((3,), dtype=np.float64)
        radius = (
            float(np.linalg.norm(patch - 1.0)) / min_scale
            + float(np.max(shift))
            + float(np.max(smooth_amp))
            + 4.0
        )
        start = np.floor(cp - radius).astype(np.int64)
        end = np.ceil(cp + radius + 1.0).astype(np.int64)
        return start, end

    def _prefetch_envelope_bbox_selected(
        self,
        record: _Record,
        cp_index: int | FiberLocation,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        start_selected, end_selected = self._prefetch_source_bbox(record, cp_index)
        selected_shape = np.asarray(
            _volume_shape_zyx(record.volume, label="3D prefetch selected base volume"),
            dtype=np.int64,
        )
        start = np.maximum(np.asarray(start_selected, dtype=np.int64), 0)
        end = np.minimum(np.asarray(end_selected, dtype=np.int64), selected_shape)
        extent = np.maximum(end - start, 0)
        valid_voxels = int(np.prod(extent, dtype=np.int64)) if bool(np.all(extent > 0)) else 0
        return int(valid_voxels), start.astype(np.int64), end.astype(np.int64)

    def _build_geometry_maps(
        self,
        params: _Augment3DParams,
        *,
        device: torch.device,
        forward_start_zyx: np.ndarray | None = None,
        forward_end_zyx: np.ndarray | None = None,
        profile_timings_ms: dict[str, float] | None = None,
        volume_spacing_base: float | None = None,
        base_shape_zyx: tuple[int, int, int] | None = None,
    ) -> _GeometryMaps3D:
        del forward_start_zyx, forward_end_zyx
        patch_shape = tuple(int(v) for v in self.config.patch_shape_zyx)

        def record(name: str, started: float) -> None:
            if profile_timings_ms is not None:
                profile_timings_ms[name] = profile_timings_ms.get(name, 0.0) + (
                    time.perf_counter() - started
                ) * 1000.0

        backend = str(self.config.geometry_backend)
        native_eligible = (
            device.type == "cpu"
            and volume_spacing_base is not None
            and base_shape_zyx is not None
        )
        native_function = None
        if native_eligible and backend in {"auto", "native"}:
            try:
                from vc.volume import generate_fiber3d_coords

                native_function = generate_fiber3d_coords
            except ImportError as exc:
                if backend == "native":
                    raise RuntimeError(
                        "geometry_backend='native' requires a rebuilt volume-cartographer "
                        "binding with generate_fiber3d_coords"
                    ) from exc
        elif backend == "native":
            raise RuntimeError(
                "geometry_backend='native' requires CPU generation with volume scale and shape"
            )

        if native_function is not None:
            smooth = params.smooth
            smooth_mode = "none" if smooth is None else str(smooth.mode)
            smooth_controls = [] if smooth is None else [
                np.ascontiguousarray(control, dtype=np.float32)
                for control in smooth.controls
            ]
            smooth_origin = np.zeros((3,), dtype=np.float32) if smooth is None else np.ascontiguousarray(
                smooth.control_min_zyx, dtype=np.float32
            )
            smooth_spacing = np.ones((3,), dtype=np.float32) if smooth is None else np.ascontiguousarray(
                smooth.control_spacing_zyx, dtype=np.float32
            )
            started = time.perf_counter()
            coords_base, valid_base = native_function(
                patch_shape,
                np.ascontiguousarray(params.cp_local_zyx, dtype=np.float32),
                np.ascontiguousarray(params.cp_volume_zyx, dtype=np.float32),
                np.ascontiguousarray(params.output_to_source_zyx, dtype=np.float32),
                smooth_mode,
                smooth_controls,
                smooth_origin,
                smooth_spacing,
                float(volume_spacing_base),
                tuple(int(v) for v in base_shape_zyx),
            )
            record("geometry_native_ms", started)
            if profile_timings_ms is not None:
                profile_timings_ms["geometry_native_used"] = 1.0
                profile_timings_ms["geometry_tiles"] = 1.0
            return _GeometryMaps3D(
                backward_source_zyx=None,
                params=params,
                backward_source_base_zyx=np.asarray(coords_base, dtype=np.float32),
                valid_base=np.asarray(valid_base, dtype=np.uint8).astype(bool, copy=False),
            )
        if profile_timings_ms is not None:
            profile_timings_ms["geometry_native_used"] = 0.0

        start = time.perf_counter()
        cp_local = torch.as_tensor(params.cp_local_zyx, dtype=torch.float32, device=device)
        cp_volume = torch.as_tensor(params.cp_volume_zyx, dtype=torch.float32, device=device)
        inv_matrix = torch.as_tensor(params.output_to_source_zyx, dtype=torch.float32, device=device)
        record("geometry_metadata_ms", start)
        start = time.perf_counter()
        controls = _smooth_control_tensors(params.smooth, device=device)
        record("geometry_controls_ms", start)

        requested_depth = int(self.config.geometry_tile_depth)
        tile_depth = patch_shape[0] if requested_depth <= 0 else min(
            requested_depth, patch_shape[0]
        )
        tiled = tile_depth < patch_shape[0]
        fuse_sampling_maps = tiled and volume_spacing_base is not None and base_shape_zyx is not None
        backward_source = (
            torch.empty((*patch_shape, 3), dtype=torch.float32, device=device)
            if tiled and not fuse_sampling_maps
            else None
        )
        coords_base = (
            np.empty((*patch_shape, 3), dtype=np.float32) if fuse_sampling_maps else None
        )
        valid_base = np.empty(patch_shape, dtype=np.bool_) if fuse_sampling_maps else None
        base_shape = (
            np.asarray(base_shape_zyx, dtype=np.float32) if fuse_sampling_maps else None
        )
        for z0 in range(0, patch_shape[0], tile_depth):
            z1 = min(z0 + tile_depth, patch_shape[0])
            tile_shape = (z1 - z0, patch_shape[1], patch_shape[2])
            start = time.perf_counter()
            local = _zyx_grid(tile_shape, device=device)
            if z0:
                local[..., 0].add_(float(z0))
            record("geometry_grid_ms", start)
            start = time.perf_counter()
            rel_output = local - cp_local.view(1, 1, 1, 3)
            record("geometry_center_ms", start)
            start = time.perf_counter()
            rel_aug = torch.einsum("...i,ji->...j", rel_output, inv_matrix)
            record("geometry_affine_ms", start)
            start = time.perf_counter()
            rel_source = _smooth_backward_rel_zyx(rel_aug, params.smooth, controls)
            record("geometry_smooth_ms", start)
            start = time.perf_counter()
            translated = cp_volume.view(1, 1, 1, 3) + rel_source
            record("geometry_translate_ms", start)
            if fuse_sampling_maps:
                assert coords_base is not None and valid_base is not None and base_shape is not None
                start_fused = time.perf_counter()
                tile_base = coords_base[z0:z1]
                np.multiply(
                    translated.detach().cpu().numpy(),
                    float(volume_spacing_base),
                    out=tile_base,
                )
                record("geometry_scale_ms", start_fused)
                start_fused = time.perf_counter()
                tile_valid = np.isfinite(tile_base).all(axis=-1)
                tile_valid &= (tile_base[..., 0] >= 0.0) & (
                    tile_base[..., 0] <= float(base_shape[0] - 1.0)
                )
                tile_valid &= (tile_base[..., 1] >= 0.0) & (
                    tile_base[..., 1] <= float(base_shape[1] - 1.0)
                )
                tile_valid &= (tile_base[..., 2] >= 0.0) & (
                    tile_base[..., 2] <= float(base_shape[2] - 1.0)
                )
                valid_base[z0:z1] = tile_valid
                record("geometry_valid_ms", start_fused)
            elif tiled:
                assert backward_source is not None
                backward_source[z0:z1].copy_(translated)
            else:
                backward_source = translated
        assert backward_source is not None or coords_base is not None
        if profile_timings_ms is not None:
            profile_timings_ms["geometry_tiles"] = float(
                math.ceil(patch_shape[0] / tile_depth)
            )
        return _GeometryMaps3D(
            backward_source_zyx=backward_source,
            params=params,
            backward_source_base_zyx=coords_base,
            valid_base=valid_base,
        )

    def _source_points_to_output_np(
        self,
        points_volume_zyx: np.ndarray,
        params: _Augment3DParams | _GeometryMaps3D,
        *, array_backend: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(params, _GeometryMaps3D):
            params = params.params
        if array_backend:
            # Sparse refinement queries share the same ordered shear mapping;
            # native SciPy interpolation avoids dozens of tiny Torch dispatches.
            relative = np.asarray(points_volume_zyx, dtype=np.float64) - params.cp_volume_zyx
            mapped = params.cp_local_zyx + _smooth_forward_rel_zyx(relative, params.smooth) @ params.source_to_output_zyx.T
            valid = np.isfinite(mapped).all(axis=1) & (mapped >= 0).all(axis=1) & (mapped <= np.asarray(self.config.patch_shape_zyx)-1).all(axis=1)
            return mapped, valid
        points_np = np.asarray(points_volume_zyx, dtype=np.float32)
        points = torch.as_tensor(
            points_np,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        cp_volume = torch.as_tensor(params.cp_volume_zyx, dtype=torch.float32, device=points.device)
        cp_local = torch.as_tensor(params.cp_local_zyx, dtype=torch.float32, device=points.device)
        rel_source = points - cp_volume.view(1, 3)
        rel_aug = _smooth_forward_rel_zyx(rel_source, params.smooth)
        matrix = torch.as_tensor(params.source_to_output_zyx, dtype=torch.float32, device=points.device)
        mapped = cp_local.view(1, 3) + torch.einsum("...i,ji->...j", rel_aug, matrix)
        patch = torch.as_tensor(self.config.patch_shape_zyx, dtype=torch.float32, device=points.device)
        valid = (
            torch.isfinite(mapped).all(dim=-1)
            & (mapped[..., 0] >= 0.0)
            & (mapped[..., 0] <= patch[0] - 1.0)
            & (mapped[..., 1] >= 0.0)
            & (mapped[..., 1] <= patch[1] - 1.0)
            & (mapped[..., 2] >= 0.0)
            & (mapped[..., 2] <= patch[2] - 1.0)
        )
        return (
            mapped.detach().cpu().numpy().astype(np.float32),
            valid.detach().cpu().numpy().astype(bool),
        )

    def _sample_volume_patch(
        self,
        record: _Record,
        geometry: _GeometryMaps3D,
        *,
        device: torch.device,
        profile_timings_ms: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, None]:
        start = time.perf_counter()
        if geometry.backward_source_base_zyx is not None:
            coords_base = geometry.backward_source_base_zyx
        else:
            assert geometry.backward_source_zyx is not None
            coords_selected = geometry.backward_source_zyx.detach().cpu().numpy().astype(
                np.float32, copy=False
            )
            coords_base = np.ascontiguousarray(
                coords_selected * float(record.volume_spacing_base)
            )
        if profile_timings_ms is not None:
            profile_timings_ms["coord_to_numpy_ms"] = (
                time.perf_counter() - start
            ) * 1000.0
        start = time.perf_counter()
        if geometry.valid_base is not None:
            valid = geometry.valid_base
        else:
            base_shape = np.asarray(record.base_shape_zyx, dtype=np.float32)
            valid = np.isfinite(coords_base).all(axis=-1)
            valid &= (coords_base[..., 0] >= 0.0) & (coords_base[..., 0] <= float(base_shape[0] - 1.0))
            valid &= (coords_base[..., 1] >= 0.0) & (coords_base[..., 1] <= float(base_shape[1] - 1.0))
            valid &= (coords_base[..., 2] >= 0.0) & (coords_base[..., 2] <= float(base_shape[2] - 1.0))
        if profile_timings_ms is not None:
            profile_timings_ms["coord_valid_ms"] = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        result = record.sampler.sample_coord_batch(coords_base, np.ascontiguousarray(valid))
        if profile_timings_ms is not None:
            profile_timings_ms["volume_sample_ms"] = (time.perf_counter() - start) * 1000.0
            profile_timings_ms["coordinate_points"] = float(coords_base.size // 3)
            profile_timings_ms["valid_coordinate_points"] = float(np.count_nonzero(valid))
            for key, value in result.stats.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    profile_timings_ms[f"vc_{key}"] = float(value)
        start = time.perf_counter()
        # The native sampler initializes uncovered output to zero. Keep its
        # compact uint8 result through shared storage and H2D; complete CUDA
        # augmentation performs the only float32 conversion.
        sampled_np = np.asarray(result.image, dtype=np.uint8)
        sampled = torch.as_tensor(sampled_np, dtype=torch.uint8, device=device)
        if profile_timings_ms is not None:
            profile_timings_ms["volume_tensor_ms"] = (time.perf_counter() - start) * 1000.0
        return sampled, None

    def _apply_value_augmentation(
        self,
        volume: torch.Tensor,
        params: _Augment3DParams,
    ) -> torch.Tensor:
        return apply_value_augmentation_3d(self.config, volume, params)

    def _line_window_for_labels(
        self,
        record: _Record,
        cp_index: int | FiberLocation,
        params: _Augment3DParams,
    ) -> tuple[np.ndarray, int]:
        line_points = np.asarray(record.fiber.line_points_zyx, dtype=np.float64) / float(
            record.volume_spacing_base
        )
        line_index = (cp_index.segment_index if isinstance(cp_index, FiberLocation)
                      else control_point_line_index(record.fiber, int(cp_index)))
        if line_points.shape[0] < 2:
            raise ValueError("fiber line must contain at least two points")
        if record.dataset_config.get("dense_supervision", False) or isinstance(cp_index, FiberLocation):
            # Later parts of the same fiber can re-enter the patch. Arc-window
            # truncation would falsely label them negative.
            return line_points, line_index
        cumulative = np.concatenate(
            [
                np.zeros((1,), dtype=np.float64),
                np.cumsum(np.linalg.norm(np.diff(line_points, axis=0), axis=1)),
            ]
        )
        patch_diag = float(np.linalg.norm(np.asarray(self.config.patch_shape_zyx, dtype=np.float64)))
        anchor = float(cumulative[line_index])
        start_arc = anchor - patch_diag
        end_arc = anchor + patch_diag
        start = max(0, int(np.searchsorted(cumulative, start_arc, side="right") - 2))
        end = min(
            int(line_points.shape[0]),
            int(np.searchsorted(cumulative, end_arc, side="left") + 3),
        )
        if end - start < 2:
            start = max(0, line_index - 1)
            end = min(int(line_points.shape[0]), line_index + 2)
        del params
        return line_points[start:end], line_index - start

    def _cp_only_target_spec(
        self,
        record: _Record,
        cp_index: int | FiberLocation,
        params: _Augment3DParams,
        segment_starts: np.ndarray,
        segment_ends: np.ndarray,
        segment_bbox_lo: np.ndarray,
        segment_bbox_hi: np.ndarray,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tangent_out = self._target_tangent_output_zyx(record, cp_index, params)
        return (
            _TARGET_MODE_CP_ONLY,
            segment_starts,
            segment_ends,
            segment_bbox_lo,
            segment_bbox_hi,
            tangent_out.astype(np.float32),
        )

    def _target_tangent_output_zyx(
        self,
        record: _Record,
        cp_index: int | FiberLocation,
        params: _Augment3DParams,
    ) -> np.ndarray:
        tangent_src = self._line_tangent_volume_zyx(record, cp_index).astype(np.float64)
        tangent_points = np.stack(
            [
                np.asarray(params.cp_volume_zyx, dtype=np.float64) - tangent_src,
                np.asarray(params.cp_volume_zyx, dtype=np.float64) + tangent_src,
            ],
            axis=0,
        )
        tangent_out_points, tangent_valid = self._source_points_to_output_np(
            tangent_points,
            params,
        )
        if bool(tangent_valid.all()):
            tangent_out = tangent_out_points[1] - tangent_out_points[0]
        else:
            tangent_out = tangent_src @ np.asarray(params.source_to_output_zyx, dtype=np.float64).T
        tangent_norm = float(np.linalg.norm(tangent_out))
        if not math.isfinite(tangent_norm) or tangent_norm <= 1.0e-12:
            tangent_out = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            tangent_out = tangent_out / tangent_norm
        return tangent_out.astype(np.float32)

    def _line_segment_spec(
        self,
        record: _Record,
        cp_index: int | FiberLocation,
        params: _Augment3DParams,
        *,
        patch_shape: tuple[int, int, int],
        radius: float,
        profile_timings_ms: dict[str, float] | None = None,
        refinement_orientation: np.ndarray | None = None,
        refinement_material: np.ndarray | None = None,
        refinement_baseline: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        start = time.perf_counter()
        line_points, _local_line_index = self._line_window_for_labels(
            record,
            cp_index,
            params,
        )
        if profile_timings_ms is not None:
            profile_timings_ms["target_line_window_ms"] = (
                time.perf_counter() - start
            ) * 1000.0
        if line_points.shape[0] < 2:
            raise ValueError("label line window must contain at least two points")
        start = time.perf_counter()
        points_out, _points_valid = self._source_points_to_output_np(line_points, params)
        segment_start = points_out[:-1].astype(np.float32)
        segment_end = points_out[1:].astype(np.float32)
        segment_mid_source = (line_points[:-1] + line_points[1:]) * 0.5
        segment_orientation = None
        normal_sample_mid_base = segment_mid_source * float(record.volume_spacing_base)
        if refinement_orientation is not None:
            from .refinement import interpolate_orientation, orientation_angles
            if (refinement_material is None or refinement_baseline is None or
                    len(refinement_orientation) != len(line_points) or
                    len(refinement_material) != len(line_points)):
                raise ValueError("refined target orientation/material geometry mismatch")
            midpoint_state = interpolate_orientation(
                refinement_orientation,
                refinement_material,
                0.5 * (refinement_material[:-1] + refinement_material[1:]),
            )
            segment_orientation = orientation_angles(midpoint_state)
            baseline_nodes = interpolate(
                refinement_baseline[:, :3], refinement_baseline[:, 3], refinement_material)
            normal_sample_mid_base = 0.5 * (baseline_nodes[:-1] + baseline_nodes[1:])
        valid_segments = np.isfinite(segment_start).all(axis=1) & np.isfinite(segment_end).all(axis=1)
        if record.label_breaks:
            valid_segments[list(record.label_breaks)] = False
        if record.dataset_config.get("dense_supervision", False) or isinstance(cp_index, FiberLocation):
            # Whole fibers can be long: reject disjoint segment boxes in NumPy
            # before exact clipping. Keep source order and boundary-touching
            # segments; this changes neither accepted segments nor target math.
            valid_segments &= np.all(
                np.maximum(segment_start, segment_end) >= np.full(3, -radius, dtype=np.float64), axis=1,
            )
            valid_segments &= np.all(
                np.minimum(segment_start, segment_end)
                <= np.asarray(patch_shape, dtype=np.float64) - 1.0 + radius, axis=1,
            )
        segment_start = segment_start[valid_segments]
        segment_end = segment_end[valid_segments]
        segment_mid_source = segment_mid_source[valid_segments]
        segment_tangent_source = np.diff(line_points, axis=0)[valid_segments]
        normal_sample_mid_base = normal_sample_mid_base[valid_segments]
        if segment_orientation is not None:
            segment_orientation = segment_orientation[valid_segments]
        if profile_timings_ms is not None:
            profile_timings_ms["target_points_to_output_ms"] = (
                time.perf_counter() - start
            ) * 1000.0
        if segment_start.shape[0] == 0:
            raise ValueError("label line window has no finite output-space segments")
        start = time.perf_counter()
        patch_hi = np.asarray(patch_shape, dtype=np.float64) - 1.0
        starts: list[np.ndarray] = []
        ends: list[np.ndarray] = []
        bbox_los: list[np.ndarray] = []
        bbox_his: list[np.ndarray] = []
        source_midpoints: list[np.ndarray] = []
        normal_sample_midpoints: list[np.ndarray] = []
        source_tangents: list[np.ndarray] = []
        orientations: list[float] = []
        bbox_voxels = 0
        orientation_values = (np.zeros(len(segment_start)) if segment_orientation is None
                              else segment_orientation)
        for p0_raw, p1_raw, midpoint_source, normal_midpoint, tangent_source, angle in zip(
            segment_start, segment_end, segment_mid_source, normal_sample_mid_base,
            segment_tangent_source, orientation_values, strict=True
        ):
            clipped = _clip_segment_to_aabb(
                p0_raw,
                p1_raw,
                lo=-np.full((3,), radius, dtype=np.float64),
                hi=patch_hi + radius,
            )
            if clipped is None:
                continue
            p0 = clipped[0].astype(np.float32)
            p1 = clipped[1].astype(np.float32)
            vec = (p1 - p0).astype(np.float32)
            length2 = np.float32(np.sum(vec * vec))
            if not np.isfinite(length2) or float(length2) <= 1.0e-12:
                continue
            lo = np.floor(np.minimum(p0, p1) - radius).astype(np.int64)
            hi = np.ceil(np.maximum(p0, p1) + radius).astype(np.int64) + 1
            lo = np.maximum(lo, 0)
            hi = np.minimum(hi, np.asarray(patch_shape, dtype=np.int64))
            if bool(np.any(hi <= lo)):
                continue
            bbox_voxels += int(np.prod(hi - lo))
            starts.append(p0)
            ends.append(p1)
            bbox_los.append(lo.astype(np.int64))
            bbox_his.append(hi.astype(np.int64))
            source_midpoints.append(midpoint_source)
            normal_sample_midpoints.append(normal_midpoint)
            source_tangents.append(tangent_source)
            orientations.append(float(angle))
        if profile_timings_ms is not None:
            profile_timings_ms["target_clip_ms"] = (time.perf_counter() - start) * 1000.0
            profile_timings_ms["target_spec_segments"] = float(len(starts))
            profile_timings_ms["target_spec_bbox_voxels"] = float(bbox_voxels)
        if not starts:
            raise ValueError("label line window has no patch-overlapping segments")
        normals_out = None
        if self.config.presence_target_mode == "anisotropic_gaussian":
            midpoints = np.stack(source_midpoints, axis=0).astype(np.float64)
            normals_base = self._sample_native_normals(
                record,
                np.stack(normal_sample_midpoints, axis=0).astype(np.float64),
                profile_timings_ms,
            )
            if refinement_orientation is not None:
                from .refinement import rotate_normals_about_tangent
                # Rotation is defined in the unaugmented source frame. Clipping
                # and uniform scale do not change the segment direction.
                normals_base = rotate_normals_about_tangent(
                    normals_base,
                    np.stack(source_tangents, axis=0),
                    np.asarray(orientations),
                )
            normals_out = self._transform_surface_normals_to_output(
                midpoints,
                normals_base,
                params,
            )
            output_segments = np.stack(ends, axis=0) - np.stack(starts, axis=0)
            tangents = output_segments / np.linalg.norm(output_segments, axis=1, keepdims=True)
            perpendicular = normals_out - tangents * np.sum(normals_out * tangents, axis=1, keepdims=True)
            if np.any(np.linalg.norm(perpendicular, axis=1) <= 1.0e-6):
                raise InvalidNormalSampleError("Gaussian target normal is parallel to its fiber segment")
        return (
            np.stack(starts, axis=0).astype(np.float32),
            np.stack(ends, axis=0).astype(np.float32),
            np.stack(bbox_los, axis=0).astype(np.int64),
            np.stack(bbox_his, axis=0).astype(np.int64),
            normals_out,
        )

    def _build_target_spec(
        self,
        record: _Record,
        cp_index: int | FiberLocation,
        params: _Augment3DParams,
        profile_timings_ms: dict[str, float] | None = None,
        *,
        refinement_orientation: np.ndarray | None = None,
        refinement_baseline: np.ndarray | None = None,
        refinement_material: np.ndarray | None = None,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        patch_shape = tuple(int(v) for v in self.config.patch_shape_zyx)
        radius = float(self.config.presence_radius_voxels)
        if self.config.presence_target_mode == "anisotropic_gaussian":
            radius *= (
                float(self.config.presence_gaussian_width_ratio)
                * float(self.config.presence_gaussian_truncate_sigma)
            )
        if self.config.presence_sideband_weight > 0.0:
            radius = max(
                radius,
                float(self.config.presence_sideband_half_width_base)
                / float(record.volume_spacing_base),
            )
        segment_starts, segment_ends, segment_bbox_lo, segment_bbox_hi, segment_normals = self._line_segment_spec(
            record,
            cp_index,
            params,
            patch_shape=patch_shape,
            radius=radius,
            profile_timings_ms=profile_timings_ms,
            refinement_orientation=refinement_orientation,
            refinement_material=refinement_material,
            refinement_baseline=refinement_baseline,
        )
        if not _uses_dense_fiber_supervision(record):
            target = self._cp_only_target_spec(
                record,
                cp_index,
                params,
                segment_starts,
                segment_ends,
                segment_bbox_lo,
                segment_bbox_hi,
            )
            return (*target, None)
        return (
            _TARGET_MODE_DENSE_LINE,
            segment_starts,
            segment_ends,
            segment_bbox_lo,
            segment_bbox_hi,
            self._target_tangent_output_zyx(record, cp_index, params),
            segment_normals,
        )

    def _sample_native_normals(self, record, points, timings=None):
        """Shared native validity/encoding-aware path for embedding and refinement."""
        from vc.volume import NormalSampler
        manifest = _resolve_path(record.dataset_config["lasagna_manifest_path"], self.config.config_dir)
        if manifest not in self._embedding_normal_records:
            opened = time.perf_counter()
            self._embedding_normal_records[manifest] = NormalSampler(
                manifest, self.config.volume_cache_dir or "")
            if timings is not None:
                timings["embedding_normal_open_ms"] = (time.perf_counter()-opened)*1000
        sampler = self._embedding_normal_records[manifest]

        xyz = np.ascontiguousarray(points[:, ::-1], dtype=np.float32)
        started = time.perf_counter()
        normals, valid = sampler.sample(xyz, threads=1)
        if timings is not None:
            timings["embedding_normals_ms"] = timings.get("embedding_normals_ms", 0.) + (time.perf_counter()-started)*1000
        if not valid.all():
            raise InvalidNormalSampleError(
                f"invalid normal (including zero grad_mag) "
                f"manifest={manifest} invalid={np.count_nonzero(~valid)}/{len(valid)}")
        return normals[:, ::-1]

    def _transform_surface_normals_to_output(self, points_volume_zyx, normals_zyx, params):
        """Push surface-normal covectors through the exact sparse augmentation map."""

        points = np.asarray(points_volume_zyx, dtype=np.float64)
        normals = np.asarray(normals_zyx, dtype=np.float64)
        if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("segment normal transformation requires matching Nx3 points/normals")
        offsets = np.concatenate((np.eye(3), -np.eye(3)), axis=0) * 0.5
        query = (points[None, :, :] + offsets[:, None, :]).reshape(-1, 3)
        mapped, _ = self._source_points_to_output_np(query, params, array_backend=True)
        mapped = mapped.reshape(6, len(points), 3)
        jacobian = (mapped[:3] - mapped[3:]).transpose(1, 2, 0)
        if not np.isfinite(jacobian).all() or np.any(np.linalg.cond(jacobian) > 1.0e8):
            raise InvalidNormalSampleError("singular augmentation Jacobian for Gaussian target normal")
        transformed = np.linalg.solve(jacobian.transpose(0, 2, 1), normals[..., None])[..., 0]
        lengths = np.linalg.norm(transformed, axis=1)
        if not np.isfinite(transformed).all() or np.any(lengths <= 1.0e-10):
            raise InvalidNormalSampleError("invalid transformed Gaussian target normal")
        return (transformed / lengths[:, None]).astype(np.float32)

    def _build_embedding_geometry(self, record, record_index, params, timings=None):
        cfg = self.config.embedding
        if not cfg.enabled:
            return None

        eligible = build_eligibility(
            self.config.patch_shape_zyx, cfg.output_stride,
            lambda points: self._output_points_to_source_volume_np(points, params) * record.volume_spacing_base,
            record.fiber.line_points_zyx, lambda points: self._sample_native_normals(record, points, timings), record.base_shape_zyx)
        return EmbeddingGeometry(eligible, _stable_seed(self.config.seed, params.stream_index, "embedding"),
                                 cfg.pairs(self.config.patch_shape_zyx), cfg.output_stride, cfg.weight,
                                 cfg.objective, cfg.negatives_per_anchor, cfg.temperature, cfg.margin,
                                 short_range_channels=cfg.short_range_channels)

    def load_raw_sample(
        self, sample_index: int, *, sample_index_limit: int | None = None,
        sample_mode: str = "random", device: torch.device | str = "cpu",
        profile: bool = False,
    ) -> FiberTrace3DRawSample:
        """Replace invalid-normal samples within their original scroll/group.

        Queue identity remains canonical. Replacement draws and augmentations
        are keyed only by that identity and attempt, not worker/rank state.
        """
        data_index = _bounded_data_sample_index(int(sample_index), sample_index_limit)
        if self.config.embedding.paired_patches and sample_index_limit and sample_index_limit % 2:
            raise ValueError("paired sample index limit must be even")
        original = self.descriptor_for_sample_index(data_index, sample_mode=sample_mode)
        descriptor = original
        for attempt in range(129):
            augmentation_index = int(sample_index) if attempt == 0 else (
                _stable_seed(self.config.seed, "missing_normal_augmentation", sample_index, attempt) % (2**63-1))
            try:
                return self._load_raw_sample_once(sample_index, sample_index_limit=sample_index_limit,
                    sample_mode=sample_mode, device=device, profile=profile,
                    descriptor=descriptor, augmentation_index=augmentation_index)
            except InvalidNormalSampleError as exc:
                self._missing_normal_samples += 1
                if self._missing_normal_samples == 1 or self._missing_normal_samples % 100 == 0:
                    print(f"fiber loader: skip invalid-normal sample={sample_index} attempt={attempt} "
                          f"worker_skipped={self._missing_normal_samples} fiber={descriptor[0].fiber.path} {exc}", flush=True)
                if attempt == 128:
                    raise RuntimeError(f"no normal-supported sample after 128 replacements: "
                        f"sample={sample_index} original_fiber={original[0].fiber.path}: {exc}") from exc
                descriptor = self._replacement_descriptor(original, sample_index, attempt+1, sample_mode)
        raise AssertionError("unreachable")

    def _replacement_descriptor(self, original, sample_index, attempt, sample_mode):
        record = original[0]
        if self.config.embedding.paired_patches:
            # Never replace one partner with a different fiber.
            return self._arc_sampler.sample_within(
                self._rng(sample_index, f"pair_normal_retry_{attempt}"), [original[1]])
        def group(r):
            return (r.dataset_config.get("scroll_id", r.volume_path),
                    r.dataset_config.get("sampling_group"))
        key = group(record)
        if key not in self._replacement_pools:
            self._replacement_pools[key] = tuple(i for i,r in enumerate(self.records) if group(r) == key)
        ids = self._replacement_pools[key]
        rng = np.random.default_rng(_stable_seed(self.config.seed, "missing_normal_replacement", sample_index, attempt))
        if sample_mode == "random" and self._arc_sampler is not None:
            return self._arc_sampler.sample_within(rng, ids)
        counts = np.cumsum([len(self.records[i].fiber.control_points_xyz) for i in ids])
        flat = int(rng.integers(int(counts[-1])))
        local = int(np.searchsorted(counts, flat, side="right"))
        cp = flat - (int(counts[local-1]) if local else 0)
        return self.records[ids[local]], ids[local], cp

    @staticmethod
    def _refined_location(curve, location):
        segment = min(len(curve) - 2, max(0, int(np.searchsorted(curve[:, 3], location) - 1)))
        fraction = (location - curve[segment, 3]) / (curve[segment+1, 3] - curve[segment, 3])
        point = (1-fraction)*curve[segment, :3] + fraction*curve[segment+1, :3]
        return FiberLocation(segment, fraction, tuple(point[::-1]))

    def _load_raw_sample_once(
        self,
        sample_index: int,
        *,
        sample_index_limit: int | None = None,
        sample_mode: str = "random",
        device: torch.device | str = "cpu",
        profile: bool = False,
        descriptor=None,
        augmentation_index: int | None = None,
    ) -> FiberTrace3DRawSample:
        stream_index = int(sample_index)
        total_start = time.perf_counter()
        cpu_total_start = time.process_time()
        timings: dict[str, float] | None = {} if profile else None
        resolved_device = torch.device(device)
        start = time.perf_counter()
        data_index = _bounded_data_sample_index(stream_index, sample_index_limit)
        record, record_index, cp_index = descriptor
        fixed_test = self._refinement_store is not None and self.config.refinement_split == "test"
        # Freeze all crop/augmentation decisions on the original GT descriptor.
        fixed_params = (self._sample_augment_params(record, cp_index, augmentation_index,
            value_index=stream_index // 2 if self.config.embedding.paired_patches else None)
            if fixed_test else None)
        refinement_current = None
        refinement_orientation = None
        refinement_baseline = None
        if self._refinement_store is not None:
            key = self._refinement_keys[record_index]
            if fixed_test:
                baseline = self._refinement_store.baseline(key)
                refinement_current = resample(baseline[:, :3], baseline[:, 3],
                    max(2, int(np.ceil(baseline[-1, 3] / self.config.refinement.spacing_base)) + 1))
            else:
                refinement_current, refinement_orientation = self._refinement_store.read_state(key)
            if fixed_test:
                refinement_orientation = np.zeros((len(refinement_current), 2), dtype=np.float64)
                refinement_orientation[:, 0] = 1.0
            refinement_baseline = self._refinement_store.baseline(key)
            reference_arc = arclength(record.fiber.line_points_zyx.astype(np.float64))
            if isinstance(cp_index, FiberLocation):
                location = reference_arc[cp_index.segment_index] + cp_index.fraction * (
                    reference_arc[cp_index.segment_index + 1] - reference_arc[cp_index.segment_index])
            else:
                location = reference_arc[control_point_line_index(record.fiber, cp_index)]
            cp_index = self._refined_location(refinement_current, location)
            record = replace(record, fiber=replace(record.fiber,
                line_points_xyz=refinement_current[:, :3][:, ::-1].astype(np.float32)))
        if timings is not None:
            timings["descriptor_ms"] = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        params = fixed_params if fixed_test else self._sample_augment_params(record, cp_index, augmentation_index,
            value_index=stream_index // 2 if self.config.embedding.paired_patches else None)
        refinement_geometry = None
        if refinement_current is not None and self.config.refinement.update_enabled:
            patch_points, valid = self._source_points_to_output_np(
                refinement_current[:, :3] / record.volume_spacing_base, params)
            # Include one outside/context node on either side. Separate visits
            # to a curved fiber may be disjoint; never span an outside gap.
            near = np.flatnonzero(valid)
            if len(near):
                groups = np.split(near, np.flatnonzero(np.diff(near) > 1) + 1)
                group = min(groups, key=lambda g: np.min(np.abs(g - cp_index.segment_index)))
                section_start, section_stop = max(0, int(group[0])-1), min(len(refinement_current), int(group[-1])+2)
                current = refinement_current[section_start:section_stop].copy()
                if len(current) >= 3:
                    baseline = refinement_baseline
                    lo = max(0, int(np.searchsorted(baseline[:, 3], current[0, 3])-1))
                    hi = min(len(baseline), int(np.searchsorted(baseline[:, 3], current[-1, 3]))+2)
                    local_baseline = baseline[lo:hi].copy()
                    observation_start = section_start
                    if fixed_test:
                        local_orientation = refinement_orientation[section_start:section_stop].copy()
                        key, current, local_orientation = self._refinement_store.test_patch(key, data_index, params,
                            self.config.patch_shape_zyx, record.volume_spacing_base,
                            section_start, current, local_baseline, local_orientation)
                        observation_start = 0
                        # Keep other visits of the same fiber as original GT;
                        # never replace the whole target with a clipped polyline.
                        refinement_current[section_start:section_stop] = current
                        refinement_orientation[section_start:section_stop] = local_orientation
                        cp_index = self._refined_location(refinement_current, location)
                        record = replace(record, fiber=replace(record.fiber,
                            line_points_xyz=refinement_current[:, :3][:, ::-1].astype(np.float32)),
                            label_breaks=tuple(i for i in (section_start-1, section_stop-1)
                                              if 0 <= i < len(refinement_current)-1))
                    ref = interpolate(baseline[:, :3], baseline[:, 3], current[:, 3])
                    try:
                        normals = self._sample_native_normals(record, ref, timings)
                    except InvalidNormalSampleError:
                        # Invalid support means no geometric evidence. Preserve
                        # valid CT/GT training samples, without a fake normal.
                        normals = None
                    if normals is not None:
                        refinement_geometry = RefinementObservation(
                            key, observation_start, current, local_baseline, normals, params,
                            record.volume_spacing_base, self.config.patch_shape_zyx,
                            self.config.refinement, self.config.embedding.output_stride,
                            independent_piece=fixed_test,
                            orientation=refinement_orientation[section_start:section_stop].copy(),
                            orientation_sigma_thickness_output=float(self.config.presence_radius_voxels),
                            orientation_width_ratio=float(self.config.presence_gaussian_width_ratio),
                        )
        if timings is not None:
            timings["augment_params_ms"] = (time.perf_counter() - start) * 1000.0
        embedding_start = time.perf_counter()
        embedding_geometry = self._build_embedding_geometry(record, record_index, params, timings)
        if self.config.embedding.paired_patches:
            embedding_geometry = replace(embedding_geometry, pair_id=stream_index // 2,
                pair_member=stream_index % 2, fiber_id=f"{record_index}:{record.fiber.path}")
        if timings is not None:
            timings["embedding_geometry_ms"] = (time.perf_counter() - embedding_start) * 1000.0
        start = time.perf_counter()
        geometry = self._build_geometry_maps(
            params,
            device=resolved_device,
            profile_timings_ms=timings,
            volume_spacing_base=float(record.volume_spacing_base),
            base_shape_zyx=record.base_shape_zyx,
        )
        if timings is not None:
            timings["geometry_ms"] = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        volume, valid = self._sample_volume_patch(
            record,
            geometry,
            device=resolved_device,
            profile_timings_ms=timings,
        )
        if timings is not None:
            timings["sample_volume_total_ms"] = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        (
            target_mode,
            segment_starts,
            segment_ends,
            segment_bbox_lo,
            segment_bbox_hi,
            target_tangent,
            target_segment_normals,
        ) = self._build_target_spec(
            record,
            cp_index,
            params,
            timings,
            refinement_orientation=(refinement_orientation
                                    if self.config.refinement.orientation_enabled else None),
            refinement_baseline=(refinement_baseline
                                 if self.config.refinement.orientation_enabled else None),
            refinement_material=(refinement_current[:, 3]
                                 if (refinement_current is not None and
                                     self.config.refinement.orientation_enabled) else None),
        )
        if timings is not None:
            timings["target_spec_total_ms"] = (time.perf_counter() - start) * 1000.0
        load_wall_ms = (time.perf_counter() - total_start) * 1000.0
        load_cpu_ms = (time.process_time() - cpu_total_start) * 1000.0
        if timings is not None:
            timings["raw_sample_total_ms"] = load_wall_ms
            timings["raw_sample_cpu_ms"] = load_cpu_ms
        sample = FiberTrace3DSample(
            presence_target_mode=self.config.presence_target_mode,
            presence_sideband_weight=self.config.presence_sideband_weight,
            volume_spacing_base=float(record.volume_spacing_base),
            embedding_geometry=embedding_geometry,
            refinement_geometry=refinement_geometry,
            stream_index=stream_index,
            data_index=int(data_index),
            record_index=int(record_index),
            fiber_path="" if record.fiber.path is None else str(record.fiber.path),
            control_point_index=(-1 if isinstance(cp_index, FiberLocation) else int(cp_index)),
            volume=volume.view(1, *self.config.patch_shape_zyx),
            valid_mask=None,
            cp_local_zyx=torch.as_tensor(params.cp_local_zyx, dtype=torch.float32, device=resolved_device),
            crop_origin_zyx=torch.as_tensor(
                np.floor(params.cp_volume_zyx - params.cp_local_zyx).astype(np.float32),
                dtype=torch.float32,
                device=resolved_device,
            ),
            target_mode=int(target_mode),
            target_segment_starts_zyx=torch.as_tensor(
                segment_starts,
                dtype=torch.float32,
                device=resolved_device,
            ),
            target_segment_ends_zyx=torch.as_tensor(
                segment_ends,
                dtype=torch.float32,
                device=resolved_device,
            ),
            target_segment_bbox_lo_zyx=torch.as_tensor(
                segment_bbox_lo,
                dtype=torch.long,
            ),
            target_segment_bbox_hi_zyx=torch.as_tensor(
                segment_bbox_hi,
                dtype=torch.long,
            ),
            target_tangent_zyx=torch.as_tensor(
                target_tangent,
                dtype=torch.float32,
                device=resolved_device,
            ),
            target_segment_normals_zyx=None
            if target_segment_normals is None
            else torch.as_tensor(
                target_segment_normals,
                dtype=torch.float32,
                device=resolved_device,
            ),
            profile_timings_ms=timings,
        )
        return FiberTrace3DRawSample(
            sample=sample,
            augmentation=params,
            load_wall_ms=load_wall_ms,
            load_cpu_ms=load_cpu_ms,
        )

    def augment_raw_sample(
        self,
        raw: FiberTrace3DRawSample,
        *,
        profile: bool = False,
    ) -> FiberTrace3DSample:
        return augment_fiber_trace_3d_raw_sample(self.config, raw, profile=profile)

    def load_sample(
        self,
        sample_index: int,
        *,
        sample_index_limit: int | None = None,
        sample_mode: str = "random",
        device: torch.device | str = "cpu",
        profile: bool = False,
    ) -> FiberTrace3DSample:
        raw = self.load_raw_sample(
            sample_index,
            sample_index_limit=sample_index_limit,
            sample_mode=sample_mode,
            device=device,
            profile=profile,
        )
        return self.augment_raw_sample(raw, profile=profile)

    def load_batch(
        self,
        start_sample_index: int,
        *,
        sample_index_limit: int | None = None,
        sample_mode: str = "random",
        device: torch.device | str = "cpu",
        profile: bool = False,
    ) -> FiberTrace3DBatch:
        total_start = time.perf_counter()
        wall_start_ns = time.time_ns()
        cpu_total_start = time.process_time()
        samples = [
            self.load_sample(
                int(start_sample_index) + offset,
                sample_index_limit=sample_index_limit,
                sample_mode=sample_mode,
                device=device,
                profile=profile,
            )
            for offset in range(int(self.config.batch_size))
        ]
        return collate_fiber_trace_3d_samples(
            samples,
            profile=profile,
            total_start=total_start,
            cpu_total_start=cpu_total_start,
            wall_start_ns=wall_start_ns,
        )

    def _chunk_requests_and_valid_voxels_for_sample_index(
        self,
        sample_index: int,
        *,
        sample_index_limit: int | None = None,
        sample_mode: str = "random",
    ) -> tuple[int, list[ZarrChunkRequest]]:
        stream_index = int(sample_index)
        data_index = _bounded_data_sample_index(stream_index, sample_index_limit)
        record, _record_index, cp_index = self.descriptor_for_sample_index(
            data_index,
            sample_mode=sample_mode,
        )
        valid_voxels, start_selected, end_selected = self._prefetch_envelope_bbox_selected(
            record,
            cp_index,
        )
        return int(valid_voxels), record.sampler.chunk_requests_for_bbox(
            start_selected,
            end_selected,
        )

    def chunk_requests_for_sample_index(
        self,
        sample_index: int,
        *,
        sample_index_limit: int | None = None,
        sample_mode: str = "random",
    ) -> list[ZarrChunkRequest]:
        _valid_voxels, requests = self._chunk_requests_and_valid_voxels_for_sample_index(
            sample_index,
            sample_index_limit=sample_index_limit,
            sample_mode=sample_mode,
        )
        return requests

    def prefetch(
        self,
        start_sample_index: int,
        sample_count: int,
        *,
        workers: int | None = None,
        sample_index_limit: int | None = None,
        sample_mode: str = "random",
    ) -> dict[str, Any]:
        total_samples = int(sample_count)
        if total_samples < 0:
            raise ValueError("sample_count must be >= 0")
        worker_count = max(1, int(self.config.prefetch_workers if workers is None else workers))
        producer_count = min(
            max(1, int(self.config.prefetch_sampler_workers)),
            max(1, total_samples),
        )
        retry_seconds = float(self.config.volume_cache_retry_seconds)
        if retry_seconds <= 0.0:
            retry_seconds = 600.0
        counters = _PrefetchCounters3D()
        start_sample = int(start_sample_index)
        end_sample = start_sample + total_samples
        seen: set[tuple[str, str]] = set()
        completed_chunks: set[tuple[str, str]] = set()
        chunk_waiters: dict[tuple[str, str], set[int]] = {}
        sample_pending: dict[int, set[tuple[str, str]]] = {}
        sample_requests_closed: set[int] = set()
        complete_raw_samples: set[int] = set()
        next_safe_raw_sample = start_sample
        start = time.perf_counter()
        print(
            "fiber_trace_3d prefetch: "
            f"samples={total_samples} workers={worker_count} "
            f"sampler_workers={producer_count} sample_mode={sample_mode} "
            f"sample_index_limit={0 if sample_index_limit is None else int(sample_index_limit)} "
            "mode=augmentation_envelope_bbox",
            flush=True,
        )

        def print_progress(*, final: bool = False) -> None:
            elapsed = max(time.perf_counter() - start, 1.0e-9)
            sample_rate = counters.samples_done / elapsed
            remaining_samples = total_samples - counters.samples_done
            sample_eta = (
                remaining_samples / sample_rate
                if remaining_samples > 0 and sample_rate > 0.0
                else 0.0
            )
            download_rate = counters.download_done / elapsed
            if counters.samples_done > 0 and counters.samples_done < total_samples:
                observed_chunks_per_sample = counters.unique_chunks_seen / max(counters.samples_done, 1)
                observed_download_ratio = counters.queued_for_download / max(counters.unique_chunks_seen, 1)
                estimated_total_downloads = counters.queued_for_download + int(
                    math.ceil(remaining_samples * observed_chunks_per_sample * observed_download_ratio)
                )
            else:
                estimated_total_downloads = counters.queued_for_download
            remaining_downloads = estimated_total_downloads - counters.download_done
            download_eta = (
                remaining_downloads / download_rate
                if remaining_downloads > 0 and download_rate > 0.0
                else 0.0
            )
            mib = counters.bytes_downloaded / (1024.0 * 1024.0)
            mib_s = mib / elapsed
            bar_width = 24
            sample_filled = int(math.floor(bar_width * counters.samples_done / max(total_samples, 1)))
            sample_bar = "#" * sample_filled + "-" * (bar_width - sample_filled)
            if counters.queued_for_download > 0:
                download_filled = int(
                    math.floor(bar_width * counters.download_done / max(counters.queued_for_download, 1))
                )
            else:
                download_filled = 0
            download_bar = "#" * download_filled + "-" * (bar_width - download_filled)
            if counters.samples_done < total_samples:
                sample_prefix = (
                    f"fiber_trace_3d prefetch samples[{sample_bar}] "
                    f"{counters.samples_done}/{total_samples} "
                    f"eta={_format_seconds(sample_eta)} "
                )
            else:
                sample_prefix = "fiber_trace_3d prefetch "
            line = (
                sample_prefix
                + f"downloads[{download_bar}] {counters.download_done}/{counters.queued_for_download} "
                f"eta={_format_seconds(download_eta)} "
                f"idx={counters.max_exclusive_sample_index} "
                f"chunks={counters.unique_chunks_seen} hits={counters.cache_hits} "
                f"queued={counters.queued_download_futures} transfers={worker_count} "
                f"samplers={producer_count} "
                f"valid_voxels={counters.valid_voxels} "
                f"skipped={counters.samples_skipped} "
                f"missing={counters.known_missing + counters.newly_missing} "
                f"downloaded={counters.downloaded} errors={counters.download_errors} "
                f"{mib:.1f} MiB {mib_s:.1f} MiB/s"
            )
            if counters.first_error:
                error = counters.first_error
                if len(error) > 160:
                    error = error[:157] + "..."
                line += f" first_error={error!r}"
            print(line, end="\n" if final else "\r", flush=True)

        def build_requests(sample_index: int) -> tuple[int, int, list[ZarrChunkRequest]]:
            valid_voxels, requests = self._chunk_requests_and_valid_voxels_for_sample_index(
                sample_index,
                sample_index_limit=sample_index_limit,
                sample_mode=sample_mode,
            )
            return int(sample_index), int(valid_voxels), requests

        def advance_safe_prefix() -> None:
            nonlocal next_safe_raw_sample
            while next_safe_raw_sample < end_sample and next_safe_raw_sample in complete_raw_samples:
                counters.max_exclusive_sample_index = max(
                    counters.max_exclusive_sample_index,
                    int(next_safe_raw_sample) + 1,
                )
                next_safe_raw_sample += 1

        def mark_sample_complete(raw_sample_index: int) -> None:
            if raw_sample_index not in sample_requests_closed:
                return
            if sample_pending.get(raw_sample_index):
                return
            complete_raw_samples.add(raw_sample_index)
            advance_safe_prefix()

        def close_sample_requests(raw_sample_index: int) -> None:
            sample_requests_closed.add(raw_sample_index)
            mark_sample_complete(raw_sample_index)

        def mark_chunk_complete(identity: tuple[str, str]) -> None:
            if identity in completed_chunks:
                return
            completed_chunks.add(identity)
            for raw_sample_index in chunk_waiters.pop(identity, set()):
                pending = sample_pending.get(raw_sample_index)
                if pending is None:
                    continue
                pending.discard(identity)
                mark_sample_complete(raw_sample_index)

        def classify_or_submit(
            request: ZarrChunkRequest,
            raw_sample_index: int,
            download_queue: list[tuple[int, int, tuple[str, str]]],
            queued_downloads: dict[tuple[str, str], tuple[int, int, ZarrChunkRequest]],
        ) -> None:
            nonlocal download_sequence
            identity = (request.store_identity, request.key)
            if identity in completed_chunks:
                return
            sample_pending.setdefault(raw_sample_index, set()).add(identity)
            chunk_waiters.setdefault(identity, set()).add(raw_sample_index)
            if identity in seen:
                queued = queued_downloads.get(identity)
                if queued is not None and int(raw_sample_index) < queued[0]:
                    download_sequence += 1
                    queued_downloads[identity] = (int(raw_sample_index), download_sequence, queued[2])
                    heapq.heappush(download_queue, (int(raw_sample_index), download_sequence, identity))
                return
            seen.add(identity)
            counters.unique_chunks_seen += 1
            if _existing_data_path(request) is not None:
                counters.cache_hits += 1
                mark_chunk_complete(identity)
                return
            empty_path = _empty_marker_path(request)
            if empty_path is not None and empty_path.is_file():
                counters.known_missing += 1
                mark_chunk_complete(identity)
                return
            counters.queued_for_download += 1
            download_sequence += 1
            queued_downloads[identity] = (int(raw_sample_index), download_sequence, request)
            heapq.heappush(download_queue, (int(raw_sample_index), download_sequence, identity))

        def consume_download_result(request: ZarrChunkRequest, result: dict[str, Any]) -> None:
            counters.download_done += 1
            status = str(result.get("status", "error"))
            if status == "cache_hit":
                counters.cache_hits += 1
                mark_chunk_complete((request.store_identity, request.key))
            elif status == "known_missing":
                counters.known_missing += 1
                mark_chunk_complete((request.store_identity, request.key))
            elif status == "new_missing":
                counters.newly_missing += 1
                mark_chunk_complete((request.store_identity, request.key))
            elif status == "downloaded":
                counters.downloaded += 1
                counters.bytes_downloaded += int(result.get("bytes", 0))
                mark_chunk_complete((request.store_identity, request.key))
            else:
                counters.download_errors += 1
                if not counters.first_error:
                    counters.first_error = str(result.get("error", "prefetch download failed"))

        next_sample = start_sample
        last_progress = 0.0
        download_sequence = 0
        previous_torch_threads = int(torch.get_num_threads())
        torch_threads_changed = False
        if previous_torch_threads != 1:
            torch.set_num_threads(1)
            torch_threads_changed = True
        producer_executor = ThreadPoolExecutor(max_workers=producer_count)
        download_executor = ThreadPoolExecutor(max_workers=worker_count)
        normal_shutdown = False
        try:
            producer_futures: dict[Future[tuple[int, int, list[ZarrChunkRequest]]], int] = {}
            download_futures: dict[Future[dict[str, Any]], tuple[ZarrChunkRequest, float]] = {}
            queued_downloads: dict[tuple[str, str], tuple[int, int, ZarrChunkRequest]] = {}
            download_queue: list[tuple[int, int, tuple[str, str]]] = []
            producer_results: dict[int, tuple[str, Any]] = {}
            next_producer_result = start_sample

            def update_download_queue_counters() -> None:
                counters.queued_download_futures = len(download_futures) + len(queued_downloads)

            def submit_downloads() -> None:
                while len(download_futures) < worker_count:
                    request: ZarrChunkRequest | None = None
                    while download_queue:
                        priority, sequence, identity = heapq.heappop(download_queue)
                        queued = queued_downloads.get(identity)
                        if queued is None:
                            continue
                        queued_priority, queued_sequence, queued_request = queued
                        if priority != queued_priority or sequence != queued_sequence:
                            continue
                        queued_downloads.pop(identity, None)
                        request = queued_request
                        break
                    if request is None:
                        return
                    future = download_executor.submit(
                        _download_prefetch_request,
                        request,
                        retry_seconds=retry_seconds,
                    )
                    download_futures[future] = (request, time.perf_counter())

            def submit_producers() -> None:
                nonlocal next_sample
                while next_sample < end_sample and len(producer_futures) < producer_count:
                    future = producer_executor.submit(build_requests, next_sample)
                    producer_futures[future] = next_sample
                    next_sample += 1

            def consume_ready_producer_results() -> None:
                nonlocal next_producer_result
                while next_producer_result in producer_results:
                    status, payload = producer_results.pop(next_producer_result)
                    sample_index_for_error = next_producer_result
                    next_producer_result += 1
                    if status == "error":
                        exc = payload
                        counters.samples_done += 1
                        counters.samples_skipped += 1
                        if not counters.first_sample_skip:
                            counters.first_sample_skip = f"sample={sample_index_for_error}: {exc}"
                        sample_pending.setdefault(sample_index_for_error, set())
                        close_sample_requests(sample_index_for_error)
                        submit_producers()
                        continue

                    _raw_sample_index, sample_valid_voxels, requests = payload
                    counters.samples_done += 1
                    sample_pending.setdefault(sample_index_for_error, set())
                    counters.valid_voxels += int(sample_valid_voxels)
                    for request in requests:
                        classify_or_submit(request, sample_index_for_error, download_queue, queued_downloads)
                    close_sample_requests(sample_index_for_error)
                    submit_downloads()
                    submit_producers()

            submit_producers()
            submit_downloads()
            update_download_queue_counters()
            print_progress()
            while producer_futures or download_futures or queued_downloads:
                consume_ready_producer_results()
                submit_downloads()
                active = set(producer_futures) | set(download_futures)
                if not active:
                    break
                done, _ = wait(active, timeout=0.25, return_when=FIRST_COMPLETED)
                for future in done:
                    if future in producer_futures:
                        sample_index_for_error = producer_futures.pop(future)
                        try:
                            producer_results[sample_index_for_error] = ("ok", future.result())
                        except ValueError as exc:
                            producer_results[sample_index_for_error] = ("error", exc)
                        consume_ready_producer_results()
                    else:
                        request, _submitted_at = download_futures.pop(future)
                        consume_download_result(request, future.result())
                        submit_downloads()
                now = time.perf_counter()
                if now - last_progress >= 0.5:
                    update_download_queue_counters()
                    print_progress()
                    last_progress = now
            update_download_queue_counters()
            print_progress(final=True)
            normal_shutdown = True
        except BaseException:
            for future in producer_futures:
                future.cancel()
            for future in download_futures:
                future.cancel()
            producer_executor.shutdown(wait=False, cancel_futures=True)
            download_executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            if normal_shutdown:
                producer_executor.shutdown(wait=True)
                download_executor.shutdown(wait=True)
            if torch_threads_changed:
                torch.set_num_threads(previous_torch_threads)

        elapsed = time.perf_counter() - start
        return {
            "samples": counters.samples_done,
            "skipped_samples": counters.samples_skipped,
            "chunks": counters.unique_chunks_seen,
            "cache_hits": counters.cache_hits,
            "queued_for_download": counters.queued_for_download,
            "download_done": counters.download_done,
            "downloaded": counters.downloaded,
            "missing": counters.known_missing + counters.newly_missing,
            "known_missing": counters.known_missing,
            "new_missing": counters.newly_missing,
            "errors": counters.download_errors,
            "elapsed_s": elapsed,
            "mib": counters.bytes_downloaded / (1024.0 * 1024.0),
            "bytes": counters.bytes_downloaded,
            "workers": worker_count,
            "producer_workers": producer_count,
            "valid_voxels": counters.valid_voxels,
            "first_error": counters.first_error,
            "first_sample_skip": counters.first_sample_skip,
            "max_exclusive_sample_index": counters.max_exclusive_sample_index,
        }


def config_from_mapping(raw: dict[str, Any], *, config_dir: Path | None = None) -> FiberTrace3DConfig:
    path = Path(config_dir or ".") / "_fiber_trace_3d_inline.json"
    del path
    datasets = raw.get("datasets")
    if raw.get("_array_records") is not None and datasets is None:
        datasets = [{"array_records": True}]
    cfg = FiberTrace3DConfig(
       datasets=tuple(dict(entry) for entry in datasets),
        sampling_strategy=str(raw.get("sampling_strategy", "control_points")),
        embedding=EmbeddingConfig.from_mapping(raw),
        refinement=RefinementConfig.parse(raw),
        refinement_split=raw.get("_refinement_split", "train"),
        batch_size=int(raw.get("batch_size", 1)),
        patch_shape_zyx=_as_zyx3(raw.get("patch_shape_zyx", [192, 192, 192]), key="patch_shape_zyx"),
        seed=int(raw.get("seed", 1)),
        cp_margin_voxels=int(raw.get("cp_margin_voxels", 4)),
        presence_radius_voxels=float(raw.get("presence_radius_voxels", 2.0)),
        presence_target_mode=str(raw.get("presence_target_mode", "binary")).lower(),
        presence_gaussian_width_ratio=float(raw.get("presence_gaussian_width_ratio", 4.0)),
        presence_gaussian_truncate_sigma=float(raw.get("presence_gaussian_truncate_sigma", 3.0)),
        presence_sideband_weight=float(raw.get("presence_sideband_weight", 0.0)),
        presence_sideband_half_width_base=float(raw.get("presence_sideband_half_width_base", 64.0)),
        presence_sideband_half_height_base=float(raw.get("presence_sideband_half_height_base", 16.0)),
        presence_negative_edge_margin_voxels=(
            None
            if raw.get("presence_negative_edge_margin_voxels") is None
            else int(raw.get("presence_negative_edge_margin_voxels"))
        ),
        augment_enabled=bool(raw.get("augment_enabled", True)),
        augment_shift_zyx=_as_float_zyx3(raw.get("augment_shift_zyx", [0.0, 0.0, 0.0]), key="augment_shift_zyx"),
        augment_rotation_degrees=float(raw.get("augment_rotation_degrees", 0.0)),
        augment_scale_min=float(raw.get("augment_scale_min", 1.0)),
        augment_scale_max=float(raw.get("augment_scale_max", 1.0)),
        augment_flip_probability=float(raw.get("augment_flip_probability", 0.0)),
        augment_brightness=float(raw.get("augment_brightness", 0.0)),
        augment_contrast_min=float(raw.get("augment_contrast_min", 1.0)),
        augment_contrast_max=float(raw.get("augment_contrast_max", 1.0)),
        augment_gamma_min=float(raw.get("augment_gamma_min", 1.0)),
        augment_gamma_max=float(raw.get("augment_gamma_max", 1.0)),
        augment_noise_std=float(raw.get("augment_noise_std", 0.0)),
        augment_blur_sigma=float(raw.get("augment_blur_sigma", 0.0)),
        augment_smooth_displacement_mode=str(
            raw.get("augment_smooth_displacement_mode", "none")
        ).lower(),
        augment_smooth_displacement_amplitude_zyx=_as_float_zyx3(
            raw.get("augment_smooth_displacement_amplitude_zyx", [0.0, 0.0, 0.0]),
            key="augment_smooth_displacement_amplitude_zyx",
        ),
        augment_smooth_displacement_control_spacing_zyx=_as_float_zyx3(
            raw.get("augment_smooth_displacement_control_spacing_zyx", [16.0, 16.0, 16.0]),
            key="augment_smooth_displacement_control_spacing_zyx",
        ),
        augment_smooth_displacement_probability=float(
            raw.get("augment_smooth_displacement_probability", 0.0)
        ),
        augment_anisotropic_blur_probability=float(
            raw.get("augment_anisotropic_blur_probability", 0.0)
        ),
        augment_anisotropic_blur_sigma_along=float(
            raw.get("augment_anisotropic_blur_sigma_along", 0.0)
        ),
        augment_anisotropic_blur_sigma_across=float(
            raw.get("augment_anisotropic_blur_sigma_across", 0.0)
        ),
        augment_anisotropic_blur_orientation=str(
            raw.get("augment_anisotropic_blur_orientation", "fiber")
        ).lower(),
        augment_anisotropic_blur_roll_degrees=float(
            raw.get("augment_anisotropic_blur_roll_degrees", 0.0)
        ),
        image_normalization=str(raw.get("image_normalization", "zscore")),
        round_source_to_chunk_boundaries=bool(raw.get("round_source_to_chunk_boundaries", True)),
        geometry_tile_depth=int(raw.get("geometry_tile_depth", 0)),
        geometry_backend=str(raw.get("geometry_backend", "auto")).lower(),
        value_augmentation_backend=str(
            raw.get("value_augmentation_backend", "auto")
        ).lower(),
        prefetch_workers=int(raw.get("prefetch_workers", 16)),
        prefetch_sampler_workers=_parse_worker_count(
            raw.get("prefetch_sampler_workers", 2),
            key="prefetch_sampler_workers",
        ),
        volume_cache_dir=None if raw.get("volume_cache_dir") is None else str(raw.get("volume_cache_dir")),
        volume_cache_memory_mib=_parse_volume_cache_memory_mib(raw.get("volume_cache_memory_mib")),
        volume_io_threads=(
            None if raw.get("volume_io_threads") is None else int(raw.get("volume_io_threads"))
        ),
        volume_decode_threads=(
            None
            if raw.get("volume_decode_threads") is None
            else int(raw.get("volume_decode_threads"))
        ),
        volume_sample_threads=(
            None
            if raw.get("volume_sample_threads") is None
            else int(raw.get("volume_sample_threads"))
        ),
        volume_chunk_locality_fast_path=bool(
            raw.get("volume_chunk_locality_fast_path", True)
        ),
        volume_collect_locality_stats=bool(
            raw.get("volume_collect_locality_stats", False)
        ),
        config_dir=config_dir,
    )
    if "_array_records" in raw:
        object.__setattr__(cfg, "_array_records", raw["_array_records"])
    return cfg
