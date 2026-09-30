"""Reusable Fiber3D operations for the ordered per-sample CPU pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
import queue
import threading
import time
import warnings
from typing import Any, Callable

import torch

from vesuvius.neural_tracing.fiber_trace_3d.loader import (
    FiberTrace3DBatch,
    FiberTrace3DConfig,
    FiberTrace3DLoader,
    FiberTrace3DRawSample,
    FiberTrace3DSample,
    FiberTrace3DSharedTensorSlots,
    augment_fiber_trace_3d_raw_sample,
    apply_value_augmentation_batch_3d,
    apply_value_augmentation_suffix_batch_3d,
    collate_fiber_trace_3d_samples,
    load_config,
    prepare_fiber_trace_3d_cuda_suffix,
)
from vesuvius.neural_tracing.ordered_sample_pipeline import (
    OrderedTwoStageSamplePipeline,
)
from vesuvius.neural_tracing.fiber_trace_3d.targets import materialize_targets


@dataclass(frozen=True)
class IdentitySampleIndex:
    def __call__(self, index: int) -> int:
        return int(index)


def _pipeline_config(source):
    """Accept a resolved test config without reloading the training datasets."""
    return source if isinstance(source, FiberTrace3DConfig) else load_config(source)


@dataclass(frozen=True)
class EvaluationSampleIndex:
    batch_size: int
    global_batch_indices: tuple[int, ...]
    start_sample_index: int = 0

    def __call__(self, index):
        batch, offset = divmod(index, self.batch_size)
        return self.start_sample_index + self.global_batch_indices[batch] * self.batch_size + offset


@dataclass(frozen=True)
class DistributedSampleIndex:
    """Map a rank-local consecutive stream onto the canonical DDP sample stream."""

    batch_size: int
    world_size: int
    rank: int
    start_step: int = 0

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.world_size <= 0:
            raise ValueError("batch_size and world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if self.start_step < 0:
            raise ValueError("start_step must be nonnegative")

    def __call__(self, index: int) -> int:
        local_batch, offset = divmod(int(index), int(self.batch_size))
        global_batch = (
            (int(self.start_step) + local_batch) * int(self.world_size) + int(self.rank)
        )
        return global_batch * int(self.batch_size) + offset


@dataclass(frozen=True)
class VirtualBatchSampleIndex:
    """Map physical-rank work onto full logical rank-local batches."""

    batch_size: int
    physical_world_size: int
    rank: int
    virtual_world_size: int
    start_step: int = 0

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.physical_world_size <= 0:
            raise ValueError("batch_size and physical_world_size must be positive")
        if self.virtual_world_size < self.physical_world_size:
            raise ValueError(
                "virtual_world_size must be at least physical_world_size"
            )
        if not 0 <= self.rank < self.physical_world_size:
            raise ValueError("rank must be in [0, physical_world_size)")
        if self.start_step < 0:
            raise ValueError("start_step must be nonnegative")

    @property
    def local_batches_per_step(self) -> int:
        return (
            int(self.virtual_world_size)
            - int(self.rank)
            + int(self.physical_world_size)
            - 1
        ) // int(self.physical_world_size)

    def __call__(self, index: int) -> int:
        local_batch, offset = divmod(int(index), int(self.batch_size))
        optimizer_step, local_batch_in_step = divmod(
            local_batch,
            self.local_batches_per_step,
        )
        virtual_rank = (
            int(self.rank)
            + local_batch_in_step * int(self.physical_world_size)
        )
        if virtual_rank >= int(self.virtual_world_size):
            raise RuntimeError("virtual batch mapping produced an invalid rank")
        global_batch = (
            (int(self.start_step) + optimizer_step) * int(self.virtual_world_size)
            + virtual_rank
        )
        return global_batch * int(self.batch_size) + offset


@dataclass(frozen=True)
class FiberTrace3DLoadOperation:
    loader: FiberTrace3DLoader
    sample_index_limit: int | None
    sample_mode: str
    index_map: Callable[[int], int]
    profile: bool

    def __call__(self, index: int) -> FiberTrace3DRawSample:
        return self.loader.load_raw_sample(
            self.index_map(index),
            sample_index_limit=self.sample_index_limit,
            sample_mode=self.sample_mode,
            device="cpu",
            profile=self.profile,
        )


@dataclass(frozen=True)
class FiberTrace3DLoadFactory:
    config_path: str
    sample_index_limit: int | None
    sample_mode: str
    index_map: Callable[[int], int] = IdentitySampleIndex()
    profile: bool = True
    single_core: bool = False

    def __call__(self) -> FiberTrace3DLoadOperation:
        warnings.filterwarnings(
            "ignore",
            message=r"skipping NML fiber component with branch node:.*",
            category=RuntimeWarning,
        )
        config = _pipeline_config(self.config_path)
        if self.single_core and getattr(config, "_array_records", None) is None:
            from .cooperative_resources import initialize_worker_volume_cache
            initialize_worker_volume_cache({"volume_cache_memory_mib": config.volume_cache_memory_mib})
            config = replace(config, volume_io_threads=1, volume_decode_threads=1, volume_sample_threads=1)
        return FiberTrace3DLoadOperation(
            FiberTrace3DLoader(config),
            self.sample_index_limit,
            self.sample_mode,
            self.index_map,
            self.profile,
        )


@dataclass(frozen=True)
class FiberTrace3DFusedLoadOperation:
    loader: FiberTrace3DLoader
    sample_index_limit: int | None
    sample_mode: str
    index_map: Callable[[int], int]
    profile: bool

    def __call__(self, index: int) -> FiberTrace3DSample:
        return self.loader.load_sample(
            self.index_map(index),
            sample_index_limit=self.sample_index_limit,
            sample_mode=self.sample_mode,
            device="cpu",
            profile=self.profile,
        )


@dataclass(frozen=True)
class FiberTrace3DFusedLoadFactory:
    config_path: str
    sample_index_limit: int | None
    sample_mode: str
    index_map: Callable[[int], int] = IdentitySampleIndex()
    profile: bool = True

    def __call__(self) -> FiberTrace3DFusedLoadOperation:
        warnings.filterwarnings(
            "ignore",
            message=r"skipping NML fiber component with branch node:.*",
            category=RuntimeWarning,
        )
        return FiberTrace3DFusedLoadOperation(
            FiberTrace3DLoader(_pipeline_config(self.config_path)),
            self.sample_index_limit,
            self.sample_mode,
            self.index_map,
            self.profile,
        )


@dataclass(frozen=True)
class FiberTrace3DAugmentOperation:
    config: object
    profile: bool
    defer_cuda_suffix: str | None = None

    def __call__(self, raw: FiberTrace3DRawSample) -> FiberTrace3DSample | FiberTrace3DRawSample:
        if self.defer_cuda_suffix:
            return prepare_fiber_trace_3d_cuda_suffix(
                self.config,
                raw,
                profile=self.profile,
                mode=self.defer_cuda_suffix,
            )
        return augment_fiber_trace_3d_raw_sample(
            self.config,
            raw,
            profile=self.profile,
        )


@dataclass(frozen=True)
class FiberTrace3DAugmentFactory:
    config_path: str
    profile: bool = True
    defer_cuda_suffix: str | None = None

    def __call__(self) -> FiberTrace3DAugmentOperation:
        return FiberTrace3DAugmentOperation(
            _pipeline_config(self.config_path),
            self.profile,
            self.defer_cuda_suffix,
        )


@dataclass(frozen=True)
class SampleQueueSource:
    """Sampling semantics; transport ordinals never seed sample generation."""

    config: Any
    sample_mode: str = "random"
    sample_index_limit: int | None = None
    defer_cuda_value_augmentation: str | None = None


@dataclass(frozen=True)
class _QueueDescriptors:
    rows: Any
    slot_count: int

    def descriptor(self, index):
        generation, source, canonical = self.rows[index % self.slot_count].tolist()
        if generation != index:
            raise RuntimeError("sample descriptor was reused before release")
        return int(source), int(canonical)

    def __call__(self, index):
        return self.descriptor(index)[1]


@dataclass(frozen=True)
class _RoutedFactory:
    sources: tuple[SampleQueueSource, ...]
    descriptors: _QueueDescriptors
    stage: str
    profile: bool

    def __call__(self):
        # Source loaders are initialized lazily in each existing worker, once.
        operations = {}

        def run(value):
            if self.stage == "load":
                source_id, canonical = self.descriptors.descriptor(value)
            else:
                source_id, value = value
            if source_id not in operations:
                source = self.sources[source_id]
                factory = (
                    FiberTrace3DLoadFactory(source.config, source.sample_index_limit,
                        source.sample_mode, profile=self.profile, single_core=True)
                    if self.stage == "load" else
                    FiberTrace3DAugmentFactory(source.config, profile=self.profile,
                        defer_cuda_suffix=source.defer_cuda_value_augmentation)
                )
                operations[source_id] = factory()
            result = operations[source_id](canonical if self.stage == "load" else value)
            return source_id, result

        return run


class _RoutedTransport:
    def __init__(self, banks, descriptors):
        self.banks = banks
        self.descriptors = descriptors
        self.slot_count = descriptors.slot_count

    def _location(self, index):
        return index % self.slot_count, index // self.slot_count

    def encode(self, index, value, *, final):
        source, sample = value
        if self.descriptors.descriptor(index)[0] != source:
            raise RuntimeError("sample source differs from admitted job")
        return source, self.banks[source].encode(index, sample, final=final)

    def decode(self, index, value, *, final):
        source, payload = value
        if self.descriptors.descriptor(index)[0] != source:
            raise RuntimeError("sample result source differs from admitted job")
        sample = self.banks[source].decode(index, payload, final=final)
        return sample if final else (source, sample)

    def batch_tensors(self, indices, *, allow_wrap=True):
        source = self.descriptors.descriptor(indices[0])[0]
        if any(self.descriptors.descriptor(index)[0] != source for index in indices):
            raise ValueError("one batch must come from one sample source")
        return self.banks[source].batch_tensors(indices, allow_wrap=allow_wrap)


class FiberTrace3DSampleQueue:
    """Persistent optimized workers shared by explicitly ordered train/test jobs.

    Admission is bounded by *leased plus future* slots. Owners decide phase order
    before admission; no worker waits for work queued behind its dependency.
    Different source dtypes keep uint8 training transfers intact. All sources use
    the same slot-ownership ledger and worker pool, not independent data loaders.
    """

    def __init__(self, *, sources, workers, slot_count, initial_in_flight,
                 load_affinity_workers=None, initial_active_workers=None, profile=False,
                 own_worker_sessions=False, on_process_start=None):
        if not sources:
            raise ValueError("sample queue requires at least one source")
        self.sources = dict(sources)
        self.names = tuple(sources)
        self._source_ids = {name: index for index, name in enumerate(self.names)}
        rows = torch.full((slot_count, 3), -1, dtype=torch.int64).share_memory_()
        self.descriptors = _QueueDescriptors(rows, slot_count)
        banks = []
        for source in self.sources.values():
            config = _pipeline_config(source.config)
            banks.append(FiberTrace3DSharedTensorSlots(
                start_index=0, slot_count=slot_count, patch_shape_zyx=config.patch_shape_zyx,
                stream_index_map=self.descriptors,
                volume_dtype=torch.uint8 if source.defer_cuda_value_augmentation == "all" else torch.float32,
                embedding_stride=config.embedding.output_stride if config.embedding.enabled else 0))
        self.transport = _RoutedTransport(tuple(banks), self.descriptors)
        factory_args = (tuple(self.sources.values()), self.descriptors)
        self.pipeline = OrderedTwoStageSamplePipeline(
            stage1_factory=_RoutedFactory(*factory_args, "load", profile),
            stage2_factory=_RoutedFactory(*factory_args, "augment", profile),
            start_index=0, item_count=2**62, stage1_workers=0, stage2_workers=0,
            shared_workers=workers, elastic_workers=True,
            initial_active_workers=workers if initial_active_workers is None else initial_active_workers,
            raw_queue_capacity=max(1, min(slot_count, workers)),
            result_queue_capacity=max(1, min(slot_count, workers)),
            max_in_flight=slot_count, initial_in_flight=min(slot_count, max(1, initial_in_flight)),
            shared_stage1_affinity_workers=load_affinity_workers,
            transport=self.transport, manual_admission=True, own_worker_sessions=own_worker_sessions,
            on_process_start=on_process_start)
        self._lock = threading.RLock()
        self._next = 0
        self._owned = set()
        self._ready = {}
        self.profile = profile

    def submit(self, source, canonical_indices):
        indices = tuple(int(index) for index in canonical_indices)
        if not indices or len(indices) > self.transport.slot_count:
            raise ValueError("sample batch must fit the bounded shared slots")
        source_id = self._source_ids[source]
        with self._lock:
            ticket = tuple(range(self._next, self._next + len(indices)))
            if any(index % self.transport.slot_count in self._owned for index in ticket):
                return None
            for ordinal, canonical in zip(ticket, indices):
                self.descriptors.rows[ordinal % self.transport.slot_count] = torch.tensor(
                    (ordinal, source_id, canonical), dtype=torch.int64)
                self._owned.add(ordinal % self.transport.slot_count)
            self._next += len(indices)
            self.pipeline.admit_until(self._next)
            return ticket

    def poll(self, limit=64):
        with self._lock:
            for _ in range(limit):
                result = self.pipeline.try_next_ordered(release_credit=False)
                if result is None:
                    break
                self._ready[result.index] = result

    def batch(self, ticket):
        with self._lock:
            self.poll()
            if not all(index in self._ready for index in ticket):
                return None
            source_id = self.descriptors.descriptor(ticket[0])[0]
            source = self.sources[self.names[source_id]]
            return collate_pipeline_results(
                [self._ready[index] for index in ticket],
                self.transport.batch_tensors(list(ticket), allow_wrap=True),
                defer_cuda_value_augmentation=source.defer_cuda_value_augmentation,
                profile=self.profile, total_start=time.perf_counter())

    def release(self, ticket):
        with self._lock:
            for index in ticket:
                if index not in self._ready:
                    raise RuntimeError("cannot release an undelivered sample")
                self.pipeline.release(index)
                del self._ready[index]
                self._owned.remove(index % self.transport.slot_count)

    def close(self):
        self.pipeline.close()
        self._ready.clear()
        self._owned.clear()


def collate_pipeline_results(results, volumes, *, defer_cuda_value_augmentation=None,
                             profile=False, total_start=None, cpu_start=None, wall_start_ns=None):
    """Shared collation for rank-local and persistent canonical sample owners."""
    samples = []
    deferred_kernels, deferred_params = [], []
    for result in results:
        value = result.value
        if defer_cuda_value_augmentation is not None:
            if not isinstance(value, FiberTrace3DRawSample):
                raise RuntimeError("training sample pipeline lost deferred augmentation")
            sample = value.sample
            deferred_kernels.append(value.blur_kernels)
            deferred_params.append(value.augmentation)
        else:
            if not isinstance(value, FiberTrace3DSample):
                raise RuntimeError("training sample pipeline returned an invalid sample")
            sample = value
        if profile:
            timings = dict(sample.profile_timings_ms or {})
            timings.update(
                pipeline_stage1_service_ms=float(result.stage1_service_ms),
                pipeline_stage1_cpu_ms=float(result.stage1_cpu_ms),
                pipeline_raw_queue_ms=float(result.raw_queue_ms),
                pipeline_stage2_service_ms=float(result.stage2_service_ms),
                pipeline_stage2_cpu_ms=float(result.stage2_cpu_ms),
                pipeline_result_queue_ms=float(result.result_queue_ms),
                pipeline_reorder_wait_ms=float(result.reorder_wait_ms),
            )
            sample = replace(sample, profile_timings_ms=timings)
        samples.append(sample)
    return collate_fiber_trace_3d_samples(
        samples, volume_batch=volumes,
        deferred_augmentation_params=tuple(deferred_params) if defer_cuda_value_augmentation is not None else None,
        deferred_blur_kernels=tuple(deferred_kernels) if defer_cuda_value_augmentation is not None else None,
        profile=profile, total_start=total_start, cpu_total_start=cpu_start, wall_start_ns=wall_start_ns,
    )


class FiberTrace3DTrainingBatchPipeline:
    """Present deterministic batches backed by the bounded per-sample pipeline."""

    def __init__(
        self,
        *,
        config_path: str,
        index_map: Callable[[int], int],
        batch_size: int,
        patch_shape_zyx: tuple[int, int, int],
        batch_count: int,
        workers: int,
        load_affinity_workers: int,
        slot_count: int,
        initial_in_flight: int,
        sample_index_limit: int | None,
        sample_mode: str,
        profile: bool,
        defer_cuda_value_augmentation: str | None = None,
        batch_sizes: tuple[int, ...] | None = None,
        single_core_load: bool = False,
        embedding_stride: int = 0,
    ) -> None:
        if int(batch_count) <= 0:
            raise ValueError("batch_count must be positive")
        if int(slot_count) % int(batch_size) != 0:
            raise ValueError("slot_count must be a multiple of batch_size")
        if int(initial_in_flight) < int(batch_size):
            raise ValueError("initial_in_flight must be at least one batch")
        self.batch_size = int(batch_size)
        self.batch_sizes = (self.batch_size,) if batch_sizes is None else tuple(batch_sizes)
        if not self.batch_sizes or any(type(size) is not int or not 0 < size <= self.batch_size for size in self.batch_sizes):
            raise ValueError("batch_sizes must be a nonempty pattern within batch_size")
        if int(batch_count) % len(self.batch_sizes):
            raise ValueError("batch_count must cover whole batch-size patterns")
        self._batch_count, self._batch_ordinal = int(batch_count), 0
        self._variable_batches = batch_sizes is not None
        self.profile = bool(profile)
        self.defer_cuda_value_augmentation = defer_cuda_value_augmentation
        self._leased: list[int] = []
        self.transport = FiberTrace3DSharedTensorSlots(
            embedding_stride=embedding_stride,
            start_index=0,
            slot_count=int(slot_count),
            patch_shape_zyx=patch_shape_zyx,
            stream_index_map=index_map,
            volume_dtype=(
                torch.uint8
                if self.defer_cuda_value_augmentation == "all"
                else torch.float32
            ),
        )
        self.pipeline = OrderedTwoStageSamplePipeline(
            stage1_factory=FiberTrace3DLoadFactory(
                config_path,
                sample_index_limit,
                sample_mode,
                index_map=index_map,
                profile=profile,
                single_core=single_core_load,
            ),
            stage2_factory=FiberTrace3DAugmentFactory(
                config_path,
                profile=profile,
                defer_cuda_suffix=self.defer_cuda_value_augmentation,
            ),
            start_index=0,
            item_count=(int(batch_count) // len(self.batch_sizes)) * sum(self.batch_sizes),
            stage1_workers=0,
            stage2_workers=0,
            raw_queue_capacity=max(self.batch_size, min(int(slot_count), int(workers))),
            result_queue_capacity=max(self.batch_size, min(int(slot_count), int(workers))),
            max_in_flight=int(slot_count),
            multiprocessing_context="forkserver",
            transport=self.transport,
            shared_workers=int(workers),
            initial_in_flight=int(initial_in_flight),
            shared_stage1_affinity_workers=int(load_affinity_workers),
        )

    def __iter__(self) -> "FiberTrace3DTrainingBatchPipeline":
        return self

    def __next__(self) -> FiberTrace3DBatch:
        if self._leased:
            raise RuntimeError("previous Fiber3D pipeline batch was not released")
        if self._batch_ordinal >= self._batch_count:
            raise StopIteration
        take = self.batch_sizes[self._batch_ordinal % len(self.batch_sizes)]
        total_start = time.perf_counter()
        cpu_start = time.process_time()
        wall_start_ns = time.time_ns()
        results = [
            self.pipeline.next_ordered(release_credit=False)
            for _ in range(take)
        ]
        indices = [int(result.index) for result in results]
        volumes = self.transport.batch_tensors(indices, allow_wrap=self._variable_batches)
        batch = collate_pipeline_results(
            results, volumes, defer_cuda_value_augmentation=self.defer_cuda_value_augmentation,
            profile=self.profile, total_start=total_start, cpu_start=cpu_start, wall_start_ns=wall_start_ns,
        )
        self._leased = indices
        self._batch_ordinal += 1
        return batch

    def release_current(self) -> None:
        for index in self._leased:
            self.pipeline.release(index)
        self._leased.clear()

    def close(self) -> None:
        self.release_current()
        self.pipeline.close()


@dataclass(frozen=True)
class _CudaPreparedBatch:
    batch: FiberTrace3DBatch
    pinned_batch: FiberTrace3DBatch
    transfer_started: torch.cuda.Event
    transfer_finished: torch.cuda.Event
    augmentation_finished: torch.cuda.Event
    preparation_finished: torch.cuda.Event
    queued_at: float
    source_wait_ms: float
    pin_memory_ms: float
    pinned_bytes: int


@dataclass(frozen=True)
class _PreparationFailure:
    error: BaseException


_PREPARATION_END = object()


def _batch_tensor_bytes(batch: FiberTrace3DBatch) -> int:
    return sum(
        int(value.numel()) * int(value.element_size())
        for value in vars(batch).values()
        if isinstance(value, torch.Tensor)
    ) + sum(
        int(item.eligible.numel()) * int(item.eligible.element_size())
        for item in batch.embedding_geometry or () if item is not None and item.eligible is not None
    )


def apply_deferred_batch_augmentation(batch, config, *, mode):
    """Complete the same deferred value transform for training/evaluation/views.

    Call after transfer, under disabled autocast. The mode is explicit because
    suffix-only batches also retain augmentation parameters for reproducibility.
    """
    if batch.deferred_blur_kernels is None:
        return batch
    if mode == "all":
        if batch.deferred_augmentation_params is None:
            raise RuntimeError("CUDA augmentation parameters are missing")
        volume = apply_value_augmentation_batch_3d(config, batch.volume,
            batch.deferred_augmentation_params, batch.deferred_blur_kernels)
    else:
        volume = apply_value_augmentation_suffix_batch_3d(batch.volume, batch.deferred_blur_kernels)
    return replace(batch, volume=volume, deferred_augmentation_params=None, deferred_blur_kernels=None)


class MacroBatchTargetSource:
    """Bounded host staging and geometry-only normalization prepass.

    The value augmentation suffix changes images only. Reuse the real target
    voxelizer on the training device with a storage-free final-shape image view.
    Never transfer the real image or run the model in this prepass.
    """

    def __init__(self, source, *, config, device, output_stride, batch_sizes, conditioned=False):
        self.source, self.config = source, config
        self.device, self.output_stride = device, output_stride
        self.batch_sizes = tuple(batch_sizes)
        self.conditioned = conditioned
        if not self.batch_sizes or any(n <= 0 for n in self.batch_sizes):
            raise ValueError("target source requires nonempty local microbatches")
        self.pending = []
        self.prepare_geometry = lambda operation: operation()

    def __iter__(self):
        return self

    def __next__(self):
        if not self.pending:
            prepared = []
            counts = None
            for size in self.batch_sizes:
                raw = next(self.source)
                try:
                    # Queue views release/reuse their slots on next(); take
                    # ownership before returning credits, including geometry.
                    batch = replace(raw, **{name: value.clone() for name, value in vars(raw).items()
                                           if isinstance(value, torch.Tensor)},
                                    embedding_geometry=None if raw.embedding_geometry is None else tuple(
                                        None if item is None else replace(item, eligible=item.eligible.clone())
                                        for item in raw.embedding_geometry))
                finally:
                    self.source.release_current()
                if batch.volume.shape[0] != size:
                    raise RuntimeError("target prepass microbatch differs from canonical partition")
                batch_counts = self.prepare_geometry(lambda: self._counts(batch))
                counts = batch_counts.clone() if counts is None else counts + batch_counts
                prepared.append(batch)
            local_counts = counts.cpu()
            self.pending = [replace(batch, normalization_counts=local_counts) for batch in reversed(prepared)]
        return self.pending.pop()

    def _counts(self, batch):
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=False):
            geometry = replace(batch, volume=torch.empty(0), valid_mask=None).to(self.device)
            geometry = replace(geometry, volume=torch.empty(1, device=self.device).expand(
                len(batch.volume), 1, *self.config.patch_shape_zyx))
            targets = materialize_targets(geometry, self.config, output_stride=self.output_stride)
            indices = targets.direction_indices_bzyx
            if self.conditioned:
                direction_count = (
                    targets.presence_target[
                        indices[:, 0], 0, indices[:, 1], indices[:, 2], indices[:, 3]
                    ].sum() * 12
                    if targets.presence_target_mode == "anisotropic_gaussian"
                    else targets.presence_target.new_tensor(indices.shape[0] * 12)
                )
                return torch.stack((
                    direction_count,
                    direction_count.new_tensor(torch.unique(indices[:, 0]).numel() * 2),
                    direction_count.new_tensor(len(batch.volume) * 2),
                )).cpu()
            if targets.presence_target_mode == "anisotropic_gaussian":
                positive_groups = (targets.presence_target > 0.0).flatten(1).any(dim=1).sum().float()
                negative_groups = (targets.presence_target == 0.0).flatten(1).any(dim=1).sum().float()
                distribution_groups = (
                    ((targets.presence_target > 0.0) & (targets.presence_target < 1.0))
                    .flatten(1).any(dim=1).sum().float()
                )
            else:
                positive_groups = targets.presence_target.new_tensor(
                    torch.unique(indices[:, 0]).numel()
                )
                negative_groups = (
                    (targets.presence_target <= 0.5).flatten(1).any(dim=1).sum().float()
                )
            direction_count = (
                targets.presence_target[
                    indices[:, 0], 0, indices[:, 1], indices[:, 2], indices[:, 3]
                ].sum() * 6
                if targets.presence_target_mode == "anisotropic_gaussian"
                else targets.presence_target.new_tensor(indices.shape[0] * 6)
            )
            counts = torch.stack((
                direction_count,
                positive_groups,
                negative_groups))
            if targets.presence_target_mode == "anisotropic_gaussian":
                counts = torch.cat((counts, distribution_groups.reshape(1)))
            if self.config.embedding.enabled:
                from .embedding import embedding_pair_counts
                counts = torch.cat((counts, counts.new_tensor(embedding_pair_counts(batch))))
            return counts.cpu()

    def release_current(self):
        pass  # Returned batches own storage, not shared-slot leases.

    def close(self):
        self.source.close()
        self.pending.clear()


class FiberTrace3DCudaBatchPipeline:
    """Prepare two future training batches on a dedicated CUDA stream."""

    def __init__(
        self,
        source: FiberTrace3DTrainingBatchPipeline,
        *,
        config: Any,
        device: torch.device,
        output_stride: int,
        depth: int = 2,
        value_augmentation_suffix: str | None = None,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA batch preparation requires a CUDA device")
        if int(depth) <= 0:
            raise ValueError("CUDA batch preparation depth must be positive")
        self.source = source
        self.config = config
        # Resolve the caller's current device before starting the preparation
        # thread: CUDA's current device is thread-local, and set_device rejects
        # an index-less torch.device('cuda').
        self.device = (torch.device("cuda", torch.cuda.current_device())
                       if device.index is None else device)
        self.output_stride = int(output_stride)
        self.depth = int(depth)
        self.value_augmentation_suffix = value_augmentation_suffix
        self.stream = torch.cuda.Stream(device=self.device)
        self._results: queue.Queue[object] = queue.Queue()
        self._available = threading.Semaphore(self.depth)
        self._stop = threading.Event()
        self._pause_condition = threading.Condition()
        self._paused = False
        self._submit_lock = threading.Lock()
        self._closed = False
        self.max_pinned_bytes = 0
        if isinstance(source, MacroBatchTargetSource):
            source.prepare_geometry = self._prepare_geometry
        self._thread = threading.Thread(
            target=self._run,
            name="fiber3d-cuda-batch-preparation",
            daemon=False,
        )
        self._thread.start()

    def __iter__(self) -> "FiberTrace3DCudaBatchPipeline":
        return self

    def _wait_until_running(self) -> bool:
        with self._pause_condition:
            self._pause_condition.wait_for(
                lambda: self._stop.is_set() or not self._paused
            )
            return not self._stop.is_set()

    def _submit_cuda(self, pinned: FiberTrace3DBatch) -> _CudaPreparedBatch | None:
        while self._wait_until_running():
            with self._submit_lock:
                with self._pause_condition:
                    if self._paused:
                        continue
                with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                    transfer_started = torch.cuda.Event(enable_timing=True)
                    transfer_finished = torch.cuda.Event(enable_timing=True)
                    augmentation_finished = torch.cuda.Event(enable_timing=True)
                    preparation_finished = torch.cuda.Event(enable_timing=True)
                    transfer_started.record(self.stream)
                    device_batch = pinned.to(self.device, non_blocking=True)
                    transfer_finished.record(self.stream)
                    with torch.autocast(device_type="cuda", enabled=False):
                        device_batch = apply_deferred_batch_augmentation(
                            device_batch, self.config, mode=self.value_augmentation_suffix)
                    augmentation_finished.record(self.stream)
                    device_batch = materialize_targets(
                        device_batch,
                        self.config,
                        profile=False,
                        output_stride=self.output_stride,
                    )
                    preparation_finished.record(self.stream)
                return _CudaPreparedBatch(
                    batch=device_batch,
                    pinned_batch=pinned,
                    transfer_started=transfer_started,
                    transfer_finished=transfer_finished,
                    augmentation_finished=augmentation_finished,
                    preparation_finished=preparation_finished,
                    queued_at=time.perf_counter(),
                    source_wait_ms=0.0,
                    pin_memory_ms=0.0,
                    pinned_bytes=_batch_tensor_bytes(pinned),
                )
        return None

    def _prepare_geometry(self, operation):
        # Queue reads remain outside this lock: evaluation may need the same
        # queue's credits while the producer waits at its admission boundary.
        while self._wait_until_running():
            with self._submit_lock:
                with self._pause_condition:
                    if self._paused:
                        continue
                with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                    return operation()
        raise StopIteration

    def _run(self) -> None:
        try:
            torch.cuda.set_device(self.device)
            while not self._stop.is_set():
                self._available.acquire()
                if self._stop.is_set():
                    self._available.release()
                    break
                if not self._wait_until_running():
                    self._available.release()
                    break
                source_started = time.perf_counter()
                try:
                    cpu_batch = next(self.source)
                except StopIteration:
                    self._results.put(_PREPARATION_END)
                    return
                source_wait_ms = (time.perf_counter() - source_started) * 1000.0
                pin_started = time.perf_counter()
                try:
                    pinned = cpu_batch.pin_memory()
                finally:
                    self.source.release_current()
                pin_memory_ms = (time.perf_counter() - pin_started) * 1000.0
                prepared = self._submit_cuda(pinned)
                if prepared is None:
                    self._available.release()
                    break
                prepared = replace(
                    prepared,
                    source_wait_ms=source_wait_ms,
                    pin_memory_ms=pin_memory_ms,
                )
                self.max_pinned_bytes = max(
                    self.max_pinned_bytes,
                    prepared.pinned_bytes * (self.depth + 1),
                )
                self._results.put(prepared)
        except BaseException as error:
            self._results.put(_PreparationFailure(error))

    def __next__(self) -> FiberTrace3DBatch:
        if self._closed:
            raise RuntimeError("CUDA batch preparation pipeline is closed")
        wait_started = time.perf_counter()
        result = self._results.get()
        self._available.release()
        queue_wait_ms = (time.perf_counter() - wait_started) * 1000.0
        if result is _PREPARATION_END:
            raise StopIteration
        if isinstance(result, _PreparationFailure):
            raise RuntimeError("Fiber3D CUDA batch preparation failed") from result.error
        if not isinstance(result, _CudaPreparedBatch):
            raise RuntimeError("Fiber3D CUDA batch preparation returned invalid state")
        ready_wait_started = time.perf_counter()
        result.preparation_finished.synchronize()
        ready_wait_ms = (time.perf_counter() - ready_wait_started) * 1000.0
        transfer_ms = result.transfer_started.elapsed_time(result.transfer_finished)
        augmentation_ms = result.transfer_finished.elapsed_time(
            result.augmentation_finished
        )
        target_ms = result.augmentation_finished.elapsed_time(result.preparation_finished)
        current_stream = torch.cuda.current_stream(self.device)
        result.batch.record_stream(current_stream)
        profile_timings = dict(result.batch.profile_timings_ms or {})
        profile_timings.update(
            cuda_pipeline_source_wait_ms=float(result.source_wait_ms),
            cuda_pipeline_pin_memory_ms=float(result.pin_memory_ms),
            cuda_pipeline_queue_residence_ms=max(
                0.0, (time.perf_counter() - result.queued_at) * 1000.0
            ),
            cuda_pipeline_exposed_queue_wait_ms=float(queue_wait_ms),
            cuda_pipeline_exposed_event_wait_ms=float(ready_wait_ms),
            cuda_pipeline_transfer_ms=float(transfer_ms),
            cuda_pipeline_value_augmentation_ms=float(augmentation_ms),
            cuda_pipeline_target_ms=float(target_ms),
            cuda_pipeline_pinned_bytes=float(result.pinned_bytes),
        )
        return replace(result.batch, profile_timings_ms=profile_timings)

    def pause(self) -> None:
        if self._closed:
            return
        with self._pause_condition:
            self._paused = True
        with self._submit_lock:
            self.stream.synchronize()

    def resume(self) -> None:
        if self._closed:
            return
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()
        for _ in range(self.depth + 1):
            self._available.release()
        self.source.close()
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise RuntimeError("Fiber3D CUDA batch preparation thread did not stop")
        while True:
            try:
                self._results.get_nowait()
            except queue.Empty:
                break
