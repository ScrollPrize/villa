"""Reusable Fiber3D operations for the ordered per-sample CPU pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
import threading

import torch
import warnings
from typing import Callable

from vesuvius.neural_tracing.fiber_trace_3d.loader import (
    FiberTrace3DBatch,
    FiberTrace3DConfig,
    FiberTrace3DLoader,
    FiberTrace3DRawSample,
    FiberTrace3DSample,
    FiberTrace3DSharedTensorSlots,
    augment_fiber_trace_3d_raw_sample,
    collate_fiber_trace_3d_samples,
    load_config,
)
from vesuvius.neural_tracing.ordered_sample_pipeline import (
    OrderedTwoStageSamplePipeline,
)


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

    def __call__(self, raw: FiberTrace3DRawSample) -> FiberTrace3DSample:
        return augment_fiber_trace_3d_raw_sample(
            self.config,
            raw,
            profile=self.profile,
        )


@dataclass(frozen=True)
class FiberTrace3DAugmentFactory:
    config_path: str
    profile: bool = True

    def __call__(self) -> FiberTrace3DAugmentOperation:
        return FiberTrace3DAugmentOperation(_pipeline_config(self.config_path), self.profile)


@dataclass(frozen=True)
class SampleQueueSource:
    """Sampling semantics; transport ordinals never seed sample generation."""

    config: Any
    sample_mode: str = "random"
    sample_index_limit: int | None = None


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
                    FiberTrace3DAugmentFactory(source.config, profile=self.profile)
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
                stream_index_map=self.descriptors))
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
            volumes, masks = self.transport.batch_tensors(list(ticket), allow_wrap=True)
            return collate_pipeline_results(
                [self._ready[index] for index in ticket], volumes, masks,
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


def collate_pipeline_results(results, volumes, masks, *, profile=False,
                             total_start=None, cpu_start=None, wall_start_ns=None):
    samples: list[FiberTrace3DSample] = []
    for result in results:
        sample = result.value
        if not isinstance(sample, FiberTrace3DSample):
            raise RuntimeError("training sample pipeline returned an invalid sample")
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
        samples, volume_batch=volumes, valid_mask_batch=masks,
        profile=profile, total_start=total_start, cpu_total_start=cpu_start,
        wall_start_ns=wall_start_ns,
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
    ) -> None:
        if int(batch_count) <= 0:
            raise ValueError("batch_count must be positive")
        if int(slot_count) % int(batch_size) != 0:
            raise ValueError("slot_count must be a multiple of batch_size")
        if int(initial_in_flight) < int(batch_size):
            raise ValueError("initial_in_flight must be at least one batch")
        self.batch_size = int(batch_size)
        self.profile = bool(profile)
        self._leased: list[int] = []
        self.transport = FiberTrace3DSharedTensorSlots(
            start_index=0,
            slot_count=int(slot_count),
            patch_shape_zyx=patch_shape_zyx,
            stream_index_map=index_map,
        )
        self.pipeline = OrderedTwoStageSamplePipeline(
            stage1_factory=FiberTrace3DLoadFactory(
                config_path,
                sample_index_limit,
                sample_mode,
                index_map=index_map,
                profile=profile,
            ),
            stage2_factory=FiberTrace3DAugmentFactory(config_path, profile=profile),
            start_index=0,
            item_count=int(batch_count) * self.batch_size,
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
        total_start = time.perf_counter()
        cpu_start = time.process_time()
        wall_start_ns = time.time_ns()
        results = [
            self.pipeline.next_ordered(release_credit=False)
            for _ in range(self.batch_size)
        ]
        indices = [int(result.index) for result in results]
        volumes, masks = self.transport.batch_tensors(indices)
        batch = collate_pipeline_results(
            results, volumes, masks, profile=self.profile, total_start=total_start,
            cpu_start=cpu_start, wall_start_ns=wall_start_ns)
        self._leased = indices
        return batch

    def release_current(self) -> None:
        for index in self._leased:
            self.pipeline.release(index)
        self._leased.clear()

    def close(self) -> None:
        self.release_current()
        self.pipeline.close()
