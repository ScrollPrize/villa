"""Persistent CPU sample ownership across GPU-cohort changes.

Keep the complete current effective batch leased until the update commits.
This makes speculative H2D preparation disposable: after pause/resize the next
cohort can read the same canonical samples without rewinding the loader pool.
The existing pipeline credit window bounds all retained/future samples.
"""

import time

import torch

from .loader import FiberTrace3DSharedTensorSlots
from .sample_pipeline import (
    FiberTrace3DLoadFactory, FiberTrace3DAugmentFactory, IdentitySampleIndex,
    collate_pipeline_results,
)
from vesuvius.neural_tracing.ordered_sample_pipeline import OrderedTwoStageSamplePipeline


class CanonicalSampleBroker:
    def __init__(self, pipeline, transport, *, effective_batch_size,
                 defer_cuda_value_augmentation=None, profile=False):
        if type(effective_batch_size) is not int or effective_batch_size < 1:
            raise ValueError("effective batch must be positive")
        if pipeline.start_index % effective_batch_size or pipeline.item_count % effective_batch_size:
            raise ValueError("sample pipeline must span whole effective updates")
        if pipeline.max_in_flight < effective_batch_size:
            raise ValueError("persistent sample credits must fit one complete effective batch")
        if not pipeline.elastic_workers:
            raise ValueError("persistent broker requires a resizable shared CPU pool")
        self.pipeline, self.transport = pipeline, transport
        self.effective_batch_size = effective_batch_size
        self.frontier = pipeline.start_index
        self.ready = {}
        self.defer_cuda_value_augmentation = defer_cuda_value_augmentation
        self.profile = profile

    @classmethod
    def create(cls, *, config_path, loader_config, start_step, max_steps,
               effective_batch_size, workers, slot_count, initial_in_flight,
               sample_index_limit=None, defer_cuda_value_augmentation=None, profile=False):
        start = start_step * effective_batch_size
        count = (max_steps - start_step) * effective_batch_size
        if count <= 0:
            raise ValueError("persistent sample broker requires remaining updates")
        if slot_count < effective_batch_size:
            raise ValueError("sample slots must fit the whole effective batch")
        transport = FiberTrace3DSharedTensorSlots(
            embedding_stride=loader_config.embedding.output_stride if loader_config.embedding.enabled else 0,
            start_index=start, slot_count=slot_count,
            patch_shape_zyx=loader_config.patch_shape_zyx,
            stream_index_map=IdentitySampleIndex(),
            volume_dtype=torch.uint8 if defer_cuda_value_augmentation == "all" else torch.float32,
        )
        pipeline = OrderedTwoStageSamplePipeline(
            stage1_factory=FiberTrace3DLoadFactory(config_path, sample_index_limit, "random", profile=profile, single_core=True),
            stage2_factory=FiberTrace3DAugmentFactory(config_path, profile=profile,
                                                    defer_cuda_suffix=defer_cuda_value_augmentation),
            start_index=start, item_count=count, stage1_workers=0, stage2_workers=0,
            raw_queue_capacity=max(1, workers), result_queue_capacity=max(1, workers),
            max_in_flight=slot_count, transport=transport, shared_workers=workers,
            initial_in_flight=min(slot_count, max(1, initial_in_flight)),
            elastic_workers=True, initial_active_workers=0,
        )
        return cls(pipeline, transport, effective_batch_size=effective_batch_size,
                   defer_cuda_value_augmentation=defer_cuda_value_augmentation, profile=profile)

    def poll(self, limit=64):
        for _ in range(limit):
            try:
                result = self.pipeline.try_next_ordered(release_credit=False)
            except StopIteration:
                break
            if result is None:
                break
            if result.index in self.ready:
                raise RuntimeError("duplicate canonical sample")
            self.ready[result.index] = result

    def batch(self, indices):
        indices = tuple(indices)
        if not indices or indices != tuple(range(indices[0], indices[0] + len(indices))):
            raise ValueError("batch must be a nonempty contiguous canonical range")
        if indices[0] < self.frontier or indices[-1] >= self.pipeline.end_index:
            raise ValueError("batch is outside the live canonical stream")
        if not all(index in self.ready for index in indices):
            return None
        started = time.perf_counter()
        volumes = self.transport.batch_tensors(list(indices), allow_wrap=True)
        return collate_pipeline_results(
            [self.ready[index] for index in indices], volumes,
            defer_cuda_value_augmentation=self.defer_cuda_value_augmentation,
            profile=self.profile, total_start=started,
        )

    def commit(self, completed_step):
        target = completed_step * self.effective_batch_size
        if target != self.frontier + self.effective_batch_size:
            raise ValueError("commit must advance exactly one effective update")
        indices = range(self.frontier, target)
        if not all(index in self.ready for index in indices):
            raise RuntimeError("cannot commit an update whose samples were not delivered")
        # Call only after *all* GPU workers have completed the optimizer update;
        # no current-batch CPU view is then needed by H2D/preparation.
        for index in indices:
            self.pipeline.release(index)
            del self.ready[index]
        self.frontier = target

    def set_worker_limit(self, count):
        self.pipeline.set_worker_limit(count)

    def workers_within_limit(self):
        self.pipeline.poll_ready()
        return self.pipeline.workers_within_limit()

    def close(self):
        self.pipeline.close()
        self.ready.clear()
