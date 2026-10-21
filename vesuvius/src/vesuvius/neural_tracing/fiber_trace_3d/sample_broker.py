"""Canonical sample ownership over the common train/evaluation worker queue."""

from .sample_pipeline import FiberTrace3DSampleQueue, SampleQueueSource


def _validate_capacity(slots, effective_batch_size, test_sample_count, test_batch_size):
    if effective_batch_size < 1 or slots < effective_batch_size:
        raise ValueError("persistent sample slots must fit the whole effective batch")
    if min(test_sample_count, test_batch_size) > slots:
        raise ValueError("persistent sample slots must fit one logical test batch")


class CanonicalSampleBroker:
    def __init__(self, queue, *, effective_batch_size, start_step, max_steps,
                 test_interval=0, test_sample_count=0, test_batch_size=1,
                 test_start_sample_index=0, initial_test=True):
        _validate_capacity(queue.pipeline.max_in_flight, effective_batch_size,
                           test_sample_count, test_batch_size)
        self.queue = queue
        self.pipeline, self.transport = queue.pipeline, queue.transport
        self.effective_batch_size = effective_batch_size
        self.frontier = start_step * effective_batch_size
        self.end_index = max_steps * effective_batch_size
        self.test_interval = test_interval
        self.test_start = test_start_sample_index
        self.test_count = test_sample_count
        self.test_step = start_step if self.test_count and initial_test else None
        self._train = {}
        self._test = {}
        self._test_leases = {}
        self._next_train = self.frontier
        self._next_test = self.test_start
        self._test_released = 0
        self._test_done = set()

    @classmethod
    def create(cls, *, config_path, loader_config, start_step, max_steps,
               effective_batch_size, workers, slot_count, initial_in_flight,
               sample_index_limit=None, defer_cuda_value_augmentation=None, profile=False,
               training_overrides=None, on_process_start=None):
        from pathlib import Path
        from .loader import FiberTrace3DLoader, config_from_mapping
        from .train import _load_raw_config, _make_test_loader_raw_config, _resolve_dense_test_selection
        raw = _load_raw_config(config_path)
        training = raw.get("training", {})
        if training_overrides:
            training.update(training_overrides)
        sources = {"train": SampleQueueSource(loader_config, "random", sample_index_limit,
                                              defer_cuda_value_augmentation)}
        loaders = {"train": FiberTrace3DLoader(loader_config)}
        count, offset, batch = 0, 0, 1
        interval = int(training.get("test_interval", 0))
        if raw.get("test_datasets") and interval > 0:
            config = config_from_mapping(_make_test_loader_raw_config(raw, training),
                                         config_dir=Path(config_path).resolve().parent)
            loader = FiberTrace3DLoader(config)
            loaders["test"] = loader
            count, offset, mode = _resolve_dense_test_selection(training,
                loader_sample_count=loader.sample_count, default_count=0)
            batch = config.batch_size
            sources["test"] = SampleQueueSource(config, mode)
        _validate_capacity(slot_count, effective_batch_size, count, batch)
        queue = FiberTrace3DSampleQueue(sources=sources, workers=workers,
            slot_count=slot_count, initial_in_flight=initial_in_flight,
            initial_active_workers=0, profile=profile, own_worker_sessions=True,
            on_process_start=on_process_start)
        result = cls(queue, effective_batch_size=effective_batch_size, start_step=start_step,
            max_steps=max_steps, test_interval=interval, test_sample_count=count,
            test_batch_size=batch, test_start_sample_index=offset)
        result.loaders = loaders
        result.dataset_metadata = {name: loader.dataset_metadata for name, loader in loaders.items()}
        return result

    @property
    def ready(self):
        return self._train

    def poll(self, limit=64):
        self.queue.poll(limit)
        if self.test_step is not None:
            while self._next_test < self.test_start + self.test_count:
                ticket = self.queue.submit("test", (self._next_test,))
                if ticket is None:
                    break
                self._test[self._next_test] = ticket[0]
                self._next_test += 1
        else:
            step = self.frontier // self.effective_batch_size
            cutoff = self.end_index
            if self.test_interval > 0 and self.test_count:
                cutoff = min(cutoff, (step // self.test_interval + 1) * self.test_interval * self.effective_batch_size)
            while self._next_train < cutoff:
                ticket = self.queue.submit("train", (self._next_train,))
                if ticket is None:
                    break
                self._train[self._next_train] = ticket[0]
                self._next_train += 1

    def batch(self, indices):
        indices = tuple(indices)
        if not indices or indices != tuple(range(indices[0], indices[0] + len(indices))):
            raise ValueError("batch must be a nonempty contiguous canonical range")
        if indices[0] < self.frontier or indices[-1] >= self.end_index:
            raise ValueError("batch is outside the live canonical stream")
        if not all(index in self._train for index in indices):
            return None
        return self.queue.batch(tuple(self._train[index] for index in indices))

    def test_batch(self, step, indices):
        if self.test_step != step:
            raise RuntimeError(f"test request at update {step} outside scheduled test phase {self.test_step}")
        if any(not self.test_start <= index < self.test_start + self.test_count for index in indices):
            raise ValueError("test request outside canonical test selection")
        for index in indices:
            if index not in self._test and index in self._test_done:
                ticket = self.queue.submit("test", (index,))
                if ticket is None:
                    return None
                self._test[index] = ticket[0]
        if not all(index in self._test for index in indices):
            return None
        batch = self.queue.batch(tuple(self._test[index] for index in indices))
        if batch is None:
            return None
        for index in indices:
            self._test_leases[index] = self._test_leases.get(index, 0) + 1
        return batch

    def release_test(self, step, indices):
        if step != self.test_step:
            raise RuntimeError("test completion belongs to a different phase")
        released = []
        for index in indices:
            leases = self._test_leases.get(index, 0)
            if leases <= 0 or index not in self._test:
                raise RuntimeError(f"test sample {index} has no live lease")
            if leases == 1:
                released.append(self._test.pop(index))
                del self._test_leases[index]
                self._test_done.add(index)
            else:
                self._test_leases[index] = leases - 1
        if released:
            self.queue.release(tuple(released))
        self._test_released = len(self._test_done)

    def begin_test(self, step):
        frontier = self.frontier // self.effective_batch_size
        if step != frontier:
            self.commit(step)
        if self.test_count and self.test_step != step:
            raise RuntimeError("test begins outside the scheduled update boundary")

    def end_test(self, step):
        if not self.test_count:
            return
        if self.test_step != step or self._test_released != self.test_count or self._test or self._test_leases:
            raise RuntimeError("test phase ended with unconsumed sample ownership")
        self.test_step = None
        self._next_test = self.test_start
        self._test_released = 0
        self._test_done.clear()
        self._test_leases.clear()

    def commit(self, completed_step):
        target = completed_step * self.effective_batch_size
        if target != self.frontier + self.effective_batch_size:
            raise ValueError("commit must advance exactly one effective update")
        indices = tuple(range(self.frontier, target))
        if not all(index in self._train for index in indices):
            raise RuntimeError("cannot commit an update whose samples were not admitted")
        self.queue.release(tuple(self._train[index] for index in indices))
        for index in indices:
            del self._train[index]
        self.frontier = target
        if self.test_interval > 0 and self.test_count and completed_step % self.test_interval == 0:
            self.test_step = completed_step

    def set_worker_limit(self, count):
        self.pipeline.set_worker_limit(count)

    def workers_within_limit(self):
        self.pipeline.poll_ready()
        return self.pipeline.workers_within_limit()

    def close(self):
        self.queue.close()
        self._train.clear()
        self._test.clear()
        self._test_leases.clear()
