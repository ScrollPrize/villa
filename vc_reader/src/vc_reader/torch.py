"""Optional PyTorch integration: whole CUDA batches, ordinary tensor outputs.

PyTorch is imported only when this module or Pipeline.batch() is used. Dataset
selection, batching and multiprocessing remain the wrapped loader's job.
"""
from collections.abc import Mapping
import operator
import os

import numpy as np
import torch

from .cuda import DeviceArray

__all__ = ["CudaLoader"]


def _main_process():
    if torch.utils.data.get_worker_info() is not None:
        raise RuntimeError("CUDA batches must run in the training process; wrap the DataLoader with CudaLoader")


def _targets(pipeline, batch):
    _main_process()
    if pipeline.backend != "cuda":
        raise ValueError("batch processing requires a CUDA pipeline")
    if not isinstance(batch, Mapping):
        raise TypeError("a batch must be a mapping of target names to batched arrays")
    targets = {k: v for k, v in batch.items() if k in pipeline.targets and v is not None}
    if not targets:
        raise ValueError("batch has no declared array targets")
    count = None
    for key, value in targets.items():
        if not isinstance(value, (torch.Tensor, np.ndarray, DeviceArray)):
            raise TypeError(f"{key}: expected a Tensor, NumPy array or DeviceArray")
        if not value.ndim or value.shape[0] == 0:
            raise ValueError(f"{key}: batch dimension must be nonempty")
        if count is not None and value.shape[0] != count:
            raise ValueError("all targets must have the same batch size")
        count = value.shape[0]
        if isinstance(value, torch.Tensor) and value.requires_grad:
            raise ValueError("augmentation does not support autograd inputs")
    return targets, count


def _seeds(values, count):
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            raise ValueError("seeds must be on CPU")
        # Tensor.__index__ converts through signed int64 even for uint64.
        # tolist() preserves the full range emitted by augment.seed().
        values = values.tolist()
    result = [operator.index(value) for value in values]
    if len(result) != count:
        raise ValueError("seeds must contain one integer per sample")
    if any(value < 0 or value >= 1 << 64 for value in result):
        raise ValueError("seeds must be unsigned 64-bit integers")
    return result


def _params(params, count):
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise TypeError("params must map parameter names to CPU scalar sequences")
    result = {}
    for name, values in params.items():
        if not isinstance(name, str):
            raise TypeError("parameter names must be strings")
        if isinstance(values, DeviceArray) or (isinstance(values, torch.Tensor) and values.device.type != "cpu"):
            raise ValueError("parameters must be on CPU")
        values = np.asarray(values)
        if values.shape != (count,):
            raise ValueError(f"parameter {name} needs one value per sample")
        result[name] = values.tolist()
    return result


def _device(device, targets):
    resident = []
    for value in targets.values():
        if isinstance(value, DeviceArray):
            value._ready()
            resident.append(torch.device("cuda", value.device))
        elif isinstance(value, torch.Tensor) and value.is_cuda:
            resident.append(value.device)
    selected = torch.device(device) if device is not None else (resident[0] if resident else torch.device("cuda"))
    if selected.type != "cuda":
        raise ValueError("device must be a CUDA device")
    if selected.index is None:
        selected = torch.device("cuda", torch.cuda.current_device())
    if any(other != selected for other in resident):
        raise ValueError("CUDA targets must already be on the selected device")
    return selected


def _record(value, stream):
    if isinstance(value, torch.Tensor) and value.is_cuda:
        value.record_stream(stream)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _record(item, stream)


class _PendingBatch:
    def __init__(self, result, outputs, owner, inputs, stream, event):
        self.result, self.outputs, self.owner = result, outputs, owner
        self.inputs, self.stream, self.event = inputs, stream, event

    def wait(self):
        # Only this batch's producer work must finish. Never synchronize the
        # entire device (which would also wait for the training model).
        self.event.synchronize()
        self.owner.check()
        consumer = torch.cuda.current_stream(self.stream.device)
        for value in self.outputs.values():
            _record(value, consumer)
        result = self.result
        self.close()
        return result

    def close(self):
        # Also used on early iterator close and exceptions. In particular, keep
        # pinned host buffers and imported native allocations alive until copied.
        if self.event is not None:
            self.event.synchronize()
        self.inputs = self.owner = self.outputs = self.result = self.event = None


def _submit(pipeline, batch, seeds, device=None, stream=None, params=None):
    targets, count = _targets(pipeline, batch)
    seeds = _seeds(seeds, count)
    params = _params(params, count)
    device = _device(device, targets)
    stream = stream if stream is not None else torch.cuda.current_stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    arrays, allocated = {}, []

    def allocate(shape, dtype, index):
        # NumPy supplies the dtype mapping, never an image buffer. Returning
        # torch-owned allocations lets record_stream protect later consumers.
        dtype = torch.from_numpy(np.empty((), dtype=np.dtype(dtype))).dtype
        result = torch.empty(tuple(shape), dtype=dtype, device=torch.device("cuda", index))
        allocated.append(result)  # retained even if native execution raises
        return result

    try:
        with torch.cuda.device(device), torch.cuda.stream(stream):
            for key, value in targets.items():
                if isinstance(value, DeviceArray):
                    value = torch.from_dlpack(value)
                elif isinstance(value, np.ndarray):
                    value = torch.from_numpy(np.require(value, requirements=["C", "A", "W"]))
                if value.device.type not in ("cpu", "cuda"):
                    raise ValueError(f"{key}: expected CPU or CUDA input")
                value = value.to(device, non_blocking=True)
                _record(value, stream)
                continuous = pipeline.targets[key]._constructor.__name__ not in ("Label", "ValidMask")
                if continuous and value.dtype == torch.uint8:
                    value = value.to(torch.float32)
                if continuous and value.dtype not in (torch.float32, torch.float64):
                    raise TypeError(f"{key}: continuous targets require uint8, float32 or float64")
                arrays[key] = value.contiguous()
            outputs, _, owner = pipeline._native.run_batch(
                arrays, seeds, params, False, True, {}, allocator=allocate, stream=stream.cuda_stream)
            event = stream.record_event()
        result = {k: v for k, v in batch.items() if k not in targets}
        result.update(outputs)
        return _PendingBatch(result, outputs, owner, (targets, arrays), stream, event)
    except BaseException:
        # Pending native reads/copies must not outlive Python-owned inputs or
        # outputs, even when validation/allocation fails midway through a batch.
        stream.synchronize()
        raise


def _batch(pipeline, batch, *, seeds, device=None, params=None):
    pending = _submit(pipeline, batch, seeds, device, params=params)
    try:
        return pending.wait()
    finally:
        pending.close()


class CudaLoader:
    """Wrap an iterable of batch dictionaries with CUDA preprocessing.

    The wrapped DataLoader should return CPU arrays, ideally pinned tensors.
    Its workers never execute the CUDA pipeline. Already-resident CUDA arrays
    are also accepted when the input iterable runs in the training process.
    Only declared targets move to CUDA; metadata is preserved. uint8 continuous
    targets become float32 on device; labels retain their dtype.

    prefetch=True queues one batch ahead on a private CUDA stream. Each batch
    is completed and validated before yielding ordinary PyTorch tensors. Use
    prefetch=False for strictly on-demand iteration. Close an iterator when
    retaining it after an early break to promptly release its pending batch.

    Default seeds depend on seed, epoch and the sample's position in this
    rank's iteration order. set_epoch() changes augmentation seeds, not the
    underlying sampler. seed_key selects a CPU batch field of explicit uint64
    seeds for reproducibility independent of ordering, batching or rank.
    """
    def __init__(self, loader, pipeline, *, device="cuda", seed=0, seed_key=None, prefetch=True):
        _main_process()
        if pipeline.backend != "cuda":
            raise ValueError("CudaLoader requires a CUDA pipeline")
        if seed_key is not None and (not isinstance(seed_key, str) or seed_key in pipeline.targets):
            raise ValueError("seed_key must name a metadata field, not an array target")
        if not isinstance(prefetch, bool):
            raise TypeError("prefetch must be a bool")
        self.loader, self.pipeline = loader, pipeline
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("device must be a CUDA device")
        self.seed, self.seed_key = operator.index(seed), seed_key
        self.prefetch, self.epoch, self._pid = prefetch, 0, os.getpid()

    def __len__(self):
        return len(self.loader)

    def set_epoch(self, epoch):
        """Set augmentation epoch; call your distributed sampler separately."""
        self.epoch = operator.index(epoch)

    def __iter__(self):
        _main_process()
        if os.getpid() != self._pid:
            raise RuntimeError("create CudaLoader in the training process, not before forking it")
        # Snapshot the epoch when the iterator is created, not on first next().
        return self._iterate(self.epoch)

    def _iterate(self, epoch):
        from .augment import seed as make_seed
        iterator = iter(self.loader)
        stream = torch.cuda.Stream(device=self.device)
        offset = 0
        pending = None

        def submit(batch):
            nonlocal offset
            _, count = _targets(self.pipeline, batch)
            seeds = batch[self.seed_key] if self.seed_key is not None else [
                make_seed(self.seed, epoch, offset + i) for i in range(count)]
            result = _submit(self.pipeline, batch, seeds, self.device, stream)
            offset += count
            return result

        try:
            if not self.prefetch:
                for batch in iterator:
                    pending = submit(batch)
                    result = pending.wait()
                    pending = None
                    yield result
                return
            try:
                pending = submit(next(iterator))
            except StopIteration:
                return
            while pending is not None:
                result = pending.wait()
                pending = None
                try:
                    batch = next(iterator)
                except StopIteration:
                    pass
                else:
                    pending = submit(batch)
                yield result
        finally:
            if pending is not None:
                pending.close()
