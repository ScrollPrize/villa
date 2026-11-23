"""Optional PyTorch integration: whole CUDA batches, ordinary tensor outputs.

PyTorch is imported only when this module or Pipeline.batch() is used. Dataset
selection, batching and multiprocessing remain the wrapped loader's job.
"""
from collections.abc import Mapping
import functools
import operator
import os

import numpy as np
import torch

from .cuda import DeviceArray
from .transforms._pipeline import _batch_params as _params, _batch_seeds as _seeds

__all__ = ["CudaLoader", "structure_tensor", "sample_array", "surface_points", "surface_derivatives"]


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
            raise ValueError("transform does not support autograd inputs")
    return targets, count


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
        # pinned host buffers and imported C++ allocations alive until copied.
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
        allocated.append(result)  # retained even if C++ execution raises
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
                # uint8 continuous targets upload as uint8 and become float32 on the device.
                if pipeline._engine.continuous(key) and value.dtype not in (torch.uint8, torch.float32, torch.float64):
                    raise TypeError(f"{key}: continuous targets require uint8, float32 or float64")
                arrays[key] = value.contiguous()
            outputs, _, owner = pipeline._engine.run_batch(arrays, seeds, params, allocate, stream.cuda_stream)
            event = stream.record_event()
        result = {k: v for k, v in batch.items() if k not in targets}
        result.update(outputs)
        return _PendingBatch(result, outputs, owner, (targets, arrays), stream, event)
    except BaseException:
        # Pending C++ reads/copies must not outlive Python-owned inputs or
        # outputs, even when validation/allocation fails midway through a batch.
        stream.synchronize()
        raise


def _batch(pipeline, batch, *, seeds, device=None, params=None):
    pending = _submit(pipeline, batch, seeds, device, params=params)
    try:
        return pending.wait()
    finally:
        pending.close()


@functools.lru_cache(maxsize=64)
def _structure_pipeline(sigma, rho, scale, backend, backward):
    from . import transforms as A
    targets = {"image": A.Image(valid=None, refill=False)}
    if backward:
        targets["cotangent"] = A.ScalarField()
    return A.Pipeline([A.StructureTensor(sigma=sigma, rho=rho, scale=scale,
                       gradient="cotangent" if backward else "", output="result")],
                      targets=targets, backend=backend)


def _structure_run(image, sigma, rho, scale, cotangent=None):
    pipeline = _structure_pipeline(sigma, rho, scale, "cuda" if image.is_cuda else "cpu", cotangent is not None)
    batch = {"image": image.detach() if image.is_cuda else image.detach().cpu()}
    if cotangent is not None:
        batch["cotangent"] = cotangent.detach() if image.is_cuda else cotangent.detach().cpu()
    if image.is_cuda:
        return _batch(pipeline, batch, seeds=[0] * len(image), device=image.device)["result"]
    return torch.stack([pipeline({k: v[i] for k, v in batch.items()}, seed=0)["result"] for i in range(len(image))]).to(image.device)


class _StructureTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, sigma, rho, scale):
        ctx.save_for_backward(image)
        ctx.options = sigma, rho, scale
        return _structure_run(image, sigma, rho, scale)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, cotangent):
        image, = ctx.saved_tensors
        return _structure_run(image, *ctx.options, cotangent=cotangent), None, None, None


def structure_tensor(image, *, sigma=1.0, rho=1.5, scale=1.0):
    """Pavel structure components of (B,1,Y,X) or (B,1,Z,Y,X), with first-order autograd.

    Native CPU/CUDA kernels (CPU staging for MPS) use zero boundaries and return yy,yx,xx or
    zz,zy,zx,yy,yx,xx. sigma smooths input and rho smooths components; zero
    disables either smoothing. Input/output preserve float32 or float64.
    """
    if image.ndim not in (4, 5) or image.shape[1] != 1 or not all(image.shape):
        raise ValueError("structure_tensor requires nonempty (B,1,Y,X) or (B,1,Z,Y,X)")
    if image.dtype not in (torch.float32, torch.float64) or image.device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("structure_tensor requires float32/float64 on CPU, CUDA or MPS")
    return _StructureTensor.apply(image, float(sigma), float(rho), float(scale))


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
    rank's iteration order. set_epoch() changes transform seeds, not the
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
        """Set transform epoch; call your distributed sampler separately."""
        self.epoch = operator.index(epoch)

    def __iter__(self):
        _main_process()
        if os.getpid() != self._pid:
            raise RuntimeError("create CudaLoader in the training process, not before forking it")
        # Snapshot the epoch when the iterator is created, not on first next().
        return self._iterate(self.epoch)

    def _iterate(self, epoch):
        from .transforms import seed as make_seed
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


def _resident_forward(values, points, order, boundary, fill):
    from . import _reader, cuda
    out = values.new_empty((values.shape[0], values.shape[1], points.shape[1]))
    view = lambda x: cuda._StreamInput(x.detach(), _reader._dlpack_stream()) if x.is_cuda else x.detach().numpy()
    _reader._sample_dense(view(values), view(points), view(out), order, boundary, fill)
    return out


def _resident_backward(values, points, cotangent, need_values, need_points, order, boundary, fill):
    from . import _reader, cuda
    vg = torch.empty_like(values) if need_values else values.new_empty(0)
    pg = torch.empty_like(points) if need_points else points.new_empty(0)
    view = lambda x: cuda._StreamInput(x.detach(), _reader._dlpack_stream()) if x.is_cuda else x.detach().numpy()
    _reader._sample_dense_backward(view(values), view(points), view(cotangent.contiguous()), view(vg), view(pg), order, boundary, fill)
    return vg, pg


# Dispatcher schemas keep native calls opaque to Dynamo/AOTAutograd. Meta kernels
# describe allocations, so compiled training retains surrounding tensor graphs.
_resident_ops = torch.library.Library("vc_reader", "FRAGMENT")
_resident_ops.define("resident_sample(Tensor values, Tensor points, int order, str boundary, float fill) -> Tensor")
_resident_ops.define("resident_sample_backward(Tensor values, Tensor points, Tensor cotangent, bool need_values, bool need_points, int order, str boundary, float fill) -> (Tensor, Tensor)")
for _dispatch in ("CPU", "CUDA"):
    _resident_ops.impl("resident_sample", _resident_forward, _dispatch)
    _resident_ops.impl("resident_sample_backward", _resident_backward, _dispatch)
_resident_ops.impl("resident_sample", lambda values, points, order, boundary, fill:
                   values.new_empty((values.shape[0], values.shape[1], points.shape[1])), "Meta")
_resident_ops.impl("resident_sample_backward", lambda values, points, cotangent, need_values, need_points, order, boundary, fill:
                   (torch.empty_like(values) if need_values else values.new_empty(0),
                    torch.empty_like(points) if need_points else points.new_empty(0)), "Meta")


class _SampleArray(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, points, order, boundary, fill):
        ctx.save_for_backward(values, points)
        ctx.options = order, boundary, fill
        return torch.ops.vc_reader.resident_sample(values, points, order, boundary, fill)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, cotangent):
        values, points = ctx.saved_tensors
        vg, pg = torch.ops.vc_reader.resident_sample_backward(
            values, points, cotangent, ctx.needs_input_grad[0], ctx.needs_input_grad[1], *ctx.options)
        return vg if ctx.needs_input_grad[0] else None, pg if ctx.needs_input_grad[1] else None, None, None, None


def sample_array(values, points, *, order=1, boundary="zeros", fill=0):
    """Sample B,C,[Z,]Y,X at B,...,D array indices; return B,C,... float32.

    Native CPU/CUDA forward and first-order gradients for values and coordinates.
    CUDA value gradients use atomic addition, like torch grid_sample. Coordinate
    gradients and forward values have no competing floating-point writes.
    MPS stages through CPU and retains autograd. Double backward is unsupported.
    """
    if values.ndim not in (4, 5) or points.ndim < 3 or points.shape[0] != values.shape[0] or points.shape[-1] != values.ndim-2:
        raise ValueError('expected B,C,[Z,]Y,X values and B,...,D points')
    if values.device != points.device:
        raise ValueError('values and points must be on the same device')
    device = values.device
    if device.type == 'mps':
        return sample_array(values.cpu(), points.cpu(), order=order, boundary=boundary, fill=fill).to(device)
    shape = points.shape[1:-1]
    result = _SampleArray.apply(values.float().contiguous(), points.float().reshape(values.shape[0], -1, points.shape[-1]).contiguous(), int(order), boundary, float(fill))
    return result.reshape(values.shape[:2] + shape)


def _surface_derivatives_impl(xyz, valid):
    """Prepare reusable bicubic derivatives for fixed B,H,W,3 surface grids."""
    from . import _reader, cuda
    xyz, valid = xyz.float().contiguous(), valid.bool().contiguous()
    if xyz.device.type == "mps":
        return surface_derivatives(xyz.cpu(), valid.cpu()).to(xyz.device)
    out = xyz.new_empty((xyz.shape[0], 3, *xyz.shape[1:]))
    view = lambda x: cuda._StreamInput(x.detach(), _reader._dlpack_stream()) if x.is_cuda else x.detach().numpy()
    _reader._prepare_surfaces(view(xyz), view(valid), view(out))
    return out


def surface_points(xyz, valid, rc, *, derivatives=None, border=False):
    """Bicubic positions, row/column tangents and support for B,H,W,3 grids.

    Queries B,Q,2 use row,column indices. Invalid queries return zero vectors
    and false support. This operation builds fixed supervision: inputs requiring
    gradients must instead use a differentiable field sampling operation.
    """
    if any(t.requires_grad for t in (xyz, rc)):
        raise ValueError('surface_points takes fixed supervision grids and queries')
    device = xyz.device
    if device.type == 'mps':
        return tuple(t.to(device) for t in surface_points(xyz.cpu(), valid.cpu(), rc.cpu(), border=border))
    xyz, valid, rc = xyz.float().contiguous(), valid.bool().contiguous(), rc.float().contiguous()
    if derivatives is None:
        derivatives = surface_derivatives(xyz, valid)
    return torch.ops.vc_reader.resident_surface(xyz, valid, rc, derivatives, border)


def _surface_points_impl(xyz, valid, rc, derivatives, border):
    from . import _reader, cuda
    view = lambda x: cuda._StreamInput(x, _reader._dlpack_stream()) if x.is_cuda else x.numpy()
    device = xyz.device
    shape = (*rc.shape[:-1], 3)
    p, row, col = (xyz.new_empty(shape) for _ in range(3))
    ok = torch.empty(rc.shape[:-1], device=device, dtype=torch.bool)
    _reader._sample_surfaces(view(xyz), view(valid), view(derivatives), view(rc), view(p), view(row), view(col), view(ok), 2, border)
    return p, row, col, ok


def _surface_meta(xyz, valid, rc, derivatives, border):
    shape = (*rc.shape[:-1], 3)
    return (xyz.new_empty(shape), xyz.new_empty(shape), xyz.new_empty(shape),
            torch.empty(rc.shape[:-1], device=xyz.device, dtype=torch.bool))


_resident_ops.define("resident_derivatives(Tensor xyz, Tensor valid) -> Tensor")
_resident_ops.define("resident_surface(Tensor xyz, Tensor valid, Tensor rc, Tensor derivatives, bool border) -> (Tensor, Tensor, Tensor, Tensor)")
for _dispatch in ("CPU", "CUDA"):
    _resident_ops.impl("resident_derivatives", _surface_derivatives_impl, _dispatch)
    _resident_ops.impl("resident_surface", _surface_points_impl, _dispatch)
_resident_ops.impl("resident_derivatives", lambda xyz, valid: xyz.new_empty((xyz.shape[0], 3, *xyz.shape[1:])), "Meta")
_resident_ops.impl("resident_surface", _surface_meta, "Meta")


def surface_derivatives(xyz, valid):
    """Reusable bicubic derivatives for fixed B,H,W,3 supervision grids."""
    if xyz.requires_grad:
        raise ValueError('surface_derivatives takes fixed supervision grids')
    if xyz.device.type == 'mps':
        return surface_derivatives(xyz.cpu(), valid.cpu()).to(xyz.device)
    return torch.ops.vc_reader.resident_derivatives(xyz.float().contiguous(), valid.bool().contiguous())
