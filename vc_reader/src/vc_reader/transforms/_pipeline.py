"""Container adaptation, execution and replay for native transform pipelines."""
from collections.abc import Mapping
from copy import deepcopy
import operator

import numpy as np

from .. import _transforms
from ._config import _Config, _unwrap


def _as_array(value, inplace=False):
    if isinstance(value, np.ndarray):
        array, converter = value, None
    elif type(value).__module__.split(".")[0] == "torch":
        import torch
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.requires_grad:
            raise ValueError("transform requires CPU tensors without autograd")
        array, converter = value.numpy(), torch.from_numpy
    else:
        raise TypeError("targets must be NumPy arrays or CPU Torch tensors")
    if not array.dtype.isnative:
        raise TypeError("transform requires native-endian arrays")
    if inplace and (not array.flags.c_contiguous or not array.flags.writeable or not array.flags.aligned):
        raise ValueError("inplace transform requires writable contiguous aligned arrays")
    return np.require(array, requirements=["C", "A"]), converter


def _is_torch(value):
    return type(value).__module__.split(".")[0] == "torch"


def _batch_seeds(values, count):
    if _is_torch(values):
        if values.device.type != "cpu":
            raise ValueError("seeds must be on CPU")
        # Tensor.__index__ converts through signed int64 even for uint64.
        # tolist() preserves the full range emitted by transforms.seed().
        values = values.tolist()
    result = [operator.index(value) for value in values]
    if len(result) != count:
        raise ValueError("seeds must contain one integer per sample")
    if any(value < 0 or value >= 1 << 64 for value in result):
        raise ValueError("seeds must be unsigned 64-bit integers")
    return result


def _batch_params(params, count):
    from ..cuda import DeviceArray
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise TypeError("params must map parameter names to CPU scalar sequences")
    result = {}
    for name, values in params.items():
        if not isinstance(name, str):
            raise TypeError("parameter names must be strings")
        if isinstance(values, DeviceArray) or (_is_torch(values) and values.device.type != "cpu"):
            raise ValueError("parameters must be on CPU")
        values = np.asarray(values)
        if values.shape != (count,):
            raise ValueError(f"parameter {name} needs one value per sample")
        result[name] = values.tolist()
    return result


def _direct_setup(config, data, targets, compact):
    """The pipeline, sample and bare-array field of a direct call; see ``_Config.__call__``."""
    from .. import cuda
    bare = not isinstance(data, Mapping)
    key = None
    if bare:
        if targets is not None:
            raise ValueError("targets apply to dict calls")
        key, target = _transforms.direct_input(config._config)
        sample, declared = {key: data}, {key: target}
    else:
        sample = data
        declared = _transforms.default_targets() if targets is None else _unwrap(targets)
    backend = "cuda" if any(cuda.is_device_array(v) for v in sample.values()) else "cpu"
    if targets is not None and not all(isinstance(v, _Config) for v in targets.values()):
        return Pipeline._direct(config, declared, compact, backend), sample, key
    schema = None if targets is None else tuple(sorted((k, repr(v)) for k, v in targets.items()))
    cache_key = (bare, backend, bool(compact), schema)
    pipeline = config._direct_pipelines.get(cache_key)
    if pipeline is None:
        pipeline = config._direct_pipelines[cache_key] = Pipeline._direct(config, declared, compact, backend)
    return pipeline, sample, key


def _direct_result(out, key):
    return out if key is None else out[_transforms.direct_output(key, list(out))]


def _direct(config, data, targets, compact, inplace):
    from .. import cuda
    pipeline, sample, key = _direct_setup(config, data, targets, compact)
    torch_cuda = [k for k, v in sample.items() if _is_torch(v) and getattr(v, "is_cuda", False)]
    if torch_cuda:
        sample = {k: cuda.wrap(v) if k in torch_cuda else v for k, v in sample.items()}
    out = pipeline(sample, seed=_transforms.DIRECT_SEED, inplace=inplace)
    if torch_cuda:
        import torch
        out = {k: torch.from_dlpack(v) if isinstance(v, cuda.DeviceArray) else v for k, v in out.items()}
    return _direct_result(out, key)


def _direct_batch(config, data, targets, compact):
    pipeline, sample, key = _direct_setup(config, data, targets, compact)
    arrays = {k: v for k, v in sample.items() if k in pipeline.targets and v is not None}
    sizes = {v.shape[0] if v.ndim else 0 for v in arrays.values()}
    if len(sizes) != 1 or not (count := sizes.pop()):
        raise ValueError("batch targets need one nonempty, shared leading dimension")
    if pipeline.backend == "cuda":
        return _direct_result(pipeline.batch(sample, seeds=[_transforms.DIRECT_SEED] * count), key)
    return _direct_result(pipeline._host_batch(sample, [_transforms.DIRECT_SEED] * count), key)


class Plan:
    """Sample-specific replay record, including data-dependent deformations."""
    def __init__(self, pipeline, plan):
        self._pipeline, self._plan = pipeline, plan

    def apply(self, sample):
        return self._pipeline.apply(sample, self)

    def __getstate__(self):
        return self._pipeline, self._plan.dump()

    def __setstate__(self, state):
        self._pipeline = state[0]
        self._plan = self._pipeline._engine.load_plan(state[1])


class Pipeline:
    """Construct once; pass an explicit seed for each sample occurrence.

    Arrays use (C, Y, X)/(C, Z, Y, X); points use (N, D). Metadata passes
    through and undeclared arrays raise. Inputs are preserved by default.
    Pass inplace=True when handing off fresh reader buffers: transforms may
    modify them and outputs may share their storage. Always use returned arrays,
    since geometry and shape-changing transforms allocate new buffers.
    compact=True keeps eligible float32 median/intensity chains in a lossless
    palette representation. Weighted statistics can differ slightly in rounding;
    compact=False retains the ordinary reduction order. Outputs keep their dtype,
    except uint8 continuous targets (intensities 0..255, e.g. CT reads), which
    leave as float32 with the values float32 input would give; value-only
    transforms (median, Equalize, CLAHE, HistogramMatching) and blurs read them
    directly, and CUDA uploads them as uint8.
    backend="cpu" is the default. "cuda" requires the optional CUDA build and a
    device; "auto" selects CUDA when available, otherwise CPU. CUDA uses dense
    buffers regardless of compact. NumPy/CPU Torch inputs return host arrays;
    vc_reader.cuda.DeviceArray inputs and outputs remain on device. Device
    samples support image, scalar, label and validity-mask targets, all on one
    device. Calls complete before returning; DLPack consumers may use any stream.
    Only the primary benchmark transforms (and OneOf) support CUDA. Unsupported
    transforms raise when CUDA is selected; execution errors never retry on CPU.
    """
    def __init__(self, transforms, *, targets=None, compact=True, backend="cpu"):
        self.compact = bool(compact)
        self._requested_backend = backend
        self.transforms = tuple(deepcopy(tuple(transforms)))
        self._declared_targets = deepcopy(targets)
        declared = _transforms.default_targets() if targets is None else _unwrap(self._declared_targets)
        self._engine = _transforms.Pipeline(_unwrap(self.transforms), declared, self.compact, backend)
        self.targets = self._engine.targets

    @classmethod
    def _direct(cls, config, targets, compact, backend):
        """One transform outside a pipeline: always applied, run with DIRECT_SEED."""
        self = cls.__new__(cls)
        self.compact, self._requested_backend = bool(compact), backend
        self.transforms, self._declared_targets = (config,), targets
        self._engine = _transforms.direct_pipeline(config._config, targets, self.compact, backend)
        self.targets = self._engine.targets
        return self

    def __getstate__(self):
        return self.transforms, self._declared_targets, self.compact, self._requested_backend

    def __setstate__(self, state):
        self.__init__(state[0], targets=state[1], compact=state[2], backend=state[3])

    @property
    def backend(self):
        """Resolved execution backend (cpu or cuda)."""
        return self._engine.backend

    def _inputs(self, sample, inplace=False):
        arrays, converters = {}, {}
        for key, value in sample.items():
            if key in self.targets:
                if value is not None:
                    arrays[key], converters[key] = _as_array(value, inplace)
            elif isinstance(value, np.ndarray) or type(value).__module__.split(".")[0] == "torch":
                raise ValueError(f"array field {key!r} needs a target declaration")
        return arrays, converters

    def _device_inputs(self, sample):
        from ..cuda import DeviceArray
        if not any(isinstance(v, DeviceArray) for v in sample.values()):
            return None
        if self.backend != "cuda":
            raise ValueError("device inputs require a CUDA pipeline")
        arrays = {}
        for key, value in sample.items():
            if key in self.targets and value is not None:
                if not isinstance(value, DeviceArray):
                    raise TypeError("device samples require DeviceArray for every array target")
                value._ready()
                arrays[key] = value._array
            elif isinstance(value, (DeviceArray, np.ndarray)):
                raise ValueError(f"array field {key!r} needs a target declaration")
        return arrays

    @staticmethod
    def _device_result(result, inputs=()):
        from ..cuda import DeviceArray, _ExportOwners
        arrays, sources, plan, outputs = result
        owner = _ExportOwners([outputs, *(v._export_owner for v in inputs)])
        return ({k: [DeviceArray(*level, _export_owner=owner) for level in v] if isinstance(v, list) else DeviceArray(*v, _export_owner=owner)
                 for k, v in arrays.items()}, sources, plan)

    @staticmethod
    def _outputs(sample, converters, result):
        arrays, sources, _ = result
        out = {k: v for k, v in sample.items() if k not in converters}

        def convert(value, converter):
            return [convert(v, converter) for v in value] if isinstance(value, list) else converter(value)
        for key, value in arrays.items():
            converter = converters.get(key, converters.get(sources[key]))
            out[key] = value if converter is None else convert(value, converter)
        return out

    def _host_batch(self, sample, seeds, params=None):
        """CPU (B, ...) batch in one native call; outputs stacked when samples agree in shape."""
        arrays, converters = self._inputs(sample)
        count = {v.shape[0] if v.ndim else 0 for v in arrays.values()}
        if len(count) != 1 or not (n := count.pop()):
            raise ValueError("batch targets need one nonempty, shared leading dimension")
        result = self._engine.run_host_batch(arrays, _batch_seeds(seeds, n), _batch_params(params, n))
        return self._outputs(sample, converters, (*result, None))

    def __call__(self, sample, *, seed, inplace=False, params=None):
        device = self._device_inputs(sample)
        if device is not None:
            result = self._device_result(self._engine.run_device(device, seed, False, inplace, params=params or {}),
                                         [sample[k] for k in device] if inplace else ())
            return self._outputs(sample, dict.fromkeys(device), result)
        arrays, converters = self._inputs(sample, inplace)
        result = self._engine.run(arrays, seed, False, inplace, params=params or {})
        return self._outputs(sample, converters, result)

    def sample(self, sample, *, seed, params=None):
        device = self._device_inputs(sample)
        if device is not None:
            return Plan(self, self._engine.run_device(device, seed, True, False, params=params or {})[2])
        arrays, _ = self._inputs(sample)
        return Plan(self, self._engine.run(arrays, seed, True, params=params or {})[2])

    def batch(self, batch, *, seeds, device=None, params=None):
        """Transform a batch, returning PyTorch CUDA tensors and unchanged metadata.

        A batch whose targets are all vc_reader.cuda.DeviceArray runs without
        PyTorch and returns DeviceArray outputs, complete when the call returns.
        Otherwise this requires the optional PyTorch integration. Both need a CUDA pipeline. Dense
        targets use (B, C, Y, X) or (B, C, Z, Y, X); seeds has one uint64 per
        sample. NumPy/CPU targets are uploaded once per target; CUDA inputs stay
        on device. uint8 continuous targets (intensities 0..255) become float32
        on the GPU; other dtypes are preserved. Inputs are not modified.

        Work uses PyTorch's current stream and completes/checks once per batch.
        CudaLoader uses the same executor with one batch of GPU prefetch.
        params maps each Param(name) to one CPU scalar per sample. Compatible
        kernel stages execute across samples in a single launch. Filters,
        reductions and FFTs retain their necessary stages; randomness and
        statistics remain independent for each sample.
        Uniform outputs are stacked tensors; differing sample shapes are lists.
        Undeclared fields, including tensor metadata, pass through unchanged.
        """
        device_batch = self._device_batch(batch, seeds, device, params)
        if device_batch is not None:
            return device_batch
        from ..torch import _batch
        return _batch(self, batch, seeds=seeds, device=device, params=params)

    def _device_batch(self, batch, seeds, device, params):
        from .. import _reader
        from ..cuda import DeviceArray, empty
        if not isinstance(batch, Mapping):
            return None
        targets = {k: v for k, v in batch.items() if k in self.targets and v is not None}
        if not targets or not all(isinstance(v, DeviceArray) for v in targets.values()):
            return None
        if self.backend != "cuda":
            raise ValueError("batch processing requires a CUDA pipeline")
        counts = {v.shape[0] if v.ndim else 0 for v in targets.values()}
        if len(counts) != 1 or not (count := counts.pop()):
            raise ValueError("all targets must share one nonempty batch dimension")
        selected = {v.device for v in targets.values()}
        if len(selected) != 1 or (device is not None and _cuda_index(device) not in selected):
            raise ValueError("CUDA targets must already be on the selected device")
        arrays = {}
        for key, value in targets.items():
            value._ready()
            if self._engine.continuous(key) and value.dtype not in (np.uint8, np.float32, np.float64):
                raise TypeError(f"{key}: continuous targets require uint8, float32 or float64")
            arrays[key] = value
        # Outputs are the allocator's DeviceArrays (stacked, or one per sample when shapes differ).
        outputs, _, owner = self._engine.run_batch(
            {k: v._array for k, v in arrays.items()}, _batch_seeds(seeds, count), _batch_params(params, count),
            lambda shape, dtype, index: empty(tuple(shape), np.dtype(dtype), device=index),
            _reader._dlpack_stream())
        owner.wait()
        result = {k: v for k, v in batch.items() if k not in targets}
        result.update(outputs)
        return result

    def apply(self, sample, plan):
        if not isinstance(plan, Plan) or plan._pipeline is not self:
            raise ValueError("plan belongs to another pipeline; use plan.apply(sample)")
        device = self._device_inputs(sample)
        if device is not None:
            result = self._device_result(self._engine.replay_device(device, plan._plan))
            return self._outputs(sample, dict.fromkeys(device), result)
        arrays, converters = self._inputs(sample)
        return self._outputs(sample, converters, self._engine.replay(arrays, plan._plan))

