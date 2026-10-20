"""Container adaptation, execution and replay for native transform pipelines."""
from copy import deepcopy

import numpy as np

from .. import _transforms
from ._config import _unwrap, Image, Label, ValidMask


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

    @property
    def steps(self):
        return self._plan.steps


class Pipeline:
    """Construct once; pass an explicit seed for each sample occurrence.

    Arrays use (C, Y, X)/(C, Z, Y, X); points use (N, D). Metadata passes
    through and undeclared arrays raise. Inputs are preserved by default.
    Pass inplace=True when handing off fresh reader buffers: transforms may
    modify them and outputs may share their storage. Always use returned arrays,
    since geometry and shape-changing transforms allocate new buffers.
    compact=True keeps eligible float32 median/intensity chains in a lossless
    palette representation. Weighted statistics can differ slightly in rounding;
    compact=False retains the ordinary reduction order. Outputs keep their dtype.
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
        self.targets = deepcopy(targets) if targets is not None else {
            "image": Image(), "mask": Label(), "padding_mask": ValidMask(),
        }
        self._engine = _transforms.Pipeline(_unwrap(self.transforms), _unwrap(self.targets), self.compact, backend)

    def with_targets(self, **targets):
        """Return a new pipeline with added or replaced target declarations.

        Values describe how arrays are treated, e.g. extra_label=Label() or
        confidence=ScalarField(). Supply the actual arrays under those names
        when calling the returned pipeline. Existing targets, transforms,
        compact mode and the requested backend are preserved; this pipeline
        is unchanged. Replay plans belong to their original pipeline.
        """
        return Pipeline(self.transforms, targets={**self.targets, **targets},
                        compact=self.compact, backend=self._requested_backend)

    def __getstate__(self):
        return self.transforms, self.targets, self.compact, self._requested_backend

    def __setstate__(self, state):
        self.__init__(state[0], targets=state[1], compact=state[2] if len(state) > 2 else True,
                      backend=state[3] if len(state) > 3 else "cpu")

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
        arrays, sources, timings, plan = result[:4]
        owner = _ExportOwners([result[4] if len(result) == 5 else None,
                               *(v._export_owner for v in inputs)])
        return ({k: [DeviceArray(*level, _export_owner=owner) for level in v] if isinstance(v, list) else DeviceArray(*v, _export_owner=owner)
                 for k, v in arrays.items()}, sources, timings, plan)

    @staticmethod
    def _outputs(sample, converters, result, profile):
        arrays, sources, timings, _ = result
        out = {k: v for k, v in sample.items() if k not in converters}
        for key, value in arrays.items():
            converter = converters.get(key, converters.get(sources[key]))
            out[key] = value if converter is None else (
                [converter(level) for level in value] if isinstance(value, list) else converter(value)
            )
        if profile:
            out["_transform_perf"] = timings
        return out

    def __call__(self, sample, *, seed, profile=False, inplace=False, params=None):
        device = self._device_inputs(sample)
        if device is not None:
            result = self._device_result(self._engine.run_device(device, seed, False, profile, inplace, params=params or {}),
                                         [sample[k] for k in device] if inplace else ())
            return self._outputs(sample, dict.fromkeys(device), result, profile)
        arrays, converters = self._inputs(sample, inplace)
        result = self._engine.run(arrays, seed, False, profile, inplace, params=params or {})
        return self._outputs(sample, converters, result, profile)

    def map_points(self, sample, *, shape, seed):
        """Map CPU point metadata without allocating or transforming an image.

        Declare each array as Points. shape is the input spatial (Y, X) or
        (Z, Y, X) extent. Flip, Rotate90, Transpose, CropPad and Affine use the
        same native geometry and seed as dense CPU/CUDA execution. Other
        transforms raise. Returns NumPy arrays or CPU tensors matching inputs;
        point IDs and other non-array metadata pass through unchanged.
        """
        arrays, converters = self._inputs(sample)
        return self._outputs(sample, converters,
                             self._engine.map_points(arrays, shape, seed), False)

    def sample(self, sample, *, seed, params=None):
        device = self._device_inputs(sample)
        if device is not None:
            return Plan(self, self._engine.run_device(device, seed, True, False, False, params=params or {})[3])
        arrays, _ = self._inputs(sample)
        return Plan(self, self._engine.run(arrays, seed, True, False, params=params or {})[3])

    def batch(self, batch, *, seeds, device=None, params=None):
        """Transform a batch, returning PyTorch CUDA tensors and unchanged metadata.

        Requires the optional PyTorch integration and a CUDA pipeline. Dense
        targets use (B, C, Y, X) or (B, C, Z, Y, X); seeds has one uint64 per
        sample. NumPy/CPU targets are uploaded once per target; CUDA inputs stay
        on device. uint8 continuous targets are cast to float32 on the GPU;
        other dtypes are preserved. Inputs are not modified.

        Work uses PyTorch's current stream and completes/checks once per batch.
        CudaLoader uses the same executor with one batch of GPU prefetch.
        params maps each Param(name) to one CPU scalar per sample. Compatible
        kernel stages execute across samples in a single launch. Filters,
        reductions and FFTs retain their necessary stages; randomness and
        statistics remain independent for each sample.
        Uniform outputs are stacked tensors; differing sample shapes are lists.
        Undeclared fields, including tensor metadata, pass through unchanged.
        """
        from ..torch import _batch
        return _batch(self, batch, seeds=seeds, device=device, params=params)

    def apply(self, sample, plan):
        if not isinstance(plan, Plan) or plan._pipeline is not self:
            raise ValueError("plan belongs to another pipeline; use plan.apply(sample)")
        device = self._device_inputs(sample)
        if device is not None:
            result = self._device_result(self._engine.replay_device(device, plan._plan))
            return self._outputs(sample, dict.fromkeys(device), result, False)
        arrays, converters = self._inputs(sample)
        return self._outputs(sample, converters, self._engine.replay(arrays, plan._plan), False)

