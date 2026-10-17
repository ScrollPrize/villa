"""Native augmentation with CPU and optional CUDA execution.

Python only constructs configurations and adapts dictionaries/array containers.
Parameter sampling, target routing, transforms, and replay run without the GIL.
"""
from copy import deepcopy
from functools import wraps
import numpy as np
from .. import _augment as _native


def _unwrap(value):
    if isinstance(value, _Config):
        return value._native
    if isinstance(value, (tuple, list)):
        return [_unwrap(v) for v in value]
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    return value


class _Config:
    """Picklable constructor recipe; contains no transform implementation."""
    def __init__(self, constructor, args, kwargs):
        self._constructor = constructor
        self._args, self._kwargs = deepcopy(args), deepcopy(kwargs)
        self._native = constructor(*_unwrap(self._args), **_unwrap(self._kwargs))

    def __reduce__(self):
        # Names are used only to reconstruct configuration after spawn/pickle.
        return _restore_config, (self._constructor.__name__, self._args, self._kwargs)

    def __repr__(self):
        args = [repr(v) for v in self._args]
        args.extend(f"{k}={v!r}" for k, v in self._kwargs.items())
        return f"{self._constructor.__name__}({', '.join(args)})"


def _restore_config(name, args, kwargs):
    return _Config(getattr(_native, name), args, kwargs)


def _factory(constructor):
    @wraps(constructor)
    def configure(*args, **kwargs):
        return _Config(constructor, args, kwargs)
    return configure


_constructors = (
    "Uniform Normal Choice BalancedContrast Image Label ScalarField ValidMask Points "
    "Vectors VectorField NormalField CoordinateField OneOf "
    "BrightnessMultiply BrightnessAdd Contrast Gamma Invert GaussianNoise RicianNoise "
    "ClipOutliers GaussianBlur MedianFilter Sharpen BrightnessGradient LocalContrast "
    "LocalGamma LocalSmoothing BlankRectangle SliceIllumination Smear Flip Rotate90 "
    "Transpose CropPad Affine Elastic SheetCompression LowResolution ThickSlice "
    "Morphology RandomBinaryOperator RemoveComponents ReplaceLabel Regions OneHot "
    "MaskImage DeepSupervision ToPseudo2D FromPseudo2D MedialSurface StripGeometry SurfacePerturb"
).split()
globals().update({name: _factory(getattr(_native, name)) for name in _constructors})
seed = _native.seed


def translate_crop_box(box, points, *, seed, **kwargs):
    """Translate a ZYX start/stop box, retaining supervised (N, 3) points.

    Filter invalid/unsupervised points before calling, then read the returned box.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return tuple(_native.translate_crop_box(box, points.ravel().tolist(), seed, **kwargs))


def _as_array(value, inplace=False):
    if isinstance(value, np.ndarray):
        array, converter = value, None
    elif type(value).__module__.split(".")[0] == "torch":
        import torch
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.requires_grad:
            raise ValueError("augmentation requires CPU tensors without autograd")
        array, converter = value.numpy(), torch.from_numpy
    else:
        raise TypeError("targets must be NumPy arrays or CPU Torch tensors")
    if not array.dtype.isnative:
        raise TypeError("augmentation requires native-endian arrays")
    if inplace and (not array.flags.c_contiguous or not array.flags.writeable or not array.flags.aligned):
        raise ValueError("inplace augmentation requires writable contiguous aligned arrays")
    return np.require(array, requirements=["C", "A"]), converter


class Plan:
    """Sample-specific native replay record, including data-dependent deformations."""
    def __init__(self, pipeline, native):
        self._pipeline, self._native = pipeline, native

    def apply(self, sample):
        return self._pipeline.apply(sample, self)

    def __getstate__(self):
        return self._pipeline, self._native.dump()

    def __setstate__(self, state):
        self._pipeline = state[0]
        self._native = self._pipeline._native.load_plan(state[1])

    @property
    def steps(self):
        return self._native.steps


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
    buffers regardless of compact, with the same host-array input/output API.
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
        self._native = _native.Pipeline(_unwrap(self.transforms), _unwrap(self.targets), self.compact, backend)

    def __getstate__(self):
        return self.transforms, self.targets, self.compact, self._requested_backend

    def __setstate__(self, state):
        self.__init__(state[0], targets=state[1], compact=state[2] if len(state) > 2 else True,
                      backend=state[3] if len(state) > 3 else "cpu")

    @property
    def backend(self):
        """Resolved execution backend (cpu or cuda)."""
        return self._native.backend

    def _inputs(self, sample, inplace=False):
        arrays, converters = {}, {}
        for key, value in sample.items():
            if key in self.targets:
                if value is not None:
                    arrays[key], converters[key] = _as_array(value, inplace)
            elif isinstance(value, np.ndarray) or type(value).__module__.split(".")[0] == "torch":
                raise ValueError(f"array field {key!r} needs a target declaration")
        return arrays, converters

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
            out["_aug_perf"] = timings
        return out

    def __call__(self, sample, *, seed, profile=False, inplace=False):
        arrays, converters = self._inputs(sample, inplace)
        result = self._native.run(arrays, seed, False, profile, inplace)
        return self._outputs(sample, converters, result, profile)

    def sample(self, sample, *, seed):
        arrays, _ = self._inputs(sample)
        return Plan(self, self._native.run(arrays, seed, True, False)[3])

    def apply(self, sample, plan):
        if not isinstance(plan, Plan) or plan._pipeline is not self:
            raise ValueError("plan belongs to another pipeline; use plan.apply(sample)")
        arrays, converters = self._inputs(sample)
        return self._outputs(sample, converters, self._native.replay(arrays, plan._native), False)


__all__ = [*_constructors, "Pipeline", "Plan", "seed", "translate_crop_box"]
