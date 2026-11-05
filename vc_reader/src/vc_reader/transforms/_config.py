"""Picklable native transform configurations."""
from copy import deepcopy
from functools import wraps

from .. import _transforms


def _unwrap(value):
    if isinstance(value, _Config):
        return value._config
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
        self._config = constructor(*_unwrap(self._args), **_unwrap(self._kwargs))
        self._direct_pipelines = {}  # (bare, backend, compact, targets) -> Pipeline for direct calls

    def __reduce__(self):
        # Names are used only to reconstruct configuration after spawn/pickle.
        return _restore_config, (self._constructor.__name__, self._args, self._kwargs)

    def __call__(self, data, *, targets=None, compact=True, inplace=False):
        """Apply this transform on its own.

        Outside a Pipeline a transform is not random: it always applies (p is
        ignored) and samples everything else from a fixed seed, so the same
        input gives the same output. Use a Pipeline for random augmentation.

        ``data`` is a channel-first (C, Y, X) / (C, Z, Y, X) NumPy array, Torch
        tensor (CPU or CUDA) or DeviceArray, standing for the field the
        transform reads by default; the call returns the field it creates, else
        the transformed input. A sample dict works like ``Pipeline.__call__``
        (``targets`` as for Pipeline) and returns a dict. CUDA inputs run on
        the CUDA backend and stay on device. ``inplace`` as for Pipeline.
        Probabilities inside the transform (p_channel, p_axis, ...) and
        parameter ranges are drawn from the fixed seed too: pass plain values.
        """
        from ._pipeline import _direct
        return _direct(self, data, targets, compact, inplace)

    def batch(self, data, *, targets=None, compact=True):
        """``__call__`` for each sample of a batch: (B, C, Y, X) / (B, C, Z, Y, X).

        Same inputs and outputs as ``__call__`` with a leading batch dimension.
        CUDA tensors and DeviceArrays run as one ``Pipeline.batch`` call and
        return CUDA tensors; host inputs run sample by sample.
        """
        from ._pipeline import _direct_batch
        return _direct_batch(self, data, targets, compact)

    def __repr__(self):
        args = [repr(v) for v in self._args]
        args.extend(f"{k}={v!r}" for k, v in self._kwargs.items())
        return f"{self._constructor.__name__}({', '.join(args)})"


def _restore_config(name, args, kwargs):
    return _Config(getattr(_transforms, name), args, kwargs)


def _factory(constructor):
    @wraps(constructor)
    def configure(*args, **kwargs):
        return _Config(constructor, args, kwargs)
    return configure


_constructors = (
    "Param Uniform Normal Choice Image Label ScalarField ValidMask Points "
    "Vectors VectorField NormalField CoordinateField OneOf "
    "BrightnessMultiply BrightnessAdd Contrast Gamma Invert Normalize GaussianNoise RicianNoise "
    "ClipOutliers GaussianBlur MedianFilter Sharpen BrightnessGradient LocalContrast "
    "LocalGamma LocalSmoothing BlankRectangle SliceIllumination Smear Flip Rotate90 "
    "Transpose CropPad Affine Elastic SheetCompression LowResolution ThickSlice "
    "Morphology RandomBinaryOperator RemoveComponents ConnectedComponents RidgeTrace ReplaceLabel Regions OneHot "
    "MaskImage DeepSupervision ToPseudo2D FromPseudo2D MedialSurface StripGeometry SurfacePerturb OrientedBlur "
    "DistanceTransform StructureTensor Ridges CoherenceEnhancingDiffusion "
    "BiasField PlasmaBrightnessContrast PlasmaShadow Illumination Vignetting RandomToneCurve "
    "Equalize CLAHE HistogramMatching PixelDistributionAdaptation FDA MultiplicativeNoise ShotNoise "
    "AdditiveNoise SaltAndPepper MotionBlur Defocus ZoomBlur UnsharpMask GibbsRinging RingingOvershoot "
    "ForegroundCrop ChannelMisalignment GridDropout StripeMasking ForegroundDropout ObjectDropout"
).split()
globals().update({name: _factory(getattr(_transforms, name)) for name in _constructors})
seed = _transforms.seed


