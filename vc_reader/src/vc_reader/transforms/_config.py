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

    def __reduce__(self):
        # Names are used only to reconstruct configuration after spawn/pickle.
        return _restore_config, (self._constructor.__name__, self._args, self._kwargs)

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
    "Param Uniform Normal Choice BalancedContrast Image Label ScalarField ValidMask Points "
    "Vectors VectorField NormalField CoordinateField OneOf "
    "BrightnessMultiply BrightnessAdd Contrast Gamma Invert Normalize GaussianNoise RicianNoise "
    "ClipOutliers GaussianBlur MedianFilter Sharpen BrightnessGradient LocalContrast "
    "LocalGamma LocalSmoothing BlankRectangle SliceIllumination Smear Flip Rotate90 "
    "Transpose CropPad Affine Elastic SheetCompression LowResolution ThickSlice "
    "Morphology RandomBinaryOperator RemoveComponents ReplaceLabel Regions OneHot "
    "MaskImage DeepSupervision ToPseudo2D FromPseudo2D MedialSurface StripGeometry SurfacePerturb OrientedBlur "
    "DistanceTransform StructureTensor "
    "BiasField PlasmaBrightnessContrast PlasmaShadow Illumination Vignetting RandomToneCurve "
    "Equalize CLAHE HistogramMatching PixelDistributionAdaptation FDA MultiplicativeNoise ShotNoise "
    "AdditiveNoise SaltAndPepper MotionBlur Defocus ZoomBlur UnsharpMask GibbsRinging RingingOvershoot "
    "ForegroundCrop ChannelMisalignment GridDropout StripeMasking ForegroundDropout ObjectDropout"
).split()
globals().update({name: _factory(getattr(_transforms, name)) for name in _constructors})
seed = _transforms.seed


