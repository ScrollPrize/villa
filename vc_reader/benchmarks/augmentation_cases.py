"""Explicit augmentation workloads shared by benchmarks and documentation previews."""
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Case:
    name: str
    kwargs: dict = field(default_factory=dict)
    planar: dict = field(default_factory=dict)
    reference: bool = False
    sample: str = 'image'
    output: str = 'image'
    channel: int = 0
    note: str = ''

    def parameters(self, spatial_dims=3):
        if spatial_dims not in (2, 3):
            raise ValueError('spatial_dims must be 2 or 3')
        return {**self.kwargs, **(self.planar if spatial_dims == 2 else {})}

    def construct(self, augment, reference, spatial_dims=3):
        kwargs = self.parameters(spatial_dims)
        if self.reference:
            kwargs['reference'] = reference
        return getattr(augment, self.name)(**kwargs)

    def recipe(self, spatial_dims=3):
        arguments = [f'{key}={value!r}' for key, value in self.parameters(spatial_dims).items()]
        if self.reference:
            arguments.insert(0, 'reference=reference')
        return f'A.{self.name}(' + ', '.join(arguments) + ')'


# Planar overrides are explicit: YX sizes/directions, Y for Z-only operations,
# and a Y-aligned oriented blur. Strip/surface cases always use a 2D companion.
CASES = (
    Case('AdditiveNoise', {'scale': 0.05, 'distribution': 'laplace', 'mean': 0}),
    Case('Affine', {'degrees': 5, 'translate': 2, 'scale': 0.95},
         sample='both'),
    Case('BiasField', {'std': 0.3, 'scale': 0.025}),
    Case('BlankRectangle', {'size': 16, 'value': 0, 'count': 3}),
    Case('BrightnessAdd', {'shift': 0.1}),
    Case('BrightnessGradient', {'scale': 12, 'loc': 0.5, 'strength': 0.3}),
    Case('BrightnessMultiply', {'factor': 1.1}),
    Case('ChannelMisalignment', {'degrees': 5, 'translate': 2, 'scale': 1, 'channels': (1,)},
         sample='channels', channel=1, note='Second of two identical CT channels.'),
    Case('CLAHE', {'clip_limit': 2, 'tile_shape': (16, 16, 16), 'bins': 256},
         planar={'tile_shape': (16, 16)}),
    Case('ClipOutliers', {'lower': 1, 'upper': 99, 'p_retain_std': 1}),
    Case('Contrast', {'factor': 1.2}),
    Case('CropPad', {'shape': (112, 112, 112)},
         planar={'shape': (112, 112)}, sample='both'),
    Case('Defocus', {'radius': 2}),
    Case('Elastic', {'sigma': 32, 'magnitude': 8, 'border': True},
         sample='both'),
    Case('Equalize', {'bins': 256}),
    Case('FDA', {'blend': 1, 'fraction': 0.1},
         reference=True),
    Case('Flip', {'axes': 'x', 'p_axis': 1},
         sample='both'),
    Case('ForegroundCrop', {'shape': (112, 112, 112)},
         planar={'shape': (112, 112)}, sample='both'),
    Case('ForegroundDropout', {'size': (16, 16, 16), 'count': 3},
         planar={'size': (16, 16)}, sample='both'),
    Case('Gamma', {'gamma': 0.8}),
    Case('GaussianBlur', {'sigma': 1}),
    Case('GaussianNoise', {'std': 0.05}),
    Case('GibbsRinging', {'cutoff': 0.7}),
    Case('GridDropout', {'unit_size': (16, 16, 16), 'ratio': 0.4},
         planar={'unit_size': (16, 16)}),
    Case('HistogramMatching', {'blend': 1, 'bins': 256},
         reference=True),
    Case('Illumination', {'strength': 0.3, 'mode': 'linear', 'direction': (1, 1, 1)},
         planar={'direction': (1, 1)}),
    Case('Invert', {}),
    Case('LocalContrast', {'scale': 12, 'loc': 0.5, 'factor': 1.2}),
    Case('LocalGamma', {'scale': 12, 'loc': 0.5, 'gamma': 0.8}),
    Case('LocalSmoothing', {'scale': 12, 'loc': 0.5, 'strength': 0.7, 'sigma': 1}),
    Case('LowResolution', {'scale': 0.5}),
    Case('MedianFilter', {'size': 3}),
    Case('Morphology', {'operation': 'close', 'radius': 1},
         sample='labels', output='mask', note='Foreground mask derived from CT > 0.5; not manual labels.'),
    Case('MotionBlur', {'length': 5, 'direction': (1, 1, 1), 'samples': 9},
         planar={'direction': (1, 1)}),
    Case('MultiplicativeNoise', {'low': 0.8, 'high': 1.2}),
    Case('Normalize', {'mode': 'zscore'}),
    Case('ObjectDropout', {'drop_probability': 0.5},
         sample='both'),
    Case('OrientedBlur', {'sigma_along': 2, 'sigma_across': 0.5, 'axis': (1, 0, 0)},
         planar={'axis': (1, 0)}),
    Case('PixelDistributionAdaptation', {'blend': 1, 'mode': 'standard'},
         reference=True),
    Case('PlasmaBrightnessContrast', {'brightness': 0.2, 'contrast': 0.3, 'scale': 0.025, 'persistence': 0.5, 'octaves': 5}),
    Case('PlasmaShadow', {'strength': 0.5, 'scale': 0.025, 'persistence': 0.5, 'octaves': 5}),
    Case('RandomBinaryOperator', {'channels': (0,), 'radius': 1, 'operations': ('dilate',)},
         sample='binary', note='Binary input derived from CT > 0.5; not manual labels.'),
    Case('RandomToneCurve', {'scale': 0.1}),
    Case('RemoveComponents', {'on': 'mask', 'max_fraction': 0.25, 'p_fill': 0},
         sample='labels', output='mask', note='Foreground mask derived from CT > 0.5; not manual labels.'),
    Case('RicianNoise', {'std': 0.05}),
    Case('RingingOvershoot', {'cutoff': 0.7, 'radius': 4}),
    Case('Rotate90', {'planes': ('yx',), 'turns': (1,)},
         sample='both'),
    Case('SaltAndPepper', {'amount': 0.02, 'salt_fraction': 0.5}),
    Case('Sharpen', {'strength': 0.25, 'p_clamp': 0}),
    Case('SheetCompression', {'strength': 0.2, 'intensity_sigma': 1, 'smoothing': 2, 'axes': 'z'},
         planar={'axes': 'y'}, sample='both'),
    Case('ShotNoise', {'photons': 100}),
    Case('SliceIllumination', {'defects': 3, 'width': 4, 'reduction': 0.5, 'base_p': 0.1, 'base_reduction': 0.5},
         planar={'axis': 'y'}),
    Case('Smear', {'shift': (3, 0), 'alpha': 0.25},
         planar={'shift': (3,), 'axis': 'y'}),
    Case('StripeMasking', {'unit_size': (16, 16, 16), 'ratio': 0.25, 'axes': 'zyx'},
         planar={'unit_size': (16, 16), 'axes': 'yx'}),
    Case('ThickSlice', {'scale': 0.5, 'axes': 'z'},
         planar={'axes': 'y'}, sample='both'),
    Case('Transpose', {'order': 'zxy'},
         planar={'order': 'xy'}, sample='both'),
    Case('UnsharpMask', {'strength': 0.5, 'sigma': 1}),
    Case('Vignetting', {'strength': 0.5, 'radius': 0.8}),
    Case('ZoomBlur', {'strength': 0.15, 'samples': 9}),
    Case('StripGeometry', {'degrees': 5, 'smooth_offset': 2, 'stride': 16},
         sample='strip', note='2D companion: central Z plane from the CT crop.'),
    Case('SurfacePerturb', {'blobs': 3, 'amplitude': 1, 'points_affected': 100, 'p': 1},
         sample='surface', output='surface', note='Synthetic coordinate surface with CT-derived height; not an anatomical surface.'),
)


def input_sample(image, kind):
    """Build only the requested fixture, outside measured augmentation calls."""
    if kind == 'image':
        return {'image': image}
    if kind in ('both', 'binary', 'labels'):
        labels = (image > .5).astype(np.int16)
        if kind == 'both':
            return {'image': image, 'mask': labels}
        if kind == 'binary':
            return {'image': labels.astype(np.float32)}
        return {'mask': labels}
    if kind == 'channels':
        return {'image': np.concatenate([image, image])}
    if kind in ('strip', 'surface'):
        plane = image if image.ndim == 3 else np.ascontiguousarray(image[:, image.shape[1] // 2])
        if kind == 'strip':
            return {'image': plane}
        y, x = np.indices(plane.shape[1:], dtype=np.float32)
        return {'surface': np.stack([plane[0] * 2, y, x])}
    raise ValueError(f'unknown input kind: {kind}')
