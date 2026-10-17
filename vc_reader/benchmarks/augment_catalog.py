"""Explicit full-catalog workloads; primary configurations reuse bench_augment."""
from dataclasses import dataclass
import numpy as np
from vc_reader import augment as A
from bench_augment import cases


# Preprocessing, derived targets, reshaping and pipeline control. Normalize is
# deliberately retained in the default gallery alongside the augmentations.
UTILITY_TRANSFORMS = frozenset({
    'ReplaceLabel', 'Regions', 'OneHot', 'MaskImage', 'DeepSupervision',
    'ToPseudo2D', 'FromPseudo2D', 'MedialSurface', 'DistanceTransform',
    'StructureTensor', 'OneOf',
})


@dataclass
class Workload:
    name: str
    transforms: list
    sample: str = 'image'
    output: str = 'image'
    channel: int = 0
    note: str = ''


def catalog(size, reference, *, include_utilities=False):
    result = [Workload(c.name, c.native, c.inputs) for c in cases((size,) * 3)]
    for case in result:
        if case.name=='Elastic':
            case.transforms=[A.Elastic(sigma=32,magnitude=8,border=True)]
            case.note='Strong cubic B-spline example: 64-voxel knot spacing, requested RMS displacement 8 voxels, with border clamping.'
    def add(name, transform, sample='image', output='image', channel=0, note=''):
        result.append(Workload(name, transform if isinstance(transform, list) else [transform], sample, output, channel, note))
    add('ClipOutliers', A.ClipOutliers(lower=1, upper=99, p_retain_std=1))
    add('CropPad', A.CropPad((max(2,size*7//8),)*3), 'both')
    add('Affine', A.Affine(degrees=5, translate=2, scale=.95), 'both')
    add('SheetCompression', A.SheetCompression(strength=.2, intensity_sigma=1, smoothing=2, axes='z'), 'both')
    add('Morphology', A.Morphology(operation='close',radius=1), 'labels', 'mask')
    add('RemoveComponents', A.RemoveComponents(on='mask',max_fraction=.25,p_fill=0), 'labels','mask')
    add('ReplaceLabel', A.ReplaceLabel(value=1,replacement=2), 'labels','mask')
    add('Regions', A.Regions(labels=[0,1]), 'labels','mask',1)
    add('OneHot', A.OneHot(labels=[0,1]), 'labels','onehot',1)
    add('MaskImage', A.MaskImage(fill=0), 'signed')
    add('DeepSupervision', A.DeepSupervision([1,.5,.25]), 'labels','mask/1',note='Preview shows the half-resolution level; timing includes all three levels.')
    add('ToPseudo2D', A.ToPseudo2D(), 'both',note='Metadata reshape. Preview restores ZYX shape for display only.')
    add('FromPseudo2D', [A.ToPseudo2D(),A.FromPseudo2D()], 'both',note='Requires ToPseudo2D in the same pipeline; timing includes that metadata setup.')
    add('MedialSurface', A.MedialSurface(tube=True,open=False,close=True), 'labels','mask_skel')
    add('Normalize', A.Normalize('zscore'))
    add('OrientedBlur', A.OrientedBlur(sigma_along=2,sigma_across=.5,axis=(1,0,0)))
    add('DistanceTransform', A.DistanceTransform(), 'labels','mask_sdt')
    add('StructureTensor', A.StructureTensor(sigma=1,rho=1.5,field='components'),output='image_st',note='Preview shows the zz component; timing generates all six tensor components.')
    add('OneOf', A.OneOf([A.Gamma(.8),A.GaussianNoise(.05)]),note='Seed determines the selected branch; timing includes selection.')
    add('BiasField', A.BiasField(std=.3,scale=.025))
    add('PlasmaBrightnessContrast', A.PlasmaBrightnessContrast(brightness=.2,contrast=.3,scale=.025,persistence=.5,octaves=5))
    add('PlasmaShadow', A.PlasmaShadow(strength=.5,scale=.025,persistence=.5,octaves=5))
    add('Illumination', A.Illumination(strength=.3,mode='linear',direction=(1,1,1)))
    add('Vignetting', A.Vignetting(strength=.5,radius=.8))
    add('RandomToneCurve', A.RandomToneCurve(scale=.1))
    add('Equalize', A.Equalize(bins=256))
    add('CLAHE', A.CLAHE(clip_limit=2,tile_shape=(16,16,16),bins=256))
    add('HistogramMatching', A.HistogramMatching(reference,blend=1,bins=256))
    add('PixelDistributionAdaptation', A.PixelDistributionAdaptation(reference,blend=1,mode='standard'))
    add('FDA', A.FDA(reference,blend=1,fraction=.1))
    add('MultiplicativeNoise', A.MultiplicativeNoise(low=.8,high=1.2))
    add('ShotNoise', A.ShotNoise(photons=100))
    add('AdditiveNoise', A.AdditiveNoise(scale=.05,distribution='laplace',mean=0))
    add('SaltAndPepper', A.SaltAndPepper(amount=.02,salt_fraction=.5))
    add('MotionBlur', A.MotionBlur(length=5,direction=(1,1,1),samples=9))
    add('Defocus', A.Defocus(radius=2))
    add('ZoomBlur', A.ZoomBlur(strength=.15,samples=9))
    add('UnsharpMask', A.UnsharpMask(strength=.5,sigma=1))
    add('GibbsRinging', A.GibbsRinging(cutoff=.7))
    add('RingingOvershoot', A.RingingOvershoot(cutoff=.7,radius=4))
    add('ForegroundCrop', A.ForegroundCrop((max(2,size*7//8),)*3), 'both')
    add('ChannelMisalignment', A.ChannelMisalignment(degrees=5,translate=2,scale=1,channels=(1,)), 'channels',channel=1,note='Two input channels (second is a copy of the CT); only channel 1 moves. Both channels are included in timing.')
    add('GridDropout', A.GridDropout(unit_size=(16,16,16),ratio=.4))
    add('StripeMasking', A.StripeMasking(unit_size=(16,16,16),ratio=.25,axes='zyx'))
    add('ForegroundDropout', A.ForegroundDropout(size=(16,16,16),count=3), 'both')
    add('ObjectDropout', A.ObjectDropout(drop_probability=.5), 'both')
    add('StripGeometry', A.StripGeometry(degrees=5,smooth_offset=2,stride=16), 'strip',note='2D-only companion: central Z slice, not a cubic-volume timing.')
    add('SurfacePerturb', A.SurfacePerturb(blobs=3,amplitude=1,points_affected=100,p=1), 'surface','surface',note='2D-only companion: synthetic coordinate surface with CT-derived height. Preview is displacement magnitude, not an anatomical surface.')
    return [case for case in result if include_utilities or case.name not in UTILITY_TRANSFORMS]


def inputs(image):
    labels=(image>.5).astype(np.int16)
    signed=labels.copy();signed[signed==0]=-1
    plane=np.ascontiguousarray(image[:,image.shape[1]//2])
    y,x=np.indices(plane.shape[1:],dtype=np.float32)
    surface=np.stack([plane[0]*2,y,x])
    return dict(image=dict(image=image),both=dict(image=image,mask=labels),
                binary=dict(image=labels.astype(np.float32)),labels=dict(mask=labels),
                signed=dict(image=image,mask=signed),channels=dict(image=np.concatenate([image,image])),
                strip=dict(image=plane),surface=dict(surface=surface))


def describe(transform):
    def value(v):
        if isinstance(v,np.ndarray):
            return f'reference_volume(shape={v.shape}, dtype={v.dtype})'
        return repr(v)
    return transform._constructor.__name__+'('+', '.join([value(v) for v in transform._args]+[f'{k}={value(v)}' for k,v in transform._kwargs.items()])+')'
