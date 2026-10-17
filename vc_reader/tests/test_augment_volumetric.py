"""Volumetric semantics, numerical references, replay and CPU/CUDA parity."""
import pickle

import numpy as np
import pytest
from vc_reader import augment as A
from vc_reader.augment.cuda import is_available


@pytest.fixture(params=['cpu', 'cuda'])
def backend(request):
    if request.param == 'cuda' and not is_available():
        pytest.skip('CUDA build/device unavailable')
    return request.param


def image(shape=(5, 7, 9), dtype=np.float32, channels=2):
    return np.random.default_rng(810).uniform(.1, .9, (channels, *shape)).astype(dtype)


def catalog(reference):
    return [
        A.BiasField(), A.PlasmaBrightnessContrast(), A.PlasmaShadow(),
        A.Illumination(mode='linear'), A.Illumination(mode='corner'),
        A.Illumination(mode='gaussian'), A.Vignetting(), A.RandomToneCurve(),
        A.Equalize(), A.CLAHE(tile_shape=(3, 4, 5)),
        A.HistogramMatching(reference), A.PixelDistributionAdaptation(reference),
        A.PixelDistributionAdaptation(reference, mode='minmax'), A.FDA(reference),
        A.MultiplicativeNoise(), A.ShotNoise(), A.AdditiveNoise(distribution='uniform'),
        A.AdditiveNoise(distribution='gaussian'), A.AdditiveNoise(distribution='laplace'),
        A.AdditiveNoise(distribution='beta'), A.SaltAndPepper(), A.MotionBlur(),
        A.Defocus(radius=1.5), A.ZoomBlur(), A.UnsharpMask(), A.GibbsRinging(),
        A.RingingOvershoot(), A.ForegroundCrop((3, 4, 5)), A.ChannelMisalignment(),
        A.GridDropout(unit_size=(3, 4, 5)), A.StripeMasking(unit_size=(3, 4, 5)),
        A.ForegroundDropout(size=(3, 3, 3)), A.ObjectDropout(),
    ]


@pytest.mark.parametrize('index', range(33))
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_catalog_replay_targets_and_parity(backend, index, dtype):
    source = image(dtype=dtype)
    mask = (source[:1] > .6).astype(np.int16)
    sample = dict(image=source, mask=mask)
    transform = catalog(image(dtype=dtype) * .6 + .2)[index]
    pipeline = A.Pipeline([transform], backend=backend)
    actual = pipeline(sample, seed=19)
    expected = A.Pipeline([transform], backend='cpu')(sample, seed=19)
    np.testing.assert_allclose(actual['image'], expected['image'], atol=2e-6 if dtype == np.float32 else 3e-12, rtol=3e-6 if dtype == np.float32 else 3e-12)
    np.testing.assert_array_equal(actual['mask'], expected['mask'])
    assert actual['image'].dtype == dtype
    assert np.isfinite(actual['image']).all()
    np.testing.assert_array_equal(source, image(dtype=dtype))
    for result in [pipeline(sample, seed=19), pipeline.sample(sample, seed=19).apply(sample),
                   pickle.loads(pickle.dumps(pipeline))(sample, seed=19),
                   pickle.loads(pickle.dumps(pipeline.sample(sample, seed=19))).apply(sample)]:
        for key in sample:
            np.testing.assert_array_equal(result[key], actual[key])
    if index not in (27, 32):
        np.testing.assert_array_equal(actual['mask'], mask)


@pytest.mark.parametrize('shape', [(1, 1, 1), (1, 3, 5), (3, 1, 5), (3, 5, 1)])
def test_singletons_and_constant_channels(backend, shape):
    source = np.full((1, *shape), .4, np.float64)
    sample = dict(image=source, mask=np.zeros_like(source, dtype=np.uint8))
    for transform in catalog(source):
        result = A.Pipeline([transform], backend=backend)(sample, seed=14)
        assert np.isfinite(result['image']).all(), transform
    for transform in [A.Equalize(), A.CLAHE(), A.RandomToneCurve(), A.MotionBlur(), A.Defocus(), A.ZoomBlur(), A.UnsharpMask(), A.GibbsRinging(), A.RingingOvershoot()]:
        out = A.Pipeline([transform], backend=backend)(dict(image=source), seed=0)['image']
        np.testing.assert_allclose(out, source, atol=2e-14)


def run(transform, source, backend):
    return A.Pipeline([transform], backend=backend)(dict(image=source), seed=17)['image']


@pytest.mark.parametrize('radius',[.1,1.5,2,5])
@pytest.mark.parametrize('shape',[(3,5,37),(1,3,5)])
def test_defocus_interior_runs_and_wide_boundary_kernel(backend,radius,shape):
    source=image(shape,np.float64,1)
    r=int(np.ceil(radius));padded=np.pad(source[0],r,mode='edge')
    expected=np.zeros(shape,np.float64);count=0
    for z in range(-r,r+1):
        for y in range(-r,r+1):
            for x in range(-r,r+1):
                if z*z+y*y+x*x<=radius*radius:
                    expected+=padded[r+z:r+z+shape[0],r+y:r+y+shape[1],r+x:r+x+shape[2]]
                    count+=1
    expected/=count
    np.testing.assert_allclose(run(A.Defocus(radius=radius),source,backend)[0],expected,rtol=2e-14,atol=2e-14)


@pytest.mark.parametrize('radius',[1,4,12])
def test_ringing_overshoot_keeps_boundary_and_tap_order(backend,radius):
    import math
    source=image((1,5,37),np.float64,1);expected=source[0].copy();cutoff=.7
    weights=[]
    for j in range(-radius,radius+1):
        w=math.sin(math.pi*cutoff*j)/(math.pi*j) if j else cutoff
        weights.append(w*(.5+.5*math.cos(math.pi*j/(radius+1))))
    for axis in range(3):
        result=np.zeros_like(expected)
        for j,w in zip(range(-radius,radius+1),weights):
            indices=np.clip(np.arange(expected.shape[axis])+j,0,expected.shape[axis]-1)
            result+=w*np.take(expected,indices,axis=axis)
        expected=result/sum(weights)
    np.testing.assert_allclose(run(A.RingingOvershoot(cutoff=cutoff,radius=radius),source,backend)[0],expected,rtol=2e-14,atol=2e-14)


def test_closed_form_intensity_operations(backend):
    source = image(dtype=np.float64, channels=1)
    z = np.linspace(-1, 1, source.shape[1])[:, None, None]
    actual = run(A.Illumination(strength=.3, direction=(1, 0, 0)), source, backend)
    np.testing.assert_allclose(actual, source * (1 + .3 * z), atol=1e-14)
    reference = (source * 3 + 7).copy()
    for mode in ('standard', 'minmax'):
        actual = run(A.PixelDistributionAdaptation(reference, blend=1, mode=mode), source, backend)
        np.testing.assert_allclose(actual, reference, atol=3e-14)
    for transform in [A.BiasField(std=0), A.PlasmaBrightnessContrast(brightness=0, contrast=0),
                      A.PlasmaShadow(strength=0), A.Illumination(strength=0), A.Vignetting(strength=0),
                      A.RandomToneCurve(scale=0), A.MultiplicativeNoise(low=1, high=1),
                      A.AdditiveNoise(scale=0, mean=0), A.SaltAndPepper(amount=0),
                      A.MotionBlur(length=0), A.Defocus(radius=0), A.ZoomBlur(strength=0),
                      A.UnsharpMask(strength=0), A.GibbsRinging(cutoff=1), A.RingingOvershoot(cutoff=1),
                      A.HistogramMatching(reference, blend=0), A.FDA(reference, blend=0)]:
        np.testing.assert_allclose(run(transform, source, backend), source, atol=2e-15, err_msg=repr(transform))


def test_histogram_known_values(backend):
    source = np.array([0., 0., 1., 3.]).reshape(1, 1, 1, 4)
    reference = np.array([2., 2., 6., 10.]).reshape(source.shape)
    np.testing.assert_allclose(run(A.Equalize(bins=4), source, backend).ravel(), [0, 0, 1.5, 3])
    np.testing.assert_allclose(run(A.HistogramMatching(reference, bins=9, blend=1), source, backend), reference)
    # Matching a constant reference must be exact regardless of histogram bins.
    np.testing.assert_allclose(run(A.HistogramMatching(np.full_like(source, 7), blend=1), source, backend), 7)


@pytest.mark.parametrize('shape', [(5, 7, 9), (4, 8, 6), (1, 3, 7)])
def test_frequency_filters_against_numpy_fft(backend, shape):
    source = image(shape, np.float64, 1)
    reference = np.random.default_rng(45).uniform(0, 2, source.shape)
    spectrum = np.fft.fftn(source[0])
    def lowpass(fraction):
        return np.logical_and.reduce(np.broadcast_arrays(*[
            (np.abs(np.fft.fftfreq(size)) <= fraction / 2 + 1e-15).reshape(tuple(size if j == axis else 1 for j in range(3)))
            for axis, size in enumerate(shape)]))
    expected = np.fft.ifftn(spectrum * lowpass(.6)).real
    np.testing.assert_allclose(run(A.GibbsRinging(cutoff=.6), source, backend)[0], expected, atol=2e-13)
    selected = lowpass(.3)
    rf = np.fft.fftn(reference[0])
    expected_spectrum = spectrum.copy()
    amplitude = .4 * np.abs(spectrum) + .6 * np.abs(rf)
    expected_spectrum[selected] = amplitude[selected] * np.exp(1j * np.angle(spectrum[selected]))
    expected = np.fft.ifftn(expected_spectrum).real
    np.testing.assert_allclose(run(A.FDA(reference, blend=.6, fraction=.3), source, backend)[0], expected, atol=3e-13)


def test_blurs_against_scipy(backend):
    ndimage = pytest.importorskip('scipy.ndimage')
    source = image(dtype=np.float64, channels=1)
    blurred = ndimage.gaussian_filter(source[0], .8, mode='nearest', truncate=4)
    np.testing.assert_allclose(run(A.UnsharpMask(strength=.4, sigma=.8), source, backend)[0], source[0] + .4 * (source[0] - blurred), atol=2e-14)
    footprint = np.zeros((3, 3, 3))
    footprint[1, 1, :] = footprint[1, :, 1] = footprint[:, 1, 1] = 1
    expected = ndimage.convolve(source[0], footprint / footprint.sum(), mode='nearest')
    np.testing.assert_allclose(run(A.Defocus(radius=1), source, backend)[0], expected, atol=2e-14)
    expected = ndimage.convolve1d(source[0], np.ones(3) / 3, axis=0, mode='nearest')
    np.testing.assert_allclose(run(A.MotionBlur(length=2, direction=(1, 0, 0), samples=3), source, backend)[0], expected, atol=2e-14)


def test_noise_statistics(backend):
    source = np.full((1, 32, 32, 32), .5, np.float64)
    shot = run(A.ShotNoise(photons=80), source, backend)
    assert abs(shot.mean() - .5) < .002
    assert abs(shot.var() - .5 / 80) < .0003
    for transform, variance in [(A.AdditiveNoise(distribution='uniform', scale=.1), .1**2 / 3),
                                 (A.AdditiveNoise(distribution='laplace', scale=.1), 2 * .1**2),
                                 (A.AdditiveNoise(distribution='beta', scale=1, alpha=2, beta=2), .05),
                                 (A.MultiplicativeNoise(low=.8, high=1.2), .5**2 * .4**2 / 12)]:
        out = run(transform, source, backend)
        assert abs(out.mean() - .5) < .005
        assert abs(out.var() - variance) < variance * .07


def test_fields_are_volumetric(backend):
    source = np.full((1, 12, 13, 14), .5, np.float64)
    for transform in [A.BiasField(std=.4, scale=.2), A.PlasmaShadow(scale=.2),
                      A.PlasmaBrightnessContrast(brightness=0, contrast=.3, scale=.2), A.Vignetting()]:
        # Plasma contrast requires image variation; add a ramp for that case.
        original = image((12, 13, 14), np.float64, 1) if 'PlasmaBrightnessContrast' in repr(transform) else source
        out = run(transform, original, backend)
        delta = out - original
        for axis in (1, 2, 3):
            assert np.max(np.abs(np.diff(delta, axis=axis))) > 1e-6, transform


def test_crop_retains_foreground_and_geometry(backend):
    source = np.arange(9 * 11 * 13, dtype=np.float64).reshape(1, 9, 11, 13)
    mask = np.zeros_like(source, dtype=np.uint64)
    mask[0, 8, 10, 12] = 2**63 + 9
    sample = dict(image=source, mask=mask, points=np.array([[8., 10., 12.]]))
    targets = dict(image=A.Image(), mask=A.Label(), points=A.Points())
    pipeline = A.Pipeline([A.ForegroundCrop((4, 5, 6))], targets=targets, backend=backend)
    for seed in range(5):
        out = pipeline(sample, seed=seed)
        assert np.count_nonzero(out['mask']) == 1
        assert out['mask'].max() == mask.max()
        point = out['points'][0].astype(int)
        assert out['mask'][(0, *point)] == mask.max()
        assert out['image'][(0, *point)] == source[0, 8, 10, 12]


def test_dropout_target_semantics(backend):
    source = np.ones((2, 7, 9, 11), np.float32)
    mask = np.zeros((1, 7, 9, 11), np.uint64)
    mask[:, 1:3, 1:4, 1:4] = 2**63 + 1
    mask[:, 4:6, 5:8, 6:10] = 2**63 + 3
    sample = dict(image=source, mask=mask)
    pipeline = A.Pipeline([A.ObjectDropout(drop_probability=1, fill=-2, channels=[0])], backend=backend)
    result = pipeline(sample, seed=0)
    assert np.all(result['mask'] == 0)
    np.testing.assert_array_equal(result['image'][1], source[1])
    np.testing.assert_array_equal(result['image'][0] == -2, mask[0] != 0)
    transform = A.ForegroundDropout(size=(5, 5, 5), count=3, fill=-1)
    result = A.Pipeline([transform], backend=backend)(sample, seed=13)
    assert np.any(result['image'] == -1)
    assert np.all(result['image'][:, mask[0] == 0] == 1)
    np.testing.assert_array_equal(result['mask'], mask)
    # Sampled erasure is replayed even when the supplied mask contents change.
    plan = A.Pipeline([transform], backend=backend).sample(sample, seed=13)
    np.testing.assert_array_equal(plan.apply(dict(sample, mask=np.zeros_like(mask)))['image'], result['image'])


def test_misalignment_moves_only_selected_channels(backend):
    source = image(channels=2)
    mask = (source[:1] > .5).astype(np.int8)
    transform = A.ChannelMisalignment(degrees=0, scale=1, translate=1, channels=[1], border=True)
    actual = A.Pipeline([transform], backend=backend)(dict(image=source, mask=mask), seed=0)
    np.testing.assert_array_equal(actual['image'][0], source[0])
    np.testing.assert_array_equal(actual['mask'], mask)
    assert not np.array_equal(actual['image'][1], source[1])


def test_invalid_inputs_and_validity(backend):
    source = image(channels=1)
    with pytest.raises(ValueError):
        run(A.ShotNoise(), -source, backend)
    for transform in [A.BiasField(std=-1), A.AdditiveNoise(distribution='beta', alpha=0),
                      A.GibbsRinging(cutoff=0), A.CLAHE(tile_shape=(0, 2, 2))]:
        with pytest.raises(ValueError):
            run(transform, source, backend)
    with pytest.raises(ValueError):
        run(A.BiasField(), source[:, 0], backend)
    valid = np.ones_like(source, dtype=np.uint8)
    valid[:, 0] = 0
    for transform in [A.BiasField(), A.ShotNoise(), A.UnsharpMask(), A.GridDropout(fill=2)]:
        out = A.Pipeline([transform], backend=backend)(dict(image=source, padding_mask=valid), seed=2)
        assert np.all(out['image'][:, 0] == 0)


def test_clahe_axis_permutation(backend):
    source = image((6, 6, 6), np.float64, 1)
    result = run(A.CLAHE(tile_shape=(3, 3, 3), bins=16), source, backend)
    permuted = run(A.CLAHE(tile_shape=(3, 3, 3), bins=16), source.transpose(0, 3, 1, 2).copy(), backend)
    np.testing.assert_allclose(permuted, result.transpose(0, 3, 1, 2), atol=2e-15)


@pytest.mark.parametrize('index', range(33))
def test_resident_transforms(index):
    from vc_reader import cuda
    if not is_available() or not cuda.is_available():
        pytest.skip('CUDA reader and augmentation builds required')
    source = image()
    sample = dict(image=source, mask=(source[:1] > .6).astype(np.int16))
    transform = catalog(source * .7 + .1)[index]
    pipeline = A.Pipeline([transform], backend='cuda')
    expected = pipeline(sample, seed=23)
    resident = {key: cuda.asarray(value) for key, value in sample.items()}
    result = pipeline(resident, seed=23)
    replayed = pipeline.sample(resident, seed=23).apply(resident)
    for key in sample:
        np.testing.assert_array_equal(result[key].numpy(), expected[key])
        np.testing.assert_array_equal(replayed[key].numpy(), expected[key])
        np.testing.assert_array_equal(resident[key].numpy(), sample[key])


def test_resident_mixed_chain():
    from vc_reader import cuda
    if not is_available() or not cuda.is_available():
        pytest.skip('CUDA reader and augmentation builds required')
    source = image((7, 9, 11))
    sample = dict(image=source, mask=(source[:1] > .5).astype(np.int16))
    transforms = [A.ForegroundCrop((5, 7, 9)), A.Flip('x'),
                  A.BiasField(std=.1), A.Normalize('minmax'),
                  A.ShotNoise(photons=100), A.CLAHE(tile_shape=(3, 4, 5)),
                  A.ChannelMisalignment(degrees=2, translate=1, scale=1),
                  A.GibbsRinging(cutoff=.7), A.ObjectDropout(drop_probability=.3),
                  A.UnsharpMask(strength=.2, sigma=.5), A.Normalize('zscore')]
    pipeline = A.Pipeline(transforms, backend='cuda')
    host = pipeline(sample, seed=123)
    device = pipeline({k: cuda.asarray(v) for k, v in sample.items()}, seed=123)
    for key in sample:
        np.testing.assert_array_equal(device[key].numpy(), host[key])
    cpu = A.Pipeline(transforms, backend='cpu')(sample, seed=123)
    np.testing.assert_allclose(cpu['image'], host['image'], atol=3e-6, rtol=3e-6)


def test_output_dtype_overflow_is_rejected(backend):
    source = np.full((1, 2, 3, 4), np.finfo(np.float32).max, np.float32)
    with pytest.raises(ValueError):
        run(A.MultiplicativeNoise(low=2, high=2), source, backend)


def test_reference_snapshot_and_shared_noise(backend):
    source = image(channels=1)
    reference = source * 2 + 1
    transform = A.PixelDistributionAdaptation(reference, blend=1)
    expected = reference.copy()
    reference[:] = -99
    np.testing.assert_allclose(run(transform, source, backend), expected, atol=5e-7)
    source = np.repeat(source, 2, axis=0)
    for transform in [A.MultiplicativeNoise(shared_noise=True), A.AdditiveNoise(shared_noise=True),
                      A.ShotNoise(photons=100, shared_noise=True), A.BiasField(std=.2, shared=True)]:
        result = run(transform, source, backend)
        np.testing.assert_array_equal(result[0], result[1])


@pytest.mark.parametrize('transform', [A.Defocus(radius=2), A.MotionBlur(length=3), A.ZoomBlur(strength=.2),
                                     A.GibbsRinging(cutoff=.7), A.UnsharpMask(strength=.5)])
def test_float32_intermediate_overflow_protection(backend, transform):
    # The result fits float32, while unnormalized sums / FFT coefficients do not.
    source=np.full((1,5,7,9),1e37,np.float32)
    actual=run(transform,source,backend)
    np.testing.assert_allclose(actual,source,rtol=3e-6)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize('transform', [A.Illumination(mode='gaussian',radius=1e-30),
                                     A.MultiplicativeNoise(low=1e50,high=1e50)])
def test_extreme_parameters_preserve_representable_result(backend, transform):
    source=np.full((1,5,7,9),1e-20,np.float32)
    actual=run(transform,source,backend)
    expected=run(transform,source,'cpu')
    np.testing.assert_allclose(actual,expected,rtol=3e-6,atol=0)
