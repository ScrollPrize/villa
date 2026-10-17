"""Complete CPU/CUDA catalog: routing, labels, sampled fields and pyramids."""
import pickle

import numpy as np
import pytest
from vc_reader import augment as A
from vc_reader.augment.cuda import is_available

pytestmark = pytest.mark.skipif(not is_available(), reason="CUDA build/device unavailable")


def assert_same(actual, expected, *, exact=False):
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key], exact=exact)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_same(a, b, exact=exact)
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        if exact or expected.dtype.kind != 'f':
            np.testing.assert_array_equal(actual, expected)
        else:
            tol = 4e-6 if expected.dtype == np.float32 else 5e-12
            np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol)
    else:
        assert actual == expected


def host(value):
    if isinstance(value, list):
        return [host(v) for v in value]
    return value.numpy() if hasattr(value, 'numpy') else value


def verify(ops, sample, targets=None, seed=12, resident=True):
    cpu = A.Pipeline(ops, targets=targets, compact=False)
    gpu = A.Pipeline(ops, targets=targets, backend='cuda')
    expected = cpu(sample, seed=seed)
    actual = gpu(sample, seed=seed)
    assert_same(actual, expected)
    for replay in (gpu.sample(sample, seed=seed), pickle.loads(pickle.dumps(gpu.sample(sample, seed=seed)))):
        assert_same(replay.apply(sample), actual, exact=True)
    assert_same(pickle.loads(pickle.dumps(gpu))(sample, seed=seed), actual, exact=True)
    if resident:
        from vc_reader import cuda
        if cuda.is_available():
            device = {k: cuda.asarray(v) if isinstance(v, np.ndarray) else v for k, v in sample.items()}
            result = gpu(device, seed=seed)
            assert_same({k: host(v) for k, v in result.items()}, actual, exact=True)
    return actual


def sample(shape, dtype=np.float32):
    rng = np.random.default_rng(94)
    image = rng.normal(size=(2, *shape)).astype(dtype)
    mask = rng.integers(-1, 4, (2, *shape), dtype=np.int16)
    return dict(image=image, mask=mask)


CASES = [
    A.ClipOutliers(lower=7, upper=82), A.ClipOutliers(p_retain_std=0),
    A.Affine(degrees=5, translate=.4, scale=.9),
    A.SheetCompression(axes='y', intensity_sigma=.6, smoothing=.8),
    A.SheetCompression(axes='x', threshold=.4),
    A.Morphology(operation='dilate'), A.Morphology(operation='erode', neutral_border=True),
    A.Morphology(operation='open'), A.Morphology(operation='close', neutral_border=True),
    A.RemoveComponents(channels=(0, 1), max_fraction=.9, p_fill=1),
    A.ReplaceLabel(value=-1, replacement=4),
    A.Regions(labels=[-1, [1, 2]], output='regions'),
    A.OneHot(labels=[-1, 0, 1, 2, 3], remove_source=True),
    A.OneHot(labels=[1, 2], append_to='image', remove_source=True),
    A.MaskImage(fill=.25),
    A.DeepSupervision(scales=[1, .6, .25]),
    A.DeepSupervision(scales=[.5], output='pyramid'),
]


@pytest.mark.parametrize('shape', [(5, 7), (3, 5, 7)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('op', CASES, ids=repr)
def test_remaining_catalog(op, shape, dtype):
    verify([op], sample(shape, dtype))


@pytest.mark.parametrize('shape', [(5, 7), (3, 5, 7)])
def test_crop_pad_and_empty_components(shape):
    s = sample(shape)
    verify([A.CropPad(tuple(v + 2 for v in shape))], s)
    verify([A.CropPad(tuple(max(1, v - 2) for v in shape))], s)
    for value in (0., 1.):
        verify([A.SheetCompression(), A.RemoveComponents(max_fraction=.5)], {'image': np.full((1, *shape), value, np.float32)})


@pytest.mark.parametrize('full', [True, False])
@pytest.mark.parametrize('seed', range(8))
def test_component_selection_and_fill(full, seed):
    s = sample((4, 5, 7))
    s['image'] = (s['image'] > .9).astype(np.float64)
    verify([A.RemoveComponents(max_fraction=.2, p_fill=1, full_connectivity=full)], s, seed=seed)


@pytest.mark.parametrize('dtype', [np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64])
def test_label_dtypes(dtype):
    s = sample((3, 5, 7))
    s['mask'] = (s['mask'] > 1).astype(dtype)
    for op in [A.Morphology(), A.ReplaceLabel(value=0, replacement=1), A.Regions(labels=[0, 1]), A.OneHot(labels=[0, 1]), A.DeepSupervision([.6])]:
        verify([op], s)


@pytest.mark.parametrize('op', [A.ReplaceLabel(), A.Regions(labels=[1]), A.OneHot(labels=[1]), A.Morphology(), A.RemoveComponents(on='mask')])
def test_large_label_ids_rejected(op):
    source = {'mask': np.full((1, 3, 4, 5), 2**63 + 1, np.uint64)}
    for backend in ('cpu', 'cuda'):
        with pytest.raises(ValueError, match='integer IDs'):
            A.Pipeline([op], backend=backend)(source, seed=1)


@pytest.mark.parametrize('points', [0, 1, 10, 200])
@pytest.mark.parametrize('degenerate', [False, True])
def test_strip_and_surface(points, degenerate):
    yy, xx = np.indices((7, 9), dtype=np.float64)
    surface = np.stack([np.zeros_like(xx), yy, xx])
    if degenerate:
        surface[:] = 0
    source = dict(image=(xx+yy)[None], surface=surface)
    targets = dict(image=A.Image(), surface=A.CoordinateField())
    verify([A.StripGeometry(smooth_offset=.3, stride=3, degrees=5, p_flip_x=.5),
            A.SurfacePerturb(points_affected=points, amplitude=.25, p=1)], source, targets)


def test_pseudo2d_chain_and_oneof():
    verify([A.ToPseudo2D(), A.OneOf([A.Affine(degrees=4), A.ClipOutliers()]), A.FromPseudo2D()], sample((3, 5, 7)))
    verify([A.Morphology(footprint='repeated_cross', radius=2)], sample((3, 5, 7)))


def test_deep_supervision_must_be_last():
    with pytest.raises(ValueError, match='last transform'):
        A.Pipeline([A.DeepSupervision([.5]), A.ReplaceLabel()], backend='cuda')(sample((3, 5, 7)), seed=1)


def test_replacement_range_only_checked_when_used():
    source = dict(mask=np.zeros((1, 3, 4, 5), np.uint8))
    verify([A.ReplaceLabel(value=1, replacement=-1)], source)
    for backend in ('cpu', 'cuda'):
        with pytest.raises(ValueError, match='representable'):
            A.Pipeline([A.ReplaceLabel(value=0, replacement=-1)], backend=backend)(source, seed=1)


def test_native_batch_pyramid_stays_on_device():
    from vc_reader import cuda
    if not cuda.is_available():
        pytest.skip('CUDA reader build required')
    batch = np.stack([sample((3, 5, 7))['mask']] * 2)
    device = cuda.asarray(batch)
    pipeline = A.Pipeline([A.DeepSupervision([1, .5])], backend='cuda')
    arrays, _, owner = pipeline._native.run_batch({'mask': device._native}, [12, 13], {}, False, False, {})
    owner.export()
    assert isinstance(arrays['mask'], list) and len(arrays['mask']) == 2
    expected = [A.Pipeline([A.DeepSupervision([1, .5])])({'mask': b}, seed=12)['mask'] for b in batch]
    for level, spec in enumerate(arrays['mask']):
        result = cuda.DeviceArray(*spec)
        np.testing.assert_array_equal(result.numpy(), np.stack([v[level] for v in expected]))
