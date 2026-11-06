"""SheetCompression pulls parallel sheets together along an axis, in voxels, with all targets aligned.

The phantom is binary (sheets 1, gaps 0) and intensity_sigma=0, so the sampling coordinate has a closed form:
x + strength * (gap voxels in 0..x).
"""
import numpy as np
import pytest

from build_support import load_build
load_build()

from vc_reader import transforms as A

SHEETS, THICKNESS, PERIOD, FIRST = 16, 3, 10, 3
LENGTH = SHEETS * PERIOD
Z = np.arange(LENGTH)
SHEET_ID = np.where((Z - FIRST) % PERIOD < THICKNESS, (Z - FIRST) // PERIOD + 1, 0)
RAMP = Z.astype(np.float32)
BACKENDS = ['cpu', 'cuda'] if A._transforms.cuda_available() else ['cpu']
backends = pytest.mark.parametrize('backend', BACKENDS)
cases = pytest.mark.parametrize('backend,strength', [(b, s) for b in BACKENDS for s in (.1, .2, .35)])


def along(values, position=0, dims=3, lateral=8):
    """Broadcast a profile along axis `position` of a (1, ...) volume with `dims` spatial axes."""
    shape = [1] + [LENGTH if a == position else lateral for a in range(dims)]
    return np.broadcast_to(values.reshape([1] + [LENGTH if a == position else 1 for a in range(dims)]), shape).copy()


def phantom(lateral=8):
    return along((SHEET_ID > 0).astype(np.float32), lateral=lateral), along(SHEET_ID.astype(np.int16), lateral=lateral)


def source(strength):
    """Input index sampled at each output index (not clamped to the volume)."""
    return Z + strength * np.cumsum(SHEET_ID == 0)


def centroids(profile):
    """Intensity-weighted centroids of the runs above .5."""
    edges = np.flatnonzero(np.diff(np.r_[0, profile > .5, 0]))
    return [float((Z[a:b] * profile[a:b]).sum() / profile[a:b].sum()) for a, b in zip(edges[::2], edges[1::2])]


def compress(sample, targets, backend, strength=.35, axes='z', smoothing=0, **options):
    step = A.SheetCompression(strength=strength, intensity_sigma=0, smoothing=smoothing, axes=axes, **options)
    return A.Pipeline([step], targets=targets, backend=backend)(sample, seed=0)


def run(backend, strength=.35, padding=None, lateral=8, smoothing=0, **options):
    image, ids = phantom(lateral)
    valid = np.ones_like(image)
    if padding:
        valid[:, slice(*padding)] = 0
    targets = {'image': A.Image(), 'seg': A.Label(), 'ramp': A.ScalarField(), 'padding_mask': A.ValidMask()}
    sample = {'image': image, 'seg': ids, 'ramp': along(RAMP, lateral=lateral), 'padding_mask': valid}
    out = compress(sample, targets, backend, strength, smoothing=smoothing, **options)
    return {key: value[0, :, 4, 4] if lateral == 8 else value for key, value in out.items()}


@cases
@pytest.mark.parametrize('options', [{}, {'threshold': .4}])
def test_sampling_coordinate_is_in_voxels_and_moves_forward(backend, strength, options):
    out = run(backend, strength, **options)
    ramp, mask, outside = out['ramp'], out['padding_mask'], source(strength) > LENGTH - 1
    np.testing.assert_allclose(ramp, np.minimum(source(strength), LENGTH - 1), atol=1e-3)
    assert (ramp >= Z - 1e-3).all() and ramp[LENGTH // 2] > LENGTH // 2 + 1
    # The tail pulled in from outside the volume is marked invalid and blank.
    assert outside.sum() > 5 and abs(int(mask.size - mask.sum()) - int(outside.sum())) <= 1
    assert (mask[:10] == 1).all() and (mask[-3:] == 0).all()
    np.testing.assert_array_equal(out['image'][mask == 0], 0)


@backends
@pytest.mark.parametrize('dims,axes,position', [(3, 'z', 0), (3, 'y', 1), (3, 'x', 2), (2, 'y', 0), (2, 'x', 1)])
def test_every_axis_gives_the_same_closed_form_in_3d_and_2d(backend, dims, axes, position):
    sample = {'image': along((SHEET_ID > 0).astype(np.float32), position, dims), 'ramp': along(RAMP, position, dims)}
    out = compress(sample, {'image': A.Image(), 'ramp': A.ScalarField()}, backend, axes=axes)
    columns = np.moveaxis(out['ramp'], 1 + position, 1).reshape(LENGTH, -1)
    np.testing.assert_allclose(columns, np.broadcast_to(np.minimum(source(.35), LENGTH - 1)[:, None], columns.shape), atol=1e-3)


@cases
def test_sheets_move_together_by_the_expected_ratio_and_all_are_kept(backend, strength):
    out = run(backend, strength)
    found, labels = centroids(out['image']), out['seg']
    # Gaps shrink by 1 / (1 + strength) and sheets keep their thickness; interpolated edges shift centroids a little.
    spacing = np.diff(found).mean() / PERIOD
    assert spacing == pytest.approx((THICKNESS + (PERIOD - THICKNESS) / (1 + strength)) / PERIOD, abs=.02) and spacing < 1
    assert len(found) == SHEETS
    assert sorted(set(labels[labels > 0].tolist())) == list(range(1, SHEETS + 1))
    assert not ((labels[:-1] > 0) & (labels[1:] > 0) & (labels[:-1] != labels[1:])).any()


@backends
@pytest.mark.parametrize('strength', [.1, .35])
def test_image_labels_and_validity_stay_aligned_with_padding(backend, strength):
    out = run(backend, strength, padding=(60, 80))
    mask, labels, coordinate = out['padding_mask'], out['seg'], source(strength)
    assert (mask[(coordinate > 60) & (coordinate < 79)] == 0).all() and (mask[coordinate < 59] == 1).all()
    found = centroids(out['image'])
    assert len(found) == SHEETS - 2  # Padding blanks the image sheets that were inside it, and only those.
    label_centers = [np.flatnonzero(labels == k).mean() for k in range(1, SHEETS + 1)]
    for center in found:
        if (mask[int(center) - 3:int(center) + 4] == 0).any():
            continue  # A sheet cut by the edge of the padding has a shifted centroid.
        # Labels are nearest-neighbour and the image linear, hence half a voxel plus rounding.
        assert min(abs(center - other) for other in label_centers) < .6


@backends
def test_smoothing_leaves_a_laterally_uniform_volume_unchanged(backend):
    plain, smooth = (run(backend, lateral=16, smoothing=s) for s in (0, 2))
    for key in ('image', 'ramp', 'padding_mask'):
        np.testing.assert_allclose(smooth[key], plain[key], atol=1e-3)
    assert (smooth['seg'] != plain['seg']).mean() < .01  # A rounding error can cross a nearest-neighbour tie.


@backends
def test_smoothing_blurs_the_displacement_across_the_axis_only(backend):
    ndimage = pytest.importorskip('scipy.ndimage')
    image = phantom(16)[0]
    image[..., 8:] = 1  # Right half is solid: no gaps, no displacement.
    field = ndimage.gaussian_filter(.35 * np.cumsum(1 - image.astype(np.float64), axis=1), (0, 0, 2, 2), mode='mirror', truncate=6)
    expected = np.clip(Z.reshape(1, -1, 1, 1) + field, 0, LENGTH - 1)
    out = compress({'image': image, 'ramp': along(RAMP, lateral=16)}, {'image': A.Image(), 'ramp': A.ScalarField()},
                   backend, smoothing=2)
    np.testing.assert_allclose(out['ramp'], expected, atol=.05)  # Kernel truncation differs by ~.02.
    assert abs(expected[0, 80, :, 7] - expected[0, 80, :, 8]).max() > 0


@cases
def test_points_follow_their_sheet(backend, strength):
    image, ids = phantom()
    centers = (FIRST + PERIOD * np.arange(SHEETS) + 1).astype(np.float32)
    points = np.column_stack([centers, np.full((SHEETS, 2), 4, np.float32)])
    targets = {'image': A.Image(), 'seg': A.Label(), 'points': A.Points(order='zyx', outside='keep')}
    out = compress({'image': image, 'seg': ids, 'points': points}, targets, backend, strength)
    # The point at the middle of sheet k lands where the sampling coordinate equals its input index.
    np.testing.assert_allclose(out['points'][:, 0], np.interp(centers, source(strength), Z), atol=1e-3)
    for k, z in enumerate(out['points'][:, 0], start=1):
        assert out['seg'][0, :, 4, 4][int(round(float(z)))] == k
