"""Independent references for the shared kernels used by the optimization pass."""
import numpy as np
import pytest

from vc_reader import augment as A


@pytest.mark.parametrize("shape", [(7, 11), (5, 7, 11), (1, 3, 2)])
@pytest.mark.parametrize("radius", [0, 1, 3])
@pytest.mark.parametrize("operation", ["dilate", "erode", "open", "close"])
@pytest.mark.parametrize("neutral", [False, True])
@pytest.mark.parametrize("dtype", [np.int16, np.uint64, np.float64])
def test_morphology_small_and_large_footprints(shape, radius, operation, neutral, dtype):
    ndi = pytest.importorskip("scipy.ndimage")
    labels = np.random.default_rng(19).integers(-1, 4, size=(2, *shape), dtype=np.int16)
    labels = (np.maximum(labels, 0) if dtype == np.uint64 else labels).astype(dtype)
    coords = np.indices((2 * radius + 1,) * len(shape)) - radius
    footprint = (coords * coords).sum(axis=0) <= radius * radius
    expected = []
    for channel in labels:
        binary = channel > 0
        if operation in ("dilate", "close"):
            binary = ndi.binary_dilation(binary, structure=footprint, border_value=0)
        if operation != "dilate":
            binary = ndi.binary_erosion(binary, structure=footprint, border_value=int(neutral))
        if operation == "open":
            binary = ndi.binary_dilation(binary, structure=footprint, border_value=0)
        expected.append(binary * channel.max())
    out = A.Pipeline([A.Morphology(operation, radius=radius, neutral_border=neutral)])(
        {"mask": labels}, seed=1)["mask"]
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == labels.dtype


def test_morphology_preserves_negative_zero():
    labels = np.full((1, 3, 5), -2., dtype=np.float64)
    out = A.Pipeline([A.Morphology("close")])({"mask": labels}, seed=1)["mask"]
    assert (out == 0).all()
    assert np.signbit(out).all()


@pytest.mark.parametrize("size", [2, 3, 4, 7])
def test_median_tracks_large_rank_jumps_and_empty_bins(size):
    ndi = pytest.importorskip("scipy.ndimage")
    image = np.broadcast_to(((np.arange(257) // 3) % 2) * 255., (1, 5, 9, 257)).copy()
    # Include all 256 palette values globally, but leave large gaps in most
    # local histograms. Moving across stripes jumps between the extreme ranks.
    image[0, 0, 0, :256] = np.arange(256)
    expected = ndi.median_filter(image[0], size=size, mode="reflect")
    output = A.Pipeline([A.MedianFilter(size)])({"image": image}, seed=123)["image"]
    np.testing.assert_array_equal(output[0], expected)


@pytest.mark.parametrize("shape", [(9, 11), (7, 9, 11), (1, 3, 2)])
@pytest.mark.parametrize("border", [False, True])
@pytest.mark.parametrize("scale", [.8, 1.2])
def test_affine_interior_and_padding_reference(shape, border, scale):
    ndi = pytest.importorskip("scipy.ndimage")
    image = np.random.default_rng(17).normal(size=(2, *shape))
    center = (np.asarray(shape) - 1) / 2
    matrix = np.eye(len(shape)) / scale
    offset = center - (center + .25) / scale
    expected = np.stack([
        ndi.affine_transform(c, matrix, offset, order=1, prefilter=False,
                             mode="nearest" if border else "grid-constant", cval=-.75)
        for c in image
    ])
    out = A.Pipeline([A.Affine(scale=scale, translate=.25, border=border)],
                     targets={"image": A.Image(fill=-.75)})(
        {"image": image}, seed=1)["image"]
    np.testing.assert_allclose(out, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("shape", [(7, 11), (5, 7, 11)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_local_gamma_separable_field_reference(shape, dtype):
    special = pytest.importorskip("scipy.special")
    image = np.random.default_rng(21).normal(size=(2, *shape)).astype(dtype)
    field = np.ones(shape)
    for axis, extent in enumerate(shape):
        x = np.arange(extent, dtype=np.float64)
        kernel = .5 * (special.erf((x + .5 - .4 * extent) / (2. * np.sqrt(2.)))
                       - special.erf((x - .5 - .4 * extent) / (2. * np.sqrt(2.))))
        broadcast = [1] * len(shape)
        broadcast[axis] = extent
        field *= kernel.reshape(broadcast)
    field = (field - field.min()) / (field.max() - field.min())
    expected = []
    for channel in image.astype(np.float64):
        low, span = channel.min(), channel.max() - channel.min()
        norm = (channel - low) / span
        expected.append(((norm * (1 - field) + norm ** .8 * field) * span + low).astype(dtype))
    out = A.Pipeline([A.LocalGamma(scale=2, loc=.4, gamma=.8)])({"image": image}, seed=1)["image"]
    np.testing.assert_allclose(out, expected, rtol=2e-6 if dtype == np.float32 else 2e-14,
                               atol=2e-6 if dtype == np.float32 else 2e-14)
