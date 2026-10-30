"""CPU crop sampling: vc_reader crops (crops.scalar_crops) equal their plain NumPy definition up to float32
interpolation, and the per-crop z-score equals its NumPy definition bit for bit."""
import numpy as np
import pytest
import zarr
from vc_reader.reader import open_zarr

from vesuvius.neural_tracing.tracers.shared.crop_sampling.crops import scalar_crops
from vesuvius.neural_tracing.tracers.shared.crop_sampling.normalization import (
    ZSCORE_EPSILON, ZSCORE_METHOD, normalize_ct)
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolumeSpec
from vesuvius.neural_tracing.tracers.shared.geometry import CropSpec, crop_local_grid

RECORD = dict(method=ZSCORE_METHOD)


def reference_crop(raw, start, pos, frame, crop):
    """Trilinear CT/255 at every crop point, zero outside ``raw``, in the sampler's order of operations."""
    g = crop_local_grid(crop).reshape(-1, 3).astype(np.float64)
    w = [pos[r] + frame[r, 0]*g[:, 0] + frame[r, 1]*g[:, 1] + frame[r, 2]*g[:, 2] for r in range(3)]
    q = np.stack([w[2]-start[0], w[1]-start[1], w[0]-start[2]], 1)
    i0 = np.floor(q).astype(np.int64)
    fz, fy, fx = (q-i0).T
    value = np.zeros(len(q))
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                idx = i0+(dz, dy, dx)
                inside = ((idx >= 0) & (idx < raw.shape[1:])).all(1)
                v = np.zeros(len(q))
                v[inside] = raw[0][tuple(idx[inside].T)]
                weight = (fz if dz else 1-fz)*(fy if dy else 1-fy)*(fx if dx else 1-fx)
                value += np.where(inside, weight*v, 0.)
    return (value*(1./255.)).astype(np.float32).reshape(crop.depth, crop.height or crop.width, crop.width)


@pytest.mark.parametrize('crop', [CropSpec(depth=20, width=16, behind=10, spacing=.5),
                                  CropSpec(depth=12, width=10, height=6, behind=3.5, spacing=1.)])
@pytest.mark.parametrize('input_scale', [1, 2])
def test_scalar_crops_match_reference(tmp_path, crop, input_scale):
    """Crop geometry (``crop_grids``: positions, frames and the CT scale) and trilinear values, zero outside the
    array, on a chunked local Zarr; rows near the edge reach past the array."""
    rng = np.random.default_rng(input_scale)
    raw = rng.integers(0, 256, (40, 44, 48), dtype=np.uint8)
    array = zarr.create_array(str(tmp_path/'ct'/'0'), shape=raw.shape, chunks=(16, 16, 16), dtype='u1', fill_value=0,
                              zarr_format=2, compressors=None)
    array[:] = raw
    spec = CTVolumeSpec(str(tmp_path/'ct'), ct_level=0, ct_grid_scale=4., grid_scale=4.*input_scale)
    ct = open_zarr(spec.ct_zarr, level=spec.ct_level)
    frames = [np.linalg.qr(rng.normal(size=(3, 3)))[0]*1.5 for _ in range(4)]
    positions = [rng.uniform(2, 38, 3)/input_scale for _ in range(4)]
    items = [dict(pos=p, frame=f) for p, f in zip(positions, frames)]
    crops = scalar_crops(items, ct, spec, crop, normalize=False).numpy()
    for crop_values, p, f in zip(crops, positions, frames):
        expected = reference_crop(raw[None], np.zeros(3), p*input_scale, f*input_scale, crop)
        np.testing.assert_allclose(crop_values[0], expected, rtol=0, atol=1e-5)


def test_normalize_matches_numpy():
    rng = np.random.default_rng(0)
    for image in (rng.random((9, 17, 13), dtype=np.float32), np.full((4, 5, 6), .25, np.float32)):
        mean, std = image.mean(dtype=np.float32), image.std(dtype=np.float32)
        expected = (image-mean)/np.float32(max(float(std), ZSCORE_EPSILON))
        normalize_ct(image, RECORD)
        assert np.array_equal(image.view(np.uint32), expected.view(np.uint32))
    for bad in (np.nan, np.inf):
        image = rng.random((4, 5, 6), dtype=np.float32)
        image[1, 2, 3] = bad
        with pytest.raises(ValueError, match='nonfinite'):
            normalize_ct(image, RECORD)
