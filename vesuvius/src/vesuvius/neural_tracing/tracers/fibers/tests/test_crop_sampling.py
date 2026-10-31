"""Crop sampling: vc_reader crops (crops.scalar_crops) equal their plain NumPy definition up to float32
interpolation, the per-crop z-score equals its NumPy definition within float32 rounding, and the CUDA crops
(crops.cuda_scalar_crops) match the CPU ones."""
import numpy as np
import pytest
import torch
from vc_reader.reader import ZarrWriter, open_zarr

from vesuvius.neural_tracing.tracers.shared.crop_sampling.crops import crop_grids, cuda_scalar_crops, scalar_crops
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


def ct_items(tmp_path, input_scale):
    """A chunked local uint8 Zarr CT, its spec and four oriented crop items, some reaching past the array."""
    rng = np.random.default_rng(input_scale)
    raw = rng.integers(0, 256, (40, 44, 48), dtype=np.uint8)
    array = ZarrWriter.create(tmp_path/'ct'/'0', shape=raw.shape, chunks=(16, 16, 16), dtype='u1', fill_value=0,
                              compressor=None)
    array[:] = raw
    spec = CTVolumeSpec(str(tmp_path/'ct'), ct_level=0, ct_grid_scale=4., grid_scale=4.*input_scale,
                        ct_normalization=RECORD)
    frames = [np.linalg.qr(rng.normal(size=(3, 3)))[0]*1.5 for _ in range(4)]
    positions = [rng.uniform(2, 38, 3)/input_scale for _ in range(4)]
    return raw, spec, [dict(pos=p, frame=f) for p, f in zip(positions, frames)]


@pytest.mark.parametrize('crop', [CropSpec(depth=20, width=16, behind=10, spacing=.5),
                                  CropSpec(depth=12, width=10, height=6, behind=3.5, spacing=1.)])
@pytest.mark.parametrize('input_scale', [1, 2])
def test_scalar_crops_match_reference(tmp_path, crop, input_scale):
    """Crop geometry (``crop_grids``: positions, frames and the CT scale) and trilinear values, zero outside the
    array, on a chunked local Zarr; rows near the edge reach past the array."""
    raw, spec, items = ct_items(tmp_path, input_scale)
    ct = open_zarr(spec.ct_zarr, level=spec.ct_level)
    crops = scalar_crops(items, ct, spec, crop, normalize=False).numpy()
    for crop_values, item in zip(crops, items):
        expected = reference_crop(raw[None], np.zeros(3), item['pos']*input_scale, item['frame']*input_scale, crop)
        np.testing.assert_allclose(crop_values[0], expected, rtol=0, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA device')
@pytest.mark.parametrize('crop', [CropSpec(depth=20, width=16, behind=10, spacing=.5),
                                  CropSpec(depth=12, width=10, height=6, behind=3.5, spacing=1.)])
@pytest.mark.parametrize('input_scale', [1, 2])
def test_cuda_scalar_crops_match_cpu(tmp_path, crop, input_scale):
    """The tracer's GPU crops (``use_gpu_crops``: ``ZarrReader.cuda()`` + ``cuda_scalar_crops``) equal the CPU
    crops, raw and z-scored."""
    _, spec, items = ct_items(tmp_path, input_scale)
    ct = open_zarr(spec.ct_zarr, level=spec.ct_level)
    cuda = ct.cuda(cache_bytes=8*int(np.prod(ct.chunks)), device=torch.cuda.current_device())
    gpu = cuda_scalar_crops(items, cuda, spec, crop).cpu().numpy()
    cpu = scalar_crops(items, ct, spec, crop).numpy()
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-4)
    raw_cpu = scalar_crops(items, ct, spec, crop, normalize=False).numpy()[:, 0]
    out = torch.empty(raw_cpu.shape, dtype=torch.float32, device=torch.device('cuda', cuda.device))
    size, origins, axes = crop_grids(items, crop, spec.input_scale)
    cuda.read_affine_batch(size, origins, axes, order=1, out=out)
    np.testing.assert_allclose(out.cpu().numpy()/255., raw_cpu, rtol=0, atol=1e-6)


def test_normalize_matches_numpy():
    rng = np.random.default_rng(0)
    for image in (rng.random((9, 17, 13), dtype=np.float32), np.full((4, 5, 6), .25, np.float32)):
        mean, std = image.mean(dtype=np.float64), image.std(dtype=np.float64)
        expected = (image-mean)/max(std, ZSCORE_EPSILON)
        normalize_ct(image, RECORD)
        np.testing.assert_allclose(image, expected, atol=1e-6)
    for bad in (np.nan, np.inf):
        image = rng.random((4, 5, 6), dtype=np.float32)
        image[1, 2, 3] = bad
        with pytest.raises(ValueError, match='finite'):
            normalize_ct(image, RECORD)
