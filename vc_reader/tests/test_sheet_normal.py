"""Independent reference for the graph-linking anisotropic normal recipe."""
import json
import numpy as np
import pytest
from scipy.ndimage import correlate1d

from build_support import load_build
load_build()

from vc_reader import _reader
from vc_reader.reader import ZarrReader
from vc_reader.sheet_normal import sample_sheet_normals, sheet_normals, sheet_normal_regions


def reference(image, offset):
    gradients = []
    for sigma in (2., 1.):
        r = int(3 * sigma + .5)
        t = np.arange(-r, r+1, dtype=np.float64)
        smooth = np.exp(-.5*t*t/(sigma*sigma))
        smooth /= smooth.sum()
        derivative = smooth*t/(sigma*sigma)
        components = []
        for component in range(3):
            value = image.astype(np.float32) * np.float32(1/255)
            for axis in range(3):
                value = correlate1d(value, derivative if axis == 2-component else smooth, axis=axis)
                selection = [slice(None)]*3
                selection[axis] = slice(6, 103, 3)
                value = value[tuple(selection)]
            components.append(value)
        gradients.append(np.stack(components, -1).reshape(-1, 3).astype(np.float64))
    zz, yy, xx = np.meshgrid(*([np.arange(33)*3]*3), indexing='ij')
    delta = np.stack((xx, yy, zz), -1).reshape(-1, 3) - (48 + np.asarray(offset, np.float64))
    mag = np.sqrt((gradients[1]**2).sum(-1)).astype(np.float32)
    floor = max(np.median(mag), 1e-8)
    normal = np.zeros(3)
    for step in range(4):
        g = gradients[bool(step)]
        exponent = (delta**2).sum(-1)/24**2
        if step:
            exponent += (delta @ normal)**2*(1/4**2 - 1/24**2)
        weight = np.exp(-.5*exponent)
        if step:
            weight /= np.maximum(mag, floor)
        tensor = (g*weight[:, None]).T @ g / weight.sum()
        values, vectors = np.linalg.eigh(tensor)
        if values[-1] <= 1e-12 or (values[-1]-values[-2])/values[-1] <= 1e-6:
            return np.zeros(4)
        normal = vectors[:, -1]
        gap = (values[-1]-values[-2])/values[-1]
    return np.r_[normal, gap]


def examples():
    z, y, x = np.indices((109,109,109), dtype=np.float32)-54
    rng = np.random.default_rng(713)
    distance = .8*x+.5*y+.3*z
    return np.stack([
        np.clip(120+70*np.sin(distance*.3), 0, 255).astype(np.uint8),
        np.clip(120+9*np.sin((distance+.004*y*y-.003*z*z)*.5)+rng.normal(0,2,x.shape), 0,255).astype(np.uint8),
        np.full(x.shape, 43, np.uint8),
        np.clip(120+30*np.sin(x*.4)+30*np.sin(y*.4)+30*np.sin(z*.4), 0,255).astype(np.uint8),
    ])


@pytest.mark.parametrize('backend', ['cpu','cuda'])
def test_reference(backend):
    image = examples()
    offsets = np.array([[0,0,0],[2.8,.31,1.42],[1,2,1],[0,0,0]], np.float32)
    if backend == 'cuda':
        from vc_reader.cuda import is_available, asarray
        if not is_available():
            pytest.skip('CUDA build/device required')
        result = sheet_normals(asarray(image), offsets).numpy()
    else:
        result = sheet_normals(image, offsets)
    expected = np.stack([reference(i, o) for i,o in zip(image,offsets)])
    np.testing.assert_allclose(result[:,3], expected[:,3], atol=2e-5)
    valid = expected[:,3] > 0
    alignment = np.abs((result[valid,:3]*expected[valid,:3]).sum(-1))
    np.testing.assert_allclose(alignment, 1, atol=2e-6)
    np.testing.assert_array_equal(result[~valid], 0)


def test_lattice_and_support():
    starts, offsets, valid = sheet_normal_regions([[54,54,54],[55.25,56.5,54.125],[53.99,54,54],[60,60,60]], [109]*3)
    np.testing.assert_array_equal(starts[:2], 0)
    np.testing.assert_allclose(offsets[1], [1.25,2.5,.125])
    np.testing.assert_array_equal(valid, [True,True,False,False])
    with pytest.raises(ValueError):
        sheet_normal_regions([[np.nan,0,0]], [109]*3)


def test_invalid_and_float_input():
    image = examples()[:1]
    np.testing.assert_allclose(sheet_normals(image.astype(np.float32)), sheet_normals(image), atol=1e-7)
    np.testing.assert_array_equal(sheet_normals(image, valid=[False]), 0)
    with pytest.raises((ValueError, TypeError)):
        sheet_normals(image.astype(np.float64))
    with pytest.raises(ValueError):
        sheet_normals(image, [[3,0,0]])
    assert sheet_normals(np.empty((0,109,109,109),np.uint8)).shape == (0,4)


def _normal_volume(root):
    values = np.ascontiguousarray(examples()[0])
    (root/'.zarray').write_text(json.dumps(dict(zarr_format=2, shape=values.shape, chunks=values.shape, dtype='|u1',
                                                compressor=None, filters=None, fill_value=0, order='C')))
    (root/'0.0.0').write_bytes(values.tobytes())
    return values


POINTS = [[54., 54., 54.], [55.25, 56.5, 54.125], [53.99, 54., 54.]]


def test_regions_and_sampling_match_explicit_patches(tmp_path):
    values = _normal_volume(tmp_path)
    starts, offsets, valid = sheet_normal_regions(POINTS, values.shape)
    np.testing.assert_array_equal(starts, [[0, 0, 0], [0, 0, 0], [0, 0, -3]])
    np.testing.assert_array_equal(valid, [True, True, False])
    np.testing.assert_array_equal(offsets[1], np.float32([1.25, 2.5, .125]))
    with ZarrReader.open(tmp_path) as reader:
        result = sample_sheet_normals(reader, POINTS)
    expected = sheet_normals(np.stack([values]*3), offsets, valid=valid)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result[2], 0)
    with pytest.raises(ValueError):
        sheet_normal_regions([[np.nan, 0, 0]], values.shape)


def test_sampling_on_the_device_matches_the_cpu(tmp_path, torch_cuda):
    _normal_volume(tmp_path)
    with ZarrReader.open(tmp_path) as reader, reader.cuda(cache_bytes=64 << 20) as device:
        expected = sample_sheet_normals(reader, POINTS)
        result = sample_sheet_normals(device, POINTS)
        np.testing.assert_allclose(result.numpy(), expected, atol=2e-5)


def test_cpu_normal_output():
    patches = examples()[:1]
    expected = sheet_normals(patches)
    out = np.empty((1,4),np.float32)
    assert sheet_normals(patches, out=out) is out
    np.testing.assert_array_equal(out, expected)
    for bad in [np.empty((4,),np.float32), np.empty((1,4),np.int32), np.empty((1,8),np.float32)[:,::2]]:
        with pytest.raises(ValueError):
            sheet_normals(patches,out=bad)
    floats = patches.astype(np.float32)
    with pytest.raises(ValueError, match='overlap'):
        sheet_normals(floats,out=floats.reshape(-1)[:4].reshape(1,4))


def test_normal_output_other_stream(torch_cuda):
    torch = torch_cuda
    host = examples()[:2]
    expected = sheet_normals(host)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        patches = torch.as_tensor(host,device='cuda')
        out = torch.empty((2,4),device='cuda')
        stream.synchronize()
        _reader._set_stream(stream.cuda_stream)
        try:
            assert sheet_normals(patches,out=out) is out
            result = out.clone()
            with pytest.raises(ValueError, match='out'):
                sheet_normals(patches,out=torch.empty((2,4),device='cuda',dtype=torch.int32))
            floats = patches.float()
            stream.synchronize()
            with pytest.raises(ValueError, match='overlap'):
                sheet_normals(floats,out=floats.flatten()[:8].reshape(2,4))
        finally:
            stream.synchronize()
            _reader._set_stream(0)
    np.testing.assert_allclose(result.cpu(),expected,atol=2e-5)
