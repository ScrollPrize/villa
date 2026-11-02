import numpy as np
from build_support import load_build
load_build()
from vc_reader import _reader


def test_display_conversion_and_palette():
    values = np.array([-1, 1.9, 255, np.nan, np.inf], np.float32)
    output = np.empty(5, np.uint8)
    coverage = np.empty_like(output)
    _reader.display_to_uint8(values, output, coverage)
    np.testing.assert_array_equal(output, [0, 1, 255, 0, 0])
    np.testing.assert_array_equal(coverage, [1, 1, 1, 0, 0])
    ramp = _reader.window_level_lut(10, 110)
    assert ramp[10] == 0xff000000 and ramp[60] == 0xff7f7f7f and ramp[110] == 0xffffffff


def test_pathfinding_entered_pixel_cost_and_ties():
    costs = np.array([[1, 100, 1], [1, 1, 1], [np.inf, np.inf, 1]], np.float32)
    path = _reader.shortest_path(costs, 0, 8)
    assert path == [0, 4, 8]
    assert _reader.shortest_path(costs, 4, 4) == [4]
    costs[1] = np.inf
    assert _reader.shortest_path(costs, 0, 8) == []


def test_native_image_writers_with_independent_decoders(tmp_path):
    import io
    from PIL import Image
    pixels = np.zeros((16, 20, 3), np.uint8)
    pixels[:] = [220, 30, 80]
    png = _reader.encode_png(pixels)
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(png))), pixels)
    jpeg = _reader.encode_jpeg(pixels, 95)
    assert np.max(np.abs(np.asarray(Image.open(io.BytesIO(jpeg))).astype(int) - pixels)) <= 3
    for suffix in ['png', 'jpg', 'tif']:
        path = tmp_path / ('image.' + suffix)
        _reader.write_image(path, pixels)
        assert Image.open(path).size == (20, 16)
        _reader.write_image(path, np.zeros_like(pixels))
        assert not np.asarray(Image.open(path)).any()


def test_bulk_display_kernels():
    pixels = np.array([[0, 1], [2, 255]], np.uint8)
    palette = np.arange(256, dtype=np.uint32) | np.uint32(0xff000000)
    output = np.empty((2, 2), np.uint32)
    _reader.apply_lut(pixels, palette, output)
    np.testing.assert_array_equal(output, palette[pixels])
    base = np.array([0xff204060, 0xff204060], np.uint32)
    overlay = np.array([0x80402010, 0], np.uint32)
    result = np.empty_like(base)
    _reader.blend(base, overlay, 1., result)
    np.testing.assert_array_equal(result, [0xff504040, 0xff204060])


def test_slab_compositing_cpu_cuda():
    import pytest
    values = np.full((2, 8, 9), 200, np.uint8)
    values[0] = 100
    mask = np.ones_like(values)
    colors = _reader.window_level_lut(0, 255)
    opacity = [0.5] * 256
    projection = _reader.SlabProjection()
    offsets = np.zeros((2, 2), np.float32)
    depths = np.zeros(2, np.float32)
    rgb, coverage = _reader.composite_slab(values, mask, colors, opacity, projection, offsets, depths)
    np.testing.assert_array_equal(rgb[:-1, :-1], 125)
    assert coverage[:-1, :-1].all()
    assert not coverage[-1].any() and not coverage[:, -1].any()
    # Exercise lighting, masked neighbors and fractional layer positions.
    values[1] = np.arange(72, dtype=np.uint8).reshape(8, 9) * 3
    mask[1, 2:4, 3:5] = 0
    offsets[1] = [0.25, -0.3]
    cpu = _reader.composite_slab(values, mask, colors, opacity, projection, offsets, depths, lighting=.8)
    try:
        gpu = _reader.composite_slab(values, mask, colors, opacity, projection, offsets, depths, lighting=.8, cuda=True)
    except RuntimeError as error:
        if 'CUDA' in str(error) or 'cuda' in str(error) or 'driver' in str(error):
            pytest.skip(str(error))
        raise
    np.testing.assert_array_equal(cpu[1], gpu[1])
    np.testing.assert_allclose(cpu[0].astype(int), gpu[0].astype(int), atol=1)
    np.testing.assert_allclose(_reader.opacity_lut(.2, .8, .7, gamma=1.7),
                               _reader.opacity_lut(.2, .8, .7, gamma=1.7, cuda=True), atol=1e-6)


def test_cuda_bulk_display_matches_cpu():
    import pytest
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    pixels = np.arange(256, dtype=np.uint8).reshape(16,16)
    palette = np.arange(256, dtype=np.uint32) | np.uint32(0xff000000)
    output = torch.empty((16,16), dtype=torch.uint32, device='cuda')
    _reader.apply_lut(torch.from_numpy(pixels).cuda(), torch.from_numpy(palette).cuda(), output)
    np.testing.assert_array_equal(output.cpu().numpy(), palette[pixels])
    base = torch.tensor([0xff204060,0xff204060],dtype=torch.uint32,device='cuda')
    over = torch.tensor([0x80402010,0],dtype=torch.uint32,device='cuda')
    mixed = torch.empty_like(base)
    _reader.blend(base, over, 1., mixed)
    np.testing.assert_array_equal(mixed.cpu().numpy(),[0xff504040,0xff204060])
