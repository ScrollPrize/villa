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

