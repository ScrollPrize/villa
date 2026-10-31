"""vc_reader.images: plain image files read and written, checked with tifffile and PIL."""
import numpy as np
import pytest
import tifffile
from PIL import Image

from vc_reader import images


@pytest.mark.parametrize('dtype', ['u1', 'i1', 'u2', 'i2', 'u4', 'i4', 'u8', 'i8', 'f4', 'f8'])
@pytest.mark.parametrize('compression', ['none', 'lzw', 'deflate', 'packbits', 'zstd'])
def test_write_tiff_values(tmp_path, dtype, compression):
    image = np.arange(64 * 48, dtype=dtype).reshape(64, 48)
    images.write_tiff(tmp_path / 'a.tif', image, compression=compression, tile=(32, 32))
    assert tifffile.TiffFile(tmp_path / 'a.tif').pages[0].is_tiled
    assert np.array_equal(tifffile.imread(tmp_path / 'a.tif'), image)
    assert np.array_equal(images.read(tmp_path / 'a.tif'), image)


def test_write_tiff_channels_pages_and_refusals(tmp_path):
    rgb = np.arange(30, dtype=np.uint8).reshape(2, 5, 3)
    images.write_tiff(tmp_path / 'c.tif', rgb)
    assert tifffile.TiffFile(tmp_path / 'c.tif').pages[0].photometric.name == 'RGB'
    assert np.array_equal(tifffile.imread(tmp_path / 'c.tif'), rgb)
    assert np.array_equal(images.read(tmp_path / 'c.tif'), rgb)

    stack = np.arange(4 * 32 * 40, dtype=np.uint16).reshape(4, 32, 40)
    images.write_tiff(tmp_path / 's.tif', stack, compression='lzw')
    assert np.array_equal(tifffile.imread(tmp_path / 's.tif'), stack)
    assert np.array_equal(images.read(tmp_path / 's.tif'), stack)

    images.write_tiff(tmp_path / 'n.tif', stack[:3, :3, :4], pages=True)  # would be channels unasked
    assert images.read(tmp_path / 'n.tif').shape == (3, 3, 4)
    colour = np.arange(4 * 32 * 40 * 3, dtype=np.uint8).reshape(4, 32, 40, 3)
    images.write_tiff(tmp_path / 'p.tif', colour)
    assert np.array_equal(images.read(tmp_path / 'p.tif'), colour)

    with pytest.raises(ValueError, match='compression'):
        images.write_tiff(tmp_path / 'b.tif', rgb, compression='jpeg')
    with pytest.raises(ValueError, match='dtype'):
        images.write_tiff(tmp_path / 'b.tif', np.zeros((4, 4), bool))
    assert not list(tmp_path.glob('b.tif*'))


def test_tiff_writer_tiles_in_any_order(tmp_path):
    image = np.arange(100 * 70 * 3, dtype=np.uint8).reshape(100, 70, 3)
    with images.TiffWriter(tmp_path / 't.tif', image.shape, np.uint8, tile=(32, 32), compression='lzw') as writer:
        assert tuple(writer.tile) == (32, 32)  # uncapped by image size
        positions = [(r, c) for r in range(0, 100, 32) for c in range(0, 70, 32)]
        for r, c in reversed(positions[1:]):
            writer.write_tile(r, c, image[r:r + 32, c:c + 32].copy())
    expected = image.copy()
    expected[:32, :32] = 0  # never written
    assert np.array_equal(tifffile.imread(tmp_path / 't.tif'), expected)
    assert not list(tmp_path.glob('*.tmp'))

    with images.TiffWriter(tmp_path / 'r.tif', image.shape, np.uint8, tile=(32, 32)) as writer:
        writer.write_tiles(image[r:r + 32, c:c + 32].copy() for r in range(0, 100, 32) for c in range(0, 70, 32))
    assert np.array_equal(tifffile.imread(tmp_path / 'r.tif'), image)
    with pytest.raises(ValueError, match='holds'), images.TiffWriter(tmp_path / 'd.tif', (32, 32), np.uint16, tile=(32, 32)) as writer:
        writer.write_tile(0, 0, np.zeros((32, 32), np.uint8))
    with pytest.raises(ValueError, match='channels'), images.TiffWriter(tmp_path / 'd.tif', (32, 32), np.uint8, tile=(32, 32)) as writer:
        writer.write_tile(0, 0, np.zeros((32, 32, 3), np.uint8))

    with pytest.raises(RuntimeError), images.TiffWriter(tmp_path / 'gone.tif', (64, 64), np.uint8, tile=(16, 16)) as writer:
        writer.write_tile(0, 0, np.zeros((16, 16), np.uint8))
        raise RuntimeError
    assert not list(tmp_path.glob('gone*'))
    with pytest.raises(ValueError, match='aligned'), images.TiffWriter(tmp_path / 'o.tif', (64, 64), np.uint8, tile=(16, 16)) as writer:
        writer.write_tile(3, 0, np.zeros((16, 16), np.uint8))


def test_open_reads_windows(tmp_path):
    image = np.arange(512 * 400, dtype=np.uint16).reshape(512, 400)
    tifffile.imwrite(tmp_path / 'w.tif', image, tile=(64, 64), compression='zlib')
    with images.open(tmp_path / 'w.tif') as reader:
        assert reader.shape == image.shape and reader.dtype == 'uint16' and reader.axes == ('y', 'x')
        assert np.array_equal(reader[100:228, 30:40], image[100:228, 30:40])

    stack = np.arange(3 * 20 * 30, dtype=np.uint8).reshape(3, 20, 30)
    tifffile.imwrite(tmp_path / 'z.tif', stack, compression='lzw', photometric='minisblack')
    with images.open(tmp_path / 'z.tif') as reader:
        assert reader.axes == ('z', 'y', 'x') and reader.shape == stack.shape
        assert np.array_equal(reader[1], stack[1])

    with tifffile.TiffWriter(tmp_path / 'mixed.tif') as writer:
        writer.write(np.ones((20, 30), np.uint8))
        writer.write(np.ones((5, 6), np.uint8) * 2)
    assert images.read(tmp_path / 'mixed.tif').shape == (20, 30)

    big = np.arange(600 * 600, dtype=np.uint16).reshape(600, 600)
    tifffile.imwrite(tmp_path / 'big.tif', big, compression='lzw')
    assert np.array_equal(images.read(tmp_path / 'big.tif'), big)

    tifffile.imwrite(tmp_path / 'four.tif', np.zeros((5, 2, 30, 40), np.uint8), photometric='minisblack')
    with pytest.raises(ValueError, match='declare'):
        images.open(tmp_path / 'four.tif')


def test_named_pages(tmp_path):
    stack = np.arange(2 * 20 * 30, dtype=np.float32).reshape(2, 20, 30)
    with images.TiffWriter(tmp_path / 'n.tif', (20, 30), np.float32, pages=2) as writer:
        writer.write_page(stack[0], name='first')
        writer.write_page(stack[1], name='second')
    with tifffile.TiffFile(tmp_path / 'n.tif') as tif:
        assert [page.tags['PageName'].value for page in tif.pages] == ['first', 'second']
    assert np.array_equal(images.read(tmp_path / 'n.tif'), stack)


def test_pixel_size(tmp_path):
    with images.TiffWriter(tmp_path / 'r.tif', (32, 32), np.uint8, tile=(16, 16), pixel_um=2.5) as writer:
        writer.write_tile(0, 0, np.ones((16, 16), np.uint8))
    with images.open(tmp_path / 'r.tif') as reader:
        rows = reader.resolved.storage_from_array.matrix
        assert reader.resolved.storage_units == 'micrometer' and abs(rows[0][0] - 2.5) < 1e-5


def test_first_page(tmp_path):
    stack = np.arange(3 * 20 * 30, dtype=np.uint8).reshape(3, 20, 30)
    images.write_tiff(tmp_path / 's.tif', stack)
    assert np.array_equal(images.read(tmp_path / 's.tif', page=1), stack[1])
    images.write_tiff(tmp_path / 'p.tif', stack[0])
    assert np.array_equal(images.read(tmp_path / 'p.tif', page=0), stack[0])


def test_memmap(tmp_path):
    image = np.arange(4000 * 300, dtype=np.uint16).reshape(4000, 300)
    images.write_tiff(tmp_path / 'm.tif', image)
    mapped = images.memmap(tmp_path / 'm.tif')
    assert isinstance(mapped, np.memmap) and np.array_equal(mapped, image)

    created = images.memmap(tmp_path / 'c.tif', (4000, 300), np.float32)
    created[10, 20] = 7
    created.flush()
    del created
    expected = np.zeros((4000, 300), np.float32)
    expected[10, 20] = 7
    assert np.array_equal(tifffile.imread(tmp_path / 'c.tif'), expected)

    images.write_tiff(tmp_path / 'lzw.tif', image, compression='lzw')
    images.write_tiff(tmp_path / 'tiled.tif', image, tile=(64, 64))
    images.write_tiff(tmp_path / 'pages.tif', np.stack([image, image]))
    for name in ('lzw.tif', 'tiled.tif', 'pages.tif'):
        with pytest.raises(ValueError, match='memory-mapped'):
            images.memmap(tmp_path / name)


def test_png_and_jpeg(tmp_path):
    image = np.arange(1200, dtype=np.uint16).reshape(30, 40)
    Image.fromarray(image).save(tmp_path / 'a.png')
    assert np.array_equal(images.read(tmp_path / 'a.png'), image)
    with images.open(tmp_path / 'a.png') as reader:
        assert reader.shape == image.shape
    rgb = np.full((16, 16, 3), 200, np.uint8)
    Image.fromarray(rgb).save(tmp_path / 'a.jpg')
    assert images.read(tmp_path / 'a.jpg').shape == (16, 16, 3)
