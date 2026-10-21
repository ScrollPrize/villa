"""Extra channels follow the writer's layout and preserve their dtype."""
import numpy as np
import pytest
import tifffile

from vesuvius.tifxyz.writer import TifxyzWriter


@pytest.mark.parametrize('tile_size,shape,tiled', [(None, (32, 48), False),
                                                 (1024, (32, 48), True),
                                                 (1024, (7, 9), False)])
def test_extra_channel_layout(tmp_path, tile_size, shape, tiled):
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    writer = TifxyzWriter(tmp_path/'surface', tile_size=tile_size, compression=None, overwrite=True)
    writer._ensure_directory()
    writer.write_tiff('x.tif', data)
    writer.write_mask(data > 0)
    writer.write_extra_channel('labels', data)
    path = tmp_path/'surface'/'labels.tif'
    with tifffile.TiffFile(path) as tif:
        assert tif.pages[0].is_tiled == tiled
        assert tif.pages[0].dtype == data.dtype
        if tile_size is None:
            assert len(tif.pages[0].dataoffsets) == 1
    np.testing.assert_array_equal(tifffile.imread(path), data)
    if tile_size is None:
        np.testing.assert_array_equal(tifffile.memmap(path), data)
