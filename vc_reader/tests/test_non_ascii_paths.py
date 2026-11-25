"""Local paths with characters outside ASCII name the same files in the reader as in Python.

Python hands the reader UTF-8. Microsoft's C++ library reads a narrow string in the system code page unless
told otherwise, which turned ``café.zarr`` into ``cafÃ©.zarr`` on a code page 1252 system. The directory here
mixes scripts, so no single Windows code page holds all of it and a round trip through one cannot pass either.
"""
import json

import numpy as np
import pytest
import tifffile

from build_support import load_build
load_build()

from test_source_storage import array, pyramid
from vc_reader import images, tifxyz
from vc_reader.reader import ZarrWriter, atomic_write, open_node, open_zarr

NAME = 'café 日本語 Ω'


@pytest.fixture
def folder(tmp_path):
    path = tmp_path / NAME
    try:
        path.mkdir()
    except (OSError, UnicodeError):
        pytest.skip('this file system cannot hold the test directory name')
    return path


def only_entry(parent):
    # Nothing was written under a second, differently decoded name.
    return [p.name for p in parent.iterdir()] == [NAME]


def test_reads_a_local_array_and_group(folder):
    values = np.arange(4 * 5 * 6, dtype=np.uint8).reshape(4, 5, 6)
    array(folder / 'ct.zarr', values, attrs={'voxel': 7.91})
    for location in (folder / 'ct.zarr', str(folder / 'ct.zarr')):
        with open_zarr(location) as reader:
            assert np.array_equal(reader[...], values)
            assert reader.attrs['voxel'] == 7.91
    pyramid(folder / 'volume.zarr')
    group = open_node(folder / 'volume.zarr')
    assert sorted(group.keys()) == ['0', '1', '2']
    with open_zarr(folder / 'volume.zarr' / '1') as reader:
        assert (reader[...] == 2).all()


def test_missing_array_reports_the_name(folder):
    with pytest.raises(FileNotFoundError) as error:
        open_zarr(folder / 'absent.zarr')
    assert NAME in str(error.value)


def test_writer_writes_where_python_looks(folder):
    data = np.arange(30, dtype=np.uint16).reshape(5, 6)
    writer = ZarrWriter.create(folder / 'out.zarr', shape=(5, 6), chunks=(2, 3), dtype='u2', cname='zstd')
    writer[:] = data
    assert json.loads((folder / 'out.zarr' / '.zarray').read_text())['chunks'] == [2, 3]
    assert sorted(p.name for p in (folder / 'out.zarr').iterdir() if not p.name.startswith('.')) == \
        ['0.0', '0.1', '1.0', '1.1', '2.0', '2.1']
    assert np.array_equal(open_zarr(folder / 'out.zarr')[:], data)
    assert np.array_equal(ZarrWriter.open(folder / 'out.zarr')[:], data)
    atomic_write(folder / 'a' / 'file.bin', b'bytes')
    assert (folder / 'a' / 'file.bin').read_bytes() == b'bytes'
    assert only_entry(folder.parent)


def test_tifxyz_and_images(folder):
    r, c = np.meshgrid(np.arange(20, dtype=np.float32), np.arange(30, dtype=np.float32), indexing='ij')
    grid = np.stack([c * 2 + 10, r * 3 + 20, r + c], -1).astype(np.float32)
    surface = tifxyz.Surface(grid, scale=(.05, .1))
    tifxyz.write(folder / 'surface', surface, tile=16)
    assert np.array_equal(tifffile.imread(folder / 'surface' / 'x.tif'), grid[..., 0])
    back = tifxyz.read(folder / 'surface')
    assert np.array_equal(back.xyz, grid) and back.uuid == surface.uuid

    image = np.arange(64 * 48, dtype=np.uint16).reshape(64, 48)
    images.write_tiff(folder / 'a.tif', image, tile=(32, 32))
    assert np.array_equal(tifffile.imread(folder / 'a.tif'), image)
    assert np.array_equal(images.read(folder / 'a.tif'), image)
    with images.open(folder / 'a.tif') as reader:
        assert np.array_equal(reader[10:20, 5:9], image[10:20, 5:9])
    assert only_entry(folder.parent)
