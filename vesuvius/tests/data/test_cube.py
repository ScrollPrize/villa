"""Cube API compatibility, NRRD order and download-cache lifecycle."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import nrrd
import numpy as np
import pytest

from vesuvius import Cube, list_cubes
from vesuvius.data.volume import Cube as LegacyCube
from vesuvius.data import cube as cube_module
from vc_reader.reader import ObjectStore


def test_inventory_and_legacy_import():
    assert LegacyCube is Cube
    from vesuvius.utils import list_cubes as utility_list
    assert utility_list is list_cubes
    assert '02256_02512_04816' in list_cubes()[1][54][7.91]


@pytest.fixture
def cube_source(tmp_path, monkeypatch):
    origin = '00001_00002_00003'
    volume = np.arange(60, dtype=np.uint16).reshape(3, 4, 5)
    labels = (volume % 7).astype(np.uint8)
    source = tmp_path / 'source'
    source.mkdir()
    for kind, data in [('volume', volume), ('mask', labels)]:
        nrrd.write(str(source / f'{origin}_{kind}.nrrd'), data, index_order='F')
    handler = partial(SimpleHTTPRequestHandler, directory=str(source))
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    monkeypatch.setattr(cube_module, 'list_cubes', lambda: {1: {54: {7.91: {origin: base}}}})
    try:
        yield volume, labels
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def open_cube(**kwargs):
    return Cube(1, 54, 7.91, 1, 2, 3, **kwargs)


def test_uncached_pair_order_and_temporary_cleanup(cube_source, monkeypatch):
    downloaded = []
    original = ObjectStore.read_file

    def record(self, name, path):
        downloaded.append(Path(path))
        return original(self, name, path)

    monkeypatch.setattr(ObjectStore, 'read_file', record)
    cube = open_cube()
    volume, labels = cube_source
    np.testing.assert_array_equal(cube[:, :, :][0], volume)
    np.testing.assert_array_equal(cube[1, :, 2][1], labels[1, :, 2])
    assert cube.volume.dtype == volume.dtype
    assert cube.mask.dtype == labels.dtype
    assert len(downloaded) == 2
    assert all(not path.parent.exists() for path in downloaded)
    with pytest.raises(IndexError, match='three elements'):
        cube[0]


def test_cache_reuse_and_normalization(cube_source, tmp_path, monkeypatch):
    cached = open_cube(cache=True, cache_dir=tmp_path / 'cache')

    def no_download(*args, **kwargs):
        pytest.fail('Cached cube should not download again')

    monkeypatch.setattr(ObjectStore, 'read_file', no_download)
    normalized = open_cube(cache=True, cache_dir=tmp_path / 'cache', normalize=True)
    np.testing.assert_array_equal(normalized[:, :, :][0], cached.volume / 65535)
    np.testing.assert_array_equal(normalized[:, :, :][1], cached.mask)


def test_cache_toggles(cube_source, tmp_path):
    cube = open_cube()
    cube.activate_caching(tmp_path / 'cache')
    assert cube.cache
    assert len(list((tmp_path / 'cache').rglob('*.nrrd'))) == 2
    cube.deactivate_caching()
    assert not cube.cache
    np.testing.assert_array_equal(cube[:, :, :][0], cube_source[0])
    np.testing.assert_array_equal(cube[:, :, :][1], cube_source[1])


def test_unknown_cube():
    with pytest.raises(ValueError, match='Cube not found'):
        Cube(-1, 54, 7.91, 0, 0, 0)
