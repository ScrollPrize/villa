import json
import pickle

import numpy as np
import pytest
import tifffile

from build_support import load_build
load_build()
from vc_reader import tifxyz


def _grid(rows=20, cols=30):
    r, c = np.meshgrid(np.arange(rows, dtype=np.float32), np.arange(cols, dtype=np.float32), indexing='ij')
    return np.stack([c * 2 + 10, r * 3 + 20, 30 + np.sin(r / 3) * np.cos(c / 4)], -1).astype(np.float32)


def test_round_trip_is_readable_by_tifffile(tmp_path):
    surface = tifxyz.Surface(_grid(), scale=(.05, .1), extra={'name': 'a'})
    tifxyz.write(tmp_path / 's', surface, tile=16)
    assert surface.uuid
    for axis, name in enumerate('xyz'):
        assert np.array_equal(tifffile.imread(tmp_path / 's' / f'{name}.tif'), _grid()[..., axis])
    assert (tifffile.imread(tmp_path / 's' / 'mask.tif') == 255).all()
    meta = json.loads((tmp_path / 's' / 'meta.json').read_text())
    assert meta['scale'] == [.1, .05] and meta['format'] == 'tifxyz' and meta['type'] == 'seg' and meta['name'] == 'a'
    back = tifxyz.read(tmp_path / 's')
    assert np.array_equal(back.xyz, surface.xyz) and back.valid.all() and back.valid.dtype == bool
    assert back.scale == (.05, .1) and back.uuid == surface.uuid and back.extra['name'] == 'a'
    assert back.full_shape == (400, 300)
    assert tifxyz.read_metadata(tmp_path / 's').uuid == surface.uuid
    assert not back.xyz.flags.writeable


def test_existing_directory_needs_overwrite(tmp_path):
    (tmp_path / 's').mkdir()
    with pytest.raises(FileExistsError):
        tifxyz.write(tmp_path / 's', tifxyz.Surface(_grid()))
    tifxyz.write(tmp_path / 's', tifxyz.Surface(_grid()), overwrite=True)
    assert tifxyz.read(tmp_path / 's').shape == (20, 30)


def test_reads_other_writers_and_mask(tmp_path):
    path = tmp_path / 's'
    path.mkdir()
    grid = _grid(40, 40)
    for axis, name in enumerate('xyz'):
        tifffile.imwrite(path / f'{name}.tif', grid[..., axis], tile=(16, 16), compression='zlib')
    mask = np.ones((40, 40), np.uint8)
    mask[3, 4] = 0
    tifffile.imwrite(path / 'mask.tif', mask)
    (path / 'meta.json').write_text(json.dumps({'uuid': 'u', 'scale': [1, 1], 'bbox': [[-1, -1, -1], [-1, -1, -1]]}))
    surface = tifxyz.read(path)
    assert not surface.valid[3, 4] and surface.valid.sum() == 1599
    assert (surface.xyz[3, 4] == -1).all()
    assert surface.bbox[0] == grid[..., 0].min()
    assert tifxyz.read(path, mask=False).valid.all()
    pages = np.ones((2, 40, 40), np.uint8)
    pages[0, 7, 8] = 0
    tifffile.imwrite(path / 'mask.tif', pages)
    assert tifxyz.read(path).valid.sum() == 1599 and not tifxyz.read(path).valid[7, 8]
    finer = np.full((80, 80), 255, np.uint8)
    finer[3, 5] = 0
    tifffile.imwrite(path / 'mask.tif', finer)
    assert tifxyz.read(path).valid.sum() == 1599 and not tifxyz.read(path).valid[1, 2]
    tifffile.imwrite(path / 'mask.tif', np.ones((5, 5), np.uint8))
    with pytest.raises(ValueError, match='mask shape'):
        tifxyz.read(path)


def test_list_warns_on_bad_metadata(tmp_path):
    tifxyz.write(tmp_path / 'good', tifxyz.Surface(_grid(), uuid='g'))
    tifxyz.write(tmp_path / 'bad', tifxyz.Surface(_grid(), uuid='b'))
    (tmp_path / 'bad' / 'meta.json').write_text('{"scale": []}')
    with pytest.warns(UserWarning, match='scale'):
        entries = tifxyz.list_surfaces(tmp_path)
    assert [e.metadata.uuid for e in entries] == ['g']


def test_smooth_is_catmull_rom_inside():
    grid = _grid()
    surface = tifxyz.Surface(grid)
    rows, cols = np.meshgrid(np.array([5.25, 7.5], np.float32), np.array([6.5, 11.75], np.float32), indexing='ij')
    xyz, valid = surface.points_at(rows, cols, interpolation='smooth')
    assert valid.all()
    def spline(t):
        return np.array([-.5 * t + t * t - .5 * t ** 3, 1 - 2.5 * t * t + 1.5 * t ** 3, .5 * t + 2 * t * t - 1.5 * t ** 3, -.5 * t * t + .5 * t ** 3])
    for j in range(2):
        for i in range(2):
            r, c = float(rows[j, i]), float(cols[j, i])
            r0, c0 = int(r), int(c)
            expected = spline(r - r0) @ grid[r0 - 1:r0 + 3, c0 - 1:c0 + 3, 2].astype(np.float64) @ spline(c - c0)
            assert xyz[j, i, 2] == pytest.approx(expected, abs=2e-4)


def test_resample_matches_points_and_marks_missing():
    grid = _grid()
    grid[5, 5] = -1
    surface = tifxyz.Surface(grid)
    xyz, valid = surface.resample((4, 4), (.5, .5), (5, 5))
    rows, cols = np.meshgrid(4 + .5 * np.arange(5, dtype=np.float32), 4 + .5 * np.arange(5, dtype=np.float32), indexing='ij')
    other, valid2 = surface.points_at(rows, cols)
    assert np.array_equal(valid, valid2) and np.array_equal(xyz[valid], other[valid2])
    assert not valid[2, 2] and np.isnan(xyz[2, 2]).all()
    _, _, normals = surface.resample((0, 0), (1, 1), (3, 3), normals=True)
    assert tifxyz.Surface(grid, mask=np.ones(grid.shape[:2], bool)).valid[0, 0]
    assert np.allclose(np.linalg.norm(normals, axis=-1), 1)


def test_retarget_and_pickle(tmp_path):
    surface = tifxyz.Surface(_grid(), scale=(.05, .05), area=3.0, uuid='u', extra={'k': [1, 2]})
    half = surface.retarget(2.0)
    assert half.scale == (.1, .1) and half.area is None
    assert np.array_equal(half.xyz, surface.xyz / np.float32(2))
    assert half.bbox == half.bounds()
    again = pickle.loads(pickle.dumps(surface))
    assert np.array_equal(again.xyz, surface.xyz) and again.scale == surface.scale and again.extra == {'k': [1, 2]} and again.uuid == 'u'


def test_resample_full_is_scaled_stored_positions():
    surface = tifxyz.Surface(_grid(), scale=(.5, .25))
    assert surface.full_shape == (40, 120)
    xyz, valid = surface.resample_full((2, 8), (3, 4))
    other, _ = surface.resample((1, 2), (.5, .25), (3, 4))
    assert valid.all() and np.array_equal(xyz, other)
