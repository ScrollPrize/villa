"""Bicubic tifxyz voxelization, including the existing output formats."""
import sys

import numpy as np
import pytest
from vc_reader import tifxyz
from vc_reader.reader import open_zarr

from vesuvius.label_processing import voxelize_objs as voxelizer


def write_surface(path, *, scale=(1, 1), mask=None, curved=False):
    row, col = np.mgrid[:5, :5].astype(np.float32)
    z = 2 + 0.25 * col**2 if curved else np.full_like(row, 2)
    xyz = np.stack((col + 2, row + 2, z), axis=-1)
    tifxyz.write(path, tifxyz.Surface(xyz, scale=scale, mask=mask), tile=16)
    return path


def sample(path, transform=None, shape=(10, 10, 10)):
    return voxelizer.process_surface(path, True, True, transform, shape)


def test_discovery_and_stable_order(tmp_path):
    b = write_surface(tmp_path / 'b')
    a = write_surface(tmp_path / 'a')
    nested = write_surface(tmp_path / 'nested' / 'c')
    assert voxelizer.find_surfaces(a) == [str(a)]
    assert voxelizer.find_surfaces(tmp_path) == [str(a), str(b)]
    assert voxelizer.find_surfaces(tmp_path, True) == [str(a), str(b), str(nested)]


def test_plane_coverage_frames_and_uv_scale(tmp_path):
    ids, normals, frames = sample(write_surface(tmp_path / 'plane', scale=(.1, .2)))
    expected = np.zeros((10, 10, 10), bool)
    expected[2, 2:7, 2:7] = True
    np.testing.assert_array_equal(ids, np.flatnonzero(expected))
    np.testing.assert_allclose(normals, np.tile([0, 0, 1], (25, 1)))
    np.testing.assert_allclose(frames, np.tile(np.eye(3), (25, 1, 1)))


def test_bicubic_positions_and_tangents_are_used(tmp_path, monkeypatch):
    path = write_surface(tmp_path / 'curved', curved=True)
    surface = tifxyz.read(path)
    calls = []

    class CheckedSurface:
        xyz, valid, shape = surface.xyz, surface.valid, surface.shape

        def points_at(self, rows, cols, **kwargs):
            assert kwargs == {'interpolation': 'bicubic', 'tangents': True}
            result = surface.points_at(rows, cols, **kwargs)
            linear, _ = surface.points_at(rows, cols, interpolation='linear')
            calls.append(np.nanmax(np.abs(result[0] - linear)))
            return result

    monkeypatch.setattr(voxelizer.tifxyz, 'read', lambda _: CheckedSurface())
    ids, normals, frames = sample(path)
    assert max(calls) > .05
    assert len(np.unique(ids // 100)) > 1
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(frames @ frames.transpose(0, 2, 1),
                               np.tile(np.eye(3), (len(ids), 1, 1)), atol=1e-6)
    # At x=4 (stored column 2), dz/dx = 1 for this quadratic patch.
    index = np.flatnonzero(ids == np.ravel_multi_index((3, 4, 4), (10, 10, 10)))[0]
    np.testing.assert_allclose(normals[index], np.array([-1, 0, 1]) / np.sqrt(2), atol=1e-6)


def test_masks_empty_and_outside_are_not_clamped(tmp_path):
    mask = np.ones((5, 5), np.uint8)
    mask[2, 2] = 0
    ids, _, _ = sample(write_surface(tmp_path / 'hole', mask=mask))
    assert np.ravel_multi_index((2, 4, 4), (10, 10, 10)) not in ids
    assert len(ids) > 0
    assert not sample(write_surface(tmp_path / 'empty', mask=np.zeros_like(mask)))[0].size
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 3] = -20
    transform = {'matrix': matrix[:3], 'perm': (0, 1, 2)}
    assert not sample(tmp_path / 'hole', transform)[0].size


def test_affine_frames_use_transformed_derivatives(tmp_path):
    path = write_surface(tmp_path / 'plane')
    # x'=x, y'=y, z'=z+x; translation is irrelevant to the directions.
    matrix = np.eye(4, dtype=np.float32)
    matrix[2, 0] = 1
    matrix[1, 3] = 1
    ids, normals, frames = sample(path, {'matrix': matrix[:3], 'perm': (0, 1, 2)})
    assert len(ids)
    np.testing.assert_allclose(normals, np.tile(np.array([-1, 0, 1]) / np.sqrt(2), (len(ids), 1)), atol=1e-6)
    np.testing.assert_allclose(frames[:, 0], np.tile(np.array([1, 0, 1]) / np.sqrt(2), (len(ids), 1)), atol=1e-6)


@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('fmt', ['zarr', 'tifs'])
def test_cli_outputs(tmp_path, monkeypatch, workers, fmt):
    input_path = tmp_path / 'surfaces'
    write_surface(input_path / 'b')
    write_surface(input_path / 'a')
    output = tmp_path / 'labels'
    normals = tmp_path / 'normals'
    frames = tmp_path / 'frames'
    argv = ['voxelize_objs', str(input_path), '--spatial-shape', '10', '10', '10',
            '--output_path', str(output), '--format', fmt, '--num_workers', str(workers)]
    if fmt == 'zarr':
        argv += ['--output_normals', '--normals_output_path', str(normals),
                 '--output_surface_frame', '--surface_frame_output_path', str(frames),
                 '--chunk_size', '4']
    monkeypatch.setattr(sys, 'argv', argv)
    voxelizer.main()
    expected = np.zeros((10, 10, 10), np.uint16)
    expected[2, 2:7, 2:7] = 2
    if fmt == 'tifs':
        import tifffile
        files = sorted(output.glob('*.tif'))
        assert len(files) == 10
        np.testing.assert_array_equal(np.stack([tifffile.imread(p) for p in files]), expected)
    else:
        for level in range(voxelizer.PYRAMID_LEVELS):
            step = 2**level
            np.testing.assert_array_equal(open_zarr(output / str(level))[:], expected[::step, ::step, ::step])
        n = open_zarr(normals / '0')[:]
        f = open_zarr(frames / '0')[:]
        np.testing.assert_array_equal(n[expected != 0], np.tile([0, 0, 1], (25, 1)))
        np.testing.assert_array_equal(f[expected != 0], np.tile(np.eye(3), (25, 1, 1)))
        assert not np.any(n[expected == 0])


def test_label_overflow_is_rejected_before_sampling(tmp_path, monkeypatch):
    monkeypatch.setattr(voxelizer, 'find_surfaces', lambda *args: ['surface'] * 256)
    monkeypatch.setattr(sys, 'argv', ['voxelize_objs', str(tmp_path), '--spatial-shape', '2', '2', '2', '--label-dtype', 'uint8'])
    with pytest.raises(SystemExit, match='2'):
        voxelizer.main()
