import json

import numpy as np
import pytest
import tifffile

from preview_fiber_merge import CTReader, load_surface, place_on_grid, source_rms, surface_metrics, upsample_surface


def test_surface_validation_detects_twisted_quad_and_single_quad_island():
    xyz = np.array([[[10., 0, 0], [11, 0, 0]], [[10, 1, 0], [11, 1, 0]]])
    metrics = surface_metrics(xyz)
    assert metrics['valid_quads'] == metrics['single_quad_regions'] == 1
    assert metrics['opposing_triangle_quads'] == 0
    xyz[1, 1] = [10, 0, 0]
    assert surface_metrics(xyz)['opposing_triangle_quads'] == 1


def test_component_preview_keeps_every_part_and_preserves_gap(tmp_path):
    components = []
    for name, x in [('first', 0), ('second', 40)]:
        folder = tmp_path / name
        folder.mkdir()
        for axis, value in zip('xyz', [100+x, 200, 300]):
            tifffile.imwrite(folder / f'{axis}.tif', np.full((2, 3), value, dtype=np.float32))
        components.append({'path': name, 'origin': [x, 0], 'spacing': 10})
    xyz, origin, spacing = load_surface(tmp_path, {'components': components})
    assert xyz.shape == (2, 7, 3)
    np.testing.assert_equal(origin, [0, 0])
    assert spacing == 10
    assert (xyz[:, :3, 0] == 100).all()
    assert (xyz[:, 3] == -1).all()
    assert (xyz[:, 4:, 0] == 140).all()


def test_preview_interpolation_preserves_xyz_and_does_not_bridge_holes():
    y, x = np.mgrid[:3, :3].astype(float)
    xyz = np.stack([10+x, 20+y, 30+x+2*y], axis=-1)
    dense, valid = upsample_surface(xyz, 2)
    np.testing.assert_allclose(dense[1, 1], [10.5, 20.5, 31.5])
    assert valid.all()
    xyz[2, 2] = -1
    dense, valid = upsample_surface(xyz, 2)
    assert valid[:2].all()
    assert not valid[2:, 2:].any()
    assert np.all(dense[2:, 2:] == -1)


def test_common_grid_uses_origin_and_preserves_missing_values():
    xyz = np.full((2, 3, 3), 7.)
    xyz[0, 1] = -1
    placed = place_on_grid(xyz, np.array([10, 20]), np.array([0, 0]), (5, 5), 10)
    np.testing.assert_equal(placed[2:4, 1:4][1], xyz[1])
    assert np.isnan(placed[2, 2]).all()
    assert np.isnan(placed[0]).all()
    with pytest.raises(ValueError, match='lattice'):
        place_on_grid(xyz, np.array([11, 20]), np.array([0, 0]), (5, 5), 10)


def test_ct_xyz_order_scale_and_missing_chunks(tmp_path):
    metadata = {'shape': [4, 4, 4], 'chunks': [2, 2, 2], 'dtype': '|u1', 'order': 'C',
                'compressor': None, 'filters': None, 'zarr_format': 2}
    (tmp_path / '.zarray').write_text(json.dumps(metadata))
    np.arange(8, dtype=np.uint8).tofile(tmp_path / '0.0.0')
    reader = CTReader(tmp_path, .5)
    # XYZ [2,0,2] becomes ZYX [1,0,1], whose C-order value is 5.
    points = np.array([[2, 0, 2], [6, 6, 6], [-1, -1, -1], [20, 0, 0]], dtype=float)
    sampled = reader.sample(points)
    assert sampled[0] == 5
    assert np.isnan(sampled[1:]).all()
    assert reader.missing == {(1, 1, 1)}


def test_source_disagreement_counts_only_available_sources():
    stack = np.array([[[[0., 0, 0], [1, 2, 3]]], [[[0., 0, 4], [np.nan]*3]]])
    rms, count, mean = source_rms(stack)
    np.testing.assert_equal(count, [[2, 1]])
    np.testing.assert_allclose(rms, [[2, 0]])
    np.testing.assert_allclose(mean, [[[0, 0, 2], [1, 2, 3]]])
