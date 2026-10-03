"""Spiral reads Automated Fiber Volume fibers through the native fiber loader."""
import json
import sqlite3

import numpy as np
import pytest

from afv_fixture import BASE_SHAPE_ZYX, fiber, write_afv
from afv_input import iter_afv_fibers, validate_afv_container
from spiral_helpers import (
    _fiber_data_point_collection, load_fiber_point_collection, load_fiber_point_collections)


@pytest.fixture
def afv(tmp_path):
    return write_afv(tmp_path / 'fibers.afv', [fiber(100, count=300, step=0.8), fiber(200, tag='V'), fiber(300)])


def test_fibers_match_the_native_loader(afv, tmp_path):
    fibers = list(iter_afv_fibers(afv))
    assert len(fibers) == 3
    logical, data, origin = fibers[0]
    assert len(data['line_points']) == 300  # Assembled from two storage blocks.
    assert data['hv_classification']['manual_tag'] == 'H'
    path = tmp_path / 'native-fiber.json'
    path.write_text(json.dumps(data))
    native = load_fiber_point_collection(path, 7, base_shape_zyx=BASE_SHAPE_ZYX)
    direct = _fiber_data_point_collection(data, logical, 7, base_shape_zyx=BASE_SHAPE_ZYX)
    np.testing.assert_array_equal(
        [p['p'] for p in native['points'].values()],
        [p['p'] for p in direct['points'].values()])
    assert direct['metadata']['input_coordinate_scale'] == 1
    assert direct['metadata']['hv_classification'] == native['metadata']['hv_classification']
    assert origin['read_only'] is True
    with pytest.raises(ValueError, match='incompatible'):
        _fiber_data_point_collection(data, logical, 7, base_shape_zyx=[100, 100, 100])


def test_z_range_selects_fibers_and_is_validated(afv):
    selected = [data['line_points'][0][2] for _, data, _ in iter_afv_fibers(afv, z_range=(150, 250))]
    assert selected == [200]
    assert list(iter_afv_fibers(afv, z_range=(-100, -1))) == []
    with pytest.raises(ValueError, match='Invalid AFV z range'):
        next(iter_afv_fibers(afv, z_range=(100, 0)))


def test_fit_z_range_selects_fibers_on_a_downsampled_volume(afv):
    def heights(z_range):
        pcls, _ = load_fiber_point_collections(
            None, 0, base_shape_zyx=[100, 75, 75], z_range=z_range, automated_fiber_volume=afv)
        return [next(iter(pcl['points'].values()))['p'][2] for pcl in pcls.values()]
    assert heights((40, 60)) == [50]  # Native Z=200 at a quarter of the AFV domain.
    assert heights((0, 100)) == [25, 50, 75]


def test_requires_the_coordinate_domain(tmp_path):
    path = write_afv(tmp_path / 'no-domain.afv', [fiber(100)], frame={'vc_open_data_coordinate_space': 'test'})
    with pytest.raises(ValueError, match='coordinate domain'):
        validate_afv_container(path)


def test_reads_the_coordinate_domain_from_root(tmp_path):
    path = write_afv(tmp_path / 'root.afv', [fiber(100)], frame={'vc_open_data_coordinate_space': 'test'},
                     root={'coordinate_base_shape_zyx': BASE_SHAPE_ZYX})
    assert validate_afv_container(path)['coordinate_base_shape_zyx'] == BASE_SHAPE_ZYX
    (_, data, _), = iter_afv_fibers(path)
    assert data['coordinate_base_shape_zyx'] == BASE_SHAPE_ZYX
    conflict = write_afv(tmp_path / 'conflict.afv', [fiber(100)], root={'coordinate_base_shape_zyx': [100, 75, 75]})
    with pytest.raises(ValueError, match='disagree'):
        validate_afv_container(conflict)


def test_rejects_an_unrelated_sqlite_database(tmp_path):
    path = tmp_path / 'unrelated.afv'
    sqlite3.connect(path).close()
    with pytest.raises(ValueError, match='Unsupported AFV'):
        next(iter_afv_fibers(path))
