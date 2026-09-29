"""Uploading an Automated Fiber Volume to the Spiral service and loading it
next to the native fibers."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from afv_fixture import BASE_SHAPE_ZYX, fiber, write_afv
from fit_session import resolve_dataset_root
from spiral_helpers import load_fiber_point_collections
from spiral_service import ApiError, ServiceState, bind_service_paths
from test_spiral_service_v2 import _write_scroll_spec


@pytest.fixture
def small_afv(tmp_path):
    return write_afv(tmp_path / 'fibers.afv', [fiber(100), fiber(200, tag='V'), fiber(300)])


@pytest.fixture
def state(tmp_path):
    root = tmp_path / 'dataset'
    root.mkdir()
    _write_scroll_spec(root)
    (root / 'umbilicus.json').write_text('{}')
    (root / 'fibers').mkdir()
    output = tmp_path / 'output'
    output.mkdir()
    resolution = bind_service_paths(resolve_dataset_root(root), output, tmp_path / 'cache')
    instance = ServiceState(dataset_root=str(root), dataset_resolution=resolution,
                            startup_run={'z_begin': 0, 'z_end': 10})
    yield instance
    instance.close()


def manifest(data):
    return {'kind': 'afv', 'id': 'fibers.afv', 'files': [
        {'name': 'fibers.afv', 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}]}


def upload(state, data):
    started = state.begin_upload(manifest(data))
    if started.get('deduplicated'):
        return started['input']
    uid = started['upload_id']
    state.receive_upload_file(uid, 'fibers.afv', io.BytesIO(data), len(data))
    return state.finalize_upload(uid)['input']


def test_upload_is_reused_and_selected(state, small_afv):
    data = small_afv.read_bytes()
    record = upload(state, data)
    published = Path(record['path'])
    assert record['kind'] == 'afv'
    assert published.suffix == '.afv'
    assert published.read_bytes() == data
    assert state.begin_upload(manifest(data))['deduplicated']
    restarted = ServiceState(dataset_root=state.dataset_root,
                             dataset_resolution=state.dataset_resolution)
    try:
        assert restarted.begin_upload(manifest(data))['input']['path'] == str(published)
    finally:
        restarted.close()
    resolved = state._dataset_session_request({'paths': {'automated_fiber_volume': str(published)},
                                               'run': {'config': {'input_use_fibers': True}}})
    assert resolved['paths']['automated_fiber_volume'] == str(published)
    assert resolved['paths']['fibers'] == state.dataset_resolution.to_dict()['resolved']['fibers']
    loaded, next_id = load_fiber_point_collections(None, 1, base_shape_zyx=BASE_SHAPE_ZYX,
                                                   automated_fiber_volume=str(published))
    assert len(loaded) == 3 and next_id == 4
    assert all(pcl['metadata']['read_only'] for pcl in loaded.values())


def test_rejects_unuploaded_path(state, small_afv):
    with pytest.raises(ApiError):
        state._dataset_session_request({'paths': {'automated_fiber_volume': str(small_afv)},
                                        'run': {'config': {'input_use_fibers': True}}})


def test_rejects_invalid_container_and_digest(state, small_afv):
    with pytest.raises(ApiError, match='AFV'):
        upload(state, b'not a fiber database')
    data = small_afv.read_bytes()
    uid = state.begin_upload(manifest(data))['upload_id']
    bad = data[:-1] + bytes([data[-1] ^ 1])
    with pytest.raises(ApiError, match='(?i)(digest|sha-?256|hash)'):
        state.receive_upload_file(uid, 'fibers.afv', io.BytesIO(bad), len(bad))
        state.finalize_upload(uid)


def test_native_dataset_path_is_unchanged(state):
    expected = state.dataset_resolution.to_dict()['resolved']['fibers']
    request = state._dataset_session_request({'paths': {}, 'run': {'config': {'input_use_fibers': True}}})
    assert request['paths']['fibers'] == expected


def test_omitting_afv_keeps_classic_fibers(state):
    request = state._dataset_session_request({'paths': {}, 'run': {'config': {'input_use_fibers': True}}})
    assert request['paths']['automated_fiber_volume'] == ''
    assert request['paths']['fibers'] == state.dataset_resolution.to_dict()['resolved']['fibers']


def test_native_and_afv_fibers_load_together_and_separately(small_afv, tmp_path):
    native_dir = tmp_path / 'native'
    native_dir.mkdir()
    (native_dir / 'user-fiber.json').write_text(
        json.dumps(dict(fiber(250), coordinate_base_shape_zyx=BASE_SHAPE_ZYX)))
    native, native_next = load_fiber_point_collections(str(native_dir), 7, base_shape_zyx=BASE_SHAPE_ZYX)
    combined, combined_next = load_fiber_point_collections(
        str(native_dir), 7, base_shape_zyx=BASE_SHAPE_ZYX, automated_fiber_volume=str(small_afv))
    assert len(native) == 1 and len(combined) == 4
    assert native_next == 8 and combined_next == 11
    np.testing.assert_array_equal([p['p'] for p in combined[7]['points'].values()],
                                  [p['p'] for p in native[7]['points'].values()])
    assert combined[7]['source_file'] == native[7]['source_file']
    assert combined[7]['metadata'] == native[7]['metadata']
    assert sum(p['metadata'].get('read_only', False) for p in combined.values()) == 3
    standalone, _ = load_fiber_point_collections(None, 7, base_shape_zyx=BASE_SHAPE_ZYX,
                                                 automated_fiber_volume=str(small_afv))
    assert len(standalone) == 3


def test_native_fiber_directory_ignores_afv_files(small_afv, tmp_path):
    native_dir = tmp_path / 'native'
    native_dir.mkdir()
    (native_dir / 'fibers.afv').write_bytes(small_afv.read_bytes())
    loaded, next_id = load_fiber_point_collections(str(native_dir), 1, base_shape_zyx=BASE_SHAPE_ZYX)
    assert loaded == {} and next_id == 1
