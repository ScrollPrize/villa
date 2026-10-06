"""CT-only crop reads: shared geometry read once, worker shared storage, coordinate dtype rounding."""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import torch

from model_fixtures import array_at, config as small_config
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume, FiberVolumeSpec
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, frame_from_heading
from vesuvius.neural_tracing.fiber_follow.data.observations import image_crop, ObservationBuilder


def volume(root):
    z, y, x = np.indices((80, 80, 80))
    array_at(root/'ct/0', (x+2*y+z).clip(0, 255).astype(np.uint8))
    from test_ct_normalization import record
    spec = FiberVolumeSpec('', ct_zarr=str(root/'ct'), ct_level=0, ct_grid_scale=4., inputs='ct', load_presence=False)
    spec.ct_normalization = record(spec)
    return FiberVolume(spec, cache_bytes=1 << 20)


def config(**kwargs):
    return small_config(**dict(dict(fine=CropSpec(depth=16, width=12, behind=7, spacing=.5), n_future=4, n_history=8),
                               **kwargs))


def item(cfg):
    frame = frame_from_heading(np.array([.3, .4, .8660254]))
    angle = .67
    c, s = np.cos(angle), np.sin(angle)
    frame = frame@np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return dict(pos=np.array([20., 20., 20.]), frame=frame, hist_local=np.zeros((cfg.n_history, 3)),
                hmask=np.zeros(cfg.n_history), seed_valid=True, seed_pos=np.array([19., 20., 19.]),
                seed_tangent=np.array([.8, 0, .6]), seed_age=2.,
                observed_path=np.array([[19., 20., 19.], [20., 20., 19.5], [20., 20., 20.]]))


def test_shared_crop_geometry_is_sampled_once_with_independent_outputs(tmp_path):
    vol = volume(tmp_path)
    assert vol.presence is None
    crop = CropSpec(depth=7, width=5, behind=3, spacing=.5)
    a = dict(pos=np.array([20., 20., 20.]), frame=np.eye(3))
    b = dict(pos=a['pos'].copy(), frame=frame_from_heading(np.array([.4, .3, .8])))
    items = [a, dict(pos=a['pos'].copy(), frame=a['frame'].copy()), b, a]
    expected = torch.cat([image_crop([item], vol, crop) for item in items])
    assert expected.shape[1] == 1
    with patch.object(vol.ct, 'read', wraps=vol.ct.read) as ct_read, ThreadPoolExecutor(2) as pool:
        actual = image_crop(items, vol, crop, pool)
        assert ct_read.call_count == 2
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual[0].zero_()
    torch.testing.assert_close(actual[1:], expected[1:], rtol=0, atol=0)


def test_worker_images_keep_shared_storage_and_identical_values(tmp_path):
    cfg = config()
    vol = volume(tmp_path)
    builder = ObservationBuilder(cfg)
    state = item(cfg)
    expected = builder.images([state], vol)
    assert expected['fine'].shape[1] == 1
    with patch('torch.utils.data.get_worker_info', return_value=object()):
        actual = builder.images([state], vol)
    assert actual['fine'].is_shared()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_scalar_crop_reuse_preserves_coordinate_dtype_rounding(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.data.crop_sampling import scalar_crops
    vol = volume(tmp_path)
    vol.input_scale = 3.
    crop = CropSpec(depth=3, width=3, behind=1, spacing=.5)
    pos = np.array([20.1, 20.2, 20.3], np.float32)
    frame = frame_from_heading(np.array([.3, .4, .8])).astype(np.float32)
    items = [dict(pos=pos, frame=frame), dict(pos=pos.astype(np.float64), frame=frame.astype(np.float64))]
    expected = torch.cat([scalar_crops([state], vol, crop) for state in items])
    assert not torch.equal(expected[0], expected[1])
    torch.testing.assert_close(scalar_crops(items, vol, crop), expected, rtol=0, atol=0)
