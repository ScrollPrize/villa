"""Per-crop z-score CT inputs, init/resume/inference persistence and the calibration guard."""
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.data import ct_normalization as norm
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolumeSpec, FiberVolume
from vesuvius.neural_tracing.fiber_follow.data.observations import augment_image_pair, image_crop
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec


def record(spec, center=50., noise=4.):
    return dict(method=norm.METHOD, volume=norm.volume_key(spec), center=center, noise=noise,
                threshold=center+3*noise)


def volume(tmp_path):
    root = tmp_path/'ct'/'0'; root.mkdir(parents=True)
    meta = dict(zarr_format=2, shape=[48,16,16], chunks=[16,16,16], dtype='|u1',
                compressor=None, fill_value=0, order='C', filters=None)
    (root/'.zarray').write_text(json.dumps(meta))
    rng = np.random.default_rng(72)
    for z, level in enumerate((50, 51, 120)):
        raw = np.clip(rng.normal(level, 3 if z < 2 else 20, (16,)*3), 0, 255).astype(np.uint8)
        (root/f'{z}.0.0').write_bytes(raw.tobytes())
    return FiberVolumeSpec('', ct_zarr=str(root.parent), ct_level=0, inputs='ct',
                           load_presence=False, grid_scale=1., ct_grid_scale=1.)


def test_zscore_uses_all_pixels_without_clipping_or_background_sentinel():
    image = np.arange(512, dtype=np.float32).reshape(8, 8, 8)/512
    image[0, 0, 0] = 100.
    expected = (image.astype(np.float64)-image.mean(dtype=np.float64))/image.std(dtype=np.float64)
    norm.normalize_ct(image, dict(method=norm.ZSCORE_METHOD))
    np.testing.assert_allclose(image, expected, atol=2e-6)
    assert abs(image.mean()) < 1e-6 and abs(image.std()-1) < 1e-6
    assert image.max() > 4
    for value in (0., .37):
        constant = np.full((8, 8, 8), value, np.float32)
        norm.normalize_ct(constant, dict(method=norm.ZSCORE_METHOD))
        assert np.isfinite(constant).all() and not constant.any()
    # Augmentation never masks values below the old -4 sentinel.
    image = np.linspace(-8, 8, 64, dtype=np.float32).reshape(1, 4, 4, 4)
    before = image.copy()
    augment_image_pair(torch.from_numpy(image), (1., 0., 0.), np.random.default_rng(1), zscore=True)
    np.testing.assert_array_equal(image, before)


def test_zscore_init_resume_and_inference_reuse_the_checkpoint_policy(tmp_path, monkeypatch):
    spec = volume(tmp_path)
    crop, items = CropSpec(8, 8, 4, 1.), [dict(pos=np.array([8., 8., 36.]), frame=np.eye(3))]
    with pytest.raises(ValueError, match='calibration is required'):
        image_crop(items, FiberVolume(spec), crop, input_mode='ct')
    key = norm.volume_key(spec)
    document = dict(method=norm.ZSCORE_METHOD, volumes={key: dict(
        method=norm.ZSCORE_METHOD, volume=key, epsilon=norm.ZSCORE_EPSILON,
        shape=[48, 16, 16], chunks=[16, 16, 16], dtype='|u1')})
    monkeypatch.setattr(norm, 'calibrate', lambda *a, **kw: pytest.fail('Z-score must not calibrate'))
    # Init weights and resume take the checkpoint document exactly; a lost JSON is restored from it.
    out = tmp_path/'run'
    assert norm.prepare_normalization(out, [spec], resume=document) == document
    saved = (out/'ct_normalization.json').read_bytes()
    (out/'ct_normalization.json').unlink()
    assert norm.prepare_normalization(out, [spec], resume=document) == document
    assert (out/'ct_normalization.json').read_bytes() == saved
    assert pickle.loads(pickle.dumps(spec)).ct_normalization == spec.ct_normalization
    image = image_crop(items, FiberVolume(spec), crop, input_mode='ct')
    assert abs(float(image.mean())) < 1e-5
    assert abs(float(image.std(correction=0))-1) < 1e-5
    bad = copy.deepcopy(document)
    bad['volumes'][key]['epsilon'] = 1e-3
    (out/'ct_normalization.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='differs'):
        norm.prepare_normalization(out, [spec], resume=document)
    with pytest.raises(ValueError, match='differs from the checkpoint'):
        norm.prepare_normalization(out, [spec], known=document)
    # A new inference volume gets its own record under the checkpoint's policy.
    second = volume(tmp_path/'second')
    inferred = norm.prepare_normalization(tmp_path/'infer', [second], known=document)
    assert inferred['method'] == second.ct_normalization['method'] == norm.ZSCORE_METHOD
    assert norm.volume_key(second) in inferred['volumes'] and norm.volume_key(second) not in document['volumes']
    with pytest.raises(ValueError, match='different volume'):
        norm.validate_record(spec.ct_normalization, second)
    meta_path = Path(second.ct_zarr)/'0'/'.zarray'
    meta = json.loads(meta_path.read_text()); meta['shape'][0] += 1
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='metadata changed'):
        norm.prepare_normalization(tmp_path/'infer', [second])
