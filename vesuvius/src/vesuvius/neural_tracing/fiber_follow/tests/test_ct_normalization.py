"""Default CT calibration, restart persistence, masks, and shared crop semantics."""
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.shared import ct_normalization as norm
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolumeSpec, FiberVolume
from vesuvius.neural_tracing.fiber_follow.regression.data import augment_image_pair, image_crop
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


def test_resume_reuses_exact_json_and_checkpoint_without_calibrating(tmp_path, monkeypatch):
    spec = volume(tmp_path)
    out = tmp_path/'run'
    document = norm.prepare_normalization(out, [spec])
    assert 47 <= spec.ct_normalization['center'] <= 53
    saved = (out/'ct_normalization.json').read_bytes()
    monkeypatch.setattr(norm, 'calibrate', lambda *a, **kw: pytest.fail('Recomputed calibration on resume'))
    assert norm.prepare_normalization(out, [spec], resume=document) == document
    assert (out/'ct_normalization.json').read_bytes() == saved
    (out/'ct_normalization.json').unlink()
    assert norm.prepare_normalization(out, [spec], resume=document) == document
    assert (out/'ct_normalization.json').read_bytes() == saved
    assert pickle.loads(pickle.dumps(spec)).ct_normalization == spec.ct_normalization
    bad = copy.deepcopy(document)
    bad['volumes'][norm.volume_key(spec)]['noise'] += 1
    (out/'ct_normalization.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='differs'):
        norm.prepare_normalization(out, [spec], resume=document)


def test_new_inference_volume_gets_own_estimate_and_metadata_changes_fail(tmp_path):
    spec = volume(tmp_path)
    known = norm.prepare_normalization(tmp_path/'train', [spec])
    second = volume(tmp_path/'second')
    infer = norm.prepare_normalization(tmp_path/'infer', [second], known=known)
    assert norm.volume_key(second) in infer['volumes']
    assert norm.volume_key(second) not in known['volumes']
    with pytest.raises(ValueError, match='different volume'):
        norm.validate_record(spec.ct_normalization, second)
    meta_path = Path(second.ct_zarr)/'0'/'.zarray'
    meta = json.loads(meta_path.read_text()); meta['shape'][0] += 1
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='metadata changed'):
        norm.prepare_normalization(tmp_path/'infer', [second])


def test_continuous_foreground_threshold_and_robust_statistics():
    rng = np.random.default_rng(83)
    image = rng.uniform(0, 1, (24, 20, 20)).astype(np.float32)
    r = dict(threshold=62., noise=4.)
    image[0,0,:6] = [0., np.nan, np.inf, 62/255, 61.999/255, 62.001/255]
    original = image.copy()
    sample = original[::4, ::4, ::4]
    bins = np.floor(sample[np.isfinite(sample) & (sample > np.float32(62/255))]*255+.5)
    center = np.quantile(bins, .5, method='inverted_cdf')
    scale = max(8., 1.4826*np.quantile(np.abs(bins-center), .5, method='inverted_cdf'))
    expected = np.where(np.isfinite(original) & (original > np.float32(62/255)),
                        np.clip((original*255-center)/scale, -4, 4), -4)
    norm.normalize_ct(image, r)
    np.testing.assert_allclose(image, expected, atol=2e-6)
    assert np.all(image[0,0,:5] == -4)
    assert image[0,0,5] > -4


def test_empty_and_thin_foreground_and_bright_spikes_are_safe():
    r = dict(threshold=62., noise=4.)
    for value in (0., 50/255, np.nan):
        image = np.full((8, 9, 9), value, np.float32)
        norm.normalize_ct(image, r)
        assert np.all(image == -4)
    image = np.full((8, 9, 9), 50/255, np.float32)
    image[1,1,1] = 100/255  # Misses the sampled grid: full foreground fallback.
    norm.normalize_ct(image, r)
    assert abs(image[1,1,1]) < 1e-6
    assert np.count_nonzero(image > -4) == 1
    base = np.full((24,24,24), 100/255, np.float32)
    spikes = base.copy(); spikes.ravel()[::101] = 1.
    norm.normalize_ct(base, r); norm.normalize_ct(spikes, r)
    np.testing.assert_allclose(spikes.ravel()[1:101], base.ravel()[1:101], atol=1e-6)


def test_scalar_sampler_normalizes_ct_and_requires_calibration(tmp_path):
    spec = volume(tmp_path)
    vol = FiberVolume(spec)
    crop = CropSpec(depth=8, width=8, behind=4, spacing=1.)
    items = [dict(pos=np.array([8.,8.,36.]), frame=np.eye(3))]
    with pytest.raises(ValueError, match='calibration is required'):
        image_crop(items, vol, crop, input_mode='ct')
    norm.prepare_normalization(tmp_path/'run', [spec])
    image = image_crop(items, vol, crop, input_mode='ct')
    assert torch.isfinite(image).all() and image.min() < 0 and image.max() > 0
    items[0]['pos'][:] = -100
    assert torch.all(image_crop(items, vol, crop, input_mode='ct') == -4)


def test_augmentations_preserve_black_mask_and_auxiliary_channels():
    image = torch.zeros(8, 16, 16, 16)
    image[0] = -4; image[0,4:12,4:12,4:12] = .5
    image[1:] = torch.rand_like(image[1:])
    background = image[0] == -4; directions = image[2:].clone()
    augment_image_pair(image, (1.3, .1, .07), np.random.default_rng(13), blur_sigma=1., drop_presence=True)
    assert torch.all(image[0][background] == -4)
    assert image[0][~background].std() > 0
    assert image[1].count_nonzero() == 0
    assert torch.equal(image[2:], directions)
