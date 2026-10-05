"""Model crops from another CT level (FiberVolumeSpec.crop_ct_level) while frames/seed headings keep ct_level."""
import json

import numpy as np
import torch

from model_fixtures import array_at
from test_ct_crops import config, item
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume, FiberVolumeSpec, model_crop_volume
from vesuvius.neural_tracing.fiber_follow.data.observations import image_crop
from vesuvius.neural_tracing.fiber_follow.data.datasets import ct_source_spec, same_sources
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec


def two_levels(root):
    z, y, x = np.indices((80, 80, 80))
    fine = (x+2*y+z).clip(0, 255).astype(np.uint8)
    array_at(root/'ct/0', fine)
    array_at(root/'ct/1', fine.reshape(40, 2, 40, 2, 40, 2).mean((1, 3, 5)).astype(np.uint8))
    from test_ct_normalization import record
    split = FiberVolumeSpec('', ct_zarr=str(root/'ct'), ct_level=0, ct_grid_scale=4., inputs='ct', load_presence=False,
                            crop_ct_level=1, crop_ct_grid_scale=8.)
    split.ct_normalization = record(split)
    coarse = FiberVolumeSpec('', ct_zarr=str(root/'ct'), ct_level=1, ct_grid_scale=8., inputs='ct', load_presence=False)
    coarse.ct_normalization = record(coarse)
    return FiberVolume(split, cache_bytes=1 << 20), FiberVolume(coarse, cache_bytes=1 << 20)


def test_model_crops_read_the_crop_level_and_frames_keep_the_main_level(tmp_path):
    split, coarse = two_levels(tmp_path)
    assert split.input_scale == 2 and split.ct.shape == (80, 80, 80)  # frames, seed headings: level 0
    view = model_crop_volume(split)
    assert view is split.crop_view and view.input_scale == 1 and view.ct.shape == (40, 40, 40)
    assert model_crop_volume(coarse) is coarse
    cfg = config(fine=CropSpec(depth=16, width=12, behind=7, spacing=1.))
    state = item(cfg)
    # Exactly the crop a level-1-only volume gives.
    torch.testing.assert_close(image_crop([state], split, cfg.fine), image_crop([state], coarse, cfg.fine), rtol=0, atol=0)


def test_crop_level_is_optional_metadata_and_never_changes_holdout_identity(tmp_path):
    single = FiberVolumeSpec('', ct_zarr='ct', ct_level=0)
    assert 'crop_ct_level' not in single.to_dict()
    split = FiberVolumeSpec('', ct_zarr='ct', ct_level=0, crop_ct_level=1, crop_ct_grid_scale=2.)
    assert FiberVolumeSpec(**split.to_dict()) == split
    source = dict(name='a', kind='afv', ct='s3://x', grid_scale=2., ct_level=0, ct_grid_scale=1., validation={})
    assert ct_source_spec(dict(source, crop_ct_level=1, crop_ct_grid_scale=2.), None).crop_ct_level == 1
    assert ct_source_spec(source, None).crop_ct_level is None
    assert same_sources(dict(sources=[source]), dict(sources=[dict(source, crop_ct_level=1, crop_ct_grid_scale=2.)]))
    assert not same_sources(dict(sources=[source]), dict(sources=[dict(source, validation={'seed': 1})]))


def test_level1_dataset_config_moves_only_model_crops():
    from pathlib import Path
    root = Path(__file__).parents[1]/'configs'
    base = json.loads((root/'mixed_ct_datasets_paris50.json').read_text())
    level1 = json.loads((root/'mixed_ct_datasets_paris50_level1.json').read_text())
    for a, b in zip(base['sources'], level1['sources']):
        assert {k: v for k, v in b.items() if not k.startswith('crop_ct_')} == a
        assert b['crop_ct_level'] == a.get('ct_level', 0)+1
        assert b['crop_ct_grid_scale'] == 2*a.get('ct_grid_scale', 4. if a['kind'] == 'paris4' else 1.)
