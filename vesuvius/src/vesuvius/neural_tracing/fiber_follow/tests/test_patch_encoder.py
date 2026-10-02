"""Patch-token lattice: physical centres, sampling, history cues, stem layout and crop validity."""
from dataclasses import replace

import pytest
import torch

from model_fixtures import aligned_config
from vesuvius.neural_tracing.fiber_follow.models.model import DirectConfig, build_model
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, crop_local_grid


def test_token_lattice_centres_sampling_history_cues_and_stem_share_one_grid():
    cfg = aligned_config()
    model = build_model(cfg)
    encoder = model.encoder
    xyz = encoder.token_xyz.reshape(*cfg.token_shape, 3)
    expected = torch.tensor([1.5-(cfg.fine.width-1)/2, 1.5-(cfg.fine.width-1)/2, 1.5-cfg.fine.behind])*cfg.fine.spacing
    torch.testing.assert_close(xyz[0, 0, 0], expected)
    torch.testing.assert_close(xyz[1, 1, 1]-xyz[0, 0, 0], torch.full((3,), 4*cfg.fine.spacing))
    point = xyz[2, 1, 2].reshape(1, 1, 3)
    fields = xyz.permute(3, 0, 1, 2)[None]
    sampled, support = model.sample_local(fields, point)
    assert support.all()
    torch.testing.assert_close(sampled, point)
    sampled, support = model.sample_local(fields, point+10000)
    assert not support.any() and not sampled.any()
    references = torch.zeros(1, cfg.n_history+1, 3)
    references[:, -1] = point[:, 0]+.4*cfg.fine.spacing
    mask = torch.zeros(1, cfg.n_history+1, dtype=torch.bool)
    mask[:, -1] = True
    rendered = encoder.conditioning(references, mask)
    assert rendered[0, 2, 1, 2, 2] == 1 and rendered[..., 2].sum() == 1
    assert encoder.patch_projection.kernel_size == (6, 6, 6) and encoder.patch_projection.stride == (4, 4, 4)
    image = torch.randn(1, cfg.input_channels, cfg.fine.depth, cfg.fine.width, cfg.fine.width)
    with torch.no_grad():
        stem = encoder.stem(image)
    assert stem.shape == encoder.patch_projection(image).shape == (1, cfg.hidden, *cfg.token_shape)
    assert torch.isfinite(stem).all() and torch.count_nonzero(stem) > 0
    torch.testing.assert_close(build_model(replace(cfg, stem_channels=0)).encoder.token_xyz, encoder.token_xyz,
                               rtol=0, atol=0)
    with pytest.raises(ValueError, match='multiples of four'):
        replace(cfg, fine=CropSpec(depth=25, width=20, behind=8))


def test_production_crop_and_tokens_are_laterally_centered():
    cfg = DirectConfig(encoder='patch4', token_only=True)
    image_grid = torch.from_numpy(crop_local_grid(cfg.fine))
    assert image_grid.shape == (120, 104, 104, 3)
    torch.testing.assert_close(image_grid[0, 0, 0, :2], -image_grid[0, -1, -1, :2])
    xyz = build_model(cfg).encoder.token_xyz.reshape(*cfg.token_shape, 3)
    torch.testing.assert_close(xyz[0, 0, 0, :2], -xyz[0, -1, -1, :2])
