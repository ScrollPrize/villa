"""Fresh BasicBlockD image stem: layout, geometry and compiled gradients."""
from dataclasses import replace

import pytest
import torch

from slab_fixtures import cfg, slab_batch
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model
from vesuvius.neural_tracing.fiber_follow.regression.train import prepare_training, training_prediction
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms


@pytest.mark.parametrize('activation_checkpointing', [False, True])
def test_fresh_stem_learns_through_compiled_geometry_and_candidate_scoring(activation_checkpointing):
    torch.manual_seed(21)
    model = build_model(cfg(encoder='patch4', token_only=True, stem_channels=32,
        stem_blocks=2, recurrent_refinement_steps=1, activation_checkpointing=activation_checkpointing))
    prepare_training(model, backend='eager')
    batch = slab_batch(model.cfg)
    candidates = torch.zeros(2, 1, model.cfg.n_future, 3)
    candidates[..., 2] = model.planes
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        out = training_prediction(model, batch['x'], batch['hist'], batch['hmask'],
                                  candidates, confidence_threshold=1.)
        terms = loss_terms({k: v for k, v in out.items() if not k.startswith('candidate_')}, batch, model.cfg)
        loss = (terms['geometry_per_state']+.5*terms['confidence_per_state']).mean()
        loss = loss+out['candidate_hazard_logits'].square().mean()
        loss.backward()
        projection = model.encoder.stem.projection.weight.grad
        assert torch.isfinite(projection).all() and projection.abs().sum() > 0
        for name, p in model.encoder.stem.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            if p.ndim == 5:
                assert p.grad.abs().sum() > 0, name
        opt.step()


@pytest.mark.parametrize('shape', [(24, 20, 20), (28, 24, 24)])
def test_stem_grid_matches_existing_patch_tokens(shape):
    config = cfg(encoder='patch4', token_only=True, stem_channels=32)
    config = replace(config, fine=replace(config.fine, depth=shape[0], width=shape[1]))
    model = build_model(config)
    stem = model.encoder.stem
    image = torch.randn(1, config.input_channels, *shape)
    actual = stem(image)
    assert actual.shape == model.encoder.patch_projection(image).shape == (1, config.hidden, *config.token_shape)
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
    original = build_model(replace(config, stem_channels=0))
    torch.testing.assert_close(model.encoder.token_xyz, original.encoder.token_xyz, rtol=0, atol=0)


def test_stem_uses_shared_basicblock_d_at_full_half_and_quarter_resolution():
    from vesuvius.models.build.resblocks import BasicBlockD
    config = cfg(encoder='patch4', token_only=True, stem_channels=32, stem_blocks=2)
    stem = build_model(config).encoder.stem
    blocks = [m for m in stem.modules() if isinstance(m, BasicBlockD)]
    assert len(blocks) == 5
    assert [b.output_channels for b in blocks] == [32, 32, 32, 64, 64]
    assert [tuple(b.stride) for b in blocks] == [(1, 1, 1), (2, 2, 2), (1, 1, 1), (2, 2, 2), (1, 1, 1)]
    image = torch.randn(1, config.input_channels, 24, 20, 20)
    with torch.no_grad():
        full = stem.input(image)
        half = stem.blocks[0](full)
        quarter = stem.blocks[1](half)
    assert full.shape == (1, 32, 24, 20, 20)
    assert half.shape == (1, 32, 12, 10, 10)
    assert quarter.shape == (1, 64, 6, 5, 5)
    assert all(isinstance(b.conv1.norm, torch.nn.InstanceNorm3d) and
               isinstance(b.nonlin2, torch.nn.ReLU) for b in blocks)
    assert isinstance(blocks[1].skip[0], torch.nn.AvgPool3d)
    assert isinstance(blocks[3].skip[0], torch.nn.AvgPool3d)


