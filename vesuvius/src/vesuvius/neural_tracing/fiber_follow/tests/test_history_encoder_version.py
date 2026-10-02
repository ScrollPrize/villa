"""Fine history geometry, padding, and explicit legacy checkpoint compatibility."""
import copy

import pytest
import torch

from model_fixtures import config as cfg, slab_batch
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model
from vesuvius.neural_tracing.fiber_follow.regression.train import (
    build_parser, checkpoint_config, resolve_history_encoder,
)
from model_fixtures import REQUIRED


@pytest.mark.parametrize('variant,shapes', [
    ('fine', [(32, 8, 65, 65), (64, 4, 33, 33), (128, 2, 17, 17)]),
    ('legacy', [(8, 8, 33, 33), (16, 4, 17, 17), (32, 2, 9, 9)]),
])
def test_history_stages_preserve_full_resolution_and_token_layout(variant, shapes):
    model = build_model(cfg(history_encoder=variant))
    encoder = model.history_encoder
    batch = slab_batch(model.cfg, 1)
    batch['x']['history_valid'][:, 1:] = False
    actual = []
    handles = [block.register_forward_hook(lambda m, x, y: actual.append(tuple(y.shape[1:])))
               for block in encoder.convolution[1::2]]
    try:
        with torch.no_grad():
            tokens, padding = model.encode_history(batch['x'])
    finally:
        for handle in handles:
            handle.remove()
    assert actual == shapes
    count = 578 if variant == 'fine' else 162
    assert encoder.tokens_per_slab == count
    assert tokens.shape == (1, 8*count, model.cfg.hidden)
    assert not padding[:, :count].any() and padding[:, count:].all()
    assert tokens[:, count:].eq(0).all()
    assert torch.isfinite(tokens).all()
    # Physical extent is unchanged; normalized lateral spacing halves in v15.
    xyz = encoder.xyz.reshape(*encoder.token_shape, 3)
    torch.testing.assert_close(xyz[0, 0, 0], torch.zeros(3))
    torch.testing.assert_close(xyz[-1, -1, -1], torch.ones(3))
    assert xyz[0, 0, 1, 0] == (1/16 if variant == 'fine' else 1/8)


@pytest.mark.parametrize('encoder,token_only,stem', [('conv', False, 0),
    ('patch4', False, 0), ('patch4', True, 0), ('patch4', True, 4)])
def test_old_checkpoint_without_history_setting_loads_original_weights(encoder, token_only, stem):
    model = build_model(cfg(encoder=encoder, token_only=token_only, stem_channels=stem,
                           history_encoder='legacy'))
    metadata = model.cfg.to_dict()
    metadata.pop('history_encoder')
    checkpoint = dict(architecture=model.architecture, model_cfg=metadata)
    original = copy.deepcopy(checkpoint)
    restored_cfg = checkpoint_config(checkpoint)
    assert restored_cfg.history_encoder == 'legacy'
    assert restored_cfg.architecture.endswith('_v14')
    assert resolve_history_encoder(None, checkpoint) == 'legacy'
    assert checkpoint == original
    restored = build_model(restored_cfg)
    restored.load_state_dict(model.state_dict(), strict=True)
    batch = slab_batch(restored_cfg, 1)
    with torch.no_grad():
        actual = restored.encode_history(batch['x'])
        expected = model.encode_history(batch['x'])
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match='History encoder must match'):
        resolve_history_encoder('fine', checkpoint)
    with pytest.raises(ValueError, match='architecture'):
        checkpoint_config(dict(checkpoint, model_cfg=dict(metadata, history_encoder='fine')))


def test_fine_checkpoint_defaults_and_explicit_selection():
    assert build_parser().parse_args(REQUIRED).history_encoder is None
    assert resolve_history_encoder(None) == 'fine'
    assert resolve_history_encoder('legacy') == 'legacy'
    model_cfg = cfg()
    checkpoint = dict(architecture=model_cfg.architecture, model_cfg=model_cfg.to_dict())
    assert checkpoint_config(checkpoint).history_encoder == 'fine'
    assert resolve_history_encoder(None, checkpoint) == 'fine'
    assert checkpoint['architecture'].endswith('_v15')
    with pytest.raises(ValueError, match='History encoder must match'):
        resolve_history_encoder('legacy', checkpoint)
    checkpoint['model_cfg'].pop('history_encoder')
    with pytest.raises(ValueError, match='architecture'):
        checkpoint_config(checkpoint)
    with pytest.raises(ValueError, match='History encoder'):
        cfg(history_encoder='unsupported')
