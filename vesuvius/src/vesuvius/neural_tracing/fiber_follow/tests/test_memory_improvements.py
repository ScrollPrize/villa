"""History encoder layout: full-resolution fine stages, spatial plus path tokens, and padding."""
import torch

from model_fixtures import aligned_config, aligned_batch
from vesuvius.neural_tracing.fiber_follow.models.history_slabs import HistoryEncoder


def test_history_tokens_keep_fine_stages_spatial_bank_path_tokens_and_padding():
    c = aligned_config()
    encoder = HistoryEncoder(c)
    batch = aligned_batch(c)['x']
    batch['history_path_valid'][0, 0, 0] = False
    batch['history_path_points'][0, 0, 0] = float('nan')
    stages = []
    handles = [block.register_forward_hook(lambda m, x, y: stages.append(tuple(y.shape[1:])))
               for block in encoder.convolution[1::2]]
    try:
        tokens, padding = encoder(batch['history_slabs'], batch['history_valid'], batch['history_pose'],
                                  path_points=batch['history_path_points'], path_tangents=batch['history_path_tangents'],
                                  path_valid=batch['history_path_valid'])
    finally:
        for handle in handles:
            handle.remove()
    assert stages == [(32, 8, 65, 65), (64, 4, 33, 33), (128, 2, 17, 17)]
    assert encoder.spatial_tokens_per_slab == 578 and encoder.tokens_per_slab == 581
    assert tokens.shape == (2, 8*581, c.hidden)
    assert padding[0, 578] and not padding[1, :2*581].any() and padding[:, 2*581:].all()
    assert torch.isfinite(tokens).all() and tokens[padding].eq(0).all()
    xyz = encoder.xyz.reshape(*encoder.token_shape, 3)
    torch.testing.assert_close(xyz[0, 0, 0], torch.zeros(3))
    torch.testing.assert_close(xyz[-1, -1, -1], torch.ones(3))
    tokens[:, 579].square().sum().backward()
    assert encoder.path_projection[-1].weight.grad.abs().sum() > 0
    assert encoder.convolution[0].weight.grad.abs().sum() > 0
