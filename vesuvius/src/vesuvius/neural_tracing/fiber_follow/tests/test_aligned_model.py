"""The aligned model end to end: compiled training decisions match inference and train every part."""
import copy

import torch

from model_fixtures import aligned_config, aligned_batch
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model
from vesuvius.neural_tracing.fiber_follow.regression.train import prepare_training, training_prediction
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms

# Inputs of zero-initialized output layers: no gradient until those layers move.
ZERO_INIT_UPSTREAM = {'history_encoder.path_projection.0.weight', 'history_encoder.path_projection.0.bias',
                      'path_geometry.embed.0.weight', 'path_geometry.embed.0.bias'}


def test_aligned_model_compiles_matches_inference_and_trains_every_part():
    torch.manual_seed(21)
    model = build_model(aligned_config())
    assert model.architecture == 'axial_patch4_residual_stem_tokens_fiber_slabs_v17'
    compiled = prepare_training(copy.deepcopy(model), backend='eager')
    batch = aligned_batch(model.cfg)
    args = batch['x'], batch['hist'], batch['hmask']
    # Threshold one keeps every refinement active in both the compacting and fixed-slot paths.
    eager = model(*args, confidence_threshold=1.)
    trained = training_prediction(compiled, *args, confidence_threshold=1.)
    for key in ('points', 'hazard_logits', 'refinement_points', 'refinement_mask'):
        torch.testing.assert_close(eager[key], trained[key], atol=1e-6, rtol=1e-5)
    assert trained['refinement_points'].shape[1] == model.cfg.recurrent_refinement_steps+1
    for m, out in ((model, eager), (compiled, trained)):
        terms = loss_terms(out, batch, m.cfg)
        ((terms['geometry_per_state']+.5*terms['confidence_per_state']).mean()+out['hazard_logits'].square().mean()).backward()
    for (name, p), (_, q) in zip(model.named_parameters(), compiled.named_parameters()):
        if name in ZERO_INIT_UPSTREAM:
            assert q.grad is None or q.grad.eq(0).all(), name
            continue
        assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0, name
        torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4, msg=name)
