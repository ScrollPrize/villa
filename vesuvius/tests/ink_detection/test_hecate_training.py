"""Loss/format regressions; opt-in Hub parity tests use real CT fixtures."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from vesuvius.ink_detection.data.resolution_distillation import teacher_targets, distillation_loss, student_input
from vesuvius.ink_detection.models.hecate_checkpoint import model_config, sampling_from_config, load_hecate, export_ema


def test_soft_targets_keep_background_as_negative_and_exclude_padding():
    valid = torch.ones(1, 1, 64, 8, 8)
    valid[:, :, [0, 63]] = 0
    foreground = torch.ones_like(valid, dtype=torch.bool)
    foreground[..., :4] = False
    output = {'ink_3d_logits': torch.zeros_like(valid), 'ink': torch.zeros(1, 1, 8, 8)}
    targets = teacher_targets(output, valid, 4, foreground)
    assert torch.all(targets['volume_weight'][:, :, [0, -1]] == .75)
    assert torch.all(targets['volume'][..., :1] == 0)
    assert torch.all(targets['volume'][..., 1:] == .5)
    student = {'ink_3d_logits': torch.zeros_like(targets['volume'], requires_grad=True),
               'ink': torch.zeros_like(targets['projection'], requires_grad=True)}
    loss, _ = distillation_loss(student, targets)
    loss.backward()
    assert student['ink_3d_logits'].grad[..., :1].min() > 0
    assert torch.all(student['ink_3d_logits'].grad[..., 1:] == 0)


def test_unclipped_normalization_and_area_reduction():
    raw = torch.full((1, 1, 64, 256, 256), 255.)
    assert student_input(raw, 1).max() == 1.275
    low = student_input(raw, 4)
    assert low.shape == (1, 1, 16, 64, 64) and low.min() == 1


@pytest.mark.parametrize('sampling', [2.4, 9.6])
def test_public_recipe_rejects_mismatched_normalization(sampling):
    cfg = model_config(sampling)
    assert sampling_from_config(cfg) == sampling
    cfg['image_normalization']['divisor'] = 256
    with pytest.raises(ValueError, match='Normalization'):
        sampling_from_config(cfg)


@pytest.fixture
def assets():
    path = os.environ.get('HECATE_TEST_ASSETS')
    if not path:
        pytest.skip('Set HECATE_TEST_ASSETS to real CT/checkpoint fixtures')
    return Path(path)


@pytest.mark.parametrize('sampling', [2.4, 9.6])
def test_real_ct_hub_parity_export_and_gradient(assets, tmp_path, sampling):
    device = os.environ.get('HECATE_TEST_DEVICE', 'cuda')
    spec = importlib.util.spec_from_file_location('hub_hecate_reference', assets/'hecate.py')
    hub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hub)
    checkpoint = assets/f'hecate_{sampling}um.pth'
    model, config = load_hecate(checkpoint, device, training=True)
    model.eval()
    reference = hub.load_model(checkpoint, device)
    crop = np.load(assets/f'test_{sampling}.npy')
    depth, width, _ = config['patch_size']
    errors = []
    for reverse in (False, True):
        oriented = crop[::-1] if reverse else crop
        start = oriented.shape[0]//2-depth//2
        raw = np.ascontiguousarray(oriented[start:start+depth, :width, :width])
        image = torch.from_numpy(raw[None, None]).to(device).float()/config['image_normalization']['divisor']
        valid = image.ne(0).any(2, keepdim=True).expand_as(image)
        for mixed in (False, True):
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed and device.startswith('cuda')):
                output = model(image, valid)
                expected2 = reference(image, valid)
                expected3 = reference.forward_3d(image)
            error2 = (output['ink']-expected2).abs().max().item()
            error3 = (output['ink_3d_logits']-expected3).abs().max().item()
            assert error2 == error3 == 0
            errors.append(dict(reverse=reverse, bf16=mixed, max_2d_error=error2, max_3d_error=error3))
    del reference
    model.zero_grad(set_to_none=True)
    output = model(image, valid)
    # Deliberately perturbed soft teacher probabilities, using real CT inputs.
    target = output['ink'].detach().sigmoid()*.8
    torch.nn.functional.binary_cross_entropy_with_logits(output['ink'], target).backward()
    for parameter in (model.canonical.backbone.conv1.weight, model.canonical.decoder.logit.weight,
                      model.canonical.decoder.depth_collapse.attn_conv.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    exported = export_ema(checkpoint, tmp_path/'export.pth')
    payload = torch.load(exported, map_location='cpu', weights_only=True)
    assert set(payload) == {'format_version', 'architecture_family', 'sampling_um', 'weight_source', 'config', 'ema_model'}
    assert payload['config']['datasets'] == [{'segments_path': '.', 'volume_scale': '0'}]
    reloaded = hub.load_model(exported, device)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(image, valid), model(image, valid)['ink'], atol=0, rtol=0)
    print(json.dumps({'sampling_um': sampling, 'parity': errors, 'gradient_reaches_encoder_3d_and_attention': True, 'export_loads_in_hub': True}))
