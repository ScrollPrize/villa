"""Shared-classifier algebra, gradients, and canonical initialization contracts."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from vesuvius.ink_detection.models import canonical_projection as cp
from vesuvius.ink_detection.training.multiteacher_loss import joint_loss
from vesuvius.ink_detection.data.normalization import normalize_image
from vesuvius.ink_detection.config import NormalizationConfig


class TinyCanonical(nn.Module):
    def __init__(self, with_norm=True):
        super().__init__()
        self.backbone = nn.Conv3d(1, 4, 3, padding=1)
        self.normalization = nn.BatchNorm3d(1) if with_norm else nn.Identity()
        self.decoder = nn.Module()
        self.decoder.logit = nn.Conv2d(4, 1, 1)
        self.decoder.depth_collapse = nn.Module()
        self.decoder.depth_collapse.attn_conv = nn.Conv3d(4, 1, 1)

    def forward_features(self, x):
        return F.avg_pool3d(self.backbone(self.normalization(x)), (1, 4, 4)), []


@pytest.fixture
def model(monkeypatch):
    torch.manual_seed(12)
    monkeypatch.setattr(cp, "canonical_runtime", lambda _: SimpleNamespace(RegressionModel=TinyCanonical))
    return cp.CanonicalLogitProjection("unused")


def batch():
    x = torch.rand(2, 1, 64, 16, 16)
    valid = torch.ones_like(x)
    valid[:, :, [0, 63]] = 0
    return {"image": x, "valid_3d": valid,
            "labels_2d": torch.randint(0, 2, (2, 1, 16, 16)).float(),
            "mask_2d": torch.ones(2, 1, 16, 16)}


def test_initialized_2d_matches_canonical_feature_pooling(model):
    b = batch()
    f, _ = model.canonical.forward_features(b["image"][:, :, 1:63])
    a = model.canonical.decoder.depth_collapse.attn_conv(f).softmax(2)
    expected = model.canonical.decoder.logit((a*f).sum(2))
    out = model(b["image"], b["valid_3d"])
    torch.testing.assert_close(out["native_2d_logits"], expected, atol=1e-6, rtol=1e-5)
    expected_full = F.interpolate(expected.sigmoid(), size=(16, 16), mode="bilinear", align_corners=False)
    torch.testing.assert_close(out["ink"].sigmoid(), expected_full, atol=1e-6, rtol=1e-5)
    assert out["ink_3d_logits"].shape == b["image"].shape
    assert not out["depth_weights"][:, :, [0, 63]].any()
    torch.testing.assert_close(out["depth_weights"].sum(2), torch.ones_like(out["ink"]))


def test_both_losses_reach_shared_classifier_and_human_reaches_attention(model):
    b = batch()
    out = model(b["image"], b["valid_3d"])
    loss, metrics = joint_loss(out, b, torch.rand_like(b["image"]), torch.tensor([1., .2]),
                               gradient_diagnostics=True)
    loss.backward()
    for p in [model.canonical.backbone.weight, model.canonical.decoder.logit.weight,
              model.canonical.decoder.depth_collapse.attn_conv.weight, model.depth_coordinate_scale]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert metrics["gradient_at_3d_logits/human_squared"] > 0
    assert metrics["gradient_at_3d_logits/teacher_squared"] > 0
    # Isolate the human signal: it must train the actual 3D classifier alone.
    model.zero_grad()
    out = model(b["image"], b["valid_3d"])
    F.binary_cross_entropy_with_logits(out["ink"], b["labels_2d"]).backward()
    assert model.canonical.decoder.logit.weight.grad.abs().sum() > 0


def test_3d_only_does_not_execute_attention(model):
    b = batch()
    expected = model(b["image"])["ink_3d_logits"]
    def forbidden(*_):
        raise AssertionError("Attention executed in 3D-only inference")
    handle = model.canonical.decoder.depth_collapse.attn_conv.register_forward_hook(forbidden)
    torch.testing.assert_close(model.forward_3d(b["image"]), expected)
    handle.remove()


def test_invalid_support_has_finite_zero_loss_and_gradients(model):
    b = batch()
    b["valid_3d"].zero_()
    b["mask_2d"].zero_()
    out = model(b["image"], b["valid_3d"])
    assert not out["depth_weights"].any()
    loss, _ = joint_loss(out, b, torch.zeros_like(b["image"]), torch.ones(2))
    loss.backward()
    assert loss == 0
    assert all(p.grad is None or not p.grad.any() for p in model.parameters())


def test_checkpoint_strictness_ema_independence_and_fixed_bn(model):
    state = deepcopy(model.canonical.state_dict())
    model.load_canonical_state({"state_dict": state})
    bad = dict(state)
    bad.pop("decoder.logit.bias")
    with pytest.raises(RuntimeError):
        model.load_canonical_state({"state_dict": bad})
    ema = deepcopy(model).eval()
    model.train()
    assert not model.canonical.normalization.training
    before = model.canonical.normalization.running_mean.clone()
    model(batch()["image"])
    torch.testing.assert_close(before, model.canonical.normalization.running_mean)
    with torch.no_grad():
        model.canonical.decoder.logit.bias.add_(1)
    assert not torch.equal(model.canonical.decoder.logit.bias, ema.canonical.decoder.logit.bias)


def test_canonical_normalization_is_fixed_clip_divide():
    import numpy as np
    config = NormalizationConfig.from_value({"mode": "clip_zscore", "clip_min": 0,
                                            "clip_max": 200, "mean": 0, "std": 200})
    raw = np.array([-1, 0, 50, 100, 200, 255], dtype=np.float32)
    np.testing.assert_array_equal(normalize_image(raw, config), [0, 0, .25, .5, 1, 1])


@pytest.mark.parametrize("tied", [False, True])
def test_spatial_max_pool_preserves_values_and_gradients(tied):
    from vesuvius.ink_detection.models.deterministic_pool import SpatialMaxPool3d
    torch.manual_seed(23)
    x = torch.randn(2, 3, 5, 13, 12)
    if tied:
        x = x.round()
    x.requires_grad_()
    other = x.detach().clone().requires_grad_()
    expected = F.max_pool3d(x, (1, 3, 3), (1, 2, 2), (0, 1, 1))
    actual = SpatialMaxPool3d()(other)
    assert torch.equal(actual, expected)
    g = torch.randn_like(actual)
    actual.backward(g)
    expected.backward(g)
    torch.testing.assert_close(other.grad, x.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('depth', [64, 65, 129])
def test_full_segment_orientation_precedes_crop(depth):
    import numpy as np
    from vesuvius.ink_detection.data.multiteacher import read_oriented_patch
    source = np.arange(depth, dtype=np.float32)[:, None, None] * np.ones((1, 3, 4))
    middle = depth // 2
    for reverse in (False, True):
        oriented = source[::-1] if reverse else source
        expected = oriented[middle-32:middle+32, :256, :256]
        np.testing.assert_array_equal(read_oriented_patch(source, 0, 0, reverse_depth=reverse), expected)


def test_orientation_preprocessing_aligns_student_teacher_raw_and_excludes_calibration(tmp_path):
    import json
    import numpy as np
    from vesuvius.ink_detection.data.multiteacher import FlatDistillationDataset
    record = {'scroll': 'phercparis4', 'segment': 'test', 'teacher_id': 0,
              'image': 'ct', 'inklabels': 'ink', 'supervision_mask': 'mask'}
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps({'seed': 27, 'segments': [record]}))
    np.savez(tmp_path/'patches.npz', train_0=[[0, 0]], val_0=[[0, 0], [0, 0]],
             heldout_0=np.ones((16, 16), dtype=bool))
    norm = {'mode': 'clip_zscore', 'clip_min': 0, 'clip_max': 200, 'mean': 0, 'std': 200}
    dataset = FlatDistillationDataset(path, [norm, norm], validation=True, augment=False,
        student_normalization=norm, valid_depth_margin=1,
        segment_depth_reversals={'phercparis4/test': True},
        orientation_calibration_exclusions={'phercparis4/test': [0]})
    assert dataset.validation_draws == [(0, 1)]
    ct = np.broadcast_to(np.arange(65, dtype=np.float32)[:, None, None], (65, 256, 256))
    planes = np.ones((1, 256, 256), dtype=np.uint8)
    dataset._open = lambda name: ct if name == 'ct' else planes
    sample = dataset[0]
    torch.testing.assert_close(sample['raw'][0, :, 0, 0], torch.arange(64, 0, -1).float())
    torch.testing.assert_close(sample['image'], sample['raw']/200)
    torch.testing.assert_close(sample['teacher_image'], sample['image'])
    assert sample['reverse_depth']
    assert sample['labels_2d'].all() and sample['mask_2d'].all()
    assert not sample['valid_3d'][0, [0, 63]].any()
    with pytest.raises(ValueError, match='every segment'):
        FlatDistillationDataset(path, [norm, norm], segment_depth_reversals={})


def test_orientation_cannot_change_during_resume():
    from vesuvius.ink_detection.training.multiteacher import validate_resume_signature
    old = {'segment_depth_reversals': {'phercparis4/test': False}}
    new = {'segment_depth_reversals': {'phercparis4/test': True}}
    with pytest.raises(ValueError, match='Resume configuration'):
        validate_resume_signature(old, new)


def test_ambiguous_segments_are_excluded_from_both_splits(tmp_path):
    import json
    import numpy as np
    from vesuvius.ink_detection.data.multiteacher import FlatDistillationDataset
    records = [{'scroll': 'phercparis4', 'segment': name} for name in ['keep', 'exclude']]
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps({'seed': 27, 'segments': records}))
    np.savez(tmp_path/'patches.npz', train_0=[[0, 0]], train_1=[[0, 0]],
             val_0=[[0, 0]], val_1=[[0, 0]], heldout_0=[[1]], heldout_1=[[1]])
    for validation in (False, True):
        dataset = FlatDistillationDataset(path, ['percentile_minmax']*2, validation=validation,
            excluded_segments=['phercparis4/exclude'])
        assert dataset.groups['phercparis4'] == [0]
        assert all(dataset.locate(i)[0] == 0 for i in range(min(100, len(dataset))))


def test_excluded_scrolls_do_not_enter_validation_macro():
    from vesuvius.ink_detection.training.multiteacher_reporting import ScrollMetrics
    metrics = ScrollMetrics([{'scroll': 'active'}, {'scroll': 'excluded'}], 'cpu')
    metrics.state[0, :10] = torch.tensor([1, 0, 0, 1, 2, 2, 2, 2, 0, 1])
    result = metrics.compute(SimpleNamespace(reduce=lambda state, reduction: state))
    assert not any(k.startswith('val/excluded/') for k in result)
    assert result['val/macro/bce'] == 1


def test_canonical_ema_teacher_export_is_exact_and_rejects_corruption(model, tmp_path):
    from vesuvius.ink_detection.training.multiteacher_handoff import export_backbone_teacher, verify_exported_teacher
    source = tmp_path/'checkpoint.pth'
    teacher = tmp_path/'teacher.pth'
    state = deepcopy(model.state_dict())
    config = {'model_type': 'multiteacher_3d_projection', 'projection': {'kind': 'canonical_logits'},
              'image_normalization': {'mode': 'clip_zscore', 'std': 200},
              'model_config': {'canonical': {'source_dir': 'test'}}}
    torch.save({'config': config, 'ema_model': state, 'optimizer_step': 10000}, source)
    report = export_backbone_teacher(source, teacher, 10000)
    assert report['component'] == 'canonical_3d' and report['weights'] == 'ema_model'
    payload = torch.load(teacher, weights_only=False)
    assert payload['config'] == config
    assert all(torch.equal(v, payload['ema_model'][k]) for k, v in state.items())
    payload['ema_model']['canonical.decoder.logit.bias'].add_(1)
    torch.save(payload, teacher)
    with pytest.raises(ValueError, match='not exactly'):
        verify_exported_teacher(source, teacher)


def test_frozen_canonical_teacher_skips_projection_and_has_no_gradients(model, monkeypatch):
    from vesuvius.ink_detection.training import multiteacher_loss as loss_module
    from vesuvius.ink_detection.training.dynamic_labels import _freeze_for_inference
    frozen = _freeze_for_inference(deepcopy(model), device=torch.device('cpu'), dtype=torch.float32)
    class Precise(nn.Module):
        def forward(self, image):
            return {'ink': torch.zeros_like(image)}
    monkeypatch.setattr(loss_module, 'load_frozen_ink_model',
                        lambda path, **_: frozen if path == 'ema' else Precise())
    teachers = loss_module.FrozenInkTeachers({'paris4': {'checkpoint': 'precise', 'weight': 1},
        'coarse': {'checkpoint': 'ema', 'weight': .2, 'background_threshold': 50}}, 'cpu')
    b = batch()
    b.update(teacher_image=b['image'], raw=torch.full_like(b['image'], 100),
             teacher_id=torch.tensor([0, 1]))
    expected = frozen.forward_3d(b['teacher_image'][1:]).sigmoid()
    def forbidden(*_):
        raise AssertionError('Frozen 3D teacher executed attention')
    handle = frozen.canonical.decoder.depth_collapse.attn_conv.register_forward_hook(forbidden)
    targets, weights, original = teachers.generate(b)
    handle.remove()
    torch.testing.assert_close(targets[1:], expected)
    assert not targets.requires_grad and not any(p.requires_grad for p in frozen.parameters())
    torch.testing.assert_close(weights, torch.tensor([1., .2]))
