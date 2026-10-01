import json
import numpy as np
import torch
import pytest
from reviewed_fiber_evaluation import REPO
from vesuvius.neural_tracing.fiber_trace_3d.tta import (
    TRANSFORMS, IDENTITY, reframe, transform_input, inverse_space,
    encode_aggregate, midpoint_median, selected_manifest)
from vesuvius.neural_tracing.fiber_trace_3d.direction import encode_lasagna_direction_3x2


def test_all_transforms():
    assert len(set(TRANSFORMS)) == 13
    rng = np.random.default_rng(7)
    vectors = np.concatenate([rng.normal(size=(17, 3)), np.eye(3), np.zeros((1, 3))])
    expected = torch.tensor(encode_lasagna_direction_3x2(vectors)).T[None, :, :, None, None]
    expected = torch.cat([expected, torch.full_like(expected[:, :1], .7)], dim=1)
    canonical = reframe(expected, IDENTITY)
    for transform in TRANSFORMS:
        p, flips = transform
        mapping = [2 - p[2 - a] for a in range(3)]
        signs = [-1 if flips[2 - a] else 1 for a in range(3)]
        transformed = vectors[:, mapping] * signs
        encoded = torch.tensor(encode_lasagna_direction_3x2(transformed)).T[None, :, :, None, None]
        encoded = torch.cat([encoded, torch.full_like(encoded[:, :1], .7)], dim=1)
        actual = reframe(transform_input(encoded, transform), transform)
        torch.testing.assert_close(actual, canonical, atol=5e-7, rtol=1e-6)
        field = torch.arange(120).reshape(1, 1, 4, 5, 6)
        assert torch.equal(inverse_space(transform_input(field, transform), transform), field)
    torch.testing.assert_close(encode_aggregate(canonical), expected, atol=5e-7, rtol=1e-6)


def test_median_and_projection():
    assert midpoint_median(torch.tensor([0., 1., 3., 8.])).item() == 2
    assert midpoint_median(torch.tensor([0., 1., 3.])).item() == 1
    value = torch.ones(1, 7, 2, 2, 2) * 4
    encoded = encode_aggregate(value)
    assert torch.isfinite(encoded).all() and encoded.min() >= 0 and encoded.max() <= 1


def test_spatial_gradient_direction_oracle():
    # Independent convention oracle: differentiate a transformed scalar ramp,
    # rather than reuse the channel mapping/sign formula being tested.
    z, y, x = torch.meshgrid(*(torch.arange(5.) for _ in range(3)), indexing='ij')
    ramp = (x * 2 + y * 3 - z * 5)[None, None]
    expected = encode_lasagna_direction_3x2(torch.tensor([2., 3., -5.]))
    for transform in TRANSFORMS:
        transformed = transform_input(ramp, transform)[0, 0]
        gz, gy, gx = torch.gradient(transformed)
        encoded = encode_lasagna_direction_3x2(torch.stack((gx, gy, gz), dim=-1))
        raw = torch.cat((encoded.movedim(-1, 0)[None], torch.ones_like(ramp)), dim=1)
        restored = encode_aggregate(reframe(raw, transform))
        torch.testing.assert_close(restored[0, :6, 2, 2, 2], expected, atol=1e-6, rtol=1e-6)


def test_manifest(tmp_path):
    path = tmp_path/'inference.lasagna.json'
    groups = {f'mean_{c}': {'channels': [f'mean_{c}'], 'zarr': f'{c}/1', 'scaledown': 1}
              for c in ('presence', 'nx', 'ny')}
    path.write_text(json.dumps({'groups': groups, 'source_to_base': 1}))
    result = json.loads(selected_manifest(path, 'mean').read_text())
    assert set(result['groups']) == {'presence', 'nx', 'ny'}
    assert result['groups']['nx']['zarr'] == 'nx/1'
    assert selected_manifest(path, 'mean').exists()


def test_adapter_end_to_end(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.tta import TtaPredictAdapter, VARIANTS
    from lasagna.tiled_predict3d import write_lasagna_product_manifest
    adapter = TtaPredictAdapter({'model_3d': {'output_channels': 7},
                                 'training': {'mixed_precision': 'off'}},
                                zarr_path_prefix=tmp_path/'fiber')
    class Model(torch.nn.Module):
        calls = 0
        def forward(self, x):
            self.calls += 1
            return torch.cat((torch.full_like(x.repeat(1, 6, 1, 1, 1), .5), x), dim=1)
    model = Model()
    tile = torch.linspace(.1, .9, 64).reshape(1, 1, 4, 4, 4)
    result = adapter.run_tile_inference(model, tile, device=torch.device('cpu'))
    assert model.calls == 13 and result.shape == (1, 14, 4, 4, 4)
    assert VARIANTS == ('mean', 'median')
    for offset in (0, 7):
        torch.testing.assert_close(result[:, offset:offset+6], torch.full_like(result[:, :6], .5))
        torch.testing.assert_close(result[:, offset+6:offset+7], tile)
    tensors = adapter.product_tensors_from_output(result)
    assert len(tensors) == 2
    for output in adapter.output_products:
        channels = adapter.finalize_product_slab(output, tensors[output.name][0].numpy())
        assert len(channels) == 3
        assert all(a.dtype == np.uint8 and a.shape == (4, 4, 4) for a in channels.values())
    path = tmp_path/'inference.lasagna.json'
    write_lasagna_product_manifest(output_path=path, products=adapter.output_products,
                                  base_shape_zyx=(4, 4, 4))
    for variant in VARIANTS:
        assert len(json.loads(selected_manifest(path, variant).read_text())['groups']) == 3


def test_tta_rejects_tf32(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.tta import TtaPredictAdapter
    adapter = TtaPredictAdapter({'model_3d': {'output_channels': 7},
                                 'training': {'mixed_precision': 'off'},
                                 'inference': {'allow_tf32': True}},
                                zarr_path_prefix=tmp_path/'fiber')
    with pytest.raises(ValueError, match='allow_tf32=False'):
        adapter.load_model(device=torch.device('cpu'))
