import json
import os

import numpy as np
import pytest
import torch

from config import BACKFILLABLE_CONFIG_DEFAULTS, Config
from fit_session import fit_input, input_source_enabled
from patch_normals import PatchNormals, override_patch_normals
from pack_resident_pools import open_pool


class Cache:
    def gather(self, indices):
        values = torch.tensor([[0, 0, 0], [255, 128, 128], [128, 128, 1]], dtype=torch.uint8)
        return values[indices[..., 2]]

    def gather_with_exclusion(self, indices):
        values = self.gather(indices)
        return values, (values != 0).any(dim=-1)


def test_patch_precedence_and_floor_coordinates():
    store = PatchNormals(Cache(), 2., (1, 1, 3), (0, 1))
    points = torch.tensor([[0., 0., 0.], [0., 0., 3.9], [0., 0., 4.],
                           [0., 0., 6.], [-.1, 0., 4.]])
    fallback = torch.tensor([[0., 1., 0.]]).expand(5, 3)
    weights = torch.tensor([.5, .5, 0., .5, .5])
    normals, actual = override_patch_normals({'patch_normals': store}, points, fallback, weights)
    torch.testing.assert_close(normals, torch.tensor([[0., 1., 0.], [0., 0., 1.],
                                                     [-1., 0., 0.], [0., 1., 0.], [0., 1., 0.]]))
    torch.testing.assert_close(actual, torch.tensor([.5, 1., 1., .5, .5]))


def test_optional_input_is_off_for_old_and_new_configs():
    assert not input_source_enabled({}, 'patch_normals')
    assert Config().dense_normals_source == 'lasagna'
    assert BACKFILLABLE_CONFIG_DEFAULTS['dense_normals_source'] == 'lasagna'
    assert Config().dense_normals_patch_signed is False
    assert BACKFILLABLE_CONFIG_DEFAULTS['dense_normals_patch_signed'] is False
    assert not fit_input('patch_normals').required({})
    preferred = {'dense_normals_source': 'patch_preferred'}
    assert fit_input('patch_normals').required(preferred)
    assert not fit_input('patch_normals').required({**preferred, 'input_use_normals': False})
    assert fit_input('normal_x').required(preferred)
    assert fit_input('normal_y').required(preferred)


def test_v4_pool_layout_is_validated(tmp_path):
    meta = {'format': 'respool', 'version': 4, 'rows': 1, 'brick_shape': [1, 1, 1],
            'channels': ['nx'], 'channel_layout': [{'dtype': '<f4', 'file': 'channel_0.u8',
                                                   'shape': [1, 1], 'order': 'C'}]}
    (tmp_path / 'meta.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='unsupported v4 channel layout'):
        open_pool(tmp_path)
    meta['channel_layout'][0]['dtype'] = '|u1'
    (tmp_path / 'meta.json').write_text(json.dumps(meta))
    np.save(tmp_path / 'table.npy', np.zeros((1, 1, 1), dtype=np.int32))
    np.save(tmp_path / 'brick_coords.npy', np.zeros((1, 3), dtype=np.int32))
    (tmp_path / 'channel_0.u8').write_bytes(b'\x80')
    _, _, _, pools = open_pool(tmp_path)
    assert pools[0][0, 0] == 128


class HalfCovered:
    def sample(self, points, *, return_exclusion=False):
        normals = torch.zeros_like(points)
        normals[..., 2] = -1  # preserve signed nz/ny/nx; loss remains axial
        present = torch.arange(len(points), device=points.device) % 2 == 0
        return (normals, present, present) if return_exclusion else (normals, present)


def test_phase_normal_sampler_uses_patch_coverage():
    from sdt_losses import sample_lasagna_normals_nearest
    data = torch.zeros((3, 4, 4, 4), dtype=torch.uint8)
    data[0] = 128
    data[1] = 255  # fallback along y
    volume = dict(backend='dense_test', volume=data, shape=(4, 4, 4),
                  lasagna_scale=1, z_origin=0, patch_normals=HalfCovered())
    normal, valid = sample_lasagna_normals_nearest(volume, torch.ones((2, 3)))
    torch.testing.assert_close(normal, torch.tensor([[0., 0., -1.], [0., 1., 0.]]))
    assert valid.all()


@pytest.mark.parametrize('signed,predicted,expected', [
    (False, [0., 0., 1.], .5),
    (False, [0., 0., -1.], .5),
    (True, [0., 0., 1.], .5),  # outward is opposite the inward patch target
    (True, [0., 0., -1.], 1.5),
    (True, [0., -1., 0.], .5),  # reversed Lasagna fallback still matches
])
def test_dense_loss_replaces_lasagna_target(monkeypatch, signed, predicted, expected):
    import losses
    class Identity:
        def inv(self, points):
            return points
    data = torch.zeros((3, 8, 8, 8), dtype=torch.uint8)
    data[0] = 128
    data[1] = 255
    volume = dict(backend='dense_test', volume=data, shape=(8, 8, 8),
                  lasagna_scale=1, z_origin=0, y_origin=-4, x_origin=-4,
                  patch_normals=HalfCovered())
    prediction = torch.tensor(predicted, requires_grad=True)
    monkeypatch.setattr(losses, 'get_radial_normal_in_scroll_space',
                        lambda transform, points, **kw: prediction.expand_as(points))
    cfg = Config().as_dict()
    cfg['dense_normals_patch_signed'] = signed
    result = dict(losses.iter_lasagna_losses(
        Identity(), torch.tensor(1.), volume, 2, 8, compute_spacing=False,
        cfg=cfg, z_begin=1, z_end=3))
    torch.testing.assert_close(result['dense_normals'], torch.tensor(expected))
    result['dense_normals'].backward()
    if signed:
        # Inward targets are -x: gradient descent must increase outward x,
        # including when the predicted orientation is reversed.
        torch.testing.assert_close(prediction.grad[2], torch.tensor(-.5))


def test_signed_option_without_patch_normals_keeps_lasagna_unsigned(monkeypatch):
    import losses
    class Identity:
        def inv(self, points):
            return points
    data = torch.zeros((3, 8, 8, 8), dtype=torch.uint8)
    data[0], data[1] = 128, 255
    volume = dict(backend='dense_test', volume=data, shape=(8, 8, 8),
                  lasagna_scale=1, z_origin=0, y_origin=-4, x_origin=-4)
    monkeypatch.setattr(losses, 'get_radial_normal_in_scroll_space',
                        lambda transform, points, **kw: torch.tensor([0., -1., 0.]).expand_as(points))
    cfg = Config().as_dict()
    cfg['dense_normals_patch_signed'] = True
    result = dict(losses.iter_lasagna_losses(
        Identity(), torch.tensor(1.), volume, 2, 8, compute_spacing=False,
        cfg=cfg, z_begin=1, z_end=3))
    torch.testing.assert_close(result['dense_normals'], torch.tensor(0.))


@pytest.mark.skipif(not os.environ.get('PATCH_NORMALS_TEST_EXPORT'),
                    reason='Set PATCH_NORMALS_TEST_EXPORT to a real patch export')
def test_real_patch_export():
    from pathlib import Path
    import zarr
    from patch_normals import load_patch_normals
    root = Path(os.environ['PATCH_NORMALS_TEST_EXPORT'])
    meta, _, coords, pools = open_pool(root / 'signed_normals_u8.respool')
    brick = np.array(meta['brick_shape'])
    row = len(coords) // 2
    occupied = np.flatnonzero(np.any(np.stack([p[row] for p in pools]) != 0, axis=0))[:32]
    assert len(occupied)
    index = np.array(np.unravel_index(occupied, tuple(brick))).T + coords[row] * brick
    manifest = json.loads((root / 'manifest.json').read_text())
    cell = manifest['cell_size_fitter_voxels']
    store = load_patch_normals(root, z_begin=float(index[:, 0].min() * cell),
                               z_end=float((index[:, 0].max() + 1) * cell), device='cpu')
    try:
        points = torch.from_numpy((index + .5) * cell).float()
        normal, valid = store.sample(points)
        assert valid.all()
        original = np.stack([
            zarr.open(str(root / manifest['channels']['n' + axis]), mode='r').vindex[
                index[:, 0], index[:, 1], index[:, 2]]
            for axis in 'zyx'], axis=-1)
        np.testing.assert_allclose(normal.numpy(), original, atol=.012, rtol=0)
        out, weight = override_patch_normals(
            {'patch_normals': store}, points, torch.ones_like(normal), torch.zeros(len(points)))
        assert torch.equal(out, normal) and weight.eq(1).all()
        _, outside = store.sample(torch.tensor([[-1., 0., 0.]]))
        assert not outside.any()
    finally:
        store.close()
