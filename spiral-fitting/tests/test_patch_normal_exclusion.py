"""CPU-only exclusion geometry, mask caching, and loss integration checks."""
import json
import os

import numpy as np
import pytest
import torch

from config import Config, BACKFILLABLE_CONFIG_DEFAULTS
from patch_normal_exclusion import (
    PatchExclusionMask, build_exclusion_mask, pack_presence_bits)
from patch_normals import load_patch_normals, override_patch_normals


def sparse_coverage(coverage, brick):
    brick = np.array(brick)
    grid = (np.array(coverage.shape) + brick - 1) // brick
    padded = np.zeros(grid * brick, dtype=bool)
    padded[tuple(slice(0, n) for n in coverage.shape)] = coverage
    cells = padded.reshape(grid[0], brick[0], grid[1], brick[1], grid[2], brick[2])
    cells = cells.transpose(0, 2, 4, 1, 3, 5).reshape(-1, int(np.prod(brick)))
    live = cells.any(axis=-1)
    table = np.zeros(len(cells), dtype=np.int32)
    table[live] = np.arange(1, int(live.sum()) + 1)
    bits = pack_presence_bits(np.concatenate([np.zeros((1, cells.shape[1]), bool), cells[live]]))
    coords = np.concatenate([np.full((1, 3), -1), np.argwhere(live.reshape(grid))]).astype(np.int32)
    return table.reshape(grid), bits, coords


def lookup(mask, points, brick):
    brick = torch.tensor(brick)
    b, local = points // brick, points % brick
    linear = (local[:, 0] * brick[1] + local[:, 1]) * brick[2] + local[:, 2]
    return mask.gather_at(b, linear // 64, linear % 64)


@pytest.mark.parametrize('radius', [0., .5, 1., 2.2, 4., 8.])
def test_mask_matches_brute_force_euclidean_distance_across_tiles(radius):
    shape, brick = (19, 21, 23), (3, 4, 5)
    coverage = np.zeros(shape, dtype=bool)
    sources = np.array([[0, 0, 0], [5, 8, 9], [6, 7, 10], [18, 20, 22], [10, 10, 10]])
    coverage[tuple(sources.T)] = True
    table, bits, _ = sparse_coverage(coverage, brick)
    out_table, out_bits = build_exclusion_mask(
        table, bits, brick, shape, radius, tile_bricks=2, workers=2)
    points = np.indices(shape).reshape(3, -1).T.copy()
    expected = (((points[:, None] - sources)**2).sum(axis=-1).min(axis=1) <= radius**2)
    mask = PatchExclusionMask(out_table, out_bits, 'cpu')
    np.testing.assert_array_equal(lookup(mask, torch.from_numpy(points), brick).numpy(), expected)
    assert lookup(mask, torch.empty((0, 3), dtype=torch.long), brick).shape == (0,)
    # No lookup may reference a nonexistent mixed row.
    assert out_table.max() < len(out_bits)
    mask.close()
    assert mask.bits is None


@pytest.mark.parametrize('filled', [False, True])
def test_uniform_bricks_share_rows(filled):
    shape, brick = (16, 16, 16), (4, 4, 4)
    table, bits, _ = sparse_coverage(np.full(shape, filled), brick)
    out_table, out_bits = build_exclusion_mask(table, bits, brick, shape, 4., tile_bricks=2)
    assert (out_table == int(filled)).all()
    assert out_bits.shape == (2, 1)  # no bitmap per brick


def test_process_build_matches_serial_rows(monkeypatch):
    import patch_normal_exclusion as module
    monkeypatch.setattr(module, '_PROCESS_MIN_CELLS', 0)
    coverage = np.zeros((16, 16, 16), dtype=bool)
    coverage[4, 4, 4] = True
    coverage[12, 12, 12] = True
    table, bits, _ = sparse_coverage(coverage, (4, 4, 4))
    args = (table, bits, (4, 4, 4), coverage.shape, 2.2)
    serial = build_exclusion_mask(*args, tile_bricks=1, workers=1)
    parallel = build_exclusion_mask(*args, tile_bricks=1, workers=2)
    for expected, actual in zip(serial, parallel):
        np.testing.assert_array_equal(actual, expected)


def write_export(path, coverage, cell_size=2.):
    brick = (4, 4, 4)
    table, bits, coords = sparse_coverage(coverage, brick)
    root = path / 'signed_normals_u8.respool'
    root.mkdir()
    np.save(root / 'table.npy', table)
    np.save(root / 'brick_coords.npy', coords)
    present = np.unpackbits(bits.view(np.uint8), axis=-1, bitorder='little', count=64).astype(bool)
    for i, value in enumerate([255, 128, 128]):
        np.where(present, value, 0).astype(np.uint8).tofile(root / f'channel_{i}.u8')
    meta = dict(format='respool', version=4, rows=len(bits), brick_shape=list(brick),
                array_shape=list(coverage.shape), channels=['nx', 'ny', 'nz'],
                channel_layout=[dict(dtype='|u1', file=f'channel_{i}.u8',
                                     shape=[len(bits), 64], order='C') for i in range(3)],
                normal_encoding=dict(name='signed_unit_vector_u8', components=['nx', 'ny', 'nz'],
                                     offset=128, scale=127, missing_vector=[0, 0, 0]),
                source_metadata=dict(cell_size_fitter_voxels=cell_size))
    (root / 'meta.json').write_text(json.dumps(meta))
    (path / 'manifest.json').write_text(json.dumps(dict(
        artifact_type='signed_patch_normal_volume', format_version=1, complete=True,
        cell_size_fitter_voxels=cell_size, shape_zyx=list(coverage.shape))))
    return root


@pytest.mark.parametrize('dtype', [torch.float32, torch.bool])
def test_exact_patch_near_excluded_far_fallback_and_bounds(tmp_path, dtype):
    coverage = np.zeros((16, 16, 20), dtype=bool)
    coverage[8, 8, 8] = True
    write_export(tmp_path, coverage)
    store = load_patch_normals(tmp_path, z_begin=0, z_end=32, device='cpu')
    # Centers: exact, 8 vx away, 10 vx away, diagonal sqrt(72)>8, out of bounds.
    points = (torch.tensor([[8, 8, 8], [8, 8, 12], [8, 8, 13], [8, 11, 11], [-1, 8, 8]]) + .5) * 2
    fallback = torch.tensor([0., 1., 0.]).expand(5, 3)
    weights = torch.full((5,), .5 if dtype == torch.float32 else True, dtype=dtype)
    try:
        normal, weight, present = override_patch_normals(
            {'patch_normals': store}, points, fallback, weights, return_presence=True)
        assert present.tolist() == [True, False, False, False, False]
        assert weight.tolist() == [1, 0, weights[2].item(), weights[3].item(), weights[4].item()]
        torch.testing.assert_close(normal[0], torch.tensor([0., 0., 1.]))
        torch.testing.assert_close(normal[1:], fallback[1:])  # no nearest-normal filling
        from sdt_losses import sample_lasagna_normals_nearest
        data = torch.zeros((3, 32, 32, 40), dtype=torch.uint8)
        data[0], data[1] = 128, 255
        volume = dict(backend='dense_test', volume=data, shape=(32, 32, 40),
                      lasagna_scale=1, z_origin=0, patch_normals=store)
        _, phase_valid = sample_lasagna_normals_nearest(volume, points)
        assert phase_valid.tolist() == [True, False, True, True, False]
        empty = store.sample(torch.empty((0, 3)), return_exclusion=True)
        assert [x.shape for x in empty] == [(0, 3), (0,), (0,)]
    finally:
        store.close()


def test_source_halo_outside_roi_and_zero_radius(tmp_path):
    coverage = np.zeros((16, 12, 12), dtype=bool)
    coverage[7, 5, 5] = True  # outside fit ROI, still near points just inside it
    write_export(tmp_path, coverage)
    points = (torch.tensor([[8, 5, 5], [7, 5, 5]]) + .5) * 2
    for radius in (8., 0.):
        store = load_patch_normals(tmp_path, z_begin=16, z_end=32, device='cpu', exclusion_radius=radius)
        try:
            _, present, near = store.sample(points, return_exclusion=True)
            assert present.tolist() == [False, False]
            assert near.tolist() == [radius > 0, False]
        finally:
            store.close()


def test_cache_reuse_and_invalidation(tmp_path, monkeypatch):
    import patch_normal_exclusion as module
    coverage = np.zeros((8, 8, 8), dtype=bool)
    coverage[3, 3, 3] = True
    root = write_export(tmp_path, coverage)
    cache = tmp_path / 'cache'
    calls = []
    build = module.build_exclusion_mask
    def counted(*args, **kwargs):
        calls.append(1)
        return build(*args, **kwargs)
    monkeypatch.setattr(module, 'build_exclusion_mask', counted)
    def load(radius=8., z_end=16):
        store = load_patch_normals(tmp_path, z_begin=0, z_end=z_end, device='cpu',
                                   exclusion_radius=radius, cache_directory=cache)
        store.close()
    load()
    load()
    assert len(calls) == 1
    load(radius=6.)
    load(z_end=12)
    assert len(calls) == 3
    channel = root / 'channel_0.u8'
    st = channel.stat()
    os.utime(channel, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    load()
    assert len(calls) == 4


def test_full_volume_cache_serves_different_z_range(tmp_path, monkeypatch):
    import patch_normal_exclusion as module
    coverage = np.zeros((16, 12, 12), dtype=bool)
    coverage[7, 5, 5] = True
    root = write_export(tmp_path, coverage)
    table, bits, _ = sparse_coverage(coverage, (4, 4, 4))
    cache_path = module.exclusion_cache_path(
        tmp_path / 'cache', root, shape=coverage.shape, brick=(4, 4, 4),
        z_roi=None, radius=4.)
    module.save_exclusion_mask(
        cache_path, *build_exclusion_mask(table, bits, (4, 4, 4), coverage.shape, 4.))
    full_bits = module.load_exclusion_mask(cache_path, table.shape, 1)[1]
    monkeypatch.setattr(module, 'build_exclusion_mask',
                        lambda *args, **kwargs: pytest.fail('full-volume cache was not reused'))
    store = load_patch_normals(tmp_path, z_begin=16, z_end=32, device='cpu',
                               cache_directory=tmp_path / 'cache')
    try:
        points = (torch.tensor([[8, 5, 5], [8, 11, 11]]) + .5) * 2
        _, present, near = store.sample(points, return_exclusion=True)
        assert present.tolist() == [False, False]
        assert near.tolist() == [True, False]
        assert len(store.cache.exclusion.bits) < len(full_bits)
    finally:
        store.close()


def test_exclusion_setting_is_rebuild_scoped():
    key = 'dense_normals_patch_exclusion_radius'
    assert Config().as_dict()[key] == 8.
    assert BACKFILLABLE_CONFIG_DEFAULTS[key] == 8.
    assert Config.catalog()['schema']['fields'][key]['runtime_impact'] == 'new_fit'
    with pytest.raises(ValueError):
        Config({key: -1.})


@pytest.mark.parametrize('all_excluded', [False, True])
def test_normal_loss_ignores_excluded_samples_including_denominator(monkeypatch, all_excluded):
    import losses
    class Identity:
        def inv(self, points):
            return points
    class Coverage:
        def sample(self, points, *, return_exclusion=False):
            normal = torch.zeros_like(points)
            normal[:, 2] = -1
            i = torch.arange(len(points)) % 3
            present = (i == 0) & ~torch.tensor(all_excluded)
            near = (i != 2) | torch.tensor(all_excluded)
            return (normal, present, near) if return_exclusion else (normal, present)
    data = torch.zeros((3, 8, 8, 8), dtype=torch.uint8)
    data[0], data[1] = 128, 255
    volume = dict(backend='dense_test', volume=data, shape=(8, 8, 8),
                  lasagna_scale=1, z_origin=0, y_origin=-4, x_origin=-4,
                  patch_normals=Coverage())
    prediction = torch.tensor([0., 0., 1.], requires_grad=True)
    monkeypatch.setattr(losses, 'get_radial_normal_in_scroll_space',
                        lambda transform, points, **kw: prediction.expand_as(points))
    cfg = Config().as_dict()
    cfg['dense_normals_patch_signed'] = True
    loss = dict(losses.iter_lasagna_losses(
        Identity(), torch.tensor(1.), volume, 2, 6, compute_spacing=False,
        cfg=cfg, z_begin=1, z_end=3))['dense_normals']
    torch.testing.assert_close(loss, torch.tensor(0. if all_excluded else .5))
    loss.backward()
    torch.testing.assert_close(prediction.grad[2], torch.tensor(0. if all_excluded else -.5))
