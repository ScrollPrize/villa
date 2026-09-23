"""Small direct-export round trips; no checkpoint or GPU needed."""
import numpy as np
import pytest
import torch

from patch_normal_compact_export import CompactPatchNormalWriter
from patch_normals import load_patch_normals


@pytest.mark.parametrize('radius', [0., 2.])
def test_direct_compact_export_roundtrip(tmp_path, radius):
    output = tmp_path / 'patch-normals'
    writer = CompactPatchNormalWriter(output, [8, 4, 4], 2., 4., 2, {}, brick_edge=2)
    # Two nonadjacent tiles, including a normal with quantized zero components.
    positions = np.array([[1., 1., 1.], [3., 3., 3.], [9., 1., 1.],
                          [11., 3., 3.], [1., 3., 1.]], dtype=np.float32)
    normals = np.array([[0., 0., 1.], [1., 0., 0.], [0., -1., 0.],
                        [0., 0., -1.], [1., 0., 0.]], dtype=np.float32)
    for indices in ([0, 1, 4], [2, 3]):
        writer.append(dict(position_zyx=positions[indices], normal_zyx=normals[indices],
                           sign_valid=np.ones(len(indices), dtype=bool)))
    writer.finish(output, {})
    assert set(p.name for p in output.iterdir()) == {
        'table.npy', 'brick_coords.npy', 'bits.npy', 'prefix.npy',
        'offsets.npy', 'values.u8', 'meta.json',
    }

    store = load_patch_normals(output, z_begin=0, z_end=16, device='cpu',
                               exclusion_radius=radius, cache_directory=tmp_path / 'cache')
    assert store.cache.prepacked_cache_used
    grid = torch.from_numpy(np.indices((8, 4, 4)).reshape(3, -1).T.copy())
    got = store.cache.gather(grid).numpy()
    expected = np.zeros((8, 4, 4, 3), dtype=np.uint8)
    cells = np.floor(positions / 2).astype(int)
    expected[tuple(cells.T)] = np.rint(normals[:, ::-1] * 127).astype(np.int16) + 128
    np.testing.assert_array_equal(got, expected.reshape(-1, 3))
    _, present, near = store.sample(torch.from_numpy(positions), return_exclusion=True)
    assert present.all()
    assert near.all()
    if radius:
        assert len(list(output.glob('exclusion-*.npz'))) == 1
    store.close()


def test_direct_compact_export_rejects_duplicate_cells(tmp_path):
    writer = CompactPatchNormalWriter(tmp_path / 'out', [2, 2, 2], 2., 4., 2, {}, brick_edge=2)
    with pytest.raises(ValueError, match='duplicate'):
        writer.append(dict(position_zyx=np.array([[1., 1., 1.], [1.5, 1., 1.]]),
                           normal_zyx=np.array([[0., 0., 1.]] * 2),
                           sign_valid=np.array([True, True])))


def test_cpu_tiled_export_selects_direct_compact_writer(tmp_path, monkeypatch):
    import patch_normal_tiled as tiled

    monkeypatch.setattr(tiled, '_read_quads', lambda _work: np.empty((0, 4, 3)))

    def stage(directory, batches, **_kwargs):
        list(batches)
        directory.mkdir()
        (directory / '0_0_0.bin').write_bytes(b'')
        return np.asarray([[0, 0, 0]]), {}

    monkeypatch.setattr(tiled, 'stage_quads', stage)
    monkeypatch.setattr(tiled, 'pool_tile',
                        lambda _work: ((0, 0, 0), None, None))
    monkeypatch.setattr(tiled, 'filter_tile',
                        lambda *_args, **_kwargs: dict(
                            position_zyx=np.asarray([[1., 1., 1.]]),
                            normal_zyx=np.asarray([[0., 0., 1.]]),
                            sign_valid=np.asarray([True])))
    output = tmp_path / 'patch-normals'
    tiled.export_filtered([tmp_path], output, {}, coordinate_scale=1., z_roi=(0, 4),
                          cell_size=2., surface_spacing=1., width=3, tile_edge=4,
                          workers=1, device='cpu', chunk_size=100, metadata={},
                          output_format='compact')
    store = load_patch_normals(output, z_begin=0, z_end=4, device='cpu', exclusion_radius=0)
    assert store.cache.gather(torch.tensor([[0, 0, 0]])).tolist() == [[255, 128, 128]]
    store.close()
