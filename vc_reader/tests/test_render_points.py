"""Prepared geometry uses the same native renderer as surface and plane rasters."""
import numpy as np
import pytest

from build_support import load_build
load_build()

from vc_reader.reader import ZarrWriter, open_zarr


@pytest.mark.parametrize("fill", [-7, 7])
def test_prepared_geometry_layers_masks_and_volume_fill(tmp_path, fill):
    z, y, x = np.mgrid[:8, :9, :10]
    values = (100 * z + 10 * y + x).astype(np.float32)
    path = tmp_path / "ct.zarr"
    ZarrWriter.create(path, shape=values.shape, chunks=(3, 4, 5), dtype="f4", fill_value=fill)[:] = values
    reader = open_zarr(path)
    # Singleton row, isolated valid points, anisotropic level placement, and
    # directions whose length is meaningful (not renormalized by the renderer).
    xyz = np.array([[[3, 4, 8], [4, 6, 12], [100, 100, 100], [2, 2, 4]]], np.float32)
    normals = np.zeros_like(xyz)
    normals[..., 2] = 4
    valid = np.array([[True, False, True, True]])
    rendered = reader.render_points(xyz, normals, valid, offsets=(-1, 0, 1), scale_zyx=(.25, .5, 1))
    np.testing.assert_allclose(rendered[0, [0, 2, 3]], [323, np.nan if fill < 0 else fill, 212])
    assert np.isnan(rendered[0, 1])
    assert rendered.dtype == np.float32


def test_prepared_fractional_positions_and_default_offset(tmp_path):
    values = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
    path = tmp_path / "ct.zarr"
    ZarrWriter.create(path, shape=values.shape, chunks=(2, 2, 2), dtype="u1")[:] = values
    reader = open_zarr(path)
    xyz = np.array([[[1.5, 1.5, 1.5], [4, 4, 4], [np.nan, 1, 1]]], np.float32)
    result = reader.render_points(xyz, np.zeros_like(xyz), np.ones((1, 3), bool))
    np.testing.assert_allclose(result[0, :2], [31.5, 0])
    assert np.isnan(result[0, 2])
    with pytest.raises(ValueError, match="H,W"):
        reader.render_points(xyz, np.zeros((1, 2, 3)), np.ones((1, 3), bool))


def test_prepared_geometry_spans_native_tiles(tmp_path):
    z, y, x = np.mgrid[:4, :4, :4]
    path = tmp_path / "ct.zarr"
    ZarrWriter.create(path, shape=z.shape, chunks=(2, 2, 2), dtype="f4")[:] = 100 * z + 10 * y + x
    row, col = np.mgrid[:67, :69].astype(np.float32)
    xyz = np.stack((col / 40, row / 40, np.ones_like(row)), -1)
    result = open_zarr(path).render_points(xyz, np.zeros_like(xyz), np.ones(row.shape, bool))
    np.testing.assert_allclose(result, 100 + row / 4 + col / 40, atol=2e-5)


def _pyramid(tmp_path):
    """Level 0: 16^3 uint8 x + 2y + 3z in 8^3 chunks; level 1: 8^3 of 200."""
    from vc_reader.reader import create_group, open_group
    root = tmp_path / "vol.zarr"
    create_group(root)
    z, y, x = np.mgrid[:16, :16, :16]
    ZarrWriter.create(root / "0", shape=(16, 16, 16), chunks=(8, 8, 8), dtype="u1")[:] = (x + 2 * y + 3 * z).astype(np.uint8)
    ZarrWriter.create(root / "1", shape=(8, 8, 8), chunks=(8, 8, 8), dtype="u1")[:] = 200
    return open_group(root, memory_cache=4 << 20)


def test_grid_views_choose_levels_and_draw_progressive_frames(tmp_path):
    from vc_reader import grid
    from vc_reader.tifxyz import Surface
    group = _pyramid(tmp_path)
    levels = [group.level(i).grid_level() for i in range(2)]
    assert grid.voxel_extent(levels[1]) == (2, 2, 2)
    assert grid.level_for_view(levels, 1.0) == 0 and grid.level_for_view(levels, 0.25) == 1
    assert grid.fallback_levels(levels, 0, 16, 16, 1.0) == 1

    # A z plane at z = 4 over x 2..9, y 3..6: nothing cached, then the reads complete it.
    plane = ((2, 3, 4), (1, 0, 0), (0, 1, 0), (0, 0, 1), 8, 4)
    pending = grid.Pending()
    frame = grid.sample_grid_cached(levels, plane=plane, pending=pending)
    assert frame.values.shape == (1, 4, 8) and (frame.level == grid.LEVEL_MISSING).all()
    assert pending.reads == (1, 1)
    parts = []
    assert grid.load(pending, lambda: parts.append(1) or True) == "" and len(parts) == 2
    pending.clear()
    frame = grid.sample_grid_cached(levels, plane=plane, pending=pending)
    yy, xx = np.mgrid[3:7, 2:10]
    assert pending.empty and (frame.level == 0).all()
    np.testing.assert_array_equal(frame.values[0], xx + 2 * yy + 12)

    # Layers reduced by a composite; a pixel mask; a tifxyz surface raster.
    top = grid.sample_grid(levels[:1], plane=plane, offsets=(0, 1), composite="max")
    np.testing.assert_array_equal(top.values[0], xx + 2 * yy + 15)
    mask = np.zeros((4, 8), np.uint8)
    mask[1, 1] = 1
    masked = grid.sample_grid(levels[:1], plane=plane, pixels=mask)
    assert masked.level[1, 1] == 0 and masked.level[0, 0] == grid.LEVEL_NONE
    with pytest.raises(ValueError, match="height, width"):
        grid.sample_grid(levels[:1], plane=plane, pixels=np.ones((2, 2), np.uint8))
    r, c = np.mgrid[:4, :6].astype(np.float32)
    surface = Surface(np.stack((2 * c, 3 * r, np.full_like(r, 10)), -1))
    raster = (0.5, 0.5, 0.25, 0.25, 8, 4)
    assert grid.pixels_per_voxel(surface, raster) == pytest.approx(2.0)
    along = grid.sample_grid(levels, surface=surface, raster=raster)
    assert along.values[0, 0, 0] == pytest.approx(1 + 2 * 1.5 + 30, abs=1e-3)


def test_grid_views_on_the_gpu_match_the_cpu(tmp_path):
    from vc_reader import cuda, grid
    if not cuda.is_available():
        pytest.skip("CUDA build/device required")
    group = _pyramid(tmp_path)
    plane = ((2.5, 3.25, 4.5), (1, 0, 0), (0, 1, 0), (0, 0, 1), 8, 4)
    readers = [group.level(i) for i in range(2)]
    # One shared cache of two bricks: the preview's level-1 brick makes room for the
    # two level-0 bricks the frame needs.
    pool = grid.CudaPool(2 * 32 ** 3)
    cudas = [r.cuda(pool=pool) for r in readers]
    on_gpu = [r.grid_level(cuda=c) for r, c in zip(readers, cudas)]
    assert on_gpu[0].gpu
    pending = grid.Pending()
    grid.sample_grid_cached(on_gpu, plane=plane, pending=pending)
    assert grid.load(pending) == ""
    assert 0 < pool.resident_bytes <= pool.cache_bytes
    gpu = grid.sample_grid_cached(on_gpu, plane=plane)
    cpu = grid.sample_grid([r.grid_level() for r in readers], plane=plane)
    assert gpu.cuda_failure == "" and (gpu.level == 0).all()
    np.testing.assert_allclose(gpu.values, cpu.values, atol=1e-3)
