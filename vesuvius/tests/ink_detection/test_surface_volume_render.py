from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tifffile
import zarr

from vesuvius.ink_detection.preprocessing.render_surface_volume import (
    ChunkReader,
    level_surface,
    plane_offsets,
    render_surface_volume,
    render_tile,
)
from vesuvius.tifxyz_label_transfer.core import Surface


class _ChunkedArray:
    """Minimal zarr-like array: shape, chunks and slicing over a numpy volume."""

    def __init__(self, data: np.ndarray, chunks: tuple[int, int, int]) -> None:
        self.data = data
        self.shape = data.shape
        self.chunks = chunks
        self.reads = 0

    def __getitem__(self, key):
        self.reads += 1
        return self.data[key]


def linear_volume(shape=(24, 40, 40)) -> np.ndarray:
    z, y, x = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    return (2 * (z + y + x)).astype(np.uint8)


def flat_surface(height: int, width: int, z: float = 10.0) -> Surface:
    rows, cols = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    return Surface(x=cols + 4, y=rows + 4, z=np.full((height, width), z, np.float32))


class PlaneOffsetTests(unittest.TestCase):
    def test_offsets_descend_symmetrically_like_published_volumes(self) -> None:
        offsets = plane_offsets(28)
        self.assertEqual(offsets[0], 13.5)
        self.assertEqual(offsets[-1], -13.5)
        np.testing.assert_array_equal(np.diff(offsets), -1.0)
        np.testing.assert_array_equal(plane_offsets(1), [0.0])


class RenderTileTests(unittest.TestCase):
    def test_planes_sample_along_the_surface_normal(self) -> None:
        volume = linear_volume()
        surface = flat_surface(20, 24)
        offsets = plane_offsets(7)
        reader = ChunkReader(_ChunkedArray(volume, (8, 8, 8)), cache_chunks=2, workers=2)

        tile = render_tile(surface, (20, 24), (0, 20, 0, 24), offsets, reader)

        # Canvas pixels sample the mesh at their centres (x = col + 4.5), and the
        # normal of x=col, y=row is +z, so plane j sits at z = 10 + offsets[j].
        # Interior pixels only: normals are clamped one grid cell in from the edge.
        rows, cols = np.mgrid[2:17, 2:21]
        for j, offset in enumerate(offsets):
            expected = 2 * ((10 + offset) + (rows + 4.5) + (cols + 4.5))
            np.testing.assert_array_equal(tile[j, 2:17, 2:21], expected)

    def test_half_voxel_planes_interpolate_trilinearly(self) -> None:
        volume = linear_volume()
        surface = flat_surface(20, 24)
        offsets = plane_offsets(6)
        reader = ChunkReader(_ChunkedArray(volume, (8, 8, 8)), cache_chunks=64, workers=2)

        tile = render_tile(surface, (20, 24), (0, 20, 0, 24), offsets, reader)

        rows, cols = np.mgrid[2:17, 2:21]
        for j, offset in enumerate(offsets):
            expected = 2 * ((10 + offset) + (rows + 4.5) + (cols + 4.5))
            np.testing.assert_array_equal(tile[j, 2:17, 2:21], expected)

    def test_pixels_off_the_mesh_stay_zero(self) -> None:
        surface = flat_surface(20, 24)
        surface.x[:, 12:] = -1
        surface = Surface(x=surface.x, y=surface.y, z=surface.z)
        reader = ChunkReader(_ChunkedArray(linear_volume(), (8, 8, 8)), cache_chunks=64, workers=2)

        tile = render_tile(surface, (20, 24), (0, 20, 0, 24), plane_offsets(3), reader)

        self.assertTrue(np.all(tile[:, :, 12:] == 0))
        self.assertTrue(np.all(tile[:, 2:17, 2:10] > 0))

    def test_only_chunks_near_the_surface_are_read(self) -> None:
        array = _ChunkedArray(linear_volume((64, 40, 40)), (8, 8, 8))
        reader = ChunkReader(array, cache_chunks=1024, workers=2)

        render_tile(flat_surface(20, 24), (20, 24), (0, 20, 0, 24), plane_offsets(3), reader)

        # z in [9, 12) touches chunk row 1 only; y, x in [4, 28) touch rows 0..3.
        self.assertEqual({key[0] for key in reader.cache}, {1})
        self.assertEqual(array.reads, len(reader.cache))


class LevelSurfaceTests(unittest.TestCase):
    def test_level_two_matches_a_2x2x2_mean_pyramid(self) -> None:
        z, y, x = np.meshgrid(*(np.arange(n) for n in (64, 96, 96)), indexing="ij")
        level0 = (z + y + x).astype(np.uint8)
        # Rounded 4x4x4 block means, as zarr_tasks builds integer pyramids.
        level2 = np.rint(
            level0.astype(np.float32).reshape(16, 4, 24, 4, 24, 4).mean(axis=(1, 3, 5))
        ).astype(np.uint8)
        surface = flat_surface(80, 80, z=30.0)
        reader0 = ChunkReader(_ChunkedArray(level0, (16, 16, 16)), cache_chunks=256, workers=2)
        reader2 = ChunkReader(_ChunkedArray(level2, (8, 8, 8)), cache_chunks=256, workers=2)
        offsets = plane_offsets(3)

        full = render_tile(surface, (80, 80), (0, 80, 0, 80), offsets * 4, reader0)
        small = render_tile(level_surface(surface, 2), (20, 20), (0, 20, 0, 20), offsets, reader2)

        # A linear field commutes with block means: level 2 equals the 4x4 mean of level 0.
        pooled = full.astype(np.float32).reshape(3, 20, 4, 20, 4).mean(axis=(2, 4))
        np.testing.assert_allclose(small[:, 2:18, 2:18], pooled[:, 2:18, 2:18], atol=1.0)
        # Without the voxel-centre shift every pixel would be off by 3 * 0.375 * 4 = 4.5.

    def test_invalid_vertices_stay_invalid(self) -> None:
        surface = flat_surface(20, 24)
        surface.x[:, 12:] = -1
        surface = Surface(x=surface.x, y=surface.y, z=surface.z)
        self.assertFalse(level_surface(surface, 1).valid[:, 12:].any())


class RenderSurfaceVolumeTests(unittest.TestCase):
    def test_writes_a_flat_inference_input_and_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh = root / "seg.tifxyz"
            mesh.mkdir()
            surface = flat_surface(10, 12)
            # Stored at half resolution: the canvas is 20 x 24.
            for name, field in (("x", surface.x), ("y", surface.y), ("z", surface.z)):
                stored = field.copy()
                if name in ("x", "y"):
                    stored = 4 + (stored - 4) * 2
                tifffile.imwrite(mesh / f"{name}.tif", stored)
            (mesh / "meta.json").write_text(json.dumps({"scale": [0.5, 0.5], "format": "tifxyz"}))
            scan = zarr.open_group(str(root / "scan.zarr"), mode="w")
            scan.create_array("0", data=linear_volume(), chunks=(8, 8, 8))

            out = render_surface_volume(
                str(mesh), str(root / "scan.zarr"), root / "out.zarr", depth=5, workers=2
            )

            rendered = zarr.open_group(str(out), mode="r")
            self.assertEqual(rendered["0"].shape, (5, 20, 24))
            self.assertEqual(rendered.attrs["plane_offsets"], [2.0, 1.0, 0.0, -1.0, -2.0])
            self.assertEqual(int(rendered["0"][2, 10, 10]), 2 * (10 + 14.5 + 14.5))
            self.assertFalse((root / "out.zarr.partial").exists())
            with self.assertRaises(FileExistsError):
                render_surface_volume(str(mesh), str(root / "scan.zarr"), out, depth=5)


if __name__ == "__main__":
    unittest.main()
