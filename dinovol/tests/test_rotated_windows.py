"""Rotated source windows: reads, label reads, mesh crops and full samples in the rotated frame."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dinovol_2.dataset.object_labels import MeshSegment, crop_mesh_objects
from dinovol_2.dataset.ssl_zarr_dataset import SSLZarrDataset, SourcePlan
from dinovol_2.dataset.volume_reader import open_volume

from test_cuda_pipeline import collate, write_array

# Window z, y, x step along volume -x, +y, +z: a proper rotation (determinant 1).
QUARTER_TURN = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def plane_segment(z=20.0, n=60):
    u, v = np.meshgrid(np.arange(n, dtype=np.float32), np.arange(n, dtype=np.float32), indexing="ij")
    zyx = np.stack([np.full_like(u, z), 5 + 0.5 * u, 5 + 0.5 * v], axis=-1)
    valid = np.ones((n, n), dtype=bool)
    seg = MeshSegment(segment_id="plane", zyx=zyx, valid=valid, ink_state=np.zeros((n, n), np.int8),
                      bbox_lo=zyx.reshape(-1, 3).min(0), bbox_hi=zyx.reshape(-1, 3).max(0), has_ink=False)
    seg.ensure_spatial_index()
    return seg


class MeshFrameTests(unittest.TestCase):
    def test_identity_frame_matches_axis_aligned_crops(self):
        start, lo, hi = np.array([12.0, 8.0, 8.0]), np.array([4.0, 2.0, 2.0]), np.array([12.0, 20.0, 20.0])
        plain, _ = crop_mesh_objects([plane_segment()], start + lo, start + hi, rng=np.random.default_rng(0))
        framed, _ = crop_mesh_objects([plane_segment()], lo, hi, frame=(start, np.eye(3)), rng=np.random.default_rng(0))
        self.assertEqual(len(plain), 1)
        np.testing.assert_allclose(framed[0][0], plain[0][0], atol=1e-5)

    def test_rotated_frame_maps_points_into_the_window(self):
        origin = np.array([0.0, 0.0, 30.0])
        lo, hi = np.array([0.0, 4.0, 10.0]), np.array([24.0, 30.0, 30.0])
        sheets, _ = crop_mesh_objects([plane_segment(z=20.0)], lo, hi, frame=(origin, QUARTER_TURN),
                                      rng=np.random.default_rng(0))
        self.assertEqual(len(sheets), 1)
        coords = sheets[0][0] + lo
        # Window x steps along volume z, so the plane z = 20 lies at window x = 20 - origin z.
        np.testing.assert_allclose(coords[:, 2], 20.0 - origin[0], atol=1e-4)
        volume = origin + coords @ QUARTER_TURN.T
        np.testing.assert_allclose(volume[:, 0], 20.0, atol=1e-4)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class RotatedWindowTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        z, y, x = np.meshgrid(*(np.arange(64),) * 3, indexing="ij")
        self.raw = (x + 2 * y + 3 * z).astype(np.uint8)
        write_array(root / "volume", self.raw)
        write_array(root / "volume", self.raw[::2, ::2, ::2], level=1)
        surface = np.zeros_like(self.raw)
        surface[:, ::4] = 255
        write_array(root / "surface", surface)
        write_array(root / "surface", surface[::2, ::2, ::2], level=1)
        self.config = dict(datasets=[dict(volume_path=str(root / "volume"), volume_scale=0,
            surface_path=str(root / "surface"), pixel_size_um=8)], epoch_length=4, vol_trim_pct=1,
            source_sampling_size=[7, 32, 32], global_crop_size=[4, 16, 16], local_crop_size=[4, 8, 8],
            global_crop_scale=[.18, .45], local_crop_scale=[.023, .057], num_local_crops=2,
            normalization_scheme="robust", augmentation="full", occupancy_level=1, source_rotation="uniform",
            cross_view_token_patch_size=[4] * 3,
            object_points=dict(enabled=True, use_metadata_meshes=False, points_per_sample=16,
                               points_per_object=8, max_objects=4))
        self.root = root

    def tearDown(self):
        self.folder.cleanup()

    def test_rotated_reads_follow_the_frame(self):
        dataset = SSLZarrDataset(copy.deepcopy(self.config))
        try:
            volume = dataset.volumes[0]
            array = dataset._get_volume_array(volume)
            origin = (10.0, 10.0, 40.0)
            window = dataset._read_plan_window(SourcePlan(volume, array, origin, None, axes=QUARTER_TURN))
            u = np.stack(np.meshgrid(*(np.arange(n) for n in dataset.source_sampling_size), indexing="ij"), -1)
            p = np.asarray(origin) + u @ QUARTER_TURN.T
            expected = self.raw[tuple(p[..., axis].astype(int) for axis in range(3))]
            np.testing.assert_array_equal(window, expected)
            np.testing.assert_allclose(dataset._to_window(SourcePlan(volume, array, origin, None, axes=QUARTER_TURN), p), u)
            # The affine label path with an identity frame matches direct and coarse reads.
            starts, lo, hi = (5, 6, 7), np.array([1, 2, 3]), np.array([6, 20, 25])
            for path, level, factor in (("surface", 0, 1), ("surface", 1, 2)):
                labels = dataset._get_array(str(self.root / path), level)
                plain = dataset._read_plan_region(SourcePlan(volume, array, starts, None), labels, lo, hi, factor)
                framed = dataset._read_plan_region(SourcePlan(volume, array, starts, None, axes=np.eye(3)), labels, lo, hi, factor)
                np.testing.assert_array_equal(framed, plain)
        finally:
            dataset.close()

    def test_rotated_samples_through_the_loader(self):
        from dinovol_2.dataset.cuda_pipeline import BatchDataLoader
        dataset = SSLZarrDataset(copy.deepcopy(self.config), do_augmentations=True)
        try:
            plans = [dataset._draw_source_plan(0.0) for _ in range(20)]
            self.assertTrue(all(plan.axes is not None for plan in plans))
            for plan in plans:
                np.testing.assert_allclose(plan.axes @ plan.axes.T, np.eye(3), atol=1e-6)
                self.assertAlmostEqual(float(np.linalg.det(plan.axes)), 1.0, places=6)
            for result in BatchDataLoader(dataset, batch_size=2, collate_fn=collate((4, 16, 16))):
                self.assertEqual(tuple(result["collated_global_crops"].shape), (4, 1, 4, 16, 16))
                self.assertEqual(tuple(result["collated_local_crops"].shape), (4, 1, 4, 8, 8))
                self.assertTrue(torch.isfinite(result["global_token_targets"]).all())
                self.assertEqual(tuple(result["object_point_coordinates"].shape), (4, 16, 3))
        finally:
            dataset.close()
        with self.assertRaisesRegex(ValueError, "point_supervision"):
            SSLZarrDataset({**copy.deepcopy(self.config), "point_supervision": {"enabled": True}})


if __name__ == "__main__":
    unittest.main()
