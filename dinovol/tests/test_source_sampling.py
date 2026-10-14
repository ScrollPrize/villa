from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import aiohttp
import numpy as np
import zarr

from dinovol_2.augmentation.transforms.noise.gaussian_blur import GaussianBlurTransform
from dinovol_2.dataset.ssl_zarr_dataset import SSLZarrDataset, window_start_cells


def _flatten_transforms(transform):
    children = [transform]
    for child in getattr(transform, "transforms", None) or []:
        children += _flatten_transforms(child)
    for child in getattr(transform, "list_of_transforms", None) or []:
        children += _flatten_transforms(child)
    if getattr(transform, "transform", None) is not None:
        children += _flatten_transforms(transform.transform)
    return children


def _downsample(values: np.ndarray) -> np.ndarray:
    # Mean pooling with uint8 truncation, so sparse blocks can round to zero.
    d, h, w = (size // 2 for size in values.shape)
    return values[:2 * d, :2 * h, :2 * w].reshape(d, 2, h, 2, w, 2).mean(axis=(1, 3, 5)).astype(np.uint8)


def _make_masked_pyramid(root: Path) -> Path:
    rng = np.random.default_rng(0)
    level = np.zeros((96, 96, 96), dtype=np.uint8)
    level[40:80, 10:60, 20:90] = rng.integers(1, 256, size=(40, 50, 70))
    level[5, 5, 5] = 1  # isolated voxel that rounds away in coarser levels
    path = root / "masked.zarr"
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    for index in range(4):
        group.create_array(str(index), shape=level.shape, chunks=(16, 16, 16), dtype="uint8", fill_value=0)[:] = level
        level = _downsample(level)
    return path


def _make_scroll_pyramid(root: Path) -> Path:
    """A masked 'scroll': a z cylinder (radius 30) of wraps 3 voxels thick with 3-voxel air
    gaps, inside a 6-voxel nonzero air halo (the dilated SAM2 mask), zeros outside."""
    z, y, x = np.meshgrid(*[np.arange(96)] * 3, indexing="ij")
    r = np.hypot(y - 48.0, x - 48.0)
    level = np.where(r < 36, 20, 0).astype(np.uint8)
    level[(r < 30) & (r % 6 < 3)] = 200
    path = root / "scroll.zarr"
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    for index in range(4):
        group.create_array(str(index), shape=level.shape, chunks=(16, 16, 16), dtype="uint8", fill_value=0)[:] = level
        level = _downsample(level)
    return path


class ObjectMaskSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.zarr_path = _make_scroll_pyramid(Path(self.tempdir.name))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def dataset(self, **overrides) -> SSLZarrDataset:
        config = {
            "epoch_length": 8, "nonzero_threshold": 0.3, "occupancy_level": 2,
            "object_mask": {"enabled": True, "gap_um": 30.0, "z_trim_pct": 0.05},
            "global_crop_size": [16, 16, 16], "local_crop_size": [8, 8, 8],
            "source_sampling_size": [24, 24, 24], "num_local_crops": 1,
            "datasets": [{"volume_path": str(self.zarr_path), "volume_scale": 0, "pixel_size_um": 8.0}],
            **overrides,
        }
        return SSLZarrDataset(config, do_augmentations=False)

    def test_windows_stay_inside_the_object_and_the_z_limits(self):
        dataset = self.dataset()
        volume = dataset.volumes[0]
        array = dataset._get_volume_array(volume)
        np.random.seed(0)
        starts = np.array([dataset._draw_source_plan(0.3).starts for _ in range(400)])
        size = 24
        # z: never in the bottom or top 5% of the 96 slices
        self.assertGreaterEqual(starts[:, 0].min(), 5)
        self.assertLessEqual(starts[:, 0].max() + size, 96 - 5)
        # y, x: the farthest window voxel from the axis stays inside the scroll (radius 30),
        # up to one 4-voxel mask cell, and never reaches the zeros outside the SAM2 mask
        for z0, y0, x0 in starts:
            dy = max(abs(y0 - 48.0), abs(y0 + size - 1 - 48.0))
            dx = max(abs(x0 - 48.0), abs(x0 + size - 1 - 48.0))
            self.assertLess(np.hypot(dy, dx), 30 + 4)
            self.assertTrue(np.all(dataset._read_source_window(array, (z0, y0, x0)) > 0))
        # uniform over the allowed region: both z ends and all four sides are reached
        self.assertLess(starts[:, 0].min(), 10)
        self.assertGreater(starts[:, 0].max(), 91 - size - 6)
        for axis in (1, 2):
            self.assertLess(starts[:, axis].min(), 48 - 12 - 8)
            self.assertGreater(starts[:, axis].max(), 48 - 12 + 8)
        dataset.close()

    def test_samples_are_produced_and_weights_follow_valid_starts(self):
        dataset = self.dataset()
        volume = dataset.volumes[0]
        self.assertIsNone(volume.occupancy)
        self.assertEqual(volume.occupancy_factor, 4)
        self.assertGreater(volume.valid_crop_starts, 0)
        self.assertEqual(volume.weight, 1.0)
        sample = dataset[0]
        self.assertEqual(tuple(sample["global_views"][0].shape), (1, 16, 16, 16))
        dataset.close()

    def test_start_cells_match_a_voxel_exact_check(self):
        rng = np.random.default_rng(1)
        inside = rng.random((10, 12, 11)) < 0.85
        factor, window = 3, np.array([7, 5, 8])
        lo, hi = np.array([2, 0, 1]), np.array([28, 36, 30])
        cells = window_start_cells(inside, factor, window, lo, hi)
        voxels = inside.repeat(factor, 0).repeat(factor, 1).repeat(factor, 2)
        for c in np.ndindex(*inside.shape):
            exact = True
            for offset in np.ndindex(factor, factor, factor):
                s = np.asarray(c) * factor + np.asarray(offset)
                if np.any(s < lo) or np.any(s + window > hi) or not voxels[tuple(slice(a, a + w) for a, w in zip(s, window))].all():
                    exact = False
                    break
            self.assertEqual(bool(cells[c]), exact, c)


class SourceSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.zarr_path = _make_masked_pyramid(Path(self.tempdir.name))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def dataset(self, **overrides) -> SSLZarrDataset:
        config = {
            "epoch_length": 8, "vol_trim_pct": 1.0, "nonzero_threshold": 0.3,
            "global_crop_size": [16, 16, 16], "local_crop_size": [8, 8, 8],
            "source_sampling_size": [24, 24, 24], "num_local_crops": 1,
            "datasets": [{"volume_path": str(self.zarr_path), "volume_scale": 0}],
            **overrides,
        }
        return SSLZarrDataset(config, do_augmentations=False)

    def draw(self, dataset, count, seed=3):
        reads = []
        read = dataset._read_source_window

        def counting_read(array, starts):
            reads.append(tuple(starts))
            return read(array, starts)

        dataset._read_source_window = counting_read
        np.random.seed(seed)
        plans = [dataset._next_source_crop(0.3) for _ in range(count)]
        return [(plan.starts, crop) for plan, crop in plans], reads

    def test_occupancy_skips_rejected_reads_without_changing_samples(self):
        plain, plain_reads = self.draw(self.dataset(), 20)
        dataset = self.dataset(occupancy_level=3)
        guided, guided_reads = self.draw(dataset, 20)
        self.assertEqual(dataset.volumes[0].occupancy_factor, 8)
        self.assertEqual([starts for starts, _ in guided], [starts for starts, _ in plain])
        for (_, expected), (_, actual) in zip(plain, guided):
            np.testing.assert_array_equal(actual, expected)
        self.assertLess(len(guided_reads), len(plain_reads))

    def test_occupied_fraction_bounds_the_nonzero_fraction(self):
        dataset = self.dataset(occupancy_level=3)
        volume = dataset.volumes[0]
        array = dataset._get_volume_array(volume)
        np.random.seed(0)
        for _ in range(200):
            starts = dataset._sample_source_start(volume.usable_bbox)
            crop = dataset._read_source_window(array, starts)
            self.assertGreaterEqual(dataset._occupied_fraction(volume, starts) + 1e-12, np.count_nonzero(crop) / crop.size)
        dataset.close()

    def test_read_ahead_samples_pass_the_threshold(self):
        dataset = self.dataset(occupancy_level=3, read_ahead=2)
        for index in range(5):
            sample = dataset[index]
            self.assertEqual(tuple(sample["global_views"][0].shape), (1, 16, 16, 16))
            self.assertEqual(len(dataset._pending_reads), 2)
        plan, crop = dataset._next_source_crop(0.3)
        self.assertGreaterEqual(np.count_nonzero(crop) / crop.size, 0.3)
        dataset.close()
        self.assertIsNone(dataset._read_executor)
        self.assertEqual(len(dataset._pending_reads), 0)

    def test_transient_read_errors_are_retried(self):
        dataset = self.dataset(read_retry_backoff=0.0)
        array = dataset._get_volume_array(dataset.volumes[0])
        expected = dataset._read_source_window(array, (30, 10, 20))

        class FlakyArray:
            def __init__(self, failures):
                self.failures = failures
                self.calls = 0

            def __getitem__(self, selection):
                self.calls += 1
                if self.calls <= self.failures:
                    raise aiohttp.ClientPayloadError("Response payload is not completed")
                return array[selection]

        flaky = FlakyArray(failures=2)
        np.testing.assert_array_equal(dataset._read_source_window(flaky, (30, 10, 20)), expected)
        self.assertEqual(flaky.calls, 3)
        with self.assertRaises(aiohttp.ClientPayloadError):
            dataset._read_source_window(FlakyArray(failures=dataset.read_retries + 1), (30, 10, 20))

    def test_gaussian_blur_can_be_disabled(self):
        def has_blur(dataset):
            pipelines = dataset.global_transforms + dataset.local_transforms
            return [any(isinstance(t, GaussianBlurTransform) for t in _flatten_transforms(p)) for p in pipelines]

        self.assertTrue(all(has_blur(self.dataset())))
        self.assertFalse(any(has_blur(self.dataset(gaussian_blur=False))))


if __name__ == "__main__":
    unittest.main()
