from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import aiohttp
import numpy as np
import zarr

from dinovol_2.augmentation.transforms.noise.gaussian_blur import GaussianBlurTransform
from dinovol_2.dataset.ssl_zarr_dataset import SSLZarrDataset


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
