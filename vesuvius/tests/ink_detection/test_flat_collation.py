"""Flat collation preserves targets and leaves native loaders unchanged."""
import unittest

import torch
from torch.utils.data import DataLoader, default_collate

from vesuvius.ink_detection.data.collation import ink_collate_fn


class FlatCollationTests(unittest.TestCase):
    def samples(self, depth=17, dtype=torch.float32):
        torch.manual_seed(42)
        return [
            {
                "image": torch.randn(1, depth, 9, 11),
                "inklabels": torch.randint(0, 2, (1, depth, 9, 11)).to(dtype),
                "supervision_mask": torch.randint(0, 2, (1, depth, 9, 11)).to(dtype),
                "surface_mask": torch.ones(1, depth, 9, 11),
                "segment_name": f"segment-{index}",
            }
            for index in range(2)
        ]

    def test_flat_loader_preserves_projected_targets_and_auxiliary_fields(self):
        for depth in (1, 3, 17):
            for dtype in (torch.float32, torch.float64, torch.uint8, torch.int64):
                with self.subTest(depth=depth, dtype=dtype):
                    samples = self.samples(depth, dtype)
                    snapshots = [{k: v.clone() for k, v in s.items() if torch.is_tensor(v)} for s in samples]
                    before = default_collate(samples)
                    after = next(iter(DataLoader(samples, batch_size=2, collate_fn=ink_collate_fn("flat"))))
                    for key in ("inklabels", "supervision_mask"):
                        self.assertTrue(torch.equal(before[key].amax(2), after[key].amax(2)))
                        self.assertEqual(after[key].shape[2], 1)
                        self.assertEqual(before[key].dtype, after[key].dtype)
                    for key in ("image", "surface_mask"):
                        self.assertTrue(torch.equal(before[key], after[key]))
                    self.assertEqual(before["segment_name"], after["segment_name"])
                    for sample, snapshot in zip(samples, snapshots):
                        for key, value in snapshot.items():
                            self.assertTrue(torch.equal(sample[key], value))

    def test_native_loaders_keep_the_complete_depth(self):
        samples = self.samples()
        expected = default_collate(samples)
        for mode in ("full_3d", "full_3d_single_wrap"):
            with self.subTest(mode=mode):
                result = next(iter(DataLoader(samples, batch_size=2, collate_fn=ink_collate_fn(mode))))
                for key in ("image", "inklabels", "supervision_mask", "surface_mask"):
                    self.assertTrue(torch.equal(expected[key], result[key]))
                    self.assertEqual(result[key].shape[2], 17)

    def test_post_augmentation_nan_and_ignored_targets_keep_loss_semantics(self):
        samples = self.samples(depth=3)
        samples[0]["inklabels"][0, 0, 0, 0] = float("nan")
        samples[1]["supervision_mask"].zero_()
        for sample in samples:
            for key in ("image", "inklabels", "supervision_mask", "surface_mask"):
                sample[key] = sample[key].transpose(1, 2).flip(3)
        before = default_collate(samples)
        after = next(iter(DataLoader(samples, batch_size=2, collate_fn=ink_collate_fn("flat"))))
        self.assertTrue(torch.equal(before["inklabels"].amax(2) > 0, after["inklabels"].amax(2) > 0))
        self.assertTrue(torch.equal(before["supervision_mask"].amax(2) <= 0, after["supervision_mask"].amax(2) <= 0))


if __name__ == "__main__":
    unittest.main()
