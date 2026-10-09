import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import detectability_probe as dp  # noqa: E402

UM = 40.0  # coarse, so a 200 px test window is 8 mm across and holds several lines


def sheet_stack(c=62, h=200, w=200):
    z = np.arange(c, dtype=np.float32)[:, None, None]
    s = 40 + 120 * np.exp(-0.5 * ((z - 31) / 4.0) ** 2)
    return (s + np.zeros((c, h, w), np.float32)).astype(np.uint8)


class DetectabilityProbeTests(unittest.TestCase):
    def test_a_detector_that_responds_gets_a_threshold(self):
        stack = sheet_stack()

        def sees(s):
            return np.clip((s[24:38].astype(np.float32).max(0) - 150.0) / 20.0, 0, 1)

        result = dp.probe(stack, sees, amplitudes=(8, 32, 64), micron_per_pixel=UM)
        self.assertIsNotNone(result["sensitivity"])
        self.assertLessEqual(result["sensitivity"], 64)

    def test_a_silent_detector_is_blind(self):
        stack = sheet_stack()
        rng = np.random.default_rng(3)
        noise = rng.random(stack.shape[1:]).astype(np.float32) * 0.2
        result = dp.probe(stack, lambda s: noise, amplitudes=(8, 32, 64), micron_per_pixel=UM)
        self.assertIsNone(result["sensitivity"])

    def test_output_that_looks_like_the_mask_without_reacting_is_blind(self):
        # the lift is measured against the same pixels before planting, so a detector whose
        # output already resembles the strokes cannot pass without responding to them
        stack = sheet_stack()
        looks_right = dp.script_mask(stack.shape[1:], UM) * 0.9
        result = dp.probe(stack, lambda s: looks_right, amplitudes=(8, 32, 64), micron_per_pixel=UM)
        self.assertIsNone(result["sensitivity"])
        self.assertTrue(all(row["lift"] == 0.0 for row in result["rows"]))

    def test_plant_puts_ink_on_the_requested_face(self):
        stack = sheet_stack()
        near, mask = dp.plant(stack, 60, "near", UM)
        far, _ = dp.plant(stack, 60, "far", UM)
        m = mask.astype(bool)
        dn = (near.astype(np.float32) - stack)[:, m].mean(1)
        df = (far.astype(np.float32) - stack)[:, m].mean(1)
        self.assertLess(int(np.argmax(dn)), int(np.argmax(df)))

    def test_window_too_small_for_two_lines_is_refused(self):
        with self.assertRaises(ValueError):
            dp.script_mask((40, 40), UM)


if __name__ == "__main__":
    unittest.main()
