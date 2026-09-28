import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import orientation_check as oc  # noqa: E402

UM = 40.0  # coarse, so a 200 px test window is 8 mm across and holds several lines


def sheet_stack(c=62, h=200, w=200):
    """A sheet brighter on its near side -- the orientation the fake model knows."""
    z = np.arange(c, dtype=np.float32)[:, None, None]
    s = 40 + 120 * np.exp(-0.5 * ((z - 31) / 4.0) ** 2) + 30 * (z < 31)
    return (s + np.zeros((c, h, w), np.float32)).astype(np.uint8)


def fake_model(clean):
    """Reads ink on the near face only, and only in the sheet's own orientation.

    Those are the two behaviours measured on the real checkpoint: the wrong layer order answers
    with nothing, and ink where the model expects none does not count as ink.
    """
    def predict(s, reverse):
        ref = (clean[::-1] if reverse else clean).astype(np.float32)
        v = (s[::-1] if reverse else s).astype(np.float32)
        if v[18:31].mean() <= v[32:45].mean():
            return np.zeros(v.shape[1:], np.float32)
        return np.clip((v[22:27] - ref[22:27]).max(0) / 20.0, 0, 1)
    return predict


class OrientationTests(unittest.TestCase):
    def test_verdict_follows_the_data_not_the_planted_face(self):
        stack = sheet_stack()
        flipped = np.ascontiguousarray(stack[::-1])
        a = oc.decide(stack, fake_model(stack), amplitude=64, micron_per_pixel=UM)
        b = oc.decide(flipped, fake_model(flipped), amplitude=64, micron_per_pixel=UM)
        self.assertEqual(a["verdict"], ("forward", "near"))
        self.assertEqual(b["verdict"], ("reverse", "far"))

    def test_a_blind_model_gets_no_verdict(self):
        stack = sheet_stack()

        def blind(s, reverse):
            return np.zeros(s.shape[1:], np.float32)

        result = oc.decide(stack, blind, amplitude=64, micron_per_pixel=UM)
        self.assertIsNone(result["verdict"])
        for v in result["combinations"].values():
            self.assertEqual(v["lift"], 0.0)

    def test_plant_puts_ink_on_the_requested_face(self):
        stack = sheet_stack()
        near, mask = oc.plant(stack, 60, "near", UM)
        far, _ = oc.plant(stack, 60, "far", UM)
        m = mask.astype(bool)
        dn = (near.astype(np.float32) - stack)[:, m].mean(1)
        df = (far.astype(np.float32) - stack)[:, m].mean(1)
        self.assertLess(int(np.argmax(dn)), int(np.argmax(df)))

    def test_window_too_small_for_two_lines_is_refused(self):
        with self.assertRaises(ValueError):
            oc.script_mask((40, 40), UM)


if __name__ == "__main__":
    unittest.main()
