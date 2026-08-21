import sys
import unittest
from pathlib import Path


SPIRAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPIRAL_DIR))

import render_ink


class RenderInkPathTests(unittest.TestCase):
    def test_default_lasagna_dir_is_sibling_of_spiral_fitting(self):
        script = Path("/checkout/spiral-fitting/render_ink.py")

        actual = Path(render_ink.default_lasagna_dir(script))

        self.assertEqual(actual, Path("/checkout/lasagna"))


if __name__ == "__main__":
    unittest.main()
