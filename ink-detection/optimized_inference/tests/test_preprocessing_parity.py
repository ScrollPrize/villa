"""The optimized path must hand the model the same numbers training did.

Training (train_resnet3d_3d_decoder.py, train_timesformer_og.py) and the
reference inference paths (inference_resnet3d.py, inference_timesformer.py) all
clip to [0, 200] and then apply A.Normalize(mean=0, std=1), which divides by
albumentations' default max_pixel_value of 255.

The optimized path divided by CFG.max_clip_value instead, so every optimized
inference fed the model values 255/200 = 1.275x larger than anything it saw
while training. Nothing compared the two branches, so it went unnoticed.

The config constants are read with ast rather than imported: pinning a
preprocessing invariant should not require torch, zarr and tifffile.
"""

import ast
import unittest
from pathlib import Path

import albumentations as A
import numpy as np

INFERENCE_PY = Path(__file__).resolve().parents[1] / "inference.py"
IN_CHANS = 26
TRAINING_DIVISOR = 255.0  # albumentations' A.Normalize default max_pixel_value


def cfg_defaults(*names: str) -> dict:
    """Read dataclass field defaults out of inference.py without importing it."""
    tree = ast.parse(INFERENCE_PY.read_text())
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in names and node.value is not None:
                found[node.target.id] = ast.literal_eval(node.value)
    missing = set(names) - set(found)
    if missing:
        raise AssertionError(f"missing from inference.py CFG: {sorted(missing)}")
    return found


def reference_preprocess(tile: np.ndarray) -> np.ndarray:
    """What training and the reference inference paths do."""
    clipped = np.clip(tile, 0, 200)
    return A.Compose([A.Normalize(mean=[0] * IN_CHANS, std=[1] * IN_CHANS)])(
        image=clipped
    )["image"]


def optimized_preprocess(tile: np.ndarray, clip: int, divisor: float) -> np.ndarray:
    """What optimized_inference does, driven by the constants it ships with."""
    clipped = np.clip(tile, 0, clip)
    return A.Compose([A.ToFloat(max_value=divisor)])(image=clipped)["image"]


def _tile(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(64, 64, IN_CHANS), dtype=np.uint8)


class TestPreprocessingParity(unittest.TestCase):
    def setUp(self):
        cfg = cfg_defaults("max_clip_value", "normalize_divisor")
        self.clip = cfg["max_clip_value"]
        self.divisor = cfg["normalize_divisor"]

    def test_matches_reference_on_random_tiles(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                tile = _tile(seed)
                np.testing.assert_allclose(
                    optimized_preprocess(tile, self.clip, self.divisor),
                    reference_preprocess(tile),
                    rtol=0,
                    atol=1e-6,
                )

    def test_clip_ceiling_maps_below_one(self):
        """A voxel at the clip ceiling is 200/255, not 1.0.

        Dividing by the clip value instead sends it to exactly 1.0, which is the
        signature of the bug this test exists to catch.
        """
        tile = np.full((8, 8, IN_CHANS), 255, dtype=np.uint8)
        got = float(optimized_preprocess(tile, self.clip, self.divisor).max())
        self.assertAlmostEqual(got, 200.0 / TRAINING_DIVISOR, places=6)

    def test_divisor_is_not_tied_to_the_clip(self):
        """The two constants are independent, and must stay that way."""
        self.assertEqual(self.clip, 200)
        self.assertEqual(self.divisor, TRAINING_DIVISOR)


if __name__ == "__main__":
    unittest.main()
