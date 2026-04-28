import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import runtime_contracts  # noqa: E402


class RuntimeContractsTests(unittest.TestCase):
    def test_normalize_model_type_accepts_new_decoder_variant(self):
        self.assertEqual(
            runtime_contracts.normalize_model_type("ResNet3D-152-3D-Decoder"),
            "resnet3d-152-3d-decoder",
        )

    def test_normalize_model_type_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            runtime_contracts.normalize_model_type("resnet3d-101")

    def test_cpu_image_rejects_inference(self):
        with self.assertRaisesRegex(RuntimeError, "GPU image target"):
            runtime_contracts.validate_image_role_for_step("inference", "cpu")

    def test_cpu_image_allows_utility_steps(self):
        for step in ("prepare", "reduce", "aggregate-profiling"):
            runtime_contracts.validate_image_role_for_step(step, "cpu")


if __name__ == "__main__":
    unittest.main()
