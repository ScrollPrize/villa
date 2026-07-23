import sys
import unittest
from pathlib import Path


LASAGNA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LASAGNA_ROOT))

from train_unet_3d import infer_model_patch_size


class CheckpointArchitectureTests(unittest.TestCase):
    def test_old_checkpoint_architecture_is_inferred_from_encoder_depth(self):
        state_dict = {
            "shared_encoder.stages.0.blocks.0.weight": object(),
            "shared_encoder.stages.6.blocks.0.weight": object(),
            "shared_decoder.stages.5.weight": object(),
        }

        self.assertEqual(infer_model_patch_size(state_dict), 256)

    def test_architecture_inference_rejects_unrecognized_state_dict(self):
        self.assertIsNone(
            infer_model_patch_size({"some_other_model.weight": object()})
        )


if __name__ == "__main__":
    unittest.main()
