import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import model_primus  # noqa: E402

try:
    from vesuvius.models.build.build_network_from_config import NetworkFromConfig
except Exception as exc:  # pragma: no cover - environment-dependent skip path
    NetworkFromConfig = None
    VESUVIUS_IMPORT_ERROR = exc
else:
    VESUVIUS_IMPORT_ERROR = None


@unittest.skipIf(NetworkFromConfig is None, f"vesuvius package unavailable: {VESUVIUS_IMPORT_ERROR}")
class ModelPrimusIntegrationTests(unittest.TestCase):
    def test_loads_real_network_from_config_checkpoint_envelope(self):
        model_config = {
            "architecture_type": "primus_s",
            "patch_size": [16, 16, 16],
            "train_patch_size": [16, 16, 16],
            "input_shape": [16, 16, 16],
            "patch_embed_size": [8, 8, 8],
            "in_channels": 1,
            "targets": {"ink": {"out_channels": 1, "activation": "none"}},
            "decoder_head_channels": 4,
            "drop_path_rate": 0.0,
            "proj_drop_rate": 0.0,
            "attn_drop_rate": 0.0,
        }
        source_model = NetworkFromConfig(model_primus._PrimusConfigShim(model_config))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "production_model.pt"
            torch.save(
                {
                    "config": {
                        "model_config": model_config,
                        "targets": model_config["targets"],
                    },
                    "model_state_dict": source_model.state_dict(),
                },
                checkpoint_path,
            )

            wrapper = model_primus.load_model(str(checkpoint_path), torch.device("cpu"), num_frames=16)
            with torch.no_grad():
                out = wrapper.forward(torch.randn(1, 1, 16, 16, 16))

            self.assertEqual(wrapper.target_key, "ink")
            self.assertEqual(tuple(out.shape), (1, 1, 16, 16, 16))


if __name__ == "__main__":
    unittest.main()
