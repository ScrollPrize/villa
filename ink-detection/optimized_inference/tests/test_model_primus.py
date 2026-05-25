import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import model_primus  # noqa: E402


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.seen_shape = None

    def forward(self, x):
        self.seen_shape = tuple(x.shape)
        return {"fiber": x + self.weight, "ink": x * self.weight}


class ModelPrimusTests(unittest.TestCase):
    def test_wrapper_preserves_optimized_inference_volume_shape(self):
        model = RecordingModel()
        wrapper = model_primus.PrimusWrapper(model, torch.device("cpu"), target_key="ink")

        x = torch.ones(2, 1, 5, 8, 8)
        out = wrapper.forward(x)

        self.assertEqual(model.seen_shape, (2, 1, 5, 8, 8))
        self.assertEqual(tuple(out.shape), (2, 1, 5, 8, 8))

    def test_wrapper_adds_single_channel_for_4d_tiles(self):
        model = RecordingModel()
        wrapper = model_primus.PrimusWrapper(model, torch.device("cpu"), target_key="ink")

        wrapper.forward(torch.ones(2, 5, 8, 8))

        self.assertEqual(model.seen_shape, (2, 1, 5, 8, 8))

    def test_load_model_uses_checkpoint_config_and_state_dict(self):
        fake_module_name = "vesuvius.models.build.build_network_from_config"
        old_modules = {
            name: sys.modules.get(name)
            for name in (
                "vesuvius",
                "vesuvius.models",
                "vesuvius.models.build",
                fake_module_name,
            )
        }

        class FakeNetworkFromConfig(nn.Module):
            received_model_config = None

            def __init__(self, mgr):
                super().__init__()
                FakeNetworkFromConfig.received_model_config = dict(mgr.model_config)
                self.model = RecordingModel()

            def forward(self, x):
                return self.model(x)

        fake_module = types.ModuleType(fake_module_name)
        fake_module.NetworkFromConfig = FakeNetworkFromConfig
        sys.modules["vesuvius"] = types.ModuleType("vesuvius")
        sys.modules["vesuvius.models"] = types.ModuleType("vesuvius.models")
        sys.modules["vesuvius.models.build"] = types.ModuleType("vesuvius.models.build")
        sys.modules[fake_module_name] = fake_module

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_path = Path(tmpdir) / "production_model.pt"
                checkpoint = {
                    "config": {
                        "patch_size": [5, 8, 8],
                        "in_channels": 1,
                        "targets": {"ink": {"activation": "sigmoid"}},
                    },
                    "model_state_dict": {"model.weight": torch.ones(1)},
                }
                torch.save(checkpoint, checkpoint_path)

                wrapper = model_primus.load_model(str(checkpoint_path), torch.device("cpu"), num_frames=5)
                out = wrapper.forward(torch.ones(1, 1, 5, 8, 8))

                self.assertEqual(tuple(out.shape), (1, 1, 5, 8, 8))
                self.assertEqual(wrapper.target_key, "ink")
                self.assertEqual(
                    FakeNetworkFromConfig.received_model_config["train_patch_size"],
                    [5, 8, 8],
                )
        finally:
            for name, module in old_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


if __name__ == "__main__":
    unittest.main()
