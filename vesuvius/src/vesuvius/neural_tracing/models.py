import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from vesuvius.models.build.build_network_from_config import NetworkFromConfig
from youssef_mae import Vesuvius3dViTModel


def _infer_spacing(config):
    """
    Try to read voxel spacing from the first dataset's meta.json. Falls back to
    an explicit config override or (1, 1, 1) if anything is missing.
    """
    # User override takes precedence
    explicit = config.get("spacing")
    if explicit is not None:
        try:
            spacing = tuple(float(s) for s in explicit)
            if len(spacing) == 3:
                return spacing
        except Exception:
            pass

    datasets = config.get("datasets") or []
    if not datasets:
        return (1.0, 1.0, 1.0)

    first = datasets[0]
    volume_path = first.get("volume_path")
    if not volume_path:
        return (1.0, 1.0, 1.0)

    meta_path = Path(volume_path).with_name("meta.json")
    try:
        meta = json.loads(meta_path.read_text())
        base_spacing = tuple(float(x) for x in meta.get("voxelsize", []))
        if len(base_spacing) != 3:
            raise ValueError("voxelsize not length 3")
        scale = int(first.get("volume_scale", 0))
        scaled = tuple(s * (2 ** scale) for s in base_spacing)
        return scaled
    except Exception:
        # Best-effort fallback if the meta file is missing or malformed
        return (1.0, 1.0, 1.0)


class NeuralTracingNet(nn.Module):
    """
    Thin wrapper around NetworkFromConfig so neural_tracing can use the main
    vesuvius model builder without pulling in ConfigManager.
    """

    def __init__(self, config):
        super().__init__()
        self.deep_supervision = bool(config.get("enable_deep_supervision", False))
        spacing = _infer_spacing(config)
        mgr = SimpleNamespace(
            model_name="neural_tracing",
            train_patch_size=(config["crop_size"],) * 3,
            train_batch_size=config["batch_size"],
            spacing=spacing,
            targets={
                "uv": {
                    "out_channels": config["step_count"] * 2,
                    "activation": "none",
                    "separate_decoder": True,
                }
            },
            model_config=config.get("model_config", {}),
            autoconfigure=False,
            enable_deep_supervision=self.deep_supervision,
            in_channels=5,
        )

        self.net = NetworkFromConfig.create_with_input_channels(mgr, input_channels=5)

    def forward(self, x, timesteps=None):
        # timesteps is accepted for backward compatibility; NetworkFromConfig ignores it
        out = self.net(x)["uv"]
        if not self.deep_supervision and isinstance(out, (list, tuple)) and len(out) > 0:
            out = out[0]
        return out


def make_model(config):

    if config['model_type'] == 'unet':
        return NeuralTracingNet(config)
    elif config['model_type'] == 'vit':
        return Vesuvius3dViTModel(
            mae_ckpt_path=config['model_config'].get('mae_ckpt_path', None),
            in_channels=5,
            out_channels=config['step_count'] * 2,
            input_size=config['crop_size'],
            patch_size=8,  # TODO: infer automatically from volume_scale and pretraining crop size
        )
    else:
        raise RuntimeError('unexpected model_type, should be unet or vit')


def load_checkpoint(checkpoint_path, model):
    print(f'loading checkpoint {checkpoint_path}... ')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
