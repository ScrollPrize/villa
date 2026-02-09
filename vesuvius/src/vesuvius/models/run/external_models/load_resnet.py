import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import torch


class _ScaleInputWrapper(torch.nn.Module):
    """Scale raw voxel intensities before forwarding into an external model."""

    def __init__(self, model: torch.nn.Module, scale: float = 255.0):
        super().__init__()
        self.model = model
        self.scale = float(scale)

    def forward(self, x):
        return self.model(x.float() / self.scale)


def _resolve_external_paths(model_path: Union[str, Path]) -> Optional[Tuple[Path, Path]]:
    path = Path(model_path)

    if path.is_file():
        if path.suffix != ".pth":
            return None
        model_file = path.parent / "ink_model.py"
        if not model_file.exists():
            return None
        return path, model_file

    if path.is_dir():
        model_file = path / "ink_model.py"
        if not model_file.exists():
            return None

        checkpoints = sorted(p for p in path.glob("*.pth") if p.is_file())
        if not checkpoints:
            raise ValueError(
                f"External model directory '{path}' contains ink_model.py but no .pth checkpoint."
            )
        if len(checkpoints) > 1:
            ckpt_names = ", ".join(p.name for p in checkpoints)
            raise ValueError(
                f"External model directory '{path}' has multiple .pth files ({ckpt_names}). "
                "Pass --model_path as a specific .pth file."
            )
        return checkpoints[0], model_file

    return None


def try_load_external_resnet34_model(
    model_path: Union[str, Path],
    device: torch.device,
    patch_size=None,
    verbose: bool = False,
):
    """Try loading an external InkDetectionModel checkpoint.

    Returns a model_info dict compatible with Inferer on success, else None.
    """
    resolved = _resolve_external_paths(model_path)
    if resolved is None:
        return None

    checkpoint_path, model_file = resolved

    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        return None

    if not isinstance(state_dict, dict):
        return None

    required_keys = {"backbone.conv1.weight", "decoder.logit.weight"}
    if not required_keys.issubset(state_dict.keys()):
        return None

    if patch_size is not None:
        if len(patch_size) != 3 or len(set(patch_size)) != 1:
            raise ValueError(
                f"Ink model requires cubic 3D patch_size, got {patch_size}. "
                "Use --patch_size 256,256,256 (or another equal-dimension cube)."
            )
        cube_size = int(patch_size[0])
    else:
        cube_size = 256

    module_name = f"vesuvius_external_ink_model_{checkpoint_path.stem}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, str(model_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load model module from: {model_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "InkDetectionModel"):
        raise AttributeError(f"Expected `InkDetectionModel` in {model_file}")

    model = module.InkDetectionModel(size=cube_size)
    model.load_state_dict(state_dict, strict=True)
    model = _ScaleInputWrapper(model, scale=255.0).to(device)

    if device.type == "cuda":
        try:
            if verbose:
                print("Compiling external ResNet model with torch.compile")
            model = torch.compile(model, mode="default")
        except Exception as e:
            if verbose:
                print(f"torch.compile failed for external ResNet model, using eager mode: {e}")

    if verbose:
        print(f"Loaded external ResNet model from {checkpoint_path} with cube size {cube_size}")

    return {
        "network": model,
        "patch_size": (cube_size, cube_size, cube_size),
        "num_input_channels": 1,
        "num_seg_heads": 1,
        "model_config": {"model_name": "InkDetectionModel"},
        "targets": {},
        "normalization_scheme": "none",
        "resolved_checkpoint_path": str(checkpoint_path),
    }
