"""
ResNet3D model loader for ink detection inference.
"""
import gc
import logging
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnetall import generate_model

logger = logging.getLogger(__name__)

BOTTLENECK_BLOCKS_BY_DEPTH = {
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
    200: (3, 24, 36, 3),
}


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_state_dict"), dict):
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format for ResNet3D inference")

    if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    # Clone tensors into plain contiguous CPU storage so checkpoint-backed storage
    # is not referenced during/after load_state_dict. This has been more stable on
    # cluster jobs than keeping the original loaded storages alive.
    cloned_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned_state_dict[key] = value.detach().cpu().contiguous().clone()
        else:
            cloned_state_dict[key] = value
    return cloned_state_dict


def _stage_block_count(state_dict: dict[str, torch.Tensor], layer_name: str) -> int:
    prefix = f"backbone.{layer_name}."
    indices = {
        int(parts[2])
        for key in state_dict
        if key.startswith(prefix)
        for parts in [key.split(".")]
        if len(parts) > 2 and parts[2].isdigit()
    }
    return (max(indices) + 1) if indices else 0


def infer_resnet_depth(state_dict: dict[str, torch.Tensor]) -> int | None:
    stage_counts = tuple(
        _stage_block_count(state_dict, layer_name)
        for layer_name in ("layer1", "layer2", "layer3", "layer4")
    )
    has_bottleneck_blocks = any(".bn3." in key for key in state_dict)
    if has_bottleneck_blocks:
        for depth, expected_counts in BOTTLENECK_BLOCKS_BY_DEPTH.items():
            if stage_counts == expected_counts:
                return depth
    return None


def _set_eval_flags(module: nn.Module) -> None:
    """Set eval-mode flags without calling module.eval()/train(False)."""
    stack = [module]
    seen: set[int] = set()
    count = 0
    while stack:
        submodule = stack.pop()
        submodule_id = id(submodule)
        if submodule_id in seen:
            continue
        seen.add(submodule_id)
        submodule.training = False
        count += 1
        stack.extend(submodule.children())
    logger.info("Set training=False on %d submodules without module.eval()", count)


class Decoder(nn.Module):
    """Decoder module for ResNet3D that upsamples feature maps to full resolution."""

    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i] + encoder_dims[i - 1], encoder_dims[i - 1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i - 1]),
                nn.ReLU(inplace=True),
            )
            for i in range(1, len(encoder_dims))
        ])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps) - 1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i - 1], f_up], dim=1)
            f_down = self.convs[i - 1](f)
            feature_maps[i - 1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask


class RegressionPLModel(pl.LightningModule):
    """ResNet3D model for ink detection inference."""

    def __init__(
        self,
        pred_shape=(1, 1),
        size=64,
        enc="resnet3d-50",
        with_norm=False,
        num_frames=30,
        model_depth=50,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = generate_model(
            model_depth=int(model_depth),
            n_input_channels=1,
            forward_features=True,
            n_classes=1039,
        )

        with torch.no_grad():
            dummy_input = torch.rand(1, 1, num_frames, 256, 256)
            feat_maps = self.backbone(dummy_input)
            encoder_dims = [x.size(1) for x in feat_maps]

        self.decoder = Decoder(encoder_dims=encoder_dims, upscale=1)

        if self.hparams.with_norm:
            self.normalization = nn.BatchNorm3d(num_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x[:, None]
        if self.hparams.with_norm:
            x = self.normalization(x)

        feat_maps = self.backbone(x)
        feat_maps_pooled = [torch.max(f, dim=2)[0] for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask

    def get_output_scale_factor(self) -> int:
        return 4


class ResNet3DWrapper:
    """Wrapper for ResNet3D model that implements InferenceModel protocol."""

    def __init__(
        self,
        model: RegressionPLModel,
        device: torch.device,
        checkpoint_keepalive: object | None = None,
    ):
        self.model = model
        self.device = device
        self.checkpoint_keepalive = checkpoint_keepalive

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_output_scale_factor(self) -> int:
        return self.model.get_output_scale_factor()

    def eval(self):
        self.model.eval()

    def to(self, device: torch.device):
        self.model.to(device)
        self.device = device


def load_model(
    model_path: str,
    device: torch.device,
    num_frames: int = 30,
    model_depth: int | None = None,
) -> ResNet3DWrapper:
    """
    Load and initialize a ResNet3D model.

    Args:
        model_path: Path to model checkpoint
        device: Torch device to load model onto
        num_frames: Number of input frames/layers
        model_depth: Optional requested ResNet depth. When omitted, infer from checkpoint.

    Returns:
        Wrapped model implementing InferenceModel protocol
    """
    try:
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning(f"Full checkpoint load failed: {e}, retrying with weights_only=True")
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

        state_dict = _extract_state_dict(checkpoint)
        inferred_depth = infer_resnet_depth(state_dict)
        selected_depth = int(model_depth or inferred_depth or 50)
        if inferred_depth is not None and model_depth is not None and int(model_depth) != inferred_depth:
            logger.warning(
                "Requested ResNet3D-%s but checkpoint looks like ResNet3D-%s; using checkpoint depth",
                model_depth,
                inferred_depth,
            )
            selected_depth = inferred_depth

        has_input_norm = any(key.startswith("normalization.") for key in state_dict)
        logger.info(
            "Loading ResNet3D-%s model from: %s with %s frames (input_norm=%s)",
            selected_depth,
            model_path,
            num_frames,
            has_input_norm,
        )

        model = RegressionPLModel(
            pred_shape=(1, 1),
            enc=f"resnet3d-{selected_depth}",
            with_norm=has_input_norm,
            num_frames=num_frames,
            model_depth=selected_depth,
        )
        if os.getenv("ALLOW_DATA_PARALLEL", "0") == "1":
            device_count = torch.cuda.device_count()
            if device_count > 1:
                model = nn.DataParallel(model)
                logger.info(f"Model wrapped with DataParallel for {device_count} GPUs")

        logger.info("About to move ResNet3D model to device %s", device)
        model.to(device)
        logger.info("ResNet3D model moved to device %s", device)
        incompat = model.load_state_dict(state_dict, strict=False)
        if incompat.missing_keys or incompat.unexpected_keys:
            logger.warning(
                "Checkpoint load completed with %s missing and %s unexpected keys",
                len(incompat.missing_keys),
                len(incompat.unexpected_keys),
            )
        else:
            logger.info("Model weights loaded cleanly")
        # Drop temporary checkpoint/state-dict references before eval/inference so
        # compute-node jobs do not retain extra CPU storages unnecessarily.
        del state_dict
        checkpoint = None
        gc.collect()
        logger.info("About to set ResNet3D model to eval mode")
        _set_eval_flags(model)
        logger.info("ResNet3D model set to eval mode")
        logger.info(f"ResNet3D model loaded successfully on {device}")

        return ResNet3DWrapper(model, device, checkpoint_keepalive=None)

    except Exception as e:
        logger.error(f"Failed to load ResNet3D model: {e}")
        raise
