"""
ResNet3D model for ink detection inference.
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

BOTTLE_NECK_BLOCKS_BY_DEPTH = {
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
    200: (3, 24, 36, 3),
}


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_state_dict"), dict):
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format for ResNet3D inference")

    normalized = {}
    for key, value in state_dict.items():
        while key.startswith("module.") or key.startswith("model."):
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("model."):
                key = key[len("model."):]
        normalized[key] = value.detach().cpu().contiguous().clone() if torch.is_tensor(value) else value
    return normalized


def _stage_block_count(state_dict, layer_name):
    prefix = f"backbone.{layer_name}."
    indices = {
        int(parts[2])
        for key in state_dict
        if key.startswith(prefix)
        for parts in [key.split(".")]
        if len(parts) > 2 and parts[2].isdigit()
    }
    return (max(indices) + 1) if indices else 0


def infer_resnet_depth(state_dict):
    stage_counts = tuple(
        _stage_block_count(state_dict, layer_name)
        for layer_name in ("layer1", "layer2", "layer3", "layer4")
    )
    has_bottleneck_blocks = any(".bn3." in key for key in state_dict)
    if has_bottleneck_blocks:
        for depth, expected_counts in BOTTLE_NECK_BLOCKS_BY_DEPTH.items():
            if stage_counts == expected_counts:
                return depth
    return None


# ----------------------------- Decoder ---------------------------------------
class Decoder(nn.Module):
    """Decoder module for ResNet3D that upsamples feature maps to full resolution."""
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i]+encoder_dims[i-1], encoder_dims[i-1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i-1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps)-1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i-1], f_up], dim=1)
            f_down = self.convs[i-1](f)
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask

# ------------------------------- Model ---------------------------------------
class RegressionPLModel(pl.LightningModule):
    """ResNet3D for ink detection inference."""
    def __init__(
        self,
        pred_shape=(1, 1),
        size=64,
        enc='resnet3d-50',
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

        # Initialize decoder based on backbone output dimensions
        # Get encoder dims by doing a forward pass with dummy input
        with torch.no_grad():
            dummy_input = torch.rand(1, 1, num_frames, 256, 256)
            feat_maps = self.backbone(dummy_input)
            encoder_dims = [x.size(1) for x in feat_maps]

        self.decoder = Decoder(encoder_dims=encoder_dims, upscale=1)

        if self.hparams.with_norm:
            self.normalization = nn.BatchNorm3d(num_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B,1,C,H,W) where C is the temporal/depth dimension
        if x.ndim == 4:
            x = x[:, None]
        if self.hparams.with_norm:
            x = self.normalization(x)

        # Get feature maps from backbone
        feat_maps = self.backbone(x)

        # Max pool along temporal dimension
        feat_maps_pooled = [torch.max(f, dim=2)[0] for f in feat_maps]

        # Decode to mask
        pred_mask = self.decoder(feat_maps_pooled)

        return pred_mask

    def get_output_scale_factor(self) -> int:
        """ResNet3D outputs 16x16, needs 4x scale to reach 64x64 tiles."""
        return 4  # 16x16 -> 64x64


class ResNet3DWrapper:
    """Wrapper for ResNet3D model that implements InferenceModel protocol."""

    def __init__(self, model: RegressionPLModel, device: torch.device, checkpoint_keepalive=None):
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
    Load and initialize a ResNet3D model with the old max-z + 2D decoder.

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
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except Exception as e:
            logger.warning(f"Full checkpoint load failed: {e}, retrying with weights_only=True")
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)

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

        # Move to device and load weights before optional DataParallel wrapping so
        # unprefixed checkpoint keys match the plain module names.
        model.to(device)
        incompat = model.load_state_dict(state_dict, strict=False)
        if incompat.missing_keys or incompat.unexpected_keys:
            logger.warning(
                "Checkpoint load completed with %s missing and %s unexpected keys",
                len(incompat.missing_keys),
                len(incompat.unexpected_keys),
            )
        else:
            logger.info("Model weights loaded cleanly")

        del state_dict
        checkpoint = None
        gc.collect()

        # Setup multi-GPU if explicitly enabled.
        if os.getenv("ALLOW_DATA_PARALLEL", "0") == "1" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logger.info(f"Model wrapped with DataParallel for {torch.cuda.device_count()} GPUs")

        model.eval()

        logger.info(f"ResNet3D model loaded successfully on {device}")

        # Wrap model
        wrapper = ResNet3DWrapper(model, device, checkpoint_keepalive=None)
        return wrapper

    except Exception as e:
        logger.error(f"Failed to load ResNet3D model: {e}")
        raise
