"""
ResNet3D-152 + 3D decoder model for ink detection inference.
"""
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnetall import generate_model

logger = logging.getLogger(__name__)


class ResConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.shortcut = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = F.relu(self.gn1(self.conv1(x)), inplace=True)
        x = self.gn2(self.conv2(x))
        x = F.relu(x + identity, inplace=True)
        return x


class DepthAttentionCollapse(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.attn_conv = nn.Conv3d(in_ch, 1, 1)

    def forward(self, x):
        attn = self.attn_conv(x)
        attn = F.softmax(attn, dim=2)
        return (x * attn).sum(dim=2)


class AuxHead(nn.Module):
    def __init__(self, in_ch, target_size=64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.conv = nn.Conv2d(in_ch, 1, 1)
        self.target_size = target_size

    def forward(self, x):
        x = self.pool(x).squeeze(2)
        x = self.conv(x)
        if x.shape[-1] != self.target_size:
            x = F.interpolate(
                x,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
        return x


class Decoder3DUNet(nn.Module):
    def __init__(
        self,
        encoder_dims=(256, 512, 1024, 2048),
        decoder_dims=(64, 128, 256, 512),
        deep_supervision=True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        self.channel_reduce = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(enc_d, dec_d, 1, bias=False),
                    nn.GroupNorm(min(32, dec_d), dec_d),
                    nn.ReLU(inplace=True),
                )
                for enc_d, dec_d in zip(encoder_dims, decoder_dims)
            ]
        )

        self.decoder_blocks = nn.ModuleList(
            [
                ResConvBlock3D(decoder_dims[i] + decoder_dims[i - 1], decoder_dims[i - 1])
                for i in range(len(decoder_dims) - 1, 0, -1)
            ]
        )

        self.depth_collapse = DepthAttentionCollapse(decoder_dims[0])
        self.logit = nn.Conv2d(decoder_dims[0], 1, 1)

        if deep_supervision:
            self.aux_head_s2 = AuxHead(decoder_dims[2])
            self.aux_head_s1 = AuxHead(decoder_dims[1])

    def forward(self, feat_maps):
        feats = [self.channel_reduce[i](feat_maps[i]) for i in range(4)]
        aux_outputs = []

        x = feats[3]
        x = F.interpolate(x, size=feats[2].shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, feats[2]], dim=1)
        x = self.decoder_blocks[0](x)
        if self.deep_supervision and self.training:
            aux_outputs.append(self.aux_head_s2(x))

        x = F.interpolate(x, size=feats[1].shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, feats[1]], dim=1)
        x = self.decoder_blocks[1](x)
        if self.deep_supervision and self.training:
            aux_outputs.append(self.aux_head_s1(x))

        x = F.interpolate(x, size=feats[0].shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, feats[0]], dim=1)
        x = self.decoder_blocks[2](x)

        x = self.depth_collapse(x)
        x = self.logit(x)

        if self.deep_supervision and self.training:
            return x, aux_outputs
        return x


class RegressionModel(nn.Module):
    def __init__(self, with_norm=False):
        super().__init__()
        self.backbone = generate_model(
            model_depth=152,
            n_input_channels=1,
            forward_features=True,
            n_classes=1039,
        )
        self.decoder = Decoder3DUNet(
            encoder_dims=(256, 512, 1024, 2048),
            decoder_dims=(64, 128, 256, 512),
            deep_supervision=True,
        )
        self.normalization = nn.BatchNorm3d(num_features=1) if with_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x[:, None]
        if self.normalization is not None:
            x = self.normalization(x)
        feat_maps = self.backbone(x)
        return self.decoder(feat_maps)

    def get_output_scale_factor(self) -> int:
        return 4


class ResNet3DDecoderWrapper:
    def __init__(self, model: RegressionModel, device: torch.device):
        self.model = model
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_output_scale_factor(self) -> int:
        return self.model.get_output_scale_factor()

    def eval(self):
        self.model.eval()

    def to(self, device: torch.device):
        self.model.to(device)
        self.device = device


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint does not contain a usable state_dict")

    for key in ("state_dict", "model_state_dict"):
        state_dict = checkpoint.get(key)
        if isinstance(state_dict, dict):
            if state_dict and all(
                isinstance(name, str) and isinstance(value, torch.Tensor)
                for name, value in state_dict.items()
            ):
                return state_dict
            raise RuntimeError(f"Checkpoint field '{key}' is not a valid tensor state_dict")

    if checkpoint and all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in checkpoint.items()
    ):
        return checkpoint

    raise RuntimeError("Checkpoint does not contain a usable state_dict")


def _strip_known_prefixes(state_dict):
    normalized = OrderedDict()
    for key, value in state_dict.items():
        while key.startswith("model.") or key.startswith("module."):
            if key.startswith("model."):
                key = key[len("model.") :]
            if key.startswith("module."):
                key = key[len("module.") :]
        normalized[key] = value
    return normalized


def load_model(model_path: str, device: torch.device, num_frames: int = 62) -> ResNet3DDecoderWrapper:
    try:
        logger.info(
            "Loading ResNet3D-152 3D decoder model from: %s with %s frames",
            model_path,
            num_frames,
        )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = _strip_known_prefixes(_extract_state_dict(checkpoint))
        with_norm = any(key.startswith("normalization.") for key in state_dict)

        model = RegressionModel(with_norm=with_norm)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning("Missing checkpoint keys: %s", missing_keys[:10])
        if unexpected_keys:
            logger.warning("Unexpected checkpoint keys: %s", unexpected_keys[:10])

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logger.info("Model wrapped with DataParallel for %s GPUs", torch.cuda.device_count())

        model.to(device)
        model.eval()
        logger.info("ResNet3D-152 3D decoder model loaded successfully on %s", device)
        return ResNet3DDecoderWrapper(model, device)
    except Exception as exc:
        logger.error("Failed to load ResNet3D-152 3D decoder model: %s", exc)
        raise
