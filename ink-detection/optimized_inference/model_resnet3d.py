"""
ResNet3D model for ink detection inference.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.resnetall import generate_model

logger = logging.getLogger(__name__)

# ----------------------------- Model Config ----------------------------------
class ResNet3DConfig:
    """Default configuration for ResNet3D model."""
    in_chans = 30  # Number of input frames/layers
    output_scale_factor = 4  # Model outputs 16x16, needs 4x interpolation to 64x64

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
    def __init__(self, pred_shape=(1, 1), size=64, enc='resnet50', with_norm=False):
        super().__init__()
        self.save_hyperparameters()

        # Select backbone based on encoder type
        if self.hparams.enc == 'resnet34':
            self.backbone = generate_model(model_depth=34, n_input_channels=1, forward_features=True, n_classes=700)
            # Try to load pretrained weights if available
            try:
                state_dict = torch.load('./r3d34_K_200ep.pth', weights_only=False)["state_dict"]
                conv1_weight = state_dict['conv1.weight']
                state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
                self.backbone.load_state_dict(state_dict, strict=False)
                logger.info("Loaded pretrained weights for ResNet34")
            except Exception as e:
                logger.warning(f"Could not load pretrained weights for ResNet34: {e}")

        elif self.hparams.enc == 'resnet101':
            self.backbone = generate_model(model_depth=101, n_input_channels=1, forward_features=True, n_classes=1039)
            try:
                state_dict = torch.load('./r3d101_KM_200ep.pth', weights_only=False)["state_dict"]
                conv1_weight = state_dict['conv1.weight']
                state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
                self.backbone.load_state_dict(state_dict, strict=False)
                logger.info("Loaded pretrained weights for ResNet101")
            except Exception as e:
                logger.warning(f"Could not load pretrained weights for ResNet101: {e}")

        else:  # Default: resnet50
            self.backbone = generate_model(model_depth=50, n_input_channels=1, forward_features=True, n_classes=700)
            try:
                state_dict = torch.load('./r3d50_K_200ep.pth', weights_only=False)["state_dict"]
                conv1_weight = state_dict['conv1.weight']
                state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
                self.backbone.load_state_dict(state_dict, strict=False)
                logger.info("Loaded pretrained weights for ResNet50")
            except Exception as e:
                logger.warning(f"Could not load pretrained weights for ResNet50: {e}")

        # Initialize decoder based on backbone output dimensions
        # Get encoder dims by doing a forward pass with dummy input
        with torch.no_grad():
            dummy_input = torch.rand(1, 1, ResNet3DConfig.in_chans, 256, 256)
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
        return ResNet3DConfig.output_scale_factor


class ResNet3DWrapper:
    """Wrapper for ResNet3D model that implements InferenceModel protocol."""

    def __init__(self, model: RegressionPLModel, device: torch.device):
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


def load_model(model_path: str, device: torch.device, encoder: str = 'resnet50') -> ResNet3DWrapper:
    """
    Load and initialize the ResNet3D model.

    Args:
        model_path: Path to model checkpoint
        device: Torch device to load model onto
        encoder: Encoder type ('resnet34', 'resnet50', 'resnet101')

    Returns:
        Wrapped model implementing InferenceModel protocol
    """
    try:
        logger.info(f"Loading ResNet3D model from: {model_path} with encoder: {encoder}")

        # Try to load with PyTorch Lightning first
        try:
            model = RegressionPLModel.load_from_checkpoint(model_path, strict=False, enc=encoder)
            logger.info("Model loaded with PyTorch Lightning")
        except Exception as e:
            logger.warning(f"PyTorch Lightning loading failed: {e}, trying manual loading")
            # Fallback to manual loading
            model = RegressionPLModel(pred_shape=(1, 1), enc=encoder)
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("Model loaded manually")

        # Setup multi-GPU if available
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logger.info(f"Model wrapped with DataParallel for {torch.cuda.device_count()} GPUs")

        # Move to device and set eval mode
        model.to(device)
        model.eval()

        logger.info(f"ResNet3D model loaded successfully on {device}")

        # Wrap model
        wrapper = ResNet3DWrapper(model, device)
        return wrapper

    except Exception as e:
        logger.error(f"Failed to load ResNet3D model: {e}")
        raise
