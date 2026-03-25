"""
TimeSformer model for ink detection inference.
"""
import logging
import torch
import torch.nn as nn
import pytorch_lightning as pl
from timesformer_pytorch import TimeSformer

logger = logging.getLogger(__name__)

# ------------------------------- Model ---------------------------------------
class RegressionPLModel(pl.LightningModule):
    """TimeSformer for ink detection inference."""
    def __init__(self, pred_shape=(1, 1), size=64, enc='', with_norm=False, num_frames=26):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = TimeSformer(
            dim=512,
            image_size=64,
            patch_size=16,
            num_frames=num_frames,   # frames = layers (dynamic)
            num_classes=16,            # 4x4 logits
            channels=1,                # single-channel per frame
            depth=8,
            heads=6,
            dim_head=64,
            attn_dropout=0.1,
            ff_dropout=0.1,
        )

        if self.hparams.with_norm:
            self.normalization = nn.BatchNorm3d(num_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input from dataset: (B,1,C,H,W). TimeSformer lib expects (B,frames,channels,H,W).
        if x.ndim == 4:
            x = x[:, None]
        if self.hparams.with_norm:
            x = self.normalization(x)
        x = torch.permute(x, (0, 2, 1, 3, 4))  # -> (B,C,1,H,W)
        x = self.backbone(x)
        x = x.view(-1, 1, 4, 4)                # (B,1,4,4)
        return x

    def get_output_scale_factor(self) -> int:
        """TimeSformer outputs 4x4 logits, needs 16x scale to reach 64x64 tiles."""
        return 16  # 4x4 -> 64x64


class TimeSformerWrapper:
    """Wrapper for TimeSformer model that implements InferenceModel protocol."""

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


def load_model(model_path: str, device: torch.device, num_frames: int = 26) -> TimeSformerWrapper:
    """
    Load and initialize the TimeSformer model.

    Args:
        model_path: Path to model checkpoint
        device: Torch device to load model onto
        num_frames: Number of input frames/layers

    Returns:
        Wrapped model implementing InferenceModel protocol
    """
    try:
        logger.info(f"Loading TimeSformer model from: {model_path} with {num_frames} frames")

        # Try to load with PyTorch Lightning first
        try:
            model = RegressionPLModel.load_from_checkpoint(model_path, strict=False, num_frames=num_frames)
            logger.info("Model loaded with PyTorch Lightning")
        except Exception as e:
            logger.warning(f"PyTorch Lightning loading failed: {e}, trying manual loading")
            # Fallback to manual loading
            model = RegressionPLModel(pred_shape=(1, 1), num_frames=num_frames)
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logger.info("Model loaded manually")

        # Setup multi-GPU if available
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logger.info(f"Model wrapped with DataParallel for {torch.cuda.device_count()} GPUs")

        # Move to device and set eval mode
        model.to(device)
        model.eval()

        logger.info(f"TimeSformer model loaded successfully on {device}")

        # Wrap model
        wrapper = TimeSformerWrapper(model, device)
        return wrapper

    except Exception as e:
        logger.error(f"Failed to load TimeSformer model: {e}")
        raise
