"""Canonical ResNet-152 weights with a shared voxel/column ink classifier.

The canonical implementation remains in ink-detection/optimized_inference.
Load its package by path rather than copying architecture code into vesuvius.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F
from vesuvius.ink_detection.models.deterministic_pool import SpatialMaxPool3d


def canonical_source_dir():
    """Use the architecture shipped with this checkout, independent of training paths."""
    root = Path(__file__).resolve().parents[5] / "ink-detection" / "optimized_inference"
    if not (root / "model_resnet3d_3d_decoder.py").is_file():
        raise FileNotFoundError("Use a complete repository checkout containing the canonical runtime")
    return root


def canonical_runtime(source_dir):
    root = Path(source_dir).resolve()
    name = "_vesuvius_canonical_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations=[str(root)])
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load canonical runtime at {root}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(name + ".model_resnet3d_3d_decoder")


class CanonicalLogitProjection(nn.Module):
    """Input: normalized B1x64x256x256; native decoder XY stride: four.

    Full-grid outputs are interpolated for supervision/display, not additional
    resolved XY detail. Only central input planes 1:63 are used. Both losses
    share projection_3d_logits as their differentiable ancestor.
    """
    def __init__(self, source_dir, with_norm=True, freeze_batchnorm_stats=True):
        super().__init__()
        self.canonical = canonical_runtime(source_dir).RegressionModel(with_norm=with_norm)
        if hasattr(self.canonical.backbone, "maxpool"):
            pool = self.canonical.backbone.maxpool
            if (pool.kernel_size != (1, 3, 3) or pool.stride != (1, 2, 2)
                    or pool.padding != (0, 1, 1) or pool.ceil_mode):
                raise ValueError("Canonical encoder spatial max-pool contract changed")
            self.canonical.backbone.maxpool = SpatialMaxPool3d()
        self.depth_coordinate_scale = nn.Parameter(torch.zeros(()))
        self.freeze_batchnorm_stats = bool(freeze_batchnorm_stats)
        self.canonical.decoder.deep_supervision = False
        # Retain these tensors for strict checkpoint loading; neither unused
        # classification FC nor auxiliary heads participates in this objective.
        for name, parameter in self.canonical.named_parameters():
            if name.startswith(("backbone.fc.", "decoder.aux_head_")):
                parameter.requires_grad_(False)
        self.train()

    def load_canonical_state(self, payload):
        state = payload["state_dict"]
        self.canonical.load_state_dict(state, strict=True)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_batchnorm_stats:
            for module in self.canonical.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def _features_logits(self, image):
        if image.ndim != 5 or image.shape[1:3] != (1, 64):
            raise ValueError("Canonical joint input must be B1x64xHxW")
        features, _ = self.canonical.forward_features(image[:, :, 1:63])
        head = self.canonical.decoder.logit
        # The existing Conv2d classifier is exactly a pointwise Conv3d when
        # its kernel gains a singleton depth dimension. No new ink head.
        logits = F.conv3d(features.float(), head.weight.float().unsqueeze(2),
                          None if head.bias is None else head.bias.float())
        return features, logits.float()

    @staticmethod
    def _full_grid(logits, image):
        full = F.interpolate(logits, size=(62, *image.shape[-2:]),
                             mode="trilinear", align_corners=False)
        return F.pad(full, (0, 0, 0, 0, 1, 1), value=-20.)

    def forward_3d(self, image):
        """Return full-grid logits without executing attention or projection."""
        _, logits = self._features_logits(image)
        return self._full_grid(logits, image)

    def forward(self, image, valid=None):
        features, logits = self._features_logits(image)
        scores = self.canonical.decoder.depth_collapse.attn_conv(features).float()
        z = (torch.arange(62, device=scores.device, dtype=torch.float32) - 31) / 31
        scores = scores + self.depth_coordinate_scale * z.view(1, 1, -1, 1, 1)
        if valid is None:
            native_valid = torch.ones_like(logits, dtype=torch.bool)
        else:
            native_valid = F.max_pool3d(valid[:, :, 1:63].float(), (1, 4, 4)).bool()
        weights = scores.masked_fill(~native_valid, -1e4).softmax(2) * native_valid
        weights = weights / weights.sum(2, keepdim=True).clamp_min(1e-8)
        native_2d = (weights * logits).sum(2)
        # Production canonical inference interpolates probabilities, not logits.
        probability = F.interpolate(native_2d.sigmoid(), size=image.shape[-2:],
                                    mode="bilinear", align_corners=False)
        ink = torch.logit(probability.clamp(torch.finfo(torch.float32).eps,
                                           1 - torch.finfo(torch.float32).eps))
        attention = F.interpolate(weights, size=(62, *image.shape[-2:]),
                                  mode="trilinear", align_corners=False)
        attention = F.pad(attention, (0, 0, 0, 0, 1, 1))
        return {"ink": ink, "ink_3d_logits": self._full_grid(logits, image),
                "depth_weights": attention, "projection_3d_logits": logits,
                "native_2d_logits": native_2d}
