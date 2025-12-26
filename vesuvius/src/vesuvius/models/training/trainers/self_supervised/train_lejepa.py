"""
LeJEPA pretraining trainer.

Implements Latent-Euclidean JEPA (LeJEPA) for unsupervised representation learning
using SIGReg (Sketched Isotropic Gaussian Regularization) loss.

Unlike traditional JEPA, LeJEPA:
- Has NO teacher-student network (single encoder)
- Has NO exponential moving average (EMA)
- Has NO stop-gradient operations
- Uses statistical regularization (SIGReg) instead of architectural tricks

Supports both 2D and 3D modes:
- 3D mode: Multi-scale cropping from 3D volumes
- 2D mode: Uses rotated 2D plane slices from 3D volumes (from dataset),
           multi-view generation with area-based cropping, and 2D augmentations

Reference: https://arxiv.org/abs/2511.08544
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
import numpy as np
from pathlib import Path
from typing import Tuple, List

from vesuvius.models.training.train import BaseTrainer
from vesuvius.models.training.loss.sigreg import SIGRegLoss
from vesuvius.models.training.lr_schedulers import get_scheduler
from vesuvius.models.utils import empty_cache

from vesuvius.models.build.primus_wrapper import PrimusEncoder
from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms


class ProjectionMLP(nn.Module):
    """
    Projection head for LeJEPA.

    Maps encoder embeddings to a lower-dimensional space where SIGReg operates.
    Architecture: embed_dim -> 2048 -> 2048 -> proj_dim with BatchNorm and GELU.
    """

    def __init__(self, embed_dim: int, proj_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TrainLeJEPA(BaseTrainer):
    """
    LeJEPA pretraining trainer with SIGReg loss.

    Creates multiple views of the same input via augmentation and trains
    a single encoder to produce invariant representations while maintaining
    an isotropic Gaussian distribution in embedding space.

    Supports both 2D and 3D modes:
    - 3D: Uses 3D volumes directly with volume-based cropping
    - 2D: Dataset provides rotated 2D plane slices, trainer handles
          area-based cropping and 2D augmentations
    """

    def __init__(self, mgr=None, verbose: bool = True):
        # LeJEPA-specific hyperparameters
        self.lejepa_lambda = 0.02  # Balance between invariance and SIGReg
        self.num_global_views = 2
        self.num_local_views = 6
        self.sigreg_num_slices = 256
        self.proj_dim = 128  # Projection dimension
        # Multi-crop scales following the paper/README defaults
        self.global_crop_scale = (0.3, 1.0)
        self.local_crop_scale = (0.05, 0.3)

        # Training hyperparameters
        self.grad_clip = 1.0
        self.initial_lr = 5e-4
        self.weight_decay = 0.05
        self.warmup_duration = 0  # No warmup, cosine only
        self.vit_patch_size = (16, 16, 16)

        # 2D mode configuration (rotated plane extraction)
        self.is_2d_mode = False
        self.plane_rotation_max_degrees = 30  # Max in-plane rotation
        self.plane_tilt_max_degrees = 15  # Max out-of-plane tilt

        # Override from config if provided
        if mgr is not None:
            self.lejepa_lambda = getattr(mgr, "lejepa_lambda", self.lejepa_lambda)
            self.num_global_views = getattr(
                mgr, "num_global_views", self.num_global_views
            )
            self.num_local_views = getattr(mgr, "num_local_views", self.num_local_views)
            self.sigreg_num_slices = getattr(
                mgr, "sigreg_num_slices", self.sigreg_num_slices
            )
            self.proj_dim = getattr(mgr, "proj_dim", self.proj_dim)
            vit_patch_size = getattr(mgr, "vit_patch_size", None)
            if vit_patch_size is None:
                model_cfg = getattr(mgr, "model_config", {}) or {}
                vit_patch_size = model_cfg.get("patch_embed_size", self.vit_patch_size)
            if isinstance(vit_patch_size, (list, tuple)):
                self.vit_patch_size = tuple(vit_patch_size)

            # Crop scale ranges for multi-scale views
            global_scale = getattr(mgr, "global_crop_scale", self.global_crop_scale)
            local_scale = getattr(mgr, "local_crop_scale", self.local_crop_scale)
            self.global_crop_scale = tuple(global_scale) if isinstance(global_scale, (list, tuple)) else (global_scale, global_scale)
            self.local_crop_scale = tuple(local_scale) if isinstance(local_scale, (list, tuple)) else (local_scale, local_scale)

            # Detect 2D mode based on patch size
            train_patch_size = getattr(mgr, "train_patch_size", self.vit_patch_size)
            self.is_2d_mode = len(train_patch_size) == 2

            if self.is_2d_mode:
                print("LeJEPA: 2D mode enabled - using rotated plane slices")
                # Update vit_patch_size for 2D
                self.vit_patch_size = tuple(self.vit_patch_size[-2:]) if len(self.vit_patch_size) == 3 else self.vit_patch_size

                # 2D-specific config from manager
                self.plane_rotation_max_degrees = getattr(
                    mgr, "plane_rotation_max_degrees", self.plane_rotation_max_degrees
                )
                self.plane_tilt_max_degrees = getattr(
                    mgr, "plane_tilt_max_degrees", self.plane_tilt_max_degrees
                )

                # Configure dataset for 2D slice sampling with rotation
                mgr.slice_sampling_enabled = True
                mgr.slice_sample_planes = getattr(mgr, "slice_sample_planes", None) or ["z"]
                mgr.slice_plane_weights = getattr(mgr, "slice_plane_weights", None) or {"z": 1.0}

                # Plane patch sizes - must cover same as train_patch_size
                patch_h, patch_w = train_patch_size
                mgr.slice_plane_patch_sizes = (
                    getattr(mgr, "slice_plane_patch_sizes", None)
                    or {plane: (patch_h, patch_w) for plane in mgr.slice_sample_planes}
                )

                # Configure rotated plane extraction
                mgr.slice_random_rotation_planes = getattr(
                    mgr, "slice_random_rotation_planes",
                    {plane: {"max_degrees": self.plane_rotation_max_degrees, "probability": 1.0}
                     for plane in mgr.slice_sample_planes}
                )
                mgr.slice_random_tilt_planes = getattr(
                    mgr, "slice_random_tilt_planes",
                    {plane: {
                        "probability": 1.0,
                        "axes": {"x": self.plane_tilt_max_degrees, "y": self.plane_tilt_max_degrees}
                    } for plane in mgr.slice_sample_planes}
                )

            # Set training config
            mgr.initial_lr = getattr(mgr, "initial_lr", self.initial_lr)
            mgr.weight_decay = getattr(mgr, "weight_decay", self.weight_decay)
            mgr.warmup_duration = getattr(mgr, "warmup_duration", self.warmup_duration)

            # LeJEPA is pure unsupervised - no targets needed
            if not hasattr(mgr, "targets"):
                mgr.targets = {}
            mgr.targets["lejepa"] = {
                "num_classes": 1,
                "out_channels": 1,
                "weight": 1.0,
            }

            # Model config for PrimusEncoder
            if not hasattr(mgr, "model_config"):
                mgr.model_config = {}
            mgr.model_config["patch_embed_size"] = self.vit_patch_size
            # Disable patch dropping for LeJEPA (we want full representations)
            mgr.model_config["patch_drop_rate"] = 0.0

            # Dataset config for pure unsupervised training
            # These settings ensure LeJEPA works without requiring labels
            mgr.allow_unlabeled_data = True  # Accept volumes without labels
            mgr.min_labeled_ratio = 0  # No minimum labeled voxel requirement
            mgr.min_bbox_percent = 0  # No bounding box coverage requirement
            mgr.skip_patch_validation = False  # Still validate patches exist
            # Enable unlabeled foreground detection to filter patches by image content
            # This ensures we use patches with actual data, not empty regions
            mgr.unlabeled_foreground_enabled = True
            # Note: unlabeled_foreground_threshold and unlabeled_foreground_bbox_threshold
            # are already read from dataset_config by the config manager

        super().__init__(mgr, verbose)

        self.training_stage = None
        self.current_epoch = 0
        self.global_step = 0

        # Build augmentation pipelines
        self._build_augmentations()

    def _build_augmentations(self):
        """
        Build augmentation pipeline using the standard training transforms.

        Both global and local views use the same augmentation pipeline.
        The differentiation comes from multi-scale cropping (local views are
        cropped to a smaller size and resized back).

        For 2D mode, uses 2D-specific augmentations including:
        - RandomHorizontalFlip
        - RandomRotation (in-plane)
        - GaussianBlur
        - Intensity augmentations
        """
        patch_size = tuple(self.mgr.train_patch_size)

        if self.is_2d_mode:
            # Build 2D augmentation pipeline
            self.augmentation = self._build_2d_augmentation()
            print(f"LeJEPA: Using 2D augmentations for patch size {patch_size}")
        else:
            self.augmentation = create_training_transforms(patch_size)

    def _build_2d_augmentation(self):
        """
        Build 2D augmentation pipeline for LeJEPA matching reference implementation.

        Augmentations (matching LeJEPA reference paper):
        - MirrorTransform (H/V flip)
        - ContrastTransform / MultiplicativeBrightness (like ColorJitter)
        - GammaTransform (non-linear intensity)
        - GaussianBlurTransform
        - SimulateLowResolutionTransform (multi-scale robustness)
        - SharpeningTransform (texture variation)
        - InvertImageTransform (like RandomSolarize)
        """
        from vesuvius.models.augmentation.transforms.intensity.contrast import ContrastTransform, BGContrast
        from vesuvius.models.augmentation.transforms.intensity.gamma import GammaTransform
        from vesuvius.models.augmentation.transforms.intensity.brightness import MultiplicativeBrightnessTransform
        from vesuvius.models.augmentation.transforms.intensity.inversion import InvertImageTransform
        from vesuvius.models.augmentation.transforms.spatial.low_resolution import SimulateLowResolutionTransform
        from vesuvius.models.augmentation.transforms.noise.gaussian_blur import GaussianBlurTransform
        from vesuvius.models.augmentation.transforms.noise import SharpeningTransform
        from vesuvius.models.augmentation.transforms.spatial.mirroring import MirrorTransform
        from vesuvius.models.augmentation.transforms.utils.random import RandomTransform
        from vesuvius.models.augmentation.transforms.utils.compose import ComposeTransforms
        from vesuvius.models.augmentation.transforms.utils.oneoftransform import OneOfTransform

        transforms = [
            # Geometric - H and V flip
            MirrorTransform(allowed_axes=(0, 1)),

            # Intensity variations (like ColorJitter) - one of contrast or brightness
            OneOfTransform([
                RandomTransform(
                    ContrastTransform(
                        contrast_range=BGContrast((0.7, 1.3)),
                        preserve_range=True,
                        synchronize_channels=True,
                        p_per_channel=1.0,
                    ),
                    apply_probability=0.5,
                ),
                RandomTransform(
                    MultiplicativeBrightnessTransform(
                        multiplier_range=BGContrast((0.7, 1.3)),
                        synchronize_channels=True,
                        p_per_channel=1.0,
                    ),
                    apply_probability=0.5,
                ),
            ]),

            # Gamma (non-linear intensity)
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=0,
                    synchronize_channels=True,
                    p_per_channel=1,
                    p_retain_stats=1
                ),
                apply_probability=0.3
            ),

            # Gaussian blur
            RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.1, 2.0),
                    synchronize_channels=True,
                    synchronize_axes=True,
                    p_per_channel=1.0,
                ),
                apply_probability=0.5,
            ),

            # Low resolution simulation - multi-scale robustness
            RandomTransform(
                SimulateLowResolutionTransform(
                    scale=(0.5, 1.0),
                    synchronize_channels=True,
                    synchronize_axes=True,
                    ignore_axes=None,
                ),
                apply_probability=0.25
            ),

            # Sharpening - texture variation
            RandomTransform(
                SharpeningTransform(
                    strength=(0.1, 1.0),
                    p_same_for_each_channel=1.0,
                    p_per_channel=1.0,
                ),
                apply_probability=0.2
            ),

            # Inversion (like RandomSolarize from reference)
            RandomTransform(
                InvertImageTransform(
                    p_invert_image=1,
                    p_synchronize_channels=1.0,
                    p_per_channel=1.0,
                ),
                apply_probability=0.1
            ),
        ]

        return ComposeTransforms(transforms)

    def _build_model(self):
        """Build PrimusEncoder and projection head for LeJEPA pretraining."""
        # Get model configuration
        config_name = getattr(self.mgr, "primus_variant", None)
        if isinstance(config_name, str):
            config_name = config_name.strip()
            if config_name.lower().startswith("primus_"):
                config_name = config_name.split("_", 1)[1]
            config_name = config_name.upper()
        if not config_name:
            arch = self.mgr.model_config.get("architecture_type")
            if isinstance(arch, str) and arch.lower().startswith("primus_"):
                config_name = arch.split("_", 1)[1].upper()
        if not config_name:
            config_name = "M"
        patch_embed_size = self.mgr.model_config.get(
            "patch_embed_size", self.vit_patch_size
        )
        input_shape = tuple(self.mgr.train_patch_size)

        # Build encoder
        self.encoder = PrimusEncoder(
            input_channels=1,  # Single channel CT/volumetric data
            config_name=config_name,
            patch_embed_size=patch_embed_size,
            input_shape=input_shape,
            drop_path_rate=self.mgr.model_config.get("drop_path_rate", 0.1),
            patch_drop_rate=0.0,  # No patch dropping for LeJEPA
            proj_drop_rate=self.mgr.model_config.get("proj_drop_rate", 0.0),
            attn_drop_rate=self.mgr.model_config.get("attn_drop_rate", 0.0),
        )

        # Enable gradient checkpointing if configured (trades compute for memory)
        grad_ckpt = self.mgr.model_config.get("gradient_checkpointing", False)
        if grad_ckpt and hasattr(self.encoder, 'eva') and hasattr(self.encoder.eva, 'set_grad_checkpointing'):
            self.encoder.eva.set_grad_checkpointing(True)
            print("LeJEPA: Gradient checkpointing enabled for memory efficiency")

        # Build projection head (critical for LeJEPA!)
        self.projector = ProjectionMLP(
            embed_dim=self.encoder.embed_dim,
            proj_dim=self.proj_dim,
        )

        # Combine into a single module for optimizer
        self.model = nn.ModuleDict({
            "encoder": self.encoder,
            "projector": self.projector,
        })

        return self.model

    def _build_loss(self):
        """Build SIGReg loss function."""
        self.criterion = SIGRegLoss(
            num_slices=self.sigreg_num_slices,
            lambd=self.lejepa_lambda,
        )
        # Return empty dict for compatibility - we handle loss internally
        return {"lejepa": [(self.criterion, 1.0)]}

    def _get_optimizer(self, model):
        """Create AdamW optimizer."""
        # Collect parameters from encoder and projector
        params = list(self.encoder.parameters()) + list(self.projector.parameters())

        optimizer = torch.optim.AdamW(
            params,
            lr=self.mgr.initial_lr,
            weight_decay=self.mgr.weight_decay,
            betas=(0.9, 0.95),
        )

        empty_cache(self.device)
        return optimizer

    def _get_scheduler(self, optimizer):
        """
        Get cosine annealing scheduler with warmup.
        """
        total_steps = self.mgr.max_steps_per_epoch * self.mgr.max_epoch
        warmup_steps = getattr(self.mgr, 'warmup_steps', int(0.01 * total_steps))

        scheduler = get_scheduler(
            'cosine_warmup',
            optimizer,
            initial_lr=self.mgr.initial_lr,
            max_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr=self.mgr.initial_lr / 1000,
        )

        print(f"Cosine warmup: {total_steps} steps, {warmup_steps} warmup")

        return scheduler, True  # per-iteration scheduler

    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation pipeline to a batch of images.

        Args:
            x: (B, C, D, H, W) tensor

        Returns:
            Augmented tensor (B, C, D, H, W)
        """
        augmented = []
        for i in range(x.shape[0]):
            # Extract single sample (C, D, H, W)
            sample = x[i]
            # Apply augmentation - transforms expect dict with 'image' key
            data_dict = {"image": sample}
            aug_dict = self.augmentation(**data_dict)
            augmented.append(aug_dict["image"])

        return torch.stack(augmented, dim=0)

    def _random_resized_crop_3d(self, x: torch.Tensor, scale_range: tuple, target_size: tuple) -> torch.Tensor:
        """
        3D RandomResizedCrop following DINO/LeJEPA reference implementation.

        Uses volume-based scaling: scale=0.3 means 30% of volume,
        so crop_dim = input_dim * scale^(1/3).

        Args:
            x: (B, C, D, H, W) tensor - batch of 3D volumes
            scale_range: (min_scale, max_scale) - fraction of volume to crop
            target_size: (D, H, W) - output size after resizing

        Returns:
            crops: (B, C, D, H, W) tensor of cropped and resized volumes
        """
        B, C, D, H, W = x.shape
        min_scale, max_scale = scale_range

        crops = []
        for i in range(B):
            # Sample random scale (volume fraction)
            scale = torch.empty(1).uniform_(min_scale, max_scale).item()

            # Volume-based: crop_dim = input_dim * scale^(1/3)
            scale_per_dim = scale ** (1 / 3)
            crop_d = max(1, int(D * scale_per_dim))
            crop_h = max(1, int(H * scale_per_dim))
            crop_w = max(1, int(W * scale_per_dim))

            # Random position (axis-aligned)
            d_start = torch.randint(0, max(1, D - crop_d + 1), (1,)).item()
            h_start = torch.randint(0, max(1, H - crop_h + 1), (1,)).item()
            w_start = torch.randint(0, max(1, W - crop_w + 1), (1,)).item()

            # Extract crop
            crop = x[i : i + 1, :, d_start : d_start + crop_d, h_start : h_start + crop_h, w_start : w_start + crop_w]

            # Resize to target size
            crop = F.interpolate(crop, size=target_size, mode="trilinear", align_corners=False)
            crops.append(crop[0])

        return torch.stack(crops, dim=0)

    def _random_resized_crop_2d(self, x: torch.Tensor, scale_range: tuple, target_size: tuple) -> torch.Tensor:
        """
        2D RandomResizedCrop following DINO/LeJEPA reference implementation.

        Uses area-based scaling: scale=0.3 means 30% of area,
        so crop_dim = input_dim * scale^(1/2).

        Args:
            x: (B, C, H, W) tensor - batch of 2D images
            scale_range: (min_scale, max_scale) - fraction of area to crop
            target_size: (H, W) - output size after resizing

        Returns:
            crops: (B, C, H, W) tensor of cropped and resized images
        """
        B, C, H, W = x.shape
        min_scale, max_scale = scale_range

        crops = []
        for i in range(B):
            # Sample random scale (area fraction)
            scale = torch.empty(1).uniform_(min_scale, max_scale).item()

            # Area-based: crop_dim = input_dim * scale^(1/2)
            scale_per_dim = scale ** 0.5
            crop_h = max(1, int(H * scale_per_dim))
            crop_w = max(1, int(W * scale_per_dim))

            # Random position
            h_start = torch.randint(0, max(1, H - crop_h + 1), (1,)).item()
            w_start = torch.randint(0, max(1, W - crop_w + 1), (1,)).item()

            # Extract crop
            crop = x[i : i + 1, :, h_start : h_start + crop_h, w_start : w_start + crop_w]

            # Resize to target size
            crop = F.interpolate(crop, size=target_size, mode="bilinear", align_corners=False)
            crops.append(crop[0])

        return torch.stack(crops, dim=0)

    def _prepare_multi_view_batch(self, batch):
        """
        Generate multiple views of the input via augmentation with multi-scale cropping.

        Following DINO/LeJEPA reference: both global and local views use RandomResizedCrop
        with different scale ranges, then resize to full resolution.

        For 3D mode:
            Global views: random crop (30-100% volume), resize to full, augment
            Local views: random crop (5-30% volume), resize to full, augment
            Returns: tensors of shape (B, C, D, H, W)

        For 2D mode (rotated plane slices from dataset):
            Global views: random crop (30-100% area), resize to full, augment
            Local views: random crop (5-30% area), resize to full, augment
            Returns: tensors of shape (B, C, H, W)

        Returns:
            global_views: list of tensors
            local_views: list of tensors
        """
        x = batch["image"]  # (B, C, D, H, W) for 3D or (B, C, H, W) for 2D

        if self.is_2d_mode:
            # 2D mode: input is (B, C, H, W) from rotated plane slices
            target_size = x.shape[2:]  # (H, W)

            # Global views: random crop (30-100% area), resize to full, augment
            global_views = []
            for _ in range(self.num_global_views):
                crop = self._random_resized_crop_2d(x, self.global_crop_scale, target_size)
                aug_crop = self._apply_augmentation(crop)
                global_views.append(aug_crop)

            # Local views: random crop (5-30% area), resize to full, augment
            local_views = []
            for _ in range(self.num_local_views):
                crop = self._random_resized_crop_2d(x, self.local_crop_scale, target_size)
                aug_crop = self._apply_augmentation(crop)
                local_views.append(aug_crop)
        else:
            # 3D mode: input is (B, C, D, H, W)
            target_size = x.shape[2:]  # (D, H, W)

            # Global views: random crop (30-100% volume), resize to full, augment
            global_views = []
            for _ in range(self.num_global_views):
                crop = self._random_resized_crop_3d(x, self.global_crop_scale, target_size)
                aug_crop = self._apply_augmentation(crop)
                global_views.append(aug_crop)

            # Local views: random crop (5-30% volume), resize to full, augment
            local_views = []
            for _ in range(self.num_local_views):
                crop = self._random_resized_crop_3d(x, self.local_crop_scale, target_size)
                aug_crop = self._apply_augmentation(crop)
                local_views.append(aug_crop)

        return global_views, local_views

    def _forward_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder with global average pooling and projection.

        Args:
            x: (B, C, D, H, W) for 3D or (B, C, H, W) for 2D input tensor

        Returns:
            emb: (B, embed_dim) embedding tensor (for potential linear probe)
            proj: (B, proj_dim) projection tensor (for SIGReg loss)
        """
        # Encoder returns list with single feature map
        features = self.encoder(x)
        feat = features[0]  # (B, embed_dim, d, h, w) for 3D or (B, embed_dim, h, w) for 2D

        # Global average pooling over spatial dimensions
        if self.is_2d_mode:
            # 2D: pool over (h, w) dimensions
            emb = feat.mean(dim=(2, 3))  # (B, embed_dim)
        else:
            # 3D: pool over (d, h, w) dimensions
            emb = feat.mean(dim=(2, 3, 4))  # (B, embed_dim)

        # Project to lower-dimensional space for SIGReg
        proj = self.projector(emb)  # (B, proj_dim)

        return emb, proj

    def _get_model_outputs(self, model, data_dict):
        """Override to handle LeJEPA multi-view forward pass."""
        # Prepare multi-view batch with multi-scale cropping
        global_views, local_views = self._prepare_multi_view_batch(data_dict)
        all_views = global_views + local_views

        # Forward all views through encoder and projector
        all_proj = []
        for view in all_views:
            view = view.to(self.device)
            _, proj = self._forward_encoder(view)
            all_proj.append(proj)

        # Stack: (V, B, proj_dim)
        all_proj = torch.stack(all_proj, dim=0)
        global_proj = all_proj[: self.num_global_views]

        # Store for loss computation
        self._global_proj = global_proj
        self._all_proj = all_proj

        # Return dummy values for compatibility with BaseTrainer
        inputs = data_dict["image"].to(self.device)
        targets_dict = {"lejepa": inputs}
        outputs_dict = {"lejepa": inputs}  # Placeholder

        return inputs, targets_dict, outputs_dict

    def _compute_train_loss(self, outputs, targets_dict, loss_fns):
        """Compute LeJEPA loss with SIGReg."""
        loss, loss_dict = self.criterion(
            self._global_proj, self._all_proj, global_step=self.global_step
        )

        self.global_step += 1

        # Return loss dict with "lejepa" key for base trainer compatibility
        task_losses = {
            "lejepa": loss_dict["loss"],  # Total loss under target name
            "invariance_loss": loss_dict["invariance_loss"],
            "sigreg_loss": loss_dict["sigreg_loss"],
        }

        return loss, task_losses

    def _validation_step(self, model, data_dict, loss_fns, use_amp):
        """Override validation step for LeJEPA."""
        if use_amp:
            if self.device.type == "cuda":
                context = torch.amp.autocast("cuda")
            else:
                context = torch.amp.autocast(self.device.type)
        else:
            context = nullcontext()

        with context:
            # Prepare multi-view batch
            global_views, local_views = self._prepare_multi_view_batch(data_dict)
            all_views = global_views + local_views

            # Forward all views
            all_proj = []
            for view in all_views:
                view = view.to(self.device)
                _, proj = self._forward_encoder(view)
                all_proj.append(proj)

            all_proj = torch.stack(all_proj, dim=0)
            global_proj = all_proj[: self.num_global_views]

            # Compute loss
            loss, loss_dict = self.criterion(
                global_proj, all_proj, global_step=self.global_step
            )

        inputs = data_dict["image"].to(self.device)
        targets_dict = {"lejepa": inputs}
        outputs_dict = {"lejepa": inputs}

        # Get spatial features from original input for similarity visualization
        with torch.no_grad():
            spatial_features = self.encoder(inputs)[0]  # (B, embed_dim, d, h, w)

        # Store debug data for visualization
        self._debug_data = {
            'original': inputs.detach(),
            'global_views': [gv.detach() for gv in global_views],
            'local_views': [lv.detach() for lv in local_views],
            'all_proj': all_proj.detach(),
            'loss_dict': loss_dict,
            'spatial_features': spatial_features.detach(),  # For similarity visualization
        }

        # Return loss dict with "lejepa" key for base trainer compatibility
        # Plus detailed losses for logging
        task_losses = {
            "lejepa": loss_dict["loss"],  # Total loss under target name
            "invariance_loss": loss_dict["invariance_loss"],
            "sigreg_loss": loss_dict["sigreg_loss"],
        }

        return task_losses, inputs, targets_dict, outputs_dict

    def _compute_validation_loss(self, outputs, targets_dict, loss_fns):
        """Validation loss is computed in _validation_step."""
        # This is called by base class but we handle loss in _validation_step
        return {"lejepa": 0.0}

    def _compute_pca_heatmap(self, feat: torch.Tensor) -> np.ndarray:
        """
        Compute DINOv2-style PCA visualization of spatial features.

        Args:
            feat: (B, embed_dim, D, H, W) for 3D or (B, embed_dim, H, W) for 2D

        Returns:
            rgb_img: (H, W, 3) RGB image with PC1->R, PC2->G, PC3->B
        """
        from sklearn.decomposition import PCA

        feat = feat.detach().cpu()

        if self.is_2d_mode:
            # 2D: (B, C, H, W)
            B, C, H, W = feat.shape
            feat_slice = feat[0]  # (C, H, W)
        else:
            # 3D: (B, C, D, H, W) - get center slice
            B, C, D, H, W = feat.shape
            center_d = D // 2
            feat_slice = feat[0, :, center_d, :, :]  # (C, H, W)

        # Reshape to (H*W, C) for PCA
        feat_flat = feat_slice.permute(1, 2, 0).reshape(-1, C).numpy()  # (H*W, C)

        # PCA to 3 components
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(feat_flat)  # (H*W, 3)

        # Reshape back to spatial (H, W, 3)
        pca_spatial = pca_result.reshape(H, W, 3)

        # Normalize each channel independently to [0, 1]
        for i in range(3):
            channel = pca_spatial[:, :, i]
            pca_spatial[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)

        return pca_spatial

    def _save_lejepa_debug(self, save_path: str, epoch: int):
        """
        Save LeJEPA-specific debug visualization showing:
        - Original input and augmented views
        - PCA of embeddings
        - Embedding histogram (should be Gaussian)
        - Cosine similarity matrix between views
        - Spatial embedding similarity heatmap (DINOv2-style)
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
        except ImportError:
            print("matplotlib or sklearn not available for debug visualization")
            return None

        debug_data = getattr(self, '_debug_data', None)
        if debug_data is None:
            return None

        original = debug_data['original']  # [1, C, D, H, W] or [1, C, H, W]
        global_views = debug_data['global_views']  # list of tensors
        local_views = debug_data['local_views']  # list of tensors
        all_proj = debug_data['all_proj']  # [V, B, K] embeddings

        # Get center slice for 3D data
        def get_center_slice(vol):
            vol = vol.detach().cpu().numpy()
            if vol.ndim == 5:  # [B, C, D, H, W]
                d = vol.shape[2] // 2
                return vol[0, 0, d]  # [H, W]
            elif vol.ndim == 4:  # [B, C, H, W]
                return vol[0, 0]
            return vol

        # Create figure with GridSpec for proper layout
        from matplotlib.gridspec import GridSpec
        n_global = len(global_views)
        n_local = len(local_views)
        n_views = n_global + n_local

        fig = plt.figure(figsize=(16, 18))
        # 4 rows: views, embeddings, PCA heatmap, loss info
        # Heights: 1 for views, 1.2 for embeddings, 1.2 for heatmap, 0.3 for loss
        gs = GridSpec(4, 7, figure=fig, height_ratios=[1, 1.2, 1.2, 0.3],
                      hspace=0.3, wspace=0.3)

        # Row 1: Original + Global views + Local views (7 columns)
        n_cols_row1 = 1 + n_global + min(n_local, 4)  # Cap local views shown

        # Original input
        ax = fig.add_subplot(gs[0, 0])
        img = get_center_slice(original)
        ax.imshow(img, cmap='gray')
        ax.set_title('Original', fontsize=10)
        ax.axis('off')

        # Global views
        for i, gv in enumerate(global_views):
            ax = fig.add_subplot(gs[0, 1 + i])
            img = get_center_slice(gv)
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Global {i+1}', fontsize=10)
            ax.axis('off')

        # Local views (show up to 4)
        for i, lv in enumerate(local_views[:4]):
            ax = fig.add_subplot(gs[0, 1 + n_global + i])
            img = get_center_slice(lv)
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Local {i+1}', fontsize=10)
            ax.axis('off')

        # Row 2: PCA of embeddings + Embedding histogram + View similarity
        embeddings = all_proj.detach().cpu().numpy()  # [V, B, K]
        V, B, K = embeddings.shape
        embeddings_flat = embeddings.reshape(V * B, K)

        # PCA (spans columns 0-1)
        ax = fig.add_subplot(gs[1, 0:2])
        if embeddings_flat.shape[0] > 2:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(embeddings_flat)

            # Color by view type
            colors = []
            labels = []
            for v in range(V):
                if v < n_global:
                    colors.extend(['blue'] * B)
                    labels.extend([f'Global {v+1}'] * B)
                else:
                    colors.extend(['orange'] * B)
                    labels.extend([f'Local {v-n_global+1}'] * B)

            for v in range(V):
                start_idx = v * B
                end_idx = (v + 1) * B
                color = 'blue' if v < n_global else 'orange'
                label = f'Global {v+1}' if v < n_global else f'Local {v-n_global+1}'
                ax.scatter(pca_result[start_idx:end_idx, 0],
                          pca_result[start_idx:end_idx, 1],
                          c=color, alpha=0.6, s=20, label=label)

            # Fixed axis limits so scale is consistent across epochs
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)

            # Show actual spread as text annotation
            spread = np.std(pca_result)
            ax.text(0.02, 0.98, f'spread: {spread:.4f}', transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            ax.set_title('PCA of Embeddings', fontsize=10)
            ax.legend(fontsize=8, loc='best')

        # Embedding histogram (spans columns 2-4)
        ax = fig.add_subplot(gs[1, 2:5])
        ax.hist(embeddings_flat.flatten(), bins=50, density=True, alpha=0.7, color='steelblue')
        # Overlay Gaussian for reference
        x = np.linspace(embeddings_flat.min(), embeddings_flat.max(), 100)
        std = embeddings_flat.std()
        mean = embeddings_flat.mean()
        gaussian = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        ax.plot(x, gaussian, 'r-', linewidth=2, label=f'N({mean:.2f}, {std:.2f}²)')
        ax.set_xlabel('Embedding Value')
        ax.set_ylabel('Density')
        ax.set_title('Embedding Distribution (should → Gaussian)', fontsize=10)
        ax.legend(fontsize=8)

        # Cosine similarity matrix (spans columns 5-6)
        ax = fig.add_subplot(gs[1, 5:7])
        # Compute mean embedding per view
        view_means = embeddings.mean(axis=1)  # [V, K]
        # Normalize for cosine similarity
        view_means_norm = view_means / (np.linalg.norm(view_means, axis=1, keepdims=True) + 1e-8)
        sim_matrix = view_means_norm @ view_means_norm.T

        im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_title('View Similarity (cosine)', fontsize=10)
        view_labels = [f'G{i+1}' for i in range(n_global)] + [f'L{i+1}' for i in range(n_local)]
        ax.set_xticks(range(n_views))
        ax.set_yticks(range(n_views))
        ax.set_xticklabels(view_labels, fontsize=8)
        ax.set_yticklabels(view_labels, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Row 3: PCA feature heatmap (DINOv2-style)
        spatial_features = debug_data.get('spatial_features')
        if spatial_features is not None:
            # Handle both 2D (B, C, H, W) and 3D (B, C, D, H, W) feature maps
            if self.is_2d_mode:
                B, C, H_feat, W_feat = spatial_features.shape
            else:
                B, C, D, H_feat, W_feat = spatial_features.shape

            # Compute PCA heatmap
            pca_rgb = self._compute_pca_heatmap(spatial_features)

            # Get original image center slice
            orig_slice = get_center_slice(original)
            H_orig, W_orig = orig_slice.shape

            # Upsample PCA heatmap to original resolution for better display
            from scipy.ndimage import zoom
            scale_h = H_orig / H_feat
            scale_w = W_orig / W_feat
            pca_rgb_upsampled = zoom(pca_rgb, (scale_h, scale_w, 1), order=1)

            # Normalize original for display
            orig_normalized = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min() + 1e-8)

            # Panel 1: Original image (spans columns 0-1)
            ax = fig.add_subplot(gs[2, 0:2])
            ax.imshow(orig_normalized, cmap='gray')
            title = 'Original' if self.is_2d_mode else 'Original (center slice)'
            ax.set_title(title, fontsize=10)
            ax.axis('off')

            # Panel 2: PCA RGB heatmap (spans columns 2-4)
            ax = fig.add_subplot(gs[2, 2:5])
            ax.imshow(pca_rgb_upsampled)
            ax.set_title('PCA Features (RGB=PC1,2,3)', fontsize=10)
            ax.axis('off')

            # Panel 3: Overlay blend (spans columns 5-6)
            ax = fig.add_subplot(gs[2, 5:7])
            # Blend original grayscale with PCA colors
            orig_rgb = np.stack([orig_normalized]*3, axis=-1)
            blended = 0.4 * orig_rgb + 0.6 * pca_rgb_upsampled
            blended = np.clip(blended, 0, 1)
            ax.imshow(blended)
            ax.set_title('Overlay (40% orig + 60% PCA)', fontsize=10)
            ax.axis('off')

        # Row 4: Loss info (spans all columns)
        ax = fig.add_subplot(gs[3, :])
        ax.axis('off')
        loss_info = debug_data.get('loss_dict', {})
        info_text = f"Epoch {epoch+1} | "
        info_text += f"Total Loss: {loss_info.get('loss', 0):.4f} | "
        info_text += f"Invariance: {loss_info.get('invariance_loss', 0):.4f} | "
        info_text += f"SIGReg: {loss_info.get('sigreg_loss', 0):.4f}"
        ax.text(0.5, 0.5, info_text, transform=ax.transAxes, fontsize=11,
                ha='center', va='center', family='monospace')

        mode_str = "2D" if self.is_2d_mode else "3D"
        plt.suptitle(f'LeJEPA Debug ({mode_str}) - Epoch {epoch+1}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Save
        save_path = Path(save_path)
        save_path = save_path.with_suffix('.png')  # Use PNG for static image
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return str(save_path)
