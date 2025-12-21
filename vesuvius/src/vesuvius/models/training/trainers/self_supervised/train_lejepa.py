"""
LeJEPA pretraining trainer.

Implements Latent-Euclidean JEPA (LeJEPA) for unsupervised representation learning
using SIGReg (Sketched Isotropic Gaussian Regularization) loss.

Unlike traditional JEPA, LeJEPA:
- Has NO teacher-student network (single encoder)
- Has NO exponential moving average (EMA)
- Has NO stop-gradient operations
- Uses statistical regularization (SIGReg) instead of architectural tricks

Reference: https://arxiv.org/abs/2511.08544
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from contextlib import nullcontext

from vesuvius.models.training.train import BaseTrainer
from vesuvius.models.training.loss.sigreg import SIGRegLoss
from vesuvius.models.utils import empty_cache

from vesuvius.models.build.primus_wrapper import PrimusEncoder
from vesuvius.models.augmentation.transforms.utils.compose import ComposeTransforms
from vesuvius.models.augmentation.transforms.spatial.mirroring import MirrorTransform
from vesuvius.models.augmentation.transforms.intensity.brightness import (
    MultiplicativeBrightnessTransform,
)
from vesuvius.models.augmentation.transforms.intensity.gaussian_noise import (
    GaussianNoiseTransform,
)
from vesuvius.models.augmentation.transforms.intensity.contrast import (
    ContrastTransform,
)
from vesuvius.models.augmentation.transforms.intensity.gamma import GammaTransform


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
    """

    def __init__(self, mgr=None, verbose: bool = True):
        # LeJEPA-specific hyperparameters
        self.lejepa_lambda = 0.02  # Balance between invariance and SIGReg
        self.num_global_views = 2
        self.num_local_views = 6
        self.sigreg_num_slices = 256
        self.proj_dim = 128  # Projection dimension
        self.local_crop_size = (64, 64, 64)  # Size for local view crops

        # Training hyperparameters
        self.grad_clip = 1.0
        self.initial_lr = 5e-4
        self.weight_decay = 0.05
        self.warmup_duration = 1  # 1 epoch warmup (matching original)
        self.vit_patch_size = (16, 16, 16)

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
            self.vit_patch_size = getattr(mgr, "vit_patch_size", self.vit_patch_size)

            # Local crop size for multi-scale views
            local_crop = getattr(mgr, "local_crop_size", self.local_crop_size)
            if isinstance(local_crop, (list, tuple)):
                self.local_crop_size = tuple(local_crop)
            else:
                self.local_crop_size = (local_crop, local_crop, local_crop)

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
            mgr.skip_patch_validation = True  # Enumerate all patches, don't filter

        super().__init__(mgr, verbose)

        self.training_stage = None
        self.current_epoch = 0
        self.global_step = 0

        # Build augmentation pipelines
        self._build_augmentations()

    def _build_augmentations(self):
        """
        Build global and local augmentation pipelines.

        Global views: lighter augmentations (small brightness/contrast changes)
        Local views: stronger augmentations (larger noise, gamma, contrast)
        """
        # Global view augmentation (lighter)
        self.global_aug = ComposeTransforms(
            [
                MirrorTransform(allowed_axes=(0, 1, 2)),
                MultiplicativeBrightnessTransform(
                    multiplier_range=(0.9, 1.1),
                    synchronize_channels=True,
                    p_per_channel=1.0,
                ),
                GaussianNoiseTransform(
                    noise_variance=(0.0, 0.02),
                    p_per_channel=0.5,
                    synchronize_channels=False,
                ),
            ]
        )

        # Local view augmentation (stronger)
        self.local_aug = ComposeTransforms(
            [
                MirrorTransform(allowed_axes=(0, 1, 2)),
                MultiplicativeBrightnessTransform(
                    multiplier_range=(0.7, 1.3),
                    synchronize_channels=True,
                    p_per_channel=1.0,
                ),
                ContrastTransform(
                    contrast_range=(0.75, 1.25),
                    preserve_range=True,
                    synchronize_channels=True,
                    p_per_channel=0.8,
                ),
                GammaTransform(
                    gamma=(0.7, 1.5),
                    p_invert_image=0.0,
                    synchronize_channels=True,
                    p_per_channel=0.8,
                    p_retain_stats=1.0,
                ),
                GaussianNoiseTransform(
                    noise_variance=(0.0, 0.1),
                    p_per_channel=0.8,
                    synchronize_channels=False,
                ),
            ]
        )

    def _build_model(self):
        """Build PrimusEncoder and projection head for LeJEPA pretraining."""
        # Get model configuration
        config_name = getattr(self.mgr, "primus_variant", "M")
        patch_embed_size = self.mgr.model_config.get(
            "patch_embed_size", self.vit_patch_size
        )
        input_shape = tuple(self.mgr.patch_size)

        # Build encoder
        self.encoder = PrimusEncoder(
            input_channels=1,  # Single channel CT/volumetric data
            config_name=config_name,
            patch_embed_size=patch_embed_size,
            input_shape=input_shape,
            drop_path_rate=getattr(self.mgr.model_config, "drop_path_rate", 0.1),
            patch_drop_rate=0.0,  # No patch dropping for LeJEPA
            proj_drop_rate=getattr(self.mgr.model_config, "proj_drop_rate", 0.0),
            attn_drop_rate=getattr(self.mgr.model_config, "attn_drop_rate", 0.0),
        )

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
        Get learning rate scheduler with warmup + cosine annealing.

        Follows the original LeJEPA paper:
        - Linear warmup for warmup_duration epochs
        - Cosine annealing to min_lr = initial_lr / 1000
        """
        # We need the dataloader to compute steps - defer to first epoch
        # Return a placeholder that will be replaced
        scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=1)
        return scheduler, True  # Step-based scheduler

    def _setup_scheduler(self, optimizer, steps_per_epoch):
        """Set up the actual scheduler once we know steps_per_epoch."""
        warmup_steps = steps_per_epoch * self.warmup_duration
        total_steps = steps_per_epoch * self.mgr.max_epoch

        s1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
        s2 = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.mgr.initial_lr / 1000,
        )
        scheduler = SequentialLR(optimizer, [s1, s2], milestones=[warmup_steps])

        return scheduler

    def _apply_augmentation(self, x: torch.Tensor, aug_pipeline) -> torch.Tensor:
        """
        Apply augmentation pipeline to a batch of images.

        Args:
            x: (B, C, D, H, W) tensor
            aug_pipeline: ComposeTransforms pipeline

        Returns:
            Augmented tensor (B, C, D, H, W)
        """
        augmented = []
        for i in range(x.shape[0]):
            # Extract single sample (C, D, H, W)
            sample = x[i]
            # Apply augmentation - transforms expect dict with 'image' key
            data_dict = {"image": sample}
            aug_dict = aug_pipeline(**data_dict)
            augmented.append(aug_dict["image"])

        return torch.stack(augmented, dim=0)

    def _random_crop_3d(self, x: torch.Tensor, crop_size: tuple) -> torch.Tensor:
        """
        Extract random 3D crops from batch of volumes.

        Args:
            x: (B, C, D, H, W) tensor
            crop_size: (cd, ch, cw) crop dimensions

        Returns:
            crops: (B, C, cd, ch, cw) tensor of random crops
        """
        B, C, D, H, W = x.shape
        cd, ch, cw = crop_size

        # Ensure crop fits within volume
        cd = min(cd, D)
        ch = min(ch, H)
        cw = min(cw, W)

        crops = []
        for i in range(B):
            # Random start positions
            d_start = torch.randint(0, max(1, D - cd + 1), (1,)).item()
            h_start = torch.randint(0, max(1, H - ch + 1), (1,)).item()
            w_start = torch.randint(0, max(1, W - cw + 1), (1,)).item()

            crop = x[i, :, d_start:d_start+cd, h_start:h_start+ch, w_start:w_start+cw]
            crops.append(crop)

        return torch.stack(crops, dim=0)

    def _prepare_multi_view_batch(self, batch):
        """
        Generate multiple views of the input via augmentation with multi-scale cropping.

        Global views: full resolution with light augmentation
        Local views: random crops resized back to full resolution with strong augmentation

        Returns:
            global_views: list of (B, C, D, H, W) tensors
            local_views: list of (B, C, D, H, W) tensors
        """
        x = batch["image"]  # (B, C, D, H, W)
        target_size = x.shape[2:]  # (D, H, W)

        # Global views: full resolution
        global_views = [
            self._apply_augmentation(x, self.global_aug)
            for _ in range(self.num_global_views)
        ]

        # Local views: random crops resized to full resolution
        local_views = []
        for _ in range(self.num_local_views):
            # Extract random crop
            crop = self._random_crop_3d(x, self.local_crop_size)
            # Resize back to target size (encoder expects fixed input size)
            crop = F.interpolate(crop, size=target_size, mode='trilinear', align_corners=False)
            # Apply strong augmentation
            aug_crop = self._apply_augmentation(crop, self.local_aug)
            local_views.append(aug_crop)

        return global_views, local_views

    def _forward_encoder(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder with global average pooling and projection.

        Args:
            x: (B, C, D, H, W) input tensor

        Returns:
            emb: (B, embed_dim) embedding tensor (for potential linear probe)
            proj: (B, proj_dim) projection tensor (for SIGReg loss)
        """
        # Encoder returns list with single feature map
        features = self.encoder(x)
        feat = features[0]  # (B, embed_dim, d, h, w)

        # Global average pooling over spatial dimensions
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

        return loss, loss_dict

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

        return loss_dict, inputs, targets_dict, outputs_dict

    def _compute_validation_loss(self, outputs, targets_dict, loss_fns):
        """Validation loss is computed in _validation_step."""
        # This is called by base class but we handle loss in _validation_step
        return {"lejepa": 0.0}

    def _train_epoch(self, model, dataloader, optimizer, loss_fns, scaler, epoch, use_amp=True):
        """Override to handle LeJEPA training loop with cosine annealing scheduler."""
        model.train()
        self.encoder.train()
        self.projector.train()

        total_loss = 0.0
        total_invariance_loss = 0.0
        total_sigreg_loss = 0.0
        num_batches = 0

        # Set up proper scheduler on first epoch
        if epoch == 0 and not hasattr(self, '_scheduler_initialized'):
            steps_per_epoch = len(dataloader)
            self.scheduler = self._setup_scheduler(optimizer, steps_per_epoch)
            self._scheduler_initialized = True

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=self.rank != 0)

        for batch_idx, data_dict in enumerate(pbar):
            optimizer.zero_grad()

            if use_amp and self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    inputs, targets_dict, outputs_dict = self._get_model_outputs(
                        model, data_dict
                    )
                    loss, loss_dict = self._compute_train_loss(
                        outputs_dict, targets_dict, loss_fns
                    )

                scaler.scale(loss).backward()

                if self.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.projector.parameters()),
                        self.grad_clip
                    )

                scaler.step(optimizer)
                scaler.update()
            else:
                inputs, targets_dict, outputs_dict = self._get_model_outputs(
                    model, data_dict
                )
                loss, loss_dict = self._compute_train_loss(
                    outputs_dict, targets_dict, loss_fns
                )

                loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.projector.parameters()),
                        self.grad_clip
                    )

                optimizer.step()

            # Step the scheduler (per-step)
            if hasattr(self, 'scheduler'):
                self.scheduler.step()

            total_loss += loss_dict["loss"]
            total_invariance_loss += loss_dict["invariance_loss"]
            total_sigreg_loss += loss_dict["sigreg_loss"]
            num_batches += 1

            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(
                {
                    "loss": f"{loss_dict['loss']:.4f}",
                    "inv": f"{loss_dict['invariance_loss']:.4f}",
                    "sig": f"{loss_dict['sigreg_loss']:.4f}",
                    "lr": f"{current_lr:.2e}",
                }
            )

        avg_loss = total_loss / max(num_batches, 1)
        avg_inv = total_invariance_loss / max(num_batches, 1)
        avg_sig = total_sigreg_loss / max(num_batches, 1)

        return {
            "loss": avg_loss,
            "invariance_loss": avg_inv,
            "sigreg_loss": avg_sig,
        }
