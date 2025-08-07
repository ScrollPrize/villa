import torch
from torch import autocast
import os
from torch import nn
from torch._dynamo import OptimizedModule
from typing import Tuple, Union
from tqdm import tqdm
from contextlib import nullcontext

from vesuvius.models.training.train import BaseTrainer
from vesuvius.models.training.trainers.self_supervised.warmup_lr import Lin_incr_LRScheduler, PolyLRScheduler_offset
from vesuvius.models.utils import empty_cache

from vesuvius.models.build.build_network_from_config import NetworkFromConfig


class MAEMSELoss(nn.Module):
    """MSE loss for Masked Autoencoder that only computes loss on masked regions."""
    def __init__(self):
        super().__init__()
        
    def forward(self, model_output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss only on masked regions.
        
        Args:
            model_output: Reconstructed image
            target: Original image
            mask: Binary mask (1 for unmasked/visible, 0 for masked)
        
        Returns:
            Loss value
        """
        # Compute reconstruction loss only on masked patches (where mask == 0)
        reconstruction_loss = (model_output - target) ** 2
        # Apply inverse mask (1 - mask) to focus on masked regions
        masked_loss = reconstruction_loss * (1 - mask)
        # Average over masked regions
        loss = torch.sum(masked_loss) / torch.sum(1 - mask)
        return loss


class TrainEVAMAE(BaseTrainer):
    def __init__(self, mgr=None, verbose: bool = True):
        # Configure MAE-specific targets before parent init
        if mgr is not None:
            # Set up MAE reconstruction target
            if not hasattr(mgr, 'targets'):
                mgr.targets = {}
            mgr.targets['mae'] = {
                'num_classes': 1,  # Regression task
                'weight': 1.0
            }
        
        super().__init__(mgr, verbose)
        
        # MAE-specific configuration
        self.drop_path_rate = 0.2
        self.attention_drop_rate = 0.2
        self.grad_clip = 1
        self.initial_lr = 3e-5
        self.weight_decay = 5e-3
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.training_stage = None
        self.mask_ratio = getattr(mgr, 'mask_ratio', 0.75)  # Default to 75% masking
        self.vit_patch_size = getattr(mgr, 'vit_patch_size', (16, 16, 16))
        self.current_epoch = 0
        
    def _build_loss(self):
        """Build MAE loss function."""
        # Return a dict with a single 'mae' target that uses our MAE loss
        return {'mae': [(MAEMSELoss(), 1.0)]}
    
    def _get_optimizer(self, model):
        """Override to create MAE-specific optimizer with warmup."""
        # Determine training stage based on current epoch
        if self.current_epoch < self.warmup_duration_whole_net:
            stage = "warmup_all"
        else:
            stage = "train"
        
        if hasattr(self, 'training_stage') and self.training_stage == stage and hasattr(self, 'optimizer'):
            return self.optimizer
        
        params = model.parameters()
        
        if stage == "warmup_all":
            print("Training whole net with warmup")
            optimizer = torch.optim.AdamW(
                params, self.initial_lr, weight_decay=self.weight_decay, amsgrad=False, betas=(0.9, 0.98)
            )
            self.training_stage = stage
        else:
            print("Training whole net with default schedule")
            if hasattr(self, 'training_stage') and self.training_stage == "warmup_all" and hasattr(self, 'optimizer'):
                # Keep existing optimizer to maintain momentum
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.AdamW(
                    params,
                    self.initial_lr,
                    weight_decay=self.weight_decay,
                    amsgrad=False,
                    betas=(0.9, 0.98)
                )
            self.training_stage = stage
        
        empty_cache(self.device)
        return optimizer
    
    def _get_scheduler(self, optimizer):
        """Override to create MAE-specific learning rate scheduler."""
        if self.current_epoch < self.warmup_duration_whole_net:
            return Lin_incr_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net)
        else:
            return PolyLRScheduler_offset(
                optimizer, self.initial_lr, self.mgr.max_epoch, self.warmup_duration_whole_net
            )
    
    def _get_model_outputs(self, model, data_dict):
        """Override to handle MAE forward pass with masking."""
        inputs = data_dict["image"].to(self.device)
        
        # MAE forward pass should return (reconstruction, keep_indices)
        # The model needs to handle masking internally
        outputs, keep_indices = model(inputs)
        
        # Create mask from keep_indices for loss calculation
        image_size = inputs.shape[2:]  # (D, H, W)
        mask = self.create_mask(keep_indices, image_size, self.vit_patch_size)
        
        # Store for loss calculation
        self._current_mask = mask
        self._current_inputs = inputs
        
        # Return in format expected by parent class
        # We'll handle the MAE loss in _compute_train_loss
        targets_dict = {'mae': inputs}
        outputs_dict = {'mae': outputs}
        
        return inputs, targets_dict, outputs_dict
    
    def _compute_train_loss(self, outputs, targets_dict, loss_fns):
        """Override to compute MAE loss with masking."""
        # Get the MAE loss function
        mae_loss_fn = loss_fns['mae'][0][0]  # First loss function for 'mae' target
        
        # Compute MAE loss using the mask we stored
        mae_output = outputs['mae']
        mae_target = targets_dict['mae']
        
        # Use the mask we computed in _get_model_outputs
        loss = mae_loss_fn(mae_output, mae_target, self._current_mask)
        
        task_losses = {'mae': loss.detach().cpu().item()}
        return loss, task_losses
    
    def _compute_validation_loss(self, outputs, targets_dict, loss_fns):
        """Override to compute MAE validation loss."""
        # Similar to train loss but for validation
        mae_loss_fn = loss_fns['mae'][0][0]
        mae_output = outputs['mae']
        mae_target = targets_dict['mae']
        
        # For validation, we need to recompute the mask
        # This is a simplified approach - ideally the model should provide this
        with torch.no_grad():
            # Assuming the model stores the mask or we can access it somehow
            # For now, use a default mask
            if hasattr(self, '_current_mask'):
                mask = self._current_mask
            else:
                # Fallback: create a dummy mask
                mask = torch.ones_like(mae_target[:, :1])  # Shape (B, 1, D, H, W)
        
        loss = mae_loss_fn(mae_output, mae_target, mask)
        task_losses = {'mae': loss.detach().cpu().item()}
        return task_losses
    
    @staticmethod
    def create_mask(
            keep_indices: torch.Tensor, image_size: Tuple[int, int, int], patch_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Create a mask tensor (1 for unmasked, 0 for masked) based on keep_indices.

        Args:
            keep_indices (torch.Tensor): Tensor of shape (B, num_kept_patches) indicating retained patches.
            image_size (Tuple[int, int, int]): Size of the full image (D, H, W).
            patch_size (Tuple[int, int, int]): Size of each patch (D_patch, H_patch, W_patch).

        Returns:
            torch.Tensor: Mask tensor of shape (B, 1, D, H, W) with 1 for unmasked and 0 for masked.
        """
        B, num_kept_patches = keep_indices.shape
        D, H, W = image_size
        D_patch, H_patch, W_patch = patch_size
        
        # Calculate the number of patches along each dimension
        num_patches_d = D // D_patch
        num_patches_h = H // H_patch
        num_patches_w = W // W_patch
        num_patches = num_patches_d * num_patches_h * num_patches_w
        
        # Create a flat mask of 0s with shape (B, num_patches)
        flat_mask = torch.zeros(B, num_patches, device=keep_indices.device)
        
        # Set retained patches to 1
        flat_mask.scatter_(1, keep_indices, 1)
        
        # Reshape to patch grid and expand to full image size
        mask = flat_mask.view(B, num_patches_d, num_patches_h, num_patches_w)
        mask = (
            mask.repeat_interleave(D_patch, dim=1).repeat_interleave(H_patch, dim=2).repeat_interleave(W_patch, dim=3)
        )
        mask = mask.unsqueeze(1)  # Add channel dimension (B, 1, D, H, W)
        return mask
