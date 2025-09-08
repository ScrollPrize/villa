import torch
import os
from torch import nn
from torch._dynamo import OptimizedModule
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

        self.mask_ratio = 0.75  # Default to 75% masking
        self.drop_path_rate = 0.2  # Stochastic depth
        self.attention_drop_rate = 0.0  # Attention dropout
        self.grad_clip = 1
        self.initial_lr = 3e-4
        self.weight_decay = 5e-2
        self.warmup_duration_whole_net = 50  # lin increase whole network
        self.vit_patch_size = (8, 8, 8)
        self.decoder_depth = 2  # Number of Eva decoder layers for MAE
        self.decoder_num_heads = 12  # Number of heads in decoder
        
        if mgr is not None:
            self.mask_ratio = getattr(mgr, 'mask_ratio', self.mask_ratio)
            self.vit_patch_size = getattr(mgr, 'vit_patch_size', self.vit_patch_size)
            
            # Override learning rate, weight decay, and warmup for MAE training
            mgr.initial_lr = self.initial_lr
            mgr.weight_decay = self.weight_decay
            mgr.warmup_duration = self.warmup_duration_whole_net
            
            # Set only_spatial_and_intensity flag for MAE training
            mgr.only_spatial_and_intensity = True
            
            if not hasattr(mgr, 'targets'):
                mgr.targets = {}
            mgr.targets['mae'] = {
                'num_classes': 1,  # Regression task
                'out_channels': 1,  # Reconstruct single channel input
                'weight': 1.0
            }
            
            if not hasattr(mgr, 'model_config'):
                mgr.model_config = {}
            
            mgr.model_config['patch_drop_rate'] = self.mask_ratio
            mgr.model_config['drop_path_rate'] = self.drop_path_rate
            mgr.model_config['attn_drop_rate'] = self.attention_drop_rate
            mgr.model_config['patch_embed_size'] = self.vit_patch_size
            mgr.model_config['decoder_depth'] = self.decoder_depth
            mgr.model_config['decoder_num_heads'] = self.decoder_num_heads
        
        super().__init__(mgr, verbose)
        
        self.training_stage = None
        self.current_epoch = 0
        self._is_mae_training = True  # Flag to keep model in training mode during validation
        
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
                params, self.mgr.initial_lr, weight_decay=self.mgr.weight_decay, amsgrad=False, betas=(0.9, 0.98)
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
                    self.mgr.initial_lr,
                    weight_decay=self.mgr.weight_decay,
                    amsgrad=False,
                    betas=(0.9, 0.98)
                )
            self.training_stage = stage
        
        empty_cache(self.device)
        return optimizer
    
    def _get_scheduler(self, optimizer):
        """Override to create initial MAE scheduler (warmup)."""
        # Always start with warmup scheduler
        # The _update_scheduler_for_epoch method will handle switching at epoch 50
        scheduler = Lin_incr_LRScheduler(optimizer, self.mgr.initial_lr, self.warmup_duration_whole_net)
        return scheduler, False  # Epoch-based scheduler
    
    def _get_model_outputs(self, model, data_dict):
        """Override to handle MAE forward pass with masking."""
        inputs = data_dict["image"].to(self.device)
        
        # Call model with return_mae_mask=True to get restoration mask
        outputs, restoration_mask = model(inputs, return_mae_mask=True)
        
        # - Shape: (B, 1, W, H, D) at full resolution
        # - Values: True/1 for visible patches, False/0 for masked patches
        # loss computes on masked patches (where mask == 0)
        
        mask = restoration_mask.float() if restoration_mask.dtype == torch.bool else restoration_mask
        
        self._current_mask = mask
        self._current_inputs = inputs
        
        targets_dict = {'mae': inputs}
        outputs_dict = outputs
        
        return inputs, targets_dict, outputs_dict
    
    def _compute_train_loss(self, outputs, targets_dict, loss_fns):
        """Override to compute MAE loss with masking."""
        # Get the MAE loss function
        mae_loss_fn = loss_fns['mae'][0][0]  # First loss function for 'mae' target
        
        mae_output = outputs['mae']
        mae_target = targets_dict['mae']
        
        loss = mae_loss_fn(mae_output, mae_target, self._current_mask)
        
        task_losses = {'mae': loss.detach().cpu().item()}
        return loss, task_losses
    
    def _validation_step(self, model, data_dict, loss_fns, use_amp):
        """Override validation step to use MAE-specific forward pass."""
        from contextlib import nullcontext
        
        inputs = data_dict["image"].to(self.device)
        targets_dict = {
            k: v.to(self.device)
            for k, v in data_dict.items()
            if k not in ["image", "patch_info", "is_unlabeled"]
        }
        
        if use_amp:
            if self.device.type == 'cuda':
                context = torch.amp.autocast('cuda')
            else:
                context = torch.amp.autocast(self.device.type)
        else:
            context = nullcontext()
        
        with context:
            outputs, restoration_mask = model(inputs, return_mae_mask=True)
            mask = restoration_mask.float() if restoration_mask.dtype == torch.bool else restoration_mask
            self._current_mask = mask
            targets_dict['mae'] = inputs
            
            task_losses = self._compute_validation_loss(outputs, targets_dict, loss_fns)
        
        return task_losses, inputs, targets_dict, outputs
    
    def _compute_validation_loss(self, outputs, targets_dict, loss_fns):
        """Override to compute MAE validation loss."""
        mae_loss_fn = loss_fns['mae'][0][0]
        mae_output = outputs['mae']
        mae_target = targets_dict['mae']
        
        if hasattr(self, '_current_mask'):
            mask = self._current_mask
        else:
            raise RuntimeError(
                "No mask found for validation loss computation. "
                "This should have been set in _validation_step."
            )
        
        loss = mae_loss_fn(mae_output, mae_target, mask)
        task_losses = {'mae': loss.detach().cpu().item()}
        return task_losses
    
    def _update_scheduler_for_epoch(self, scheduler, optimizer, epoch):
        """
        Override to switch from warmup to main scheduler at epoch 50.
        """
        # Track current epoch for other methods that might need it
        self.current_epoch = epoch
        
        # Check if we need to switch schedulers
        if epoch == 0 and self.training_stage != "warmup_all":
            # First epoch - use warmup scheduler
            print(f"[Epoch {epoch}] Initializing warmup scheduler (first {self.warmup_duration_whole_net} epochs)")
            scheduler = Lin_incr_LRScheduler(optimizer, self.mgr.initial_lr, self.warmup_duration_whole_net)
            self.training_stage = "warmup_all"
            return scheduler, False  # Epoch-based scheduler
            
        elif epoch == self.warmup_duration_whole_net and self.training_stage != "train":
            # Switch to main training scheduler
            print(f"[Epoch {epoch}] Switching from warmup to PolyLR scheduler")
            scheduler = PolyLRScheduler_offset(
                optimizer, self.mgr.initial_lr, self.mgr.max_epoch, self.warmup_duration_whole_net
            )
            self.training_stage = "train"
            return scheduler, False  # Epoch-based scheduler
        
        # Otherwise keep the current scheduler
        return scheduler, False
    
