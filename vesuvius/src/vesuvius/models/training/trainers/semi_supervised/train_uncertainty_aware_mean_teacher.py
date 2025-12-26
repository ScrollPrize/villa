from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

from vesuvius.models.training.train import BaseTrainer
from vesuvius.models.training.trainers.semi_supervised import ramps
from vesuvius.models.training.trainers.semi_supervised.two_stream_batch_sampler import TwoStreamBatchSampler


# reimplemented from https://github.com/HiLab-git/SSL4MIS/blob/master/code/train_uncertainty_aware_mean_teacher_3D.py


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax of both sides and returns MSE loss

    Returns element-wise squared differences
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    mse_loss = (input_softmax - target_softmax) ** 2
    
    return mse_loss


class TrainUncertaintyAwareMeanTeacher(BaseTrainer):
    def __init__(self, mgr=None, verbose: bool = True):
        super().__init__(mgr, verbose)
        
        self.ema_decay = getattr(mgr, 'ema_decay', 0.99)
        self.consistency_weight = getattr(mgr, 'consistency_weight', 0.1)
        self.consistency_rampup = getattr(mgr, 'consistency_rampup', 200.0)
        self.uncertainty_threshold_start = getattr(mgr, 'uncertainty_threshold_start', 0.75)
        self.uncertainty_threshold_end = getattr(mgr, 'uncertainty_threshold_end', 1.0)
        self.uncertainty_T = getattr(mgr, 'uncertainty_T', 4)  # Number of stochastic forward passes

        # Validate uncertainty_T is even (required for paired forward passes)
        if self.uncertainty_T % 2 != 0:
            old_T = self.uncertainty_T
            self.uncertainty_T = self.uncertainty_T - 1
            if self.uncertainty_T < 2:
                self.uncertainty_T = 2
            print(f"Warning: uncertainty_T={old_T} is odd. Adjusted to {self.uncertainty_T} for paired forward passes.")

        self.noise_scale = getattr(mgr, 'noise_scale', 0.1)  # Noise scale for stochastic augmentation
        self.labeled_batch_size = getattr(mgr, 'labeled_batch_size', mgr.train_batch_size // 2)
        
        # Semi-supervised data split parameters
        self.labeled_ratio = getattr(mgr, 'labeled_ratio', 1.0)  # Fraction of data to use as labeled
        self.num_labeled = getattr(mgr, 'num_labeled', None)
        
        # Deep supervision complicates per-sample masking; keep it off for SSL trainers
        self.mgr.enable_deep_supervision = False

        # Disable scaling augmentation - it requires padding which causes issues
        # with consistency loss (padded regions get different noise, creating fake disagreement)
        self.mgr.no_scaling = True

        self.ema_model = None
        self.global_step = 0
        self.labeled_indices = None
        self.unlabeled_indices = None
    
    def _create_ema_model(self, model):
        """Create an EMA (teacher) model from the student model."""
        ema_model = self._build_model()
        ema_model = ema_model.to(self.device)
        
        for param_student, param_teacher in zip(model.parameters(), ema_model.parameters()):
            param_teacher.data.copy_(param_student.data)
            param_teacher.requires_grad = False
        
        ema_model.eval()
        return ema_model
    
    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    
    def _get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.consistency_weight * ramps.sigmoid_rampup(epoch, self.consistency_rampup)

    def _get_noise_bounds(self, inputs):
        """Compute noise clamp bounds relative to input statistics.

        The original hardcoded ±0.2 clamp assumes z-score normalized data (std~1).
        This method adapts the bounds to the actual input scale, so the noise
        remains proportionally meaningful regardless of normalization scheme.
        """
        input_std = inputs.std()
        # Clamp noise to roughly 2x noise_scale * std
        # For z-score data with std~1, this gives ~0.2 matching original behavior
        bound = 2.0 * self.noise_scale * max(input_std.item(), 0.1)
        return bound

    def _configure_dataloaders(self, train_dataset, val_dataset=None):
        """
        Override to use TwoStreamBatchSampler for semi-supervised learning.
        This ensures each batch contains both labeled and unlabeled samples.
        """
        if val_dataset is None:
            val_dataset = train_dataset
        
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        
        if hasattr(self.mgr, 'seed'):
            np.random.seed(self.mgr.seed)
            if self.mgr.verbose:
                print(f"Using seed {self.mgr.seed} for labeled/unlabeled split")
        
        np.random.shuffle(indices)
        
        # Determine labeled/unlabeled indices using fast path (file-level)
        labeled_idx, unlabeled_idx = [], []
        used_fast_path = False
        if hasattr(train_dataset, 'get_labeled_unlabeled_patch_indices'):
            try:
                li, ui = train_dataset.get_labeled_unlabeled_patch_indices()
                li_set, ui_set = set(li), set(ui)
                labeled_idx = [i for i in indices if i in li_set]
                unlabeled_idx = [i for i in indices if i in ui_set]
                used_fast_path = True
                if self.mgr.verbose:
                    print("Using dataset fast-path for labeled/unlabeled split")
            except Exception as e:
                print(f"Fast-path split failed: {e}")
        if not used_fast_path:
            raise ValueError(
                "Dataset does not support fast labeled/unlabeled split. "
                "Use DatasetOrchestrator with data_format='image' (adapter='image') or implement "
                "get_labeled_unlabeled_patch_indices() on your dataset."
            )

        # Determine how many labeled to use
        if self.num_labeled is not None:
            num_labeled = min(self.num_labeled, len(labeled_idx))
        else:
            num_labeled = int(self.labeled_ratio * max(1, len(labeled_idx)))
        num_labeled = max(num_labeled, self.labeled_batch_size) if labeled_idx else 0

        # Build final ordered lists
        self.labeled_indices = labeled_idx[:num_labeled]
        self.unlabeled_indices = unlabeled_idx

        # Ensure we have enough labeled samples for batch composition
        if len(self.labeled_indices) < self.labeled_batch_size and len(labeled_idx) > len(self.labeled_indices):
            extra = self.labeled_batch_size - len(self.labeled_indices)
            self.labeled_indices = self.labeled_indices + labeled_idx[len(self.labeled_indices):len(self.labeled_indices)+extra]

        unlabeled_batch_size = self.mgr.train_batch_size - self.labeled_batch_size
        if len(self.unlabeled_indices) < unlabeled_batch_size:
            raise ValueError(
                f"Insufficient unlabeled data for semi-supervised training. "
                f"Need at least {unlabeled_batch_size} unlabeled samples per batch, "
                f"but only have {len(self.unlabeled_indices)} unlabeled samples total. "
                f"Either reduce labeled_batch_size ({self.labeled_batch_size}), "
                f"reduce labeled_ratio ({self.labeled_ratio}), or increase dataset size."
            )
        
        print(
            f"Semi-supervised split (patch-level): {len(self.labeled_indices)} labeled patches "
            f"(from {len(labeled_idx)}), {len(self.unlabeled_indices)} unlabeled patches"
        )
        print(
            f"Batch composition: {self.labeled_batch_size} labeled + {unlabeled_batch_size} unlabeled = {self.mgr.train_batch_size} total")
        
        batch_sampler = TwoStreamBatchSampler(
            primary_indices=self.labeled_indices,
            secondary_indices=self.unlabeled_indices,
            batch_size=self.mgr.train_batch_size,
            secondary_batch_size=unlabeled_batch_size
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            pin_memory=(True if self.device == 'cuda' else False),
            num_workers=self.mgr.train_num_dataloader_workers
        )
        
        # --- choose validation indices ---
        # If an external validation dataset is provided (e.g., via --val-dir),
        # its indices are independent from the training dataset. In that case
        # evaluate over the full validation set (or a sampler can downselect later).
        # Check if datasets share the same source (not just object identity)
        train_path = getattr(train_dataset, 'data_path', None)
        val_path = getattr(val_dataset, 'data_path', None)
        same_source = (train_path == val_path) if (train_path and val_path) else False

        if val_dataset is not train_dataset and not same_source:
            if self.mgr.verbose:
                print("Using external validation dataset for uncertainty-aware mean teacher; evaluating on full validation set")
            val_indices = list(range(len(val_dataset)))
        else:
            # Same source - use only labeled indices for validation
            train_val_split = self.mgr.tr_val_split
            val_split = int(np.floor((1 - train_val_split) * max(1, len(self.labeled_indices))))
            val_indices = self.labeled_indices[-val_split:] if val_split > 0 else self.labeled_indices[-5:]

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            sampler=SubsetRandomSampler(val_indices),
            pin_memory=(True if self.device == 'cuda' else False),
            num_workers=self.mgr.train_num_dataloader_workers
        )
        
        self.train_indices = list(self.labeled_indices)
        
        return train_dataloader, val_dataloader, self.labeled_indices, val_indices
    
    def _get_model_outputs(self, model, data_dict):
        """Override to handle both labeled and unlabeled data"""
        inputs = data_dict["image"].to(self.device)
        targets_dict = {
            k: v.to(self.device)
            for k, v in data_dict.items()
            if k not in ["image", "patch_info", "is_unlabeled", "dataset_indices"]
        }
        
        batch_size = inputs.shape[0]

        # Build unlabeled mask for this batch
        def as_float_mask(val):
            if isinstance(val, torch.Tensor):
                return val.to(self.device).float()
            if isinstance(val, (list, tuple)):
                return torch.tensor(val, device=self.device, dtype=torch.float32)
            if isinstance(val, bool):
                return torch.full((batch_size,), float(val), device=self.device)
            return torch.zeros(batch_size, device=self.device)

        if model.training and batch_size == self.mgr.train_batch_size:
            # TwoStreamBatchSampler order: [labeled..., unlabeled...]
            is_unlabeled = torch.cat([
                torch.zeros(self.labeled_batch_size, device=self.device),
                torch.ones(batch_size - self.labeled_batch_size, device=self.device)
            ], dim=0)
        else:
            # Validation/other: default to labeled unless dataset provides flags
            is_unlabeled = as_float_mask(data_dict.get('is_unlabeled', None))
        
        outputs = model(inputs)
        
        # Store is_unlabeled separately - don't add to targets_dict as it breaks visualization
        if model.training:
            targets_dict['is_unlabeled'] = is_unlabeled
        
        return inputs, targets_dict, outputs
    
    def _compute_uncertainty(self, unlabeled_inputs, autocast_ctx):
        """Compute uncertainty using multiple stochastic forward passes"""
        batch_size = unlabeled_inputs.shape[0]
        T = self.uncertainty_T
        
        volume_batch_r = unlabeled_inputs.repeat(2, 1, 1, 1, 1)
        stride = volume_batch_r.shape[0] // 2
        
        # Get number of output channels (assuming single task or first task)
        num_classes = self.mgr.out_channels[0] if isinstance(self.mgr.out_channels, tuple) else self.mgr.out_channels
        
        if unlabeled_inputs.dim() == 5:  # 3D case: [B, C, D, H, W]
            _, c, d, h, w = unlabeled_inputs.shape
            preds = torch.zeros([stride * T, num_classes, d, h, w]).to(self.device)
        else:  # 2D case: [B, C, H, W]
            _, c, h, w = unlabeled_inputs.shape
            preds = torch.zeros([stride * T, num_classes, h, w]).to(self.device)
        
        noise_bound = self._get_noise_bounds(volume_batch_r)
        with torch.no_grad():
            for i in range(T // 2):
                noise = torch.clamp(
                    torch.randn_like(volume_batch_r) * self.noise_scale,
                    -noise_bound, noise_bound
                )
                ema_inputs = volume_batch_r + noise
                with autocast_ctx:
                    ema_outputs = self.ema_model(ema_inputs)
                    # Extract the first task's output from the dictionary
                    first_task = list(ema_outputs.keys())[0]
                    if first_task == '_inputs':
                        first_task = list(ema_outputs.keys())[1] if len(ema_outputs.keys()) > 1 else list(ema_outputs.keys())[0]
                    preds[2 * stride * i:2 * stride * (i + 1)] = ema_outputs[first_task]
        
        preds = F.softmax(preds, dim=1)
        if unlabeled_inputs.dim() == 5:
            preds = preds.reshape(T, stride, num_classes, d, h, w)
        else:
            preds = preds.reshape(T, stride, num_classes, h, w)
        
        mean_pred = torch.mean(preds, dim=0)  # [stride, num_classes, ...]
        
        # Compute entropy as uncertainty measure
        uncertainty = -1.0 * torch.sum(mean_pred * torch.log(mean_pred + 1e-6), dim=1, keepdim=True)
        
        return uncertainty, mean_pred
    
    def _compute_train_loss(self, outputs, targets_dict, loss_fns, autocast_ctx=None):
        """
        Override to add consistency loss with uncertainty weighting
        """
        
        # Get unlabeled mask
        is_unlabeled = targets_dict.get('is_unlabeled', None)
        
        # doesnt really make sense to use this trainer without unlabeled data
        if is_unlabeled is None or not is_unlabeled.any():
            raise ValueError(
                "UncertaintyAwareMeanTeacher trainer requires unlabeled data but none was found in this batch. "
                "This semi-supervised trainer does not make sense without unlabeled data. "
                "Please ensure your dataset has unlabeled samples and the labeled_ratio is < 1.0."
            )
        
        labeled_mask = ~is_unlabeled.bool()
        unlabeled_mask = is_unlabeled.bool()
        
        # filter outputs and targets for labeled data only -- we dont want to (and probably would not even be able to)
        # attempt to compute supervised loss on unlabeled data
        labeled_outputs = {}
        labeled_targets = {}
        
        for key, value in outputs.items():
            if key != '_inputs':  # Skip our temporary storage
                labeled_outputs[key] = value[labeled_mask]
        
        for key, value in targets_dict.items():
            if key != 'is_unlabeled':  # Skip the unlabeled flag
                labeled_targets[key] = value[labeled_mask]
        
        # Compute supervised loss only on labeled data
        total_loss, task_losses = super()._compute_train_loss(labeled_outputs, labeled_targets, loss_fns)
        
        inputs = outputs.get('_inputs', None)
        if inputs is None:
            raise ValueError("_inputs not found in outputs. This is required for consistency loss computation.")
        
        unlabeled_inputs = inputs[unlabeled_mask]
        
        if autocast_ctx is None:
            from contextlib import nullcontext
            autocast_ctx = nullcontext()
        
        uncertainty, mean_pred = self._compute_uncertainty(unlabeled_inputs, autocast_ctx)
        
        noise_bound = self._get_noise_bounds(unlabeled_inputs)
        with torch.no_grad():
            noise = torch.clamp(
                torch.randn_like(unlabeled_inputs) * self.noise_scale,
                -noise_bound, noise_bound
            )
            teacher_inputs = unlabeled_inputs + noise
            with autocast_ctx:
                teacher_outputs = self.ema_model(teacher_inputs)
        
        first_task = list(outputs.keys())[0]
        if first_task == '_inputs':
            first_task = list(outputs.keys())[1] if len(outputs.keys()) > 1 else None
            if first_task is None:
                raise ValueError("No task outputs found besides _inputs")
        
        student_unlabeled = outputs[first_task][unlabeled_mask]
        teacher_unlabeled = teacher_outputs[first_task]
        
        # Compute consistency loss (element-wise)
        consistency_dist = softmax_mse_loss(student_unlabeled, teacher_unlabeled)
        
        # Apply uncertainty-based weighting
        # Use sigmoid ramp-up for threshold
        current_iter = self.global_step

        max_steps_per_epoch = getattr(self.mgr, 'max_steps_per_epoch', 100)
        max_epochs = getattr(self.mgr, 'max_epoch', 100)
        max_iterations = max_steps_per_epoch * max_epochs

        # Use num_classes for max entropy threshold (not hardcoded log(2) for binary)
        num_classes = self.mgr.out_channels[0] if isinstance(self.mgr.out_channels, tuple) else self.mgr.out_channels
        max_entropy = np.log(max(num_classes, 2))  # Ensure at least log(2)
        threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(current_iter, max_iterations)) * max_entropy

        mask = (uncertainty < threshold).float()
        while mask.dim() < consistency_dist.dim():
            mask = mask.unsqueeze(1)

        masked_loss = mask * consistency_dist
        consistency_loss = torch.sum(masked_loss) / (torch.sum(mask) + 1e-8)

        # Compute effective epoch from global_step for rampup
        effective_epoch = self.global_step / max(max_steps_per_epoch, 1)
        consistency_weight = self._get_current_consistency_weight(effective_epoch)
        
        weighted_consistency_loss = consistency_weight * consistency_loss
        total_loss = total_loss + weighted_consistency_loss
        
        task_losses['consistency'] = consistency_loss.detach().cpu().item()
        
        return total_loss, task_losses
    
    def _train_step(self, model, data_dict, loss_fns, use_amp, autocast_ctx, epoch, step, verbose=False,
                    scaler=None, optimizer=None, num_iters=None, grad_accumulate_n=1):
        """Override to store inputs in outputs and update EMA model"""
        
        self.global_step = epoch * (num_iters or getattr(self.mgr, 'max_steps_per_epoch', 100)) + step
        
        with autocast_ctx:
            inputs, targets_dict, outputs = self._get_model_outputs(model, data_dict)
            outputs['_inputs'] = inputs
            total_loss, task_losses = self._compute_train_loss(outputs, targets_dict, loss_fns, autocast_ctx)
        
        scaled_loss = total_loss / grad_accumulate_n
        scaler.scale(scaled_loss).backward()
        
        optimizer_stepped = False
        if (step + 1) % grad_accumulate_n == 0 or (step + 1) == num_iters:
            scaler.unscale_(optimizer)
            grad_clip = getattr(self.mgr, 'gradient_clip', 12.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_stepped = True
            
            # Update EMA model after optimizer step
            self._update_ema_variables(model, self.ema_model, self.ema_decay, self.global_step)

        # Capture unlabeled debug sample on first step of each epoch
        if step == 0 and hasattr(self, 'labeled_batch_size'):
            unlabeled_start = self.labeled_batch_size
            if inputs.shape[0] > unlabeled_start:
                # Capture first unlabeled sample
                self._debug_unlabeled_input = inputs[unlabeled_start:unlabeled_start+1].detach().clone()
                self._debug_unlabeled_student_pred = {
                    k: v[unlabeled_start:unlabeled_start+1].detach().clone()
                    for k, v in outputs.items() if k != '_inputs'
                }
                # Get teacher (EMA) predictions for pseudo-labels
                with torch.no_grad():
                    with autocast_ctx:
                        teacher_out = self.ema_model(self._debug_unlabeled_input)
                        self._debug_unlabeled_pseudo_label = {
                            k: v.detach().clone() for k, v in teacher_out.items()
                        }

        # Remove _inputs from outputs to avoid issues downstream
        outputs.pop('_inputs', None)
        
        # Remove is_unlabeled from targets_dict before returning to avoid breaking debug gif capture
        # The base trainer's debug sample capture logic expects only actual targets in targets_dict
        targets_dict_clean = {k: v for k, v in targets_dict.items() if k != 'is_unlabeled'}

        return total_loss, task_losses, inputs, targets_dict_clean, outputs, optimizer_stepped
    
    def _get_additional_checkpoint_data(self):
        """Return EMA model state for checkpoint saving."""
        if self.ema_model is not None:
            return {'ema_model': self.ema_model.state_dict()}
        return {}

    def _initialize_training(self):
        """Override to initialize EMA model after base initialization"""
        training_state = super()._initialize_training()

        # Create EMA model after the student model is initialized
        model = training_state['model']
        self.ema_model = self._create_ema_model(model)

        # Restore EMA model from checkpoint if available
        if hasattr(self, '_checkpoint_ema_state') and self._checkpoint_ema_state is not None:
            try:
                self.ema_model.load_state_dict(self._checkpoint_ema_state)
                print("Restored EMA model from checkpoint")
                del self._checkpoint_ema_state
            except Exception as e:
                print(f"Warning: Failed to restore EMA model from checkpoint: {e}")
                print("Using freshly initialized EMA model")
        else:
            print(f"Created fresh EMA model with decay factor: {self.ema_decay}")

        print(f"Uncertainty estimation using {self.uncertainty_T} forward passes")
        print(f"Consistency weight ramp-up over {self.consistency_rampup} epochs")

        return training_state
