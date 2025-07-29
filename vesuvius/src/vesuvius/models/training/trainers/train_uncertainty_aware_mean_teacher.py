import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from contextlib import nullcontext
from tqdm import tqdm
from vesuvius.models.training.train import BaseTrainer
from vesuvius.models.training.auxiliary_tasks import compute_auxiliary_loss
import gc
from collections import deque
from torch.utils.data import Sampler
import random


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def softmax_mse_loss(input_logits, target_logits, use_sigmoid=False):
    """Takes softmax on both sides and returns MSE loss
    
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if use_sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
    
    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss


class TwoStreamBatchSampler(Sampler):
    """Samples batches with a fixed number of labeled and unlabeled samples.
    
    Args:
        labeled_indices: List of indices for labeled samples
        unlabeled_indices: List of indices for unlabeled samples
        batch_size: Total batch size
        labeled_batch_size: Number of labeled samples per batch
    """
    
    def __init__(self, labeled_indices, unlabeled_indices, batch_size, labeled_batch_size):
        self.labeled_indices = labeled_indices
        self.unlabeled_indices = unlabeled_indices
        self.batch_size = batch_size
        self.labeled_batch_size = labeled_batch_size
        self.unlabeled_batch_size = batch_size - labeled_batch_size
        
        assert self.labeled_batch_size > 0, "Need at least one labeled sample per batch"
        assert self.unlabeled_batch_size > 0, "Need at least one unlabeled sample per batch"
        assert len(self.labeled_indices) >= self.labeled_batch_size, "Not enough labeled samples"
        assert len(self.unlabeled_indices) >= self.unlabeled_batch_size, "Not enough unlabeled samples"
        
    def __iter__(self):
        # Shuffle indices
        labeled_indices = self.labeled_indices.copy()
        unlabeled_indices = self.unlabeled_indices.copy()
        random.shuffle(labeled_indices)
        random.shuffle(unlabeled_indices)
        
        # Calculate number of batches based on labeled data
        n_batches = len(labeled_indices) // self.labeled_batch_size
        
        for i in range(n_batches):
            # Get labeled samples for this batch
            labeled_batch = labeled_indices[i * self.labeled_batch_size:(i + 1) * self.labeled_batch_size]
            
            # Get unlabeled samples (cycle through if needed)
            unlabeled_start = (i * self.unlabeled_batch_size) % len(unlabeled_indices)
            unlabeled_end = unlabeled_start + self.unlabeled_batch_size
            
            if unlabeled_end <= len(unlabeled_indices):
                unlabeled_batch = unlabeled_indices[unlabeled_start:unlabeled_end]
            else:
                # Wrap around
                unlabeled_batch = unlabeled_indices[unlabeled_start:] + unlabeled_indices[:unlabeled_end - len(unlabeled_indices)]
            
            # Combine and yield
            batch = labeled_batch + unlabeled_batch
            yield batch
    
    def __len__(self):
        return len(self.labeled_indices) // self.labeled_batch_size


class UncertaintyAwareMeanTeacher3DTrainer(BaseTrainer):
    """
    Uncertainty-aware Mean Teacher trainer for semi-supervised learning.
    
    This trainer implements the uncertainty-aware mean teacher approach where:
    - A student model is trained with labeled data using supervised loss
    - A teacher model (EMA of student) provides pseudo-labels for unlabeled data
    - Uncertainty estimation filters out unreliable pseudo-labels
    - Consistency loss enforces agreement between student and teacher on unlabeled data
    """
    
    def __init__(self, mgr=None, verbose: bool = True):
        super().__init__(mgr, verbose)
        
        # Mean teacher specific parameters
        self.ema_decay = getattr(mgr, 'ema_decay', 0.99)
        self.consistency_weight = getattr(mgr, 'consistency_weight', 0.1)
        self.consistency_rampup = getattr(mgr, 'consistency_rampup', 200.0)
        self.uncertainty_threshold_start = getattr(mgr, 'uncertainty_threshold_start', 0.75)
        self.uncertainty_threshold_end = getattr(mgr, 'uncertainty_threshold_end', 1.0)
        self.uncertainty_T = getattr(mgr, 'uncertainty_T', 8)  # Number of stochastic forward passes
        self.noise_scale = getattr(mgr, 'noise_scale', 0.1)  # Noise scale for stochastic augmentation
        
    def _create_ema_model(self, model):
        """Create an EMA (teacher) model from the student model."""
        ema_model = self._build_model()
        ema_model = ema_model.to(self.device)
        
        # Initialize EMA model with student weights
        for param_student, param_teacher in zip(model.parameters(), ema_model.parameters()):
            param_teacher.data.copy_(param_student.data)
            param_teacher.requires_grad = False  # Teacher doesn't need gradients
            
        return ema_model
    
    def update_ema_variables(self, model, ema_model, alpha, global_step):
        """Update EMA model weights using exponential moving average."""
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    
    def get_current_consistency_weight(self, epoch, consistency_rampup):
        """Get consistency weight with ramp-up schedule."""
        return self.consistency_weight * sigmoid_rampup(epoch, consistency_rampup)
    
    def compute_uncertainty_and_teacher_outputs(self, ema_model, unlabeled_inputs, T=8):
        """
        Compute uncertainty and teacher predictions using multiple stochastic forward passes.
        
        Returns:
            uncertainty: Uncertainty map for each spatial location
            mean_predictions: Dict of mean predictions for each task across T forward passes
        """
        B, C, D, H, W = unlabeled_inputs.shape
        
        # Collect predictions for all tasks
        all_task_preds = {}
        
        with torch.no_grad():
            for i in range(T):
                # Add noise for stochastic prediction
                noise = torch.clamp(
                    torch.randn_like(unlabeled_inputs) * self.noise_scale,
                    -0.2 * self.noise_scale,
                    0.2 * self.noise_scale
                )
                noisy_inputs = unlabeled_inputs + noise
                
                # Forward pass through teacher
                outputs = ema_model(noisy_inputs)
                
                # Ensure we have dict outputs
                if not isinstance(outputs, dict):
                    outputs = {'default': outputs}
                
                # Collect predictions for each task
                for task_name, pred in outputs.items():
                    if task_name not in all_task_preds:
                        all_task_preds[task_name] = []
                    all_task_preds[task_name].append(pred)
        
        # Compute mean predictions and uncertainty for each task
        mean_predictions = {}
        uncertainties = []
        
        for task_name, preds_list in all_task_preds.items():
            # Stack predictions: (T, B, C, D, H, W)
            stacked_preds = torch.stack(preds_list, dim=0)
            
            # Compute mean prediction (raw logits)
            mean_pred_logits = torch.mean(stacked_preds, dim=0)
            mean_predictions[task_name] = mean_pred_logits
            
            # For uncertainty, we need to apply appropriate activation
            # Check if this is a single-channel output (binary classification)
            out_channels = stacked_preds.shape[2]  # Channel dimension is 2 for (T, B, C, D, H, W)
            
            if out_channels == 1:
                # Binary task - use sigmoid
                activated_preds = torch.sigmoid(stacked_preds)
                mean_activated = torch.mean(activated_preds, dim=0)
                # Uncertainty for binary: entropy of mean prediction
                eps = 1e-6
                uncertainty = -1.0 * (mean_activated * torch.log(mean_activated + eps) + 
                                    (1 - mean_activated) * torch.log(1 - mean_activated + eps))
            else:
                # Multi-class task - use softmax
                activated_preds = F.softmax(stacked_preds, dim=2)  # Softmax over channel dimension
                mean_activated = torch.mean(activated_preds, dim=0)
                # Uncertainty for multi-class: entropy of mean prediction
                uncertainty = -1.0 * torch.sum(mean_activated * torch.log(mean_activated + 1e-6), dim=1, keepdim=True)
            
            uncertainties.append(uncertainty)
        
        # Average uncertainty across tasks
        avg_uncertainty = torch.mean(torch.stack(uncertainties, dim=0), dim=0)
        
        return avg_uncertainty, mean_predictions
    
    def _initialize_training(self):
        """Override to add EMA model initialization and TwoStreamBatchSampler."""
        # Get base training components
        training_state = super()._initialize_training()
        
        # Create EMA model
        ema_model = self._create_ema_model(training_state['model'])
        ema_model.train()  # Keep in train mode for batch norm statistics
        
        training_state['ema_model'] = ema_model
        
        # Replace the train dataloader with TwoStreamBatchSampler
        train_dataset = training_state['train_dataset']
        train_indices = training_state['train_indices']
        
        # Get labeled/unlabeled indices based on volume metadata
        print("\nIdentifying labeled/unlabeled samples based on volume metadata...")
        labeled_indices = []
        unlabeled_indices = []
        
        # Get the first target name to access volume info
        first_target = list(train_dataset.target_volumes.keys())[0]
        
        for idx in train_indices:
            # Get the patch info without loading the actual data
            patch_info = train_dataset.valid_patches[idx]
            volume_idx = patch_info['volume_index']
            
            # Check if this volume has a label
            volume_info = train_dataset.target_volumes[first_target][volume_idx]
            has_label = volume_info.get('has_label', True)  # Default to True for backward compatibility
            
            if has_label:
                labeled_indices.append(idx)
            else:
                unlabeled_indices.append(idx)
        
        print(f"Found {len(labeled_indices)} labeled and {len(unlabeled_indices)} unlabeled samples in training set")
        
        # Create TwoStreamBatchSampler if we have both labeled and unlabeled data
        if len(labeled_indices) > 0 and len(unlabeled_indices) > 0:
            batch_size = self.mgr.train_batch_size
            # Use half the batch for labeled, half for unlabeled (same as reference)
            labeled_batch_size = getattr(self.mgr, 'labeled_batch_size', batch_size // 2)
            
            # Ensure we have valid batch sizes
            labeled_batch_size = min(labeled_batch_size, len(labeled_indices))
            unlabeled_batch_size = batch_size - labeled_batch_size
            
            if unlabeled_batch_size > 0 and len(unlabeled_indices) >= unlabeled_batch_size:
                print(f"Using TwoStreamBatchSampler with {labeled_batch_size} labeled + {unlabeled_batch_size} unlabeled per batch")
                
                batch_sampler = TwoStreamBatchSampler(
                    labeled_indices=labeled_indices,
                    unlabeled_indices=unlabeled_indices,
                    batch_size=batch_size,
                    labeled_batch_size=labeled_batch_size
                )
                
                # Create new dataloader with the TwoStreamBatchSampler
                from torch.utils.data import DataLoader
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_sampler=batch_sampler,
                    pin_memory=(True if self.device.type == 'cuda' else False),
                    num_workers=self.mgr.train_num_dataloader_workers
                )
                
                training_state['train_dataloader'] = train_dataloader
                # Store labeled batch size for use in train_step
                training_state['labeled_batch_size'] = labeled_batch_size
            else:
                print(f"WARNING: Not enough unlabeled data for TwoStreamBatchSampler. Using standard dataloader.")
                print(f"  Batch size: {batch_size}, Unlabeled batch size: {unlabeled_batch_size}, Available unlabeled: {len(unlabeled_indices)}")
                training_state['labeled_batch_size'] = None
        else:
            print(f"WARNING: Cannot use mean teacher training without both labeled and unlabeled data!")
            print(f"  Labeled samples: {len(labeled_indices)}, Unlabeled samples: {len(unlabeled_indices)}")
            training_state['labeled_batch_size'] = None
        
        return training_state
    
    def _train_step(self, model, ema_model, data_dict, loss_fns, use_amp, autocast_ctx, 
                    epoch, step, global_step, labeled_batch_size=None, verbose=False):
        """
        Execute a single training step with mean teacher approach.
        
        This includes:
        1. Supervised loss on labeled data
        2. Consistency loss on unlabeled data with uncertainty masking
        """
        if epoch == 0 and step == 0 and verbose:
            print("Items from the first batch -- Double check that your shapes and values are expected:")
            for item, val in data_dict.items():
                if isinstance(val, dict):
                    print(f"{item}: (dictionary with keys: {list(val.keys())})")
                    for sub_key, sub_val in val.items():
                        print(
                            f"  {sub_key}: {sub_val.dtype}, {sub_val.shape}, min {sub_val.min()} max {sub_val.max()}")
                else:
                    print(f"{item}: {val.dtype}, {val.shape}, min {val.min()} max {val.max()}")
        
        inputs = data_dict["image"].to(self.device, dtype=torch.float32)
        targets_dict = {
            k: v.to(self.device, dtype=torch.float32)
            for k, v in data_dict.items()
            if k not in ["image", "patch_info", "is_unlabeled"]
        }
        
        # Check if we have unlabeled data in this batch
        if labeled_batch_size is not None:
            # Using TwoStreamBatchSampler - fixed ordering: first labeled_batch_size are labeled, rest are unlabeled
            num_labeled = labeled_batch_size
            num_unlabeled = inputs.shape[0] - labeled_batch_size
            labeled_mask = torch.zeros(inputs.shape[0], dtype=torch.bool, device=self.device)
            labeled_mask[:labeled_batch_size] = True
            is_unlabeled = ~labeled_mask
            if verbose and step == 0:
                print(f"\n[DEBUG] Using TwoStreamBatchSampler: {num_labeled} labeled, {num_unlabeled} unlabeled samples")
        else:
            # Fall back to is_unlabeled flag from dataset
            is_unlabeled = data_dict.get("is_unlabeled", None)
            if is_unlabeled is not None:
                is_unlabeled = is_unlabeled.to(self.device)
                labeled_mask = ~is_unlabeled
                num_labeled = labeled_mask.sum().item()
                num_unlabeled = is_unlabeled.sum().item()
                if verbose and step == 0:
                    print(f"\n[DEBUG] Batch has mixed data: {num_labeled} labeled, {num_unlabeled} unlabeled samples")
            else:
                # All data is labeled
                labeled_mask = torch.ones(inputs.shape[0], dtype=torch.bool, device=self.device)
                num_labeled = inputs.shape[0]
                num_unlabeled = 0
                if verbose and step == 0:
                    print(f"\n[DEBUG] Batch has only labeled data: {num_labeled} samples")
        
        with autocast_ctx:
            # Forward pass through student
            outputs = model(inputs)
            
            # Compute supervised loss only on labeled data
            total_loss = 0.0
            task_losses = {}
            
            if num_labeled > 0:
                # Extract labeled samples
                labeled_outputs = {k: v[labeled_mask] for k, v in outputs.items()}
                labeled_targets = {k: v[labeled_mask] for k, v in targets_dict.items()}
                
                # Compute supervised loss
                supervised_loss, supervised_task_losses = self._compute_train_loss(
                    labeled_outputs, labeled_targets, loss_fns
                )
                total_loss += supervised_loss
                
                # Store task losses with 'supervised_' prefix
                for k, v in supervised_task_losses.items():
                    task_losses[f'supervised_{k}'] = v
            
            # Compute consistency loss on unlabeled data
            if num_unlabeled > 0:
                # Get consistency weight with ramp-up
                consistency_weight = self.get_current_consistency_weight(
                    global_step,  # Use global_step directly like reference
                    self.consistency_rampup
                )
                
                # Extract unlabeled samples
                unlabeled_inputs = inputs[is_unlabeled]
                
                # Compute uncertainty and teacher predictions for all tasks
                uncertainty, teacher_predictions = self.compute_uncertainty_and_teacher_outputs(
                    ema_model, unlabeled_inputs, T=self.uncertainty_T
                )
                
                # Get student predictions for unlabeled data
                student_unlabeled_outputs = {k: v[is_unlabeled] for k, v in outputs.items()}
                
                # Compute consistency loss with uncertainty masking
                consistency_loss = 0.0
                
                # Compute threshold with ramp-up
                progress = global_step / self.mgr.max_epoch
                threshold = (self.uncertainty_threshold_start + 
                           (self.uncertainty_threshold_end - self.uncertainty_threshold_start) * 
                           sigmoid_rampup(global_step, self.mgr.max_epoch)) * np.log(2)
                
                # Create uncertainty mask
                mask = (uncertainty < threshold).float()
                
                # Compute consistency loss for each task
                for task_name in student_unlabeled_outputs.keys():
                    student_pred = student_unlabeled_outputs[task_name]
                    
                    # Get teacher prediction for this task (already computed)
                    teacher_pred = teacher_predictions[task_name]
                    
                    # Determine if we should use sigmoid based on output channels
                    use_sigmoid = (student_pred.shape[1] == 1)  # Single channel = binary task
                    
                    # Compute MSE between activated outputs
                    consistency_dist = softmax_mse_loss(student_pred, teacher_pred, use_sigmoid=use_sigmoid)
                    
                    # Apply uncertainty mask and average
                    masked_consistency = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)
                    
                    consistency_loss += masked_consistency
                    task_losses[f'consistency_{task_name}'] = masked_consistency.detach().cpu().item()
                
                # Add weighted consistency loss to total
                weighted_consistency_loss = consistency_weight * consistency_loss
                total_loss += weighted_consistency_loss
                
                # Log consistency metrics
                task_losses['consistency_weight'] = consistency_weight
                task_losses['uncertainty_threshold'] = threshold
                task_losses['uncertainty_mask_ratio'] = mask.mean().item()
                task_losses['consistency_loss_total'] = consistency_loss.detach().cpu().item()
                task_losses['weighted_consistency_loss'] = weighted_consistency_loss.detach().cpu().item()
                
            # Log batch composition
            task_losses['num_labeled'] = num_labeled
            task_losses['num_unlabeled'] = num_unlabeled
            task_losses['rampup_progress'] = min(1.0, global_step / self.consistency_rampup)
        
        return total_loss, task_losses, inputs, targets_dict, outputs
    
    def train(self):
        # Initialize all training components including EMA model
        training_state = self._initialize_training()
        
        # Unpack the state
        model = training_state['model']
        ema_model = training_state['ema_model']
        optimizer = training_state['optimizer']
        scheduler = training_state['scheduler']
        is_per_iteration_scheduler = training_state['is_per_iteration_scheduler']
        loss_fns = training_state['loss_fns']
        scaler = training_state['scaler']
        train_dataset = training_state['train_dataset']
        val_dataset = training_state['val_dataset']
        train_dataloader = training_state['train_dataloader']
        val_dataloader = training_state['val_dataloader']
        train_indices = training_state['train_indices']
        val_indices = training_state['val_indices']
        use_amp = training_state['use_amp']
        start_epoch = training_state['start_epoch']
        ckpt_dir = training_state['ckpt_dir']
        model_ckpt_dir = training_state['model_ckpt_dir']
        labeled_batch_size = training_state.get('labeled_batch_size', None)
        
        # Initialize wandb if configured
        self._initialize_wandb(train_dataset, val_dataset, train_indices, val_indices)
        
        # Track validation loss history and checkpoints
        val_loss_history = {}
        checkpoint_history = deque(maxlen=3)
        best_checkpoints = []
        debug_gif_history = deque(maxlen=3)
        best_debug_gifs = []
        
        global_step = 0
        grad_accumulate_n = self.mgr.gradient_accumulation
        
        print(f"\nUncertainty-Aware Mean Teacher Training Configuration:")
        print(f"  EMA decay: {self.ema_decay}")
        print(f"  Consistency weight: {self.consistency_weight}")
        print(f"  Consistency ramp-up: {self.consistency_rampup} iterations")
        print(f"  Uncertainty T (forward passes): {self.uncertainty_T}")
        print(f"  Uncertainty threshold: {self.uncertainty_threshold_start} -> {self.uncertainty_threshold_end}")
        print(f"  Noise scale: {self.noise_scale}")
        
        # Training loop
        for epoch in range(start_epoch, self.mgr.max_epoch):
            model.train()
            ema_model.train()
            
            if getattr(self.mgr, 'max_steps_per_epoch', None) is not None and self.mgr.max_steps_per_epoch > 0:
                num_iters = min(len(train_dataloader), self.mgr.max_steps_per_epoch)
            else:
                num_iters = len(train_dataloader)
            
            epoch_losses = {t_name: [] for t_name in self.mgr.targets}
            # Add tracking for mean teacher specific losses
            epoch_losses.update({
                f'supervised_{t_name}': [] for t_name in self.mgr.targets
            })
            epoch_losses.update({
                f'consistency_{t_name}': [] for t_name in self.mgr.targets
            })
            epoch_losses['consistency_weight'] = []
            epoch_losses['uncertainty_mask_ratio'] = []
            
            train_iter = iter(train_dataloader)
            pbar = tqdm(range(num_iters), desc=f'Epoch {epoch + 1}/{self.mgr.max_epoch}')
            
            for i in pbar:
                if i % grad_accumulate_n == 0:
                    optimizer.zero_grad(set_to_none=True)
                
                data_dict = next(train_iter)
                global_step += 1
                
                # Setup autocast context
                if use_amp and self.device.type in ['cuda', 'cpu']:
                    autocast_ctx = torch.amp.autocast(self.device.type)
                else:
                    autocast_ctx = nullcontext()
                
                # Execute training step with mean teacher
                total_loss, task_losses, inputs, targets_dict, outputs = self._train_step(
                    model=model,
                    ema_model=ema_model,
                    data_dict=data_dict,
                    loss_fns=loss_fns,
                    use_amp=use_amp,
                    autocast_ctx=autocast_ctx,
                    epoch=epoch,
                    step=i,
                    global_step=global_step,
                    labeled_batch_size=labeled_batch_size,
                    verbose=self.mgr.verbose
                )
                
                # Accumulate losses for tracking
                for loss_name, loss_value in task_losses.items():
                    if loss_name in epoch_losses:
                        epoch_losses[loss_name].append(loss_value)
                
                # Update gradients
                optimizer_stepped = self._update_gradients(
                    scaler=scaler,
                    total_loss=total_loss,
                    optimizer=optimizer,
                    model=model,
                    step=i,
                    num_iters=num_iters,
                    grad_accumulate_n=grad_accumulate_n
                )
                
                # Update EMA model after optimizer step
                if optimizer_stepped:
                    self.update_ema_variables(model, ema_model, self.ema_decay, global_step)
                    if is_per_iteration_scheduler:
                        scheduler.step()
                
                # Update progress bar with mean teacher specific metrics
                loss_str_parts = []
                for t in self.mgr.targets:
                    if f'supervised_{t}' in epoch_losses and len(epoch_losses[f'supervised_{t}']) > 0:
                        loss_str_parts.append(f"sup_{t}: {np.mean(epoch_losses[f'supervised_{t}'][-100:]):.4f}")
                    if f'consistency_{t}' in epoch_losses and len(epoch_losses[f'consistency_{t}']) > 0:
                        loss_str_parts.append(f"cons_{t}: {np.mean(epoch_losses[f'consistency_{t}'][-100:]):.4f}")
                
                if 'uncertainty_mask_ratio' in epoch_losses and len(epoch_losses['uncertainty_mask_ratio']) > 0:
                    loss_str_parts.append(f"mask: {np.mean(epoch_losses['uncertainty_mask_ratio'][-100:]):.2f}")
                
                pbar.set_postfix_str(" | ".join(loss_str_parts))
                
                # Get current learning rate for logging
                current_lr = optimizer.param_groups[0]['lr']
                
                # Log metrics to wandb
                if self.mgr.wandb_project:
                    metrics = self._prepare_metrics_for_logging(
                        epoch=epoch,
                        step=global_step,
                        epoch_losses=epoch_losses,
                        current_lr=current_lr
                    )
                    # Add mean teacher specific metrics
                    mean_teacher_params = [
                        'consistency_weight', 'uncertainty_threshold', 'uncertainty_mask_ratio',
                        'consistency_loss_total', 'weighted_consistency_loss',
                        'num_labeled', 'num_unlabeled', 'rampup_progress'
                    ]
                    for param in mean_teacher_params:
                        if param in task_losses:
                            metrics[param] = task_losses[param]
                    
                    # Add constant parameters (logged for reference)
                    metrics['ema_decay'] = self.ema_decay
                    metrics['noise_scale'] = self.noise_scale
                    metrics['uncertainty_T'] = self.uncertainty_T
                    
                    import wandb
                    wandb.log(metrics)
                
                del data_dict, inputs, targets_dict, outputs
            
            # Step per-epoch schedulers
            if not is_per_iteration_scheduler:
                scheduler.step()
            
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            print(f"\n[Train] Epoch {epoch + 1} completed.")
            # Print supervised and consistency losses separately
            for t_name in self.mgr.targets:
                if f'supervised_{t_name}' in epoch_losses:
                    sup_loss = np.mean(epoch_losses[f'supervised_{t_name}']) if epoch_losses[f'supervised_{t_name}'] else 0
                    cons_loss = np.mean(epoch_losses[f'consistency_{t_name}']) if epoch_losses[f'consistency_{t_name}'] else 0
                    print(f"  {t_name}: Supervised Loss = {sup_loss:.4f}, Consistency Loss = {cons_loss:.4f}")
            
            # Validation (using student model)
            if epoch % 1 == 0:
                model.eval()
                with torch.no_grad():
                    val_losses = {t_name: [] for t_name in self.mgr.targets}
                    frames_array = None
                    
                    val_dataloader_iter = iter(val_dataloader)
                    
                    if hasattr(self.mgr, 'max_val_steps_per_epoch') and self.mgr.max_val_steps_per_epoch is not None and self.mgr.max_val_steps_per_epoch > 0:
                        num_val_iters = min(len(val_indices), self.mgr.max_val_steps_per_epoch)
                    else:
                        num_val_iters = len(val_indices)
                    
                    val_pbar = tqdm(range(num_val_iters), desc=f'Validation {epoch + 1}')
                    
                    for i in val_pbar:
                        try:
                            data_dict = next(val_dataloader_iter)
                        except StopIteration:
                            val_dataloader_iter = iter(val_dataloader)
                            data_dict = next(val_dataloader_iter)
                        
                        # Execute validation step
                        task_losses, inputs, targets_dict, outputs = self._validation_step(
                            model=model,
                            data_dict=data_dict,
                            loss_fns=loss_fns,
                            use_amp=use_amp
                        )
                        
                        # Accumulate validation losses
                        for t_name, loss_value in task_losses.items():
                            val_losses[t_name].append(loss_value)
                        
                        # Create debug visualization on first batch
                        if i == 0:
                            from vesuvius.utils.plotting import save_debug
                            
                            b_idx = 0
                            found_non_zero = False
                            
                            first_target = next(iter(targets_dict.values()))
                            if torch.any(first_target[0] != 0):
                                found_non_zero = True
                            else:
                                for b in range(first_target.shape[0]):
                                    if torch.any(first_target[b] != 0):
                                        b_idx = b
                                        found_non_zero = True
                                        break
                            
                            if found_non_zero:
                                inputs_first = inputs[b_idx: b_idx + 1]
                                targets_dict_first = {t_name: t_tensor[b_idx: b_idx + 1] 
                                                    for t_name, t_tensor in targets_dict.items()}
                                outputs_dict_first = {t_name: p_tensor[b_idx: b_idx + 1] 
                                                    for t_name, p_tensor in outputs.items()}
                                
                                debug_img_path = f"{ckpt_dir}/{self.mgr.model_name}_debug_epoch{epoch}.gif"
                                frames_array = save_debug(
                                    input_volume=inputs_first,
                                    targets_dict=targets_dict_first,
                                    outputs_dict=outputs_dict_first,
                                    tasks_dict=self.mgr.targets,
                                    epoch=epoch,
                                    save_path=debug_img_path
                                )
                                debug_gif_history.append((epoch, debug_img_path))
                        
                        loss_str = " | ".join([f"{t}: {np.mean(val_losses[t]):.4f}"
                                             for t in self.mgr.targets if len(val_losses[t]) > 0])
                        val_pbar.set_postfix_str(loss_str)
                        
                        # Log validation metrics to wandb
                        if self.mgr.wandb_project:
                            global_step += 1
                            val_metrics = {"epoch": epoch, "step": global_step}
                            for t_name, loss_value in task_losses.items():
                                val_metrics[f"val_loss_{t_name}"] = loss_value
                            
                            total_step_loss = sum(task_losses.values())
                            val_metrics["val_loss_total"] = total_step_loss / len(task_losses) if task_losses else 0
                            
                            if i == 0 and 'frames_array' in locals() and frames_array is not None:
                                import wandb
                                val_metrics["debug_gif"] = wandb.Video(frames_array)
                            
                            import wandb
                            wandb.log(val_metrics)
                        
                        del outputs, inputs, targets_dict
                    
                    # Calculate average validation losses
                    print(f"\n[Validation] Epoch {epoch + 1} summary:")
                    total_val_loss = 0.0
                    for t_name in self.mgr.targets:
                        val_avg = np.mean(val_losses[t_name]) if val_losses[t_name] else 0
                        print(f"  Task '{t_name}': Avg validation loss = {val_avg:.4f}")
                        total_val_loss += val_avg
                    
                    avg_val_loss = total_val_loss / len(self.mgr.targets) if self.mgr.targets else 0
                    val_loss_history[epoch] = avg_val_loss
                    
                    # Handle epoch end operations
                    checkpoint_history, best_checkpoints, ckpt_path = self._on_epoch_end(
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        train_dataset=train_dataset,
                        ckpt_dir=ckpt_dir,
                        model_ckpt_dir=model_ckpt_dir,
                        checkpoint_history=checkpoint_history,
                        best_checkpoints=best_checkpoints,
                        avg_val_loss=avg_val_loss
                    )
                    
                    # Also save EMA model state in checkpoint
                    ema_ckpt_path = ckpt_path.replace('.pth', '_ema.pth')
                    torch.save(ema_model.state_dict(), ema_ckpt_path)
                    
                    # Manage debug GIFs
                    if epoch in [e for e, _ in debug_gif_history]:
                        from vesuvius.models.training.save_checkpoint import manage_debug_gifs
                        debug_gif_history, best_debug_gifs = manage_debug_gifs(
                            debug_gif_history=debug_gif_history,
                            best_debug_gifs=best_debug_gifs,
                            epoch=epoch,
                            gif_path=next(p for e, p in debug_gif_history if e == epoch),
                            validation_loss=avg_val_loss,
                            checkpoint_dir=ckpt_dir,
                            model_name=self.mgr.model_name,
                            max_recent=3,
                            max_best=2
                        )
        
        print('Training Finished!')
        
        # Save final checkpoint
        from vesuvius.models.training.save_checkpoint import save_final_checkpoint
        final_model_path = save_final_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            max_epoch=self.mgr.max_epoch,
            model_ckpt_dir=model_ckpt_dir,
            model_name=self.mgr.model_name,
            model_config=model.final_config,
            train_dataset=train_dataset
        )
        
        # Also save final EMA model
        final_ema_path = final_model_path.replace('.pth', '_ema.pth')
        torch.save(ema_model.state_dict(), final_ema_path)
        print(f"Saved final EMA model to: {final_ema_path}")