"""
Mean Teacher Trainer for semi-supervised learning with uncertainty-aware consistency loss.
Based on: https://github.com/HiLab-git/SSL4MIS/blob/master/code/train_uncertainty_aware_mean_teacher_3D.py
"""

# Mean Teacher Training Constants
LABELED_RATIO = 0.1  # 10% of data is labeled
TEST_RATIO = 0.1  # 10% of data for test set
UNLABELED_BATCH_RATIO = 1  # 1:1 ratio of unlabeled to labeled samples per batch

# EMA and Consistency Parameters
EMA_DECAY = 0.999  # Exponential moving average decay for teacher model
CONSISTENCY_WEIGHT = 1.0  # Weight for consistency loss
CONSISTENCY_RAMPUP = 40  # Number of epochs for consistency weight rampup

# Uncertainty Parameters
UNCERTAINTY_THRESHOLD_BASE = 0.75  # Base threshold for uncertainty filtering
UNCERTAINTY_THRESHOLD_RAMPUP = 0.25  # Additional rampup for uncertainty threshold
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext
import copy
import itertools
from itertools import cycle
import gc
from torch.utils.data import DataLoader, SubsetRandomSampler, Sampler
from vesuvius.models.datasets.samplers import TwoStreamBatchSampler
from .train import BaseTrainer
from vesuvius.models.build.build_network_from_config import NetworkFromConfig




class MeanTeacherTrainer(BaseTrainer):
    """
    Mean Teacher trainer for semi-supervised learning.
    Implements uncertainty-aware consistency regularization.
    """
    
    def __init__(self, mgr=None, verbose=True):
        super().__init__(mgr, verbose)
        
        # Mean teacher specific parameters - use constants defined at top of file
        self.ema_decay = getattr(self.mgr, 'ema_decay', EMA_DECAY)
        self.consistency_weight = getattr(self.mgr, 'consistency_weight', CONSISTENCY_WEIGHT)
        self.consistency_rampup = getattr(self.mgr, 'consistency_rampup', CONSISTENCY_RAMPUP)
        self.labeled_ratio = getattr(self.mgr, 'labeled_ratio', LABELED_RATIO)
        self.test_ratio = getattr(self.mgr, 'test_ratio', TEST_RATIO)
        self.unlabeled_batch_ratio = getattr(self.mgr, 'unlabeled_batch_ratio', UNLABELED_BATCH_RATIO)
        
        # Uncertainty parameters - use constants defined at top of file
        self.uncertainty_threshold_base = getattr(self.mgr, 'uncertainty_threshold_base', UNCERTAINTY_THRESHOLD_BASE)
        self.uncertainty_threshold_rampup = getattr(self.mgr, 'uncertainty_threshold_rampup', UNCERTAINTY_THRESHOLD_RAMPUP)
        
        self.teacher_model = None
        self.student_model = None
        
    def _build_model(self):
        """Build both student and teacher models"""
        # Build student model (trainable)
        self.student_model = super()._build_model()
        
        # Build teacher model (EMA of student)
        self.teacher_model = copy.deepcopy(self.student_model)

        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        print("Built student and teacher models for Mean Teacher training")
        return self.student_model
    
    def _update_ema_variables(self, alpha, global_step):
        """Update teacher model using exponential moving average"""
        # Use the global_step to adjust alpha if needed
        alpha = min(1 - 1 / (global_step + 1), alpha)
        
        for ema_param, param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
            
    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))
    
    def get_current_consistency_weight(self, epoch):
        """Get consistency weight with rampup schedule"""
        return self.consistency_weight * self.sigmoid_rampup(epoch, self.consistency_rampup)
    
    def compute_uncertainty(self, predictions):
        """
        Compute uncertainty using entropy of predictions.
        
        Parameters
        ----------
        predictions : torch.Tensor
            Softmax predictions from teacher model [B, C, ...]
            
        Returns
        -------
        uncertainty : torch.Tensor
            Uncertainty map [B, 1, ...]
        """
        # Add small epsilon for numerical stability
        epsilon = 1e-6
        uncertainty = -1.0 * torch.sum(predictions * torch.log(predictions + epsilon), dim=1, keepdim=True)
        return uncertainty
    
    def get_uncertainty_threshold(self, iter_num, max_iterations):
        """Get dynamic uncertainty threshold with rampup"""
        rampup = self.uncertainty_threshold_rampup * self.sigmoid_rampup(iter_num, max_iterations)
        threshold = (self.uncertainty_threshold_base + rampup) * np.log(2)  # np.log(2) is max entropy for binary
        return threshold
    
    def _configure_dataloaders(self, train_dataset, val_dataset=None):
        """Configure dataloaders with labeled/unlabeled split"""
        if val_dataset is None:
            val_dataset = train_dataset
            
        dataset_size = len(train_dataset)
        
        # Automatically detect which samples have labels
        print("\nDetecting labeled vs unlabeled samples...")
        labeled_indices = []
        unlabeled_indices = []
        
        # Access the dataset's patch information directly
        for idx, patch_info in enumerate(tqdm(train_dataset.valid_patches, desc="Checking patches for labels")):
            vol_idx = patch_info["volume_index"]
            
            # Extract patch coordinates
            if train_dataset.is_2d_dataset:
                y, x = patch_info["y"], patch_info["x"]
                dy, dx = patch_info["dy"], patch_info["dx"]
            else:
                z, y, x = patch_info["z"], patch_info["y"], patch_info["x"]
                dz, dy, dx = patch_info["dz"], patch_info["dy"], patch_info["dx"]
            
            has_label = False
            
            # Check each target's label array for this specific patch region
            for target_name in self.mgr.targets:
                if target_name in train_dataset.target_volumes:
                    volume_info = train_dataset.target_volumes[target_name][vol_idx]
                    label_array = volume_info['data']['label']
                    
                    # Extract the patch region from label array
                    if train_dataset.is_2d_dataset:
                        label_patch = label_array[y:y+dy, x:x+dx]
                    else:
                        label_patch = label_array[z:z+dz, y:y+dy, x:x+dx]
                    
                    # Check if this patch region has any non-zero values
                    if np.any(label_patch != 0):
                        has_label = True
                        break
            
            if has_label:
                labeled_indices.append(idx)
            else:
                unlabeled_indices.append(idx)


        
        # Shuffle indices
        np.random.shuffle(labeled_indices)
        np.random.shuffle(unlabeled_indices)
        
        # Split off test set from labeled data
        test_split = int(np.floor(self.test_ratio * len(labeled_indices)))
        test_indices = labeled_indices[:test_split]
        labeled_indices = labeled_indices[test_split:]
        
        print(f"\nDataset split:")
        print(f"  Total samples: {dataset_size}")
        print(f"  Test samples: {len(test_indices)} ({len(test_indices)/dataset_size*100:.1f}%)")
        print(f"  Labeled samples: {len(labeled_indices)} ({len(labeled_indices)/dataset_size*100:.1f}%)")
        print(f"  Unlabeled samples: {len(unlabeled_indices)} ({len(unlabeled_indices)/dataset_size*100:.1f}%)")
        
        batch_size = self.mgr.train_batch_size
        
        # Calculate labeled and unlabeled samples per batch
        # If we have no unlabeled data, use all labeled
        if len(unlabeled_indices) == 0:
            labeled_bs = batch_size
            unlabeled_bs = 0
        else:
            labeled_bs = max(1, batch_size // (1 + self.unlabeled_batch_ratio))
            unlabeled_bs = batch_size - labeled_bs
        
        print(f"\nBatch composition:")
        print(f"  Labeled samples per batch: {labeled_bs}")
        print(f"  Unlabeled samples per batch: {unlabeled_bs}")
        
        # Create combined sampler for training
        train_sampler = TwoStreamBatchSampler(
            labeled_indices, 
            unlabeled_indices,
            batch_size=batch_size,
            secondary_batch_size=unlabeled_bs
        )
        
        # Store labeled batch size for use in training
        self.labeled_bs = labeled_bs
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            pin_memory=(True if self.device == 'cuda' else False),
            num_workers=self.mgr.train_num_dataloader_workers
        )
        
        # Validation uses test indices
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            sampler=SubsetRandomSampler(test_indices),
            pin_memory=(True if self.device == 'cuda' else False),
            num_workers=self.mgr.train_num_dataloader_workers
        )
        
        return train_dataloader, val_dataloader, labeled_indices + unlabeled_indices, test_indices
    
    def train(self):
        """Main training loop for Mean Teacher"""
        # Setup S3 if needed
        from vesuvius.models.utilities.s3_utils import detect_s3_paths, setup_multiprocessing_for_s3
        if detect_s3_paths(self.mgr):
            print("\nDetected S3 paths in configuration")
            setup_multiprocessing_for_s3()
            
        # Configure datasets
        train_dataset = self._configure_dataset(is_training=True)
        val_dataset = self._configure_dataset(is_training=False)
        
        self.mgr.auto_detect_channels(train_dataset)
        
        # Build models (student and teacher)
        model = self._build_model()  # This creates both student and teacher
        
        # Setup training components
        optimizer = self._get_optimizer(model)
        loss_fns = self._build_loss()
        scheduler, is_per_iteration_scheduler = self._get_scheduler(optimizer)
        
        # Initialize weights
        from vesuvius.utils.utils import init_weights_he
        model.apply(lambda module: init_weights_he(module, neg_slope=0.2))
        self.teacher_model.apply(lambda module: init_weights_he(module, neg_slope=0.2))
        
        # Move models to device
        model = model.to(self.device)
        self.teacher_model = self.teacher_model.to(self.device)
        
        # Compile models if on CUDA
        if self.device.type == 'cuda':
            model = torch.compile(model)
            self.teacher_model = torch.compile(self.teacher_model)
            
        # Setup mixed precision
        use_amp = not getattr(self.mgr, 'no_amp', False)
        if not use_amp:
            print("Automatic Mixed Precision (AMP) is disabled")
        scaler = self._get_scaler(self.device.type, use_amp=use_amp)
        
        # Configure dataloaders with labeled/unlabeled split
        train_dataloader, val_dataloader, train_indices, val_indices = self._configure_dataloaders(
            train_dataset, val_dataset
        )
        
        # Setup Weights & Biases if configured
        if self.mgr.wandb_project:
            import wandb
            wandb.init(
                entity=self.mgr.wandb_entity,
                project=self.mgr.wandb_project,
                group=self.mgr.model_name,
                config=self.mgr.convert_to_dict()
            )
            
        # Training state
        start_epoch = 0
        val_loss_history = {}
        checkpoint_history = []
        best_checkpoints = []
        debug_gif_history = []
        best_debug_gifs = []
        
        # Load checkpoint if resuming
        checkpoint_loaded = False
        if hasattr(self.mgr, 'resume_checkpoint') and self.mgr.resume_checkpoint:
            from vesuvius.models.utilities.load_checkpoint import load_checkpoint
            loaded_data = load_checkpoint(
                checkpoint_path=self.mgr.resume_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self.device
            )
            if loaded_data:
                start_epoch = loaded_data.get('epoch', 0) + 1
                checkpoint_loaded = True
                print(f"\nResumed from checkpoint at epoch {start_epoch}")
                
                # Also load teacher model if available
                if 'teacher_state_dict' in loaded_data:
                    self.teacher_model.load_state_dict(loaded_data['teacher_state_dict'])
                    print("Loaded teacher model from checkpoint")
                    
        # Create checkpoint directory
        import os
        from pathlib import Path
        from datetime import datetime
        
        output_dir = Path(self.mgr.ckpt_out_base)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        
        if hasattr(self.mgr, 'experiment_name') and self.mgr.experiment_name:
            ckpt_dir = output_dir / self.mgr.experiment_name / f"{self.mgr.model_name}_{timestamp}"
        else:
            ckpt_dir = output_dir / f"{self.mgr.model_name}_{timestamp}"
            
        ckpt_dir = str(ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Save configuration
        from vesuvius.models.training.save_checkpoint import cleanup_old_configs
        import yaml
        config_path = os.path.join(ckpt_dir, "config.yaml")
        config_dict = self.mgr.convert_to_dict()
        with open(config_path, 'w') as f:
            yaml.safe_dump(config_dict, f, sort_keys=False)
        cleanup_old_configs(ckpt_dir, self.mgr.model_name, keep_latest=1)
        
        print(f"\nCheckpoints will be saved to: {ckpt_dir}")
        print(f"Starting Mean Teacher training from epoch {start_epoch}")
        
        # Initialize checkpoint tracking
        from collections import deque
        val_loss_history = {}  # {epoch: validation_loss}
        checkpoint_history = deque(maxlen=3)
        best_checkpoints = []
        debug_gif_history = deque(maxlen=3)
        best_debug_gifs = []  # List of (val_loss, epoch, gif_path)
        
        # Training loop
        global_step = start_epoch * len(train_dataloader)
        
        for epoch in range(start_epoch, self.mgr.max_epoch):
            model.train()
            self.teacher_model.train()
            
            epoch_losses = {t_name: [] for t_name in self.mgr.targets}
            epoch_losses['consistency'] = []
            
            num_iters = len(train_dataloader)
            grad_accumulate_n = getattr(self.mgr, 'grad_accumulate_n', 1)
            
            pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{self.mgr.max_epoch}')
            
            for i, data_dict in enumerate(pbar):
                global_step += 1
                
                # Get inputs and targets
                inputs = data_dict["image"].to(self.device, dtype=torch.float32)
                targets_dict = {
                    k: v.to(self.device, dtype=torch.float32)
                    for k, v in data_dict.items()
                    if k not in ["image", "ignore_masks", "patch_info"]
                }
                
                # Split batch into labeled and unlabeled
                labeled_inputs = inputs[:self.labeled_bs]
                unlabeled_inputs = inputs[self.labeled_bs:]
                
                # Add noise to unlabeled inputs for teacher
                noise = torch.clamp(torch.randn_like(unlabeled_inputs) * 0.1, -0.2, 0.2)
                ema_inputs = unlabeled_inputs + noise
                
                optimizer.zero_grad()

                if use_amp:
                    context = torch.amp.autocast('cuda') if self.device.type == 'cuda' else nullcontext()
                else:
                    context = nullcontext()
                    
                with context:
                    # Forward pass through student model
                    outputs = model(inputs)
                    
                    # Forward pass through teacher model (no gradients)
                    with torch.no_grad():
                        ema_outputs = self.teacher_model(ema_inputs)
                        
                    # Compute supervised loss on labeled data
                    total_loss = 0.0
                    
                    for t_name, t_gt in targets_dict.items():
                        # Get task configuration
                        task_weight = self.mgr.targets[t_name].get("weight", 1.0)
                        task_losses = loss_fns[t_name]
                        
                        # Only use labeled portion for supervised loss
                        t_pred_labeled = outputs[t_name][:self.labeled_bs]
                        t_gt_labeled = t_gt[:self.labeled_bs]
                        
                        task_total_loss = 0.0
                        for loss_fn, loss_weight in task_losses:
                            loss_value = loss_fn(t_pred_labeled, t_gt_labeled)
                            task_total_loss += loss_weight * loss_value
                            
                        weighted_loss = task_weight * task_total_loss
                        total_loss += weighted_loss
                        epoch_losses[t_name].append(task_total_loss.detach().cpu().item())
                        
                    # Compute consistency loss on unlabeled data
                    consistency_weight = self.get_current_consistency_weight(global_step // 150)
                    
                    if consistency_weight > 0 and unlabeled_inputs.shape[0] > 0:
                        # Get predictions for consistency
                        for t_name in self.mgr.targets:
                            student_pred_unlabeled = outputs[t_name][self.labeled_bs:]
                            teacher_pred_unlabeled = ema_outputs[t_name]
                            
                            # Apply activation to get probabilities for uncertainty computation
                            activation = self.mgr.targets[t_name].get("activation", "sigmoid")
                            if activation == "sigmoid":
                                teacher_prob = torch.sigmoid(teacher_pred_unlabeled)
                            elif activation == "softmax":
                                teacher_prob = torch.softmax(teacher_pred_unlabeled, dim=1)
                            else:
                                teacher_prob = teacher_pred_unlabeled
                                
                            # Compute uncertainty from teacher predictions
                            uncertainty = self.compute_uncertainty(teacher_prob)
                            
                            # Get uncertainty threshold
                            threshold = self.get_uncertainty_threshold(global_step, self.mgr.max_epoch * num_iters)
                            
                            # Create mask for low-uncertainty regions
                            mask = (uncertainty < threshold).float()
                            
                            # Compute consistency loss using softmax_mse_loss approach
                            # Apply softmax/sigmoid to logits and compute MSE
                            if activation == "sigmoid":
                                student_softmax = torch.sigmoid(student_pred_unlabeled)
                                teacher_softmax = torch.sigmoid(teacher_pred_unlabeled)
                            elif activation == "softmax":
                                student_softmax = torch.softmax(student_pred_unlabeled, dim=1)
                                teacher_softmax = torch.softmax(teacher_pred_unlabeled, dim=1)
                            else:
                                student_softmax = student_pred_unlabeled
                                teacher_softmax = teacher_pred_unlabeled
                            
                            # Compute MSE loss (matching softmax_mse_loss implementation)
                            consistency_dist = (student_softmax - teacher_softmax.detach()) ** 2
                            
                            # Apply uncertainty mask
                            if consistency_dist.dim() > mask.dim():
                                # Average over channel dimension if needed
                                consistency_dist = consistency_dist.mean(dim=1, keepdim=True)
                                
                            masked_consistency = mask * consistency_dist
                            
                            # Average over spatial dimensions
                            consistency_loss = masked_consistency.sum() / (2 * mask.sum() + 1e-16)
                            
                            # Add to total loss
                            total_loss += consistency_weight * consistency_loss
                            epoch_losses['consistency'].append(consistency_loss.detach().cpu().item())
                            
                    # Scale loss for gradient accumulation
                    total_loss = total_loss / grad_accumulate_n
                    
                # Backward pass
                scaler.scale(total_loss).backward()
                
                # Update weights if accumulation complete
                if (i + 1) % grad_accumulate_n == 0 or (i + 1) == num_iters:
                    scaler.unscale_(optimizer)
                    grad_clip = getattr(self.mgr, 'gradient_clip', 12.0)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Update teacher model via EMA
                    self._update_ema_variables(self.ema_decay, global_step)
                    
                    if is_per_iteration_scheduler:
                        scheduler.step()
                        
                # Update progress bar
                loss_str = " | ".join([f"{t}: {np.mean(epoch_losses[t][-100:]):.4f}"
                                     for t in list(self.mgr.targets.keys()) + ['consistency']
                                     if len(epoch_losses[t]) > 0])
                pbar.set_postfix_str(loss_str)
                
                # Log to wandb
                if self.mgr.wandb_project:
                    wandb.log({
                        **{
                            f"loss_{t_name}": epoch_losses[t_name][-1]
                            for t_name in self.mgr.targets
                            if len(epoch_losses[t_name]) > 0
                        },
                        "loss_consistency": epoch_losses['consistency'][-1] if len(epoch_losses['consistency']) > 0 else 0,
                        "consistency_weight": consistency_weight,
                        "loss_total": total_loss.detach().cpu().item()
                    })
                    
                del data_dict, inputs, targets_dict, outputs, ema_outputs
                
            # Step per-epoch schedulers
            if not is_per_iteration_scheduler:
                scheduler.step()
                
            # Garbage collection
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
            # Print epoch summary
            print(f"\n[Train] Epoch {epoch + 1} completed.")
            for t_name in self.mgr.targets:
                avg_loss = np.mean(epoch_losses[t_name]) if epoch_losses[t_name] else 0
                print(f"  {t_name}: Avg Loss = {avg_loss:.4f}")
            avg_consistency = np.mean(epoch_losses['consistency']) if epoch_losses['consistency'] else 0
            print(f"  consistency: Avg Loss = {avg_consistency:.4f}")
            
            # Save checkpoint
            from vesuvius.models.training.save_checkpoint import save_checkpoint, manage_checkpoint_history
            
            ckpt_path = os.path.join(
                ckpt_dir,
                f"{self.mgr.model_name}_epoch{epoch}.pth"
            )
            
            # Save both student and teacher models
            checkpoint_data = save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                checkpoint_path=ckpt_path,
                model_config=model.final_config if hasattr(model, 'final_config') else None,
                train_dataset=train_dataset,
                additional_data={
                    'teacher_state_dict': self.teacher_model.state_dict(),
                    'ema_decay': self.ema_decay,
                    'global_step': global_step
                }
            )
            
            checkpoint_history.append((epoch, ckpt_path))
            
            del checkpoint_data
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
            # Validation
            if epoch % 1 == 0:
                model.eval()
                self.teacher_model.eval()
                
                with torch.no_grad():
                    val_losses = {t_name: [] for t_name in self.mgr.targets}
                    
                    val_dataloader_iter = iter(val_dataloader)
                    
                    if hasattr(self.mgr, 'max_val_steps_per_epoch') and self.mgr.max_val_steps_per_epoch and self.mgr.max_val_steps_per_epoch > 0:
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
                            
                        inputs = data_dict["image"].to(self.device, dtype=torch.float32)
                        targets_dict = {
                            k: v.to(self.device, dtype=torch.float32)
                            for k, v in data_dict.items()
                            if k not in ["image", "ignore_masks", "patch_info"]
                        }
                        
                        if use_amp:
                            context = torch.amp.autocast('cuda') if self.device.type == 'cuda' else nullcontext()
                        else:
                            context = nullcontext()
                            
                        with context:
                            # Use student model for validation
                            outputs = model(inputs)
                            
                            for t_name, t_gt in targets_dict.items():
                                t_pred = outputs[t_name]
                                task_losses = loss_fns[t_name]
                                
                                task_total_loss = 0.0
                                for loss_fn, loss_weight in task_losses:
                                    loss_value = loss_fn(t_pred, t_gt)
                                    task_total_loss += loss_weight * loss_value
                                    
                                val_losses[t_name].append(task_total_loss.detach().cpu().item())
                                
                            # Save debug image for first batch
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
                                    targets_dict_first = {
                                        t_name: t_tensor[b_idx: b_idx + 1]
                                        for t_name, t_tensor in targets_dict.items()
                                    }
                                    outputs_dict_first = {
                                        t_name: p_tensor[b_idx: b_idx + 1]
                                        for t_name, p_tensor in outputs.items()
                                    }
                                    
                                    debug_img_path = f"{ckpt_dir}/{self.mgr.model_name}_debug_epoch{epoch}.gif"
                                    save_debug(
                                        input_volume=inputs_first,
                                        targets_dict=targets_dict_first,
                                        outputs_dict=outputs_dict_first,
                                        tasks_dict=self.mgr.targets,
                                        epoch=epoch,
                                        save_path=debug_img_path
                                    )
                                    debug_gif_history.append((epoch, debug_img_path))
                                    
                        del data_dict, inputs, targets_dict, outputs
                        
                # Print validation summary
                print(f"\n[Validation] Epoch {epoch + 1} completed.")
                avg_val_loss = 0
                for t_name in self.mgr.targets:
                    avg_loss = np.mean(val_losses[t_name]) if val_losses[t_name] else 0
                    print(f"  {t_name}: Avg Loss = {avg_loss:.4f}")
                    avg_val_loss += avg_loss
                    
                avg_val_loss /= len(self.mgr.targets)
                
                # Track validation history
                val_loss_history[epoch] = avg_val_loss
                
                # Manage checkpoints based on validation loss
                if epoch > 0:
                    from vesuvius.models.training.save_checkpoint import manage_checkpoint_history, manage_debug_gifs
                    
                    checkpoint_history, best_checkpoints = manage_checkpoint_history(
                        checkpoint_history=checkpoint_history,
                        best_checkpoints=best_checkpoints,
                        epoch=epoch,
                        checkpoint_path=ckpt_path,
                        validation_loss=avg_val_loss,
                        checkpoint_dir=ckpt_dir,
                        model_name=self.mgr.model_name,
                        max_recent=3,
                        max_best=2
                    )
                    
                    # Only manage debug gifs if we have any
                    if debug_gif_history:
                        # Get the most recent debug gif path if it exists for this epoch
                        recent_gif_path = None
                        for gif_epoch, gif_path in debug_gif_history:
                            if gif_epoch == epoch:
                                recent_gif_path = gif_path
                                break
                        
                        if recent_gif_path:
                            debug_gif_history, best_debug_gifs = manage_debug_gifs(
                                debug_gif_history=debug_gif_history,
                                best_debug_gifs=best_debug_gifs,
                                epoch=epoch,
                                gif_path=recent_gif_path,
                                validation_loss=avg_val_loss,
                                checkpoint_dir=ckpt_dir,
                                model_name=self.mgr.model_name,
                                max_recent=3,
                                max_best=2
                            )
                    
                # Log validation to wandb
                if self.mgr.wandb_project:
                    wandb.log({
                        **{f"val_loss_{t_name}": np.mean(val_losses[t_name])
                           for t_name in self.mgr.targets if val_losses[t_name]},
                        "val_loss_avg": avg_val_loss
                    })
                    
        # Save final checkpoint
        from vesuvius.models.training.save_checkpoint import save_final_checkpoint
        save_final_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=self.mgr.max_epoch - 1,
            ckpt_dir=ckpt_dir,
            model_name=self.mgr.model_name,
            model_config=model.final_config if hasattr(model, 'final_config') else None,
            train_dataset=train_dataset,
            additional_data={
                'teacher_state_dict': self.teacher_model.state_dict(),
                'ema_decay': self.ema_decay,
                'global_step': global_step
            }
        )
        
        print(f"\nTraining completed! Final checkpoint saved to: {ckpt_dir}")
        
        if self.mgr.wandb_project:
            wandb.finish()