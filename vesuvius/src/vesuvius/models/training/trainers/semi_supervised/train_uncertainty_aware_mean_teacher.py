import os
from contextlib import nullcontext
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
        self.uncertainty_T = getattr(mgr, 'uncertainty_T', 8)  # Number of stochastic forward passes
        self.noise_scale = getattr(mgr, 'noise_scale', 0.1)  # Noise scale for stochastic augmentation
        self.labeled_batch_size = getattr(mgr, 'labeled_batch_size', mgr.train_batch_size // 2)
        
        # Semi-supervised data split parameters
        self.labeled_ratio = getattr(mgr, 'labeled_ratio', 0.1)  # Fraction of data to use as labeled
        self.num_labeled = getattr(mgr, 'num_labeled', None)
        
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
        
        if self.num_labeled is not None:
            num_labeled = min(self.num_labeled, dataset_size)
        else:
            num_labeled = int(self.labeled_ratio * dataset_size)
        
        num_labeled = max(num_labeled, self.labeled_batch_size)
        
        self.labeled_indices = indices[:num_labeled]
        self.unlabeled_indices = indices[num_labeled:]
        
        unlabeled_batch_size = self.mgr.train_batch_size - self.labeled_batch_size
        if len(self.unlabeled_indices) < unlabeled_batch_size:
            raise ValueError(
                f"Insufficient unlabeled data for semi-supervised training. "
                f"Need at least {unlabeled_batch_size} unlabeled samples per batch, "
                f"but only have {len(self.unlabeled_indices)} unlabeled samples total. "
                f"Either reduce labeled_batch_size ({self.labeled_batch_size}), "
                f"reduce labeled_ratio ({self.labeled_ratio}), or increase dataset size."
            )
        
        print(f"Semi-supervised split: {num_labeled} labeled, {len(self.unlabeled_indices)} unlabeled")
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
        
        train_val_split = self.mgr.tr_val_split
        val_split = int(np.floor((1 - train_val_split) * num_labeled))
        val_indices = self.labeled_indices[-val_split:] if val_split > 0 else self.labeled_indices[-5:]
        
        from torch.utils.data import SubsetRandomSampler
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            sampler=SubsetRandomSampler(val_indices),
            pin_memory=(True if self.device == 'cuda' else False),
            num_workers=self.mgr.train_num_dataloader_workers
        )
        
        self.train_indices = indices[:num_labeled]
        
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
        
        if self.training and batch_size == self.mgr.train_batch_size:
            # TwoStreamBatchSampler orders data like so : [labeled..., unlabeled...]
            # i.e :  labeled_batch_size samples are labeled, rest are unlabeled
            is_unlabeled = torch.zeros(batch_size, device=self.device)
            is_unlabeled[self.labeled_batch_size:] = 1.0
        else:
            # During validation, all samples are labeled
            is_unlabeled = torch.zeros(batch_size, device=self.device)
        
        outputs = model(inputs)
        
        targets_dict['is_unlabeled'] = is_unlabeled
        
        return inputs, targets_dict, outputs
    
    def _compute_uncertainty(self, unlabeled_inputs, autocast_ctx):
        """Compute uncertainty using multiple stochastic forward passes"""
        batch_size = unlabeled_inputs.shape[0]
        T = self.uncertainty_T
        
        volume_batch_r = unlabeled_inputs.repeat(2, 1, 1, 1, 1)
        stride = volume_batch_r.shape[0] // 2
        
        if unlabeled_inputs.dim() == 5:  # 3D case: [B, C, D, H, W]
            _, c, d, h, w = unlabeled_inputs.shape
            preds = torch.zeros([stride * T, self.mgr.num_classes, d, h, w]).to(self.device)
        else:  # 2D case: [B, C, H, W]
            _, c, h, w = unlabeled_inputs.shape
            preds = torch.zeros([stride * T, self.mgr.num_classes, h, w]).to(self.device)
        
        with torch.no_grad():
            for i in range(T // 2):
                noise = torch.clamp(
                    torch.randn_like(volume_batch_r) * self.noise_scale,
                    -0.2, 0.2
                )
                ema_inputs = volume_batch_r + noise
                with autocast_ctx:
                    preds[2 * stride * i:2 * stride * (i + 1)] = self.ema_model(ema_inputs)
        
        preds = F.softmax(preds, dim=1)
        if unlabeled_inputs.dim() == 5:
            preds = preds.reshape(T, stride, self.mgr.num_classes, d, h, w)
        else:
            preds = preds.reshape(T, stride, self.mgr.num_classes, h, w)
        
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
        
        with torch.no_grad():
            noise = torch.clamp(
                torch.randn_like(unlabeled_inputs) * self.noise_scale,
                -0.2, 0.2
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
        
        # Compute consistency loss (element-wise)
        consistency_dist = softmax_mse_loss(student_unlabeled, teacher_outputs)
        
        # Apply uncertainty-based weighting
        # Use sigmoid ramp-up for threshold
        current_iter = self.global_step
        
        max_steps_per_epoch = getattr(self.mgr, 'max_steps_per_epoch', 100)
        max_epochs = getattr(self.mgr, 'max_epoch', 100)
        max_iterations = max_steps_per_epoch * max_epochs
        threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(current_iter, max_iterations)) * np.log(2)
        
        mask = (uncertainty < threshold).float()
        while mask.dim() < consistency_dist.dim():
            mask = mask.unsqueeze(1)
        
        masked_loss = mask * consistency_dist
        consistency_loss = torch.sum(masked_loss) / (2 * torch.sum(mask) + 1e-16)
        consistency_weight = self._get_current_consistency_weight(self.global_step // 150)
        
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
        
        # Remove _inputs from outputs to avoid issues downstream
        outputs.pop('_inputs', None)
        
        return total_loss, task_losses, inputs, targets_dict, outputs, optimizer_stepped
    
    def train(self):
        """Override train method to add EMA model initialization"""
        training_state = self._initialize_training()
        
        # Create EMA model after the student model is initialized
        model = training_state['model']
        self.ema_model = self._create_ema_model(model)
        print(f"Created EMA model with decay factor: {self.ema_decay}")
        print(f"Uncertainty estimation using {self.uncertainty_T} forward passes")
        print(f"Consistency weight ramp-up over {self.consistency_rampup} epochs")
        
        # Call the parent train method with our EMA model ready
        # The parent train will use our overridden _train_step and _compute_train_loss
        super().train()