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


def prob_mse_loss(student_logits, target_probs):
    """MSE loss between student softmax and target probabilities.

    Use this when the target is already softmax probabilities (e.g., from
    ensemble aggregation) rather than raw logits.

    Args:
        student_logits: Raw logits from student model [B, C, ...]
        target_probs: Softmax probabilities from teacher [B, C, ...]

    Returns:
        Element-wise squared differences [B, C, ...]
    """
    assert student_logits.size() == target_probs.size()
    student_softmax = F.softmax(student_logits, dim=1)
    return (student_softmax - target_probs) ** 2


class TrainUncertaintyAwareMeanTeacher(BaseTrainer):
    def __init__(self, mgr=None, verbose: bool = True):
        super().__init__(mgr, verbose)
        
        self.ema_decay = getattr(mgr, 'ema_decay', 0.99)
        self.consistency_weight = getattr(mgr, 'consistency_weight', 0.1)
        self.consistency_rampup = getattr(mgr, 'consistency_rampup', 200.0)
        self.uncertainty_threshold_start = getattr(mgr, 'uncertainty_threshold_start', 0.75)
        self.uncertainty_threshold_end = getattr(mgr, 'uncertainty_threshold_end', 1.0)
        self.uncertainty_T = getattr(mgr, 'uncertainty_T', 4)  # Number of stochastic forward passes

        # Validate minimum uncertainty_T
        if self.uncertainty_T < 2:
            print(f"Warning: uncertainty_T={self.uncertainty_T} is less than 2. Setting to minimum of 2.")
            self.uncertainty_T = 2

        self.noise_scale = getattr(mgr, 'noise_scale', 0.1)  # Noise scale for stochastic augmentation

        # Ensemble model configuration for multi-model uncertainty estimation
        self.ensemble_model_paths = getattr(mgr, 'ensemble_model_paths', []) or []
        self.ensemble_aggregation = getattr(mgr, 'ensemble_aggregation', 'max_confidence')
        self.ensemble_models = []

        # Validate ensemble_aggregation
        valid_aggregations = ('max_confidence', 'average')
        if self.ensemble_aggregation not in valid_aggregations:
            print(f"Warning: Unknown ensemble_aggregation '{self.ensemble_aggregation}'. "
                  f"Using 'max_confidence'. Valid options: {valid_aggregations}")
            self.ensemble_aggregation = 'max_confidence'

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

    def _load_ensemble_models(self):
        """Load additional pre-trained models for ensemble uncertainty estimation.

        Uses the same load_checkpoint pattern as the main model loading,
        but only loads model weights (not optimizer/scheduler state).
        Each ensemble model is frozen and set to eval mode.

        Returns:
            List of loaded models, empty if no paths configured or all fail to load.
        """
        from vesuvius.models.utilities.load_checkpoint import load_checkpoint
        from vesuvius.models.training.lr_schedulers import get_scheduler

        ensemble_models = []

        for path in self.ensemble_model_paths:
            try:
                print(f"Loading ensemble model from: {path}")

                # Build a fresh model with the same architecture
                model = self._build_model()
                model = model.to(self.device)

                # Create dummy optimizer/scheduler (required by load_checkpoint signature)
                dummy_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                dummy_scheduler = get_scheduler(
                    scheduler_type='poly',
                    optimizer=dummy_optimizer,
                    initial_lr=0.01,
                    max_steps=100
                )

                # Load weights only
                model, _, _, _, loaded = load_checkpoint(
                    checkpoint_path=path,
                    model=model,
                    optimizer=dummy_optimizer,
                    scheduler=dummy_scheduler,
                    mgr=self.mgr,
                    device=self.device,
                    load_weights_only=True
                )

                if loaded:
                    # Set to eval mode and freeze all parameters
                    model.eval()
                    for param in model.parameters():
                        param.requires_grad = False
                    ensemble_models.append(model)
                    print(f"  Successfully loaded ensemble model ({len(model.state_dict())} tensors)")
                else:
                    print(f"  Warning: Failed to load ensemble model from {path}")

            except Exception as e:
                print(f"  Error loading ensemble model from {path}: {e}")

        return ensemble_models

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

    def _get_first_task_key(self, outputs):
        """Extract the first task key from model outputs, skipping '_inputs'."""
        for key in outputs.keys():
            if key != '_inputs':
                return key
        raise ValueError("No task outputs found in model outputs")

    def _aggregate_max_confidence(self, all_preds):
        """Aggregate predictions by taking max-confidence prediction per voxel.

        For each voxel, selects the prediction from whichever model has the
        highest max-class probability (i.e., most confident prediction).

        Args:
            all_preds: [num_passes, B, num_classes, ...] tensor of softmax predictions

        Returns:
            [B, num_classes, ...] tensor where each voxel has the prediction from
            the most confident model (highest max probability across classes)
        """
        # all_preds: [T, B, C, D, H, W] or [T, B, C, H, W]

        # Get max probability per voxel per model: [T, B, 1, ...]
        max_probs, _ = all_preds.max(dim=2, keepdim=True)  # max across classes

        # Find which model has highest confidence at each voxel: [1, B, 1, ...]
        _, best_model_idx = max_probs.max(dim=0, keepdim=True)  # max across models

        # Expand best_model_idx to match all_preds shape for gather
        # From [1, B, 1, ...] to [1, B, C, ...]
        expand_shape = list(all_preds.shape)
        expand_shape[0] = 1
        best_model_idx = best_model_idx.expand(*expand_shape)

        # Gather predictions from best model at each voxel
        aggregated = torch.gather(all_preds, dim=0, index=best_model_idx).squeeze(0)

        return aggregated

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
                "Use ZarrDataset or implement "
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
        """Compute uncertainty using ensemble of models with configurable aggregation.

        Pass distribution for T total passes:
        - min(len(ensemble_models), T) passes: one with each additional pre-trained model
        - Remaining passes: EMA teacher model

        Aggregation strategy (configurable via ensemble_aggregation):
        - 'max_confidence': For each voxel, take prediction from most confident model
        - 'average': Average all predictions (original behavior)
        """
        batch_size = unlabeled_inputs.shape[0]
        T = self.uncertainty_T
        num_ensemble = len(self.ensemble_models)

        # Calculate pass distribution
        # Up to num_ensemble for additional models, rest for EMA
        ensemble_passes = min(num_ensemble, T)  # Don't exceed T
        ema_passes = max(0, T - num_ensemble)

        # Get number of output channels
        num_classes = self.mgr.out_channels[0] if isinstance(self.mgr.out_channels, tuple) else self.mgr.out_channels

        noise_bound = self._get_noise_bounds(unlabeled_inputs)

        # Collect all predictions: list of [B, num_classes, ...] tensors
        all_preds = []

        with torch.no_grad():
            # 1. Ensemble model passes (no noise - these are frozen pre-trained models)
            for i, ensemble_model in enumerate(self.ensemble_models[:ensemble_passes]):
                with autocast_ctx:
                    ensemble_outputs = ensemble_model(unlabeled_inputs)
                    first_task = self._get_first_task_key(ensemble_outputs)
                    ensemble_pred = F.softmax(ensemble_outputs[first_task], dim=1)
                    all_preds.append(ensemble_pred)

            # 2. EMA teacher passes with noise (fills remaining T - num_ensemble passes)
            for i in range(ema_passes):
                noise = torch.clamp(
                    torch.randn_like(unlabeled_inputs) * self.noise_scale,
                    -noise_bound, noise_bound
                )
                ema_inputs = unlabeled_inputs + noise
                with autocast_ctx:
                    ema_outputs = self.ema_model(ema_inputs)
                    first_task = self._get_first_task_key(ema_outputs)
                    ema_pred = F.softmax(ema_outputs[first_task], dim=1)
                    all_preds.append(ema_pred)

        # Stack all predictions: [num_passes, B, num_classes, ...]
        all_preds = torch.stack(all_preds, dim=0)

        # Aggregation based on configured strategy
        if self.ensemble_aggregation == 'max_confidence':
            mean_pred = self._aggregate_max_confidence(all_preds)
        else:
            # Backward compatible: average predictions
            mean_pred = torch.mean(all_preds, dim=0)

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
        
        # Compute uncertainty and get ensemble-aggregated predictions as pseudo-labels
        # mean_pred combines predictions from:
        # - Frozen pre-trained ensemble models (clean inputs, authoritative)
        # - EMA teacher (noised inputs for diversity)
        uncertainty, mean_pred = self._compute_uncertainty(unlabeled_inputs, autocast_ctx)

        first_task = list(outputs.keys())[0]
        if first_task == '_inputs':
            first_task = list(outputs.keys())[1] if len(outputs.keys()) > 1 else None
            if first_task is None:
                raise ValueError("No task outputs found besides _inputs")

        student_unlabeled = outputs[first_task][unlabeled_mask]

        # Compute consistency loss using ensemble-aggregated predictions as target
        # mean_pred is already softmax probabilities, so use prob_mse_loss
        consistency_dist = prob_mse_loss(student_unlabeled, mean_pred)
        
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
        """Override to initialize EMA model and ensemble models after base initialization"""
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

        # Load ensemble models if configured
        if self.ensemble_model_paths:
            self.ensemble_models = self._load_ensemble_models()
            num_ensemble = len(self.ensemble_models)
            ensemble_passes = min(num_ensemble, self.uncertainty_T)
            ema_passes = max(0, self.uncertainty_T - num_ensemble)

            print(f"Loaded {num_ensemble} ensemble models for uncertainty estimation")
            if num_ensemble > self.uncertainty_T:
                print(f"  Warning: More ensemble models ({num_ensemble}) than available passes ({self.uncertainty_T}). "
                      f"Using first {self.uncertainty_T} ensemble models.")
            print(f"Pass distribution: {ensemble_passes} ensemble + {ema_passes} EMA = {self.uncertainty_T} total")
        else:
            self.ensemble_models = []
            print(f"No ensemble models configured - using EMA teacher only")
            print(f"Pass distribution: {self.uncertainty_T} EMA passes")

        print(f"Uncertainty estimation using {self.uncertainty_T} forward passes")
        print(f"Aggregation strategy: {self.ensemble_aggregation}")
        print(f"Consistency weight ramp-up over {self.consistency_rampup} epochs")

        return training_state
