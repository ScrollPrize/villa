import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch.utils.data import DataLoader

from vesuvius.models.training.train import BaseTrainer
from vesuvius.models.training.trainers.semi_supervised.two_stream_batch_sampler import TwoStreamBatchSampler
from vesuvius.models.training.trainers.semi_supervised import ramps


class TrainMeanTeacher(BaseTrainer):
    """
    Regular Mean Teacher trainer (no uncertainty masking), adapted from SSL4MIS.

    - Uses a two-stream batch sampler: first part labeled, remainder unlabeled.
    - Supervised loss computed only on labeled subset (via BaseTrainer losses).
    - Consistency loss on unlabeled subset between student and EMA teacher predictions.
    - EMA teacher updated after each optimizer step.
    """

    def __init__(self, mgr=None, verbose: bool = True):
        super().__init__(mgr, verbose)

        # Core hyperparameters
        self.ema_decay = getattr(mgr, 'ema_decay', 0.99)
        self.consistency_weight = getattr(mgr, 'consistency_weight', 0.1)
        self.consistency_rampup = getattr(mgr, 'consistency_rampup', 200.0)
        self.noise_scale = getattr(mgr, 'noise_scale', 0.1)  # Input noise for teacher
        # Ignore EMA consistency loss for the first `warmup` epochs
        # If not provided, defaults to 0 (no warmup)
        self.warmup = getattr(mgr, 'warmup', 0)

        # Semi-supervised sampling
        self.labeled_batch_size = getattr(mgr, 'labeled_batch_size', mgr.train_batch_size // 2)
        self.labeled_ratio = getattr(mgr, 'labeled_ratio', 0.1)
        self.num_labeled = getattr(mgr, 'num_labeled', None)

        # Runtime state
        self.ema_model = None
        self.global_step = 0
        self.labeled_indices = None
        self.unlabeled_indices = None

        mgr.enable_deep_supervision = False

    # --- EMA helpers --- #
    def _create_ema_model(self, model):
        ema_model = self._build_model()
        ema_model = ema_model.to(self.device)
        for param_student, param_teacher in zip(model.parameters(), ema_model.parameters()):
            param_teacher.data.copy_(param_student.data)
            param_teacher.requires_grad = False
        # Train mode mirrors SSL4MIS behavior if dropout/BN are present; harmless otherwise
        ema_model.train()
        return ema_model

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def _get_current_consistency_weight(self, epoch_like):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.consistency_weight * ramps.sigmoid_rampup(epoch_like, self.consistency_rampup)

    # --- Dataloaders (two-stream) --- #
    def _configure_dataloaders(self, train_dataset, val_dataset=None):
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
                f"Insufficient unlabeled data: need at least {unlabeled_batch_size}, have {len(self.unlabeled_indices)}."
            )

        print(f"Semi-supervised split: {num_labeled} labeled, {len(self.unlabeled_indices)} unlabeled")
        print(
            f"Batch composition: {self.labeled_batch_size} labeled + {unlabeled_batch_size} unlabeled = {self.mgr.train_batch_size} total")

        batch_sampler = TwoStreamBatchSampler(
            primary_indices=self.labeled_indices,
            secondary_indices=self.unlabeled_indices,
            batch_size=self.mgr.train_batch_size,
            secondary_batch_size=unlabeled_batch_size,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            pin_memory=(True if self.device == 'cuda' else False),
            num_workers=self.mgr.train_num_dataloader_workers,
        )

        # --- choose validation indices ---
        # If an external validation dataset is provided (e.g., via --val-dir),
        # its indices are independent from the training dataset. In that case
        # evaluate over the full validation set (or a sampler can downselect later).
        if val_dataset is not train_dataset:
            if self.mgr.verbose:
                print("Using external validation dataset for mean teacher; evaluating on full validation set")
            val_indices = list(range(len(val_dataset)))
        else:
            train_val_split = self.mgr.tr_val_split
            val_split = int(np.floor((1 - train_val_split) * num_labeled))
            val_indices = self.labeled_indices[-val_split:] if val_split > 0 else self.labeled_indices[-5:]

        from torch.utils.data import SubsetRandomSampler
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            sampler=SubsetRandomSampler(val_indices),
            pin_memory=(True if self.device == 'cuda' else False),
            num_workers=self.mgr.train_num_dataloader_workers,
        )

        self.train_indices = indices[:num_labeled]

        return train_dataloader, val_dataloader, self.labeled_indices, val_indices

    # --- Forward helpers --- #
    def _get_model_outputs(self, model, data_dict):
        inputs = data_dict["image"].to(self.device)
        targets_dict = {
            k: v.to(self.device)
            for k, v in data_dict.items()
            if k not in ["image", "patch_info", "is_unlabeled", "dataset_indices", "regression_keys"]
            and hasattr(v, "to")
        }

        batch_size = inputs.shape[0]

        if model.training and batch_size == self.mgr.train_batch_size:
            # TwoStreamBatchSampler orders labeled first, then unlabeled
            is_unlabeled = torch.zeros(batch_size, device=self.device)
            is_unlabeled[self.labeled_batch_size:] = 1.0
        else:
            is_unlabeled = torch.zeros(batch_size, device=self.device)

        outputs = model(inputs)

        # Store unlabeled mask separately in training
        if model.training:
            targets_dict['is_unlabeled'] = is_unlabeled

        # Handle deep supervision targets if enabled (use BaseTrainer util)
        if getattr(self.mgr, 'enable_deep_supervision', False):
            targets_dict = self._downsample_targets_for_ds(outputs, targets_dict)

        return inputs, targets_dict, outputs

    # --- Loss --- #
    def _compute_train_loss(self, outputs, targets_dict, loss_fns, autocast_ctx=None):
        # Supervised on labeled data only
        is_unlabeled = targets_dict.get('is_unlabeled', None)
        if is_unlabeled is None or not is_unlabeled.any():
            raise ValueError(
                "MeanTeacher trainer requires unlabeled data in each training batch. "
                "Ensure TwoStreamBatchSampler is used and batch composition is correct."
            )

        labeled_mask = is_unlabeled == 0
        unlabeled_mask = is_unlabeled == 1

        labeled_outputs = {k: v[labeled_mask] for k, v in outputs.items() if k != '_inputs'}
        labeled_targets = {k: v[labeled_mask] for k, v in targets_dict.items() if k != 'is_unlabeled'}

        total_loss, task_losses = super()._compute_train_loss(labeled_outputs, labeled_targets, loss_fns)

        # Consistency loss on unlabeled subset (skip during warmup epochs)
        do_consistency = True
        current_epoch = getattr(self, '_current_epoch', 0)
        if getattr(self, 'warmup', 0) > 0 and current_epoch < self.warmup:
            do_consistency = False

        if do_consistency:
            inputs = outputs.get('_inputs', None)
            if inputs is None:
                raise ValueError("_inputs not found in outputs; required for teacher consistency computation.")

            unlabeled_inputs = inputs[unlabeled_mask]
            if autocast_ctx is None:
                autocast_ctx = nullcontext()

            with torch.no_grad():
                noise = torch.clamp(torch.randn_like(unlabeled_inputs) * self.noise_scale, -0.2, 0.2)
                teacher_inputs = unlabeled_inputs + noise
                with autocast_ctx:
                    teacher_outputs = self.ema_model(teacher_inputs)

            # Choose first task head for consistency, skip sentinel keys
            first_task = next((k for k in outputs.keys() if k != '_inputs'), None)
            if first_task is None:
                raise ValueError("No task outputs found for consistency loss.")

            student_unlabeled = outputs[first_task][unlabeled_mask]
            teacher_unlabeled = teacher_outputs[first_task]

            # MSE on softmax probabilities (regular mean teacher)
            student_soft = F.softmax(student_unlabeled, dim=1)
            teacher_soft = F.softmax(teacher_unlabeled, dim=1)
            consistency_loss = torch.mean((student_soft - teacher_soft) ** 2)

            consistency_weight = self._get_current_consistency_weight(self.global_step // 150)
            weighted_consistency = consistency_weight * consistency_loss

            total_loss = total_loss + weighted_consistency
            task_losses['consistency'] = consistency_loss.detach().cpu().item()
        else:
            # During warmup, ignore teacher/consistency loss
            task_losses['consistency'] = 0.0

        return total_loss, task_losses

    # --- Training step override to update EMA --- #
    def _train_step(self, model, data_dict, loss_fns, use_amp, autocast_ctx, epoch, step, verbose=False,
                    scaler=None, optimizer=None, num_iters=None, grad_accumulate_n=1):
        self.global_step = epoch * (num_iters or getattr(self.mgr, 'max_steps_per_epoch', 100)) + step
        self._current_epoch = epoch

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

            # EMA update after student step
            self._update_ema_variables(model, self.ema_model, self.ema_decay, self.global_step)

        outputs.pop('_inputs', None)
        targets_dict_clean = {k: v for k, v in targets_dict.items() if k != 'is_unlabeled'}

        return total_loss, task_losses, inputs, targets_dict_clean, outputs, optimizer_stepped

    def _initialize_training(self):
        training_state = super()._initialize_training()
        model = training_state['model']
        self.ema_model = self._create_ema_model(model)
        print(f"Created EMA model (mean teacher) with decay: {self.ema_decay}")
        print(f"Consistency rampup over {self.consistency_rampup} epochs-equivalent (per-150-step units)")
        if getattr(self, 'warmup', 0) > 0:
            print(f"Warmup enabled: ignoring EMA consistency loss for first {self.warmup} epoch(s)")
        return training_state
