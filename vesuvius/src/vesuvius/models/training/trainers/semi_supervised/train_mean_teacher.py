import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch.utils.data import DataLoader, SubsetRandomSampler

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
        self.labeled_ratio = getattr(mgr, 'labeled_ratio', 1.0)
        self.num_labeled = getattr(mgr, 'num_labeled', None)

        # Runtime state
        self.ema_model = None
        self.global_step = 0
        self.labeled_indices = None
        self.unlabeled_indices = None

        mgr.enable_deep_supervision = False

        # Disable scaling augmentation - it requires padding which causes issues
        # with consistency loss (padded regions get different noise, creating fake disagreement)
        mgr.no_scaling = True

        # One-time validation debug flag
        self._val_debug_done = False

    # --- EMA helpers --- #
    def _create_ema_model(self, model):
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

    def _get_current_consistency_weight(self, epoch_like):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.consistency_weight * ramps.sigmoid_rampup(epoch_like, self.consistency_rampup)

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

        # Re-evaluate labeled/unlabeled from dataset flags to ensure supervision uses true labeled patches
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

        if self.num_labeled is not None:
            num_labeled = min(self.num_labeled, len(labeled_idx))
        else:
            num_labeled = int(self.labeled_ratio * max(1, len(labeled_idx)))
        num_labeled = max(num_labeled, self.labeled_batch_size) if labeled_idx else 0

        # Build final ordered indices list: labeled first, then unlabeled
        self.labeled_indices = labeled_idx[:num_labeled]
        self.unlabeled_indices = unlabeled_idx

        # If we still need more samples to fill the batch composition, draw extra from remaining labeled
        if len(self.labeled_indices) < self.labeled_batch_size and len(labeled_idx) > len(self.labeled_indices):
            extra = self.labeled_batch_size - len(self.labeled_indices)
            self.labeled_indices += labeled_idx[len(self.labeled_indices):len(self.labeled_indices)+extra]

        unlabeled_batch_size = self.mgr.train_batch_size - self.labeled_batch_size
        if len(self.unlabeled_indices) < unlabeled_batch_size:
            raise ValueError(
                f"Insufficient unlabeled data: need at least {unlabeled_batch_size}, have {len(self.unlabeled_indices)}."
            )

        print(f"Semi-supervised split: {len(self.labeled_indices)} labeled (from {len(labeled_idx)}), {len(self.unlabeled_indices)} unlabeled")
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
            pin_memory=(True if getattr(self.device, 'type', str(self.device)) == 'cuda' else False),
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
            val_split = int(np.floor((1 - train_val_split) * max(1, len(self.labeled_indices))))
            val_indices = self.labeled_indices[-val_split:] if val_split > 0 else self.labeled_indices[-min(5, len(self.labeled_indices)):]

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            sampler=SubsetRandomSampler(val_indices),
            pin_memory=(True if getattr(self.device, 'type', str(self.device)) == 'cuda' else False),
            num_workers=self.mgr.train_num_dataloader_workers,
        )

        self.train_indices = list(self.labeled_indices)

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
            # Warmup: trust dataset flags; After warmup: rely on sampler ordering [labeled..., unlabeled...]
            in_warmup = getattr(self, 'warmup', 0) > 0 and getattr(self, '_current_epoch', 0) < self.warmup
            ds_mask = as_float_mask(data_dict.get('is_unlabeled', None))
            if in_warmup:
                is_unlabeled = ds_mask
            else:
                is_unlabeled = torch.cat([
                    torch.zeros(self.labeled_batch_size, device=self.device),
                    torch.ones(batch_size - self.labeled_batch_size, device=self.device)
                ], dim=0)
        else:
            # Validation/other: default to labeled unless dataset provides flags
            is_unlabeled = as_float_mask(data_dict.get('is_unlabeled', None))

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
        current_epoch = getattr(self, '_current_epoch', 0)
        in_warmup = getattr(self, 'warmup', 0) > 0 and current_epoch < self.warmup
        # Only enforce presence of unlabeled data when not in warmup
        if not in_warmup and (is_unlabeled is None or not is_unlabeled.any()):
            raise ValueError(
                "MeanTeacher trainer requires unlabeled data in each training batch. "
                "Ensure TwoStreamBatchSampler is used and batch composition is correct."
            )

        labeled_mask = is_unlabeled == 0 if is_unlabeled is not None else None
        unlabeled_mask = is_unlabeled == 1 if is_unlabeled is not None else None

        if labeled_mask is not None:
            labeled_outputs = {k: v[labeled_mask] for k, v in outputs.items() if k != '_inputs'}
            labeled_targets = {k: v[labeled_mask] for k, v in targets_dict.items() if k != 'is_unlabeled'}
        else:
            labeled_outputs = {k: v for k, v in outputs.items() if k != '_inputs'}
            labeled_targets = {k: v for k, v in targets_dict.items() if k != 'is_unlabeled'}

        total_loss, task_losses = super()._compute_train_loss(labeled_outputs, labeled_targets, loss_fns)

        # Consistency loss on unlabeled subset (skip during warmup epochs)
        do_consistency = True
        if in_warmup:
            do_consistency = False

        if do_consistency:
            inputs = outputs.get('_inputs', None)
            if inputs is None:
                raise ValueError("_inputs not found in outputs; required for teacher consistency computation.")

            unlabeled_inputs = inputs[unlabeled_mask]
            if autocast_ctx is None:
                autocast_ctx = nullcontext()

            noise_bound = self._get_noise_bounds(unlabeled_inputs)
            with torch.no_grad():
                noise = torch.clamp(torch.randn_like(unlabeled_inputs) * self.noise_scale, -noise_bound, noise_bound)
                teacher_inputs = unlabeled_inputs + noise
                with autocast_ctx:
                    teacher_outputs = self.ema_model(teacher_inputs)

            # Select task head for consistency, prefer explicit config if provided
            consistency_target = getattr(self.mgr, 'consistency_target', None)
            first_task = None
            if consistency_target is not None and consistency_target in outputs:
                first_task = consistency_target
            else:
                # Fallback to the first available prediction key
                first_task = next((k for k in outputs.keys() if k != '_inputs'), None)
            if first_task is None:
                raise ValueError("No task outputs found for consistency loss.")

            student_unlabeled = outputs[first_task][unlabeled_mask]
            teacher_unlabeled = teacher_outputs[first_task]

            # Consistency on probabilities: handle binary (1-channel) vs multi-class
            if student_unlabeled.shape[1] == 1:
                # Binary case: use sigmoid probabilities
                student_prob = torch.sigmoid(student_unlabeled)
                teacher_prob = torch.sigmoid(teacher_unlabeled)
                consistency_loss = F.mse_loss(student_prob, teacher_prob)
            else:
                # Multi-class: MSE on softmax probabilities
                student_soft = F.softmax(student_unlabeled, dim=1)
                teacher_soft = F.softmax(teacher_unlabeled, dim=1)
                consistency_loss = F.mse_loss(student_soft, teacher_soft)

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

            # One-time debug: inspect shapes and label stats on first step
            if epoch == 0 and step == 0 and (not self.is_distributed or self.rank == 0):
                try:
                    # Prediction head shape
                    pred_keys = [k for k in outputs.keys() if k != '_inputs']
                    if pred_keys:
                        pk = pred_keys[0]
                        ps = tuple(outputs[pk].shape)
                        print(f"[MT DEBUG] pred[{pk}] shape: {ps}")
                    # Labeled/unlabeled counts
                    is_unlabeled = targets_dict.get('is_unlabeled', None)
                    if is_unlabeled is not None:
                        lbl_mask = (is_unlabeled == 0)
                        ulb_mask = (is_unlabeled == 1)
                        print(f"[MT DEBUG] labeled count: {int(lbl_mask.sum().item())} | unlabeled count: {int(ulb_mask.sum().item())}")
                    else:
                        lbl_mask = None
                    # Ground truth stats for supervised subset
                    for k, v in list(targets_dict.items()):
                        if k in ('is_unlabeled', 'skel'):
                            continue
                        t = v[lbl_mask] if lbl_mask is not None else v
                        try:
                            t_min = float(t.min().item())
                            t_max = float(t.max().item())
                            pos_frac = float((t > 0).float().mean().item())
                            print(f"[MT DEBUG] gt[{k}] shape: {tuple(t.shape)} | min/max: {t_min}/{t_max} | pos_frac: {pos_frac:.6f}")
                        except Exception as e:
                            print(f"[MT DEBUG] gt[{k}] stats error: {e}")
                except Exception as e:
                    print(f"[MT DEBUG] error during debug logging: {e}")

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

    # --- Validation step override for first-batch debug --- #
    def _validation_step(self, model, data_dict, loss_fns, use_amp):
        task_losses, inputs, targets_dict, outputs = super()._validation_step(
            model=model,
            data_dict=data_dict,
            loss_fns=loss_fns,
            use_amp=use_amp,
        )

        # One-time debug info on first validation batch (rank 0 only)
        if not self._val_debug_done and (not self.is_distributed or self.rank == 0):
            self._val_debug_done = True
            try:
                # Prediction head shape
                pred_keys = [k for k in outputs.keys()]
                if pred_keys:
                    pk = pred_keys[0]
                    ps = tuple(outputs[pk][0].shape if isinstance(outputs[pk], (list, tuple)) else outputs[pk].shape)
                    print(f"[MT DEBUG] [val] pred[{pk}] shape: {ps}")

                # Ground truth stats
                for k, v in list(targets_dict.items()):
                    if k in ('is_unlabeled', 'skel'):
                        continue
                    t = v[0] if isinstance(v, (list, tuple)) else v
                    try:
                        t_min = float(t.min().item())
                        t_max = float(t.max().item())
                        pos_frac = float((t > 0).float().mean().item())
                        print(f"[MT DEBUG] [val] gt[{k}] shape: {tuple(t.shape)} | min/max: {t_min}/{t_max} | pos_frac: {pos_frac:.6f}")
                    except Exception as e:
                        print(f"[MT DEBUG] [val] gt[{k}] stats error: {e}")
            except Exception as e:
                print(f"[MT DEBUG] [val] error during debug logging: {e}")

        return task_losses, inputs, targets_dict, outputs

    def _initialize_training(self):
        training_state = super()._initialize_training()
        model = training_state['model']
        self.ema_model = self._create_ema_model(model)
        print(f"Created EMA model (mean teacher) with decay: {self.ema_decay}")
        print(f"Consistency rampup over {self.consistency_rampup} epochs-equivalent (per-150-step units)")
        if getattr(self, 'warmup', 0) > 0:
            print(f"Warmup enabled: ignoring EMA consistency loss for first {self.warmup} epoch(s)")
        return training_state

    def _update_dataloaders_for_epoch(self,
                                      train_dataloader,
                                      val_dataloader,
                                      train_dataset,
                                      val_dataset,
                                      epoch):
        """
        During warmup epochs, switch to a supervised dataloader over labeled data only.
        After warmup, (re)enable the two-stream semi-supervised dataloader.
        """
        in_warmup = getattr(self, 'warmup', 0) > 0 and epoch < self.warmup
        # If we don't have a labeled split yet, keep existing dataloaders
        if self.labeled_indices is None or self.unlabeled_indices is None:
            return train_dataloader, val_dataloader

        if in_warmup:
            # Build a standard supervised dataloader over labeled subset
            from torch.utils.data import Subset, SubsetRandomSampler
            from torch.utils.data.distributed import DistributedSampler
            per_device_batch = self.mgr.train_batch_size
            labeled_subset = Subset(train_dataset, self.labeled_indices)
            if self.is_distributed:
                sampler = DistributedSampler(labeled_subset, num_replicas=self.world_size, rank=self.rank,
                                             shuffle=True, drop_last=False)
            else:
                sampler = SubsetRandomSampler(list(range(len(labeled_subset))))
            pin_mem = True if self.device == torch.device('cuda') or self.device.type == 'cuda' else False
            dl_kwargs = {}
            if self.mgr.train_num_dataloader_workers and self.mgr.train_num_dataloader_workers > 0:
                dl_kwargs['prefetch_factor'] = 2
            new_train_dl = DataLoader(
                labeled_subset,
                batch_size=per_device_batch,
                sampler=sampler,
                shuffle=False,
                pin_memory=pin_mem,
                num_workers=self.mgr.train_num_dataloader_workers,
                **dl_kwargs
            )
            return new_train_dl, val_dataloader
        else:
            # Ensure we are using the two-stream loader after warmup
            new_train_dl, new_val_dl, _, _ = self._configure_dataloaders(train_dataset, val_dataset)
            return new_train_dl, new_val_dl
