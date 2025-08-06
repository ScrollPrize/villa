import multiprocessing
import sys

### this is at the top because s3fs/fsspec do not work with fork
if __name__ == '__main__' and len(sys.argv) > 1:
    # Quick check for S3 paths in command line args
    if any('s3://' in str(arg) for arg in sys.argv) or '--config-path' in sys.argv:
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

from pathlib import Path
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
from vesuvius.models.training.lr_schedulers import get_scheduler, PolyLRScheduler
from torch.utils.data import DataLoader, SubsetRandomSampler
from vesuvius.utils.utils import init_weights_he
from vesuvius.models.datasets import NapariDataset, ImageDataset, ZarrDataset
from vesuvius.utils.plotting import save_debug
from vesuvius.models.build.build_network_from_config import NetworkFromConfig

from vesuvius.models.training.loss.losses import _create_loss
from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.models.training.save_checkpoint import (
    save_checkpoint,
    manage_checkpoint_history,
    manage_debug_gifs,
    cleanup_old_configs,
    save_final_checkpoint
)

from vesuvius.models.utilities.load_checkpoint import load_checkpoint
from vesuvius.models.utilities.get_accelerator import get_accelerator
from vesuvius.models.utilities.compute_gradient_norm import compute_gradient_norm
from vesuvius.models.utilities.s3_utils import detect_s3_paths, setup_multiprocessing_for_s3
from vesuvius.models.training.auxiliary_tasks import compute_auxiliary_loss
from vesuvius.models.training.wandb_logging import save_train_val_filenames
from vesuvius.models.utilities.cli_utils import update_config_from_args
from vesuvius.models.configuration.config_utils import configure_targets
from vesuvius.models.evaluation.connected_components import ConnectedComponentsMetric
from vesuvius.models.evaluation.critical_components import CriticalComponentsMetric
from vesuvius.models.evaluation.iou_dice import IOUDiceMetric
from vesuvius.models.evaluation.hausdorff import HausdorffDistanceMetric


from itertools import cycle
from contextlib import nullcontext
from collections import deque
import gc



class BaseTrainer:
    def __init__(self,
                 mgr=None,
                 verbose: bool = True):
        """
        Initialize the trainer with a config manager instance

        Parameters
        ----------
        mgr : ConfigManager, optional
            If provided, use this config manager instance instead of creating a new one
        verbose : bool
            Whether to print verbose output
        """
        if mgr is not None:
            self.mgr = mgr
        else:
            from vesuvius.models.configuration.config_manager import ConfigManager
            self.mgr = ConfigManager(verbose)

        self.device = get_accelerator()

    # --- build model --- #
    def _build_model(self):
        if not hasattr(self.mgr, 'model_config') or self.mgr.model_config is None:
            print("Initializing model_config with defaults")
            self.mgr.model_config = {
                "train_patch_size": self.mgr.train_patch_size,
                "in_channels": self.mgr.in_channels,
                "model_name": self.mgr.model_name,
                "autoconfigure": self.mgr.autoconfigure,
                "conv_op": "nn.Conv2d" if len(self.mgr.train_patch_size) == 2 else "nn.Conv3d"
            }

        model = NetworkFromConfig(self.mgr)
        return model

    # --- configure dataset --- #
    def _configure_dataset(self, is_training=True):
        data_format = getattr(self.mgr, 'data_format', 'zarr').lower()

        if data_format == 'napari':
            dataset = NapariDataset(mgr=self.mgr, is_training=is_training)
        elif data_format == 'image':
            dataset = ImageDataset(mgr=self.mgr, is_training=is_training)
        elif data_format == 'zarr':
            dataset = ZarrDataset(mgr=self.mgr, is_training=is_training)
        else:
            raise ValueError(f"Unsupported data format: {data_format}. "
                             f"Supported formats are: 'napari', 'image', 'zarr'")

        print(f"Using {data_format} dataset format ({'training' if is_training else 'validation'})")

        return dataset

    # --- losses ---- #
    def _build_loss(self):
        loss_fns = {}
        self._deferred_losses = {}  # losses that may be added later in training (at a selected epoch)
        
        for task_name, task_info in self.mgr.targets.items():
            task_losses = []
            deferred_losses = []

            if "losses" in task_info:
                print(f"Target {task_name} using multiple losses:")
                for loss_cfg in task_info["losses"]:
                    loss_name = loss_cfg["name"]
                    loss_weight = loss_cfg.get("weight", 1.0)
                    loss_kwargs = loss_cfg.get("kwargs", {})
                    start_epoch = loss_cfg.get("start_epoch", 0)

                    weight = loss_kwargs.get("weight", None)
                    ignore_index = loss_kwargs.get("ignore_index", -100)
                    pos_weight = loss_kwargs.get("pos_weight", None)

                    try:
                        loss_fn = _create_loss(
                            name=loss_name,
                            loss_config=loss_kwargs,
                            weight=weight,
                            ignore_index=ignore_index,
                            pos_weight=pos_weight,
                            mgr=self.mgr
                        )
                        
                        if start_epoch > 0:
                            # Store for later addition
                            deferred_losses.append({
                                'loss_fn': loss_fn,
                                'weight': loss_weight,
                                'start_epoch': start_epoch,
                                'name': loss_name
                            })
                            print(f"  - {loss_name} (weight: {loss_weight}) - will start at epoch {start_epoch}")
                        else:
                            # Add immediately
                            task_losses.append((loss_fn, loss_weight))
                            print(f"  - {loss_name} (weight: {loss_weight})")
                    except RuntimeError as e:
                        raise ValueError(
                            f"Failed to create loss function '{loss_name}' for target '{task_name}': {str(e)}")

            loss_fns[task_name] = task_losses
            if deferred_losses:
                self._deferred_losses[task_name] = deferred_losses

        return loss_fns

    def _update_loss_for_epoch(self, loss_fns, epoch):
        if hasattr(self, '_deferred_losses') and self._deferred_losses:
            task_names = list(self._deferred_losses.keys())
            
            for task_name in task_names:
                deferred_list = self._deferred_losses[task_name]

                losses_to_add = []
                remaining_deferred = []
                
                for deferred_loss in deferred_list:
                    if epoch >= deferred_loss['start_epoch']:
                        losses_to_add.append(deferred_loss)
                    else:
                        remaining_deferred.append(deferred_loss)

                for loss_info in losses_to_add:
                    loss_fns[task_name].append((loss_info['loss_fn'], loss_info['weight']))
                    print(f"\nEpoch {epoch}: Adding {loss_info['name']} to task '{task_name}' (weight: {loss_info['weight']})")

                if remaining_deferred:
                    self._deferred_losses[task_name] = remaining_deferred
                else:
                    del self._deferred_losses[task_name]
        
        return loss_fns

    # --- optimizer ---- #
    def _get_optimizer(self, model):

        optimizer_config = {
            'name': self.mgr.optimizer,
            'learning_rate': self.mgr.initial_lr,
            'weight_decay': self.mgr.weight_decay
        }

        return create_optimizer(optimizer_config, model)

    # --- scheduler --- #
    def _get_scheduler(self, optimizer):

        scheduler_type = getattr(self.mgr, 'scheduler', 'poly')
        scheduler_kwargs = getattr(self.mgr, 'scheduler_kwargs', {})

        scheduler = get_scheduler(
            scheduler_type=scheduler_type,
            optimizer=optimizer,
            initial_lr=self.mgr.initial_lr,
            max_steps=self.mgr.max_epoch,
            **scheduler_kwargs
        )

        print(f"Using {scheduler_type} learning rate scheduler")
        
        # set some per iteration schedulers so we can easily step them once per iter vs once per epoch
        per_iter_schedulers = ['onecycle', 'cyclic', 'cosine_warmup']
        is_per_iteration = scheduler_type.lower() in per_iter_schedulers
        
        return scheduler, is_per_iteration

    # --- scaler --- #
    def _initialize_evaluation_metrics(self):

        metrics = {}
        for task_name, task_config in self.mgr.targets.items():
            task_metrics = []

            num_classes = task_config.get('num_classes', 2)
            task_metrics.append(ConnectedComponentsMetric(num_classes=num_classes))

            # if num_classes == 2:
            #     task_metrics.append(CriticalComponentsMetric())

            task_metrics.append(IOUDiceMetric(num_classes=num_classes))
            # task_metrics.append(HausdorffDistanceMetric(num_classes=num_classes))
            metrics[task_name] = task_metrics
        
        return metrics
    
    def _get_scaler(self, device_type='cuda', use_amp=True):
        # for cuda, we can use a grad scaler for mixed precision training if amp is enabled
        # for mps or cpu, or when amp is disabled, we create a dummy scaler that does nothing

        class DummyScaler:
            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                pass

            def step(self, optimizer):
                optimizer.step()

            def update(self):
                pass


        if device_type == 'cuda' and use_amp:
            # Use GradScaler for both float16 and bfloat16
            # even though according to every single thing i've read you dont need grad scaling with bf16
            # my model learns essentially nothing with it disabled. i have no idea why.
            if torch.cuda.is_bf16_supported():
                print("Using GradScaler with bfloat16 autocast")
            else:
                print("Using GradScaler with float16 autocast")
            return torch.amp.GradScaler('cuda')
        else:
            # Not using amp or not on cuda - no gradient scaling needed
            return DummyScaler()

    # --- dataloaders --- #
    def _configure_dataloaders(self, train_dataset, val_dataset=None):

        if val_dataset is None:
            val_dataset = train_dataset
            
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))

        if hasattr(self.mgr, 'seed'):
            np.random.seed(self.mgr.seed)
            if self.mgr.verbose:
                print(f"Using seed {self.mgr.seed} for train/val split")

        np.random.shuffle(indices)

        train_val_split = self.mgr.tr_val_split
        split = int(np.floor(train_val_split * dataset_size))
        train_indices, val_indices = indices[:split], indices[split:]
        batch_size = self.mgr.train_batch_size

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      sampler=SubsetRandomSampler(train_indices),
                                      pin_memory=(True if self.device == 'cuda' else False),
                                      num_workers=self.mgr.train_num_dataloader_workers
                                      )

        val_dataloader = DataLoader(val_dataset,
                                    batch_size=1,
                                    sampler=SubsetRandomSampler(val_indices),
                                    pin_memory=(True if self.device == 'cuda' else False),
                                    num_workers=self.mgr.train_num_dataloader_workers
                                    )

        return train_dataloader, val_dataloader, train_indices, val_indices

    def _initialize_training(self):
        # Check for S3 paths and set up multiprocessing if needed
        if detect_s3_paths(self.mgr):
            print("\nDetected S3 paths in configuration")
            setup_multiprocessing_for_s3()

        # the is_training flag forces the dataset to perform augmentations
        # we put augmentations in the dataset class so we can use the __getitem__ method
        # for free multi processing of augmentations 
        train_dataset = self._configure_dataset(is_training=True)
        val_dataset = self._configure_dataset(is_training=False)
        

        self.mgr.auto_detect_channels(train_dataset)
        model = self._build_model()
        optimizer = self._get_optimizer(model)
        loss_fns = self._build_loss()
        scheduler, is_per_iteration_scheduler = self._get_scheduler(optimizer)

        model.apply(lambda module: init_weights_he(module, neg_slope=0.2))
        model = model.to(self.device)

        if self.device.type == 'cuda':
            model = torch.compile(model)

        use_amp = not getattr(self.mgr, 'no_amp', False)
        
        # betti loss (and maybe some future ones) require disabling amp because they call .detach()
        # which breaks the grad chain
        for task_name, task_info in self.mgr.targets.items():
            if "losses" in task_info:
                for loss_cfg in task_info["losses"]:
                    if loss_cfg["name"] == "BettiMatchingLoss":
                        use_amp = False
                        print(f"Automatic Mixed Precision (AMP) disabled due to BettiMatchingLoss (incompatible with gradient computation)")
                        break
            if not use_amp:
                break
        
        if not use_amp and getattr(self.mgr, 'no_amp', False):
            print("Automatic Mixed Precision (AMP) is disabled")
        
        scaler = self._get_scaler(self.device.type, use_amp=use_amp)
        train_dataloader, val_dataloader, train_indices, val_indices = self._configure_dataloaders(train_dataset,
                                                                                                   val_dataset)
        os.makedirs(self.mgr.ckpt_out_base, exist_ok=True)
        model_ckpt_dir = os.path.join(self.mgr.ckpt_out_base, self.mgr.model_name)
        os.makedirs(model_ckpt_dir, exist_ok=True)

        now = datetime.now()
        date_str = now.strftime('%m%d%y')
        time_str = now.strftime('%H%M')
        ckpt_dir = os.path.join('checkpoints', f"{self.mgr.model_name}_{date_str}{time_str}")
        os.makedirs(ckpt_dir, exist_ok=True)

        start_epoch = 0
        if hasattr(self.mgr, 'checkpoint_path') and self.mgr.checkpoint_path:
            model, optimizer, scheduler, start_epoch, checkpoint_loaded = load_checkpoint(
                checkpoint_path=self.mgr.checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                mgr=self.mgr,
                device=self.device,
                load_weights_only=getattr(self.mgr, 'load_weights_only', False)
            )

            if checkpoint_loaded and self.mgr.load_weights_only:
                scheduler, is_per_iteration_scheduler = self._get_scheduler(optimizer)

        return {
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'is_per_iteration_scheduler': is_per_iteration_scheduler,
            'loss_fns': loss_fns,
            'scaler': scaler,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'train_dataloader': train_dataloader,
            'val_dataloader': val_dataloader,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'use_amp': use_amp,
            'start_epoch': start_epoch,
            'ckpt_dir': ckpt_dir,
            'model_ckpt_dir': model_ckpt_dir
        }

    def _initialize_wandb(self, train_dataset, val_dataset, train_indices, val_indices):
        """Initialize Weights & Biases logging if configured."""
        if self.mgr.wandb_project:
            import wandb  # lazy import in case it's not available
            train_val_splits = save_train_val_filenames(self, train_dataset, val_dataset, train_indices, val_indices)
            mgr_config = self.mgr.convert_to_dict()
            mgr_config.update({'train_val_splits': train_val_splits})

            wandb.init(
                entity=self.mgr.wandb_entity,
                project=self.mgr.wandb_project,
                group=self.mgr.model_name,
                config=mgr_config
            )

    def _get_model_outputs(self, model, data_dict):
        inputs = data_dict["image"].to(self.device)
        targets_dict = {
            k: v.to(self.device)
            for k, v in data_dict.items()
            if k not in ["image", "patch_info", "is_unlabeled"]
        }
        
        outputs = model(inputs)
        
        return inputs, targets_dict, outputs

    def _train_step(self, model, data_dict, loss_fns, use_amp, autocast_ctx, epoch, step, verbose=False,
                    scaler=None, optimizer=None, num_iters=None, grad_accumulate_n=1):
        """Execute a single training step including gradient updates."""
        global_step = step

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

        with autocast_ctx:
            inputs, targets_dict, outputs = self._get_model_outputs(model, data_dict)
            total_loss, task_losses = self._compute_train_loss(outputs, targets_dict, loss_fns)

        # Handle gradient accumulation, clipping, and optimizer step
        # Scale loss by accumulation steps to maintain same effective batch size
        scaled_loss = total_loss / grad_accumulate_n

        # backward
        scaler.scale(scaled_loss).backward()

        optimizer_stepped = False
        if (step + 1) % grad_accumulate_n == 0 or (step + 1) == num_iters:
            scaler.unscale_(optimizer)
            grad_clip = getattr(self.mgr, 'gradient_clip', 12.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = True

        return total_loss, task_losses, inputs, targets_dict, outputs, optimizer_stepped

    def _compute_train_loss(self, outputs, targets_dict, loss_fns):
        total_loss = 0.0
        task_losses = {}

        for t_name, t_gt in targets_dict.items():
            if t_name == 'skel' or t_name.endswith('_skel') or t_name == 'is_unlabeled':
                continue  # Skip skeleton data and metadata as they're not predicted outputs
            t_pred = outputs[t_name]
            task_loss_fns = loss_fns[t_name]  # List of (loss_fn, weight) tuples
            task_weight = self.mgr.targets[t_name].get("weight", 1.0)

            task_total_loss = 0.0
            for loss_fn, loss_weight in task_loss_fns:
                # this naming is extremely confusing, i know. we route all loss through the aux helper
                # because it just simplifies adding addtl losses for aux tasks if present.
                # TODO: rework this janky setup to make it more clear
                # Get skeleton data if available
                skeleton_data = targets_dict.get(f'{t_name}_skel', None)
                loss_value = compute_auxiliary_loss(loss_fn, t_pred, t_gt, outputs,
                                                    self.mgr.targets[t_name], skeleton_data)
                task_total_loss += loss_weight * loss_value

            weighted_loss = task_weight * task_total_loss
            total_loss += weighted_loss

            # Store the actual loss value (after task weighting but before grad accumulation scaling)
            task_losses[t_name] = task_total_loss.detach().cpu().item()

        return total_loss, task_losses


    def _validation_step(self, model, data_dict, loss_fns, use_amp):
        inputs = data_dict["image"].to(self.device)
        targets_dict = {
            k: v.to(self.device)
            for k, v in data_dict.items()
            if k not in ["image", "patch_info", "is_unlabeled"]
        }

        if use_amp:
            if self.device.type == 'cuda':
                if torch.cuda.is_bf16_supported():
                    context = torch.amp.autocast('cuda', dtype=torch.bfloat16)
                else:
                    context = torch.amp.autocast('cuda')
            else:
                context = torch.amp.autocast(self.device.type)
            
        else:
            context = nullcontext()

        with context:
            outputs = model(inputs)
            task_losses = self._compute_validation_loss(outputs, targets_dict, loss_fns)
        
        return task_losses, inputs, targets_dict, outputs

    def _compute_validation_loss(self, outputs, targets_dict, loss_fns):
        task_losses = {}

        for t_name, t_gt in targets_dict.items():
            if t_name == 'skel' or t_name.endswith('_skel') or t_name == 'is_unlabeled':
                continue  # Skip skeleton data and metadata as they're not predicted outputs
            t_pred = outputs[t_name]
            task_loss_fns = loss_fns[t_name]  # List of (loss_fn, weight) tuples

            task_total_loss = 0.0
            for loss_fn, loss_weight in task_loss_fns:
                # Get skeleton data if available
                skeleton_data = targets_dict.get(f'{t_name}_skel', None)
                loss_value = compute_auxiliary_loss(loss_fn, t_pred, t_gt, outputs,
                                                    self.mgr.targets[t_name], skeleton_data)
                task_total_loss += loss_weight * loss_value

            task_losses[t_name] = task_total_loss.detach().cpu().item()

        return task_losses

    def _on_epoch_end(self, epoch, model, optimizer, scheduler, train_dataset,
                       ckpt_dir, model_ckpt_dir, checkpoint_history, best_checkpoints,
                       avg_val_loss):
        """Handle end-of-epoch operations: checkpointing, cleanup, etc."""
        ckpt_path = os.path.join(
            ckpt_dir,
            f"{self.mgr.model_name}_epoch{epoch}.pth"
        )

        checkpoint_data = save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            checkpoint_path=ckpt_path,
            model_config=model.final_config,
            train_dataset=train_dataset
        )

        checkpoint_history.append((epoch, ckpt_path))

        del checkpoint_data
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Manage checkpoint history
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

        cleanup_old_configs(
            model_ckpt_dir=model_ckpt_dir,
            model_name=self.mgr.model_name,
            keep_latest=1
        )

        return checkpoint_history, best_checkpoints, ckpt_path

    def _prepare_metrics_for_logging(self, epoch, step, epoch_losses, current_lr=None, val_losses=None):

        # this is a separate method just so i have an easy way to accumulate metrics
        # TODO: make this easier

        metrics = {"epoch": epoch, "step": step}

        # Add training losses
        for t_name in self.mgr.targets:
            if t_name in epoch_losses and len(epoch_losses[t_name]) > 0:
                # Use recent average for training losses
                metrics[f"train_loss_{t_name}"] = np.mean(epoch_losses[t_name][-100:])

        # Add total training loss
        if epoch_losses:
            recent_losses = [np.mean(losses[-100:]) for losses in epoch_losses.values() if len(losses) > 0]
            if recent_losses:
                metrics["train_loss_total"] = np.mean(recent_losses)

        # Add learning rate if provided
        if current_lr is not None:
            metrics["learning_rate"] = current_lr

        # Add validation losses if provided
        if val_losses is not None:
            total_val_loss = 0.0
            for t_name in self.mgr.targets:
                if t_name in val_losses and len(val_losses[t_name]) > 0:
                    val_avg = np.mean(val_losses[t_name])
                    metrics[f"val_loss_{t_name}"] = val_avg
                    total_val_loss += val_avg

            # Add total validation loss
            if self.mgr.targets:
                metrics["val_loss_total"] = total_val_loss / len(self.mgr.targets)

        return metrics

    def train(self):

        training_state = self._initialize_training()

        # Unpack the state
        model = training_state['model']
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

        self._initialize_wandb(train_dataset, val_dataset, train_indices, val_indices)

        val_loss_history = {}  # {epoch: validation_loss}
        checkpoint_history = deque(maxlen=3)
        best_checkpoints = []
        debug_gif_history = deque(maxlen=3)
        best_debug_gifs = []  # List of (val_loss, epoch, gif_path)

        global_step = 0
        grad_accumulate_n = self.mgr.gradient_accumulation

        early_stopping_patience = getattr(self.mgr, 'early_stopping_patience', 20)
        if early_stopping_patience > 0:
            best_val_loss = float('inf')
            patience_counter = 0
            print(f"Early stopping enabled with patience: {early_stopping_patience} epochs")
        else:
            print("Early stopping disabled")

        # ---- training! ----- #
        for epoch in range(start_epoch, self.mgr.max_epoch):
            # Update loss functions for this epoch
            loss_fns = self._update_loss_for_epoch(loss_fns, epoch)
            
            model.train()

            if getattr(self.mgr, 'max_steps_per_epoch', None) is not None and self.mgr.max_steps_per_epoch > 0:
                num_iters = min(len(train_dataloader), self.mgr.max_steps_per_epoch)
            else:
                num_iters = len(train_dataloader)

            epoch_losses = {t_name: [] for t_name in self.mgr.targets}
            train_iter = iter(train_dataloader)
            pbar = tqdm(range(num_iters), desc=f'Epoch {epoch + 1}/{self.mgr.max_epoch}')
            
            # Variables to store train samples for debug visualization
            train_sample_input = None
            train_sample_targets = None
            train_sample_outputs = None

            print(f"Using optimizer : {optimizer.__class__.__name__}")
            print(f"Using scheduler : {scheduler.__class__.__name__} (per-iteration: {is_per_iteration_scheduler})")
            print(f"Initial learning rate : {self.mgr.initial_lr}")
            print(f"Gradient accumulation steps : {grad_accumulate_n}")

            for i in pbar:
                if i % grad_accumulate_n == 0:
                    optimizer.zero_grad(set_to_none=True)

                data_dict = next(train_iter)
                global_step += 1
                
                # Setup autocast context
                if use_amp and self.device.type == 'cuda':
                    if torch.cuda.is_bf16_supported():
                        autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)
                    else:
                        autocast_ctx = torch.amp.autocast('cuda')
                elif use_amp and self.device.type in ['cpu', 'mlx']:
                    autocast_ctx = torch.amp.autocast(self.device.type)
                else:
                    autocast_ctx = nullcontext()

                # Execute training step
                total_loss, task_losses, inputs, targets_dict, outputs, optimizer_stepped = self._train_step(
                    model=model,
                    data_dict=data_dict,
                    loss_fns=loss_fns,
                    use_amp=use_amp,
                    autocast_ctx=autocast_ctx,
                    epoch=epoch,
                    step=i,
                    verbose=self.mgr.verbose,
                    scaler=scaler,
                    optimizer=optimizer,
                    num_iters=num_iters,
                    grad_accumulate_n=grad_accumulate_n
                )

                for t_name, loss_value in task_losses.items():
                    epoch_losses[t_name].append(loss_value)
                

                if i == 0 and train_sample_input is None:
                    # Find first sample with non-zero target
                    first_target_key = list(targets_dict.keys())[0]
                    first_target = targets_dict[first_target_key]
                    
                    b_idx = 0
                    found_non_zero = False
                    for b in range(first_target.shape[0]):
                        if torch.any(first_target[b] != 0):
                            b_idx = b
                            found_non_zero = True
                            break
                    
                    if found_non_zero:
                        train_sample_input = inputs[b_idx: b_idx + 1]
                        # First collect all targets including skel
                        train_sample_targets_all = {}
                        for t_name, t_tensor in targets_dict.items():
                            train_sample_targets_all[t_name] = t_tensor[b_idx: b_idx + 1]
                        # Now create train_sample_targets without skel for save_debug
                        train_sample_targets = {}
                        for t_name, t_tensor in train_sample_targets_all.items():
                            if t_name != 'skel':
                                train_sample_targets[t_name] = t_tensor
                        train_sample_outputs = {}
                        for t_name, p_tensor in outputs.items():
                            train_sample_outputs[t_name] = p_tensor[b_idx: b_idx + 1]

                if optimizer_stepped and is_per_iteration_scheduler:
                    scheduler.step()

                loss_str = " | ".join([f"{t}: {np.mean(epoch_losses[t][-100:]):.4f}"
                                       for t in self.mgr.targets if len(epoch_losses[t]) > 0])
                pbar.set_postfix_str(loss_str)

                current_lr = optimizer.param_groups[0]['lr']

                # Log metrics to wandb once per step
                if self.mgr.wandb_project:
                    metrics = self._prepare_metrics_for_logging(
                        epoch=epoch,
                        step=global_step,
                        epoch_losses=epoch_losses,
                        current_lr=current_lr
                    )
                    import wandb
                    wandb.log(metrics)

                del data_dict, inputs, targets_dict, outputs

            if not is_per_iteration_scheduler:
                scheduler.step()

            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            print(f"\n[Train] Epoch {epoch + 1} completed.")
            for t_name in self.mgr.targets:
                avg_loss = np.mean(epoch_losses[t_name]) if epoch_losses[t_name] else 0
                print(f"  {t_name}: Avg Loss = {avg_loss:.4f}")

            # ---- validation ----- #
            if epoch % 1 == 0:
                model.eval()
                with torch.no_grad():
                    val_losses = {t_name: [] for t_name in self.mgr.targets}
                    frames_array = None
                    
                    # Initialize evaluation metrics
                    evaluation_metrics = self._initialize_evaluation_metrics()

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

                        task_losses, inputs, targets_dict, outputs = self._validation_step(
                            model=model,
                            data_dict=data_dict,
                            loss_fns=loss_fns,
                            use_amp=use_amp
                        )

                        for t_name, loss_value in task_losses.items():
                            val_losses[t_name].append(loss_value)
                        
                        # Compute evaluation metrics for each task
                        for t_name in self.mgr.targets:
                            if t_name in outputs and t_name in targets_dict:
                                for metric in evaluation_metrics[t_name]:
                                    if isinstance(metric, CriticalComponentsMetric) and i >= 10:
                                        continue
                                    metric.update(pred=outputs[t_name], gt=targets_dict[t_name])

                        if i == 0:
                                # Find first non-zero sample for debug visualization
                                b_idx = 0
                                found_non_zero = False

                                # Check if the first sample is non-zero
                                first_target = next(iter(targets_dict.values()))
                                if torch.any(first_target[0] != 0):
                                    found_non_zero = True
                                else:
                                    # Look for a non-zero sample
                                    for b in range(first_target.shape[0]):
                                        if torch.any(first_target[b] != 0):
                                            b_idx = b
                                            found_non_zero = True
                                            break

                                if found_non_zero:
                                    # Slicing shape: [1, c, z, y, x ]
                                    inputs_first = inputs[b_idx: b_idx + 1]

                                    targets_dict_first_all = {}
                                    for t_name, t_tensor in targets_dict.items():
                                        targets_dict_first_all[t_name] = t_tensor[b_idx: b_idx + 1]

                                    outputs_dict_first = {}
                                    for t_name, p_tensor in outputs.items():
                                        outputs_dict_first[t_name] = p_tensor[b_idx: b_idx + 1]

                                    debug_img_path = f"{ckpt_dir}/{self.mgr.model_name}_debug_epoch{epoch}.gif"
                                    
                                    # Extract skeleton data if using SkeletonRecallTrainer
                                    skeleton_dict = None
                                    train_skeleton_dict = None
                                    # Check if skeleton data is available in the batch
                                    if 'skel' in targets_dict_first_all:
                                        skeleton_dict = {'segmentation': targets_dict_first_all.get('skel')}
                                    # Check if train_sample_targets_all exists (from earlier training step)
                                    if 'train_sample_targets_all' in locals() and train_sample_targets_all and 'skel' in train_sample_targets_all:
                                        train_skeleton_dict = {'segmentation': train_sample_targets_all.get('skel')}
                                    
                                    targets_dict_first = {}
                                    for t_name, t_tensor in targets_dict_first_all.items():
                                        if t_name != 'skel':
                                            targets_dict_first[t_name] = t_tensor
                                    
                                    frames_array = save_debug(
                                        input_volume=inputs_first,
                                        targets_dict=targets_dict_first,
                                        outputs_dict=outputs_dict_first,
                                        tasks_dict=self.mgr.targets,
                                        # dictionary, e.g. {"sheet": {"activation":"sigmoid"}, "normals": {"activation":"none"}}
                                        epoch=epoch,
                                        save_path=debug_img_path,
                                        train_input=train_sample_input,
                                        train_targets_dict=train_sample_targets,
                                        train_outputs_dict=train_sample_outputs,
                                        skeleton_dict=skeleton_dict,
                                        train_skeleton_dict=train_skeleton_dict
                                    )
                                    debug_gif_history.append((epoch, debug_img_path))

                        loss_str = " | ".join([f"{t}: {np.mean(val_losses[t]):.4f}"
                                               for t in self.mgr.targets if len(val_losses[t]) > 0])
                        val_pbar.set_postfix_str(loss_str)

                        del outputs, inputs, targets_dict

                    print(f"\n[Validation] Epoch {epoch + 1} summary:")
                    total_val_loss = 0.0
                    for t_name in self.mgr.targets:
                        val_avg = np.mean(val_losses[t_name]) if val_losses[t_name] else 0
                        print(f"  Task '{t_name}': Avg validation loss = {val_avg:.4f}")
                        total_val_loss += val_avg

                    avg_val_loss = total_val_loss / len(self.mgr.targets) if self.mgr.targets else 0
                    val_loss_history[epoch] = avg_val_loss
                    
                    print("\n[Validation Metrics]")
                    metric_results = {}
                    for t_name in self.mgr.targets:
                        if t_name in evaluation_metrics:
                            print(f"  Task '{t_name}':")
                            for metric in evaluation_metrics[t_name]:
                                aggregated = metric.aggregate()
                                for metric_name, value in aggregated.items():
                                    full_metric_name = f"{t_name}_{metric_name}"
                                    metric_results[full_metric_name] = value
                                    display_name = f"{metric.name}_{metric_name}"
                                    print(f"    {display_name}: {value:.4f}")

                    if self.mgr.wandb_project:
                        val_metrics = {"epoch": epoch, "step": global_step}
                        for t_name in self.mgr.targets:
                            if t_name in val_losses and len(val_losses[t_name]) > 0:
                                val_metrics[f"val_loss_{t_name}"] = np.mean(val_losses[t_name])
                        val_metrics["val_loss_total"] = avg_val_loss
                        
                        # Add evaluation metrics to wandb
                        for metric_name, value in metric_results.items():
                            val_metrics[f"val_{metric_name}"] = value

                        if 'frames_array' in locals() and frames_array is not None:
                            import wandb
                            # Stack the list of frames into a proper numpy array (T, H, W, C)
                            frames_np = np.stack(frames_array, axis=0)
                            # Convert BGR to RGB for wandb
                            frames_np = frames_np[..., ::-1]
                            # Transpose to (frames, channels, height, width) as required by wandb
                            frames_np = np.transpose(frames_np, (0, 3, 1, 2))
                            
                            val_metrics["debug_gif"] = wandb.Video(frames_np, format="gif")

                        import wandb
                        wandb.log(val_metrics)

                    # Early stopping check
                    if early_stopping_patience > 0:
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            patience_counter = 0
                            print(f"[Early Stopping] New best validation loss: {best_val_loss:.4f}")
                        else:
                            patience_counter += 1
                            print(f"[Early Stopping] No improvement for {patience_counter}/{early_stopping_patience} epochs")

                        if patience_counter >= early_stopping_patience:
                            print(f"\n[Early Stopping] Validation loss did not improve for {early_stopping_patience} epochs.")
                            print(f"Best validation loss: {best_val_loss:.4f}")
                            print("Stopping training early.")
                            break
                    
                    # Handle epoch end operations (checkpointing, cleanup)
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

                    # Manage debug GIFs
                    if epoch in [e for e, _ in debug_gif_history]:
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

def main():
    """Main entry point for the training script."""
    import argparse
    import ast

    parser = argparse.ArgumentParser(
        description="Train Vesuvius neural networks for ink detection and segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("-i", "--input", required=True,
                        help="Input directory containing images/ and labels/ subdirectories.")
    parser.add_argument("-o", "--output", default="checkpoints",
                        help="Output directory for saving checkpoints and configurations (default: checkpoints)")
    parser.add_argument("--format", choices=["image", "zarr", "napari"],
                        help="Data format (image: tif, png, or jpg files, zarr: Zarr arrays, napari: Napari layers). If not specified, will attempt to auto-detect.")

    # Optional arguments
    parser.add_argument("--batch-size", type=int,
                        help="Training batch size (default: from config or 2)")
    parser.add_argument("--patch-size", type=str,
                        help="Patch size as comma-separated values, e.g., '192,192,192' for 3D or '256,256' for 2D")
    parser.add_argument("--loss", type=str,
                        help="Loss functions as a list, e.g., '[SoftDiceLoss, BCEWithLogitsLoss]' or comma-separated")
    parser.add_argument("--train-split", type=float,
                        help="Training/validation split ratio (0.0-1.0, default: 0.95)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/validation split (default: 42)")
    parser.add_argument("--config", "--config-path", dest="config_path", type=str, required=True,
                        help="Path to configuration YAML file (required)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output for debugging")
    parser.add_argument("--max-epoch", type=int, default=1000,
                        help="Maximum number of epochs (default: 1000)")
    parser.add_argument("--max-steps-per-epoch", type=int, default=200,
                        help="Maximum training steps per epoch (if not set, uses all data)")
    parser.add_argument("--max-val-steps-per-epoch", type=int, default=30,
                        help="Maximum validation steps per epoch (if not set, uses all data)")
    parser.add_argument("--full-epoch", action="store_true",
                        help="Iterate over entire train and validation datasets once per epoch (overrides max-steps-per-epoch)")
    parser.add_argument("--model-name", type=str,
                        help="Model name for checkpoints and logging (default: from config or 'Model')")
    parser.add_argument("--nonlin", type=str, choices=["LeakyReLU", "ReLU", "SwiGLU", "swiglu", "GLU", "glu"],
                        help="Activation function to use in the model (default: from config or 'LeakyReLU')")
    parser.add_argument("--se", action="store_true", help="Enable squeeze and excitation modules in the encoder")
    parser.add_argument("--se-reduction-ratio", type=float, default=0.0625,
                        help="Squeeze excitation reduction ratio (default: 0.0625 = 1/16)")
    parser.add_argument("--pool-type", type=str, choices=["avg", "max", "conv"],
                        help="Type of pooling to use in encoder ('avg', 'max', or 'conv' for strided convolutions). Default: 'conv'")
    parser.add_argument("--optimizer", type=str,
                        help="Optimizer to use for training (default: from config or 'AdamW, available options in models/optimizers.py')")
    parser.add_argument("--no-spatial", action="store_true",
                        help="Disable spatial/geometric transformations (rotations, flips, etc.) during training")
    parser.add_argument("--grad-clip", type=float, default=12.0,
                        help="Gradient clipping value (default: 12.0)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable Automatic Mixed Precision (AMP) for training")
    parser.add_argument("--skip-intensity-sampling", dest="skip_intensity_sampling", 
                        action="store_true", default=True,
                        help="Skip intensity sampling during dataset initialization (default: True)")
    parser.add_argument("--no-skip-intensity-sampling", dest="skip_intensity_sampling",
                        action="store_false",
                        help="Enable intensity sampling during dataset initialization")
    parser.add_argument("--early-stopping-patience", type=int, default=20,
                        help="Number of epochs to wait for validation loss improvement before early stopping (default: 5, set to 0 to disable)")

    # Trainer selection
    parser.add_argument("--trainer", type=str, default="base",
                        help="Trainer class to use (default: base). Options: base, uncertainty_aware_mean_teacher")

    # Learning rate scheduler arguments
    parser.add_argument("--scheduler", type=str,
                        help="Learning rate scheduler type (default: from config or 'poly')")
    parser.add_argument("--warmup-steps", type=int,
                        help="Number of warmup steps for cosine_warmup scheduler (default: 10%% of first cycle)")

    # Weights & Biases arguments
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Weights & Biases project name (default: from config; wandb logging disabled if not set anywhere)")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="Weights & Biases team/username (default: from config)")

    args = parser.parse_args()

    # Load configuration first to check if we have data_paths
    from vesuvius.models.configuration.config_manager import ConfigManager
    mgr = ConfigManager(verbose=args.verbose)

    if not Path(args.config_path).exists():
        print(f"\nError: Config file does not exist: {args.config_path}")
        print("\nPlease provide a valid configuration file.")
        print("\nExample usage:")
        print("  vesuvius.train --config path/to/config.yaml --input path/to/data --output path/to/output")
        print("\nFor more options, use: vesuvius.train --help")
        sys.exit(1)

    mgr.load_config(args.config_path)
    print(f"Loaded configuration from: {args.config_path}")

    # Check if input exists
    if not Path(args.input).exists():
        raise ValueError(f"Input directory does not exist: {args.input}")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    update_config_from_args(mgr, args)

    # Select trainer based on --trainer argument
    trainer_name = args.trainer.lower()
    mgr.trainer_class = trainer_name  # Store trainer class in config manager
    
    if trainer_name == "uncertainty_aware_mean_teacher":
        mgr.allow_unlabeled_data = True
        from vesuvius.models.training.trainers.train_uncertainty_aware_mean_teacher import UncertaintyAwareMeanTeacher3DTrainer
        trainer = UncertaintyAwareMeanTeacher3DTrainer(mgr=mgr, verbose=args.verbose)
        print("Using Uncertainty-Aware Mean Teacher Trainer for semi-supervised 3D training")
    elif trainer_name == "base":
        trainer = BaseTrainer(mgr=mgr, verbose=args.verbose)
        print("Using Base Trainer for supervised training")
    else:
        raise ValueError(f"Unknown trainer: {trainer_name}. Available options: base, uncertainty_aware_mean_teacher")

    print("Starting training...")
    trainer.train()
    print("Training completed!")


if __name__ == '__main__':
    main()
