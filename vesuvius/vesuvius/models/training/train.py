import multiprocessing
import sys

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
from torch.optim.lr_scheduler import CosineAnnealingLR
from vesuvius.models.training.lr_schedulers import get_scheduler, PolyLRScheduler
from torch.optim import AdamW, SGD
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

from itertools import cycle
from contextlib import nullcontext
from collections import deque
import gc

from vesuvius.models.utilities.load_checkpoint import load_checkpoint
from vesuvius.models.utilities.get_accelerator import get_accelerator
from vesuvius.models.utilities.compute_gradient_norm import compute_gradient_norm
from vesuvius.models.utilities.s3_utils import detect_s3_paths, setup_multiprocessing_for_s3
from vesuvius.models.training.auxiliary_tasks import (
    compute_auxiliary_loss,
    preserve_auxiliary_targets,
    restore_auxiliary_targets,
    apply_auxiliary_tasks_from_config
)


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
        for task_name, task_info in self.mgr.targets.items():
            task_losses = []

            if "losses" in task_info:
                print(f"Target {task_name} using multiple losses:")
                for loss_cfg in task_info["losses"]:
                    loss_name = loss_cfg["name"]
                    loss_weight = loss_cfg.get("weight", 1.0)
                    loss_kwargs = loss_cfg.get("kwargs", {})

                    weight = loss_kwargs.get("weight", None)
                    ignore_index = loss_kwargs.get("ignore_index", -100)
                    pos_weight = loss_kwargs.get("pos_weight", None)

                    if hasattr(self.mgr,
                               'compute_loss_on_labeled_only') and self.mgr.compute_loss_on_labeled_only and ignore_index is None:
                        ignore_index = -100
                        print(f"  Setting ignore_index=-100 for {loss_name} due to compute_loss_on_labeled_only=True")

                    try:
                        loss_fn = _create_loss(
                            name=loss_name,
                            loss_config=loss_kwargs,
                            weight=weight,
                            ignore_index=ignore_index,
                            pos_weight=pos_weight
                        )
                        task_losses.append((loss_fn, loss_weight))
                        print(f"  - {loss_name} (weight: {loss_weight})")
                    except RuntimeError as e:
                        raise ValueError(
                            f"Failed to create loss function '{loss_name}' for target '{task_name}': {str(e)}")

            loss_fns[task_name] = task_losses

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
    def _get_scaler(self, device_type='cuda', use_amp=True):
        # for cuda, we can use a grad scaler for mixed precision training if amp is enabled
        # for mps or cpu, or when amp is disabled, we create a dummy scaler that does nothing
        if device_type == 'cuda' and use_amp:
            return torch.cuda.amp.GradScaler()
        else:
            class DummyScaler:
                def scale(self, loss):
                    return loss

                def unscale_(self, optimizer):
                    pass

                def step(self, optimizer):
                    optimizer.step()

                def update(self):
                    pass

            return DummyScaler()

    # --- dataloaders --- #
    def _configure_dataloaders(self, train_dataset, val_dataset=None):

        if val_dataset is None:
            val_dataset = train_dataset

        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
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
    
    def _save_train_val_filenames(self, train_dataset, val_dataset, train_indices, val_indices, ckpt_dir):
        """
        Save the filenames of the volumes used in training and validation sets along with patch locations.
        
        Parameters
        ----------
        train_dataset : Dataset
            Training dataset
        val_dataset : Dataset  
            Validation dataset
        train_indices : list
            Indices used for training patches
        val_indices : list
            Indices used for validation patches
        ckpt_dir : str
            Directory to save the split information
        """
        import json
        
        # Extract volume information and patch locations from datasets
        train_patches = []
        val_patches = []
        train_volumes = set()
        val_volumes = set()
        
        # Get volume IDs and patch locations for training set
        for idx in train_indices:
            patch_info = train_dataset.valid_patches[idx]
            vol_idx = patch_info["volume_index"]
            position = patch_info["position"]  # [z, y, x] coordinates
            
            # Get volume ID if available
            volume_id = f"volume_{vol_idx}"  # Default if no volume_ids
            if hasattr(train_dataset, 'volume_ids'):
                first_target = list(train_dataset.volume_ids.keys())[0]
                if vol_idx < len(train_dataset.volume_ids[first_target]):
                    volume_id = train_dataset.volume_ids[first_target][vol_idx]
            
            train_volumes.add(volume_id)
            train_patches.append({
                "patch_index": idx,
                "volume_id": volume_id,
                "volume_index": vol_idx,
                "position": position  # [z, y, x] coordinates
            })
        
        # Get volume IDs and patch locations for validation set
        for idx in val_indices:
            patch_info = val_dataset.valid_patches[idx]
            vol_idx = patch_info["volume_index"]
            position = patch_info["position"]  # [z, y, x] coordinates
            
            # Get volume ID if available
            volume_id = f"volume_{vol_idx}"  # Default if no volume_ids
            if hasattr(val_dataset, 'volume_ids'):
                first_target = list(val_dataset.volume_ids.keys())[0]
                if vol_idx < len(val_dataset.volume_ids[first_target]):
                    volume_id = val_dataset.volume_ids[first_target][vol_idx]
            
            val_volumes.add(volume_id)
            val_patches.append({
                "patch_index": idx,
                "volume_id": volume_id,
                "volume_index": vol_idx,
                "position": position  # [z, y, x] coordinates
            })
        
        # Save split information with patch details
        split_info = {
            "metadata": {
                "train_patch_count": len(train_indices),
                "val_patch_count": len(val_indices),
                "train_volume_count": len(train_volumes),
                "val_volume_count": len(val_volumes),
                "train_volumes": sorted(list(train_volumes)),
                "val_volumes": sorted(list(val_volumes)),
                "train_val_split": self.mgr.tr_val_split,
                "patch_size": self.mgr.train_patch_size,
                "timestamp": datetime.now().isoformat()
            },
            "train_patches": train_patches,
            "val_patches": val_patches
        }
        
        # Save to JSON file
        split_file = os.path.join(ckpt_dir, "train_val_split.json")
        with open(split_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"\nSaved train/validation split information to: {split_file}")
        print(f"Training volumes ({len(train_volumes)}): {sorted(list(train_volumes))}")
        print(f"Validation volumes ({len(val_volumes)}): {sorted(list(val_volumes))}")
        print(f"Training patches: {len(train_indices)} with locations saved")
        print(f"Validation patches: {len(val_indices)} with locations saved")
        
        # Log to wandb if enabled
        if self.mgr.wandb_project:
            import wandb
            wandb.log({
                "train_volumes": len(train_volumes),
                "val_volumes": len(val_volumes),
                "train_patches": len(train_indices),
                "val_patches": len(val_indices),
                "train_val_split_ratio": self.mgr.tr_val_split
            })

    def train(self):
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

        if not hasattr(torch, 'no_op'):
            class NullContextManager:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass

            torch.no_op = lambda: NullContextManager()

        use_amp = not getattr(self.mgr, 'no_amp', False)
        if not use_amp:
            print("Automatic Mixed Precision (AMP) is disabled")

        scaler = self._get_scaler(self.device.type, use_amp=use_amp)
        train_dataloader, val_dataloader, train_indices, val_indices = self._configure_dataloaders(train_dataset,
                                                                                                   val_dataset)

        # Initialise wandb if wandb_project is set
        if self.mgr.wandb_project:
            import wandb  # lazy import in case it's not available
            
            # Prepare config manager parameters
            mgr_config = self.mgr.convert_to_dict()
            
            # Initialize wandb with config manager parameters
            wandb.init(
                entity=self.mgr.wandb_entity,
                project=self.mgr.wandb_project,
                group=self.mgr.model_name,
                config=mgr_config
            )
            
            # Log the final model configuration after model is built
            if hasattr(model, 'final_config'):
                wandb.config.update({"model_final_config": model.final_config})
                
            # Log model architecture summary
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            wandb.config.update({
                "model_total_parameters": total_params,
                "model_trainable_parameters": trainable_params,
                "model_architecture": str(model.__class__.__name__)
            })
            
            # Save configs as artifacts for better tracking
            artifact = wandb.Artifact(
                name=f"{self.mgr.model_name}_configs",
                type="model_configs",
                description="Model and training configurations"
            )
            
            # Save config manager parameters as JSON
            import json
            with artifact.new_file("config_manager.json") as f:
                json.dump(mgr_config, f, indent=2)
                
            # Save final model config if available
            if hasattr(model, 'final_config'):
                with artifact.new_file("model_final_config.json") as f:
                    json.dump(model.final_config, f, indent=2)
                    
            # Log the artifact
            wandb.log_artifact(artifact)
            
            # Create a summary table with key parameters
            summary_table = wandb.Table(columns=["Parameter", "Value"])
            summary_table.add_data("Model Name", self.mgr.model_name)
            summary_table.add_data("Model Architecture", str(model.__class__.__name__))
            summary_table.add_data("Total Parameters", f"{total_params:,}")
            summary_table.add_data("Trainable Parameters", f"{trainable_params:,}")
            summary_table.add_data("Batch Size", self.mgr.train_batch_size)
            summary_table.add_data("Patch Size", str(self.mgr.train_patch_size))
            summary_table.add_data("Learning Rate", self.mgr.initial_lr)
            summary_table.add_data("Optimizer", optimizer.__class__.__name__)
            summary_table.add_data("Scheduler", scheduler.__class__.__name__)
            summary_table.add_data("Max Epochs", self.mgr.max_epoch)
            summary_table.add_data("Train/Val Split", self.mgr.tr_val_split)
            summary_table.add_data("Gradient Accumulation", grad_accumulate_n)
            summary_table.add_data("Mixed Precision", "Enabled" if use_amp else "Disabled")
            summary_table.add_data("Device", str(self.device))
            summary_table.add_data("Number of Targets", len(self.mgr.targets))
            summary_table.add_data("Targets", ", ".join(self.mgr.targets.keys()))
            
            wandb.log({"training_configuration": summary_table})

        start_epoch = 0

        # track the validation loss so we can save the best checkpoints
        val_loss_history = {}  # {epoch: validation_loss}
        checkpoint_history = deque(maxlen=3)
        best_checkpoints = []
        debug_gif_history = deque(maxlen=3)
        best_debug_gifs = []  # List of (val_loss, epoch, gif_path)

        os.makedirs(self.mgr.ckpt_out_base, exist_ok=True)
        model_ckpt_dir = os.path.join(self.mgr.ckpt_out_base, self.mgr.model_name)
        os.makedirs(model_ckpt_dir, exist_ok=True)

        now = datetime.now()
        date_str = now.strftime('%m%d%y')
        time_str = now.strftime('%H%M')
        ckpt_dir = os.path.join('checkpoints', f"{self.mgr.model_name}_{date_str}{time_str}")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Save train/validation filenames
        self._save_train_val_filenames(train_dataset, val_dataset, train_indices, val_indices, ckpt_dir)

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
        else:
            start_epoch = 0

        global_step = 0
        grad_accumulate_n = self.mgr.gradient_accumulation

        # ---- training! ----- #
        for epoch in range(start_epoch, self.mgr.max_epoch):
            model.train()

            if getattr(self.mgr, 'max_steps_per_epoch', None) and self.mgr.max_steps_per_epoch > 0:
                num_iters = min(len(train_dataloader), self.mgr.max_steps_per_epoch)
            else:
                num_iters = len(train_dataloader)

            epoch_losses = {t_name: [] for t_name in self.mgr.targets}
            train_iter = iter(train_dataloader)
            pbar = tqdm(range(num_iters), desc=f'Epoch {epoch + 1}/{self.mgr.max_epoch}')

            print(f"Using optimizer : {optimizer.__class__.__name__}")
            print(f"Using scheduler : {scheduler.__class__.__name__} (per-iteration: {is_per_iteration_scheduler})")
            print(f"Initial learning rate : {self.mgr.initial_lr}")
            print(f"Gradient accumulation steps : {grad_accumulate_n}")

            for i in pbar:
                if i % grad_accumulate_n == 0:
                    optimizer.zero_grad(set_to_none=True)

                data_dict = next(train_iter)

                if epoch == 0 and i == 0 and self.mgr.verbose:
                    print("Items from the first batch -- Double check that your shapes and values are expected:")
                    for item, val in data_dict.items():
                        if isinstance(val, dict):
                            print(f"{item}: (dictionary with keys: {list(val.keys())})")
                            for sub_key, sub_val in val.items():
                                print(
                                    f"  {sub_key}: {sub_val.dtype}, {sub_val.shape}, min {sub_val.min()} max {sub_val.max()}")
                        else:
                            print(f"{item}: {val.dtype}, {val.shape}, min {val.min()} max {val.max()}")

                global_step += 1

                inputs = data_dict["image"].to(self.device, dtype=torch.float32)
                ignore_masks = None
                if "ignore_masks" in data_dict:
                    ignore_masks = {t_name: mask.to(self.device) for t_name, mask in data_dict["ignore_masks"].items()}

                targets_dict = {
                    k: v.to(self.device, dtype=torch.float32)
                    for k, v in data_dict.items()
                    if k not in ["image", "ignore_masks", "patch_info"]
                }

                if use_amp and self.device.type in ['cuda', 'cpu']:
                    autocast_ctx = torch.amp.autocast(self.device.type)
                else:
                    autocast_ctx = nullcontext()

                with autocast_ctx:
                    outputs = model(inputs)
                    total_loss = 0.0

                    for t_name, t_gt in targets_dict.items():
                        t_pred = outputs[t_name]
                        task_losses = loss_fns[t_name]  # List of (loss_fn, weight) tuples
                        task_weight = self.mgr.targets[t_name].get("weight", 1.0)

                        task_total_loss = 0.0
                        for loss_fn, loss_weight in task_losses:
                            t_gt_masked = t_gt
                            if ignore_masks is not None and t_name in ignore_masks:
                                ignore_mask = ignore_masks[t_name]

                                if i == 0 and task_losses.index((loss_fn, loss_weight)) == 0:
                                    print(f"Using custom ignore mask for target {t_name}")

                                ignore_label = getattr(loss_fn, 'ignore_index', -100)

                                if ignore_mask.dim() == t_gt.dim() - 1:
                                    ignore_mask = ignore_mask.unsqueeze(1)

                                ignore_tensor = torch.tensor(ignore_label, dtype=t_gt.dtype, device=t_gt.device)
                                t_gt_masked = torch.where(ignore_mask == 1, ignore_tensor, t_gt)

                            # Compute loss
                            # Use auxiliary loss computation helper
                            loss_value = compute_auxiliary_loss(loss_fn, t_pred, t_gt_masked, outputs,
                                                                self.mgr.targets[t_name])
                            task_total_loss += loss_weight * loss_value

                        weighted_loss = task_weight * task_total_loss
                        total_loss += weighted_loss

                        # Store the actual loss value (after task weighting but before grad accumulation scaling)
                        epoch_losses[t_name].append(task_total_loss.detach().cpu().item())

                    # Scale loss by accumulation steps to maintain same effective batch size
                    total_loss = total_loss / grad_accumulate_n

                if self.mgr.wandb_project:
                    wandb.log({
                        **{
                            f"loss_{t_name}": epoch_losses[t_name][-1]
                            for t_name in self.mgr.targets
                        },
                        "loss_total": total_loss.detach().cpu().item()
                    })

                # backward
                scaler.scale(total_loss).backward()

                if (i + 1) % grad_accumulate_n == 0 or (i + 1) == num_iters:
                    scaler.unscale_(optimizer)
                    grad_clip = getattr(self.mgr, 'gradient_clip', 12.0)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    if is_per_iteration_scheduler:
                        scheduler.step()

                loss_str = " | ".join([f"{t}: {np.mean(epoch_losses[t][-100:]):.4f}"
                                       for t in self.mgr.targets if len(epoch_losses[t]) > 0])
                pbar.set_postfix_str(loss_str)

                del data_dict, inputs, targets_dict, outputs
                if ignore_masks is not None:
                    del ignore_masks

            # Step per-epoch schedulers once after each epoch
            if not is_per_iteration_scheduler:
                scheduler.step()

            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            print(f"\n[Train] Epoch {epoch + 1} completed.")
            for t_name in self.mgr.targets:
                avg_loss = np.mean(epoch_losses[t_name]) if epoch_losses[t_name] else 0
                print(f"  {t_name}: Avg Loss = {avg_loss:.4f}")

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

            # ---- validation ----- #
            if epoch % 1 == 0:
                model.eval()
                with torch.no_grad():
                    val_losses = {t_name: [] for t_name in self.mgr.targets}

                    val_dataloader_iter = iter(val_dataloader)

                    if hasattr(self.mgr,
                               'max_val_steps_per_epoch') and self.mgr.max_val_steps_per_epoch and self.mgr.max_val_steps_per_epoch > 0:
                        num_val_iters = min(len(val_indices), self.mgr.max_val_steps_per_epoch)
                    else:
                        num_val_iters = len(val_indices)  # Use all data

                    val_pbar = tqdm(range(num_val_iters), desc=f'Validation {epoch + 1}')

                    for i in val_pbar:
                        try:
                            data_dict = next(val_dataloader_iter)
                        except StopIteration:
                            val_dataloader_iter = iter(val_dataloader)
                            data_dict = next(val_dataloader_iter)

                        inputs = data_dict["image"].to(self.device, dtype=torch.float32)
                        ignore_masks = None
                        if "ignore_masks" in data_dict:
                            ignore_masks = {
                                t_name: mask.to(self.device, dtype=torch.float32)
                                for t_name, mask in data_dict["ignore_masks"].items()
                            }

                        targets_dict = {
                            k: v.to(self.device, dtype=torch.float32)
                            for k, v in data_dict.items()
                            if k not in ["image", "ignore_masks", "patch_info"]
                        }

                        if use_amp:
                            context = (
                                torch.amp.autocast('cuda') if self.device.type == 'cuda'
                                else nullcontext()
                            )
                        else:
                            context = nullcontext()

                        with context:
                            outputs = model(inputs)

                            for t_name, t_gt in targets_dict.items():
                                t_pred = outputs[t_name]
                                task_losses = loss_fns[t_name]  # List of (loss_fn, weight) tuples

                                task_total_loss = 0.0
                                for loss_fn, loss_weight in task_losses:
                                    t_gt_masked = t_gt
                                    if ignore_masks is not None and t_name in ignore_masks:
                                        ignore_mask = ignore_masks[t_name].to(self.device, dtype=torch.float32)

                                        if hasattr(loss_fn, 'ignore_index'):
                                            ignore_label = loss_fn.ignore_index
                                        else:
                                            ignore_label = -100

                                        if ignore_mask.dim() == t_gt.dim() - 1:
                                            ignore_mask = ignore_mask.unsqueeze(1)

                                        # Apply mask to target: set regions where mask is 1 to ignore_label
                                        t_gt_masked = torch.where(ignore_mask == 1,
                                                                  torch.tensor(ignore_label, dtype=t_gt.dtype,
                                                                               device=self.device), t_gt)

                                    # Compute loss
                                    loss_value = loss_fn(t_pred, t_gt_masked)
                                    task_total_loss += loss_weight * loss_value

                                val_losses[t_name].append(task_total_loss.detach().cpu().item())

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

                                # Only create debug gif if we found a non-zero sample
                                if found_non_zero:
                                    # Slicing shape: [1, c, z, y, x ]
                                    inputs_first = inputs[b_idx: b_idx + 1]

                                    targets_dict_first = {}
                                    for t_name, t_tensor in targets_dict.items():
                                        targets_dict_first[t_name] = t_tensor[b_idx: b_idx + 1]

                                    outputs_dict_first = {}
                                    for t_name, p_tensor in outputs.items():
                                        outputs_dict_first[t_name] = p_tensor[b_idx: b_idx + 1]

                                    debug_img_path = f"{ckpt_dir}/{self.mgr.model_name}_debug_epoch{epoch}.gif"
                                    save_debug(
                                        input_volume=inputs_first,
                                        targets_dict=targets_dict_first,
                                        outputs_dict=outputs_dict_first,
                                        tasks_dict=self.mgr.targets,
                                        # dictionary, e.g. {"sheet": {"activation":"sigmoid"}, "normals": {"activation":"none"}}
                                        epoch=epoch,
                                        save_path=debug_img_path
                                    )
                                    debug_gif_history.append((epoch, debug_img_path))
                                    
                                    # Log debug GIF to wandb
                                    if self.mgr.wandb_project:
                                        wandb.log({"debug_gif": wandb.Image(debug_img_path)})

                            loss_str = " | ".join([f"{t}: {np.mean(val_losses[t]):.4f}"
                                                   for t in self.mgr.targets if len(val_losses[t]) > 0])

                            val_pbar.set_postfix_str(loss_str)

                            del outputs, inputs, targets_dict
                            if ignore_masks is not None:
                                del ignore_masks

                    print(f"\n[Validation] Epoch {epoch + 1} summary:")
                    total_val_loss = 0.0
                    val_loss_dict = {}
                    for t_name in self.mgr.targets:
                        val_avg = np.mean(val_losses[t_name]) if val_losses[t_name] else 0
                        print(f"  Task '{t_name}': Avg validation loss = {val_avg:.4f}")
                        total_val_loss += val_avg
                        val_loss_dict[f"val_loss_{t_name}"] = val_avg

                    # Average validation loss across all tasks
                    avg_val_loss = total_val_loss / len(self.mgr.targets) if self.mgr.targets else 0
                    val_loss_history[epoch] = avg_val_loss
                    val_loss_dict["val_loss_total"] = avg_val_loss

                    # Log validation losses to wandb
                    if self.mgr.wandb_project:
                        wandb.log(val_loss_dict)



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

                    cleanup_old_configs(
                        model_ckpt_dir=model_ckpt_dir,
                        model_name=self.mgr.model_name,
                        keep_latest=1
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


def detect_data_format(data_path):
    """
    Automatically detect the data format based on file extensions in the input directory.

    Parameters
    ----------
    data_path : Path
        Path to the data directory containing images/ and labels/ subdirectories

    Returns
    -------
    str or None
        Detected format ('zarr' or 'image') or None if cannot be determined
    """
    data_path = Path(data_path)
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"

    if not images_dir.exists():
        return None

    # Check for zarr directories and image files
    zarr_count = 0
    image_count = 0

    # Check images directory
    for item in images_dir.iterdir():
        if item.is_dir() and item.suffix == '.zarr':
            zarr_count += 1
        elif item.is_file() and item.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
            image_count += 1

    # Also check labels directory if it exists
    if labels_dir.exists():
        for item in labels_dir.iterdir():
            if item.is_dir() and item.suffix == '.zarr':
                zarr_count += 1
            elif item.is_file() and item.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
                image_count += 1

    # Determine format based on what was found
    if zarr_count > 0 and image_count == 0:
        # Only zarr files found
        return 'zarr'
    elif image_count > 0:
        # If there are any image files, it's image format
        # (even if there are zarr files too, as they may have been created during training)
        return 'image'
    else:
        # No recognized files found
        return None


def configure_targets(mgr, loss_list=None):
    """
    Detect available targets from the data directory and apply optional loss_list.
    """
    # Save existing auxiliary tasks before detection
    auxiliary_targets = preserve_auxiliary_targets(mgr.targets if hasattr(mgr, 'targets') and mgr.targets else {})

    # Detect data-based targets if not yet configured
    if not getattr(mgr, 'targets', None):
        data_path = Path(mgr.data_path)
        images_dir = data_path / "images"
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")

        targets = set()
        if mgr.data_format == "zarr":
            for d in images_dir.iterdir():
                if d.is_dir() and d.suffix == '.zarr' and '_' in d.stem:
                    targets.add(d.stem.rsplit('_', 1)[1])
        elif mgr.data_format.lower() == "image":
            for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']:
                for f in images_dir.glob(ext):
                    if '_' in f.stem:
                        targets.add(f.stem.rsplit('_', 1)[1])
        elif mgr.data_format == "napari":
            print("Warning: target detection not implemented for napari format.")
        if targets:
            mgr.targets = {}
            for t in sorted(targets):
                mgr.targets[t] = {
                    "out_channels": 2,
                    "activation": "softmax",
                    "loss_fn": "CrossEntropyLoss"
                }
            print(f"Detected targets from data: {sorted(targets)}")
        else:
            print("No targets detected from data. Please configure targets in config file.")

    # Re-add auxiliary targets
    if auxiliary_targets:
        mgr.targets = restore_auxiliary_targets(mgr.targets, auxiliary_targets)
        print(f"Re-added auxiliary targets: {list(auxiliary_targets.keys())}")

    # Re-apply auxiliary tasks from config
    apply_auxiliary_tasks_from_config(mgr)

    # Apply loss_list to configured targets, if provided
    if loss_list:
        names = list(mgr.targets.keys())
        for i, tname in enumerate(names):
            fn = loss_list[i] if i < len(loss_list) else loss_list[-1]
            mgr.targets[tname]["loss_fn"] = fn
            print(f"Applied {fn} to target '{tname}'")


def update_config_from_args(mgr, args):
    """
    Update ConfigManager with command line arguments.
    """
    # Only set data_path if input is provided
    if args.input is not None:
        mgr.data_path = Path(args.input)
        # Save data_path to dataset_config
        if not hasattr(mgr, 'dataset_config'):
            mgr.dataset_config = {}
        mgr.dataset_config["data_path"] = str(mgr.data_path)

        if args.format:
            mgr.data_format = args.format;
            print(f"Using specified data format: {mgr.data_format}")
        else:
            detected = detect_data_format(mgr.data_path)
            if detected:
                mgr.data_format = detected;
                print(f"Auto-detected data format: {mgr.data_format}")
            else:
                raise ValueError("Data format could not be determined. Please specify --format.")

        # Save data_format to dataset_config
        mgr.dataset_config["data_format"] = mgr.data_format
    else:
        # When using data_paths, we don't need data_path or data_format
        print("No input directory specified - using data_paths from config")

    # Set checkpoint output directory
    mgr.ckpt_out_base = Path(args.output)
    mgr.tr_info["ckpt_out_base"] = str(mgr.ckpt_out_base)

    # Update optional parameters if provided
    if args.batch_size is not None:
        mgr.train_batch_size = args.batch_size
        mgr.tr_configs["batch_size"] = args.batch_size

    if args.patch_size is not None:
        # Parse patch size from string like "192,192,192" or "256,256"
        try:
            patch_size = [int(x.strip()) for x in args.patch_size.split(',')]
            mgr.update_config(patch_size=patch_size)
        except ValueError as e:
            raise ValueError(
                f"Invalid patch size format: {args.patch_size}. Expected comma-separated integers like '192,192,192'")

    if args.train_split is not None:
        if not 0.0 <= args.train_split <= 1.0:
            raise ValueError(f"Train split must be between 0.0 and 1.0, got {args.train_split}")
        mgr.tr_val_split = args.train_split
        mgr.tr_info["tr_val_split"] = args.train_split

    if args.loss_on_label_only:
        mgr.compute_loss_on_labeled_only = True
        mgr.tr_info["compute_loss_on_labeled_only"] = True

    if args.max_epoch is not None:
        mgr.max_epoch = args.max_epoch
        mgr.tr_configs["max_epoch"] = args.max_epoch

    if args.max_steps_per_epoch is not None:
        mgr.max_steps_per_epoch = args.max_steps_per_epoch
        mgr.tr_configs["max_steps_per_epoch"] = args.max_steps_per_epoch

    if args.max_val_steps_per_epoch is not None:
        mgr.max_val_steps_per_epoch = args.max_val_steps_per_epoch
        mgr.tr_configs["max_val_steps_per_epoch"] = args.max_val_steps_per_epoch

    # Handle model name
    if args.model_name is not None:
        mgr.model_name = args.model_name
        mgr.tr_info["model_name"] = args.model_name
        if mgr.verbose:
            print(f"Set model name: {mgr.model_name}")

    # Handle nonlinearity/activation function
    if args.nonlin is not None:
        if not hasattr(mgr, 'model_config') or mgr.model_config is None:
            mgr.model_config = {}
        mgr.model_config["nonlin"] = args.nonlin
        if mgr.verbose:
            print(f"Set activation function: {args.nonlin}")

    # Handle squeeze and excitation
    if args.se:
        if not hasattr(mgr, 'model_config') or mgr.model_config is None:
            mgr.model_config = {}
        mgr.model_config["squeeze_excitation"] = True
        mgr.model_config["squeeze_excitation_reduction_ratio"] = args.se_reduction_ratio
        if mgr.verbose:
            print(f"Enabled squeeze and excitation with reduction ratio: {args.se_reduction_ratio}")

    # Handle optimizer selection
    if args.optimizer is not None:
        mgr.optimizer = args.optimizer
        mgr.tr_configs["optimizer"] = args.optimizer
        if mgr.verbose:
            print(f"Set optimizer: {mgr.optimizer}")

    # Handle loss functions
    if args.loss is not None:
        import ast
        # parse loss list
        try:
            loss_list = ast.literal_eval(args.loss)
            loss_list = loss_list if isinstance(loss_list, list) else [loss_list]
        except Exception:
            loss_list = [s.strip() for s in args.loss.split(',')]
        configure_targets(mgr, loss_list)

    # Handle no_spatial flag
    if args.no_spatial:
        mgr.no_spatial = True
        if hasattr(mgr, 'dataset_config'):
            mgr.dataset_config['no_spatial'] = True
        if mgr.verbose:
            print(f"Disabled spatial transformations (--no-spatial flag set)")

    # Handle gradient clipping
    if args.grad_clip is not None:
        mgr.gradient_clip = args.grad_clip
        mgr.tr_configs["gradient_clip"] = args.grad_clip
        if mgr.verbose:
            print(f"Set gradient clipping: {mgr.gradient_clip}")

    # Handle scheduler selection
    if args.scheduler is not None:
        mgr.scheduler = args.scheduler
        mgr.tr_configs["scheduler"] = args.scheduler
        if mgr.verbose:
            print(f"Set learning rate scheduler: {mgr.scheduler}")

        # If using cosine_warmup, handle its specific parameters
        if args.scheduler == "cosine_warmup":
            if not hasattr(mgr, 'scheduler_kwargs'):
                mgr.scheduler_kwargs = {}

            # Set warmup steps if provided
            if args.warmup_steps is not None:
                mgr.scheduler_kwargs["warmup_steps"] = args.warmup_steps
                # Save scheduler_kwargs to tr_configs
                mgr.tr_configs["scheduler_kwargs"] = mgr.scheduler_kwargs
                if mgr.verbose:
                    print(f"Set warmup steps: {args.warmup_steps}")

    # Handle no_amp flag
    if args.no_amp:
        mgr.no_amp = True
        mgr.tr_configs["no_amp"] = True
        if mgr.verbose:
            print(f"Disabled Automatic Mixed Precision (AMP)")

    # Handle Weights & Biases arguments
    mgr.wandb_project = args.wandb_project
    mgr.wandb_entity = args.wandb_entity


def main():
    """Main entry point for the training script."""
    import argparse
    import ast

    parser = argparse.ArgumentParser(
        description="Train Vesuvius neural networks for ink detection and segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments (input is optional for self-supervised pretraining with data_paths)
    parser.add_argument("-i", "--input",
                        help="Input directory containing images/, labels/, and optionally masks/ subdirectories. Optional when using --pretrain with data_paths in config.")
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
    parser.add_argument("--loss-on-label-only", action="store_true",
                        help="Compute loss only on labeled regions (use masks for loss calculation)")
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
    parser.add_argument("--model-name", type=str,
                        help="Model name for checkpoints and logging (default: from config or 'Model')")
    parser.add_argument("--nonlin", type=str, choices=["LeakyReLU", "ReLU", "SwiGLU", "swiglu", "GLU", "glu"],
                        help="Activation function to use in the model (default: from config or 'LeakyReLU')")
    parser.add_argument("--se", action="store_true", help="Enable squeeze and excitation modules in the encoder")
    parser.add_argument("--se-reduction-ratio", type=float, default=0.0625,
                        help="Squeeze excitation reduction ratio (default: 0.0625 = 1/16)")
    parser.add_argument("--optimizer", type=str,
                        help="Optimizer to use for training (default: from config or 'AdamW, available options in models/optimizers.py')")
    parser.add_argument("--no-spatial", action="store_true",
                        help="Disable spatial/geometric transformations (rotations, flips, etc.) during training")
    parser.add_argument("--grad-clip", type=float, default=12.0,
                        help="Gradient clipping value (default: 12.0)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable Automatic Mixed Precision (AMP) for training")

    # Trainer selection
    parser.add_argument("--trainer", type=str, default="base",
                        help="Trainer class to use (default: base). Options: base, self_supervised, mean_teacher")

    # Self-supervised specific arguments (only used when --trainer self_supervised)
    parser.add_argument("--mask-ratio", type=float, default=0.75,
                        help="Mask ratio for self-supervised pretraining (default: 0.75)")
    parser.add_argument("--mask-patch-size", type=str,
                        help="Mask patch size for self-supervised training as comma-separated values, e.g., '8,16,16' for 3D")

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

    # Check if input is required
    if args.input is None:
        # Check if we're using self-supervised trainer with data_paths
        if args.trainer == "self_supervised" and mgr.dataset_config.get('data_paths'):
            print("Using data_paths from config for self-supervised pretraining")
        else:
            raise ValueError("--input is required unless using self_supervised trainer with data_paths in config")
    else:
        if not Path(args.input).exists():
            raise ValueError(f"Input directory does not exist: {args.input}")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    update_config_from_args(mgr, args)

    # Select trainer based on --trainer argument
    trainer_name = args.trainer.lower()
    if trainer_name == "self_supervised":
        # Configure self-supervised specific parameters
        if not hasattr(mgr, 'dataset_config'):
            mgr.dataset_config = {}

        mgr.dataset_config["mask_ratio"] = args.mask_ratio

        # Parse mask patch size if provided
        if args.mask_patch_size:
            try:
                mask_patch_size = [int(x.strip()) for x in args.mask_patch_size.split(',')]
                mgr.dataset_config["mask_patch_size"] = mask_patch_size
            except ValueError:
                raise ValueError(f"Invalid mask patch size format: {args.mask_patch_size}")

        from vesuvius.models.training.self_supervised_trainer import SelfSupervisedTrainer
        trainer = SelfSupervisedTrainer(mgr=mgr, verbose=args.verbose)
        print("Using Self-Supervised Trainer for self-supervised pretraining")
    elif trainer_name == "mean_teacher":
        from vesuvius.models.training.train_mean_teacher import MeanTeacherTrainer
        trainer = MeanTeacherTrainer(mgr=mgr, verbose=args.verbose)
        print("Using Mean Teacher Trainer for semi-supervised training")
    elif trainer_name == "base":
        trainer = BaseTrainer(mgr=mgr, verbose=args.verbose)
        print("Using Base Trainer for supervised training")
    else:
        raise ValueError(f"Unknown trainer: {trainer_name}. Available options: base, self_supervised, mean_teacher")

    print("Starting training...")
    trainer.train()
    print("Training completed!")


if __name__ == '__main__':
    main()
