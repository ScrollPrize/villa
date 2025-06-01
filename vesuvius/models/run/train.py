from pathlib import Path
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.training.lr_schedulers import get_scheduler, PolyLRScheduler
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils.utils import init_weights_he
from models.datasets import NapariDataset, ImageDataset, ZarrDataset
from utils.plotting import save_debug
from models.build.build_network_from_config import NetworkFromConfig

from models.training.loss.losses import _create_loss
from models.training.optimizers import create_optimizer
from itertools import cycle
from contextlib import nullcontext
from collections import deque   
import gc                      


def compute_gradient_norm(model):
    """Compute the L2 norm of gradients across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


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
            loss_fn_name = task_info.get("loss_fn", "BCEDiceLoss")
            print(f"Target {task_name} using loss function: {loss_fn_name}")
            
            loss_config = task_info.get("loss_kwargs", {})
            weight = loss_config.get("weight", None)
            ignore_index = loss_config.get("ignore_index", -100)
            pos_weight = loss_config.get("pos_weight", None)
            
            # If compute_loss_on_labeled_only is set and no ignore_index is specified, use -100
            if hasattr(self.mgr, 'compute_loss_on_labeled_only') and self.mgr.compute_loss_on_labeled_only and ignore_index is None:
                ignore_index = -100
                print(f"Setting ignore_index=-100 for target '{task_name}' due to compute_loss_on_labeled_only=True")
            
            try:
                loss_fns[task_name] = _create_loss(
                    name=loss_fn_name,
                    loss_config=loss_config,
                    weight=weight,
                    ignore_index=ignore_index,
                    pos_weight=pos_weight
                )
            except RuntimeError as e:
                raise ValueError(f"Failed to create loss function '{loss_fn_name}' for target '{task_name}': {str(e)}")

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
            return torch.amp.GradScaler()
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

        device_type = 'mps' if hasattr(torch, 'mps') and torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

        # For MPS device, set num_workers=0 to avoid pickling error
        if device_type == 'mps':
            num_workers = 0
        else:
            # For CUDA or CPU, use the configured number of workers
            # If train_num_dataloader_workers is not set, default to 4
            if hasattr(self.mgr, 'train_num_dataloader_workers'):
                num_workers = self.mgr.train_num_dataloader_workers
            else:
                num_workers = 4

        train_dataloader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                sampler=SubsetRandomSampler(train_indices),
                                pin_memory=(device_type == 'cuda'),  # pin_memory=True for CUDA
                                num_workers=num_workers)

        val_dataloader = DataLoader(val_dataset,
                                    batch_size=1,
                                    sampler=SubsetRandomSampler(val_indices),
                                    pin_memory=(device_type == 'cuda'), 
                                    num_workers=num_workers)

        return train_dataloader, val_dataloader, train_indices, val_indices


    def train(self):

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

        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA device")
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS device (Apple Silicon)")
        else:
            device = torch.device('cpu')
            print("Using CPU device")

        model.apply(lambda module: init_weights_he(module, neg_slope=0.2))
        model = model.to(device)

        # Only compile the model if it's on CUDA (not supported on MPS/CPU)
        if device.type == 'cuda':
            model = torch.compile(model, mode="default", fullgraph=False)

        if not hasattr(torch, 'no_op'):
            class NullContextManager:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass

            torch.no_op = lambda: NullContextManager()


        # Check if AMP is disabled
        use_amp = not getattr(self.mgr, 'no_amp', False)
        if not use_amp:
            print("Automatic Mixed Precision (AMP) is disabled")
        elif device.type == 'cuda':
            print("Using Automatic Mixed Precision (AMP) for training")
        
        scaler = self._get_scaler(device.type, use_amp=use_amp)
        train_dataloader, val_dataloader, train_indices, val_indices = self._configure_dataloaders(train_dataset, val_dataset)

        if model.save_config:
            self.mgr.save_config()

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
        
        # Create the checkpoint directory once at the start of training
        now = datetime.now()
        date_str = now.strftime('%m%d%y')
        time_str = now.strftime('%H%M')
        ckpt_dir = os.path.join('checkpoints', f"{self.mgr.model_name}_{date_str}{time_str}")
        os.makedirs(ckpt_dir, exist_ok=True)

        valid_checkpoint = (self.mgr.checkpoint_path is not None and 
                           self.mgr.checkpoint_path != "" and 
                           Path(self.mgr.checkpoint_path).exists())

        if valid_checkpoint:
            print(f"Loading checkpoint from {self.mgr.checkpoint_path}")
            checkpoint = torch.load(self.mgr.checkpoint_path, map_location=device)

            if 'model_config' in checkpoint:
                print("Found model configuration in checkpoint, using it to initialize the model")

                if hasattr(self.mgr, 'targets') and 'targets' in checkpoint['model_config']:
                    self.mgr.targets = checkpoint['model_config']['targets']
                    print(f"Updated targets from checkpoint: {self.mgr.targets}")
            
            if 'normalization_scheme' in checkpoint:
                print(f"Found normalization scheme in checkpoint: {checkpoint['normalization_scheme']}")
                self.mgr.normalization_scheme = checkpoint['normalization_scheme']
                if hasattr(self.mgr, 'dataset_config'):
                    self.mgr.dataset_config['normalization_scheme'] = checkpoint['normalization_scheme']
            
            if 'intensity_properties' in checkpoint:
                print("Found intensity properties in checkpoint")
                self.mgr.intensity_properties = checkpoint['intensity_properties']
                if hasattr(self.mgr, 'dataset_config'):
                    self.mgr.dataset_config['intensity_properties'] = checkpoint['intensity_properties']
                print("Loaded intensity properties:")
                for key, value in checkpoint['intensity_properties'].items():
                    print(f"  {key}: {value:.4f}")

                if model.autoconfigure != checkpoint['model_config'].get('autoconfigure', True):
                    print("Model autoconfiguration differs, rebuilding model from checkpoint config")

                    class ConfigWrapper:
                        def __init__(self, config_dict, base_mgr):
                            self.__dict__.update(config_dict)
                            for attr_name in dir(base_mgr):
                                if not attr_name.startswith('__') and not hasattr(self, attr_name):
                                    setattr(self, attr_name, getattr(base_mgr, attr_name))

                    config_wrapper = ConfigWrapper(checkpoint['model_config'], self.mgr)
                    model = NetworkFromConfig(config_wrapper)

                    model = model.to(device)
                    if device.type == 'cuda':
                        model = torch.compile(model)
                    optimizer = self._get_optimizer(model)

            # Load model weights
            model.load_state_dict(checkpoint['model'])

            if not self.mgr.load_weights_only:
                # Only load optimizer, scheduler, epoch if we are NOT in "weights_only" mode
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming training from epoch {start_epoch + 1}")
            else:
                start_epoch = 0
                scheduler, is_per_iteration_scheduler = self._get_scheduler(optimizer)
                print("Loaded model weights only; starting new training run from epoch 1.")

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
            pbar = tqdm(range(num_iters), desc=f'Epoch {epoch+1}/{self.mgr.max_epoch}')
            
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
                                print(f"  {sub_key}: {sub_val.dtype}, {sub_val.shape}, min {sub_val.min()} max {sub_val.max()}")
                        else:
                            print(f"{item}: {val.dtype}, {val.shape}, min {val.min()} max {val.max()}")

                global_step += 1
                inputs = data_dict["image"].to(device, dtype=torch.float32)
                ignore_masks = None
                if "ignore_masks" in data_dict:
                    ignore_masks = {t_name: mask.to(device) for t_name, mask in data_dict["ignore_masks"].items()}

                targets_dict = {
                    k: v.to(device, dtype=torch.float32)
                    for k, v in data_dict.items()
                    if k not in ["image", "ignore_masks"]
                }

                # Only use autocast if AMP is enabled
                if use_amp and device.type in ['cuda', 'cpu']:
                    autocast_ctx = torch.amp.autocast(device.type)
                else:
                    autocast_ctx = nullcontext()

                with autocast_ctx:
                    outputs = model(inputs)
                    total_loss = 0.0

                    for t_name, t_gt in targets_dict.items():
                        t_pred = outputs[t_name]
                        t_loss_fn = loss_fns[t_name]
                        task_weight = self.mgr.targets[t_name].get("weight", 1.0)

                        if ignore_masks is not None and t_name in ignore_masks:
                            ignore_mask = ignore_masks[t_name]

                            if i == 0:
                                print(f"Using custom ignore mask for target {t_name}")

                            ignore_label = getattr(t_loss_fn, 'ignore_index', -100)

                            if ignore_mask.dim() == t_gt.dim() - 1:
                                ignore_mask = ignore_mask.unsqueeze(1)

                            ignore_tensor = torch.tensor(ignore_label, dtype=t_gt.dtype, device=t_gt.device)
                            t_gt = torch.where(ignore_mask == 1, ignore_tensor, t_gt)

                        t_loss = t_loss_fn(t_pred, t_gt)
                        
                        weighted_loss = task_weight * t_loss
                        total_loss += weighted_loss
                        
                        # Store the actual loss value (after task weighting but before grad accumulation scaling)
                        epoch_losses[t_name].append(t_loss.detach().cpu().item())

                    # Scale loss by accumulation steps to maintain same effective batch size
                    total_loss = total_loss / grad_accumulate_n
                
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
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            print(f"\n[Train] Epoch {epoch + 1} completed.")
            for t_name in self.mgr.targets:
                avg_loss = np.mean(epoch_losses[t_name]) if epoch_losses[t_name] else 0
                print(f"  {t_name}: Avg Loss = {avg_loss:.4f}")

            # Use the checkpoint directory created at the start of training
            ckpt_path = os.path.join(
                ckpt_dir,
                f"{self.mgr.model_name}_epoch{epoch}.pth"
                )
            
            # Save checkpoint with model weights and training state
            # Include the model configuration directly in the checkpoint
            # Also include normalization information
            checkpoint_data = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'model_config': model.final_config  # Save the model configuration
            }
            
            # Add normalization information from the dataset
            if hasattr(train_dataset, 'normalization_scheme'):
                checkpoint_data['normalization_scheme'] = train_dataset.normalization_scheme
            if hasattr(train_dataset, 'intensity_properties'):
                checkpoint_data['intensity_properties'] = train_dataset.intensity_properties
                
            torch.save(checkpoint_data, ckpt_path)
            print(f"Checkpoint saved to: {ckpt_path}")
            
            # Add to checkpoint history
            checkpoint_history.append((epoch, ckpt_path))
            
            # Explicit cleanup after checkpoint save
            del checkpoint_data
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # ---- validation ----- #
            if epoch % 1 == 0:
                model.eval()
                with torch.no_grad():
                    val_losses = {t_name: [] for t_name in self.mgr.targets}

                    val_dataloader_iter = iter(val_dataloader)

                    if hasattr(self.mgr, 'max_val_steps_per_epoch') and self.mgr.max_val_steps_per_epoch and self.mgr.max_val_steps_per_epoch > 0:
                        num_val_iters = min(len(val_indices), self.mgr.max_val_steps_per_epoch)
                    else:
                        num_val_iters = len(val_indices)  # Use all data 

                    val_pbar = tqdm(range(num_val_iters), desc=f'Validation {epoch+1}')
                    
                    for i in val_pbar:
                        try:
                            data_dict = next(val_dataloader_iter)
                        except StopIteration:
                            val_dataloader_iter = iter(val_dataloader)
                            data_dict = next(val_dataloader_iter)

                        inputs = data_dict["image"].to(device, dtype=torch.float32)
                        ignore_masks = None
                        if "ignore_masks" in data_dict:
                            ignore_masks = {
                                t_name: mask.to(device, dtype=torch.float32)
                                for t_name, mask in data_dict["ignore_masks"].items()
                            }

                        targets_dict = {
                            k: v.to(device, dtype=torch.float32)
                            for k, v in data_dict.items()
                            if k != "image" and k != "ignore_masks"
                        }

                        # Only use autocast if AMP is enabled
                        if use_amp:
                            context = (
                                torch.amp.autocast(device.type) if device.type == 'cuda' 
                                else torch.amp.autocast('cpu') if device.type == 'cpu' 
                                else torch.no_op() if device.type == 'mps' 
                                else torch.no_op()
                            )
                        else:
                            context = nullcontext()

                        with context:
                            outputs = model(inputs)
                            
                            for t_name, t_gt in targets_dict.items():
                                t_pred = outputs[t_name]
                                t_loss_fn = loss_fns[t_name]

                                if ignore_masks is not None and t_name in ignore_masks:
                                    ignore_mask = ignore_masks[t_name].to(device, dtype=torch.float32)

                                    if hasattr(t_loss_fn, 'ignore_index'):
                                        ignore_label = t_loss_fn.ignore_index
                                    else:
                                        ignore_label = -100

                                    if ignore_mask.dim() == t_gt.dim() - 1:
                                        ignore_mask = ignore_mask.unsqueeze(1)

                                    # Apply mask to target: set regions where mask is 1 to ignore_label
                                    t_gt = torch.where(ignore_mask == 1, torch.tensor(ignore_label, dtype=t_gt.dtype, device=t_gt.device), t_gt)

                                t_loss = t_loss_fn(t_pred, t_gt)
                                val_losses[t_name].append(t_loss.detach().cpu().item())

                            if i == 0:
                                b_idx = 0  
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
                                    tasks_dict=self.mgr.targets, # dictionary, e.g. {"sheet": {"activation":"sigmoid"}, "normals": {"activation":"none"}}
                                    epoch=epoch,
                                    save_path=debug_img_path
                                )
                                debug_gif_history.append((epoch, debug_img_path))
                            

                            loss_str = " | ".join([f"{t}: {np.mean(val_losses[t]):.4f}" 
                                                  for t in self.mgr.targets if len(val_losses[t]) > 0])
                            val_pbar.set_postfix_str(loss_str)
                            
                            del outputs, inputs, targets_dict
                            if ignore_masks is not None:
                                del ignore_masks

                    print(f"\n[Validation] Epoch {epoch + 1} summary:")
                    total_val_loss = 0.0
                    for t_name in self.mgr.targets:
                        val_avg = np.mean(val_losses[t_name]) if val_losses[t_name] else 0
                        print(f"  Task '{t_name}': Avg validation loss = {val_avg:.4f}")
                        total_val_loss += val_avg
                    
                    # Average validation loss across all tasks
                    avg_val_loss = total_val_loss / len(self.mgr.targets) if self.mgr.targets else 0
                    val_loss_history[epoch] = avg_val_loss
                    
                    # Update best checkpoints
                    if epoch in [e for e, _ in checkpoint_history]:
                        ckpt_path = next(p for e, p in checkpoint_history if e == epoch)
                        best_checkpoints.append((avg_val_loss, epoch, ckpt_path))
                        best_checkpoints.sort(key=lambda x: x[0])  # Sort by validation loss
                        
                        # Keep only 2 best checkpoints
                        if len(best_checkpoints) > 2:
                            _, _, removed_path = best_checkpoints.pop(-1)  # Remove worst
                            # Check if the removed checkpoint is not in the last 3
                            if removed_path not in [p for _, p in checkpoint_history]:
                                if Path(removed_path).exists():
                                    Path(removed_path).unlink()
                                    print(f"Removed checkpoint with higher validation loss: {removed_path}")
                    
                    # Update best debug gifs
                    if epoch in [e for e, _ in debug_gif_history]:
                        gif_path = next(p for e, p in debug_gif_history if e == epoch)
                        best_debug_gifs.append((avg_val_loss, epoch, gif_path))
                        best_debug_gifs.sort(key=lambda x: x[0])  # Sort by validation loss
                        
                        # Keep only 2 best debug gifs
                        if len(best_debug_gifs) > 2:
                            _, _, removed_gif = best_debug_gifs.pop(-1)  # Remove worst
                            # Check if the removed gif is not in the last 3
                            if removed_gif not in [p for _, p in debug_gif_history]:
                                if Path(removed_gif).exists():
                                    Path(removed_gif).unlink()
                                    print(f"Removed debug gif with higher validation loss: {removed_gif}")
                    
                    # Clean up checkpoints not in last 3 or best 2
                    all_checkpoints_to_keep = set()
                    # Add last 3
                    for _, ckpt_path in checkpoint_history:
                        all_checkpoints_to_keep.add(Path(ckpt_path))
                    # Add best 2
                    for _, _, ckpt_path in best_checkpoints[:2]:
                        all_checkpoints_to_keep.add(Path(ckpt_path))
                    
                    # Remove checkpoints not in the keep set
                    ckpt_dir_path = Path(ckpt_dir)
                    for ckpt_file in ckpt_dir_path.glob(f"{self.mgr.model_name}_epoch*.pth"):
                        if ckpt_file not in all_checkpoints_to_keep:
                            ckpt_file.unlink()
                            print(f"Removed checkpoint: {ckpt_file}")
                    
                    # Clean up debug gifs not in last 3 or best 2
                    all_gifs_to_keep = set()
                    # Add last 3
                    for _, gif_path in debug_gif_history:
                        all_gifs_to_keep.add(Path(gif_path))
                    # Add best 2
                    for _, _, gif_path in best_debug_gifs[:2]:
                        all_gifs_to_keep.add(Path(gif_path))
                    
                    # Remove gifs not in the keep set
                    for gif_file in ckpt_dir_path.glob(f"{self.mgr.model_name}_debug_epoch*.gif"):
                        if gif_file not in all_gifs_to_keep:
                            gif_file.unlink()
                            print(f"Removed debug gif: {gif_file}")
                    
                    # Print current checkpoint status
                    print(f"\nCheckpoint management:")
                    print(f"  Last 3 checkpoints: {[f'epoch{e}' for e, _ in checkpoint_history]}")
                    if best_checkpoints:
                        print(f"  Best 2 checkpoints: {[f'epoch{e} (loss={l:.4f})' for l, e, _ in best_checkpoints[:2]]}")
                    
                    # Clean up old config files - keep only the latest
                    ckpt_dir_parent = Path(model_ckpt_dir)
                    all_configs = sorted(
                        ckpt_dir_parent.glob(f"{self.mgr.model_name}_*.yaml"),
                        key=lambda x: x.stat().st_mtime
                    )
                    while len(all_configs) > 1:
                        oldest = all_configs.pop(0)
                        oldest.unlink()
                        print(f"Removed old config: {oldest}")


        print('Training Finished!')

        final_model_path = f"{model_ckpt_dir}/{self.mgr.model_name}_final.pth"

        final_checkpoint_data = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': self.mgr.max_epoch - 1,
            'model_config': model.final_config
        }
        if hasattr(train_dataset, 'normalization_scheme'):
            final_checkpoint_data['normalization_scheme'] = train_dataset.normalization_scheme
        if hasattr(train_dataset, 'intensity_properties'):
            final_checkpoint_data['intensity_properties'] = train_dataset.intensity_properties
            
        torch.save(final_checkpoint_data, final_model_path)

        print(f"Final model saved to {final_model_path}")
        print(f"Model configuration is embedded in the checkpoint")


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
    auxiliary_targets = {}
    if hasattr(mgr, 'targets') and mgr.targets:
        for target_name, target_info in mgr.targets.items():
            if target_info.get('auxiliary_task', False):
                auxiliary_targets[target_name] = target_info
    
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
            for ext in ['*.tif','*.tiff','*.png','*.jpg','*.jpeg']:
                for f in images_dir.glob(ext):
                    if '_' in f.stem:
                        targets.add(f.stem.rsplit('_',1)[1])
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
        mgr.targets.update(auxiliary_targets)
        print(f"Re-added auxiliary targets: {list(auxiliary_targets.keys())}")
    
    # Re-apply auxiliary tasks from config
    if hasattr(mgr, '_apply_auxiliary_tasks'):
        mgr._apply_auxiliary_tasks()

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
    mgr.data_path = Path(args.input)
    # Save data_path to dataset_config
    if not hasattr(mgr, 'dataset_config'):
        mgr.dataset_config = {}
    mgr.dataset_config["data_path"] = str(mgr.data_path)

    if args.format:
        mgr.data_format = args.format; print(f"Using specified data format: {mgr.data_format}")
    else:
        detected = detect_data_format(mgr.data_path)
        if detected:
            mgr.data_format = detected; print(f"Auto-detected data format: {mgr.data_format}")
        else:
            raise ValueError("Data format could not be determined. Please specify --format.")
    
    # Save data_format to dataset_config
    mgr.dataset_config["data_format"] = mgr.data_format

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
            raise ValueError(f"Invalid patch size format: {args.patch_size}. Expected comma-separated integers like '192,192,192'")

    if args.train_split is not None:
        if not 0.0 <= args.train_split <= 1.0:
            raise ValueError(f"Train split must be between 0.0 and 1.0, got {args.train_split}")
        mgr.tr_val_split = args.train_split
        mgr.tr_info["tr_val_split"] = args.train_split

    if args.loss_on_label_only:
        mgr.compute_loss_on_labeled_only = True
        mgr.tr_info["compute_loss_on_labeled_only"] = True

    # Handle max steps per epoch if provided
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
                       help="Input directory containing images/, labels/, and optionally masks/ subdirectories")
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
    parser.add_argument("--config-path", type=str,
                       help="Path to configuration YAML file (if not provided, uses defaults)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output for debugging")
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
    
    # Learning rate scheduler arguments
    parser.add_argument("--scheduler", type=str, 
                       help="Learning rate scheduler type (default: from config or 'poly')")
    parser.add_argument("--warmup-steps", type=int,
                       help="Number of warmup steps for cosine_warmup scheduler (default: 10%% of first cycle)")

    args = parser.parse_args()


    if not Path(args.input).exists():
        raise ValueError(f"Input directory does not exist: {args.input}")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    from models.configuration.config_manager import ConfigManager
    mgr = ConfigManager(verbose=args.verbose)

    if args.config_path:
        if not Path(args.config_path).exists():
            raise ValueError(f"Config file does not exist: {args.config_path}")
        mgr.load_config(args.config_path)
        print(f"Loaded configuration from: {args.config_path}")
    else:
        mgr.tr_info = {}
        mgr.tr_configs = {}
        mgr.model_config = {}
        mgr.dataset_config = {}
        mgr._init_attributes()
        print("Initialized with default configuration")

    update_config_from_args(mgr, args)

    trainer = BaseTrainer(mgr=mgr, verbose=args.verbose)

    print("Starting training...")
    trainer.train()
    print("Training completed!")


if __name__ == '__main__':
    main()
