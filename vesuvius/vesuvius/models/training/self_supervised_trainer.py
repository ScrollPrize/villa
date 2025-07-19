"""
Self-supervised Trainer for unsupervised pretraining of volumetric models.
"""
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext

from .train import BaseTrainer
from vesuvius.models.datasets import SelfSupervisedPretrainDataset


class SelfSupervisedTrainer(BaseTrainer):
    """
    Trainer class for self-supervised pretraining of volumetric models.
    """
    
    def __init__(self, mgr=None, verbose: bool = True):
        super().__init__(mgr, verbose)

        if not hasattr(self.mgr, 'model_config'):
            self.mgr.model_config = {}
        self.mgr.model_config["self_supervised_mode"] = True
    
    def _configure_dataset(self, is_training=True):
        dataset = SelfSupervisedPretrainDataset(mgr=self.mgr, is_training=is_training)
        print(f"Using self-supervised pretraining dataset ({'training' if is_training else 'validation'})")
        return dataset
    
    def _update_mask_ratio(self, dataset, epoch):
        """The mask ratio of the target can be changed throughout training if
         for example you'd like to try something like curriculum learning.
        """
        if hasattr(dataset, 'set_mask_ratio'): # by default the mask ratio is used from start to end (ie. no curriculum)
            target_mask_ratio = dataset.mask_ratio
            dataset.set_mask_ratio(target_mask_ratio)
            print(f"\n[Self-Supervised] Epoch {epoch + 1}: Mask ratio = {target_mask_ratio:.2%}")
            return target_mask_ratio
        return None
    
    def _prepare_self_supervised_batch(self, data_dict):
        inputs = data_dict["masked_image"].to(self.device, dtype=torch.float32)
        original_image = data_dict["image"].to(self.device, dtype=torch.float32)
        mask = data_dict["mask"].to(self.device, dtype=torch.float32)

        batch_size = original_image.shape[0]
        non_zero_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for b in range(batch_size):
            non_zero_mask[b] = torch.any(original_image[b] != 0)

        is_valid = torch.any(non_zero_mask)
        targets_dict = {"reconstruction": original_image}
        self_supervised_mask = {"reconstruction": mask}
        
        return inputs, targets_dict, self_supervised_mask, is_valid, non_zero_mask
    
    def _compute_self_supervised_loss(self, outputs, targets_dict, self_supervised_mask, loss_fns):
        total_loss = 0.0
        losses_dict = {}
        
        for t_name, t_gt in targets_dict.items():
            t_pred = outputs[t_name]
            task_losses = loss_fns[t_name]
            task_weight = self.mgr.targets[t_name].get("weight", 1.0)
            
            task_total_loss = 0.0
            for loss_fn, loss_weight in task_losses:
                # Self-supervised loss expects mask as third argument
                loss_value = loss_fn(t_pred, t_gt, self_supervised_mask[t_name])
                task_total_loss += loss_weight * loss_value
            
            weighted_loss = task_weight * task_total_loss
            total_loss += weighted_loss
            losses_dict[t_name] = task_total_loss.detach().cpu().item()
        
        return total_loss, losses_dict
    
    def _compute_self_supervised_val_loss(self, outputs, targets_dict, self_supervised_mask, loss_fns, non_zero_mask=None):
        val_losses = {}
        
        for t_name, t_gt in targets_dict.items():
            t_pred = outputs[t_name]
            task_losses = loss_fns[t_name]
            
            task_total_loss = 0.0
            for loss_fn, loss_weight in task_losses:
                if non_zero_mask is not None and torch.any(non_zero_mask):
                    t_pred_filtered = t_pred[non_zero_mask]
                    t_gt_filtered = t_gt[non_zero_mask]
                    mask_filtered = self_supervised_mask[t_name][non_zero_mask]
                    
                    if t_pred_filtered.shape[0] > 0:
                        loss_value = loss_fn(t_pred_filtered, t_gt_filtered, mask_filtered)
                    else:
                        loss_value = torch.tensor(0.0, device=self.device, requires_grad=True)
                else:
                    loss_value = loss_fn(t_pred, t_gt, self_supervised_mask[t_name])
                
                task_total_loss += loss_weight * loss_value
            
            val_losses[t_name] = task_total_loss.detach().cpu().item()
        
        return val_losses
    
    def train(self):
        from vesuvius.models.utilities.s3_utils import detect_s3_paths, setup_multiprocessing_for_s3
        if detect_s3_paths(self.mgr):
            print("\nDetected S3 paths in configuration")
            setup_multiprocessing_for_s3()

        train_dataset = self._configure_dataset(is_training=True)
        val_dataset = self._configure_dataset(is_training=False)

        self.mgr.auto_detect_channels(train_dataset)
        model = self._build_model()
        optimizer = self._get_optimizer(model)
        loss_fns = self._build_loss()
        scheduler, is_per_iteration_scheduler = self._get_scheduler(optimizer)

        from vesuvius.utils.utils import init_weights_he
        model.apply(lambda module: init_weights_he(module, neg_slope=0.2))
        model = model.to(self.device)
        
        if self.device.type == 'cuda':
            model = torch.compile(model)

        use_amp = not getattr(self.mgr, 'no_amp', False)
        if not use_amp:
            print("Automatic Mixed Precision (AMP) is disabled")
        scaler = self._get_scaler(self.device.type, use_amp=use_amp)

        train_dataloader, val_dataloader, train_indices, val_indices = self._configure_dataloaders(train_dataset, val_dataset)

        if self.mgr.wandb_project:
            import wandb
            wandb.init(
                entity=self.mgr.wandb_entity,
                project=self.mgr.wandb_project,
                group=self.mgr.model_name,
                config=self.mgr.convert_to_dict()
            )

        start_epoch = 0
        val_loss_history = {}
        checkpoint_history = []
        best_checkpoints = []
        debug_gif_history = []
        best_debug_gifs = []

        import os
        from datetime import datetime
        from collections import deque
        os.makedirs(self.mgr.ckpt_out_base, exist_ok=True)
        model_ckpt_dir = os.path.join(self.mgr.ckpt_out_base, self.mgr.model_name)
        os.makedirs(model_ckpt_dir, exist_ok=True)
        
        now = datetime.now()
        date_str = now.strftime('%m%d%y')
        time_str = now.strftime('%H%M')
        ckpt_dir = os.path.join('checkpoints', f"{self.mgr.model_name}_{date_str}{time_str}")
        os.makedirs(ckpt_dir, exist_ok=True)

        if hasattr(self.mgr, 'checkpoint_path') and self.mgr.checkpoint_path:
            from vesuvius.models.utilities.load_checkpoint import load_checkpoint
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
        
        global_step = 0
        grad_accumulate_n = self.mgr.gradient_accumulation
        
        # Training loop
        for epoch in range(start_epoch, self.mgr.max_epoch):
            self._update_mask_ratio(train_dataset, epoch)
            if hasattr(val_dataset, 'set_mask_ratio'):
                val_dataset.set_mask_ratio(train_dataset.mask_ratio)
            
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

                inputs, targets_dict, self_supervised_mask, is_valid, _ = self._prepare_self_supervised_batch(data_dict)
                
                # Skip if all samples are zero
                if not is_valid:
                    continue

                if use_amp and self.device.type in ['cuda', 'cpu']:
                    autocast_ctx = torch.amp.autocast()
                else:
                    autocast_ctx = nullcontext()
                
                with autocast_ctx:
                    outputs = model(inputs)
                    total_loss, loss_values = self._compute_self_supervised_loss(outputs, targets_dict, self_supervised_mask, loss_fns)

                    for t_name, loss_val in loss_values.items():
                        epoch_losses[t_name].append(loss_val)

                    total_loss = total_loss / grad_accumulate_n

                if self.mgr.wandb_project:
                    import wandb
                    wandb.log({
                        **{f"loss_{t_name}": loss_values[t_name] for t_name in self.mgr.targets},
                        "loss_total": total_loss.detach().cpu().item()
                    })
                
                # Backward pass
                scaler.scale(total_loss).backward()
                
                # Optimizer step
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

            if not is_per_iteration_scheduler:
                scheduler.step()

            import gc
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            print(f"\n[Train] Epoch {epoch + 1} completed.")
            for t_name in self.mgr.targets:
                avg_loss = np.mean(epoch_losses[t_name]) if epoch_losses[t_name] else 0
                print(f"  {t_name}: Avg Loss = {avg_loss:.4f}")

            from vesuvius.models.training.save_checkpoint import save_checkpoint
            ckpt_path = os.path.join(ckpt_dir, f"{self.mgr.model_name}_epoch{epoch}.pth")
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
            
            # Validation
            if epoch % 1 == 0:
                model.eval()
                with torch.no_grad():
                    val_losses = {t_name: [] for t_name in self.mgr.targets}
                    val_dataloader_iter = iter(val_dataloader)

                    if hasattr(self.mgr, 'max_val_steps_per_epoch') and self.mgr.max_val_steps_per_epoch and self.mgr.max_val_steps_per_epoch > 0:
                        num_val_iters = min(len(val_indices), self.mgr.max_val_steps_per_epoch)
                    else:
                        num_val_iters = len(val_indices)
                    
                    val_pbar = tqdm(range(num_val_iters), desc=f'Validation {epoch+1}')
                    
                    for i in val_pbar:
                        try:
                            data_dict = next(val_dataloader_iter)
                        except StopIteration:
                            val_dataloader_iter = iter(val_dataloader)
                            data_dict = next(val_dataloader_iter)

                        inputs, targets_dict, self_supervised_mask, is_valid, val_non_zero_mask = self._prepare_self_supervised_batch(data_dict)

                        if not is_valid:
                            continue
                        
                        # Forward pass
                        if use_amp:
                            context = (
                                torch.cuda.amp.autocast(dtype=torch.float16) if self.device.type == 'cuda' 
                                else nullcontext()
                            )
                        else:
                            context = nullcontext()
                        
                        with context:
                            outputs = model(inputs)
                            batch_val_losses = self._compute_self_supervised_val_loss(
                                outputs, targets_dict, self_supervised_mask, loss_fns, val_non_zero_mask
                            )
                            
                            for t_name, loss_val in batch_val_losses.items():
                                val_losses[t_name].append(loss_val)
                        
                        # Debug visualization for first batch
                        if i == 0 and is_valid:
                            from vesuvius.utils.plotting import save_debug
                            
                            # Find first non-zero sample
                            b_idx = 0
                            for b in range(val_non_zero_mask.shape[0]):
                                if val_non_zero_mask[b]:
                                    b_idx = b
                                    break
                            
                            # Extract first valid sample
                            inputs_first = inputs[b_idx: b_idx + 1]
                            targets_dict_first = {t: v[b_idx: b_idx + 1] for t, v in targets_dict.items()}
                            outputs_dict_first = {t: v[b_idx: b_idx + 1] for t, v in outputs.items()}
                            
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

                    from vesuvius.models.training.save_checkpoint import manage_checkpoint_history, manage_debug_gifs, cleanup_old_configs
                    
                    checkpoint_history = deque(checkpoint_history, maxlen=3)
                    checkpoint_history, best_checkpoints = manage_checkpoint_history(
                        checkpoint_history=list(checkpoint_history),
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
                        debug_gif_history = deque(debug_gif_history, maxlen=3)
                        debug_gif_history, best_debug_gifs = manage_debug_gifs(
                            debug_gif_history=list(debug_gif_history),
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