from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import wandb

# Internal imports
from models.build.build_network_from_config import NetworkFromConfig
from models.config_manager import ConfigManager
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, BCELoss
from utils.models.helpers import init_weights_he

from vesuvius.models.visualization.plotting import save_debug_gif, export_data_dict_as_tif




class BaseTrainer:
    def __init__(self,
                 config_file: str,
                 verbose: bool = True,
                 debug_dataloader: bool = False):

        self.mgr = ConfigManager(config_file)
        self.debug_dataloader = debug_dataloader

        wandb.init(
            # If you have a project name, set it here:
            project="my_3d_segmentation_project",
            # Use model_name as the run name, or any other naming scheme:
            name=self.mgr.model_name,
        )

        if self.debug_dataloader:
            dataset = self._configure_dataset()
            export_data_dict_as_tif(
                dataset=dataset,
                num_batches=50,
                out_dir="debug_dir"
            )
            print("Debug dataloader plots generated; exiting training early.")
            return

    # --- build model --- #
    def _build_model(self):

        model = NetworkFromConfig(self.mgr)
        model.apply(lambda module: init_weights_he(module, neg_slope=1e-2))

        return model

    def _compose_augmentations(self):
        from vesuvius.augmentation.compose import Compose
        from vesuvius.augmentation.spatial.rotation import RandomRotate90
        from vesuvius.augmentation.spatial.noise import RandomZoom
        from vesuvius.augmentation.noise.intensity import RandomIntensityScaling
        from vesuvius.augmentation.noise.noise import GaussianNoise
        
        augmentations = Compose([
            RandomRotate90(axes=(1, 2), p=0.5),  # Rotate in Y-X plane
            RandomZoom(min_zoom=0.9, max_zoom=1.1, p=0.5),
            RandomIntensityScaling(scale_range=(0.9, 1.1), p=0.5),
            GaussianNoise(mean=0.0, std=0.02, p=0.3)
        ])
        
        return augmentations

    # --- configure dataset --- #
    def _configure_dataset(self):
        from vesuvius.data.vc_dataset import VCDataset
        
        # Convert targets dictionary to the list of dictionaries format expected by VCDataset
        vc_targets = []
        for task_name, task_info in self.mgr.targets.items():
            # For binary segmentation, ensure we have 2 output channels (background + foreground)
            # following nnUNet v2 convention
            out_channels = task_info.get("out_channels", 2)
            
            # If this is a binary segmentation task, force 2 channels
            if out_channels == 1 and task_info.get("is_segmentation", True):
                out_channels = 2
                if self.mgr.verbose:
                    print(f"Forcing 2 output channels for task '{task_name}' to match nnUNet convention (background+foreground)")
            
            vc_targets.append({
                "name": task_name,
                "out_channels": out_channels,
                "activation": task_info.get("activation", "softmax" if out_channels > 1 else "sigmoid")
            })
        
        dataset = VCDataset(
            input_path=self.mgr.data_dir,
            targets=vc_targets,
            patch_size=self.mgr.train_patch_size,
            num_input_channels=self.mgr.in_channels,
            step_size=0.5,  # Default for training
            mode='train',  # Set to train mode
            verbose=self.mgr.verbose,
            # Pass through any Volume-specific parameters from config
            scroll_id=self.mgr.get("scroll_id", None),
            energy=self.mgr.get("energy", None),
            resolution=self.mgr.get("resolution", None),
            segment_id=self.mgr.get("segment_id", None),
            cache=self.mgr.get("cache", True),
            normalize=self.mgr.get("normalize", True),
            normalization_scheme=self.mgr.get("normalization_scheme", "zscore"),
            return_as_type=self.mgr.get("return_as_type", "np.float32"),
            return_as_tensor=True,  # Always return tensors for training
        )
        
        return dataset

    # --- losses ---- #
    def _build_loss(self):
        # if you override this you need to allow for a loss fn to apply to every single
        # possible target in the dictionary of targets . the easiest is probably
        # to add it in losses.loss, import it here, and then add it to the map
        LOSS_FN_MAP = {
            "BCEDiceLoss": BCEDiceLoss,
            "CrossEntropyLoss": CrossEntropyLoss,
            "MSELoss": MSELoss,
            "MaskedCosineLoss": MaskedCosineLoss,
        }

        loss_fns = {}
        for task_name, task_info in self.mgr.targets.items():
            loss_fn = task_info.get("loss_fn", "DC_and_CE_loss")
            if loss_fn not in LOSS_FN_MAP:
                raise ValueError(f"Loss function {loss_fn} not found in LOSS_FN_MAP. Add it to the mapping and try again.")
            loss_kwargs = task_info.get("loss_kwargs", {})
            loss_fns[task_name] = LOSS_FN_MAP[loss_fn](**loss_kwargs)

        return loss_fns

    # --- optimizer ---- #

    def _get_optimizer(self, model):
        from vesuvius.models.training.optimizers import create_optimizer
        
        return create_optimizer(
            model=model,
            optimizer_type=self.mgr.optimizer,
            initial_lr=self.mgr.initial_lr,
            weight_decay=self.mgr.weight_decay
        )

    # --- scheduler --- #
    def _get_scheduler(self, optimizer):
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=self.mgr.max_epoch,
                                      eta_min=0)
        return scheduler

    # --- scaler --- #
    def _get_scaler(self):
        scaler = torch.amp.GradScaler("cuda")
        return scaler

    # --- dataloaders --- #
    def _configure_dataloaders(self, dataset):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        train_val_split = self.mgr.tr_val_split
        split = int(np.floor(train_val_split * dataset_size))
        train_indices, val_indices = indices[:split], indices[split:]
        batch_size = self.mgr.train_batch_size

        train_dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=SubsetRandomSampler(train_indices),
                                pin_memory=True,
                                num_workers=self.mgr.train_num_dataloader_workers)
        val_dataloader = DataLoader(dataset,
                                    batch_size=1,
                                    sampler=SubsetRandomSampler(val_indices),
                                    pin_memory=True,
                                    num_workers=self.mgr.train_num_dataloader_workers)

        return train_dataloader, val_dataloader


    def train(self):

        model = self._build_model()
        optimizer = self._get_optimizer(model)
        loss_fns = self._build_loss()
        dataset = self._configure_dataset()
        scheduler = self._get_scheduler(optimizer)
        scaler = self._get_scaler()

        device = torch.device('cuda')
        model = model.to(device)
        model = torch.compile(model)

        train_dataloader, val_dataloader = self._configure_dataloaders(dataset)

        if model.save_config:
            self.mgr.save_config()



        start_epoch = 0

        if not self.mgr.checkpoint_path:
            os.makedirs(self.mgr.ckpt_out_base, exist_ok=True)

        if self.mgr.checkpoint_path is not None and Path(self.mgr.checkpoint_path).exists():
            print(f"Loading checkpoint from {self.mgr.checkpoint_path}")
            checkpoint = torch.load(self.mgr.checkpoint_path, map_location=device)

            # Always load model weights
            model.load_state_dict(checkpoint['model'])

            if not self.mgr.load_weights_only:
                # Only load optimizer, scheduler, epoch if we are NOT in "weights_only" mode
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming training from epoch {start_epoch + 1}")
            else:
                # Start a 'new run' from epoch 0 or 1
                start_epoch = 0
                scheduler = self._get_scheduler(optimizer)
                print("Loaded model weights only; starting new training run from epoch 1.")


        # TODO: this has some leftover tensorboard crap in it, remove it and finalize wandb switch
        writer = SummaryWriter(log_dir=self.mgr.tensorboard_log_dir)
        global_step = 0
        grad_accumulate_n = self.mgr.gradient_accumulation

        # ---- training! ----- #
        for epoch in range(start_epoch, self.mgr.max_epoch):
            model.train()

            train_running_losses = {t_name: 0.0 for t_name in self.mgr.targets}
            pbar = tqdm(enumerate(train_dataloader), total=self.mgr.max_steps_per_epoch)
            steps = 0

            for i, data_dict in pbar:
                if i >= self.mgr.max_steps_per_epoch:
                    break

                if epoch == 0 and i == 0:
                    for item in data_dict:
                        print(f"Items from the first batch -- Double check that your shapes and values are expected:")
                        print(f"{item}: {data_dict[item].dtype}")
                        print(f"{item}: {data_dict[item].shape}")
                        print(f"{item}: min : {data_dict[item].min()} max : {data_dict[item].max()}")

                global_step += 1

                # VCDataset returns data with "data" key instead of "image"
                inputs = data_dict["data"].to(device, dtype=torch.float32)
                
                # Handle targets for VCDataset format
                # Create an empty targets_dict and populate based on target names from config
                targets_dict = {}
                for t_name, t_info in self.mgr.targets.items():
                    # Determine correct number of output channels (2 for binary segmentation tasks)
                    is_segmentation = t_info.get("is_segmentation", True)
                    out_channels = t_info.get("out_channels", 2 if is_segmentation else 1)
                    
                    # For binary segmentation, enforce 2 channels following nnUNet convention
                    if out_channels == 1 and is_segmentation:
                        out_channels = 2
                    
                    # VCDataset should provide targets with task name as keys
                    # If no target with this name exists, create dummy zeros tensor
                    if t_name in data_dict and data_dict[t_name] is not None:
                        # Get target data and ensure correct shape
                        target = data_dict[t_name].to(device, dtype=torch.float32)
                        
                        # Binary segmentation with 2 output channels requires special handling
                        # If target has single channel but model expects 2, convert to one-hot encoding
                        if target.shape[1] == 1 and out_channels == 2:
                            # Convert binary mask to one-hot encoding (background + foreground)
                            # Channel 0 = background (1 - mask)
                            # Channel 1 = foreground (mask)
                            background = 1.0 - target
                            foreground = target
                            target = torch.cat([background, foreground], dim=1)
                        
                        targets_dict[t_name] = target
                    else:
                        # Create empty tensor with correct shape if target not available
                        targets_dict[t_name] = torch.zeros(
                            (inputs.shape[0], out_channels, *inputs.shape[2:]),
                            device=device, dtype=torch.float32
                        )

                # forward
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    total_loss = 0.0
                    per_task_losses = {}

                    for t_name, t_gt in targets_dict.items():

                        t_pred = outputs[t_name]
                        t_loss_fn = loss_fns[t_name]
                        task_weight = self.mgr.targets[t_name].get("weight", 1.0)

                        # check if skeleton is available in data_dict (e.g. "fibers_skel")
                        skel_key = f"{t_name}_skel"
                        if skel_key in data_dict:
                            t_skel = data_dict[skel_key].to(device, dtype=torch.float32)
                            t_loss = t_loss_fn(t_pred, t_gt, t_skel) * task_weight
                        else:
                            t_loss = t_loss_fn(t_pred, t_gt) * task_weight

                        total_loss += t_loss
                        train_running_losses[t_name] += t_loss.item()
                        per_task_losses[t_name] = t_loss.item()

                # backward
                # loss \ accumulation steps to maintain same effective batch size
                total_loss = total_loss / grad_accumulate_n
                # backward
                scaler.scale(total_loss).backward()

                if (i + 1) % grad_accumulate_n == 0 or (i + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                steps += 1

                desc_parts = []
                for t_name in self.mgr.targets:
                    avg_t_loss = train_running_losses[t_name] / steps
                    desc_parts.append(f"{t_name}: {avg_t_loss:.4f}")

                desc_str = f"Epoch {epoch + 1} => " + " | ".join(desc_parts)
                pbar.set_description(desc_str)

            pbar.close()


            for t_name in self.mgr.targets:
                epoch_avg = train_running_losses[t_name] / steps
                writer.add_scalar(f"train/{t_name}_loss", epoch_avg, epoch)
                avg_t_loss = train_running_losses[t_name] / steps
                wandb.log({f"train/{t_name}_loss": avg_t_loss}, step=global_step)


            print(f"[Train] Epoch {epoch + 1} completed.")

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch
            }, f"{self.mgr.ckpt_out_base}/{self.mgr.model_name}_{epoch + 1}.pth")

            # clean up old checkpoints -- currently just keeps 10 newest
            all_checkpoints = sorted(
                self.mgr.ckpt_out_base.glob(f"{self.mgr.model_name}_*.pth"),
                key=lambda x: x.stat().st_mtime
            )

            # if more than 10, remove the oldest
            while len(all_checkpoints) > 10:
                oldest = all_checkpoints.pop(0)
                oldest.unlink()  #

            # ---- validation ----- #
            if epoch % 1 == 0:
                model.eval()
                with torch.no_grad():
                    val_running_losses = {t_name: 0.0 for t_name in self.mgr.targets}
                    val_steps = 0

                    pbar = tqdm(enumerate(val_dataloader), total=self.mgr.max_val_steps_per_epoch)
                    for i, data_dict in pbar:
                        if i >= self.mgr.max_val_steps_per_epoch:
                            break

                        # VCDataset returns data with "data" key instead of "image"
                        inputs = data_dict["data"].to(device, dtype=torch.float32)
                        
                        # Handle targets for VCDataset format
                        # Create an empty targets_dict and populate based on target names from config
                        targets_dict = {}
                        for t_name, t_info in self.mgr.targets.items():
                            # Determine correct number of output channels (2 for binary segmentation tasks)
                            is_segmentation = t_info.get("is_segmentation", True)
                            out_channels = t_info.get("out_channels", 2 if is_segmentation else 1)
                            
                            # For binary segmentation, enforce 2 channels following nnUNet convention
                            if out_channels == 1 and is_segmentation:
                                out_channels = 2
                            
                            # VCDataset should provide targets with task name as keys
                            # If no target with this name exists, create dummy zeros tensor
                            if t_name in data_dict and data_dict[t_name] is not None:
                                # Get target data and ensure correct shape
                                target = data_dict[t_name].to(device, dtype=torch.float32)
                                
                                # Binary segmentation with 2 output channels requires special handling
                                # If target has single channel but model expects 2, convert to one-hot encoding
                                if target.shape[1] == 1 and out_channels == 2:
                                    # Convert binary mask to one-hot encoding (background + foreground)
                                    # Channel 0 = background (1 - mask)
                                    # Channel 1 = foreground (mask)
                                    background = 1.0 - target
                                    foreground = target
                                    target = torch.cat([background, foreground], dim=1)
                                
                                targets_dict[t_name] = target
                            else:
                                # Create empty tensor with correct shape if target not available
                                targets_dict[t_name] = torch.zeros(
                                    (inputs.shape[0], out_channels, *inputs.shape[2:]),
                                    device=device, dtype=torch.float32
                                )

                        with torch.amp.autocast("cuda"):
                            outputs = model(inputs)
                            total_val_loss = 0.0
                            for t_name, t_gt in targets_dict.items():
                                t_pred = outputs[t_name]
                                t_loss_fn = loss_fns[t_name]
                                t_loss = t_loss_fn(t_pred, t_gt)

                                total_val_loss += t_loss
                                val_running_losses[t_name] += t_loss.item()

                            val_steps += 1

                            if i == 0:
                                b_idx = 0  # pick which sample in the batch to visualize
                                # Slicing shape: [1, c, z, y, x ]
                                inputs_first = inputs[b_idx: b_idx + 1]

                                targets_dict_first = {}
                                for t_name, t_tensor in targets_dict.items():
                                    targets_dict_first[t_name] = t_tensor[b_idx: b_idx + 1]

                                outputs_dict_first = {}
                                for t_name, p_tensor in outputs.items():
                                    outputs_dict_first[t_name] = p_tensor[b_idx: b_idx + 1]

                                # create debug gif
                                save_debug_gif(
                                    input_volume=inputs_first,
                                    targets_dict=targets_dict_first,
                                    outputs_dict=outputs_dict_first,
                                    tasks_dict=self.mgr.targets, # your dictionary, e.g. {"sheet": {"activation":"sigmoid"}, "normals": {"activation":"none"}}
                                    epoch=epoch,
                                    save_path=f"{self.mgr.model_name}_debug.gif"
                                )

                                wandb.log({"val_gif": wandb.Video(f"{self.mgr.model_name}_debug.gif")},
                                          step=global_step)

                    desc_parts = []
                    for t_name in self.mgr.targets:
                        avg_loss_for_t = val_running_losses[t_name] / val_steps
                        desc_parts.append(f"{t_name} {avg_loss_for_t:.4f}")
                        wandb.log({"val_loss": avg_loss_for_t}, step=global_step)
                    desc_str = "Val: " + " | ".join(desc_parts)
                    pbar.set_description(desc_str)

                pbar.close()

                # Final avg for each task
                for t_name in self.mgr.targets:
                    val_avg = val_running_losses[t_name] / val_steps
                    print(f"Task '{t_name}', epoch {epoch + 1} avg val loss: {val_avg:.4f}")

            scheduler.step()

        print('Training Finished!')
        torch.save(model.state_dict(), f'{self.mgr.model_name}_final.pth')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train script for MultiTaskUnet.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to your config file. Use the same one you used for training!")
    parser.add_argument("--debug_dataloader", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    trainer = BaseTrainer(
        config_file=args.config_path,
        verbose=args.verbose,
        debug_dataloader=args.debug_dataloader
    )

    trainer.train()





# During training, you'll get a dict with all outputs
# outputs = model(input_tensor)
# sheet_pred = outputs['sheet']          # Shape: [B, 1, D, H, W]
# normals_pred = outputs['normals']      # Shape: [B, 3, D, H, W]
# affinities_pred = outputs['affinities']  # Shape: [B, N_affinities, D, H, W]