from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from helpers import init_weights_he
import albumentations as A
import volumentations as V
import wandb


from dataloading.dataset_old import ZarrSegmentationDataset3D
from dataloading.dataset import MultiTask3dDataset
from dataloading.wk2_dataset import wkDataset
from training.visualization.plotting import save_debug_gif, export_data_dict_as_tif
from builders.build_network_from_config import NetworkFromConfig
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, BCELoss
from training.losses.losses import MaskedCosineLoss, BCEDiceLoss
from configuration.config_manager import ConfigManager


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

        # --- Augmentations (2D + 3D) ---
        image_transforms = A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(),
                #A.Illumination(),
            ], p=0.2),

            A.OneOf([
                A.GaussNoise()
            ], p=0.2),

            A.OneOf([
                A.MotionBlur(),
                #A.Defocus(),
                A.Downscale(),
                A.AdvancedBlur()
            ], p=0.2),
        ], p=1.0)

        vol_transform = A.Compose([
            A.CoarseDropout3D(
                fill=0.5,
                num_holes_range=(1, 4),
                hole_depth_range=(0.1, 0.5),
                hole_height_range=(0.1, 0.5),
                hole_width_range=(0.1, 0.5)
            )
        ], p=0.5)

        return image_transforms, vol_transform

    # --- configure dataset --- #
    def _configure_dataset(self):

        image_transforms, volume_transforms = self._compose_augmentations()

        dataset = MultiTask3dDataset(mgr=self.mgr,
                                     image_transforms=image_transforms,
                                     volume_transforms=volume_transforms)

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
        if self.mgr.optimizer == "SGD":
            optimizer = SGD(
                model.parameters(),
                lr=self.mgr.initial_lr,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.mgr.weight_decay
            )
        else:
            optimizer = AdamW(
                model.parameters(),
                lr=self.mgr.initial_lr,
                weight_decay=self.mgr.weight_decay
            )
        return optimizer

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

                inputs = data_dict["image"].to(device, dtype=torch.float32)
                targets_dict = {
                    k: v.to(device, dtype=torch.float32)
                    for k, v in data_dict.items()
                    if k != "image"
                }

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

                        inputs = data_dict["image"].to(device, dtype=torch.float32)
                        targets_dict = {
                            k: v.to(device, dtype=torch.float32)
                            for k, v in data_dict.items()
                            if k != "image"
                        }

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