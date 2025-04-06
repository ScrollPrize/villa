import os
import numpy as np
import torch
from torch import autocast, nn
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from PIL import Image, ImageDraw, ImageFont  # for adding text overlays


class nnUNetTrainerWorkshop(nnUNetTrainerDA5):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # Save GIFs in the same folder as checkpoints (self.output_folder)
        self.visualization_folder = self.output_folder

    def normalize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize a 2D image to uint8 (0-255). If the image is nearly constant, returns a zero image.
        """
        img = img.astype(np.float32)
        min_val = np.min(img)
        max_val = np.max(img)
        if max_val - min_val > 1e-8:
            norm = (img - min_val) / (max_val - min_val)
        else:
            norm = img - min_val
        norm = (norm * 255).astype(np.uint8)
        return norm

    def save_gif_from_train_val_batches(self, train_batch: dict, train_logits: torch.Tensor,
                                          val_batch: dict, val_logits: torch.Tensor, filename: str):
        """
        Save an animated GIF for a single 3D volume (the first case in each batch) using predictions.
        Each frame corresponds to one z-slice.
        Each frame is a composite image with two rows:
          - Top row (Train): raw, target, and prediction from the training batch
          - Bottom row (Val): raw, target, and prediction from the validation batch
        A text label ("Train" or "Val") is added to the corresponding row.
        """
        # ===== Process the training batch =====
        # Convert raw data tensor to numpy.
        train_raw_np = train_batch['data'].cpu().numpy()
        # Handle target: if it's a list, use the first element.
        train_target_obj = train_batch['target']
        if isinstance(train_target_obj, list):
            train_target_np = train_target_obj[0].cpu().numpy()
        else:
            train_target_np = train_target_obj.cpu().numpy()
        train_logits_np = train_logits.cpu().numpy()

        # Use the first case in the batch.
        train_raw_vol = train_raw_np[0]  # expected shape: [C, Z, H, W] or [Z, H, W]
        train_target_vol = train_target_np[0]
        train_logit_vol = train_logits_np[0]  # shape: [C, Z, H, W]

        # If there is a channel dimension, take the first channel for raw and target,
        # and for logits choose channel 1 (foreground).
        if train_raw_vol.ndim == 4:
            train_raw_vol = train_raw_vol[0]
        if train_target_vol.ndim == 4:
            train_target_vol = train_target_vol[0]
        if train_logit_vol.ndim == 4:
            train_logit_vol = train_logit_vol[1]

        # ===== Process the validation batch =====
        val_raw_np = val_batch['data'].cpu().numpy()
        val_target_obj = val_batch['target']
        if isinstance(val_target_obj, list):
            val_target_np = val_target_obj[0].cpu().numpy()
        else:
            val_target_np = val_target_obj.cpu().numpy()
        val_logits_np = val_logits.cpu().numpy()

        val_raw_vol = val_raw_np[0]
        val_target_vol = val_target_np[0]
        val_logit_vol = val_logits_np[0]

        if val_raw_vol.ndim == 4:
            val_raw_vol = val_raw_vol[0]
        if val_target_vol.ndim == 4:
            val_target_vol = val_target_vol[0]
        if val_logit_vol.ndim == 4:
            val_logit_vol = val_logit_vol[1]

        # ===== Build composite frames =====
        # Assume the number of slices is the same for train and validation volumes.
        num_slices = train_raw_vol.shape[0]
        frames = []
        # Prepare a font for overlaying text.
        font = ImageFont.load_default()

        for z in range(num_slices):
            # --- Train row ---
            train_raw_slice = train_raw_vol[z]
            train_target_slice = train_target_vol[z]
            train_logit_slice = train_logit_vol[z]

            train_raw_norm = self.normalize_image(train_raw_slice)
            if train_target_slice.max() > 1:
                train_target_norm = self.normalize_image(train_target_slice)
            else:
                train_target_norm = (train_target_slice * 255).astype(np.uint8)
            train_logit_norm = self.normalize_image(train_logit_slice)

            # Concatenate horizontally: Raw | Target | Prediction.
            train_row = np.concatenate((train_raw_norm, train_target_norm, train_logit_norm), axis=1)
            # Convert to PIL image to add text.
            train_img = Image.fromarray(train_row).convert("RGB")
            draw_train = ImageDraw.Draw(train_img)
            draw_train.text((5, 5), "Train", fill=(255, 0, 0), font=font)  # red text
            train_row_with_text = np.array(train_img)

            # --- Validation row ---
            val_raw_slice = val_raw_vol[z]
            val_target_slice = val_target_vol[z]
            val_logit_slice = val_logit_vol[z]

            val_raw_norm = self.normalize_image(val_raw_slice)
            if val_target_slice.max() > 1:
                val_target_norm = self.normalize_image(val_target_slice)
            else:
                val_target_norm = (val_target_slice * 255).astype(np.uint8)
            val_logit_norm = self.normalize_image(val_logit_slice)

            val_row = np.concatenate((val_raw_norm, val_target_norm, val_logit_norm), axis=1)
            val_img = Image.fromarray(val_row).convert("RGB")
            draw_val = ImageDraw.Draw(val_img)
            draw_val.text((5, 5), "Val", fill=(0, 255, 0), font=font)  # green text
            val_row_with_text = np.array(val_img)

            # --- Combine train and validation rows vertically ---
            composite = np.concatenate((train_row_with_text, val_row_with_text), axis=0)
            frames.append(composite)

        # ----- Save as animated GIF using PIL to avoid frame artifacts -----
        # Convert each frame (a NumPy array) to a PIL Image in RGB.
        pil_frames = [Image.fromarray(frame).convert("RGB") for frame in frames]
        # Save the frames as a GIF.
        # The 'disposal=2' setting ensures that each frame is fully cleared before drawing the next.
        pil_frames[0].save(
            filename,
            save_all=True,
            append_images=pil_frames[1:],
            loop=0,
            duration=100,  # duration in milliseconds (adjust as needed)
            disposal=2
        )

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            first_train_batch = None
            for batch_id in range(self.num_iterations_per_epoch):
                batch = next(self.dataloader_train)
                # Save the first training batch for visualization.
                if batch_id == 0:
                    first_train_batch = batch
                train_outputs.append(self.train_step(batch))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []

                # Get the first validation batch.
                first_val_batch = next(self.dataloader_val)

                # --- Compute predictions for the training batch ---
                train_data = first_train_batch['data'].to(self.device, non_blocking=True)
                with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                    train_output = self.network(train_data)
                if self.enable_deep_supervision:
                    train_output = train_output[0]
                train_probs = torch.softmax(train_output, dim=1)

                # --- Compute predictions for the validation batch ---
                val_data = first_val_batch['data'].to(self.device, non_blocking=True)
                with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                    val_output = self.network(val_data)
                if self.enable_deep_supervision:
                    val_output = val_output[0]
                val_probs = torch.softmax(val_output, dim=1)

                # Save the GIF showing both train and validation batches.
                gif_filename = os.path.join(self.output_folder, f'epoch_{epoch}_train_val.gif')
                self.save_gif_from_train_val_batches(first_train_batch, train_probs,
                                                     first_val_batch, val_probs, gif_filename)
                self.print_to_log_file(f"Saved GIF visualization to {gif_filename}")

                val_outputs.append(self.validation_step(first_val_batch))
                for batch_id in range(1, self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
