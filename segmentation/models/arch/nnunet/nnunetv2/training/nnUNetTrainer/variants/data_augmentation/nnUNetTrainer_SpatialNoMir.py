from typing import Union, Tuple, List
import os

import numpy as np
import torch
from torch import autocast, nn
from PIL import Image, ImageDraw, ImageFont

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import nnUNetTrainerNoMirroring

from batchgeneratorsv2.transforms.spatial.sinusoidal_wave import SineWaveDeformation
from batchgeneratorsv2.transforms.spatial.rotation import ArbitraryRotationTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.intensity.illumination import InhomogeneousSliceIlluminationTransform
from batchgeneratorsv2.transforms.noise.extranoisetransforms import BlankRectangleTransform
from nnunetv2.utilities.helpers import empty_cache, dummy_context

class nnUNetTrainer_SpatialNoMir(nnUNetTrainerNoMirroring):
    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.0,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False, mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.3),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        transforms.append(RandomTransform(
            BlankRectangleTransform(
                rectangle_size=((max(1, patch_size[0] // 5), patch_size[0] // 3),
                                (max(1, patch_size[1] // 5), patch_size[1] // 3),
                                (max(1, patch_size[2] // 5), patch_size[2] // 3)),
                rectangle_value=np.mean,
                num_rectangles=(1, 5),
                force_square=False,
                p_per_sample=0.4,
                p_per_channel=0.5
            ), apply_probability=0.20
        ))
        transforms.append(RandomTransform(
            InhomogeneousSliceIlluminationTransform(
                num_defects=(2, 5),  # Range for number of defects
                defect_width=(5, 50),  # Range for defect width
                mult_brightness_reduction_at_defect=(0.3, 0.7),  # Range for brightness reduction
                base_p=(0.2, 0.4),  # Base probability range
                base_red=(0.5, 0.9),  # Base reduction range
                p_per_sample=1.0,  # Probability per sample
                per_channel=True,  # Apply per channel
                p_per_channel=0.5  # Probability per channel
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            ArbitraryRotationTransform(
            rotation_angle_range=(-np.pi / 8, np.pi / 8),  # +/- 15 degrees
            p_per_axis=0.5  # 50% chance of rotation per axis
            ), apply_probability=0.75
        ))
        transforms.append(RandomTransform(
            SineWaveDeformation(
                    min_peaks=1,
                    max_peaks=2,
                    min_magnitude=0.0,
                    max_magnitude=0.50,
                    boundary_mode='constant',  # 'constant', 'nearest', 'reflect', or 'mirror'
                    constant_value=0.5,  # Fill value when boundary_mode is 'constant'
                    single_axis=True,  # Use same random axis for all waves
                    fixed_axis=None # random axis to apply
                ), apply_probability=0.15
        ))


        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)


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