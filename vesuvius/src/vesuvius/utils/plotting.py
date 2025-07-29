import cv2
import imageio
import numpy as np
import torch
from pathlib import Path
import importlib

def apply_activation_if_needed(scalar_or_vector, activation_str):
    if not activation_str or activation_str.lower() == "none":
        return scalar_or_vector

    if activation_str.lower() == "sigmoid":
        return 1.0 / (1.0 + np.exp(-scalar_or_vector))

    return scalar_or_vector

def add_text_label(img: np.ndarray, label: str, position: str = "top") -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    color = (255, 255, 255)  # White text

    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    img_height, img_width = img.shape[:2]
    needed_width = text_width + 10

    if img_width < needed_width:
        padding_width = needed_width - img_width
        img = np.pad(img, ((0, 0), (0, padding_width), (0, 0)), mode='constant', constant_values=0)

    img_with_label = img.copy()
    
    if position == "top":
        cv2.rectangle(img_with_label, (0, 0), (text_width + 10, text_height + baseline + 10), (0, 0, 0), -1)
        cv2.putText(img_with_label, label, (5, text_height + 5), font, font_scale, color, thickness)
    
    return img_with_label

def convert_slice_to_bgr(
    slice_2d_or_3d: np.ndarray,
    show_magnitude: bool = False,
    dynamic_range: bool = True
) -> np.ndarray:


    def minmax_scale_to_8bit(img: np.ndarray) -> np.ndarray:
        min_val = img.min()
        max_val = img.max()
        eps = 1e-6
        if (max_val - min_val) > eps:
            img_scaled = (img - min_val) / (max_val - min_val)
        else:
            # Instead of zeros, you could use a constant value:
            img_scaled = np.full_like(img, 0.5, dtype=np.float32)
        return (img_scaled * 255).astype(np.uint8)

    # -----------------------------------------
    # Case 1: Single-channel [H, W]
    # -----------------------------------------
    if slice_2d_or_3d.ndim == 2:
        if dynamic_range:
            slice_8u = minmax_scale_to_8bit(slice_2d_or_3d)
        else:
            # Old clamp approach
            slice_clamped = np.clip(slice_2d_or_3d, 0, 1)
            slice_8u = (slice_clamped * 255).astype(np.uint8)
        return cv2.cvtColor(slice_8u, cv2.COLOR_GRAY2BGR)

    # -----------------------------------------
    # Case 2: Multi-channel [C, H, W]
    # -----------------------------------------
    elif slice_2d_or_3d.ndim == 3:
        if slice_2d_or_3d.shape[0] == 3:
            # shape => [3, H, W]
            if dynamic_range:
                # Per-channel local min..max
                ch_list = []
                for ch_idx in range(3):
                    ch_8u = minmax_scale_to_8bit(slice_2d_or_3d[ch_idx])
                    ch_list.append(ch_8u)
                mapped_normals = np.stack(ch_list, axis=0)  # shape => [3, H, W]
            else:
                # Old clamp approach for normals => [-1..1] mapped to [0..255]
                normal_slice = np.clip(slice_2d_or_3d, -1, 1)
                mapped_normals = ((normal_slice * 0.5) + 0.5) * 255
                mapped_normals = np.clip(mapped_normals, 0, 255).astype(np.uint8)

            # Reorder to [H, W, 3]
            bgr_normals = np.transpose(mapped_normals, (1, 2, 0))

            if show_magnitude:
                mag = np.linalg.norm(slice_2d_or_3d, axis=0)  # => [H, W]
                if dynamic_range:
                    mag_8u = minmax_scale_to_8bit(mag)
                else:
                    mag_8u = minmax_scale_to_8bit(mag)
                mag_bgr = cv2.cvtColor(mag_8u, cv2.COLOR_GRAY2BGR)
                return np.hstack([mag_bgr, bgr_normals])

            return bgr_normals

        else:
            if slice_2d_or_3d.shape[0] == 2:
                ch_8u = minmax_scale_to_8bit(slice_2d_or_3d[1])
            else:
                ch_8u = minmax_scale_to_8bit(slice_2d_or_3d[0])
            return cv2.cvtColor(ch_8u, cv2.COLOR_GRAY2BGR)

    else:
        raise ValueError(
            f"convert_slice_to_bgr expects shape [H, W] or [C, H, W] where C is the number of channels, got {slice_2d_or_3d.shape}"
        )

def save_debug(
    input_volume: torch.Tensor,          # shape [1, C, Z, H, W] for 3D or [1, C, H, W] for 2D
    targets_dict: dict,                 # e.g. {"sheet": tensor([1, Z, H, W]), "normals": tensor([3, Z, H, W])}
    outputs_dict: dict,                 # same shape structure
    tasks_dict: dict,                   # e.g. {"sheet": {"activation":"sigmoid"}, "normals": {"activation":"none"}}
    epoch: int,
    save_path: str = "debug.gif",       # Will be modified to PNG for 2D data
    show_normal_magnitude: bool = True, # We'll set this to False below to avoid extra sub-panels
    fps: int = 5,
    train_input: torch.Tensor = None,   # Optional train sample input
    train_targets_dict: dict = None,    # Optional train sample targets
    train_outputs_dict: dict = None     # Optional train sample outputs
):

    inp_np = input_volume.cpu().numpy()[0]  # => shape [C, Z, H, W] for 3D or [C, H, W] for 2D
    is_2d = len(inp_np.shape) == 3  # [C, H, W] format for 2D data
    
    if is_2d:
        save_path = save_path.replace('.gif', '.png')
    
    if inp_np.shape[0] == 1:
        # single-channel => shape [Z, H, W] for 3D or [H, W] for 2D
        inp_np = inp_np[0]

    targets_np, preds_np = {}, {}
    for t_name, t_tensor in targets_dict.items():
        arr_np = t_tensor.cpu().numpy()  # Remove [0] indexing - we'll handle it below

        loss_fn = tasks_dict[t_name].get("loss_fn", "")
        if loss_fn == "CrossEntropyLoss":
            # For CrossEntropyLoss, targets are class indices, not one-hot
            # First, handle batch dimension if present
            if arr_np.ndim == 4:  # [B, 1, Z, H, W] for 3D
                arr_np = arr_np[0, 0]  # => [Z, H, W]
            elif arr_np.ndim == 3 and is_2d:  # [B, 1, H, W] for 2D
                arr_np = arr_np[0, 0]  # => [H, W]
            elif arr_np.ndim == 3 and not is_2d:  # [B, Z, H, W] for 3D
                arr_np = arr_np[0]  # => [Z, H, W]
            elif arr_np.ndim == 2:  # Already [H, W] for 2D
                pass
            else:
                while arr_np.ndim > (2 if is_2d else 3):
                    arr_np = arr_np[0]

            num_classes = tasks_dict[t_name].get("out_channels", 2)
            arr_np = arr_np.astype(int)
            
            if is_2d:
                # 2D: arr_np shape is [H, W]
                one_hot = np.eye(num_classes)[arr_np]  # [H, W, num_classes]
                arr_np = one_hot.transpose(2, 0, 1)  # [num_classes, H, W]
            else:
                # 3D: arr_np shape is [Z, H, W]
                one_hot = np.eye(num_classes)[arr_np]  # [Z, H, W, num_classes]
                arr_np = one_hot.transpose(3, 0, 1, 2)  # [num_classes, Z, H, W]
        else:
            # For non-CrossEntropyLoss targets, extract batch dimension
            if arr_np.ndim > (3 if is_2d else 4):
                arr_np = arr_np[0]  # Remove batch dimension
            elif arr_np.ndim == (3 if is_2d else 4):
                arr_np = arr_np[0]  # Remove batch dimension
                
        targets_np[t_name] = arr_np

    for t_name, p_tensor in outputs_dict.items():
        arr_np = p_tensor.cpu().numpy()[0]  # => [C, Z, H, W] for 3D or [C, H, W] for 2D
        activation_str = tasks_dict[t_name].get("activation", "none")
        if arr_np.shape[0] == 1:
            arr_np = apply_activation_if_needed(arr_np, activation_str)
        preds_np[t_name] = arr_np

    train_inp_np = None
    train_targets_np = {}
    train_preds_np = {}
    
    if train_input is not None and train_targets_dict is not None and train_outputs_dict is not None:

        train_inp_np = train_input.cpu().numpy()[0]  # => shape [C, Z, H, W] for 3D or [C, H, W] for 2D
        if train_inp_np.shape[0] == 1:
            train_inp_np = train_inp_np[0]

        for t_name, t_tensor in train_targets_dict.items():
            arr_np = t_tensor.cpu().numpy()

            loss_fn = tasks_dict[t_name].get("loss_fn", "")
            if loss_fn == "CrossEntropyLoss":
                if arr_np.ndim == 4:  # [B, 1, Z, H, W] for 3D
                    arr_np = arr_np[0, 0]  # => [Z, H, W]
                elif arr_np.ndim == 3 and is_2d:  # [B, 1, H, W] for 2D
                    arr_np = arr_np[0, 0]  # => [H, W]
                elif arr_np.ndim == 3 and not is_2d:  # [B, Z, H, W] for 3D
                    arr_np = arr_np[0]  # => [Z, H, W]
                elif arr_np.ndim == 2:  # Already [H, W] for 2D
                    pass
                else:
                    while arr_np.ndim > (2 if is_2d else 3):
                        arr_np = arr_np[0]

                num_classes = tasks_dict[t_name].get("out_channels", 2)
                arr_np = arr_np.astype(int)
                
                if is_2d:
                    one_hot = np.eye(num_classes)[arr_np]  # [H, W, num_classes]
                    arr_np = one_hot.transpose(2, 0, 1)  # [num_classes, H, W]
                else:
                    one_hot = np.eye(num_classes)[arr_np]  # [Z, H, W, num_classes]
                    arr_np = one_hot.transpose(3, 0, 1, 2)  # [num_classes, Z, H, W]
            else:
                if arr_np.ndim > (3 if is_2d else 4):
                    arr_np = arr_np[0]
                elif arr_np.ndim == (3 if is_2d else 4):
                    arr_np = arr_np[0]
                    
            train_targets_np[t_name] = arr_np

        for t_name, p_tensor in train_outputs_dict.items():
            arr_np = p_tensor.cpu().numpy()[0]  # => [C, Z, H, W] for 3D or [C, H, W] for 2D
            activation_str = tasks_dict[t_name].get("activation", "none")
            if arr_np.shape[0] == 1:
                arr_np = apply_activation_if_needed(arr_np, activation_str)
            train_preds_np[t_name] = arr_np

    show_normal_magnitude = False

    if is_2d:
        rows = []
        task_names = sorted(list(targets_dict.keys()))
        
        # -----------------------------
        # VAL ROW 1: [Val Input] + [Val GT tasks]
        # -----------------------------
        val_row1_imgs = []

        inp_slice = inp_np  # Already in correct shape for 2D
        val_input_img = add_text_label(convert_slice_to_bgr(inp_slice), "Val Input")
        val_row1_imgs.append(val_input_img)

        for t_name in task_names:
            gt_slice = targets_np[t_name]
            if gt_slice.shape[0] == 1:
                slice_2d = gt_slice[0]  # shape => [H, W]
            else:
                slice_2d = gt_slice  # shape => [C, H, W]
            gt_img = add_text_label(convert_slice_to_bgr(slice_2d), f"Val GT {t_name}")
            val_row1_imgs.append(gt_img)
        
        rows.append(np.hstack(val_row1_imgs))
        
        # ----------------------------------------
        # VAL ROW 2: [blank tile] + [Val pred tasks]
        # ----------------------------------------
        val_row2_imgs = []

        blank_tile = np.zeros_like(convert_slice_to_bgr(inp_slice))
        val_row2_imgs.append(blank_tile)

        for t_name in task_names:
            pd_slice = preds_np[t_name]
            if pd_slice.shape[0] == 1:
                slice_2d = pd_slice[0]  # shape => [H, W]
                bgr_pred = convert_slice_to_bgr(slice_2d)
            else:
                bgr_pred = convert_slice_to_bgr(pd_slice, show_magnitude=show_normal_magnitude)
            pred_img = add_text_label(bgr_pred, f"Val Pred {t_name}")
            val_row2_imgs.append(pred_img)
        
        rows.append(np.hstack(val_row2_imgs))

        if train_inp_np is not None:
            # -----------------------------
            # TRAIN ROW 1: [Train Input] + [Train GT tasks]
            # -----------------------------
            train_row1_imgs = []

            train_inp_slice = train_inp_np if train_inp_np.ndim == 2 else train_inp_np
            train_input_img = add_text_label(convert_slice_to_bgr(train_inp_slice), "Train Input")
            train_row1_imgs.append(train_input_img)

            for t_name in task_names:
                train_gt_slice = train_targets_np[t_name]
                if train_gt_slice.shape[0] == 1:
                    slice_2d = train_gt_slice[0]  # shape => [H, W]
                else:
                    slice_2d = train_gt_slice  # shape => [C, H, W]
                train_gt_img = add_text_label(convert_slice_to_bgr(slice_2d), f"Train GT {t_name}")
                train_row1_imgs.append(train_gt_img)
            
            rows.append(np.hstack(train_row1_imgs))
            
            # ----------------------------------------
            # TRAIN ROW 2: [blank tile] + [Train pred tasks]
            # ----------------------------------------
            train_row2_imgs = []

            blank_tile = np.zeros_like(convert_slice_to_bgr(train_inp_slice))
            train_row2_imgs.append(blank_tile)

            for t_name in task_names:
                train_pd_slice = train_preds_np[t_name]
                if train_pd_slice.shape[0] == 1:
                    slice_2d = train_pd_slice[0]  # shape => [H, W]
                    bgr_pred = convert_slice_to_bgr(slice_2d)
                else:
                    bgr_pred = convert_slice_to_bgr(train_pd_slice, show_magnitude=show_normal_magnitude)
                train_pred_img = add_text_label(bgr_pred, f"Train Pred {t_name}")
                train_row2_imgs.append(train_pred_img)
            
            rows.append(np.hstack(train_row2_imgs))

        max_width = max(row.shape[1] for row in rows)
        padded_rows = []
        for row in rows:
            if row.shape[1] < max_width:
                padding = max_width - row.shape[1]
                row = np.pad(row, ((0, 0), (0, padding), (0, 0)), mode='constant', constant_values=0)
            padded_rows.append(row)

        final_img = np.vstack(padded_rows)
        
        # Save as PNG
        out_dir = Path(save_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Epoch {epoch}] Saving PNG to: {save_path}")
        imageio.imwrite(save_path, final_img)
        
    else:
        frames = []
        z_dim = inp_np.shape[0] if inp_np.ndim == 3 else inp_np.shape[1]
        task_names = sorted(list(targets_dict.keys()))

        for z_idx in range(z_dim):
            rows = []
            
            # -----------------------------
            # VAL ROW 1: [Val Input] + [Val GT tasks]
            # -----------------------------
            val_row1_imgs = []

            if inp_np.ndim == 3:
                inp_slice = inp_np[z_idx]          # shape => [H, W]
            else:
                inp_slice = inp_np[:, z_idx, :, :] # shape => [C, H, W]
            val_input_img = add_text_label(convert_slice_to_bgr(inp_slice), "Val Input")
            val_row1_imgs.append(val_input_img)

            for t_name in task_names:
                gt_slice = targets_np[t_name]
                if gt_slice.shape[0] == 1:
                    slice_2d = gt_slice[0, z_idx, :, :]  # shape => [H, W]
                else:
                    slice_2d = gt_slice[:, z_idx, :, :]  # shape => [3, H, W] or however
                gt_img = add_text_label(convert_slice_to_bgr(slice_2d), f"Val GT {t_name}")
                val_row1_imgs.append(gt_img)

            rows.append(np.hstack(val_row1_imgs))

            # ----------------------------------------
            # VAL ROW 2: [blank tile] + [Val pred tasks]
            # ----------------------------------------
            val_row2_imgs = []

            blank_tile = np.zeros_like(convert_slice_to_bgr(inp_slice))
            val_row2_imgs.append(blank_tile)

            for t_name in task_names:
                pd_slice = preds_np[t_name]
                if pd_slice.shape[0] == 1:
                    slice_2d = pd_slice[0, z_idx, :, :]
                    bgr_pred = convert_slice_to_bgr(slice_2d)
                else:
                    slice_3d = pd_slice[:, z_idx, :, :]
                    bgr_pred = convert_slice_to_bgr(slice_3d, show_magnitude=show_normal_magnitude)
                pred_img = add_text_label(bgr_pred, f"Val Pred {t_name}")
                val_row2_imgs.append(pred_img)

            rows.append(np.hstack(val_row2_imgs))

            if train_inp_np is not None:
                # -----------------------------
                # TRAIN ROW 1: [Train Input] + [Train GT tasks]
                # -----------------------------
                train_row1_imgs = []

                if train_inp_np.ndim == 3:
                    train_inp_slice = train_inp_np[z_idx]          # shape => [H, W]
                else:
                    train_inp_slice = train_inp_np[:, z_idx, :, :] # shape => [C, H, W]
                train_input_img = add_text_label(convert_slice_to_bgr(train_inp_slice), "Train Input")
                train_row1_imgs.append(train_input_img)

                for t_name in task_names:
                    train_gt_slice = train_targets_np[t_name]
                    if train_gt_slice.shape[0] == 1:
                        slice_2d = train_gt_slice[0, z_idx, :, :]  # shape => [H, W]
                    else:
                        slice_2d = train_gt_slice[:, z_idx, :, :]  # shape => [3, H, W] or however
                    train_gt_img = add_text_label(convert_slice_to_bgr(slice_2d), f"Train GT {t_name}")
                    train_row1_imgs.append(train_gt_img)
                
                rows.append(np.hstack(train_row1_imgs))
                
                # ----------------------------------------
                # TRAIN ROW 2: [blank tile] + [Train pred tasks]
                # ----------------------------------------
                train_row2_imgs = []

                blank_tile = np.zeros_like(convert_slice_to_bgr(train_inp_slice))
                train_row2_imgs.append(blank_tile)

                for t_name in task_names:
                    train_pd_slice = train_preds_np[t_name]
                    if train_pd_slice.shape[0] == 1:
                        slice_2d = train_pd_slice[0, z_idx, :, :]
                        bgr_pred = convert_slice_to_bgr(slice_2d)
                    else:
                        slice_3d = train_pd_slice[:, z_idx, :, :]
                        bgr_pred = convert_slice_to_bgr(slice_3d, show_magnitude=show_normal_magnitude)
                    train_pred_img = add_text_label(bgr_pred, f"Train Pred {t_name}")
                    train_row2_imgs.append(train_pred_img)
                
                rows.append(np.hstack(train_row2_imgs))

            max_width = max(row.shape[1] for row in rows)
            padded_rows = []
            for row in rows:
                if row.shape[1] < max_width:
                    padding = max_width - row.shape[1]
                    row = np.pad(row, ((0, 0), (0, padding), (0, 0)), mode='constant', constant_values=0)
                padded_rows.append(row)

            final_img = np.vstack(padded_rows)
            frames.append(final_img)

        out_dir = Path(save_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Epoch {epoch}] Saving GIF to: {save_path}")
        imageio.mimsave(save_path, frames, fps=fps)

        # Return frames array for wandb logging
        # Convert list of BGR images to array with shape (frames, height, width, channels)
        frames_array = np.stack(frames, axis=0)
        # Convert BGR to RGB for wandb
        frames_array = frames_array[..., ::-1]
        # Transpose to (frames, channels, height, width) as required by wandb
        frames_array = np.transpose(frames_array, (0, 3, 1, 2))
        return frames_array

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import tifffile
def export_data_dict_as_tif(
    dataset,
    num_batches: int = 5,
    out_dir: str = "debug_tifs"
):
    """
    Writes each entry in `data_dict` to a multi-page TIFF, one file per key.
    Assumes batch_size=1 => shape [B, C, D, H, W].
    The output TIFF for each key has shape [C*D, H, W] (multi-page stack),
    preserving exact values (no scaling or axis reorder).
    """
    os.makedirs(out_dir, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, data_dict in enumerate(loader):
        if i >= num_batches:
            break

        # We'll assume batch_size=1 for simpler visualization
        for key, tensor in data_dict.items():
            # shape => [B, C, D, H, W]. Take B=0
            arr_4d = tensor[0].cpu().numpy()  # shape => [C, D, H, W]

            # Flatten [C,D] into one dimension: => [C*D, H, W]
            c, d, h, w = arr_4d.shape
            arr_pages = arr_4d.reshape(c * d, h, w)

            # Write the multi-page TIFF exactly as-is
            out_path = os.path.join(out_dir, f"batch_{i}_{key}.tif")
            tifffile.imwrite(out_path, arr_pages, dtype=arr_pages.dtype)

            print(f"Wrote {out_path} with shape {arr_pages.shape} "
                  f"(original [C,D,H,W] => [C*D,H,W]).")
