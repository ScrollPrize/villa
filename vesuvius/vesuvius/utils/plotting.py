import cv2
import imageio
import numpy as np
import torch
from pathlib import Path
import importlib

def apply_activation_if_needed(scalar_or_vector, activation_str):
    """
    scalar_or_vector: a NumPy array (e.g., shape [D, H, W] or [C, D, H, W])
    activation_str: e.g., "none", "sigmoid", "softmax", etc.

    Returns a NumPy array with the activation applied if appropriate.
    """
    if not activation_str or activation_str.lower() == "none":
        return scalar_or_vector

    if activation_str.lower() == "sigmoid":
        return 1.0 / (1.0 + np.exp(-scalar_or_vector))

    # If "softmax" is needed, implement as appropriate
    return scalar_or_vector

def convert_slice_to_bgr(
    slice_2d_or_3d: np.ndarray,
    show_magnitude: bool = False,
    dynamic_range: bool = True
) -> np.ndarray:
    """
    Converts a slice (single-channel, multi-channel, or 3-channel normal) into a BGR image
    by doing min..max scaling to the full range of values present in *that slice*.

    Args:
        slice_2d_or_3d (np.ndarray): shape [H, W] for single-channel, [3, H, W] for 3-channel (processed as BGR), 
                                     or [C, H, W] for any number of channels (each channel visualized side by side).
        show_magnitude (bool): If True and data is 3-channel, we horizontally stack an extra panel
                               showing that slice's magnitude, also min–max scaled to [0..255].
        dynamic_range (bool): If True, we scale by the slice's min and max. If False, we clamp
                              to [-1..1] for 3-channel (old approach) or [0..1] for single-channel.

    Returns:
        np.ndarray: BGR image of shape [H, W, 3], or shape [H, 2*W, 3] if magnitude panel is shown,
                    or shape [H, C*W, 3] for multi-channel data with C channels.
    """

    def minmax_scale_to_8bit(img: np.ndarray) -> np.ndarray:
        """Scales `img` so that its min -> 0 and max -> 255 for this slice only."""
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
        # Special handling for 3-channel data (likely normals)
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

            # Optionally add a separate magnitude panel
            if show_magnitude:
                # Magnitude of the original float channels
                mag = np.linalg.norm(slice_2d_or_3d, axis=0)  # => [H, W]
                if dynamic_range:
                    mag_8u = minmax_scale_to_8bit(mag)
                else:
                    # If you like, you could do the same [-1..1] clamp, but magnitude rarely is negative
                    mag_8u = minmax_scale_to_8bit(mag)
                mag_bgr = cv2.cvtColor(mag_8u, cv2.COLOR_GRAY2BGR)
                # Combine horizontally: [H, W + W, 3]
                return np.hstack([mag_bgr, bgr_normals])

            return bgr_normals
        
        # Other multi-channel cases (2-channel, 4-channel, etc.)
        else:
            # For 2-channel data (one-hot encoded binary segmentation), show the foreground channel
            if slice_2d_or_3d.shape[0] == 2:
                # Channel 1 is the foreground channel in one-hot encoding
                ch_8u = minmax_scale_to_8bit(slice_2d_or_3d[1])
            else:
                # For other multi-channel cases, show the first channel
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
    fps: int = 5
):
    """
    Creates debug visualizations for both 2D and 3D data.
    
    For 3D data:
    - Creates a multi-panel GIF for debugging by animating through z-slices.
    
    For 2D data:
    - Creates a single image (PNG) with the same panel layout.
    
    The top row will show: [Input slice] + [GT for each task].
    The bottom row will show: [blank tile] + [Prediction for each task].
    If a task has 3 channels (normals), we visualize them as a single BGR.
    """

    # Convert input volume to NumPy
    inp_np = input_volume.cpu().numpy()[0]  # => shape [C, Z, H, W] for 3D or [C, H, W] for 2D

    # Detect if the input is 2D or 3D
    is_2d = len(inp_np.shape) == 3  # [C, H, W] format for 2D data
    
    if is_2d:
        # For 2D data, override the save path extension to PNG
        save_path = save_path.replace('.gif', '.png')
    
    if inp_np.shape[0] == 1:
        # single-channel => shape [Z, H, W] for 3D or [H, W] for 2D
        inp_np = inp_np[0]

    # Convert targets & predictions to NumPy (and apply activation if needed)
    targets_np, preds_np = {}, {}
    for t_name, t_tensor in targets_dict.items():
        arr_np = t_tensor.cpu().numpy()  # Remove [0] indexing - we'll handle it below
        
        # Check if this is a CrossEntropyLoss target (class indices)
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
                # Try to extract the first element
                while arr_np.ndim > (2 if is_2d else 3):
                    arr_np = arr_np[0]
            
            # We'll convert to one-hot for visualization
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

    # If you want to remove the normal magnitude sub-panel, set show_normal_magnitude=False:
    show_normal_magnitude = False

    # Determine if we're processing 2D or 3D data
    if is_2d:
        # Process 2D data - create a single image
        # -----------------------------
        # TOP ROW: [Input] + [GT tasks]
        # -----------------------------
        top_row_imgs = []

        # 1) Input slice for 2D
        inp_slice = inp_np  # Already in correct shape for 2D
        top_row_imgs.append(convert_slice_to_bgr(inp_slice))

        # 2) Each GT for 2D
        task_names = sorted(list(targets_dict.keys()))
        for t_name in task_names:
            gt_slice = targets_np[t_name]
            if gt_slice.shape[0] == 1:
                slice_2d = gt_slice[0]  # shape => [H, W]
            else:
                slice_2d = gt_slice  # shape => [C, H, W]
            top_row_imgs.append(convert_slice_to_bgr(slice_2d))

        top_row = np.hstack(top_row_imgs)

        # ----------------------------------------
        # BOTTOM ROW: [blank tile] + [pred tasks]
        # ----------------------------------------
        bottom_row_imgs = []

        # 1) A blank tile that matches the input-slice shape
        blank_tile = np.zeros_like(convert_slice_to_bgr(inp_slice))
        bottom_row_imgs.append(blank_tile)

        # 2) Predictions for each task - 2D
        for t_name in task_names:
            pd_slice = preds_np[t_name]
            if pd_slice.shape[0] == 1:
                slice_2d = pd_slice[0]  # shape => [H, W]
                bgr_pred = convert_slice_to_bgr(slice_2d)
                bottom_row_imgs.append(bgr_pred)
            else:
                # For multi-channel 2D data
                bgr_normals = convert_slice_to_bgr(pd_slice, show_magnitude=show_normal_magnitude)
                bottom_row_imgs.append(bgr_normals)

        bottom_row = np.hstack(bottom_row_imgs)

        # Vertical stack -> final image 
        final_img = np.vstack([top_row, bottom_row])
        
        # Save as PNG
        out_dir = Path(save_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Epoch {epoch}] Saving PNG to: {save_path}")
        imageio.imwrite(save_path, final_img)
        
    else:
        # Process 3D data - create a GIF
        frames = []
        z_dim = inp_np.shape[0] if inp_np.ndim == 3 else inp_np.shape[1]
        task_names = sorted(list(targets_dict.keys()))

        for z_idx in range(z_dim):
            # -----------------------------
            # TOP ROW: [Input] + [GT tasks]
            # -----------------------------
            top_row_imgs = []

            # 1) Input slice
            if inp_np.ndim == 3:
                inp_slice = inp_np[z_idx]          # shape => [H, W]
            else:
                inp_slice = inp_np[:, z_idx, :, :] # shape => [C, H, W]
            top_row_imgs.append(convert_slice_to_bgr(inp_slice))

            # 2) Each GT
            for t_name in task_names:
                gt_slice = targets_np[t_name]
                if gt_slice.shape[0] == 1:
                    slice_2d = gt_slice[0, z_idx, :, :]  # shape => [H, W]
                else:
                    slice_2d = gt_slice[:, z_idx, :, :]  # shape => [3, H, W] or however
                top_row_imgs.append(convert_slice_to_bgr(slice_2d))

            top_row = np.hstack(top_row_imgs)

            # ----------------------------------------
            # BOTTOM ROW: [blank tile] + [pred tasks]
            # ----------------------------------------
            bottom_row_imgs = []

            # 1) A blank tile that matches the input-slice shape
            blank_tile = np.zeros_like(convert_slice_to_bgr(inp_slice))
            bottom_row_imgs.append(blank_tile)

            # 2) Predictions for each task
            for t_name in task_names:
                pd_slice = preds_np[t_name]
                if pd_slice.shape[0] == 1:
                    slice_2d = pd_slice[0, z_idx, :, :]
                    bgr_pred = convert_slice_to_bgr(slice_2d)
                    bottom_row_imgs.append(bgr_pred)
                else:
                    slice_3d = pd_slice[:, z_idx, :, :]
                    # Because we set show_normal_magnitude=False,
                    # this should return just one BGR panel
                    bgr_normals = convert_slice_to_bgr(slice_3d, show_magnitude=show_normal_magnitude)
                    bottom_row_imgs.append(bgr_normals)

            bottom_row = np.hstack(bottom_row_imgs)

            # Vertical stack -> final image for this slice
            final_img = np.vstack([top_row, bottom_row])
            frames.append(final_img)

        # Save frames as GIF
        out_dir = Path(save_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Epoch {epoch}] Saving GIF to: {save_path}")
        imageio.mimsave(save_path, frames, fps=fps)



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
