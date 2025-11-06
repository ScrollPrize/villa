import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from PIL import Image, ImageDraw
import multiprocessing as mp


# Worker must be at module scope for multiprocessing pickling (spawn context)
def _save_gif_worker(frames_list, path, _fps):
    try:
        if not frames_list:
            # No frames to write; treat as success but nothing done
            return
        pil_frames = []
        for frame in frames_list:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            pil_frames.append(Image.fromarray(frame))
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=max(1, 1000 // max(1, _fps)),
            loop=0,
        )
        for pf in pil_frames:
            pf.close()
    except Exception:
        # Ensure non-zero exit code on failure so parent can detect it
        import sys, traceback
        traceback.print_exc()
        sys.exit(1)


def minmax_scale_to_8bit(arr_np, clip_quantile: float = 0.005):
    """Convert array to 8-bit by scaling to 0-255 range with optional outlier clipping (default 0.5/99.5 percentiles)."""
    if not 0.0 <= clip_quantile < 0.5:
        raise ValueError("clip_quantile must be in the range [0.0, 0.5)")

    # Ensure float32 for computation
    if arr_np.dtype != np.float32 and arr_np.dtype != np.float64:
        arr_np = arr_np.astype(np.float32)
    else:
        arr_np = arr_np.astype(np.float32, copy=False)

    if clip_quantile > 0.0:
        lower = float(np.quantile(arr_np, clip_quantile))
        upper = float(np.quantile(arr_np, 1.0 - clip_quantile))
        if upper > lower:
            arr_np = np.clip(arr_np, lower, upper)

    min_val = arr_np.min()
    max_val = arr_np.max()
    if max_val > min_val:
        arr_np = (arr_np - min_val) / (max_val - min_val) * 255
    else:
        arr_np = np.zeros_like(arr_np, dtype=np.float32)
    return np.clip(arr_np, 0, 255).astype(np.uint8)


def add_text_label(img, text):
    """Add text label to the top of an image"""
    # Ensure img is proper format
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    
    h, w = img.shape[:2]
    label_height = 30
    
    # Create labeled image
    labeled_img = np.zeros((h + label_height, w, 3), dtype=np.uint8)
    labeled_img[label_height:, :, :] = img
    
    # Use PIL for text rendering to avoid OpenCV segfaults
    pil_img = Image.fromarray(labeled_img)
    draw = ImageDraw.Draw(pil_img)
    draw.text((5, 5), text, fill=(255, 255, 255))
    return np.array(pil_img, dtype=np.uint8)


def _apply_activation(array_np: np.ndarray, activation: Optional[str], *, is_surface: bool) -> np.ndarray:
    if activation is None or str(activation).lower() in {"none", "identity"} or is_surface:
        return array_np

    activation_l = str(activation).lower()
    arr = array_np.astype(np.float32, copy=False)
    tensor = torch.from_numpy(arr)

    if activation_l.startswith("sigmoid"):
        return torch.sigmoid(tensor).numpy()
    if activation_l.startswith("softmax"):
        return torch.softmax(tensor, dim=0).numpy()
    if activation_l.startswith("tanh"):
        return torch.tanh(tensor).numpy()
    return array_np


def _vector_to_bgr(vector_3ch):
    """Map a 3×H×W vector field to a BGR image using directional colouring."""
    if vector_3ch.shape[0] != 3:
        raise ValueError(f"Expected 3-channel vector field, got shape {vector_3ch.shape}")

    vec = vector_3ch.astype(np.float32, copy=False)
    norm = np.linalg.norm(vec, axis=0, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    unit = vec / norm

    rgb = (np.transpose(unit, (1, 2, 0)) * 0.5 + 0.5).clip(0.0, 1.0)
    rgb_uint8 = (rgb * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)


def _surface_frame_to_bgr(frame_slice):
    """Render a 9-channel surface frame slice as a concatenated BGR panel."""
    if frame_slice.shape[0] != 9:
        raise ValueError(f"Surface frame slice must have 9 channels, got {frame_slice.shape}")

    h = frame_slice.shape[-2]
    w = frame_slice.shape[-1]

    reshaped = frame_slice.reshape(3, 3, h, w)
    panels = []
    separator = np.zeros((h, 2, 3), dtype=np.uint8)
    for idx in range(3):
        panel = _vector_to_bgr(reshaped[idx])
        panels.append(panel)
        panels.append(separator.copy())

    combined = np.hstack(panels[:-1])  # drop last separator
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = ("T_u", "T_v", "N")
    offset = 0
    for idx, label in enumerate(labels):
        x = idx * (w + 2) + 10
        cv2.putText(combined, label, (x, 25), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return combined


def convert_slice_to_bgr(slice_2d_or_3d, *, task_name: str | None = None, task_cfg: Dict | None = None):
    """Convert a slice to BGR format for visualization"""
    task_cfg = task_cfg or {}
    task_type = task_cfg.get("visualization") or task_cfg.get("type")
    is_surface = (
        (task_name is not None and task_name.endswith("surface_frame"))
        or task_type == "surface_frame"
        or (slice_2d_or_3d.ndim >= 3 and slice_2d_or_3d.shape[0] == 9)
    )

    if slice_2d_or_3d.ndim == 2:
        # Single channel - convert to BGR
        ch_8u = minmax_scale_to_8bit(slice_2d_or_3d)
        return cv2.cvtColor(ch_8u, cv2.COLOR_GRAY2BGR)

    elif slice_2d_or_3d.ndim == 3:
        if is_surface:
            try:
                return _surface_frame_to_bgr(slice_2d_or_3d)
            except ValueError:
                pass
        if slice_2d_or_3d.shape[0] == 1:
            # Single channel with channel dimension
            ch_8u = minmax_scale_to_8bit(slice_2d_or_3d[0])
            return cv2.cvtColor(ch_8u, cv2.COLOR_GRAY2BGR)
        
        elif slice_2d_or_3d.shape[0] == 3:
            # RGB or normal map - just transpose and scale
            rgb = np.transpose(slice_2d_or_3d, (1, 2, 0))
            return minmax_scale_to_8bit(rgb)
        
        elif slice_2d_or_3d.shape[0] == 2:
            # Binary segmentation - use foreground channel (channel 1)
            ch_8u = minmax_scale_to_8bit(slice_2d_or_3d[1])
            return cv2.cvtColor(ch_8u, cv2.COLOR_GRAY2BGR)
        
        else:
            is_affinity = (
                task_type == "affinity"
                or (task_name is not None and "affinity" in task_name.lower())
            )
            if is_affinity:
                # Aggregate across channels for visualization
                agg = slice_2d_or_3d.mean(axis=0)
                ch_8u = minmax_scale_to_8bit(agg)
            else:
                ch_8u = minmax_scale_to_8bit(slice_2d_or_3d[0])
            return cv2.cvtColor(ch_8u, cv2.COLOR_GRAY2BGR)
    
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {slice_2d_or_3d.shape}")


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
    train_outputs_dict: dict = None,    # Optional train sample outputs
    skeleton_dict: dict = None,         # Optional skeleton data for visualization
    train_skeleton_dict: dict = None,   # Optional train skeleton data
    apply_activation: bool = True       # Whether to apply activation functions
):
    """
    Save debug visualization as GIF (3D) or PNG (2D).

    Returns
    -------
    tuple[list[np.ndarray] | None, np.ndarray | None]
        (frames_for_gif, preview_image) where preview_image is a single 2D panel
        extracted from the visualization (middle Z slice for 3D data).
    """
    
    # Get input array
    # Convert BFloat16 to Float32 before numpy conversion
    if input_volume.dtype == torch.bfloat16:
        input_volume = input_volume.float()
    inp_np = input_volume.cpu().numpy()[0]  # Remove batch dim
    is_2d = len(inp_np.shape) == 3  # [C, H, W] format for 2D data
    
    if is_2d:
        save_path = save_path.replace('.gif', '.png')
    
    # Remove channel dim if single channel
    if inp_np.shape[0] == 1:
        inp_np = inp_np[0]

    # Process all targets
    targets_np = {}
    for t_name, t_tensor in targets_dict.items():
        # Convert BFloat16 to Float32 before numpy conversion
        if t_tensor.dtype == torch.bfloat16:
            t_tensor = t_tensor.float()
        arr_np = t_tensor.cpu().numpy()
        # Remove batch dimension if present
        while arr_np.ndim > (3 if is_2d else 4):
            arr_np = arr_np[0]
        targets_np[t_name] = arr_np

    # Process all predictions
    preds_np = {}
    for t_name, p_tensor in outputs_dict.items():
        # Convert BFloat16 to Float32 before numpy conversion
        if p_tensor.dtype == torch.bfloat16:
            p_tensor = p_tensor.float()
        arr_np = p_tensor.cpu().numpy()
        # Remove batch dimension if present
        if arr_np.ndim > (3 if is_2d else 4):
            arr_np = arr_np[0]
        
        task_cfg = tasks_dict.get(t_name, {}) if tasks_dict else {}
        activation = task_cfg.get("activation", None)
        is_surface_frame = arr_np.shape[0] == 9 or t_name.endswith("surface_frame")

        if apply_activation:
            arr_np = _apply_activation(arr_np, activation, is_surface=is_surface_frame)
        
        preds_np[t_name] = arr_np

    # Process train data if provided
    train_inp_np = None
    train_targets_np = {}
    train_preds_np = {}
    
    if train_input is not None and train_targets_dict is not None and train_outputs_dict is not None:
        # Convert BFloat16 to Float32 before numpy conversion
        if train_input.dtype == torch.bfloat16:
            train_input = train_input.float()
        train_inp_np = train_input.cpu().numpy()[0]
        if train_inp_np.shape[0] == 1:
            train_inp_np = train_inp_np[0]

        # Process all train targets
        for t_name, t_tensor in train_targets_dict.items():
            # Convert BFloat16 to Float32 before numpy conversion
            if t_tensor.dtype == torch.bfloat16:
                t_tensor = t_tensor.float()
            arr_np = t_tensor.cpu().numpy()
            # Remove batch dimension if present
            while arr_np.ndim > (3 if is_2d else 4):
                arr_np = arr_np[0]
            train_targets_np[t_name] = arr_np

        # Process all train predictions
        for t_name, p_tensor in train_outputs_dict.items():
            # Convert BFloat16 to Float32 before numpy conversion
            if p_tensor.dtype == torch.bfloat16:
                p_tensor = p_tensor.float()
            arr_np = p_tensor.cpu().numpy()
            # Remove batch dimension if present
            if arr_np.ndim > (3 if is_2d else 4):
                arr_np = arr_np[0]
            
            task_cfg = tasks_dict.get(t_name, {}) if tasks_dict else {}
            activation = task_cfg.get("activation", None)
            is_surface_frame = arr_np.shape[0] == 9 or t_name.endswith("surface_frame")

            if apply_activation:
                arr_np = _apply_activation(arr_np, activation, is_surface=is_surface_frame)
            
            train_preds_np[t_name] = arr_np

    # Create visualization
    # Get actual prediction tasks (not skeleton data)
    pred_task_names = sorted(list(preds_np.keys()))
    
    def _pad_rows_to_uniform_width(rows_list: list[np.ndarray]) -> list[np.ndarray]:
        if not rows_list:
            return rows_list
        max_width = max(row.shape[1] for row in rows_list)
        if max_width == 0:
            return rows_list
        padded_rows = []
        for row in rows_list:
            pad_width = max_width - row.shape[1]
            if pad_width > 0:
                if row.ndim == 3:
                    pad_config = ((0, 0), (0, pad_width), (0, 0))
                elif row.ndim == 2:
                    pad_config = ((0, 0), (0, pad_width))
                else:
                    raise ValueError(f"Unexpected row ndim={row.ndim} for debug visualization stacking.")
                row = np.pad(row, pad_config, mode="constant", constant_values=0)
            padded_rows.append(row)
        return padded_rows

    if is_2d:
        # Build image grid for 2D
        rows = []
        
        # Val row: input, targets (including skels), preds
        val_imgs = [add_text_label(convert_slice_to_bgr(inp_np), "Val Input")]
        
        # Show all targets (including skeleton data)
        for t_name in sorted(targets_np.keys()):
            gt = targets_np[t_name]
            gt_slice = gt[0] if gt.shape[0] == 1 else gt
            label = f"Skel {t_name.replace('_skel', '')}" if t_name.endswith('_skel') else f"GT {t_name}"
            val_imgs.append(add_text_label(convert_slice_to_bgr(gt_slice, task_name=t_name, task_cfg=tasks_dict.get(t_name, {})), label))
        
        # Show predictions (only for actual model outputs)
        for t_name in pred_task_names:
            pred = preds_np[t_name]
            pred_slice = pred[0] if pred.ndim == 3 and pred.shape[0] == 1 else pred
            val_imgs.append(add_text_label(convert_slice_to_bgr(pred_slice, task_name=t_name, task_cfg=tasks_dict.get(t_name, {})), f"Pred {t_name}"))
        
        rows.append(np.hstack(val_imgs))
        
        # Train row if available
        if train_inp_np is not None:
            train_imgs = [add_text_label(convert_slice_to_bgr(train_inp_np), "Train Input")]
            
            # Show all train targets (including skeleton data)
            for t_name in sorted(train_targets_np.keys()):
                gt = train_targets_np[t_name]
                gt_slice = gt[0] if gt.shape[0] == 1 else gt
                label = f"Skel {t_name.replace('_skel', '')}" if t_name.endswith('_skel') else f"GT {t_name}"
                train_imgs.append(add_text_label(convert_slice_to_bgr(gt_slice, task_name=t_name, task_cfg=tasks_dict.get(t_name, {})), label))
            
            # Show train predictions (only for actual model outputs)
            for t_name in pred_task_names:
                if t_name in train_preds_np:
                    pred = train_preds_np[t_name]
                    pred_slice = pred[0] if pred.ndim == 3 and pred.shape[0] == 1 else pred
                    train_imgs.append(add_text_label(convert_slice_to_bgr(pred_slice, task_name=t_name, task_cfg=tasks_dict.get(t_name, {})), f"Pred {t_name}"))
            
            rows.append(np.hstack(train_imgs))
        
        # Stack rows and save
        rows = _pad_rows_to_uniform_width(rows)
        final_img = np.vstack(rows)
        out_dir = Path(save_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Epoch {epoch}] Saving PNG to: {save_path}")
        # Use PIL for saving
        Image.fromarray(final_img).save(save_path)

        preview_img = np.ascontiguousarray(final_img, dtype=np.uint8)
        return None, preview_img

    else:
        # Build frames for 3D GIF
        frames = []
        preview_frame = None
        z_dim = inp_np.shape[0] if inp_np.ndim == 3 else inp_np.shape[1]
        mid_z_idx = max(z_dim // 2, 0)

        for z_idx in range(z_dim):
            rows = []
            
            # Get slices
            inp_slice = inp_np[z_idx] if inp_np.ndim == 3 else inp_np[:, z_idx, :, :]
            
            # Val row
            val_imgs = [add_text_label(convert_slice_to_bgr(inp_slice), "Val Input")]
            
            # Show all targets (including skeleton data)
            for t_name in sorted(targets_np.keys()):
                gt = targets_np[t_name]
                if gt.shape[0] == 1:
                    gt_slice = gt[0, z_idx, :, :]
                else:
                    gt_slice = gt[:, z_idx, :, :]
                label = f"Skel {t_name.replace('_skel', '')}" if t_name.endswith('_skel') else f"GT {t_name}"
                val_imgs.append(add_text_label(convert_slice_to_bgr(gt_slice, task_name=t_name, task_cfg=tasks_dict.get(t_name, {})), label))
            
            # Show predictions (only for actual model outputs)
            for t_name in pred_task_names:
                pred = preds_np[t_name]
                if pred.ndim == 4:
                    if pred.shape[0] == 1:
                        pred_slice = pred[0, z_idx, :, :]
                    else:
                        pred_slice = pred[:, z_idx, :, :]
                else:
                    pred_slice = pred[z_idx, :, :]
                val_imgs.append(add_text_label(convert_slice_to_bgr(pred_slice, task_name=t_name, task_cfg=tasks_dict.get(t_name, {})), f"Pred {t_name}"))
            
            rows.append(np.hstack(val_imgs))
            
            # Train row if available
            if train_inp_np is not None:
                train_slice = train_inp_np[z_idx] if train_inp_np.ndim == 3 else train_inp_np[:, z_idx, :, :]
                train_imgs = [add_text_label(convert_slice_to_bgr(train_slice), "Train Input")]
                
                # Show all train targets (including skeleton data)
                for t_name in sorted(train_targets_np.keys()):
                    gt = train_targets_np[t_name]
                    if gt.shape[0] == 1:
                        gt_slice = gt[0, z_idx, :, :]
                    else:
                        gt_slice = gt[:, z_idx, :, :]
                    label = f"Skel {t_name.replace('_skel', '')}" if t_name.endswith('_skel') else f"GT {t_name}"
                    train_imgs.append(add_text_label(convert_slice_to_bgr(gt_slice, task_name=t_name, task_cfg=tasks_dict.get(t_name, {})), label))
                
                # Show train predictions (only for actual model outputs)
                for t_name in pred_task_names:
                    if t_name in train_preds_np:
                        pred = train_preds_np[t_name]
                        if pred.ndim == 4:
                            if pred.shape[0] == 1:
                                pred_slice = pred[0, z_idx, :, :]
                            else:
                                pred_slice = pred[:, z_idx, :, :]
                        else:
                            pred_slice = pred[z_idx, :, :]
                        train_imgs.append(add_text_label(convert_slice_to_bgr(pred_slice, task_name=t_name, task_cfg=tasks_dict.get(t_name, {})), f"Pred {t_name}"))
                
                rows.append(np.hstack(train_imgs))
            
            # Stack rows for this frame
            rows = _pad_rows_to_uniform_width(rows)
            frame = np.vstack(rows)
            # Ensure frame is uint8 and contiguous
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            frames.append(frame)

            if z_idx == mid_z_idx:
                preview_frame = frame.copy()
        
        # Save GIF in a subprocess to avoid crashing main training process on encoder segfaults
        out_dir = Path(save_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Epoch {epoch}] Saving GIF to: {save_path}")

        # Use spawn context for better isolation
        ctx = mp.get_context("spawn")
        proc = ctx.Process(target=_save_gif_worker, args=(frames, str(save_path), fps))
        proc.start()
        proc.join(30)  # timeout safeguard (seconds)

        if proc.is_alive():
            proc.terminate()
            print("Warning: GIF save timed out; skipping debug visualization")
            if preview_frame is None and frames:
                preview_frame = frames[len(frames) // 2].copy()
            return None, preview_frame

        if proc.exitcode == 0:
            print(f"Successfully saved GIF to: {save_path}")
            if preview_frame is None and frames:
                preview_frame = frames[len(frames) // 2].copy()
            return frames, preview_frame
        else:
            print(f"Warning: GIF save failed in subprocess (exit code {proc.exitcode}); skipping")
            if preview_frame is None and frames:
                preview_frame = frames[len(frames) // 2].copy()
            return None, preview_frame


def apply_activation_if_needed(activation_str):
    """This function is no longer needed but kept for compatibility"""
    pass
