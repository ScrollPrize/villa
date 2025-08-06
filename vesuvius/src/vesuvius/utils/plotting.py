import torch
import cv2
import imageio
import numpy as np
from pathlib import Path


def minmax_scale_to_8bit(arr_np):
    """Convert array to 8-bit by scaling to 0-255 range"""
    # Ensure float32 for computation
    if arr_np.dtype != np.float32 and arr_np.dtype != np.float64:
        arr_np = arr_np.astype(np.float32)
    
    min_val = arr_np.min()
    max_val = arr_np.max()
    if max_val > min_val:
        arr_np = (arr_np - min_val) / (max_val - min_val) * 255
    else:
        arr_np = np.zeros_like(arr_np, dtype=np.float32) * 255
    return np.clip(arr_np, 0, 255).astype(np.uint8)


def add_text_label(img, text):
    """Add text label to the top of an image"""
    h, w = img.shape[:2]
    label_height = 30
    labeled_img = np.zeros((h + label_height, w, 3), dtype=np.uint8)
    labeled_img[label_height:, :, :] = img
    cv2.putText(labeled_img, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return labeled_img


def convert_slice_to_bgr(slice_2d_or_3d):
    """Convert a slice to BGR format for visualization"""
    if slice_2d_or_3d.ndim == 2:
        # Single channel - convert to BGR
        ch_8u = minmax_scale_to_8bit(slice_2d_or_3d)
        return cv2.cvtColor(ch_8u, cv2.COLOR_GRAY2BGR)
    
    elif slice_2d_or_3d.ndim == 3:
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
            # Multi-channel - just use first channel
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
    """Save debug visualization as GIF (3D) or PNG (2D)"""
    
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
        
        # Apply activation based on number of channels
        if not apply_activation:
            # No activation requested
            pass
        elif arr_np.shape[0] == 1:
            # Single channel - apply sigmoid
            arr_np = torch.sigmoid(torch.from_numpy(arr_np)).numpy()
        elif arr_np.shape[0] == 2:
            # Two channels - apply softmax
            arr_np = torch.softmax(torch.from_numpy(arr_np), dim=0).numpy()
        elif arr_np.shape[0] > 2:
            # More than 2 channels - apply argmax
            arr_np = torch.argmax(torch.from_numpy(arr_np), dim=0).numpy()
        
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
            
            # Apply activation based on number of channels
            if not apply_activation:
                # No activation requested
                pass
            elif arr_np.shape[0] == 1:
                # Single channel - apply sigmoid
                arr_np = torch.sigmoid(torch.from_numpy(arr_np)).numpy()
            elif arr_np.shape[0] == 2:
                # Two channels - apply softmax
                arr_np = torch.softmax(torch.from_numpy(arr_np), dim=0).numpy()
            elif arr_np.shape[0] > 2:
                # More than 2 channels - apply argmax
                arr_np = torch.argmax(torch.from_numpy(arr_np), dim=0).numpy()
            
            train_preds_np[t_name] = arr_np

    # Create visualization
    # Get actual prediction tasks (not skeleton data)
    pred_task_names = sorted(list(preds_np.keys()))
    
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
            val_imgs.append(add_text_label(convert_slice_to_bgr(gt_slice), label))
        
        # Show predictions (only for actual model outputs)
        for t_name in pred_task_names:
            pred = preds_np[t_name]
            pred_slice = pred[0] if pred.ndim == 3 and pred.shape[0] == 1 else pred
            val_imgs.append(add_text_label(convert_slice_to_bgr(pred_slice), f"Pred {t_name}"))
        
        rows.append(np.hstack(val_imgs))
        
        # Train row if available
        if train_inp_np is not None:
            train_imgs = [add_text_label(convert_slice_to_bgr(train_inp_np), "Train Input")]
            
            # Show all train targets (including skeleton data)
            for t_name in sorted(train_targets_np.keys()):
                gt = train_targets_np[t_name]
                gt_slice = gt[0] if gt.shape[0] == 1 else gt
                label = f"Skel {t_name.replace('_skel', '')}" if t_name.endswith('_skel') else f"GT {t_name}"
                train_imgs.append(add_text_label(convert_slice_to_bgr(gt_slice), label))
            
            # Show train predictions (only for actual model outputs)
            for t_name in pred_task_names:
                if t_name in train_preds_np:
                    pred = train_preds_np[t_name]
                    pred_slice = pred[0] if pred.ndim == 3 and pred.shape[0] == 1 else pred
                    train_imgs.append(add_text_label(convert_slice_to_bgr(pred_slice), f"Pred {t_name}"))
            
            rows.append(np.hstack(train_imgs))
        
        # Stack rows and save
        final_img = np.vstack(rows)
        out_dir = Path(save_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Epoch {epoch}] Saving PNG to: {save_path}")
        imageio.imwrite(save_path, final_img)
        
    else:
        # Build frames for 3D GIF
        frames = []
        z_dim = inp_np.shape[0] if inp_np.ndim == 3 else inp_np.shape[1]
        
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
                val_imgs.append(add_text_label(convert_slice_to_bgr(gt_slice), label))
            
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
                val_imgs.append(add_text_label(convert_slice_to_bgr(pred_slice), f"Pred {t_name}"))
            
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
                    train_imgs.append(add_text_label(convert_slice_to_bgr(gt_slice), label))
                
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
                        train_imgs.append(add_text_label(convert_slice_to_bgr(pred_slice), f"Pred {t_name}"))
                
                rows.append(np.hstack(train_imgs))
            
            # Stack rows for this frame
            frame = np.vstack(rows)
            # Ensure frame is uint8 and contiguous
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            frames.append(frame)
        
        # Save GIF
        out_dir = Path(save_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Epoch {epoch}] Saving GIF to: {save_path}")
        try:
            # Ensure all frames are properly formatted before saving
            frames_final = [np.ascontiguousarray(f, dtype=np.uint8) for f in frames]
            imageio.mimsave(save_path, frames_final, fps=fps, loop=0)
            # Return frames list for wandb logging (wandb expects list of frames or (T, H, W, C) array)
            return frames_final
        except Exception as e:
            print(f"Error saving GIF: {e}")
            print(f"Skipping debug visualization due to error")
            return None


def apply_activation_if_needed(activation_str):
    """This function is no longer needed but kept for compatibility"""
    pass