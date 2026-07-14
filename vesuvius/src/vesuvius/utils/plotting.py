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


def minmax_scale_to_8bit(
    arr_np,
    clip_quantile: float = 0.005,
    value_range: Optional[tuple[float, float]] = None,
):
    """Convert array to 8-bit by scaling to 0-255 range with optional outlier clipping (default 0.5/99.5 percentiles)."""
    if not 0.0 <= clip_quantile < 0.5:
        raise ValueError("clip_quantile must be in the range [0.0, 0.5)")

    # Ensure float32 for computation
    if arr_np.dtype != np.float32 and arr_np.dtype != np.float64:
        arr_np = arr_np.astype(np.float32)
    else:
        arr_np = arr_np.astype(np.float32, copy=False)

    if value_range is not None:
        lower, upper = value_range
        lower = float(lower)
        upper = float(upper)
        if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
            arr_np = np.clip(arr_np, lower, upper)
    elif clip_quantile > 0.0:
        lower = float(np.quantile(arr_np, clip_quantile))
        upper = float(np.quantile(arr_np, 1.0 - clip_quantile))
        if upper > lower:
            arr_np = np.clip(arr_np, lower, upper)

    min_val = arr_np.min()
    max_val = arr_np.max()
    if max_val > min_val:
        arr_np = (arr_np - min_val) / (max_val - min_val) * 255
    else:
        # Uniform value - scale assuming label range [0, max(2, value)]
        # This ensures: 0→0 (black), 1→127 (gray), 2→255 (white)
        uniform_val = float(min_val)
        assumed_max = max(2.0, uniform_val)
        if assumed_max > 0:
            scaled = (uniform_val / assumed_max) * 255
        else:
            scaled = 0.0
        arr_np = np.full_like(arr_np, scaled, dtype=np.float32)
    return np.clip(arr_np, 0, 255).astype(np.uint8)


def _resolve_display_signal(
    arr_np: np.ndarray,
    *,
    is_2d_run: bool,
    task_name: str | None = None,
    task_cfg: Dict | None = None,
):
    """Resolve the signal actually rendered by convert_slice_to_bgr for consistent global scaling."""
    task_cfg = task_cfg or {}
    task_type = task_cfg.get("visualization") or task_cfg.get("type")
    is_spatial_3d_volume = arr_np.ndim == 3 and not is_2d_run
    is_surface = (
        (task_name is not None and task_name.endswith("surface_frame"))
        or task_type == "surface_frame"
        or (arr_np.ndim >= 3 and arr_np.shape[0] == 9 and not is_spatial_3d_volume)
    )
    if is_surface:
        return None

    if arr_np.ndim == 2:
        return arr_np

    # In 3D runs, [Z, H, W] tensors are spatial volumes, not channel-first 2D tensors.
    if arr_np.ndim == 3 and not is_2d_run:
        return arr_np

    if arr_np.ndim == 3:
        if arr_np.shape[0] == 1:
            return arr_np[0]
        if arr_np.shape[0] == 3:
            return np.transpose(arr_np, (1, 2, 0))
        if arr_np.shape[0] == 2:
            return arr_np[1]
        is_affinity = (
            task_type == "affinity"
            or (task_name is not None and "affinity" in task_name.lower())
        )
        return arr_np.mean(axis=0) if is_affinity else arr_np[0]

    if arr_np.ndim == 4:
        if arr_np.shape[0] == 1:
            return arr_np[0]
        if arr_np.shape[0] == 3:
            return np.transpose(arr_np, (1, 2, 3, 0))
        if arr_np.shape[0] == 2:
            return arr_np[1]
        is_affinity = (
            task_type == "affinity"
            or (task_name is not None and "affinity" in task_name.lower())
        )
        return arr_np.mean(axis=0) if is_affinity else arr_np[0]

    return arr_np


def _compute_display_value_range(
    arr_np: np.ndarray,
    *,
    is_2d_run: bool,
    task_name: str | None = None,
    task_cfg: Dict | None = None,
    clip_quantile: float = 0.005,
) -> Optional[tuple[float, float]]:
    """Compute robust global visualization bounds used for all slices of a panel."""
    signal = _resolve_display_signal(
        arr_np,
        is_2d_run=is_2d_run,
        task_name=task_name,
        task_cfg=task_cfg,
    )
    if signal is None:
        return None

    vals = signal.astype(np.float32, copy=False)
    finite_mask = np.isfinite(vals)
    if not np.any(finite_mask):
        return (0.0, 1.0)
    finite_vals = vals[finite_mask]

    if clip_quantile > 0.0 and finite_vals.size > 1:
        lower = float(np.quantile(finite_vals, clip_quantile))
        upper = float(np.quantile(finite_vals, 1.0 - clip_quantile))
    else:
        lower = float(finite_vals.min())
        upper = float(finite_vals.max())

    if not np.isfinite(lower) or not np.isfinite(upper):
        return (0.0, 1.0)

    # Keep a stable non-degenerate range for nearly-uniform panels.
    if upper <= lower:
        lower = float(finite_vals.min())
        upper = float(finite_vals.max())
        if upper <= lower:
            uniform_val = lower
            assumed_max = max(2.0, uniform_val)
            if assumed_max <= 0.0:
                assumed_max = 1.0
            return (0.0, float(assumed_max))

    return (lower, upper)


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


def _format_aux_panel_label(name: str) -> str:
    raw = str(name).strip()
    if raw.startswith("guide_"):
        target_name = raw[len("guide_"):].replace("_", " ")
        return f"Guide {target_name}"
    return raw.replace("_", " ")


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


# Categorical palette for multiclass segmentation visualization. Kept small
# and stable: bg=black, class 1=red, class 2=green, class 3=grey (ignore),
# classes 4+ cycle through distinct hues. Stored BGR for OpenCV.
_SEG_PALETTE_BGR = np.array(
    [
        [0,   0,   0],    # 0: background
        [0,   0,   255],  # 1: red
        [0,   255, 0],    # 2: green
        [128, 128, 128],  # 3: grey (conventional ignore slot)
        [255, 0,   0],    # 4: blue
        [0,   255, 255],  # 5: yellow
        [255, 0,   255],  # 6: magenta
        [255, 255, 0],    # 7: cyan
        [0,   165, 255],  # 8: orange
        [255, 255, 255],  # 9: white
    ],
    dtype=np.uint8,
)


def _is_multiclass_segmentation(task_cfg: Dict | None) -> bool:
    """A target is multiclass seg when out_channels > 2 and the activation is
    softmax-ish (``"none"`` / ``"softmax"``). out_channels == 2 is treated as
    binary and handled by the existing grayscale branches.
    """
    if not task_cfg:
        return False
    out_channels = int(task_cfg.get("out_channels", 0) or 0)
    if out_channels <= 2:
        return False
    activation = str(task_cfg.get("activation", "none") or "none").lower()
    return activation in {"none", "softmax", "identity"}


def _indices_to_bgr(indices_2d: np.ndarray) -> np.ndarray:
    """Map a (H, W) integer array of class indices to a BGR uint8 image."""
    idx = np.asarray(indices_2d)
    if idx.ndim != 2:
        raise ValueError(f"Expected (H, W), got shape {idx.shape}")
    idx = np.clip(np.rint(idx).astype(np.int64), 0, len(_SEG_PALETTE_BGR) - 1)
    return _SEG_PALETTE_BGR[idx]


def _logits_to_class_indices(logits_c_hw: np.ndarray) -> np.ndarray:
    """(C, H, W) -> (H, W) argmax across channels."""
    return np.argmax(logits_c_hw, axis=0)


# BGR channel (index) used for each foreground class: class 1 -> R,
# class 2 -> G, class 3 -> B. Matches the GT palette so the same class
# shows the same hue in GT and prediction.
_FG_CHANNEL_ORDER_BGR = (2, 1, 0)  # R, G, B in BGR order


def _softmax_probs(logits_c_hw: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax across channel 0, returning (C, H, W)."""
    x = logits_c_hw.astype(np.float32, copy=False)
    m = x.max(axis=0, keepdims=True)
    e = np.exp(x - m)
    return e / np.maximum(e.sum(axis=0, keepdims=True), 1e-12)


def _logits_to_rgb_cloud(logits_c_hw: np.ndarray) -> np.ndarray:
    """(C, H, W) logits -> (H, W, 3) BGR showing the softmax probability cloud.

    Background (channel 0) is dropped so it renders black. Foreground
    classes land in R/G/B (class 1 -> R, class 2 -> G, class 3 -> B) to
    match the categorical GT palette; extra foreground classes cycle
    through the same three channels. Soft / uncertain predictions show up
    as dimmer pixels and color blends (the probability "cloud").
    """
    probs = _softmax_probs(logits_c_hw)  # (C, H, W), sums to 1 over axis 0
    if probs.shape[0] <= 1:
        return cv2.cvtColor(
            np.clip(probs[0] * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
        )
    fg = probs[1:]  # drop background
    h, w = fg.shape[1], fg.shape[2]
    bgr = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(fg.shape[0]):
        ch = _FG_CHANNEL_ORDER_BGR[i % 3]
        # Use max so cycled channels don't erase each other for high C.
        bgr[..., ch] = np.maximum(bgr[..., ch], fg[i])
    return np.clip(bgr * 255.0, 0, 255).astype(np.uint8)


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


def convert_slice_to_bgr(
    slice_2d_or_3d,
    *,
    task_name: str | None = None,
    task_cfg: Dict | None = None,
    value_range: Optional[tuple[float, float]] = None,
):
    """Convert a slice to BGR format for visualization"""
    task_cfg = task_cfg or {}
    task_type = task_cfg.get("visualization") or task_cfg.get("type")
    is_surface = (
        (task_name is not None and task_name.endswith("surface_frame"))
        or task_type == "surface_frame"
        or (slice_2d_or_3d.ndim >= 3 and slice_2d_or_3d.shape[0] == 9)
    )
    is_multiclass = _is_multiclass_segmentation(task_cfg)

    if slice_2d_or_3d.ndim == 2:
        if is_multiclass:
            return _indices_to_bgr(slice_2d_or_3d)
        # Single channel - convert to BGR
        ch_8u = minmax_scale_to_8bit(slice_2d_or_3d, value_range=value_range)
        return cv2.cvtColor(ch_8u, cv2.COLOR_GRAY2BGR)

    elif slice_2d_or_3d.ndim == 3:
        if is_surface:
            try:
                return _surface_frame_to_bgr(slice_2d_or_3d)
            except ValueError:
                pass
        if slice_2d_or_3d.shape[0] == 1:
            if is_multiclass:
                return _indices_to_bgr(slice_2d_or_3d[0])
            # Single channel with channel dimension
            ch_8u = minmax_scale_to_8bit(slice_2d_or_3d[0], value_range=value_range)
            return cv2.cvtColor(ch_8u, cv2.COLOR_GRAY2BGR)

        elif is_multiclass and slice_2d_or_3d.shape[0] >= 2:
            # Multiclass prediction logits (C, H, W): softmax -> RGB
            # probability cloud. Keeps the full confidence range visible.
            return _logits_to_rgb_cloud(slice_2d_or_3d)

        elif slice_2d_or_3d.shape[0] == 3:
            # RGB or normal map - just transpose and scale
            rgb = np.transpose(slice_2d_or_3d, (1, 2, 0))
            return minmax_scale_to_8bit(rgb, value_range=value_range)

        elif slice_2d_or_3d.shape[0] == 2:
            # Binary segmentation - use foreground channel (channel 1)
            ch_8u = minmax_scale_to_8bit(slice_2d_or_3d[1], value_range=value_range)
            return cv2.cvtColor(ch_8u, cv2.COLOR_GRAY2BGR)
        
        else:
            is_affinity = (
                task_type == "affinity"
                or (task_name is not None and "affinity" in task_name.lower())
            )
            if is_affinity:
                # Aggregate across channels for visualization
                agg = slice_2d_or_3d.mean(axis=0)
                ch_8u = minmax_scale_to_8bit(agg, value_range=value_range)
            else:
                ch_8u = minmax_scale_to_8bit(slice_2d_or_3d[0], value_range=value_range)
            return cv2.cvtColor(ch_8u, cv2.COLOR_GRAY2BGR)
    
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {slice_2d_or_3d.shape}")


def _scale_with_fixed_value_range_to_8bit(
    arr_np: np.ndarray,
    value_range: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    """Scale an array to uint8 using a fixed global range when provided."""
    arr = arr_np.astype(np.float32, copy=False)
    if value_range is None:
        return minmax_scale_to_8bit(arr)

    lower, upper = value_range
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return minmax_scale_to_8bit(arr)

    arr = np.clip(arr, lower, upper)
    arr = (arr - lower) / (upper - lower) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _gray_volume_to_bgr(volume_zhw: np.ndarray, value_range: Optional[tuple[float, float]]) -> np.ndarray:
    gray_8u = _scale_with_fixed_value_range_to_8bit(volume_zhw, value_range)
    return np.repeat(gray_8u[..., None], 3, axis=-1)


def _render_3d_volume_to_bgr(
    arr_np: np.ndarray,
    *,
    task_name: str | None = None,
    task_cfg: Dict | None = None,
    value_range: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    """Render a full 3D tensor/volume to a Z-stacked uint8 BGR panel volume."""
    task_cfg = task_cfg or {}
    task_type = task_cfg.get("visualization") or task_cfg.get("type")
    is_surface = (
        (task_name is not None and task_name.endswith("surface_frame"))
        or task_type == "surface_frame"
        or (arr_np.ndim >= 4 and arr_np.shape[0] == 9)
    )

    is_multiclass = _is_multiclass_segmentation(task_cfg)

    if arr_np.ndim == 3:
        if is_multiclass:
            return np.stack([_indices_to_bgr(arr_np[z]) for z in range(arr_np.shape[0])], axis=0)
        return _gray_volume_to_bgr(arr_np, value_range)

    if arr_np.ndim != 4:
        raise ValueError(f"Expected 3D volume or 4D channel-first tensor, got shape {arr_np.shape}")

    if is_surface:
        return np.stack(
            [_surface_frame_to_bgr(arr_np[:, z_idx, :, :]) for z_idx in range(arr_np.shape[1])],
            axis=0,
        )

    if arr_np.shape[0] == 1:
        if is_multiclass:
            return np.stack(
                [_indices_to_bgr(arr_np[0, z]) for z in range(arr_np.shape[1])], axis=0
            )
        return _gray_volume_to_bgr(arr_np[0], value_range)

    if is_multiclass and arr_np.shape[0] >= 2:
        # Multiclass pred logits (C, Z, H, W): softmax -> RGB probability
        # cloud per slice. Uses the full confidence range instead of a
        # hard argmax so uncertain regions are visible.
        return np.stack(
            [_logits_to_rgb_cloud(arr_np[:, z]) for z in range(arr_np.shape[1])],
            axis=0,
        )

    if arr_np.shape[0] == 3:
        rgb_zhwc = np.transpose(arr_np, (1, 2, 3, 0))
        return _scale_with_fixed_value_range_to_8bit(rgb_zhwc, value_range)

    if arr_np.shape[0] == 2:
        return _gray_volume_to_bgr(arr_np[1], value_range)

    is_affinity = (
        task_type == "affinity"
        or (task_name is not None and "affinity" in task_name.lower())
    )
    reduced = arr_np.mean(axis=0) if is_affinity else arr_np[0]
    return _gray_volume_to_bgr(reduced, value_range)


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
    aux_outputs_dict: dict = None,      # Optional display-only aux outputs
    train_aux_outputs_dict: dict = None,# Optional display-only train aux outputs
    skeleton_dict: dict = None,         # Optional skeleton data for visualization
    train_skeleton_dict: dict = None,   # Optional train skeleton data
    apply_activation: bool = True,      # Whether to apply activation functions
    save_media: bool = True,            # Whether to write the debug image/GIF to disk
    # Unlabeled sample visualization for semi-supervised training
    unlabeled_input: torch.Tensor = None,       # Optional unlabeled sample input
    unlabeled_pseudo_dict: dict = None,         # Teacher predictions (pseudo-labels)
    unlabeled_outputs_dict: dict = None         # Student predictions on unlabeled data
):
    """
    Save debug visualization as GIF (3D) or PNG (2D).

    Returns
    -------
    tuple[list[np.ndarray] | None, np.ndarray | None]
        (frames_for_gif, preview_image) where preview_image is a single 2D panel
        extracted from the visualization (middle Z slice for 3D data).
    """
    
    tasks_dict = tasks_dict or {}

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

    aux_outputs_np = {}
    for aux_name, aux_tensor in (aux_outputs_dict or {}).items():
        if aux_tensor.dtype == torch.bfloat16:
            aux_tensor = aux_tensor.float()
        arr_np = aux_tensor.cpu().numpy()
        while arr_np.ndim > (3 if is_2d else 4):
            arr_np = arr_np[0]
        aux_outputs_np[aux_name] = arr_np

    # Process train data if provided
    train_inp_np = None
    train_targets_np = {}
    train_preds_np = {}
    train_aux_outputs_np = {}
    
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

        for aux_name, aux_tensor in (train_aux_outputs_dict or {}).items():
            if aux_tensor.dtype == torch.bfloat16:
                aux_tensor = aux_tensor.float()
            arr_np = aux_tensor.cpu().numpy()
            while arr_np.ndim > (3 if is_2d else 4):
                arr_np = arr_np[0]
            train_aux_outputs_np[aux_name] = arr_np

    # Process unlabeled data if provided (for semi-supervised training visualization)
    unlabeled_inp_np = None
    unlabeled_pseudo_np = {}
    unlabeled_preds_np = {}

    if unlabeled_input is not None and unlabeled_pseudo_dict is not None and unlabeled_outputs_dict is not None:
        # Convert BFloat16 to Float32 before numpy conversion
        if unlabeled_input.dtype == torch.bfloat16:
            unlabeled_input = unlabeled_input.float()
        unlabeled_inp_np = unlabeled_input.cpu().numpy()[0]
        if unlabeled_inp_np.shape[0] == 1:
            unlabeled_inp_np = unlabeled_inp_np[0]

        # Process pseudo-labels (teacher predictions)
        for t_name, p_tensor in unlabeled_pseudo_dict.items():
            if p_tensor.dtype == torch.bfloat16:
                p_tensor = p_tensor.float()
            arr_np = p_tensor.cpu().numpy()
            if arr_np.ndim > (3 if is_2d else 4):
                arr_np = arr_np[0]

            task_cfg = tasks_dict.get(t_name, {}) if tasks_dict else {}
            activation = task_cfg.get("activation", None)
            is_surface_frame = arr_np.shape[0] == 9 or t_name.endswith("surface_frame")

            if apply_activation:
                arr_np = _apply_activation(arr_np, activation, is_surface=is_surface_frame)

            unlabeled_pseudo_np[t_name] = arr_np

        # Process student predictions on unlabeled data
        for t_name, p_tensor in unlabeled_outputs_dict.items():
            if p_tensor.dtype == torch.bfloat16:
                p_tensor = p_tensor.float()
            arr_np = p_tensor.cpu().numpy()
            if arr_np.ndim > (3 if is_2d else 4):
                arr_np = arr_np[0]

            task_cfg = tasks_dict.get(t_name, {}) if tasks_dict else {}
            activation = task_cfg.get("activation", None)
            is_surface_frame = arr_np.shape[0] == 9 or t_name.endswith("surface_frame")

            if apply_activation:
                arr_np = _apply_activation(arr_np, activation, is_surface=is_surface_frame)

            unlabeled_preds_np[t_name] = arr_np

    # Create visualization
    # Get actual prediction tasks (not skeleton data)
    pred_task_names = sorted(list(preds_np.keys()))

    # Compute stable global display ranges per panel/tensor to avoid per-slice flicker artifacts.
    val_input_range = _compute_display_value_range(inp_np, is_2d_run=is_2d)
    target_ranges = {
        t_name: _compute_display_value_range(
            t_arr,
            is_2d_run=is_2d,
            task_name=t_name,
            task_cfg=tasks_dict.get(t_name, {}),
        )
        for t_name, t_arr in targets_np.items()
    }
    pred_ranges = {
        t_name: _compute_display_value_range(
            p_arr,
            is_2d_run=is_2d,
            task_name=t_name,
            task_cfg=tasks_dict.get(t_name, {}),
        )
        for t_name, p_arr in preds_np.items()
    }
    aux_ranges = {
        aux_name: _compute_display_value_range(
            aux_arr,
            is_2d_run=is_2d,
            task_name=aux_name,
            task_cfg={},
        )
        for aux_name, aux_arr in aux_outputs_np.items()
    }

    train_input_range = (
        _compute_display_value_range(train_inp_np, is_2d_run=is_2d) if train_inp_np is not None else None
    )
    train_target_ranges = {
        t_name: _compute_display_value_range(
            t_arr,
            is_2d_run=is_2d,
            task_name=t_name,
            task_cfg=tasks_dict.get(t_name, {}),
        )
        for t_name, t_arr in train_targets_np.items()
    }
    train_pred_ranges = {
        t_name: _compute_display_value_range(
            p_arr,
            is_2d_run=is_2d,
            task_name=t_name,
            task_cfg=tasks_dict.get(t_name, {}),
        )
        for t_name, p_arr in train_preds_np.items()
    }
    train_aux_ranges = {
        aux_name: _compute_display_value_range(
            aux_arr,
            is_2d_run=is_2d,
            task_name=aux_name,
            task_cfg={},
        )
        for aux_name, aux_arr in train_aux_outputs_np.items()
    }

    unlabeled_input_range = (
        _compute_display_value_range(unlabeled_inp_np, is_2d_run=is_2d) if unlabeled_inp_np is not None else None
    )
    unlabeled_pseudo_ranges = {
        t_name: _compute_display_value_range(
            p_arr,
            is_2d_run=is_2d,
            task_name=t_name,
            task_cfg=tasks_dict.get(t_name, {}),
        )
        for t_name, p_arr in unlabeled_pseudo_np.items()
    }
    unlabeled_pred_ranges = {
        t_name: _compute_display_value_range(
            p_arr,
            is_2d_run=is_2d,
            task_name=t_name,
            task_cfg=tasks_dict.get(t_name, {}),
        )
        for t_name, p_arr in unlabeled_preds_np.items()
    }
    
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
        val_imgs = [add_text_label(convert_slice_to_bgr(inp_np, value_range=val_input_range), "Val Input")]
        for aux_name in sorted(aux_outputs_np.keys()):
            aux_arr = aux_outputs_np[aux_name]
            aux_slice = aux_arr[0] if aux_arr.ndim == 3 and aux_arr.shape[0] == 1 else aux_arr
            val_imgs.append(
                add_text_label(
                    convert_slice_to_bgr(
                        aux_slice,
                        value_range=aux_ranges.get(aux_name),
                    ),
                    _format_aux_panel_label(aux_name),
                )
            )
        
        # Show all targets (including skeleton data)
        for t_name in sorted(targets_np.keys()):
            gt = targets_np[t_name]
            gt_slice = gt[0] if gt.shape[0] == 1 else gt
            label = f"Skel {t_name.replace('_skel', '')}" if t_name.endswith('_skel') else f"GT {t_name}"
            val_imgs.append(
                add_text_label(
                    convert_slice_to_bgr(
                        gt_slice,
                        task_name=t_name,
                        task_cfg=tasks_dict.get(t_name, {}),
                        value_range=target_ranges.get(t_name),
                    ),
                    label,
                )
            )
        
        # Show predictions (only for actual model outputs)
        for t_name in pred_task_names:
            pred = preds_np[t_name]
            pred_slice = pred[0] if pred.ndim == 3 and pred.shape[0] == 1 else pred
            val_imgs.append(
                add_text_label(
                    convert_slice_to_bgr(
                        pred_slice,
                        task_name=t_name,
                        task_cfg=tasks_dict.get(t_name, {}),
                        value_range=pred_ranges.get(t_name),
                    ),
                    f"Pred {t_name}",
                )
            )
        
        rows.append(np.hstack(val_imgs))
        
        # Train row if available
        if train_inp_np is not None:
            train_imgs = [add_text_label(convert_slice_to_bgr(train_inp_np, value_range=train_input_range), "Train Input")]
            for aux_name in sorted(train_aux_outputs_np.keys()):
                aux_arr = train_aux_outputs_np[aux_name]
                aux_slice = aux_arr[0] if aux_arr.ndim == 3 and aux_arr.shape[0] == 1 else aux_arr
                train_imgs.append(
                    add_text_label(
                        convert_slice_to_bgr(
                            aux_slice,
                            value_range=train_aux_ranges.get(aux_name),
                        ),
                        _format_aux_panel_label(aux_name),
                    )
                )
            
            # Show all train targets (including skeleton data)
            for t_name in sorted(train_targets_np.keys()):
                gt = train_targets_np[t_name]
                gt_slice = gt[0] if gt.shape[0] == 1 else gt
                label = f"Skel {t_name.replace('_skel', '')}" if t_name.endswith('_skel') else f"GT {t_name}"
                train_imgs.append(
                    add_text_label(
                        convert_slice_to_bgr(
                            gt_slice,
                            task_name=t_name,
                            task_cfg=tasks_dict.get(t_name, {}),
                            value_range=train_target_ranges.get(t_name),
                        ),
                        label,
                    )
                )
            
            # Show train predictions (only for actual model outputs)
            for t_name in pred_task_names:
                if t_name in train_preds_np:
                    pred = train_preds_np[t_name]
                    pred_slice = pred[0] if pred.ndim == 3 and pred.shape[0] == 1 else pred
                    train_imgs.append(
                        add_text_label(
                            convert_slice_to_bgr(
                                pred_slice,
                                task_name=t_name,
                                task_cfg=tasks_dict.get(t_name, {}),
                                value_range=train_pred_ranges.get(t_name),
                            ),
                            f"Pred {t_name}",
                        )
                    )

            rows.append(np.hstack(train_imgs))

        # Unlabeled row if available (for semi-supervised training)
        if unlabeled_inp_np is not None:
            unlabeled_imgs = [add_text_label(convert_slice_to_bgr(unlabeled_inp_np, value_range=unlabeled_input_range), "Unlabeled")]

            # Show pseudo-labels (teacher predictions)
            for t_name in sorted(unlabeled_pseudo_np.keys()):
                pseudo = unlabeled_pseudo_np[t_name]
                pseudo_slice = pseudo[0] if pseudo.ndim == 3 and pseudo.shape[0] == 1 else pseudo
                unlabeled_imgs.append(
                    add_text_label(
                        convert_slice_to_bgr(
                            pseudo_slice,
                            task_name=t_name,
                            task_cfg=tasks_dict.get(t_name, {}),
                            value_range=unlabeled_pseudo_ranges.get(t_name),
                        ),
                        f"Pseudo {t_name}",
                    )
                )

            # Show student predictions on unlabeled data
            for t_name in pred_task_names:
                if t_name in unlabeled_preds_np:
                    pred = unlabeled_preds_np[t_name]
                    pred_slice = pred[0] if pred.ndim == 3 and pred.shape[0] == 1 else pred
                    unlabeled_imgs.append(
                        add_text_label(
                            convert_slice_to_bgr(
                                pred_slice,
                                task_name=t_name,
                                task_cfg=tasks_dict.get(t_name, {}),
                                value_range=unlabeled_pred_ranges.get(t_name),
                            ),
                            f"Pred {t_name}",
                        )
                    )

            rows.append(np.hstack(unlabeled_imgs))

        # Stack rows and save
        rows = _pad_rows_to_uniform_width(rows)
        final_img = np.vstack(rows)
        preview_img = np.ascontiguousarray(final_img, dtype=np.uint8)
        if save_media:
            out_dir = Path(save_path).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Epoch {epoch}] Saving PNG to: {save_path}")
            # Use PIL for saving
            Image.fromarray(final_img).save(save_path)
        return None, preview_img

    else:
        # Build frames for 3D GIF
        frames = []
        z_dim = inp_np.shape[0] if inp_np.ndim == 3 else inp_np.shape[1]
        mid_z_idx = max(z_dim // 2, 0)

        val_input_bgr = _render_3d_volume_to_bgr(inp_np, value_range=val_input_range)
        target_panel_volumes = {
            t_name: _render_3d_volume_to_bgr(
                t_arr,
                task_name=t_name,
                task_cfg=tasks_dict.get(t_name, {}),
                value_range=target_ranges.get(t_name),
            )
            for t_name, t_arr in targets_np.items()
        }
        pred_panel_volumes = {
            t_name: _render_3d_volume_to_bgr(
                p_arr,
                task_name=t_name,
                task_cfg=tasks_dict.get(t_name, {}),
                value_range=pred_ranges.get(t_name),
            )
            for t_name, p_arr in preds_np.items()
        }
        aux_panel_volumes = {
            aux_name: _render_3d_volume_to_bgr(
                aux_arr,
                value_range=aux_ranges.get(aux_name),
            )
            for aux_name, aux_arr in aux_outputs_np.items()
        }

        train_input_bgr = (
            _render_3d_volume_to_bgr(train_inp_np, value_range=train_input_range)
            if train_inp_np is not None else None
        )
        train_target_panel_volumes = {
            t_name: _render_3d_volume_to_bgr(
                t_arr,
                task_name=t_name,
                task_cfg=tasks_dict.get(t_name, {}),
                value_range=train_target_ranges.get(t_name),
            )
            for t_name, t_arr in train_targets_np.items()
        }
        train_pred_panel_volumes = {
            t_name: _render_3d_volume_to_bgr(
                p_arr,
                task_name=t_name,
                task_cfg=tasks_dict.get(t_name, {}),
                value_range=train_pred_ranges.get(t_name),
            )
            for t_name, p_arr in train_preds_np.items()
        }
        train_aux_panel_volumes = {
            aux_name: _render_3d_volume_to_bgr(
                aux_arr,
                value_range=train_aux_ranges.get(aux_name),
            )
            for aux_name, aux_arr in train_aux_outputs_np.items()
        }

        unlabeled_input_bgr = (
            _render_3d_volume_to_bgr(unlabeled_inp_np, value_range=unlabeled_input_range)
            if unlabeled_inp_np is not None else None
        )
        unlabeled_pseudo_panel_volumes = {
            t_name: _render_3d_volume_to_bgr(
                p_arr,
                task_name=t_name,
                task_cfg=tasks_dict.get(t_name, {}),
                value_range=unlabeled_pseudo_ranges.get(t_name),
            )
            for t_name, p_arr in unlabeled_pseudo_np.items()
        }
        unlabeled_pred_panel_volumes = {
            t_name: _render_3d_volume_to_bgr(
                p_arr,
                task_name=t_name,
                task_cfg=tasks_dict.get(t_name, {}),
                value_range=unlabeled_pred_ranges.get(t_name),
            )
            for t_name, p_arr in unlabeled_preds_np.items()
        }

        def _build_3d_frame(z_idx: int) -> np.ndarray:
            rows = []

            val_imgs = [add_text_label(val_input_bgr[z_idx], "Val Input")]
            for aux_name in sorted(aux_outputs_np.keys()):
                val_imgs.append(add_text_label(aux_panel_volumes[aux_name][z_idx], _format_aux_panel_label(aux_name)))
            for t_name in sorted(targets_np.keys()):
                label = f"Skel {t_name.replace('_skel', '')}" if t_name.endswith('_skel') else f"GT {t_name}"
                val_imgs.append(add_text_label(target_panel_volumes[t_name][z_idx], label))
            for t_name in pred_task_names:
                val_imgs.append(add_text_label(pred_panel_volumes[t_name][z_idx], f"Pred {t_name}"))
            rows.append(np.hstack(val_imgs))

            if train_input_bgr is not None:
                train_imgs = [add_text_label(train_input_bgr[z_idx], "Train Input")]
                for aux_name in sorted(train_aux_outputs_np.keys()):
                    train_imgs.append(add_text_label(train_aux_panel_volumes[aux_name][z_idx], _format_aux_panel_label(aux_name)))
                for t_name in sorted(train_targets_np.keys()):
                    label = f"Skel {t_name.replace('_skel', '')}" if t_name.endswith('_skel') else f"GT {t_name}"
                    train_imgs.append(add_text_label(train_target_panel_volumes[t_name][z_idx], label))
                for t_name in pred_task_names:
                    if t_name in train_pred_panel_volumes:
                        train_imgs.append(add_text_label(train_pred_panel_volumes[t_name][z_idx], f"Pred {t_name}"))
                rows.append(np.hstack(train_imgs))

            if unlabeled_input_bgr is not None:
                unlabeled_imgs = [add_text_label(unlabeled_input_bgr[z_idx], "Unlabeled")]
                for t_name in sorted(unlabeled_pseudo_np.keys()):
                    unlabeled_imgs.append(add_text_label(unlabeled_pseudo_panel_volumes[t_name][z_idx], f"Pseudo {t_name}"))
                for t_name in pred_task_names:
                    if t_name in unlabeled_pred_panel_volumes:
                        unlabeled_imgs.append(add_text_label(unlabeled_pred_panel_volumes[t_name][z_idx], f"Pred {t_name}"))
                rows.append(np.hstack(unlabeled_imgs))

            rows = _pad_rows_to_uniform_width(rows)
            return np.ascontiguousarray(np.vstack(rows), dtype=np.uint8)

        if not save_media:
            return None, _build_3d_frame(mid_z_idx)

        for z_idx in range(z_dim):
            frame = _build_3d_frame(z_idx)
            frames.append(frame)

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
            preview_frame = frames[mid_z_idx].copy() if frames else None
            return None, preview_frame

        if proc.exitcode == 0:
            print(f"Successfully saved GIF to: {save_path}")
            preview_frame = frames[mid_z_idx].copy() if frames else None
            return frames, preview_frame
        else:
            print(f"Warning: GIF save failed in subprocess (exit code {proc.exitcode}); skipping")
            preview_frame = frames[mid_z_idx].copy() if frames else None
            return None, preview_frame


def apply_activation_if_needed(activation_str):
    """This function is no longer needed but kept for compatibility"""
    pass
