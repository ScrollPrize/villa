import os
import numpy as np
import torch
import vc_reader.transforms as A
import napari
from pathlib import Path
import torch.nn as nn
from tqdm.auto import tqdm
import contextlib
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
import itertools

from vesuvius.models.run.blending import generate_gaussian_map
from vesuvius.utils.models.helpers import compute_steps_for_sliding_window

# Model loader class
class ModelLoader:
    """
    Class for loading models from checkpoints with optional config manager integration.
    """
    
    def __init__(self, checkpoint_path, config_manager=None):
        self.checkpoint_path = Path(checkpoint_path)
        self.config_manager = config_manager
        
    def load(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
        checkpoint_data = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        # Try to extract model configuration
        model_config = {}
        if 'model_config' in checkpoint_data:
            model_config = checkpoint_data['model_config']
        elif 'config' in checkpoint_data:
            model_config = checkpoint_data['config']
        elif self.config_manager is not None:
            # Use config from config manager
            model_config = self.config_manager.model_config
            print(f"Using model configuration from config manager")
        else:
            print(f"Warning: No model configuration found in checkpoint. Using empty config.")
        
        from vesuvius.models.build.build_network_from_config import NetworkFromConfig
        
        class MinimalConfigManager:
            def __init__(self, model_config):
                self.model_config = model_config
                self.inference_config = model_config
                self.targets = model_config.get('targets', {})
                self.train_patch_size = model_config.get('patch_size', (128, 128, 128))
                self.train_batch_size = model_config.get('batch_size', 1)
                self.in_channels = model_config.get('in_channels', 1)
                self.autoconfigure = model_config.get('autoconfigure', False)
                
                self.spacing = [1] * len(self.train_patch_size)
                self.model_name = "Loaded_Model"
                
        if self.config_manager is not None:
            mgr = self.config_manager
        else:
            mgr = MinimalConfigManager(model_config)
            
        model = NetworkFromConfig(mgr)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        model_state_dict = None
        if 'model' in checkpoint_data:
            print("Loading weights from 'model' key in checkpoint")
            model_state_dict = checkpoint_data['model']
        elif 'model_state_dict' in checkpoint_data:
            model_state_dict = checkpoint_data['model_state_dict']
        elif 'state_dict' in checkpoint_data:
            model_state_dict = checkpoint_data['state_dict']
        else:
            model_state_dict = checkpoint_data
        
        # Check if this is a compiled model state dict (keys will have "_orig_mod." prefix)
        is_compiled_state_dict = any("_orig_mod." in key for key in model_state_dict.keys())
        
        # If we're on CUDA and the state dict is from a compiled model,
        # compile our model before loading weights
        if device.type == 'cuda' and is_compiled_state_dict:
            print("On CUDA with compiled model checkpoint - compiling model before loading weights")
            model = torch.compile(model)
            
        # Load the state dict
        try:
            model.load_state_dict(model_state_dict, strict=False)
            print("Model weights loaded successfully")
        except Exception as e:
            print(f"Warning: Error loading state dict: {str(e)}")
            print("Attempting to continue with partially loaded weights")
                
        return model, model_config

def _as_outputs(outputs) -> Dict[str, torch.Tensor]:
    if isinstance(outputs, torch.Tensor):
        return {"output": outputs}
    if isinstance(outputs, tuple) and all(isinstance(o, torch.Tensor) for o in outputs):
        return {f"output_{i}": tensor for i, tensor in enumerate(outputs)}
    if isinstance(outputs, dict):
        return outputs
    raise ValueError(f"Unexpected output format: {type(outputs)}")


def sliding_window_inference(model: nn.Module,
                             image: np.ndarray,
                             patch_size: Tuple[int, ...],
                             overlap: float = 0.25,
                             batch_size: int = 2,
                             sigma_scale: float = 8.0,
                             post_transform: Optional[Callable] = None) -> Dict[str, np.ndarray]:
    """Gaussian-blended sliding-window prediction of an in-memory image.

    Patch positions and blend weights are those of ``vesuvius.predict`` /
    ``vesuvius.blend_logits``: per-axis steps from
    ``compute_steps_for_sliding_window`` with step fraction ``1 - overlap``,
    weights from ``generate_gaussian_map`` (sigma = patch / sigma_scale).
    ``image`` is spatial ``(Y, X)`` / ``(Z, Y, X)`` or channel-first
    ``(C, ...)``; spatial axes smaller than the patch are zero-padded.
    Returns ``(1, C_out, ...)`` arrays per model output.
    """
    patch_size = tuple(int(p) for p in patch_size)
    dims = len(patch_size)
    if image.ndim == dims:
        image = image[None]
    elif image.ndim != dims + 1:
        raise ValueError(f"{dims}D patch needs a {dims}D or channel-first {dims + 1}D image, got shape {image.shape}")

    device = next(model.parameters()).device
    model.eval()
    inputs = A.Normalize("minmax", eps=1e-5)(torch.as_tensor(np.asarray(image), dtype=torch.float32, device=device))[None]

    spatial = tuple(inputs.shape[2:])
    padded = tuple(max(s, p) for s, p in zip(spatial, patch_size))
    if padded != spatial:
        pad = []
        for s, p in reversed(list(zip(spatial, padded))):
            pad += [0, p - s]
        inputs = torch.nn.functional.pad(inputs, pad, mode="constant", value=0.0)

    step = 1.0 - overlap
    positions = list(itertools.product(*(compute_steps_for_sliding_window(s, p, step)
                                          for s, p in zip(padded, patch_size))))
    weight = torch.as_tensor(generate_gaussian_map(patch_size, sigma_scale=sigma_scale)[0], device=device)
    weight_sum = torch.zeros(padded, dtype=torch.float32, device=device)
    sums: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for start in tqdm(range(0, len(positions), batch_size), desc="Sliding window"):
            batch_pos = positions[start:start + batch_size]
            windows = [tuple(slice(o, o + p) for o, p in zip(pos, patch_size)) for pos in batch_pos]
            batch = torch.cat([inputs[(slice(None), slice(None)) + w] for w in windows])
            for name, pred in _as_outputs(model(batch)).items():
                acc = sums.get(name)
                if acc is None:
                    acc = sums[name] = torch.zeros((pred.shape[1],) + padded, dtype=torch.float32, device=device)
                for i, w in enumerate(windows):
                    acc[(slice(None),) + w] += pred[i].float() * weight
            for w in windows:
                weight_sum[w] += weight

    crop = tuple(slice(0, s) for s in spatial)
    weight_sum = weight_sum[crop]
    result = {}
    for name, acc in sums.items():
        acc = acc[(slice(None),) + crop]
        blended = torch.where(weight_sum > 0, acc / weight_sum, acc)[None]
        if post_transform is not None:
            blended = post_transform(blended)
        result[name] = blended.cpu().numpy()
    return result


def run_inference(viewer: napari.Viewer,
                  layer: napari.layers.Layer,
                  checkpoint_path: str,
                  roi_size: Optional[Tuple[int, ...]] = None,
                  overlap: float = 0.25,
                  batch_size: int = 2,
                  sigma_scale: float = 8.0,
                  activation_type: str = "none") -> List[napari.layers.Layer]:

    config_manager = None
    try:
        from .main_window import _config_manager
        if _config_manager is not None:
            config_manager = _config_manager
            print("Using global config manager from main_window")
    except (ImportError, AttributeError):
        print("No global config manager available")

    loader = ModelLoader(checkpoint_path, config_manager=config_manager)
    model, model_config = loader.load()
    model_name = Path(checkpoint_path).stem
    image_data = layer.data

    if roi_size is None:
        for key in ('patch_size', 'train_patch_size'):
            if key in model_config:
                roi_size = tuple(model_config[key])
                print(f"Using {key} from model_config: {roi_size}")
                break
        else:
            raise ValueError("No patch_size or train_patch_size found in model config. Cannot proceed with inference.")

    # Dimensionality comes from the model patch, not the raw data: napari
    # images are often (C, H, W), which is 2D inference with a 2D patch.
    if len(roi_size) not in (2, 3):
        raise ValueError(f"roi_size must be 2D or 3D, got {roi_size}")
    print(f"Final roi_size for inference: {roi_size}")
    print(f"Data shape: {image_data.shape}, Using {len(roi_size)}D inference")

    post_transform = None
    activation = activation_type.lower()
    if activation == "sigmoid":
        post_transform = torch.sigmoid
    elif activation == "softmax":
        post_transform = lambda x: torch.softmax(x, dim=1)
    elif activation == "argmax":
        post_transform = lambda x: torch.argmax(x, dim=1, keepdim=True).float()

    results = sliding_window_inference(model, np.asarray(image_data), roi_size, overlap=overlap,
                                       batch_size=batch_size, sigma_scale=sigma_scale,
                                       post_transform=post_transform)
    new_layers = []
    for output_name, result in results.items():
        new_layers.append(viewer.add_image(result, name=f"{layer.name}_{model_name}_{output_name}",
                                           colormap='gray', blending='translucent'))

    print(f"Inference completed. Added {len(new_layers)} new layers to the viewer.")
    return new_layers
