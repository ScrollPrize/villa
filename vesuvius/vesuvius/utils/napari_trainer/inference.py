import os
import numpy as np
import torch
import napari
from pathlib import Path
import torch.nn as nn
from tqdm.auto import tqdm
import contextlib
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from monai.inferers import SlidingWindowInferer, SaliencyInferer, Inferer
from monai.transforms import AsDiscrete

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
        
        from models.build.build_network_from_config import NetworkFromConfig
        
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

class MonaiInferer:
    def __init__(self, 
                model: nn.Module,
                roi_size: Union[List[int], Tuple[int, ...]], 
                sw_batch_size: int = 1,
                overlap: float = 0.25,
                mode: str = "constant",
                sigma_scale: float = 0.125,
                padding_mode: str = "constant",
                cval: float = 0.0,
                sw_device = None,
                device = None,
                progress: bool = True,
                cache_roi_weight_map: bool = False,
                cpu_thresh = None,
                buffer_steps = None,
                buffer_dim: int = -1,):
        
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()
        
        self.sliding_window_inferer = SlidingWindowInferer(
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            mode=mode,
            sigma_scale=sigma_scale,
            padding_mode=padding_mode,
            cval=cval,
            sw_device=sw_device,
            device=self.device,
            progress=progress,
            cache_roi_weight_map=cache_roi_weight_map,
            cpu_thresh=cpu_thresh,
            buffer_steps=buffer_steps,
            buffer_dim=buffer_dim,
        )
        
    def infer(self, inputs: torch.Tensor, post_transform: Optional[Callable] = None) -> Dict[str, np.ndarray]:

        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(inputs)
        
        if inputs.dtype != torch.float32:
            inputs = inputs.to(dtype=torch.float32)
            
        if inputs.ndim == 2:  # (H, W)
            inputs = inputs.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif inputs.ndim == 3:  # (D, H, W) or (C, H, W)
            # Assume channel dimension is missing
            inputs = inputs.unsqueeze(0)  # (1, D, H, W) or (1, C, H, W)
        elif inputs.ndim == 4:  # (C, D, H, W) - 3D with channels or (B, C, H, W) - 2D with batch
            # Check if this is 3D data with channels or 2D data with batch
            if len(self.sliding_window_inferer.roi_size) == 3:  # 3D inference
                inputs = inputs.unsqueeze(0)  # (1, C, D, H, W)
            # Otherwise, leave as is - already has batch dimension
        

        inputs = inputs.to(device=self.device)
        
        def normalize(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-5)
            
        inputs = normalize(inputs)
        
        with torch.no_grad():
            outputs = self.sliding_window_inferer(inputs, self.model)
        
        result = {}
        
        if isinstance(outputs, torch.Tensor):
            # Single output tensor
            outputs = {"output": outputs}
        elif isinstance(outputs, tuple) and all(isinstance(o, torch.Tensor) for o in outputs):
            # Tuple of tensors
            outputs = {f"output_{i}": tensor for i, tensor in enumerate(outputs)}
        elif not isinstance(outputs, dict):
            raise ValueError(f"Unexpected output format: {type(outputs)}")
        
        for name, tensor in outputs.items():
            if post_transform is not None:
                tensor = post_transform(tensor)
                
            array = tensor.detach().cpu().numpy()
            result[name] = array
            
        return result


def run_inference(viewer: napari.Viewer, 
                       layer: napari.layers.Layer, 
                       checkpoint_path: str,
                       roi_size: Optional[Tuple[int, ...]] = None,
                       overlap: float = 0.25,
                       sw_batch_size: int = 2,
                       mode: str = "constant",
                       sigma_scale: float = 0.125,
                       padding_mode: str = "constant",
                       cval: float = 0.0,
                       sw_device = None,
                       progress: bool = True,
                       cache_roi_weight_map: bool = False,
                       cpu_thresh = None,
                       buffer_steps = None,
                       buffer_dim: int = -1,
                       with_coord: bool = False,
                       use_saliency: bool = False,
                       target_layers: Optional[List[str]] = None,
                       cam_name: str = "GradCAM",
                       class_idx: Optional[int] = None,
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
    checkpoint_path = Path(checkpoint_path)
    model_name = checkpoint_path.stem
    image_data = layer.data
    
    is_3d = image_data.ndim == 3 and image_data.shape[0] > 1
    
    if roi_size is None:
        if 'patch_size' in model_config:
            config_patch_size = model_config['patch_size']
            print(f"Using patch_size from model_config: {config_patch_size}")
            roi_size = tuple(config_patch_size)
        elif 'train_patch_size' in model_config:
            config_patch_size = model_config['train_patch_size']
            print(f"Using train_patch_size from model_config: {config_patch_size}")
            roi_size = tuple(config_patch_size)
        else:
            raise ValueError("No patch_size or train_patch_size found in model config. Cannot proceed with inference.")
    
    print(f"Final roi_size for inference: {roi_size}")
    print(f"Data shape: {image_data.shape}, Using {'3D' if is_3d else '2D'} inference")
    
    if is_3d and len(roi_size) != 3:
        raise ValueError(f"3D data requires 3D roi_size, but got {roi_size}")
    elif not is_3d and len(roi_size) != 2:
        raise ValueError(f"2D data requires 2D roi_size, but got {roi_size}")
    
    model_dims = len(roi_size)
    data_dims = 3 if is_3d else 2
    if model_dims != data_dims:
        raise ValueError(f"Model dimensionality ({model_dims}D) must match data dimensionality ({data_dims}D)")
    
    # Create inferer with all sliding window parameters
    inferer = MonaiInferer(
        model=model,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode=mode,
        sigma_scale=sigma_scale,
        padding_mode=padding_mode,
        cval=cval,
        sw_device=sw_device,
        progress=progress,
        cache_roi_weight_map=cache_roi_weight_map,
        cpu_thresh=cpu_thresh,
        buffer_steps=buffer_steps,
        buffer_dim=buffer_dim,
    )
    
    # Set up post-processing
    post_transform = None
    if activation_type.lower() == "sigmoid":
        post_transform = torch.sigmoid
    elif activation_type.lower() == "softmax":
        def softmax_transform(x):
            return torch.softmax(x, dim=1)
        post_transform = softmax_transform
    elif activation_type.lower() == "argmax":
        post_transform = AsDiscrete(argmax=True)
    
    # Run inference
    new_layers = []
    
    try:
        results = inferer.infer(image_data, post_transform=post_transform)
        
        for output_name, result in results.items():
            layer_name = f"{layer.name}_{model_name}_{output_name}"
            
            # Add as image layer
            new_layer = viewer.add_image(
                result,
                name=layer_name,
                colormap='gray',
                blending='translucent'
            )
            new_layers.append(new_layer)
            
    except Exception as e:
        print(f"Inference error: {str(e)}")
        raise e
    
    print(f"Inference completed. Added {len(new_layers)} new layers to the viewer.")
    return new_layers
