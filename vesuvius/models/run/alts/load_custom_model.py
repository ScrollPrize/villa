import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

from models.build.build_network_from_config import NetworkFromConfig
from models.config_manager import ConfigManager


def load_vesuvius_model_for_inference(
    config_file: str,
    checkpoint_path: Optional[str] = None,
    device_str: str = 'cuda',
    use_mirroring: bool = True,
    patch_size: Optional[Tuple[int, int, int]] = None,
    verbose: bool = False,
    rank: int = 0
) -> Dict[str, Any]:
    """
    Load a trained NetworkFromConfig model for inference with the same interface
    as nnUNet's load_model_for_inference function.
    
    Args:
        config_file: Path to the configuration file
        checkpoint_path: Optional path to a checkpoint file
        device_str: Device to run inference on ('cuda' or 'cpu')
        use_mirroring: Enable test time augmentation via mirroring
        patch_size: Optional override for the patch size
        verbose: Enable detailed output messages during loading
        rank: Process rank for distributed processing
        
    Returns:
        model_info: Dictionary with model information and parameters
    """
    if verbose and rank == 0:
        print(f"Loading Vesuvius model from config: {config_file}")
        print(f"Test time augmentation (mirroring): {'enabled' if use_mirroring else 'disabled'}")
    
    # Create the config manager
    mgr = ConfigManager(config_file)
    
    # Build the network
    network = NetworkFromConfig(mgr)
    
    # Move to correct device
    device = torch.device(device_str)
    network = network.to(device)
    
    # Load checkpoint if provided
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        if verbose and rank == 0:
            print(f"Loading checkpoint from {checkpoint_path}")
        
        try:
            # Try with weights_only=False first (required for PyTorch 2.0+)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if verbose and rank == 0:
                print("Loaded checkpoint with weights_only=False")
        except (TypeError, ValueError):
            # Fallback for older PyTorch versions
            if verbose and rank == 0:
                print("Falling back to loading without weights_only parameter")
            checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            # Our standard format from train.py
            network.load_state_dict(checkpoint['model'])
            epoch = checkpoint.get('epoch', 0)
            if verbose and rank == 0:
                print(f"Loaded weights from epoch {epoch}")
        elif 'network_weights' in checkpoint:
            # nnUNet format
            network.load_state_dict(checkpoint['network_weights'])
            if verbose and rank == 0:
                print(f"Loaded weights from nnUNet format checkpoint")
        else:
            # Direct model state dict
            network.load_state_dict(checkpoint)
            if verbose and rank == 0:
                print(f"Loaded weights directly from state dict")
    
    # Set to evaluation mode
    network.eval()
    
    # Compile by default unless explicitly disabled
    should_compile = True
    if 'nnUNet_compile' in os.environ.keys():
        should_compile = os.environ['nnUNet_compile'].lower() in ('true', '1', 't')
    
    if should_compile:
        if verbose and rank == 0:
            print('Using torch.compile for potential performance improvement')
        try:
            network = torch.compile(network)
        except Exception as e:
            if verbose and rank == 0:
                print(f"Warning: Could not compile model: {e}")
                print("Continuing with uncompiled model")
    
    # Get input/output information
    in_channels = mgr.in_channels
    
    # Use the first target as the main target for compatibility with inference
    if hasattr(network, 'targets') and network.targets:
        targets = network.targets
        target_name = list(targets.keys())[0]
        target_info = targets[target_name]
        out_channels = target_info.get("out_channels", 2)  # Default to 2 for binary segmentation
    else:
        # Fallback if targets not available
        target_name = "segmentation"
        out_channels = 2  # Default to standard binary segmentation
    
    # Use provided patch_size or from config
    if patch_size is None:
        if hasattr(network, 'patch_size'):
            patch_size = network.patch_size
        else:
            patch_size = mgr.train_patch_size
    
    # Return with the same interface as nnUNet's load_model_for_inference
    model_info = {
        'network': network,
        'checkpoint': checkpoint if checkpoint_path is not None else None,
        'num_input_channels': in_channels,
        'num_seg_heads': out_channels,
        'patch_size': patch_size,
        'use_mirroring': use_mirroring,
        'mirror_axes': (0, 1, 2) if use_mirroring else None,
        'verbose': verbose
    }
    
    if verbose and rank == 0:
        print(f"Model loaded successfully from {config_file}")
        if checkpoint_path:
            print(f"Using checkpoint: {checkpoint_path}")
        print(f"Model type: {type(network).__name__}")
        print(f"Model is on device: {next(network.parameters()).device}")
        print(f"Input channels: {in_channels}")
        print(f"Output channels: {out_channels}")
        print(f"Expected patch size: {patch_size}")
    
    return model_info