"""
Batch handling utilities for Theseus optimization.

This module provides helper functions for dealing with tensor batch dimensions
consistently across the codebase, especially for Theseus compatibility.
"""

from typing import Dict, List, Tuple, Union, Optional, Any
import torch
import logging

logger = logging.getLogger(__name__)


def ensure_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor has a batch dimension.
    
    If tensor doesn't have a batch dimension (first dimension), add one.
    This is useful for Theseus compatibility, which expects all tensors to have
    a batch dimension.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Tensor with batch dimension
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        
    if len(tensor.shape) == 0:  # Scalar tensor
        return tensor.unsqueeze(0)
    elif len(tensor.shape) == 1:  # 1D tensor, treat as a single vector without batch
        return tensor.unsqueeze(0)
    else:  # Already has batch dimension or higher-dimensional tensor
        return tensor


def get_batch_size(tensors: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]) -> int:
    """
    Get the batch size from tensor(s).
    
    Args:
        tensors: Single tensor, list of tensors, or dict of tensors
        
    Returns:
        Batch size (from first dimension)
    
    Raises:
        ValueError: If no valid tensors are found
    """
    if isinstance(tensors, torch.Tensor):
        return tensors.shape[0]
    elif isinstance(tensors, list):
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor) and len(tensor.shape) > 0:
                return tensor.shape[0]
    elif isinstance(tensors, dict):
        for name, tensor in tensors.items():
            if isinstance(tensor, torch.Tensor) and len(tensor.shape) > 0:
                return tensor.shape[0]
                
    raise ValueError("No valid tensors found to determine batch size")


def validate_batch_consistency(tensors_dict: Dict[str, torch.Tensor]) -> Tuple[bool, Optional[str], int]:
    """
    Validate that all tensors have consistent batch dimensions.
    
    Args:
        tensors_dict: Dictionary of tensors to validate
        
    Returns:
        Tuple of:
            - bool: True if all batch dimensions are consistent
            - str: Error message if inconsistent, None otherwise
            - int: The common batch size if consistent
    """
    batch_sizes = {}
    
    for name, tensor in tensors_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
            
        if len(tensor.shape) > 0:
            batch_size = tensor.shape[0]
            batch_sizes[name] = batch_size
    
    if not batch_sizes:
        return True, None, 1  # No tensors to validate, assume batch_size=1
        
    # Get unique batch sizes
    unique_batch_sizes = set(batch_sizes.values())
    
    if len(unique_batch_sizes) == 1:
        # All batch sizes are the same
        common_batch_size = next(iter(unique_batch_sizes))
        return True, None, common_batch_size
    else:
        # Inconsistent batch sizes
        error_msg = f"Inconsistent batch dimensions: {batch_sizes}"
        return False, error_msg, max(unique_batch_sizes)


def normalize_batch_sizes(tensors_dict: Dict[str, torch.Tensor], target_batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """
    Normalize batch dimensions across all tensors.
    
    If target_batch_size is None, use the maximum batch size found.
    For tensors with batch_size=1, broadcast to the target batch size.
    For tensors with batch_size>target, keep only the first target_batch_size elements.
    
    Args:
        tensors_dict: Dictionary of tensors to normalize
        target_batch_size: Target batch size (if None, use maximum found)
        
    Returns:
        Dictionary with normalized tensor batch dimensions
    """
    # Get current batch sizes
    batch_sizes = {}
    for name, tensor in tensors_dict.items():
        if isinstance(tensor, torch.Tensor) and len(tensor.shape) > 0:
            batch_sizes[name] = tensor.shape[0]
    
    if not batch_sizes:
        return tensors_dict  # No tensors to normalize
        
    # Determine target batch size
    if target_batch_size is None:
        target_batch_size = max(batch_sizes.values())
    
    # Create normalized dict
    normalized_dict = {}
    
    for name, tensor in tensors_dict.items():
        if not isinstance(tensor, torch.Tensor):
            normalized_dict[name] = tensor
            continue
            
        if len(tensor.shape) == 0:
            # Scalar tensor - unsqueeze twice to get [batch_size, 1]
            normalized = tensor.unsqueeze(0).unsqueeze(0).expand(target_batch_size, 1)
        elif len(tensor.shape) == 1:
            # 1D tensor - unsqueeze to get [batch_size, dim]
            dim = tensor.shape[0]
            normalized = tensor.unsqueeze(0).expand(target_batch_size, dim)
        else:
            # Already has batch dimension
            current_batch = tensor.shape[0]
            
            if current_batch == target_batch_size:
                # Already correct
                normalized = tensor
            elif current_batch == 1:
                # Broadcast batch dimension
                normalized = tensor.expand(target_batch_size, *tensor.shape[1:])
            elif current_batch > target_batch_size:
                # Take first target_batch_size elements
                normalized = tensor[:target_batch_size]
            else:
                # Batch size is too small but not 1
                # Repeat the tensor to reach target_batch_size
                repeats_needed = (target_batch_size + current_batch - 1) // current_batch
                repeated = tensor.repeat(repeats_needed, *([1] * (len(tensor.shape) - 1)))
                normalized = repeated[:target_batch_size]
                
        normalized_dict[name] = normalized
        
    return normalized_dict


def reshape_to_batch_dim_3d(tensor: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
    """
    Reshape tensor to have shape [batch_size, 3] for 3D points.
    
    This is particularly useful for 3D points that need to be in the proper
    shape for Theseus optimization.
    
    Args:
        tensor: Input tensor with 3D point data
        batch_size: Target batch size (if None, preserve existing batch)
        
    Returns:
        Tensor with shape [batch_size, 3]
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    
    # Handle scalar or 1D tensor
    if len(tensor.shape) == 0:
        raise ValueError("Cannot reshape scalar tensor to [batch_size, 3]")
    elif len(tensor.shape) == 1:
        if tensor.shape[0] != 3:
            raise ValueError(f"1D tensor must have exactly 3 elements to reshape to 3D point, got {tensor.shape[0]}")
        if batch_size is None:
            batch_size = 1
        return tensor.reshape(1, 3).expand(batch_size, 3)
        
    # Handle batch of points (2D+ tensor)
    if len(tensor.shape) == 2:
        # Already [batch, 3] or similar
        if tensor.shape[1] != 3:
            raise ValueError(f"2D tensor must have shape [batch_size, 3], got {tensor.shape}")
            
        if batch_size is not None and tensor.shape[0] != batch_size:
            if tensor.shape[0] == 1:
                # Broadcast single batch to target batch size
                return tensor.expand(batch_size, 3)
            elif batch_size == 1:
                # Take first element when reducing to batch_size=1
                return tensor[0:1, :]
            else:
                # Error for general case of batch size mismatch
                raise ValueError(f"Cannot change batch size from {tensor.shape[0]} to {batch_size} without losing data")
        
        return tensor
    else:
        # 3D+ tensor needs reshaping
        if tensor.shape[-1] != 3 and tensor.numel() % 3 != 0:
            raise ValueError(f"Cannot reshape tensor of shape {tensor.shape} to [batch_size, 3]")
            
        # Try to preserve batch dimension if possible
        if batch_size is None:
            batch_size = tensor.shape[0]
            
        # Check if total elements is compatible with [batch_size, 3]
        total_elements = tensor.numel()
        if total_elements != batch_size * 3:
            raise ValueError(f"Cannot reshape tensor with {total_elements} elements to shape [{batch_size}, 3]")
            
        return tensor.reshape(batch_size, 3)