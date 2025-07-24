"""
Factory and orchestrator for auxiliary tasks in Vesuvius models.

This module provides factory functions for creating auxiliary tasks and
helper functions for managing auxiliary targets during training.
"""

from typing import Dict, List, Tuple, Any, Optional
import torch


def create_auxiliary_task(task_type: str, aux_task_name: str, aux_config: Dict[str, Any], 
                         source_target_name: str) -> Dict[str, Any]:
    """
    Factory function to create auxiliary task configurations.
    
    Parameters
    ----------
    task_type : str
        Type of auxiliary task ('distance_transform' or 'surface_normals')
    aux_task_name : str
        Name for the auxiliary task
    aux_config : dict
        Configuration for the auxiliary task
    source_target_name : str
        Name of the source target this auxiliary task depends on
        
    Returns
    -------
    dict
        Target configuration for the auxiliary task
    """
    if task_type == "distance_transform":
        from .aux_distance_transform import create_distance_transform_config
        return create_distance_transform_config(aux_task_name, aux_config, source_target_name)
    elif task_type == "surface_normals":
        from .aux_surface_normals import create_surface_normals_config
        return create_surface_normals_config(aux_task_name, aux_config, source_target_name)
    else:
        raise ValueError(f"Unknown auxiliary task type: {task_type}")


def compute_auxiliary_loss(loss_fn, t_pred: torch.Tensor, t_gt_masked: torch.Tensor, 
                          outputs: Dict[str, torch.Tensor], target_info: Dict[str, Any]) -> torch.Tensor:
    """
    Handle special loss computation for auxiliary tasks.
    
    Parameters
    ----------
    loss_fn : torch.nn.Module
        Loss function to use
    t_pred : torch.Tensor
        Predicted tensor
    t_gt_masked : torch.Tensor
        Ground truth tensor (possibly masked)
    outputs : dict
        Dictionary of all model outputs
    target_info : dict
        Target configuration information
        
    Returns
    -------
    torch.Tensor
        Computed loss value
    """
    # Check if this target has a source_target (indicating it's an auxiliary task)
    if 'source_target' in target_info:
        source_target_name = target_info['source_target']
        if source_target_name in outputs:
            # Try to pass source predictions as keyword argument
            # This way, losses that don't expect it won't break
            try:
                loss_value = loss_fn(t_pred, t_gt_masked, source_pred=outputs[source_target_name])
            except TypeError:
                # Fallback to standard call if loss doesn't accept source_pred
                loss_value = loss_fn(t_pred, t_gt_masked)
        else:
            loss_value = loss_fn(t_pred, t_gt_masked)
    else:
        loss_value = loss_fn(t_pred, t_gt_masked)
    
    return loss_value


def preserve_auxiliary_targets(targets: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Extract auxiliary targets from targets dictionary.
    
    Parameters
    ----------
    targets : dict
        Dictionary of all targets
        
    Returns
    -------
    dict
        Dictionary containing only auxiliary targets
    """
    auxiliary_targets = {}
    if targets:
        for target_name, target_info in targets.items():
            if target_info.get('auxiliary_task', False):
                auxiliary_targets[target_name] = target_info
    return auxiliary_targets


def restore_auxiliary_targets(targets: Dict[str, Dict[str, Any]], 
                            auxiliary_targets: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Re-add auxiliary targets after detection.
    
    Parameters
    ----------
    targets : dict
        Current targets dictionary
    auxiliary_targets : dict
        Dictionary of auxiliary targets to restore
        
    Returns
    -------
    dict
        Updated targets dictionary with auxiliary targets restored
    """
    if auxiliary_targets:
        targets.update(auxiliary_targets)
    return targets


def apply_auxiliary_tasks_from_config(mgr) -> None:
    """
    Apply auxiliary tasks from configuration manager.
    
    Parameters
    ----------
    mgr : ConfigManager
        Configuration manager instance
    """
    if hasattr(mgr, '_apply_auxiliary_tasks'):
        mgr._apply_auxiliary_tasks()
