"""
Checkpoint saving and management utilities for model training.
"""
import os
from pathlib import Path
from collections import deque
import torch


def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path, 
                   model_config=None, train_dataset=None, additional_data=None):
    """
    Save a training checkpoint with all necessary state information.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to save
    optimizer : torch.optim.Optimizer
        The optimizer state to save
    scheduler : torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler state to save
    epoch : int
        Current epoch number
    checkpoint_path : str or Path
        Path where to save the checkpoint
    model_config : dict, optional
        Model configuration to include in checkpoint
    train_dataset : Dataset, optional
        Training dataset to extract normalization info from
    additional_data : dict, optional
        Any additional data to include in the checkpoint
        
    Returns
    -------
    dict
        The checkpoint dictionary that was saved
    """
    # Unwrap compiled or DDP-wrapped modules for clean state_dict keys
    def _unwrap_model(m):
        # DDP unwrap
        if hasattr(m, 'module'):
            m = m.module
        # torch.compile unwrap
        # Compiled modules typically have attribute `_orig_mod`
        if hasattr(m, '_orig_mod'):
            try:
                m = m._orig_mod
            except Exception:
                pass
        return m

    model_to_save = _unwrap_model(model)

    checkpoint_data = {
        'model': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    
    if model_config is not None:
        checkpoint_data['model_config'] = model_config
    
    if train_dataset is not None:
        if hasattr(train_dataset, 'normalization_scheme'):
            checkpoint_data['normalization_scheme'] = train_dataset.normalization_scheme
        if hasattr(train_dataset, 'intensity_properties'):
            checkpoint_data['intensity_properties'] = train_dataset.intensity_properties
    
    if additional_data is not None:
        checkpoint_data.update(additional_data)
    
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")
    
    return checkpoint_data


def manage_checkpoint_history(checkpoint_history, best_checkpoints, epoch, 
                            checkpoint_path, validation_loss, 
                            checkpoint_dir, model_name,
                            max_recent=3, max_best=2):
    """
    Manage checkpoint history by keeping only the most recent and best checkpoints.
    
    Parameters
    ----------
    checkpoint_history : deque
        Deque of (epoch, path) tuples for recent checkpoints
    best_checkpoints : list
        List of (val_loss, epoch, path) tuples for best checkpoints
    epoch : int
        Current epoch number
    checkpoint_path : str or Path
        Path to the current checkpoint
    validation_loss : float
        Validation loss for the current epoch
    checkpoint_dir : str or Path
        Directory containing checkpoints
    model_name : str
        Model name used in checkpoint filenames
    max_recent : int, default=3
        Maximum number of recent checkpoints to keep
    max_best : int, default=2
        Maximum number of best checkpoints to keep
        
    Returns
    -------
    tuple
        Updated (checkpoint_history, best_checkpoints)
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_dir = Path(checkpoint_dir)
    
    checkpoint_history.append((epoch, str(checkpoint_path)))
    
    if epoch in [e for e, _ in checkpoint_history]:
        ckpt_path = next(p for e, p in checkpoint_history if e == epoch)
        best_checkpoints.append((validation_loss, epoch, ckpt_path))
        best_checkpoints.sort(key=lambda x: x[0]) 
        
        if len(best_checkpoints) > max_best:
            _, _, removed_path = best_checkpoints.pop(-1) 
            if removed_path not in [p for _, p in checkpoint_history]:
                if Path(removed_path).exists():
                    Path(removed_path).unlink()
                    print(f"Removed checkpoint with higher validation loss: {removed_path}")
    
    all_checkpoints_to_keep = set()
    
    for _, ckpt_path in checkpoint_history:
        all_checkpoints_to_keep.add(Path(ckpt_path))
    
    for _, _, ckpt_path in best_checkpoints[:max_best]:
        all_checkpoints_to_keep.add(Path(ckpt_path))
    
    for ckpt_file in checkpoint_dir.glob(f"{model_name}_epoch*.pth"):
        if ckpt_file not in all_checkpoints_to_keep:
            ckpt_file.unlink()
            print(f"Removed checkpoint: {ckpt_file}")
    
    print(f"\nCheckpoint management:")
    # Display 1-based epoch numbers for user-facing logs
    print(f"  Last {max_recent} checkpoints: {[f'epoch{e+1}' for e, _ in checkpoint_history]}")
    if best_checkpoints:
        print(f"  Best {max_best} checkpoints: {[f'epoch{e+1} (loss={l:.4f})' for l, e, _ in best_checkpoints[:max_best]]}")
    
    return checkpoint_history, best_checkpoints


def manage_debug_gifs(debug_gif_history, best_debug_gifs, epoch, 
                     gif_path, validation_loss,
                     checkpoint_dir, model_name,
                     max_recent=3, max_best=2):
    """
    Manage debug GIF history by keeping only the most recent and best GIFs.
    
    Parameters
    ----------
    debug_gif_history : deque
        Deque of (epoch, path) tuples for recent debug GIFs
    best_debug_gifs : list
        List of (val_loss, epoch, path) tuples for best debug GIFs
    epoch : int
        Current epoch number
    gif_path : str or Path
        Path to the current debug GIF
    validation_loss : float
        Validation loss for the current epoch
    checkpoint_dir : str or Path
        Directory containing checkpoints and debug GIFs
    model_name : str
        Model name used in GIF filenames
    max_recent : int, default=3
        Maximum number of recent GIFs to keep
    max_best : int, default=2
        Maximum number of best GIFs to keep
        
    Returns
    -------
    tuple
        Updated (debug_gif_history, best_debug_gifs)
    """
    gif_path = Path(gif_path)
    checkpoint_dir = Path(checkpoint_dir)
    
    debug_gif_history.append((epoch, str(gif_path)))
    
    if epoch in [e for e, _ in debug_gif_history]:
        gif_path_str = next(p for e, p in debug_gif_history if e == epoch)
        best_debug_gifs.append((validation_loss, epoch, gif_path_str))
        best_debug_gifs.sort(key=lambda x: x[0]) 
        
        if len(best_debug_gifs) > max_best:
            _, _, removed_gif = best_debug_gifs.pop(-1)
            if removed_gif not in [p for _, p in debug_gif_history]:
                if Path(removed_gif).exists():
                    Path(removed_gif).unlink()
                    print(f"Removed debug gif with higher validation loss: {removed_gif}")
    
    all_gifs_to_keep = set()
    
    for _, gif_path_str in debug_gif_history:
        all_gifs_to_keep.add(Path(gif_path_str))
    
    for _, _, gif_path_str in best_debug_gifs[:max_best]:
        all_gifs_to_keep.add(Path(gif_path_str))
    
    for gif_file in checkpoint_dir.glob(f"{model_name}_debug_epoch*.gif"):
        if gif_file not in all_gifs_to_keep:
            gif_file.unlink()
            print(f"Removed debug gif: {gif_file}")
    
    return debug_gif_history, best_debug_gifs


def cleanup_old_configs(model_ckpt_dir, model_name, keep_latest=1):
    """
    Clean up old configuration files, keeping only the most recent ones.
    
    Parameters
    ----------
    model_ckpt_dir : str or Path
        Directory containing model checkpoints and configs
    model_name : str
        Model name used in config filenames
    keep_latest : int, default=1
        Number of latest config files to keep
    """
    ckpt_dir_parent = Path(model_ckpt_dir)
    all_configs = sorted(
        ckpt_dir_parent.glob(f"{model_name}_*.yaml"),
        key=lambda x: x.stat().st_mtime
    )
    
    while len(all_configs) > keep_latest:
        oldest = all_configs.pop(0)
        oldest.unlink()
        print(f"Removed old config: {oldest}")


def save_final_checkpoint(model, optimizer, scheduler, max_epoch,
                         model_ckpt_dir, model_name, 
                         model_config=None, train_dataset=None):
    """
    Save the final model checkpoint at the end of training.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to save
    optimizer : torch.optim.Optimizer
        The optimizer state to save
    scheduler : torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler state to save
    max_epoch : int
        Maximum number of epochs (for recording in checkpoint)
    model_ckpt_dir : str or Path
        Directory to save the final checkpoint
    model_name : str
        Model name for the checkpoint filename
    model_config : dict, optional
        Model configuration to include in checkpoint
    train_dataset : Dataset, optional
        Training dataset to extract normalization info from
        
    Returns
    -------
    str
        Path to the saved final checkpoint
    """
    final_model_path = f"{model_ckpt_dir}/{model_name}_final.pth"
    
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=max_epoch - 1,
        checkpoint_path=final_model_path,
        model_config=model_config,
        train_dataset=train_dataset
    )
    
    print(f"Final model saved to {final_model_path}")
    print(f"Model configuration is embedded in the checkpoint")
    
    return final_model_path
