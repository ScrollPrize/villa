import torch
from pathlib import Path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, mgr, device, load_weights_only=False):
    """
    Load checkpoint and update model, optimizer, scheduler, and configuration.
    
    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the checkpoint file
    model : torch.nn.Module
        The model to load weights into
    optimizer : torch.optim.Optimizer
        The optimizer to load state into
    scheduler : torch.optim.lr_scheduler
        The scheduler to load state into
    mgr : ConfigManager
        Configuration manager instance
    device : torch.device
        Device to load the checkpoint on
    load_weights_only : bool
        If True, only load model weights, not optimizer/scheduler state
    
    Returns
    -------
    tuple
        (model, optimizer, scheduler, start_epoch, checkpoint_loaded)
        - model: Updated model (might be rebuilt)
        - optimizer: Updated optimizer (might be recreated)
        - scheduler: Updated scheduler  
        - start_epoch: Epoch to start training from
        - checkpoint_loaded: Boolean indicating if checkpoint was successfully loaded
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Check if checkpoint exists
    valid_checkpoint = (checkpoint_path is not None and 
                       str(checkpoint_path) != "" and 
                       checkpoint_path.exists())
    
    if not valid_checkpoint:
        print(f"No valid checkpoint found at {checkpoint_path}")
        return model, optimizer, scheduler, 0, False
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=load_weights_only)
    
    # Handle model configuration from checkpoint
    if 'model_config' in checkpoint:
        print("Found model configuration in checkpoint, using it to initialize the model")
        
        # Update targets from checkpoint
        if hasattr(mgr, 'targets') and 'targets' in checkpoint['model_config']:
            mgr.targets = checkpoint['model_config']['targets']
            print(f"Updated targets from checkpoint: {mgr.targets}")
    
    # Handle normalization scheme
    if 'normalization_scheme' in checkpoint:
        print(f"Found normalization scheme in checkpoint: {checkpoint['normalization_scheme']}")
        mgr.normalization_scheme = checkpoint['normalization_scheme']
        if hasattr(mgr, 'dataset_config'):
            mgr.dataset_config['normalization_scheme'] = checkpoint['normalization_scheme']
    
    # Handle intensity properties
    if 'intensity_properties' in checkpoint:
        print("Found intensity properties in checkpoint")
        mgr.intensity_properties = checkpoint['intensity_properties']
        if hasattr(mgr, 'dataset_config'):
            mgr.dataset_config['intensity_properties'] = checkpoint['intensity_properties']
        print("Loaded intensity properties:")
        for key, value in checkpoint['intensity_properties'].items():
            print(f"  {key}: {value:.4f}")
    
    # Check if model needs to be rebuilt due to autoconfigure mismatch
    if 'model_config' in checkpoint:
        checkpoint_autoconfigure = checkpoint['model_config'].get('autoconfigure', True)
        if hasattr(model, 'autoconfigure') and model.autoconfigure != checkpoint_autoconfigure:
            print("Model autoconfiguration differs, rebuilding model from checkpoint config")
            
            # Import here to avoid circular imports
            from models.build.build_network_from_config import NetworkFromConfig
            
            # Create a config wrapper that combines checkpoint config with mgr
            class ConfigWrapper:
                def __init__(self, config_dict, base_mgr):
                    self.__dict__.update(config_dict)
                    # Copy any missing attributes from base_mgr
                    for attr_name in dir(base_mgr):
                        if not attr_name.startswith('__') and not hasattr(self, attr_name):
                            setattr(self, attr_name, getattr(base_mgr, attr_name))
            
            config_wrapper = ConfigWrapper(checkpoint['model_config'], mgr)
            model = NetworkFromConfig(config_wrapper)
            
            # Move model to device and compile if CUDA
            model = model.to(device)
            if device.type == 'cuda':
                model = torch.compile(model)
            
            # Recreate optimizer with new model
            from models.training.optimizers import create_optimizer
            optimizer_config = {
                'name': mgr.optimizer,
                'learning_rate': mgr.initial_lr,
                'weight_decay': mgr.weight_decay
            }
            optimizer = create_optimizer(optimizer_config, model)
    
    # Load model weights
    model.load_state_dict(checkpoint['model'])
    
    # Determine start epoch
    start_epoch = 0
    
    if not load_weights_only:
        # Only load optimizer, scheduler, epoch if we are NOT in "weights_only" mode
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        # In weights_only mode, reinitialize scheduler
        from models.training.lr_schedulers import get_scheduler
        
        scheduler_type = getattr(mgr, 'scheduler', 'poly')
        scheduler_kwargs = getattr(mgr, 'scheduler_kwargs', {})
        
        scheduler, is_per_iteration_scheduler = get_scheduler(
            scheduler_type=scheduler_type,
            optimizer=optimizer,
            initial_lr=mgr.initial_lr,
            max_steps=mgr.max_epoch,
            **scheduler_kwargs
        )
        print("Loaded model weights only; starting new training run from epoch 1.")
    
    return model, optimizer, scheduler, start_epoch, True
