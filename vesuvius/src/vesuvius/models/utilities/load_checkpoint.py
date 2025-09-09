import torch
from pathlib import Path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, mgr, device, load_weights_only=False):

    checkpoint_path = Path(checkpoint_path)

    valid_checkpoint = (checkpoint_path is not None and 
                       str(checkpoint_path) != "" and 
                       checkpoint_path.exists())
    
    if not valid_checkpoint:
        print(f"No valid checkpoint found at {checkpoint_path}")
        return model, optimizer, scheduler, 0, False
    
    print(f"Loading checkpoint from {checkpoint_path}")
    # Load raw object; do not assume a particular structure yet
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        print("Found model configuration in checkpoint, using it to initialize the model")

        if hasattr(mgr, 'targets') and 'targets' in checkpoint['model_config']:
            mgr.targets = checkpoint['model_config']['targets']
            print(f"Updated targets from checkpoint: {mgr.targets}")

    if isinstance(checkpoint, dict) and 'normalization_scheme' in checkpoint:
        print(f"Found normalization scheme in checkpoint: {checkpoint['normalization_scheme']}")
        mgr.normalization_scheme = checkpoint['normalization_scheme']
        if hasattr(mgr, 'dataset_config'):
            mgr.dataset_config['normalization_scheme'] = checkpoint['normalization_scheme']

    if isinstance(checkpoint, dict) and 'intensity_properties' in checkpoint:
        print("Found intensity properties in checkpoint")
        mgr.intensity_properties = checkpoint['intensity_properties']
        if hasattr(mgr, 'dataset_config'):
            mgr.dataset_config['intensity_properties'] = checkpoint['intensity_properties']
        print("Loaded intensity properties:")
        for key, value in checkpoint['intensity_properties'].items():
            print(f"  {key}: {value:.4f}")

    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        checkpoint_autoconfigure = checkpoint['model_config'].get('autoconfigure', True)
        force_rebuild = bool(getattr(mgr, 'rebuild_from_checkpoint_config', False))
        if force_rebuild or (hasattr(model, 'autoconfigure') and model.autoconfigure != checkpoint_autoconfigure):
            reason = "(forced by flag)" if force_rebuild else "(autoconfigure mismatch)"
            print(f"Rebuilding model from checkpoint config {reason}")

            from vesuvius.models.build.build_network_from_config import NetworkFromConfig
            
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

            model = model.to(device)

            from vesuvius.models.training.optimizers import create_optimizer
            optimizer_config = {
                'name': mgr.optimizer,
                'learning_rate': mgr.initial_lr,
                'weight_decay': mgr.weight_decay
            }
            optimizer = create_optimizer(optimizer_config, model)
            # Update mgr.model_config to reflect the rebuilt model
            try:
                mgr.model_config = checkpoint['model_config']
            except Exception:
                pass

    # --- Resolve model state dict --- #
    model_state = None
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
            model_state = checkpoint['model']
        elif 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
            model_state = checkpoint['state_dict']
        else:
            # Heuristic: if values look like tensors, treat as state_dict directly
            if all(hasattr(v, 'shape') for v in checkpoint.values()):
                model_state = checkpoint
    
    if model_state is None:
        raise ValueError(
            "Unsupported checkpoint format. Expected a dict with 'model' or 'state_dict', or a raw state_dict.")

    # Strip potential DataParallel/Distributed prefixes
    def _strip_module_prefix(sd):
        return { (k[7:] if k.startswith('module.') else k): v for k, v in sd.items() }

    model_state = _strip_module_prefix(model_state)

    # Optionally filter to matching keys/shapes when loading weights only
    target_state = model.state_dict()
    allow_partial = bool(load_weights_only)
    if allow_partial:
        model_state = {k: v for k, v in model_state.items()
                       if k in target_state and target_state[k].shape == v.shape}

    # Choose strictness: default to non-strict when weights-only
    strict = getattr(mgr, 'load_strict', not allow_partial)
    strict_loaded = False
    try:
        incompatible = model.load_state_dict(model_state, strict=strict)
        strict_loaded = True
        # Log any missing/unexpected keys for transparency
        try:
            missing = list(getattr(incompatible, 'missing_keys', []))
            unexpected = list(getattr(incompatible, 'unexpected_keys', []))
            if missing:
                print(f"Missing keys while loading: {len(missing)}")
                print(f"  e.g., {missing[:10]}")
            if unexpected:
                print(f"Unexpected keys while loading: {len(unexpected)}")
                print(f"  e.g., {unexpected[:10]}")
        except Exception:
            pass
    except RuntimeError as e:
        if strict and not allow_partial:
            print("Strict state_dict load failed. Falling back to partial non-strict load of matching weights.")
            # Filter to matching keys/shapes and load non-strict
            model_state_partial = {k: v for k, v in model_state.items()
                                   if k in target_state and target_state[k].shape == v.shape}
            missing_before = len([k for k in target_state.keys() if k not in model_state_partial])
            unexpected_before = len([k for k in model_state.keys() if k not in target_state])
            incompatible = model.load_state_dict(model_state_partial, strict=False)
            print(f"Loaded {len(model_state_partial)} matching tensors; "
                  f"missing in target: {missing_before}, unexpected in checkpoint: {unexpected_before}")
            strict_loaded = False
            load_weights_only = True  # treat as weights-only if we had to fall back
        else:
            raise

    start_epoch = 0
    
    if not load_weights_only and strict_loaded and isinstance(checkpoint, dict) and 'optimizer' in checkpoint and 'scheduler' in checkpoint:
        # Only load optimizer, scheduler, epoch if we are NOT in "weights_only" mode
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = int(checkpoint.get('epoch', -1)) + 1
        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print("Optimizer/scheduler state restored; starting from epoch 1")
    else:
        # In weights_only mode, reinitialize scheduler
        from vesuvius.models.training.lr_schedulers import get_scheduler
        
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
