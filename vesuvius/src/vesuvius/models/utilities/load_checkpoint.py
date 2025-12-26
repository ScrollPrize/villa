import re
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
        print("Found model configuration in checkpoint")

        if hasattr(mgr, 'targets') and 'targets' in checkpoint['model_config']:
            mgr.targets = checkpoint['model_config']['targets']
            print("Loaded targets from checkpoint (may be overridden by config)")

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

    def _rebuild_model_from_checkpoint_config():
        nonlocal model, optimizer
        from vesuvius.models.build.build_network_from_config import NetworkFromConfig
        from vesuvius.models.training.optimizers import create_optimizer

        # Create a config wrapper that combines checkpoint config with mgr
        class ConfigWrapper:
            def __init__(self, config_dict, base_mgr):
                self.__dict__.update(config_dict)
                # Copy any missing attributes from base_mgr
                for attr_name in dir(base_mgr):
                    if not attr_name.startswith('__') and not hasattr(self, attr_name):
                        setattr(self, attr_name, getattr(base_mgr, attr_name))

        config_wrapper = ConfigWrapper(checkpoint['model_config'], mgr)
        # Ensure deep supervision setting does not alter decoder sharing on rebuild.
        # If the checkpoint used a shared decoder (separate_decoders == False), forcing
        # deep supervision here would switch to separate decoders and break strict loads.
        try:
            sep_dec = bool(checkpoint['model_config'].get('separate_decoders', False))
            if not sep_dec:
                # Avoid DS-enforced decoder split during rebuild for exact arch match
                setattr(config_wrapper, 'enable_deep_supervision', False)
        except Exception:
            pass
        model = NetworkFromConfig(config_wrapper)
        model = model.to(device)

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

    # --- Infer separate_decoders from checkpoint keys if not accurately saved ---
    # Old checkpoints may have saved separate_decoders: False even when separate decoders were used
    # Detect the actual architecture from state dict keys
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        raw_state = checkpoint.get('model') or checkpoint.get('state_dict') or {}
        has_task_decoders = any(k.startswith('task_decoders.') and '.decoder.' in k for k in raw_state.keys())
        has_shared_decoder = any(k.startswith('shared_decoder.') for k in raw_state.keys())

        saved_sep_dec = checkpoint['model_config'].get('separate_decoders', False)
        inferred_sep_dec = has_task_decoders and not has_shared_decoder

        if inferred_sep_dec and not saved_sep_dec:
            print(f"Detected separate decoders from checkpoint keys (overriding saved separate_decoders=False)")
            checkpoint['model_config']['separate_decoders'] = True

    # Prefer rebuilding the model from checkpoint config to ensure an exact architectural match
    # unless the user explicitly requested weights-only load and wants to keep current model.
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        checkpoint_autoconfigure = checkpoint['model_config'].get('autoconfigure', True)
        force_rebuild = bool(getattr(mgr, 'rebuild_from_checkpoint_config', False))
        # Rebuild if resuming full training (default) or if explicitly forced
        if not load_weights_only or force_rebuild:
            reason = "(forced by flag)" if force_rebuild else "(align to checkpoint config)"
            print(f"Rebuilding model from checkpoint config {reason}")
            _rebuild_model_from_checkpoint_config()

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

    # Strip potential wrapper prefixes (DDP 'module.' and torch.compile '._orig_mod.')
    def _strip_wrapper_prefixes(sd):
        prefixes = ('module.', '_orig_mod.')
        def strip_key(k: str) -> str:
            changed = True
            while changed:
                changed = False
                for p in prefixes:
                    if k.startswith(p):
                        k = k[len(p):]
                        changed = True
            return k
        return { strip_key(k): v for k, v in sd.items() }

    model_state = _strip_wrapper_prefixes(model_state)

    # Align checkpoint keys with target model keys regarding DDP ('module.') prefix.
    # If the model is already wrapped in DDP (model.state_dict has 'module.'), but the
    # checkpoint was saved from an unwrapped model (no 'module.'), prepend the prefix.
    try:
        target_keys = list(model.state_dict().keys())
        ckpt_keys = list(model_state.keys())
        target_has_module = all(k.startswith('module.') for k in target_keys) if target_keys else False
        ckpt_has_module = any(k.startswith('module.') for k in ckpt_keys) if ckpt_keys else False
        if target_has_module and not ckpt_has_module:
            model_state = {f"module.{k}": v for k, v in model_state.items()}
        # Conversely, if target does not have 'module.' but checkpoint does (rare after stripping), strip it.
        elif (not target_has_module) and ckpt_has_module:
            model_state = { (k[7:] if k.startswith('module.') else k): v for k, v in model_state.items() }
    except Exception:
        # If any issue occurs during prefix alignment, proceed with original model_state
        pass

    # Optionally filter to matching keys/shapes when loading weights only
    target_state = model.state_dict()
    allow_partial = bool(load_weights_only)

    # Handle backward compatibility for encoder reference in task decoders.
    # Old models stored encoder as a submodule in each task decoder, creating duplicate
    # state_dict keys like task_decoders.<task>.encoder.* alongside shared_encoder.*.
    # New models don't have these duplicate keys.
    task_decoder_encoder_pattern = re.compile(r'^task_decoders\.([^.]+)\.encoder\.(.+)$')

    # Check if model expects task_decoders.<task>.encoder.* keys but checkpoint doesn't have them
    # (loading new checkpoint into old model) - populate from shared_encoder.*
    model_expects_decoder_encoder = any(task_decoder_encoder_pattern.match(k) for k in target_state.keys())
    ckpt_has_decoder_encoder = any(task_decoder_encoder_pattern.match(k) for k in model_state.keys())

    if model_expects_decoder_encoder and not ckpt_has_decoder_encoder:
        # Old model expects task_decoders.<task>.encoder.* but new checkpoint doesn't have them
        # Copy from shared_encoder.* to task_decoders.<task>.encoder.*
        task_names = set()
        for k in target_state.keys():
            m = task_decoder_encoder_pattern.match(k)
            if m:
                task_names.add(m.group(1))

        added_keys = []
        for task_name in task_names:
            for k, v in list(model_state.items()):
                if k.startswith('shared_encoder.'):
                    new_key = f'task_decoders.{task_name}.encoder.{k[len("shared_encoder."):]}'
                    if new_key in target_state and target_state[new_key].shape == v.shape:
                        model_state[new_key] = v
                        added_keys.append(new_key)
        if added_keys:
            print(f"Populated {len(added_keys)} task_decoders.<task>.encoder.* keys from shared_encoder.* for backward compatibility")

    elif ckpt_has_decoder_encoder and not model_expects_decoder_encoder:
        # Old checkpoint has task_decoders.<task>.encoder.* but new model doesn't need them
        # Filter out these duplicate keys to avoid unexpected key warnings
        removed_keys = [k for k in model_state.keys() if task_decoder_encoder_pattern.match(k)]
        if removed_keys:
            model_state = {k: v for k, v in model_state.items() if not task_decoder_encoder_pattern.match(k)}
            print(f"Filtered out {len(removed_keys)} duplicate task_decoders.<task>.encoder.* keys from old checkpoint")

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
            # If we had mismatches and we are not in weights-only mode, rebuild from ckpt config and retry strict load
            if (missing or unexpected) and not load_weights_only and isinstance(checkpoint, dict) and 'model_config' in checkpoint:
                print("Model-Checkpoint mismatch detected. Rebuilding model from checkpoint config for exact match.")
                _rebuild_model_from_checkpoint_config()
                target_state = model.state_dict()  # refresh
                incompatible = model.load_state_dict(model_state, strict=True)
                missing = list(getattr(incompatible, 'missing_keys', []))
                unexpected = list(getattr(incompatible, 'unexpected_keys', []))
                if missing or unexpected:
                    print("Rebuilt from checkpoint config, but still found key mismatches. Will fall back if needed.")
        except Exception:
            pass
    except RuntimeError as e:
        if strict and not allow_partial:
            # Try a rebuild for exact match first
            if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
                print("Strict state_dict load failed. Attempting to rebuild model from checkpoint config.")
                _rebuild_model_from_checkpoint_config()
                try:
                    incompatible = model.load_state_dict(model_state, strict=True)
                    strict_loaded = True
                except RuntimeError:
                    strict_loaded = False

            if not strict_loaded:
                print("Strict state_dict load failed. Falling back to partial non-strict load of matching weights.")
                # Debug: print a few key samples to understand mismatch
                try:
                    ch_keys = list(model_state.keys())
                    tg_keys = list(model.state_dict().keys())
                    print(f"Checkpoint has {len(ch_keys)} tensors; Model expects {len(tg_keys)} tensors")
                    print(f"  ckpt key sample: {ch_keys[:3]} ... {ch_keys[-3:]}")
                    print(f"  model key sample: {tg_keys[:3]} ... {tg_keys[-3:]}")
                except Exception:
                    pass
                # Filter to matching keys/shapes and load non-strict
                target_state = model.state_dict()  # refresh in case of rebuild
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

        # Re-create a fresh scheduler instance. get_scheduler returns a single
        # scheduler object (not a tuple). The per-iteration flag is handled
        # by BaseTrainer._get_scheduler when needed.
        scheduler_type = getattr(mgr, 'scheduler', 'poly')
        scheduler_kwargs = getattr(mgr, 'scheduler_kwargs', {})

        scheduler = get_scheduler(
            scheduler_type=scheduler_type,
            optimizer=optimizer,
            initial_lr=mgr.initial_lr,
            max_steps=mgr.max_epoch,
            **scheduler_kwargs
        )
        print("Loaded model weights only; starting new training run from epoch 1.")
    
    return model, optimizer, scheduler, start_epoch, True
