from pathlib import Path
from vesuvius.models.utilities.data_format_utils import detect_data_format


def update_config_from_args(mgr, args):
    if args.input is not None:
        mgr.data_path = Path(args.input)
        if not hasattr(mgr, 'dataset_config'):
            mgr.dataset_config = {}
        mgr.dataset_config["data_path"] = str(mgr.data_path)

        if args.format:
            mgr.data_format = args.format;
            print(f"Using specified data format: {mgr.data_format}")
        else:
            detected = detect_data_format(mgr.data_path)
            if detected:
                mgr.data_format = detected;
                print(f"Auto-detected data format: {mgr.data_format}")
            else:
                raise ValueError("Data format could not be determined. Please specify --format.")

        mgr.dataset_config["data_format"] = mgr.data_format
    else:
        print("No input directory specified - using data_paths from config")

    mgr.ckpt_out_base = Path(args.output)
    mgr.tr_info["ckpt_out_base"] = str(mgr.ckpt_out_base)

    # Skip image/zarr preflight checks if requested
    if hasattr(args, 'skip_image_checks') and args.skip_image_checks:
        mgr.skip_image_checks = True
        if hasattr(mgr, 'dataset_config'):
            mgr.dataset_config['skip_image_checks'] = True
        if mgr.verbose:
            print("Skipping image/zarr preflight checks as requested (--skip-image-checks)")

    if args.batch_size is not None:
        mgr.train_batch_size = args.batch_size
        mgr.tr_configs["batch_size"] = args.batch_size

    if args.patch_size is not None:
        # Parse patch size from string like "192,192,192" or "256,256"
        try:
            patch_size = [int(x.strip()) for x in args.patch_size.split(',')]
            mgr.update_config(patch_size=patch_size)
        except ValueError as e:
            raise ValueError(
                f"Invalid patch size format: {args.patch_size}. Expected comma-separated integers like '192,192,192'")

    if args.train_split is not None:
        if not 0.0 <= args.train_split <= 1.0:
            raise ValueError(f"Train split must be between 0.0 and 1.0, got {args.train_split}")
        mgr.tr_val_split = args.train_split
        mgr.tr_info["tr_val_split"] = args.train_split

    if args.seed is not None:
        mgr.seed = args.seed
        mgr.tr_info["seed"] = args.seed
        if mgr.verbose:
            print(f"Set random seed for train/val split: {mgr.seed}")

    if args.max_epoch is not None:
        mgr.max_epoch = args.max_epoch
        mgr.tr_configs["max_epoch"] = args.max_epoch

    if args.max_steps_per_epoch is not None:
        mgr.max_steps_per_epoch = args.max_steps_per_epoch
        mgr.tr_configs["max_steps_per_epoch"] = args.max_steps_per_epoch

    if args.max_val_steps_per_epoch is not None:
        mgr.max_val_steps_per_epoch = args.max_val_steps_per_epoch
        mgr.tr_configs["max_val_steps_per_epoch"] = args.max_val_steps_per_epoch
    
    # Handle full_epoch flag - overrides max_steps_per_epoch and max_val_steps_per_epoch
    if args.full_epoch:
        mgr.max_steps_per_epoch = None  # None means use all data
        mgr.max_val_steps_per_epoch = None  # None means use all data
        mgr.tr_configs["max_steps_per_epoch"] = None
        mgr.tr_configs["max_val_steps_per_epoch"] = None
        if mgr.verbose:
            print(f"Full epoch mode enabled - will iterate over entire train and validation datasets")

    if args.model_name is not None:
        mgr.model_name = args.model_name
        mgr.tr_info["model_name"] = args.model_name
        if mgr.verbose:
            print(f"Set model name: {mgr.model_name}")

    if args.nonlin is not None:
        if not hasattr(mgr, 'model_config') or mgr.model_config is None:
            mgr.model_config = {}
        mgr.model_config["nonlin"] = args.nonlin
        if mgr.verbose:
            print(f"Set activation function: {args.nonlin}")

    if args.se:
        if not hasattr(mgr, 'model_config') or mgr.model_config is None:
            mgr.model_config = {}
        mgr.model_config["squeeze_excitation"] = True
        mgr.model_config["squeeze_excitation_reduction_ratio"] = args.se_reduction_ratio
        if mgr.verbose:
            print(f"Enabled squeeze and excitation with reduction ratio: {args.se_reduction_ratio}")

    if args.pool_type is not None:
        if not hasattr(mgr, 'model_config') or mgr.model_config is None:
            mgr.model_config = {}
        mgr.model_config["pool_type"] = args.pool_type
        
        if mgr.verbose:
            if args.pool_type == 'conv':
                print(f"Set pooling type: conv (using strided convolutions, no pooling)")
            else:
                print(f"Set pooling type: {args.pool_type}")

    if args.optimizer is not None:
        mgr.optimizer = args.optimizer
        mgr.tr_configs["optimizer"] = args.optimizer
        if mgr.verbose:
            print(f"Set optimizer: {mgr.optimizer}")

    if args.loss is not None:
        import ast
        from vesuvius.models.configuration.config_utils import configure_targets
        try:
            loss_list = ast.literal_eval(args.loss)
            loss_list = loss_list if isinstance(loss_list, list) else [loss_list]
        except Exception:
            loss_list = [s.strip() for s in args.loss.split(',')]
        configure_targets(mgr, loss_list)

    if args.no_spatial:
        mgr.no_spatial = True
        if hasattr(mgr, 'dataset_config'):
            mgr.dataset_config['no_spatial'] = True
        if mgr.verbose:
            print(f"Disabled spatial transformations (--no-spatial flag set)")

    if hasattr(args, 'rotation_axes') and args.rotation_axes is not None:
        axis_name_to_index = {
            'z': 0,
            'depth': 0,
            'd': 0,
            'y': 1,
            'height': 1,
            'h': 1,
            'x': 2,
            'width': 2,
            'w': 2,
        }
        index_to_axis_name = {0: 'z', 1: 'y', 2: 'x'}

        tokens = [tok.strip().lower() for tok in args.rotation_axes.split(',') if tok.strip()]
        normalized_indices = []

        if not tokens:
            normalized_indices = []
        else:
            seen = set()
            special_handled = False
            for token in tokens:
                if token in ('all', 'xyz', 'zyx'):
                    normalized_indices = [0, 1, 2]
                    special_handled = True
                    break
                if token == 'none':
                    normalized_indices = []
                    special_handled = True
                    break
                if token in axis_name_to_index:
                    idx = axis_name_to_index[token]
                else:
                    try:
                        idx = int(token)
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid value '{token}' for --rotation-axes. "
                            "Use a comma-separated subset of x,y,z (width,height,depth), digits 0-2, or 'none'."
                        ) from exc
                    if idx not in index_to_axis_name:
                        raise ValueError(
                            f"Invalid axis index '{idx}' for --rotation-axes. Valid indices are 0, 1, 2."
                        )
                if idx not in seen:
                    normalized_indices.append(idx)
                    seen.add(idx)
            if not special_handled and not normalized_indices:
                raise ValueError(
                    "No valid axes provided to --rotation-axes. Provide a comma-separated subset of x,y,z or use 'none'."
                )

        mgr.allowed_rotation_axes = tuple(normalized_indices)
        axis_names = [index_to_axis_name[idx] for idx in normalized_indices]
        if hasattr(mgr, 'dataset_config'):
            mgr.dataset_config['rotation_axes'] = axis_names
        if mgr.verbose:
            if axis_names:
                print(f"Restricting random rotations to axes: {', '.join(axis_names)}")
            else:
                print("Disabling random rotations (rotation_axes=none)")

    # Handle skip_intensity_sampling (default is True now)
    if hasattr(args, 'skip_intensity_sampling'):
        mgr.skip_intensity_sampling = args.skip_intensity_sampling
        if hasattr(mgr, 'dataset_config'):
            mgr.dataset_config['skip_intensity_sampling'] = args.skip_intensity_sampling
        if mgr.verbose:
            if args.skip_intensity_sampling:
                print(f"Skipping intensity sampling (default behavior)")
            else:
                print(f"Intensity sampling enabled (--no-skip-intensity-sampling flag set)")

    if args.grad_clip is not None:
        mgr.gradient_clip = args.grad_clip
        mgr.tr_configs["gradient_clip"] = args.grad_clip
        if mgr.verbose:
            print(f"Set gradient clipping: {mgr.gradient_clip}")

    # Gradient accumulation steps
    if hasattr(args, 'gradient_accumulation') and args.gradient_accumulation is not None:
        if args.gradient_accumulation < 1:
            raise ValueError(f"--grad-accum/--gradient-accumulation must be >= 1, got {args.gradient_accumulation}")
        mgr.gradient_accumulation = int(args.gradient_accumulation)
        mgr.tr_configs["gradient_accumulation"] = int(args.gradient_accumulation)
        if mgr.verbose:
            print(f"Set gradient accumulation steps: {mgr.gradient_accumulation}")

    if args.scheduler is not None:
        mgr.scheduler = args.scheduler
        mgr.tr_configs["scheduler"] = args.scheduler
        if mgr.verbose:
            print(f"Set learning rate scheduler: {mgr.scheduler}")

        if args.scheduler == "cosine_warmup":
            if not hasattr(mgr, 'scheduler_kwargs'):
                mgr.scheduler_kwargs = {}

            if args.warmup_steps is not None:
                mgr.scheduler_kwargs["warmup_steps"] = args.warmup_steps
                # Save scheduler_kwargs to tr_configs
                mgr.tr_configs["scheduler_kwargs"] = mgr.scheduler_kwargs
                if mgr.verbose:
                    print(f"Set warmup steps: {args.warmup_steps}")

    if args.no_amp:
        mgr.no_amp = True
        mgr.tr_configs["no_amp"] = True
        if mgr.verbose:
            print(f"Disabled Automatic Mixed Precision (AMP)")

    if hasattr(args, 'amp_dtype') and args.amp_dtype is not None:
        mgr.amp_dtype = args.amp_dtype.lower()
        mgr.tr_configs["amp_dtype"] = mgr.amp_dtype
        if mgr.verbose:
            print(f"Set AMP dtype: {mgr.amp_dtype}")

    if hasattr(args, 'early_stopping_patience') and args.early_stopping_patience is not None:
        mgr.early_stopping_patience = args.early_stopping_patience
        mgr.tr_configs["early_stopping_patience"] = args.early_stopping_patience
        if mgr.verbose:
            if args.early_stopping_patience == 0:
                print(f"Early stopping disabled")
            else:
                print(f"Set early stopping patience: {args.early_stopping_patience} epochs")

    # SSL warmup (mean teacher): ignore EMA consistency loss for N epochs
    if hasattr(args, 'ssl_warmup') and args.ssl_warmup is not None:
        mgr.warmup = int(args.ssl_warmup)
        mgr.tr_configs["ssl_warmup"] = int(args.ssl_warmup)
        if mgr.verbose:
            if args.ssl_warmup > 0:
                print(f"SSL warmup enabled: ignoring EMA consistency loss for first {args.ssl_warmup} epoch(s)")
            else:
                print("SSL warmup disabled (0 epochs)")

    # Semi-supervised sampling controls
    if hasattr(args, 'labeled_ratio') and args.labeled_ratio is not None:
        if not (0.0 <= args.labeled_ratio <= 1.0):
            raise ValueError(f"--labeled-ratio must be in [0,1], got {args.labeled_ratio}")
        mgr.labeled_ratio = float(args.labeled_ratio)
        mgr.tr_configs["labeled_ratio"] = float(args.labeled_ratio)
        if mgr.verbose:
            print(f"Set labeled patch ratio: {mgr.labeled_ratio}")

    if hasattr(args, 'num_labeled') and args.num_labeled is not None:
        if args.num_labeled < 0:
            # Convention: -1 means use all labeled patches
            mgr.num_labeled = None
            mgr.tr_configs["num_labeled"] = None
            if mgr.verbose:
                print("Using all labeled patches (num_labeled=-1)")
        else:
            mgr.num_labeled = int(args.num_labeled)
            mgr.tr_configs["num_labeled"] = int(args.num_labeled)
            if mgr.verbose:
                print(f"Set absolute labeled patch count: {mgr.num_labeled}")

    if hasattr(args, 'labeled_batch_size') and args.labeled_batch_size is not None:
        if args.labeled_batch_size < 1:
            raise ValueError(f"--labeled-batch-size must be >=1, got {args.labeled_batch_size}")
        mgr.labeled_batch_size = int(args.labeled_batch_size)
        mgr.tr_configs["labeled_batch_size"] = int(args.labeled_batch_size)
        if mgr.verbose:
            print(f"Set labeled batch size: {mgr.labeled_batch_size}")



    # Checkpoint/weights loading controls
    if hasattr(args, 'checkpoint_path') and args.checkpoint_path is not None:
        mgr.checkpoint_path = Path(args.checkpoint_path)
        mgr.tr_info["checkpoint_path"] = str(mgr.checkpoint_path)
        if mgr.verbose:
            print(f"Set checkpoint path: {mgr.checkpoint_path}")

    if hasattr(args, 'load_weights_only') and args.load_weights_only:
        mgr.load_weights_only = True
        mgr.tr_info["load_weights_only"] = True
        if mgr.verbose:
            print("Will load model weights only (ignore optimizer/scheduler)")

    if hasattr(args, 'rebuild_from_ckpt_config') and args.rebuild_from_ckpt_config:
        mgr.rebuild_from_checkpoint_config = True
        mgr.tr_info["rebuild_from_checkpoint_config"] = True
        if mgr.verbose:
            print("Will rebuild model from checkpoint's model_config before loading weights")

    mgr.wandb_project = args.wandb_project
    mgr.wandb_entity = args.wandb_entity
