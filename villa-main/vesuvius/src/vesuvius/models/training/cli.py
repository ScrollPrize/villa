import argparse
import multiprocessing
import os
import socket
import subprocess
import sys
from pathlib import Path

import torch

from vesuvius.models.configuration.config_manager import ConfigManager
from vesuvius.models.datasets.intensity_properties import load_intensity_props_formatted
from vesuvius.models.utilities.cli_utils import update_config_from_args


def _maybe_set_spawn_start_method(argv):
    # s3fs/fsspec can misbehave with fork; force spawn if s3/config is present.
    if not argv:
        return
    if any('s3://' in str(arg) for arg in argv) or '--config-path' in argv or '--config' in argv:
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass


def main(argv=None):
    """Main entry point for the training script."""
    if argv is None:
        argv = sys.argv[1:]

    _maybe_set_spawn_start_method(argv)

    parser = argparse.ArgumentParser(
        description="Train Vesuvius neural networks for ink detection and segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    grp_required = parser.add_argument_group("Required")
    grp_paths = parser.add_argument_group("Paths & Format")
    grp_data = parser.add_argument_group("Data & Splits")
    grp_model = parser.add_argument_group("Model")
    grp_train = parser.add_argument_group("Training Control")
    grp_optim = parser.add_argument_group("Optimization")
    grp_sched = parser.add_argument_group("Scheduler")
    grp_trainer = parser.add_argument_group("Trainer Selection")
    grp_logging = parser.add_argument_group("Logging & Tracking")

    # Required
    grp_required.add_argument("-i", "--input",
                              help="Input directory containing images/ and labels/ subdirectories.")
    grp_required.add_argument("--config", "--config-path", dest="config_path", type=str, required=True,
                              help="Path to configuration YAML file")

    # Paths & Format
    grp_paths.add_argument("-o", "--output", default="checkpoints",
                           help="Output directory for saving checkpoints and configs")
    grp_paths.add_argument("--format", choices=["image", "zarr", "napari"],
                           help="Data format (auto-detected if omitted)")
    grp_paths.add_argument("--val-dir", type=str,
                           help="Optional validation directory with images/ and labels/")
    grp_paths.add_argument("--checkpoint", "--checkpoint-path", dest="checkpoint_path", type=str,
                           help="Path to checkpoint (.pt/.pth) or weights-only state_dict file")
    grp_paths.add_argument("--load-weights-only", action="store_true",
                           help="Load only model weights from checkpoint; ignore optimizer/scheduler and allow partial load")
    grp_paths.add_argument("--rebuild-from-ckpt-config", action="store_true",
                           help="Rebuild model from checkpoint's model_config before loading weights")
    grp_paths.add_argument("--intensity-properties-json", type=str, default=None,
                           help="nnU-Net style intensity properties JSON for CT normalization")
    grp_paths.add_argument("--skip-image-checks", action="store_true",
                           help="Skip expensive image/zarr existence checks and conversions; assumes images.zarr/labels.zarr already exist")

    # Data & Splits
    grp_data.add_argument("--batch-size", type=int,
                          help="Training batch size")
    grp_data.add_argument("--patch-size", type=str,
                          help="Patch size CSV, e.g. '192,192,192' (3D) or '256,256' (2D)")
    grp_data.add_argument("--loss", type=str,
                          help="Loss functions, e.g. '[SoftDiceLoss, BCEWithLogitsLoss]' or CSV")
    grp_data.add_argument("--train-split", type=float,
                          help="Training/validation split ratio in [0,1]")
    grp_data.add_argument("--seed", type=int, default=42,
                          help="Random seed for split/initialization")
    grp_data.add_argument("--skip-intensity-sampling", dest="skip_intensity_sampling",
                          action="store_true", default=True,
                          help="Skip intensity sampling during dataset init")
    grp_data.add_argument("--no-skip-intensity-sampling", dest="skip_intensity_sampling",
                          action="store_false",
                          help="Enable intensity sampling during dataset init")
    grp_data.add_argument("--no-spatial", action="store_true",
                          help="Disable spatial/geometric augmentations")
    grp_data.add_argument("--rotation-axes", type=str,
                          help="Comma-separated axes (subset of x,y,z / width,height,depth) that may be rotated; e.g. 'z' keeps the depth axis upright")

    # Model
    grp_model.add_argument("--model-name", type=str,
                           help="Model name for checkpoints and logging")
    grp_model.add_argument("--nonlin", type=str, choices=["LeakyReLU", "ReLU", "SwiGLU", "swiglu", "GLU", "glu"],
                           help="Activation function")
    grp_model.add_argument("--se", action="store_true", help="Enable squeeze and excitation modules in the encoder")
    grp_model.add_argument("--se-reduction-ratio", type=float, default=0.0625,
                           help="Squeeze excitation reduction ratio")
    grp_model.add_argument("--pool-type", type=str, choices=["avg", "max", "conv"],
                           help="Type of pooling in encoder ('conv' = strided conv)")

    # Training Control
    grp_train.add_argument("--max-epoch", type=int, default=1000,
                           help="Maximum number of epochs")
    grp_train.add_argument("--max-steps-per-epoch", type=int, default=250,
                           help="Max training steps per epoch (use all data if unset)")
    grp_train.add_argument("--max-val-steps-per-epoch", type=int, default=50,
                           help="Max validation steps per epoch (use all data if unset)")
    grp_train.add_argument("--full-epoch", action="store_true",
                           help="Iterate over entire train/val set per epoch (overrides max-steps)")
    grp_train.add_argument("--early-stopping-patience", type=int, default=0,
                           help="Epochs to wait for val loss improvement (0 disables)")
    grp_train.add_argument("--ddp", action="store_true",
                           help="Enable DistributedDataParallel (use with torchrun)")
    grp_train.add_argument("--val-every-n", dest="val_every_n", type=int, default=1,
                           help="Perform validation every N epochs (1=every epoch)")
    grp_train.add_argument("--gpus", type=str, default=None,
                           help="Comma-separated GPU device IDs to use, e.g. '0,1,3'. With DDP, length must equal WORLD_SIZE")
    grp_train.add_argument("--nproc-per-node", type=int, default=None,
                           help="Number of processes to spawn locally for DDP (use instead of torchrun)")
    grp_train.add_argument("--master-addr", type=str, default="127.0.0.1",
                           help="Master address for DDP when spawning without torchrun")
    grp_train.add_argument("--master-port", type=int, default=None,
                           help="Master port for DDP when spawning without torchrun (default: auto)")

    # Optimization
    grp_optim.add_argument("--optimizer", type=str,
                           help="Optimizer (see models/optimizers.py)")
    grp_optim.add_argument("--grad-accum", "--gradient-accumulation", dest="gradient_accumulation", type=int, default=None,
                           help="Number of steps to accumulate gradients before optimizer.step()")
    grp_optim.add_argument("--grad-clip", type=float, default=12.0,
                           help="Gradient clipping value")
    grp_optim.add_argument("--amp-dtype", type=str, choices=["float16", "bfloat16"], default="float16",
                           help="Autocast dtype when AMP is enabled (float16 uses GradScaler; bfloat16 skips scaling)")
    grp_optim.add_argument("--no-amp", action="store_true",
                           help="Disable Automatic Mixed Precision (AMP)")

    # Scheduler
    grp_sched.add_argument("--scheduler", type=str,
                           help="Learning rate scheduler (default: from config or 'poly')")
    grp_sched.add_argument("--warmup-steps", type=int,
                           help="Number of warmup steps for cosine_warmup scheduler")

    # Trainer Selection
    grp_trainer.add_argument("--trainer", "--tr", type=str, default="base",
                             help="Trainer: base, surface_frame, mean_teacher, uncertainty_aware_mean_teacher, primus_mae, unet_mae, finetune_mae_unet")
    grp_trainer.add_argument("--ssl-warmup", type=int, default=None,
                             help="Semi-supervised: epochs to ignore EMA consistency loss (0 disables)")
    # Semi-supervised sampling controls (used by mean_teacher/uncertainty_aware_mean_teacher)
    grp_trainer.add_argument("--labeled-ratio", type=float, default=None,
                             help="Fraction of labeled patches to use (0-1). If set, overrides trainer default")
    grp_trainer.add_argument("--num-labeled", type=int, default=None,
                             help="Absolute number of labeled patches to use (overrides --labeled-ratio if provided)")
    grp_trainer.add_argument("--labeled-batch-size", type=int, default=None,
                             help="Number of labeled patches per batch (rest are unlabeled) for two-stream sampler")

    # Only valid for finetune_mae_unet: path to the pretrained MAE checkpoint to initialize from
    grp_trainer.add_argument("--pretrained_checkpoint", type=str, default=None,
                             help="Pretrained MAE checkpoint path (required when --trainer finetune_mae_unet). Invalid for other trainers.")

    # Logging & Tracking
    grp_logging.add_argument("--wandb-project", type=str, default=None,
                             help="Weights & Biases project (omit to disable wandb)")
    grp_logging.add_argument("--wandb-entity", type=str, default=None,
                             help="Weights & Biases team/username")
    grp_logging.add_argument("--wandb-run-name", type=str, default=None,
                             help="Optional custom name for the Weights & Biases run")
    grp_logging.add_argument("--wandb-resume", nargs='?', const='allow', default=None,
                             help="Weights & Biases resume mode or run id. Provide a resume policy ('allow', 'auto', 'must', 'never') or a run id (defaults to 'allow' if flag used without value).")
    grp_logging.add_argument("--profile-augmentations", action="store_true",
                             help="Collect per-augmentation timing and report per-epoch totals")
    grp_logging.add_argument("--verbose", action="store_true",
                             help="Enable verbose debug output")

    args = parser.parse_args(argv)

    mgr = ConfigManager(verbose=args.verbose)

    if not Path(args.config_path).exists():
        print(f"\nError: Config file does not exist: {args.config_path}")
        print("\nPlease provide a valid configuration file.")
        print("\nExample usage:")
        print("  vesuvius.train --config path/to/config.yaml --input path/to/data --output path/to/output")
        print("\nFor more options, use: vesuvius.train --help")
        sys.exit(1)

    mgr.load_config(args.config_path)
    print(f"Loaded configuration from: {args.config_path}")

    # Resolve input path: CLI arg takes precedence, else use config's data_path
    if args.input is not None:
        input_path = Path(args.input)
    elif mgr.data_path is not None:
        input_path = mgr.data_path
        args.input = str(input_path)  # Update args so downstream code sees it
    else:
        print("\nError: No input directory specified.")
        print("Provide --input on the command line OR set data_path in your YAML config.")
        sys.exit(1)

    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_path}")

    if args.val_dir is not None and not Path(args.val_dir).exists():
        raise ValueError(f"Validation directory does not exist: {args.val_dir}")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    update_config_from_args(mgr, args)

    # Validation frequency
    if hasattr(args, 'val_every_n') and args.val_every_n is not None:
        if int(args.val_every_n) < 1:
            raise ValueError(f"--val-every-n must be >= 1, got {args.val_every_n}")
        setattr(mgr, 'val_every_n', int(args.val_every_n))
        mgr.tr_configs["val_every_n"] = int(args.val_every_n)
        if args.verbose:
            print(f"Validate every {args.val_every_n} epoch(s)")

    # Enable DDP if requested or if torchrun sets WORLD_SIZE>1
    if getattr(args, 'ddp', False) or int(os.environ.get('WORLD_SIZE', '1')) > 1:
        setattr(mgr, 'use_ddp', True)
        # In DDP, --batch-size is per-GPU; no extra adjustment needed.

    # Parse GPUs selection if provided
    if getattr(args, 'gpus', None):
        try:
            gpu_ids = [int(x) for x in str(args.gpus).split(',') if x.strip() != '']
        except ValueError as exc:
            raise ValueError("--gpus must be a comma-separated list of integers, e.g. '0,1,3'") from exc
        setattr(mgr, 'gpu_ids', gpu_ids)

    if args.val_dir is not None:
        mgr.val_data_path = Path(args.val_dir)

    # If user supplies intensity properties JSON, load and inject into config for CT normalization
    if args.intensity_properties_json is not None:
        ip_path = Path(args.intensity_properties_json)
        if not ip_path.exists():
            raise ValueError(f"Intensity properties JSON not found: {ip_path}")
        props = load_intensity_props_formatted(ip_path, channel=0)
        if not props:
            raise ValueError(f"Failed to parse intensity properties JSON: {ip_path}")
        if hasattr(mgr, 'update_config'):
            mgr.update_config(normalization_scheme='ct', intensity_properties=props)
        else:
            mgr.dataset_config = getattr(mgr, 'dataset_config', {})
            mgr.dataset_config['normalization_scheme'] = 'ct'
            mgr.dataset_config['intensity_properties'] = props
        setattr(mgr, 'skip_intensity_sampling', True)
        print("Using provided intensity properties for CT normalization. Sampling disabled.")

    # If DDP is requested but not launched with torchrun, optionally self-spawn processes
    if getattr(mgr, 'use_ddp', False) and int(os.environ.get('WORLD_SIZE', '1')) == 1:
        # Determine process count
        nproc = args.nproc_per_node
        if nproc is None:
            # Default to number of requested GPUs, else CUDA device count, else 1
            gpu_ids = getattr(mgr, 'gpu_ids', None)
            if gpu_ids:
                nproc = len(gpu_ids)
            elif torch.cuda.is_available():
                try:
                    nproc = torch.cuda.device_count()
                except Exception:
                    nproc = 1
            else:
                nproc = 1

        if nproc > 1:
            # Validate GPU mapping length if provided
            gpu_ids = getattr(mgr, 'gpu_ids', None)
            if gpu_ids and len(gpu_ids) != nproc:
                raise ValueError(f"--gpus specifies {len(gpu_ids)} GPUs but --nproc-per-node is {nproc}. They must match.")

            # Find a free port if not provided
            master_port = args.master_port
            if master_port is None:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((args.master_addr, 0))
                    master_port = s.getsockname()[1]

            print(f"Spawning {nproc} DDP processes (master {args.master_addr}:{master_port}) without torchrun...")

            # Rebuild argv without the spawn-only flags; children don't need them
            skip_next = False
            child_argv = []
            for a in argv:
                if skip_next:
                    skip_next = False
                    continue
                if a in ("--nproc-per-node", "--master-addr", "--master-port"):
                    skip_next = True
                    continue
                child_argv.append(a)

            procs = []
            for rank in range(nproc):
                env = os.environ.copy()
                env.update({
                    'RANK': str(rank),
                    'LOCAL_RANK': str(rank),
                    'WORLD_SIZE': str(nproc),
                    'MASTER_ADDR': args.master_addr,
                    'MASTER_PORT': str(master_port),
                })
                cmd = [sys.executable, sys.argv[0], *child_argv]
                # Use unbuffered -u for timely logs on Windows/Unix
                if '-u' not in cmd:
                    cmd.insert(1, '-u')
                procs.append(subprocess.Popen(cmd, env=env))

            exit_code = 0
            for p in procs:
                ret = p.wait()
                if ret != 0:
                    exit_code = ret
            sys.exit(exit_code)
        else:
            print("DDP requested but only one process determined; proceeding single-process.")

    trainer_name = args.trainer.lower()
    mgr.trainer_class = trainer_name

    # Enforce usage of --pretrained_checkpoint only for the MAE finetune trainer, and require it there
    if getattr(args, 'pretrained_checkpoint', None):
        if trainer_name != "finetune_mae_unet":
            raise ValueError("--pretrained_checkpoint is only valid when using --trainer finetune_mae_unet")
        # Stash onto mgr so the finetune trainer can load it
        setattr(mgr, 'pretrained_mae_checkpoint', args.pretrained_checkpoint)
        mgr.tr_info["pretrained_mae_checkpoint"] = args.pretrained_checkpoint
    elif trainer_name == "finetune_mae_unet":
        # For finetune trainer the pretrained checkpoint is mandatory
        raise ValueError("Missing --pretrained_checkpoint: required for --trainer finetune_mae_unet")

    if trainer_name == "uncertainty_aware_mean_teacher":
        mgr.allow_unlabeled_data = True
        from vesuvius.models.training.trainers.semi_supervised.train_uncertainty_aware_mean_teacher import TrainUncertaintyAwareMeanTeacher
        trainer = TrainUncertaintyAwareMeanTeacher(mgr=mgr, verbose=args.verbose)
        print("Using Uncertainty-Aware Mean Teacher Trainer for semi-supervised 3D training")
    elif trainer_name == "mean_teacher":
        mgr.allow_unlabeled_data = True
        from vesuvius.models.training.trainers.semi_supervised.train_mean_teacher import TrainMeanTeacher
        trainer = TrainMeanTeacher(mgr=mgr, verbose=args.verbose)
        print("Using Regular Mean Teacher Trainer for semi-supervised training")
    elif trainer_name == "primus_mae":
        mgr.allow_unlabeled_data = True
        from vesuvius.models.training.trainers.self_supervised.train_eva_mae import TrainEVAMAE
        trainer = TrainEVAMAE(mgr=mgr, verbose=args.verbose)
        print("Using EVA (Primus) Architecture for MAE Pretraining")
    elif trainer_name == "unet_mae":
        mgr.allow_unlabeled_data = True
        from vesuvius.models.training.trainers.self_supervised.train_unet_mae import TrainUNetMAE
        trainer = TrainUNetMAE(mgr=mgr, verbose=args.verbose)
        print("Using UNet-style MAE Trainer (NetworkFromConfig)")
    elif trainer_name == "finetune_mae_unet":
        from vesuvius.models.training.trainers.self_supervised.train_finetune_mae_unet import TrainFineTuneMAEUNet
        trainer = TrainFineTuneMAEUNet(mgr=mgr, verbose=args.verbose)
        print("Using Fine-Tune MAE->UNet Trainer (NetworkFromConfig)")
    elif trainer_name == "lejepa":
        mgr.allow_unlabeled_data = True
        from vesuvius.models.training.trainers.self_supervised.train_lejepa import TrainLeJEPA
        trainer = TrainLeJEPA(mgr=mgr, verbose=args.verbose)
        print("Using LeJEPA Trainer (Primus + SIGReg) for unsupervised pretraining")
    elif trainer_name == "mutex_affinity":
        from vesuvius.models.training.trainers.mutex_affinity_trainer import MutexAffinityTrainer
        trainer = MutexAffinityTrainer(mgr=mgr, verbose=args.verbose)
        print("Using Mutex Affinity Trainer")
    elif trainer_name == "base":
        from vesuvius.models.training.train import BaseTrainer
        trainer = BaseTrainer(mgr=mgr, verbose=args.verbose)
        print("Using Base Trainer for supervised training")
    elif trainer_name == "surface_frame":
        from vesuvius.models.training.trainers.surface_frame_trainer import SurfaceFrameTrainer
        trainer = SurfaceFrameTrainer(mgr=mgr, verbose=args.verbose)
        print("Using Surface Frame Trainer")
    else:
        raise ValueError(
            "Unknown trainer: {trainer}. Available options: base, surface_frame, mutex_affinity, mean_teacher, "
            "uncertainty_aware_mean_teacher, primus_mae, unet_mae, finetune_mae_unet, lejepa".format(trainer=trainer_name)
        )

    print("Starting training...")
    trainer.train()
    print("Training completed!")


if __name__ == '__main__':
    main()
