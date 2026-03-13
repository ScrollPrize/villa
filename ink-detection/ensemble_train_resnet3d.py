import argparse
import os
import os.path as osp

from train_resnet3d_lib.config import log
from train_resnet3d_lib.runtime import ensemble as ensemble_runtime


__all__ = ["parse_args", "run_ensemble", "main"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_run_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument(
        "--epochs",
        type=str,
        default=None,
        help="CSV epoch override, e.g. 1,2,4,5,7,10,14,19,24,29",
    )
    parser.add_argument("--target_count", type=int, default=10)
    parser.add_argument("--dense_prefix", type=int, default=4)

    parser.add_argument(
        "--avg_mode",
        type=str,
        choices=list(ensemble_runtime.SUPPORTED_AVG_MODES),
        default=ensemble_runtime.AVG_MODE_EQUAL,
        help="Checkpoint averaging mode.",
    )
    parser.add_argument(
        "--sigma_rel",
        type=float,
        default=0.10,
        help="Relative sigma used for fake_ema_power weighting (Section 3-inspired).",
    )

    parser.add_argument(
        "--metadata_json",
        type=str,
        default=None,
        help="Optional metadata source for old runs that do not have metadata.snapshot.json.",
    )
    parser.add_argument("--run_stitch", action="store_true")

    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="Lightning precision mode. Use --precision auto to derive from metadata use_amp.",
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=None,
        help="Override metadata valid batch size for stitch runs only.",
    )
    parser.add_argument(
        "--stitch_outputs_path",
        type=str,
        default=None,
        help="Optional outputs_path override passed to stitch workflow.",
    )
    parser.add_argument(
        "--stitch_run_name",
        type=str,
        default=None,
        help="Optional run_name override passed to stitch workflow.",
    )
    return parser.parse_args()


def _resolve_metadata_source(training_run_dir: str, metadata_json: str | None) -> str | None:
    if metadata_json is not None:
        candidate = osp.expanduser(str(metadata_json))
        if not osp.isabs(candidate):
            candidate = osp.abspath(candidate)
        if not osp.isfile(candidate):
            raise FileNotFoundError(f"--metadata_json not found: {candidate}")
        return candidate

    run_snapshot = osp.join(training_run_dir, "metadata.snapshot.json")
    if osp.isfile(run_snapshot):
        return run_snapshot
    return None


def _build_stitch_args(args, *, metadata_path: str, ensemble_checkpoint_path: str):
    return argparse.Namespace(
        metadata_json=metadata_path,
        init_ckpt_path=ensemble_checkpoint_path,
        resume_from_ckpt=None,
        checkpoint_group=None,
        checkpoint_run_prefix="stitch",
        run_name=args.stitch_run_name,
        outputs_path=args.stitch_outputs_path,
        devices=int(args.devices),
        accelerator=str(args.accelerator),
        precision=str(args.precision),
        check_val_every_n_epoch=1,
        valid_batch_size=args.valid_batch_size,
    )


def run_ensemble(args):
    training_run_dir = ensemble_runtime.resolve_training_run_dir(args.training_run_dir)
    output_dir = ensemble_runtime.resolve_output_dir(training_run_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    checkpoints_by_epoch = ensemble_runtime.discover_epoch_checkpoints(training_run_dir)
    available_epochs = sorted(checkpoints_by_epoch.keys())
    selected_epochs = ensemble_runtime.resolve_selected_epochs(
        available_epochs,
        epochs_csv=args.epochs,
        target_count=int(args.target_count),
        dense_prefix=int(args.dense_prefix),
    )
    selected_checkpoint_paths = ensemble_runtime.resolve_checkpoint_paths_for_epochs(
        checkpoints_by_epoch,
        selected_epochs,
    )
    selected_checkpoint_paths = ensemble_runtime.as_absolute_paths(selected_checkpoint_paths)

    weights = ensemble_runtime.compute_averaging_weights(
        selected_epochs,
        avg_mode=args.avg_mode,
        sigma_rel=float(args.sigma_rel),
    )
    averaged_state_dict = ensemble_runtime.average_checkpoints(
        selected_checkpoint_paths,
        weights=weights,
    )
    ensemble_checkpoint_path = ensemble_runtime.write_ensemble_checkpoint(
        output_dir,
        averaged_state_dict,
        avg_mode=args.avg_mode,
    )

    metadata_source = _resolve_metadata_source(training_run_dir, args.metadata_json)
    metadata_snapshot_path = None
    if metadata_source is not None:
        metadata_snapshot_path = ensemble_runtime.copy_metadata_snapshot(metadata_source, output_dir)

    manifest = {
        "training_run_dir": training_run_dir,
        "output_dir": output_dir,
        "avg_mode": str(args.avg_mode),
        "sigma_rel": float(args.sigma_rel),
        "epochs_override": args.epochs,
        "target_count": int(args.target_count),
        "dense_prefix": int(args.dense_prefix),
        "available_epochs": available_epochs,
        "selected_epochs": selected_epochs,
        "selected_checkpoints": selected_checkpoint_paths,
        "weights": [float(weight) for weight in weights],
        "ensemble_checkpoint": ensemble_checkpoint_path,
        "metadata_source": metadata_source,
        "metadata_snapshot": metadata_snapshot_path,
    }
    manifest_path = ensemble_runtime.write_manifest(output_dir, manifest)
    log(
        "ensemble complete "
        f"mode={args.avg_mode!r} checkpoints={len(selected_checkpoint_paths)} "
        f"output={ensemble_checkpoint_path!r} manifest={manifest_path!r}"
    )

    if args.run_stitch:
        if metadata_source is None:
            raise ValueError(
                "--run_stitch requires metadata: expected either <training_run_dir>/metadata.snapshot.json "
                "or explicit --metadata_json"
            )
        from stitch_train_resnet3d import run_stitch_jobs

        stitch_args = _build_stitch_args(
            args,
            metadata_path=metadata_source,
            ensemble_checkpoint_path=ensemble_checkpoint_path,
        )
        log(
            "starting stitch handoff "
            f"metadata_json={stitch_args.metadata_json!r} init_ckpt_path={stitch_args.init_ckpt_path!r}"
        )
        run_stitch_jobs(stitch_args)

    return {
        "training_run_dir": training_run_dir,
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "ensemble_checkpoint_path": ensemble_checkpoint_path,
        "selected_epochs": selected_epochs,
        "weights": weights,
        "metadata_source": metadata_source,
        "metadata_snapshot_path": metadata_snapshot_path,
    }


def main():
    args = parse_args()
    run_ensemble(args)


if __name__ == "__main__":
    main()
