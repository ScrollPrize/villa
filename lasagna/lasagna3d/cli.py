"""lasagna3d — analysis & tooling CLI for lasagna 3D training.

Usage:
    python -m lasagna3d dataset vis --train-config CONFIG --vis-dir OUT
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="lasagna3d")
    sub = parser.add_subparsers(dest="group", required=True)

    dataset = sub.add_parser("dataset", help="Dataset inspection/analysis")
    dataset_sub = dataset.add_subparsers(dest="command", required=True)

    vis = dataset_sub.add_parser(
        "vis",
        help="Visualize dataset samples as 3-plane jpegs with surface overlays.",
    )
    vis.add_argument("--train-config", required=True,
                     help="Path to the lasagna training JSON config.")
    vis.add_argument("--vis-dir", required=True,
                     help="Output directory for jpeg visualizations.")
    vis.add_argument("--num-samples", type=int, default=10,
                     help="How many samples to render per dataset (default: 10).")
    vis.add_argument("--seed", type=int, default=0,
                     help="Seed for deterministic shuffle (default: 0).")
    vis.add_argument("--patch-size", type=int, default=None,
                     help="Model-architecture patch size used to BUILD the "
                          "model (NetworkFromConfig autoconfigures from "
                          "this). Only needed as a fallback for old "
                          "checkpoints that don't embed `patch_size`. Does "
                          "NOT change the dataset patch size — the dataset "
                          "always uses the training config's `patch_size`.")
    vis.add_argument("--model", type=str, default=None,
                     help="Optional checkpoint path. If set, runs inference "
                          "and adds rows for prediction + diff (full + "
                          "scale-space sum). Loss values are added to the "
                          "title.")
    vis.add_argument("--inference-tile-size", type=int, default=None,
                     help="Cubic patch size of the CT crop the model runs "
                          "on (single forward). The dataset is unaffected "
                          "and always emits the training config patch size. "
                          "Loss + residuals are computed at min(this, "
                          "config patch_size); vis is rendered at the "
                          "config patch size with pred padded/cropped. "
                          "Default: training config patch size.")
    vis.add_argument("--num-workers", type=int, default=None,
                     help="DataLoader workers for parallel extraction "
                          "and render thread pool size. Default: number "
                          "of CPU cores (os.cpu_count()). 0 runs "
                          "everything on the main thread.")

    args = parser.parse_args()

    if args.group == "dataset" and args.command == "vis":
        from lasagna3d.dataset_vis import run_dataset_vis
        run_dataset_vis(
            train_config=args.train_config,
            vis_dir=args.vis_dir,
            num_samples=args.num_samples,
            seed=args.seed,
            patch_size=args.patch_size,
            num_workers=args.num_workers,
            model_path=args.model,
            inference_tile_size=args.inference_tile_size,
        )
        return

    parser.print_help()
    sys.exit(1)
