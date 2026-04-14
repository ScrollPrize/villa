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
    vis.add_argument("--idx", type=str, default=None,
                     help="Comma-separated list of specific patch indices "
                          "to render (e.g. `--idx 17,42,108`). Bypasses "
                          "--num-samples/--seed shuffle and scans exactly "
                          "those patches, in order, in every dataset.")
    vis.add_argument("--same-surface-threshold", type=float, default=None,
                     help="Opt-in voxel-median distance threshold for the "
                          "same-surface merge inside compute_patch_labels. "
                          "Duplicate wraps closer than this are collapsed "
                          "into one surface for EDT, validity, and the "
                          "loss — the vis still draws both contours but "
                          "with a shared label/color. Overrides the "
                          "training config's `same_surface_threshold`.")

    overlap = dataset_sub.add_parser(
        "overlap",
        help="Per-patch pairwise surface-overlap diagnostics (JSONL out).",
    )
    overlap.add_argument("--train-config", required=True,
                         help="Path to the lasagna training JSON config.")
    overlap.add_argument("--out", required=True,
                         help="Output JSONL path (one record per patch).")
    overlap.add_argument("--num-samples", type=int, default=None,
                         help="Patches to scan per dataset. Default: all.")
    overlap.add_argument("--seed", type=int, default=0,
                         help="Seed for per-dataset deterministic shuffle.")
    overlap.add_argument("--patch-size", type=int, default=None,
                         help="Unused (kept for CLI symmetry with 'vis').")
    overlap.add_argument("--num-workers", type=int, default=None,
                         help="DataLoader workers. Default: os.cpu_count(). "
                              "0 runs everything on the main thread.")
    overlap.add_argument("--vis-dir", type=str, default=None,
                         help="If set, render a `dataset vis`-style JPEG for "
                              "each emitted patch into this directory. "
                              "Filenames embed the worst-pair p1 so sorting "
                              "by name puts the worst patches first.")
    overlap.add_argument("--vis-top-k", type=int, default=None,
                         help="With --vis-dir, render only the K worst "
                              "patches (by p1) after scanning completes. "
                              "Default: render every emitted patch "
                              "inline during the scan.")
    overlap.add_argument("--model", type=str, default=None,
                         help="Optional checkpoint path passed through to "
                              "the vis renderer; same semantics as "
                              "`dataset vis --model`. Only meaningful with "
                              "--vis-dir.")
    overlap.add_argument("--inference-tile-size", type=int, default=None,
                         help="Inference tile size forwarded to the vis "
                              "renderer; same semantics as `dataset vis "
                              "--inference-tile-size`. Only meaningful with "
                              "--vis-dir + --model.")
    overlap.add_argument("--idx", type=str, default=None,
                         help="Comma-separated list of specific patch "
                              "indices to scan (e.g. `--idx 17,42,108`). "
                              "Bypasses --num-samples/--seed shuffle and "
                              "scans exactly those patches, in order, in "
                              "every dataset.")

    args = parser.parse_args()

    def _parse_idx(s):
        if s is None:
            return None
        return [int(tok) for tok in s.split(",") if tok.strip()]

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
            explicit_indices=_parse_idx(args.idx),
            same_surface_threshold=args.same_surface_threshold,
        )
        return

    if args.group == "dataset" and args.command == "overlap":
        from lasagna3d.dataset_overlap import run_dataset_overlap
        run_dataset_overlap(
            train_config=args.train_config,
            out_path=args.out,
            num_samples=args.num_samples,
            seed=args.seed,
            patch_size=args.patch_size,
            num_workers=args.num_workers,
            vis_dir=args.vis_dir,
            vis_top_k=args.vis_top_k,
            model_path=args.model,
            inference_tile_size=args.inference_tile_size,
            explicit_indices=_parse_idx(args.idx),
        )
        return

    parser.print_help()
    sys.exit(1)
