#!/usr/bin/env python3
"""Attach winding-inference crossings to verified patches.

The conventional invocation resolves both inputs from a Spiral dataset and
writes ``winding_patch_assignments`` beside ``winding_inference``::

    python scripts/spiral/build_winding_patch_assignments.py \
        --dataset /path/to/dataset

Input and output paths can be overridden for non-conventional layouts.  The
result is optional fit input: patch removals or edits later invalidate only the
affected assignments, not the artifact as a whole.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fit_session import conventional_input_paths, load_scroll_spec
from winding_patch_assignments import build_winding_patch_assignments


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True,
        help="Spiral dataset root containing spiral-scroll.json",
    )
    parser.add_argument(
        "--winding-inference",
        help="source winding-inference store (default: resolved dataset input)",
    )
    parser.add_argument(
        "--verified-patches",
        help="verified patch directory (default: resolved dataset input)",
    )
    parser.add_argument(
        "--output",
        help="output directory (default: <dataset>/winding_patch_assignments)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=2.5,
        help="maximum crossing-to-patch distance in voxels (default: 2.5)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=250_000,
        help="crossings queried per surface-index batch (default: 250000)",
    )
    parser.add_argument(
        "--patch-workers", type=int, default=0,
        help="parallel patch-loading processes (default: auto, up to 8)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="replace an existing output artifact",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="skip source array checksum verification",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = Path(args.dataset).resolve()
    scroll = load_scroll_spec(dataset)
    paths = conventional_input_paths(dataset, scroll)
    winding_inference = args.winding_inference or paths.winding_inference
    verified_patches = args.verified_patches or paths.verified_patches
    output = args.output or str(dataset / "winding_patch_assignments")
    build_winding_patch_assignments(
        winding_inference,
        verified_patches,
        output,
        tolerance=args.tolerance,
        chunk_size=args.chunk_size,
        patch_workers=args.patch_workers,
        force=args.force,
        verify=not args.no_verify,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
