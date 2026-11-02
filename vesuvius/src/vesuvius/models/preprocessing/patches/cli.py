from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .generate import generate_patch_caches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-volume patch caches for a Vesuvius dataset.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the training config YAML.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help=(
            "Dataset root with images/ and labels/ subdirectories. "
            "Takes precedence over the config's data_path, mirroring "
            "vesuvius.train -i/--input."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override cache directory (defaults to <data_path>/.patches_cache).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute caches even if they already exist.",
    )

    args = parser.parse_args()

    if args.input is not None and not args.input.is_dir():
        parser.error(f"Input is not a directory: {args.input}")

    result = generate_patch_caches(
        config_path=args.config,
        input_path=args.input,
        cache_dir=args.cache_dir,
        force=args.force,
    )

    print(
        f"Patch cache generation complete: "
        f"{result.written_caches} cache(s) written, "
        f"{result.scanned_volumes} volumes scanned, "
        f"{result.skipped_volumes} skipped."
    )
    print(
        f"Patches found: {result.total_fg_patches} FG, "
        f"{result.total_bg_patches} BG, "
        f"{result.total_unlabeled_fg_patches} unlabeled FG."
    )

    if result.total_volumes == 0:
        print(
            "Error: no volumes were found, so no cache was written. "
            "Point find_patches at the dataset root with -i/--input "
            "(the same directory vesuvius.train receives), or set "
            "data_path in the config YAML.",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
