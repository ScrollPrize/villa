from __future__ import annotations

import argparse
import os
from pathlib import Path

from .generate import generate_patch_caches


def _configure_blosc_threads() -> None:
    """
    Configure BLOSC threads for patch cache generation stability.

    If BLOSC_NTHREADS is not set, default to 1 to avoid allocator issues seen
    on some systems during long zarr scans. Users can override by exporting
    BLOSC_NTHREADS before running this command.
    """
    env_value = os.environ.get("BLOSC_NTHREADS")
    if env_value is None:
        desired_threads = 1
        emit_default_msg = True
    else:
        emit_default_msg = False
        try:
            desired_threads = int(env_value)
        except ValueError:
            desired_threads = 1
            print(f"Invalid BLOSC_NTHREADS={env_value!r}; falling back to 1.")
        if desired_threads < 1:
            desired_threads = 1

    try:
        from numcodecs import blosc

        blosc.set_nthreads(desired_threads)
        if emit_default_msg:
            print("BLOSC_NTHREADS not set; defaulting to 1 for patch-cache stability.")
    except Exception:
        # numcodecs/blosc may not be available in all environments.
        pass


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
    _configure_blosc_threads()

    result = generate_patch_caches(
        config_path=args.config,
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


if __name__ == "__main__":
    main()
