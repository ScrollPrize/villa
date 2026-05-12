"""Pre-warm autoreg_mesh prefilter caches before any DDP launch.

The autoreg_mesh dataset builds a per-sample "valid split plans" cache that
is keyed by (segment-uuid set + frontier widths + crop size + ...). The
first build is expensive (sequential or fork-pool); the cache file is
written to {segments_path}/.patch_cache/valid_plans_<md5>.pkl and re-used
by every subsequent rank/process.

For the 8-GPU launch we want the cache fully warm before any DDP rank
starts. This script runs the build in a SINGLE process with no fork pool
(prefilter_num_workers=1) — that's the safest path: we have not yet opened
any S3 zarr handles (the C1 refactor defers that to worker_init), so fork
safety is moot, but single-process keeps the prewarm trivially debuggable.

Usage:
    uv run python scripts/pre_warm_prefilter.py <config.json>

Pass criteria:
    - Each of the 5 segments_path dirs ends up with a
      .patch_cache/valid_plans_*.pkl file.
    - Re-running this script with the same config logs `cache hit` for
      every dataset entry's prefilter key.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

from vesuvius.neural_tracing.autoreg_mesh.config import load_autoreg_mesh_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("config", type=str, help="Path to autoreg_mesh JSON config")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help=(
            "Override prefilter_num_workers (default 8). Fork-based pool is "
            "safe at pre-warm time because the lazy-zarr refactor keeps no S3 "
            "handles open during AutoregMeshDataset construction. Drop to 1 "
            "only for debugging."
        ),
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        print(f"[pre_warm_prefilter] config not found: {cfg_path}", flush=True)
        return 2

    cfg = load_autoreg_mesh_config(cfg_path)
    cfg = copy.deepcopy(cfg)
    cfg["prefilter_num_workers"] = int(args.workers)
    cfg["prefilter_show_progress"] = True

    print(
        f"[pre_warm_prefilter] config={cfg_path}",
        flush=True,
    )
    print(
        f"[pre_warm_prefilter] datasets={len(cfg['datasets'])} "
        f"prefilter_num_workers={cfg['prefilter_num_workers']}",
        flush=True,
    )
    for idx, entry in enumerate(cfg["datasets"]):
        print(
            f"[pre_warm_prefilter]   [{idx}] segments_path={entry['segments_path']} "
            f"volume_path={entry['volume_path']}",
            flush=True,
        )

    # Importing here so the print banner appears even if dataset import
    # would later fail.
    from vesuvius.neural_tracing.autoreg_mesh.dataset import AutoregMeshDataset

    t0 = time.perf_counter()
    dataset = AutoregMeshDataset(cfg)
    elapsed = time.perf_counter() - t0
    n_samples = len(getattr(dataset, "sample_index", []) or [])
    print(
        f"[pre_warm_prefilter] DONE in {elapsed:.1f}s; sample_index size={n_samples}",
        flush=True,
    )

    # Re-construct the dataset once to verify the cache hits.
    t1 = time.perf_counter()
    dataset2 = AutoregMeshDataset(cfg)
    elapsed2 = time.perf_counter() - t1
    n_samples2 = len(getattr(dataset2, "sample_index", []) or [])
    print(
        f"[pre_warm_prefilter] cache re-read in {elapsed2:.1f}s; sample_index size={n_samples2}",
        flush=True,
    )
    if n_samples != n_samples2:
        print(
            f"[pre_warm_prefilter] WARN: sample_index size changed across reads: "
            f"{n_samples} -> {n_samples2}",
            flush=True,
        )

    # Find written cache files (for the human reading the log).
    print("[pre_warm_prefilter] cache files on disk:", flush=True)
    for entry in cfg["datasets"]:
        seg_path = Path(entry["segments_path"])
        cache_dir = seg_path / ".patch_cache"
        if cache_dir.exists():
            for f in sorted(cache_dir.glob("valid_plans_*.pkl")):
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"  {f} ({size_mb:.1f} MiB)", flush=True)
        else:
            print(f"  WARN: no cache dir at {cache_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
