#!/usr/bin/env python
"""Cast winding-model rays from a fitted Spiral checkpoint and store the
decoded absolute-winding predictions.

Rays are seeded on the fitted surface every ``--winding-step`` windings, at
``--seed-spacing`` voxels along the winding and along z, and cast along the
outward surface normal with the seed crossing at the ray midpoint. Each slab
inference yields a grid of columns whose crossing peaks are decoded with
phase-level dedup and anchored to the seed's absolute winding index.

Output is a single zarr group:
  winding     int16, full volume shape from --reference-zarr (downsampled by
              --output-downsample), fill -1; sparse: only inferred chunks
              exist. Per voxel: neighborhood vote over all overlapping
              slabs' observations, weighted by confidence and proximity to
              each slab's seed anchor
  confidence  uint8, same shape: the winner's vote share x its best
              observation prob
  points/*    flat per-crossing records (xyz, winding, prob, seed, slab)
  strips/offsets   CSR offsets into points/* (one strip per decoded column)

--prob-volume observations are reduced on the GPU into either rendered-kernel
statistics or registered-phase consensus statistics, compressed per output
chunk in the scratch dir, and merged in parallel (--merge-workers), so memory
and temporary storage stay bounded.
Coordinates in points/xyz are voxels of the reference zarr's scale-0 array
(the Spiral fit's coordinate space); the checkpoint, decode parameters and
grid definition are stored in the group attrs.

Multi-GPU runs use one plain worker process per GPU (no DDP/torch.distributed;
inference has no gradient sync, so processes just shard the slab list).

Example:

    python infer_winding_volume.py \\
        /path/to/checkpoint_fitted.ckpt out_windings.zarr \\
        --reference-zarr /path/to/volume.ome.zarr --volume-scale 0 \\
        --model-ckpt /path/to/winding_model/ckpt_final.pth \\
        --z-range 9000 15000 --winding-step 8 --seed-spacing 96 \\
        --gpus 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

# Output zarr chunk edge; prob-volume spill buckets are full z/y/x chunks.
_OUTPUT_CHUNK = 128
# Raw prob-volume record used only by the portable CPU fallback.
_PROB_RECORD = np.dtype([("key", "<u8"), ("val", "u1")])
# Spill records are partial reductions, not individual slab observations.
# ``total/count`` reproduce mean combine exactly and ``maximum`` reproduces
# max combine.  This is substantially larger per record, but GPU reduction
# removes repeated voxel observations before anything reaches disk.
_PROB_AGG_RECORD = np.dtype([
    ("key", "<u8"), ("total", "<u8"), ("count", "<u4"), ("maximum", "u1")
])
# Phase-consensus partials.  Cosine/sine merge fractional phase without
# letting an integer winding-count disagreement move a crossing.  Density is
# the local phase derivative in windings per scale-0 voxel; it converts the
# merged phase residual back to a physical distance at finalization.
_PHASE_AGG_RECORD = np.dtype([
    ("key", "<u8"), ("cosine", "<f4"), ("sine", "<f4"),
    ("density", "<f4"), ("weight", "<f4"), ("count", "<u4"),
])
# Topology-free overlap mode keeps the synchronized integer level in ``key``:
# ``linear_output_voxel * 65536 + signed_winding``.  Squared weight permits an
# effective independent-observation gate, so an almost-zero edge vote cannot
# merely unlock a full-strength proposal from one slab.
_PHASE_LABEL_AGG_RECORD = np.dtype([
    ("key", "<u8"), ("cosine", "<f4"), ("sine", "<f4"),
    ("density", "<f4"), ("weight", "<f4"), ("weight_sq", "<f4"),
    ("count", "<u4"),
])
# One decoded observation assigned to an output chunk for winding voting.
# ``order`` is its deterministic global point order; the rasterizer uses it
# with the source chunk encoded in ``key`` to reproduce the legacy stable-sort
# accumulation order exactly.
_WINDING_RECORD = np.dtype([
    ("key", "<u8"), ("order", "<u8"), ("winding", "<i2"),
    ("prob", "u1"), ("level", "u1"),
])
# Inference-time reduction of source-voxel votes. ``key`` packs the linear
# output voxel and signed winding candidate; neighborhood expansion happens
# once per reduced entry on the GPUs during finalization.
_WINDING_AGG_RECORD = np.dtype([
    ("key", "<u8"), ("total", "<f4"), ("maximum", "u1"),
])
_PROB_KEY_MASK = (1 << 21) - 1
# H100-class devices have ample memory after this model's forward (~10 GiB).
# Accumulating several adjacent batches greatly increases spatial overlap and
# therefore the reduction ratio, while remaining bounded.
_PROB_GPU_FLUSH_RECORDS = 1 << 26


def _append_prob_aggregates(path, records):
    """Append one independently compressed aggregate block."""
    from numcodecs import Blosc

    # Byte shuffle is both faster and smaller than bitshuffle for this
    # structured record (u64/u64/u32/u8).  On representative full-resolution
    # spills it reduced encode time by ~28% and bytes by ~22%.
    codec = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)
    encoded = codec.encode(np.ascontiguousarray(records))
    with open(path, "ab") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)


def _iter_prob_aggregates(path):
    """Yield aggregate arrays from concatenated compressed blocks."""
    from numcodecs import Blosc

    codec = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)
    with open(path, "rb") as handle:
        while True:
            header = handle.read(8)
            if not header:
                return
            if len(header) != 8:
                raise RuntimeError(f"truncated probability spill header: {path}")
            size = struct.unpack("<Q", header)[0]
            encoded = handle.read(size)
            if len(encoded) != size:
                raise RuntimeError(f"truncated probability spill block: {path}")
            yield np.frombuffer(codec.decode(encoded), dtype=_PROB_AGG_RECORD)


def _append_phase_aggregates(path, records):
    """Append one independently compressed phase-consensus block."""
    from numcodecs import Blosc

    codec = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)
    encoded = codec.encode(np.ascontiguousarray(records))
    with open(path, "ab") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)


def _iter_phase_aggregates(path):
    """Yield phase-consensus arrays from concatenated compressed blocks."""
    from numcodecs import Blosc

    codec = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)
    with open(path, "rb") as handle:
        while True:
            header = handle.read(8)
            if not header:
                return
            if len(header) != 8:
                raise RuntimeError(f"truncated phase spill header: {path}")
            size = struct.unpack("<Q", header)[0]
            encoded = handle.read(size)
            if len(encoded) != size:
                raise RuntimeError(f"truncated phase spill block: {path}")
            yield np.frombuffer(codec.decode(encoded), dtype=_PHASE_AGG_RECORD)


def _append_phase_label_aggregates(path, records):
    """Append one independently compressed labeled-phase block."""
    from numcodecs import Blosc

    codec = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)
    encoded = codec.encode(np.ascontiguousarray(records))
    with open(path, "ab") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)


def _iter_phase_label_aggregates(path):
    """Yield labeled-phase arrays from concatenated compressed blocks."""
    from numcodecs import Blosc

    codec = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)
    with open(path, "rb") as handle:
        while True:
            header = handle.read(8)
            if not header:
                return
            if len(header) != 8:
                raise RuntimeError(
                    f"truncated labeled-phase spill header: {path}")
            size = struct.unpack("<Q", header)[0]
            encoded = handle.read(size)
            if len(encoded) != size:
                raise RuntimeError(
                    f"truncated labeled-phase spill block: {path}")
            yield np.frombuffer(
                codec.decode(encoded), dtype=_PHASE_LABEL_AGG_RECORD)


def _append_winding_records(path, records):
    """Append one independently compressed winding-observation block."""
    from numcodecs import Blosc

    codec = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)
    encoded = codec.encode(np.ascontiguousarray(records))
    with open(path, "ab") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)


def _iter_winding_records(path):
    """Yield winding observations from concatenated compressed blocks."""
    from numcodecs import Blosc

    codec = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)
    with open(path, "rb") as handle:
        while True:
            header = handle.read(8)
            if not header:
                return
            if len(header) != 8:
                raise RuntimeError(f"truncated winding spill header: {path}")
            size = struct.unpack("<Q", header)[0]
            encoded = handle.read(size)
            if len(encoded) != size:
                raise RuntimeError(f"truncated winding spill block: {path}")
            yield np.frombuffer(codec.decode(encoded), dtype=_WINDING_RECORD)


def _append_winding_aggregates(path, records):
    """Append one compressed source-vote aggregate block."""
    from numcodecs import Blosc

    codec = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)
    encoded = codec.encode(np.ascontiguousarray(records))
    with open(path, "ab") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)


def _iter_winding_aggregates(path):
    """Yield source-vote aggregates from concatenated compressed blocks."""
    from numcodecs import Blosc

    codec = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)
    with open(path, "rb") as handle:
        while True:
            header = handle.read(8)
            if not header:
                return
            if len(header) != 8:
                raise RuntimeError(f"truncated winding aggregate header: {path}")
            size = struct.unpack("<Q", header)[0]
            encoded = handle.read(size)
            if len(encoded) != size:
                raise RuntimeError(f"truncated winding aggregate block: {path}")
            yield np.frombuffer(
                codec.decode(encoded), dtype=_WINDING_AGG_RECORD)


def _spiral_scripts_dir():
    """volume-cartographer/scripts/spiral, for the raw-spiral seed fallback.

    Resolved from $SPIRAL_SCRIPTS_DIR, or the villa checkout layout relative
    to this file (vesuvius/src/vesuvius/neural_tracing/winding_models).
    """
    override = os.environ.get("SPIRAL_SCRIPTS_DIR")
    candidates = [Path(override)] if override else []
    here = Path(__file__).resolve()
    candidates += [parent / "volume-cartographer/scripts/spiral"
                   for parent in here.parents]
    for candidate in candidates:
        if (candidate / "transforms.py").is_file():
            return str(candidate)
    raise FileNotFoundError(
        "cannot locate volume-cartographer/scripts/spiral; set "
        "SPIRAL_SCRIPTS_DIR (only needed for --seed-source spiral)")


def _load_checkpoint_cfg(path):
    """The fit checkpoint's cfg/z-range metadata, with lazily mapped tensors."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _native_phase_cache_model_cfg(path):
    """Model geometry stored in a lossless headless native-phase cache."""
    metadata_path = Path(path) / "zarr.json"
    with metadata_path.open() as handle:
        attrs = json.load(handle).get("attributes", {})
    if attrs.get("artifact_type") != "winding_native_phase_cache":
        raise ValueError(f"not a native winding phase cache: {path}")
    return {
        "model": {"use_crossing_head": False},
        "ray_length": int(attrs["ray_length"]),
        "transverse_size": int(attrs["transverse_size"]),
        "column_stride": int(attrs["column_stride"]),
        "spacing": float(attrs["spacing"]),
        "sampling": str(attrs["sampling"]),
        "crossing_sigma_wv": float(attrs["crossing_sigma_wv"]),
    }

ANCHOR_TOLERANCE = 6.0  # max |t_peak - midpoint| for the seed anchor, samples
EDGE_MARGIN = 8  # ignore peaks this close to the ray ends, samples
TRANSFORM_CHUNK = 200_000  # points per SpiralAndTransform.inv call


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("fit_checkpoint", help="fitted Spiral checkpoint (.ckpt)")
    parser.add_argument("output", help="output zarr group path")
    parser.add_argument("--reference-zarr", required=True,
                        help="volume zarr (multiscale group or array); scale-0 "
                             "shape defines the output coordinate space and the "
                             "selected scale is sampled for inference")
    parser.add_argument("--volume-scale", type=int, default=0,
                        help="pyramid level of the reference zarr to sample")
    parser.add_argument("--model-ckpt", required=True,
                        help="winding model checkpoint (.pth)")
    parser.add_argument("--umbilicus", default=None,
                        help="umbilicus.json (default: search near checkpoint / "
                             "SPIRAL_DATASET)")
    parser.add_argument("--seed-source", choices=("meshes", "spiral"),
                        default="meshes",
                        help="'meshes' (default) seeds rays on the checkpoint's "
                             "exported _spliced winding meshes, which are "
                             "snapped onto satisfied patches (real sheets); "
                             "'spiral' seeds on the raw fitted spiral surface, "
                             "which can sit a winding or more off the true "
                             "sheet and then mislabels the anchor")
    parser.add_argument("--meshes-dir", default=None,
                        help="winding-mesh directory (w###_spliced_* tifxyz "
                             "dirs); default: newest dir under "
                             "<checkpoint dir>/meshes")
    parser.add_argument("--z-range", type=int, nargs=2, default=None,
                        metavar=("Z_MIN", "Z_MAX"),
                        help="z range to seed rays in, scale-0 voxels "
                             "(default: the checkpoint's fitted z range)")
    parser.add_argument("--winding-range", type=int, nargs=2, default=None,
                        metavar=("FIRST", "LAST"),
                        help="seed winding range (default: checkpoint's "
                             "output_first_winding..shell_outer_winding_idx)")
    parser.add_argument("--winding-step", type=int, default=8,
                        help="seed a ray sheet every this many windings")
    parser.add_argument("--seed-spacing", type=float, default=96.0,
                        help="seed spacing along the winding and along z, voxels")
    parser.add_argument("--column-step", type=int, default=4,
                        help="decode every Nth column per transverse axis "
                             "(1 = every column of the, possibly upsampled, "
                             "grid)")
    parser.add_argument(
        "--slab-center-width", type=float, default=None,
        help="optional scale-0-voxel width of the central square retained "
             "from each slab for both decoded points and crossing_prob. "
             "Use a value modestly larger than the cache's seed spacing "
             "to assign each seed a local tile and avoid repeatedly "
             "materializing the heavily overlapping full slab footprint")
    parser.add_argument("--column-upsample", type=int, default=1,
                        help="transversely interpolate the model's output "
                             "fields onto this-factor-finer column grid before "
                             "decoding (must divide the model's column stride; "
                             "e.g. 2 turns a stride-2 model into per-voxel "
                             "columns). The phase field is transversely smooth "
                             "at stride scale, so linear interpolation is "
                             "near-exact; passage kernels are rendered from "
                             "the interpolated phase and stay unit-height.")
    parser.add_argument("--threshold", type=float, default=0.285,
                        help="crossing peak threshold")
    parser.add_argument("--min-distance", type=int, default=3,
                        help="peak NMS distance, samples")
    parser.add_argument("--min-prob-keep", type=float, default=0.0,
                        help="drop decoded crossings below this probability")
    parser.add_argument("--max-level", type=int, default=8,
                        help="drop crossings more than this many windings from "
                             "the seed anchor (accuracy decays with distance; "
                             "keep <= 2x winding-step so slabs still overlap)")
    parser.add_argument("--prob-volume", action="store_true",
                        help="also write a 'crossing_prob' uint8 evidence "
                             "array (not calibrated probability for headless "
                             "models): unit-height passage kernels or merged "
                             "registered phase, selected by --prob-combine, "
                             "sampled along every decoded column and "
                             "combined per voxel over overlapping slabs "
                             "(--prob-combine), independent of anchor "
                             "success. Meant for --output-downsample >= 2; at "
                             "full resolution it gets very large.")
    parser.add_argument("--archive-workers", type=int, default=None,
                        help="processes materializing the required point and "
                             "strip archives into independent Zarr chunks "
                             "(default: min(8, cpus))")
    parser.add_argument("--prob-volume-floor", type=float, default=0.15,
                        help="combined crossing_prob values below this are "
                             "zeroed at merge time (keeps the array sparse)")
    parser.add_argument(
        "--prob-combine", choices=("mean", "max", "phase", "phase-label"),
                        default="mean",
                        help="per-voxel combine over overlapping slabs' "
                             "prob-volume observations; mean suppresses "
                             "single-slab edge bias, max is the old behavior, "
                             "'phase' robustly merges registered phase modulo "
                             "one winding, while 'phase-label' keeps the "
                             "synchronized absolute level separate and gates "
                             "on effective weighted support (headless models)")
    parser.add_argument("--prob-ray-margin", type=int, default=32,
                        help="prob-volume records drop this many samples at "
                             "each ray end, where the model systematically "
                             "over-predicts")
    parser.add_argument("--prob-column-margin", type=int, default=2,
                        help="prob-volume records drop this many columns at "
                             "each transverse slab edge")
    parser.add_argument("--merge-workers", type=int, default=None,
                        help="processes folding spilled prob-volume records "
                             "into the output zarr; each holds one output "
                             "chunk. Default min(32, cpus)")
    parser.add_argument("--raster-workers", type=int, default=None,
                        help="deprecated compatibility option; reduced "
                             "winding chunks are rasterized on --gpus")
    parser.add_argument("--prob-column-step", type=int, default=None,
                        help="column stride for the crossing_prob sampling "
                             "only (default: --column-step). 1 samples every "
                             "model column (4-voxel line spacing, dense at "
                             "pyramid level 2) without inflating the decoded "
                             "strips/points")
    parser.add_argument("--prob-phase-level-half-life", type=float, default=2.0,
                        help="with --prob-combine phase, halve a slab's merge "
                             "weight this many windings from its seed anchor")
    parser.add_argument("--prob-phase-max-level", type=float, default=None,
                        help="with --prob-combine phase, ignore samples farther "
                             "than this many windings from the seed (default: "
                             "--max-level + 0.5 passage support)")
    parser.add_argument("--prob-phase-edge-taper", type=int, default=8,
                        help="with --prob-combine phase, cosine-taper this many "
                             "samples inward from the retained ray/column "
                             "margins")
    parser.add_argument("--prob-phase-agreement-power", type=float, default=1.0,
                        help="power applied to circular phase agreement when "
                             "rendering phase-consensus crossing evidence")
    parser.add_argument("--prob-phase-min-observations", type=int, default=2,
                        help="minimum distinct slab observations required for "
                             "phase-consensus crossing evidence")
    parser.add_argument(
        "--prob-phase-min-effective-observations", type=float, default=1.5,
        help="with --prob-combine phase-label, minimum Kish effective slab "
             "support (sum(w)^2/sum(w^2)) for a winding proposal")
    parser.add_argument(
        "--prob-phase-min-weight", type=float, default=0.5,
        help="with --prob-combine phase-label, minimum total tapered anchor "
             "weight for a winding proposal")
    parser.add_argument("--prob-phase-band-sigma", type=float, default=4.0,
                        help="with --prob-combine phase, project only samples "
                             "within this many crossing-kernel sigmas of an "
                             "integer passage (default 4; values outside this "
                             "band cannot affect the stored uint8 evidence)")
    parser.add_argument(
        "--phase-registration", choices=("anchor", "overlap"),
        default="anchor",
        help="phase gauge used during reconstruction. 'anchor' preserves the "
             "legacy assumption that every fitted seed midpoint is an exact "
             "integer crossing; 'overlap' estimates one continuous correction "
             "per slab from world-space phase agreement between spatially "
             "overlapping cached slabs, using seed winding only as a weak "
             "gauge prior and making no fitted-spiral topology assumption")
    parser.add_argument(
        "--phase-sync-radius", type=float, default=192.0,
        help="with --phase-registration overlap, maximum world-space seed "
             "distance considered for overlap probes")
    parser.add_argument(
        "--phase-sync-neighbors", type=int, default=24,
        help="with --phase-registration overlap, maximum spatial probe "
             "neighbors retained per slab")
    parser.add_argument(
        "--phase-sync-workers", type=int, default=None,
        help="processes reading cached slabs for overlap synchronization "
             "(default min(16, cpus))")
    parser.add_argument(
        "--phase-sync-block-size", type=int, default=256,
        help="selected slabs per overlap-probe task")
    parser.add_argument(
        "--phase-sync-transverse-margin", type=int, default=8,
        help="ignore overlap probes this many full-resolution samples from "
             "a cached slab's transverse edges")
    parser.add_argument(
        "--phase-sync-ray-margin", type=int, default=32,
        help="ignore overlap probes this many samples from ray ends")
    parser.add_argument(
        "--phase-sync-taper", type=int, default=12,
        help="raised-cosine reliability taper inside synchronization margins")
    parser.add_argument(
        "--phase-sync-min-density", type=float, default=0.01,
        help="minimum local phase derivative in windings/voxel for an "
             "overlap constraint")
    parser.add_argument(
        "--phase-sync-iterations", type=int, default=5,
        help="robust graph-solve IRLS iterations")
    parser.add_argument(
        "--phase-sync-huber", type=float, default=0.25,
        help="Huber transition for overlap residuals, in windings")
    parser.add_argument(
        "--phase-sync-prior-weight", type=float, default=0.02,
        help="weak fitted-seed gauge prior relative to one overlap edge")
    parser.add_argument(
        "--phase-sync-prior-huber", type=float, default=0.5,
        help="Huber transition for fitted-seed gauge priors, in windings")
    parser.add_argument(
        "--phase-sync-max-correction", type=float, default=4.0,
        help="maximum continuous phase correction applied to one slab")
    parser.add_argument(
        "--phase-sync-recompute", action="store_true",
        help="discard a compatible synchronization solution already present "
             "in OUTPUT.tmp and rebuild it")
    parser.add_argument("--output-downsample", type=int, default=4,
                        help="downsample factor of the winding/confidence arrays "
                             "relative to scale-0")
    parser.add_argument("--gpus", default=None,
                        help="comma-separated GPU indices (default: all visible)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model in each worker (adds "
                             "~a minute of warmup per GPU; pays off on long "
                             "runs)")
    parser.add_argument("--extract-threads", type=int, default=8,
                        help="slab-extraction threads per GPU worker")
    parser.add_argument("--volume-cache-bytes", type=int, default=None,
                        help="volume chunk cache per GPU worker (default: the "
                             "model checkpoint's volume_cache_bytes)")
    parser.add_argument("--decode-workers", type=int, default=3,
                        help="decode threads per GPU worker (the peak decode "
                             "is CPU-bound and overlaps the GPU forward)")
    parser.add_argument("--max-slabs", type=int, default=None,
                        help="cap the number of slabs (benchmark/smoke runs)")
    parser.add_argument(
        "--max-slabs-selection", choices=("random", "first"),
        default="random",
        help="selection used by --max-slabs (default random; 'first' keeps "
             "a contiguous cache range for I/O throughput benchmarks)")
    parser.add_argument("--seed-rays-npz", default=None,
                        help="reuse a previously computed seed-ray file")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--native-phase-only", action="store_true",
        help="write a lossless native-stride phase/validity cache to OUTPUT "
             "and defer point, probability, and winding materialization")
    modes.add_argument(
        "--phase-cache", default=None,
        help="reconstruct the normal output from a --native-phase-only Zarr, "
             "skipping volume extraction and neural-model inference")
    parser.add_argument(
        "--phase-cache-allow-relocated-inputs", action="store_true",
        help="allow --model-ckpt and --reference-zarr paths to differ from "
             "the paths recorded in a copied phase cache. Model inference is "
             "not rerun; cached model geometry is used, while the supplied "
             "reference Zarr must exist and defines the output shape")
    parser.add_argument(
        "--phase-cache-winding-stride", type=int, default=1,
        help="reconstruct every Nth cached seed-winding sheet (and always "
             "the final sheet); only applies with --phase-cache. A stride "
             "above 1 reduces cache reads but requires --max-level to cover "
             "the wider retained-anchor gap")
    args = parser.parse_args()
    if args.native_phase_only:
        derived_options = {
            "--column-step", "--column-upsample", "--threshold",
            "--min-distance", "--min-prob-keep", "--max-level",
            "--prob-volume", "--archive-workers", "--prob-volume-floor",
            "--prob-combine", "--prob-ray-margin", "--prob-column-margin",
            "--merge-workers", "--raster-workers", "--prob-column-step",
            "--prob-phase-level-half-life", "--prob-phase-max-level",
            "--prob-phase-edge-taper", "--prob-phase-agreement-power",
            "--prob-phase-min-observations",
            "--prob-phase-min-effective-observations",
            "--prob-phase-min-weight", "--prob-phase-band-sigma",
            "--output-downsample", "--slab-center-width",
            "--phase-cache-winding-stride",
            "--phase-cache-allow-relocated-inputs",
            "--phase-registration", "--phase-sync-radius",
            "--phase-sync-neighbors", "--phase-sync-workers",
            "--phase-sync-block-size", "--phase-sync-transverse-margin",
            "--phase-sync-ray-margin", "--phase-sync-taper",
            "--phase-sync-min-density", "--phase-sync-iterations",
            "--phase-sync-huber", "--phase-sync-prior-weight",
            "--phase-sync-prior-huber", "--phase-sync-max-correction",
            "--phase-sync-recompute",
            "--decode-workers",
        }
        supplied = sorted({
            option
            for token in sys.argv[1:]
            for option in derived_options
            if token == option or token.startswith(option + "=")
        })
        if supplied:
            parser.error(
                "--native-phase-only does not materialize derived outputs; "
                "remove these reconstruction-only options: "
                + ", ".join(supplied))
    return args


# --------------------------------------------------------------------------
# Stage A: seed rays from the fit
# --------------------------------------------------------------------------

def _find_umbilicus(args):
    if args.umbilicus:
        return args.umbilicus
    candidates = [Path(args.fit_checkpoint).parent / "umbilicus.json"]
    if os.environ.get("SPIRAL_DATASET"):
        candidates.insert(0, Path(os.environ["SPIRAL_DATASET"]) / "umbilicus.json")
    candidates += [p / "umbilicus.json"
                   for p in Path(args.fit_checkpoint).resolve().parents]
    found = next((str(c) for c in candidates if c.is_file()), None)
    if found is None:
        raise FileNotFoundError("pass --umbilicus or set SPIRAL_DATASET")
    return found


def build_seed_rays_from_meshes(args):
    """Seed rays on the fit's exported ``_spliced`` winding meshes.

    The spliced variants replace the raw spiral surface with the actual
    verified-patch geometry wherever those patches are satisfied, so seeds
    land on real sheets and the ray-midpoint crossing genuinely carries the
    seed's winding index. The raw spiral surface can sit a winding or more
    away from the sheet (its splice tolerance alone is ~0.5 windings).
    """
    import re
    import vesuvius.tifxyz as tifxyz

    checkpoint = _load_checkpoint_cfg(args.fit_checkpoint)
    cfg = dict(checkpoint["cfg"])
    z_begin, z_end = int(checkpoint["z_begin"]), int(checkpoint["z_end"])
    z_lo, z_hi = args.z_range if args.z_range else (z_begin, z_end)
    z_lo, z_hi = max(z_lo, z_begin), min(z_hi, z_end)
    if z_lo >= z_hi:
        raise ValueError(f"empty z range [{z_lo}, {z_hi})")
    dr_per_winding = float(cfg.get("initial_dr_per_winding", 16.0))
    del checkpoint

    if args.meshes_dir:
        meshes_dir = Path(args.meshes_dir)
    else:
        root = Path(args.fit_checkpoint).parent / "meshes"
        subdirs = sorted((p for p in root.iterdir() if p.is_dir()),
                         key=lambda p: p.stat().st_mtime)
        if not subdirs:
            raise FileNotFoundError(f"no mesh exports under {root}; pass "
                                    "--meshes-dir or --seed-source spiral")
        meshes_dir = subdirs[-1]

    if args.winding_range:
        first, last = args.winding_range
    else:
        first = int(cfg.get("output_first_winding", 10))
        last = int(cfg.get("shell_outer_winding_idx")
                   or int(cfg["gap_expander_num_windings"]) - 1)
    wanted = set(range(first, last + 1, args.winding_step))

    pattern = re.compile(r"^w(?P<index>\d+)_spliced")
    mesh_paths, all_mesh_paths = {}, {}
    for path in meshes_dir.iterdir():
        match = pattern.match(path.name)
        if match:
            all_mesh_paths[int(match["index"])] = path
            if int(match["index"]) in wanted:
                mesh_paths[int(match["index"])] = path
    if not mesh_paths:
        raise FileNotFoundError(
            f"no w###_spliced meshes for windings {sorted(wanted)} in "
            f"{meshes_dir}")

    umbilicus_path = _find_umbilicus(args)

    def load_grid(path):
        surface = tifxyz.read_tifxyz(path)
        grid = np.stack([surface._x, surface._y, surface._z],
                        axis=-1).astype(np.float64)
        valid = surface.valid_vertex_mask & np.isfinite(grid).all(-1)
        grid[~valid] = np.nan
        return grid, valid

    def neighbor_vertices(winding):
        """Vertices of an adjacent winding's mesh and the sign of its offset."""
        for delta in (1, -1):
            path = all_mesh_paths.get(winding + delta)
            if path is None:
                continue
            grid, valid = load_grid(path)
            vertices = grid[valid & (np.abs(grid[..., 2] - (z_lo + z_hi) / 2)
                                     <= (z_hi - z_lo) / 2 + 200)]
            if len(vertices):
                return vertices, float(delta)
        return None, 0.0

    from scipy.spatial import cKDTree

    origins, directions, windings = [], [], []
    for winding, path in sorted(mesh_paths.items()):
        grid, valid = load_grid(path)

        meta = json.loads((path / "meta.json").read_text())
        step = 1.0 / float(meta["scale"][0])
        stride = max(1, int(round(args.seed_spacing / step)))

        tangent_z = np.gradient(grid, axis=0)
        tangent_theta = np.gradient(grid, axis=1)
        normal = np.cross(tangent_theta, tangent_z)
        norm = np.linalg.norm(normal, axis=-1)

        sub = np.zeros(grid.shape[:2], dtype=bool)
        sub[::stride, ::stride] = True
        keep = (
            sub & valid
            & np.isfinite(normal).all(-1) & (norm > 1e-8)
            & (grid[..., 2] >= z_lo) & (grid[..., 2] < z_hi)
        )
        points = grid[keep]
        normals = normal[keep] / norm[keep][:, None]
        if not len(points):
            continue

        # Orient toward the next-higher winding: the umbilicus-radial
        # direction flips where sheets fold back, but "toward winding w+1"
        # is outward by definition. Sign each normal by its projection onto
        # the offset to the nearest vertex of the adjacent winding's mesh.
        vertices, delta = neighbor_vertices(winding)
        if vertices is None:
            raise RuntimeError(
                f"winding {winding} has no adjacent mesh to orient against")
        _, nearest = cKDTree(vertices).query(points, workers=-1)
        toward = (vertices[nearest] - points) * delta
        sign = np.sign(np.einsum("nc,nc->n", normals, toward))
        sign[sign == 0] = 1.0
        normals *= sign[:, None]

        origins.append(points)
        directions.append(normals)
        windings.append(np.full(len(points), winding, dtype=np.int16))
        print(f"[seed] winding {winding}: {len(points)} rays "
              f"({path.name})", flush=True)

    if not origins:
        raise RuntimeError("no seed rays inside the requested z range")
    return {
        "seed_xyz": np.concatenate(origins).astype(np.float32),
        "direction_xyz": np.concatenate(directions).astype(np.float32),
        "seed_winding": np.concatenate(windings),
        "dr_per_winding": dr_per_winding,
        "z_range": np.array([z_lo, z_hi]),
        "seed_windings": np.array(sorted(mesh_paths)),
        "umbilicus_path": str(umbilicus_path),
        "meshes_dir": str(meshes_dir),
    }


def build_seed_rays(args, device="cuda:0"):
    """Reconstruct the fitted surface and return one ray per seed point.

    Returns dict of arrays: origin_xyz, direction_xyz, seed_winding, plus the
    ray/slab geometry scalars.
    """
    import torch

    spiral_dir = _spiral_scripts_dir()
    if spiral_dir not in sys.path:
        sys.path.insert(0, spiral_dir)
    from transforms import SpiralAndTransform
    from umbilicus import json_umbilicus_z_to_yx
    from sample_spiral import get_spiral_yxs

    checkpoint = _load_checkpoint_cfg(args.fit_checkpoint)
    cfg = dict(checkpoint["cfg"])
    for key in ("num_flow_integration_steps", "flow_integration_solver",
                "num_flow_timesteps", "num_flow_stages", "flow_bounds_z_margin",
                "flow_bounds_radius", "flow_voxel_resolution", "flow_field_type",
                "gap_expander_logit_resolution", "gap_expander_num_windings",
                "gap_expander_lr_scale", "linear_z_resolution",
                "initial_dr_per_winding"):
        cfg.setdefault(f"model_{key}", cfg[key])

    z_begin, z_end = int(checkpoint["z_begin"]), int(checkpoint["z_end"])
    z_lo, z_hi = args.z_range if args.z_range else (z_begin, z_end)
    z_lo, z_hi = max(z_lo, z_begin), min(z_hi, z_end)
    if z_lo >= z_hi:
        raise ValueError(f"empty z range [{z_lo}, {z_hi}) after clamping to the "
                         f"checkpoint's [{z_begin}, {z_end})")

    umbilicus_path = args.umbilicus
    if umbilicus_path is None:
        candidates = [Path(args.fit_checkpoint).parent / "umbilicus.json"]
        if os.environ.get("SPIRAL_DATASET"):
            candidates.insert(
                0, Path(os.environ["SPIRAL_DATASET"]) / "umbilicus.json")
        candidates += [p / "umbilicus.json"
                       for p in Path(args.fit_checkpoint).resolve().parents]
        umbilicus_path = next((str(c) for c in candidates if c.is_file()), None)
        if umbilicus_path is None:
            raise FileNotFoundError("pass --umbilicus or set SPIRAL_DATASET")

    device = torch.device(device)
    z_values = np.arange(z_begin, z_end)
    centre = json_umbilicus_z_to_yx(umbilicus_path, coordinate_scale=1.0)
    umbilicus_zyx = torch.from_numpy(np.concatenate(
        [z_values[:, None], centre(z_values)], axis=-1
    ).astype(np.float32)).to(device)
    radius = int(cfg["model_flow_bounds_radius"])
    margin = int(cfg["model_flow_bounds_z_margin"])
    model = SpiralAndTransform(
        # More steps than the fit used: the checkpoint's 3-step rk4 leaves the
        # forward and inverse maps mutually inconsistent by up to ~2 windings
        # (p95); at 24 steps the round trip closes to <0.5 voxel, and the
        # inference-mode cost is milliseconds.
        flow_integration_steps=max(
            24, int(cfg["model_num_flow_integration_steps"])),
        flow_integration_solver=str(cfg["model_flow_integration_solver"]),
        flow_min_corner_zyx=torch.tensor(
            [z_begin - margin, -radius, -radius], dtype=torch.int64,
            device=device),
        flow_max_corner_zyx=torch.tensor(
            [z_end + margin, radius, radius], dtype=torch.int64, device=device),
        umbilicus_zyx=umbilicus_zyx,
        config=cfg,
        spiral_outward_sense=str(
            checkpoint.get("spiral_outward_sense") or "CW"),
    ).to(device)
    model.load_state_dict(checkpoint["spiral_and_transform"])
    model.eval()

    if args.winding_range:
        first, last = args.winding_range
    else:
        first = int(cfg.get("output_first_winding", 10))
        last = int(cfg.get("shell_outer_winding_idx")
                   or int(cfg["model_gap_expander_num_windings"]) - 1)
    seed_windings = list(range(first, last + 1, args.winding_step))

    with torch.inference_mode():
        transform = model.get_slice_to_spiral_transform()
        dr_per_winding = float(model.get_dr_per_winding())
        yxs_by_winding = get_spiral_yxs(
            last + 1, model.get_dr_per_winding(), args.seed_spacing,
            group_by_winding=True, device=str(device))
        z_rows = torch.arange(z_lo, z_hi, args.seed_spacing,
                              dtype=torch.float32, device=device)

        # One (nz, ntheta, 3) scroll-space grid per seed winding; a single
        # batched inverse-transform call over all windings amortizes the ODE
        # integration overhead.
        spiral_chunks, layout = [], []
        for winding in seed_windings:
            yxs = yxs_by_winding[winding]
            grid = torch.cat([
                z_rows[:, None, None].expand(-1, yxs.shape[0], 1),
                yxs[None].expand(z_rows.shape[0], -1, 2),
            ], dim=-1)
            layout.append((winding, grid.shape[0], grid.shape[1]))
            spiral_chunks.append(grid.reshape(-1, 3))
        flat = torch.cat(spiral_chunks)
        print(f"[seed] inverse-transforming {flat.shape[0]:,} surface points "
              f"({len(seed_windings)} windings)", flush=True)
        pieces = [transform.inv(flat[i:i + TRANSFORM_CHUNK]).cpu()
                  for i in range(0, flat.shape[0], TRANSFORM_CHUNK)]
        scroll_flat = torch.cat(pieces).numpy().astype(np.float64)

    origins, directions, windings = [], [], []
    cursor = 0
    for winding, nz, ntheta in layout:
        grid = scroll_flat[cursor:cursor + nz * ntheta].reshape(nz, ntheta, 3)
        cursor += nz * ntheta
        # Tangents by central differences on the (z, theta) grid.
        tangent_z = np.gradient(grid, axis=0)
        tangent_theta = np.gradient(grid, axis=1)
        normal = np.cross(tangent_theta, tangent_z)
        norm = np.linalg.norm(normal, axis=-1, keepdims=True)
        # Orient outward: away from the umbilicus in the yx plane.
        z_idx = np.clip(grid[..., 0].astype(int) - z_begin, 0, z_end - z_begin - 1)
        umb_yx = centre(z_idx + z_begin)
        radial = grid[..., 1:] - umb_yx
        sign = np.sign(np.einsum("...c,...c", normal[..., 1:], radial))
        sign[sign == 0] = 1.0
        normal = normal * (sign / np.maximum(norm[..., 0], 1e-12))[..., None]

        keep = (
            np.isfinite(normal).all(-1)
            & (norm[..., 0] > 1e-6)
            & (grid[..., 0] >= z_lo) & (grid[..., 0] < z_hi)
        )
        points = grid[keep]
        normals = normal[keep]
        # zyx -> xyz for the slab extractor
        origins.append(points[:, ::-1])
        directions.append(normals[:, ::-1])
        windings.append(np.full(len(points), winding, dtype=np.int16))
        print(f"[seed] winding {winding}: {len(points)} rays", flush=True)

    return {
        "seed_xyz": np.concatenate(origins).astype(np.float32),
        "direction_xyz": np.concatenate(directions).astype(np.float32),
        "seed_winding": np.concatenate(windings),
        "dr_per_winding": dr_per_winding,
        "z_range": np.array([z_lo, z_hi]),
        "seed_windings": np.array(seed_windings),
        "umbilicus_path": str(umbilicus_path),
    }


# --------------------------------------------------------------------------
# Stage B: per-GPU inference workers
# --------------------------------------------------------------------------

def _selected_columns(columns, column_stride, step, args, *, margin=0):
    """Column indices after edge exclusion and optional center ownership."""
    selected = np.arange(int(margin), int(columns) - int(margin), int(step))
    center_width = getattr(args, "slab_center_width", None)
    if center_width is not None and len(selected):
        center = (int(columns) - 1) * float(column_stride) / 2.0
        physical = selected.astype(np.float64) * float(column_stride)
        selected = selected[
            np.abs(physical - center) <= float(center_width) / 2.0]
    return selected


def _decode_slab_phase(prob, phase, valid_cols, frame, ray_length,
                       column_stride, seed_winding, args,
                       phase_offset=0.0):
    """Vectorized phase-passage decode over every selected column at once.

    Passages are the integer levels of the registered phase: per segment
    (i, i+1] the level count is the floor difference (monotone phase), and
    each passage position interpolates linearly inside its segment —
    matching phase_passages column-for-column without the per-column
    Python loop, which dominated decode time on fine column grids.
    """
    columns = phase.shape[0]
    center = int(round((columns * column_stride - 1) / 2 / column_stride))
    anchor = int(round((ray_length - 1) / 2.0))
    if not valid_cols[center, center, anchor]:
        return None

    sel = _selected_columns(
        columns, column_stride, args.column_step, args)
    if not len(sel):
        return None
    if args.column_step == 1 and len(sel) == columns:
        sub_phase, sub_valid = phase, valid_cols
        sub_prob = prob
    else:
        index = np.ix_(sel, sel)
        sub_phase, sub_valid = phase[index], valid_cols[index]
        sub_prob = None if prob is None else prob[index]
    flat_shape = (len(sel) * len(sel), ray_length)
    # float32 throughout: the dense per-sample ops dominate the decode, and
    # float32 keeps positions accurate to ~1e-4 samples — far below any
    # physical scale here.
    registered = (
        np.ascontiguousarray(sub_phase, dtype=np.float32).reshape(flat_shape)
        - np.float32(phase[center, center, anchor])
        + np.float32(phase_offset)
    )
    valid_flat = sub_valid.reshape(flat_shape)
    prob_flat = None if prob is None else sub_prob.reshape(flat_shape)

    # Events per segment: the floor difference counts the integers passed.
    level_lo = np.floor(registered)
    counts = (level_lo[:, 1:] - level_lo[:, :-1]).astype(np.int32)
    np.clip(counts, 0, None, out=counts)
    segment_col, segment_idx = np.nonzero(counts)
    if not len(segment_col):
        return None
    reps = counts[segment_col, segment_idx].astype(np.int64)
    col = np.repeat(segment_col, reps)
    idx = np.repeat(segment_idx, reps)
    segment = np.repeat(np.arange(len(reps)), reps)
    first = np.cumsum(reps) - reps
    within = np.arange(reps.sum()) - np.repeat(first, reps)
    levels = np.repeat(level_lo[segment_col, segment_idx], reps) \
        + (within + 1.0).astype(np.float32)
    base = registered[col, idx]
    step = np.maximum(registered[col, idx + 1] - base, np.float32(1e-9))
    positions = idx + np.clip((levels - base) / step, 0.0, 1.0)

    keep = (positions >= EDGE_MARGIN) & (positions < ray_length - EDGE_MARGIN)
    col, positions, levels, segment = (
        col[keep], positions[keep], levels[keep], segment[keep])
    samples = np.rint(positions).astype(np.int64)
    keep = valid_flat[col, samples]
    col, positions, levels, samples, segment = (
        col[keep], positions[keep], levels[keep], samples[keep], segment[keep])
    if not len(col):
        return None
    if prob_flat is not None:
        confidence = prob_flat[col, samples]
    else:
        # Headless models need the passage kernel only at decoded crossings.
        # Reconstruct those values directly instead of materializing a dense
        # 125x125x384 CPU field.  passage_kernels renders the first integer
        # passage of every crossed segment; the nearest rendered passage can
        # only be this segment's, the preceding segment's, or the following
        # segment's passage because segments are ordered within each column.
        # Gather the sparse crossing segments before widening to float64.
        # Widening the full dense field allocated and copied ~48 MB per slab
        # on the full-resolution column grid for values that were never read.
        render_base = registered[segment_col, segment_idx].astype(np.float64)
        render_next = registered[segment_col, segment_idx + 1].astype(np.float64)
        render_step = np.maximum(
            render_next - render_base, 1e-9)
        render_position = segment_idx.astype(np.float64) + np.clip(
            (level_lo[segment_col, segment_idx].astype(np.float64) + 1.0
             - render_base) / render_step,
            0.0, 1.0,
        )
        previous = np.full(len(render_position), np.inf)
        following = np.full(len(render_position), np.inf)
        same_previous = segment_col[1:] == segment_col[:-1]
        previous[1:][same_previous] = render_position[:-1][same_previous]
        following[:-1][same_previous] = render_position[1:][same_previous]
        sample64 = samples.astype(np.float64)
        distance = np.minimum.reduce([
            np.abs(sample64 - render_position[segment]),
            np.abs(sample64 - previous[segment]),
            np.abs(sample64 - following[segment]),
        ])
        sigma = float(getattr(args, "passage_sigma_samples", 1.0))
        confidence = np.exp(-0.5 * (distance / sigma) ** 2).astype(np.float32)
    levels = np.rint(levels).astype(np.int64)
    strong = (confidence >= args.min_prob_keep) \
        & (np.abs(levels) <= args.max_level)
    col, positions, levels, confidence = (
        col[strong], positions[strong], levels[strong], confidence[strong])
    if not len(col):
        return None

    # Events were built column-major and every filter preserves order, so
    # they are already grouped by column with ascending positions.
    ijk = np.stack([
        (sel[col // len(sel)] * column_stride).astype(np.float64),
        (sel[col % len(sel)] * column_stride).astype(np.float64),
        positions,
    ], axis=-1)
    per_column = np.bincount(col, minlength=len(sel) * len(sel))
    offsets = np.r_[0, np.cumsum(per_column[per_column > 0])]
    return (
        frame.to_world(ijk).astype(np.float32),
        (seed_winding + levels).astype(np.int16),
        np.clip(confidence * 255, 0, 255).astype(np.uint8),
        offsets.astype(np.int64),
    )


def decode_slab(prob, phase, valid_cols, frame, ray_length, column_stride,
                seed_winding, slab_id, args, phase_offset=0.0):
    """Decode one slab's columns into absolute-winding crossing strips.

    ``valid_cols`` is the slab validity sampled at the column grid,
    [columns, columns, ray_length]. Returns (xyz f32 [N,3], winding i16 [N],
    prob u8 [N], offsets) or None when the seed anchor cannot be recovered.

    With ``args.phase_decode`` (models without a crossing head) crossings
    are the integer passages of the phase registered at the seed crossing:
    duplicate-free by construction, so no phase-level dedup, and decoded
    for all columns in one vectorized pass (_decode_slab_phase). ``prob``
    then holds unit-height passage kernels (_passage_prob), so per-crossing
    confidence is deliberately uniform (~1) and the ``min_prob_keep`` gate
    is inert; the seed anchor is taken at the ray midpoint directly (the
    seeder placed the seed crossing there), so seed validation lives with
    the seeder in this mode.
    """
    from vesuvius.neural_tracing.winding_models.winding_targets import (
        extract_peaks,
    )

    if bool(getattr(args, "phase_decode", False)):
        return _decode_slab_phase(
            prob, phase, valid_cols, frame, ray_length, column_stride,
            seed_winding, args, phase_offset=phase_offset)

    columns = prob.shape[0]
    center = int(round((prob.shape[0] * column_stride - 1) / 2 / column_stride))
    midpoint = (ray_length - 1) / 2.0

    def column_peaks(a, b):
        peaks = extract_peaks(prob[a, b], threshold=args.threshold,
                              min_distance=args.min_distance)
        peaks = peaks[(peaks >= EDGE_MARGIN) & (peaks < ray_length - EDGE_MARGIN)]
        return peaks[valid_cols[a, b, peaks]]

    # Anchor: the center column's peak nearest the ray midpoint is the
    # seed crossing and carries the seed's absolute winding index.
    anchor_peaks = column_peaks(center, center)
    if not len(anchor_peaks):
        return None
    anchor = anchor_peaks[np.argmin(np.abs(anchor_peaks - midpoint))]
    if abs(anchor - midpoint) > ANCHOR_TOLERANCE:
        return None
    phase_anchor = phase[center, center, anchor]

    def column_crossings(a, b):
        """(fractional positions, winding levels, confidences) for a column."""
        peaks = column_peaks(a, b)
        if len(peaks) < 1:
            return peaks.astype(np.float64), peaks.astype(int), peaks
        levels = np.rint(phase[a, b, peaks] - phase_anchor).astype(int)
        # Phase-level dedup: keep the strongest peak per winding level.
        keep = []
        for level in np.unique(levels):
            group = np.nonzero(levels == level)[0]
            keep.append(group[np.argmax(prob[a, b, peaks[group]])])
        keep = np.sort(np.asarray(keep))
        peaks, levels = peaks[keep], levels[keep]
        return peaks.astype(np.float64), levels, prob[a, b, peaks]

    xyz_out, winding_out, prob_out, offsets = [], [], [], [0]
    total = 0
    selected = _selected_columns(
        columns, column_stride, args.column_step, args)
    for a in selected:
        for b in selected:
            positions, levels, confidence = column_crossings(a, b)
            if len(positions) < 1:
                continue
            strong = (confidence >= args.min_prob_keep) \
                & (np.abs(levels) <= args.max_level)
            positions, levels, confidence = (
                positions[strong], levels[strong], confidence[strong])
            if len(positions) < 1:
                continue
            ijk = np.stack([
                np.full(len(positions), a * column_stride, dtype=np.float64),
                np.full(len(positions), b * column_stride, dtype=np.float64),
                positions,
            ], axis=-1)
            xyz_out.append(frame.to_world(ijk).astype(np.float32))
            winding_out.append((seed_winding + levels).astype(np.int16))
            prob_out.append(
                np.clip(confidence * 255, 0, 255).astype(np.uint8))
            total += len(positions)
            offsets.append(total)
    if not xyz_out:
        return None
    return (
        np.concatenate(xyz_out),
        np.concatenate(winding_out),
        np.concatenate(prob_out),
        np.asarray(offsets, dtype=np.int64),
    )


def _decoded_shard_paths(result_path):
    """Raw append-only arrays belonging to one GPU's result metadata."""
    prefix = Path(result_path).with_suffix("")
    return {
        name: Path(f"{prefix}.{name}.bin")
        for name in ("xyz", "winding", "prob", "strip_slab", "strip_length")
    }


class _DecodedShardWriter:
    """Bounded append-only storage for one GPU's decoded point stream.

    The former worker retained every array in Python lists and then built
    full-size concatenations for ``np.savez``.  Raw typed streams preserve the
    exact deterministic order, require no final worker-side copy, and can be
    memory-mapped in bounded blocks by finalization.
    """

    def __init__(self, result_path):
        self.paths = _decoded_shard_paths(result_path)
        self.handles = {
            name: open(path, "wb") for name, path in self.paths.items()
        }
        self.num_points = 0
        self.num_strips = 0

    def add(self, index, decoded):
        if decoded is None:
            return
        xyz, winding, prob, offsets = decoded
        lengths = np.diff(offsets).astype(np.uint32, copy=False)
        self.num_points += len(xyz)
        self.num_strips += len(lengths)
        np.ascontiguousarray(xyz, dtype=np.float32).tofile(self.handles["xyz"])
        np.ascontiguousarray(winding, dtype=np.int16).tofile(
            self.handles["winding"])
        np.ascontiguousarray(prob, dtype=np.uint8).tofile(
            self.handles["prob"])
        np.full(len(lengths), int(index), dtype=np.int64).tofile(
            self.handles["strip_slab"])
        lengths.tofile(self.handles["strip_length"])

    def close(self):
        for handle in self.handles.values():
            handle.close()


class _NativePhaseCacheWriter:
    """Write one GPU's native phase, validity, and geometric frames."""

    def __init__(self, group_path, gpu):
        import zarr

        group = zarr.open_group(str(group_path), mode="r+")
        name = f"shard_{gpu}"
        self.phase = group["phase"][name]
        self.valid = group["valid"][name]
        self.frame = group["frame"][name]
        self.available = group["available"][name]

    @staticmethod
    def _runs(indices):
        indices = np.asarray(indices, dtype=np.int64)
        starts = np.r_[0, np.flatnonzero(np.diff(indices) != 1) + 1]
        return zip(starts, np.r_[starts[1:], len(indices)])

    def add(self, indices, phase, valid, frames):
        indices = np.asarray(indices, dtype=np.int64)
        phase = np.asarray(phase, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool)
        frame_values = np.stack([
            np.stack([
                item.origin, item.axis_a, item.axis_b, item.direction,
            ]) for item in frames
        ]).astype(np.float64)
        for begin, end in self._runs(indices):
            destination = slice(int(indices[begin]), int(indices[end - 1]) + 1)
            source = slice(int(begin), int(end))
            self.phase[destination] = phase[source]
            self.valid[destination] = valid[source]
            self.frame[destination] = frame_values[source]
            # Availability is committed last, so interrupted cache writes are
            # never mistaken for complete slabs during reconstruction.
            self.available[destination] = True


class _NativePhaseCacheReader:
    """Random-access view of globally ordered, physically sharded phase data."""

    def __init__(self, group_path):
        import zarr

        self.group = zarr.open_group(str(group_path), mode="r")
        shards = list(self.group.attrs["phase_shards"])
        self.bounds = np.asarray(
            [int(shards[0]["lo"])] + [int(item["hi"]) for item in shards],
            dtype=np.int64)
        self.names = [str(item["name"]) for item in shards]

    def read(self, global_index):
        from vesuvius.neural_tracing.winding_models.volume_slab_extractor import (
            SlabFrame,
        )

        global_index = int(global_index)
        shard = int(np.searchsorted(
            self.bounds[1:], global_index, side="right"))
        local = global_index - int(self.bounds[shard])
        name = self.names[shard]
        if not bool(self.group["available"][name][local]):
            return None
        phase = np.asarray(self.group["phase"][name][local], dtype=np.float32)
        valid = np.asarray(self.group["valid"][name][local], dtype=bool)
        values = np.asarray(self.group["frame"][name][local], dtype=np.float64)
        spacing = float(self.group.attrs["spacing"])
        frame = SlabFrame(
            origin=values[0], axis_a=values[1], axis_b=values[2],
            direction=values[3], spacing=spacing)
        return phase, valid, frame


def gpu_worker(gpu, args, shard_path, result_path, progress_queue=None):
    """One inference process per GPU: extract slabs, run the model, decode."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from vesuvius.neural_tracing.winding_models.volume_slab_extractor import (
        VolumeSlabExtractor,
    )
    from vesuvius.neural_tracing.winding_models.winding_model import WindingModel

    shard = np.load(shard_path)
    seeds = shard["seed_xyz"]
    directions = shard["direction_xyz"]
    seed_windings = shard["seed_winding"]
    phase_offsets = (
        np.asarray(shard["phase_offset"], dtype=np.float32)
        if "phase_offset" in shard
        else np.zeros(len(seeds), dtype=np.float32))

    device = torch.device(f"cuda:{gpu}")
    phase_reader = None
    if getattr(args, "phase_cache", None):
        import zarr

        cache_attrs = zarr.open_group(args.phase_cache, mode="r").attrs
        model_cfg = {
            "ray_length": int(cache_attrs["ray_length"]),
            "transverse_size": int(cache_attrs["transverse_size"]),
            "column_stride": int(cache_attrs["column_stride"]),
            "spacing": float(cache_attrs["spacing"]),
            "sampling": str(cache_attrs["sampling"]),
            "crossing_sigma_wv": float(cache_attrs["crossing_sigma_wv"]),
        }
        model = None
        phase_reader = _NativePhaseCacheReader(args.phase_cache)
    else:
        checkpoint = torch.load(args.model_ckpt, map_location="cpu",
                                weights_only=False)
        model_cfg = checkpoint["config"]
        model = WindingModel(model_cfg.get("model"))
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        if args.compile:
            model = torch.compile(model)

    ray_length = int(model_cfg.get("ray_length", 384))
    transverse = int(model_cfg.get("transverse_size", 128))
    spacing = float(model_cfg.get("spacing", 1.0))
    column_stride = int(model_cfg.get("column_stride", 4))
    # Optional transverse interpolation onto a finer column grid; downstream
    # everything works at the effective (decode) stride.
    column_upsample = max(1, int(getattr(args, "column_upsample", 1)))
    if column_stride % column_upsample:
        raise ValueError(
            f"--column-upsample {column_upsample} must divide the model's "
            f"column stride {column_stride}")
    decode_stride = column_stride // column_upsample
    columns_out = (transverse // column_stride - 1) * column_upsample + 1

    reference = Path(args.reference_zarr)
    extractor = None
    if phase_reader is None:
        extractor = VolumeSlabExtractor(
            [VolumeSlabExtractor.scaled_volume_path(
                reference, args.volume_scale)],
            transverse_size=transverse,
            ray_length=ray_length,
            spacing=spacing,
            sampling=str(model_cfg.get("sampling", "trilinear")),
            tile_size=int(model_cfg.get("tile_size", 64)),
            cache_bytes=int(
                args.volume_cache_bytes
                if args.volume_cache_bytes is not None
                else model_cfg.get("volume_cache_bytes", 0)),
            io_threads=int(model_cfg.get("volume_io_threads", 1)),
            segment_to_volume_xyz=[
                VolumeSlabExtractor.load_segment_to_volume_transform(
                    reference, args.volume_scale, segment_downscale=1,
                    use_registration=False,
                )
            ],
        )

    midpoint_t = (ray_length - 1) / 2.0

    # A few locally stored chunks of these volumes are truncated; the
    # training dataloader skips samples that touch them and so do we.
    malformed_chunk = "decoded chunk byte size does not match full chunk shape"

    def extract(index):
        if phase_reader is not None:
            cached = phase_reader.read(int(shard["global_index"][index]))
            if cached is None:
                return index, None, None, None
            phase, valid, frame = cached
            return index, phase, valid, frame
        origin = seeds[index].astype(np.float64) \
            - midpoint_t * spacing * directions[index].astype(np.float64)
        try:
            return index, *extractor.extract(0, directions[index], origin)
        except RuntimeError as exc:
            if malformed_chunk not in str(exc):
                raise
            return index, None, None, None

    phase_only = bool(getattr(args, "native_phase_only", False))
    phase_writer = (
        _NativePhaseCacheWriter(args.output, gpu) if phase_only else None)
    decoded_writer = None if phase_only else _DecodedShardWriter(result_path)
    # Prob-volume partials spill straight to compressed per-output-chunk files:
    # neither this worker nor the merge stage holds the full record set.
    spill = None
    out_shape = None if phase_only else _output_shape(
        reference, args.output_downsample)
    phase_mode = getattr(args, "prob_combine", "mean")
    phase_prob = (
        not phase_only and getattr(args, "prob_volume", False)
        and phase_mode in ("phase", "phase-label"))
    if not phase_only and getattr(args, "prob_volume", False):
        if phase_mode == "phase-label":
            spill = _PhaseLabelSpillWriter(
                Path(result_path).parent / f"phase_label_spill_{gpu}",
                out_shape, _OUTPUT_CHUNK)
        elif phase_prob:
            spill = _PhaseSpillWriter(
                Path(result_path).parent / f"phase_spill_{gpu}",
                out_shape, _OUTPUT_CHUNK)
        else:
            spill = _ProbSpillWriter(
                Path(result_path).parent / f"prob_spill_{gpu}",
                out_shape, _OUTPUT_CHUNK)
    if spill is None:
        prob_accumulator = None
    elif phase_mode == "phase-label":
        prob_accumulator = _GpuPhaseLabelAccumulator(spill)
    elif phase_prob:
        prob_accumulator = _GpuPhaseAccumulator(spill)
    else:
        prob_accumulator = _GpuProbAccumulator(spill)
    winding_accumulator = None
    if not phase_only:
        winding_spill = _WindingAggregateSpillWriter(
            Path(result_path).parent / f"winding_spill_{gpu}",
            out_shape, _OUTPUT_CHUNK, args.output_downsample)
        winding_accumulator = _GpuWindingAccumulator(
            winding_spill, device, seed_windings)
    decoded_slabs = 0
    started = time.time()

    # The GPU forward and vectorized NumPy decode run in a pipeline. NumPy's
    # dense operations release the GIL, so threads scale well here and avoid
    # pickling a ~30 MB phase/valid pair into a process for every slab (and
    # pickling the multi-megabyte decoded result back again).
    from concurrent.futures import (
        FIRST_COMPLETED, wait,
    )

    futures = set()
    completed = {}
    next_result = 0
    max_pending = args.decode_workers * 6

    def merge_ready():
        nonlocal next_result
        while next_result in completed:
            decoded, records = completed.pop(next_result)
            if not phase_only:
                _merge_decoded(
                    decoded_writer, spill, winding_accumulator,
                    next_result, decoded, records)
            next_result += 1

    def drain(block):
        nonlocal futures
        done = {f for f in futures if f.done()}
        if block and not done and futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
        for future in done:
            index, decoded, records = future.result()
            completed[index] = (decoded, records)
        futures -= done
        merge_ready()

    def submit(decode_pool, batch):
        nonlocal decoded_slabs
        if phase_only:
            _probs, phases, _records = _forward_batch(
                batch, model, device, 1, args=None,
                column_stride=column_stride, out_shape=None)
            phase_writer.add(
                [item[0] for item in batch], phases,
                [item[2] for item in batch], [item[3] for item in batch])
            decoded_slabs += len(batch)
            if progress_queue is not None:
                progress_queue.put(len(batch))
            return
        if phase_reader is None:
            probs, phases, prob_records = _forward_batch(
                batch, model, device, column_upsample, args=args,
                column_stride=decode_stride, out_shape=out_shape,
                phase_offsets=[phase_offsets[item[0]] for item in batch],
                phase_seed_windings=[
                    seed_windings[item[0]] for item in batch])
        else:
            native_phase = torch.from_numpy(
                np.stack([item[1] for item in batch])).to(device)
            probs, phases, prob_records = _postprocess_model_fields(
                native_phase, None, batch, column_upsample, args=args,
                column_stride=decode_stride, out_shape=out_shape,
                phase_offsets=[phase_offsets[item[0]] for item in batch],
                phase_seed_windings=[
                    seed_windings[item[0]] for item in batch])
        if prob_accumulator is not None:
            prob_accumulator.add_batch(prob_records)
        for sample, (index, _, slab_valid, frame) in enumerate(batch):
            # The interpolated grid has (H - 1) * factor + 1 columns at the
            # decode stride, one short of a plain [::decode_stride] slice.
            valid_cols = np.ascontiguousarray(
                slab_valid[::decode_stride, ::decode_stride]
                [:columns_out, :columns_out])
            futures.add(decode_pool.submit(
                _decode_task, args, ray_length, decode_stride, int(index),
                int(seed_windings[index]),
                probs[sample] if probs is not None else None, phases[sample],
                valid_cols, frame, float(phase_offsets[index]),
                prob_records_done=prob_records is not None))
        decoded_slabs += len(batch)
        if progress_queue is not None:
            progress_queue.put(len(batch))
        while len(futures) > max_pending:
            drain(block=True)
        drain(block=False)

    with ThreadPoolExecutor(args.extract_threads) as pool, \
            ThreadPoolExecutor(args.decode_workers) as decode_pool, \
            torch.inference_mode():
        # Bounded extraction prefetch: each buffered slab is ~30 MB, so the
        # window must stay small — executor.map would submit every ray up
        # front and let extraction run hundreds of GB ahead of the consumer
        # while the model warms up.
        from collections import deque

        extract_window = deque()
        next_ray = 0

        def refill():
            nonlocal next_ray
            while (len(extract_window) < args.extract_threads * 3
                   and next_ray < len(seeds)):
                extract_window.append(pool.submit(extract, next_ray))
                next_ray += 1

        refill()
        pending = []
        skipped = 0
        while extract_window:
            item = extract_window.popleft().result()
            refill()
            if item[1] is None:  # slab touched a truncated volume chunk
                skipped += 1
                decoded_slabs += 1
                # Extraction is consumed in index order, so any earlier
                # decoded results can now be merged deterministically.
                completed[int(item[0])] = (None, None)
                merge_ready()
                if progress_queue is not None:
                    progress_queue.put(1)
                continue
            pending.append(item)
            if len(pending) < args.batch_size:
                continue
            batch, pending = pending, []
            submit(decode_pool, batch)
        if pending:
            submit(decode_pool, pending)
        while futures:
            drain(block=True)
        if prob_accumulator is not None:
            prob_accumulator.close()
        if winding_accumulator is not None:
            winding_accumulator.close()
        if spill is not None:
            spill.flush()
        if skipped:
            print(f"[gpu{gpu}] skipped {skipped} slabs on truncated volume "
                  "chunks", flush=True)

    compute_elapsed = time.time() - started
    if decoded_writer is not None:
        decoded_writer.close()
    # Large arrays are already in bounded append-only streams.  The NPZ is
    # metadata only, so the parent never reloads a multi-hundred-GB shard.
    np.savez(
        result_path,
        num_points=(0 if decoded_writer is None else decoded_writer.num_points),
        num_strips=(0 if decoded_writer is None else decoded_writer.num_strips),
        slabs=decoded_slabs,
        elapsed=compute_elapsed,
    )
    elapsed = time.time() - started
    print(f"[gpu{gpu}] done: {decoded_slabs} slabs in {elapsed:.1f}s "
          f"({decoded_slabs / max(elapsed, 1e-9):.2f} slabs/s)", flush=True)


def _output_shape(reference, output_downsample):
    """Shape of the downsampled output arrays for a reference zarr."""
    scale0 = VolumeSlabExtractorShape(Path(reference))
    down = int(output_downsample)
    return tuple(int(np.ceil(s / down)) for s in scale0)


def _initialize_native_phase_cache(args, rays, bounds, gpus, model_cfg):
    """Create the lossless, independently writable native-phase artifact."""
    import zarr
    from zarr.codecs import BloscCodec, BloscShuffle, PackBits

    group = zarr.open_group(args.output, mode="w")
    phase_group = group.create_group("phase")
    valid_group = group.create_group("valid")
    frame_group = group.create_group("frame")
    available_group = group.create_group("available")
    rays_group = group.create_group("rays")
    for key in ("seed_xyz", "direction_xyz", "seed_winding"):
        value = np.asarray(rays[key])
        first_chunk = max(1, min(len(value), 1 << 16))
        rays_group.create_array(
            key, data=value, chunks=(first_chunk, *value.shape[1:]))

    ray_length = int(model_cfg.get("ray_length", 384))
    transverse = int(model_cfg.get("transverse_size", 128))
    column_stride = int(model_cfg.get("column_stride", 4))
    native_columns = transverse // column_stride
    compressor = BloscCodec(
        cname="zstd", clevel=1, shuffle=BloscShuffle.shuffle)
    shards = []
    for slot, gpu in enumerate(gpus):
        lo, hi = int(bounds[slot]), int(bounds[slot + 1])
        count = hi - lo
        name = f"shard_{gpu}"
        # One slab per independently compressed chunk retains random access;
        # 32 chunks per physical shard keeps the cache to ~4,400 files rather
        # than one file per ray on the reported 139k-ray run.
        phase_group.create_array(
            name, shape=(count, native_columns, native_columns, ray_length),
            chunks=(1, native_columns, native_columns, ray_length),
            shards=(min(32, max(1, count)), native_columns, native_columns,
                    ray_length),
            dtype="float32", compressors=[compressor])
        valid_group.create_array(
            name, shape=(count, transverse, transverse, ray_length),
            chunks=(1, transverse, transverse, ray_length),
            shards=(min(32, max(1, count)), transverse, transverse, ray_length),
            dtype="bool", filters=[PackBits()], compressors=[compressor])
        compact_chunk = max(1, min(count, 1 << 12))
        frame_group.create_array(
            name, shape=(count, 4, 3), chunks=(compact_chunk, 4, 3),
            dtype="float64", compressors=[compressor])
        available_group.create_array(
            name, shape=(count,), chunks=(compact_chunk,), dtype="bool",
            filters=[PackBits()], compressors=[compressor], fill_value=False)
        shards.append({"name": name, "gpu": int(gpu), "lo": lo, "hi": hi})

    group.attrs.update({
        "artifact_type": "winding_native_phase_cache",
        "format_version": 1,
        "phase_dtype": "float32",
        "fit_checkpoint": str(Path(args.fit_checkpoint).resolve()),
        "model_ckpt": str(Path(args.model_ckpt).resolve()),
        "reference_zarr": str(Path(args.reference_zarr).resolve()),
        "volume_scale": int(args.volume_scale),
        "ray_length": ray_length,
        "transverse_size": transverse,
        "column_stride": column_stride,
        "native_columns": native_columns,
        "spacing": float(model_cfg.get("spacing", 1.0)),
        "sampling": str(model_cfg.get("sampling", "trilinear")),
        "crossing_sigma_wv": float(model_cfg.get("crossing_sigma_wv", 1.0)),
        "phase_shards": shards,
        "z_range": [int(value) for value in rays["z_range"]],
        "seed_windings": [int(value) for value in rays["seed_windings"]],
        "dr_per_winding": float(rays["dr_per_winding"]),
        "umbilicus": str(rays["umbilicus_path"]),
        "complete": False,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })


def _load_native_phase_cache(args):
    """Load cached rays and validate immutable model geometry."""
    import zarr

    group = zarr.open_group(str(args.phase_cache), mode="r")
    if group.attrs.get("artifact_type") != "winding_native_phase_cache":
        raise ValueError(f"not a native winding phase cache: {args.phase_cache}")
    if not bool(group.attrs.get("complete", False)):
        raise RuntimeError(
            f"native phase cache is incomplete: {args.phase_cache}")
    if Path(args.output).resolve() == Path(args.phase_cache).resolve():
        raise ValueError("reconstruction OUTPUT must differ from --phase-cache")
    expected_model = str(Path(args.model_ckpt).resolve())
    cached_model = str(group.attrs["model_ckpt"])
    allow_relocated = bool(getattr(
        args, "phase_cache_allow_relocated_inputs", False))
    if expected_model != cached_model and not allow_relocated:
        raise ValueError(
            f"phase cache model mismatch: {cached_model} != {expected_model}")
    expected_reference = str(Path(args.reference_zarr).resolve())
    cached_reference = str(group.attrs["reference_zarr"])
    if expected_reference != cached_reference and not allow_relocated:
        raise ValueError(
            "phase cache reference mismatch: "
            f"{cached_reference} != {expected_reference}")
    if allow_relocated and (
            expected_model != cached_model
            or expected_reference != cached_reference):
        print(
            "[warning] using relocated phase-cache inputs; cached neural "
            "geometry is authoritative and the supplied reference path is "
            "used only for output coordinates/shape", flush=True)
    if int(args.volume_scale) != int(group.attrs["volume_scale"]):
        raise ValueError(
            "phase cache volume-scale mismatch: "
            f"{group.attrs['volume_scale']} != {args.volume_scale}")
    rays_group = group["rays"]
    rays = {
        key: np.asarray(rays_group[key][:])
        for key in ("seed_xyz", "direction_xyz", "seed_winding")
    }
    rays.update({
        "z_range": np.asarray(group.attrs["z_range"]),
        "seed_windings": np.asarray(group.attrs["seed_windings"]),
        "dr_per_winding": float(group.attrs["dr_per_winding"]),
        "umbilicus_path": str(group.attrs["umbilicus"]),
    })
    return rays


def _subsample_cached_winding_sheets(rays, stride):
    """Keep regularly spaced anchor sheets while preserving both bounds."""
    stride = int(stride)
    if stride <= 1:
        return rays
    available = np.asarray(rays["seed_windings"])
    retained = available[::stride]
    if len(available) and (not len(retained) or retained[-1] != available[-1]):
        retained = np.r_[retained, available[-1]]
    keep = np.isin(rays["seed_winding"], retained)
    result = dict(rays)
    for key in ("seed_xyz", "direction_xyz", "seed_winding", "global_index"):
        if key in result:
            result[key] = np.asarray(result[key])[keep]
    result["seed_windings"] = retained
    return result


class _ProbSpillWriter:
    """Write exact partial prob-volume reductions by output chunk.

    Raw CPU-fallback observations are reduced at bounded intervals.  The
    normal CUDA path calls :meth:`add_aggregates` with a much larger reduction
    spanning adjacent batches.  Spill size therefore follows the union of
    covered voxels per interval instead of ``samples x slabs``.
    """

    def __init__(self, directory, out_shape, chunk, flush_records=1 << 22):
        self.directory = Path(directory)
        # Stale spills from an earlier run in the same scratch directory
        # would silently duplicate records: appends must start clean.
        if self.directory.exists():
            shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True)
        self.out_shape = np.asarray(out_shape, dtype=np.int64)
        self.chunk = int(chunk)
        self.ny_chunks = -(-int(out_shape[1]) // self.chunk)
        self.nx_chunks = -(-int(out_shape[2]) // self.chunk)
        self.flush_records = int(flush_records)
        self.buffers = {}
        self.buffered = 0

    def add(self, voxels_zyx, values):
        voxels = np.asarray(voxels_zyx, dtype=np.int64)
        values = np.asarray(values, dtype=np.uint8)
        inside = ((voxels >= 0) & (voxels < self.out_shape)).all(-1)
        if not inside.all():
            voxels, values = voxels[inside], values[inside]
        if not len(voxels):
            return
        records = np.empty(len(voxels), dtype=_PROB_RECORD)
        records["key"] = (voxels[:, 0] << 42) + (voxels[:, 1] << 21) + voxels[:, 2]
        records["val"] = values
        buckets = (
            ((voxels[:, 0] // self.chunk) * self.ny_chunks
             + voxels[:, 1] // self.chunk) * self.nx_chunks
            + voxels[:, 2] // self.chunk
        )
        order = np.argsort(buckets, kind="stable")
        buckets, records = buckets[order], records[order]
        starts = np.r_[0, np.flatnonzero(np.diff(buckets)) + 1]
        for start, end in zip(starts, np.r_[starts[1:], len(buckets)]):
            self.buffers.setdefault(int(buckets[start]), []).append(
                records[start:end]
            )
        self.buffered += len(records)
        if self.buffered >= self.flush_records:
            self.flush()

    def add_aggregates(self, keys, totals, counts, maxima, *, bucket_sorted=False):
        """Append already-reduced partial aggregates."""
        keys = np.asarray(keys, dtype=np.uint64)
        if not len(keys):
            return
        records = np.empty(len(keys), dtype=_PROB_AGG_RECORD)
        records["key"] = keys
        records["total"] = np.asarray(totals, dtype=np.uint64)
        records["count"] = np.asarray(counts, dtype=np.uint32)
        records["maximum"] = np.asarray(maxima, dtype=np.uint8)
        z = (keys >> 42).astype(np.int64)
        y = ((keys >> 21) & _PROB_KEY_MASK).astype(np.int64)
        x = (keys & _PROB_KEY_MASK).astype(np.int64)
        buckets = (((z // self.chunk) * self.ny_chunks + y // self.chunk)
                   * self.nx_chunks + x // self.chunk)
        if not bucket_sorted and len(buckets) > 1:
            order = np.argsort(buckets, kind="stable")
            buckets, records = buckets[order], records[order]
        starts = np.r_[0, np.flatnonzero(np.diff(buckets)) + 1]
        for start, end in zip(starts, np.r_[starts[1:], len(buckets)]):
            bucket = int(buckets[start])
            zy, bx = divmod(bucket, self.nx_chunks)
            bz, by = divmod(zy, self.ny_chunks)
            _append_prob_aggregates(
                self.directory / f"{bz:05d}_{by:05d}_{bx:05d}.rec",
                records[start:end])

    def add_sorted_aggregates(
        self, keys, totals, counts, maxima, bucket_ids, bucket_starts
    ):
        """Append aggregates using bucket runs already found on the GPU.

        The old sorted path reconstructed z/y/x, divided every key into a
        bucket, and allocated one full structured-record copy on the CPU.
        Full-resolution batches contain tens of millions of records, making
        that redundant pass several hundred MB per flush.  Building one
        bucket-sized structured array at a time keeps the exact spill format
        while avoiding both the pass and the large temporary.
        """
        keys = np.asarray(keys)
        totals = np.asarray(totals)
        counts = np.asarray(counts)
        maxima = np.asarray(maxima)
        bucket_ids = np.asarray(bucket_ids)
        starts = np.asarray(bucket_starts, dtype=np.int64)
        ends = np.r_[starts[1:], len(keys)]
        for bucket_value, start, end in zip(bucket_ids, starts, ends):
            records = np.empty(int(end - start), dtype=_PROB_AGG_RECORD)
            records["key"] = keys[start:end]
            records["total"] = totals[start:end]
            records["count"] = counts[start:end]
            records["maximum"] = maxima[start:end]
            bucket = int(bucket_value)
            zy, bx = divmod(bucket, self.nx_chunks)
            bz, by = divmod(zy, self.ny_chunks)
            _append_prob_aggregates(
                self.directory / f"{bz:05d}_{by:05d}_{bx:05d}.rec",
                records,
            )

    def flush(self):
        for bucket, pieces in self.buffers.items():
            raw = np.concatenate(pieces)
            order = np.argsort(raw["key"], kind="stable")
            keys, values = raw["key"][order], raw["val"][order]
            starts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1]
            aggregate = np.empty(len(starts), dtype=_PROB_AGG_RECORD)
            aggregate["key"] = keys[starts]
            aggregate["total"] = np.add.reduceat(
                values.astype(np.uint64), starts)
            aggregate["count"] = np.diff(
                np.r_[starts, len(keys)]).astype(np.uint32)
            aggregate["maximum"] = np.maximum.reduceat(values, starts)
            zy, bx = divmod(bucket, self.nx_chunks)
            bz, by = divmod(zy, self.ny_chunks)
            _append_prob_aggregates(
                self.directory / f"{bz:05d}_{by:05d}_{bx:05d}.rec",
                aggregate)
        self.buffers.clear()
        self.buffered = 0


class _PhaseSpillWriter:
    """Write bounded registered-phase aggregates by output chunk."""

    def __init__(self, directory, out_shape, chunk):
        self.directory = Path(directory)
        if self.directory.exists():
            shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True)
        self.out_shape = tuple(int(value) for value in out_shape)
        self.chunk = int(chunk)
        self.ny_chunks = -(-self.out_shape[1] // self.chunk)
        self.nx_chunks = -(-self.out_shape[2] // self.chunk)

    def add_sorted_aggregates(
        self, keys, cosine, sine, density, weight, count,
        bucket_ids, bucket_starts,
    ):
        keys = np.asarray(keys)
        cosine = np.asarray(cosine)
        sine = np.asarray(sine)
        density = np.asarray(density)
        weight = np.asarray(weight)
        count = np.asarray(count)
        starts = np.asarray(bucket_starts, dtype=np.int64)
        ends = np.r_[starts[1:], len(keys)]
        for bucket_value, start, end in zip(bucket_ids, starts, ends):
            records = np.empty(int(end - start), dtype=_PHASE_AGG_RECORD)
            records["key"] = keys[start:end]
            records["cosine"] = cosine[start:end]
            records["sine"] = sine[start:end]
            records["density"] = density[start:end]
            records["weight"] = weight[start:end]
            records["count"] = count[start:end]
            bucket = int(bucket_value)
            zy, bx = divmod(bucket, self.nx_chunks)
            bz, by = divmod(zy, self.ny_chunks)
            _append_phase_aggregates(
                self.directory / f"{bz:05d}_{by:05d}_{bx:05d}.rec",
                records,
            )

    def flush(self):
        """Compatibility with the legacy buffered spill writer."""


class _PhaseLabelSpillWriter:
    """Write synchronized ``(voxel, winding)`` phase aggregates by chunk."""

    def __init__(self, directory, out_shape, chunk):
        self.directory = Path(directory)
        if self.directory.exists():
            shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True)
        self.out_shape = tuple(int(value) for value in out_shape)
        self.chunk = int(chunk)
        self.ny_chunks = -(-self.out_shape[1] // self.chunk)
        self.nx_chunks = -(-self.out_shape[2] // self.chunk)

    def add_sorted_aggregates(
        self, keys, cosine, sine, density, weight, weight_sq, count,
        bucket_ids, bucket_starts,
    ):
        values = tuple(np.asarray(value) for value in (
            keys, cosine, sine, density, weight, weight_sq, count))
        starts = np.asarray(bucket_starts, dtype=np.int64)
        ends = np.r_[starts[1:], len(values[0])]
        for bucket_value, start, end in zip(bucket_ids, starts, ends):
            records = np.empty(
                int(end - start), dtype=_PHASE_LABEL_AGG_RECORD)
            for name, value in zip(
                ("key", "cosine", "sine", "density", "weight",
                 "weight_sq", "count"), values
            ):
                records[name] = value[start:end]
            bucket = int(bucket_value)
            zy, bx = divmod(bucket, self.nx_chunks)
            bz, by = divmod(zy, self.ny_chunks)
            _append_phase_label_aggregates(
                self.directory / f"{bz:05d}_{by:05d}_{bx:05d}.rec",
                records,
            )

    def flush(self):
        """Compatibility with the legacy buffered spill writer."""


def _phase_taper_torch(indices, low, high, width):
    """Raised-cosine weight inside inclusive retained bounds."""
    import torch

    width = int(width)
    if width <= 0:
        return torch.ones_like(indices, dtype=torch.float32)
    distance = torch.minimum(indices - int(low), int(high) - indices).float()
    fraction = ((distance + 1.0) / (width + 1.0)).clamp(0.0, 1.0)
    return torch.sin(fraction * (0.5 * torch.pi)).square()


def _phase_volume_records_cuda(phase, valid, frame, ray_length,
                               column_stride, args, out_shape,
                               phase_offset=0.0, seed_winding=0,
                               label_aware=False):
    """Project one slab's registered phase as one observation per voxel.

    Fractional phase is represented on the unit circle.  Multiple samples
    from this slab that round to the same world voxel are combined before the
    slab contributes, so rotated sampling grids cannot give one slab extra
    cross-slab weight.
    """
    import torch

    columns = phase.shape[0]
    column_margin = int(getattr(args, "prob_column_margin", 2))
    ray_margin = int(getattr(args, "prob_ray_margin", 32))
    step = int(args.prob_column_step or args.column_step)
    sel = torch.as_tensor(
        _selected_columns(
            columns, column_stride, step, args, margin=column_margin),
        dtype=torch.int64, device=phase.device)
    samples = torch.arange(
        ray_margin, ray_length - ray_margin,
        dtype=torch.int64, device=phase.device)
    if not len(sel) or not len(samples):
        return None

    selected_phase = phase.index_select(0, sel).index_select(1, sel)
    selected_valid = valid.index_select(0, sel).index_select(1, sel) \
        .index_select(2, samples).bool()
    center = int(round(
        (columns * int(column_stride) - 1) / 2 / int(column_stride)))
    anchor = int(round((ray_length - 1) / 2.0))
    registered = selected_phase.index_select(2, samples) \
        - phase[center, center, anchor] + float(phase_offset)

    max_level = getattr(args, "prob_phase_max_level", None)
    if max_level is None:
        max_level = float(args.max_level) + 0.5
    selected_valid &= registered.abs() <= float(max_level)
    if not bool(selected_valid.any()):
        return None

    # Density is windings per scale-0 voxel. Compute only the retained
    # columns/samples: building a full upsampled derivative field allocated
    # millions of values per slab that the crossing band immediately threw
    # away.
    previous = (samples - 1).clamp_min(0)
    following = (samples + 1).clamp_max(ray_length - 1)
    denominator = (following - previous).to(torch.float32) \
        * float(frame.spacing)
    density = (
        selected_phase.index_select(2, following)
        - selected_phase.index_select(2, previous)
    ).abs() / denominator[None, None, :]
    density.clamp_min_(1e-6)

    # A uint8 Gaussian is zero after rounding beyond ~3.53 sigma. Restricting
    # projection to four sigma therefore removes samples that cannot affect
    # the stored result, while retaining a little margin for cross-slab phase
    # reconciliation. This cuts the dominant world projection, key sort, and
    # spill traffic by the winding-gap / kernel-width ratio.
    band_sigma = float(getattr(args, "prob_phase_band_sigma", 4.0))
    if band_sigma > 0:
        residual = (registered - torch.round(registered)).abs()
        physical_distance = residual / density
        kernel_sigma = (
            float(getattr(args, "passage_sigma_samples", 1.0))
            * float(getattr(args, "model_spacing", frame.spacing)))
        selected_valid &= physical_distance <= band_sigma * kernel_sigma
        if not bool(selected_valid.any()):
            return None

    half_life = float(getattr(args, "prob_phase_level_half_life", 2.0))
    taper_width = int(getattr(args, "prob_phase_edge_taper", 8))
    column_taper = _phase_taper_torch(
        sel, int(sel[0]), int(sel[-1]), taper_width)
    ray_taper = _phase_taper_torch(
        samples, ray_margin, ray_length - ray_margin - 1, taper_width)
    selected = torch.nonzero(selected_valid, as_tuple=False)
    if not len(selected):
        return None
    ai, bi, ki = selected.unbind(1)
    retained_phase = registered[ai, bi, ki]
    retained_density = density[ai, bi, ki]
    sample_weight = torch.pow(
        torch.tensor(0.5, dtype=torch.float32, device=phase.device),
        retained_phase.abs().float() / half_life,
    ) * column_taper[ai] * column_taper[bi] * ray_taper[ki]
    keep_weight = sample_weight > 0
    ai, bi, ki, retained_phase, retained_density, sample_weight = (
        value[keep_weight] for value in (
            ai, bi, ki, retained_phase, retained_density, sample_weight))
    if not len(ai):
        return None

    i = sel[ai].to(torch.float64) * float(column_stride)
    j = sel[bi].to(torch.float64) * float(column_stride)
    k = samples[ki].to(torch.float64)
    origin = torch.as_tensor(
        frame.origin, dtype=torch.float64, device=phase.device)
    axis_a = torch.as_tensor(
        frame.axis_a, dtype=torch.float64, device=phase.device)
    axis_b = torch.as_tensor(
        frame.axis_b, dtype=torch.float64, device=phase.device)
    direction = torch.as_tensor(
        frame.direction, dtype=torch.float64, device=phase.device)
    spacing = float(frame.spacing)
    xyz = [
        torch.round((origin[d] + spacing * (
            i * axis_a[d] + j * axis_b[d] + k * direction[d]
        )) / int(args.output_downsample)).to(torch.int64)
        for d in range(3)
    ]
    shape = tuple(int(value) for value in out_shape)
    inside = torch.ones(len(i), dtype=torch.bool, device=phase.device)
    inside &= (xyz[2] >= 0) & (xyz[2] < shape[0])
    inside &= (xyz[1] >= 0) & (xyz[1] < shape[1])
    inside &= (xyz[0] >= 0) & (xyz[0] < shape[2])
    if not bool(inside.any()):
        return None

    if label_aware:
        linear = (
            (xyz[2] * int(shape[1]) + xyz[1]) * int(shape[2]) + xyz[0]
        )[inside]
        winding = (
            int(seed_winding) + torch.round(retained_phase[inside]).to(
                torch.int64))
        if bool(((winding < -32768) | (winding > 32767)).any()):
            raise ValueError("synchronized winding does not fit int16")
        key = linear * 65536 + (winding + 32768)
    else:
        key = ((xyz[2] << 42) + (xyz[1] << 21) + xyz[0])[inside]
    angle = retained_phase[inside].float() * (2.0 * torch.pi)
    sample_weight = sample_weight[inside].float()
    cosine = torch.cos(angle) * sample_weight
    sine = torch.sin(angle) * sample_weight
    density_weight = retained_density[inside].float() * sample_weight

    key, order = torch.sort(key)
    cosine, sine, density_weight, sample_weight = (
        cosine[order], sine[order], density_weight[order], sample_weight[order])
    unique, inverse = torch.unique_consecutive(key, return_inverse=True)
    cosine_sum = torch.zeros(
        len(unique), dtype=torch.float32, device=phase.device)
    sine_sum = torch.zeros_like(cosine_sum)
    density_sum = torch.zeros_like(cosine_sum)
    weight_sum = torch.zeros_like(cosine_sum)
    slab_weight = torch.zeros_like(cosine_sum)
    cosine_sum.scatter_add_(0, inverse, cosine)
    sine_sum.scatter_add_(0, inverse, sine)
    density_sum.scatter_add_(0, inverse, density_weight)
    weight_sum.scatter_add_(0, inverse, sample_weight)
    slab_weight.scatter_reduce_(
        0, inverse, sample_weight, reduce="amax", include_self=False)
    denominator = weight_sum.clamp_min(1e-12)
    # Normalize sampling multiplicity within this slab; slab_weight remains
    # the observation's cross-slab influence.
    cosine_out = cosine_sum / denominator * slab_weight
    sine_out = sine_sum / denominator * slab_weight
    density_out = density_sum / denominator * slab_weight
    count = torch.ones(len(unique), dtype=torch.int64, device=phase.device)
    if label_aware:
        return (
            unique, cosine_out, sine_out, density_out, slab_weight,
            slab_weight.square(), count,
        )
    return unique, cosine_out, sine_out, density_out, slab_weight, count


def _aggregate_phase_cuda(keys, cosine, sine, density, weight, count):
    """Reduce registered-phase sufficient statistics on CUDA."""
    import torch

    keys = torch.cat(keys)
    cosine = torch.cat(cosine)
    sine = torch.cat(sine)
    density = torch.cat(density)
    weight = torch.cat(weight)
    count = torch.cat(count)
    keys, order = torch.sort(keys)
    cosine, sine, density, weight, count = (
        cosine[order], sine[order], density[order], weight[order], count[order])
    unique, inverse = torch.unique_consecutive(keys, return_inverse=True)
    outputs = [
        torch.zeros(len(unique), dtype=torch.float32, device=keys.device)
        for _ in range(4)
    ]
    for output, values in zip(outputs, (cosine, sine, density, weight)):
        output.scatter_add_(0, inverse, values.float())
    out_count = torch.zeros(
        len(unique), dtype=torch.int64, device=keys.device)
    out_count.scatter_add_(0, inverse, count.to(torch.int64))
    return unique, *outputs, out_count


class _GpuPhaseAccumulator:
    """Bounded cross-batch registered-phase reduction and spill."""

    def __init__(self, writer, flush_records=_PROB_GPU_FLUSH_RECORDS):
        from concurrent.futures import ThreadPoolExecutor

        self.writer = writer
        self.flush_records = int(flush_records)
        self.parts = [[] for _ in range(6)]
        self.records = 0
        self._write_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="phase-spill")
        self._pending_write = None

    def add_batch(self, observations):
        observations = [item for item in observations if item is not None]
        if not observations:
            return
        batch = _aggregate_phase_cuda(*[
            [item[index] for item in observations] for index in range(6)
        ])
        for destination, value in zip(self.parts, batch):
            destination.append(value)
        self.records += len(batch[0])
        if self.records >= self.flush_records:
            self.flush()

    def flush(self):
        import torch

        if not self.records:
            return
        reduced = _aggregate_phase_cuda(*self.parts)
        keys = reduced[0]
        z = keys >> 42
        y = (keys >> 21) & _PROB_KEY_MASK
        x = keys & _PROB_KEY_MASK
        buckets = (((z // self.writer.chunk) * self.writer.ny_chunks
                    + y // self.writer.chunk) * self.writer.nx_chunks
                   + x // self.writer.chunk)
        order = torch.argsort(buckets, stable=True)
        buckets = buckets[order]
        reduced = tuple(value[order] for value in reduced)
        bucket_ids, bucket_counts = torch.unique_consecutive(
            buckets, return_counts=True)
        bucket_starts = torch.cumsum(bucket_counts, 0) - bucket_counts
        self._finish_pending_write()
        write_args = tuple(value.cpu().numpy() for value in reduced) + (
            bucket_ids.cpu().numpy(), bucket_starts.cpu().numpy())
        self._pending_write = self._write_pool.submit(
            self.writer.add_sorted_aggregates, *write_args)
        self.parts = [[] for _ in range(6)]
        self.records = 0

    def _finish_pending_write(self):
        if self._pending_write is not None:
            self._pending_write.result()
            self._pending_write = None

    def close(self):
        self.flush()
        self._finish_pending_write()
        self._write_pool.shutdown()


def _aggregate_phase_label_cuda(
    keys, cosine, sine, density, weight, weight_sq, count
):
    """Reduce synchronized labeled-phase statistics on CUDA."""
    import torch

    values = [torch.cat(parts) for parts in (
        keys, cosine, sine, density, weight, weight_sq, count)]
    keys_value = values[0]
    keys_value, order = torch.sort(keys_value)
    values = [keys_value] + [value[order] for value in values[1:]]
    unique, inverse = torch.unique_consecutive(
        keys_value, return_inverse=True)
    outputs = [
        torch.zeros(len(unique), dtype=torch.float32, device=keys_value.device)
        for _ in range(5)
    ]
    for output, value in zip(outputs, values[1:6]):
        output.scatter_add_(0, inverse, value.float())
    out_count = torch.zeros(
        len(unique), dtype=torch.int64, device=keys_value.device)
    out_count.scatter_add_(0, inverse, values[6].to(torch.int64))
    return unique, *outputs, out_count


class _GpuPhaseLabelAccumulator:
    """Bounded cross-batch ``(voxel, winding)`` phase reduction."""

    def __init__(self, writer, flush_records=_PROB_GPU_FLUSH_RECORDS):
        from concurrent.futures import ThreadPoolExecutor

        self.writer = writer
        self.flush_records = int(flush_records)
        self.parts = [[] for _ in range(7)]
        self.records = 0
        self._write_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="phase-label-spill")
        self._pending_write = None

    def add_batch(self, observations):
        observations = [item for item in observations if item is not None]
        if not observations:
            return
        batch = _aggregate_phase_label_cuda(*[
            [item[index] for item in observations] for index in range(7)
        ])
        for destination, value in zip(self.parts, batch):
            destination.append(value)
        self.records += len(batch[0])
        if self.records >= self.flush_records:
            self.flush()

    def flush(self):
        import torch

        if not self.records:
            return
        reduced = _aggregate_phase_label_cuda(*self.parts)
        keys = reduced[0]
        linear = keys >> 16
        plane = int(self.writer.out_shape[1]) * int(self.writer.out_shape[2])
        z = linear // plane
        remainder = linear % plane
        y = remainder // int(self.writer.out_shape[2])
        x = remainder % int(self.writer.out_shape[2])
        buckets = (((z // self.writer.chunk) * self.writer.ny_chunks
                    + y // self.writer.chunk) * self.writer.nx_chunks
                   + x // self.writer.chunk)
        order = torch.argsort(buckets, stable=True)
        buckets = buckets[order]
        reduced = tuple(value[order] for value in reduced)
        bucket_ids, bucket_counts = torch.unique_consecutive(
            buckets, return_counts=True)
        bucket_starts = torch.cumsum(bucket_counts, 0) - bucket_counts
        self._finish_pending_write()
        write_args = tuple(value.cpu().numpy() for value in reduced) + (
            bucket_ids.cpu().numpy(), bucket_starts.cpu().numpy())
        self._pending_write = self._write_pool.submit(
            self.writer.add_sorted_aggregates, *write_args)
        self.parts = [[] for _ in range(7)]
        self.records = 0

    def _finish_pending_write(self):
        if self._pending_write is not None:
            self._pending_write.result()
            self._pending_write = None

    def close(self):
        self.flush()
        self._finish_pending_write()
        self._write_pool.shutdown()


def _prob_volume_records(prob, valid_cols, frame, ray_length, column_stride,
                         args):
    """Quantized crossing-prob samples along the decoded column grid.

    ``valid_cols`` as in decode_slab. Returns (voxel zyx int32 [M, 3],
    value uint8 [M]) with one record (the slab's max) per output voxel this
    slab touches: one observation per overlapping slab, combined at merge
    time (--prob-combine).

    The slab's borders are excluded (--prob-ray-margin samples at each ray
    end, --prob-column-margin columns at each transverse edge): the model
    systematically over-predicts where its context is cut off, and border
    records paint those biases as plane artifacts cross-cutting the sheets.
    No probability floor applies here — the floor is a display threshold,
    applied to the combined value at merge time so the mean stays unbiased.
    """
    columns = prob.shape[0]
    column_margin = int(getattr(args, "prob_column_margin", 2))
    ray_margin = int(getattr(args, "prob_ray_margin", 32))
    sel = _selected_columns(
        columns, column_stride,
        args.prob_column_step or args.column_step, args,
        margin=column_margin)
    samples = np.arange(ray_margin, ray_length - ray_margin)
    if not len(sel) or not len(samples):
        return None
    transverse = sel * column_stride
    grid_i, grid_j, grid_k = np.meshgrid(
        transverse.astype(np.float64), transverse.astype(np.float64),
        samples.astype(np.float64), indexing="ij")
    ijk = np.stack([grid_i, grid_j, grid_k], axis=-1).reshape(-1, 3)

    values = prob[np.ix_(sel, sel, samples)].reshape(-1)
    keep = valid_cols[np.ix_(sel, sel, samples)].reshape(-1)
    if not keep.any():
        return None
    world = frame.to_world(ijk[keep])
    voxels = np.rint(world[:, ::-1] / args.output_downsample).astype(np.int64)
    values = np.clip(values[keep] * 255, 0, 255).astype(np.uint8)

    # Max per voxel within the slab: sort by (voxel key, value) and keep the
    # last record of each key run.
    key = (voxels[:, 0] << 42) + (voxels[:, 1] << 21) + voxels[:, 2]
    order = np.lexsort((values, key))
    key, voxels, values = key[order], voxels[order], values[order]
    last = np.ones(len(key), dtype=bool)
    last[:-1] = key[1:] != key[:-1]
    return voxels[last].astype(np.int32), values[last]


def _passage_prob(phase, sigma_samples):
    """Unit-height passage kernels; shared with the training visualization
    (winding_targets.passage_kernels)."""
    from vesuvius.neural_tracing.winding_models.winding_targets import (
        passage_kernels,
    )

    return passage_kernels(phase, sigma_samples)


def _passage_prob_torch(phase, sigma_samples):
    """CUDA equivalent of :func:`_passage_prob` for one phase field.

    Float64 is intentional: it mirrors ``passage_kernels`` before uint8
    quantization, preserving the stored probability volume exactly.
    """
    import torch

    phase = phase.double()
    level = torch.floor(phase)
    crossed = level[..., 1:] > level[..., :-1]
    step = torch.diff(phase, dim=-1).clamp_min(1e-9)
    fraction = ((level[..., :-1] + 1.0 - phase[..., :-1]) / step).clamp(0, 1)
    positions = torch.arange(
        phase.shape[-1] - 1, dtype=torch.float64, device=phase.device
    ) + fraction
    big = 1e9
    last = torch.cummax(torch.where(crossed, positions, -big), dim=-1).values
    ahead = torch.cummin(
        torch.where(crossed, positions, big).flip(-1), dim=-1
    ).values.flip(-1)
    pad_shape = phase.shape[:-1] + (1,)
    samples = torch.arange(
        phase.shape[-1], dtype=torch.float64, device=phase.device)
    forward = samples - torch.cat([
        torch.full(pad_shape, -big, dtype=torch.float64, device=phase.device),
        last,
    ], dim=-1)
    backward = torch.cat([
        ahead,
        torch.full(pad_shape, big, dtype=torch.float64, device=phase.device),
    ], dim=-1) - samples
    distance = torch.minimum(forward, backward).clamp_min(0)
    return torch.exp(
        -0.5 * (distance / float(sigma_samples)).square()).float()


def _prob_volume_records_cuda(prob, valid, frame, ray_length, column_stride,
                              args, out_shape):
    """GPU projection and within-slab max reduction, returned as packed keys.

    This is numerically equivalent to :func:`_prob_volume_records`: world
    coordinates and rounding use float64, values quantize to uint8 before
    lexicographic ``(key, value)`` reduction, and each slab contributes at
    most one (maximum) observation to an output voxel.
    """
    import torch

    columns = prob.shape[0]
    column_margin = int(getattr(args, "prob_column_margin", 2))
    ray_margin = int(getattr(args, "prob_ray_margin", 32))
    step = int(args.prob_column_step or args.column_step)
    sel = torch.as_tensor(
        _selected_columns(
            columns, column_stride, step, args, margin=column_margin),
        dtype=torch.int64, device=prob.device)
    samples = torch.arange(
        ray_margin, ray_length - ray_margin,
        dtype=torch.int64, device=prob.device)
    if not len(sel) or not len(samples):
        return None

    selected_valid = valid.index_select(0, sel).index_select(1, sel) \
        .index_select(2, samples)
    if not bool(selected_valid.any()):
        return None
    values = (prob.index_select(0, sel).index_select(1, sel)
              .index_select(2, samples) * 255).clamp(0, 255).to(torch.uint8)

    transverse = sel.to(torch.float64) * float(column_stride)
    sample64 = samples.to(torch.float64)
    i = transverse[:, None, None]
    j = transverse[None, :, None]
    k = sample64[None, None, :]
    origin = torch.as_tensor(frame.origin, dtype=torch.float64,
                             device=prob.device)
    axis_a = torch.as_tensor(frame.axis_a, dtype=torch.float64,
                             device=prob.device)
    axis_b = torch.as_tensor(frame.axis_b, dtype=torch.float64,
                             device=prob.device)
    direction = torch.as_tensor(frame.direction, dtype=torch.float64,
                                device=prob.device)
    spacing = float(frame.spacing)
    xyz = [
        torch.round((origin[d] + spacing * (
            i * axis_a[d] + j * axis_b[d] + k * direction[d]
        )) / int(args.output_downsample)).to(torch.int64)
        for d in range(3)
    ]
    shape = tuple(int(v) for v in out_shape)
    inside = selected_valid
    inside = inside & (xyz[2] >= 0) & (xyz[2] < shape[0])
    inside = inside & (xyz[1] >= 0) & (xyz[1] < shape[1])
    inside = inside & (xyz[0] >= 0) & (xyz[0] < shape[2])
    if not bool(inside.any()):
        return None
    key = (xyz[2] << 42) + (xyz[1] << 21) + xyz[0]
    key = key[inside]
    values = values[inside]
    key, order = torch.sort(key)
    values = values[order]
    unique, inverse = torch.unique_consecutive(key, return_inverse=True)
    maxima = torch.zeros(len(unique), dtype=torch.uint8, device=prob.device)
    maxima.scatter_reduce_(
        0, inverse, values, reduce="amax", include_self=False)
    return unique, maxima


def _aggregate_prob_cuda(keys, totals, counts, maxima):
    """Reduce sorted or unsorted partial aggregates exactly on CUDA."""
    import torch

    keys = torch.cat(keys)
    totals = torch.cat(totals)
    counts = torch.cat(counts)
    maxima = torch.cat(maxima)
    keys, order = torch.sort(keys)
    totals, counts, maxima = totals[order], counts[order], maxima[order]
    unique, inverse = torch.unique_consecutive(keys, return_inverse=True)
    out_total = torch.zeros(len(unique), dtype=torch.int64, device=keys.device)
    out_count = torch.zeros_like(out_total)
    out_max = torch.zeros_like(out_total)
    out_total.scatter_add_(0, inverse, totals.to(torch.int64))
    out_count.scatter_add_(0, inverse, counts.to(torch.int64))
    out_max.scatter_reduce_(
        0, inverse, maxima.to(torch.int64), reduce="amax", include_self=False)
    return unique, out_total, out_count, out_max.to(torch.uint8)


class _GpuProbAccumulator:
    """Bounded cross-batch CUDA reduction feeding a prob spill writer.

    Compression and filesystem writes run on one background thread.  At most
    one completed CPU flush is outstanding, so this overlaps spill I/O with
    subsequent extraction/model/decode work without unbounded host memory.
    """

    def __init__(self, writer, flush_records=_PROB_GPU_FLUSH_RECORDS):
        from concurrent.futures import ThreadPoolExecutor

        self.writer = writer
        self.flush_records = int(flush_records)
        self.keys, self.totals, self.counts, self.maxima = [], [], [], []
        self.records = 0
        self._write_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="prob-spill")
        self._pending_write = None

    def add_batch(self, observations):
        import torch

        observations = [item for item in observations if item is not None]
        if not observations:
            return
        keys = [item[0] for item in observations]
        values = [item[1] for item in observations]
        batch = _aggregate_prob_cuda(
            keys,
            [value.to(torch.int64) for value in values],
            [torch.ones_like(value, dtype=torch.int64) for value in values],
            values,
        )
        self.keys.append(batch[0])
        self.totals.append(batch[1])
        self.counts.append(batch[2])
        self.maxima.append(batch[3])
        self.records += len(batch[0])
        if self.records >= self.flush_records:
            self.flush()

    def flush(self):
        import torch

        if not self.records:
            return
        keys, totals, counts, maxima = _aggregate_prob_cuda(
            self.keys, self.totals, self.counts, self.maxima)
        z = keys >> 42
        y = (keys >> 21) & _PROB_KEY_MASK
        x = keys & _PROB_KEY_MASK
        buckets = (((z // self.writer.chunk) * self.writer.ny_chunks
                    + y // self.writer.chunk) * self.writer.nx_chunks
                   + x // self.writer.chunk)
        order = torch.argsort(buckets, stable=True)
        buckets = buckets[order]
        keys, totals, counts, maxima = (
            keys[order], totals[order], counts[order], maxima[order])
        bucket_ids, bucket_counts = torch.unique_consecutive(
            buckets, return_counts=True)
        bucket_starts = torch.cumsum(bucket_counts, 0) - bucket_counts

        # Bound host memory to one outstanding write. Waiting here still lets
        # the previous write overlap all batches accumulated since the last
        # flush, plus the CUDA reduction and bucket sort above.
        self._finish_pending_write()
        write_args = (
            keys.cpu().numpy(),
            totals.cpu().numpy(),
            counts.to(torch.int32).cpu().numpy(),
            maxima.cpu().numpy(),
            bucket_ids.cpu().numpy(),
            bucket_starts.cpu().numpy(),
        )
        self._pending_write = self._write_pool.submit(
            self.writer.add_sorted_aggregates, *write_args)
        self.keys.clear()
        self.totals.clear()
        self.counts.clear()
        self.maxima.clear()
        self.records = 0

    def _finish_pending_write(self):
        if self._pending_write is not None:
            self._pending_write.result()
            self._pending_write = None

    def close(self):
        """Flush all CUDA aggregates and wait for the bounded writer."""
        self.flush()
        self._finish_pending_write()
        self._write_pool.shutdown()


class _WindingAggregateSpillWriter:
    """Append GPU-reduced source-voxel winding votes by spatial chunk."""

    def __init__(self, directory, out_shape, chunk, output_downsample=1):
        self.directory = Path(directory)
        if self.directory.exists():
            shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True)
        self.out_shape = tuple(int(value) for value in out_shape)
        self.chunk = int(chunk)
        self.output_downsample = int(output_downsample)
        self.ny_chunks = -(-self.out_shape[1] // self.chunk)
        self.nx_chunks = -(-self.out_shape[2] // self.chunk)
        self.grid_shape = np.array([
            -(-size // self.chunk) for size in self.out_shape
        ], dtype=np.int64)

    def add_sorted(self, keys, totals, maxima, bucket_ids, bucket_starts):
        keys = np.asarray(keys)
        totals = np.asarray(totals)
        maxima = np.asarray(maxima)
        starts = np.asarray(bucket_starts, dtype=np.int64)
        ends = np.r_[starts[1:], len(keys)]
        for bucket_value, start, end in zip(bucket_ids, starts, ends):
            records = np.empty(int(end - start), dtype=_WINDING_AGG_RECORD)
            records["key"] = keys[start:end]
            records["total"] = totals[start:end]
            records["maximum"] = maxima[start:end]
            bucket = int(bucket_value)
            zy, bx = divmod(bucket, self.nx_chunks)
            bz, by = divmod(zy, self.ny_chunks)
            own = np.array([bz, by, bx], dtype=np.int64)
            linear = records["key"] >> np.uint64(16)
            plane = self.out_shape[1] * self.out_shape[2]
            z = (linear // np.uint64(plane)).astype(np.int64)
            remainder = linear % np.uint64(plane)
            y = (remainder // np.uint64(self.out_shape[2])).astype(np.int64)
            x = (remainder % np.uint64(self.out_shape[2])).astype(np.int64)
            local = np.stack([z, y, x], axis=-1) - own * self.chunk
            all_records = np.ones(len(records), dtype=bool)
            choices = [[
                (0, all_records),
                (-1, (own[axis] > 0) & (local[:, axis] == 0)),
                (1, (own[axis] + 1 < self.grid_shape[axis])
                 & (local[:, axis] == self.chunk - 1)),
            ] for axis in range(3)]
            from itertools import product

            # Copy only face/edge/corner records into adjacent target buckets.
            # The final raster task therefore reads one bucket, rather than
            # rereading 26 complete source buckets to obtain a one-voxel halo.
            for z_choice, y_choice, x_choice in product(*choices):
                delta = np.array([
                    z_choice[0], y_choice[0], x_choice[0]], dtype=np.int64)
                mask = z_choice[1] & y_choice[1] & x_choice[1]
                if not mask.any():
                    continue
                target = own + delta
                _append_winding_aggregates(
                    self.directory / (
                        f"{target[0]:05d}_{target[1]:05d}_"
                        f"{target[2]:05d}.rec"),
                    records[mask],
                )


def _aggregate_winding_cuda(keys, totals, maxima):
    """Reduce source ``(voxel, winding)`` vote aggregates on CUDA."""
    import torch

    keys = torch.cat(keys)
    totals = torch.cat(totals)
    maxima = torch.cat(maxima)
    keys, order = torch.sort(keys)
    totals, maxima = totals[order], maxima[order]
    unique, inverse = torch.unique_consecutive(keys, return_inverse=True)
    out_total = torch.zeros(
        len(unique), dtype=torch.float32, device=keys.device)
    out_max = torch.zeros(
        len(unique), dtype=torch.uint8, device=keys.device)
    out_total.scatter_add_(0, inverse, totals.float())
    out_max.scatter_reduce_(
        0, inverse, maxima, reduce="amax", include_self=False)
    return unique, out_total, out_max


class _GpuWindingAccumulator:
    """Reduce decoded observations before any neighborhood expansion.

    A full-resolution run emits many observations of the same physical source
    voxel. Keeping the 3x3x3 operation factored until finalization changes
    neither the vote kernel nor its candidate maxima. The required all-points
    archive is still written; rasterization simply no longer rereads it or
    creates 27 vote records per archived point.
    """

    def __init__(self, writer, device, seed_windings,
                 flush_records=_PROB_GPU_FLUSH_RECORDS):
        from concurrent.futures import ThreadPoolExecutor

        self.writer = writer
        self.device = device
        self.seed_windings = np.asarray(seed_windings)
        self.flush_records = int(flush_records)
        self.keys, self.totals, self.maxima = [], [], []
        self.records = 0
        self._write_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="winding-spill")
        self._pending_write = None

    def add(self, index, decoded):
        import torch

        if decoded is None:
            return
        xyz, winding, prob, _offsets = decoded
        xyz_t = torch.as_tensor(xyz, dtype=torch.float32, device=self.device)
        winding_t = torch.as_tensor(
            winding, dtype=torch.int64, device=self.device)
        prob_t = torch.as_tensor(prob, dtype=torch.uint8, device=self.device)
        voxels = torch.round(
            xyz_t[:, [2, 1, 0]] / int(self.writer.output_downsample)
        ).to(torch.int64)
        shape = self.writer.out_shape
        inside = (
            (voxels[:, 0] >= 0) & (voxels[:, 0] < shape[0])
            & (voxels[:, 1] >= 0) & (voxels[:, 1] < shape[1])
            & (voxels[:, 2] >= 0) & (voxels[:, 2] < shape[2])
        )
        voxels, winding_t, prob_t = (
            voxels[inside], winding_t[inside], prob_t[inside])
        if not len(voxels):
            return
        level = (winding_t - int(self.seed_windings[index])).abs().clamp_max(64)
        weight = prob_t.float() * torch.pow(
            torch.tensor(0.5, dtype=torch.float32, device=self.device),
            level.float() / _VOTE_LEVEL_HALF_LIFE,
        )
        linear = ((voxels[:, 0] * shape[1] + voxels[:, 1]) * shape[2]
                  + voxels[:, 2])
        key = linear * 65536 + (winding_t + 32768)
        # Keep raw tensors until the bounded cross-slab flush. Sorting every
        # slab separately did the same work twice and added ~139k tiny CUDA
        # sorts on the reported run.
        self.keys.append(key)
        self.totals.append(weight)
        self.maxima.append(prob_t)
        self.records += len(key)
        if self.records >= self.flush_records:
            self.flush()

    def flush(self):
        import torch

        if not self.records:
            return
        keys, totals, maxima = _aggregate_winding_cuda(
            self.keys, self.totals, self.maxima)
        linear = keys >> 16
        plane = self.writer.out_shape[1] * self.writer.out_shape[2]
        z = linear // plane
        remainder = linear % plane
        y = remainder // self.writer.out_shape[2]
        x = remainder % self.writer.out_shape[2]
        buckets = (((z // self.writer.chunk) * self.writer.ny_chunks
                    + y // self.writer.chunk) * self.writer.nx_chunks
                   + x // self.writer.chunk)
        order = torch.argsort(buckets, stable=True)
        buckets = buckets[order]
        keys, totals, maxima = keys[order], totals[order], maxima[order]
        bucket_ids, bucket_counts = torch.unique_consecutive(
            buckets, return_counts=True)
        bucket_starts = torch.cumsum(bucket_counts, 0) - bucket_counts
        self._finish_pending_write()
        write_args = (
            keys.cpu().numpy(), totals.cpu().numpy(), maxima.cpu().numpy(),
            bucket_ids.cpu().numpy(), bucket_starts.cpu().numpy(),
        )
        self._pending_write = self._write_pool.submit(
            self.writer.add_sorted, *write_args)
        self.keys.clear()
        self.totals.clear()
        self.maxima.clear()
        self.records = 0

    def _finish_pending_write(self):
        if self._pending_write is not None:
            self._pending_write.result()
            self._pending_write = None

    def close(self):
        self.flush()
        self._finish_pending_write()
        self._write_pool.shutdown()


def _upsample_columns(field, factor):
    """Transversely interpolate a [B, H, W, L] tensor onto a finer grid.

    With ``align_corners=True`` and output size ``(H - 1) * factor + 1``,
    output index k lands exactly at input position ``k / factor``: existing
    columns are reproduced bit-exactly and new columns interpolate linearly
    between them. Linear is near-exact for the phase field, whose measured
    transverse curvature at stride scale is ~0.003 windings; the ray axis
    is untouched.
    """
    import torch.nn.functional as F

    batch, height, width, length = field.shape
    factor = int(factor)
    return F.interpolate(
        field[:, None],
        size=((height - 1) * factor + 1, (width - 1) * factor + 1, length),
        mode="trilinear",
        align_corners=True,
    )[:, 0]


def _crop_native_fields_to_center(
        phase, prob_tensor, batch, column_upsample, column_stride, args):
    """Crop before interpolation while preserving the full-grid samples.

    Linear interpolation is local, so retaining the two native grid lines
    bracketing the requested center tile produces exactly the same values as
    upsampling the complete field and cropping afterward. Frames and validity
    masks are shifted with the crop so downstream world coordinates are
    unchanged.
    """
    center_width = getattr(args, "slab_center_width", None)
    if center_width is None:
        return phase, prob_tensor
    native_stride = int(column_stride) * int(column_upsample)
    columns = int(phase.shape[1])
    physical_center = (columns - 1) * native_stride / 2.0
    low = physical_center - float(center_width) / 2.0
    high = physical_center + float(center_width) / 2.0
    native_low = max(0, int(np.floor(low / native_stride)))
    native_high = min(columns - 1, int(np.ceil(high / native_stride)))
    if native_low == 0 and native_high == columns - 1:
        return phase, prob_tensor

    phase = phase[:, native_low:native_high + 1,
                  native_low:native_high + 1]
    if prob_tensor is not None:
        prob_tensor = prob_tensor[:, native_low:native_high + 1,
                                  native_low:native_high + 1]
    physical_low = native_low * native_stride
    physical_high = native_high * native_stride + 1
    adjusted = []
    for index, image, slab_valid, frame in batch:
        origin = np.asarray(frame.origin) + float(frame.spacing) * physical_low \
            * (np.asarray(frame.axis_a) + np.asarray(frame.axis_b))
        local_frame = type(frame)(
            origin=origin,
            axis_a=frame.axis_a,
            axis_b=frame.axis_b,
            direction=frame.direction,
            spacing=frame.spacing,
        )
        adjusted.append((
            index, image,
            slab_valid[physical_low:physical_high,
                       physical_low:physical_high],
            local_frame,
        ))
    batch[:] = adjusted
    return phase, prob_tensor


def _postprocess_model_fields(
    phase, prob_tensor, batch, column_upsample=1, *, args=None,
    column_stride=None, out_shape=None, phase_offsets=None,
    phase_seed_windings=None,
):
    """Interpolate and derive dense products from native model fields."""
    import torch

    phase = phase.float()
    if args is not None:
        phase, prob_tensor = _crop_native_fields_to_center(
            phase, prob_tensor, batch, column_upsample,
            column_stride, args)
    valid = torch.from_numpy(np.stack([b[2] for b in batch])).to(phase.device)
    if column_upsample > 1:
        phase = _upsample_columns(phase, column_upsample)
    if prob_tensor is not None:
        prob_tensor = torch.sigmoid(prob_tensor.float())
        if column_upsample > 1:
            prob_tensor = _upsample_columns(prob_tensor, column_upsample)
        probs = prob_tensor.cpu().numpy()
    else:
        # Headless strip decoding computes passage confidence only at its
        # sparse events. A dense passage field is needed solely for the
        # optional probability volume and is rendered on the GPU below.
        probs = None
    records = None
    if args is not None and args.prob_volume:
        records = []
        for sample, (_, _, _slab_valid, frame) in enumerate(batch):
            phase_offset = (
                0.0 if phase_offsets is None else float(phase_offsets[sample]))
            phase_mode = getattr(args, "prob_combine", "mean")
            if phase_mode in ("phase", "phase-label"):
                records.append(_phase_volume_records_cuda(
                    phase[sample], valid[sample].bool(), frame,
                    phase.shape[-1], int(column_stride), args, out_shape,
                    phase_offset=phase_offset,
                    seed_winding=(
                        0 if phase_seed_windings is None
                        else int(phase_seed_windings[sample])),
                    label_aware=phase_mode == "phase-label"))
                continue
            if prob_tensor is None:
                center = int(round(
                    (phase.shape[1] * int(column_stride) - 1)
                    / 2 / int(column_stride)))
                anchor = int(round((phase.shape[-1] - 1) / 2.0))
                sample_prob = _passage_prob_torch(
                    phase[sample] - phase[sample, center, center, anchor]
                    + phase_offset,
                    float(getattr(args, "passage_sigma_samples", 1.0)))
            else:
                sample_prob = prob_tensor[sample]
            records.append(_prob_volume_records_cuda(
                sample_prob, valid[sample].bool(), frame, phase.shape[-1],
                int(column_stride), args, out_shape))
    phases = phase.cpu().numpy()
    return probs, phases, records


def _forward_batch(batch, model, device, column_upsample=1, *, args=None,
                   column_stride=None, out_shape=None, phase_offsets=None,
                   phase_seed_windings=None):
    import torch

    images = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
    valid = torch.from_numpy(np.stack([b[2] for b in batch])).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(images, valid.bool())
    return _postprocess_model_fields(
        out["phase"], out.get("crossing_logits"), batch, column_upsample,
        args=args, column_stride=column_stride, out_shape=out_shape,
        phase_offsets=phase_offsets,
        phase_seed_windings=phase_seed_windings)


def _decode_task(args, ray_length, column_stride, index, seed_winding, prob,
                 phase, valid_cols, frame, phase_offset=0.0,
                 prob_records_done=False):
    """Decode-pool entry point: strips plus optional prob-volume records."""
    if prob is None and args.prob_volume and not prob_records_done:
        # Headless model: render integer-passage kernels as the prob field.
        # The slab's phase is registered at the seed crossing (the ray
        # midpoint sample of the center column, where the seeder placed it)
        # so integer levels sit on predicted sheet positions — the same
        # registration decode_slab anchors its winding levels to.
        columns = phase.shape[0]
        center = int(round((columns * column_stride - 1) / 2 / column_stride))
        anchor = int(round((ray_length - 1) / 2.0))
        prob = _passage_prob(
            phase - phase[center, center, anchor] + float(phase_offset),
            float(getattr(args, "passage_sigma_samples", 1.0)),
        )
    records = None
    if args.prob_volume and not prob_records_done:
        records = _prob_volume_records(
            prob, valid_cols, frame, ray_length, column_stride, args)
    decoded = decode_slab(
        prob, phase, valid_cols, frame, ray_length, column_stride,
        seed_winding, index, args, phase_offset=phase_offset)
    return index, decoded, records


def _merge_decoded(writer, spill, winding_accumulator,
                   index, decoded, records):
    if records is not None and spill is not None:
        spill.add(records[0], records[1])
    winding_accumulator.add(index, decoded)
    writer.add(index, decoded)


# --------------------------------------------------------------------------
# Stage C: merge worker shards into the output zarr
# --------------------------------------------------------------------------

# Vote weight halves for every this many windings of distance between an
# observation and its slab's seed anchor (winding-count accuracy decays with
# that distance), and neighbouring-voxel votes carry this fraction of the
# observation's weight.
_VOTE_LEVEL_HALF_LIFE = 2.0
_VOTE_NEIGHBOR_WEIGHT = 0.5


def _render_winding_chunk(v, winding, prob, level, lo, shape):
    """Render one chunk from ordered nearby observations.

    Both the in-memory reference rasterizer and the bounded spill rasterizer
    use this helper, keeping candidate reduction, tie-breaking, confidence,
    and sparse-write semantics in one implementation.
    """
    if not len(v):
        return None
    weight = prob.astype(np.float32) * np.float32(0.5) ** (
        np.minimum(level, 64).astype(np.float32) / _VOTE_LEVEL_HALF_LIFE)
    offsets = np.array(
        np.meshgrid(*([(-1, 0, 1)] * 3), indexing="ij"), dtype=np.int64
    ).reshape(3, -1).T
    offset_weight = np.where(
        (offsets == 0).all(-1), 1.0, _VOTE_NEIGHBOR_WEIGHT
    ).astype(np.float32)

    spread = len(offsets)
    position = (v[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    vote_weight = (weight[:, None] * offset_weight[None, :]).reshape(-1)
    vote_cand = np.repeat(winding.astype(np.int64), spread)
    vote_prob = np.repeat(prob, spread)
    hi = lo + shape
    in_column = ((position >= lo) & (position < hi)).all(-1)
    position, vote_weight, vote_cand, vote_prob = (
        position[in_column], vote_weight[in_column],
        vote_cand[in_column], vote_prob[in_column])

    local = position - lo
    flat = (local[:, 0] * shape[1] + local[:, 1]) * shape[2] + local[:, 2]
    vote_key = flat * 65536 + (vote_cand + 32768)
    srt = np.argsort(vote_key, kind="stable")
    vote_key, vote_weight, vote_prob = (
        vote_key[srt], vote_weight[srt], vote_prob[srt])
    run = np.r_[0, np.flatnonzero(np.diff(vote_key)) + 1]
    totals = np.add.reduceat(vote_weight, run)
    best_prob = np.maximum.reduceat(vote_prob, run)
    run_flat = vote_key[run] >> 16
    run_cand = (vote_key[run] & 0xFFFF) - 32768

    winner_order = np.lexsort((totals, run_flat))
    run_flat, run_cand, totals, best_prob = (
        run_flat[winner_order], run_cand[winner_order],
        totals[winner_order], best_prob[winner_order])
    voxel_start = np.r_[0, np.flatnonzero(np.diff(run_flat)) + 1]
    voxel_total = np.add.reduceat(totals, voxel_start)
    last = np.r_[run_flat[1:] != run_flat[:-1], True]
    win_flat, win_cand = run_flat[last], run_cand[last]
    win_weight, win_prob = totals[last], best_prob[last]

    core = ((v >= lo) & (v < hi)).all(-1)
    core_local = v[core] - lo
    occupied = np.unique(
        (core_local[:, 0] * shape[1] + core_local[:, 1]) * shape[2]
        + core_local[:, 2])
    if not len(occupied):
        return None
    slot = np.searchsorted(occupied, win_flat)
    keep = (slot < len(occupied)) & (
        occupied[np.minimum(slot, len(occupied) - 1)] == win_flat)
    win_flat, win_cand, win_weight, win_prob, voxel_total = (
        win_flat[keep], win_cand[keep], win_weight[keep],
        win_prob[keep], voxel_total[keep])
    if not len(win_flat):
        return None

    dense_winding = np.full(shape, -1, dtype=np.int16)
    dense_confidence = np.zeros(shape, dtype=np.uint8)
    zz = win_flat // (shape[1] * shape[2])
    rest = win_flat % (shape[1] * shape[2])
    yy, xx = rest // shape[2], rest % shape[2]
    dense_winding[zz, yy, xx] = win_cand.astype(np.int16)
    dense_confidence[zz, yy, xx] = np.clip(np.rint(
        win_prob.astype(np.float64) * win_weight
        / np.maximum(voxel_total, 1e-9)), 0, 255).astype(np.uint8)
    return dense_winding, dense_confidence


def _rasterize_winding_votes(group_path, out_shape, chunk, voxels, winding,
                             prob, level, workers=None):
    """Neighborhood-vote rasterization of the decoded crossings.

    Every observation votes for its winding in the 3^3 voxels around it
    (full weight on its own voxel, ``_VOTE_NEIGHBOR_WEIGHT`` on the 26
    neighbours — observations of the same physical crossing from different
    slabs land within a voxel of each other), weighted by its confidence
    and halved per ``_VOTE_LEVEL_HALF_LIFE`` windings of anchor distance.
    The old scatter kept the single highest-prob observation, which let one
    confident far-from-anchor miscount overwrite near-anchor consensus
    (measured: 16% of contested voxels, median anchor distance 5).

    A voxel is written only where at least one observation itself lands, so
    the raster keeps its old sparsity. ``winding`` gets the top-weighted
    candidate; ``confidence`` becomes vote share x the winner's best
    observation prob — cross-slab agreement scaled by peak strength, which
    reduces to the old per-point prob where only one slab observed.

    Output chunks are processed one at a time with observations gathered
    from all 26 neighbouring chunks too, so votes pool correctly across chunk
    borders, memory stays tightly bounded, and no chunk is written twice.
    """
    import zarr
    from tqdm import tqdm

    out_shape = np.asarray(out_shape, dtype=np.int64)
    inside = ((voxels >= 0) & (voxels < out_shape)).all(-1)
    voxels, winding, prob, level = (
        voxels[inside], winding[inside], prob[inside], level[inside])
    if not len(voxels):
        return
    nz = -(-int(out_shape[0]) // chunk)
    ny = -(-int(out_shape[1]) // chunk)
    nx = -(-int(out_shape[2]) // chunk)
    column_of = (((voxels[:, 0] // chunk) * ny + voxels[:, 1] // chunk) * nx
                 + voxels[:, 2] // chunk)
    order = np.argsort(column_of, kind="stable")
    voxels, winding, prob, level = (
        voxels[order], winding[order], prob[order], level[order])

    columns, starts = np.unique(column_of[order], return_index=True)
    ends = np.r_[starts[1:], len(order)]
    ranges = {
        int(column): (int(start), int(end))
        for column, start, end in zip(columns, starts, ends)
    }
    group = zarr.open_group(str(group_path), mode="r+")
    winding_arr = group["winding"]
    confidence_arr = group["confidence"]

    def write_column(column):
        zy, bx = divmod(int(column), nx)
        bz, by = divmod(zy, ny)
        lo = np.array([bz * chunk, by * chunk, bx * chunk])
        hi = np.minimum(lo + chunk, out_shape)
        shape = hi - lo

        pieces = [
            np.arange(*ranges[
                ((bz + dz) * ny + (by + dy)) * nx + (bx + dx)])
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if 0 <= bz + dz < nz and 0 <= by + dy < ny
            and 0 <= bx + dx < nx
            and ((bz + dz) * ny + (by + dy)) * nx + (bx + dx) in ranges
        ]
        idx = np.concatenate(pieces)
        v = voxels[idx]
        near = ((v >= lo - 1) & (v < hi + 1)).all(-1)
        idx, v = idx[near], v[near]
        if not len(idx):
            return
        rendered = _render_winding_chunk(
            v, winding[idx], prob[idx], level[idx], lo, shape)
        if rendered is None:
            return
        dense_winding, dense_confidence = rendered
        winding_arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = dense_winding
        confidence_arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = \
            dense_confidence

    from concurrent.futures import ThreadPoolExecutor

    workers = min(workers or min(8, os.cpu_count() or 1), len(columns))
    progress = lambda it: tqdm(  # noqa: E731
        it, total=len(columns), desc="writing winding", unit="chunk")
    if workers <= 1:
        for _ in progress(map(write_column, columns)):
            pass
    else:
        with ThreadPoolExecutor(workers) as pool:
            for _ in progress(pool.map(write_column, columns)):
                pass


class _WindingSpillWriter:
    """Spatially partition decoded observations for bounded rasterization.

    Records normally go only to their own output chunk.  An observation on a
    chunk face/edge/corner is additionally copied to the adjacent target
    chunks reached by its 3x3x3 neighborhood.  Consequently each raster task
    reads one file rather than rereading all 26 neighboring files.
    """

    def __init__(self, directory, out_shape, chunk):
        self.directory = Path(directory)
        if self.directory.exists():
            shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True)
        self.out_shape = np.asarray(out_shape, dtype=np.int64)
        self.chunk = int(chunk)
        self.grid_shape = np.array([
            -(-int(size) // self.chunk) for size in out_shape
        ], dtype=np.int64)

    def add(self, voxels, winding, prob, level, point_order):
        voxels = np.asarray(voxels, dtype=np.int64)
        winding = np.asarray(winding, dtype=np.int16)
        prob = np.asarray(prob, dtype=np.uint8)
        level = np.asarray(level, dtype=np.uint8)
        point_order = np.asarray(point_order, dtype=np.uint64)
        inside = ((voxels >= 0) & (voxels < self.out_shape)).all(-1)
        if not inside.all():
            voxels, winding, prob, level, point_order = (
                value[inside]
                for value in (voxels, winding, prob, level, point_order)
            )
        if not len(voxels):
            return

        own = voxels // self.chunk
        local = voxels - own * self.chunk
        all_records = np.ones(len(voxels), dtype=bool)
        choices = []
        for axis in range(3):
            choices.append([
                (0, all_records),
                (-1, (own[:, axis] > 0) & (local[:, axis] == 0)),
                (1, (own[:, axis] + 1 < self.grid_shape[axis])
                 & (local[:, axis] == self.chunk - 1)),
            ])

        from itertools import product

        ny, nx = int(self.grid_shape[1]), int(self.grid_shape[2])
        for z_choice, y_choice, x_choice in product(*choices):
            dz, mz = z_choice
            dy, my = y_choice
            dx, mx = x_choice
            mask = mz & my & mx
            if not mask.any():
                continue
            target = own[mask] + np.array([dz, dy, dx], dtype=np.int64)
            buckets = (target[:, 0] * ny + target[:, 1]) * nx + target[:, 2]
            order = np.argsort(buckets, kind="stable")
            buckets = buckets[order]
            selected = np.flatnonzero(mask)[order]
            starts = np.r_[0, np.flatnonzero(np.diff(buckets)) + 1]
            for start, end in zip(starts, np.r_[starts[1:], len(buckets)]):
                idx = selected[start:end]
                records = np.empty(len(idx), dtype=_WINDING_RECORD)
                records["key"] = (
                    (voxels[idx, 0].astype(np.uint64) << np.uint64(42))
                    + (voxels[idx, 1].astype(np.uint64) << np.uint64(21))
                    + voxels[idx, 2].astype(np.uint64)
                )
                records["order"] = point_order[idx]
                records["winding"] = winding[idx]
                records["prob"] = prob[idx]
                records["level"] = level[idx]
                bucket = int(buckets[start])
                zy, bx = divmod(bucket, nx)
                bz, by = divmod(zy, ny)
                _append_winding_records(
                    self.directory / f"{bz:05d}_{by:05d}_{bx:05d}.rec",
                    records,
                )


def _write_winding_bucket(task):
    """Rasterize one output chunk from its spatial observation spill."""
    import zarr

    group_path, out_shape, chunk, bz, by, bx, path = task
    records = np.concatenate(list(_iter_winding_records(path)))
    key = records["key"]
    voxels = np.stack([
        key >> np.uint64(42),
        (key >> np.uint64(21)) & np.uint64(_PROB_KEY_MASK),
        key & np.uint64(_PROB_KEY_MASK),
    ], axis=-1).astype(np.int64)

    # Match _rasterize_winding_votes exactly: source chunks are visited in
    # z/y/x order, and observations retain their original global point order
    # within each source chunk before the stable candidate reduction.
    out_shape = np.asarray(out_shape, dtype=np.int64)
    ny = -(-int(out_shape[1]) // chunk)
    nx = -(-int(out_shape[2]) // chunk)
    source = (((voxels[:, 0] // chunk) * ny + voxels[:, 1] // chunk) * nx
              + voxels[:, 2] // chunk)
    order = np.lexsort((records["order"], source))
    voxels, records = voxels[order], records[order]

    lo = np.array([bz * chunk, by * chunk, bx * chunk], dtype=np.int64)
    hi = np.minimum(lo + chunk, out_shape)
    shape = hi - lo
    near = ((voxels >= lo - 1) & (voxels < hi + 1)).all(-1)
    v = voxels[near]
    winding = records["winding"][near]
    prob = records["prob"][near]
    level = records["level"][near]
    if not len(v):
        return 0
    rendered = _render_winding_chunk(
        v, winding, prob, level, lo, shape)
    if rendered is None:
        return 0
    dense_winding, dense_confidence = rendered
    group = zarr.open_group(group_path, mode="r+")
    group["winding"][
        lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]
    ] = dense_winding
    group["confidence"][
        lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]
    ] = dense_confidence
    return int((dense_winding >= 0).sum())


def _rasterize_winding_spill(group_path, directory, out_shape, chunk,
                              workers=None):
    """Rasterize spatial spills independently, one bounded chunk per task."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    tasks = []
    for path in sorted(Path(directory).glob("*.rec")):
        bz, by, bx = (int(part) for part in path.stem.split("_"))
        tasks.append((str(group_path), tuple(out_shape), int(chunk),
                      bz, by, bx, str(path)))
    if not tasks:
        return
    workers = min(workers or min(8, os.cpu_count() or 1), len(tasks))
    progress = lambda it: tqdm(  # noqa: E731
        it, total=len(tasks), desc="writing winding", unit="chunk")
    if workers <= 1:
        for _ in progress(map(_write_winding_bucket, tasks)):
            pass
    else:
        with ProcessPoolExecutor(
                workers, mp_context=mp.get_context("spawn")) as pool:
            for _ in progress(pool.map(_write_winding_bucket, tasks,
                                        chunksize=1)):
                pass


def _render_winding_aggregates_cuda(
    voxels, winding, total, maximum, lo, shape, gpu
):
    """Apply the 3x3x3 vote kernel to reduced source entries on one GPU."""
    import torch

    if not len(voxels):
        return None
    device = torch.device(f"cuda:{gpu}")
    offsets = torch.cartesian_prod(*[
        torch.arange(-1, 2, device=device, dtype=torch.int64)
    ] * 3)
    offset_weight = torch.where(
        (offsets == 0).all(-1),
        torch.tensor(1.0, device=device),
        torch.tensor(_VOTE_NEIGHBOR_WEIGHT, device=device),
    )
    lo_t = torch.as_tensor(lo, dtype=torch.int64, device=device)
    shape_t = torch.as_tensor(shape, dtype=torch.int64, device=device)
    partial_keys, partial_totals, partial_maxima = [], [], []
    block = 500_000
    with torch.inference_mode():
        for start in range(0, len(voxels), block):
            end = min(start + block, len(voxels))
            v = torch.as_tensor(
                np.ascontiguousarray(voxels[start:end]), device=device)
            candidate = torch.as_tensor(
                np.ascontiguousarray(winding[start:end]),
                dtype=torch.int64, device=device)
            source_total = torch.as_tensor(
                np.ascontiguousarray(total[start:end]),
                dtype=torch.float32, device=device)
            source_max = torch.as_tensor(
                np.ascontiguousarray(maximum[start:end]),
                dtype=torch.uint8, device=device)
            position = (v[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
            values = (
                source_total[:, None] * offset_weight[None, :]
            ).reshape(-1)
            candidates = candidate[:, None].expand(-1, 27).reshape(-1)
            maxima = source_max[:, None].expand(-1, 27).reshape(-1)
            local = position - lo_t
            inside = ((local >= 0) & (local < shape_t)).all(-1)
            local, values, candidates, maxima = (
                local[inside], values[inside], candidates[inside], maxima[inside])
            flat = (local[:, 0] * int(shape[1]) + local[:, 1]) \
                * int(shape[2]) + local[:, 2]
            keys = flat * 65536 + (candidates + 32768)
            reduced = _aggregate_winding_cuda([keys], [values], [maxima])
            partial_keys.append(reduced[0])
            partial_totals.append(reduced[1])
            partial_maxima.append(reduced[2])
        keys, totals, maxima = _aggregate_winding_cuda(
            partial_keys, partial_totals, partial_maxima)
        keys = keys.cpu().numpy()
        totals = totals.cpu().numpy()
        maxima = maxima.cpu().numpy()

    run_flat = keys >> 16
    run_cand = (keys & 0xFFFF) - 32768
    winner_order = np.lexsort((totals, run_flat))
    run_flat, run_cand, totals, maxima = (
        run_flat[winner_order], run_cand[winner_order],
        totals[winner_order], maxima[winner_order])
    voxel_start = np.r_[0, np.flatnonzero(np.diff(run_flat)) + 1]
    voxel_total = np.add.reduceat(totals, voxel_start)
    last = np.r_[run_flat[1:] != run_flat[:-1], True]
    win_flat, win_cand = run_flat[last], run_cand[last]
    win_weight, win_prob = totals[last], maxima[last]

    core = ((voxels >= lo) & (voxels < lo + shape)).all(-1)
    core_local = voxels[core] - lo
    occupied = np.unique(
        (core_local[:, 0] * shape[1] + core_local[:, 1]) * shape[2]
        + core_local[:, 2])
    if not len(occupied):
        return None
    slot = np.searchsorted(occupied, win_flat)
    keep = (slot < len(occupied)) & (
        occupied[np.minimum(slot, len(occupied) - 1)] == win_flat)
    win_flat, win_cand, win_weight, win_prob, voxel_total = (
        win_flat[keep], win_cand[keep], win_weight[keep],
        win_prob[keep], voxel_total[keep])
    if not len(win_flat):
        return None

    dense_winding = np.full(shape, -1, dtype=np.int16)
    dense_confidence = np.zeros(shape, dtype=np.uint8)
    zz = win_flat // (shape[1] * shape[2])
    rest = win_flat % (shape[1] * shape[2])
    yy, xx = rest // shape[2], rest % shape[2]
    dense_winding[zz, yy, xx] = win_cand.astype(np.int16)
    dense_confidence[zz, yy, xx] = np.clip(np.rint(
        win_prob.astype(np.float64) * win_weight
        / np.maximum(voxel_total, 1e-9)), 0, 255).astype(np.uint8)
    return dense_winding, dense_confidence


def _write_winding_aggregate_bucket(task, gpu):
    """Merge one source-vote chunk and render it on ``gpu``."""
    import zarr

    group_path, out_shape, chunk, bz, by, bx, paths = task
    pieces = [
        records
        for path in paths
        for records in _iter_winding_aggregates(path)
    ]
    if not pieces:
        return 0
    records = np.concatenate(pieces)
    order = np.argsort(records["key"], kind="stable")
    keys = records["key"][order]
    starts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1]
    totals = np.add.reduceat(records["total"][order], starts)
    maxima = np.maximum.reduceat(records["maximum"][order], starts)
    keys = keys[starts]

    linear = keys >> np.uint64(16)
    winding = ((keys & np.uint64(0xFFFF)).astype(np.int64) - 32768)
    plane = int(out_shape[1]) * int(out_shape[2])
    z = linear // np.uint64(plane)
    remainder = linear % np.uint64(plane)
    y = remainder // np.uint64(out_shape[2])
    x = remainder % np.uint64(out_shape[2])
    voxels = np.stack([z, y, x], axis=-1).astype(np.int64)
    lo = np.array([bz * chunk, by * chunk, bx * chunk], dtype=np.int64)
    hi = np.minimum(lo + chunk, np.asarray(out_shape, dtype=np.int64))
    shape = hi - lo
    near = ((voxels >= lo - 1) & (voxels < hi + 1)).all(-1)
    rendered = _render_winding_aggregates_cuda(
        voxels[near], winding[near], totals[near], maxima[near],
        lo, shape, gpu)
    if rendered is None:
        return 0
    dense_winding, dense_confidence = rendered
    group = zarr.open_group(group_path, mode="r+")
    group["winding"][lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = dense_winding
    group["confidence"][lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = \
        dense_confidence
    return int((dense_winding >= 0).sum())


def _winding_aggregate_gpu_worker(gpu, tasks, progress_queue):
    import torch

    torch.cuda.set_device(gpu)
    for task in tasks:
        _write_winding_aggregate_bucket(task, gpu)
        progress_queue.put(1)


def _rasterize_winding_aggregate_spills(
    group_path, directories, out_shape, chunk, gpus
):
    """Merge reduced source votes and render chunks across idle GPUs."""
    import multiprocessing as mp
    import queue
    from tqdm import tqdm

    grid = {}
    for directory in directories:
        for path in Path(directory).glob("*.rec"):
            bucket = tuple(int(part) for part in path.stem.split("_"))
            grid.setdefault(bucket, {})[Path(directory)] = str(path)
    tasks = []
    for bz, by, bx in sorted(grid):
        paths = [
            str(path)
            for directory in directories
            if (path := Path(directory) /
                f"{bz:05d}_{by:05d}_{bx:05d}.rec").is_file()
        ]
        tasks.append((str(group_path), tuple(out_shape), int(chunk),
                      bz, by, bx, paths))
    if not tasks:
        return

    # Greedy size balancing prevents one GPU receiving all dense chunks.
    assignments = [[] for _ in gpus]
    loads = [0] * len(gpus)
    for task in sorted(
            tasks, key=lambda item: sum(os.path.getsize(p) for p in item[-1]),
            reverse=True):
        slot = int(np.argmin(loads))
        assignments[slot].append(task)
        loads[slot] += sum(os.path.getsize(p) for p in task[-1])

    context = mp.get_context("spawn")
    progress_queue = context.Queue()
    processes = []
    for gpu, assigned in zip(gpus, assignments):
        process = context.Process(
            target=_winding_aggregate_gpu_worker,
            args=(gpu, assigned, progress_queue))
        process.start()
        processes.append(process)
    with tqdm(total=len(tasks), desc="writing winding", unit="chunk") as bar:
        completed = 0
        while completed < len(tasks):
            try:
                update = progress_queue.get(timeout=0.5)
            except queue.Empty:
                failed = [process.exitcode for process in processes
                          if process.exitcode not in (None, 0)]
                if failed:
                    raise RuntimeError(
                        f"winding raster worker exit codes: {failed}")
                continue
            bar.update(update)
            completed += update
    for process in processes:
        process.join()
    failed = [process.exitcode for process in processes if process.exitcode]
    if failed:
        raise RuntimeError(f"winding raster worker exit codes: {failed}")


def _write_prob_bucket(task):
    """Pool entry: fold one bucket's spill files into the output array.

    A bucket holds every partial aggregate for one output chunk, so the
    footprint is one small dense chunk plus a bounded decode buffer,
    independent of slab count.
    Each record is one slab's observation of a voxel; ``combine`` folds
    them as the mean (suppresses any single slab's residual edge bias by
    the overlap count) or the max (the old behavior). The display floor
    applies to the combined value. Buckets partition the chunk grid, so no
    two workers touch one chunk.
    """
    import zarr

    group_path, name, out_shape, chunk, bz, by, bx, paths, combine, floor = task
    lo = (bz * chunk, by * chunk, bx * chunk)
    shape = tuple(
        min(chunk, out_shape[axis] - lo[axis]) for axis in range(3))
    size = shape[0] * shape[1] * shape[2]
    if combine == "max":
        dense = np.zeros(shape, dtype=np.uint8)
        dense_flat = dense.reshape(-1)
    else:
        total = np.zeros(size, dtype=np.float64)
        count = np.zeros(size, dtype=np.int64)
    for path in paths:
        for records in _iter_prob_aggregates(path):
            key = records["key"]
            z = (key >> 42).astype(np.int64) - lo[0]
            y = ((key >> 21) & _PROB_KEY_MASK).astype(np.int64) - lo[1]
            x = (key & _PROB_KEY_MASK).astype(np.int64) - lo[2]
            flat = (z * shape[1] + y) * shape[2] + x
            if combine == "max":
                # Every compressed block was reduced to unique keys before
                # spilling. Blocks can overlap each other, but direct indexed
                # folding is therefore exact and avoids maximum.at overhead.
                dense_flat[flat] = np.maximum(
                    dense_flat[flat], records["maximum"])
            else:
                # Keys are unique within this block, so indexed addition is
                # exact. It avoids allocating and clearing two chunk-sized
                # bincount temporaries for every compressed block.
                total[flat] += records["total"]
                count[flat] += records["count"]
    if combine != "max":
        dense = np.rint(
            total / np.maximum(count, 1)
        ).astype(np.uint8).reshape(shape)
    if floor > 0:
        dense[dense < floor] = 0
    array = zarr.open_group(group_path, mode="r+")[name]
    if dense.any():
        array[
            lo[0]:lo[0] + shape[0],
            lo[1]:lo[1] + shape[1],
            lo[2]:lo[2] + shape[2],
        ] = dense
    return int((dense > 0).sum())


def _render_phase_consensus(cosine, sine, density, weight, count, *,
                            sigma_voxels, agreement_power=1.0,
                            min_observations=2):
    """Render uint8 crossing evidence from phase sufficient statistics."""
    cosine = np.asarray(cosine, dtype=np.float64)
    sine = np.asarray(sine, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    count = np.asarray(count)
    supported = (weight > 0) & (count >= int(min_observations))
    evidence = np.zeros(weight.shape, dtype=np.float64)
    if supported.any():
        phase_residual = np.abs(np.arctan2(
            sine[supported], cosine[supported])) / (2.0 * np.pi)
        concentration = np.hypot(
            cosine[supported], sine[supported]
        ) / np.maximum(weight[supported], 1e-12)
        concentration = np.clip(concentration, 0.0, 1.0)
        mean_density = density[supported] / np.maximum(
            weight[supported], 1e-12)
        distance = phase_residual / np.maximum(mean_density, 1e-6)
        evidence[supported] = np.exp(
            -0.5 * (distance / float(sigma_voxels)) ** 2
        ) * concentration ** float(agreement_power)
    return np.clip(np.rint(evidence * 255.0), 0, 255).astype(np.uint8)


def _render_phase_label_consensus(
    cosine, sine, density, weight, weight_sq, count, *, sigma_voxels,
    agreement_power=1.0, min_observations=2,
    min_effective_observations=1.5, min_weight=0.5,
):
    """Render one synchronized integer-winding proposal per input row."""
    cosine, sine, density, weight, weight_sq = (
        np.asarray(value, dtype=np.float64)
        for value in (cosine, sine, density, weight, weight_sq))
    count = np.asarray(count)
    effective = weight * weight / np.maximum(weight_sq, 1e-12)
    supported = (
        (count >= int(min_observations))
        & (effective >= float(min_effective_observations))
        & (weight >= float(min_weight)))
    evidence = np.zeros(weight.shape, dtype=np.float64)
    if supported.any():
        phase_residual = np.abs(np.arctan2(
            sine[supported], cosine[supported])) / (2.0 * np.pi)
        concentration = np.clip(
            np.hypot(cosine[supported], sine[supported])
            / np.maximum(weight[supported], 1e-12), 0.0, 1.0)
        mean_density = density[supported] / np.maximum(
            weight[supported], 1e-12)
        distance = phase_residual / np.maximum(mean_density, 1e-6)
        evidence[supported] = (
            np.exp(-0.5 * (distance / float(sigma_voxels)) ** 2)
            * concentration ** float(agreement_power))
    return np.clip(np.rint(evidence * 255.0), 0, 255).astype(np.uint8)


def _write_phase_bucket(task):
    """Fold one registered-phase bucket and render crossing evidence."""
    import zarr

    (group_path, name, out_shape, chunk, bz, by, bx, paths, floor,
     sigma_voxels, agreement_power, min_observations) = task
    lo = (bz * chunk, by * chunk, bx * chunk)
    shape = tuple(
        min(chunk, out_shape[axis] - lo[axis]) for axis in range(3))
    size = shape[0] * shape[1] * shape[2]
    cosine = np.zeros(size, dtype=np.float64)
    sine = np.zeros(size, dtype=np.float64)
    density = np.zeros(size, dtype=np.float64)
    weight = np.zeros(size, dtype=np.float64)
    count = np.zeros(size, dtype=np.uint64)
    for path in paths:
        for records in _iter_phase_aggregates(path):
            key = records["key"]
            z = (key >> 42).astype(np.int64) - lo[0]
            y = ((key >> 21) & _PROB_KEY_MASK).astype(np.int64) - lo[1]
            x = (key & _PROB_KEY_MASK).astype(np.int64) - lo[2]
            flat = (z * shape[1] + y) * shape[2] + x
            # CUDA reduction makes keys unique inside each compressed block,
            # so direct indexed addition is exact and much faster than add.at.
            cosine[flat] += records["cosine"]
            sine[flat] += records["sine"]
            density[flat] += records["density"]
            weight[flat] += records["weight"]
            count[flat] += records["count"]

    dense = _render_phase_consensus(
        cosine, sine, density, weight, count,
        sigma_voxels=sigma_voxels, agreement_power=agreement_power,
        min_observations=min_observations).reshape(shape)
    if floor > 0:
        dense[dense < floor] = 0
    array = zarr.open_group(group_path, mode="r+")[name]
    if dense.any():
        array[
            lo[0]:lo[0] + shape[0],
            lo[1]:lo[1] + shape[1],
            lo[2]:lo[2] + shape[2],
        ] = dense
    return int((dense > 0).sum())


def _write_phase_label_bucket(task):
    """Fold labeled-phase records and choose the best winding per voxel."""
    import zarr

    (group_path, name, out_shape, chunk, bz, by, bx, paths, floor,
     sigma_voxels, agreement_power, min_observations,
     min_effective_observations, min_weight) = task
    lo = np.array([bz * chunk, by * chunk, bx * chunk], dtype=np.int64)
    shape = tuple(
        min(chunk, out_shape[axis] - int(lo[axis])) for axis in range(3))
    pieces = [
        records
        for path in paths
        for records in _iter_phase_label_aggregates(path)
    ]
    if not pieces:
        return 0
    records = np.concatenate(pieces)
    order = np.argsort(records["key"], kind="stable")
    keys_sorted = records["key"][order]
    starts = np.r_[0, np.flatnonzero(np.diff(keys_sorted)) + 1]
    keys = keys_sorted[starts]
    reduced = {
        name: np.add.reduceat(records[name][order], starts)
        for name in (
            "cosine", "sine", "density", "weight", "weight_sq", "count")
    }
    evidence = _render_phase_label_consensus(
        reduced["cosine"], reduced["sine"], reduced["density"],
        reduced["weight"], reduced["weight_sq"], reduced["count"],
        sigma_voxels=sigma_voxels, agreement_power=agreement_power,
        min_observations=min_observations,
        min_effective_observations=min_effective_observations,
        min_weight=min_weight)

    linear = keys >> np.uint64(16)
    plane = int(out_shape[1]) * int(out_shape[2])
    z = (linear // np.uint64(plane)).astype(np.int64) - lo[0]
    remainder = linear % np.uint64(plane)
    y = (remainder // np.uint64(out_shape[2])).astype(np.int64) - lo[1]
    x = (remainder % np.uint64(out_shape[2])).astype(np.int64) - lo[2]
    flat = (z * shape[1] + y) * shape[2] + x
    dense = np.zeros(int(np.prod(shape)), dtype=np.uint8)
    np.maximum.at(dense, flat, evidence)
    dense = dense.reshape(shape)
    if floor > 0:
        dense[dense < floor] = 0
    array = zarr.open_group(group_path, mode="r+")[name]
    if dense.any():
        array[
            lo[0]:lo[0] + shape[0],
            lo[1]:lo[1] + shape[1],
            lo[2]:lo[2] + shape[2],
        ] = dense
    return int((dense > 0).sum())


def _merge_prob_spill(group_path, name, spill_dirs, out_shape, chunk,
                      workers=None, combine="mean", floor=0.0):
    """Merge spilled prob-volume records into the zarr, bucket at a time.

    Peak memory is ``workers`` dense 128-cubed chunks plus decode buffers,
    versus the old path's full in-RAM concatenation whose footprint grew with
    slab count. ``combine`` is
    "mean" or "max" per voxel over the overlapping slabs' observations;
    ``floor`` (probability units) zeroes combined values below it.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    floor_u8 = int(round(float(floor) * 255))
    by_bucket = {}
    for directory in spill_dirs:
        for path in Path(directory).glob("*.rec"):
            bz, by, bx = (int(part) for part in path.stem.split("_"))
            by_bucket.setdefault((bz, by, bx), []).append(str(path))
    tasks = [
        (str(group_path), name, tuple(out_shape), int(chunk), bz, by, bx, paths,
         str(combine), floor_u8)
        for (bz, by, bx), paths in sorted(by_bucket.items())
    ]
    if not tasks:
        return
    workers = min(workers or min(32, os.cpu_count() or 1), len(tasks))
    progress = lambda it: tqdm(  # noqa: E731
        it, total=len(tasks), desc=f"writing {name}", unit="bucket")
    if workers <= 1:
        for _ in progress(map(_write_prob_bucket, tasks)):
            pass
    else:
        with ProcessPoolExecutor(
                workers, mp_context=mp.get_context("spawn")) as pool:
            for _ in progress(pool.map(_write_prob_bucket, tasks, chunksize=1)):
                pass


def _merge_phase_spill(group_path, name, spill_dirs, out_shape, chunk,
                       workers=None, floor=0.0, sigma_voxels=1.0,
                       agreement_power=1.0, min_observations=2):
    """Merge registered fractional phase and render one crossing kernel."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    floor_u8 = int(round(float(floor) * 255))
    by_bucket = {}
    for directory in spill_dirs:
        for path in Path(directory).glob("*.rec"):
            bz, by, bx = (int(part) for part in path.stem.split("_"))
            by_bucket.setdefault((bz, by, bx), []).append(str(path))
    tasks = [
        (str(group_path), name, tuple(out_shape), int(chunk), bz, by, bx,
         paths, floor_u8, float(sigma_voxels), float(agreement_power),
         int(min_observations))
        for (bz, by, bx), paths in sorted(by_bucket.items())
    ]
    if not tasks:
        return
    workers = min(workers or min(32, os.cpu_count() or 1), len(tasks))
    progress = lambda it: tqdm(  # noqa: E731
        it, total=len(tasks), desc=f"writing {name}", unit="bucket")
    if workers <= 1:
        for _ in progress(map(_write_phase_bucket, tasks)):
            pass
    else:
        with ProcessPoolExecutor(
                workers, mp_context=mp.get_context("spawn")) as pool:
            for _ in progress(pool.map(
                    _write_phase_bucket, tasks, chunksize=1)):
                pass


def _merge_phase_label_spill(
    group_path, name, spill_dirs, out_shape, chunk, workers=None, floor=0.0,
    sigma_voxels=1.0, agreement_power=1.0, min_observations=2,
    min_effective_observations=1.5, min_weight=0.5,
):
    """Merge synchronized integer-labeled phase proposals by output chunk."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    floor_u8 = int(round(float(floor) * 255))
    by_bucket = {}
    for directory in spill_dirs:
        for path in Path(directory).glob("*.rec"):
            bz, by, bx = (int(part) for part in path.stem.split("_"))
            by_bucket.setdefault((bz, by, bx), []).append(str(path))
    tasks = [
        (str(group_path), name, tuple(out_shape), int(chunk), bz, by, bx,
         paths, floor_u8, float(sigma_voxels), float(agreement_power),
         int(min_observations), float(min_effective_observations),
         float(min_weight))
        for (bz, by, bx), paths in sorted(by_bucket.items())
    ]
    if not tasks:
        return
    workers = min(workers or min(32, os.cpu_count() or 1), len(tasks))
    progress = lambda it: tqdm(  # noqa: E731
        it, total=len(tasks), desc=f"writing {name}", unit="bucket")
    if workers <= 1:
        for _ in progress(map(_write_phase_label_bucket, tasks)):
            pass
    else:
        with ProcessPoolExecutor(
                workers, mp_context=mp.get_context("spawn")) as pool:
            for _ in progress(pool.map(
                    _write_phase_label_bucket, tasks, chunksize=1)):
                pass


def _memmap_decoded(path, dtype, shape):
    """Map a raw decoded stream, including the valid empty-array case."""
    if not int(np.prod(shape, dtype=np.int64)):
        return np.empty(shape, dtype=dtype)
    return np.memmap(path, mode="r", dtype=dtype, shape=shape)


_ARCHIVE_WORKER_CONTEXT = None
_ARCHIVE_WORKER_GROUP = None


def _read_sharded_stream(context, name, dtype, start, end, tail_shape=()):
    """Read a global interval from contiguous per-GPU raw streams."""
    bounds = context["point_bounds" if name in ("xyz", "winding", "prob")
                     else "strip_bounds"]
    paths = context["paths"][name]
    pieces = []
    position = int(start)
    item_width = int(np.prod(tail_shape, dtype=np.int64)) if tail_shape else 1
    item_bytes = np.dtype(dtype).itemsize * item_width
    while position < end:
        shard = int(np.searchsorted(bounds[1:], position, side="right"))
        shard_end = min(int(end), int(bounds[shard + 1]))
        count = shard_end - position
        local_start = position - int(bounds[shard])
        shape = (count, *tail_shape)
        pieces.append(np.memmap(
            paths[shard], mode="r", dtype=dtype,
            offset=local_start * item_bytes, shape=shape,
        ))
        position = shard_end
    if not pieces:
        return np.empty((0, *tail_shape), dtype=dtype)
    if len(pieces) == 1:
        return pieces[0]
    return np.concatenate(pieces)


def _init_archive_worker(context):
    """Open output metadata once per archive materialization process."""
    global _ARCHIVE_WORKER_CONTEXT, _ARCHIVE_WORKER_GROUP
    import zarr

    _ARCHIVE_WORKER_CONTEXT = context
    _ARCHIVE_WORKER_GROUP = zarr.open_group(context["group_path"], mode="r+")


def _write_archive_chunk(task):
    """Write one disjoint point or strip chunk into the final archive."""
    context = _ARCHIVE_WORKER_CONTEXT
    group = _ARCHIVE_WORKER_GROUP
    kind, start, end, point_start = task
    if kind == "point":
        points = group["points"]
        points["xyz"][start:end] = _read_sharded_stream(
            context, "xyz", np.float32, start, end, (3,))
        points["winding"][start:end] = _read_sharded_stream(
            context, "winding", np.int16, start, end)
        points["prob"][start:end] = _read_sharded_stream(
            context, "prob", np.uint8, start, end)
        return end - start

    lengths = np.asarray(_read_sharded_stream(
        context, "strip_length", np.uint32, start, end), dtype=np.int64)
    slabs = np.asarray(_read_sharded_stream(
        context, "strip_slab", np.int64, start, end)).copy()
    bounds = context["strip_bounds"]
    position = start
    while position < end:
        shard = int(np.searchsorted(bounds[1:], position, side="right"))
        shard_end = min(end, int(bounds[shard + 1]))
        slabs[position - start:shard_end - start] += \
            int(context["ray_bounds"][shard])
        position = shard_end

    # Each task owns identically aligned chunks in both strip arrays. Offset
    # i is the beginning of strip i, so lengths[:-1] fills the later entries.
    offsets = np.empty(end - start, dtype=np.int64)
    if len(offsets):
        offsets[0] = int(point_start)
        if len(offsets) > 1:
            offsets[1:] = int(point_start) + np.cumsum(
                lengths[:-1], dtype=np.int64)
    group["strips"]["offsets"][start:end] = offsets
    group["strips"]["slab"][start:end] = slabs
    return end - start


def _materialize_archive(group_path, result_paths, shard_results, ray_bounds,
                         point_chunk, strip_chunk, workers=None):
    """Copy required raw point/strip streams into Zarr in parallel.

    Tasks align exactly with output chunks. Consequently processes never
    update the same chunk, retaining deterministic bytes without store locks.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    point_counts = np.asarray(
        [int(result["num_points"]) for result in shard_results],
        dtype=np.int64)
    strip_counts = np.asarray(
        [int(result["num_strips"]) for result in shard_results],
        dtype=np.int64)
    point_bounds = np.r_[0, np.cumsum(point_counts, dtype=np.int64)]
    strip_bounds = np.r_[0, np.cumsum(strip_counts, dtype=np.int64)]
    paths = {
        name: [str(_decoded_shard_paths(path)[name]) for path in result_paths]
        for name in ("xyz", "winding", "prob", "strip_slab", "strip_length")
    }
    context = {
        "group_path": str(group_path),
        "paths": paths,
        "point_bounds": point_bounds,
        "strip_bounds": strip_bounds,
        "ray_bounds": np.asarray(ray_bounds, dtype=np.int64),
    }

    point_tasks = [
        ("point", start, min(start + point_chunk, int(point_bounds[-1])), 0)
        for start in range(0, int(point_bounds[-1]), point_chunk)
    ]
    # Compute one prefix per strip chunk in a single sequential scan. This is
    # only the compact uint32 length stream (~9 GB for the reported run), and
    # permits every much larger archive chunk to be encoded independently.
    strip_tasks = []
    point_start = 0
    for start in range(0, int(strip_bounds[-1]), strip_chunk):
        end = min(start + strip_chunk, int(strip_bounds[-1]))
        strip_tasks.append(("strip", start, end, point_start))
        lengths = _read_sharded_stream(
            context, "strip_length", np.uint32, start, end)
        point_start += int(np.asarray(lengths, dtype=np.uint64).sum())
    if point_start != int(point_bounds[-1]):
        raise RuntimeError(
            "decoded point/strip mismatch while building archive tasks: "
            f"{point_start} != {int(point_bounds[-1])}")

    tasks = point_tasks + strip_tasks
    if not tasks:
        return
    workers = min(workers or min(8, os.cpu_count() or 1), len(tasks))
    with ProcessPoolExecutor(
            workers, mp_context=mp.get_context("spawn"),
            initializer=_init_archive_worker, initargs=(context,)) as pool:
        results = pool.map(_write_archive_chunk, tasks, chunksize=1)
        for _ in tqdm(results, total=len(tasks), desc="materializing archive",
                      unit="chunk"):
            pass


def write_output(args, rays, shard_results, result_paths, elapsed_total):
    import zarr

    reference = Path(args.reference_zarr)
    out_shape = _output_shape(reference, int(args.output_downsample))
    num_points = sum(int(r["num_points"]) for r in shard_results)
    num_strips = sum(int(r["num_strips"]) for r in shard_results)

    # Shards were built by np.linspace over the ray list in this fixed order.
    ray_bounds = np.linspace(
        0, len(rays["seed_xyz"]), len(shard_results) + 1).astype(int)

    group = zarr.open_group(args.output, mode="w")
    chunk = _OUTPUT_CHUNK
    group.create_array(
        "winding", shape=out_shape, chunks=(chunk, chunk, chunk),
        dtype="int16", fill_value=-1)
    group.create_array(
        "confidence", shape=out_shape, chunks=(chunk, chunk, chunk),
        dtype="uint8", fill_value=0)

    point_chunk = max(1, min(num_points, 1 << 20))
    strip_chunk = max(1, min(num_strips + 1, 1 << 20))
    points = group.create_group("points")
    points.create_array(
        "xyz", shape=(num_points, 3), dtype="float32",
        chunks=(point_chunk, 3))
    points.create_array(
        "winding", shape=(num_points,), dtype="int16", chunks=(point_chunk,))
    points.create_array(
        "prob", shape=(num_points,), dtype="uint8", chunks=(point_chunk,))
    strips = group.create_group("strips")
    strip_offsets_arr = strips.create_array(
        "offsets", shape=(num_strips + 1,), dtype="int64",
        chunks=(strip_chunk,))
    strips.create_array(
        "slab", shape=(num_strips,), dtype="int64",
        chunks=(max(1, min(num_strips, 1 << 20)),))
    strip_offsets_arr[0] = 0

    materialize_started = time.time()
    _materialize_archive(
        args.output, result_paths, shard_results, ray_bounds,
        point_chunk, strip_chunk,
        workers=getattr(args, "archive_workers", None))
    # Chunk-aligned strip tasks own offsets [0:num_strips); write the terminal
    # point count only after all workers have joined.
    strip_offsets_arr[num_strips] = num_points

    materialize_elapsed = time.time() - materialize_started
    raster_started = time.time()
    winding_spill_dirs = sorted(
        Path(args.output).with_suffix(".tmp").glob("winding_spill_*"))
    raster_gpus = list(getattr(args, "worker_gpus", ()))
    _rasterize_winding_aggregate_spills(
        args.output, winding_spill_dirs, out_shape, chunk, raster_gpus)
    raster_elapsed = time.time() - raster_started

    prob_merge_elapsed = 0.0
    if getattr(args, "prob_volume", False):
        group.create_array(
            "crossing_prob", shape=out_shape, chunks=(chunk, chunk, chunk),
            dtype="uint8", fill_value=0)
        prob_merge_started = time.time()
        if getattr(args, "prob_combine", "mean") == "phase-label":
            spill_dirs = sorted(
                Path(args.output).with_suffix(".tmp").glob(
                    "phase_label_spill_*"))
            _merge_phase_label_spill(
                args.output, "crossing_prob", spill_dirs, out_shape, chunk,
                workers=args.merge_workers,
                floor=float(getattr(args, "prob_volume_floor", 0.0)),
                sigma_voxels=(
                    float(getattr(args, "passage_sigma_samples", 1.0))
                    * float(getattr(args, "model_spacing", 1.0))),
                agreement_power=float(getattr(
                    args, "prob_phase_agreement_power", 1.0)),
                min_observations=int(getattr(
                    args, "prob_phase_min_observations", 2)),
                min_effective_observations=float(getattr(
                    args, "prob_phase_min_effective_observations", 1.5)),
                min_weight=float(getattr(
                    args, "prob_phase_min_weight", 0.5)),
            )
        elif getattr(args, "prob_combine", "mean") == "phase":
            spill_dirs = sorted(
                Path(args.output).with_suffix(".tmp").glob("phase_spill_*"))
            _merge_phase_spill(
                args.output, "crossing_prob", spill_dirs, out_shape, chunk,
                workers=args.merge_workers,
                floor=float(getattr(args, "prob_volume_floor", 0.0)),
                sigma_voxels=(
                    float(getattr(args, "passage_sigma_samples", 1.0))
                    * float(getattr(args, "model_spacing", 1.0))),
                agreement_power=float(getattr(
                    args, "prob_phase_agreement_power", 1.0)),
                min_observations=int(getattr(
                    args, "prob_phase_min_observations", 2)),
            )
        else:
            spill_dirs = sorted(
                Path(args.output).with_suffix(".tmp").glob("prob_spill_*"))
            _merge_prob_spill(
                args.output, "crossing_prob", spill_dirs, out_shape, chunk,
                workers=args.merge_workers,
                combine=getattr(args, "prob_combine", "mean"),
                floor=float(getattr(args, "prob_volume_floor", 0.0)))
        prob_merge_elapsed = time.time() - prob_merge_started

    rays_group = group.create_group("rays")
    for key in ("seed_xyz", "direction_xyz", "seed_winding", "phase_offset"):
        if key not in rays:
            continue
        value = rays[key]
        rays_group.create_array(key, shape=value.shape, dtype=value.dtype,
                                chunks=value.shape)[:] = value

    group.attrs.update({
        "fit_checkpoint": str(Path(args.fit_checkpoint).resolve()),
        "model_ckpt": str(Path(args.model_ckpt).resolve()),
        "reference_zarr": str(reference.resolve()),
        "phase_cache": (None if not getattr(args, "phase_cache", None)
                        else str(Path(args.phase_cache).resolve())),
        "volume_scale": args.volume_scale,
        "coordinate_space": "reference zarr scale-0 voxels; winding/confidence "
                            f"arrays downsampled by {args.output_downsample}",
        "output_downsample": args.output_downsample,
        "z_range": [int(v) for v in rays["z_range"]],
        "seed_windings": [int(v) for v in rays["seed_windings"]],
        "winding_step": args.winding_step,
        "seed_spacing": args.seed_spacing,
        "column_step": args.column_step,
        "column_upsample": int(getattr(args, "column_upsample", 1)),
        "slab_center_width": getattr(args, "slab_center_width", None),
        "phase_cache_winding_stride": int(getattr(
            args, "phase_cache_winding_stride", 1)),
        "phase_registration": {
            "mode": str(getattr(args, "phase_registration", "anchor")),
            "synchronization": getattr(args, "phase_sync_stats", None),
        },
        "dr_per_winding": float(rays["dr_per_winding"]),
        "umbilicus": str(rays["umbilicus_path"]),
        "decode": {
            "threshold": args.threshold,
            "min_distance": args.min_distance,
            "min_prob_keep": args.min_prob_keep,
            "max_level": args.max_level,
            "phase_decode": bool(getattr(args, "phase_decode", False)),
            "phase_level_dedup": not bool(getattr(args, "phase_decode", False)),
            "prob_source": (
                "synchronized_integer_phase_consensus"
                if getattr(args, "prob_combine", "mean") == "phase-label"
                else ("registered_phase_consensus"
                if getattr(args, "prob_combine", "mean") == "phase"
                else ("phase_passage_kernels"
                      if getattr(args, "phase_decode", False)
                      else "crossing_head"))),
            "passage_sigma_samples": float(
                getattr(args, "passage_sigma_samples", 1.0)),
            "anchor_tolerance": ANCHOR_TOLERANCE,
            "edge_margin": EDGE_MARGIN,
        },
        "winding_raster": {
            "combine": "neighborhood_vote",
            "vote_level_half_life": _VOTE_LEVEL_HALF_LIFE,
            "vote_neighbor_weight": _VOTE_NEIGHBOR_WEIGHT,
            "confidence": "vote share x winner best prob",
        },
        "prob_volume": bool(getattr(args, "prob_volume", False)),
        "prob_volume_floor": float(getattr(args, "prob_volume_floor", 0.0)),
        "prob_combine": str(getattr(args, "prob_combine", "mean")),
        "prob_ray_margin": int(getattr(args, "prob_ray_margin", 32)),
        "prob_column_margin": int(getattr(args, "prob_column_margin", 2)),
        "prob_column_step": int(getattr(args, "prob_column_step", None)
                                or args.column_step),
        "prob_phase": ({
            "level_half_life": float(getattr(
                args, "prob_phase_level_half_life", 2.0)),
            "max_level": (float(args.max_level) + 0.5
                          if getattr(args, "prob_phase_max_level", None) is None
                          else float(args.prob_phase_max_level)),
            "edge_taper": int(getattr(args, "prob_phase_edge_taper", 8)),
            "agreement_power": float(getattr(
                args, "prob_phase_agreement_power", 1.0)),
            "min_observations": int(getattr(
                args, "prob_phase_min_observations", 2)),
            "min_effective_observations": float(getattr(
                args, "prob_phase_min_effective_observations", 1.5)),
            "min_weight": float(getattr(
                args, "prob_phase_min_weight", 0.5)),
            "band_sigma": float(getattr(
                args, "prob_phase_band_sigma", 4.0)),
            "projection": "nearest_dense_grid",
        } if getattr(args, "prob_combine", "mean") in
             ("phase", "phase-label") else None),
        "runtime": {
            "gpus": str(args.gpus),
            "batch_size": int(args.batch_size),
            "compile": bool(args.compile),
            "extract_threads_per_gpu": int(args.extract_threads),
            "decode_workers_per_gpu": int(args.decode_workers),
            "volume_cache_bytes_per_gpu": (
                None if args.volume_cache_bytes is None
                else int(args.volume_cache_bytes)),
            "raster_workers": getattr(args, "raster_workers", None),
            "archive_workers": getattr(args, "archive_workers", None),
            "merge_workers": args.merge_workers,
        },
        "finalization": {
            "materialize_points_seconds": materialize_elapsed,
            "rasterize_winding_seconds": raster_elapsed,
            "merge_probability_seconds": prob_merge_elapsed,
            "archive_point_chunk": point_chunk,
            "archive_strip_chunk": strip_chunk,
        },
        "num_points": int(num_points),
        "num_strips": int(num_strips),
        "num_slabs": int(sum(int(r["slabs"]) for r in shard_results)),
        "elapsed_seconds": elapsed_total,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    print(f"[finalize] points/strips {materialize_elapsed:.1f}s, winding "
          f"raster {raster_elapsed:.1f}s, probability merge "
          f"{prob_merge_elapsed:.1f}s", flush=True)
    return num_points


def VolumeSlabExtractorShape(reference: Path):
    import zarr
    node = zarr.open(str(reference), mode="r")
    if hasattr(node, "shape"):
        return node.shape
    return node["0"].shape


# --------------------------------------------------------------------------

def main():
    args = parse_args()
    import multiprocessing as mp

    # Whether the model decodes crossings from phase passages (no crossing
    # head): detected once here so the GPU workers, decode threads, and the
    # output attrs all read the same flag off args.
    winding_model_cfg = (
        _native_phase_cache_model_cfg(args.phase_cache)
        if args.phase_cache
        else _load_checkpoint_cfg(args.model_ckpt).get("config", {}))
    args.phase_decode = not bool(
        (winding_model_cfg.get("model") or {}).get("use_crossing_head", True))
    if (args.native_phase_only or args.phase_cache) and not args.phase_decode:
        raise ValueError(
            "native phase caching currently requires a headless phase model; "
            "this checkpoint has a separate crossing head that would also "
            "need to be cached")
    if args.prob_combine in ("phase", "phase-label") and not args.phase_decode:
        raise ValueError(
            "phase probability combine modes require a headless phase model")
    if (args.prob_combine == "phase-label"
            and args.phase_registration != "overlap"):
        raise ValueError(
            "--prob-combine phase-label requires "
            "--phase-registration overlap")
    if args.prob_phase_level_half_life <= 0:
        raise ValueError("--prob-phase-level-half-life must be positive")
    if args.prob_phase_max_level is not None and args.prob_phase_max_level <= 0:
        raise ValueError("--prob-phase-max-level must be positive")
    if args.prob_phase_edge_taper < 0:
        raise ValueError("--prob-phase-edge-taper cannot be negative")
    if args.prob_phase_agreement_power < 0:
        raise ValueError("--prob-phase-agreement-power cannot be negative")
    if args.prob_phase_min_observations < 1:
        raise ValueError("--prob-phase-min-observations must be at least one")
    if args.prob_phase_min_effective_observations < 1:
        raise ValueError(
            "--prob-phase-min-effective-observations must be at least one")
    if args.prob_phase_min_weight < 0:
        raise ValueError("--prob-phase-min-weight cannot be negative")
    if args.prob_phase_band_sigma <= 0:
        raise ValueError("--prob-phase-band-sigma must be positive")
    if args.slab_center_width is not None and args.slab_center_width <= 0:
        raise ValueError("--slab-center-width must be positive")
    if args.phase_cache_winding_stride < 1:
        raise ValueError("--phase-cache-winding-stride must be at least one")
    if args.phase_cache_winding_stride > 1 and not args.phase_cache:
        raise ValueError(
            "--phase-cache-winding-stride only applies with --phase-cache")
    if args.phase_registration == "overlap" and not args.phase_cache:
        raise ValueError(
            "--phase-registration overlap requires --phase-cache")
    if args.phase_sync_radius <= 0:
        raise ValueError("--phase-sync-radius must be positive")
    if args.phase_sync_neighbors < 1:
        raise ValueError("--phase-sync-neighbors must be at least one")
    if args.phase_sync_workers is not None and args.phase_sync_workers < 1:
        raise ValueError("--phase-sync-workers must be at least one")
    if args.phase_sync_block_size < 1:
        raise ValueError("--phase-sync-block-size must be at least one")
    if args.phase_sync_transverse_margin < 0 or args.phase_sync_ray_margin < 0:
        raise ValueError("phase synchronization margins cannot be negative")
    if args.phase_sync_taper < 0:
        raise ValueError("--phase-sync-taper cannot be negative")
    if args.phase_sync_min_density <= 0:
        raise ValueError("--phase-sync-min-density must be positive")
    if args.phase_sync_iterations < 1:
        raise ValueError("--phase-sync-iterations must be at least one")
    if args.phase_sync_huber <= 0 or args.phase_sync_prior_huber <= 0:
        raise ValueError("phase synchronization Huber transitions must be positive")
    if args.phase_sync_prior_weight <= 0:
        raise ValueError("--phase-sync-prior-weight must be positive")
    if args.phase_sync_max_correction <= 0:
        raise ValueError("--phase-sync-max-correction must be positive")
    # Headless prob fields render integer-passage kernels of the label
    # kernel's width (in ray samples).
    args.passage_sigma_samples = max(
        float(winding_model_cfg.get("crossing_sigma_wv", 1.0))
        / float(winding_model_cfg.get("spacing", 1.0)),
        1e-6,
    )
    args.model_spacing = float(winding_model_cfg.get("spacing", 1.0))
    if args.phase_decode:
        print("[main] no crossing head in model checkpoint: decoding "
              "crossings from phase integer passages (prob fields render "
              "unit-height passage kernels)", flush=True)
    if args.prob_volume and args.prob_combine in ("phase", "phase-label"):
        description = (
            "synchronized integer-level phase consensus with effective "
            "weighted support" if args.prob_combine == "phase-label"
            else "registered phase consensus with anchor-distance and "
                 "edge-taper weighting")
        print(f"[main] crossing_prob uses {description}", flush=True)
    if args.phase_registration == "overlap":
        print(
            "[main] phase gauges will be synchronized from world-space slab "
            "overlaps; fitted seed winding is a weak gauge prior only",
            flush=True)

    if args.gpus is not None:
        gpus = [int(g) for g in args.gpus.split(",") if g != ""]
    else:
        import torch
        gpus = list(range(torch.cuda.device_count()))
    if not gpus:
        raise RuntimeError("no GPUs")
    args.worker_gpus = gpus

    scratch = Path(args.output).with_suffix(".tmp")
    scratch.mkdir(parents=True, exist_ok=True)
    # Workers recreate their own spill dirs, but a rerun with fewer GPUs
    # would leave higher-numbered dirs behind — and the merge globs them all.
    for pattern in (
        "prob_spill_*", "phase_spill_*", "phase_label_spill_*",
        "winding_spill_*",
    ):
        for stale in scratch.glob(pattern):
            shutil.rmtree(stale)
    legacy_winding_spill = scratch / "winding_spill"
    if legacy_winding_spill.exists():
        shutil.rmtree(legacy_winding_spill)
    for stale in scratch.glob("result_*.bin"):
        stale.unlink()

    started = time.time()
    rays_path = scratch / "seed_rays.npz"
    if args.phase_cache:
        rays = _load_native_phase_cache(args)
        rays["global_index"] = np.arange(
            len(rays["seed_xyz"]), dtype=np.int64)
        cached_count = len(rays["seed_xyz"])
        rays = _subsample_cached_winding_sheets(
            rays, args.phase_cache_winding_stride)
        if len(rays["seed_xyz"]) != cached_count:
            retained = np.asarray(rays["seed_windings"])
            largest_gap = int(np.diff(retained).max()) \
                if len(retained) > 1 else 0
            print(
                f"[main] cached winding-sheet stride "
                f"{args.phase_cache_winding_stride}: retained "
                f"{len(rays['seed_xyz']):,}/{cached_count:,} slabs on "
                f"{len(retained)} anchor sheets", flush=True)
            phase_limit = (
                float(args.max_level) + 0.5
                if args.prob_phase_max_level is None
                else float(args.prob_phase_max_level))
            if (int(args.max_level) < largest_gap
                    or (args.prob_volume
                        and args.prob_combine in ("phase", "phase-label")
                        and phase_limit < largest_gap + 0.5)):
                print(
                    f"[warning] retained anchor sheets are up to "
                    f"{largest_gap} windings apart; use --max-level "
                    f"{largest_gap} and --prob-phase-max-level "
                    f"{largest_gap + 0.5:g} to maintain overlap at the "
                    f"anchor sheets", flush=True)
    elif args.seed_rays_npz:
        rays = dict(np.load(args.seed_rays_npz, allow_pickle=False))
        rays["umbilicus_path"] = str(rays["umbilicus_path"])
    elif args.seed_source == "meshes":
        rays = build_seed_rays_from_meshes(args)
        np.savez(rays_path, **rays)
    else:
        rays = build_seed_rays(args, device=f"cuda:{gpus[0]}")
        np.savez(rays_path, **rays)
    total = len(rays["seed_xyz"])
    if args.max_slabs is not None and total > args.max_slabs:
        if args.max_slabs_selection == "first":
            keep = np.arange(args.max_slabs)
        else:
            keep = np.random.default_rng(0).choice(
                total, args.max_slabs, replace=False)
            keep.sort()
        for key in ("seed_xyz", "direction_xyz", "seed_winding",
                    "global_index"):
            if key not in rays:
                continue
            rays[key] = rays[key][keep]
        total = args.max_slabs
    if "global_index" not in rays:
        rays["global_index"] = np.arange(total, dtype=np.int64)
    args.phase_sync_stats = None
    if args.phase_registration == "overlap":
        from vesuvius.neural_tracing.winding_models.phase_overlap_sync import (
            build_phase_overlap_offsets,
        )

        phase_offset, args.phase_sync_stats = build_phase_overlap_offsets(
            args, rays, winding_model_cfg, scratch)
        rays["phase_offset"] = phase_offset
    else:
        rays["phase_offset"] = np.zeros(total, dtype=np.float32)
    print(f"[main] {total:,} rays on {len(gpus)} GPU(s)", flush=True)

    # Shard contiguously: nearby rays share volume chunks, which keeps each
    # worker's sampler cache hot.
    import queue as queue_module
    from tqdm import tqdm

    context = mp.get_context("spawn")
    progress_queue = context.Queue()
    shard_paths, result_paths, processes = [], [], []
    bounds = np.linspace(0, total, len(gpus) + 1).astype(int)
    if args.native_phase_only:
        _initialize_native_phase_cache(
            args, rays, bounds, gpus, winding_model_cfg)
    for slot, gpu in enumerate(gpus):
        lo, hi = bounds[slot], bounds[slot + 1]
        shard_path = scratch / f"shard_{gpu}.npz"
        np.savez(shard_path,
                 seed_xyz=rays["seed_xyz"][lo:hi],
                 direction_xyz=rays["direction_xyz"][lo:hi],
                 seed_winding=rays["seed_winding"][lo:hi],
                 global_index=rays["global_index"][lo:hi],
                 phase_offset=rays["phase_offset"][lo:hi])
        result_path = scratch / f"result_{gpu}.npz"
        shard_paths.append(shard_path)
        result_paths.append(result_path)
        process = context.Process(
            target=gpu_worker,
            args=(gpu, args, shard_path, result_path, progress_queue))
        process.start()
        processes.append(process)

    with tqdm(total=total, desc="inference", unit="slab",
              smoothing=0.05) as bar:
        while any(process.is_alive() for process in processes):
            try:
                bar.update(progress_queue.get(timeout=0.5))
            except queue_module.Empty:
                pass
        while True:
            try:
                bar.update(progress_queue.get_nowait())
            except queue_module.Empty:
                break
    for process in processes:
        process.join()
    failed = [p.exitcode for p in processes if p.exitcode]
    if failed:
        raise RuntimeError(f"worker exit codes: {failed}")

    shard_results = [
        dict(np.load(p))
        for p in tqdm(result_paths, desc="loading shards", unit="shard")
    ]
    inference_elapsed = time.time() - started
    if args.native_phase_only:
        import zarr

        cache = zarr.open_group(args.output, mode="r+")
        available = sum(
            int(np.count_nonzero(cache["available"][item["name"]][:]))
            for item in cache.attrs["phase_shards"])
        cache.attrs.update({
            "complete": True,
            "num_slabs": int(total),
            "num_available_slabs": int(available),
            "elapsed_seconds": float(inference_elapsed),
        })
        print(
            f"[main] wrote native phase for {available:,}/{total:,} slabs "
            f"to {args.output} in {inference_elapsed:.0f}s", flush=True)
        return
    num_points = write_output(
        args, rays, shard_results, result_paths, inference_elapsed)
    elapsed = time.time() - started
    import zarr
    output_group = zarr.open_group(args.output, mode="r+")
    output_group.attrs["inference_elapsed_seconds"] = inference_elapsed
    output_group.attrs["elapsed_seconds"] = elapsed
    slabs = sum(int(r["slabs"]) for r in shard_results)
    print(f"[main] wrote {num_points:,} crossings from {slabs:,} slabs to "
          f"{args.output} in {elapsed:.0f}s total "
          f"({slabs / max(elapsed, 1e-9):.2f} slabs/s aggregate)", flush=True)


if __name__ == "__main__":
    main()
