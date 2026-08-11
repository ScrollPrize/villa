"""Audit a surface-prediction volume against the CT it was inferred from.

Predictions can contain *phantom* positives: voxels marked foreground where the
masked CT reads exactly 0, i.e. outside the scroll (see ScrollPrize/villa#1114).
Inference-time prevention and post-hoc masking address new and existing runs;
this module answers the separate question those don't: **given a prediction
volume that already exists, how contaminated is it, and where?**

Two modes, both read-only and resumable:

``chunks``
    Zero-download audit. Zarr stores omit all-zero chunks, so the set of stored
    chunk keys is an exact map of where each volume holds data. Listing keys for
    the predictions and for the CT therefore classifies every prediction chunk
    -- supported (overlaps CT data), inside the one-chunk blend margin, or
    beyond it -- without fetching a single voxel. Runs in a minute or two per
    scroll against remote storage and is the fast way to check whether a fix
    landed, or to triage a batch of published volumes.

``voxels``
    Exact voxel-level phantom fraction. Reads chunk-aligned z-slabs of both
    volumes in bounded-memory Y-stripes (every transferred byte is used; plane
    sampling pays chunk-depth amplification on remote stores) and reports
    per-plane positives/phantoms plus the totals.

Examples
--------
    # zero-download triage of a published volume
    vesuvius.audit_ct_support chunks \\
        --predictions s3://bucket/PHerc0332/.../surface.zarr/0 \\
        --ct s3://bucket/PHerc0332/volumes/...-masked.zarr/2 \\
        --anon --output audit.json

    # exact voxel fractions, every 12th chunk slab
    vesuvius.audit_ct_support voxels \\
        --predictions preds.zarr/0 --ct ct.zarr/2 \\
        --slab-stride 12 --output survey.json
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple

import fsspec
import numpy as np

from vesuvius.data.utils import open_zarr
from vesuvius.utils.cli import HyphenUnderscoreParser

ChunkCoord = Tuple[int, ...]

# Bytes of decoded prediction+CT held at once by a `voxels` stripe. A z-slab of a
# 4k^2 volume is ~30 GB decoded, which is why the read is striped along Y.
STRIPE_BYTES = 2_000_000_000


def _filesystem(path: str, anon: bool = False):
    """fsspec filesystem plus the path with its protocol prefix stripped."""
    if "://" in path:
        protocol, rest = path.split("://", 1)
        options: Dict[str, Any] = {}
        if protocol in ("s3", "gs", "gcs") and anon:
            options["anon"] = True
        return fsspec.filesystem(protocol, **options), rest
    return fsspec.filesystem("file"), path


def _parse_chunk_key(key: str, ndim: int) -> Optional[ChunkCoord]:
    """Chunk index from a stored key, or None if the key is metadata.

    Handles the layouts zarr writes: v2 nested (``0/1/2``), v2 flat
    (``0.1.2``), and v3 (``c/0/1/2``).
    """
    key = key.strip("/")
    if not key or key.startswith(".") or key.endswith(".json"):
        return None
    parts = key.split("/")
    if parts and parts[0] == "c":            # zarr v3 chunk prefix
        parts = parts[1:]
    if len(parts) == 1:                      # v2 flat: dimensions dot-separated
        parts = parts[0].split(".")
    if len(parts) != ndim or not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def stored_chunk_coords(path: str, ndim: int, anon: bool = False) -> Set[ChunkCoord]:
    """Indices of every chunk physically present in the store.

    All-zero chunks are not written, so membership here means "this region
    holds data" -- the property the ``chunks`` audit is built on.
    """
    fs, root = _filesystem(path, anon=anon)
    root = root.rstrip("/")
    coords: Set[ChunkCoord] = set()
    for key in fs.find(root):
        rel = key[len(root):] if key.startswith(root) else key
        coord = _parse_chunk_key(rel, ndim)
        if coord is not None:
            coords.add(coord)
    return coords


def _chunk_box(coord: ChunkCoord, chunks: Sequence[int],
               shape: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
    """Half-open voxel box [start, stop) covered by a chunk index."""
    box = []
    for i, c in enumerate(coord):
        start = c * chunks[i]
        box.append((start, min(start + chunks[i], shape[i])))
    return tuple(box)


def _gap(box_a: Sequence[Tuple[int, int]], box_b: Sequence[Tuple[int, int]]) -> int:
    """Chebyshev gap in voxels between two half-open boxes (0 if they overlap)."""
    gap = 0
    for (a0, a1), (b0, b1) in zip(box_a, box_b):
        d = max(0, b0 - (a1 - 1), a0 - (b1 - 1))
        gap = max(gap, d)
    return gap


def classify_chunks(pred_coords: Iterable[ChunkCoord],
                    ct_coords: Set[ChunkCoord],
                    shape: Sequence[int],
                    pred_chunks: Sequence[int],
                    ct_chunks: Sequence[int]) -> Dict[str, Any]:
    """Bucket prediction chunks by distance to the nearest CT-bearing chunk.

    Buckets: ``supported`` (boxes overlap), ``halo_within_1_chunk`` (gap of at
    most one prediction chunk -- the reach of a blend window that runs one chunk
    past the data), and ``beyond_blend_margin`` (farther, which a blend-margin
    artifact cannot explain and points at a second mechanism).
    """
    margin = int(max(pred_chunks))
    ndim = len(shape)
    counts = {"supported": 0, "halo_within_1_chunk": 0, "beyond_blend_margin": 0}
    beyond_examples = []

    for coord in pred_coords:
        pbox = _chunk_box(coord, pred_chunks, shape)
        # Candidate CT chunks are those whose index range can reach pbox +- margin.
        ranges = []
        for i in range(ndim):
            lo = max(0, (pbox[i][0] - margin) // ct_chunks[i])
            hi = (pbox[i][1] - 1 + margin) // ct_chunks[i]
            ranges.append((lo, hi))

        best = None
        for cz in range(ranges[0][0], ranges[0][1] + 1):
            for cy in range(ranges[1][0], ranges[1][1] + 1):
                for cx in range(ranges[2][0], ranges[2][1] + 1):
                    cand = (cz, cy, cx)
                    if cand not in ct_coords:
                        continue
                    d = _gap(pbox, _chunk_box(cand, ct_chunks, shape))
                    if best is None or d < best:
                        best = d
                        if best == 0:
                            break
                if best == 0:
                    break
            if best == 0:
                break

        if best == 0:
            counts["supported"] += 1
        elif best is not None and best <= margin:
            counts["halo_within_1_chunk"] += 1
        else:
            counts["beyond_blend_margin"] += 1
            if len(beyond_examples) < 20:
                beyond_examples.append({"chunk": list(coord),
                                        "gap_voxels": None if best is None else int(best)})

    total = sum(counts.values())
    report = {"prediction_chunks": total, **counts,
              "beyond_examples": beyond_examples}
    if total:
        report["frac_supported"] = counts["supported"] / total
        report["frac_halo_within_1_chunk"] = counts["halo_within_1_chunk"] / total
        report["frac_beyond_blend_margin"] = counts["beyond_blend_margin"] / total
    return report


def _read_options(paths: Sequence[str], anon: bool,
                  storage_options: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Storage options for opening arrays, honouring anonymous object access."""
    if not anon or not any("://" in p and not p.startswith("file://") for p in paths):
        return storage_options
    options = dict(storage_options or {})
    options.setdefault("anon", True)
    return options


def audit_chunks(predictions: str, ct: str, anon: bool = False,
                 storage_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Zero-download chunk-level audit of a prediction volume against its CT."""
    read_options = _read_options((predictions, ct), anon, storage_options)
    P = open_zarr(predictions, mode="r", storage_options=read_options)
    C = open_zarr(ct, mode="r", storage_options=read_options)
    if tuple(P.shape) != tuple(C.shape):
        raise ValueError(
            f"grid mismatch: predictions {tuple(P.shape)} vs CT {tuple(C.shape)} -- "
            f"point --ct at the pyramid level that matches the prediction grid")
    if len(P.shape) != 3:
        raise ValueError(f"expected a 3D volume, got shape {tuple(P.shape)}")

    t0 = time.time()
    pred_coords = stored_chunk_coords(predictions, len(P.shape), anon=anon)
    ct_coords = stored_chunk_coords(ct, len(C.shape), anon=anon)
    report = classify_chunks(pred_coords, ct_coords, P.shape, P.chunks, C.chunks)
    report.update({
        "mode": "chunks",
        "predictions": predictions,
        "ct": ct,
        "shape": list(P.shape),
        "prediction_chunk_shape": list(P.chunks),
        "ct_chunk_shape": list(C.chunks),
        "ct_chunks_stored": len(ct_coords),
        "elapsed_seconds": round(time.time() - t0, 1),
    })
    return report


def audit_voxels(predictions: str, ct: str, threshold: int = 127,
                 slab_stride: int = 12, anon: bool = False,
                 storage_options: Optional[Dict[str, Any]] = None,
                 progress: bool = True) -> Dict[str, Any]:
    """Exact voxel-level phantom fraction over chunk-aligned z-slabs.

    A phantom voxel is one above ``threshold`` in the predictions where the CT
    reads exactly 0. Every plane inside a sampled slab is measured; the stride
    controls how many slabs are visited.
    """
    read_options = _read_options((predictions, ct), anon, storage_options)
    P = open_zarr(predictions, mode="r", storage_options=read_options)
    C = open_zarr(ct, mode="r", storage_options=read_options)
    if tuple(P.shape) != tuple(C.shape):
        raise ValueError(
            f"grid mismatch: predictions {tuple(P.shape)} vs CT {tuple(C.shape)} -- "
            f"point --ct at the pyramid level that matches the prediction grid")

    Z, Y, _ = P.shape
    pcz, pcy = P.chunks[0], P.chunks[1]
    step = max(int(slab_stride), 1) * pcz
    # Stripe along Y so peak memory stays near STRIPE_BYTES regardless of slab area.
    per_plane_row = max(1, pcz * P.shape[2])
    ystep = max(pcy, (STRIPE_BYTES // per_plane_row) // pcy * pcy)

    rows = []
    total_pos = total_phantom = 0
    slabs = list(range(0, Z, step))
    t0 = time.time()

    for i, z0 in enumerate(slabs):
        z1 = min(z0 + pcz, Z)
        pos_z = np.zeros(z1 - z0, dtype=np.int64)
        phantom_z = np.zeros(z1 - z0, dtype=np.int64)
        for y0 in range(0, Y, ystep):
            y1 = min(y0 + ystep, Y)
            p = np.asarray(P[z0:z1, y0:y1]) > threshold
            c = np.asarray(C[z0:z1, y0:y1]) > 0
            pos_z += p.sum(axis=(1, 2))
            phantom_z += (p & ~c).sum(axis=(1, 2))
            del p, c
        for k in range(z1 - z0):
            pos = int(pos_z[k])
            phantom = int(phantom_z[k])
            rows.append({"z": int(z0 + k), "positives": pos, "phantom": phantom,
                         "phantom_frac": (phantom / pos) if pos else 0.0})
        total_pos += int(pos_z.sum())
        total_phantom += int(phantom_z.sum())
        if progress:
            frac = total_phantom / total_pos if total_pos else 0.0
            print(f"[{i + 1}/{len(slabs)}] z={z0}:{z1} "
                  f"cumulative phantom fraction {frac:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    phantom_frac = (total_phantom / total_pos) if total_pos else 0.0
    return {
        "mode": "voxels",
        "predictions": predictions,
        "ct": ct,
        "threshold": threshold,
        "slab_stride": int(slab_stride),
        "planes_measured": len(rows),
        "positives": total_pos,
        "phantom": total_phantom,
        "phantom_frac": phantom_frac,
        "support_frac": 1.0 - phantom_frac,
        "elapsed_seconds": round(time.time() - t0, 1),
        "per_plane": rows,
    }


def build_parser() -> HyphenUnderscoreParser:
    parser = HyphenUnderscoreParser(
        description="Audit surface predictions against the CT they were inferred from.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--predictions", required=True,
                       help="path/URL of the prediction array (a single pyramid level)")
        p.add_argument("--ct", required=True,
                       help="path/URL of the masked CT array on the same voxel grid")
        p.add_argument("--output", default=None, help="write the JSON report here")
        p.add_argument("--anon", action="store_true",
                       help="access object storage anonymously (public buckets)")

    chunks = sub.add_parser(
        "chunks", help="zero-download chunk-level audit from stored chunk keys")
    common(chunks)

    voxels = sub.add_parser("voxels", help="exact voxel-level phantom fraction")
    common(voxels)
    voxels.add_argument("--threshold", type=int, default=127,
                        help="prediction value above which a voxel counts as foreground")
    voxels.add_argument("--slab_stride", type=int, default=12,
                        help="measure every Nth chunk-aligned z-slab (default 12)")
    voxels.add_argument("--quiet", action="store_true", help="suppress per-slab progress")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "chunks":
        report = audit_chunks(args.predictions, args.ct, anon=args.anon)
        print(f"prediction chunks: {report['prediction_chunks']:,} | "
              f"supported {report.get('frac_supported', 0):.4f} | "
              f"one-chunk halo {report.get('frac_halo_within_1_chunk', 0):.4f} | "
              f"beyond blend margin {report.get('frac_beyond_blend_margin', 0):.4f} "
              f"({report['beyond_blend_margin']:,} chunks)")
    else:
        report = audit_voxels(args.predictions, args.ct, threshold=args.threshold,
                              slab_stride=args.slab_stride, anon=args.anon,
                              progress=not args.quiet)
        print(f"planes measured: {report['planes_measured']:,} | "
              f"positives {report['positives']:,} | "
              f"phantom {report['phantom']:,} ({report['phantom_frac']:.4f}) | "
              f"support {report['support_frac']:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
