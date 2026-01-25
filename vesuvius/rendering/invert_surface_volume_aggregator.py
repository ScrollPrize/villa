#!/usr/bin/env python3
"""
invert_surface_volume_aggregator.py

Aggregate per‐segment Zarr volumes into per‐scroll Zarr volumes by taking the voxelwise maximum
across all segments of a given scroll—and optionally merging in a “predictions” Zarr—using Dask
for parallel, out‐of‐core computation.

Folder structure assumed:
    segments_root/
        <scroll_id>_<segment_id>/
            labels/    ← a uint8 3D Zarr (DirectoryStore) with shape=(Z,Y,X)
            mask/      ← a uint8 3D Zarr (DirectoryStore) with the same shape & chunks

If --predictions-zarr is provided, it should point to a directory containing
<scroll_id_predictions.zarr> stores. Each predictions Zarr must have the same overall shape
as the segment Zarrs, but may use a different chunk layout. For each scroll_id, this script
will produce:
    output_root/
        <scroll_id>_labels.zarr
        <scroll_id>_mask.zarr

Each output Zarr has the same shape, chunk‐layout, compressor, filters, etc., as the
segment inputs, and each voxel in “labels” is the maximum over:
    • all segment‐labels at that position, and
    • the predictions volume (if provided, over the same voxel‐range).

Usage:
    python invert_surface_volume_aggregator.py \
        --segments-root /path/to/segments \
        --output-root   /path/to/aggregated_zarrs \
        [--predictions-zarr /path/to/predictions] \
        [--use-distributed] \
        [--n-workers N] \
        [--threads-per-worker T] \
        [--log-level LEVEL]
"""

import os
import argparse
import logging
from collections import defaultdict
import shutil
from tqdm import tqdm
import numpy as np
import zarr

# Optionally import Dask Distributed components if --use-distributed is requested
try:
    from dask.distributed import Client, LocalCluster
    from dask.diagnostics import ProgressBar
except ImportError:
    Client = None
    LocalCluster = None
    ProgressBar = None


def find_all_segments(segments_root):
    """
    Scan 'segments_root' for subdirectories matching '<scroll_id>_<segment_id>',
    and return a dict: { scroll_id: [full_path_to_each_segment] }.
    """
    segments_by_scroll = defaultdict(list)
    for entry in os.listdir(segments_root):
        entry_path = os.path.join(segments_root, entry)
        if not os.path.isdir(entry_path):
            continue
        if "_" not in entry:
            # Skip folders that don’t follow 'scrollID_segmentID' naming
            continue
        scroll_id, _ = entry.split("_", 1)
        segments_by_scroll[scroll_id].append(entry_path)
    return segments_by_scroll


def aggregate_scroll_with_dask(
    scroll_id,
    segment_paths,
    output_root,
    predictions_root=None,
):
    """
    For a single scroll_id:
      - Gather all 'labels' and 'mask' Zarr paths from segment_paths
      - If predictions_root is provided and a <scroll_id_predictions.zarr> exists,
        open it (must match overall shape, but can have different chunking).
      - Build the union of all segment‐chunk keys. For each chunk key:
          • Compute its voxel‐range [z0:z1, y0:y1, x0:x1)
          • Read all segment blocks that have that chunk, take np.max(...)
          • If predictions exist, slice pred_array[z0:z1, y0:y1, x0:x1] and
            do np.maximum(...) with the segment max.
          • Write the result into the output Zarr if not all fill‐value.
    """

    # 1) Collect all segment 'labels' and 'mask' stores
    label_paths = []
    mask_paths = []
    for seg in segment_paths:
        lbl = os.path.join(seg, "labels")
        msk = os.path.join(seg, "mask")
        if not os.path.isdir(lbl) or not os.path.isdir(msk):
            raise RuntimeError(f"Missing 'labels/' or 'mask/' under {seg!r}")
        label_paths.append(lbl)
        mask_paths.append(msk)

    if len(label_paths) == 0:
        logging.warning(f"[{scroll_id}] No segments found; skipping.")
        return

    # 2) Open the first segment’s labels to grab metadata (shape, chunks, dtype, etc.)
    template_lbl = zarr.open(label_paths[0], mode="r")
    shape = template_lbl.shape
    chunks = template_lbl.chunks
    dtype = template_lbl.dtype
    compressor = template_lbl.compressor
    filters = template_lbl.filters
    fillvalue = template_lbl.fill_value

    # 3) Open + verify every segment’s labels and masks
    seg_lbl_arrays = []
    seg_msk_arrays = []
    for lp, mp in tqdm(
        zip(label_paths, mask_paths),
        total=len(label_paths),
        desc=f"[{scroll_id}] Verifying segments",
    ):
        seg_lbl = zarr.open(lp, mode="r")
        seg_msk = zarr.open(mp, mode="r")
        if seg_lbl.shape != shape or seg_lbl.chunks != chunks:
            raise RuntimeError(
                f"[{scroll_id}] Segment label Zarr at {lp!r} has shape={seg_lbl.shape}, "
                f"chunks={seg_lbl.chunks}, but expected shape={shape}, chunks={chunks}"
            )
        if seg_msk.shape != shape or seg_msk.chunks != chunks:
            raise RuntimeError(
                f"[{scroll_id}] Segment mask  Zarr at {mp!r} has shape={seg_msk.shape}, "
                f"chunks={seg_msk.chunks}, but expected shape={shape}, chunks={chunks}"
            )
        seg_lbl_arrays.append(seg_lbl)
        seg_msk_arrays.append(seg_msk)

    # ───────────────────────────────────────────────────────────────────────
    # 4) Open (and verify) the predictions Zarr, if provided
    pred_array = None
    if predictions_root is not None:
        pred_array = zarr.open(predictions_root, mode="r")
        if pred_array.shape != shape:
            raise RuntimeError(
                f"[{scroll_id}] Predictions Zarr at {predictions_root!r} has shape={pred_array.shape}, "
                f"but expected {shape}."
            )
        if pred_array.dtype != dtype:
            logging.warning(
                f"[{scroll_id}] Predictions Zarr dtype {pred_array.dtype} != segment dtype {dtype}. "
                f"Non‐zero voxels will be cast to {dtype}."
            )
            
    # 5) Build the union of all chunk‐keys from every segment
    chunk_keys_union = set()
    for seg_lbl in seg_lbl_arrays:
        for key in seg_lbl.store.keys():
            if key.startswith("."):
                continue
            chunk_keys_union.add(key)

    if len(chunk_keys_union) == 0:
        logging.warning(f"[{scroll_id}] No chunk files found in any segment; skipping.")
        return

    # 6) Prepare / re‐create the output Zarrs for labels + mask
    os.makedirs(output_root, exist_ok=True)
    out_lbl_path = os.path.join(output_root, f"{scroll_id}_labels.zarr")
    out_msk_path = os.path.join(output_root, f"{scroll_id}_mask.zarr")
    if os.path.exists(out_lbl_path):
        shutil.rmtree(out_lbl_path)
    if os.path.exists(out_msk_path):
        shutil.rmtree(out_msk_path)

    target_lbl = zarr.open(
        out_lbl_path,
        mode="w",
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
        filters=filters,
        fill_value=fillvalue,
    )
    target_msk = zarr.open(
        out_msk_path,
        mode="w",
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
        filters=filters,
        fill_value=fillvalue,
    )

    # 7) Loop over each chunk_key and write the max over segments + predictions
    cz, cy, cx = chunks
    fv = fillvalue

    logging.info(f"[{scroll_id}] Writing chunks (total = {len(chunk_keys_union)})")
    for key in tqdm(sorted(chunk_keys_union), desc=f"[{scroll_id}] Chunks"):
        try:
            bz, by, bx = map(int, key.split("."))
        except ValueError:
            # Skip any unexpected file in store
            continue

        # Compute voxel‐range [z0:z1, y0:y1, x0:x1)
        z0 = bz * cz
        y0 = by * cy
        x0 = bx * cx
        z1 = min(z0 + cz, shape[0])
        y1 = min(y0 + cy, shape[1])
        x1 = min(x0 + cx, shape[2])

        # ───────────────
        # (a) Gather segment “label” blocks that have this key
        block_list_lbl = []
        for seg_lbl in seg_lbl_arrays:
            if key in seg_lbl.store:
                block_arr = seg_lbl[z0:z1, y0:y1, x0:x1]
                block_list_lbl.append(block_arr)

        # (b) Slice the predictions block over the same voxel‐range (if exists)
        if pred_array is not None:
            pred_block = pred_array[z0:z1, y0:y1, x0:x1]
            if pred_block.dtype != dtype:
                pred_block = pred_block.astype(dtype, copy=False)
        else:
            pred_block = None

        # (c) Compute the voxel‐wise max for “labels”
        if block_list_lbl:
            # 1) Max over all segment blocks
            stacked_lbl = np.stack(block_list_lbl, axis=0)
            sub_lbl = np.max(stacked_lbl, axis=0)
            # 2) If predictions exist, include them
            if pred_block is not None:
                sub_lbl = np.maximum(sub_lbl, pred_block)
        else:
            # No segment wrote this chunk; if prediction is all zeros, skip
            if (pred_block is None) or np.all(pred_block == fv):
                continue
            # Otherwise, use the prediction block as-is
            sub_lbl = pred_block

        # (d) Write to target_lbl if not all fill
        if not np.all(sub_lbl == fv):
            target_lbl[z0:z1, y0:y1, x0:x1] = sub_lbl

        # ───────────────
        # Repeat for “mask” (no predictions involved)
        block_list_msk = []
        for seg_msk in seg_msk_arrays:
            if key in seg_msk.store:
                block_arr = seg_msk[z0:z1, y0:y1, x0:x1]
                block_list_msk.append(block_arr)

        if not block_list_msk:
            continue

        stacked_msk = np.stack(block_list_msk, axis=0)
        sub_msk = np.max(stacked_msk, axis=0)
        if not np.all(sub_msk == fv):
            target_msk[z0:z1, y0:y1, x0:x1] = sub_msk

    logging.info(
        f"[{scroll_id}] Done writing:\n"
        f"  • {out_lbl_path}\n"
        f"  • {out_msk_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per‐segment Zarr volumes into per‐scroll Zarr volumes using Dask."
    )
    parser.add_argument(
        "--segments-root", "-i",
        required=True,
        help="Path to the directory containing '<scroll_id>_<segment_id>/' subfolders."
    )
    parser.add_argument(
        "--output-root", "-o",
        required=True,
        help="Directory where '<scroll_id>_labels.zarr' and '<scroll_id>_mask.zarr' will be written."
    )
    parser.add_argument(
        "--predictions-zarr", "-p",
        required=False,
        help="(Optional) Path to a directory containing per‐scroll predictions Zarrs."
    )
    parser.add_argument(
        "--use-distributed",
        action="store_true",
        help="If set, launch a local Dask cluster rather than the default single‐machine scheduler."
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of Dask workers for LocalCluster (only used if --use-distributed). Default: number of CPU cores."
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="Number of threads per Dask worker (only used if --use-distributed). Default: 1."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity."
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    segments_root = args.segments_root
    output_root = args.output_root
    predictions_root = args.predictions_zarr

    if not os.path.isdir(segments_root):
        logging.error(f"Segments root does not exist or is not a directory: {segments_root!r}")
        return

    # 1) Optionally launch a local Dask cluster
    if args.use_distributed:
        if Client is None or LocalCluster is None:
            logging.error("Requested --use-distributed, but dask.distributed is not installed.")
            return
        n_workers = args.n_workers if args.n_workers is not None else os.cpu_count()
        threads_per_worker = args.threads_per_worker
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=True,
            memory_limit="auto"
        )
        client = Client(cluster)
        logging.info(
            f"Started Dask LocalCluster:\n"
            f"    • n_workers = {n_workers}\n"
            f"    • threads_per_worker = {threads_per_worker}\n"
            f"    • dashboard = {cluster.dashboard_link}"
        )
    else:
        logging.info("Using Dask’s default scheduler (no distributed cluster).")

    # 2) Discover all segments and group by scroll_id
    logging.info(f"Scanning '{segments_root}' for segment subfolders...")
    segments_by_scroll = find_all_segments(segments_root)
    if not segments_by_scroll:
        logging.error(f"No valid '<scroll_id>_<segment_id>' subfolders found under {segments_root!r}")
        if args.use_distributed:
            client.close()
            cluster.close()
        return

    # 3) For each scroll_id, call aggregate_scroll_with_dask
    for scroll_id, seg_paths in sorted(segments_by_scroll.items()):
        logging.info(f"→ Aggregating scroll_id = '{scroll_id}' with {len(seg_paths)} segments…")
        try:
            aggregate_scroll_with_dask(
                scroll_id=scroll_id,
                segment_paths=seg_paths,
                output_root=output_root,
                predictions_root=predictions_root
            )
        except Exception as e:
            logging.error(f"[{scroll_id}] Aggregation failed: {e!r}")

    # 4) If we created a Dask Client, shut it down
    if args.use_distributed:
        client.close()
        cluster.close()
        logging.info("Dask LocalCluster closed.")

    logging.info("All done.")


if __name__ == "__main__":
    main()
