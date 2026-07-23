import numpy as np
import os
import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from tqdm.auto import tqdm
import argparse
import zarr
import fsspec
import numcodecs
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from vesuvius.data.utils import open_zarr
from vesuvius.utils.io.zarr_utils import wait_for_zarr_creation
from vesuvius.utils.k8s import get_tqdm_kwargs


def _canonical_store_path(path: str):
    """Return a comparable local path or normalized remote store URL."""
    if "://" in path:
        scheme, remainder = path.split("://", 1)
        normalized = posixpath.normpath(remainder.replace("\\", "/"))
        return "remote", f"{scheme.lower()}://{normalized.rstrip('/')}"
    resolved = Path(path).resolve(strict=False)
    return "local", os.path.normcase(str(resolved))


def store_paths_overlap(first: str, second: str) -> bool:
    """Return whether two store paths are equal or one contains the other."""
    first_kind, first_path = _canonical_store_path(first)
    second_kind, second_path = _canonical_store_path(second)
    if first_kind != second_kind:
        return False
    separator = "/" if first_kind == "remote" else os.sep
    return (
        first_path == second_path
        or first_path.startswith(second_path + separator)
        or second_path.startswith(first_path + separator)
    )


def validate_output_support_paths(output_path: str, support_volume_path: str) -> None:
    """Reject output stores that could overwrite their support input."""
    if store_paths_overlap(output_path, support_volume_path):
        raise ValueError(
            "output_path must not equal, contain, or be contained by "
            "support_volume_path"
        )


@dataclass
class FinalizeConfig:
    """Bundle all finalization parameters."""
    mode: str = "binary"           # "binary" or "multiclass"
    threshold: Optional[float] = None  # None=probabilities; float in (0,1)=binarize at that probability
    is_multi_task: bool = False
    target_info: Optional[dict] = None
    support_volume_path: Optional[str] = None
    support_threshold: float = 0.0
    support_anon: bool = True


SUPPORT_STAT_COUNT_KEYS = (
    "spatial_voxels",
    "supported_voxels",
    "unsupported_voxels",
    "nonzero_voxels_before",
    "nonzero_voxels_after",
    "nonzero_voxels_removed",
)


def empty_support_mask_stats():
    """Return zeroed counters used to aggregate chunk-level mask statistics."""
    return {key: 0 for key in SUPPORT_STAT_COUNT_KEYS}


def support_mask_stats_with_fraction(totals, *, scope: str):
    """Add the issue #1114 phantom fraction and an explicit counting scope."""
    stats = {key: int(totals[key]) for key in SUPPORT_STAT_COUNT_KEYS}
    positives = stats["nonzero_voxels_before"]
    stats["phantom_fraction_before"] = (
        float(stats["nonzero_voxels_removed"] / positives) if positives else 0.0
    )
    stats["scope"] = scope
    return stats


def apply_support_mask(output_np, support_np, threshold: float = 0.0):
    """Mask finalized predictions where the source volume has no support.

    ``output_np`` is channel-first ``(C, Z, Y, X)`` data. ``support_np`` may
    be either ``(Z, Y, X)`` or a singleton-channel ``(1, Z, Y, X)`` array.
    Non-finite support values and values less than or equal to ``threshold``
    are treated as unsupported.

    Returns a masked copy and spatial-voxel statistics. A spatial voxel is
    counted once even when multiple output channels were non-zero there.
    """
    output_np = np.asarray(output_np)
    support_np = np.asarray(support_np)

    if output_np.ndim != 4:
        raise ValueError(
            f"output_np must have shape (C, Z, Y, X), got {output_np.shape}"
        )
    if support_np.ndim == 4:
        if support_np.shape[0] != 1:
            raise ValueError(
                "4D support arrays must have a singleton channel dimension; "
                f"got {support_np.shape}"
            )
        support_np = support_np[0]
    if support_np.ndim != 3:
        raise ValueError(
            f"support_np must have shape (Z, Y, X) or (1, Z, Y, X), got {support_np.shape}"
        )
    if output_np.shape[1:] != support_np.shape:
        raise ValueError(
            "Support/output spatial shape mismatch: "
            f"{support_np.shape} vs {output_np.shape[1:]}"
        )
    if not np.isfinite(threshold):
        raise ValueError(f"support threshold must be finite, got {threshold}")

    supported = np.isfinite(support_np) & (support_np > threshold)
    nonzero_before = np.any(output_np != 0, axis=0)
    removed = nonzero_before & ~supported
    nonzero_after = nonzero_before & supported

    masked = output_np.copy()
    masked[:, ~supported] = 0
    stats = {
        "spatial_voxels": int(supported.size),
        "supported_voxels": int(np.count_nonzero(supported)),
        "unsupported_voxels": int(np.count_nonzero(~supported)),
        "nonzero_voxels_before": int(np.count_nonzero(nonzero_before)),
        "nonzero_voxels_after": int(np.count_nonzero(nonzero_after)),
        "nonzero_voxels_removed": int(np.count_nonzero(removed)),
    }
    return masked, stats


def open_support_volume(path: str, anon: bool = True):
    """Open a support Zarr array using public-S3 defaults when appropriate."""
    storage_options = {"anon": bool(anon)} if path.startswith("s3://") else None
    return open_zarr(path=path, mode="r", storage_options=storage_options)


_support_store_cache = {}


def get_cached_support_volume(path: str, anon: bool = True):
    """Open a support array once per process and reuse it for later chunks."""
    key = (path, bool(anon))
    if key not in _support_store_cache:
        _support_store_cache[key] = open_support_volume(path, anon=anon)
    return _support_store_cache[key]


def validate_support_volume(support_store, spatial_shape):
    """Validate rank and shape; physical-grid alignment remains caller-asserted."""
    if not hasattr(support_store, "shape"):
        raise ValueError(
            "Support volume must point to a Zarr array (for OME-Zarr, include the "
            "resolution level such as '/2'), not a group."
        )

    shape = tuple(int(v) for v in support_store.shape)
    if len(shape) == 4 and shape[0] == 1:
        support_spatial_shape = shape[1:]
    elif len(shape) == 3:
        support_spatial_shape = shape
    else:
        raise ValueError(
            "Support volume must have shape (Z, Y, X) or (1, Z, Y, X), "
            f"got {shape}"
        )

    expected = tuple(int(v) for v in spatial_shape)
    if support_spatial_shape != expected:
        raise ValueError(
            "Support/output spatial shape mismatch: "
            f"{support_spatial_shape} vs {expected}. Choose the aligned OME-Zarr level."
        )
    return shape


def read_support_chunk(support_store, spatial_slices):
    """Read a spatial chunk from a validated 3D or singleton-channel array."""
    if len(support_store.shape) == 4:
        return np.asarray(support_store[(0,) + tuple(spatial_slices)])
    return np.asarray(support_store[tuple(spatial_slices)])


def apply_finalization(logits_np, num_classes, config: FinalizeConfig):
    """
    Apply softmax + mode logic to normalized logits, producing uint8 output.

    Pure function (no I/O).

    Args:
        logits_np: float32 array (C, Z, Y, X) of normalized logits
        num_classes: number of classes (C dimension)
        config: FinalizeConfig with mode/threshold/multi-task settings

    Returns:
        (output_uint8, is_empty) — finalized uint8 array or (None, True) for empty chunks
    """
    # A zero-filled chunk means no model contributions in this pipeline. Other
    # constant logits are valid predictions and must still be finalized.
    is_empty = not np.any(logits_np)

    if is_empty:
        return None, True

    mode = config.mode
    threshold = config.threshold
    is_multi_task = config.is_multi_task
    target_info = config.target_info

    # Convert probability threshold to a raw-logit cutoff: sigmoid(x) > T  <=>  x > ln(T/(1-T)).
    # This also works for 2-class softmax, where p_fg > T  <=>  logit[1] - logit[0] > ln(T/(1-T)).
    if threshold is not None:
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        logit_cutoff = float(np.log(threshold / (1.0 - threshold)))
    else:
        logit_cutoff = None

    if mode == "binary":
        if is_multi_task and target_info:
            target_results = []
            for target_name, info in sorted(target_info.items(), key=lambda x: x[1]['start_channel']):
                start_ch = info['start_channel']
                end_ch = info['end_channel']
                target_logits = logits_np[start_ch:end_ch]
                if target_logits.shape[0] == 1:
                    logits = target_logits[0].astype(np.float32)
                    if logit_cutoff is not None:
                        target_results.append((logits > logit_cutoff).astype(np.float32))
                    else:
                        target_results.append(1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20))))
                elif logit_cutoff is not None:
                    binary_mask = (target_logits[1] - target_logits[0] > logit_cutoff).astype(np.float32)
                    target_results.append(binary_mask)
                else:
                    exp_logits = np.exp(target_logits - np.max(target_logits, axis=0, keepdims=True))
                    softmax = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
                    target_results.append(softmax[1])
            output_data = np.stack(target_results, axis=0)
        elif num_classes == 1:
            # Single-channel logits: interpret channel as foreground logit.
            logits = logits_np[0].astype(np.float32)
            if logit_cutoff is not None:
                output_data = (logits > logit_cutoff).astype(np.float32)[np.newaxis, ...]
            else:
                probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
                output_data = probs[np.newaxis, ...]
        else:
            if logit_cutoff is not None:
                binary_mask = (logits_np[1] - logits_np[0] > logit_cutoff).astype(np.float32)
                output_data = binary_mask[np.newaxis, ...]
            else:
                exp_logits = np.exp(logits_np - np.max(logits_np, axis=0, keepdims=True))
                softmax = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
                fg_prob = softmax[1:2]
                output_data = fg_prob
    else:  # multiclass
        argmax = np.argmax(logits_np, axis=0).astype(np.float32)
        argmax = argmax[np.newaxis, ...]
        if threshold is not None:
            # In multiclass the CLI only allows the bare-flag form (arrives as 0.5); emit argmax only.
            output_data = argmax
        else:
            exp_logits = np.exp(logits_np - np.max(logits_np, axis=0, keepdims=True))
            softmax = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
            output_data = np.concatenate([softmax, argmax], axis=0)

    output_np = output_data

    if mode == "binary":
        output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
    elif threshold is not None:
        # Thresholded multiclass output is a label map. Preserve the class IDs
        # verbatim so their values do not depend on which labels happen to be
        # present in an individual chunk.
        output_np = np.clip(output_np, 0, 255).astype(np.uint8)
    else:
        # Scale to uint8 range [0, 255]
        min_val = output_np.min()
        max_val = output_np.max()
        if min_val < max_val:
            output_np = ((output_np - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            output_np = np.clip(output_np, 0, 255).astype(np.uint8)

    if mode == "multiclass":
        # A homogeneous nonzero label is still a valid multiclass prediction.
        if not np.any(output_np):
            return None, True

    return output_np, False


def compute_finalized_shape(spatial_shape, num_classes, config: FinalizeConfig):
    """
    Compute the output shape for finalized data based on mode/threshold.

    Args:
        spatial_shape: (Z, Y, X) spatial dimensions
        num_classes: number of input classes
        config: FinalizeConfig with mode/threshold/multi-task settings

    Returns:
        Output shape tuple (C_out, Z, Y, X)
    """
    mode = config.mode
    threshold = config.threshold is not None
    is_multi_task = config.is_multi_task
    target_info = config.target_info

    if mode == "binary":
        if is_multi_task and target_info:
            num_targets = len(target_info)
            return (num_targets, *spatial_shape)
        else:
            return (1, *spatial_shape)
    else:  # multiclass
        if threshold:
            return (1, *spatial_shape)
        else:
            return (num_classes + 1, *spatial_shape)


def process_chunk(
    chunk_info,
    input_path,
    output_path,
    mode,
    threshold,
    num_classes,
    spatial_shape,
    output_chunks,
    is_multi_task=False,
    target_info=None,
    support_volume_path=None,
    support_threshold=0.0,
    support_anon=True,
):
    """
    Process a single chunk of the volume in parallel.

    Args:
        chunk_info: Dictionary with chunk boundaries and indices
        input_path: Path to input zarr
        output_path: Path to output zarr
        mode: Processing mode ("binary" or "multiclass")
        threshold: Optional probability cutoff in (0,1); None emits probabilities
        num_classes: Number of classes in input
        spatial_shape: Spatial dimensions of the volume (Z, Y, X)
        output_chunks: Chunk size for output
        is_multi_task: Whether this is a multi-task model
        target_info: Dictionary with target information for multi-task models
        support_volume_path: Optional aligned CT/support Zarr array
        support_threshold: Support values <= this threshold are masked
        support_anon: Use anonymous credentials for an S3 support volume
    """
    
    chunk_idx = chunk_info['indices']
    
    spatial_slices = tuple(
        slice(idx * chunk, min((idx + 1) * chunk, shape_dim))
        for idx, chunk, shape_dim in zip(chunk_idx, output_chunks[1:], spatial_shape)
    )
    
    input_store = open_zarr(
        path=input_path,
        mode='r',
        storage_options={'anon': False} if input_path.startswith('s3://') else None
    )
    
    output_store = open_zarr(
        path=output_path,
        mode='r+',
        storage_options={'anon': False} if output_path.startswith('s3://') else None
    )
    
    input_slice = (slice(None),) + spatial_slices
    logits_np = input_store[input_slice]

    config = FinalizeConfig(
        mode=mode,
        threshold=threshold,
        is_multi_task=is_multi_task,
        target_info=target_info,
    )
    output_np, is_empty = apply_finalization(logits_np, num_classes, config)

    if is_empty:
        return {
            'chunk_idx': chunk_idx,
            'processed_voxels': 0,
            'empty': True,
            'support_stats': None,
        }

    support_stats = None
    if support_volume_path is not None:
        support_store = get_cached_support_volume(
            support_volume_path,
            anon=support_anon,
        )
        support_np = read_support_chunk(support_store, spatial_slices)
        output_np, support_stats = apply_support_mask(
            output_np,
            support_np,
            threshold=support_threshold,
        )
        if not np.any(output_np):
            return {
                'chunk_idx': chunk_idx,
                'processed_voxels': 0,
                'empty': True,
                'support_stats': support_stats,
            }

    output_slice = (slice(None),) + spatial_slices
    output_store[output_slice] = output_np
    return {
        'chunk_idx': chunk_idx,
        'processed_voxels': int(np.prod(output_np.shape)),
        'support_stats': support_stats,
    }


def finalize_logits(
    input_path: str,
    output_path: str,
    mode: str = "binary",  # "binary" or "multiclass"
    threshold: Optional[float] = None,  # None=probabilities; float in (0,1)=binarize at that probability (multiclass: any non-None => argmax)
    delete_intermediates: bool = False,  # If True, will delete the input logits after processing
    chunk_size: tuple = None,  # Optional custom chunk size for output
    num_workers: int = None,  # Number of worker processes to use
    verbose: bool = True,
    num_parts: int = 1,  # Number of parts to split processing into
    part_id: int = 0,  # Part ID for this process (0-indexed)
    support_volume_path: Optional[str] = None,
    support_threshold: float = 0.0,
    support_anon: bool = True,
):
    """
    Process merged logits and apply softmax/argmax to produce final outputs.

    Args:
        input_path: Path to the merged logits Zarr store
        output_path: Path for the finalized output Zarr store
        mode: "binary" (2 channels) or "multiclass" (>2 channels)
        threshold: Optional probability cutoff in (0,1); None emits probabilities
        delete_intermediates: Whether to delete input logits after processing
        chunk_size: Optional custom chunk size for output (Z,Y,X)
        num_workers: Number of worker processes to use for parallel processing
        verbose: Print progress messages
        num_parts: Number of parts to split the finalization process into
        part_id: Part ID for this process (0-indexed). Used for Z-axis partitioning
        support_volume_path: Optional aligned CT/support Zarr array. Values at
            or below support_threshold are removed from finalized outputs.
        support_threshold: Maximum value considered unsupported.
        support_anon: Use anonymous credentials for an S3 support volume.
    """
    if support_volume_path is not None:
        validate_output_support_paths(output_path, support_volume_path)

    tqdm_kwargs = get_tqdm_kwargs()
    if not verbose:
        tqdm_kwargs['disable'] = True

    numcodecs.blosc.use_threads = False
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() // 2)
    
    print(f"Using {num_workers} worker processes")

    # Add partitioning information
    if num_parts > 1:
        print(f"Partitioned finalization: Processing part {part_id}/{num_parts}")

    compressor = numcodecs.Blosc(
        cname='zstd',
        clevel=1,  # compression level is 1 because we're only using this for mostly empty chunks
        shuffle=numcodecs.blosc.SHUFFLE
    )
    
    threshold_enabled = threshold is not None

    print(f"Opening input logits: {input_path}")
    print(f"Mode: {mode}, Threshold: {threshold if threshold_enabled else 'off'}")
    input_store = open_zarr(
        path=input_path,
        mode='r',
        storage_options={'anon': False} if input_path.startswith('s3://') else None,
        verbose=verbose
    )
    
    input_shape = input_store.shape
    num_classes = input_shape[0]
    spatial_shape = input_shape[1:]  # (Z, Y, X)

    if support_volume_path is not None:
        if mode != "binary":
            raise ValueError("Support masking is currently supported only in binary mode.")
        if not np.isfinite(support_threshold):
            raise ValueError(
                f"support_threshold must be finite, got {support_threshold}"
            )
        support_store = open_support_volume(support_volume_path, anon=support_anon)
        validate_support_volume(support_store, spatial_shape)
        print(
            f"Support mask: {support_volume_path} "
            f"(supported when value > {support_threshold})"
        )
    
    # Check for multi-task metadata
    is_multi_task = False
    target_info = None
    if hasattr(input_store, 'attrs'):
        is_multi_task = input_store.attrs.get('is_multi_task', False)
        target_info = input_store.attrs.get('target_info', None)
    
    # Verify we have the expected number of channels based on mode
    print(f"Input shape: {input_shape}, Num classes: {num_classes}")
    if is_multi_task:
        print(f"Multi-task model detected with targets: {list(target_info.keys()) if target_info else 'None'}")
    
    if mode == "binary":
        if is_multi_task and target_info:
            # For multi-task binary, each target should have 2 channels
            expected_channels = sum(info['out_channels'] for info in target_info.values())
            if num_classes != expected_channels:
                raise ValueError(f"Multi-task binary mode expects {expected_channels} total channels, but input has {num_classes} channels.")
        elif num_classes not in (1, 2):
            raise ValueError(f"Binary mode expects 1 or 2 channels, but input has {num_classes} channels.")
        elif num_classes == 1:
            print("Detected single-channel binary logits. Using sigmoid finalization (or thresholded mask with --threshold).")
    elif mode == "multiclass" and num_classes < 2:
        raise ValueError(f"Multiclass mode expects at least 2 channels, but input has {num_classes} channels.")
    
    if chunk_size is None:
        try:
            src_chunks = input_store.chunks
            # Input chunks include class dimension - extract spatial dimensions
            output_chunks = src_chunks[1:]
            if verbose:
                print(f"Using input chunk size: {output_chunks}")
        except:
            raise ValueError("Cannot determine input chunk size. Please specify --chunk-size.")
    else:
        output_chunks = chunk_size
        if verbose:
            print(f"Using specified chunk size: {output_chunks}")
    
    if mode == "binary":
        if is_multi_task and target_info:
            # For multi-task binary, output one channel per target
            num_targets = len(target_info)
            output_shape = (num_targets, *spatial_shape)  # One mask per target
            if threshold_enabled:
                print(f"Output will have {num_targets} channels: [" + ", ".join(f"{k}_binary_mask" for k in sorted(target_info.keys())) + "]")
            else:
                print(f"Output will have {num_targets} channels: [" + ", ".join(f"{k}_softmax_fg" for k in sorted(target_info.keys())) + "]")
        else:
            if num_classes == 1:
                output_shape = (1, *spatial_shape)
                if threshold_enabled:
                    print("Output will have 1 channel: [binary_mask_from_logit]")
                else:
                    print("Output will have 1 channel: [sigmoid_fg]")
            else:
                output_shape = (1, *spatial_shape)
                if threshold_enabled:
                    print("Output will have 1 channel: [binary_mask]")
                else:
                    print("Output will have 1 channel: [softmax_fg]")
    else:  # multiclass
        if threshold_enabled:
            # If threshold is provided for multiclass, only save the argmax
            output_shape = (1, *spatial_shape)
            print("Output will have 1 channel: [argmax]")
        else:
            # For multiclass, we'll output num_classes channels (all softmax values)
            # Plus 1 channel for the argmax
            output_shape = (num_classes + 1, *spatial_shape)
            print(f"Output will have {num_classes + 1} channels: [softmax_c0...softmax_cN, argmax]")

    output_chunks = (1, *output_chunks)  # Chunk each channel separately

    # --- Create or Open Output Array ---
    if part_id == 0:
        # Part 0 creates the array
        print(f"Creating output store: {output_path}")
        print(f"  Shape: {output_shape}, Chunks: {output_chunks}")

        output_store = open_zarr(
            path=output_path,
            mode='w',
            storage_options={'anon': False} if output_path.startswith('s3://') else None,
            verbose=verbose,
            shape=output_shape,
            chunks=output_chunks,
            dtype=np.uint8,
            compressor=compressor,
            config={'write_empty_chunks': False},
            zarr_format=2,
        )
    else:
        # Other parts wait for part 0 to create the array, then open it in r+ mode
        print(f"Waiting for part 0 to create output array...")

        wait_for_zarr_creation(output_path, verbose=verbose, part_id=part_id)

        print(f"Array found! Opening in r+ mode for part {part_id}")

        # Open existing array in r+ mode (no need to specify shape/chunks)
        output_store = open_zarr(
            path=output_path,
            mode='r+',
            storage_options={'anon': False} if output_path.startswith('s3://') else None,
            verbose=verbose
        )

    def get_chunk_indices(shape, chunks):
        # For each dimension, calculate how many chunks we need
        # Skip first dimension (channels)
        spatial_shape = shape[1:]
        spatial_chunks = chunks[1:]
        
        # Generate all combinations of chunk indices
        from itertools import product
        chunk_counts = [int(np.ceil(s / c)) for s, c in zip(spatial_shape, spatial_chunks)]
        chunk_indices = list(product(*[range(count) for count in chunk_counts]))
        
        # list of dicts with indices for each chunk
        # Each dict will have 'indices' key with the chunk indices
        # we pass these to the worker functions
        chunks_info = []
        for idx in chunk_indices:
            chunks_info.append({'indices': idx})
        
        return chunks_info

    # --- Calculate Z-range for this part ---
    all_chunk_infos = get_chunk_indices(input_shape, output_chunks)

    if num_parts > 1:
        total_z = spatial_shape[0]  # Z dimension
        z_start = (part_id * total_z) // num_parts
        z_end = ((part_id + 1) * total_z) // num_parts
        print(f"Part {part_id} processing Z-range: {z_start} to {z_end} (out of {total_z})")

        # Filter chunks to only include those intersecting with this Z-range
        chunk_infos = []
        for chunk_info in all_chunk_infos:
            z_idx, y_idx, x_idx = chunk_info['indices']
            # Calculate actual Z coordinates for this chunk
            chunk_z_start = z_idx * output_chunks[1]  # output_chunks[1] is Z chunk size
            chunk_z_end = min(chunk_z_start + output_chunks[1], spatial_shape[0])

            # Check if chunk intersects with our Z-range
            if chunk_z_end > z_start and chunk_z_start < z_end:
                chunk_infos.append(chunk_info)

        print(f"Filtered to {len(chunk_infos)} chunks for part {part_id} (from {len(all_chunk_infos)} total)")
    else:
        chunk_infos = all_chunk_infos

    total_chunks = len(chunk_infos)
    print(f"Processing data in {total_chunks} chunks using {num_workers} worker processes...")
    
    # main processing function with partial application of common arguments
    # This allows us to pass only the chunk_info to the worker function
    # and keep the other parameters fixed
    process_chunk_partial = partial(
        process_chunk,
        input_path=input_path,
        output_path=output_path,
        mode=mode,
        threshold=threshold,
        num_classes=num_classes,
        spatial_shape=spatial_shape,
        output_chunks=output_chunks,
        is_multi_task=is_multi_task,
        target_info=target_info,
        support_volume_path=support_volume_path,
        support_threshold=support_threshold,
        support_anon=support_anon,
    )
    
    total_processed = 0
    empty_chunks = 0
    support_totals = empty_support_mask_stats()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:

        future_to_chunk = {executor.submit(process_chunk_partial, chunk): chunk for chunk in chunk_infos}
        
        from concurrent.futures import as_completed
        for future in tqdm(
            as_completed(future_to_chunk),
            total=total_chunks,
            desc="Processing Chunks",
            **tqdm_kwargs,
        ):
            try:
                result = future.result()
                if result.get('empty', False):
                    empty_chunks += 1
                else:
                    total_processed += result['processed_voxels']
                stats = result.get('support_stats')
                if stats:
                    for key in SUPPORT_STAT_COUNT_KEYS:
                        support_totals[key] += int(stats[key])
            except Exception as e:
                print(f"Error processing chunk: {e}")
                raise e
    
    print(f"\nOutput processing complete. Processed {total_chunks - empty_chunks} chunks, skipped {empty_chunks} empty chunks ({empty_chunks/total_chunks:.2%}).")
    if support_volume_path is not None:
        support_summary = support_mask_stats_with_fraction(
            support_totals,
            scope='chunks_with_nonempty_finalized_output',
        )
        print(
            "Support masking removed "
            f"{support_totals['nonzero_voxels_removed']:,} nonzero spatial voxels; "
            f"phantom fraction before masking was "
            f"{support_summary['phantom_fraction_before']:.4%}."
        )

    # Only part 0 updates metadata to avoid conflicts
    if part_id == 0:
        try:
            if hasattr(input_store, 'attrs') and hasattr(output_store, 'attrs'):
                for key in input_store.attrs:
                    output_store.attrs[key] = input_store.attrs[key]

                output_store.attrs['processing_mode'] = mode
                output_store.attrs['threshold_applied'] = threshold
                output_store.attrs['empty_chunks_skipped'] = empty_chunks
                output_store.attrs['total_chunks'] = total_chunks
                output_store.attrs['empty_chunk_percentage'] = float(empty_chunks/total_chunks) if total_chunks > 0 else 0.0
                if support_volume_path is not None:
                    output_store.attrs['support_mask_applied'] = True
                    output_store.attrs['support_volume_path'] = support_volume_path
                    output_store.attrs['support_threshold'] = float(support_threshold)
                    output_store.attrs['support_anonymous_access'] = bool(support_anon)
                    output_store.attrs['support_alignment_validation'] = (
                        'shape_only_physical_alignment_asserted_by_caller'
                    )
                    if num_parts == 1:
                        output_store.attrs['support_mask_stats'] = (
                            support_mask_stats_with_fraction(
                                support_totals,
                                scope='chunks_with_nonempty_finalized_output',
                            )
                        )
        except Exception as e:
            print(f"Warning: Failed to copy metadata: {e}")
    elif verbose:
        print(f"Part {part_id} skipping metadata update (handled by part 0)")

    if delete_intermediates and num_parts == 1:
        # Only delete intermediates when not using partitioning
        print(f"Deleting intermediate logits: {input_path}")
        try:
            # we have to use fsspec for s3/gs/azure paths 
            # os module does not work well with them
            if input_path.startswith(('s3://', 'gs://', 'azure://')):
                fs_protocol = input_path.split('://', 1)[0]
                fs = fsspec.filesystem(fs_protocol, anon=False if fs_protocol == 's3' else None)
                
                # Remove protocol prefix for fs operations
                path_no_prefix = input_path.split('://', 1)[1]
                
                if fs.exists(path_no_prefix):
                    fs.rm(path_no_prefix, recursive=True)
                    print(f"Successfully deleted intermediate logits (remote path)")
            elif os.path.exists(input_path):
                shutil.rmtree(input_path)
                print(f"Successfully deleted intermediate logits (local path)")
        except Exception as e:
            print(f"Warning: Failed to delete intermediate logits: {e}")
            print(f"You may need to delete them manually: {input_path}")
    elif delete_intermediates and num_parts > 1:
        print(f"Skipping intermediate deletion in partitioned mode. Delete manually after all parts complete: {input_path}")

    print(f"Final output saved to: {output_path}")


# --- Shared CLI helpers ---
def add_threshold_arguments(parser):
    """Register the shared `--threshold` / `--threshold-value` arguments on an argparse parser.

    - `--threshold` toggles thresholding on (default cutoff = 0.5).
    - `--threshold-value T` overrides the cutoff (must be in (0, 1)); requires --threshold.
    """
    parser.add_argument('--threshold', dest='threshold', action='store_true',
                        help='Binarize the probability map (default cutoff 0.5). '
                             'In multiclass mode this emits the argmax channel.')
    parser.add_argument('--threshold-value', dest='threshold_value', type=float, default=None,
                        help='Override the probability cutoff used by --threshold '
                             '(float in (0, 1)). Binary mode only.')


def add_support_mask_arguments(parser):
    """Register opt-in source-volume support masking arguments."""
    parser.add_argument(
        '--support-volume',
        dest='support_volume',
        type=str,
        default=None,
        help=(
            'Shape-aligned 3D source/support Zarr array. Finalized predictions are set '
            'to background where its value is <= --support-threshold. For an '
            "OME-Zarr pyramid, include the matching resolution level (for example '/2'). "
            'The caller must assert physical-grid alignment.'
        ),
    )
    parser.add_argument(
        '--support-threshold',
        dest='support_threshold',
        type=float,
        default=0.0,
        help='Maximum source value treated as unsupported. Default: 0.',
    )
    parser.add_argument(
        '--support-authenticated',
        dest='support_authenticated',
        action='store_true',
        help='Use configured credentials for an S3 support volume. Public S3 is anonymous by default.',
    )


def resolve_support_mask(parser, args):
    """Validate support-mask CLI arguments and return path, threshold, anon."""
    if args.support_volume is None:
        if args.support_threshold != 0.0:
            parser.error("--support-threshold requires --support-volume")
        if args.support_authenticated:
            parser.error("--support-authenticated requires --support-volume")
        return None, 0.0, True

    if args.mode != 'binary':
        parser.error("--support-volume is currently supported only with --mode binary")
    if not np.isfinite(args.support_threshold):
        parser.error(
            f"--support-threshold must be finite, got {args.support_threshold}"
        )
    return (
        args.support_volume,
        float(args.support_threshold),
        not args.support_authenticated,
    )


def resolve_threshold(parser, args):
    """Validate --threshold / --threshold-value and return the effective Optional[float] cutoff.

    Returns None if thresholding is disabled, else a float in (0, 1) (0.5 if no override).
    Calls parser.error on invalid combinations.
    """
    if args.threshold_value is not None and not args.threshold:
        parser.error("--threshold-value requires --threshold")
    if args.threshold_value is not None and not (0.0 < args.threshold_value < 1.0):
        parser.error(f"--threshold-value must be in (0, 1), got {args.threshold_value}")
    if args.mode == 'multiclass' and args.threshold_value is not None:
        parser.error("--threshold-value is not applicable in multiclass mode (argmax ignores the cutoff)")
    if not args.threshold:
        return None
    return args.threshold_value if args.threshold_value is not None else 0.5


# --- Command Line Interface ---
def main():
    """Entry point for the vesuvius.finalize command."""
    parser = argparse.ArgumentParser(description='Process merged logits to produce final outputs.')
    parser.add_argument('input_path', type=str,
                      help='Path to the merged logits Zarr store')
    parser.add_argument('output_path', type=str,
                      help='Path for the finalized output Zarr store')
    parser.add_argument('--mode', type=str, choices=['binary', 'multiclass'], default='binary',
                      help='Processing mode. "binary" for 2-class segmentation, "multiclass" for >2 classes. Default: binary')
    add_threshold_arguments(parser)
    add_support_mask_arguments(parser)
    parser.add_argument('--delete-intermediates', dest='delete_intermediates', action='store_true',
                      help='Delete intermediate logits after processing')
    parser.add_argument('--chunk-size', dest='chunk_size', type=str, default=None,
                      help='Spatial chunk size (Z,Y,X) for output Zarr. Comma-separated. If not specified, input chunks will be used.')
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=None,
                      help='Number of worker processes for parallel processing. Default: CPU_COUNT // 2')
    parser.add_argument('--quiet', dest='quiet', action='store_true',
                      help='Suppress verbose output')
    parser.add_argument('--num_parts', type=int, default=1,
                      help='Number of parts to split the finalization process into. Default: 1')
    parser.add_argument('--part_id', type=int, default=0,
                      help='Part ID for this process (0-indexed). Default: 0')

    args = parser.parse_args()

    # Validate partitioning arguments
    if args.part_id < 0 or args.part_id >= args.num_parts:
        parser.error(f"Invalid part_id {args.part_id} for num_parts {args.num_parts}. part_id must be 0 <= part_id < num_parts")

    effective_threshold = resolve_threshold(parser, args)
    support_volume_path, support_threshold, support_anon = resolve_support_mask(
        parser, args
    )

    chunks = None
    if args.chunk_size:
        try:
            chunks = tuple(map(int, args.chunk_size.split(',')))
            if len(chunks) != 3: raise ValueError()
        except ValueError:
            parser.error("Invalid chunk_size format. Expected 3 comma-separated integers (Z,Y,X).")
    
    try:
        finalize_logits(
            input_path=args.input_path,
            output_path=args.output_path,
            mode=args.mode,
            threshold=effective_threshold,
            delete_intermediates=args.delete_intermediates,
            chunk_size=chunks,
            num_workers=args.num_workers,
            verbose=not args.quiet,
            num_parts=args.num_parts,
            part_id=args.part_id,
            support_volume_path=support_volume_path,
            support_threshold=support_threshold,
            support_anon=support_anon,
        )
        return 0
    except Exception as e:
        print(f"\n--- Finalization Failed ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
