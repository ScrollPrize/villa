"""
Torch-free processing utilities for surface volume creation and result reduction.

This module contains functions that don't require PyTorch, allowing CPU-only
tasks (prepare, reduce) to run without loading torch.
"""
import os
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "0")

import logging
import shutil
import math
from typing import List, Tuple, Optional
import concurrent.futures

import numpy as np
import cv2
import zarr
import tifffile as tiff
from numcodecs import LZ4
from tqdm.auto import tqdm
import fsspec

from k8s import get_tqdm_kwargs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def path_exists(path: str) -> bool:
    """Check if path exists (supports local paths and S3 URLs)."""
    storage_options = {"anon": False, "asynchronous": False} if path.startswith("s3://") else {}
    fs, p = fsspec.core.url_to_fs(path, **storage_options)
    return fs.exists(p)


# ----------------------------- Surface Volume Creation ------------------------------

def _read_gray_any(path: str) -> np.ndarray:
    """Read a grayscale image from various formats."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".tif", ".tiff"):
            img = tiff.imread(path)
            if img is None:
                logger.warning(f"Failed to read image: {path}")
                return None
            if img.ndim > 2:
                img = img[..., 0]
            if img.dtype != np.uint8:
                # Minimal, safe conversion to uint8
                img = np.clip(img, 0, 255).astype(np.uint8)
            return img
        else:
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    except Exception:
        logger.exception(f"Exception reading image: {path}")
        return None


def create_surface_volume_zarr(
    layer_paths: List[str],
    output_path: str,
    chunk_size: int = 1024,
    max_workers: Optional[int] = None
) -> str:
    """
    Create a surface volume zarr array from a list of layer image files.

    Args:
        layer_paths: List of paths to layer image files (must be sorted in layer order)
        output_path: Path where the zarr array will be created
        chunk_size: Chunk size for zarr array (default: 1024x1024x1)
        max_workers: Number of worker threads for parallel reading (default: auto)

    Returns:
        Path to the created zarr array (same as output_path)

    Raises:
        RuntimeError: If image reading fails or size mismatches occur
    """
    if not layer_paths:
        raise ValueError("layer_paths cannot be empty")

    # Read first image to get dimensions
    first = _read_gray_any(layer_paths[0])
    if first is None:
        raise RuntimeError(f"Failed to read image: {layer_paths[0]}")
    h, w = first.shape
    c = len(layer_paths)

    # Check disk space (estimate with 50% compression ratio for LZ4)
    zarr_root = os.path.dirname(output_path) or "/tmp"
    uncompressed_gib = (h * w * c) / (1024**3)
    estimated_gib = uncompressed_gib * 0.5  # LZ4 typically achieves ~50% compression on uint8 images
    free_gib = shutil.disk_usage(zarr_root).free / (1024**3)
    logger.info(
        f"Creating surface volume zarr at {output_path} (H,W,C={h},{w},{c} ~{uncompressed_gib:.2f} GiB uncompressed, "
        f"~{estimated_gib:.2f} GiB estimated with LZ4). Free on {zarr_root}: {free_gib:.2f} GiB."
    )
    if free_gib < estimated_gib * 1.10:
        raise RuntimeError(
            f"Not enough space on {zarr_root}: need ~{estimated_gib:.2f} GiB, have {free_gib:.2f} GiB"
        )

    # Remove existing zarr if present
    if os.path.exists(output_path):
        logger.warning(f"Removing existing zarr at {output_path}")
        shutil.rmtree(output_path)

    # Create zarr v2 array with shape (H, W, C) and chunk size optimized for spatial tile access
    # Using LZ4 compression (fast with decent ratio)
    z = zarr.open(
        output_path,
        mode="w",
        shape=(h, w, c),
        chunks=(chunk_size, chunk_size, 1),
        dtype=np.uint8,
        compressor=LZ4(acceleration=1),  # acceleration=1 is default, good balance of speed/ratio
        zarr_format=2,
        config={'write_empty_chunks': False}  # Don't write chunks that are entirely fill_value
    )

    # Parallel reads with batching to limit memory usage
    if max_workers is None:
        max_workers = int(os.environ.get("MMAP_READ_WORKERS", str(min(6, (os.cpu_count() or 4)))))
    # Limit in-flight futures to prevent memory buildup
    batch_size = max_workers * 2

    def _read_and_write(idx_path):
        """Read image and write directly to zarr, return only index."""
        idx, p = idx_path
        img = _read_gray_any(p)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        if img.shape != (h, w):
            raise RuntimeError(f"Layer size mismatch: {p} has {img.shape}, expected {(h, w)}")
        z[:, :, idx] = img  # write to zarr immediately
        del img  # release memory ASAP
        return idx

    with tqdm(total=c, desc="Building surface volume", unit="layer", **get_tqdm_kwargs()) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            # Process in batches to limit memory
            for batch_start in range(0, c, batch_size):
                batch_end = min(batch_start + batch_size, c)
                batch_paths = [(i, layer_paths[i]) for i in range(batch_start, batch_end)]
                futures = [ex.submit(_read_and_write, ip) for ip in batch_paths]

                for fut in concurrent.futures.as_completed(futures):
                    idx = fut.result()
                    pbar.update(1)

    logger.info(f"Surface volume zarr created successfully at {output_path}")
    return output_path


# ----------------------------- Partition Reduction ------------------------------

def reduce_partitions(
    zarr_output_dir: str,
    num_parts: int,
    pred_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Reduce/blend all partition zarr arrays into a single prediction.

    Args:
        zarr_output_dir: Directory containing partition zarr arrays
        num_parts: Total number of partitions
        pred_shape: Output shape (H, W)

    Returns:
        Blended numpy array (H, W) with values in [0, 1]
    """
    try:
        H, W = pred_shape
        logger.info(f"Starting reduce phase: blending {num_parts} partitions from {zarr_output_dir}")

        # Initialize accumulators in memory (we're blending, so need the full arrays)
        total_pred = np.zeros((H, W), dtype=np.float32)
        total_count = np.zeros((H, W), dtype=np.float32)

        # Process each partition in chunks to avoid memory issues
        chunk_size = 1024
        with tqdm(total=num_parts, desc="Blending partitions", unit="partition", **get_tqdm_kwargs()) as pbar:
            for part_id in range(num_parts):
                mask_pred_path = os.path.join(zarr_output_dir, f"mask_pred_part_{part_id:03d}.zarr")
                mask_count_path = os.path.join(zarr_output_dir, f"mask_count_part_{part_id:03d}.zarr")

                if not os.path.exists(mask_pred_path) or not os.path.exists(mask_count_path):
                    logger.warning(f"Missing partition {part_id} at {zarr_output_dir}, skipping")
                    pbar.update(1)
                    continue

                mask_pred_z = zarr.open(mask_pred_path, mode='r')
                mask_count_z = zarr.open(mask_count_path, mode='r')

                # Process in chunks to avoid loading entire partition into memory
                for y in range(0, H, chunk_size):
                    y_end = min(y + chunk_size, H)
                    for x in range(0, W, chunk_size):
                        x_end = min(x + chunk_size, W)

                        # Read chunks and accumulate
                        pred_chunk = mask_pred_z[y:y_end, x:x_end]
                        count_chunk = mask_count_z[y:y_end, x:x_end]

                        total_pred[y:y_end, x:x_end] += pred_chunk
                        total_count[y:y_end, x:x_end] += count_chunk

                pbar.update(1)

        # Blend: divide total_pred by total_count
        logger.info("Blending accumulated predictions")
        result = total_pred / np.clip(total_count, 1e-6, None)
        result = np.clip(result, 0, 1)

        logger.info(f"Reduce completed. Final prediction shape: {result.shape}")
        return result

    except Exception as e:
        logger.error(f"Error in reduce_partitions: {e}")
        raise


# ----------------------------- TIFF Writing ------------------------------

def write_tiled_tiff(prediction: np.ndarray, output_path: str) -> None:
    """
    Write prediction array to a tiled TIFF file.

    Args:
        prediction: Float32 array with values in [0, 1], shape (H, W)
        output_path: Path to output TIFF file
    """
    try:
        logger.info(f"Writing tiled TIFF to {output_path}")

        # Convert float32 [0, 1] to uint8 [0, 255]
        pred_uint8 = (np.clip(prediction, 0, 1) * 255).astype(np.uint8)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write with tiling and compression
        tiff.imwrite(
            output_path,
            pred_uint8,
            compression='deflate',
            tile=(256, 256),
            metadata={'software': 'optimized_inference'}
        )

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Wrote tiled TIFF: {output_path} ({file_size_mb:.2f} MB)")

    except Exception as e:
        logger.error(f"Error writing tiled TIFF: {e}")
        raise
