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
from zarr.experimental.cache_store import CacheStore
from zarr.storage import LocalStore, FsspecStore

from k8s import get_tqdm_kwargs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def path_exists(path: str) -> bool:
    """Check if path exists (supports local paths and S3 URLs)."""
    if path.startswith("s3://"):
        import s3fs
        fs = s3fs.S3FileSystem(anon=False)
        return fs.exists(path)
    return os.path.exists(path)


def get_zarr_store(path: str):
    """Get zarr store for path (supports local paths and S3 URLs).

    For S3 paths, automatically applies disk-based caching using zarr3's CacheStore
    to improve performance and reduce S3 request costs.
    """
    if path.startswith("s3://"):
        # Create S3 filesystem with credentials
        fs = fsspec.filesystem(
            's3',
            anon=False,
            asynchronous=True,
            s3_additional_kwargs={'StorageClass': 'INTELLIGENT_TIERING'}
        )

        # Remove s3:// prefix to get the path for FsspecStore
        s3_path = path[5:]  # Remove 's3://'

        # Create base S3 store using zarr3's FsspecStore
        base_store = FsspecStore(fs=fs, path=s3_path, read_only=True)

        # Configure cache settings from environment variables
        cache_dir = os.environ.get("ZARR_CACHE_DIR", "./zarr_cache")
        cache_size_gb = float(os.environ.get("ZARR_CACHE_SIZE_GB", "100"))
        cache_max_age = os.environ.get("ZARR_CACHE_MAX_AGE", "infinity")

        # Convert cache size from GB to bytes
        cache_size_bytes = int(cache_size_gb * 1024 * 1024 * 1024)

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        # Create LocalStore for persistent disk-based caching
        cache_store = LocalStore(cache_dir)

        # Wrap S3 store with CacheStore for automatic caching
        cached_store = CacheStore(
            store=base_store,
            cache_store=cache_store,
            max_size=cache_size_bytes,
            max_age_seconds=cache_max_age
        )

        logger.info(
            f"Enabled zarr3 disk cache for S3 path: {path}\n"
            f"  Cache dir: {cache_dir}\n"
            f"  Cache size limit: {cache_size_gb} GB\n"
            f"  Cache max age: {cache_max_age}"
        )

        return cached_store

    return path


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
    max_workers: Optional[int] = None,
    use_compression: bool = True
) -> str:
    """
    Create a surface volume zarr array from a list of layer image files.

    Args:
        layer_paths: List of paths to layer image files (must be sorted in layer order)
        output_path: Path where the zarr array will be created (supports local paths and S3 URLs)
        chunk_size: Chunk size for zarr array (default: 1024x1024x1)
        max_workers: Number of worker threads for parallel reading (default: auto)
        use_compression: Enable LZ4 compression (default: True)

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

    # Determine if we're writing to S3 or local
    is_s3 = output_path.startswith("s3://")

    # Calculate size estimates
    uncompressed_gib = (h * w * c) / (1024**3)
    estimated_gib = uncompressed_gib * 0.5  # LZ4 typically achieves ~50% compression on uint8 images

    # Check disk space only for local paths
    if not is_s3:
        zarr_root = os.path.dirname(output_path) or "/tmp"
        free_gib = shutil.disk_usage(zarr_root).free / (1024**3)
        logger.info(
            f"Creating surface volume zarr at {output_path} (H,W,C={h},{w},{c} ~{uncompressed_gib:.2f} GiB uncompressed, "
            f"~{estimated_gib:.2f} GiB estimated with LZ4). Free on {zarr_root}: {free_gib:.2f} GiB."
        )
        if free_gib < estimated_gib * 1.10:
            raise RuntimeError(
                f"Not enough space on {zarr_root}: need ~{estimated_gib:.2f} GiB, have {free_gib:.2f} GiB"
            )
    else:
        logger.info(
            f"Creating surface volume zarr at {output_path} (H,W,C={h},{w},{c} ~{uncompressed_gib:.2f} GiB uncompressed, "
            f"~{estimated_gib:.2f} GiB estimated with LZ4). Writing to S3 with INTELLIGENT_TIERING."
        )

    # Fail if zarr already exists
    if path_exists(output_path):
        raise RuntimeError(f"Surface volume zarr already exists at {output_path}. Please remove it or use a different path.")

    # Get zarr store (handles both local and S3)
    store = get_zarr_store(output_path)

    # Create zarr v2 array with shape (H, W, C) and chunk size optimized for spatial tile access
    # Optionally use LZ4 compression (fast with decent ratio)
    compressor = LZ4(acceleration=1) if use_compression else None
    compression_msg = "with LZ4 compression" if use_compression else "without compression"
    logger.info(f"Creating zarr array {compression_msg}")

    z = zarr.open(
        store,
        mode="w",
        shape=(h, w, c),
        chunks=(chunk_size, chunk_size, 1),
        dtype=np.uint8,
        compressor=compressor,
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
    pred_shape: Tuple[int, int],
    tile_size: int
) -> Tuple:
    """
    Create a lazy tile generator that blends partition zarr arrays tile-by-tile.

    Args:
        zarr_output_dir: Directory containing partition zarr arrays
        num_parts: Total number of partitions
        pred_shape: Output shape (H, W)
        tile_size: Size of tiles to process (default 1024)

    Returns:
        Tuple of (tile_generator, shape) where tile_generator yields uint8 tiles lazily
    """
    H, W = pred_shape
    logger.info(f"Starting reduce phase: will blend {num_parts} partitions tile-by-tile (tile_size={tile_size})")

    # Open all partition zarr arrays once (outside the generator loop)
    logger.info(f"Opening {num_parts} partition zarr arrays...")
    partition_zarrs = []
    for part_id in range(num_parts):
        mask_pred_path = os.path.join(zarr_output_dir, f"mask_pred_part_{part_id:03d}.zarr")
        mask_count_path = os.path.join(zarr_output_dir, f"mask_count_part_{part_id:03d}.zarr")

        if not os.path.exists(mask_pred_path) or not os.path.exists(mask_count_path):
            logger.warning(f"Missing partition {part_id} at {zarr_output_dir}, skipping")
            continue

        # Open zarr arrays once and store references
        mask_pred_z = zarr.open(mask_pred_path, mode='r')
        mask_count_z = zarr.open(mask_count_path, mode='r')
        partition_zarrs.append((mask_pred_z, mask_count_z))

    logger.info(f"Successfully opened {len(partition_zarrs)} partition zarr arrays")

    def tile_generator():
        """Lazily generate tiles by blending partitions on-the-fly."""
        total_tiles = ((H + tile_size - 1) // tile_size) * ((W + tile_size - 1) // tile_size)

        with tqdm(total=total_tiles, desc="Processing tiles", unit="tile", **get_tqdm_kwargs()) as pbar:
            # Outer loop: iterate over tile positions
            for y in range(0, H, tile_size):
                y_end = min(y + tile_size, H)
                tile_h = y_end - y

                for x in range(0, W, tile_size):
                    x_end = min(x + tile_size, W)
                    tile_w = x_end - x

                    # Create tile-sized accumulators
                    tile_pred = np.zeros((tile_h, tile_w), dtype=np.float32)
                    tile_count = np.zeros((tile_h, tile_w), dtype=np.float32)

                    # Inner loop: accumulate from all partitions for this tile
                    for mask_pred_z, mask_count_z in partition_zarrs:
                        # Read tile from pre-opened zarr arrays
                        pred_chunk = mask_pred_z[y:y_end, x:x_end]
                        count_chunk = mask_count_z[y:y_end, x:x_end]

                        # Accumulate
                        tile_pred += pred_chunk
                        tile_count += count_chunk

                    # Blend: divide, clip, convert to uint8
                    result_tile = tile_pred / np.clip(tile_count, 1e-6, None)
                    result_tile = np.clip(result_tile, 0, 1)
                    result_uint8 = (result_tile * 255).astype(np.uint8)

                    # Yield the tile
                    yield result_uint8
                    pbar.update(1)

    return tile_generator(), pred_shape


# ----------------------------- TIFF Writing ------------------------------

def write_tiled_tiff(tile_iterator, shape: Tuple[int, int], output_path: str, tile_size: int = 1024) -> None:
    """
    Write tiles from iterator to a tiled TIFF file.

    Args:
        tile_iterator: Iterator that yields uint8 tiles
        shape: Output shape (H, W)
        output_path: Path to output TIFF file
        tile_size: Size of tiles (default 1024)
    """
    try:
        H, W = shape
        logger.info(f"Writing tiled TIFF to {output_path} with shape ({H}, {W}) and tile size {tile_size}x{tile_size}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # Write with tiling and compression using iterator
        tiff.imwrite(
            output_path,
            tile_iterator,
            shape=(H, W),
            dtype=np.uint8,
            compression='deflate',
            tile=(tile_size, tile_size),
            metadata={'software': 'optimized_inference'}
        )

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Wrote tiled TIFF: {output_path} ({file_size_mb:.2f} MB)")

    except Exception as e:
        logger.error(f"Error writing tiled TIFF: {e}")
        raise
