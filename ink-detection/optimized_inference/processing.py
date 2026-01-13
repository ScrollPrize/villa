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
from vendored_cache_store import CacheStore
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


def get_cached_zarr_store(path: str):
    """Get read-only cached zarr store for path (supports local paths and S3 URLs).

    This function is intended for read operations with performance optimization.
    For S3 paths, automatically applies disk-based caching using zarr3's CacheStore
    to improve performance and reduce S3 request costs.

    For write operations, use get_writable_zarr_store() instead.

    Args:
        path: Local file path or S3 URL (s3://bucket/path)

    Returns:
        Cached zarr store (CacheStore for S3, path string for local)
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


def get_writable_zarr_store(path: str):
    """Get writable zarr store for path (supports local paths and S3 URLs).

    This function is intended for write operations and does not apply caching.
    For read-only operations with caching, use get_cached_zarr_store() instead.

    Args:
        path: Local file path or S3 URL (s3://bucket/path)

    Returns:
        Writable zarr store (fsspec mapper for S3, path string for local)
    """
    if path.startswith("s3://"):
        # Return writable fsspec mapper for S3
        return fsspec.get_mapper(path, anon=False, s3_additional_kwargs={'StorageClass': 'INTELLIGENT_TIERING'})
    return path


def get_zarr_store(path: str):
    """Get zarr store for path (supports local paths and S3 URLs).

    .. deprecated::
        Use get_cached_zarr_store() for read operations or get_writable_zarr_store()
        for write operations instead. This function defaults to cached read behavior
        for backwards compatibility.

    Args:
        path: Local file path or S3 URL (s3://bucket/path)

    Returns:
        Cached zarr store (defaults to read-only cached behavior)
    """
    import warnings
    warnings.warn(
        "get_zarr_store() is deprecated. Use get_cached_zarr_store() for reads "
        "or get_writable_zarr_store() for writes.",
        DeprecationWarning,
        stacklevel=2
    )
    return get_cached_zarr_store(path)


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

    # Get writable zarr store (handles both local and S3)
    store = get_writable_zarr_store(output_path)

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

def create_scale_bar_tiles(
    image_w: int,
    image_h: int,
    tile_size: int,
    pixel_resolution_um: float,
    scale_bar_length_um: float,
    padding: int = 50
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Pre-render scale bar on a strip and split into tiles with alpha masks.

    The scale bar will be positioned at the top-left of the image and span
    multiple tiles if needed. Returns both the scale bar tiles and mask tiles
    for proper alpha compositing.

    Args:
        image_w: Full image width
        image_h: Full image height
        tile_size: Size of tiles (e.g., 1024)
        pixel_resolution_um: Real-world pixel resolution in micrometers
        scale_bar_length_um: Desired scale bar length in micrometers
        padding: Padding from edges in pixels (default 50)

    Returns:
        Tuple of (scale_bar_tiles, mask_tiles):
        - scale_bar_tiles: List of uint8 arrays (tile_size, tile_size) with scale bar rendered
        - mask_tiles: List of uint8 arrays (tile_size, tile_size) where 255=scale bar, 0=transparent
        Returns ([], []) if scale bar doesn't fit or isn't needed.
    """
    # Calculate scale bar dimensions in pixels
    scale_bar_width_px = int(scale_bar_length_um / pixel_resolution_um)
    line_thickness = 4  # pixels - uniform thickness for all lines (bar and ticks)
    tick_height_major = 60  # pixels - height for major ticks (0 and end), symmetric around bar
    tick_height_medium = 40  # pixels - height for medium ticks (middle)
    tick_height_minor = 25  # pixels - height for minor ticks (others)
    text_height = 25  # pixels for font

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2

    # Calculate tick mark interval (1mm for mm scale, 100um for um scale)
    if scale_bar_length_um >= 1000:
        tick_interval_um = 1000  # 1mm
        unit = "mm"
    else:
        tick_interval_um = 100  # 100um
        unit = "um"

    # Generate labels for each tick mark
    num_intervals = int(scale_bar_length_um / tick_interval_um)
    labels_with_positions = []

    for i in range(num_intervals + 1):
        # Position of this tick
        tick_x = int((i * tick_interval_um / scale_bar_length_um) * scale_bar_width_px)

        # Generate label
        if i == 0:
            label = "0"
        elif i == num_intervals:
            # Last tick - show value with unit
            value = scale_bar_length_um / (1000.0 if unit == "mm" else 1.0)
            if value == int(value):
                label = f"{int(value)}{unit}"
            else:
                label = f"{value:.1f}{unit}"
        else:
            # Middle ticks - just show the number
            value = i * tick_interval_um / (1000.0 if unit == "mm" else 1.0)
            if value == int(value):
                label = f"{int(value)}"
            else:
                label = f"{value:.1f}"

        # Determine alignment
        if i == 0:
            alignment = "left"
        elif i == num_intervals:
            alignment = "right"
        else:
            alignment = "center"

        labels_with_positions.append((tick_x, label, alignment))

    # Calculate total box dimensions
    box_width = scale_bar_width_px + 2 * padding
    box_height = tick_height_major + text_height + 3 * padding  # major tick (symmetric), text, spacing

    # Check if scale bar would exceed image width - if so, scale it down
    max_allowed_width = image_w - 2 * padding
    if box_width > max_allowed_width:
        # Scale down the scale bar
        scale_factor = max_allowed_width / box_width
        scale_bar_length_um = scale_bar_length_um * scale_factor
        scale_bar_width_px = int(scale_bar_length_um / pixel_resolution_um)

        # Recalculate labels and tick interval
        if scale_bar_length_um >= 1000:
            tick_interval_um = 1000  # 1mm
            unit = "mm"
        else:
            tick_interval_um = 100  # 100um
            unit = "um"

        # Regenerate labels for each tick mark
        num_intervals = int(scale_bar_length_um / tick_interval_um)
        labels_with_positions = []

        for i in range(num_intervals + 1):
            tick_x = int((i * tick_interval_um / scale_bar_length_um) * scale_bar_width_px)

            if i == 0:
                label = "0"
            elif i == num_intervals:
                value = scale_bar_length_um / (1000.0 if unit == "mm" else 1.0)
                label = f"{int(value)}{unit}" if value == int(value) else f"{value:.1f}{unit}"
            else:
                value = i * tick_interval_um / (1000.0 if unit == "mm" else 1.0)
                label = f"{int(value)}" if value == int(value) else f"{value:.1f}"

            alignment = "left" if i == 0 else ("right" if i == num_intervals else "center")
            labels_with_positions.append((tick_x, label, alignment))

        box_width = scale_bar_width_px + 2 * padding
        logger.warning(f"Scale bar too wide for image, scaled down to {scale_bar_length_um:.0f}um")

    # Calculate how many horizontal tiles we need
    num_tiles = math.ceil(box_width / tile_size)

    # Create canvases: width = num_tiles * tile_size, height = tile_size
    canvas_width = num_tiles * tile_size
    canvas_height = tile_size

    # Scale bar canvas (contains the actual rendered scale bar)
    scale_bar_canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    # Mask canvas (255 where scale bar exists, 0 elsewhere)
    mask_canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    # Calculate scale bar position on canvas (top-left with padding)
    box_x = padding
    box_y = padding

    # Draw black background box on scale bar canvas
    bg_box_x = box_x
    bg_box_y = box_y
    bg_box_w = scale_bar_width_px
    bg_box_h = box_height - 2 * padding

    cv2.rectangle(scale_bar_canvas,
                  (bg_box_x, bg_box_y),
                  (bg_box_x + bg_box_w, bg_box_y + bg_box_h),
                  0, -1)

    # Mark the same region in mask canvas
    cv2.rectangle(mask_canvas,
                  (bg_box_x, bg_box_y),
                  (bg_box_x + bg_box_w, bg_box_y + bg_box_h),
                  255, -1)

    # Draw main horizontal white bar (centered vertically in the major tick area)
    bar_x = bg_box_x
    bar_y = bg_box_y + tick_height_major // 2  # Center the bar in the middle of major tick area
    cv2.line(scale_bar_canvas,
             (bar_x, bar_y),
             (bar_x + scale_bar_width_px, bar_y),
             255, line_thickness, cv2.LINE_AA)

    # Draw tick marks at regular intervals with different heights
    num_intervals = int(scale_bar_length_um / tick_interval_um)

    for i in range(num_intervals + 1):
        tick_x = bar_x + int((i * tick_interval_um / scale_bar_length_um) * scale_bar_width_px)

        # Determine tick height based on position
        if i == 0 or i == num_intervals:
            # Major ticks at start and end
            half_tick_height = tick_height_major // 2
        elif i == num_intervals // 2:
            # Medium tick at middle
            half_tick_height = tick_height_medium // 2
        else:
            # Minor ticks everywhere else
            half_tick_height = tick_height_minor // 2

        # Draw vertical tick mark symmetrically around the bar
        cv2.line(scale_bar_canvas,
                 (tick_x, bar_y - half_tick_height),
                 (tick_x, bar_y + half_tick_height),
                 255, line_thickness, cv2.LINE_AA)

    # Draw text labels below the bar
    label_y = bar_y + tick_height_major // 2 + text_height

    for tick_x_offset, label, alignment in labels_with_positions:
        tick_x_abs = bar_x + tick_x_offset
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Calculate x position based on alignment
        if alignment == "left":
            text_x = tick_x_abs
        elif alignment == "center":
            text_x = tick_x_abs - text_w // 2
        else:  # right
            text_x = tick_x_abs - text_w

        cv2.putText(scale_bar_canvas, label,
                    (text_x, label_y),
                    font, font_scale, 255, font_thickness, cv2.LINE_AA)

    # Split both canvases into tiles
    scale_bar_tiles = []
    mask_tiles = []
    for i in range(num_tiles):
        tile_x_start = i * tile_size
        tile_x_end = tile_x_start + tile_size

        scale_bar_tile = scale_bar_canvas[:tile_size, tile_x_start:tile_x_end].copy()
        mask_tile = mask_canvas[:tile_size, tile_x_start:tile_x_end].copy()

        scale_bar_tiles.append(scale_bar_tile)
        mask_tiles.append(mask_tile)

    logger.info(f"Pre-rendered scale bar across {num_tiles} tiles ({box_width}px wide)")
    return scale_bar_tiles, mask_tiles


def reduce_partitions(
    zarr_output_dir: str,
    num_parts: int,
    pred_shape: Tuple[int, int],
    tile_size: int,
    add_scale_bar: bool = False,
    pixel_resolution_um: Optional[float] = None,
    scale_bar_length_um: float = 10000.0
) -> Tuple:
    """
    Create a lazy tile generator that blends partition zarr arrays tile-by-tile.

    Args:
        zarr_output_dir: Directory containing partition zarr arrays
        num_parts: Total number of partitions
        pred_shape: Output shape (H, W)
        tile_size: Size of tiles to process (default 1024)
        add_scale_bar: Whether to overlay scale bar on output (default False)
        pixel_resolution_um: Real-world pixel resolution in micrometers (None to skip scale bar)
        scale_bar_length_um: Scale bar length in micrometers (default 10000 = 1cm)

    Returns:
        Tuple of (tile_generator, shape) where tile_generator yields uint8 tiles lazily
    """
    H, W = pred_shape
    logger.info(f"Starting reduce phase: will blend {num_parts} partitions tile-by-tile (tile_size={tile_size})")

    # Cache directory for partition zarrs (network filesystem -> local /tmp)
    cache_dir = "/tmp/partition_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Open all partition zarr arrays once (outside the generator loop)
    logger.info(f"Caching and opening {num_parts} partition zarr arrays in parallel...")

    def prepare_partition(part_id):
        """Cache and open a single partition's zarr arrays."""
        mask_pred_path = os.path.join(zarr_output_dir, f"mask_pred_part_{part_id:03d}.zarr")
        mask_count_path = os.path.join(zarr_output_dir, f"mask_count_part_{part_id:03d}.zarr")

        if not os.path.exists(mask_pred_path) or not os.path.exists(mask_count_path):
            logger.warning(f"Missing partition {part_id} at {zarr_output_dir}, skipping")
            return None

        # Copy zarr directories to local cache if not already cached
        cached_pred_path = os.path.join(cache_dir, f"mask_pred_part_{part_id:03d}.zarr")
        cached_count_path = os.path.join(cache_dir, f"mask_count_part_{part_id:03d}.zarr")

        if not os.path.exists(cached_pred_path):
            shutil.copytree(mask_pred_path, cached_pred_path)

        if not os.path.exists(cached_count_path):
            shutil.copytree(mask_count_path, cached_count_path)

        # Open zarr arrays from cache
        mask_pred_z = zarr.open(cached_pred_path, mode='r')
        mask_count_z = zarr.open(cached_count_path, mode='r')

        return (mask_pred_z, mask_count_z)

    # Use ThreadPoolExecutor for parallel I/O
    partition_zarrs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, num_parts)) as executor:
        # Submit all tasks
        futures = {executor.submit(prepare_partition, part_id): part_id for part_id in range(num_parts)}

        # Collect results with progress bar
        with tqdm(total=num_parts, desc="Preparing partitions", unit="partition", **get_tqdm_kwargs()) as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    partition_zarrs.append(result)
                pbar.update(1)

    logger.info(f"Successfully cached and opened {len(partition_zarrs)} partition zarr arrays")

    # Pre-render scale bar tiles if enabled
    scale_bar_tiles = []
    mask_tiles = []
    if add_scale_bar and pixel_resolution_um is not None:
        scale_bar_tiles, mask_tiles = create_scale_bar_tiles(
            W, H, tile_size, pixel_resolution_um, scale_bar_length_um
        )

    def tile_generator():
        """Lazily generate tiles by blending partitions on-the-fly."""
        total_tiles = ((H + tile_size - 1) // tile_size) * ((W + tile_size - 1) // tile_size)

        with tqdm(total=total_tiles, desc="Processing tiles", unit="tile", **get_tqdm_kwargs()) as pbar:
            # Outer loop: iterate over tile positions
            tile_idx_x = 0  # Track which tile we're at horizontally
            for y in range(0, H, tile_size):
                tile_idx_x = 0  # Reset for each row
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

                    # Overlay scale bar if this is the first row and we have a scale bar tile for this position
                    if y == 0 and scale_bar_tiles and tile_idx_x < len(scale_bar_tiles):
                        scale_tile = scale_bar_tiles[tile_idx_x][:tile_h, :tile_w]
                        mask_tile = mask_tiles[tile_idx_x][:tile_h, :tile_w]

                        # Alpha compositing using mask: where mask==255, use scale_bar; else use original
                        result_uint8 = np.where(mask_tile == 255, scale_tile, result_uint8)

                    # Yield the tile
                    yield result_uint8
                    pbar.update(1)
                    tile_idx_x += 1

    return tile_generator(), pred_shape


# ----------------------------- TIFF Writing ------------------------------

def write_tiled_tiff(tile_iterator, shape: Tuple[int, int], output_path: str, tile_size: int = 1024, pixel_resolution_um: Optional[float] = None) -> None:
    """
    Write tiles from iterator to a tiled TIFF file.

    Args:
        tile_iterator: Iterator that yields uint8 tiles
        shape: Output shape (H, W)
        output_path: Path to output TIFF file
        tile_size: Size of tiles (default 1024)
        pixel_resolution_um: Real-world pixel resolution in micrometers (None to omit resolution metadata)
    """
    try:
        H, W = shape

        # Build tiff.imwrite kwargs
        tiff_kwargs = {
            'shape': (H, W),
            'dtype': np.uint8,
            'compression': 'deflate',
            'tile': (tile_size, tile_size),
            'metadata': {'software': 'optimized_inference'}
        }

        # Add resolution metadata if provided
        if pixel_resolution_um is not None:
            # Convert µm/pixel to DPI (dots per inch)
            # 1 inch = 25400 µm, so DPI = 25400 / (µm per pixel)
            dpi = 25400.0 / pixel_resolution_um
            tiff_kwargs['resolution'] = (dpi, dpi)
            tiff_kwargs['resolutionunit'] = 'INCH'
            logger.info(f"Writing tiled TIFF to {output_path} with shape ({H}, {W}), tile size {tile_size}x{tile_size}, "
                       f"and resolution {pixel_resolution_um:.3f} µm/pixel ({dpi:.2f} DPI)")
        else:
            logger.info(f"Writing tiled TIFF to {output_path} with shape ({H}, {W}), tile size {tile_size}x{tile_size} "
                       f"(no resolution metadata)")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # Write with tiling and compression using iterator
        tiff.imwrite(output_path, tile_iterator, **tiff_kwargs)

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Wrote tiled TIFF: {output_path} ({file_size_mb:.2f} MB)")

    except Exception as e:
        logger.error(f"Error writing tiled TIFF: {e}")
        raise
