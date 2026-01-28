#!/usr/bin/env python3
"""
Pack paired TIF image/label samples into sparse OME-Zarr format with multi-resolution pyramids.

Converts individual TIF image/label pairs into a single sparse OME-Zarr volume
with samples arranged in a grid. Empty gap regions between samples are
not written to disk (sparse storage). Multiple resolution levels are generated
for efficient multi-scale viewing/processing.

Supports both 2D (HxW) and 3D (DxHxW) TIF files.

Example usage:
    python -m vesuvius.scripts.pack_tifs_to_sparse_zarr \
        --source /path/to/tif_dataset \
        --output /path/to/output \
        --gap 64 \
        --chunk-size 64 \
        --num-levels 4 \
        --compression-level 5 \
        --target-name fiber
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tifffile
import zarr
from scipy import ndimage
from numcodecs import Blosc
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SampleInfo:
    """Metadata for a single TIF sample."""
    index: int
    filename: str
    image_path: Path
    label_path: Path
    original_coords: Optional[Dict[str, Union[str, int]]]
    split: Optional[str]
    shape: Tuple[int, ...]
    # Label validation (computed during discovery)
    has_labels: bool = True
    labeled_ratio: float = 0.0


@dataclass
class LayoutConfig:
    """Grid layout configuration."""
    num_samples: int
    sample_shape: Tuple[int, ...]
    gap_size: int
    grid_shape: Tuple[int, ...]
    cell_size: Tuple[int, ...]
    volume_shape: Tuple[int, ...]
    chunk_size: Tuple[int, ...]
    is_2d: bool


def parse_vesuvius_filename(filename: str) -> Optional[Dict[str, Union[str, int]]]:
    """
    Parse Vesuvius-style filename: '{segment}_z{z}_y{y}_x{x}.tif'.

    Returns None if filename doesn't match the pattern.
    """
    stem = Path(filename).stem
    pattern = r'^(.+?)_z(\d+)_y(\d+)_x(\d+)$'
    match = re.match(pattern, stem)
    if match:
        return {
            'segment': match.group(1),
            'z': int(match.group(2)),
            'y': int(match.group(3)),
            'x': int(match.group(4)),
        }
    return None


def _process_single_sample(
    idx: int,
    img_path: Path,
    labels_dir: Path,
    split_lookup: Dict[str, str],
) -> Optional[SampleInfo]:
    """
    Process a single image/label pair.

    Parameters
    ----------
    idx : int
        Sample index
    img_path : Path
        Path to image TIF file
    labels_dir : Path
        Directory containing label files
    split_lookup : Dict[str, str]
        Mapping from sample name to train/val split

    Returns
    -------
    SampleInfo or None
        Sample metadata, or None if label not found
    """
    # Try both .tif and .tiff extensions for labels
    label_path = labels_dir / img_path.name
    if not label_path.exists():
        alt_ext = ".tiff" if img_path.suffix == ".tif" else ".tif"
        label_path = labels_dir / (img_path.stem + alt_ext)

    if not label_path.exists():
        logger.warning(f"No label found for {img_path.name}, skipping")
        return None

    # Read shape from image
    sample_data = tifffile.imread(str(img_path))
    shape = sample_data.shape

    # Read label and compute validation metrics
    label_data = tifffile.imread(str(label_path))
    non_zero_count = int(np.count_nonzero(label_data))
    total_voxels = label_data.size
    has_labels = non_zero_count > 0
    labeled_ratio = float(non_zero_count / total_voxels) if total_voxels > 0 else 0.0

    stem = img_path.stem
    return SampleInfo(
        index=idx,
        filename=img_path.name,
        image_path=img_path,
        label_path=label_path,
        original_coords=parse_vesuvius_filename(img_path.name),
        split=split_lookup.get(stem),
        shape=shape,
        has_labels=has_labels,
        labeled_ratio=labeled_ratio,
    )


def discover_samples(
    source_dir: Path,
    splits_json: Optional[Path] = None,
    num_workers: int = 4,
) -> List[SampleInfo]:
    """
    Discover all image/label TIF pairs in source directory.

    Parameters
    ----------
    source_dir : Path
        Directory containing images/ and labels/ subdirectories
    splits_json : Path, optional
        Path to splits JSON file for train/val assignment
    num_workers : int
        Number of parallel workers for reading TIF files

    Returns
    -------
    List[SampleInfo]
        List of discovered sample metadata
    """
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Load splits if available
    split_lookup: Dict[str, str] = {}
    if splits_json and splits_json.exists():
        with open(splits_json) as f:
            splits_data = json.load(f)
        # Handle nnUNet-style splits format (list of fold dicts)
        if isinstance(splits_data, list):
            for fold_entry in splits_data:
                for name in fold_entry.get('train', []):
                    split_lookup[name] = 'train'
                for name in fold_entry.get('val', []):
                    split_lookup[name] = 'val'
        # Handle simple dict format
        elif isinstance(splits_data, dict):
            for name in splits_data.get('train', []):
                split_lookup[name] = 'train'
            for name in splits_data.get('val', []):
                split_lookup[name] = 'val'

    image_files = sorted(images_dir.glob("*.tif")) + sorted(images_dir.glob("*.tiff"))

    # Process samples in parallel
    samples = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _process_single_sample,
                idx,
                img_path,
                labels_dir,
                split_lookup,
            ): idx
            for idx, img_path in enumerate(image_files)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Discovering samples"):
            result = future.result()
            if result is not None:
                samples.append(result)

    # Sort by index to maintain consistent ordering
    samples.sort(key=lambda s: s.index)

    return samples


def calculate_layout(
    num_samples: int,
    sample_shape: Tuple[int, ...],
    gap: int,
    chunk_size: int,
) -> LayoutConfig:
    """
    Calculate optimal grid layout for packing samples.

    Parameters
    ----------
    num_samples : int
        Total number of samples to pack
    sample_shape : Tuple[int, ...]
        Shape of each sample (2D or 3D)
    gap : int
        Gap size between samples in voxels
    chunk_size : int
        Zarr chunk size per axis

    Returns
    -------
    LayoutConfig
        Complete layout configuration
    """
    is_2d = len(sample_shape) == 2
    ndim = len(sample_shape)

    # Calculate grid side length (cube/square root)
    grid_side = math.ceil(num_samples ** (1.0 / ndim))

    # Cell size includes sample + gap
    cell_size = tuple(s + gap for s in sample_shape)

    # Grid shape
    grid_shape = tuple([grid_side] * ndim)

    # Total volume dimension (per axis)
    vol_dims = []
    for i in range(ndim):
        dim = grid_side * cell_size[i]
        # Align to chunk boundary
        aligned = ((dim + chunk_size - 1) // chunk_size) * chunk_size
        vol_dims.append(aligned)

    volume_shape = tuple(vol_dims)
    chunk_shape = tuple([chunk_size] * ndim)

    return LayoutConfig(
        num_samples=num_samples,
        sample_shape=sample_shape,
        gap_size=gap,
        grid_shape=grid_shape,
        cell_size=cell_size,
        volume_shape=volume_shape,
        chunk_size=chunk_shape,
        is_2d=is_2d,
    )


def sample_to_position(sample_idx: int, layout: LayoutConfig) -> Tuple[int, ...]:
    """
    Calculate zarr position for a sample index.

    Parameters
    ----------
    sample_idx : int
        Linear index of the sample
    layout : LayoutConfig
        Grid layout configuration

    Returns
    -------
    Tuple[int, ...]
        Position (z, y, x) or (y, x) in the zarr volume
    """
    ndim = len(layout.grid_shape)
    gs = layout.grid_shape[0]  # Assuming uniform grid

    if ndim == 3:
        gz = sample_idx // (gs * gs)
        gy = (sample_idx % (gs * gs)) // gs
        gx = sample_idx % gs
        return (
            gz * layout.cell_size[0],
            gy * layout.cell_size[1],
            gx * layout.cell_size[2],
        )
    else:  # 2D
        gy = sample_idx // gs
        gx = sample_idx % gs
        return (
            gy * layout.cell_size[0],
            gx * layout.cell_size[1],
        )


def create_ome_zarr_group(
    output_path: Path,
    layout: LayoutConfig,
    dtype: str,
    compressor,
    num_levels: int,
    is_labels: bool = False,
) -> zarr.Group:
    """
    Create an OME-Zarr group with multi-resolution pyramid structure.

    Parameters
    ----------
    output_path : Path
        Output zarr path
    layout : LayoutConfig
        Grid layout configuration
    dtype : str
        Data type for the arrays
    compressor
        Blosc compressor instance
    num_levels : int
        Number of resolution levels (1 = full res only)
    is_labels : bool
        If True, use nearest-neighbor for downsampling

    Returns
    -------
    zarr.Group
        Created zarr group with resolution levels
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create the root group
    root = zarr.open_group(str(output_path), mode='w')

    # Create each resolution level
    datasets = []
    current_shape = layout.volume_shape
    current_chunks = layout.chunk_size

    for level in range(num_levels):
        scale_factor = 2 ** level

        # Calculate shape for this level
        if level == 0:
            level_shape = current_shape
            level_chunks = current_chunks
        else:
            level_shape = tuple(max(1, s // 2) for s in current_shape)
            # Keep chunks reasonable, don't let them exceed the level shape
            level_chunks = tuple(min(c, s) for c, s in zip(current_chunks, level_shape))
            current_shape = level_shape

        # Create array for this level
        root.create_dataset(
            str(level),
            shape=level_shape,
            chunks=level_chunks,
            dtype=dtype,
            fill_value=0,
            compressor=compressor,
            write_empty_chunks=False,
        )

        # Add to datasets list for metadata
        scale = [float(scale_factor)] * len(level_shape)
        datasets.append({
            'path': str(level),
            'coordinateTransformations': [
                {'type': 'scale', 'scale': scale}
            ]
        })

        logger.info(f"  Level {level}: shape={level_shape}, chunks={level_chunks}")

    # Write OME-NGFF multiscales metadata
    ndim = len(layout.volume_shape)
    if ndim == 3:
        axes = [
            {'name': 'z', 'type': 'space', 'unit': 'voxel'},
            {'name': 'y', 'type': 'space', 'unit': 'voxel'},
            {'name': 'x', 'type': 'space', 'unit': 'voxel'},
        ]
    else:
        axes = [
            {'name': 'y', 'type': 'space', 'unit': 'voxel'},
            {'name': 'x', 'type': 'space', 'unit': 'voxel'},
        ]

    root.attrs['multiscales'] = [{
        'version': '0.4',
        'name': output_path.stem,
        'axes': axes,
        'datasets': datasets,
        'type': 'gaussian' if not is_labels else 'nearest',
        'metadata': {
            'method': 'strided 2x downsampling',
            'is_packed_samples': True,
        }
    }]

    return root


def downsample_array(arr: np.ndarray, is_labels: bool = False) -> np.ndarray:
    """
    Downsample array by factor of 2 in each dimension.

    Parameters
    ----------
    arr : np.ndarray
        Input array (2D or 3D)
    is_labels : bool
        If True, use nearest-neighbor (max) for labels

    Returns
    -------
    np.ndarray
        Downsampled array
    """
    ndim = arr.ndim
    if ndim == 3:
        # Strided downsampling for 3D
        d, h, w = arr.shape
        new_d, new_h, new_w = d // 2, h // 2, w // 2
        if is_labels:
            # For labels, take max to preserve any label presence
            result = arr[0::2, 0::2, 0::2].copy()
            # Could do block max but strided is faster and usually sufficient
        else:
            # For images, use mean of 2x2x2 blocks
            result = arr[0::2, 0::2, 0::2].copy()
        return result
    else:
        # 2D
        h, w = arr.shape
        if is_labels:
            return arr[0::2, 0::2].copy()
        else:
            return arr[0::2, 0::2].copy()


def resize_volume(
    data: np.ndarray,
    target_shape: Tuple[int, ...],
    is_labels: bool = False,
) -> np.ndarray:
    """
    Resize 2D or 3D array to target shape.

    Parameters
    ----------
    data : np.ndarray
        Input array (2D or 3D)
    target_shape : Tuple[int, ...]
        Target shape to resize to
    is_labels : bool
        If True, use nearest-neighbor interpolation to preserve label values

    Returns
    -------
    np.ndarray
        Resized array
    """
    if data.shape == target_shape:
        return data

    # Calculate zoom factors
    zoom_factors = tuple(t / s for t, s in zip(target_shape, data.shape))

    # Use order=0 (nearest) for labels to preserve discrete values
    # Use order=1 (linear) for images
    order = 0 if is_labels else 1

    resized = ndimage.zoom(data, zoom_factors, order=order)

    # Ensure exact shape (zoom can be off by 1 due to rounding)
    if resized.shape != target_shape:
        # Crop or pad to exact size
        slices = tuple(slice(0, min(r, t)) for r, t in zip(resized.shape, target_shape))
        result = np.zeros(target_shape, dtype=data.dtype)
        result_slices = tuple(slice(0, min(r, t)) for r, t in zip(resized.shape, target_shape))
        result[result_slices] = resized[slices]
        return result

    return resized.astype(data.dtype)


def write_sample_to_ome_zarr(
    sample: SampleInfo,
    layout: LayoutConfig,
    images_zarr_path: str,
    labels_zarr_path: str,
    num_levels: int,
    target_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[int, str]:
    """
    Write a single sample to all resolution levels of the OME-Zarr arrays.

    Parameters
    ----------
    sample : SampleInfo
        Sample metadata
    layout : LayoutConfig
        Grid layout configuration
    images_zarr_path : str
        Path to images zarr group
    labels_zarr_path : str
        Path to labels zarr group
    num_levels : int
        Number of resolution levels
    target_shape : Tuple[int, ...], optional
        If provided, resize samples to this shape before writing

    Returns
    -------
    Tuple[int, str]
        (sample_index, status_message)
    """
    try:
        # Read TIF files
        image_data = tifffile.imread(str(sample.image_path))
        label_data = tifffile.imread(str(sample.label_path))

        # Resize if needed
        if target_shape is not None and image_data.shape != target_shape:
            image_data = resize_volume(image_data, target_shape, is_labels=False)
            label_data = resize_volume(label_data, target_shape, is_labels=True)

        # Open zarr groups (each worker opens fresh handles for thread safety)
        images_group = zarr.open_group(images_zarr_path, mode='r+')
        labels_group = zarr.open_group(labels_zarr_path, mode='r+')

        # Calculate position at full resolution
        position = sample_to_position(sample.index, layout)

        # Write to each resolution level
        current_image = image_data
        current_label = label_data
        current_pos = position
        current_shape = image_data.shape  # Use actual data shape after resize

        for level in range(num_levels):
            # Get arrays for this level
            img_arr = images_group[str(level)]
            lbl_arr = labels_group[str(level)]

            # Build slice for this level
            if layout.is_2d:
                y, x = current_pos
                sy, sx = current_shape
                slc = (slice(y, y + sy), slice(x, x + sx))
            else:
                z, y, x = current_pos
                sz, sy, sx = current_shape
                slc = (slice(z, z + sz), slice(y, y + sy), slice(x, x + sx))

            # Write data
            img_arr[slc] = current_image
            lbl_arr[slc] = current_label

            # Prepare for next level (if any)
            if level < num_levels - 1:
                current_image = downsample_array(current_image, is_labels=False)
                current_label = downsample_array(current_label, is_labels=True)
                current_pos = tuple(p // 2 for p in current_pos)
                current_shape = current_image.shape

        return (sample.index, "ok")
    except Exception as e:
        return (sample.index, f"error: {e}")


def save_mapping(
    output_dir: Path,
    samples: List[SampleInfo],
    layout: LayoutConfig,
    source_dir: Path,
    target_name: str,
    num_levels: int,
) -> None:
    """
    Save samples mapping JSON with all metadata.

    Parameters
    ----------
    output_dir : Path
        Output directory
    samples : List[SampleInfo]
        List of sample metadata
    layout : LayoutConfig
        Grid layout configuration
    source_dir : Path
        Original source directory path
    target_name : str
        Label target name
    num_levels : int
        Number of resolution levels
    """
    mapping = {
        'version': '1.0',
        'source': {
            'path': str(source_dir),
            'num_samples': len(samples),
        },
        'target_name': target_name,
        'num_levels': num_levels,
        'layout': {
            'type': '3d_grid' if not layout.is_2d else '2d_grid',
            'grid_shape': list(layout.grid_shape),
            'cell_size': list(layout.cell_size),
            'sample_shape': list(layout.sample_shape),
            'gap_size': layout.gap_size,
            'volume_shape': list(layout.volume_shape),
            'chunk_size': list(layout.chunk_size),
        },
        'samples': [],
        'splits': {'train': [], 'val': [], 'unassigned': []},
    }

    for sample in samples:
        pos = sample_to_position(sample.index, layout)
        sample_entry = {
            'index': sample.index,
            'filename': sample.filename,
            'zarr_position': list(pos),
            'sample_shape': list(sample.shape),
            'has_labels': sample.has_labels,
            'labeled_ratio': sample.labeled_ratio,
        }
        if sample.original_coords:
            sample_entry['original_coords'] = sample.original_coords
        if sample.split:
            sample_entry['split'] = sample.split

        mapping['samples'].append(sample_entry)

        # Track splits
        if sample.split == 'train':
            mapping['splits']['train'].append(sample.index)
        elif sample.split == 'val':
            mapping['splits']['val'].append(sample.index)
        else:
            mapping['splits']['unassigned'].append(sample.index)

    with open(output_dir / 'samples_mapping.json', 'w') as f:
        json.dump(mapping, f, indent=2)

    logger.info(f"Saved mapping to {output_dir / 'samples_mapping.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Pack paired TIF samples into sparse OME-Zarr format with multi-resolution pyramids",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--source', type=Path, required=True,
        help='Source directory with images/ and labels/ subdirectories',
    )
    parser.add_argument(
        '--output', type=Path, required=True,
        help='Output directory for zarr files',
    )
    parser.add_argument(
        '--gap', type=int, default=64,
        help='Gap between samples in voxels (default: 64)',
    )
    parser.add_argument(
        '--chunk-size', type=int, default=64,
        help='Zarr chunk size per axis (default: 64)',
    )
    parser.add_argument(
        '--num-levels', type=int, default=4,
        help='Number of resolution levels in pyramid (default: 4)',
    )
    parser.add_argument(
        '--compression-level', type=int, default=5,
        help='ZSTD compression level 1-9 (default: 5)',
    )
    parser.add_argument(
        '--num-workers', type=int, default=4,
        help='Number of parallel workers (default: 4)',
    )
    parser.add_argument(
        '--target-name', type=str, default='fiber',
        help='Label target name for output (default: fiber)',
    )
    parser.add_argument(
        '--splits-json', type=Path, default=None,
        help='Path to splits JSON (default: source/splits_final.json)',
    )
    parser.add_argument(
        '--resize', action='store_true',
        help='Resize samples with mismatched shapes to a common shape',
    )
    parser.add_argument(
        '--resize-shape', type=str, default=None,
        help='Target shape for resize (e.g. "384" for cubic or "384,384,384"). If not specified with --resize, uses the most common shape.',
    )
    args = parser.parse_args()

    # Determine splits file
    splits_json = args.splits_json
    if splits_json is None:
        default_splits = args.source / 'splits_final.json'
        if default_splits.exists():
            splits_json = default_splits

    # Discover samples
    logger.info(f"Discovering samples in {args.source}")
    samples = discover_samples(args.source, splits_json, num_workers=args.num_workers)

    if not samples:
        raise ValueError(f"No valid image/label pairs found in {args.source}")

    logger.info(f"Found {len(samples)} samples")

    # Analyze sample shapes
    shape_counts = Counter(s.shape for s in samples)
    shapes = set(shape_counts.keys())

    # Determine target shape for resizing (if needed)
    target_shape = None
    if len(shapes) > 1:
        if args.resize:
            # Determine target shape
            if args.resize_shape:
                # Parse user-specified shape
                parts = args.resize_shape.split(',')
                if len(parts) == 1:
                    # Single value = cubic
                    dim = int(parts[0])
                    is_2d = len(samples[0].shape) == 2
                    target_shape = (dim, dim) if is_2d else (dim, dim, dim)
                else:
                    target_shape = tuple(int(p) for p in parts)
            else:
                # Use most common shape
                target_shape = shape_counts.most_common(1)[0][0]

            # Log resize plan
            logger.info(f"Found {len(shapes)} different shapes: {dict(shape_counts)}")
            logger.info(f"Resizing all samples to target shape: {target_shape}")

            # Update sample shapes for layout calculation
            for s in samples:
                s.shape = target_shape
        else:
            raise ValueError(
                f"All samples must have the same shape. Found: {dict(shape_counts)}. "
                f"Use --resize to automatically resize samples to the dominant shape."
            )

    sample_shape = samples[0].shape
    is_2d = len(sample_shape) == 2
    logger.info(f"Sample shape: {sample_shape} ({'2D' if is_2d else '3D'})")

    # Calculate layout
    layout = calculate_layout(len(samples), sample_shape, args.gap, args.chunk_size)
    logger.info(f"Grid layout: {layout.grid_shape}")
    logger.info(f"Cell size: {layout.cell_size}")
    logger.info(f"Volume shape: {layout.volume_shape}")

    # Create output directories
    args.output.mkdir(parents=True, exist_ok=True)
    images_zarr_path = args.output / 'images' / 'volume.zarr'
    labels_zarr_path = args.output / 'labels' / f'volume_{args.target_name}.zarr'

    # Create compressor
    compressor = Blosc(cname='zstd', clevel=args.compression_level, shuffle=Blosc.BITSHUFFLE)

    # Determine dtype from first sample
    sample_data = tifffile.imread(str(samples[0].image_path))
    dtype = str(sample_data.dtype)
    logger.info(f"Data type: {dtype}")

    # Create OME-Zarr groups with pyramid structure
    logger.info(f"Creating OME-Zarr pyramids with {args.num_levels} levels...")
    logger.info("Images:")
    create_ome_zarr_group(
        images_zarr_path, layout, dtype, compressor,
        num_levels=args.num_levels, is_labels=False
    )
    logger.info("Labels:")
    create_ome_zarr_group(
        labels_zarr_path, layout, dtype, compressor,
        num_levels=args.num_levels, is_labels=True
    )

    # Write samples in parallel
    logger.info(f"Writing {len(samples)} samples with {args.num_workers} workers...")

    errors = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                write_sample_to_ome_zarr,
                sample,
                layout,
                str(images_zarr_path),
                str(labels_zarr_path),
                args.num_levels,
                target_shape,
            ): sample.index
            for sample in samples
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Writing samples"):
            idx, status = future.result()
            if status != "ok":
                errors.append((idx, status))
                logger.warning(f"Sample {idx}: {status}")

    if errors:
        logger.warning(f"{len(errors)} samples had errors")

    # Save mapping file
    save_mapping(args.output, samples, layout, args.source, args.target_name, args.num_levels)

    # Copy dataset.json if it exists
    dataset_json = args.source / 'dataset.json'
    if dataset_json.exists():
        shutil.copy(dataset_json, args.output / 'dataset.json')
        logger.info("Copied dataset.json")

    # Summary
    logger.info("=" * 50)
    logger.info("Conversion complete!")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Images: {images_zarr_path}")
    logger.info(f"  Labels: {labels_zarr_path}")
    logger.info(f"  Mapping: {args.output / 'samples_mapping.json'}")
    logger.info(f"  Resolution levels: {args.num_levels}")

    # Report split statistics
    train_count = sum(1 for s in samples if s.split == 'train')
    val_count = sum(1 for s in samples if s.split == 'val')
    unassigned = len(samples) - train_count - val_count
    logger.info(f"  Splits: {train_count} train, {val_count} val, {unassigned} unassigned")

    # Report label statistics
    labeled_count = sum(1 for s in samples if s.has_labels)
    unlabeled_count = len(samples) - labeled_count
    if unlabeled_count > 0:
        logger.warning(f"  Labels: {labeled_count} with labels, {unlabeled_count} without labels")
    else:
        logger.info(f"  Labels: All {labeled_count} samples have labels")


if __name__ == '__main__':
    main()
