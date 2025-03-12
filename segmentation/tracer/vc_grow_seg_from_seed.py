#!/usr/bin/env python3
"""
Python implementation of vc_grow_seg_from_seed.cpp.

This script grows a surface from a seed point through a volumetric dataset.
It supports multiple modes:
- explicit_seed: Use a user-provided seed point
- random_seed: Find a random valid seed point
- expansion: Grow from an existing segment

Usage:
    vc_grow_seg_from_seed.py <ome-zarr-volume> <tgt-dir> <json-params> [<seed-x> <seed-y> <seed-z>]
"""

import os
import sys
import json
import random
import logging
import datetime
import numpy as np
import zarr
import torch
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union

from tracer.core.interpolation import TrilinearInterpolator
from tracer.grow import space_tracing_quad_phys
from surfaces.quad_surface import QuadSurface
from surfaces.surface_meta import SurfaceMeta, contains, overlap

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def time_str() -> str:
    """Generate a timestamp string like in the C++ implementation."""
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d%H%M%S%f")[:-3]

def get_val(interpolator, point):
    """Sample a value at a 3D point."""
    # Convert point to batch format for interpolator
    z = torch.tensor([[point[0]]], dtype=torch.float32)
    y = torch.tensor([[point[1]]], dtype=torch.float32)
    x = torch.tensor([[point[2]]], dtype=torch.float32)
    return interpolator.evaluate(z, y, x).item()

def check_existing_segments(tgt_dir: Path, origin: np.ndarray, 
                          name_prefix: str, search_effort: int) -> bool:
    """Check if the seed point overlaps with existing segments."""
    for entry in os.listdir(tgt_dir):
        entry_path = tgt_dir / entry
        if not os.path.isdir(entry_path):
            continue
            
        if not entry.startswith(name_prefix):
            continue
            
        meta_path = entry_path / "meta.json"
        if not os.path.exists(meta_path):
            continue
            
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        if "bbox" not in meta or meta.get("format", "NONE") != "tifxyz":
            continue
            
        other = SurfaceMeta(entry_path, meta)
        if contains(other, origin, search_effort):
            logger.info(f"Found overlapping segment at location: {entry_path}")
            return True
            
    return False

def main():
    """Main function."""
    if len(sys.argv) != 4 and len(sys.argv) != 7:
        print(f"usage: {sys.argv[0]} <ome-zarr-volume> <tgt-dir> <json-params> [<seed-x> <seed-y> <seed-z>]")
        return
        
    vol_path = Path(sys.argv[1])
    tgt_dir = Path(sys.argv[2])
    params_path = sys.argv[3]
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Open dataset (zarr or numpy)
    logger.info(f"Opening dataset at {vol_path}")
    
    # Check if it's a numpy file
    if str(vol_path).endswith('.npy'):
        try:
            # Load NumPy file
            dataset = np.load(vol_path)
            logger.info(f"Loaded numpy array with shape: {dataset.shape}")
        except Exception as e:
            logger.error(f"Error loading numpy array: {e}")
            return
    else:
        # Try loading as zarr - DON'T load into memory, keep as zarr array
        try:
            # First try opening the highest resolution level (0)
            zarr_array = zarr.open(vol_path / "0", mode='r')
            logger.info(f"Opened zarr dataset from level 0 with shape: {zarr_array.shape}")
            dataset = zarr_array
        except Exception as e:
            try:
                # If that fails, try opening the root dataset
                zarr_array = zarr.open(vol_path, mode='r')
                logger.info(f"Opened zarr dataset from root with shape: {zarr_array.shape}")
                dataset = zarr_array
            except Exception as e2:
                logger.error(f"Error opening zarr dataset: {e2}")
                return
    
    # Create chunk cache
    cache_size = params.get("cache_size", 1e9)
    cache = None  # TODO: Implement chunked cache
    
    # Set up interpolator
    interpolator = TrilinearInterpolator(dataset, use_cache=True, cache_size=int(cache_size))
    
    # Initialize parameters
    name_prefix = "auto_grown_"
    tgt_overlap_count = params.get("tgt_overlap_count", 20)
    min_area_cm = params.get("min_area_cm", 0.1)  # Lowered from 0.3 to 0.1
    step_size = params.get("step_size", 20)
    search_effort = params.get("search_effort", 10)
    generations = params.get("generations", 100)
    
    # Read voxel size from metadata
    try:
        with open(vol_path / "meta.json", 'r') as f:
            voxelsize = json.load(f)["voxelsize"]
    except:
        voxelsize = 1.0
        logger.warning("Could not read voxelsize from metadata, using default: 1.0")
    
    cache_root = params.get("cache_root", "")
    mode = params.get("mode", "seed")
    
    logger.info(f"Mode: {mode}")
    logger.info(f"Step size: {step_size}")
    logger.info(f"Min area (cm²): {min_area_cm}")
    logger.info(f"Target overlap count: {tgt_overlap_count}")
    
    surfs = {}
    surfs_v = []
    src = None
    origin = np.zeros(3)
    count_overlap = 0
    
    # Expansion mode
    if mode == "expansion":
        # Go through all existing segments
        for entry in os.listdir(tgt_dir):
            entry_path = tgt_dir / entry
            if not os.path.isdir(entry_path):
                continue
                
            if not entry.startswith(name_prefix):
                continue
                
            logger.info(f"Found segment: {entry_path}")
            
            meta_path = entry_path / "meta.json"
            if not os.path.exists(meta_path):
                continue
                
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                
            if "bbox" not in meta or meta.get("format", "NONE") != "tifxyz":
                continue
                
            sm = SurfaceMeta(entry_path, meta)
            sm.read_overlapping()
            
            surfs[entry] = sm
            surfs_v.append(sm)
            
        if not surfs:
            logger.error("ERROR: no seed surfaces found in expansion mode")
            return
            
        # Shuffle the surfaces
        random.seed()
        random.shuffle(surfs_v)
        
        # Find a seed point for expansion
        for it in surfs_v:
            src = it
            surf = src.surf()
            points = surf.raw_points()
            h, w = points.shape[:2]
            
            found = False
            for r in range(10):
                if random.randint(0, 1) == 0:
                    # Choose a random point
                    y, x = random.randint(0, h-1), random.randint(0, w-1)
                    
                    if points[y, x, 0] != -1 and get_val(interpolator, points[y, x]) >= 128:
                        found = True
                        origin = points[y, x]
                        break
                else:
                    # Choose a point at the edge
                    side = random.randint(0, 3)
                    if side == 0:
                        y, x = random.randint(0, h-1), 0
                    elif side == 1:
                        y, x = 0, random.randint(0, w-1)
                    elif side == 2:
                        y, x = random.randint(0, h-1), w-1
                    else:
                        y, x = h-1, random.randint(0, w-1)
                        
                    # Calculate search direction toward center
                    search_dir = np.array([h/2 - y, w/2 - x], dtype=float)
                    search_dir = search_dir / np.linalg.norm(search_dir)
                    
                    # Search along direction
                    found = False
                    p = np.array([y, x], dtype=float)
                    max_steps = min(int(w/2/abs(search_dir[1])), int(h/2/abs(search_dir[0])))
                    
                    for i in range(max_steps):
                        p += search_dir
                        p_int = p.astype(int)
                        
                        if (p_int[0] < 0 or p_int[1] < 0 or 
                            p_int[0] >= h or p_int[1] >= w):
                            continue
                            
                        found = True
                        for r in range(5):
                            p_eval = p + r * search_dir
                            p_eval = p_eval.astype(int)
                            
                            if (p_eval[0] < 0 or p_eval[1] < 0 or
                                p_eval[0] >= h or p_eval[1] >= w or
                                points[p_eval[0], p_eval[1], 0] == -1 or
                                get_val(interpolator, points[p_eval[0], p_eval[1]]) < 128):
                                found = False
                                break
                                
                        if found:
                            p_eval = (p + 2*search_dir).astype(int)
                            if (p_eval[0] >= 0 and p_eval[1] >= 0 and
                                p_eval[0] < h and p_eval[1] < w):
                                origin = points[p_eval[0], p_eval[1]]
                                break
                            
            if not found:
                continue
                
            # Check for overlaps
            count_overlap = 0
            for comp in surfs_v:
                if comp == src:
                    continue
                if contains(comp, origin, search_effort):
                    count_overlap += 1
                if count_overlap >= tgt_overlap_count - 1:
                    break
                    
            if count_overlap < tgt_overlap_count - 1:
                break
                
        logger.info(f"Found potential overlapping starting seed {origin} with overlap {count_overlap}")
        
    else:
        if len(sys.argv) == 7:
            # Explicit seed mode
            mode = "explicit_seed"
            
            # Note: User provides seed in order [x, y, z], but we need [z, y, x] internally
            # So we need to reorder the coordinates
            seed_x = float(sys.argv[4])
            seed_y = float(sys.argv[5])
            seed_z = float(sys.argv[6])
            
            # Reorder to ZYX for internal use
            origin = np.array([seed_z, seed_y, seed_x])
            value = get_val(interpolator, origin)
            logger.info(f"Seed location {origin} value is {value} (original XYZ input: [{seed_x}, {seed_y}, {seed_z}])")
        else:
            # Random seed mode
            mode = "random_seed"
            count = 0
            succ = False
            max_attempts = 1000
            
            while count < max_attempts and not succ:
                # Choose a random point with some margin from the edges
                origin = np.array([
                    128 + random.randint(0, dataset.shape[0] - 384),
                    128 + random.randint(0, dataset.shape[1] - 384),
                    128 + random.randint(0, dataset.shape[2] - 384)
                ])
                
                count += 1
                
                # TODO: Check if chunk exists (needed for sparse datasets)
                
                # Choose a random direction
                direction = np.array([
                    random.randint(-512, 512),
                    random.randint(-512, 512),
                    random.randint(-512, 512)
                ], dtype=float)
                direction = direction / np.linalg.norm(direction)
                
                # Search along the direction
                for i in range(128):
                    p = origin + i * direction
                    value = get_val(interpolator, p)
                    
                    if value >= 128:
                        if check_existing_segments(tgt_dir, p, name_prefix, search_effort):
                            continue
                            
                        succ = True
                        origin = p
                        logger.info(f"Found seed location {origin} value: {value}")
                        break
                        
            if not succ:
                logger.error(f"ERROR: Could not find valid non-overlapping seed location after {max_attempts} attempts")
                return
    
    # Grow the surface
    logger.info(f"Growing surface from seed {origin} for {generations} generations")
    surface = space_tracing_quad_phys(
        dataset=dataset,
        scale=1.0,
        cache=cache,
        origin=origin,
        generations=generations,
        step_size=step_size,
        intensity_threshold=170,  # Use raw intensity threshold value (matching C++)
        physical_fail_threshold=0.1,    # Match C++ default value for better compatibility
        max_reference_count=6,          # Match C++ ref_max (6) from SurfaceHelpers.cpp:1106
        batch_size=64                   # Keep batch size high for faster processing
    )
    
    # Calculate area
    area_cm2 = surface.meta["area_vx2"] * voxelsize * voxelsize / 1e8
    if area_cm2 < min_area_cm:
        logger.info(f"Surface area {area_cm2} cm² is below minimum threshold {min_area_cm} cm², but saving anyway")
        # Continue with saving instead of returning
        
    # Update metadata
    surface.meta["area_cm2"] = area_cm2
    surface.meta["source"] = "vc_grow_seg_from_seed"
    surface.meta["vc_gsfs_params"] = params
    surface.meta["vc_gsfs_mode"] = mode
    surface.meta["vc_gsfs_version"] = "python-theseus"
    
    if mode == "expansion":
        surface.meta["seed_overlap"] = count_overlap
        
    # Generate UUID and save surface
    uuid = name_prefix + time_str()
    seg_dir = tgt_dir / uuid
    logger.info(f"Saving surface to {seg_dir}")
    surface.save(seg_dir, uuid)
    
    # Handle overlapping segments
    if mode == "expansion":
        # Create the current surface meta
        current = SurfaceMeta(seg_dir, surface.meta)
        current.set_surf(surface)
        
        # Create overlapping directories
        overlap_dir = current.path / "overlapping"
        os.makedirs(overlap_dir, exist_ok=True)
        
        # Mark overlaps with source segment
        with open(overlap_dir / src.name(), 'w') as f:
            pass
            
        overlap_src = src.path / "overlapping"
        os.makedirs(overlap_src, exist_ok=True)
        with open(overlap_src / current.name(), 'w') as f:
            pass
            
        # Check for overlaps with existing segments
        for s in surfs_v:
            if overlap(current, s, search_effort):
                with open(overlap_dir / s.name(), 'w') as f:
                    pass
                    
                overlap_other = s.path / "overlapping"
                os.makedirs(overlap_other, exist_ok=True)
                with open(overlap_other / current.name(), 'w') as f:
                    pass
                    
        # Check for overlaps with segments not in memory
        for entry in os.listdir(tgt_dir):
            entry_path = tgt_dir / entry
            if not os.path.isdir(entry_path) or entry in surfs:
                continue
                
            if not entry.startswith(name_prefix):
                continue
                
            if entry == current.name():
                continue
                
            meta_path = entry_path / "meta.json"
            if not os.path.exists(meta_path):
                continue
                
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                
            if "bbox" not in meta or meta.get("format", "NONE") != "tifxyz":
                continue
                
            other = SurfaceMeta(entry_path, meta)
            other.read_overlapping()
            
            if overlap(current, other, search_effort):
                with open(overlap_dir / other.name(), 'w') as f:
                    pass
                    
                overlap_other = other.path / "overlapping"
                os.makedirs(overlap_other, exist_ok=True)
                with open(overlap_other / current.name(), 'w') as f:
                    pass

if __name__ == "__main__":
    main()