"""
Surface growth and tracing algorithms.

This module implements the core surface growth algorithm for tracing surfaces
through volumetric data.

Coordinate Convention:
- 3D points are in ZYX order [z, y, x] with 0=z, 1=y, 2=x
- Grid coordinates follow YX ordering [y, x] for 2D indexing
- Volume data is accessed with ZYX ordering
- Growth directions and validation follow ZYX ordering for coordinates
"""

import logging
import time
import heapq
from typing import Optional, Tuple, List, Union, Dict, Any, Set

import numpy as np
import torch

from tracer.grid import PointGrid, STATE_NONE, STATE_LOC_VALID, STATE_COORD_VALID, STATE_PROCESSING
from tracer.optimizer import SurfaceOptimizer
from surfaces.quad_surface import QuadSurface

# Add SurfaceGrower class for simplified interface
class SurfaceGrower:
    """
    A simplified interface for growing surfaces along high-intensity regions.
    
    This class provides a user-friendly interface to the surface growing
    functionality, managing the grid, optimizer, and growth process.
    """
    
    def __init__(
        self,
        grid: PointGrid,
        interpolator: 'TrilinearInterpolator',
        intensity_threshold: float = 0.5,
        batch_size: int = 4,
        max_tries: int = 5,
        step_size: float = None
    ):
        """
        Initialize the surface grower.
        
        Args:
            grid: The point grid to grow from
            interpolator: Trilinear interpolator for the volume data
            intensity_threshold: Threshold for considering a point valid (0-1 range)
            batch_size: Number of points to optimize in each batch
            max_tries: Maximum number of optimization attempts per point
            step_size: Step size for the optimizer (if None, use default)
        """
        self.grid = grid
        
        # Default step size if not provided
        if step_size is None:
            step_size = 5.0  # Sensible default
        self.step_size = step_size
        
        # Reference to dataset from the interpolator (for optimizer)
        self.dataset = interpolator.volume
        
        # Create optimizer using the grid and dataset
        self.optimizer = SurfaceOptimizer(grid, self.dataset, None, self.step_size)
        
        # Make interpolator accessible to optimizer
        self.optimizer.interpolator = interpolator
        
        # Store parameters
        self.intensity_threshold = intensity_threshold
        self.batch_size = batch_size
        self.max_tries = max_tries
        
        # Create growth priority handler
        self.priority = GrowthPriority(grid, self.optimizer)
        
        # Track statistics
        self.total_valid_points = 0
        self.failed_points = 0
        self.consecutive_failed_generations = 0
        self.optimization_successes = 0
        self.optimization_failures = 0
        
        # Initialize with the seed quad
        valid_count = np.sum((grid.state & grid.state.dtype.type(STATE_LOC_VALID)) != 0)
        self.total_valid_points = valid_count
        
        print(f"Initialized SurfaceGrower with {valid_count} seed points, batch_size={batch_size}")
        
    def grow_one_generation(self) -> int:
        """
        Grow the surface by one generation.
        
        Returns:
            Number of new valid points added in this generation
        """
        # Get candidate points for this generation
        candidates = self.grid.get_candidate_points()
        if not candidates:
            print(f"No more candidates found, stopping")
            return 0
            
        print(f"Found {len(candidates)} candidate points")
        
        # Add candidates to priority queue
        self.priority.add_candidates(candidates)
        
        # Track how many points we add in this generation
        generation_valid_points = 0
        
        # Process candidates in batches by priority until all are processed
        while not self.priority.is_empty():
            # Get next batch of highest-priority candidates
            batch = self.priority.get_next_batch(self.batch_size)
            if not batch:
                break
                
            try:
                # Process the batch
                print(f"GROW_DEBUG: Processing batch with {len(batch)} points: {batch}")
                
                # Track optimization success
                optimization_succeeded = False
                
                try:
                    # Process with proper batch handling
                    print(f"GROW_DEBUG: Calling optimizer.optimize_points() with batch of {len(batch)} points")
                    print(f"GROW_DEBUG: Using batch_size={self.batch_size} for tensor batching")
                    
                    # Split batch into smaller sub-batches if needed for memory efficiency
                    # but maintain sub-batches of at least self.batch_size or remaining points
                    sub_batch_size = min(self.batch_size, len(batch))
                    if sub_batch_size < 1:
                        sub_batch_size = 1  # Ensure at least one point per sub-batch
                        
                    for i in range(0, len(batch), sub_batch_size):
                        sub_batch = batch[i:i+sub_batch_size]
                        print(f"GROW_DEBUG: Processing sub-batch {i//sub_batch_size+1} with {len(sub_batch)} points")
                        
                        try:
                            self.optimizer.optimize_points(sub_batch)
                            print(f"GROW_DEBUG: Sub-batch optimization completed successfully")
                            optimization_succeeded = True
                            self.optimization_successes += 1
                        except Exception as e:
                            print(f"GROW_DEBUG: Sub-batch optimization failed: {e}")
                            self.optimization_failures += 1
                            
                            # If sub-batch optimization fails, try each point individually
                            print("GROW_DEBUG: Falling back to single-point optimization")
                            for y, x in sub_batch:
                                try:
                                    print(f"GROW_DEBUG: Optimizing single point ({y},{x})")
                                    self.optimizer.optimize_points([(y, x)])
                                    print(f"GROW_DEBUG: Single point optimization for ({y},{x}) completed")
                                    optimization_succeeded = True
                                    self.optimization_successes += 1
                                except Exception as e:
                                    print(f"GROW_DEBUG: Single point optimization for ({y},{x}) failed: {e}")
                                    self.optimization_failures += 1
                
                except Exception as e:
                    print(f"GROW_DEBUG: Batch processing failed: {e}")
                    self.optimization_failures += 1
                
                # Fail if optimization completely failed
                if not optimization_succeeded:
                    print("GROW_DEBUG: Complete optimization failure - no points could be optimized!")
                    # Mark batch as failed and skip processing
                    self.failed_points += len(batch)
                    continue
                
                # Process optimized points and update fringe
                new_valid_points = 0
                for y, x in batch:
                    # Sample volume at optimized point for quality check
                    value = self.optimizer.sample_volume_at_point(y, x)
                    point = self.grid.get_point(y, x)
                    
                    # Perform additional validation checks
                    valid_point = True
                    
                    # Check if any coordinate is negative
                    if np.any(point < 0):
                        print(f"Point ({y}, {x}) has negative coordinates: {point}")
                        valid_point = False
                    
                    # Check if intensity is above threshold
                    if value < self.intensity_threshold:
                        print(f"Point ({y}, {x}) has low intensity: {value:.3f}")
                        valid_point = False
                    
                    # Only keep valid points
                    if valid_point:
                        # Mark as valid (this will allow us to expand from this point)
                        self.grid.update_state(y, x, STATE_LOC_VALID | STATE_COORD_VALID)
                        self.grid.fringe.append((y, x))
                        new_valid_points += 1
                        generation_valid_points += 1
                    else:
                        # Mark as invalid
                        self.grid.set_state(y, x, STATE_NONE)
                        self.failed_points += 1
                
                self.total_valid_points += new_valid_points
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Mark these candidates as failed but continue with next batch
                for y, x in batch:
                    self.grid.set_state(y, x, STATE_NONE)
                    self.failed_points += len(batch)
        
        return generation_valid_points
        
    def grow_generations(self, num_generations: int) -> int:
        """
        Grow the surface for a specified number of generations.
        
        Args:
            num_generations: Number of generations to grow
            
        Returns:
            Total number of valid points in the surface
        """
        for gen in range(num_generations):
            new_points = self.grow_one_generation()
            
            # Check for early termination if no new points were added
            if new_points == 0:
                print(f"No new points added in generation {gen}, stopping")
                break
                
            # Check if we've reached a reasonable point limit to avoid long tests
            if gen % 5 == 0:
                print(f"Generation {gen}: Total points = {self.total_valid_points}")
                
            # Check if the fringe is empty (no more candidates possible)
            if len(self.grid.fringe) == 0:
                print("No valid points in fringe, stopping")
                break
        
        return self.total_valid_points

logger = logging.getLogger(__name__)


class GrowthPriority:
    """Helper class for prioritizing candidate points during growth."""
    
    def __init__(self, grid: PointGrid, optimizer: SurfaceOptimizer):
        """
        Initialize growth priority tracker.
        
        Args:
            grid: The point grid being grown
            optimizer: The optimizer used for growth
        """
        self.grid = grid
        self.optimizer = optimizer
        self.intensity_cache = {}  # Cache for intensity values
        self.priority_queue = []   # Priority queue for candidates
        
    def calculate_priority(self, y: int, x: int) -> float:
        """
        Calculate priority for a candidate point.
        
        Higher priority (larger value) means the point should be grown first.
        Priority is based on a combination of:
        1. Volume intensity at the point
        2. Volume gradient magnitude (high gradient = feature boundary)
        3. Number of valid neighbors
        4. Neighbor intensity pattern (prefer growing in high intensity directions)
        5. Distance from boundary (prefer points at outer edge for growth)
        
        Args:
            y: Y coordinate of the candidate
            x: X coordinate of the candidate
            
        Returns:
            Priority value (higher = grows earlier)
        """
        # Get candidate position (initial guess from neighbors)
        position = self.grid.get_point(y, x)
        
        # Sample intensity and gradient at position
        if (y, x) not in self.intensity_cache:
            # Sample with gradient
            value, gradient = self.optimizer.sample_volume_with_gradient(y, x)
            gradient_magnitude = np.linalg.norm(gradient)
            self.intensity_cache[(y, x)] = (value, gradient_magnitude)
        else:
            value, gradient_magnitude = self.intensity_cache[(y, x)]
        
        # Get origin from grid center in 3D space
        cy, cx = self.grid.center
        center_point = self.grid.get_point(cy, cx)
        
        # Calculate distances in 3D space
        dist_from_center = np.linalg.norm(position - center_point)
        
        # Count valid neighbors (to prefer filling holes)
        neighbor_count = 0
        neighbor_intensities = []
        outward_neighbors = 0
        for dy, dx in self.grid.neighbors:
            ny, nx = y + dy, x + dx
            if self.grid.is_valid(ny, nx):
                neighbor_count += 1
                # Get the neighbor's intensity
                neighbor_value = self.optimizer.sample_volume_at_point(ny, nx)
                neighbor_intensities.append(neighbor_value)
                
                # Check if this neighbor is further from center (outward direction)
                neighbor_pos = self.grid.get_point(ny, nx)
                if np.linalg.norm(neighbor_pos - center_point) > dist_from_center:
                    outward_neighbors += 1
        
        # Calculate priority components
        
        # Intensity factor - prefer high but not maximum intensity
        # We want points that are likely at boundary (0.7-0.9) rather than interior (1.0)
        if value > 0.7 and value < 0.9:
            intensity_factor = 2.0  # High priority for boundary region
        else:
            intensity_factor = value * 1.0
        
        # Gradient factor - high gradient magnitude means we're at feature boundaries
        # Give highest priority to points near edges
        gradient_factor = min(gradient_magnitude * 0.7, 0.7)
        
        # Neighbor intensity pattern - prefer continuing in high-intensity directions
        neighbor_intensity_factor = 0.0
        if neighbor_intensities:
            # If this point is higher intensity than its neighbors, prioritize it
            avg_neighbor_intensity = sum(neighbor_intensities) / len(neighbor_intensities)
            if value > avg_neighbor_intensity:
                neighbor_intensity_factor = 0.4
        
        # Neighbor count factor - more neighbors = more stable, higher priority
        neighbor_factor = min(neighbor_count * 0.1, 0.3)
        
        # Outward bias factor - prefer points that lead to outward growth
        outward_factor = outward_neighbors * 0.3
        
        # Distance factor - prioritize points that are far from center
        # This is critical to make growth prioritize expanding to the boundary
        distance_factor = min(dist_from_center * 0.05, 1.0)
        
        # Final priority score (combination of factors)
        priority = (
            intensity_factor + 
            gradient_factor + 
            neighbor_intensity_factor + 
            neighbor_factor + 
            outward_factor + 
            distance_factor
        )
        
        if logger.level <= logging.DEBUG:
            logger.debug(f"Priority for ({y},{x}): {priority:.4f} = {intensity_factor:.2f}(int) + {gradient_factor:.2f}(grad) + {neighbor_intensity_factor:.2f}(n_int) + {neighbor_factor:.2f}(n_cnt) + {outward_factor:.2f}(out) + {distance_factor:.2f}(dist), val={value:.2f}, dist={dist_from_center:.2f}")
        
        return priority
        
    def add_candidates(self, candidates: List[Tuple[int, int]]):
        """
        Add candidates to the priority queue.
        
        Args:
            candidates: List of (y, x) coordinates for candidate points
        """
        for y, x in candidates:
            # Calculate priority (negative for max-heap behavior)
            priority = -self.calculate_priority(y, x)
            
            # Add to priority queue (heapq is min-heap, so use negative priority)
            heapq.heappush(self.priority_queue, (priority, (y, x)))
    
    def get_next_batch(self, batch_size: int = 10) -> List[Tuple[int, int]]:
        """
        Get the next batch of highest-priority candidates.
        
        Args:
            batch_size: Maximum number of candidates to return
            
        Returns:
            List of (y, x) coordinates for highest-priority candidates
        """
        batch = []
        while len(batch) < batch_size and self.priority_queue:
            _, candidate = heapq.heappop(self.priority_queue)
            y, x = candidate
            
            # We want points that are not yet valid but marked as processing
            state = self.grid.get_state(y, x)
            if (state & STATE_LOC_VALID) == 0:  # Not yet valid
                batch.append((y, x))
        
        return batch
    
    def is_empty(self) -> bool:
        """Check if there are no more candidates in the queue."""
        return len(self.priority_queue) == 0


def space_tracing_quad_phys(
    dataset: 'Union[np.ndarray, object]',  # zarr.Array or numpy array
    scale: float = 1.0,
    cache = None,  # Optional chunk cache
    origin: Optional[np.ndarray] = None,
    generations: int = 100,
    step_size: float = 10.0,
    cache_root: str = "",
    intensity_threshold: float = 170,  # Raw intensity threshold (matches C++ TH=170)
    batch_size: int = 20,  # Number of points to optimize at once
    max_points: int = 5000,  # Maximum number of points in the surface
    min_growth_rate: float = 0.05,  # Minimum growth rate before stopping (fraction of batch size)
    max_failed_generations: int = 5,  # Maximum number of consecutive failed generations before stopping
    use_fringe_expander: bool = True,  # Whether to use the new FringeExpander for growth
    distance_threshold: float = 1.5,  # Maximum distance value for accepting points (matches C++ dist_th)
    physical_fail_threshold: float = 0.1,  # Threshold for physical constraint failure (matches C++ phys_fail_th)
    max_reference_count: int = 6,  # Maximum reference count for quality assessment (matches C++ ref_max)
    initial_reference_min: int = 2  # Initial minimum reference count required (matches C++ curr_ref_min)
) -> QuadSurface:
    """
    Grow a surface from a seed point through optimization.
    
    This is the core algorithm for growing a surface through volumetric data.
    It initializes a grid around a seed point, then iteratively expands the
    surface by adding new points, optimizing their positions using a combination
    of cost functions.
    
    Args:
        dataset: The input volume data (zarr array or numpy array)
        scale: Scale factor for the surface
        cache: Optional chunk cache for efficient data access
        origin: 3D seed point for starting surface growth
        generations: Maximum number of growth generations
        step_size: Step size for surface growth
        cache_root: Cache directory path
        intensity_threshold: Threshold for considering a point valid
        batch_size: Number of points to optimize in each batch
        max_points: Maximum number of points in the surface
        min_growth_rate: Minimum growth rate before stopping (fraction of batch size)
        max_failed_generations: Maximum number of consecutive failed generations
        use_fringe_expander: Whether to use the new FringeExpander (True) or legacy GrowthPriority (False)
        distance_threshold: Maximum distance value for accepting points
        physical_fail_threshold: Threshold for physical constraint failure
        max_reference_count: Maximum reference count for quality assessment
        initial_reference_min: Initial minimum reference count required
    
    Returns:
        A QuadSurface object with the optimized surface points
    """
    # Set default origin if not provided
    if origin is None:
        # Get middle of volume as origin
        if hasattr(dataset, 'shape'):
            shape = dataset.shape
            origin = np.array([shape[0] // 2, shape[1] // 2, shape[2] // 2], dtype=np.float32)
        else:
            # Default origin at (0,0,0)
            origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    # Ensure origin is a numpy array
    if not isinstance(origin, np.ndarray):
        origin = np.array(origin, dtype=np.float32)
    
    logger.info(f"Starting surface growth at origin {origin} with step size {step_size}")
    
    # Initialize grid
    grid_size = 2 * generations + 10  # Ensure enough space for growth
    grid = PointGrid(grid_size, grid_size)
    
    # Initialize grid with volume information
    from tracer.core.interpolation import TrilinearInterpolator
    interp = TrilinearInterpolator(dataset)
    grid.initialize_at_origin(origin, step_size, use_gradients=True, 
                             volume_loader=interp.evaluate_with_gradient)
    
    # Create optimizer
    optimizer = SurfaceOptimizer(grid, dataset, cache, step_size)
    optimizer.interpolator = interp  # Ensure interpolator is accessible
    
    # Initial stats
    start_time = time.time()
    total_valid_points = 4  # Start with 4 seed points
    
    # Choose growth strategy based on parameter
    if use_fringe_expander:
        # Use the new FringeExpander implementation with C++ compatible settings
        from tracer.fringe_expansion import FringeExpander
        
        logger.info("Using FringeExpander with C++ compatible settings for surface growth")
        
        fringe_expander = FringeExpander(
            grid=grid,
            optimizer=optimizer,
            reference_radius=1,
            max_reference_count=6,  # Match C++ ref_max = 6 in SurfaceHelpers.cpp:1109
            initial_reference_min=6,  # Start high like C++ (curr_ref_min = ref_max) in SurfaceHelpers.cpp:1110
            distance_threshold=1.5,  # Match C++ dist_th = 1.5 in SurfaceHelpers.cpp:989
            max_optimization_tries=20,  # Much higher than default for more persistent optimization
            physical_fail_threshold=physical_fail_threshold,  # 0.1 from caller (matching C++)
            num_workers=min(32, batch_size)  # Use more workers for faster parallel processing
        )
        
        # Growth loop using FringeExpander
        try:
            logger.info(f"Starting growth for up to {generations} generations with max {max_points} points")
            
            # Keep track of consecutive failed generations
            consecutive_failed_generations = 0
            last_valid_count = total_valid_points
            
            for gen in range(generations):
                # Expand one generation
                logger.info(f"Starting generation {gen}")
                
                # Add detailed debugging about current state before expansion
                logger.info(f"Before generation {gen}: fringe size={len(grid.fringe)}, "
                            f"reference_min={fringe_expander.current_reference_min}, "
                            f"valid points={fringe_expander.total_valid_points}, "
                            f"rejected points={fringe_expander.total_rejected_points}")
                            
                new_points = fringe_expander.expand_one_generation()
                
                # Update total valid points
                total_valid_points = fringe_expander.total_valid_points + 4  # Add the 4 seed points
                
                # More detailed logging after expansion
                logger.info(f"Generation {gen} result: +{new_points} points, "
                            f"new fringe size={len(grid.fringe)}, "
                            f"total valid={total_valid_points}, "
                            f"rejected total={fringe_expander.total_rejected_points}")
                
                # Check for early termination conditions
                if new_points == 0:
                    logger.info(f"No new points added in generation {gen}, stopping")
                    break
                
                # Calculate growth rate
                if gen > 0:  # Skip first generation
                    growth_rate = new_points / (batch_size + 0.001)  # Avoid division by zero
                    logger.info(f"Generation {gen} growth rate: {growth_rate:.2f}")
                    
                    # Update consecutive failed generations count
                    if growth_rate < min_growth_rate:
                        consecutive_failed_generations += 1
                        logger.warning(f"Low growth rate generation ({growth_rate:.2f} < {min_growth_rate}). "
                                    f"Consecutive failed generations: {consecutive_failed_generations}")
                    else:
                        consecutive_failed_generations = 0
                    
                    # Stop if too many consecutive failed generations
                    if consecutive_failed_generations >= max_failed_generations:
                        logger.warning(f"Stopping after {consecutive_failed_generations} consecutive failed generations")
                        break
                
                # Periodic logging
                if gen % 5 == 0 or gen == generations - 1:
                    valid_count = np.sum((grid.state & grid.state.dtype.type(STATE_LOC_VALID)) != 0)
                    elapsed = time.time() - start_time
                    logger.info(f"Generation {gen}: {valid_count} valid points, " 
                                f"fringe size {len(grid.fringe)}, "
                                f"speed: {valid_count/elapsed:.1f} points/sec")
                    
                    # Check if we're stuck (no valid points in fringe)
                    if len(grid.fringe) == 0:
                        logger.info("No valid points in fringe, stopping")
                        break
                
                # Stop if we've reached the maximum number of points
                if total_valid_points >= max_points:
                    logger.info(f"Reached maximum point count ({max_points}), stopping")
                    break
        
        except Exception as e:
            logger.error(f"Error during fringe expansion: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Compute final statistics
    elapsed = time.time() - start_time
    valid_count = np.sum((grid.state & grid.state.dtype.type(STATE_LOC_VALID)) != 0)
    logger.info(f"Growth completed in {elapsed:.1f} seconds")
    logger.info(f"Final surface: {valid_count} valid points")
    logger.info(f"Speed: {valid_count/elapsed:.1f} points/sec")
    
    # Convert to QuadSurface
    surface = grid.to_quad_surface((scale, scale))
    
    # Add additional metadata
    surface.meta["origin"] = origin.tolist()
    surface.meta["step_size"] = step_size
    surface.meta["generations"] = gen if 'gen' in locals() else 0
    surface.meta["processing_time_sec"] = elapsed
    surface.meta["valid_points"] = int(valid_count)
    
    if use_fringe_expander:
        # Add FringeExpander-specific metadata
        surface.meta["fringe_expander"] = True
        surface.meta["rejected_points"] = fringe_expander.total_rejected_points
        surface.meta["recoveries"] = fringe_expander.recoveries
        surface.meta["reference_min"] = fringe_expander.current_reference_min
    else:
        # Add GrowthPriority-specific metadata
        surface.meta["fringe_expander"] = False
        surface.meta["failed_points"] = failed_points if 'failed_points' in locals() else 0
    
    surface.meta["intensity_threshold"] = intensity_threshold
    
    # Calculate surface area (rough estimate)
    area_vx2 = valid_count * (step_size ** 2)
    surface.meta["area_vx2"] = float(area_vx2)
    
    # Add stop reason
    if 'consecutive_failed_generations' in locals() and consecutive_failed_generations >= max_failed_generations:
        surface.meta["stop_reason"] = "consecutive_failed_generations"
    elif total_valid_points >= max_points:
        surface.meta["stop_reason"] = "max_points_reached"
    elif 'gen' in locals() and gen >= generations - 1:
        surface.meta["stop_reason"] = "max_generations_reached"
    else:
        surface.meta["stop_reason"] = "no_more_candidates"
    
    return surface


def grow_surface_from_seed(
    volume_path: str,
    output_path: str,
    seed_point: Optional[np.ndarray] = None,
    generations: int = 100,
    step_size: float = 10.0,
    scale: float = 1.0,
    cache_size: int = 1024 * 1024 * 1024,  # 1 GB default cache size
    intensity_threshold: float = 0.5,
    check_overlapping: bool = False,
    name_prefix: str = "auto_grown_",
    use_fringe_expander: bool = True,  # Whether to use the new FringeExpander
    distance_threshold: float = 1.5,    # Maximum distance value for accepting points
    physical_fail_threshold: float = 0.1,  # Threshold for physical constraint failure
    max_reference_count: int = 8,      # Maximum reference count for quality assessment
    initial_reference_min: int = 3,    # Initial minimum reference count required
    batch_size: int = 20               # Number of points to optimize at once
) -> QuadSurface:
    """
    Grow a surface from a seed point and save it to disk.
    
    This is a higher-level function that handles loading data and saving the
    resulting surface.
    
    Args:
        volume_path: Path to volume data (zarr format)
        output_path: Path to save the surface
        seed_point: 3D seed point (if None, use center of volume)
        generations: Maximum number of growth generations
        step_size: Step size for surface growth
        scale: Scale factor for the surface
        cache_size: Size of chunk cache in bytes
        intensity_threshold: Threshold for considering a point valid
        check_overlapping: Whether to check for overlapping segments
        name_prefix: Prefix for generated surface names
        use_fringe_expander: Whether to use the new FringeExpander (True) or legacy GrowthPriority (False)
        distance_threshold: Maximum distance value for accepting points
        physical_fail_threshold: Threshold for physical constraint failure
        max_reference_count: Maximum reference count for quality assessment
        initial_reference_min: Initial minimum reference count required
        batch_size: Number of points to optimize at once
        
    Returns:
        The grown QuadSurface
    """
    import zarr
    from pathlib import Path
    import uuid
    import datetime
    
    # Load volume data
    logger.info(f"Loading volume data from {volume_path}")
    dataset = zarr.open(volume_path, mode='r')
    
    # Initialize cache
    cache = None  # TODO: Add chunked cache implementation
    
    # Determine seed point if not provided
    if seed_point is None:
        shape = dataset.shape
        seed_point = np.array([shape[0] // 2, shape[1] // 2, shape[2] // 2], dtype=np.float32)
        logger.info(f"Using center of volume as seed point: {seed_point}")
    
    # Grow surface
    surface = space_tracing_quad_phys(
        dataset=dataset,
        scale=scale,
        cache=cache,
        origin=seed_point,
        generations=generations,
        step_size=step_size,
        intensity_threshold=intensity_threshold,
        batch_size=batch_size,
        use_fringe_expander=use_fringe_expander,
        distance_threshold=distance_threshold,
        physical_fail_threshold=physical_fail_threshold,
        max_reference_count=max_reference_count,
        initial_reference_min=initial_reference_min
    )
    
    # Create unique ID (timestamp-based like in C++ version)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    surface_id = f"{name_prefix}{timestamp}"
    
    # Add metadata
    surface.meta["seed_point"] = seed_point.tolist()
    surface.meta["source"] = "space_tracing_quad_phys"
    surface.meta["source_volume"] = str(volume_path)
    surface.meta["vc_gsfs_mode"] = "explicit_seed"
    surface.meta["vc_gsfs_version"] = "python-theseus"
    surface.meta["format"] = "tifxyz"  # For compatibility with C++ code
    surface.meta["use_fringe_expander"] = use_fringe_expander
    
    # Save surface
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    surface_path = output_path / surface_id
    
    logger.info(f"Saving surface to {surface_path}")
    surface.save(str(surface_path))
    
    # Handle overlapping if needed
    if check_overlapping:
        # TODO: Implement overlapping detection and recording
        # This would check other surfaces in the output directory
        # and create the "overlapping" directories with appropriate files
        logger.info("Overlapping detection not yet implemented")
    
    return surface