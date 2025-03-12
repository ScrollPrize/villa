"""
Implementation of the fringe expansion system for surface growing.

This module contains the fringe expansion system described in expansion_and_fringe.md,
which is responsible for candidate selection, evaluation, and fringe management during
surface growth.

Coordinate Convention:
- 3D points are in ZYX order [z, y, x] with 0=z, 1=y, 2=x
- Grid coordinates follow YX ordering [y, x] for 2D indexing
- Volume data is accessed with ZYX ordering
- Growth directions and validation follow ZYX ordering for coordinates
"""

from typing import List, Tuple, Dict, Set, Optional, Union
import logging
import random
import time
import os
import numpy as np
import torch

from tracer.grid import PointGrid, STATE_NONE, STATE_LOC_VALID, STATE_COORD_VALID, STATE_PROCESSING
from tracer.optimizer import SurfaceOptimizer
from concurrent.futures import ThreadPoolExecutor

# Check if debugging is enabled via environment variables
FRINGE_DEBUG_ENABLED = os.environ.get('FRINGE_DEBUG', '0').lower() in ('1', 'true', 'yes', 'on')
GENERAL_DEBUG_ENABLED = os.environ.get('DEBUG', '0').lower() in ('1', 'true', 'yes', 'on')

# Utility functions for debug printing
def fringe_debug(message):
    """Print fringe expander debug message only if fringe debugging is enabled"""
    if FRINGE_DEBUG_ENABLED:
        print(f"FRINGE_DEBUG: {message}")
        
def print_debug(message):
    """Print general debug message only if debugging is enabled"""
    if GENERAL_DEBUG_ENABLED:
        print(message)

# Configure logger
logger = logging.getLogger(__name__)
# This ensures the logger is properly configured
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG for more detailed logs


class FringeExpander:
    """
    Implements the fringe expansion system for surface growing.
    
    This class is responsible for:
    1. Reference point selection and evaluation
    2. Candidate point generation and optimization
    3. Quality assessment and fringe management
    4. Adaptive reference counting and recovery strategies
    
    It follows the approach described in the expansion_and_fringe.md document, including
    the reference point counting mechanism, candidate evaluation, and fringe recovery.
    """
    
    def __init__(
        self,
        grid: PointGrid,
        optimizer: SurfaceOptimizer,
        reference_radius: int = 1,
        max_reference_count: int = 8,
        initial_reference_min: int = 3,
        distance_threshold: float = 1.5,  # Match C++ dist_th = 1.5
        max_optimization_tries: int = 5,
        physical_fail_threshold: float = 0.1,  # Match C++ phys_fail_th = 0.1
        num_workers: int = 4
    ):
        """
        Initialize the fringe expander.
        
        Args:
            grid: The point grid to expand
            optimizer: The optimizer to use for candidate evaluation
            reference_radius: Radius for reference point counting
            max_reference_count: Maximum reference count for quality assessment
            initial_reference_min: Initial minimum reference count required
            distance_threshold: Maximum distance value for accepting points
            intensity_threshold: Minimum intensity value for accepting points
            max_optimization_tries: Maximum number of optimization attempts
            physical_fail_threshold: Threshold for physical constraint failure
            num_workers: Number of worker threads for parallel evaluation
        """
        self.grid = grid
        self.optimizer = optimizer
        
        # Reference counting parameters
        self.reference_radius = reference_radius
        self.max_reference_count = 6  # Match C++ default (ref_max = 6)
        
        # C++ implementation starts with high reference requirement that gradually decreases
        # This is different from our previous approach where we started low
        self.absolute_min_references = 2   # Hard minimum for any point to be considered
        self.current_reference_min = 6     # Start high like C++ implementation
        
        logger.info(f"Using absolute minimum reference count of {self.absolute_min_references} and " +
                  f"initial preferred reference count of {self.current_reference_min} (matching C++ behavior)")
        
        # Quality thresholds
        self.distance_threshold = distance_threshold  # Max allowed distance from object (C++ dist_th)
        self.physical_fail_threshold = physical_fail_threshold  # Max loss value (C++ phys_fail_th)
        self.intensity_threshold = getattr(optimizer, 'intensity_threshold', 170.0)  # Default intensity threshold
        
        # Optimization parameters
        self.max_optimization_tries = max_optimization_tries
        
        # Parallelism
        self.num_workers = num_workers
        
        # Statistics
        self.total_valid_points = 0
        self.total_rejected_points = 0
        self.recoveries = 0
        self.generation = 0
        
        # Get step_size from optimizer or grid (needed for exact C++ random range calculation)
        self.step_size = getattr(optimizer, 'step_size', 10.0)
        
        # For fringe management
        self.rest_points = []  # Points that didn't meet reference count but might be used later
        
        # Set debug flag based on environment variables
        self.debug = FRINGE_DEBUG_ENABLED or GENERAL_DEBUG_ENABLED
        
        logger.info(f"Initialized FringeExpander with reference_min={self.current_reference_min}, "
                   f"max_reference_count={self.max_reference_count}, "
                   f"physical_fail_threshold={self.physical_fail_threshold}")
        
        # Additional debug info if debugging is enabled
        if self.debug:
            fringe_debug(f"FringeExpander initialized with debugging ON")
            fringe_debug(f"  - reference_radius: {self.reference_radius}")
            fringe_debug(f"  - distance_threshold: {self.distance_threshold}")
            fringe_debug(f"  - intensity_threshold: {self.intensity_threshold}")
            fringe_debug(f"  - max_optimization_tries: {self.max_optimization_tries}")
    
    def expand_one_generation(self) -> int:
        """
        Expand the surface by one generation.
        
        This implements one complete cycle of:
        1. Collecting candidates from the fringe
        2. Evaluating candidates in parallel
        3. Adding successful candidates to the fringe
        4. Handling fringe recovery if needed
        
        Returns:
            Number of valid points added in this generation
        """
        self.generation += 1
        generation_valid_points = 0
        
        # Step 1: Collect candidates from current fringe
        candidates = self._collect_candidates_from_fringe()
        if not candidates:
            if self.debug:
                fringe_debug(f"No candidates found from fringe in generation {self.generation}")
            else:
                logger.info(f"Generation {self.generation}: No candidates found, attempting recovery")
                
            # Try fringe recovery if no candidates were found
            if self._attempt_fringe_recovery():
                logger.info(f"Fringe recovery successful, found {len(self.grid.fringe)} new fringe points")
                # Retry with new fringe
                candidates = self._collect_candidates_from_fringe()
            else:
                logger.info("Fringe recovery failed, no more candidates available")
                return 0
        
        if self.debug:
            fringe_debug(f"Generation {self.generation}: Collected {len(candidates)} candidates")
        
        # Step 2: Evaluate candidates in parallel
        successful_candidates = self._evaluate_candidates_parallel(candidates)
        
        # Step 3: Add successful candidates to fringe
        for y, x in successful_candidates:
            self.grid.fringe.append((y, x))
            generation_valid_points += 1
            self.total_valid_points += 1
        
        if self.debug:
            fringe_debug(f"Generation {self.generation}: Added {generation_valid_points} valid points to fringe")
            fringe_debug(f"Current fringe size: {len(self.grid.fringe)}")
        
        # Implement C++ reference count behavior - start high and decrease over time
        if generation_valid_points > 0:
            # C++ gradually decreases the threshold after successful generations
            # Unlike our previous approach of increasing it
            if self.generation > 5 and self.generation % 3 == 0:
                old_min = self.current_reference_min
                # Gradually decrease the threshold down to the minimum (similar to C++)
                self.current_reference_min = max(self.absolute_min_references, self.current_reference_min - 1)
                
                if self.debug:
                    fringe_debug(f"Reference threshold decreased from {old_min} to {self.current_reference_min}")
                else:
                    # Log reference threshold changes
                    logger.info(f"Reference threshold decreased to {self.current_reference_min} after {self.generation} generations")
        
        return generation_valid_points
    
    def _collect_candidates_from_fringe(self) -> List[Tuple[int, int]]:
        """
        Collect candidate points from the current fringe.
        
        This follows the approach in lines 1125-1147 of the C++ implementation:
        - Only expand from points with STATE_LOC_VALID
        - Select 4-connected neighbors
        - Mark each candidate as "processing"
        - Clear the current fringe after collecting candidates
        
        Returns:
            List of candidate points (y, x) coordinates
        """
        candidates = []
        
        # Track processed points to avoid duplicates
        processed = set()
        
        # For each valid point in the fringe
        for y, x in self.grid.fringe:
            if not (self.grid.get_state(y, x) & STATE_LOC_VALID):
                continue
            
            # Check all 4-connected neighbors
            for dy, dx in self.grid.neighbors:
                ny, nx = y + dy, x + dx
                
                # Skip if out of bounds
                if not self.grid.is_in_bounds(ny, nx):
                    continue
                    
                # Skip if already processed or already valid/processing
                point_state = self.grid.get_state(ny, nx)
                if (ny, nx) in processed or (point_state & STATE_PROCESSING) or (point_state & STATE_LOC_VALID):
                    continue
                
                # Mark as processing and add to candidates
                self.grid.update_state(ny, nx, STATE_PROCESSING)
                candidates.append((ny, nx))
                processed.add((ny, nx))
        
        # Clear the current fringe (following the reference implementation, line 1144)
        # The new fringe will be built from successful candidates
        old_fringe = self.grid.fringe
        self.grid.fringe = []
        
        if self.debug:
            fringe_debug(f"Collected {len(candidates)} candidates from {len(old_fringe)} fringe points")
        
        return candidates
    
    def _evaluate_candidates_parallel(self, candidates: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Evaluate a list of candidates in parallel.
        
        This implements the parallel candidate evaluation from lines 1158-1333
        of the C++ reference implementation, including:
        1. Reference point counting and selection
        2. Initial optimization
        3. Final quality testing
        
        Args:
            candidates: List of candidate points to evaluate
            
        Returns:
            List of successful candidates
        """
        successful_candidates = []
        
        # Enhanced logging - track rejection reasons for debugging
        rejection_reasons = {
            "insufficient_refs": 0,
            "optimization_failed": 0,
            "distance_check_failed": 0,
            "path_check_failed": 0,
            "exception": 0
        }
        
        # Use ThreadPoolExecutor for parallel evaluation
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit evaluation tasks for all candidates
            future_to_candidate = {
                executor.submit(self._evaluate_candidate_with_reason, y, x): (y, x) 
                for y, x in candidates
            }
            
            # Process results as they complete
            for future in future_to_candidate:
                y, x = future_to_candidate[future]
                try:
                    # Get result (Success flag and rejection reason if failed)
                    success, reason = future.result()
                    if success:
                        # Point was successful, add to successful candidates
                        successful_candidates.append((y, x))
                    else:
                        # Point failed, track rejection reason for debugging
                        if reason in rejection_reasons:
                            rejection_reasons[reason] += 1
                        # Might be added to rest_points in _evaluate_candidate
                        self.total_rejected_points += 1
                except Exception as e:
                    logger.error(f"Error evaluating candidate ({y},{x}): {e}")
                    # Clear processing state
                    self.grid.update_state(y, x, 0, clear=STATE_PROCESSING)
                    self.total_rejected_points += 1
                    rejection_reasons["exception"] += 1
        
        # Log rejection statistics
        total_candidates = len(candidates)
        success_count = len(successful_candidates)
        logger.info(f"Candidate evaluation: {success_count}/{total_candidates} successful "
                   f"({success_count/total_candidates*100:.1f}% success rate)")
        logger.info(f"Rejection reasons: {rejection_reasons}")
        
        return successful_candidates
        
    def _evaluate_candidate_with_reason(self, y: int, x: int) -> Tuple[bool, str]:
        """Wrapper for _evaluate_candidate that returns success and rejection reason"""
        # Call the actual evaluation function and capture pass/fail
        success = self._evaluate_candidate(y, x)
        reason = self.grid.get_point_rejection_reason(y, x) if not success else None
        return success, reason
    
    def _evaluate_candidate(self, y: int, x: int) -> bool:
        """
        Evaluate a single candidate point.
        
        This implements the candidate evaluation from lines 1158-1333
        of the C++ reference implementation, including:
        1. Reference point counting and selection
        2. Initial optimization
        3. Final quality testing
        
        Args:
            y: Y coordinate of the candidate
            x: X coordinate of the candidate
            
        Returns:
            True if the candidate was successful, False otherwise
        """
        try:
            if self.debug:
                fringe_debug(f"Evaluating candidate point ({y},{x})")
            
            # Skip if already processed by another thread
            if (self.grid.get_state(y, x) & (STATE_LOC_VALID | STATE_COORD_VALID)):
                if self.debug:
                    fringe_debug(f"Point ({y},{x}) already processed, skipping")
                return False
            
            # 1. Reference point counting and neighborhood quality assessment
            # This follows lines 1172-1214 of the C++ implementation
            ref_count, ref_points, best_ref = self._count_reference_points(y, x)
            if self.debug:
                fringe_debug(f"Point ({y},{x}) has {ref_count} reference points")
            
            # Calculate recursive reference count (simulating the rec_ref_sum)
            # This is a way to consider not just immediate neighbors but their connectivity
            rec_ref_sum = 0
            for ref_y, ref_x in ref_points:
                # For each reference point, count its reference points
                ref_count_nested = self.grid.get_neighbor_count(
                    ref_y, ref_x, self.reference_radius, STATE_LOC_VALID
                )
                # Add to recursive sum with a weight
                rec_ref_sum += min(ref_count_nested, 4) / 4.0
            
            # Log reference counts for debugging
            if self.debug:
                fringe_debug(f"Point ({y},{x}): ref_count={ref_count}, rec_ref_sum={rec_ref_sum:.2f}, min={self.current_reference_min}")
            
            # FIXED: Always use the absolute minimum from the class instance
            # absolute_min_references maintains the hard minimum (2)
            # current_reference_min is the preference metric that increases over time
            
            # Check if we have enough reference points for minimum threshold
            # C++ checks this on line 1209: if(ref_count < 2 || ref_count+0.35*rec_ref_sum < curr_ref_min)
            if ref_count < self.absolute_min_references:
                # Not enough references for absolute minimum, add to rest points
                if self.debug:
                    fringe_debug(f"Point ({y},{x}) has insufficient references ({ref_count} direct, {rec_ref_sum:.2f} recursive), absolute min={self.absolute_min_references}")
                self.grid.update_state(y, x, 0, clear=STATE_PROCESSING)
                self.grid.set_point_rejection_reason(y, x, "insufficient_refs")
                self.rest_points.append((y, x))
                return False
                
            # Use the exact same formula as C++ for evaluating reference count adequacy
            # In C++, the formula is: ref_count + 0.35 * rec_ref_sum < curr_ref_min
            combined_ref_count = ref_count + 0.35 * rec_ref_sum
            
            # If we have enough references for absolute minimum, check preference threshold
            if combined_ref_count < self.current_reference_min:
                # C++ implementation is more aggressive - checks if curr_points.size() < 10
                # In our case, that means checking if the fringe is small
                # C++ adds these points to "rest_ref_pts" but still continues optimizing them
                # See lines 1212-1228 in SurfaceHelpers.cpp
                
                # For us, be more permissive: If fringe is small (<20) or we have few rest points,
                # continue processing this candidate anyway (like C++ does)
                if len(self.grid.fringe) < 20 or len(self.rest_points) < 20:
                    # Continue with optimization (fall through) - C++ is more permissive here
                    if self.debug:
                        fringe_debug(f"Point ({y},{x}) has {ref_count} refs (combined {combined_ref_count:.2f}) below preferred {self.current_reference_min}, but continuing due to small fringe")
                else:
                    # We have enough good candidates, mark this for later
                    if self.debug:
                        fringe_debug(f"Point ({y},{x}) has {ref_count} refs (combined {combined_ref_count:.2f}) below preferred {self.current_reference_min}, adding to rest points")
                    # Add to rest points - but C++ would still try optimizing it
                    self.grid.update_state(y, x, 0, clear=STATE_PROCESSING)
                    self.rest_points.append((y, x))
                    return False
        
            # 2. Initial optimization
            # This follows lines 1216-1247 of the C++ implementation
            
            # Calculate average position of reference points
            avg_position = np.zeros(3, dtype=np.float32)
            for ref_y, ref_x in ref_points:
                avg_position += self.grid.get_point(ref_y, ref_x)
            avg_position /= len(ref_points)
            
            # Add small random offset exactly like C++ does on line 1217:
            # cv::Vec3d init = locs(best_l)+cv::Vec3d((rand()%1000)/10000.0-0.05,(rand()%1000)/10000.0-0.05,(rand()%1000)/10000.0-0.05);
            random_offset = np.random.uniform(-0.05, 0.05, 3)  # MATCH C++ EXACTLY: Offset range [-0.05, 0.05]
            
            # Initialize position based on best reference plus small random offset
            if best_ref is not None:
                best_y, best_x = best_ref
                init_position = self.grid.get_point(best_y, best_x) + random_offset
            else:
                # Fallback to average if no best reference
                init_position = avg_position + random_offset
            
            if self.debug:
                fringe_debug(f"Point ({y},{x}) initial position: {init_position}")
            
            # Set initial position
            self.grid.set_point(y, x, init_position)
            
            # Set up optimization
            self.grid.update_state(y, x, STATE_LOC_VALID | STATE_COORD_VALID)
            
            # Perform optimization with multiple attempts if needed
            loss = self._optimize_candidate(y, x)
            if self.debug:
                fringe_debug(f"Point ({y},{x}) first optimization loss: {loss}")
            
            # If loss is too high, try multiple random initializations
            # This matches C++ implementation which tries multiple initializations
            # in SurfaceHelpers.cpp lines 1230-1247 
            if loss > self.physical_fail_threshold:
                if self.debug:
                    fringe_debug(f"Point ({y},{x}) loss {loss} > threshold {self.physical_fail_threshold}, trying random initializations")
                best_loss = loss
                best_position = self.grid.get_point(y, x).copy()
                
                # C++ tries exactly 100 optimization attempts (SurfaceHelpers.cpp line 1234-1236)
                # for (int n=0;n<100;n++) {
                #     int range = step*10;
                #     locs(p) = avg + cv::Vec3d((rand()%(range*2))-range,(rand()%(range*2))-range,(rand()%(range*2))-range);
                # }
                
                # Calculate exact range like C++ does: step*10 (SurfaceHelpers.cpp line 1235)
                range_value = self.step_size * 10  # MATCH C++ EXACTLY: range = step*10
                
                for i in range(100):  # MATCH C++ EXACTLY: 100 attempts
                    # Reset to new random position using exact C++ range [-step*10, step*10]
                    random_offset = np.random.uniform(-range_value, range_value, 3)
                    new_init = avg_position + random_offset
                    self.grid.set_point(y, x, new_init)
                    
                    # Try optimization again
                    new_loss = self._optimize_candidate(y, x)
                    if self.debug:
                        fringe_debug(f"Point ({y},{x}) random init {i+1} loss: {new_loss}")
                    
                    # Keep track of best result
                    if new_loss < best_loss:
                        best_loss = new_loss
                        best_position = self.grid.get_point(y, x).copy()
                    
                    # If loss is good enough, stop early
                    if new_loss < self.physical_fail_threshold:
                        if self.debug:
                            fringe_debug(f"Point ({y},{x}) found good loss {new_loss}, stopping random search")
                        break
                
                # Restore best position found
                if best_loss < loss:
                    if self.debug:
                        fringe_debug(f"Point ({y},{x}) restoring best position with loss {best_loss}")
                    self.grid.set_point(y, x, best_position)
                    loss = best_loss
                
                # Add local optimization fallback as in C++ SurfaceHelpers.cpp lines 1297-1308
                # C++ tries local optimization with increasing radius if random initialization fails
                if best_loss > self.physical_fail_threshold:
                    if self.debug:
                        fringe_debug(f"Point ({y},{x}) still above threshold after random attempts, trying local optimization")
                    
                    # Try local optimization with increasing radius
                    for radius in range(1, 5):  # Similar to C++ max_local_opt_r=4
                        # Try local optimization with this radius
                        neighbors = self._get_neighbors_within_radius(y, x, radius)
                        if not neighbors:
                            continue
                        
                        # Create position based on average of neighbors
                        local_avg = np.zeros(3, dtype=np.float32)
                        for ny, nx in neighbors:
                            local_avg += self.grid.get_point(ny, nx)
                        local_avg /= len(neighbors)
                        
                        # Set position and optimize
                        self.grid.set_point(y, x, local_avg)
                        local_loss = self._optimize_candidate(y, x)
                        
                        if self.debug:
                            fringe_debug(f"Point ({y},{x}) local radius {radius} loss: {local_loss}")
                        
                        # Keep if better
                        if local_loss < best_loss:
                            best_loss = local_loss
                            best_position = self.grid.get_point(y, x).copy()
                            self.grid.set_point(y, x, best_position)
                            loss = best_loss
                        
                        # Early stop if good enough
                        if best_loss < self.physical_fail_threshold:
                            break
            
            # 3. Final quality test
            # This follows lines 1252-1286 of the C++ implementation
            
            # Check distance transform value at this point
            point_position = self.grid.get_point(y, x)
            distance_value = self._evaluate_distance(point_position)
            if self.debug:
                fringe_debug(f"Point ({y},{x}) distance value: {distance_value}")
            
            # Check paths to neighbors for quality
            path_checks_passed = self._check_paths_to_neighbors(y, x)
            if self.debug:
                fringe_debug(f"Point ({y},{x}) path checks passed: {path_checks_passed}")
            
            # Final quality decision - EXACTLY match C++ implementation
            # C++ uses dist_th = 1.5 (matching our self.distance_threshold)
            # C++ uses phys_fail_th = 0.1 (matching our self.physical_fail_threshold)
            # These values are defined in SurfaceHelpers.cpp lines 989 and 1102
            
            # CORRECTION: In C++, high intensity is good, not bad!
            # The distance_value is actually an intensity value in our implementation (0-255 range)
            # So we want to PASS points with HIGH intensity, not fail them
            # In SurfaceHelpers.cpp line 1263:
            # if(dtrans_value.at<float>(0,0) >= dist_th) continue;
            # The comparison is REVERSED - it fails points with intensity BELOW threshold
            
            # FIXED: Change the condition to properly check distance transform values
            # IMPORTANT: In the distance transform, LOW values (near 0.0) are GOOD!
            # They mean the point is close to the object boundary
            distance_passed = distance_value <= self.distance_threshold  # Use C++ dist_th=1.5
            loss_passed = loss < self.physical_fail_threshold  # phys_fail_th = 0.1 in C++
            
            # Always log detailed quality check results for debugging
            print(f"QUALITY CHECKS ({y},{x}): distance={distance_value:.6f} (should be ≤{self.distance_threshold}), loss={loss:.6f} (thresh={self.physical_fail_threshold}), paths={path_checks_passed}")
            
            # Final quality checks with correct distance transform logic (low is good)
            quality_passed = distance_passed and loss_passed and path_checks_passed
            
            # Log each check result
            print(f"  distance_passed: {distance_passed}, loss_passed: {loss_passed}, path_checks_passed: {path_checks_passed}")
            
            if quality_passed:
                # Success! Keep point as valid with both LOC_VALID and COORD_VALID
                print(f"PASSED ALL ({y},{x}): Point passed all quality checks")
                # Explicitly set both flags to match C++ behavior
                self.grid.update_state(y, x, STATE_LOC_VALID | STATE_COORD_VALID)
                return True
            else:
                # Failed quality checks, but still keep as COORD_VALID for future reference
                # This mirrors C++ behavior which maintains points even when they fail validation
                print(f"FAILED CHECK ({y},{x}): Failed quality checks")
                
                # Critical difference: Set STATE_COORD_VALID but clear STATE_LOC_VALID
                # This allows the point to be part of the surface but not used for reference counting
                self.grid.update_state(y, x, STATE_COORD_VALID, clear=STATE_LOC_VALID | STATE_PROCESSING)
                # Add to rest_points for potential future use like C++ does
                self.rest_points.append((y, x))
                # Track rejection reason for diagnostics
                if not distance_passed:
                    self.grid.set_point_rejection_reason(y, x, "distance_check_failed")
                    print(f"  REASON: distance value {distance_value:.6f} > threshold {self.distance_threshold} (too far from object)")
                elif not loss_passed:
                    self.grid.set_point_rejection_reason(y, x, "optimization_failed")
                    print(f"  REASON: loss value {loss:.6f} > threshold {self.physical_fail_threshold}")
                elif not path_checks_passed:
                    self.grid.set_point_rejection_reason(y, x, "path_check_failed")
                    print(f"  REASON: path checks failed")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating candidate ({y},{x}): {e}")
            # Mark as failed to ensure it's not left in processing state
            self.grid.update_state(y, x, 0, clear=STATE_PROCESSING | STATE_LOC_VALID)
            return False
    
    def _count_reference_points(self, y: int, x: int) -> Tuple[int, List[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Count valid reference points around a candidate point.
        
        This implements the reference point counting from lines 1172-1198
        of the C++ implementation.
        
        Args:
            y: Y coordinate of the candidate
            x: X coordinate of the candidate
            
        Returns:
            Tuple of (reference_count, list_of_reference_points, best_reference_point)
        """
        ref_points = []
        ref_count = 0
        
        # Search in reference radius
        min_y = max(0, y - self.reference_radius)
        max_y = min(self.grid.height, y + self.reference_radius + 1)
        min_x = max(0, x - self.reference_radius)
        max_x = min(self.grid.width, x + self.reference_radius + 1)
        
        # Find all valid reference points
        for ref_y in range(min_y, max_y):
            for ref_x in range(min_x, max_x):
                if self.grid.get_state(ref_y, ref_x) & STATE_LOC_VALID:
                    ref_count += 1
                    ref_points.append((ref_y, ref_x))
        
        # Find the best reference point (with most connected neighbors)
        best_ref = None
        best_ref_count = -1
        
        for ref_y, ref_x in ref_points:
            # Count valid neighbors for this reference
            ref_neighbor_count = self.grid.get_neighbor_count(ref_y, ref_x, 1, STATE_LOC_VALID)
            
            # Update best if this one has more neighbors
            if ref_neighbor_count > best_ref_count:
                best_ref_count = ref_neighbor_count
                best_ref = (ref_y, ref_x)
        
        return ref_count, ref_points, best_ref
    
    
    def _evaluate_distance(self, point: np.ndarray) -> float:
        """
        Evaluate the distance transform value at a point.
        
        Args:
            point: 3D point coordinates in ZYX order
            
        Returns:
            Distance value at the point
        """
        # Sample the volume at this point
        try:
            # Check for valid point coordinates
            if np.any(np.isnan(point)) or np.any(np.isinf(point)):
                logger.error(f"Cannot evaluate distance: Invalid coordinates in point: {point}")
                return float('inf')
            
            # Get distance value at the point using the optimizer's distance transform sampler
            # IMPORTANT: This should return the distance transform value, not raw intensity
            value = self.optimizer.sample_distance_at_point_3d(point)
            
            # For debugging
            if self.debug:
                fringe_debug(f"Distance value at point {point}: {value}")
            
            # Check intensity/distance value
            if value > self.distance_threshold and self.debug:
                fringe_debug(f"Point at {point} distance value {value} exceeds threshold {self.distance_threshold}")
            
            return value
        except Exception as e:
            logger.error(f"Error evaluating distance at point {point}: {e}")
            # Return a high distance value to indicate failure
            return float('inf')
    
    def _check_paths_to_neighbors(self, y: int, x: int) -> bool:
        """
        Check quality of paths to neighboring points.
        
        This follows the quality checking in lines 1267-1281 of the C++ implementation.
        
        Args:
            y: Y coordinate of the candidate
            x: X coordinate of the candidate
            
        Returns:
            True if paths to neighbors pass quality checks, False otherwise
        """
        # Get the point's coordinates
        point = self.grid.get_point(y, x)
        
        # Check paths in each cardinal direction
        for dy, dx in self.grid.neighbors:
            ny, nx = y + dy, x + dx
            
            # Skip if out of bounds
            if not self.grid.is_in_bounds(ny, nx):
                continue
                
            # Skip if not a valid point
            if not (self.grid.get_state(ny, nx) & STATE_LOC_VALID):
                continue
            
            # Get neighbor coordinates
            neighbor = self.grid.get_point(ny, nx)
            
            # Check the path between points (sample multiple points along line)
            path_quality = self._evaluate_path_quality(point, neighbor)
            
            # If any path fails quality, return False
            if not path_quality:
                return False
        
        # All paths passed
        return True
    
    def _evaluate_path_quality(self, start: np.ndarray, end: np.ndarray, samples: int = 5) -> bool:
        """
        Evaluate quality of a path between two points.
        
        Args:
            start: Starting point
            end: Ending point
            samples: Number of samples to take along the path
            
        Returns:
            True if path passes quality checks, False otherwise
        """
        try:
            # Sample points along the path
            for t in np.linspace(0, 1, samples):
                # Interpolate point
                point = start * (1 - t) + end * t
                
                # Check intensity value (distance_value is actually intensity in our case)
                intensity = self._evaluate_distance(point)
                
                # FIXED: Check against distance threshold (low values are good)
                # This matches the C++ logic: distance values should be LESS THAN dist_th
                if intensity > self.distance_threshold:
                    print(f"PATH CHECK FAILED: Point along path with distance {intensity} > {self.distance_threshold} threshold")
                    return False
            
            # All samples passed - all points along path had low distance values
            print(f"PATH CHECK PASSED: All points along path have distance < {self.distance_threshold}")
            return True
        except Exception as e:
            logger.error(f"Error evaluating path from {start} to {end}: {e}")
            return False
    
    def _attempt_fringe_recovery(self) -> bool:
        """
        Attempt to recover when fringe is empty.
        
        This implements the fringe recovery strategy from lines 1336-1347
        of the C++ implementation, with improvements to handle preference vs. minimum thresholds.
        
        Returns:
            True if recovery was successful, False otherwise
        """
        # If fringe is not empty, no need for recovery
        if len(self.grid.fringe) > 0:
            return True
        
        # FIXED: Use the class-level absolute_min_references rather than hardcoding
        
        # Try different recovery strategies based on C++ implementation
        # In SurfaceHelpers.cpp lines 1336-1347, the recovery mechanism reduces reference threshold
        # and is more aggressive
        self.recoveries += 1
        
        # 1. For C++ style recovery, be more aggressive with threshold reduction
        # Especially in later recoveries
        old_min = self.current_reference_min
        if self.current_reference_min > self.absolute_min_references:
            # C++ style: more aggressive reduction after multiple recovery attempts
            if self.recoveries > 2:
                # After multiple recoveries, make larger reductions
                reduction = min(3, self.current_reference_min - self.absolute_min_references)
                self.current_reference_min = max(self.absolute_min_references, self.current_reference_min - reduction)
            else:
                # First few recoveries, reduce by 1
                self.current_reference_min -= 1
                
            if self.debug:
                fringe_debug(f"C++ style recovery #{self.recoveries}: Reducing reference threshold from {old_min} to {self.current_reference_min}")
            else:
                logger.info(f"Recovery #{self.recoveries}: Reducing reference threshold to {self.current_reference_min} (C++ style)")
            
            # Try to recover from rest points first using the new threshold
            recovered_from_rest = False
            new_fringe = []
            
            # Check all rest points against the new threshold
            for y, x in self.rest_points:
                ref_count, _, _ = self._count_reference_points(y, x)
                if ref_count >= self.current_reference_min:
                    # This point might work now
                    new_fringe.append((y, x))
                    recovered_from_rest = True
            
            # If we recovered points from rest with the new threshold, use those
            if recovered_from_rest:
                # Sort by reference count to get the best candidates first
                sorted_fringe = []
                for y, x in new_fringe:
                    ref_count, _, _ = self._count_reference_points(y, x)
                    sorted_fringe.append((ref_count, (y, x)))
                
                sorted_fringe.sort(reverse=True)  # Sort by reference count (higher first)
                
                # C++ style: In recovery mode, take more points to increase chances of success
                # Take a subset of points with highest reference counts, but ensure we take more than standard
                recovery_count = max(20, min(40, len(sorted_fringe)))  # C++ takes more points in recovery
                new_fringe = [point for _, point in sorted_fringe[:recovery_count]]
                
                if self.debug:
                    fringe_debug(f"C++ style recovery: Recovered {len(new_fringe)} points from rest_points using reduced threshold {self.current_reference_min}")
                else:
                    logger.info(f"Recovery: Found {len(new_fringe)} points with threshold {self.current_reference_min} (C++ style recovery)")
                self.grid.fringe = new_fringe
                return True
        
        # 2. If reducing threshold didn't work or we're already at minimum, try using absolute minimum
        if self.debug:
            fringe_debug(f"Moving to recovery strategy 2: Using absolute minimum reference count {self.absolute_min_references}")
        else:
            logger.info(f"Recovery: Trying with absolute minimum of {self.absolute_min_references} references")
        
        # Even if we already tried with current_reference_min = 2, we'll try again with absolute_min_references
        # This time searching all rest points again
        new_fringe = []
        for y, x in self.rest_points:
            ref_count, _, _ = self._count_reference_points(y, x)
            if ref_count >= self.absolute_min_references:
                # This point meets absolute minimum requirement
                new_fringe.append((y, x))
        
        if new_fringe:
            # Sort by reference count
            sorted_fringe = []
            for y, x in new_fringe:
                ref_count, _, _ = self._count_reference_points(y, x)
                sorted_fringe.append((ref_count, (y, x)))
            
            sorted_fringe.sort(reverse=True)  # Sort by reference count (higher first)
            
            # C++ implementation takes more points during recovery
            # Use more aggressive recovery by taking a larger subset
            recovery_count = max(30, min(50, len(sorted_fringe)))  # More aggressive count
            new_fringe = [point for _, point in sorted_fringe[:recovery_count]]
            
            if self.debug:
                fringe_debug(f"C++ style recovery: Recovered {len(new_fringe)} points from rest_points using absolute minimum {self.absolute_min_references}")
            else:
                logger.info(f"Recovery: Found {len(new_fringe)} points with absolute minimum {self.absolute_min_references} (C++ style)")
            
            self.grid.fringe = new_fringe
            # C++ implementation sets reference threshold to absolute minimum during recovery
            # This is more aggressive than our original approach
            self.current_reference_min = self.absolute_min_references  # Force to absolute minimum like C++
            
            if self.debug:
                fringe_debug(f"Reset preference threshold to {self.current_reference_min} based on recovered points")
            return True
            
        # 3. If no rest points pass even the absolute minimum, try all valid points
        if self.debug:
            fringe_debug(f"Moving to recovery strategy 3: Collecting from all valid points")
        else:
            logger.info(f"Recovery: Trying to gather points from all valid surface points")
        
        # Collect all valid points in the used area
        bounds = self.grid.get_used_rect()
        min_x, min_y, width, height = bounds
        
        all_valid_points = []
        for y in range(min_y, min_y + height):
            for x in range(min_x, min_x + width):
                if self.grid.get_state(y, x) & STATE_LOC_VALID:
                    all_valid_points.append((y, x))
        
        # Sort valid points by number of valid neighbors (better connectivity)
        sorted_valid_points = []
        for y, x in all_valid_points:
            # Count valid neighbors for better connectivity
            neighbor_count = self.grid.get_neighbor_count(y, x, 1, STATE_LOC_VALID)
            sorted_valid_points.append((neighbor_count, (y, x)))
        
        sorted_valid_points.sort(reverse=True)  # Sort by neighbor count
        
        # C++ implementation is very aggressive in final recovery stage
        # Take many more points (this is our last chance at recovery)
        recovery_count = max(50, min(100, len(sorted_valid_points)))  # Much more aggressive
        self.grid.fringe = [point for _, point in sorted_valid_points[:recovery_count]]
        
        if self.debug:
            fringe_debug(f"Final C++ style recovery: Recovered {len(self.grid.fringe)} points from all valid points")
        else:
            logger.info(f"Recovery: Found {len(self.grid.fringe)} points from all valid surface points (C++ aggressive recovery)")
        
        # C++ style: Reset reference threshold to absolute minimum and keep it there for a while
        self.current_reference_min = self.absolute_min_references
        # Also reset the recovery counter to give this a fresh start
        self.recoveries = 0  # Reset counter to avoid increasing thresholds too quickly after recovery
        if self.debug:
            fringe_debug(f"Reset preference threshold to absolute minimum {self.absolute_min_references}")
        
        # If we found any points, recovery was successful
        return len(self.grid.fringe) > 0
    
    def expand_generations(self, num_generations: int, min_points: int = 0) -> int:
        """
        Expand the surface for a specified number of generations.
        
        Args:
            num_generations: Maximum number of generations to grow
            min_points: Minimum number of valid points to generate
            
        Returns:
            Total number of valid points added
        """
        starting_points = self.total_valid_points
        
        # Log debugging status
        logger.info(f"Starting expansion with {num_generations} generations limit, {min_points} min points")
        if FRINGE_DEBUG_ENABLED:
            fringe_debug("Fringe debugging is ENABLED via FRINGE_DEBUG environment variable")
        if GENERAL_DEBUG_ENABLED:
            print_debug("General debugging is ENABLED via DEBUG environment variable")
        
        for gen in range(num_generations):
            # Expand one generation
            new_points = self.expand_one_generation()
            
            # Check for early termination conditions
            if new_points == 0:
                logger.info(f"No new points added in generation {gen}, stopping")
                break
                
            # Log progress periodically
            if gen % 5 == 0 or gen == num_generations - 1:
                if self.debug:
                    fringe_debug(f"Generation {gen}: {self.total_valid_points} valid points, "
                               f"fringe size {len(self.grid.fringe)}, "
                               f"recoveries: {self.recoveries}")
                else:
                    # More concise version for info level
                    logger.info(f"Generation {gen}: {self.total_valid_points} valid points, {len(self.grid.fringe)} in fringe")
            
            # Check if we've reached the minimum point count
            if min_points > 0 and self.total_valid_points >= min_points:
                logger.info(f"Reached minimum point count ({min_points}), stopping")
                break
                
            # Check if the fringe is empty (no more candidates possible)
            if len(self.grid.fringe) == 0:
                logger.info("No valid points in fringe, stopping")
                break
        
        # Return total points added
        return self.total_valid_points - starting_points
        
    def _count_reference_points(self, y: int, x: int) -> Tuple[int, List[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Count reference points for a candidate and find the best reference.
        This matches the C++ implementation in SurfaceHelpers.cpp lines 1172-1214.
        
        Args:
            y: Y coordinate of the candidate
            x: X coordinate of the candidate
            
        Returns:
            (ref_count, ref_points, best_ref):
            - ref_count: Number of reference points
            - ref_points: List of (y, x) coordinates of reference points
            - best_ref: Coordinates of the best reference point (or None)
        """
        # Get reference points (valid neighbors within reference_radius)
        ref_points = []
        
        # Count reference points (LOC_VALID neighbors within reference_radius)
        for dy in range(-self.reference_radius, self.reference_radius+1):
            for dx in range(-self.reference_radius, self.reference_radius+1):
                # Skip the center point
                if dy == 0 and dx == 0:
                    continue
                
                # Check neighbor
                ny, nx = y + dy, x + dx
                if not self.grid.is_in_bounds(ny, nx):
                    continue
                    
                # Only count LOC_VALID points as references
                if self.grid.get_state(ny, nx) & STATE_LOC_VALID:
                    ref_points.append((ny, nx))
        
        # Find the best reference point (with most connected neighbors)
        best_ref = None
        best_ref_count = -1
        
        for ref_y, ref_x in ref_points:
            # Count valid neighbors for this reference
            ref_neighbor_count = self.grid.get_neighbor_count(ref_y, ref_x, 1, STATE_LOC_VALID)
            
            # Update best if this one has more neighbors
            if ref_neighbor_count > best_ref_count:
                best_ref_count = ref_neighbor_count
                best_ref = (ref_y, ref_x)
        
        return len(ref_points), ref_points, best_ref
        
    def _optimize_candidate(self, y: int, x: int) -> float:
        """
        Optimize a candidate point and normalize the loss like C++.
        
        Args:
            y: Y coordinate of candidate point
            x: X coordinate of candidate point
            
        Returns:
            Normalized loss value (lower is better)
        """
        try:
            # Convert y and x to proper integer values to avoid any variable name issues
            y_val = int(y)
            x_val = int(x)
            
            # Get the raw loss from the optimizer using the new optimize_point method
            # The optimizer.optimize_point method will now return a high but not infinite value on failure
            print(f"FRINGE_OPTIMIZE: Calling optimizer.optimize_point with y={y_val}, x={x_val}")
            raw_loss = self.optimizer.optimize_point(y_val, x_val)
            
            # Check for extreme or invalid loss values
            if np.isnan(raw_loss) or np.isinf(raw_loss):
                logger.warning(f"Invalid loss value {raw_loss} for point ({y},{x}), using fallback value")
                raw_loss = 10.0  # Use a reasonable high value, not infinite
            elif raw_loss > 100.0:
                logger.warning(f"Extremely high loss value {raw_loss} for point ({y},{x}), capping")
                raw_loss = 10.0  # Cap extremely high values
            
            # Estimate number of constraints based on reference points
            ref_count, _, _ = self._count_reference_points(y, x)
            num_constraints = max(1, ref_count * 2)  # ~2 constraints per reference
            
            # Apply normalization like C++ does in SurfaceHelpers.cpp:
            # Line 334: return sqrt(best/tgts.size());
            # Line 406: return sqrt(best/tgts.size());
            # Line 819: return sqrt(summary.final_cost/summary.num_residual_blocks);
            normalized_loss = np.sqrt(raw_loss / num_constraints)
            
            if self.debug:
                fringe_debug(f"Point ({y},{x}) loss normalized: sqrt({raw_loss:.6f} / {num_constraints}) = {normalized_loss:.6f}")
                if normalized_loss < self.physical_fail_threshold:
                    fringe_debug(f"Point ({y},{x}) normalized loss {normalized_loss:.6f} is below threshold {self.physical_fail_threshold}")
                else:
                    fringe_debug(f"Point ({y},{x}) normalized loss {normalized_loss:.6f} exceeds threshold {self.physical_fail_threshold}")
                
            return normalized_loss
        except Exception as e:
            logger.error(f"Error optimizing point ({y},{x}): {e}")
            # Return a capped high value rather than infinity
            return 5.0  # High enough to exceed threshold but not infinite
    
    def _get_neighbors_within_radius(self, y: int, x: int, radius: int) -> List[Tuple[int, int]]:
        """
        Get neighbors within a given radius for local optimization.
        This matches the C++ local optimization approach in SurfaceHelpers.cpp.
        
        Args:
            y: Y coordinate of the center point
            x: X coordinate of the center point
            radius: Radius to search for neighbors
            
        Returns:
            List of (y, x) coordinates of valid neighbors within radius
        """
        neighbors = []
        
        # Search in a square around the point
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                # Skip the center point
                if dy == 0 and dx == 0:
                    continue
                
                # Check if within radius (use L2 norm)
                if np.sqrt(dy*dy + dx*dx) > radius:
                    continue
                
                # Check if neighbor is valid
                ny, nx = y + dy, x + dx
                if not self.grid.is_in_bounds(ny, nx):
                    continue
                
                # Only use LOC_VALID points as references
                if self.grid.get_state(ny, nx) & STATE_LOC_VALID:
                    neighbors.append((ny, nx))
        
        return neighbors