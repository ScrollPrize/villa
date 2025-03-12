"""
Grid data structures for surface tracing.

This module contains classes for representing and manipulating 2D grids of
3D points used in surface tracing.
"""

from enum import IntFlag
from typing import List, Tuple, Optional, Dict, Set, Union, Any

import numpy as np
import torch


# Grid state constants (bitflags)
STATE_NONE = 0
STATE_PROCESSING = 1     # Currently being processed
STATE_LOC_VALID = 2      # Point has valid location (surface is valid here)
STATE_COORD_VALID = 4    # Point has valid coordinates (but might not be on surface)


class GridState(IntFlag):
    """State flags for grid points."""
    NONE = STATE_NONE
    PROCESSING = STATE_PROCESSING
    LOC_VALID = STATE_LOC_VALID
    COORD_VALID = STATE_COORD_VALID


class PointGrid:
    """
    A 2D grid of 3D points for surface tracing.
    
    This class represents a 2D grid of 3D points, which is the core data structure
    for the surface tracing algorithm. Each point in the grid is a 3D coordinate,
    and the grid also tracks the state of each point.
    
    Important coordinate conventions:
    - Points are stored in a [height, width, 3] array where each point follows ZYX ordering
    - The 3D coordinates are always in [z, y, x] order, consistent with cost functions
    - Grid coordinates are accessed as grid[y, x] for 2D indexing 
    """
    
    # Possible rejection reasons for point candidates
    REJECTION_REASONS = {
        "insufficient_refs": "Insufficient reference points",
        "optimization_failed": "Optimization failed to converge",
        "distance_check_failed": "Distance value exceeds threshold",
        "path_check_failed": "Path quality check failed",
        "unknown": "Unknown rejection reason"
    }
    
    def set_point_rejection_reason(self, y: int, x: int, reason: str):
        """Store the rejection reason for a point."""
        self.rejection_reasons[(y, x)] = reason
        
    def get_point_rejection_reason(self, y: int, x: int) -> str:
        """Get the rejection reason for a point."""
        return self.rejection_reasons.get((y, x), "unknown")
    
    def __init__(self, width: int, height: int, dtype: type = np.float32):
        """
        Initialize a new point grid.
        
        Args:
            width: Width of the grid
            height: Height of the grid
            dtype: Data type for the grid points
        """
        self.width = width
        self.height = height
        self.dtype = dtype
        
        # Points array - shape: [height, width, 3]
        self.points = np.zeros((height, width, 3), dtype=dtype)
        # Initialize points to invalid values (-1, -1, -1)
        self.points.fill(-1)
        
        # State matrix - shape: [height, width]
        self.state = np.zeros((height, width), dtype=np.uint8)
        
        # Boundary and rectangle tracking [min_x, min_y, max_x, max_y]
        self.boundary_rect = [width // 2, height // 2, width // 2 + 1, height // 2 + 1]
        self.center = np.array([height // 2, width // 2], dtype=np.int32)  # [y, x]
        
        # Used for tracking fringe (boundary points)
        self.fringe: List[Tuple[int, int]] = []
        
        # For optimization metrics
        self.generation = 0
        self.success_count = 0
        self.generation_max_cost: List[float] = []
        self.generation_avg_cost: List[float] = []
        
        # Track rejection reasons for points (for debugging)
        self.rejection_reasons: Dict[Tuple[int, int], str] = {}
        
        # Neighbor offsets for 4-connected grid
        self.neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
    def initialize_at_origin(self, origin: np.ndarray, step_size: float, use_gradients: bool = False, volume_loader = None):
        """
        Initialize the grid at the given origin point.
        
        Args:
            origin: 3D origin point
            step_size: Step size for the surface
            use_gradients: If True, use volume gradients to help determine initial positions
            volume_loader: Function to sample volume (for use_gradients)
        """
        # Convert to numpy array if needed
        if not isinstance(origin, np.ndarray):
            origin = np.array(origin, dtype=self.dtype)
            
        print(f"DEBUG GRID: Initializing grid at origin {origin} with step_size {step_size}")
        print(f"DEBUG GRID: Grid dimensions: {self.width}x{self.height}")
            
        # Get center coordinates
        cy, cx = self.center
        print(f"DEBUG GRID: Center coordinates: ({cy}, {cx})")
        
        # Initialize a 2x2 grid around the center
        if use_gradients and volume_loader is not None:
            # Use volume information to determine initial directions
            try:
                # Try to get gradient at origin
                print(f"DEBUG GRID: Calling volume_loader with origin {origin}")
                
                # Note: We need to pass torch tensors to the volume_loader
                # Convert origin coordinates to tensors with batch dimension
                z_tensor = torch.tensor([[float(origin[0])]], dtype=torch.float32)
                y_tensor = torch.tensor([[float(origin[1])]], dtype=torch.float32)
                x_tensor = torch.tensor([[float(origin[2])]], dtype=torch.float32)
                
                print(f"DEBUG GRID: Calling volume_loader with tensors z={z_tensor.shape}, y={y_tensor.shape}, x={x_tensor.shape}")
                _, gradient = volume_loader(z_tensor, y_tensor, x_tensor)
                print(f"DEBUG GRID: Got gradient of type {type(gradient)}")
                
                # Determine perpendicular directions to gradient for better seed point placement
                # Calculate two orthogonal vectors to the gradient for better seed placement
                try:
                    print(f"DEBUG GRID: Attempting to compute gradient norm")
                    # Check if gradient is tensor or ndarray first
                    if isinstance(gradient, torch.Tensor):
                        # Extract the first gradient in the batch
                        if len(gradient.shape) >= 3 and gradient.shape[-1] == 3:
                            print(f"DEBUG GRID: Extracting first gradient from tensor of shape {gradient.shape}")
                            gradient_np = gradient[0, 0].detach().cpu().numpy()
                        elif len(gradient.shape) == 2 and gradient.shape[-1] == 3:
                            print(f"DEBUG GRID: Extracting gradient from tensor of shape {gradient.shape}")
                            gradient_np = gradient[0].detach().cpu().numpy()
                        else:
                            print(f"DEBUG GRID: Unexpected gradient tensor shape {gradient.shape}, falling back to standard initialization")
                            # Fall back to standard initialization
                            raise ValueError(f"Unexpected gradient tensor shape {gradient.shape}")
                    elif isinstance(gradient, np.ndarray) and gradient.shape == (3,):
                        gradient_np = gradient
                    else:
                        print(f"DEBUG GRID: Gradient is not a usable tensor or ndarray, shape={getattr(gradient, 'shape', 'unknown')}")
                        # Fall back to standard initialization
                        raise ValueError(f"Gradient is not a usable tensor or ndarray")
                        
                    grad_norm = np.linalg.norm(gradient_np)
                    print(f"DEBUG GRID: Gradient norm = {grad_norm}")
                except Exception as e:
                    print(f"DEBUG GRID: Error computing gradient norm: {e}")
                    # Fall back to standard initialization
                    self._standard_initialization(origin, step_size, cy, cx)
                    return
                
                if grad_norm > 1e-6:  # Only use gradient if it's meaningful
                    # Normalize gradient
                    normalized_grad = gradient_np / grad_norm
                    
                    # Find perpendicular vectors using cross products
                    # We need 2 vectors that are perpendicular to the gradient and to each other
                    if abs(normalized_grad[0]) < abs(normalized_grad[1]) and abs(normalized_grad[0]) < abs(normalized_grad[2]):
                        perp1 = np.array([0.0, -normalized_grad[2], normalized_grad[1]])
                    elif abs(normalized_grad[1]) < abs(normalized_grad[0]) and abs(normalized_grad[1]) < abs(normalized_grad[2]):
                        perp1 = np.array([-normalized_grad[2], 0.0, normalized_grad[0]])
                    else:
                        perp1 = np.array([-normalized_grad[1], normalized_grad[0], 0.0])
                    
                    # Normalize
                    perp1 = perp1 / np.linalg.norm(perp1)
                    
                    # Second perpendicular vector (cross product of gradient and first perpendicular)
                    perp2 = np.cross(normalized_grad, perp1)
                    perp2 = perp2 / np.linalg.norm(perp2)
                    
                    # Use these vectors to create better-spaced initial points
                    delta = step_size * 0.3  # Start with smaller step for stability
                    self.points[cy, cx] = origin
                    self.points[cy, cx+1] = origin + delta * perp1
                    self.points[cy+1, cx] = origin + delta * perp2
                    self.points[cy+1, cx+1] = origin + delta * (perp1 + perp2)
                else:
                    # Fallback to standard initialization if gradient is too small
                    self._standard_initialization(origin, step_size, cy, cx)
            except Exception as e:
                print(f"Error using gradient for initialization: {e}")
                # Fallback to standard initialization
                self._standard_initialization(origin, step_size, cy, cx)
        else:
            # Use standard initialization without gradients
            self._standard_initialization(origin, step_size, cy, cx)
        
        # Set states to valid for all 4 seed points
        self.state[cy, cx] = STATE_LOC_VALID | STATE_COORD_VALID
        self.state[cy, cx+1] = STATE_LOC_VALID | STATE_COORD_VALID
        self.state[cy+1, cx] = STATE_LOC_VALID | STATE_COORD_VALID
        self.state[cy+1, cx+1] = STATE_LOC_VALID | STATE_COORD_VALID
        
        # Initialize fringe with the 4 seed points
        self.fringe = [(cy, cx), (cy, cx+1), (cy+1, cx), (cy+1, cx+1)]
        
    def _standard_initialization(self, origin, step_size, cy, cx):
        """
        Standard initialization logic for seed points.
        
        Args:
            origin: 3D origin point in ZYX order
            step_size: Step size for the surface
            cy, cx: Center coordinates
        """
        # Use a larger delta for initial grid to encourage faster growth outward
        delta = step_size * 0.8  # Larger fraction of step_size for better spacing
        print(f"DEBUG GRID: Using delta of {delta} for seed points (step_size={step_size})")
        print(f"DEBUG GRID: Origin point: {origin} in ZYX order")
        
        # Make the seed points spread out more in 3D, rather than just in XY plane
        # Note: Using ZYX coordinate ordering for all points
        self.points[cy, cx] = origin
        
        # Original offsets
        offset_x = np.array([0, 0, delta])  # Offset in X axis (last component of ZYX)
        offset_y = np.array([0, delta, 0])  # Offset in Y axis (middle component of ZYX)
        offset_diag = np.array([delta/2, delta/2, delta/2])  # Diagonal offset
        
        print(f"DEBUG GRID: X offset: {offset_x} in ZYX order")
        print(f"DEBUG GRID: Y offset: {offset_y} in ZYX order")
        print(f"DEBUG GRID: Diagonal offset: {offset_diag} in ZYX order")
        
        self.points[cy, cx+1] = origin + offset_x  # Z fixed, Y fixed, X+delta
        self.points[cy+1, cx] = origin + offset_y  # Z fixed, Y+delta, X fixed
        self.points[cy+1, cx+1] = origin + offset_diag
        
        print(f"DEBUG GRID: Seed point 1 at ({cy}, {cx}): {self.points[cy, cx]}")
        print(f"DEBUG GRID: Seed point 2 at ({cy}, {cx+1}): {self.points[cy, cx+1]}")
        print(f"DEBUG GRID: Seed point 3 at ({cy+1}, {cx}): {self.points[cy+1, cx]}")
        print(f"DEBUG GRID: Seed point 4 at ({cy+1}, {cx+1}): {self.points[cy+1, cx+1]}")
        
    def get_point(self, y: int, x: int) -> np.ndarray:
        """
        Get the point at the given coordinates.
        
        Args:
            y: Y coordinate in the grid
            x: X coordinate in the grid
            
        Returns:
            The 3D point at the given coordinates
        """
        return self.points[y, x]
    
    def set_point(self, y: int, x: int, point: np.ndarray, state: int = STATE_NONE):
        """
        Set the point at the given coordinates.
        
        Args:
            y: Y coordinate in the grid
            x: X coordinate in the grid
            point: 3D point to set
            state: State flags to set
        """
        self.points[y, x] = point
        if state != STATE_NONE:
            self.state[y, x] = state
        
        # Update boundary rectangle
        self.boundary_rect[0] = min(self.boundary_rect[0], x)
        self.boundary_rect[1] = min(self.boundary_rect[1], y)
        self.boundary_rect[2] = max(self.boundary_rect[2], x + 1)
        self.boundary_rect[3] = max(self.boundary_rect[3], y + 1)
    
    def get_state(self, y: int, x: int) -> int:
        """
        Get the state at the given coordinates.
        
        Args:
            y: Y coordinate in the grid
            x: X coordinate in the grid
            
        Returns:
            The state flags at the given coordinates
        """
        return self.state[y, x]
    
    def set_state(self, y: int, x: int, state: int):
        """
        Set the state at the given coordinates.
        
        Args:
            y: Y coordinate in the grid
            x: X coordinate in the grid
            state: State flags to set
        """
        self.state[y, x] = state
    
    def update_state(self, y: int, x: int, state: int, clear: int = STATE_NONE):
        """
        Update the state at the given coordinates.
        
        Args:
            y: Y coordinate in the grid
            x: X coordinate in the grid
            state: State flags to set
            clear: State flags to clear (default: none)
        """
        if clear != STATE_NONE:
            self.state[y, x] &= ~clear
        self.state[y, x] |= state
    
    def is_in_bounds(self, y: int, x: int) -> bool:
        """
        Check if the given coordinates are in bounds.
        
        Args:
            y: Y coordinate in the grid
            x: X coordinate in the grid
            
        Returns:
            True if the coordinates are in bounds, False otherwise
        """
        return 0 <= y < self.height and 0 <= x < self.width
    
    def is_valid(self, y: int, x: int) -> bool:
        """
        Check if the point at the given coordinates is valid.
        
        Args:
            y: Y coordinate in the grid
            x: X coordinate in the grid
            
        Returns:
            True if the point is valid (LOC_VALID), False otherwise
        """
        return self.is_in_bounds(y, x) and (self.state[y, x] & STATE_LOC_VALID) != 0
    
    def get_neighbor_count(self, y: int, x: int, radius: int = 1, 
                          state_mask: int = STATE_LOC_VALID) -> int:
        """
        Count the neighbors with the given state within the given radius.
        
        Args:
            y: Y coordinate in the grid
            x: X coordinate in the grid
            radius: Radius to search within
            state_mask: State flags to match
            
        Returns:
            Number of neighbors with the given state
        """
        count = 0
        min_y = max(0, y - radius)
        max_y = min(self.height, y + radius + 1)
        min_x = max(0, x - radius)
        max_x = min(self.width, x + radius + 1)
        
        for ny in range(min_y, max_y):
            for nx in range(min_x, max_x):
                if (self.state[ny, nx] & state_mask) == state_mask:
                    count += 1
                    
        return count
    
    def get_candidate_points(self) -> List[Tuple[int, int]]:
        """
        Get a list of candidate points for expansion.
        
        This finds points adjacent to the current fringe that are not yet
        processed or valid.
        
        Returns:
            List of (y, x) coordinates for candidate points
        """
        candidates = []
        processed = set()
        
        # Start from current fringe
        for y, x in self.fringe:
            if not self.is_valid(y, x):
                continue
                
            # Check neighbors
            for dy, dx in self.neighbors:
                ny, nx = y + dy, x + dx
                
                # Skip if already processed or out of bounds
                if not self.is_in_bounds(ny, nx) or (ny, nx) in processed:
                    continue
                    
                # Skip if already valid or being processed
                if ((self.state[ny, nx] & STATE_LOC_VALID) != 0 or 
                    (self.state[ny, nx] & STATE_PROCESSING) != 0):
                    continue
                    
                # Add to candidates and mark as processed
                candidates.append((ny, nx))
                processed.add((ny, nx))
                
                # Mark as processing
                self.update_state(ny, nx, STATE_PROCESSING)
                
        return candidates
    
    def get_used_rect(self) -> Tuple[int, int, int, int]:
        """
        Get the used rectangle (boundary) of the grid.
        
        Returns:
            (min_x, min_y, width, height) of the used area
        """
        min_x, min_y, max_x, max_y = self.boundary_rect
        return min_x, min_y, max_x - min_x, max_y - min_y
    
    def get_crop(self) -> 'PointGrid':
        """
        Get a cropped version of the grid containing only the used area.
        
        Returns:
            A new PointGrid with only the used area
        """
        min_x, min_y, width, height = self.get_used_rect()
        
        # Create new grid
        cropped = PointGrid(width, height, self.dtype)
        
        # Copy data
        cropped.points = self.points[min_y:min_y+height, min_x:min_x+width].copy()
        cropped.state = self.state[min_y:min_y+height, min_x:min_x+width].copy()
        
        # Update metadata
        cropped.boundary_rect = [0, 0, width, height]
        cropped.center = np.array([width // 2, height // 2], dtype=np.int32)
        cropped.generation = self.generation
        cropped.success_count = self.success_count
        cropped.generation_max_cost = self.generation_max_cost.copy()
        cropped.generation_avg_cost = self.generation_avg_cost.copy()
        
        return cropped
    
    def to_quad_surface(self, scale: Tuple[float, float] = (1.0, 1.0)) -> 'surfaces.QuadSurface':
        """
        Convert the grid to a QuadSurface.
        
        Args:
            scale: Scale factors for x and y dimensions
            
        Returns:
            A QuadSurface object
        """
        from surfaces import QuadSurface
        
        # Get cropped grid if needed
        min_x, min_y, width, height = self.get_used_rect()
        points = self.points[min_y:min_y+height, min_x:min_x+width].copy()
        
        # Create surface
        surface = QuadSurface(points, scale)
        
        # Add metadata
        surface.meta = {
            "generation": self.generation,
            "success_count": self.success_count,
            "generation_max_cost": self.generation_max_cost,
            "generation_avg_cost": self.generation_avg_cost,
        }
        
        return surface