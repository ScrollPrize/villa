"""
Surface optimizer for the tracing algorithm.

This module implements the SurfaceOptimizer class, which is responsible for
optimizing the surface points using the Theseus optimization framework.

Coordinate Convention:
- 3D points are in ZYX order [z, y, x] with 0=z, 1=y, 2=x
- All tensor operations maintain ZYX ordering for point coordinates
- Grid coordinates follow YX ordering [y, x] for 2D indexing
- Volume data is accessed with ZYX ordering
- Bounds checking and interpolation follow ZYX ordering
"""

from typing import List, Tuple, Dict, Optional, Union
import os
import numpy as np
import torch
import theseus as th

from tracer.grid import PointGrid, STATE_LOC_VALID, STATE_COORD_VALID, STATE_PROCESSING
from cost_functions.base.space_loss_acc import SpaceLossAcc
from cost_functions.base.dist_loss import DistLoss
from cost_functions.base.straight_loss_2 import StraightLoss2


class SqrtNormObjective(th.Objective):
    """
    An objective that applies square root normalization to the final error.
    
    This matches the C++ implementation which uses sqrt(sum/count) normalization
    as seen in SurfaceHelpers.cpp:
    - Line 334: return sqrt(best/tgts.size());
    - Line 406: return sqrt(best/tgts.size());
    - Line 819: return sqrt(summary.final_cost/summary.num_residual_blocks);
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._residual_count = 0
        # Add debug flag
        self.debug_enabled = os.environ.get('OPTIMIZER_DEBUG', '0').lower() in ('1', 'true', 'yes', 'on')
        
    def add(self, cost_function):
        """Add a cost function and track residual count."""
        super().add(cost_function)
        # Each cost function contributes its dimension to the residual count
        added_dim = cost_function.dim()
        self._residual_count += added_dim
        
        if self.debug_enabled:
            print(f"OPTIMIZER_DEBUG: Added cost function {cost_function.name} with dimension {added_dim}, "
                  f"total residuals now {self._residual_count}")
    
    def error_metric(self):
        """Calculate the error with sqrt normalization."""
        if self._residual_count == 0:
            return 0.0
        
        # Get the standard sum-of-squares error
        standard_error = super().error_metric()
        
        # Apply sqrt(error/count) normalization to match C++ implementation
        if torch.is_tensor(standard_error):
            normalized_error = torch.sqrt(standard_error / self._residual_count)
        else:
            normalized_error = np.sqrt(standard_error / self._residual_count)
        
        if self.debug_enabled:
            print(f"OPTIMIZER_DEBUG: Normalizing error from {standard_error} to {normalized_error} "
                  f"with {self._residual_count} residuals")
        
        return normalized_error

# Check if debugging is enabled via environment variables
OPTIMIZER_DEBUG_ENABLED = os.environ.get('OPTIMIZER_DEBUG', '0').lower() in ('1', 'true', 'yes', 'on')
GENERAL_DEBUG_ENABLED = os.environ.get('DEBUG', '0').lower() in ('1', 'true', 'yes', 'on')

# Utility functions for debug printing
def debug_print(message):
    """Print optimizer debug message only if optimizer debugging is enabled"""
    if OPTIMIZER_DEBUG_ENABLED:
        print(message)
        
def print_debug(message):
    """Print general debug message only if debugging is enabled"""
    if GENERAL_DEBUG_ENABLED:
        print(message)


class SurfaceOptimizer:
    """
    Optimizer for surfaces using Theseus.
    
    This class handles the optimization of surface points using the Theseus
    optimization framework. It connects the PointGrid with volume data and
    applies various cost functions to optimize point positions.
    """
    
    def __init__(self, grid: PointGrid, volume: 'Union[np.ndarray, object]', 
                 cache=None, step_size: float = 10.0):
        """
        Initialize a new surface optimizer.
        
        Args:
            grid: PointGrid containing the surface points
            volume: Volume data (zarr.Array or numpy array)
            cache: Optional chunk cache for volume data
            step_size: Step size for surface growth
        """
        self.grid = grid
        self.volume = volume
        self.step_size = step_size
        
        # Add debug attribute to the optimizer to fix 'no attribute debug' errors
        self.debug = OPTIMIZER_DEBUG_ENABLED or GENERAL_DEBUG_ENABLED
        
        # Create a unique ID that's used for all variable names
        # This ensures we avoid name conflicts between different optimizer instances
        import time
        import random
        self.current_optimization_id = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        print(f"OPTIMIZER_DEBUG: Created optimizer with session ID {self.current_optimization_id}")
        
        # Initialize interpolator for raw volume data
        from tracer.core.interpolation import TrilinearInterpolator
        debug_print("OPTIMIZER_DEBUG: Using autodiff-compatible interpolator for gradient tracking")
        self.interpolator = TrilinearInterpolator(volume)
        
        # Initialize distance transform interpolator (for path checking and evaluations)
        from tracer.core.distance_transform.interpolator import DistanceTransformInterpolator
        self.dist_interpolator = DistanceTransformInterpolator(
            volume=volume,
            threshold=170.0,  # Match C++ TH=170
            chunk_size=64,    # Match C++ CHUNK_SIZE=64
            border=16,        # Match C++ BORDER=16
            cache_size=100    # Cache size (adjust as needed)
        )
        
        # Set distance threshold for evaluation (match C++ dist_th=1.5)
        self.dist_threshold = 1.5
        
        # Dict to store optimization variables
        self.variables: Dict[Tuple[int, int], 'th.Vector'] = {}
        
        # Create a persistent variable registry to ensure proper sharing of variables
        # Key: variable name, Value: the variable object
        self.variable_registry = {}
        
        # Optimization parameters
        self.max_iterations = 50  # Default iterations for convergence
        self.step_size_opt = 0.1  # Step size for optimization
        self.cost_weights = {
            "space_loss": 1.0,  # FIXED: C++ uses weight 1.0 for space_loss from gen_space_loss
            "dist_loss": 1.0,   # Match C++ space_trace_dist_w (1.0)
            "straight_loss": 0.7,  # Match C++ StraightLoss weight (0.7)
            "anchor_loss": 0.0  # Zero out anchor loss (not used in C++ version)
        }
        
        # Enable detailed debug mode for this execution
        self.detailed_debug = True
        print(f"DEBUG OPTIMIZER INIT: Created SurfaceOptimizer with cost weights: {self.cost_weights}")
        
        # Boundary checking for volume extents
        self.volume_shape = None
        if hasattr(volume, 'shape'):
            self.volume_shape = volume.shape
    
    def sample_volume_at_point(self, y: int, x: int) -> float:
        """
        Sample the volume at the given grid point.
        
        Args:
            y: Y coordinate in the grid
            x: X coordinate in the grid
            
        Returns:
            The interpolated value at the point (0.0 if point is invalid)
        """
        # Get the point position (even if not marked as valid yet)
        # This allows us to sample candidate points during growth
        if y < 0 or x < 0 or y >= self.grid.height or x >= self.grid.width:
            print_debug(f"DEBUG: sample_volume_at_point({y},{x}): Out of bounds")
            return 0.0  # Out of bounds
            
        point = self.grid.get_point(y, x)
        if point[0] < 0:  # Uninitialized point
            print_debug(f"DEBUG: sample_volume_at_point({y},{x}): Uninitialized point {point}")
            return 0.0
            
        # Use tensor inputs with batch dimension for Theseus compatibility
        try:
            # Create input tensors with batch size 1
            z = torch.tensor([[float(point[0])]], dtype=torch.float32)
            y_val = torch.tensor([[float(point[1])]], dtype=torch.float32)
            x_val = torch.tensor([[float(point[2])]], dtype=torch.float32)
            
            # Call evaluate with proper tensor inputs
            value = self.interpolator.evaluate(z, y_val, x_val)
            print_debug(f"DEBUG: sample_volume_at_point({y},{x}) at position {point}: value = {value.item()}")
            return float(value.item())  # Ensure we return a regular float
        except Exception as e:
            print(f"ERROR: Error sampling at point({y},{x}) {point}: {e}")
            return 0.0
    
    def sample_volume_with_gradient(self, y: int, x: int) -> Tuple[float, np.ndarray]:
        """
        Sample the volume and gradient at the given grid point.
        
        Args:
            y: Y coordinate in the grid
            x: X coordinate in the grid
            
        Returns:
            Tuple of (value, gradient) at the point (0.0, zeros if point is invalid)
        """
        # Get the point position (even if not marked as valid yet)
        if y < 0 or x < 0 or y >= self.grid.height or x >= self.grid.width:
            return 0.0, np.zeros(3, dtype=np.float32)  # Out of bounds
            
        point = self.grid.get_point(y, x)
        if point[0] < 0:  # Uninitialized point
            return 0.0, np.zeros(3, dtype=np.float32)
            
        # Use tensor inputs with batch dimension for Theseus compatibility
        try:
            # Create input tensors with batch size 1
            z = torch.tensor([[float(point[0])]], dtype=torch.float32)
            y_val = torch.tensor([[float(point[1])]], dtype=torch.float32)
            x_val = torch.tensor([[float(point[2])]], dtype=torch.float32)
            
            try:
                # Call evaluate_with_gradient with proper tensor inputs
                value, gradient = self.interpolator.evaluate_with_gradient(z, y_val, x_val)
                
                # Convert the tensors to numpy arrays and extract the values
                # FIXED: Ensure proper detachment before converting to numpy
                value_np = float(value.detach().item())  # Detach before item() to avoid gradient issues
                
                # Safely extract gradient - handle different gradient tensor shapes
                try:
                    # Default case: Gradient has shape [batch, N, 3]
                    if isinstance(gradient, torch.Tensor) and len(gradient.shape) >= 3 and gradient.shape[-1] == 3:
                        # Extract the first (and only) gradient in the batch
                        gradient_np = gradient[0, 0].detach().cpu().numpy()
                    # Alternative case: Gradient has shape [batch, 3]
                    elif isinstance(gradient, torch.Tensor) and len(gradient.shape) == 2 and gradient.shape[-1] == 3:
                        gradient_np = gradient[0].detach().cpu().numpy()
                    # Handle case where gradient is a scalar or has no shape
                    elif not isinstance(gradient, torch.Tensor) or not hasattr(gradient, 'shape'):
                        print(f"Warning: Gradient is not a tensor or has no shape attribute: type={type(gradient)}")
                        gradient_np = np.zeros(3, dtype=np.float32)
                    # Fallback: Reshape or create a default gradient
                    else:
                        print(f"Warning: Unexpected gradient shape {gradient.shape}, creating default gradient")
                        gradient_np = np.zeros(3, dtype=np.float32)
                except (IndexError, AttributeError, TypeError) as e:
                    print(f"Warning: Could not process gradient properly: {e}")
                    gradient_np = np.zeros(3, dtype=np.float32)
                
                # Extra safety check to ensure gradient is always a numpy array with shape (3,)
                if not isinstance(gradient_np, np.ndarray):
                    print(f"Warning: gradient_np is not an ndarray, creating zeros array")
                    gradient_np = np.zeros(3, dtype=np.float32)
                elif gradient_np.shape != (3,):
                    print(f"Warning: Correcting gradient shape from {gradient_np.shape} to (3,)")
                    # Try to reshape if possible, otherwise create zeros
                    try:
                        if gradient_np.size >= 3:
                            gradient_np = gradient_np.flatten()[:3]
                        else:
                            gradient_np = np.zeros(3, dtype=np.float32)
                    except:
                        gradient_np = np.zeros(3, dtype=np.float32)
                
                # Final safety check to ensure dtype is float32
                gradient_np = gradient_np.astype(np.float32)
                
            except Exception as e:
                print(f"Error in evaluate_with_gradient: {e}")
                # Fall back to separate evaluation and numerical gradient
                try:
                    # Evaluate value directly
                    value = self.interpolator.evaluate(z, y_val, x_val)
                    value_np = float(value.detach().item())
                    
                    # Create a default zero gradient
                    gradient_np = np.zeros(3, dtype=np.float32)
                    
                    # Attempt to compute numerical gradient if possible
                    epsilon = 0.1
                    try:
                        # Z direction
                        z_plus = torch.tensor([[float(point[0] + epsilon)]], dtype=torch.float32)
                        z_minus = torch.tensor([[float(point[0] - epsilon)]], dtype=torch.float32)
                        val_z_plus = self.interpolator.evaluate(z_plus, y_val, x_val).item()
                        val_z_minus = self.interpolator.evaluate(z_minus, y_val, x_val).item()
                        gradient_np[0] = (val_z_plus - val_z_minus) / (2 * epsilon)
                        
                        # Y direction
                        y_plus = torch.tensor([[float(point[1] + epsilon)]], dtype=torch.float32)
                        y_minus = torch.tensor([[float(point[1] - epsilon)]], dtype=torch.float32)
                        val_y_plus = self.interpolator.evaluate(z, y_plus, x_val).item()
                        val_y_minus = self.interpolator.evaluate(z, y_minus, x_val).item()
                        gradient_np[1] = (val_y_plus - val_y_minus) / (2 * epsilon)
                        
                        # X direction
                        x_plus = torch.tensor([[float(point[2] + epsilon)]], dtype=torch.float32)
                        x_minus = torch.tensor([[float(point[2] - epsilon)]], dtype=torch.float32)
                        val_x_plus = self.interpolator.evaluate(z, y_val, x_plus).item()
                        val_x_minus = self.interpolator.evaluate(z, y_val, x_minus).item()
                        gradient_np[2] = (val_x_plus - val_x_minus) / (2 * epsilon)
                    except Exception as e2:
                        print(f"Failed to compute numerical gradient: {e2}")
                        # Keep zero gradient
                except Exception as e2:
                    print(f"Failed to fall back to basic evaluation: {e2}")
                    value_np = 0.0
                    gradient_np = np.zeros(3, dtype=np.float32)
                
            return value_np, gradient_np  # Ensure consistent float32 dtype
        except Exception as e:
            print(f"Error sampling with gradient at {point}: {e}")
            return 0.0, np.zeros(3, dtype=np.float32)
    
    def initialize_optimization_variables(self, candidate_points: List[Tuple[int, int]]):
        """
        Initialize Theseus optimization variables for the given candidate points.
        
        This creates Theseus Vector variables for each point to be optimized.
        
        Args:
            candidate_points: List of (y, x) coordinates for candidate points
        """
        # Use the persistent variable registry from the class
        # to ensure variables are shared across multiple optimization calls
        
        # Helper function to get or create point variables - simplified for reliability
        def get_point_variable(y, x):
            # Ensure y and x are converted to base Python types for string formatting
            y_val = int(y)
            x_val = int(x)
            
            # SIMPLIFIED APPROACH: Always use this naming format with unique session ID
            # This guarantees no name conflicts between different optimization runs
            unique_suffix = self.current_optimization_id
            
            # Create a simple and consistent naming scheme that is guaranteed to be unique
            simple_name = f"p{y_val}_{x_val}_{unique_suffix}"
            
            print(f"DEBUG_VAR_INIT: Creating variable for point ({y_val},{x_val}) with name '{simple_name}'")
            
            # Check if variable already exists in registry for this exact session
            if simple_name in self.variable_registry:
                print(f"DEBUG_VAR_INIT: Variable '{simple_name}' found in this session's registry")
                return self.variable_registry[simple_name]
                
            # Variable doesn't exist, need to create it
            # Get the point position
            point = self.grid.get_point(y_val, x_val).copy()
            
            if not self.grid.is_valid(y_val, x_val):
                # For new points, initialize from neighbors
                valid_neighbors = []
                for dy, dx in self.grid.neighbors:
                    ny, nx = y_val + dy, x_val + dx
                    if self.grid.is_valid(ny, nx):
                        valid_neighbors.append(self.grid.get_point(ny, nx))
                
                if valid_neighbors:
                    # Average of valid neighbors
                    point = np.mean(valid_neighbors, axis=0)
            
            # Ensure correct shape
            flat_point = point.flatten()[:3]  # Take first 3 elements in ZYX order
            
            # Log shape information
            debug_print(f"OPTIMIZER_DEBUG: Creating variable {simple_name} from point {flat_point} (ZYX order)")
            
            # CRITICAL FIX: Convert from our ZYX ordering to Theseus XYZ ordering
            # Since Theseus Point3 expects XYZ order, we need to swap coordinates
            theseus_point = np.zeros(3, dtype=np.float32)
            theseus_point[0] = flat_point[2]  # x = z
            theseus_point[1] = flat_point[1]  # y = y
            theseus_point[2] = flat_point[0]  # z = x
            
            debug_print(f"OPTIMIZER_DEBUG: Converted to Theseus format {theseus_point} (XYZ order)")
            
            # Convert to tensor properly, ensuring correct shape (batch_size, 3)
            # IMPORTANT: Set requires_grad=True to enable gradient tracking
            # Use tensor() with requires_grad=True, as this is more reliable than requires_grad_()
            point_tensor = torch.tensor(theseus_point.reshape(1, 3), dtype=torch.float32, requires_grad=True)
            
            # Double-check that requires_grad is actually True
            if not point_tensor.requires_grad:
                print(f"WARNING: point_tensor.requires_grad is False even though we set it to True")
                # Try an alternative approach to ensure gradient tracking
                point_tensor = torch.tensor(theseus_point.reshape(1, 3), dtype=torch.float32).requires_grad_(True)
            
            print_debug(f"DEBUG: Tensor shape after creation: {point_tensor.shape}, requires_grad={point_tensor.requires_grad}")
            
            try:
                # Create a new variable with the simple, unique name
                var = th.Point3(tensor=point_tensor, name=simple_name)
                print(f"DEBUG_VAR_INIT: Successfully created variable with name '{simple_name}'")
                
                # Register with unique name
                self.variable_registry[simple_name] = var
                self.variables[(y_val, x_val)] = var
                return var
                
            except Exception as e:
                print(f"ERROR: Failed to create variable with name '{simple_name}': {e}")
                try:
                    # Try with a truly basic fallback name that's still unique to this session
                    fallback_name = f"var_{len(self.variable_registry)}_{unique_suffix}"
                    print(f"DEBUG_VAR_INIT: Trying fallback name '{fallback_name}'")
                    var = th.Point3(tensor=point_tensor, name=fallback_name)
                    
                    # Register with fallback name
                    self.variable_registry[fallback_name] = var 
                    self.variables[(y_val, x_val)] = var
                    return var
                except Exception as final_e:
                    print(f"FATAL ERROR: Could not create variable under any name: {final_e}")
                    raise
        
        # Determine which variables we need
        need_vars = set()
        
        # Add the candidate points
        for y, x in candidate_points:
            need_vars.add((y, x))
            
            # Also add valid neighbors
            for dy, dx in self.grid.neighbors:
                ny, nx = y + dy, x + dx
                if self.grid.is_valid(ny, nx):
                    need_vars.add((ny, nx))
        
        # Create variables for each needed point using our helper function
        for y, x in need_vars:
            get_point_variable(y, x)
    
    def create_objective(self, candidate_points: List[Tuple[int, int]]):
        """
        Create a Theseus objective for optimizing the given candidate points.
        
        This sets up the cost functions for the optimization problem.
        
        Args:
            candidate_points: List of (y, x) coordinates for candidate points
            
        Returns:
            A Theseus Objective object
        """
        import theseus as th
        from cost_functions.base.anchor_loss import AnchorLoss
        
        # Convert all candidate points to ensure they're proper integer tuples
        safe_candidates = []
        for y, x in candidate_points:
            y_val = int(y)
            x_val = int(x)
            safe_candidates = safe_candidates + [(y_val, x_val)]
        candidate_points = safe_candidates  # Use the safe candidates instead
        
        # Create a custom objective that uses sqrt normalization to match C++
        if hasattr(self, 'detailed_debug') and self.detailed_debug:
            print(f"DEBUG CREATE OBJECTIVE: Creating SqrtNormObjective for {len(candidate_points)} candidate points: {candidate_points}")
        objective = SqrtNormObjective()
        
        # FUNDAMENTAL FIX: Create a completely fresh variable registry and variable dict for each objective
        # This is needed to avoid name conflicts - since we're creating one objective per call here
        # Each candidate point goes in its own objective with no shared variables
        self.variable_registry = {}
        self.variables = {}
        variable_registry = self.variable_registry
        
        # Helper function to get or create point variables
        def get_point_variable(y, x):
            # Ensure y and x are converted to base Python types for string formatting
            y_val = int(y)
            x_val = int(x)
            
            # CRITICAL FIX: Generate a completely unique name for EVERY variable instance
            # Include the timestamp ID plus a random number to guarantee uniqueness
            import random
            random_suffix = random.randint(10000, 99999)
            
            # Create a variable name that is guaranteed to be unique
            # This is essential to avoid any possible name conflicts
            var_name = f"p{y_val}_{x_val}_{self.current_optimization_id}_{random_suffix}"
            
            print(f"DEBUG_VAR_CREATE: Creating new variable for point ({y_val},{x_val})")
            print(f"DEBUG_VAR_CREATE: Unique variable name: '{var_name}'")
            
            # CRITICAL FIX: DO NOT check if variable exists - always create a new one
            # This ensures each cost function gets its own unique variable instance
                
            # Variable doesn't exist, need to create it
            # Get the point position
            point = self.grid.get_point(y_val, x_val).copy()
            print_debug(f"DEBUG: Initial point ({y_val},{x_val}): {point}")
            
            # Reshape to ensure we have a flat 1D array of 3 elements
            flat_point = point.flatten()[:3]  # Take first 3 elements in case of weird shapes
            
            # Ensure points are not negative and within volume bounds
            if self.volume_shape is not None:
                print_debug(f"DEBUG: Volume shape: {self.volume_shape}")
                # Enforce minimum coordinate of 0.1 (slightly above 0)
                original = flat_point.copy()
                flat_point[0] = max(0.1, flat_point[0])
                flat_point[1] = max(0.1, flat_point[1])
                flat_point[2] = max(0.1, flat_point[2])
                
                # Ensure points are within volume bounds with margin
                if flat_point[0] >= self.volume_shape[0] - 1:
                    flat_point[0] = self.volume_shape[0] - 1.1
                if flat_point[1] >= self.volume_shape[1] - 1:
                    flat_point[1] = self.volume_shape[1] - 1.1
                if flat_point[2] >= self.volume_shape[2] - 1:
                    flat_point[2] = self.volume_shape[2] - 1.1
                
                if not np.array_equal(original, flat_point):
                    print_debug(f"DEBUG: Point adjusted from {original} to {flat_point}")
            
            # CRITICAL FIX: Convert from our ZYX ordering to Theseus XYZ ordering
            # For Point3 in Theseus, we need XYZ order
            theseus_point = np.zeros(3, dtype=np.float32)
            theseus_point[0] = flat_point[2]  # x = z
            theseus_point[1] = flat_point[1]  # y = y
            theseus_point[2] = flat_point[0]  # z = x
            
            debug_print(f"OPTIMIZER_DEBUG: Converted point {flat_point} (ZYX) to {theseus_point} (XYZ)")
            
            # Convert to tensor properly, ensuring correct shape (batch_size, 3)
            # Make sure we're creating a new tensor with shape [1, 3]
            # IMPORTANT: Set requires_grad=True to enable gradient tracking
            point_tensor = torch.tensor(theseus_point.reshape(1, 3), dtype=torch.float32, requires_grad=True)
            
            # Double-check that requires_grad is actually True
            if not point_tensor.requires_grad:
                print(f"WARNING: point_tensor.requires_grad is False even though we set it to True")
                # Try an alternative approach to ensure gradient tracking
                point_tensor = torch.tensor(flat_point.reshape(1, 3), dtype=torch.float32).requires_grad_(True)
                
                # If still not working, try another approach
                if not point_tensor.requires_grad:
                    print(f"WARNING: Second attempt failed, trying with clone()")
                    raw_tensor = torch.tensor(flat_point.reshape(1, 3), dtype=torch.float32)
                    point_tensor = raw_tensor.clone().detach().requires_grad_(True)
            
            print_debug(f"DEBUG: Created tensor with shape {point_tensor.shape}, requires_grad={point_tensor.requires_grad}")
            
            # Ensure point tensor has the correct shape [batch_size, 3]
            if len(point_tensor.shape) > 2:
                # Reshape tensor if it has extra dimensions
                batch_size = point_tensor.shape[0]
                point_tensor = point_tensor.reshape(batch_size, 3)
            
            # Create a single variable with a guaranteed unique name
            try:
                # Create the variable with our unique name
                var = th.Point3(tensor=point_tensor, name=var_name)
                
                # Register the variable in our registries
                variable_registry[var_name] = var
                self.variables[(y_val, x_val)] = var
                print(f"DEBUG_VAR_CREATE: Variable created and registered with name '{var_name}'")
                return var
                
            except Exception as e:
                print(f"ERROR: Failed to create variable with name '{var_name}': {e}")
                try:
                    # If there's an error, try with an even simpler fallback name
                    import random
                    fallback_name = f"var_{len(variable_registry)}_{random.randint(10000, 99999)}"
                    print(f"DEBUG_VAR_CREATE: Trying fallback name '{fallback_name}'")
                    var = th.Point3(tensor=point_tensor, name=fallback_name)
                    
                    # Register the variable
                    variable_registry[fallback_name] = var 
                    self.variables[(y_val, x_val)] = var
                    return var
                except Exception as final_e:
                    print(f"FATAL ERROR: Could not create variable: {final_e}")
                    raise
        
        # Create shared cost weights
        space_weight = th.ScaleCostWeight(self.cost_weights["space_loss"])
        dist_weight = th.ScaleCostWeight(self.cost_weights["dist_loss"])
        straight_weight = th.ScaleCostWeight(self.cost_weights["straight_loss"])
        anchor_weight = th.ScaleCostWeight(self.cost_weights["anchor_loss"])
        
        if hasattr(self, 'detailed_debug') and self.detailed_debug:
            print(f"DEBUG COST WEIGHTS: space={self.cost_weights['space_loss']}, "
                  f"dist={self.cost_weights['dist_loss']}, "
                  f"straight={self.cost_weights['straight_loss']}, "
                  f"anchor={self.cost_weights['anchor_loss']}")
        
        # Create a set of all variables we need
        all_vars = set()
        all_vars.update(candidate_points)
        
        # Add neighbors of candidate points
        for y, x in candidate_points:
            for dy, dx in self.grid.neighbors:
                ny, nx = y + dy, x + dx
                if self.grid.is_valid(ny, nx):
                    all_vars.add((ny, nx))
        
        # Pre-initialize all variables we'll need
        for y, x in all_vars:
            get_point_variable(y, x)
        
        # Step 1: First add SpaceLossAcc for each candidate point
        for y, x in candidate_points:
            var = get_point_variable(y, x)
            if self.cost_weights["space_loss"] > 0:
                # CRITICAL FIX: Check the variable to ensure it's a proper Theseus variable
                if not hasattr(var, 'tensor'):
                    print(f"ERROR: Variable has no tensor attribute! Type: {type(var)}")
                    if isinstance(var, str):
                        print(f"ERROR: Variable is a string '{var}' not a Theseus variable!")
                        raise ValueError(f"Cannot create cost function with string variable: {var}")
                    else:
                        print(f"ERROR: Unknown variable type: {type(var)}")
                        raise ValueError(f"Cannot create cost function with invalid variable type: {type(var)}")
                
                # Generate a unique name for the cost function
                import random
                unique_cf_name = f"space_loss_{y}_{x}_{random.randint(10000, 99999)}"
                
                try:
                    # Add space loss for this point
                    cf = SpaceLossAcc(
                        point=var, 
                        interpolator=self.interpolator,
                        cost_weight=space_weight,
                        maximize=True,
                        name=unique_cf_name
                    )
                    objective.add(cf)
                    print(f"DEBUG: Successfully added SpaceLossAcc with unique name: {unique_cf_name}")
                except Exception as e:
                    print(f"ERROR: Failed to create SpaceLossAcc: {e}")
                    print(f"ERROR: Variable type: {type(var)}")
                    if hasattr(var, 'tensor'):
                        print(f"ERROR: Tensor shape: {var.tensor.shape}")
                    raise
                
                # Add AnchorLoss to enhance stability
                if self.cost_weights["anchor_loss"] > 0:
                    # CRITICAL FIX: Make sure var is a valid Theseus variable
                    if not hasattr(var, 'tensor'):
                        print(f"ERROR: Variable has no tensor attribute! Type: {type(var)}")
                        continue  # Skip anchor loss if the variable isn't valid
                    
                    # Create an anchor point variable for this point with unique name
                    # Generate a random suffix to guarantee uniqueness
                    import random
                    y_val = int(y)
                    x_val = int(x)
                    random_suffix = random.randint(10000, 99999)
                    unique_anchor_name = f"anchor{y_val}_{x_val}_{self.current_optimization_id}_{random_suffix}"
                    
                    # Create a copy of the point tensor for the anchor
                    # Ensure the tensor has the shape [batch_size, 3]
                    # IMPORTANT: We need to ensure the anchor tensor also has requires_grad=True
                    anchor_tensor = var.tensor.clone().detach().requires_grad_(True)
                    print(f"DEBUG: Creating anchor variable '{unique_anchor_name}' from tensor with shape {anchor_tensor.shape}, requires_grad={anchor_tensor.requires_grad}")
                    
                    # Ensure anchor tensor has shape [batch_size, 3]
                    if len(anchor_tensor.shape) > 2:
                        # Log original shape
                        print(f"DEBUG: Anchor tensor has too many dimensions: {anchor_tensor.shape}")
                        
                        # Reshape tensor if it has extra dimensions
                        batch_size = anchor_tensor.shape[0]
                        anchor_tensor = anchor_tensor.reshape(batch_size, 3)
                        print(f"DEBUG: Reshaped anchor tensor to {anchor_tensor.shape}")
                            
                        # Create a Theseus variable from the anchor tensor
                        print(f"DEBUG: Creating anchor variable with unique name: '{unique_anchor_name}'")
                        anchor_var = th.Point3(tensor=anchor_tensor, name=unique_anchor_name)
                        
                        # Store in registry with the unique name
                        variable_registry[unique_anchor_name] = anchor_var
                    # Create the anchor variable if not done above 
                    if not 'anchor_var' in locals():
                        # Create a Theseus variable from the anchor tensor
                        print(f"DEBUG: Creating anchor variable with unique name: '{unique_anchor_name}'")
                        anchor_var = th.Point3(tensor=anchor_tensor, name=unique_anchor_name)
                        
                        # Store in registry with the unique name
                        variable_registry[unique_anchor_name] = anchor_var
                    
                    # Add more debug for creating AnchorLoss
                    print(f"DEBUG: Adding AnchorLoss for point ({y},{x})")
                    print_debug(f"DEBUG: Point tensor shape before AnchorLoss: {var.tensor.shape}")
                    print_debug(f"DEBUG: Anchor tensor shape before AnchorLoss: {anchor_var.tensor.shape}")
                    
                    # Validate tensor shapes for AnchorLoss - they must be [batch_size, 3]
                    if len(var.tensor.shape) > 2:
                        print_debug(f"DEBUG: Reshaping point tensor from {var.tensor.shape} to [batch_size, 3]")
                        if var.tensor.shape[-1] == 3:
                            var.tensor = var.tensor.reshape(-1, 3)
                        else:
                            raise ValueError(f"Cannot reshape point tensor with shape {var.tensor.shape} to [batch_size, 3]")
                    
                    if len(anchor_var.tensor.shape) > 2:
                        print_debug(f"DEBUG: Reshaping anchor tensor from {anchor_var.tensor.shape} to [batch_size, 3]")
                        if anchor_var.tensor.shape[-1] == 3:
                            anchor_var.tensor = anchor_var.tensor.reshape(-1, 3)
                        else:
                            raise ValueError(f"Cannot reshape anchor tensor with shape {anchor_var.tensor.shape} to [batch_size, 3]")
                    
                    # Create a unique name for the AnchorLoss cost function
                    import random
                    anchor_cf_name = f"anchor_loss_{y}_{x}_{random.randint(10000, 99999)}"
                    
                    # Create and add the anchor loss with detailed error checking
                    try:
                        # Make sure both variables have the .tensor attribute
                        if not hasattr(var, 'tensor'):
                            print(f"ERROR: Point variable has no tensor attribute!")
                            continue
                            
                        if not hasattr(anchor_var, 'tensor'):
                            print(f"ERROR: Anchor variable has no tensor attribute!")
                            continue

                        # Create the anchor loss with our unique name
                        anchor_cf = AnchorLoss(
                            point=var,
                            anchor_point=anchor_var,
                            interpolator=self.interpolator,
                            cost_weight=anchor_weight,
                            name=anchor_cf_name
                        )
                        objective.add(anchor_cf)
                        print(f"DEBUG: Successfully added AnchorLoss with name '{anchor_cf_name}' for point ({y},{x})")
                    except Exception as e:
                        print(f"ERROR: Failed adding AnchorLoss for point ({y},{x}): {e}")
                        print(f"ERROR: Point type: {type(var)}, Anchor type: {type(anchor_var)}")
                        if hasattr(var, 'tensor'):
                            print(f"ERROR: Point tensor shape: {var.tensor.shape}")
                        if hasattr(anchor_var, 'tensor'):
                            print(f"ERROR: Anchor tensor shape: {anchor_var.tensor.shape}")
                        # Continue without re-raising to allow other cost functions to be added
                        continue
        
        # Step 2: Add distance constraints between points - this helps maintain proper spacing
        if self.cost_weights["dist_loss"] > 0:
            # Keep track of which pairs we've added
            added_pairs = set()
            
            for y, x in all_vars:
                for dy, dx in self.grid.neighbors:
                    ny, nx = y + dy, x + dx
                    # Skip if neighbor isn't valid or we've already added this pair
                    if not self.grid.is_valid(ny, nx) or ((y, x), (ny, nx)) in added_pairs or ((ny, nx), (y, x)) in added_pairs:
                        continue
                    
                    # Get variables from registry to ensure proper sharing
                    point_a = get_point_variable(y, x)
                    point_b = get_point_variable(ny, nx)
                    
                    # Add distance constraint
                    dist_cf = DistLoss(
                        point_a=point_a,
                        point_b=point_b,
                        target_dist=float(self.step_size),
                        cost_weight=dist_weight,
                        name=f"dist_loss_{y}_{x}_{ny}_{nx}"
                    )
                    objective.add(dist_cf)
                    
                    # Mark this pair as added
                    added_pairs.add(((y, x), (ny, nx)))
        
        # Step 3: Add straightness constraints for points with at least 2 neighbors - improves smoothness
        if self.cost_weights["straight_loss"] > 0:
            candidate_set = set(candidate_points)
            
            # Add StraightLoss for each point with 2+ neighbors
            for y, x in all_vars:  # Apply to all vars, not just candidates, for better smoothness
                # Find neighbors
                neighbors = []
                for dy, dx in self.grid.neighbors:
                    ny, nx = y + dy, x + dx
                    if self.grid.is_valid(ny, nx):
                        neighbors.append((ny, nx))
                
                # Only add constraints for points with at least 2 neighbors
                if len(neighbors) >= 2:
                    for i in range(len(neighbors) - 1):
                        for j in range(i + 1, len(neighbors)):
                            n1y, n1x = neighbors[i]
                            n2y, n2x = neighbors[j]
                            
                            # Get variables from registry to ensure proper sharing
                            point_a = get_point_variable(n1y, n1x)
                            point_b = get_point_variable(y, x)
                            point_c = get_point_variable(n2y, n2x)
                            
                            # Create straightness constraint matching the expected parameter names
                            straight_cf = StraightLoss2(
                                point_a=point_a,
                                point_b=point_b,
                                point_c=point_c,
                                cost_weight=straight_weight,
                                name=f"straight_loss_{n1y}_{n1x}_{y}_{x}_{n2y}_{n2x}"
                            )
                            objective.add(straight_cf)
        
        # Return the completed objective
        return objective
    
    def optimize_points(self, candidate_points: List[Tuple[int, int]], max_iterations: Optional[int] = None):
        """
        Optimize the positions of the candidate points.
        
        Args:
            candidate_points: List of (y, x) coordinates for candidate points
            max_iterations: Maximum number of iterations (default: self.max_iterations)
        """
        # Store candidate_points in a class variable so it's available in exception handlers
        self.current_candidates = [tuple(map(int, point)) for point in candidate_points]
        
        # Convert all candidate points to ensure they're proper integer tuples
        # This helps avoid subtle type issues with formatting in variable names
        safe_candidates = []
        for y, x in candidate_points:
            y_val = int(y)
            x_val = int(x)
            safe_candidates.append((y_val, x_val))
        
        debug_print(f"OPTIMIZER_DEBUG: Starting optimize_points with {len(safe_candidates)} candidate points: {safe_candidates}")
        import logging
        logger = logging.getLogger(__name__)
        
        if max_iterations is None:
            max_iterations = self.max_iterations
        
        # Pre-optimize candidate points using volume gradients for better initialization
        self._pre_optimize_using_gradient(safe_candidates)
        
        # CRITICAL FIX: Always completely clear the variable registry for each optimization run
        # This is essential to avoid "Two different variable objects with the same name" errors
        self.variable_registry = {}
        self.variables = {}
        
        # Generate a fresh unique session ID for EVERY optimization run
        # This ensures all variables created in this specific run have globally unique names
        import time
        import random
        self.current_optimization_id = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        print(f"OPTIMIZER_DEBUG: Generated new session ID {self.current_optimization_id} for optimization run")
        
        debug_print("OPTIMIZER_DEBUG: Cleared variable registry to prevent name conflicts")
        
        # Perform a pre-check on variable naming to catch issues early
        for y, x in safe_candidates:
            var_name = f"p_{y}_{x}"
            safe_name = str(var_name)
            print(f"OPTIMIZER_DEBUG: Will create variable '{safe_name}' for point ({y},{x})")
        
        try:
            # Create objective - this initializes variables internally
            debug_print(f"OPTIMIZER_DEBUG: Creating objective for candidates: {safe_candidates}")
            # Use the safe_candidates with proper integer types instead of the original candidates
            objective = self.create_objective(safe_candidates)
            debug_print(f"OPTIMIZER_DEBUG: Objective created with {len(objective.optim_vars)} optimization variables")
            
            # Create optimizer with CholmodSparseSolver instead of default DenseCholesky
            debug_print("OPTIMIZER_DEBUG: Creating LevenbergMarquardt optimizer")
            optimizer = th.LevenbergMarquardt(
                objective=objective,
                linear_solver_cls=th.CholmodSparseSolver,  # Use Cholmod sparse solver
                max_iterations=200,  # Match C++ inner optimization iterations (line 1082 in SurfaceHelpers.cpp)
                step_size=1.0,  # Standard step size
                damping=1.0,  # Reduced damping to match C++ implementation approach
                function_tolerance=1e-3,  # Match C++ options.function_tolerance (line 1083)
                verbose=True  # Enable verbose output to see optimization progress
            )
            
            # Create Theseus layer - as recommended in examples
            layer = th.TheseusLayer(optimizer)
            
            # Ensure we only use variables that are in the objective
            valid_vars = {}
            for (y, x), var in self.variables.items():
                if var.name in objective.optim_vars:
                    valid_vars[(y, x)] = var
            
            # Replace our variables dictionary with only valid variables
            self.variables = valid_vars
            
            # Prepare input dictionary for optimization using variables from the objective
            inputs = {}
            debug_print("OPTIMIZER_DEBUG: Creating inputs dictionary")
            var_count = 0
            for (y, x), var in self.variables.items():
                # Add the variable tensor to the inputs dictionary
                inputs[var.name] = var.tensor
                var_count += 1
                debug_print(f"OPTIMIZER_DEBUG: Added variable '{var.name}' with shape {var.tensor.shape}")
            debug_print(f"OPTIMIZER_DEBUG: Created inputs dictionary with {var_count} variables")
            
            # Run optimization with error handling
            success = True
            # Initialize final_values and info to avoid reference before assignment errors
            final_values = {}
            info = None
            try:
                # Print all input tensor shapes for debugging
                debug_print(f"OPTIMIZER_DEBUG: Running optimization for {len(candidate_points)} candidates")
                debug_print(f"OPTIMIZER_DEBUG: Input dictionary contains {len(inputs)} variables")
                for name, tensor in inputs.items():
                    debug_print(f"OPTIMIZER_DEBUG: Input tensor '{name}' has shape {tensor.shape}")
                
                # According to the Theseus guide, we need to make sure all tensors have the same batch dimension (first dimension)
                # However, we can use batch_size > 1 for optimization if all tensors have the same batch size
                
                # Import the batch utilities
                from tracer.batch_utils import validate_batch_consistency, normalize_batch_sizes
                
                debug_print("OPTIMIZER_DEBUG: Normalizing all tensors to ensure consistent batch dimensions")
                
                # Validate batch consistency across all input tensors
                is_consistent, error_msg, common_batch_size = validate_batch_consistency(inputs)
                
                if not is_consistent:
                    debug_print(f"OPTIMIZER_DEBUG: {error_msg}")
                    debug_print("OPTIMIZER_DEBUG: Normalizing tensors to use a common batch size")
                    inputs = normalize_batch_sizes(inputs)
                    
                    # Verify normalization succeeded
                    is_consistent, error_msg, common_batch_size = validate_batch_consistency(inputs)
                    if not is_consistent:
                        raise ValueError(f"Failed to normalize batch dimensions: {error_msg}")
                
                debug_print(f"OPTIMIZER_DEBUG: All tensors now have consistent batch_size={common_batch_size}")
                
                # Add batch dimensions to tensors that don't have one
                num_fixed = 0
                for name, tensor in list(inputs.items()):  # Use list() to avoid dict changing during iteration
                    if not isinstance(tensor, torch.Tensor):
                        continue
                        
                    if len(tensor.shape) < 1:  # Scalar tensor
                        inputs[name] = tensor.unsqueeze(0).expand(common_batch_size)
                        num_fixed += 1
                    elif len(tensor.shape) == 1:  # 1D tensor
                        inputs[name] = tensor.unsqueeze(0).expand(common_batch_size, -1)
                        num_fixed += 1
                
                if num_fixed > 0:
                    debug_print(f"OPTIMIZER_DEBUG: Added batch dimension for {num_fixed} scalar/1D tensors")
                
                # Now check if we have the right number of dimensions - Theseus expects [batch_size, dim]
                
                # Ensure all input tensors for 3D points have shape [batch_size, 3]
                # Use batch_utils to reshape tensors correctly while preserving batch dimension
                from tracer.batch_utils import reshape_to_batch_dim_3d
                
                # First identify which tensors are 3D points (Vector3/Point3 variables)
                point_tensors = {}
                for name, tensor in list(inputs.items()):  # Use list to avoid dict mutation issues during iteration
                    if "p_" in name or "anchor_" in name:  # Naming pattern for 3D point variables
                        point_tensors[name] = tensor
                
                # Fix shape of point tensors only
                for name, tensor in point_tensors.items():
                    if len(tensor.shape) != 2 or tensor.shape[1] != 3:
                        # Found a tensor with wrong dimensions for a 3D point
                        original_shape = tensor.shape
                        debug_print(f"OPTIMIZER_DEBUG: Need to reshape point tensor '{name}' from {original_shape}")
                        
                        try:
                            # Reshape while preserving batch dimension
                            reshaped = reshape_to_batch_dim_3d(tensor, common_batch_size)
                            inputs[name] = reshaped
                            debug_print(f"OPTIMIZER_DEBUG: Reshaped tensor '{name}' from {original_shape} to {reshaped.shape}")
                        except ValueError as e:
                            error_msg = f"Failed to reshape point tensor '{name}': {e}"
                            debug_print(f"OPTIMIZER_DEBUG: {error_msg}")
                            raise ValueError(error_msg) from e
                
                # Verify all point tensors have correct shape [batch_size, 3]
                for name, tensor in point_tensors.items():
                    if len(tensor.shape) != 2 or tensor.shape[1] != 3:
                        error_msg = f"Point tensor '{name}' has invalid shape {tensor.shape} after reshaping. Expected [batch_size, 3]."
                        debug_print(f"OPTIMIZER_DEBUG: {error_msg}")
                        raise ValueError(error_msg)
                        
                debug_print(f"OPTIMIZER_DEBUG: All point tensors now have correct shape [batch_size, 3] with batch_size={common_batch_size}")
                    
                debug_print("OPTIMIZER_DEBUG: Starting optimization with TheseusLayer.forward()")
                try:
                    # IMPORTANT: Enable gradient tracking for tensor inputs
                    # First ensure inputs are all properly set to requires_grad=True
                    for name, tensor in inputs.items():
                        if isinstance(tensor, torch.Tensor) and not tensor.requires_grad:
                            inputs[name] = tensor.detach().clone().requires_grad_(True)
                            debug_print(f"OPTIMIZER_DEBUG: Enabled requires_grad for tensor {name}, now: {inputs[name].requires_grad}")
                    
                    # Verify all point variables have gradient tracking enabled
                    for name, tensor in inputs.items():
                        if isinstance(tensor, torch.Tensor) and "p_" in name:
                            debug_print(f"OPTIMIZER_DEBUG: Point variable {name} has requires_grad={tensor.requires_grad}")
                            if not tensor.requires_grad:
                                raise ValueError(f"Point variable {name} still doesn't have requires_grad=True after fixing!")

                    # DEBUGGING: Print each cost function type in the objective
                    if hasattr(self, 'detailed_debug') and self.detailed_debug:
                        print("DEBUG OPTIMIZER FUNCTIONS:")
                        for i, cf in enumerate(objective.cost_functions):
                            print(f"  Cost function {i}: {cf.__class__.__name__} - {cf.name}")
                            if hasattr(cf, "optim_vars"):
                                # DEBUG: Check each optim_var and print its type to find the str issue
                                print(f"    Optimization variables count: {len(cf.optim_vars)}")
                                for j, var in enumerate(cf.optim_vars):
                                    print(f"      Var {j} type: {type(var)}")
                                    if isinstance(var, str):
                                        print(f"      WARNING: Found string variable instead of tensor: '{var}'")
                                    elif hasattr(var, 'name'):
                                        print(f"      Var name: {var.name}")
                                    else:
                                        print(f"      Variable has no name attribute")
                            print(f"    Cost weight: {cf.weight.__class__.__name__}")

                    # Now run the optimization - NO torch.no_grad() context manager
                    print("OPTIMIZER RUN: Running TheseusLayer.forward() with gradient tracking enabled")
                    try:
                        final_values, info = layer.forward(
                            inputs,
                            optimizer_kwargs={"track_best_solution": True}
                        )
                        print("OPTIMIZER SUCCESS: Forward pass completed successfully")
                        
                        # Print optimization results summary
                        status = info.status[0] if hasattr(info, 'status') and len(info.status) > 0 else "Unknown"
                        print(f"OPTIMIZER STATUS: {status}")
                        
                        if hasattr(objective, 'error_metric') and objective.error_metric() is not None:
                            err_val = objective.error_metric()
                            print(f"OPTIMIZER ERROR: Final error metric = {err_val}")
                        
                        debug_print("OPTIMIZER_DEBUG: Optimization completed successfully")
                        
                        # Print detailed information about the final values
                        debug_print(f"OPTIMIZER_DEBUG: Final values contain {len(final_values)} variables")
                        for name, tensor in final_values.items():
                            debug_print(f"OPTIMIZER_DEBUG: Final value '{name}' has shape {tensor.shape}")
                            
                        # Verify the optimization actually moved points
                        for name, tensor in final_values.items():
                            if name in inputs and torch.allclose(inputs[name], tensor):
                                debug_print(f"OPTIMIZER_DEBUG: WARNING - Variable {name} didn't move during optimization!")
                    except Exception as e:
                        logger.error(f"Optimization forward pass failed: {e}")
                        # Create fallback values if forward fails entirely
                        debug_print(f"OPTIMIZER_DEBUG: Creating fallback values due to forward pass failure")
                        final_values = {}
                        for name, tensor in inputs.items():
                            final_values[name] = tensor  # Use original inputs as fallback
                        info = None
                        success = False
                except RuntimeError as e:
                    # Handle potential dimension mismatch errors from Theseus
                    error_str = str(e)
                    debug_print(f"OPTIMIZER_DEBUG: TheseusLayer.forward() failed with error: {error_str}")
                    
                    # Look for specific dimension mismatch errors
                    if "expand" in error_str and "must be greater or equal to the number of dimensions" in error_str:
                        debug_print("OPTIMIZER_DEBUG: Detected dimension mismatch in TheseusLayer.forward()")
                        debug_print("OPTIMIZER_DEBUG: This is likely due to tensor shape inconsistencies")
                        
                        # Print detailed input shapes again
                        debug_print("OPTIMIZER_DEBUG: Detailed input tensor shapes:")
                        for name, tensor in inputs.items():
                            print(f"  {name}: {tensor.shape} (dtype={tensor.dtype})")
                    
                    # Re-raise the exception
                    raise
                
                # Check for NaN values in the results
                has_nan = False
                missing_vars = []
                
                for (y, x), var in self.variables.items():
                    if var.name not in final_values:
                        missing_vars.append(var.name)
                        # Only log this as debug, not as warning (too noisy)
                        debug_print(f"OPTIMIZER_DEBUG: Variable {var.name} not found in final_values")
                        continue
                        
                    if torch.isnan(final_values[var.name]).any():
                        has_nan = True
                        logger.warning(f"NaN detected in optimization results for point ({y},{x})")
                
                # Only consider it a failure if we have missing main candidate points (not neighbors)
                if missing_vars:
                    # Log a summary instead of individual warnings
                    if len(missing_vars) > 5:
                        logger.debug(f"{len(missing_vars)} variables not found in final_values (first 5: {missing_vars[:5]}...)")
                    else:
                        logger.debug(f"Missing variables: {missing_vars}")
                    
                    # Calculate what percentage of variables were missing
                    missing_percentage = len(missing_vars) / len(self.variables) * 100
                    
                    # Only treat as failure if a very high percentage of variables are missing
                    # Use a higher threshold (80%) since we've now cleared the registry in optimize_points
                    if missing_percentage > 80:
                        # Important: Don't use warning level for this anymore, as it's a common situation
                        # with the Theseus optimizer where variables aren't returned
                        if self.debug:
                            logger.debug(f"Theseus optimizer didn't return {missing_percentage:.1f}% of variables")
                        success = False
                    else:
                        # Just debug log if only a reasonable percentage is missing - this is normal
                        debug_print(f"OPTIMIZER_DEBUG: Theseus optimizer didn't return {missing_percentage:.1f}% of variables (usually expected)")
                        
                if has_nan:
                    logger.warning("Optimization produced NaN values, falling back to gradient initialization")
                    success = False
                    
            except Exception as e:
                # Log error details with tensor shapes for debugging
                error_msg = str(e)
                logger.warning(f"Optimization failed with error: {error_msg}")
                
                # Safely log the input tensors that are defined
                if 'inputs' in locals():
                    try:
                        shape_info = {name: tensor.shape for name, tensor in inputs.items()}
                        logger.warning(f"Input tensor shapes: {shape_info}")
                    except Exception as shape_err:
                        logger.warning(f"Failed to log tensor shapes: {shape_err}")
                else:
                    logger.warning("Input dictionary not defined yet")
                
                # For the specific "batch not defined" error, find where it's referenced
                if "name 'batch' is not defined" in error_msg:
                    debug_print("OPTIMIZER_DEBUG: Found 'batch not defined' error - this indicates a reference to 'batch' in the code")
                    debug_print("OPTIMIZER_DEBUG: Search for 'batch' in logger.info statements or other debug code")
                    debug_print(f"OPTIMIZER_DEBUG: Variables in current scope: {list(locals().keys())}")
                    # Print a full stack trace to help locate the issue
                    import traceback
                    debug_print("OPTIMIZER_DEBUG: Stack trace:")
                    traceback.print_exc()
                
                success = False
                
        except Exception as e:
            logger.warning(f"Failed to create optimization objective: {e}")
            success = False
        
        # Update grid with optimized values if optimization succeeded
        if success:
            for (y, x), var in self.variables.items():
                # Only update candidate points and points being processed
                if (y, x) in candidate_points or (self.grid.get_state(y, x) & STATE_PROCESSING) != 0:
                    # Check if the variable exists in final_values
                    if var.name not in final_values:
                        continue
                    
                    # FIXED: Ensure proper detachment of tensors
                    optimized_point = final_values[var.name].detach().cpu().numpy()
                    
                    # IMPORTANT: Add logging to show original values - use direct print for reliability
                    original_point = self.grid.get_point(y, x).copy()
                    print(f"DIRECT DEBUG: Point ({y},{x}) original position: {original_point}")
                    print(f"DIRECT DEBUG: Point ({y},{x}) optimized position from Theseus (XYZ format): {optimized_point[0]}")
                    print(f"DIRECT DEBUG: Variable tensor shape: {final_values[var.name].shape}")
                    
                    # Print a sample of what's in final_values to understand structure
                    if (y,x) == next(iter(candidate_points)):  # Only for first point
                        print(f"DIRECT DEBUG: Sample of final_values keys: {list(final_values.keys())[:5]}")
                        print(f"DIRECT DEBUG: Original input shape for {var.name}: {inputs[var.name].shape if var.name in inputs else 'Not in inputs'}")
                    
                    # CRITICAL FIX: Convert from Theseus XYZ ordering to our ZYX ordering
                    # Theseus uses th.Point3 with XYZ order, but our grid uses ZYX order
                    # So we need to swap coordinates from tensor[0]=x, tensor[1]=y, tensor[2]=z to our z,y,x
                    optimized_point_zyx = optimized_point.copy()
                    # Swap coordinates: X from Theseus becomes Z for us (index 0)
                    # Swap coordinates: Z from Theseus becomes X for us (index 2)
                    optimized_point_zyx[0, 0] = optimized_point[0, 2]  # z = theseus.z
                    optimized_point_zyx[0, 1] = optimized_point[0, 1]  # y = theseus.y
                    optimized_point_zyx[0, 2] = optimized_point[0, 0]  # x = theseus.x
                    
                    # Use the properly ordered point
                    optimized_point = optimized_point_zyx
                    print(f"DIRECT DEBUG: Point ({y},{x}) converted position (ZYX format): {optimized_point[0]}")
                    
                    # Apply post-optimization bounds checking
                    if self.volume_shape is not None:
                        # Ensure points are positive and in bounds
                        optimized_point[0, 0] = max(0.1, min(self.volume_shape[0] - 1.1, optimized_point[0, 0]))
                        optimized_point[0, 1] = max(0.1, min(self.volume_shape[1] - 1.1, optimized_point[0, 1]))
                        optimized_point[0, 2] = max(0.1, min(self.volume_shape[2] - 1.1, optimized_point[0, 2]))
                    
                    # Calculate position delta for debugging
                    delta = optimized_point[0] - original_point
                    delta_norm = np.linalg.norm(delta)
                    
                    # Compare input and output tensors for this variable
                    if var.name in inputs:
                        input_tensor = inputs[var.name]
                        output_tensor = final_values[var.name]
                        tensor_diff = output_tensor - input_tensor
                        tensor_diff_norm = torch.norm(tensor_diff).item()
                        print(f"DIRECT DEBUG: Variable {var.name} tensor changed by {tensor_diff_norm:.6f}")
                        print(f"DIRECT DEBUG: Input tensor: {input_tensor.detach().cpu().numpy()[0]}")
                        print(f"DIRECT DEBUG: Output tensor: {output_tensor.detach().cpu().numpy()[0]}")
                    
                    # Add debug info about the optimized point - use direct print
                    print(f"DIRECT DEBUG: Updated point ({y},{x}) to position {optimized_point[0]}")
                    print(f"DIRECT DEBUG: Point ({y},{x}) position delta: {delta}, magnitude: {delta_norm:.6f}")
                    
                    # IMPORTANT: Verify whether the position actually changed after coordinate conversion
                    if delta_norm < 1e-5:
                        print(f"DIRECT DEBUG: WARNING - Point ({y},{x}) didn't move after optimization!")
                    
                    # Apply the update
                    self.grid.set_point(y, x, optimized_point)
                    
                    # Mark as valid if optimization succeeded
                    self.grid.update_state(y, x, STATE_LOC_VALID | STATE_COORD_VALID)
        
        # Track metrics
        self.grid.generation += 1
        self.grid.success_count += len(candidate_points)
        
        # Store costs if available
        if success and hasattr(info, 'best_solution') and 'objective' in locals():
            if hasattr(objective, 'error_metric') and objective.error_metric() is not None:
                # FIXED: Ensure error value is properly detached
                error_val = objective.error_metric().detach().item() if torch.is_tensor(objective.error_metric()) else objective.error_metric()
                self.grid.generation_max_cost.append(error_val)
                self.grid.generation_avg_cost.append(error_val)
            else:
                # Add a placeholder value if error_metric isn't available
                self.grid.generation_max_cost.append(-1.0)
                self.grid.generation_avg_cost.append(-1.0)
        else:
            # Add a placeholder value for failed optimizations
            self.grid.generation_max_cost.append(-1.0)
            self.grid.generation_avg_cost.append(-1.0)
            
    def optimize_point(self, y: int, x: int) -> float:
        """
        Optimize a single point and return its loss value.
        
        This is a convenience wrapper around optimize_points for single point optimization.
        
        Args:
            y: Y coordinate of the point to optimize
            x: X coordinate of the point to optimize
            
        Returns:
            The final loss value after optimization
        """
        try:
            import logging
            logger = logging.getLogger(__name__)
            
            # We need a unique session ID for this optimization
            import time
            import random
            
            # Set the session ID on the optimizer instance for this run
            # This ensures all variables created will use this same ID
            self.current_optimization_id = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
            
            # ESSENTIAL: Clear the variable registry before optimization to avoid conflicts
            # This ensures a clean slate for each optimization run
            self.variable_registry = {}
            self.variables = {}
            
            print(f"OPTIMIZE_POINT: Starting optimization for point ({y},{x}) with ID {self.current_optimization_id}")
            
            # Create a consistent variable naming format using the new ID
            expected_var_name = f"p{y}_{x}_{self.current_optimization_id}"
            print(f"OPTIMIZE_POINT: Will create variable with name '{expected_var_name}'")
            
            try:
                # Note: We've already set the current_optimization_id above, no need to do it again
                
                # Call optimize_points with the single point to optimize
                # The integer conversion is done inside optimize_points
                self.optimize_points([(y, x)])
                print(f"OPTIMIZE_POINT: Successfully completed optimization for point ({y},{x})")
                
            except Exception as e:
                logger.error(f"Failed to optimize point ({y},{x}): {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return 100.0  # Return high loss value on failure
            finally:
                # Always clean up the ID after optimization, even if there was an error
                if hasattr(self, 'current_optimization_id'):
                    delattr(self, 'current_optimization_id')
            
            # Get the loss from the optimization
            loss = self.get_last_loss()
            print(f"OPTIMIZE_POINT: Loss for point ({y},{x}): {loss}")
            return loss
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Optimization failed with error: {e}")
            return 100.0  # High but not infinite
    
    def get_last_loss(self) -> float:
        """
        Get the last optimization loss value.
        
        Returns:
            The final loss value from the last optimization, or a large value if unavailable
        """
        # If we have generation cost info, use the most recent one
        if hasattr(self.grid, 'generation_avg_cost') and self.grid.generation_avg_cost:
            last_loss = self.grid.generation_avg_cost[-1]
            # Check if the loss is valid (not a placeholder -1.0)
            if last_loss >= 0.0:
                return last_loss
        
        # Default to a moderate value if no valid loss is available
        # Not too high to prevent rejecting all points, but not too low either
        return 0.5
    
    def sample_volume_at_point_3d(self, point: np.ndarray) -> float:
        """
        Sample the raw volume at a 3D point.
        
        Args:
            point: 3D point coordinates in ZYX order
            
        Returns:
            The interpolated intensity value at the point (0.0 if out of bounds)
        """
        try:
            # Check for valid point coordinates
            if np.any(np.isnan(point)) or np.any(np.isinf(point)):
                print(f"WARNING: Invalid coordinates in point: {point}")
                return 0.0
                
            # Check if point is within volume bounds
            if self.volume_shape is not None:
                for i, coord in enumerate(point):
                    if coord < 0 or coord >= self.volume_shape[i] - 1:
                        print(f"WARNING: Point {point} is out of volume bounds {self.volume_shape}")
                        return 0.0
                        
            # Create input tensors with batch size 1
            z = torch.tensor([[float(point[0])]], dtype=torch.float32)
            y_val = torch.tensor([[float(point[1])]], dtype=torch.float32)
            x_val = torch.tensor([[float(point[2])]], dtype=torch.float32)
            
            # Call evaluate with proper tensor inputs
            value = self.interpolator.evaluate(z, y_val, x_val)
            # FIXED: Ensure tensor is detached before calling item()
            result = float(value.detach().item())  # Detach before calling item() to avoid gradient issues
            
            print_debug(f"DEBUG: Sampled raw volume value {result} at point {point}")
            return result
        except Exception as e:
            print(f"ERROR: Error sampling at 3D point {point}: {e}")
            return 0.0
            
    def sample_distance_at_point_3d(self, point: np.ndarray) -> float:
        """
        Sample the distance transform at a 3D point.
        
        Args:
            point: 3D point coordinates in ZYX order
            
        Returns:
            The distance value at the point (0.0 for object voxels, >0 for background)
        """
        try:
            # Check for valid point coordinates
            if np.any(np.isnan(point)) or np.any(np.isinf(point)):
                print(f"WARNING: Invalid coordinates in point for distance sampling: {point}")
                return float('inf')
                
            # Check if point is within volume bounds
            if self.volume_shape is not None:
                for i, coord in enumerate(point):
                    if coord < 0 or coord >= self.volume_shape[i] - 1:
                        print(f"WARNING: Point {point} is out of volume bounds {self.volume_shape}")
                        return float('inf')
            
            # Sample from distance transform interpolator
            z, y, x = point
            dist_value = self.dist_interpolator.sample(z, y, x)
            
            print_debug(f"DEBUG: Sampled distance value {dist_value} at point {point}")
            return dist_value
        except Exception as e:
            print(f"ERROR: Error sampling distance at 3D point {point}: {e}")
            return float('inf')
    
    def _pre_optimize_using_gradient(self, candidate_points: List[Tuple[int, int]]):
        """
        Pre-optimize candidate points using volume gradient information.
        
        This helps provide better initialization for the optimizer by moving points
        in the direction of increasing intensity and outward from the center.
        
        Args:
            candidate_points: List of (y, x) coordinates for candidate points
        """
        # Get center point for distance calculation (important for outward growth)
        cy, cx = self.grid.center
        center_point = self.grid.get_point(cy, cx)
        
        for y, x in candidate_points:
            # Get initial position from neighbors
            neighbors = []
            for dy, dx in self.grid.neighbors:
                ny, nx = y + dy, x + dx
                if self.grid.is_valid(ny, nx):
                    neighbors.append(self.grid.get_point(ny, nx))
            
            if not neighbors:
                continue  # Skip if no valid neighbors
                
            # Initialize from average of neighbors
            avg_pos = np.mean(neighbors, axis=0)
            self.grid.set_point(y, x, avg_pos)
            
            # Calculate direction from center to point (outward direction)
            outward_vector = avg_pos - center_point
            outward_dist = np.linalg.norm(outward_vector)
            
            if outward_dist > 0.1:  # Only normalize if non-zero
                outward_vector = outward_vector / outward_dist
            else:
                # If we're too close to center, use a random direction
                outward_vector = np.random.rand(3) - 0.5
                outward_vector = outward_vector / np.linalg.norm(outward_vector)
            
            # Sample volume with gradient for intensity guidance
            try:
                value, gradient = self.sample_volume_with_gradient(y, x)
                
                # Ensure gradient is a proper numpy array
                if not isinstance(gradient, np.ndarray):
                    print(f"Error using gradient for initialization: gradient is {type(gradient)}, not ndarray")
                    gradient = np.zeros(3, dtype=np.float32)  # Default to zero gradient
                
                # Handle case where gradient doesn't have a shape attribute (scalar float)
                # This can happen with certain volume types or when interpolation returns a float
                try:
                    grad_shape = gradient.shape
                except AttributeError:
                    print(f"Warning: gradient is a scalar value without shape attribute. Creating zero gradient.")
                    gradient = np.zeros(3, dtype=np.float32)
                
                # Combine gradient direction and outward direction
                # Start with more outward bias for points near center
                if outward_dist < self.step_size * 4:
                    # For points very near center, use mostly outward direction
                    # with just a small influence from gradient
                    combined_vector = outward_vector * 0.8 + gradient * 0.2
                else:
                    # For points already away from center, balance gradient and outward
                    combined_vector = outward_vector * 0.4 + gradient * 0.6
                    
            except Exception as e:
                print(f"Error using gradient for initialization: {e}")
                # Fall back to using only outward direction
                combined_vector = outward_vector
                
            # Normalize the combined vector
            combined_norm = np.linalg.norm(combined_vector)
            if combined_norm > 1e-6:  # Avoid division by zero
                combined_vector = combined_vector / combined_norm
                
                # Move in the computed direction, with longer steps for points near center
                # to help them move outward faster
                if outward_dist < self.step_size * 2:
                    move_scale = self.step_size * 1.2  # Bigger steps for points near center
                else:
                    move_scale = self.step_size * 0.8  # Normal steps for other points
                    
                new_pos = avg_pos + combined_vector * move_scale
                
                # Apply bounds checking
                if self.volume_shape is not None:
                    new_pos[0] = max(0.1, min(self.volume_shape[0] - 1.1, new_pos[0]))
                    new_pos[1] = max(0.1, min(self.volume_shape[1] - 1.1, new_pos[1]))
                    new_pos[2] = max(0.1, min(self.volume_shape[2] - 1.1, new_pos[2]))
                
                # Update position
                print_debug(f"DEBUG: Pre-optimizing point ({y},{x}) from {avg_pos} to {new_pos}, dist from center: {outward_dist}")
                self.grid.set_point(y, x, new_pos)