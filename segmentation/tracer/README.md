# Volume Cartographer Tracer

This project reimplements the cost functions from the Volume Cartographer C++ library using Theseus for Python. These cost functions are used for nonlinear optimization of points in 2D and 3D space.

## Cost Functions

The following cost functions have been implemented:

### Distance-based Cost Functions

- `DistLoss`: Penalizes deviations from a target distance between two 3D points
- `DistLoss2D`: Penalizes deviations from a target distance between two 2D points

### Straightness-based Cost Functions

- `StraightLoss`: Encourages three 3D points to form a straight line by minimizing 1 - dot product of normalized direction vectors
- `StraightLoss2`: Encourages three 3D points to form a straight line by minimizing the distance between the middle point and the average of the two end points
- `StraightLoss2D`: Encourages three 2D points to form a straight line by minimizing 1 - dot product of normalized direction vectors

## Usage

The cost functions can be used with Theseus for various optimization tasks. Here's a basic example:

```python
import torch
import theseus as th
from cost_functions import DistLoss

# Create optimization variables
point_a = th.Point3(tensor=torch.tensor([[0.0, 0.0, 0.0]]))
point_b = th.Point3(tensor=torch.tensor([[1.0, 0.0, 0.0]]))

# Target distance to maintain
target_dist = 2.0

# Create cost function
dist_loss = DistLoss(point_a, point_b, target_dist, th.ScaleCostWeight(1.0))

# Create objective and add cost function
objective = th.Objective()
objective.add(dist_loss)

# Create optimizer
optimizer = th.LevenbergMarquardt(
    objective,
    th.CholeskyDenseSolver,
    max_iterations=20,
    step_size=1.0,
)

# Create TheseusLayer
layer = th.TheseusLayer(optimizer)

# Optimize
with torch.no_grad():
    final_values, info = layer.forward({
        'point_a': point_a.tensor,
        'point_b': point_b.tensor
    })

# Access optimized values
optimized_point_a = final_values['point_a']
optimized_point_b = final_values['point_b']
```

## Examples

See the `examples` directory for more complex usage examples, such as:

- `optimize_curve.py`: Shows how to use `DistLoss` and `StraightLoss2` to optimize a curve with specified spacing and straightness constraints.

## Testing

The `tests` directory contains unit tests for all cost functions. Run the tests with:

```bash
python -m unittest discover tests
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- Theseus (for differentiable optimization)
- Matplotlib (for examples)
- NumPy (for examples)