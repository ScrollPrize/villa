# Volume Cartographer Tracer

A Python implementation of cost functions from the Volume Cartographer C++ library using Theseus for differentiable nonlinear optimization.

## Overview

This project provides a set of cost functions that can be used with the Theseus optimization framework to perform nonlinear optimization of points in 2D and 3D space. These cost functions are direct ports of the ones used in the Volume Cartographer C++ library, maintaining the same mathematical properties while leveraging the advantages of PyTorch and Theseus.

## Implemented Cost Functions

All cost functions from the original C++ implementation have been ported:

### Distance-based Cost Functions
- `DistLoss`: Penalizes deviations from a target distance between two 3D points
- `DistLoss2D`: Penalizes deviations from a target distance between two 2D points
- `LinChkDistLoss`: Penalizes deviations from a target 2D location using sqrt of absolute differences

### Straightness-based Cost Functions
- `StraightLoss`: Encourages three 3D points to form a straight line by minimizing 1 - dot product of normalized direction vectors
- `StraightLoss2`: Encourages three 3D points to form a straight line by minimizing the distance between the middle point and the average of the two end points
- `StraightLoss2D`: Encourages three 2D points to form a straight line by minimizing 1 - dot product of normalized direction vectors

### Z-Coordinate Constraints
- `ZCoordLoss`: Constrains a 3D point to have a specific Z coordinate
- `ZLocationLoss`: Interpolates a 3D point from a matrix and enforces a specific Z coordinate

### Surface and Volume Constraints
- `SurfaceLossD`: Interpolates a 3D point from a location and enforces position
- `SpaceLossAcc`: Samples a 3D volume and minimizes/maximizes the interpolated value
- `AnchorLoss`: Anchors points to specific positions using volume interpolation
- `SpaceLineLossAcc`: Evaluates multiple points along a line using interpolation

### Utilities
- `TrilinearInterpolator`: Helper class for 3D volume interpolation

## Usage

### Installation

```bash
# Install directly from source
pip install -e .
```

### Basic Example

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

## Cost Function Details

### Distance-based Functions

#### DistLoss
```python
DistLoss(point_a, point_b, target_distance, cost_weight, name=None)
```
- **Purpose**: Penalizes deviations from a target distance between two 3D points
- **Parameters**:
  - `point_a`, `point_b`: 3D points (th.Point3)
  - `target_distance`: Target distance to maintain
  - `cost_weight`: Weight for this cost function

#### DistLoss2D
```python
DistLoss2D(point_a, point_b, target_distance, cost_weight, name=None)
```
- **Purpose**: Penalizes deviations from a target distance between two 2D points
- **Parameters**: Similar to DistLoss but for 2D points (th.Point2)

### Straightness Functions

#### StraightLoss
```python
StraightLoss(point_a, point_b, point_c, cost_weight, name=None)
```
- **Purpose**: Encourages three 3D points to form a straight line using dot product method
- **Parameters**:
  - `point_a`, `point_b`, `point_c`: Three 3D points (th.Point3)
  - `cost_weight`: Weight for this cost function

#### StraightLoss2
```python
StraightLoss2(point_a, point_b, point_c, cost_weight, name=None)
```
- **Purpose**: Encourages three 3D points to form a straight line using midpoint method
- **Parameters**: Same as StraightLoss

### Volume-based Functions

#### SpaceLossAcc
```python
SpaceLossAcc(point, interpolator, cost_weight, name=None)
```
- **Purpose**: Samples a 3D volume at a point location and uses the interpolated value
- **Parameters**:
  - `point`: 3D point to sample (th.Point3)
  - `interpolator`: TrilinearInterpolator instance
  - `cost_weight`: Weight for this cost function

#### AnchorLoss
```python
AnchorLoss(point, anchor_point, interpolator, cost_weight, name=None)
```
- **Purpose**: Anchors a 3D point based on volume values and distance
- **Parameters**:
  - `point`: 3D point to anchor (th.Point3)
  - `anchor_point`: 3D anchor point (th.Point3)
  - `interpolator`: TrilinearInterpolator for the 3D volume
  - `cost_weight`: Weight for this cost function

## Testing

The project includes unit tests for all cost functions. Run them with:

```bash
python -m unittest discover tests
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- Theseus (for differentiable optimization)
- NumPy