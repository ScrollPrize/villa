"""
Example of using cost functions to optimize a curve.

This example demonstrates how to use the DistLoss and StraightLoss cost functions
to optimize the shape of a curve.
"""

import torch
import theseus as th
import matplotlib.pyplot as plt
import numpy as np

from cost_functions import DistLoss, StraightLoss2


def optimize_curve():
    """Optimize a curve using DistLoss and StraightLoss2 cost functions."""
    # Create a set of 3D points to form an initial curve
    n_points = 10
    
    # Start with points in a curve shape (half circle)
    initial_points = []
    for i in range(n_points):
        t = i / (n_points - 1) * np.pi
        x = np.cos(t)
        y = np.sin(t)
        if y < 0:
            y = 0
        initial_points.append([x, y, 0.0])
    
    points_tensor = torch.tensor([initial_points], dtype=torch.float64)
    
    # Create theseus Point3 variables
    points = []
    for i in range(n_points):
        points.append(th.Point3(tensor=points_tensor[:, i, :], name=f"point_{i}"))
    
    # Set up the objective
    objective = th.Objective()
    
    # Add DistLoss cost functions to maintain spacing between consecutive points
    target_dist = 0.5
    for i in range(n_points - 1):
        dist_loss = DistLoss(
            points[i], 
            points[i+1], 
            target_dist, 
            th.ScaleCostWeight(1.0),
            name=f"dist_{i}"
        )
        objective.add(dist_loss)
    
    # Add StraightLoss2 cost functions for consecutive triplets of points
    for i in range(n_points - 2):
        straight_loss = StraightLoss2(
            points[i],
            points[i+1],
            points[i+2],
            th.ScaleCostWeight(0.5),  # Lower weight to allow some curvature
            name=f"straight_{i}"
        )
        objective.add(straight_loss)
    
    # Fix the first and last points
    fix_first = th.Difference(
        points[0],
        th.Point3(tensor=points_tensor[:, 0, :]),
        th.ScaleCostWeight(10.0),
        name="fix_first"
    )
    fix_last = th.Difference(
        points[-1],
        th.Point3(tensor=points_tensor[:, -1, :]),
        th.ScaleCostWeight(10.0),
        name="fix_last"
    )
    objective.add(fix_first)
    objective.add(fix_last)
    
    # Create optimizer
    optimizer = th.LevenbergMarquardt(
        objective,
        th.CholeskyDenseSolver,
        max_iterations=50,
        step_size=1.0,
    )
    
    # Create TheseusLayer
    layer = th.TheseusLayer(optimizer)
    
    # Prepare input dictionary
    input_dict = {}
    for i in range(n_points):
        input_dict[f"point_{i}"] = points_tensor[:, i, :]
    
    # Optimize
    with torch.no_grad():
        final_values, info = layer.forward(input_dict)
    
    # Extract optimized points
    optimized_points = torch.zeros_like(points_tensor)
    for i in range(n_points):
        optimized_points[:, i, :] = final_values[f"point_{i}"]
    
    return points_tensor, optimized_points


def plot_curves(initial_points, optimized_points):
    """Plot the initial and optimized curves."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot initial curve
    ax.plot(
        initial_points[0, :, 0], 
        initial_points[0, :, 1], 
        initial_points[0, :, 2], 
        'o-', 
        label='Initial Curve'
    )
    
    # Plot optimized curve
    ax.plot(
        optimized_points[0, :, 0], 
        optimized_points[0, :, 1], 
        optimized_points[0, :, 2], 
        's-', 
        label='Optimized Curve'
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('optimized_curve.png')
    plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run the optimization
    initial_points, optimized_points = optimize_curve()
    
    # Plot the results
    plot_curves(initial_points, optimized_points)
    
    print("Optimization complete. Results saved to 'optimized_curve.png'")