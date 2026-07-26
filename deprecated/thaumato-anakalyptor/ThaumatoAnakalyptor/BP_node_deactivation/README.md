# Loopy Belief Propagation Solver with Iterated MAP estimation for the Sheet Stitching Problem in PGMax 

#This version of the BP algorithm includes dynamic node deactivation

## Overview
This project implements a **Loopy Belief Propagation Solver**  with **Iterated MAP estimation** to provide an approximate solution for the *Sheet Stitching* problem. The solution is optimized for GPU using PGMax, a Python library for graphical model inference. The original formalization of belief propagation is based on the work of Francesco Mori, while the Laplacian smoothing approach is inspired by Julian Schilliger's Graph Problem Solver.

### Key Features:
- **Loopy Belief Propagation (BP):** Propagates messages across nodes in a graph to resolve conflicting information in the winding angles.
- **Iterated MAP estimation:** Updates winding angle allocations based on maximum a posteriori estimates from BP.
- **Laplacian Smoothing:** Smooths the graph's winding angle allocation by minimizing the shifts between spatially adjacent nodes.
- **GPU-Accelerated:** Efficient implementation optimized for CUDA 12+ with JAX and PGMax.

## Prerequisites

- **CUDA 12** or a later version must be installed on your system to run this solver, as it relies on GPU acceleration for computational efficiency.

## Installation

### Installation Steps

1. **Create and activate the conda environment:**
   Run the following command to create the environment:
   ```bash
   conda create --name bp_solver python=3.12
   ```

   Activate the environment:
   ```bash
   conda activate bp_solver
   ```

    ```bash
   pip install -U "jax[cuda12]"
   pip install git+https://github.com/deepmind/PGMax.git
   conda install numba networkx tqdm
   ```


## Usage

To run the solver, use the following command:
```bash
python bp_pgmax.py path/to/graph.npz <L> --bp_iterations <iterations> --output <final_result.npz>
```

### Example:
```bash
python bp_pgmax.py path/to/graph.npz 4 --bp_iterations 500 --output final_result.npz
```

### Parameters:
- `<L>`: The absolute size of the maximum shift corrected by a LBP cycle.
- `--bp_iterations`: The number of LBP iterations per cycle.
- `--output`: File path to save the final result.
- `--Big_iterations`:  Number of cycles.
- `--mu`: Cost (per edge) of deactivating a node.

## Detailed Description

This solver addresses the **Sheet Stitching Problem** by assigning winding angles to each instance in a scroll using a combination of **Loopy Belief Propagation (BP)**, **Iterated MAP estimation**, and **Laplacian Smoothing**. Here is a step-by-step breakdown of the approach:

### 1. **Initial Assignment (Maximum Spanning Tree):**
   We begin by assigning winding angles based on angle differences propagated through a **Maximum Weight Spanning Tree (MST)** of the graph. However, due to accumulated errors and conflicting information, nodes that are spatially close but on different branches of the MST may receive incorrect winding angle assignments.

### 2. **Loopy Belief Propagation (BP):**
   BP minimizes the **loss function**, defined as the sum of the absolute weighted differences (or "shifts") between neighboring nodes. The loss function is computed as:
`$ \text{Loss} = \sum_{\text{edges}} w_{ij} \cdot \exp( \left| \frac{\theta_i - \theta_j-k_{ij}}{360} \right| )\sigma_i \sigma_j+ \sum_{\text{edges}}\mu (1-\sigma_i)$`

   where  \( k_{ij} \) and  \( w_{ij} \) is the winding angle difference and the weight of the edge between nodes \( i \) and \( j \), and \( \theta_i \) and \( \theta_j \) are the winding angles at those nodes.  \( \sigma_i=0 \) if the node  \( i \) is deactivated and \( \sigma_i=1 \) otherwise. BP propagates messages across the graph to iteratively minimize this loss.

   **Note:** The size of the BP messages (determined by \( L \), the maximum allowable shift) is constrained by available GPU memory. As such, the value of \( L \) is set based on the system's capacity to balance accuracy and performance.

### 4. **Iterated MAP estimation:**
   After each BP cycle, the winding angle allocations are updated using **Maximum A Posteriori (MAP)** estimates. This step iteratively refines the winding angles to move them closer to their optimal values while considering the constraints of the graph's topology.


### Suggestion: Handling Local Minima
   Since BP on loopy graphs is not guaranteed to converge to the global minimum, you can alternate between **Laplacian Smoothing** and **BP + MAP** cycles to help escape local minima. This approach may lead to better overall solutions.

### Trade-offs:
   While this solution generally performs well, it is not guaranteed to yield the globally optimal solution due to the nature of the problem and the limitations of BP on loopy graphs.

## Contributing
Contributions to improve the solver are welcome. Please feel free to open an issue or submit a pull request.