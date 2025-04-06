# Multi-Mesh Registration with Skeleton Constraints

Optimization-based registration to align 3D meshes with skeleton curves extracted from fiber inference TIFF volumes. It uses Open3D for mesh I/O and visualization, PyTorch for optimization, and Kimimaro for skeletonization.

## Project Structure

- **registration_pipe.py**  
  Main script that:
  - Recursively loads meshes.
  - Extracts skeleton curves from corresponding TIFF volumes.
  - Runs registration optimization.
  - Saves registered meshes preserving the input folder structure.

- **registration.py**  
  Implements the registration framework, including:
  - **Energy Terms:**
    - **Data Term:** Aligns deformed mesh vertices to target skeleton points via soft-assignment.
    - **Displacement Term:** L2 regularization on vertex displacements.
    - **Elasticity Term:** Penalizes changes in edge lengths to preserve local geometry.
    - **Self-Intersection Term:** Uses a softplus barrier to avoid mesh self-collisions.
    - **Laplacian Term:** Enforces smoothness by comparing each vertex to the average of its neighbors.
    - **ARAP Term:** Enforces local rigidity by computing optimal rotations (via SVD) and penalizing deviations.
    - **SDF Term:** Minimizes a weighted distance between deformed vertices and skeleton points.
    - **Inter-Mesh Intersection Term:** Penalizes collisions between vertices of different meshes.
  - Helper functions for neighbor computation and visualization.

- **extract_skeleton_tif.py**  
  Extracts and classifies skeleton curves from a TIFF volume using Kimimaro and PCA. Curves are categorized as "vertical" or "horizontal" based on their principal component.

- **environment.yml**  
  Conda environment specification for all required dependencies.

## Installation

```bash
conda env create -f environment.yml
conda activate register
```

To extract the example data:
```bash
7z x example_data.7z
```
## Usage

### Run the Registration Pipeline

Process all mesh files in a directory and output registered meshes:
```bash
python registration_pipe.py --mesh_root example_data/meshes --tif_root example_data/tifs --cube_label_root example_data/labels --output_root registered-meshes
```

Use the `--help` flag to see all configurable parameters (e.g., number of iterations, learning rate, and energy term weights).

### Skeleton Extraction

Extract and optionally visualize skeleton curves from a TIFF volume:
```bash
python extract_skeleton_tif.py --tif path/to/file.tif --visualize
```

## Configuration

Adjust registration and skeleton extraction parameters via command-line arguments:
- **Registration Hyperparameters:**  
  `--num_iters`, `--lr`, `--lambda_data`, `--lambda_disp`, `--lambda_elastic`, `--lambda_self`, `--lambda_lap`, `--lambda_arap`, `--lambda_sdf`, `--lambda_inter`, etc.
- **Skeleton Processing:**  
  `--skel_origin`, `--skel_axis`, `--target_skel_points`, etc.

## Visualization

When enabled (via `--visualize`), the pipeline displays:
- The initial mesh with extracted skeleton curves.
- The registered mesh post-optimization.
