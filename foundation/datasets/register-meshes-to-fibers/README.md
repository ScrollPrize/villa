# Multi-Mesh Registration with Skeleton Constraints

Optimization-based registration to align 3D meshes with skeleton curves extracted from fiber inference TIFF volumes. It uses Open3D for mesh I/O and visualization, PyTorch for optimization, and Kimimaro for skeletonization.

## Project Structure

- **registration_pipe.py**  
  Main script that:
  - Recursively loads meshes and groups meshes that share a cube/TIFF context.
  - Extracts skeleton curves from corresponding TIFF volumes.
  - Runs registration optimization, including inter-mesh penalties within each group.
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

- **export_deformation_control_points.py**
  Exports sparse displacement controls from original and registered OBJ vertex pairs. The output JSON records `source_xyz`, `target_xyz`, and `displacement_xyz` values that can seed downstream volume-deformation experiments.

- **build_deformation_field.py**
  Interpolates sparse deformation controls into a coarse displacement grid saved as a compressed `.npz` file. The field is stored as a `zyx` grid whose vectors are `xyz` displacements.

- **warp_volume_with_field.py**
  Applies a displacement field to a `.npy`, `.tif`, `.tiff`, or `.zarr` volume using backward trilinear sampling. This is a reference path for evaluating deformation fields before adding heavier tiled resampling.

- **reference_volume_warp_pipeline.py**
  Runs the lightweight reference workflow end to end: original/registered mesh pairs become sparse controls, a coarse displacement field, and a warped `.npy`, TIFF, or Zarr volume.

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

### Export Deformation Controls

After registering meshes, export sparse control points for downstream volume-warp experiments:
```bash
python export_deformation_control_points.py \
  --mesh-pair example_data/meshes/cube_a/1_hz.obj registered-meshes/cube_a/1_hz_registered.obj \
  --output deformation-controls.json \
  --max-points-per-mesh 5000
```

Pass `--mesh-pair` multiple times to combine controls from several registered meshes. `--max-points-per-mesh 0` keeps every vertex.

Build a coarse displacement field from those sparse controls:
```bash
python build_deformation_field.py \
  --controls deformation-controls.json \
  --output deformation-field.npz \
  --grid-shape 256,256,128 \
  --spacing 8,8,8 \
  --origin 0,0,0 \
  --k 8 \
  --query-chunk-size 250000 \
  --control-chunk-size 50000
```

The `.npz` output contains `displacement_xyz`, `origin_xyz`, `spacing_xyz`, and `coordinate_order`. This is not a final volume resampler; it is an explicit intermediate field for evaluating and iterating on whole-volume deformation strategies. Use `--query-chunk-size` to cap how many grid points are interpolated per IDW batch, and `--control-chunk-size` to cap how many sparse controls are compared per sub-batch. Together they avoid allocating one full query-by-control distance matrix for large fields.

Apply a field to a test volume stored as a NumPy array, TIFF stack, or Zarr array:
```bash
python warp_volume_with_field.py \
  --volume volume.npy \
  --field deformation-field.npz \
  --output warped-volume.npy \
  --volume-spacing 1,1,1 \
  --volume-origin 0,0,0 \
  --fill-value 0 \
  --chunk-depth 64
```

Use `.tif` or `.tiff` for `--volume` or `--output` to read or write TIFF stacks, and `.zarr` to read or write Zarr arrays. TIFF and Zarr support use lazy imports, so `.npy` workflows do not require those dependencies at import time. Zarr inputs stay backed by the Zarr store instead of being converted to a full NumPy array up front. When `--volume` points at a Zarr group, the warper reads array key `0` by default; pass `--zarr-array-key` to select another multiscale level, or include the array key in the path, such as `volume.zarr/1`.

The reference warper resamples the displacement field onto the input volume grid and uses backward sampling, so a positive `x` displacement samples from lower input `x` coordinates. If `--volume-spacing` or `--volume-origin` are omitted, the field spacing and origin are reused. Use `--chunk-depth` to process the output volume in z-slice bands instead of allocating the full output sampling grid at once; chunked `.npy` output is written directly through a NumPy memmap, and chunked `.zarr` output is written directly to the Zarr store. `--chunk-depth 0` keeps the full-volume path.

Run the reference workflow in one command:
```bash
python reference_volume_warp_pipeline.py \
  --mesh-pair example_data/meshes/cube_a/1_hz.obj registered-meshes/cube_a/1_hz_registered.obj \
  --volume volume.tif \
  --output-dir deformation-run \
  --warped-output warped-volume.tif \
  --grid-shape auto \
  --field-spacing 8,8,8 \
  --field-origin 0,0,0 \
  --volume-spacing 1,1,1 \
  --volume-origin 0,0,0 \
  --max-points-per-mesh 5000 \
  --field-query-chunk-size 250000 \
  --field-control-chunk-size 50000 \
  --chunk-depth 64
```

This writes `deformation-controls.json`, `deformation-field.npz`, `warped-volume.npy`, `deformation-run-manifest.json`, and `deformation-run-metrics.json` into the output directory by default. Use `--grid-shape auto` to infer a field grid that covers the input volume extent from `--field-spacing` and `--volume-spacing`. Use `--field-query-chunk-size` and `--field-control-chunk-size` to pass chunked IDW interpolation through the end-to-end workflow. The manifest records the input volume, mesh pairs, actual field grid, requested spacing/origin metadata, effective volume spacing/origin used by the warp, field query/control chunk sizes, chunk depth, output paths, output write mode, and metrics for reproducible follow-up runs. The metrics file records control count, displacement-magnitude statistics, and the fraction of backward-sampled output voxels whose source coordinates remain inside the input volume. Use `--metrics-sample-step N` to estimate those sampling metrics from every Nth voxel per axis on large volumes; the default of `1` evaluates every voxel. The output filenames can be changed with `--controls-output`, `--field-output`, `--warped-output`, `--manifest-output`, and `--metrics-output`.

Grid shapes, field spacing, volume spacing, IDW nearest-neighbor counts, and metrics sample steps must be positive. Chunk depth and mesh-control sampling limits must be non-negative. The CLIs reject invalid values before building intermediate outputs.

### Skeleton Extraction

Extract and optionally visualize skeleton curves from a TIFF volume:
```bash
python extract_skeleton_tif.py --tif path/to/fiber.tif --cube_label path/to/cube_mask.tif --label 1 --fiber_type hz --axis_order zyx --visualize
```

## Configuration

Adjust registration and skeleton extraction parameters via command-line arguments:
- **Registration Hyperparameters:**  
  `--num_iters`, `--lr`, `--lambda_data`, `--lambda_disp`, `--lambda_elastic`, `--lambda_self`, `--lambda_lap`, `--lambda_arap`, `--lambda_sdf`, `--lambda_inter`, etc.
- **Skeleton Processing:**  
  `--skel_origin`, `--skel_axis`, `--curve_type`, `--target_skel_points`, etc.
  `--curve_type auto` maps `*_hz.obj` meshes to horizontal curves and `*_vt.obj`
  meshes to vertical curves in grouped runs; pass `horizontal` or `vertical` to
  force one curve class for every mesh. `--skeleton_axis_order` is forwarded to
  skeleton extraction and controls whether PCA interprets curves as NumPy-style
  `zyx` coordinates or already converted `xyz` coordinates. For standalone
  skeleton extraction, use `--axis_order` for the same PCA coordinate-order
  control. The registration pipeline defaults `--skeleton_axis_order zyx` and
  `--skel_axis zyx`, so extracted NumPy-style skeleton coordinates are classified
  consistently and converted to mesh-style `xyz` before optimization.

## Visualization

When enabled (via `--visualize`), the pipeline displays:
- The initial mesh with extracted skeleton curves.
- The registered mesh post-optimization.
