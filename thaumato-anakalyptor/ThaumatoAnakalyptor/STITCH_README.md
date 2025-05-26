# Mesh Splitting and Stitching

This module provides functionality to split large meshes into smaller pieces for processing, and then stitch the rendered results back together into a single UV-mapped image.

## Overview

The workflow consists of two main steps:

1. **Splitting/Cutting**: Split a large mesh into smaller pieces using either:
   - `split_mesh.py`: Advanced UV-based splitting with cylindrical projection
   - `finalize_mesh.py`: Simple texture-coordinate based cutting
2. **Stitching**: Reassemble rendered images of the pieces back into the original mapping

## Files

- `split_mesh.py`: Contains the `MeshSplitter` class for UV-based mesh splitting
- `finalize_mesh.py`: Contains functions for texture-coordinate based mesh cutting
- `stitch_splits.py`: Contains both `MeshStitcher` and `FinalizeMeshStitcher` classes for stitching

## Prerequisites

- Original mesh file (.obj format)
- For `split_mesh.py`: Umbilicus path file (for UV coordinate calculation)
- Rendered images of the split/cut meshes (various formats supported: .tif, .tiff, .jpg, .png, etc.)

## Usage

### Method 1: Using split_mesh.py (Advanced UV-based splitting)

#### Step 1: Split the Mesh

```bash
python3 -m ThaumatoAnakalyptor.split_mesh \
    --mesh /path/to/original_mesh.obj \
    --umbilicus_path /path/to/umbilicus.txt \
    --split_width 50000
```

This creates:
- A `windowed_mesh_YYYYMMDDHHMMSS/` folder containing split mesh pieces
- `vertices_flattened.npy` file with UV coordinates
- `split_info.json` file with splitting metadata

#### Step 2: Render and Stitch

```bash
# Render each split mesh, then stitch
python3 -m ThaumatoAnakalyptor.stitch_splits \
    /path/to/original_mesh.obj \
    /path/to/rendered_images/ \
    --type split_mesh

# Or stitch a specific image file from each window folder
python3 -m ThaumatoAnakalyptor.stitch_splits \
    /path/to/original_mesh.obj \
    /path/to/rendered_images/ \
    --type split_mesh \
    --image_filename composite.jpg
```

### Method 2: Using finalize_mesh.py (Simple texture-based cutting)

#### Step 1: Cut the Mesh

```bash
python3 -m ThaumatoAnakalyptor.finalize_mesh \
    --input_mesh /path/to/original_mesh.obj \
    --cut_size 20000 \
    --scale_factor 2.0
```

This creates:
- `working/` folder containing cut mesh pieces
- `cut_info.json` file with cutting metadata

#### Step 2: Render and Stitch

```bash
# Render each cut mesh, then stitch
python3 -m ThaumatoAnakalyptor.stitch_splits \
    /path/to/original_mesh.obj \
    /path/to/rendered_images/ \
    --type finalize_mesh

# Or stitch a specific image file from each cut folder
python3 -m ThaumatoAnakalyptor.stitch_splits \
    /path/to/original_mesh.obj \
    /path/to/rendered_images/ \
    --type finalize_mesh \
    --image_filename prediction.png
```

### Auto-Detection

The stitcher can automatically detect which type of splitting was used:

```bash
# Basic auto-detection
python3 -m ThaumatoAnakalyptor.stitch_splits \
    /path/to/original_mesh.obj \
    /path/to/rendered_images/

# Auto-detection with specific image filename
python3 -m ThaumatoAnakalyptor.stitch_splits \
    /path/to/original_mesh.obj \
    /path/to/rendered_images/ \
    --image_filename composite.jpg
```

## File Structure

### After split_mesh.py:

```
mesh_directory/
├── original_mesh.obj                    # Original mesh
├── vertices_flattened.npy              # Flattened UV coordinates
├── vertices_colored.ply                # Debug: colored vertices
└── windowed_mesh_YYYYMMDDHHMMSS/       # Split meshes folder
    ├── split_info.json                 # Splitting metadata
    ├── original_mesh_window_0_50000.obj
    ├── original_mesh_window_0_50000_cylindrical.png
    └── ...
```

### After finalize_mesh.py:

```
mesh_directory/
├── original_mesh.obj                    # Original mesh
├── working/                            # Working folder
│   ├── cut_info.json                  # Cutting metadata
│   ├── working_mesh/
│   │   └── mesh.obj                   # First cut
│   ├── working_mesh_1/
│   │   └── mesh_1.obj                 # Second cut
│   └── ...
```

## Parameters

### split_mesh.py Parameters

- `--split_width`: Width of each split window in UV coordinates (default: 50000)
- `--mesh`: Path to the original mesh file
- `--umbilicus_path`: Path to the umbilicus file for UV calculation

### finalize_mesh.py Parameters

- `--input_mesh`: Path to the original mesh file
- `--cut_size`: Size of each cut piece along the X axis (default: 20000)
- `--scale_factor`: Scaling factor for vertices (default: 2.0)
- `--output_folder`: Folder to save the cut meshes (default: working/)

### Stitching Parameters

- `original_mesh`: Path to the original mesh file
- `renders_folder`: Path to folder containing rendered images
- `--output`: Output path for the stitched image (optional)
- `--type`: Type of stitching ('auto', 'split_mesh', 'finalize_mesh')
- `--image_filename`: Specific image filename to look for in each cut/window folder (e.g., 'composite.jpg', 'prediction.png')

## Output

The stitching process creates:

1. **Main output**: A stitched image combining all rendered pieces
2. **Coverage mask**: A debug image showing which areas were covered during stitching

## Technical Details

### split_mesh.py Process

1. Loads the original mesh and computes UV coordinates using BFS traversal
2. Scales UV coordinates to create a flattened representation
3. Splits the mesh into windows based on U-coordinate ranges
4. Uses triangle-level precision for accurate mapping

### finalize_mesh.py Process

1. Loads the original mesh and texture coordinates
2. Cuts the mesh into pieces along texture X-axis
3. **Saves triangle mapping information**: Records which triangles belong to each cut and their original UV coordinates
4. Normalizes UV coordinates for each piece (for rendering)
5. Uses triangle-level precision for accurate stitching (when triangle info is available)

### Stitching Process

Both stitchers now use triangle-level precision:
1. Load the original mesh and cutting/splitting metadata
2. **Triangle-level mapping**: For each triangle, determine its exact pixel coverage
3. Map rendered images back to original coordinate space with triangle precision
4. Handle overlapping regions appropriately
5. Create coverage masks for debugging

**Fallback support**: The FinalizeMeshStitcher includes fallback to rectangular region mapping for older cut files without triangle information.

## Troubleshooting

### Common Issues

1. **"No windowed_mesh_* or working* folder found"**
   - Make sure you've run the splitting/cutting step first
   - Check that the process was successful

2. **"No render found for window/cut"**
   - Ensure rendered images are named to match the mesh files
   - Check that image files have supported extensions

3. **"Cut/Split info file not found"**
   - The cutting/splitting process may have failed
   - Re-run with the updated code

4. **Poor stitching quality**
   - For split_mesh: Try adjusting the `split_width` parameter
   - For finalize_mesh: Try adjusting the `cut_size` parameter
   - Ensure rendered images have sufficient resolution

### Debug Output

Both stitchers create coverage mask images that show which areas were successfully filled. Black areas indicate missing or problematic data.

## Performance Considerations

- **split_mesh.py**: Advanced UV-based splitting with cylindrical projection and triangle-level stitching
- **finalize_mesh.py**: Texture-coordinate based cutting with triangle-level stitching (enhanced)
- Both methods now provide high accuracy through triangle-level processing
- Image resolution affects both quality and processing time
- Triangle-level processing provides the best quality but can be slower for very dense meshes

## Integration with Existing Pipeline

Both methods integrate with the existing ThaumatoAnakalyptor pipeline:

```bash
# Method 1: Using split_mesh
python3 -m ThaumatoAnakalyptor.split_mesh --mesh mesh.obj --umbilicus_path umbilicus.txt
python3 -m ThaumatoAnakalyptor.large_mesh_to_surface --input_mesh windowed_mesh_folder/ --scroll scroll.zarr
python3 -m ThaumatoAnakalyptor.stitch_splits mesh.obj rendered_outputs/

# Method 2: Using finalize_mesh
python3 -m ThaumatoAnakalyptor.finalize_mesh --input_mesh mesh.obj --cut_size 20000
python3 -m ThaumatoAnakalyptor.large_mesh_to_surface --input_mesh working/ --scroll scroll.zarr
python3 -m ThaumatoAnakalyptor.stitch_splits mesh.obj rendered_outputs/
```

## Choosing Between Methods

- **Use split_mesh.py when**:
  - You have an umbilicus file
  - Working with cylindrical/scroll-like meshes
  - Need advanced UV coordinate computation
  - Working with complex mesh topologies

- **Use finalize_mesh.py when**:
  - You don't have an umbilicus file
  - Working with meshes that already have good UV coordinates
  - Need simpler setup and faster initial processing
  - Working with more standard mesh structures

**Note**: Both methods now provide triangle-level stitching accuracy. The main difference is in the initial coordinate computation and splitting approach.

## Stitching Specific Images from Cut/Window Folders

The stitcher supports stitching specific image files that are located within each cut or window folder. This is useful when you have multiple processed images per cut (e.g., composite images, predictions, etc.).

### How it works

When you specify `--image_filename`, the stitcher will look for that specific file in each cut/window folder:

1. **First priority**: Look in the cut/window's own folder (same directory as the .obj file)
2. **Second priority**: Look in the renders folder with cut/window name prefix
3. **Third priority**: Look in the renders folder with just the filename

### Example folder structure

```
working/
├── cut_info.json
├── working_mesh/
│   ├── mesh.obj
│   ├── composite.jpg          # ← Found here first
│   └── prediction.png
├── working_mesh_1/
│   ├── mesh_1.obj
│   ├── composite.jpg          # ← Found here first
│   └── prediction.png
└── ...

renders_folder/
├── mesh_composite.jpg         # ← Found here second
├── mesh_1_composite.jpg
├── composite.jpg              # ← Found here third (fallback)
└── ...
```

### Usage examples

```bash
# Stitch composite.jpg from each cut folder
python3 -m ThaumatoAnakalyptor.stitch_splits \
    mesh.obj renders/ \
    --image_filename composite.jpg

# Stitch prediction.png from each window folder
python3 -m ThaumatoAnakalyptor.stitch_splits \
    mesh.obj renders/ \
    --image_filename prediction.png \
    --type split_mesh

# Output will be named automatically: mesh_composite_stitched.png
```

## Parameters 