
# Installation

1. Activate your virtual environment. Instructions for `direnv`.

```sh
echo 'layout pyenv 3.12.5' > .envrc 
direnv allow
```

2. Install dependencies. `scrollcase` is installed as an editable package.

```sh
pip install -r requirements.txt
```

3. Install the `OCP CAD Viewer` VSCode estension to view BRep models displayed with `show()`. 

# Project structure

- `src/`: scrollcase source code
- `scripts/`: select example scripts

Within `src/`:
- `case.py`: Case construction with `build123d`
- `mesh.py`: Mesh processing
- `alignment.py`: Smallest enclosing cylinder
- `smallest_circle.py`: Helper utilities for Welzl's algorithm

# Usage

## Examples

See `scripts/` directory.

Simple case generation example at `scripts/example_basic_case.ipynb`.

## Enable logging

`scrollcase` uses the `logging` package for logs. Enable with:

```python
logging.basicConfig()  # Required in Jupyter to correctly set output stream
logging.getLogger("scrollcase").setLevel(logging.DEBUG)
```

## Mesh processing

See example `scripts/example_mesh_processing.ipynb`.

Mesh processing involves the following steps:
1. Optionally scale mesh to target size (for unscaled meshes)
2. Simplify (decimate) mesh to a target error
3. Fit the minimum bounding error
4. (Optional) Optimize rotation about the cylinder axis
5. (Optional) Smooth mesh
6. Offset by 2mm
7. Split along YZ plane
8. Remove overhangs

### Mesh smoothing

### `mesh_smooth_denoise`

Denoise smoothing. Configure the `gamma` parameter to vary smoothing amount.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/37858078-1245-4ca2-80db-0f13daa374d6" />

### `mesh_smooth_shrink_expand`

Smooth convexities/concavities by first shrinking/expanding the mesh and then expanding/shrinking back to the original value.

Shown here with 2 mm offset:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/f96841e0-b9b9-4156-9d65-cfd8d905c811" />



## Case construction

Case as viewed in the OCP CAD Viewer:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/cd32c255-e841-4cdb-8a43-0bdbd0284553" />

Enabling transparency to check clearances etc.:

<img width="1000" alt="image" src="https://github.com/user-attachments/assets/495dcffd-c353-496e-b086-b02d5a0bc729" />

### Customizing the case

The case is customized using the ScrollCase class:

<img width="600" alt="image" src="https://github.com/user-attachments/assets/d478fc5f-d35e-4726-ac6d-09b44c65b227" />


### Distribution of case sizes

Can be generated with `scratch/02-05_scroll-sizes.ipynb`.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/c957bf06-5891-4335-912b-e2bb35e2d76d" />



# Other notes
## Avoid pushing meshes in Jupyter notebooks
```
git config filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
```

