# Kaggle Visualizer

Napari helper to browse paired 3D TIFF volumes and binary labels with connected-component coloring.

## Environment

```
conda env create -f environment.yml
conda activate kaggle-visualizer
```

## Usage

Place training volumes and labels in separate folders with matching filenames (e.g., `train/001.tif` and `labels/001.tif`).

```
python -m kaggle_visualizer \
  --train-dir /path/to/train \
  --label-dir /path/to/labels \
  --log-mergers-csv /path/to/mergers.csv \
  --log-tiny-csv /path/to/tiny.csv
```

Controls:
- `n` move to the next sample (wraps around)
- `b` move to the previous sample (wraps around)
- `v` toggle isolating a single connected component (show only the selected component)
- `k` move to the next connected component within the current sample (wraps)
- `j` move to the previous connected component within the current sample (wraps)
- `g` append the current sample ID to the mergers log CSV (created if missing; duplicates ignored)
- `t` append the current sample ID to the tiny-components log CSV (created if missing; duplicates ignored)
- `c` cycle label source: auto → raw → fixed (if available)

Notes:
- Training data is shown in grayscale; labels use connected components (26-connectivity) of voxels with label value `1` colored with the `glasbey` palette from `colorcet` (labels `0` and `2` are treated as background). Before labeling, diagonal-only bridges (checked across XY, ZX, and ZY orientations) that merge separate 6-connected components are pruned, and a `_fixed` label volume is written alongside the original when pruning changes the mask.
- The overlay text shows the current sample name and index.
