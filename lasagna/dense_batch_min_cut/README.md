# dense_batch_min_cut

Small C++/OpenCV experiments for dense batched min-cut preprocessing.

## Build

Requires OpenCV with `core`, `imgcodecs`, and `imgproc`.

```bash
cmake -S lasagna/dense_batch_min_cut -B lasagna/dense_batch_min_cut/build
cmake --build lasagna/dense_batch_min_cut/build
```

## Run

```bash
cd path/to/workdir
/path/to/villa2/lasagna/dense_batch_min_cut/build/dense_batch_preprocess \
  -i path/to/image.tif
```

Outputs:

- `<stem>_binary.tif`: 8-bit binary mask from a fixed threshold of 127.
- `<stem>_dt.tif`: normalized 16-bit distance transform visualization.
- `<stem>_skeleton_zs.tif`: 8-bit Zhang-Suen thinning of the binary foreground.
- `<stem>_skeleton_dt_ordered.tif`: 8-bit distance-ordered topology-preserving
  thinning of the binary foreground.

The threshold is intentionally fixed for repeatable comparisons. The two
skeletonizers both operate on the binary foreground; the distance transform is
used for visualization and to order candidate removals in the distance-ordered
variant.
