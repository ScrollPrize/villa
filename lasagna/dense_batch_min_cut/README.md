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

- `<stem>_dt.tif`: normalized 16-bit distance transform visualization.
- `<stem>_ridges.tif`: 8-bit local maxima traced from the distance transform.
- `<stem>_skeleton.tif`: 8-bit thinned ridge skeleton.

The current ridge tracing is a deterministic local-maximum pass over the distance
transform. Skeletonization uses an in-tree Zhang-Suen thinning implementation so
the tool only depends on standard OpenCV modules.
