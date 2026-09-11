# Ink depth review — working research prototype

A CPU-only, browser-based workflow for reviewing ink candidates across individual CT surface-volume layers. It is a concrete first contribution toward [ScrollPrize/villa #192](https://github.com/ScrollPrize/villa/issues/192), **not a solution to that issue or a claim to a prize**.

## Try the real sample

Generate `new-review/review.html` with the reproduction commands below, then open it in a desktop browser. It is self-contained, works offline, and sends nothing to a server. It contains a local excerpt of the official PHercParis4 surface volume.

1. Click **Next suggestion**. This navigates to a bright feature, not necessarily ink.
2. Move the depth slider. Click the two cross-sections to inspect continuity at other depths. Toggle suggestions off to inspect the scan without overlays.
3. Use **Mark ink** or **Mark background** only where you can actually justify the annotation. Brush edits affect one layer only. Leave ambiguity unknown. **Undo** reverses a brush stroke.
4. Enter a reviewer name and **Save reviewed voxels**. Resume later with that JSON file. Save before closing or refreshing: edits are held in browser memory.
5. Convert the review JSON to TIFF training assets with the export command below. Unknown voxels are excluded by the supervision mask.

**Do not mistake the existing cyan labels or orange suggestions for verified depth-specific ink.** No annotations in this package have been scientifically verified by a human. Automated UI-test marks were undone.

## What changed compared with the available flat labels

The official sample is a 65 × 256 × 256 surface-volume crop. All 32,864 positive prior pixels occur at layer 32. The official `label_zarr.py` defines that storage convention; it does not constitute 3D ground truth. The native training pipeline separately projects surface labels along normals, controlled by projection thickness.

This prototype adds individual-depth review, three orthogonal views, a deterministic intensity-based suggestion queue, reversible edits, source-bound reviews, and export of aligned image / ink / supervision volumes. No model prediction or 2D prior becomes an automatically confirmed training label. A known background voxel and an unknown voxel remain distinct.

**Coordinates:** the demonstrated data use `(surface depth, surface y, surface x)`, not native scroll `(z,y,x)`. This prototype does not map edits back through tifxyz, modify the native trainer, or claim that surface-coordinate TIFFs can be dropped directly into its native-3D loader. That integration is a separate, necessary step for a full #192 dataset submission.

## Reproduce

Python 3.14 is preferred, matching the official repository. The standalone tool also passed its tests under Python 3.13 on macOS arm64. No GPU, model weights, paid services, or accounts were used.

From this directory:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python fetch_sample.py --output sample
.venv/bin/python depth_review.py prepare \
  --image sample/image.tif --prior sample/prior.tif \
  --manifest sample/manifest.json --coordinate-system surface_dyx \
  --output new-review
```

Open `new-review/review.html`. Then:

```sh
.venv/bin/python depth_review.py export \
  --image sample/image.tif --review /path/to/review.json \
  --output reviewed-pair
INK_REVIEW_SAMPLE=sample .venv/bin/python -m pytest -q test_depth_review.py
```

Export writes `image.tif`, `inklabels.tif`, `supervision_mask.tif`, and `review.json`. TIFF axes are ZYX for interchange; the accompanying metadata explicitly identifies the surface coordinate system. `inklabels=1` only for explicitly marked ink; `supervision_mask=1` only for reviewed ink or background. A trainer **must** apply that supervision mask. Export refuses an empty review, duplicate/out-of-range indices, a wrong image hash, or an existing output directory.

TIFF or a local Zarr array/group level `0` can be input. Supply a bounded grayscale 3D crop with at most 16,777,216 voxels. This release expects a binary prior of the same shape. It does not stream an entire scroll, support arbitrary axis permutations, or infer physical spacing. The source manifest carries crop origin and source URLs; no millimeter scale is claimed without verified render-spacing metadata.

## Actual results

Source: official PHercParis4 segment `w00_20231016151002`, level 0, origin `(depth=0, y=13568, x=24832)`; crop `(65,256,256)`. The selection is deterministic from an existing label tile; it is not an independently held-out evaluation region. See `source-manifest.json` for URLs, hashes, sizes, and the one absent label chunk interpreted as zero under Zarr semantics. Network errors other than that explicit label-chunk 404 are not converted to background.

| Measurement | Result |
|---|---:|
| Scan voxels | 4,259,840 |
| Existing annotated layers | only 32 |
| Search support: prior footprint, layers 24–40 | 558,688 voxels |
| Intensity suggestions | 42,662 voxels, 100 connected regions |
| Suggestions as a fraction of search support | 7.64% |
| CPU suggestion generation, median of 5 runs | 0.257 s (Python 3.13, macOS arm64) |
| Automatically confirmed ink | 0 voxels |

This is a count reduction for a review queue, **not** measured time savings, precision, recall, or improved ink detection. The 558,688-voxel number is the declared search interval, not an upstream performance benchmark. Image loading and HTML generation are excluded from CPU timing. Full timings are in `benchmark.json`.

The score subtracts a Gaussian local intensity trend per layer (sigma 3 pixels), divides by robust median absolute deviation, thresholds at 2.5, intersects with the 2D prior footprint within ±8 layers, and removes 6-connected components smaller than 8 voxels. Flat layers yield no suggestions. Bright fibers, edges and artifacts can trigger it; weak or dark ink can be missed. Parameters were not optimized against ground truth. The annotation UI allows painting outside suggestions.

## Official code inspected and executed

Cloned `https://github.com/ScrollPrize/villa.git`, pinned at `b218ace879d3239a63047e1e2039f327929d283e`.

- Read `vesuvius/src/vesuvius/ink_detection/data/geometry.py`, especially `project_labels_and_supervision` and projection along normals.
- Read `preprocessing/create_label_zarrs.py`, `validate_segments.py`, `clean_labels.py`, `composite_from_zarr.py`, and shared `vesuvius/label_zarr.py`.
- Executed official `create_label_group` and `create_label_array` on the real crop; writing its prior at official `LABEL_SLICE=32` reproduces the downloaded label volume voxel-for-voxel.
- Executed the new suggestion and export workflow on the real sample. Regression tests use the same real CT; format-test edits are explicitly not scientific annotations.
- Full VC3D builds, GPU training/inference and ink-accuracy comparisons have **not** been run. Importing the small official label writer by file avoids bootstrapping unrelated ML/C++ packages.

## Data and code licensing

Code: MIT. Data: retain `DATA-LICENSE.txt` and the official source terms. Generated demos contain real data; this contribution excludes input crops, generated demos and candidate images. Do not publicly redistribute the demo or label images without resolving the source terms with the organizers. Cite EduceLab-Scrolls and [Parsons et al. (2023)](https://doi.org/10.48550/arXiv.2304.02084) where applicable.

Before claiming an annotation improvement, evaluate human review time and label accuracy on independent crops. Native scan integration also requires mapping the reviewed edits through tifxyz.

To repeat the official writer check and timing (with a checkout of villa):

```sh
.venv/bin/python benchmark.py --villa /path/to/villa --sample sample --output benchmark-new.json
```

This standalone CPU tool makes no changes to model training behavior. `VALIDATION.md` records the checks actually completed.
