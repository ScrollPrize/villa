# Notebook examples

Open these notebooks in order or start with the workflow you need. Each exposes
its reader calls and processing steps directly; `_display.py` only draws figures.
Saved outputs show an executed run on the published data.

| Notebook | What it does |
| --- | --- |
| [2.5D ink training](01_ink_training_25d.ipynb) | Uses all supervised labels at XY level 3 for one PHercParis4 segment, shows annotation sources and full-canvas masks, holds out a contiguous validation section, and trains a 3D convolutional stem with depth max pooling and a 2D U-Net on nine-layer CT chunks. |
| [3D ink training](02_ink_training_3d.ipynb) | Scans a full PHercParis4 segment for supervised 512×512 seeds, caches projected CT patches on disk, trains a 3D U-Net for complete epochs, then stitches overlapping predictions into a flattened letter-scale image on a held-out segment. |
| [Zarr reader](03_zarr_reader.ipynb) | Opens groups and arrays, selects levels, reads slices and batches of blocks, reuses buffers and bounded caches, and reopens a crop offline. |
| [Transforms and CUDA](04_transforms_and_cuda.ipynb) | Shows image/label/validity semantics and reproducible CPU transforms, then a pinned DataLoader → CudaLoader → model → optimizer path. |
| [Mesh segment rendering](05_render_mesh_segment.ipynb) | Discovers registered geometry and displays CT at −1, 0 and +1 source voxel along the surface normals, with an optional CUDA comparison. |

## Run

Install the package in the Python environment used by your Jupyter kernel.
From the repository root, install the package with the extras used by the notebooks:

```bash
python -m pip install -e '.[reader,labels,torch]'
```

To install all optional Python dependencies, including benchmark tools, use:

```bash
python -m pip install -e '.[all]'
```

The `all` extra includes `reader`, `labels`, `transforms`, `torch`, `previews`
and `bench` dependencies. On a fresh build, CUDA extensions are enabled
automatically when a CUDA compiler is found; otherwise the build is CPU-only.
Discovery includes standard toolkit locations such as `/usr/local/cuda`.
For a toolkit elsewhere, set `CUDACXX=/path/to/nvcc`.

To force a CPU-only build, use:

```bash
python -m pip install -e '.[all]' \
  --config-settings=cmake.define.VC_READER_ENABLE_CUDA=OFF
```

Use `VC_READER_ENABLE_CUDA=ON` to require CUDA and fail if it is unavailable.
Explicit settings persist in the CMake build cache, so pass `ON` to enable
CUDA after a previous CPU-only build.

This builds and installs the compiled extensions. The reader notebook needs NumPy
and Pillow. The surface and training notebooks also need the `labels`
dependencies (including tifffile),
and the training and transform notebooks need PyTorch. These correspond to the package's
`reader`, `labels` and `torch` extras. Jupyter and its Python kernel must already
be installed in your chosen environment. No notebook installs packages.

From the repository root, launch `jupyter lab examples/` and run the cells in
order. Use `examples/` as the notebook working directory so the adjacent
`_display.py` helper can be imported. CUDA sections
need a CUDA-enabled build and device; both training notebooks also run on CPU.
The general reader and mesh-rendering notebooks do not import PyTorch.

An editable installation uses the current Python source but still needs its
compiled extensions rebuilt after C++/CUDA changes. If a notebook reports
`cache_max_bytes must be nonnegative` with the default settings, check which
kernel and extensions it loaded:

```python
import sys
from vc_reader import _reader, _transforms
print(sys.executable)
print(_reader.__file__)
print(_transforms.__file__)
```

Use the environment containing the current build. After reinstalling compiled
extensions, **restart the kernel and run all cells**; re-running imports does
not unload an extension already in memory.

Configuration is visible in the setup cells. Optional environment variables:

| Variable | Purpose |
| --- | --- |
| `VC_EXAMPLE_METADATA` | Local metadata snapshot; omit to use normal catalog loading. |
| `VC_EXAMPLE_CACHE` | Disk cache, default `/tmp/vc-reader-notebooks`. |
| `VC_EXAMPLE_DEVICE=cpu` | Run the training models on CPU even if Torch sees a GPU. The dedicated CUDA demonstration still checks CUDA independently. |

The first real-data run can take substantially longer than training while it
fetches CT chunks and coordinate TIFFs. Readers share a bounded chunk cache
(64 MiB normally; 256 MiB with eight read threads in the 3D notebook).
Readers close before training. Both training notebooks store prepared patches
on disk; the 3D notebook reuses completed projections across runs.
TIFF decoding, output arrays and model memory are separate from the chunk budget.
The examples never download a complete CT volume or search it for random patches.

## Data and supervision choices

The 2.5D example selects PHercParis4 segment `20231210121321`, label set
`20260918`, source volume `20260411134726`. It prints the published ink,
supervision and CT URLs plus the original annotation raster names. This segment
has no published validation mask. The notebook reserves a contiguous bottom
section, choosing a whole-tile-row boundary closest to 20% of supervised pixels.
The split depends only on supervision coverage. Full-canvas ink and supervision
views show the boundary and color the training and validation regions.

A non-overlapping grid of 512×512 tiles covers the full canvas at XY level 3,
including partial edge tiles and supervised negative-only regions. Nine stored
layers centered on the middle layer form a one-channel volumetric chunk. The window is derived
from the render header: 109 layers give middle index 54 and inputs 50–58. A label-only scan avoids fetching CT for
unsupervised tiles. Prepared patches use uint8 CT and boolean labels/support on
disk; batches convert to float32. Training and validation supports partition
all supervised pixels with valid labels and valid CT in every input layer.
Coverage is at the selected pyramid resolution; published decimation can omit
finer annotations. Training visits every training tile once per epoch, with three
epochs by default. All validation tiles contribute to metrics. Illustrations
use well-supervised tiles containing both ink and background, selected by
annotation coverage and class counts without using model predictions. Captions
report coverage and ink fraction.

The 2.5D model takes `(B,1,D,H,W)` chunks through two 3×3×3 convolutions
with four feature channels, then max-pools over depth to `(B,4,H,W)`. A 2D
U-Net produces `(B,1,H,W)` ink logits. The narrow 3D stem bounds activation
memory while retaining full XY resolution. It uses GroupNorm, CT-only normalization,
matched random XY flips, and training-weighted BCE plus soft Dice. Validation never
contributes to the optimizer or class weight. This is a same-segment spatial
split: adjacent texture can be correlated, and its metrics do not establish
transfer to other segments or scrolls.

The 3D example uses training segment `20230702185753` and holds out segment
`20230929220926`, both PHercParis4 label set `20260918`. A label-only scan at
level 3 covers the entire training canvas with a non-overlapping grid of
seeds of 512×512 full-resolution pixels. Every tile with scanned supervision is retained,
including negative-only and partial edge tiles. The scan finds 772 training
seeds covering 2,060,815 supervised level-3 pixels. Published decimation can
omit finer annotations. Validation selects 43 supervised seeds from the fixed
field of 4096×4096 full-resolution pixels used for the final stitched view. Figures show the
ink, published supervision and selected grid; the notebook prints source URLs.

Seeds project into CT level 2 with a 9.6 µm projection band and 19.2 µm context
padding. Preparation runs on CPU or CUDA and saves uint8 CT plus boolean sheet,
ink and support arrays. Completed files are reused using source/registration,
projection settings and backend in the cache key. The Dataset loads one patch
at a time and converts the current batch to float32. The executed dataset uses
about 5.1 GiB of prepared files, separate from the 4 GiB source chunk cache. Remove the printed prepared
cache directory to rebuild projections after changing the reader implementation.
Seeds without projected support are reported and excluded.

The 88,889-parameter U-Net has two downsampling stages, skip connections and
GroupNorm; only CT is normalized. Three epochs visit every prepared training
patch once per epoch, using matched 3D flips and masked weighted BCE plus soft
Dice. Class weights use training support only. Every prepared validation patch
contributes to metrics after each epoch. Voxel counts include repeated
observations where projected crop margins overlap.

The second input is geometric sheet coverage, including non-ink sheet voxels
and sheet continuing through the padded crop. The seed rectangles select crop
bounds; they do not truncate the projected sheet. Geometry coverage is
independent of label availability. Supervision and conflicts affect loss support,
not that input channel. Label-pixel-to-UV registration is explicitly declared
using the published full render-canvas convention; volume registration and
geometry ownership are discovered from metadata.

After training, the 3D notebook predicts the validation context of 4096×4096
full-resolution pixels with 225 geometry-bounded chunks, seeded by 512×512 rectangles at stride
256. Each 3D prediction is trilinearly sampled at the registered sheet's label
pixel centers, then blended in 2D with tapered overlap weights. The final
512×512 surface view shows CT, supervised ink and stitched probabilities over
the same letter-scale field. Geometry and CT validity determine rendering
coverage; labels determine only scoring support. Each supported output pixel is
scored once. These surface metrics are separate from the projected-crop voxel
metrics. One chunk and small 2D accumulators bound memory.
Projection uses the model device; CUDA outputs enter Torch through DLPack
without a host round-trip. Training chunks use the same seed size as inference.
Holding out a segment ID does not establish that different segments never cover
overlapping physical sheet regions.

For measurements of reading, collation, transform, and batch prefetch,
see the [PyTorch batch benchmark](../benchmarks/README.md#pytorch-cuda-batches).

## Recorded validation

The 2.5D notebook was executed on real PHercParis4 segment `20231210121321`
with CUDA. Its coverage tests also exercise local paired readers,
partial edge tiles, negative-only labels and rejection of an empty split.
CPU/CUDA model tests check variable-depth chunks, depth max pooling, finite
gradients and updates to the 3D convolution weights. The recorded
split contains 39 training tiles (2,217,069 supported pixels) and 13 validation
tiles (543,948 pixels). The validation section begins at level-3 Y = 4608,
covering 19.7% of supervision. Saved outputs report the three-epoch metrics alongside the
all-ink baseline; this short run demonstrates the workflow. Prepared patches occupy about 143 MiB on disk. The 2.5D notebook explicitly
caps the shared source cache at 4 GiB so this full-segment run fits for replay.

The reader, transform and mesh notebooks were executed on CPU, including
offline replay of real-data reads. CUDA runs exercised the full
CudaLoader/model loop, and CPU/CUDA equality of rendered layers. Saved notebook
outputs are from CUDA runs; finite-loss and optimizer-update assertions pass.
The notebooks report an all-ink Dice baseline alongside model metrics so a
short run dominated by one predicted class is easy to recognize. The 3D notebook reports both its projected-crop voxel metrics and the larger
stitched surface metrics. CPU/CUDA tests verify fractional surface sampling,
large coordinate origins, invalid geometry and overlap normalization. Both
notebooks use final weights.

The expanded 3D notebook was executed end to end on CUDA. All 772 training and
43 validation seeds produced usable projected patches, with 34,379,667 and
1,710,732 supported voxel observations respectively. Each of the three epochs
visited every training patch (2,316 optimizer updates total). Training and
validation took 106 seconds after preparation. Held-out voxel Dice varied
0.43 → 0.48 → 0.16 across epochs. The final stitched view has 100% rendering
coverage and 122,056 supervised output pixels; its Dice is 0.21 against an
all-ink baseline of 0.52. The larger dataset completes successfully, but this
short run does not yet give reliable letter predictions. Overlapping inference
took 132 seconds. The saved outputs show these final weights.

The expanded dataset tests cover negative-only labels, partial edge seeds,
variable-shaped disk patches, training-only weights and complete epochs on
CPU/CUDA, alongside surface sampling and overlap normalization.
Results can differ across devices and runs after optimization.

The run used CPython 3.14, Linux amd64 and an RTX 5090, with metadata SHA256
`55aa3d217732a3c75401765f7d425738a35fecab75baad6e4ef5cb8d8b9844e3` and source
fingerprint `de38ba456eedfa4e0c4c6d6b4e093f4664d635eae551bfbe923bbdf6dc4ec58d`.
Cells were executed directly as Python and their real text/PNG outputs saved;
the Jupyter application itself was not available in the validation environment.
The refreshed editable installation was also checked, exercising the same
package imports used by the notebooks.
