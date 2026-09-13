# Photogrammetry and case generation

_Notes by Giorgio Angelotti, 26/08/2026_

This document describes the workflow from photographic acquisition through photogrammetric reconstruction and production of nylon cases.

## Equipment

The acquisition setup uses:

- Sony alpha6300 camera
- Spare batteries and battery charger
- Additional high-write-speed storage cards
- Gorilla tripod
- OrangeMonkie Foldio360
    - Smart Turntable
    - Light box
    - Halo bars
- ArUco stickers printed from [`markers_0.tif`](markers_0.tif). For our printed version the measured black-square side supplied to the script is 31.5 mm.
- Scale with at least decigram precision
- A ruler
- Peli-case with foam
- Low grammature japanese paper
- Bolts and screws (M5?)

## Software

- Turntable smartphone app https://orangemonkie.com/pages/foldio360-app
- SAM 2.1 and its checkpoints [`../sam2-photogrammetry/`](../sam2-photogrammetry/)
- Modified PGS-Recon [`../pgs-recon/`](../pgs-recon/)
- Scrollcase [`../scrollcase/`](../scrollcase/)

### Photographic acquisition

#### Initial setup

1. Ensure that the ArUco stickers are flat and firmly attached all around the turntable. The more, the better.
2. Create a session-log Google Doc or Sheet. Record timestamped notes for each scroll, orientation change, measured weight, acquisition start and any additional test photographs.
3. Tape cables to the tabletop and keep the area in front of the light box clear for the conservators.
4. Configure the camera, preferably using a saved preset:
    1. OSS: off
    2. Continuous shooting: high
    3. DRO: off
    4. Creative style: Neutral
    5. White balance: custom, measured on the background or turntable
    6. Capture format: RAW
5. Set the light box, halo bars and turntable lighting to maximum. You should use the Turntable smartphone app.

#### For each scroll

1. The conservator weighs the scroll and records the result in the session log. If the scroll weighs less than 15 g, do not use full turntable speed; otherwise use the DSLR 3x setting. With the ruler, measure approximately the length and the diameter of the scroll.
2. Clean the turntable surface.
3. Acquire at least two scroll orientations, limited to faces on which the conservator is comfortable resting the object:
    1. Remove the camera and lights to give the conservator space, then step away.
    2. The conservator places the scroll on the turntable and removes its label.
    3. Replace the halo lights and return all lights to maximum.
    4. Use at least two camera heights per orientation: low, slightly above the scroll; and high, the maximum stable tripod height.
    5. Before each acquisition block, photograph a written annotation identifying the upcoming block.
    6. Frame the scroll as tightly as practical and keep the zoom fixed throughout the rotation.
    7. Use approximately f/22, manual focus and the lowest ISO that still reveals the fine surface detail. Rotate the scroll once by hand to verify focus at every angle. A good compromise is 28 mm, f/22 and 1/6 s; the session note recommends about ISO 640 when lower ISO does not expose the black surface adequately.
    8. Record the start time of every automated turn.
    9. “96 shots per height, two turns at 3x speed - 48 photos”; however, the real number will depend on how many orientation will be needed to capture the shape without blind spots

#### After acquisition

The photographs may remain in a flat list separated by the written annotation images, but organizing them by scroll, orientation and camera height makes completeness easier to verify. For example:

* PHerc172/
    * orientation1_camera_low/
        * IMG_7065.ARW
        * IMG_7066.ARW
        * …
        * IMG_7184.ARW
    * orientation1_camera_high/
        * IMG_7186.ARW
        * …
        * IMG_7305.ARW
    * orientation2_camera_low/
        * …
    * orientation2_camera_high/
        * …
* PHerc145/
    * …

Regardless of layout, verify the expected photograph count for every acquisition block before leaving the session.

### Masking

Use [`foundation/sam2-photogrammetry`](../sam2-photogrammetry/) from the Villa repository at commit `a60235b67249918953b3d3c367d08d7da54ae459`. Use Python 3.12 and install ExifTool so that `raw2jpg.py` can transfer metadata from each RAW file to its JPEG.

The input tree may contain any number of scroll and orientation directories. RAW files belong directly in an acquisition directory; the scripts create or consume these sibling directories:

```text
<root>/<scroll>/<orientation>/
├── <image>.ARW
├── JPGEnhanced/
│   └── <image>.jpg
└── Masks/
    └── <image>_mask.png
```

Create an isolated environment from the pinned checkout.

NOTE: if you don't have a GPU, `SAM2_BUILD_CUDA=0` is appropriate for a CPU or Apple-MPS functional environment; otherwise use the upstream CUDA installation for production GPU processing.

```bash
git clone https://github.com/ScrollPrize/villa.git

cd villa/foundation/sam2-photogrammetry
uv venv --python 3.12 .venv
SAM2_BUILD_CUDA=0 uv pip install \
  --python .venv/bin/python \
  -e . \
  matplotlib scikit-image piexif
```

For photographs without a calibration ruler (like the one with the ArUco stickers you should be using), download these two checkpoints into `checkpoints/` and verify their hashes. The fine-tuned checkpoint is the scroll-only variant; use the separate `photo2_ruler_t_1000.torch` variant only for acquisitions that include a ruler.

| Checkpoint | Source | SHA-256 |
| --- | --- | --- |
| `sam2.1_hiera_tiny.pt` | [Meta SAM 2.1 tiny checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) | `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69` |
| `photo2_t_5000.torch` | [ScrollPrize scroll-only fine-tuned checkpoint](https://huggingface.co/scrollprize/sam2-photogrammetry/resolve/main/photo2_t_5000.torch) | `084dba5ddd02fdef6a0b95d4bfeb8dd259ce39c2077237f967fa384e2bace0a7` |

Run the required conversion and segmentation commands from [`foundation/sam2-photogrammetry`](../sam2-photogrammetry/):

```bash
SAM_INPUT_ROOT=/path/to/photogrammetry-input

.venv/bin/python raw2jpg.py "$SAM_INPUT_ROOT" \
  --quality 100 \
  --recompute

.venv/bin/python segment.py \
  --root_dir "$SAM_INPUT_ROOT" \
  --sam2_checkpoint checkpoints/sam2.1_hiera_tiny.pt \
  --model_cfg configs/sam2.1/sam2.1_hiera_t.yaml \
  --photo_t_checkpoint checkpoints/photo2_t_5000.torch \
    --recompute
```

The model configuration is a Hydra package key and must remain `configs/sam2.1/sam2.1_hiera_t.yaml`; it is not an absolute path.

For every acquisition directory, verify that:

1. Every RAW file has one readable JPEG with matching dimensions and transferred EXIF metadata.
2. Every JPEG has one readable, same-sized `Masks/<stem>_mask.png` file with non-empty foreground.
3. The palette labels are inspected rather than assumed. `segment.py` assigns object IDs from the initial frame and may segment the scroll and fiducial markers as different nonzero labels; its direct output is not necessarily a binary scroll-only mask.
4. The masks passed to reconstruction represent the intended foreground under the selected PGS-Recon `--mask-value` semantics.


#### Optional post-processing
The following commands are optional post-processing operations, not prerequisites for producing the `JPGEnhanced/` and `Masks/` PGS input layout:

```bash
python cc-fix.py "$SAM_INPUT_ROOT" \
  --workers 12 \
  --dust-threshold 256
```

`cc-fix.py` overwrites masks in place. It retains only the largest label-1 component and removes label-2 components smaller than the dust threshold, so use it only after confirming that those label semantics are correct for the dataset.

```bash
python mask-applier.py \
  --root_dir "$SAM_INPUT_ROOT" \
  --workers 12 \
  --recompute
```

`mask-applier.py` creates a separate `Masked/` dataset containing black-background JPEGs and binary masks derived only from label 1. It is not required for the PGS input preparation above and likewise requires verified label semantics.

### PGS-Recon (validated Vesuvius Challenge version with histogram production)

PGS-Recon is an EduceLab wrapper around OpenMVG and OpenMVS, together with a tool that detects ArUco markers and scales an SfM scene to physical units. Giorgio's changes add a weighted-median scale estimator and an SVG histogram of the marker-level scale estimates.

#### Reconstruction environment

##### Storage and AWS

I usually use an Ubuntu EC2 instance with enough RAM, CPU capacity and local scratch storage for the dataset. Memory demand depends strongly on image count, resolution, descriptor preset and feature-extraction concurrency. A current-generation Intel x86 worker with 64 vCPUs, 256 GiB of RAM and at least 300 GiB of local or root scratch storage is a validated baseline; `m7id.16xlarge` is one suitable AWS instance type and includes local NVMe scratch storage.


##### Build and container provenance

Use the ready-to-build source and Docker context at [`foundation/pgs-recon`](../pgs-recon/).

The Villa snapshot is based on upstream PGS-Recon revision [`5ccaa10f4de821aaa5f98e60593fd8be69ecda3a`](https://gitlab.com/educelab/pgs-recon/-/commit/5ccaa10f4de821aaa5f98e60593fd8be69ecda3a), with the global-scaler work from Giorgio's [merge request !39](https://gitlab.com/educelab/pgs-recon/-/merge_requests/39) and the validated pause, diagnostic and resume changes. Upstream later incorporated the scaler work through [merge request !41](https://gitlab.com/educelab/pgs-recon/-/merge_requests/41).

The printable [`markers_0.tif`](markers_0.tif) was created by Seth Parker. It contains ArUco markers 0000–0034. Print it at 100% scale with automatic fit-to-page or rescaling disabled, measure the printed black-square side and pass that physical measurement to `--marker-size`.

Build from the Villa directory itself:

```bash
cd foundation/pgs-recon
docker build --tag pgs-recon:villa-validated .

docker run --rm pgs-recon:villa-validated pgs-recon --help
docker run --rm pgs-recon:villa-validated pgs-global-scaler --help
```

Before reconstruction:

1. Record the Villa commit, full container-image ID, build host and architecture, base image, installed packages and OpenMVG/OpenMVS versions.
2. Retain the `--help` output for `pgs-recon` and `pgs-global-scaler` and confirm that it includes `--stop-after`, `--resume-sfm`, `--autoscale-histogram-out`, `--autoscale-debug-dir`, scaler `--histogram-out` and scaler `--save-debug-images`.

#### Reconstruction workflow

Follow the phases below in order. Do not continue to scaling until the sparse reconstruction is accepted, and do not continue to OpenMVS until one scaled scene is accepted. Use fresh output directories for retries so that every parameter change remains attributable.

##### Run record and parameter baseline

For every reconstruction, retain:

- Exact top-level command and every generated OpenMVG/OpenMVS command
- Input and output paths
- Container-image ID and source revision
- All descriptor, matching, geometric-filtering, robustification, scaling and OpenMVS parameters

Use global SfM, SIFT, the `HIGH` descriptor preset, matching ratio `0.75`, 32 threads and robustification as the operational baseline. `HIGH` gives a useful balance between feature coverage and runtime. `ULTRA` extracts more features and is the better-quality choice when `HIGH` does not recover sufficient coverage, but it is substantially slower and more memory-intensive. `NORMAL` is useful for a quick diagnostic reconstruction, but it is not the default for the final run.

The nearest-neighbour ratio must be in `(0, 1]`; increasing it accepts progressively less distinctive matches. Keep the ratio at `0.75` initially and relax it only after inspecting a failed sparse reconstruction.

Set host paths and a run name that records the important choices:

```bash
export PGS_IMAGE=pgs-recon:villa-validated
export PGS_PHOTOS=/absolute/path/to/pgs-input
export PGS_RUN=/absolute/path/to/reconstruction
export PGS_RUN_NAME=object-high-r075-robust

mkdir -p "$PGS_RUN/sfm"
```

##### 1. Reconstruct and review sparse OpenMVG SfM

OpenMVG initializes the camera model, extracts features, computes putative matches, geometrically filters them and reconstructs the sparse scene and camera poses. With `--mvg-robust`, PGS-Recon then re-triangulates the scene from the recovered poses before stopping. This robustified scene is the baseline input to scaling.

Run through OpenMVG once and stop before scaling:

```bash
docker run --rm \
  --mount type=bind,src="$PGS_PHOTOS",dst=/workspace/photos,readonly \
  --mount type=bind,src="$PGS_RUN/sfm",dst=/workspace/output \
  "$PGS_IMAGE" \
  pgs-recon \
    --input /workspace/photos \
    --output /workspace/output \
    --name "$PGS_RUN_NAME" \
    --mvg-recon-method global \
    --describer-method SIFT \
    --describer-preset HIGH \
    --matching-ratio 0.75 \
    --threads 32 \
    --file-type ply \
    --mvg-robust \
    --stop-after sfm
```

The reconstruction directory retains both scenes:

```text
<PGS_RUN>/sfm/mvg/recon_dir/sfm_data.bin                 # before robustification
<PGS_RUN>/sfm/mvg/recon_dir/sfm_data_structured.bin      # after robustification; scale this one
```

Inspect the robustified scene in the OpenMVG Viewer. Require a coherent sparse cloud, approximately circular camera trajectories at the acquired heights, no unexplained angular gaps and at least 95% of retained views registered. Inspect the match graph, track count, final reprojection residual, intrinsics and sparse-point distribution together. A non-circular trajectory can indicate that the object moved on the turntable, in which case that acquisition block must not be used to estimate scale.

It's important to use ply as output file type, because many readers downstream that work on other format used to corrupt the input mesh while parsing.

If the sparse reconstruction is not acceptable, preserve it and change one parameter family at a time:

1. First correct missing or invalid images, mask-label mistakes, inconsistent acquisition grouping, camera metadata and obvious turntable movement.
2. Compare `sfm_data.bin` with `sfm_data_structured.bin`. Robustification is the baseline, but it can occasionally remove useful structure; preserve a no-robust branch as a diagnostic if the robustified scene is visibly worse.
3. If the inputs are valid but matching is too sparse, retry ratio `0.8`. A higher ratio accepts more ambiguous matches, so inspect false connections as well as registration coverage. Relax it further only in small, separately recorded increments when the added matches improve real coverage without introducing false geometry.
4. If `HIGH` remains insufficient, retry with `ULTRA`. Reduce `--threads` if feature extraction approaches the host's memory limit. Use `NORMAL` only when a faster, lower-detail diagnostic is useful.
5. Change the descriptor family, matching method or geometric model only as an independently named experiment only after the simpler changes above have failed.

##### 2. Estimate and review physical scale

The scaler detects the printed ArUco markers, triangulates their corners and maps the accepted SfM scene into the physical unit used for `--marker-size`. Scale only the accepted robustified scene, and apply scale before OpenMVS.

The markers are fixed in the turntable frame, whereas a deliberately reoriented or accidentally displaced object is not. Use the acquisition log, annotation images and camera trajectories to create basename-only include lists for blocks in which the object orientation remained fixed. Combine camera heights only when they belong to the same fixed orientation. Do not infer blocks from historical filename counts.

If more than one block may be valid, estimate every candidate independently. Run the standalone scaler on the accepted `sfm_data_structured.bin` so that comparing candidates does not repeat feature extraction, matching or SfM:

```bash
export PGS_SCALE_LISTS=/absolute/path/to/include-lists
export PGS_MARKER_SIZE_MM=REPLACE_WITH_MEASURED_SIDE_IN_MM

for include_path in "$PGS_SCALE_LISTS"/keep-*.txt; do
  candidate="$(basename "$include_path" .txt)"
  candidate="${candidate#keep-}"
  mkdir -p "$PGS_RUN/scale-candidates/$candidate/debug-images"

  docker run --rm \
    --mount type=bind,src="$PGS_PHOTOS",dst=/workspace/photos,readonly \
    --mount type=bind,src="$PGS_RUN/sfm",dst=/workspace/sfm,readonly \
    --mount type=bind,src="$PGS_RUN/scale-candidates/$candidate",dst=/workspace/scale-output \
    --mount type=bind,src="$PGS_SCALE_LISTS",dst=/workspace/scale-lists,readonly \
    "$PGS_IMAGE" \
    pgs-global-scaler \
      --input-scene /workspace/sfm/mvg/recon_dir/sfm_data_structured.bin \
      --output-scene /workspace/scale-output/sfm_data_scaled.bin \
      --sfm-root /workspace/photos \
      --scale-method umeyama \
      --marker-size "$PGS_MARKER_SIZE_MM" \
      --detection-method markers \
      --include-from "/workspace/scale-lists/keep-${candidate}.txt" \
      --histogram-out /workspace/scale-output/histogram.svg \
      --save-debug-images /workspace/scale-output/debug-images \
      --save-landmarks /workspace/scale-output/landmarks.ply \
      --save-scaled-landmarks /workspace/scale-output/landmarks-scaled.ply \
      --progress
done
```

The scaler defaults omitted from the command are a minimum marker size of 32 pixels and RANSAC-enabled corner triangulation. Each Umeyama histogram sample is a marker-level similarity scale estimated from at least three triangulated corners. The reported summary is a weighted median, with the number of available corners used as the weight.

Before accepting a candidate, inspect:

- marker counts and the number of triangulated corners per marker;
- debug overlays for missed or false detections;
- unscaled and scaled landmark meshes for the expected planarity and squareness;
- the complete histogram for multiple modes, broad variance or extreme tails;
- agreement between independent candidates that should represent the same fixed orientation;
- scaled object dimensions against an independent physical measurement.

Detection success or the weighted median alone is not sufficient. Reject false detections, outlier-dominated estimates and candidates assembled from incompatible acquisition blocks. Preserve the include list, scaled scene, histogram, overlays, landmark meshes and log for every candidate that informed the decision.

The integrated `--mvg-autoscale` route can stop after scaling and can write its histogram and overlays through `--autoscale-histogram-out` and `--autoscale-debug-dir`. Use it when one include list is already known to be valid. Prefer the standalone scaler above when candidates must be compared because it reuses the accepted sparse scene.

##### 3. Resume OpenMVS from the accepted scaled scene

OpenMVS converts the scaled SfM scene into an MVS scene, optionally densifies the point cloud, reconstructs the surface mesh, optionally refines it and finally textures it. In the baseline below, `--mvs-densify` enables densification, `--mask-value 0` tells densification to ignore background label 0, refinement remains enabled and `--refine-resolution-level 1` reduces the refinement image resolution by one level.

After accepting one scale candidate, resume directly from its scaled scene. Do not pass `--mvg-robust`, `--mvg-autoscale` or `--stop-after` with `--resume-sfm`: the accepted scene already contains the results of those earlier decisions, and the CLI rejects those combinations.

```bash
export PGS_CANDIDATE=REPLACE_WITH_ACCEPTED_FIXED_ORIENTATION_ID

mkdir -p "$PGS_RUN/openmvs/$PGS_CANDIDATE"

docker run --rm \
  --mount type=bind,src="$PGS_PHOTOS",dst=/workspace/photos,readonly \
  --mount type=bind,src="$PGS_RUN",dst=/workspace/run \
  "$PGS_IMAGE" \
  pgs-recon \
    --input /workspace/photos \
    --output "/workspace/run/openmvs/$PGS_CANDIDATE" \
    --name "$PGS_RUN_NAME-$PGS_CANDIDATE" \
    --resume-sfm "/workspace/run/scale-candidates/$PGS_CANDIDATE/sfm_data_scaled.bin" \
    --threads 32 \
    --file-type ply \
    --mvs-densify \
    --mask-value 0 \
    --refine-resolution-level 1
```

The resume invocation skips feature extraction, matching, SfM, robustification and scaling. A resume config may display the parser's unused default descriptor preset; use the stopped SfM run's metadata to establish that the executed feature extraction used `HIGH`. Do not rerun into a non-empty result directory.

Require finite vertices and valid faces, nonzero extents, one dominant face-connected component, a plausible silhouette and object-aligned dimensions consistent with independent measurements. A case-generation mesh must also be watertight and free of boundary edges, non-manifold edges, orientation conflicts and degenerate faces. Inspect canonical renders for holes, bridges, collapsed regions and disconnected debris.

If OpenMVS fails, retain the completed intermediate files and diagnose stages in increasing order of geometric impact:

1. If texturing fails after a valid refined PLY was produced, keep the untextured mesh for geometry and case-generation QA and diagnose `TextureMesh` separately. The PGS-Recon CLI always invokes texturing; it has no top-level disable-texturing flag.
2. If refinement fails or makes the geometry worse, retry as a separately named branch with `--no-mvs-refine`.
3. If densification fails, retry without `--mvs-densify` only as a separately validated branch; mesh reconstruction is always required.
4. If you have memory problems, try increasing the `--refine-resolution-level`.

Case generation does not intrinsically require color texture, but every omitted stage changes the retained artifacts and must be recorded. PLY is the required interchange format for topology QA.

In MeshLab, use `Filters -> Quality Measures and Computations -> Compute Topological Measures`. Alternatively, use a PLY-native topology checker and retain its exact version, command and machine-readable report with the run. Inspect the original PLY indices because some visualization readers duplicate vertices at texture seams and can falsely report disconnected components. Since the world axes are arbitrary, compare physical size with PCA-aligned or otherwise object-aligned extents rather than only the raw axis-aligned bounding box.

#### Case generation

Use [`foundation/scrollcase`](../scrollcase/) from Villa.

scrollcase contains all the code to generate the shape of the case programmatically.
By changing the code and the parameters, one can change the shape of the case.

Installation:
```bash
cd villa/foundation/scrollcase
sudo apt-get install libxrender1
uv venv --python 3.12 .venv

uv pip install --python .venv/bin/python \
  -e . \
  tqdm
```

Coordinates of the input ply mesh must already be in millimetres. Stage each validated, scaled mesh under its numeric scroll identifier:

```text
<input>/
└── <scroll-number>/
    └── <scroll-number>-scaled.ply
```

The general rule is `<input>/<scroll-number>/<scroll-number>-scaled.ply`.

Run the generator:

```bash
SCROLLCASE_INPUT=/path/to/scrollcase-input
SCROLLCASE_OUTPUT=/path/to/scrollcase-output

.venv/bin/python scripts/stl_generator.py \
  --input "$SCROLLCASE_INPUT" \
  --output "$SCROLLCASE_OUTPUT"
```

The output directory uses the numeric identifier padded to four digits and contains:

```text
<output>/
├── <padded-scroll-number>/
│   ├── <padded-scroll-number>_scroll.stl
│   ├── <padded-scroll-number>_right.stl
│   └── <padded-scroll-number>_left.stl
└── scroll_summary.csv
```

The case label combines the configured collection prefix with the padded scroll identifier. The CSV records the padded identifier, lining-interior height and lining-outer diameter in millimetres.

The generator expands and smooths the mesh, creates the scroll proxy and combines a fitted lining with the two case halves. Provide all three STLs per sample to the 3D-printing company only after verifying them.

Confirm that every input produced three readable, non-empty STLs and one CSV row. Record bounds, connected components, boundary/non-manifold topology, winding consistency, watertightness and visual renders.

The generator uses multiple workers and may finish its parent process even if a worker failed. A zero exit status is therefore insufficient. Require the padded identifier's three expected STLs, a populated `scroll_summary.csv` row, readable finite geometry and sensible assembly renders before accepting the kit.

#### Printing

For scrolls fixtured in Naples, the historical supplier is 3DNa S.r.L. in Pomigliano d'Arco (NA). The printed cases are fragile and can be damaged in transit, so local printing and hand delivery to the library are preferred when possible. If shipment is unavoidable, use substantial protective packaging.

Use HP Multi Jet Fusion (MJF) with the finest available nylon powder to obtain a uniform grain density. Filament-based printing can introduce phase artefacts in synchrotron imaging. Smooth the cases, remove every residual glass bead that could cause scan artefacts, and paint them black with a verified non-metallic paint.

For PHercParis3 and PHercParis4, the case interiors were additionally coated with polyurethane paint to make them smoother and reduce the risk of abrasion. This should become the standard for future sessions.

#### Fixturing

The conservators first wrap each scroll in very thin, low grammature Japanese paper, then close the case with the specified screws and bolts. The paper grammage and fastener specifications still need to be documented.

Transport the closed cases in a foam-filled Peli case with a fitted cut-out for each case.

When transporting more than one case, mark each with a distinct, unambiguous symbol, such as a unique sticker (e.g. Pokemon stickers)

#### Scanning remarks

1. To scan with 0.22 sample-propagation distance on BM18, an additional cylinder needs to be put on top of the mounting stage, below the case, to allow the turret-detector to come close enough.
2. During a long scan, both the nylon case and the scroll can move due either to thermal excitation or for the deterioration of the nylon. Solidary translation and rotations of case+scroll can be fixed in reconstruction by a special Nightrail's BM18 script, but there's no fixing for non-solidary movements.
3. To mitigate the effect, it could be worth researching on whether printing the case in radio-resistant resins rather than nylon could be better.