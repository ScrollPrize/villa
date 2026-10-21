# Photogrammetry and case generation

_Originally by Giorgio Angelotti, 26/08/2026; updated by Paul Henderson, 10/2026_

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

If the annotation photographs or the session log are missing, the blocks can usually be recovered from the camera metadata. Within a turntable turn, frames are about a second apart; turns of one block follow each other after a short pause (roughly 10–20 s); blocks are separated by longer breaks and usually by a zoom change, since each camera height is reframed. [`photogrammetry-tools/find_blocks.py`](../photogrammetry-tools/) groups the photographs on these cues using ExifTool, reports frame-number gaps and byte-identical duplicate files (for example `DSC01234_1.ARW` left by a repeated copy), and writes a proposed block list with a contact sheet of every burst. Timing cannot always separate a long pause inside a block from a short break between two blocks at the same zoom, and it says nothing about when the scroll was reoriented, so confirm the blocks, and which blocks share an orientation, on the contact sheet before naming them and building the directory tree with its `link` command.

### Masking

Use [`foundation/sam2-photogrammetry`](../sam2-photogrammetry/) from the Villa repository at commit `a60235b67249918953b3d3c367d08d7da54ae459` or later; `scroll-mask.py`, and the `segment.py` change that keeps host memory bounded on full-resolution photographs, were added after that commit. Use Python 3.12 and install ExifTool so that `raw2jpg.py` can transfer metadata from each RAW file to its JPEG.

The input tree may contain any number of scroll and orientation directories. RAW files belong directly in an acquisition directory; the scripts create or consume these sibling directories:

```text
<root>/<scroll>/<orientation>/
├── <image>.ARW
├── JPGEnhanced/
│   └── <image>.jpg
├── Masks/
│   └── <image>_mask.png
└── ScrollMasks/
    └── <image>_mask.png
```

`segment.py` treats each `JPGEnhanced/` folder as a video: it segments the first frame and propagates those objects through the rest. Each orientation directory must therefore contain one fixed acquisition block (one scroll orientation and camera height) and no annotation photographs.

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

.venv/bin/python scroll-mask.py "$SAM_INPUT_ROOT"
```

The model configuration is a Hydra package key and must remain `configs/sam2.1/sam2.1_hiera_t.yaml`; it is not an absolute path.

`segment.py` labels every object it finds on the first frame (scroll, background, turntable, ArUco markers, dust), ordered darkest first, so the scroll is not necessarily label 1 and markers are included only when they happened to be found on that frame. `scroll-mask.py` identifies the scroll label in each acquisition directory from its behaviour across the rotation (present in every frame, never touching the image border, centroid nearly static while markers orbit the axis) and writes binary `ScrollMasks/<stem>_mask.png` files: 255 for the largest component of the scroll label, 0 elsewhere. It prints the chosen label, area and centroid motion for each directory, and reports and skips directories where the pick is ambiguous or the scroll appears split across labels; inspect those before rerunning with `--force`. A small object that fills less than 1% of the frame is rejected as a candidate by the default `--min-area 0.01` ("no candidate label"); rerun with a lower value, e.g. `--min-area 0.003`.

Captures delivered as camera JPEGs, without RAW files, skip `raw2jpg.py`: `find_blocks.py link` hard-links each JPEG to `JPGEnhanced/<stem>.jpg` directly. Such JPEGs can be much brighter than the converted RAWs and may be zero-padded at the end by the card, which is harmless.

The markers do not need to be in the reconstruction masks. The scaler detects them in the photographs directly, and when a scroll is reoriented between blocks, markers that are fixed to the turntable are inconsistent with the scroll across orientations.

For every acquisition directory, verify that:

1. Every RAW file has one readable JPEG with matching dimensions and transferred EXIF metadata.
2. Every JPEG has one readable, same-sized `Masks/<stem>_mask.png` file with non-empty foreground.
3. `scroll-mask.py` reports `ok` with a plausible scroll area, and every JPEG has one same-sized `ScrollMasks/<stem>_mask.png` file with non-empty foreground.
4. Overlays of the `ScrollMasks/` files on their JPEGs cover the scroll and nothing else, for a sample of frames from every directory.

[`photogrammetry-tools/mask_sheet.py`](../photogrammetry-tools/) checks item 3's mask files and writes the overlay sheet for item 4.

`cc-fix.py` and `mask-applier.py` predate this step. They assume the scroll is label 1 (and the ruler label 2), which holds for earlier ruler-calibrated captures but not for turntable captures with ArUco markers, so do not use them on such captures.

#### Preparing the PGS-Recon input

PGS-Recon takes one flat directory of images. OpenMVG reads the mask for `<stem>.jpg` from `<stem>_mask.png` in the same directory, which must be the same size as the image, and ignores features wherever the mask is 0. The OpenMVS export then carries the masks into densification, where `--mask-value 0` ignores the same pixels. The binary `ScrollMasks/` files already follow this convention, so the input directory is every `JPGEnhanced/` image of the scroll with its `ScrollMasks/` mask beside it.

[`photogrammetry-tools/prepare_input.py`](../photogrammetry-tools/prepare_input.py) `WORK ID` builds this directory for `WORK/ID/ID/<blocks>/`, together with the scale include lists and the per-orientation inputs of the split route, and refuses to run if a mask is missing. By hand: hard links avoid copying the photographs (the input directory must be on the same filesystem as the masking tree). Image stems must be unique across acquisition directories, which holds for the camera's file numbering:

```bash
mkdir "$PGS_PHOTOS"
for d in "$SAM_INPUT_ROOT"/*/; do
  for jpg in "$d"JPGEnhanced/*.jpg; do
    stem=$(basename "$jpg" .jpg)
    ln "$jpg" "$PGS_PHOTOS/$stem.jpg"
    ln "$d/ScrollMasks/${stem}_mask.png" "$PGS_PHOTOS/${stem}_mask.png"
  done
done
```

Confirm that the directory holds the expected number of JPEGs, one `_mask.png` per JPEG and nothing else. Include every acquisition directory of the scroll, across orientations: scroll-only masks are what allow the orientations to be reconstructed jointly. The scaler's `--include-from` lists later restrict marker detection to selected blocks without removing any images from the scene.

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

[`photogrammetry-tools/sfm_review.py`](../photogrammetry-tools/) reports registration and residuals per block, the verified image pairs linking orientations (an orientation with none is left out of the reconstruction; see [Orientations that do not link](#orientations-that-do-not-link)), and the camera circles and axes.

It's important to use ply as output file type, because many readers downstream that work on other format used to corrupt the input mesh while parsing.

If the sparse reconstruction is not acceptable, preserve it and change one parameter family at a time:

1. First correct missing or invalid images, mask-label mistakes, inconsistent acquisition grouping, camera metadata and obvious turntable movement.
2. Compare `sfm_data.bin` with `sfm_data_structured.bin`. Robustification is the baseline, but it can occasionally remove useful structure; preserve a no-robust branch as a diagnostic if the robustified scene is visibly worse.
3. If the inputs are valid but matching is too sparse, retry ratio `0.8`. A higher ratio accepts more ambiguous matches, so inspect false connections as well as registration coverage. Relax it further only in small, separately recorded increments when the added matches improve real coverage without introducing false geometry.
4. If `HIGH` remains insufficient, retry with `ULTRA`. Reduce `--threads` if feature extraction approaches the host's memory limit. Use `NORMAL` only when a faster, lower-detail diagnostic is useful.
5. Change the descriptor family, matching method or geometric model only as an independently named experiment only after the simpler changes above have failed.

##### 2. Estimate and review physical scale

The scaler detects the printed ArUco markers, triangulates their corners and maps the accepted SfM scene into the physical unit used for `--marker-size`. Scale only the accepted robustified scene, and apply scale before OpenMVS.

The markers are fixed in the turntable frame, whereas a deliberately reoriented or accidentally displaced object is not. Use the acquisition log, annotation images (or the metadata-derived blocks described under [After acquisition](#after-acquisition)) and camera trajectories to create basename-only include lists for blocks in which the object orientation remained fixed. Combine camera heights only when they belong to the same fixed orientation. Do not infer blocks from historical filename counts.

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

[`photogrammetry-tools/scale_review.py`](../photogrammetry-tools/) tabulates the scale, triangulated corners, marker side lengths, planarity and marker ring radius of every candidate.

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

After the export to OpenMVS, `<output>/mvs/undistorted_images/` must contain one `<stem>.mask.png` for every image. Densification reports a mask that fails to load only at debug log level and then densifies that image unmasked, so check the count rather than relying on the log.

Require finite vertices and valid faces, nonzero extents, one dominant face-connected component, a plausible silhouette and object-aligned dimensions consistent with independent measurements. A case-generation mesh must also be watertight and free of boundary edges, non-manifold edges, orientation conflicts and degenerate faces. Inspect canonical renders for holes, bridges, collapsed regions and disconnected debris. [`photogrammetry-tools/mesh_qa.py`](../photogrammetry-tools/) checks the mask count, the topology and the principal-axis extents, and writes the renders.

If OpenMVS fails, retain the completed intermediate files and diagnose stages in increasing order of geometric impact:

1. If texturing fails after a valid refined PLY was produced, keep the untextured mesh for geometry and case-generation QA and diagnose `TextureMesh` separately. The PGS-Recon CLI always invokes texturing; it has no top-level disable-texturing flag.
2. If refinement fails or makes the geometry worse, retry as a separately named branch with `--no-mvs-refine`.
3. If densification fails, retry without `--mvs-densify` only as a separately validated branch; mesh reconstruction is always required.
4. If you have memory problems, try increasing the `--refine-resolution-level`.

Case generation does not intrinsically require color texture, but every omitted stage changes the retained artifacts and must be recorded. PLY is the required interchange format for topology QA.

In MeshLab, use `Filters -> Quality Measures and Computations -> Compute Topological Measures`. Alternatively, use a PLY-native topology checker and retain its exact version, command and machine-readable report with the run. Inspect the original PLY indices because some visualization readers duplicate vertices at texture seams and can falsely report disconnected components. Since the world axes are arbitrary, compare physical size with PCA-aligned or otherwise object-aligned extents rather than only the raw axis-aligned bounding box.


##### Orientations that do not link

A small or flat object can come out of SfM with only one orientation posed: after flipping, the cameras see mostly the other face, and `sfm_review.py` reports 0 verified pairs between the orientations. First try the remedies listed under step 1, and check the block grouping: blocks that the joint SfM posed together with the other orientation may belong to it (a block filed under the wrong orientation makes the split halves overlap less and the join ambiguous). If the orientations still do not link, reconstruct each one on its own and join the two halves on the rim that both saw. For that object, this route branches off after the step 1 review and replaces steps 2 and 3. The method was rehearsed on PHerc. 0525, whose joint reconstruction did link: the split result was within 0.11 mm of it in thickness.

1. **Reconstruct each half.** Give each orientation its own PGS-Recon input (hard links to its photographs and masks) and include lists per camera height, then run SfM, one scaler candidate per height and OpenMVS for each: [`photogrammetry-tools/split_recon.sh`](../photogrammetry-tools/split_recon.sh) `WORK ID ...`. Export each scaled scene to JSON (`openMVG_main_ConvertSfM_DataFormat`).
2. **Measure each half's scale.** pgs-global-scaler triangulates marker corners with a 0.1 px RANSAC threshold (`pgs-recon/dependencies/utilities/src/global_scaler.cpp`). On small objects the lens distortion at the image edges is weakly constrained, so few sightings pass. Its candidates then disagree by several percent, or fail and write the scene at scale 1. [`marker_scale.py`](../photogrammetry-tools/marker_scale.py) re-detects the markers, refits the lens distortion from them (SfM's own radial polynomial is fitted only where the object is, and on small objects it is useless at the image edges, where the markers are) and triangulates them at 1.5 px. It reports every block and picks the median factor of the blocks with complete markers and a consistent ring radius (Naples turntable: 99.5-100.8 mm); blocks of one scene agree to 0.3-0.7%. It applies equally to a joint scene, where its factor can replace the scaler's.
3. **Align the halves on the rim** with [`split_align.py align A.ply B.ply OUT --scale-a KA --scale-b KB`](../photogrammetry-tools/split_align.py) (the two `scene_dense.ply`).
   - It tries the sign flips of the principal axes, then runs point-to-plane ICP on point pairs with matching normals only, so that one face is never fitted onto the other.
   - Require one hypothesis to score clearly above the rest and an rms below 0.1 mm.
   - The reported rim scale (B relative to A, not applied) should be within about 0.5% of 1. It is an independent check on the two marker scales.
4. **Check the thickness against the photographs.** The rim fixes the object's length and width well, but its thickness only weakly. `split_align.py silhouette OUT A.json B.json MASKS_A MASKS_B` projects the joined cloud into the photographs of both orientations at several offsets of B along the thin axis, and fits the offset at which the outlines best match the masks.
   - Read only the position of the optimum: the splatted cloud overshoots the masks by a roughly constant 0.3 mm.
   - Accept the alignment if the optimum is within ±0.2 mm.
   - A larger positive optimum is partly a bias of the check on bright captures: the soft contact shadow under the object enters the masks along its resting edge, so every orientation looks thicker than it is; on such captures the optimum can reach about +1 mm. Merge and reconstruct both the unshifted and the shifted variant (step 5) and compare them in step 6. If the evidence still disagrees, use the thicker variant, which gives a looser case rather than a tight one.
5. **Merge and reconstruct once.** `split_align.py merge OUT A.json B.json OUT/sfm_data_merged.json [--thin-shift MM]` writes one OpenMVG scene holding both halves at their marker scales. [`merged_openmvs.sh`](../photogrammetry-tools/merged_openmvs.sh) `WORK ID ...` converts it to binary, which validates it, and runs OpenMVS on all photographs. This gives one watertight mesh rather than two halves to stitch.
6. **Review** with `split_align.py check OUT OUT/sfm_data_merged.json MESH.ply PGS_INPUT --sheet BLOCKS sheet.jpg` and `mesh_qa.py`.
   - Landmark reprojection must match between the two orientations, about 0.65 px median.
   - The mesh must reproduce both halves' main faces to within a few hundredths of a millimetre.
   - The mesh outline must sit equally close to the masks in both orientations (about 0.1 mm inside them).
   - Each half sees a little of the other face (A's points on the +thin face, B's on the -thin face). With the right thickness these sit on the mesh; with the halves too far apart they sit inside it by about the excess. This is the most direct measure of the thickness; the rim agreement and the outline fit are weaker and can point the other way.
   - A joint reconstruction of either side can serve as a reference: `split_align.py reference OUT A.json REF_DENSE.ply REF_SCENE.json [--scene-b B.json]` (`--scene-b` when it shares cameras with B only). Check that it really covers both faces: a joint scene that saw one face only closes the other with a mirrored copy of it.

Keep `alignment.json`, `merge.json`, the logs and every review output with the run.

##### Unattended batch

[`photogrammetry-tools/batch/batch_recon.sh`](../photogrammetry-tools/batch/batch_recon.sh) `WORK ID ...` runs steps 1-3, or the split route above, for several objects in one sudo session, after masking and [`prepare_input.py`](../photogrammetry-tools/prepare_input.py) `WORK ID`. It runs every joint SfM first, then per object either the joint route (marker scale with `marker_scale.py`, the scene multiplied into mm, OpenMVS) or, when an orientation was not posed, `split_recon.sh`, the automatic join `batch/auto_split.sh` and `merged_openmvs.sh`. The join applies the rules of [`batch/split_gate.py`](../photogrammetry-tools/batch/split_gate.py) and stops before the merged OpenMVS run when the flip is ambiguous, the rim fit or rim scale is poor or the silhouette optimum is out of range; an accepted optimum above +0.3 mm is marked for the two-variant review of step 4. Completed steps are skipped on a rerun. Review every object afterwards as for a manual run.

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

By default the case axis is the axis of the smallest enclosing cylinder. For a squashed scroll this can run across the flattened cross-section instead of along the umbilicus. To set the axis, add `<input>/<scroll-number>/<scroll-number>-axis.txt` holding either a direction (3 numbers) or two points on the umbilicus (6 numbers), in the PLY's coordinates. Only the direction is used: the case is still centred on the smallest circle enclosing the scroll about that axis, so the umbilicus may sit a few millimetres off the case axis. Record the file with the run.

To pick the two points, open the textured PLY in MeshLab (its layering shows the umbilicus best; its coordinates match the untextured mesh), choose `Edit -> Pick Points`, click where the umbilicus emerges on each end face and read the coordinates from the dialog, or save them as a `.pp` file. Check the result by viewing the scroll along the axis: the end-face layering should be concentric about it.

The top cap is engraved with `PHerc<padded number>` and `v3` by default. For an object without a number, or to engrave something else, add `<input>/<name>/<name>-label.txt` holding one or two lines (e.g. `PHerc. s.n.` and `cass. 94`); a folder name that is not a number is used unpadded for the output names.

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

The generator uses multiple workers and may finish its parent process even if a worker failed. A zero exit status is therefore insufficient. Require the padded identifier's three expected STLs, a populated `scroll_summary.csv` row, readable finite geometry and sensible assembly renders before accepting the kit. [`photogrammetry-tools/scrollcase_qa.py`](../photogrammetry-tools/) checks these, measures the clearance between the scroll and each half, and writes section and assembly renders.

#### Scan footprint

`scripts/xy_outline.py` projects each scroll along the case axis (z) and outlines the union of the x,y coordinates of its interior:

```bash
.venv/bin/python scripts/xy_outline.py "$SCROLLCASE_OUTPUT" --output /path/to/xy-outlines
```

Inputs may be the generator's output directory (searched for `*_scroll.stl`) or individual scroll STLs. The coordinates are those of the case frame: z along the case axis, origin on the axis at the base of the scroll, and the origin at the centre of the smallest circle enclosing the scroll's projection (so a lopsided fragment can look off-centre). +y is the side of the `_right` half, marked by the alignment arrow engraved on the bottom cap. For each scroll it writes `<padded-scroll-number>_xy_outline.csv`, the points of a closed counter-clockwise polyline in order, at most 1 mm apart (`x_mm,y_mm`; the script stops if a projection falls into several pieces), and `<padded-scroll-number>_xy_outline.png` showing the outline over the mesh footprint coloured by height; `xy_outlines_overview.png` shows all of them. The projection is rasterised on a 0.5 mm grid, so the outline is within about 0.5 mm of the true footprint; use `--margin` to grow it outwards, and `--cell` and `--step` to change the grid and point spacing.

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