---
title: "Community Projects"
hide_title: true
---

<head>
  <html data-theme="dark" />

  <meta
    name="description"
    content="Open-source tools built by the Vesuvius Challenge community for data access, segmentation, and ink detection to help read the buried Herculaneum scrolls."
  />

  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://scrollprize.org" />
  <meta property="og:title" content="Vesuvius Challenge" />
  <meta
    property="og:description"
    content="Open-source tools built by the Vesuvius Challenge community for data access, segmentation, and ink detection to help read the buried Herculaneum scrolls."
  />
  <meta
    property="og:image"
    content="https://scrollprize.org/img/social/opengraph.jpg"
  />

  <meta property="twitter:card" content="summary_large_image" />
  <meta property="twitter:url" content="https://scrollprize.org" />
  <meta property="twitter:title" content="Vesuvius Challenge" />
  <meta
    property="twitter:description"
    content="Open-source tools built by the Vesuvius Challenge community for data access, segmentation, and ink detection to help read the buried Herculaneum scrolls."
  />
  <meta
    property="twitter:image"
    content="https://scrollprize.org/img/social/opengraph.jpg"
  />
</head>

# 📜 Awesome Scroll Tools [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

Here are all the awesome awarded open source contributions from our community that will allow us to read the scrolls! 📚✨

Contributions are divided into four categories: _Data access/visualization_, _Segmentation_, _Ink Detection_, and _Other_.

Every category is subdivided in classes: 🌟 _Highlighted_ (for popular contributions), ⚙️ _Tools_, 📦 _Materials_, 📝 _Reports_, and 📊 _Visualization_.

Some highlighted contributions are added to this repository as submodules.

We keep this repository updated as much as we can, but research moves _fast_! 🏃💨

For state-of-the-art updates join our [Discord server](https://discord.com/invite/uTfNwwecCQ) 💬⏰

## 📊 Data access/visualization

### 🌟 Highlighted

- [vesuvius](https://github.com/scrollprize/vesuvius): Python library for accessing Vesuvius Challenge data. Allows direct access to scroll data without managing download scripts or storing terabytes of CT scans locally.

- [VC3D](https://github.com/ScrollPrize/villa/tree/main/volume-cartographer): the team’s interactive tool for viewing, segmenting, and virtually unwrapping scroll volumes.

- [Segment browser](https://github.com/jrudolph/vesuvius-browser) is a web-based tool to browse layers and open source ink detection results of all released segments. By Johannes Rudolph

### 🛠️ Tools

- [vesuvius-c](https://github.com/ScrollPrize/villa/tree/main/vesuvius-c): C library for accessing Vesuvius Challenge data (deprecated — prefer the [`vesuvius`](https://github.com/scrollprize/vesuvius) Python library or VC3D).

- [vesuvius-gui](https://github.com/jrudolph/vesuvius-gui) is a single binary GUI to render volumes and segments on-the-fly. By Johannes Rudolph

- [vesuvius-phalanx](https://github.com/mvrcii/phalanx): Python library / CLI for accessing Vesuvius data. Allows flexible access to volume and fragment scroll data. By Marcel Roth

- [llfio-chunkloader](https://github.com/climbmax123/LLFIOCunkloadingTestingAndBenching): A method to access data in chunks of (x,y,z) that is much faster and more compute-efficient than Zarr. (Written in C++ but it is possible to integrate in Python).

- [preprocessed-data](https://github.com/usc-caisplusplus/scroll-data-preprocessing): Data preprocessing code and a fully processed version of the dataset in .zarr format to allow for faster training of ink detection models. 
- [scroll-data-audit](https://github.com/Bullo27/scroll-data-audit) by Matteo Bulloni. Integrity auditor for the open-data: reconciles the catalog (`metadata.json`) against the actual Zarr arrays, filenames and scan metadata, and verifies multiscale pyramid value-correctness. Reported a Scroll 5 (PHerc0172) catalog shape error ([#1211](https://github.com/ScrollPrize/villa/issues/1211)) and certified the rest of the open-data consistent.

- [vesuvius-repro](https://github.com/TAUIL-Abd-Elilah/vesuvius-repro): Regional reproducibility spot-checks of one selected 256³ region (central 128³ scored) from each of 41 m7 artifacts across 36 scrolls. Forty TTA-off checks match at Dice 0.9983–1.0000; PHerc. Paris 4 matches at 0.9999 with TTA on. The audit exposed previously missing configuration provenance; [#1253](https://github.com/ScrollPrize/villa/pull/1253) now records TTA, and maintainers backfilled existing artifacts. Also resolves each artifact's CT level. By TAUIL Abd Elilah

- [vesuvius-catalog](https://github.com/Schurkai/vesuvius-catalog): scriptable catalog CLI/library for the open-data bucket - answers which samples have segments, ink outputs or surface predictions at which resolutions, resolves S3/HTTPS data URLs (JSON/CSV output for scripting), and includes working openers for the bucket's OME-Zarr v2 stores under zarr-python 3.
- [tifxyz-audit](https://github.com/jaideepsaipadhi/tifxyz-audit) by Jaideep Sai Padhi. Cross-scan tifxyz mesh consistency audit: reconciles a segment's meshes across its different source scans via the catalog's `transform.json` chains -- complements scroll-data-audit's volume-side reconciliation with a mesh-side check. 112/112 cross-scan pairs verified consistent across PHerc1667/PHerc0172/PHerc0332; found `area_vx2` absent from 100% of meshes built on two specific source scans ([#1468](https://github.com/ScrollPrize/villa/issues/1468)).

## Segmentation

### 🌟 Highlighted

- [Khartes](https://github.com/KhartesViewer/khartes) by Chuck is a tool to manually create and visualize segment meshes, while also visualizing a preview of the rendered segment.

### 🛠️ Tools

- [Volume Cartographer](https://github.com/educelab/volume-cartographer): the OG virtual unwrapping toolkit. Includes a graphical interface to annotate scroll segments. First built by [EduceLab](https://educelab.engr.uky.edu/); an [active fork](https://github.com/spacegaier/volume-cartographer) by Philip Allgaier contains many community contributions and is currently used by the segmentation team.
  
- [Thaumato Anakalyptor](https://github.com/schillij95/ThaumatoAnakalyptor/tree/main) is an automatic tool that combines classical methods such as threshold gradient operator based edge detectors and Deep Learning based instance segmentation of point clouds to detect, merge and render segments. It was built by Julian Schilliger (part of Grand Prize winning submission).

- [Fast Segment Rendering](https://github.com/schillij95/ThaumatoAnakalyptor/blob/main/ThaumatoAnakalyptor/sheet_to_mesh.py) by Julian Schilliger. Fast rendering of segments with GPU acceleration. Capable of saving the surface volume to multiple file formats.

    - [CPU rendering](https://github.com/schillij95/ThaumatoAnakalyptor/commit/bcd382a0ef59b2a8566ec62a474479ea9d1bb8c2) by Julian Schilliger and Giorgio Angelotti

- [Volumetric Vesuvius Labelling](https://github.com/JamesDarby345/Volumetric_Vesuvius_Labelling) by James Darby. Provide custom tooling the [napari](https://napari.org/stable/) 3d viewer that will help manually annotate volumetric masks of the scrolls to train ML models for 3D segmentation.

- [Autosegmentation preprocessing pipeline](https://github.com/giorgioangel/vesuvius_autoseg_preprocess) (work in progress) collection of scripts to pre-process volumes for autosegmentation. By Giorgio Angelotti

- [Segment2Voxel](https://github.com/giorgioangel/vesuvius-segment2voxel) by Giorgio Angelotti. Tool to create 1-voxel thick volumetric segment labels starting from mesh .obj files.

- [Volumetric Instance Labels to obj](https://github.com/JamesDarby345/Volumetric_Instance_to_Mesh/tree/main) by James Darby. Tools to create .obj mesh files from volumetric instance labels.

- [Hraun](https://github.com/SuperOptimizer/Hraun) is a collection of python tools for handling volumetric scroll data by Forrest McDonald.

- [windcheck](https://github.com/joe-carr-data/windcheck) by Josep Carreras. Finds where a traced surface passes through itself, from mesh geometry alone, and emits transverse-clean `tifxyz` outputs plus a VC3D overlay.

- [Scroll compression and masking](https://github.com/OliverDaubney/vesuvius_basic_compression) by Olivier Daubney. Script to compress and mask scroll data, greatly reducing storage requirements!

- [Mesh merging](https://github.com/schillij95/ThaumatoAnakalyptor/blob/main/ThaumatoAnakalyptor/mesh_merger.py) by Julian Schilliger. Merges multiple overlapping meshes into one continuous mesh. Flattening not included.

    - [Mesh merging prototype](https://gist.github.com/giorgioangel/b4cc56a5514335a2947adb058af2982b) by Giorgio Angelotti. Different attempt to merge existing mesh of segments by projecting them in 2D and retriangulating in the plane.

- [Meshing and chunking](https://discord.com/channels/1079907749569237093/1232307086952501313) by Santiago Pelufo

- [Volumetric segmentation model with labels](https://github.com/tspersonalgithub/march_2024_progress_submission), deep learning 3D model to separate papyrus from air, by Tim Skinner

- [Superpixels and cells](https://discord.com/channels/1079907749569237093/1221902373887279226) by Santiago Pelufo

- [Segment Flattening](https://github.com/schillij95/ThaumatoAnakalyptor/blob/main/ThaumatoAnakalyptor/slim_uv.py) by Julian Schilliger and Giorgio Angelotti. Improved flattening of scroll segments.

    - [Slim-Flatboi](https://github.com/giorgioangel/slim-flatboi) previous implementation of the SLIM algorithm with minimization of isometric distortion to flatten scroll segments. Later included in ThaumatoAnakalyptor. By Giorgio Angelotti.

- [Single Sheet Segmentation attempt](https://discord.com/channels/1079907749569237093/1179216516697296906/1179216516697296906) by Brett Olsen

- [vesuvius-blender](https://github.com/spelufo/vesuvius-blender) by Santiago Pelufo. Explore the X-ray scans in Blender.

- [vesuvius-build](https://github.com/spelufo/vesuvius-build/tree/main) by Santiago Pelufo. Scripts to build files for progressive loading of the data. Convert the tif stack to grid cells or to h5 format that can be used by Ilastik.

- [Volume Annotate](https://github.com/MosheLevy20/VolumeAnnotate) A partial reimplementation of Volume Cartographer in Python by Moshe Levy.

    - [VA-Sheet Tracer](https://github.com/teeohem96/VA-Sheet-Tracer) by Trevor, Tom, Babak and Boaz

- [vesuvius-image](https://github.com/caethan/vesuvius_image) by Brett Olsen. Tool for storing and viewing data, including efficient Zarr loading of stack of tif images later included in Khartes.

- [Quick Segment](https://github.com/educelab/quick-segment) Created by EduceLab for annotating a large air gap in Scroll 1, and then projecting from that gap to either side to create two large segments, colloquially referred to as the “Monster Segment”. Hasn’t been used for more segmentation, since it was the only large air gap we could find.

- [scrollreading](https://github.com/WillStevens/scrollreading) by Will Stevens. Experiments with using algorithms based on flood-fill to extract non-intersecting surfaces from scrolls.

- [VC with OME-Zarr & more](https://github.com/hendrikschilling/volume-cartographer) by Hendrik Schilling:
    - fast interactive OME-Zarr access and live slicing & flattening [thread](https://discord.com/channels/1079907749569237093/1286341523570688121)
    - instant flattening from VC segments without meshing (10s for one slice) [thread](https://discord.com/channels/1079907749569237093/1289946915269509251)
    - segment surface refinement (also works on obj segments) [thread](https://discord.com/channels/1079907749569237093/1290364437836075231)
    - fiber based segmentation efforts using an optimizing physics inspired surface meshing approach based on ceres-solver [thread](https://discord.com/channels/1079907749569237093/1301139262422646926)
    - non-destructive large scale interactive segment viewing and editing [thread](https://discord.com/channels/1079907749569237093/1294185795221065802)
    - automatic patch generation pipeline: vc_grow_seg_from_seed, vc_render_tifxyz, vc_tifxyz2obj: [thread](https://discord.com/channels/1079907749569237093/1312490723001499808)
    - segment tagging, segment masking, POIs, segment filters (all/filter by focus point/filter by POIs), display intersections scaling to thousands of segments [message](https://discord.com/channels/1079907749569237093/1286341523570688121/1312537855846907974)
    - low memory tiled rendering to enable GP-sized an full scroll rendering https://github.com/hendrikschilling/volume-cartographer/blob/dev-zarr/apps/src/vc_render_tifxyz.cpp
    - large segment tracing based on patch consensus: vc_grow_seg_from_segments, as documented in the [FASP submission](https://github.com/hendrikschilling/FASP?tab=readme-ov-file#vc_grow_seg_from_segments)
    - consistent winding number estimation by winding number diffusion: [vc_tifxyz_winding](https://github.com/hendrikschilling/FASP?tab=readme-ov-file#51-winding-number-assignment)
    - segment fusion & inpainting: [vc_fill_quadmesh](https://github.com/hendrikschilling/FASP?tab=readme-ov-file#vc_fill_quadmesh)

- [fast and low memory inference for the GP ink detection](https://discord.com/channels/1079907749569237093/1315006782191570975) 1/5 the memory consumption and 20x the speed compared to the baseline GP ink detection for large segments to allow GP and full scroll size ink detection and fast preview.

- [vesuvius-render](https://github.com/jrudolph/vesuvius-gui?tab=readme-ov-file#vesuvius-render) by Johannes Rudolph:
    - Fast self-contained CPU-based rendering of segments from obj files downloading data on-the-fly.

- [segmata](https://github.com/sgoutteb/segmata) by Stephane Gouttebroze:
    - Improve the segmentation process by sharpening the layers rendering, this is based on optimizing the layer 32, a further objective is to link this optimization on a inference loop (optimizing on the detected ink instead of only layers)

- [Synthetic instance labels and volume generation](https://lcparker/synthetic-pages) by lcparker
    - Generate artificial 3D volumes with corresponding instance labels for use in pretraining instance segmentation networks

- [Mask3D for instance segmentation on scroll volumes](https://lcparker/Mask3D) by lcparker
    - SOTA instance segmentation network, configured to work with scroll volumes
    - [Effects of pretraining on synthetically generated data](https://github.com/lcparker/pretraining-advantage), plus pretrained and finetuned weights for the Mask3D network
- [Affinity Prediction with Unet](https://discordapp.com/channels/1079907749569237093/1407379961417039953) by Ayush Mishra
    - Unet trained on affinity labels using pytorch_connectomics and experiments with watershed
- [Gabor Filter for surface prediction](https://www.kaggle.com/code/bluetriad/scroll4-gaborfilters/notebook?scriptVersionId=265957590) by Ayush Mishra
  
- [ScrollFiesta -- virtual meshing & unwrapping for the Herculaneum papyri](https://github.com/Hob3rMallow/scrollfiesta_public) by HariSeldon and friends - now with parameterization and GPU acceleration!

    - [GPU-accelerated MLS projection for ScrollFiesta](https://github.com/pscamillo/scrollfiesta_public/blob/cuda-mls/BENCHMARKS.md) by pscamillo — OpenMP + CUDA FP32 acceleration, byte-identical, ~6x throughput. (Note: GPU acceleration through CubeCL is now part of the main ScrollFiesta repository)

- [Scroll-specific augmentations](https://github.com/ScrollPrize/villa/pull/997) by pscamillo. GPU-native training augmentations modeling real scroll/CT distortions: Squeeze ([#997](https://github.com/ScrollPrize/villa/pull/997)), Decohesion + Warp ([#999](https://github.com/ScrollPrize/villa/pull/999)), Ring ([#1020](https://github.com/ScrollPrize/villa/pull/1020)), Streak ([#1021](https://github.com/ScrollPrize/villa/pull/1021)), Warp edge-padding fix ([#1025](https://github.com/ScrollPrize/villa/pull/1025)) (#201), each validated with ablation + benchmark.

- [winding-ruler](https://github.com/pscamillo/winding-ruler) by pscamillo. Measures winding evidence for the spiral fit: where human annotations matter, why generation fails at the published resolution, and a collection-wide winding-pitch atlas (36 scrolls). Includes [qa_holescan](https://github.com/pscamillo/winding-ruler/blob/main/concordance/qa_holescan.py), a content-level detector for silent z-slice loss in predict3d output ([#1183](https://github.com/ScrollPrize/villa/issues/1183)).
       

- [HIP/ROCm port of ScrollFiesta's MLS for AMD GPUs](https://github.com/altommo/scrollfiesta-mls-hip) by Alan Thompson (altommo) — clean-room HIP port of the MLS-midpoint kernel, validated on RX 9070 / gfx1201: ~16.8x kernel, ~5.8x multi-cube, topology-equivalent weld-safe mesh. ([upstream PR](https://github.com/Hob3rMallow/scrollfiesta_public/pull/4))
- [CubeCL port of ScrollFiesta's MLS (portable GPU: AMD/NVIDIA/WGPU)](https://github.com/altommo/scrollfiesta-mls-cubecl) by Alan Thompson (altommo) — Rust/CubeCL reimplementation of the same MLS kernel from a single source; validated on RX 9070 / gfx1201 against the HIP port, passing the 0.25-voxel weld-safety gate and 5-pass strict parity vs clean HIP.

- [vesuvius-automesh](https://github.com/spencerdavis-tx/vesuvius-automesh) by spencerdavis-tx. Fully automated, QC-gated surface harvest: masks the public surface predictions against the CT, seed-sweeps the villa tracer, and accepts output per 25 mm window under a two-part quality gate plus an independent topology check - 279 cm2 of verified Scroll 3 surface with zero manual annotation, CPU only. Ships per-window QC records for all 157 windows (passes and failures) as a labeled set of tracer failure modes.

- [Herculaneum Scroll Tools](https://github.com/axiosdevs/herculaneum-scroll-tools) by axiosdevs — CT-consistency QA for the published m7 surface predictions: voxel-exact phantom fractions measured for all 36 samples incl. all 13 grand-prize scrolls (43.3% of positives sit outside the masked CT), with a one-pass `clean` mode ([villa#1114](https://github.com/ScrollPrize/villa/issues/1114)). `audit_ct_support` audits any prediction volume in two modes: a **zero-download** chunk triage that reads only stored zarr chunk keys (~10 s per scroll, no voxel transfer) and classifies every prediction chunk as CT-supported, inside the one-chunk blend margin, or beyond it; and an exact voxel-level survey over chunk-aligned slabs. Across the batch: 1,662,405 prediction chunks, 83.1% supported, 16.9% one-chunk halo, **0 beyond the margin in every sample** — consistent with the blend-margin mechanism and with no second source. Plus cross-scan registration (2023→2025 rescan, MAD 29 µm), a winding-constraint annotator/verifier in native spiral-input format, and dual-energy high-Z ink-candidate rendering.

- [tifxyz-repair](https://github.com/Nieuwlaar/tifxyz-repair) by Nieuwlaar. Validates and repairs `tifxyz` patch metadata against VC3D's exact loader semantics: detects the stale-bbox corruption of [villa#1272](https://github.com/ScrollPrize/villa/issues/1272) (106 of 4,922 verified PHercParis4 spiral-input patches affected, independently reproducing the issue's counts), rewrites corrected bboxes in place (atomic, `--dry-run`, backups) and ships ready-to-apply boxes for every flagged patch; full-corpus audits certify all 40,782 unverified patches and all 817 published segment meshes clean. Companion PR [#1285](https://github.com/ScrollPrize/villa/pull/1285) stops the corruption at the source.

- [winding-sync](https://github.com/abundantjoe/winding-sync) by Joseph Balmaceda. Generates relative winding constraints directly from CT via structure-tensor lamina orientation, then reconciles contradictions globally as L1 integer synchronization (totally unimodular LP, exact integer solutions). Complements winding-number diffusion over existing surfaces by working from the raw volume before a surface exists. Tolerates 15% gross measurement error before dropping below 95% accuracy, versus 2% for spanning-tree propagation. Absolute winding counts are not yet calibrated; see README.

- [spiral-fit-consumer-gpu](https://github.com/7jycwjmbfn-eng/spiral-fit-consumer-gpu) by Shuhan Yang runs the spiral fitter in this repo on a 12 GB consumer GPU without the native VC extension. Drop-in replacements for the sparse CUDA cache, which otherwise raises before the first step, and for point-to-patch linking, where a measured 41-hour ETA drops to minutes. Both are checked for bitwise-identical output against the originals. Includes a scaling ladder and pool-tuning measurements from one gaming laptop.

- [TIFXYZ Doctor](https://github.com/aviad12g/tifxyz-doctor) by Aviad Cohen — deterministic QA and triage for TIFXYZ surface grids, with sparse overlays and reproducible corpus/reader checks. v0.2 adds an overlap-component-isolated benchmark on 709 official human-reviewed PHercParis4 `same_wrap` patches; its frozen cue localizes abrupt synthetic normal-offset steps while byte-identical null controls and the reported gradual-transition miss rate bound the claim.

- [spiralcheck](https://github.com/Nicodol/spiralcheck) by Nicolas Dolegieviez. Held-out evaluation for whole-scroll spiral fits: scores a finished run from its output meshes alone (CPU-only, no checkpoint, producer-agnostic) against verified patches withheld from that fit, and measures geometrically how much of the withheld evidence actually sits within touching distance of the fit's real inputs — on PHerc. Paris 4, 54.8% of a naive name-level split leaked that way, which no hash-level check can see. Also ships ground-truth-free winding-order checks around the umbilicus, a planted-defect matrix with computed null-control bounds, and `spiralcheck demo`, which runs the whole pipeline on a synthetic scroll with planted defects and needs no data.

### 📦 Materials

#### 🌟 Highlighted

- [Sheet instance annotation of cubes for Deep Learning models](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/seg-volumetric-labels/finished_cubes/) (work in progress)
    - [More cubes to annotate, help us!](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/seg-volumetric-labels/cubes/)

- [Denoised and contrast enhanced volumes](https://discord.com/channels/1079907749569237093/1249316301273436320), download [here](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_denoised_ce/), same path pattern for other scrolls.

#### Scroll Surface Predictions
- [Scroll 1, and 3 Surface Predictions](https://dl.ash2txt.org/community-uploads/bruniss/p2_submission/) by Sean Johnson
- [Scroll 4 Surface Predictions](https://dl.ash2txt.org/community-uploads/bruniss/Fiber-and-Surface-Models/Predictions/s4/) by Sean Johnson
- [Scroll Surface Prediction Repository and Writeup](https://github.com/bruniss/VC-Surface-Models) by Sean Johnson 

#### 📜 Segments
-[Large Autosegmentation of Scroll5](https://dl.ash2txt.org/community-uploads/bruniss/p2_submission/s5_initial_trace/) by Hendrik Schilling and Sean Johnson -- Unsupervised, many switches -- check readme.md

- [Scroll 2 segments](https://discord.com/channels/1079907749569237093/1079907750265499772/1245553260362858577) by Sean Johnson

- [New segments](https://discord.com/channels/1079907749569237093/1234969334535946303) by Sean Johnson

- [Large segments](http://dl.ash2txt.org/bruniss-uploads/) by Sean Johnson

- [Rescaled to 7.91um fragment surfaces and labels](https://dl.ash2txt.org/community-uploads/jrudolph/rescaled-fragments/) by Johannes Rudolph

#### 🏷️ Volumetric Labels

- [Instance segmentation labels](https://github.com/JamesDarby345/Vesuvius_3D_datasets) by James Darby

### 📝 Reports

- [Technical report on ThaumatoAnakalyptor](https://github.com/schillij95/ThaumatoAnakalyptor/blob/main/documentation/ThaumatoAnakalyptor___Technical_Report_and_Roadmap.pdf) by Julian Schilliger

- [Physical equalization of scrolls' brightness](https://github.com/giorgioangel/vesuvius_autoseg_preprocess/blob/main/equalize/Scroll_Equalizer.pdf) by Giorgio Angelotti

- [Volumetric segmentation architecture investigation](https://docs.google.com/document/d/1SX83Dhz5sJXHhSRbADcNxUmuH53BypLRny01rbizK8I/edit?usp=sharing) by James Darby

- [Instance segmentation experiments](https://discord.com/channels/1079907749569237093/1235042673899995176/1235042673899995176) by James Darby, Ryan Reszetnik, Liamo Pennimpede, Lucas Nelson

- [Probabilistic view on the offset for surface volume creation](https://discord.com/channels/1079907749569237093/1177617480366170162) by Giorgio Angelotti

- [Creating segments from intersecting horizontal and vertical fibers](https://gist.github.com/jrudolph/3e0ebbd6e731f794733c236a86ff39fb) by Johannes Rudolph

- [Phantom contamination audit of the published surface predictions](https://github.com/Schurkai/vesuvius-phantom-audit): chunk-listing-based measurement across all 36 samples with published m7 surface predictions (villa#1114) - every sample is affected, 16.9% of stored prediction chunks are certain phantoms; includes a voxel-exact-verified cleanup tool.

- [Surface geometry failure diagnostic](https://github.com/Jinhojeong/vesuvius-surface-geometry-diagnostic) by Jinho Jeong. Geometry-stratified analysis of where surface models fail in compressed regions (villa #191): compressed sheets are fused rather than missed across three architectures, the CT carries no intensity dip between contacting sheets, and neither the model's probability field nor local CT geometry supplies a boundary either. Includes oracle measurements showing the official topometrics blend does not respond to repairing these merges, a patch-mode eval entry point and a 200-patch ground-truth pool so any surface model or splitter can be scored the same way.

- [unmerge-cli](https://github.com/Jinhojeong/vesuvius-unmerge) by Jinho Jeong. Splits fused sheet instances by carrying identity through a contact from the resolved surroundings, using an anisotropic random-walk solve seeded from neighbouring instance labels. On raw predictions it refuses to split and flags the weld instead, since no boundary signal exists there. Applied to the published PHerc1218 instance labels it produced [pherc1218-topological-repair](https://www.kaggle.com/datasets/jhjeong0815/pherc1218-topological-repair), 54,377 repaired fusion sites in three tiers, listed as a companion dataset by [vesuvius-sheet-tools](https://github.com/IyanDopico/vesuvius-sheet-tools). That project independently cross-checked the flagged sites with its own intensity splitter and a matched spatial control, finding them 2.89x enriched for fused stacks over background.

### 📊 Visualization

- [Browser-based scroll viewer](https://discord.com/channels/1079907749569237093/1246129199304151052/1246129199304151052) by Yao Hsiao

- [wj-wt-ftt](https://github.com/tomhsiao1260/wj-wt-ftt) by Yao Hsiao and Dalufishe. Tool to view and annotate volumetric scrolls data.

- [Crackle Viewer](https://github.com/schillij95/Crackle-Viewer) is a tool to browse and annotate surface volumes of rendered segments, by Julian Schilliger

- [Point cloud extraction method comparer](https://github.com/giorgioangel/vesuvius-compare/) by Giorgio Angelotti. Tool to compare different point cloud extraction methods.

- [Pipeline Visualize](https://github.com/tomhsiao1260/pipeline-visualize) by Yao Hsiao. Tool to visualize the first steps of the Thaumato Anakalyptor pipeline.

- [Cell viewer and segmentation comparison](https://discord.com/channels/1079907749569237093/1162822294415097907/threads/1167722091781554290) by Yao Hsiao

- [Volume Viewer](https://github.com/tomhsiao1260/vc-whiteboard/tree/demo-3) Used by the segmentation team primarily to see which segments they have worked on already. Hosted [here](http://37.19.207.113:5174/). By Yao Hsiao
    - [Vesuvius Challenge Whiteboard](https://github.com/tomhsiao1260/vc-whiteboard/tree/dev) by Yao Hsiao and Dalufishe
 
- [Neuroglancer Mini](https://github.com/tomhsiao1260/neuroglancer-mini) A trimmed-down version of the Neuroglancer source code. By Yao Hsiao

- [Scroll Viewer](https://github.com/lukeboi/scroll-viewer) by Luke Farritor. A lightweight, extensible tool for viewing volumetric data, which runs in the browser, and is very fast.

- [Scroll Sleuth](https://github.com/Paul-G2/ScrollSleuth) by Paul Geiger. A web app that supports visual ink-searching in segment volumes via multiple display modes and segmentation tools.

- [Scroll Slab Viewer](https://github.com/Paul-G2/ScrollSlabViewer) by Paul Geiger. A 3D viewer tailored for the [Kaggle Surface Detection challenge](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection).

## Ink Detection

### 🏆 3D Ink Detection

#### 🌟 Highlighted

- [3D (volumetric) Ink detection model](https://github.com/ryanchesler/3d-ink-detection) by Ryan Chesler. Ink detection model that works on full scroll data in 3D, without segmentation nor flattening.
- [Volumetric Ink Detection for Scroll 1, 2, 3, 4](https://dl.ash2txt.org/community-uploads/bruniss/3d%20Ink%20/) by Sean Johnson

#### ⚙️ Tools

- [Large Scroll Model](https://github.com/ryanchesler/LSM/blob/main/README.md) is a 3D Unet pretrained on scroll data, by Ryan Chesler

- [UV predictions visualizer](https://gist.github.com/giorgioangel/6ae26b126f364dda751a10be0b90b36d) by Giorgio Angelotti. Script to quickly visualize the ink predictions output by Ryan Chesler's 3D model as a scatter pkot on segments. Needs the predictions Zarr for the full scroll.

- [Volumetric ink detection attempt](https://discord.com/channels/1079907749569237093/1204133327083147264) by Jorge Villaescusa

- [Inkalyzer](https://github.com/younader/Inkalyzer) by Youssef Nader. XAI package for Ink models to explain predictions and generate volumetric labels.

#### 📦 Materials

- [3D Ink labels](https://discord.com/channels/1079907749569237093/1079907750265499772/1223357870762889308) by Sean Johnsonn

- [3D Ink predictions](https://dl.ash2txt.org/community-uploads/ryan/) by Ryan Chesler. Predictions of 3D Ink models on full scrolls in Zarr format.

### 🖋️ Scroll segments-based Ink Detection

#### 🌟 Highlighted

- [Grand Prize Winner Ink Detection model](https://github.com/younader/Vesuvius-Grandprize-Winner) by Youssef Nader, Luke Farritor and Julian Schilliger

#### ⚙️ Tools

- [ScrollMAE](https://github.com/jgcarrasco/ScrollMAE) by Jorge García. Contains the necessary code to pretrain a 3D ResNet on unlabeled data and then finetune it to perform ink detection.

- [Unsupervised Ink Detection with DINO](https://github.com/jgcarrasco/dino-ink-detection) by Jorge García. Contains experiments related to detecting ink without labels, including a Colab notebook.

- [Vesuvius AutoResearch](https://github.com/mojomast/vesuvius-autoresearch) by mojomast. Autonomous, evidence-gated experiment search for Vesuvius ink-detection research, with reproducible configs, metric contracts, synthetic demo data, and promotion checks.

- [Vesuvius GP+](https://github.com/jaredlandau/Vesuvius-Grandprize-Winner-Plus) by Jared Landau. Updated version of the Grand Prize Ink Detection script with extra features.
  
- [Segment-to-segment label mapping](https://github.com/OliverDaubney/s2slabmap) by Oliver Daubney

- Runner Up Models, December, 2023
    - [Ink detection model](https://github.com/SQMah/Vesuvius-Grand-Prize-Submission/) by SQ Mah


    - [Ink detection model](https://github.com/lschlessinger1/vesuvius-grand-prize-submission) by Lou Schlessinger and Arefeh Sherafati

    - [Ink detection model](https://github.com/erdpx/vesuvius-grand-prize) by lian Rafael Dal Prá, Sean Johnson, Leonardo Scabini, Raí Fernando Dal Prá, João Vitor Brentigani Torezan, Daniel Baldin Franceschini, Bruno Pereira Kellm, Marcelo Soccol Gris, Odemir Martinez Bruno

- [Vesuvius Kintsugi](https://github.com/giorgioangel/vesuvius-kintsugi) is a tool to label floodfill surface volumes of rendered segments, by Giorgio Angelotti

- [Omit](https://onedrive.live.com/?authkey=%21ALfVTOHQOkbecQ0&id=D6F698278C30CB3E%212310&cid=D6F698278C30CB3E) is a pipeline that tries to detect ink with classical approaches (not deep learning) by Timo Meireman

- First Letters winning models, October 2023
    - [Ink detection model](https://github.com/lukeboi/scroll-first-letters) by Luke Farritor
    - [Ink detection model, 2nd place but more accurate](https://github.com/younader/Vesuvius-First-Letters) by Youssef Nader

- [Crackle Viewer](https://github.com/schillij95/Crackle-Viewer) is a tool to browse and annotate surface volumes of rendered segments, by Julian Schilliger

- [Fourth placed Kaggle model finetuning](https://github.com/lukeboi/scroll-fourth-second/blob/master/README.md) on scroll data by Luke Farritor

- [Scroll pretraining](https://github.com/younader/VesuviusPretraining) by Youssef Nader. Youssef’s original idea for pretraining on the scrolls and finetuning on the fragments, which led him to winning the First Letters Prize.

- [pre-trained DINOv2 models](https://github.com/SergeyPnev/dinov2-vesuvius) by Sergei Pnev. Self-supervised model pre-trained on scrolls 1-5 with predictions.

- [Ink detection validation harness](https://github.com/khj1222/vesuvius-challenge) by khj1222. The ink-detection tutorial trains with no held-out data, so improvements cannot be told apart from noise. This generates a `_validation_mask` for a labeled segment by holding out whole annotated regions (splitting by pixels cuts letters in half), scores predictions inside it (threshold sweep, DRD / pseudo-F-measure, per-region breakdown), sweeps checkpoints, and runs k-fold. Also includes a native-Windows walkthrough of the tutorial.

#### 📦 Materials

- [Scroll 1 Ink Labels](https://discord.com/channels/1079907749569237093/1223849912467460116). Nicola Bodill produced more accurate labels for ink detection based on the prediction of the Grand Prize winner model

- [Scroll 4 predictions](https://dl.ash2txt.org/community-uploads/luke/youssef_uploads/scroll_4/). Youssef Nader produced some predictions on Scroll 4 from his Grand prize winner model. No sure trace of ink yet

- [Ink detection masks](https://discord.com/channels/1079907749569237093/1177039383375912990/1177039383375912990). Anton Repushko shared some ink labels for Scroll 1, these labels were used by many participants for their final submission in December 2023.

- [Crackle labels on Scroll 1](https://dl.ash2txt.org/community-uploads/bruniss/) by Sean Johnson

- [Ink Generator]
(https://github.com/StewartSethA/VesuviusInkGenerator) by Seth Stewart, ink volume sample patches generated using gradient ascent

- [Ink feature visualization]
(https://github.com/StewartSethA/ink-seer) by Seth Stewart, ink feature vector clustering and visualization via t-SNE to distinguish ink from non-ink features

- [Scroll 5 Ink Labels](https://github.com/Bodillium/Herculaneum-Scroll-Labels) by Nicola Bodill. Early ink labels for Scroll 5.

#### 📝 Reports

- [Introduction to Ink Detection](https://medium.com/@jaredlandau/vesuvius-challenge-ink-detection-part-1-introduction-1cb125a56b21) by Jared Landau

- [Grand Prize Presentation](https://www.youtube.com/watch?v=F5ak1pRaqVo&ab_channel=VesuviusChallenge) by Youssef Nader and Julian Schilliger

- [First Ink on scroll 1](https://caseyhandmer.wordpress.com/2023/08/05/reading-ancient-scrolls/) by Casey Handmer

#### 📊 Visualization

- [Segment viewer](https://github.com/tomhsiao1260/segment-viewer). Used by the segmentation team primarily to see which segments they have worked on already. Hosted [here](http://37.19.207.113:5173/?mode=segment&segment=20230702185753) By Yao Hsiao and Dalufishe

### 📜 Fragment-based Ink Detection

#### 🌟 Highlighted 

- [Kaggle competition on Fragments](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/overview)
    - [1st place](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417496) by Ryan Chesler
    - [2nd place](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417255) by RTX2309
    - [3rd place](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417536) by wuyu
    - [4th place](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417779) by POSCO DX -- Heeyoung Ahn
    - [5th place](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417642) by Aksell
    - [6th place](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417274) by chumajin
    - [7th place](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417430) by OverthINKingSegmenter
    - [8th place](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417383) by Luck is all you need
    - [9th place](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417361) by still 1 fold, 2 net
    - [10th place](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417363) by Feng Qilong

- [2.5D fragment segmentation (ink detection) baseline](https://www.kaggle.com/code/tanakar/2-5d-segmentaion-baseline-inference) by Ryosuke Tanaka

- [Ink ID](https://github.com/educelab/ink-id) by Stephen Parsons

- [Iterative Labeling on fragments](https://discord.com/channels/1079907749569237093/1279263442913591349/1279263442913591349) by Youssef Nader. Applying the iterative labeling approach of the GP winning team to improve ink detection on fragments hidden layers.

- [ink detection with rescaled fragments](https://discordapp.com/channels/1079907749569237093/1400138226269814785) by Ayush Mishra

#### 📝 Reports

- [Kaggle Challenge top ink detection model analysis](https://github.com/ainatersol/Vesuvius-InkDetection/blob/main/additional_findings.md) by Ryan Chesler

- [Ink detection model resolution analysis](https://github.com/MIC-DKFZ/OverthINKingSegmenter/blob/master/vesuvius_followup_writeup.pdf) by Yannick Kirchhoff, Maximilian Rokuss and Benjamin Hamm

## Other

### ⚙️ Tools

- [Efficient Data Downloader](https://github.com/JamesDarby345/VesuviusDataDownload): scripts to efficiently download data with rclone, by James Darby

- [Improving scroll alignment with image registration](https://github.com/Paul-G2/VesuviusScrollAlignment) Scripts and a report showing how image registration can improve the alignment of scroll volumes scanned at different energies and resolutions, by Paul Geiger

### 📦 Materials

- [CT scanning campfire scrolls](https://dl.ash2txt.org/community-uploads/waynewaynehello/) Ahron Wayne replicated the carbonization process of a papyrus scroll and scanned it with his personal CT scanner

### 📝 Reports

- [Hard-Hearted Scrolls](https://uknowledge.uky.edu/cs_etds/138/), PhD Dissertation by Stephen Parsons

# Contributions
If you want to contribute and add any resource please submit a PR! 😊🚀
