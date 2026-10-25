---
title: "Tutorial: First Letters on a Mac"
sidebar_label: "First Letters on a Mac"
---

<head>
  <html data-theme="dark" />

  <meta
    name="description"
    content="Hunt for First Letters on an Apple-silicon Mac with no NVIDIA GPU: grow surfaces with GrowPatch, stream-render them from the open-data bucket, triage cheaply, and run ink detection on MPS."
  />

  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://scrollprize.org" />
  <meta property="og:title" content="Vesuvius Challenge" />
  <meta
    property="og:description"
    content="Hunt for First Letters on an Apple-silicon Mac with no NVIDIA GPU: grow surfaces with GrowPatch, stream-render them from the open-data bucket, triage cheaply, and run ink detection on MPS."
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
    content="Hunt for First Letters on an Apple-silicon Mac with no NVIDIA GPU: grow surfaces with GrowPatch, stream-render them from the open-data bucket, triage cheaply, and run ink detection on MPS."
  />
  <meta
    property="twitter:image"
    content="https://scrollprize.org/img/social/opengraph.jpg"
  />
</head>

*Last updated: October 25, 2026*

You can hunt for [First Letters](/prizes#first-letters-prizes) on an Apple-silicon laptop. You don't need an NVIDIA GPU, and you don't need to download a whole volume. This page walks through a workflow tested end to end on a 64 GB M5 Pro MacBook. The pipeline is:

**seed → grow a surface → stream-render it → triage → ink detection on MPS**

The scripts are at [antaloaalonso/mac-first-letters](https://github.com/antaloaalonso/mac-first-letters). Everything else is official tooling:
- the VC3D macOS release;
- `vesuvius` ink detection;
- the published `ink_9um` checkpoints;
- the published surface predictions and normal grids.

## 1. Set up

```bash
gh release download latest -R ScrollPrize/villa -p '*macos-arm64.zip' && unzip -d tools VC3D-*.zip
git clone --depth 1 https://github.com/ScrollPrize/villa.git
cd villa/vesuvius && uv sync --extra models-core   # ink detection without building the C++ bindings
hf download scrollprize/ink_9um hybrid_3d2d-seed42/step-075000.pth hybrid_3d2d-seed43/step-075000.pth --local-dir checkpoints/ink_9um
```

`uv sync --extra models` also builds the `volume-cartographer` Python bindings, which needs Ceres and Qt6. Ink detection doesn't use them, so on a Mac use `models-core` instead.

## 2. Grow surfaces where the sheets are clean

`vc_grow_seg_from_seed` runs on the CPU. It needs a scroll's published surface prediction and `normal-grids`, about 4.5 GB for a 2.4 µm scan, with every pyramid level present locally. It rejects the published `metadata.json` because that file has no `scan` key, so rename it and set `voxelsize` in the params JSON.

Seeds matter more than anything else here. A random seed often lands where sheets touch, and the patch then hops between windings in a crumpled "starburst". `vc_render_tifxyz` renders the same patch the same way, so it isn't a renderer artefact.

`seedmap.py` scores blocks of the surface prediction for thin, parallel, coherent sheets. `grow_seeds.sh` then grows 50-generation patches from the best-scoring seeds:
- on PHerc1203, 10/12 seeded patches were clean recto, against about 4/12 with random seeds;
- on PHerc0846A, 15/19 were clean.

<figure>
  <img src="/img/tutorials/mac-growpatch-starburst.webp" />
  <figcaption className="mt-0">A random-seed GrowPatch patch rendered with the official <code>vc_render_tifxyz</code> (left) and with the streaming renderer (right). Both show the same sheet-hopping starburst around the seed.</figcaption>
</figure>

## 3. Stream-render instead of downloading

`render.py` samples a tifxyz mesh along its normals. It fetches only the 128³ chunks of the open-data volume that the mesh touches, over plain HTTPS, and keeps them in a local cache trimmed to a size budget. Two options matter:
- `--frame-level` says which pyramid level the mesh coordinates are in;
- `--level` picks the level to sample.

Re-rendering PHerc0814 segment `46527` this way at level 2 matched the published surface volume with normalised cross-correlation 0.94 at zero shift. The published volumes use the opposite normal sign, so pass `--flip`.

The official `vc_render_tifxyz` from the same VC3D release can also stream a remote volume: pass the `https://` OME-Zarr URL as `-v`. Use whichever fits your setup. `render.py` is a small pure-Python alternative that is easy to modify.

<figure>
  <img src="/img/tutorials/mac-renderer-vs-official.webp" />
  <figcaption className="mt-0">PHerc0800 segment 20251029010146: published surface volume vs. the streaming render, centre layer (r = 0.968).</figcaption>
</figure>

`ink_9um` was trained on 2.4 µm scans reduced to 9.6 µm. Sampling pyramid level 2 directly gave predictions visually equivalent to sampling level 1 and mean-pooling 2×2×2, and it fetches 8× less data.

## 4. Triage at 1/8 the bandwidth

Before rendering a patch at full resolution, `triage.sh` renders all patches at level 3 with 13 layers and puts them in one contact sheet. Patches with crumpled or blank surfaces are easy to spot and discard at that size.

<figure>
  <img src="/img/tutorials/mac-triage-contact-sheet.webp" />
  <figcaption className="mt-0">Level-3 triage contact sheet for GrowPatch patches on PHerc1203.</figcaption>
</figure>

## 5. Ink detection on MPS

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -m vesuvius.ink_detection.inference.infer \
  segment.zarr checkpoints/ink_9um/hybrid_3d2d-seed42/step-075000.pth pred.tif \
  --ensemble-checkpoint checkpoints/ink_9um/hybrid_3d2d-seed43/step-075000.pth \
  --direction both --overlap 0.5 --blend-mode hann --batch-size 16 --no-compile
```

A 1320 × 2480 px, 31-layer segment takes about two minutes per model on an M5 Pro. Run both depth directions: ink should show in only one of them. Strokes that appear in both directions are almost always crack or tear edges. The rims of voids are the other common false positive, so mask them before you review.

**Check the pipeline on known text first.** The `ink_9um` labels include held-out validation regions on three segments:
- `pherc0139-w016`;
- `pherc0814-46527`;
- `pherc1667-w029`.

Render one of them with the steps above and compare the prediction with the published labels inside `<case>_validation_mask`. The mac-first-letters repo's `bench/` scripts do this end to end. Its `RESULTS.md` shows the streamed renders scoring within about 0.01 AUROC of the official surface volumes.

A quicker visual check is PHerc1667 `w018`: it should show rows of letters in the forward direction and noise in reverse. It is part of the training labels, so it confirms orientation and scale, not generalisation. If either check fails, look at `--flip` and the mesh frame level before you scan new surfaces.

## 6. Spiral fitting

`spiral-fitting` runs track-based fits on MPS with `FIT_SPIRAL_DEVICE=mps`. On a 1000-slice PHerc0846A band with 144k tracks, an M5 Pro reaches about 1.1 it/s (0.6 it/s while sharing the machine with other jobs). The same fit on CPU does 0.47 it/s. This needs the MPS fixes in ScrollPrize/villa PR "spiral-fitting: make weighted track sampling work on MPS". The fit holds about 4.5 GB between steps and peaks around 10 GB.

You can build a band dataset from public data alone:
- umbilicus: `vc_gen_umbilicus` on the normal grids;
- tracks: `extract_surface_tracks.py --backend cpu` on the level-2 surface prediction, about an hour per 1000 slices;
- verified patches: your clean GrowPatch patches.

Questions or results? Post them in the Discord.
