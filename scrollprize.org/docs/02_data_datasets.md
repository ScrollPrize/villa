---
title: "Curated Datasets"
---

<head>
  <html data-theme="dark" />

  <meta
    name="description"
    content="Curated Vesuvius Challenge datasets from Herculaneum scrolls: labeled papyrus fibers, volumetric instance labels, and a Grand Prize region X-ray CT volume."
  />

  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://scrollprize.org" />
  <meta property="og:title" content="Vesuvius Challenge" />
  <meta
    property="og:description"
    content="Curated Vesuvius Challenge datasets from Herculaneum scrolls: labeled papyrus fibers, volumetric instance labels, and a Grand Prize region X-ray CT volume."
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
    content="Curated Vesuvius Challenge datasets from Herculaneum scrolls: labeled papyrus fibers, volumetric instance labels, and a Grand Prize region X-ray CT volume."
  />
  <meta
    property="twitter:image"
    content="https://scrollprize.org/img/social/opengraph.jpg"
  />
</head>

import JsonLd from '@site/src/components/JsonLd';

<JsonLd data={{ "@context":"https://schema.org","@type":"Dataset","name":"Herculaneum Scrolls — Curated Datasets","description":"Task-specific curated bundles from the Herculaneum scrolls, including manually labeled papyrus fibers, volumetric instance segmentation labels, and the Scroll 1 Grand Prize region.","url":"https://scrollprize.org/data_datasets","creator":{"@type":"Organization","name":"Vesuvius Challenge","url":"https://scrollprize.org/"},"measurementTechnique":"X-ray computed tomography","keywords":["Herculaneum scrolls","papyri","X-ray CT","virtual unwrapping","ink detection","machine learning"],"isAccessibleForFree":true,"license":"https://dl.ash2txt.org/LICENSE.txt","distribution":{"@type":"DataDownload","encodingFormat":"image/tiff","contentUrl":"https://scrollprize.org/data_datasets"} }} />

# Curated Datasets

<div className="opacity-60 mb-8 italic">July 2026</div>

The data available through Vesuvius Challenge is large, frequently updated, and can be overwhelming to navigate.

Here are some organized datasets suited for particular tasks or subproblems.
Largely, these curate the segmentation efforts of our team and community.
Click one of the datasets to find a download along with more information.

## `spiral-input` (2026-07)

A collection of same-winding and multi-winding annotations, in the form of: surface patches, line annotations, and point collections.

<div className="flex flex-wrap mb-4">
  <div className="w-[41%] mr-[3%] mb-2">
    <img src="/img/data/datasets/spiral-input-multiwinding.webp" className="w-[100%]"/>
    <figcaption className="mt-[-6px]">Multi-winding annotations.</figcaption>
  </div>
  <div className="w-[52%]">
    <img src="/img/data/datasets/spiral-input-fiber.webp" className="w-[100%]"/>
    <figcaption className="mt-[-6px]">Same-winding fiber annotation.</figcaption>
  </div>
</div>

This dataset contains manual annotations of the scroll wraps, recording which parts of the surface belong to the same wrap (*winding*) of the sheet for use as ground-truth inputs when fitting a global winding solution. It is organized by scroll. Each scroll below provides some combination of: verified and unverified surface patches (grid-topology quad meshes sampled on the papyrus surface), line annotations (curves traced across the surface), point collections, same-/relative-/absolute-winding annotations, and the fitted umbilicus.

### Paris4

{/* Paris4 = Scroll 1 / PHercParis4. */}
Spiral annotations for scroll PHercParis4 (Scroll 1). Contents include ~27,000 verified and ~204,000 unverified surface patches, traced tracks, and point collections, together with `same_windings` / `relative_windings` / `abs_winding` graphs, the fitted `umbilicus`, fiber and outer-shell geometry, and the volume inputs used by the fitting pipeline (~49.6 GB total).

- [README](pathname:///data/datasets/spiral-input-PHercParis4-README.md)
- [Browse on Hugging Face](https://huggingface.co/buckets/scrollprize/datasets/tree/spiral/PHercParis4)
- [Tutorial: Spiral Fitting](tutorial_spiral) — how to fit a whole-scroll surface to these annotations

{/* TODO: add more scroll subsections here later, e.g. "### Scroll5", each following the Paris4 pattern above. */}

## `surface-labels` (2026-07)

Labeled papyrus surfaces: voxelized, recto (inside) sheet surfaces within the scroll volume.

<div className="mb-4">
  <img src="/img/data/datasets/surface-labels-kaggle.webp" className="w-[70%]"/>
  <figcaption className="mt-[-6px]">3D sub-volumes paired with surface label masks.</figcaption>
</div>

Voxelized surface segmentation that identifies the volumetric part of the papyrus we want to map. This dataset provides labels of the recto surfaces of the papyrus sheet, paired with the corresponding scroll volume data so they can be used directly as training inputs for machine-learning models.

- [README](pathname:///data/datasets/surface-labels-README.md)
- [Browse on Hugging Face](https://huggingface.co/buckets/scrollprize/datasets/tree/surfaces)

## `ink-labels` (2026-07)

Binary ink masks aligned to surface volumes (2D) and scroll volumes (3D).

<div className="mb-4">
  <img src="/img/firstletters/ink-label.webp" className="w-[55%]"/>
  <figcaption className="mt-[-6px]">Texture image (left) and the corresponding binary ink label (right).</figcaption>
</div>

Ink labels are binary images marking the location of ink on a papyrus surface. No infrared ground truth exists in scrolls — labels begin as hand-annotated ink strokes and are refined through iterative pseudo-labeling. Models trained on these labels can then detect ink elsewhere in the scrolls.

- [README](pathname:///data/datasets/ink-labels-README.md)
- [Browse on Hugging Face](https://huggingface.co/buckets/scrollprize/datasets/tree/ink)

<hr className="mt-16 mb-10 border-t-2 border-solid opacity-20" />

<details className="mb-4">
<summary className="cursor-pointer uppercase tracking-widest text-sm opacity-60">Archived datasets — earlier pipeline generation</summary>

:::warning[OUTDATED CONTENT]

The datasets below describe an earlier generation of the Vesuvius Challenge pipeline. Tools, data layouts, and results referenced here may have been superseded.

:::

## `fiber-skeletons` (2025-07)

A dataset of manually annotated papyrus fibers - the individual strands that make up a papyrus sheet.

<div className="flex flex-wrap mb-4">
  <div className="w-[55%] mr-4 mb-2">
    <img src="/img/data/datasets/skeleton-labeled-fibers.gif" className="w-[100%]"/>
    <figcaption className="mt-[-6px]">Fiber labels inside a scroll cube.</figcaption>
  </div>
  <div className="w-[34%]">
    <img src="/img/data/datasets/fibers-color.webp" className="w-[100%]"/>
    <figcaption className="mt-[-6px]">Another view of fiber skeletons.</figcaption>
    </div>
</div>

Cubes of size 256^3 or 512^3 were selected from within the scroll, and inside each cube, every papyrus fiber was traced and labeled.
The fibers have been converted to a volumetric/voxelized representation to be used as inputs to machine learning or other methods that expect 3D image data.
- [README](https://dl.ash2txt.org/datasets/fiber-skeletons/README.txt)
- [.zip download](https://dl.ash2txt.org/datasets/fiber-skeletons/fiber-skeletons.zip) (422 MB)

## `volumetric-instance-labels` (2024-10)

Volumetric instance segmentation labels.

<div className="mb-4">
  <img src="/img/data/datasets/volumetric-instance-labels.webp" className="w-[60%]"/>
  <figcaption className="mt-[-6px]">Two annotated cubes, with volumetric labels representing papyrus sheet instances.</figcaption>
</div>

This dataset contains a subset of Scroll 1, chunked into 256x256x256 cubes.
For each cube, the original scroll volume data and the instance segmentation data are provided (each in `.nrrd` format).
- [README](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumetric-instance-labels/README.txt)
- [.zip download](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumetric-instance-labels/instance-labels-harmonized.zip) (1.4 GB)

## `grand-prize-banner-region` (2024-12)

Data related to the 2023 Grand Prize (GP) region from Scroll 1.

<div className="flex flex-wrap mb-4">
  <div className="w-[15.9%] mr-4 mb-2">
    <img src="/img/data/datasets/gp_mesh.webp" className="w-[100%]"/>
    <figcaption className="mt-[-6px]">Surface mesh.</figcaption>
  </div>
  <div className="w-[40%]">
    <img src="/img/data/datasets/gp_predictions.webp" className="w-[100%]"/>
    <figcaption className="mt-[-6px]">ML predictions for medial surface segmentation.</figcaption>
    </div>
</div>

The dataset includes the scan volume and segmented surface meshes created by our segmentation team.
We also provide predictions from machine learning [models](https://dl.ash2txt.org/ml-models/) that aim to segment the medial surface of the papyrus sheet.

- [README](https://dl.ash2txt.org/datasets/grand-prize-banner-region/README.txt)
- [gp_meshes.7z](https://dl.ash2txt.org/datasets/grand-prize-banner-region/gp_meshes.7z) (288 MB)
- [gp_volume.zarr/](https://dl.ash2txt.org/datasets/grand-prize-banner-region/volumes/gp_volume.zarr) (77 GB)
- [gp_tifstack.7z](https://dl.ash2txt.org/datasets/grand-prize-banner-region/volumes/gp_tifstack.7z) (389.9 GB)
- [gp_legendary-medial-surfaces.7z](https://dl.ash2txt.org/datasets/grand-prize-banner-region/predictions/gp_legendary-medial-cubes.7z) (5.8 GB)
- [gp_legendary-medial-surfaces-softmax.7z](https://dl.ash2txt.org/datasets/grand-prize-banner-region/predictions/gp_legendary-medial-cubes-softmax.7z) (146.8 GB)

</details>
