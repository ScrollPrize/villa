---
title: "Curated Datasets"
hide_table_of_contents: true
---

<head>
  <html data-theme="dark" />

  <meta
    name="description"
    content="A $1,500,000+ machine learning and computer vision competition"
  />

  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://scrollprize.org" />
  <meta property="og:title" content="Vesuvius Challenge" />
  <meta
    property="og:description"
    content="A $1,500,000+ machine learning and computer vision competition"
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
    content="A $1,500,000+ machine learning and computer vision competition"
  />
  <meta
    property="twitter:image"
    content="https://scrollprize.org/img/social/opengraph.jpg"
  />
</head>

# Curated Datasets

The data available through Vesuvius Challenge is large, frequently updated, and can be overwhelming to navigate.

Here are some organized datasets suited for particular tasks or subproblems.
Largely, these curate the segmentation efforts of our team and community.
Click one of the datasets to find a download along with more information.

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
