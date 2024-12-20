---
title: "Curated Datasets"
hide_table_of_contents: true
---

<head>
  <html data-theme="dark" />

  <meta
    name="description"
    content="A $1,000,000+ machine learning and computer vision competition"
  />

  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://scrollprize.org" />
  <meta property="og:title" content="Vesuvius Challenge" />
  <meta
    property="og:description"
    content="A $1,000,000+ machine learning and computer vision competition"
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
    content="A $1,000,000+ machine learning and computer vision competition"
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

## `volumetric-instance-labels`

Volumetric instance segmentation labels.

<div className="mb-4">
  <img src="/img/data/datasets/volumetric-instance-labels.webp" className="w-[60%]"/>
  <figcaption className="mt-[-6px]">Two annotated cubes, with volumetric labels representing papyrus sheet instances.</figcaption>
</div>

This dataset contains a subset of Scroll 1, chunked into 256x256x256 cubes.
For each cube, the original scroll volume data and the instance segmentation data are provided (each in `.nrrd` format).
- [README](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumetric-instance-labels/README.txt)
- [.zip download](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumetric-instance-labels/instance-labels-harmonized.zip) (1.4 GB)

## `GrandPrizeBannerRegion`

Data related to the Grand Prize (GP) region from `Scroll1`. The dataset includes volumes and meshes associated to manual segmentation of surfaces.

- [README](https://dl.ash2txt.org/datasets/GrandPrizeBannerRegion/README.txt)
- [gp_meshes.7z](https://dl.ash2txt.org/datasets/GrandPrizeBannerRegion/gp_meshes.7z) (288 MB)
- [gp_volume.zarr/](https://dl.ash2txt.org/datasets/GrandPrizeBannerRegion/volumes/gp_volume.zarr) (77 GB)
- [gp_tifstack.7z](https://dl.ash2txt.org/datasets/GrandPrizeBannerRegion/volumes/gp_tifstack.7z) (389.9 GB)



