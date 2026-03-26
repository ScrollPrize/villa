---
title: "The Data"
hide_table_of_contents: true
---

<head>
  <html data-theme="dark" />

  <meta
    name="description"
    content="A $1,700,000+ machine learning and computer vision competition"
  />

  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://scrollprize.org" />
  <meta property="og:title" content="Vesuvius Challenge" />
  <meta
    property="og:description"
    content="A $1,700,000+ machine learning and computer vision competition"
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
    content="A $1,700,000+ machine learning and computer vision competition"
  />
  <meta
    property="twitter:image"
    content="https://scrollprize.org/img/social/opengraph.jpg"
  />
</head>

> **Work‑in‑progress 👷‍♀️**
> We are transitioning data hosting to a new repository. During the transition, some assets may appear in one location before the other. Both repositories follow the same organization structure.

**Quick start:** [example quick data access notebook](https://github.com/ScrollPrize/open-data/blob/main/examples/get-to-know-a-dataset.ipynb)

## Overview

A vast library of papyrus scrolls in ancient Herculaneum was buried beneath volcanic mud and ash during the 79 AD eruption of Mount Vesuvius. The scrolls were carbonized into a fragile but remarkably preserved state. The Vesuvius Challenge uses synchrotron micro‑CT imaging to study both **intact scrolls** and **detached fragments**.

Our goal is to **virtually unwrap** the scrolls from their 3D X‑ray volumes and recover ink that is invisible to the naked eye. Detached fragments include exposed ink and serve as **ground truth** for improving machine‑learning approaches to ink detection.

## Multiple datasets

This portal aggregates multiple released datasets under one Vesuvius Challenge data portal.

- **EduceLab-Scrolls**: the legacy dataset. Scrolls 1-4 and Fragments 1-6 belong to this dataset.
- **Vesuvius Challenge - CT Scans of Herculaneum Papyri**: newer scans released directly by Vesuvius Challenge. Most current releases on this portal belong to this dataset.

If you are publishing or presenting results, make sure you cite the dataset that corresponds to the scans you used.

## Data repositories

We host the dataset in **two repositories** (with the **same folder layout**):

- **Web-browsable samples:** https://data.aws.ash2txt.org/samples/
- **Open data bucket:** `s3://vesuvius-challenge-open-data/` usable with any S3‑compatible client (e.g., AWS CLI, boto3, s3fs, etc.). It's also [browsable directly](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/index.html).

An overview of the dataset can be found in the [Data Browser](data_browser).
The browser is the unified sample index for both scrolls and fragments.
For deeper exploration, use [Segments](data_segments) when you want mapped surface data, and [Curated Datasets](data_datasets) for ready-to-use bundles built for specific research tasks.

## What's included

The open data repository provides a consistent set of artifacts across scrolls and fragments:

- **Volumes**: 3D micro‑CT reconstructions of papyrus (primary input for virtually unrolling).
- **Segments**: extracted papyrus surfaces (geometry + surface‑aligned "texture" volumes).
- **Representations / Predictions**: derived products such as ML‑predicted surfaces and ink detection outputs (when available).
- **Metadata**: lightweight JSON/text files that describe scans, exports, and processing (where available).

## Formats at a glance

This is a practical "what you'll actually see on disk" summary.

| Data type                        | What it represents                                             | Common formats                                                  |
| -------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------- |
| **Volumetric scans ("volumes")** | 3D density/intensity values from CT reconstruction             | **OME‑Zarr** (primary), sometimes **TIFF stacks**               |
| **Segment surface volumes**      | 2D/3D data extracted along a papyrus surface at several depths | **OME‑Zarr** and/or **TIFF stacks** (`00.tif`, `01.tif`, …)     |
| **Surface geometry ("meshes")**  | The 3D sheet geometry and its flattened mapping                | **OBJ** meshes, plus **TIFXYZ** (x/y/z TIFF triplet + metadata) |
| **Model outputs**                | Predicted surfaces, ink probability maps, derived images       | **OME‑Zarr** (volumetric outputs), **TIFF** (image outputs)     |
| **Metadata**                     | Provenance, parameters, IDs, links between artifacts           | **JSON** (and occasional text files)                            |

### Why OME‑Zarr?

OME‑Zarr is the primary distribution format because it is cloud‑optimized (chunked, multi‑resolution) and supports **streaming / partial reads**—so you don't need to download entire terabyte‑scale volumes to get started.

## Organization on disk

Both repositories follow the same high‑level structure:

```text
{SAMPLE_ID}/
├── volumes/            # 3D reconstructed volumes (OME‑Zarr, sometimes TIFF)
├── segments/           # Extracted surfaces: meshes, surface volumes, (optional) ink results
└── representations/    # Derived artifacts (e.g., predictions)
```

You will typically browse by **sample ID** (e.g., a specific scroll or fragment), then choose the artifact you need (a volume, a segment, or a derived representation).

## Scrolls and Fragments

- Herculaneum scrolls scanned via synchrotron micro‑CT. These are the core targets for "virtual unwrapping" and reading.

- Detached fragments with exposed ink on their surfaces. These are especially useful for building and validating ML approaches (e.g., ink detection), because they provide ground truth signals.

➡️ **Browse all samples:** [Data Browser](data_browser)

Looking for scroll-only or fragment-only views? Use the same browser row-by-row and apply the relevant filters.
- [Segments](data_segments) for mapped surface exports and segment artifacts


## Documentation and references

- [EduceLab-Scrolls (2019)](https://arxiv.org/abs/2304.02084): technical paper describing the original dataset work.
- [EduceLab Data Sheet (2023)](https://drive.google.com/file/d/1I6JNrR6A9pMdANbn6uAuXbcDNwjk8qZ2/view?usp=sharing): technical description of more recent scans added to the dataset.
- [Scan at ESRF Draft Info Sheet (2025)](https://docs.google.com/document/d/1CDPgx7XhNsnLJw6uErT8Z5tgY3wnETQdvXpR5Kwu9K4/edit?usp=sharing)

## How to Cite

If you use Vesuvius Challenge data in a publication or presentation, cite the dataset that corresponds to the scans you used.

### Vesuvius Challenge - CT Scans of Herculaneum Papyri

Use this citation for newer scans released directly by Vesuvius Challenge:

> Giorgio Angelotti, Stephen Parsons, Sean Johnson, Elian Rafael Dal Prà, Johannes Rudolph, Paul Tafforeau, Alessandro Mirone, Paul Henderson, Hendrik Schilling, Forrest McDonald, David Josey, Youssef Nader, C. Seth Parker, W. Brent Seales. *Vesuvius Challenge - CT Scans of Herculaneum Papyri*. Vesuvius Challenge.

### EduceLab-Scrolls

Scrolls 1-4 and Fragments 1-6 belong to the legacy EduceLab-Scrolls dataset.

- In any published abstract, cite `EduceLab-Scrolls` as the source of the data.
- In any published manuscripts using data from EduceLab-Scrolls, reference: Parsons, S., Parker, C. S., Chapman, C., Hayashida, M., & Seales, W. B. (2023). *EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using X-ray CT*. ArXiv [Cs.CV]. https://doi.org/10.48550/arXiv.2304.02084.
- Include language similar to the following in the methods section: "Data used in the preparation of this article were obtained from the EduceLab-Scrolls dataset [above citation]."

## Support

- GitHub Issues: [Vesuvius Challenge repository](https://github.com/scrollprize/villa)
- Community Forum: [Discord](https://discord.gg/V4fJhvtaQn)

## Licenses

- [CC‑BY‑NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (unless otherwise noted for specific assets)
- Scrolls 1-4 and Fragments 1-6 are from the EduceLab-Scrolls Dataset, copyrighted by EduceLab/The University of Kentucky. Permission to use the data linked herein according to the terms outlined above is granted to Vesuvius Challenge, with additional citation requirements listed in [How to Cite](#how-to-cite).
