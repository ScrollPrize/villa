---
id: prizes
title: "New Year, New Prizes"
sidebar_label: "2025 Prizes"
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

This year, we're changing up the prizes a bit from previous phases. We’re issuing more frequent, targeted awards that address specific technical challenges.

We’re also going to offer targeted “asks” for prizes, as we become aware of what roadblocks lie in the way. We still encourage open ended solutions to problems, but will often provide more specific requirements for some progress prizes. 

_**Let’s get started with the prizes!**_

## Read Entire Scroll Prize ($200,000)
[Submission Form](https://docs.google.com/forms/d/e/1FAIpQLSed2vRgW3HECXW3E6-llYPOST550yr4L0T5AIQp45GAYRcnGQ/viewform?usp=dialog)

We’re awarding $200,000 to the first team to unwrap an entire scroll and produce readable letters within.

<details class="submission-details">
<summary>Submission criteria and requirements</summary>

1. **Segmentation**
* Estimate the total area of the scroll’s written surface (in cm2)
* Compute the total surface area of the segmented mesh (in cm2) in your submission. **You must segment 90% or more of the total estimated area**
* Segments should be flattened and shown in 2D as if the scroll were unwrapped.
* The scroll should ideally be captured by a single segmentation (or each connected component of the scroll) rather than separate overlapping segmentations.
* Segments should pass geometric sanity checks; for example, no self-intersections

2. **Ink detection**
* Your submission must contain ink predictions for the entire flattened mesh, of the same shape as the flattened surface
* The entire submission is too large to transcribe quickly, so the papyrological team will evaluate each line as:
    * ✅ **readable** (could read 85% of the characters),
    * ❌ **not readable** (couldn't),
    * 🟡 **maybe** (would have to stop and actually do the transcription to determine), or
    * 🔷 **incomplete** (line incomplete due to the physical boundaries of the scroll)
* 80% of the total complete lines (incomplete lines will not be judged) must be either 🟡 **maybe** or ✅ **readable**. Multiple papyrologists may review each line, in which case ties will be broken favorably towards the submission.

As a baseline, here's how the 2023 Grand Prize banner would have scored:

<div className="mb-4">
  <img src="/img/2024-prizes/GP_scores_sample.webp" className="w-[80%]"/>
  <figcaption className="mt-[-6px]">Visible Ink on Scroll 5 (<a href="/img/ink/2023_GP_banner_lines_score.webp">full banner</a>).</figcaption>
</div>

Total lines: 240. Complete lines: 206. Passing lines: 137. Pass rate: 137 / 206 = **67% (needs to be 80%)**.

We may reward partial work - if your unrolling works but the ink detection isn't all the way there yet, go ahead and submit!

</details>

## First Letters and Title Prizes
[Submission Form](https://docs.google.com/forms/d/e/1FAIpQLSdw43FX_uPQwBTIV8pC2y0xkwZmu6GhrwxV4n3WEbqC8Xof9Q/viewform?usp=dialog)

**First Letters: $60,000 to the first team that uncovers 10 letters within a single 4cm^2 area of any of the following scrolls:**
* Scroll 2
* Scroll 3
* Scroll 4

**First Title: $60,000 to the first team to unveil a title of any of our scrolls.**

<div className="mb-4">
  <img src="/img/data/title_example.webp" className="w-[50%]"/>
  <figcaption className="mt-[-6px]">Visible Title in a Scroll Fragment.</figcaption>
</div>

## Progress Prizes
As the challenges of autosegmentation and generalized ink detection become clearer, so do our expections for progress prize submissions.

Submissions are evaluated monthly, and multiple submissions/awards per month are permitted. The next deadline is 11:59pm Pacific, February 28, 2025! When ready, please submit using [this form](https://docs.google.com/forms/d/e/1FAIpQLSdPKXHtDgZs0kihxTMlwCvFMiJcR0b6ls55Kol9-KFZQCU_zQ/viewform?usp=header).

**Core Requirements**
1. Problem Identification and Solution
   * Address a specific challenge using competition scroll data
   * Provide clear implementation path and a demonstration of its use
   * Demonstrate significant advantages over existing solutions
2. Documentation
   * Include comprehensive documentation
   * Provide usage examples
3. Technical Integration
   * Accept standard competition formats (multipage TIFs, on-disk numpy arrays)
   * Maintain consistent output formats
   * Designed for modular integration


### Wish list

**High-Quality Surface or Fiber Predictions**
  * Input Format: image/label pairs of surface or fiber labels
  * Output Format: zarr
  * Result: High-accuracy predictions of written surfaces or fibers. Good fiber predictions would consist of long connected components without branching

**High-Quality 3d Ink Dataset**
  * File Format: Zarr or .tif slices
  * Result: Image/label pairs of high quality 3d ink labels that are “ready-to-run” , no additional cropping or preprocessing required.

**Label Generation Methods**
  * Input Format: zarr or other on-disk npy array
  * Output Format: zarr
  * Result: Image/label pairs of surface or fiber labels that are higher quality than simple voxelization of existing obj meshes

**Inpainting**
  * Input Format: .obj mesh
  * Output Format: Triangulated .obj mesh (or quad mesh) with surface normals (sanity constraints: self intersections, watertight, etc)
  * Implementation: Within the Vesuvius python library (ie: vesuvius.meshing.inpaint)
  * Result: Given a mesh obj with holes, the holes are inpainted in a way that respects the underlying surface, or interpolates accurately enough that the resultant mesh does this without using the scroll data.

**Sheet Switch Detection**
  * Input Format: .obj mesh or surface volume
  * Output Format: zarr array containing labeled switches and an overlay of the flattened segment highlighting the sheet switches
  * Result: A robust solution to programatically detecting sheet switches within generated segments or surface volumes

**Representation to Graph Tool**
* Input Format: zarr or other disk array
* Output Format: Graph with nodes and edges
* Node Attributes:
  * ID / name should be zyx position
  * Input format  [fibers, sheet patches, point clouds, objs, etc]
  * Additional information to relate the node to the input
  * Initial winding angle in degrees
  * This is w.r.t umbilicus
  * Assigned winding angle (can be initialized as None) in degrees
  * Ground Truth winding angles, derived from manual segmentations example: [scroll 1 gp banner, scroll 4 banner]
  * Certainty / Probability score of node accuracy

* Edge Attributes:
  * Connection between nodes (i, j)
  * Angle difference between node i and j, represented as (K) in degrees
  * Certainty / Probability score of edge accuracy of (K)

* Result: A tool or function that accepts a patch or chunk based representation, and returns a graph based representation

<details class="example_graph">
<summary>An example on VC3D Patch Seeds:</summary>
* Patches are nodes
* Patch position in 3D space is ID
* Patch accuracy from sheet tracer is certainty of Node
* Patch position around the umbilicus is initial winding angle [-180, 180] degree
* Overlapping patches form Edges
* Amount of overlap is certainty of Edge
* Difference in Edge Nodes is K range [-180, 180] for overlapping nodes, probably mostly in range [+- few degrees around 0]
* Solver adds the assigned winding angle around the umbilicus that indicates which wrap and angle the patch is at
</details>

**Improvements to existing graph solvers or implementation of different graph solver**
  * Input: graph structured as described above
  * Output: graph with the winding angle assigned
  * Can include manual annotation or refinements, optimizing for minimal human input

**Graph meshing**
  * Result: Given a graph with attributes pointing to the underlying data, a triangulated obj mesh with UVs is returned 

**Whole Scroll flattening**
  * Input format: obj
  * Output format: flattened obj with UV coordinates
  * Result: Given a triangulated .obj mesh with surface normals, the mesh is flattened such that the distortion present on the boundaries of current flattening methods tested on large segmentations is no longer present. 

**Improvements to VC3D**
  * Add a refresh button to the segment list, so users don’t have to relaunch VC3D to refresh the segments
  * Allow running vc_grow_seg_from_segs with a target of a previous trace instead of a seed patch
  * Integrate a surface normal cost function into the solver
  * Add functionality to manipulate patch and trace locations, similar to how a segment is manipulated in khartes or the original implementation of VC
  * Allow scroll regions to be masked rather than individual patches, forcing the solver to ignore these regions
  * Allow users to “push” or “pull” the segment forward/back from the 2d flattened view

**Integrations into Vesuvius Python Library**

Integrate the 2024 FASP submissions into the Vesuvius Python Library to enable more people to experiment and iterate upon them quickly

**VC3D Tracing / Seeding –**

_[vc_grow_seg_from_seed.cpp](https://github.com/hendrikschilling/volume-cartographer/blob/dev-next/apps/src/vc_grow_seg_from_seed.cpp)_ - > vesuvius.tracer.seed
  * Input format: zarr array containing volumetric predictions, json or other config method for parameters
  * Output format: obj meshes or tifxyz quad meshes of patches, with required metadata for next steps
  * Mode: Seed

_[vc_grow_seg_from_seed.cpp](https://github.com/hendrikschilling/volume-cartographer/blob/dev-next/apps/src/vc_grow_seg_from_seed.cpp)_ - > vesuvius.tracer.expand
  * Input format: folder containing patches from seeds, json or other config method for parameters
  * Output format: obj meshes or tifxyz quad meshes of patches, with required metadata for next steps, including overlap markers
  * Mode: Expand

_[vc_grow_seg_from_segments.cpp](https://github.com/hendrikschilling/volume-cartographer/blob/dev-next/apps/src/vc_grow_seg_from_segments.cpp)_  - > vesuvius.tracer.trace
  * Input format: folder containing patches from seeds, source/target seed, json or other config method for parameters
  * Output format: obj mesh and/or tifxyz quadmesh

<details class="implementation_details">
<summary>Implementation Details</summary>

Both call functions in core/src [surface.cpp](https://github.com/hendrikschilling/volume-cartographer/blob/dev-next/core/src/Surface.cpp) and surfacehelpers.cpp. The cost functions are located [here](https://github.com/hendrikschilling/volume-cartographer/blob/dev-next/core/include/vc/core/util/CostFunctions.hpp)

This does not have to use ceres, or any particular solver. We’re looking at functionality not strict requirements on how the solution is attained, but the performance should be equal in terms of output quality. We are willing to sacrifice some speed in this implementation if necessary, but not substantially. 

For python, a good starting point might be theseus (https://github.com/facebookresearch/theseus) which offers many of the same solvers used in the tracer.
</details>

**Spiral Fitting –**
  
We’re looking to utilize the spiral fitting created by Paul for his submission [here](https://github.com/pmh47/spiral-fitting). We want it again similar to the sheet tracer in utilization, ie: calling vesuvius.fit
  * Input format: one or many zarr arrays containing the outputs of the representation step.
  * output format: .obj mesh

We’d like to explore some methods of using this on patches generated by the tracer or patches generated by any other methods – potentially utilizing this as a generalized “stitcher”


