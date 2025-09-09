---
title: "Segments"
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

**Where are the segments?** Segments are separated by [scroll](https://dl.ash2txt.org/full-scrolls/) on our data server. Navigate to the /*.volpkg/paths directory (eg., [/full-scrolls/Scroll1/PHercParis4.volpkg/paths/](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/)).

**What are segments?** Read on:

A segment is a mapped section of the written surface inside the scroll. The scrolls have a complex interior structure because they were crumpled and charred by the weight and heat of volcanic debris. Many different tools and techniques have been employed to map the intricate 3D shapes of these papyrus sheets ([Segmentation Tutorial](tutorial3)).

<figure className="max-w-[500px]">
  <video autoPlay playsInline muted controls className="w-[100%] rounded-xl" poster="/img/tutorials/vc-extrapolation2.webp">
    <source src="/img/tutorials/vc-extrapolation2.webm" type="video/webm"/>
  </video>
  <figcaption className="mt-0">Illustration of Volume Cartographer with an algorithm extrapolating in 3D.</figcaption>
</figure>

The geometric mapping data is stored in various file formats depending on the mapping tool used (e.g., VC generates pointset files).

Ultimately, all mapping data is converted into a mesh, commonly stored in OBJ format. The mesh, together with the original scan volume, defines the segment.

Auxiliary files are also provided with each segment. These include: UV coordinates from segment flattening, surface volume, and composite image.

**UV coordinates** - `<id>.obj`

The UV coordinates define a flattened representation of the segment by storing the flattened UV positions of each vertex in the OBJ mesh. This flattening can be achieved using algorithms such as SLIM, LSCM, and ABF.

<div className="flex w-[100%]">
    <div className="w-[100%] mb-2 mr-2"><img src="/img/data/mesh.webp" className="w-[100%]"/><figcaption className="mt-0">A meshed segment prior to flattening.</figcaption></div>
</div>

**Surface volume** - `layers` directory

A surface volume is a flattened subvolume with the segment as the center layer. The surface volume is formed by extracting CT voxels both above and below the segment, guided by the surface normal vector in XYZ. Each new layer is one voxel above or below the segment. Our standard surface volume size is 32 layers in front and behind of the segment layer, 65 layers in total. This stack of layers extracts the segment so it can be inspected near the written surface for ink.

<div className="flex w-[100%]">
    <div className="w-[100%] mb-2 mr-2"><img src="/img/data/mesh2sv.webp" className="w-[100%]"/><figcaption className="mt-0">A surface volume created by flattening the region around a segment.</figcaption></div>
</div>

<figure className="max-w-[600px]">
  <img src="/img/data/surface_volume.gif"/>
  <figcaption className="mt-0">Scrubbing through layers of the surface volume of segment <a href="https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/20230827161846/layers/">20230827161846</a>.</figcaption>
</figure>

**Composite image** - `<id>.tif` or `composite.jpg`

The composite texture image is a 2D image created by applying a max filter over a specified number of layers above and below the segment map.

<div className="flex w-[100%]">
  <div className="w-[100%] mb-2 mr-2"><img src="/img/data/20231210121321.webp" className="w-[100%]"/><figcaption className="mt-0">Composite texture image of segment [20231210121321](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/20231210121321/20231210121321.tif) from PHerc. Paris 4 (Scroll 1).</figcaption></div>
</div>

Our Segmentation Team has been mapping select regions of the scrolls. The community has made this a significantly more automated process with improved tools, but it still involves considerable human input. 

<div className="flex w-[100%]">
  <div className="w-[100%] mb-2 mr-2"><img src="/img/data/segment_areas.webp" className="w-[100%]"/><figcaption className="mt-0">Total segment area created over time by our team and community.</figcaption></div>
</div>

## Data format

You can find all segment data on the [data server](https://dl.ash2txt.org/) in the [`/full-scrolls/`](https://dl.ash2txt.org/full-scrolls/) folder.

* **Scroll 1 (PHerc. Paris. 4):** [`/full-scrolls/Scroll1/PHercParis4.volpkg/paths/`](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/)
* **Scroll 2 (PHerc. Paris. 3):** [`/full-scrolls/Scroll2/PHercParis3.volpkg/paths/`](https://dl.ash2txt.org/full-scrolls/Scroll2/PHercParis3.volpkg/paths/)
* **Scroll 3 (PHerc. 332):** [`/full-scrolls/Scroll3/PHerc332.volpkg/paths/`](https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths/)
* **Scroll 4 (PHerc. 1667):** [`/full-scrolls/Scroll4/PHerc1667.volpkg/paths/`](https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/paths/)
* **Scroll 5 (PHerc. 172):** [`/full-scrolls/Scroll5/PHerc172.volpkg/paths/`](https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/paths/)

There are many different types of files provided in each `paths/<id>/` folder:
* `<id>.obj`: Mesh of the segment (see above)
  * `<id>.mtl`: Information about `<id>.obj`
* `layers/{00-64}.tif`: Surface volume of 65 layers (see above)
* `<id>.tif` or `composite.jpg`: Texture of the surrounding voxels
* `<id>_prediction*.png`: Ink prediction for the segment

* `meta.json`: Metadata of the segment
* `pointset.vcps`: Pointset of a segment. More information [here](https://www.kaggle.com/code/kglspl/simple-vcps-parser)
* `pointset.vcano`: Information about the pointset, such as which slices were annotated
* `<id>_points.obj`: Pointcloud
* `<id>.ppm`: Per-pixel map, a custom data format mapping points between the surface volume and the original 3D volume of the scroll
* `<id>_cellmap.tif`: Maps each pixel to its corresponding mesh triangle
* `<id>_mask.png`: Mask of the flattened segment
* `<id>_flat_mask.png`: Mask of the flattened segment
* `<id>_flat.png`: Mask of the flattened segment
* `<id>_flat_0.png`: Mask of the flattened segment
* `energies_flatboi.txt`: Minimization of the distortion energy
* `author.txt`: Name of the author of the segment.
* `area_cm2.txt`: Total surface area, in cm<sup>2</sup>.
