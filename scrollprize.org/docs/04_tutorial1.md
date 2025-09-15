---
title: "Tutorial: Scanning"
sidebar_label: "Scanning"
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

import { TutorialsTop } from '@site/src/components/TutorialsTop';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<TutorialsTop highlightId={2} />

### Campfire scroll

To get a basic understanding of the kind of data we’re working with, it’s useful to look at it directly. Our goal will be to look at a scroll in two ways: 1) a “word soup” of floating letters, and 2) a mesh representation:

<div className="flex flex-wrap items-start">
  <figure className="sm:w-[48.4%]">
    <img src="/img/tutorials/en-gedi-word-soup.gif"/>
    <figcaption className="mt-0">Word soup in a simulated scroll <a href="https://www.youtube.com/watch?v=tL7rhIFNtQg">(source)</a></figcaption>
  </figure>
  <figure className="sm:w-[50%]">
    <video autoPlay playsInline loop muted className="w-[100%]" poster="/img/tutorials/meshlab-small2.webp">
      <source src="/img/tutorials/meshlab-small2.webm" type="video/webm"/>
    </video>
    <figcaption className="mt-0">Mesh representation of the “campfire scroll”</figcaption>
  </figure>
</div>

We’ll use [Stephen Parsons’s](https://www2.cs.uky.edu/dri/stephen-parsons/) “campfire scroll” proxy (available on the <a href="/data">Data page</a>). He made this scroll by writing on papyrus with both carbon ink (low contrast with papyrus in CT scans) and iron gall ink (high contrast with papyrus in CT scans). We won’t give away all the reference pictures, but here is the last page of the scroll:

<figure className="max-w-[250px]">

![](/img/tutorials/campfire-last-page.webp)

<figcaption>Last page of the campfire scroll, with nonsense words</figcaption>
</figure>

Stephen then _carbonized_ the scroll through a process that is similar to that of making charcoal: he rolled up the scroll, sealed it in a cookie tin with a tiny hole poked in the top, and heated the tin in a campfire. In a hot, oxygen-poor environment like the cookie tin, the papyrus turns to carbon rather than catching on fire. He then put the resulting carbonized scroll in a benchtop CT scanner (Bruker SkyScan 1173).

<figure className="">
  <div className="flex flex-wrap">
    <img className="max-h-[250px]" src="/img/tutorials/campfire-rolled2.webp"/>
    <img className="max-h-[250px]" src="/img/tutorials/campfire-rolled.webp"/>
    <img className="max-h-[250px]" src="/img/tutorials/campfire-camp.webp"/>
    <img className="max-h-[250px]" src="/img/tutorials/campfire-container.webp"/>
    <img className="max-h-[250px]" src="/img/tutorials/campfire-result.webp"/>
    <img className="max-h-[250px]" src="/img/tutorials/campfire-scanning.webp"/>
    <img className="max-h-[250px]" src="/img/tutorials/campfire-scanning2.webp"/>
  </div>
  <figcaption className="mt-1">Making of the campfire scroll</figcaption>
</figure>

### Fiji

:::tip

We now also have a community-contributed program that is worth trying out: [Scroll Viewer](https://github.com/lukeboi/scroll-viewer).

:::

We’ll use the [“Fiji”](https://imagej.net/software/fiji/downloads) open source program, which is a versatile tool for all sorts of image-based operations. It is a distribution of ImageJ, but the main version of ImageJ doesn’t include all the features we need. (In fact, “Fiji” is a recursive acronym for “Fiji Is Just ImageJ”.)

:::info OS-specific instructions

<Tabs groupId="operating-systems">
  <TabItem value="win" label="Windows">
    * No known issues.
  </TabItem>
  <TabItem value="mac" label="macOS">

* Even if you’re using an M1/M2 processor, please use the x86_64 version.
* If you get a popup saying that the app can’t be opened, be sure to *right click > Open* to circumvent the blocking.

<figure className="max-w-[400px]">

![](/img/tutorials/right-click-open.webp)

</figure>

* Alternatively, you can use the “Open Anyway” trick: go to *“System Preferences > Security & Privacy > General > Open Anyway”*.

<figure className="max-w-[400px]">

![](/img/tutorials/macos-open-anyway.webp)

</figure>

  </TabItem>
  <TabItem value="linux" label="Linux">
    * We recommend downloading the `.app` portable application from their website, instead of using a package manager. It does require you to have the JVM installed.
  </TabItem>
</Tabs>

:::

<div>To follow along, download the tutorial data: <a href="https://gist.github.com/janpaul123/280262ebce904f7366fe4cc155593e90">campfire.zip</a>. In here there are three folders:</div>

* `raw`: the raw X-ray photos of the scroll.
* `rec`: the reconstructed 3D image volume (“rec” = “reconstructed”).
* `logs`: log files during scanning and reconstruction.

Let’s start with `raw`. These are the raw X-ray photos of the scroll, taken in 0.15 degree steps for a full 360 degree rotation, so there are 2400 photos in total (you can see these details in `logs/carbonized01-45kV_.log` and `logs/carbonized01-45kV__TS.csv`).

However, to keep this tutorial dataset small, we have resized these photos to 1/4th their size and kept only every 4th photo. This effectively reduces the voxel size of these scans from the original 26µm to 104µm. We also reduced the bit depth from 16 bits to 8 bits.

We don’t typically use these photos directly, but they are useful to understand how CT scanning works.

<figure className="max-w-[556px]">

![](/img/tutorials/imagej-import2.webp)

</figure>

1. In Fiji navigate to *“File > Import > Image Sequence”*. Then *“Browse”* to select `campfire/raw`.
2. The *“Count”* should show exactly 600.<br/>
**If the count is not exactly 600**: double-check that the unzipping of the dataset resulted in multiple directories. Sometimes zip programs put all files inside one big directory.
3. Click *“OK”* to load the image stack.

:::tip

If Fiji crashes, or you run into other issues, try to increase the memory limit: *“Edit > Options > Memory and Threads”.*

:::

Note that when you are loading a dataset that hasn’t been reduced in size beforehand, you can do that here. For example, we could have set *“Scale”* to *“25%”* and *“Step”* to *“4”*, to get a similar result.

In the resulting scan you can vaguely see some features, perhaps even some ink, but it's hard to see what is actually in there:

<figure>
  <video autoPlay playsInline loop muted className="max-w-[100%] rounded-xl" poster="/img/tutorials/imagej-raw-xrays2.webp">
    <source src="/img/tutorials/imagej-raw-xrays2.webm" type="video/webm"/>
  </video>
</figure>

### Volumes

However, this is not yet the data we use for virtual unwrapping or machine learning. For that, these x-ray projection images are combined using a process called [tomographic reconstruction](https://en.wikipedia.org/wiki/Tomographic_reconstruction). This is typically done by software that comes with the CT scanner, so we already have done this, in our case using the Bruker SkyScan NRecon tool.

The result of reconstruction is a 3D image, or volume, which is made up of 3D pixel cubes called voxels. The *voxel size* of the volume tells us the physical size of the cube, and the value stored in the voxel is an estimate of that location's radiodensity. Like the x-ray images, the volume is stored as a .tif image stack, but this time each image represents a slice along the “z” axis of the volume.

<figure className="max-w-[556px]">

![](/img/tutorials/imagej-import-rec2.webp)

</figure>

* First close the existing window with the image stack.
* Then open our reconstructed slices `campfire/rec` in the same way into Fiji (*“File > Import > Image Sequence”*).
* The count should be **477** now.

The result should look like this:

<figure>
  <video autoPlay playsInline loop muted className="w-[100%] max-w-[400px] rounded-xl" poster="/img/tutorials/imagej-slices2.webp">
    <source src="/img/tutorials/imagej-slices2.webm" type="video/webm"/>
  </video>
</figure>

In these images, radiodense materials have bright pixel values while more radiolucent materials have darker pixel values. As a result, you can see the iron gall ink light up quite clearly in these slices as bright lines which follow the wraps of the scroll. However, it’s impossible to make out any letters from this "top-down" view of the volume.

* Go to *“Plugins > Volume Viewer”*.
  * It can take a while to load — there is a status bar with a loading bar on the main Fiji window.
  * In the top bar select *“Mode > Volume (4)”*
  * And *“Interpolation: Nearest Neighbor (0)”*
  * Check all 3 checkboxes at the bottom (*“Axes”*, *“Clipping”*, and *“Slice positions”*).
* Click the *“xy”* button at the bottom so you get a top down view
  * Then click and drag up on the visualization to rotate it down.

<figure>
  <video autoPlay playsInline loop muted className="w-[100%] max-w-[700px] rounded-xl" poster="/img/tutorials/drag-up2.webp">
    <source src="/img/tutorials/drag-up2.webm" type="video/webm"/>
  </video>
</figure>

* Next, draw the alpha graph to be roughly a diagonal line.
  * The x-axis represents the brightness of the color (left = dark, right = bright).
  * The y-axis represents the opacity that we’ll draw with.
  * A diagonal line means low opacity (= high translucency) for dark colors, which makes the parts without ink transparent, so we can see the ink shine through.
  * You have to do the drawing in one go, because Fiji can hang for a while when it decides to rerender the image.

<figure>
  <video autoPlay playsInline loop muted className="w-[100%] max-w-[300px] rounded-xl" poster="/img/tutorials/alpha-graph2.webp">
    <source src="/img/tutorials/alpha-graph2.webm" type="video/webm"/>
  </video>
</figure>


* Now you can play with the *“Distance”* and *“Scale”* sliders.
  * *“Distance”* slices the volume away from the camera, revealing more of the inner structure.
  * *“Scale”* is simple zooming.
* See if you can uncover the spiral on the last page of the scroll:

<figure>
  <img src="/img/tutorials/spiral.webp" />
</figure>

Here is a video showing the whole process:

<figure>
  <video controls playsInline muted className="w-[100%] rounded-xl" poster="/img/tutorials/imagej-word-soup2.webp">
    <source src="/img/tutorials/imagej-word-soup2.webm" type="video/webm"/>
  </video>
</figure>

That is a what we call a “word soup” — letters floating in the air. Our real datasets aren’t nearly as pretty as this, but this should give you some idea of what volumetric data of papyrus scrolls looks like in the best case scenario.

However, it’s still hard to read anything. [“Tutorial 3: Segmentation and Flattening”](tutorial3) should help with this.

### Meshes

There is another way we can visualize the volume: we can convert it into triangles. A set of triangles is called a “mesh”, and it allows us to view our volume in tools that don’t support image stacks directly, which is sometimes useful. For this, we need to export to a mesh from Fiji:

* Go back to the tomographically reconstructed image stack in Fiji (or import it again using *“File > Import > Image Sequence”*).
* Go to *“Image > Type (at the very top) > RGB Color”* (this is required for the meshing algorithm).
* Go to *“File > Save As > Wavefront .OBJ (near the bottom)”*
* Now choose an intensity threshold above which we consider the object “solid”. A value of *“40”* works well for the campfire scroll.
* Press *“OK”* and save the file somewhere.
* Wait a moment until the file is saved. It should be around 400MB.

<figure className="max-w-[200px]">
  <img src="/img/tutorials/mesh-export2.webp" />
</figure>

Now, download and install [MeshLab](https://www.meshlab.net/#download).

* Use *“File > Import”*, or drag the file into MeshLab.
  * It can take a while to import; there is a progress bar at the bottom.
* In the right panel, click on the cylinder icon (see screenshot below), then click on *“Back-Face > Double”*.
* Drag around using the left mouse button to rotate the mesh.

<figure>
  <img src="/img/tutorials/meshlab.webp" className="rounded-xl" />
</figure>

You can quite clearly see the different pieces of papyrus, and even the container holding the scroll in place inside the scanner. However, all information of letters contained in the data is lost.

We’ll use Meshlab in later tutorials, so it’s useful to have seen it.

Here is a video where we navigate around the model in Meshlab:

<figure>
  <video controls playsInline muted className="w-[100%] rounded-xl" poster="/img/tutorials/meshlab2.webp">
    <source src="/img/tutorials/meshlab2.webm" type="video/webm"/>
  </video>
</figure>
