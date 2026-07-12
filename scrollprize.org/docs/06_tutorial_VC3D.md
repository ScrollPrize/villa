---
title: "Volume Cartographer 3D (VC3D)"
---

<head>
  <html data-theme="dark" />

  <meta
    name="description"
    content="Volume Cartographer 3D (VC3D) tutorial: install and launch the GUI, load scroll data, navigate volumes, and annotate fibers and windings."
  />

  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://scrollprize.org" />
  <meta property="og:title" content="Vesuvius Challenge" />
  <meta
    property="og:description"
    content="Volume Cartographer 3D (VC3D) tutorial: install and launch the GUI, load scroll data, navigate volumes, and annotate fibers and windings."
  />
  <meta property="og:image" content="https://scrollprize.org/img/social/opengraph.jpg" />

  <meta property="twitter:card" content="summary_large_image" />
  <meta property="twitter:url" content="https://scrollprize.org" />
  <meta property="twitter:title" content="Vesuvius Challenge" />
  <meta
    property="twitter:description"
    content="Volume Cartographer 3D (VC3D) tutorial: install and launch the GUI, load scroll data, navigate volumes, and annotate fibers and windings."
  />
  <meta property="twitter:image" content="https://scrollprize.org/img/social/opengraph.jpg" />
</head>

import ChatCallout from '@site/src/components/ChatWidget/ChatCallout';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Volume Cartographer 3D (VC3D)

*Last updated: July 12, 2026*

<ChatCallout prefill="Help me get started with VC3D" />

:::info[Draft]

This is an early draft that will eventually replace the [Volume Cartographer (Segmentation)](/tutorial_VC) tutorial. Many sections are still incomplete.

:::

:::note

VC3D is updated frequently. Follow along in Discord for the latest changes.

:::

## Installing VC3D
Downloads for all operating systems are available on the [releases page](https://github.com/ScrollPrize/villa/releases) of the villa repository.


<Tabs groupId="operating-systems">
  <TabItem value="mac-apple-silicon" label="Mac (Apple Silicon)">

  - Download the macos .dmg file from the [releases page](https://github.com/ScrollPrize/villa/releases) of the villa repository.
  - Double click the .dmg file, and drag/drop it into applications (**you may have to right click the dmg and click 'open'.**)

  <div className="mb-4 max-w-[760px]">
    <video autoPlay playsInline loop muted className="w-[100%] rounded-xl" poster="/img/tutorials/vc3d/macos-install-poster.webp">
      <source src="/img/tutorials/vc3d/macos-install.webm" type="video/webm"/>
    </video>
  </div>

</TabItem>
  <TabItem value="windows" label="Windows">

  - Download the windows installer .zip file from the [releases page](https://github.com/ScrollPrize/villa/releases) of the villa repository.
  - Extract the contents of the .zip file to a folder of your choice.
  - Double click the install file, click "show more details" and then "run anyway"
  - Follow the prompts, clicking yes or next where necessary
  
  </TabItem>
  <TabItem value="linux" label="Linux">
  
   Currently, the application must be built from source on linux. There is an install script at the volume-cartographer root 
   (villa/volume-cartographer/build_from_src_debian.sh) that will handle this build for you
    ```bash
    cd villa/volume-cartographer
    chmod +x build_from_src_debian.sh
    ./build_from_src_debian.sh
    ```
  
</TabItem>

<TabItem value="docker" label="Docker">
  Prebuild docker containers are hosted on the GitHub container registry. To use them, run the following command:
  ```bash
  docker pull ghcr.io/scrollprize/villa/volume-cartographer:stable
  ```

  </TabItem>
</Tabs>


## Launching the GUI

Depending on your install method or operating system, the application may be launched in different ways.
- **Mac:** Open the application from the Applications folder
- **Windows:** Open the application from the Start menu
- **Linux:** Navigate to the build folder and run the VC3D app. example: `cd build/bin && ./vc3d`

## Data

- **File → Open Data Catalog…**

<div className="mb-4 max-w-[760px]">
  <video autoPlay playsInline loop muted className="w-[100%] rounded-xl" poster="/img/tutorials/vc3d/data-catalog-poster.webp">
    <source src="/img/tutorials/vc3d/data-catalog.webm" type="video/webm"/>
  </video>
</div>

- Select a scroll → **Open Sample**
- Wait for any available segments to download

## Data Viewing and Navigation

- Use the **View** menu to customize the bottom access bar

<div className="mb-4 max-w-[760px]">
  <video autoPlay playsInline loop muted className="w-[100%] rounded-xl" poster="/img/tutorials/vc3d/view-tab-poster.webp">
    <source src="/img/tutorials/vc3d/view-tab.webm" type="video/webm"/>
  </video>
</div>

## Volume Package

<div className="mb-4">
  <img src="/img/tutorials/vc3d/volume-package.webp" className="w-[35%] rounded-xl"/>
</div>

- Select the segmentation directory that corresponds to your target volume
- Select the target volume
- Use the filter to reduce the active segments
  - Note: this helps improve GUI responsiveness if there are many segments
- Select any segment to view

<div className="border-l-4 border-red-500 bg-red-500/10 px-3 py-2 my-3 text-sm">
  <strong>Editor note:</strong> Unselecting segmentation directories seems to break VC3D — at least in testing with PHerc0814. Confirm before publishing.
</div>

## Viewer Controls

<div className="mb-4">
  <img src="/img/tutorials/vc3d/viewer-controls.webp" className="w-[55%] rounded-xl"/>
</div>

- **Focus point** — hover the mouse in 3D; the keybinds **Ctrl + Left click** or **r** move the focus point to those coordinates
- **Intersection thickness** — adjust segment line thickness in the 3D volume windows
- **Volume window** — manually threshold/window the volume data
- **Use axis-aligned slice planes**
- **Show axis overlays in XY view** — toggles visibility of the YZ axis plane adjustment tool
- **Max displayed resolution** — increase if streaming is slow

<div className="border-l-4 border-red-500 bg-red-500/10 px-3 py-2 my-3 text-sm">
  <strong>Editor note:</strong> Re "Use axis-aligned slice planes" — can we get rid of this? Also, can we hide the red axis adjuster when the pane isn't in view?
</div>

## Winding Annotation

VC3D can be used to create winding annotations for the spiral fit (see [the inputs section](tutorial_spiral#what-goes-in) of the spiral
fitting document for more details on how these are used). VC3D outputs these primarily as  *patches (segmentations)* or *point collections*. 

When generating data for the spiral, we can think of the inputs broadly as two types of annotations:
- same-winding annotations (fibers, patches, kolleisis, points along surface preds, etc)
- relative or different winding annotations (generally points which move outward radially *across* sheets rather than *along* them)

A same-winding annotation is simply some set of points which say "these points are all part of the same sheet of the scroll". Conversely, a
relative or different winding annotation is a set of points which say "these points are part of different sheets of the scroll, and they are this many windings apart"

These can be generated in an infinite number of ways. VC3D has built-in tools for generating a few examples of them: 

### Fibers (*same-winding*)

A tool to manually add control points to a line annotation (which we often use to target fibers), with interpolation and extrapolation assistance via lasagna normals.

- Note: lasagna normals must be present to use this tool

<div className="mb-4">
  <img src="/img/tutorials/vc3d/fiber-pane.webp" className="w-[75%] rounded-xl"/>
</div>

To start a line annotation: **Ctrl + Right click** on your starting point in the 3D volume, then select **Line Annotation**. The first time, you will be prompted to load the lasagna normals.

<div className="mb-4 max-w-[760px]">
  <video autoPlay playsInline loop muted className="w-[100%] rounded-xl" poster="/img/tutorials/vc3d/start-line-annotation-poster.webp">
    <source src="/img/tutorials/vc3d/start-line-annotation.webm" type="video/webm"/>
  </video>
</div>

A line annotation workspace will launch, with 4 viewers:

- **Top left:** XY plane
- **Top right:** orthogonal plane, similar to the YZ plane
- **Middle:** a flattened fiber view
- **Bottom:** a flattened fiber view, orthogonal to the middle plane

**To pan along a fiber:**

- Move the mouse pointer over the middle or bottom view
- **Shift + Wheel** while the mouse pointer is in one of the top viewers

<div className="mb-4 max-w-[760px]">
  <video autoPlay playsInline loop muted className="w-[100%] rounded-xl" poster="/img/tutorials/vc3d/fiber-annotating-poster.webp">
    <source src="/img/tutorials/vc3d/fiber-annotating.webm" type="video/webm"/>
  </video>
</div>

**To annotate:**

- Press **Spacebar** to freeze panning via the mouse pointer
- **Left click** to add a control point
- A reoptimization takes place after each control point is added
- To delete a control point, use **Ctrl + Right click → Delete control point**


### Patches

<div className="border-l-4 border-red-500 bg-red-500/10 px-3 py-2 my-3 text-sm">
  <strong>Editor note:</strong> I think we need to hide 95% of these settings and options.
</div>

<div className="mb-4 max-w-[760px]">
  <video autoPlay playsInline loop muted className="w-[100%] rounded-xl" poster="/img/tutorials/vc3d/seg-panel-poster.webp">
    <source src="/img/tutorials/vc3d/seg-panel.webm" type="video/webm"/>
  </video>
</div>

## Tracer

<div className="border-l-4 border-red-500 bg-red-500/10 px-3 py-2 my-3 text-sm">
  <strong>Editor note:</strong> Do we want or expect people to want to run tracer?
</div>

- Input: surface predictions
- Seeding, Expansion, Tracing
