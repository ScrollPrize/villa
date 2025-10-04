![Volume Cartographer](docs/imgs/banner.svg)

**Volume Cartographer** is a toolkit and set of cross-platform C++ libraries for
virtually unwrapping volumetric datasets. It was designed to recover text from
CT scans of ancient, badly damaged manuscripts, but can be applied in many
volumetric analysis applications.

This repository (a.k.a. **"VC3D"**) is a version of [EduceLab](https://educelab.engr.uky.edu/)'s original.
Via Vesuvius Challenge, it has gone through a few forks:
[EduceLab](https://github.com/educelab/volume-cartographer) (original) ->
[Julian Schilliger](https://github.com/schillij95/volume-cartographer-papyrus) (optical flow segmentation) ->
[Philip Allgaier](https://github.com/spacegaier/volume-cartographer) (performance/usability improvements) ->
[Hendrik Schilling and Sean Johnson](https://github.com/hendrikschilling/volume-cartographer) (various improvements) ->
this repository (developed and maintained by the Vesuvius Challenge team).
The original Volume Cartographer is a general purpose toolkit for virtual unwrapping, while this version is specialized for the frontier challenges of the Herculaneum papyri.

### Installation Instructions
Due to a complex set of dependencies, it is *highly* recommended to use the docker image. We host an up-to-date image on the [github container registry](https://github.com/ScrollPrize/villa/pkgs/container/villa%2Fvolume-cartographer) which can be pulled with a simple command :

```bash
docker pull ghcr.io/scrollprize/villa/volume-cartographer:edge
```

If you want to install vc3d from source, the easiest path is to look at the [dockerfile](ubuntu-24.04-noble.Dockerfile) and adapt for your environment. Building from source presently requires a *nix like environment for atomic rename support. If you are on Windows, either use the docker image or WSL. 

[installation instructions for docker](https://docs.docker.com/engine/install/)

[running docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/)


### Basic introduction to VC3D 
The document below is a basic introduction to the ui and keybindings available in VC3D , more detailed documentation is available in the [segmentation tutorial](https://scrollprize.org/segmentation) on the scrollprize website.
> [!WARNING]
> if you are using Ubuntu , the default open file limit is 1024, and you may encounter errors when running vc3d. To fix this, run `ulimit -n 750000` in your terminal.

### Launching VC3D

If you're using docker : 

```bash
xhost +local:docker
sudo docker run -it --rm \
  -v "/path/to/data/:/path/to/data/" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -e QT_QPA_PLATFORM=xcb \
  -e QT_X11_NO_MITSHM=1 \
ghcr.io/scrollprize/villa/volume-cartographer:edge
```

From source : 

```bash
/path/to/build/bin/VC3D
```

![vc3d initial ui](docs/imgs/vc3d_ui_initial.png)

### Data 
VC3D requires a few changes to the data you may already have downloaded. All data must be in OME-Zarr format, of dtype uint8, and contain a meta.json file. To check if your zarr is in uint8 already, open a resolution group zarray file (located at /path/to.zarr/0/.zarray) look at the dtype field. "|u1" is uint8, and "|u2" is uint16. 

The meta.json contains the following information. The only real change from a standard VC meta.json is the inclusion of the `format:"zarr"` key.
```json
{"height":3550,"max":65535.0,"min":0.0,"name":"PHerc0332, 7.91 - zarr - s3_raw","slices":9778,"type":"vol","uuid":"20231117143551-zarr-s3_raw","voxelsize":7.91,"width":3400, "format":"zarr"}
```
Your folder structure should resemble this: 
```
.
└── scroll1.volpkg/
    ├── volumes/
    │   ├── s1_uint8_ome.zarr -- this is your volume data/
    │   │   └── meta.json - REQUIRED!
    │   └── 050_entire_scroll1_ome.zarr -- this is your surface prediction/
    │       └── meta.json - REQUIRED!
    ├── paths 
    └── config.json - REQUIRED!
```
### Opening a volume package and navigating the UI
First , click `File -> Open volpkg` and select the volpkg you wish to work with (select the folder ending in .volpkg)

On the left side of the UI, you have a few dock widgets. The first being the Volume Package / Surface List. The most important things to note here are the `Volume` dropdown list, which selects the currently displayed volume, and the `paths` surface list, which selects the currently displayed segment. Selecting a `Surface ID` should populate the `Surface` volume view in the top left, and initialize the rest the volume viewers.

There are a number of widgets, each with their own purpose. From top to bottom, and left to right : 

**Volume Package**

_Main UI widget for interacting with the volume package and surface list_

<img src="docs/imgs/volume-package-ui.png" style="height: 500px">

- `Reload Surfaces` : checks for new segmentations in the currently selected directory (by default, this is {volpkg}/paths
- `Filters` : Apply a single (or multiple) filters to show or hide specific surface ids from the current set
  - the most important one to remember here is `Current Segment Only` , which will hide the intersection overlays of all other segmentations in the volume viewers (this can greatly speed up ui interaction)
- `Approved`, `Defective`, `Reviewed`, `Revisit`, `Inspect`  : tags which can be applied to the surface meta.json , and filtered against
- `Generate Surface Mask` and `Append Surface Mask` : Create a binary mask representing the valid surface, and optionally append the current selected volume to it as a multipage tif

**Location**

_Main UI widget for adjusting the ROI and display of the volume viewers_

<img src="docs/imgs/location-dock-ui.png" style="height: 500px">

- `Focus` : Displays the current location of the focus point (in XYZ) -- also can be used to set the focus point by modifying the values in the text box
- `Zoom` : Can be used in place of scroll wheel based zooming , mostly exists to alleviate touchy track pad zooming
- `Overlay Volume` : Selects the volume to overlay onto the base volume 
- `Overlay opacity` : Modifies the opacity of the overlay volume 
- `Axis overlay opacity` : Modifies the opacity of the plane slice axis overlays
- `Overlay threshold` : The value of the overlay array below which will not be rendered on the base volume (useful for removing background)
- `Overlay colormap` : The colormap to use for the overlay volume
- `Use axis aligned slices` : by default, vc3d attempts to align your XZ and YZ planes orthogonal to the normal of the current selected surface, this checkbox will instead use the plane normal as the axis to slice along
- `Show axis overlays in XY view` : if you do not wish to view the axis overlays in the XY view, this checkbox will disable them

_Keybinds:_
> - `Right mouse + drag` : pans the current viewer
> - `Scroll wheel` : zooms the current viewer
> - `Ctrl + left mouse button` : centers the focus point on the cursor position
> - `Scroll wheel click + drag` : rotates the slicing pane in the XZ or XY volume viewers
> - `Spacebar` : toggles the overlay on / off
> - `C` : toggle a composite view of the current surface in the segmentation window (parameters of which can be located in the "composite" dock tab, next to the location tab)

**Segmentation**

_Primary entry point for interacting with the segmentation, more info in the [segmentation docs](docs/segmentation.md)_
 
<img src="docs/imgs/segmentation-widget-ui.png" style="height: 500px">

- `Enable editing` : Must be checked to enable any of the actions in this widget. Creates a copy of the base surface upon which we perform any of the following actions
- `Surface Growth` - grouping of settings mostly pertaining to the `Grow` action
  - `Steps` : The number of "generations" the growth action will undertake
  - `Allowed Directions` : Limits the directions of growth _relative to the flattened 2d quad surface!_ (i.e if you look in the segmentation window in the top left, and you set 'up' it will grow towards the top of the window)
  - `Limit Z range` : Constrains the growth to a selected `Z` range
  - `Volume` : the volume to pass to the `tracer()` call _should be a surface prediction volume!_ 
- `Editing` - grouping of settings which mostly apply to drag/push/pull actions for mesh deformation. All mesh deformation-type actions are performed on a gaussian-like area which has a circular shape centered at the current mouse location and whose strength is reduced as we reach the edges. _Radius is in quad vertices in the 2d flattened surface._
  - `Max Radius` : the maximum radius of the area to be affected by the action
  - `Sigma` : the _strength_ of the push/pull on affected vertices other the original one (aka how quickly the influence tapers as we step away from the original point)
  - `Push Pull Step` : The number of "offset steps" to take along the surface normal (in voxels) for each push/pull action (this parameter only affects the push/pull in the segmentation window that is applied with `F` and `G`)
- `Direction Fields`
  - `Zarr folder` : the path containing the direction field you want to use for optimization (ex: `/path/to/scroll.volpkg/fiber-directions.zarr/horizontal`) _these are optional_ 
    - `Orientation` : the orientation/type of the direction field (from horizontal, vertical, or normal)
    - `Scale level` : the zarr resolution from which these were computed
    - `Weight` : the weight to apply to the field when optimizing.
- `Corrections` : group which controls / manages the "correction points" for the current growth session
  - `New correction set` : creates a new point collection containing a single "correction" 
      
_Keybinds:_
> - `F` or `G` : push/pull the current surface in a positive or negative direction along the surface normal (only works in the surface
> - `1`, `2`, `3`, `4`, `5` : select a direction to grow in -- left, up , down, right , all , respectively
> - `T` : create a new correction set

**Seeding** 

_Widget for creating initial segmentations which can be used for traces or later growth actions_

<img src="docs/imgs/seeding-ui.png" style="height: 500px">

- `Switch to point mode` : toggles the seeding widget into draw mode (not recommended)
- `Source Collection` : the source point collection to use as seed points (will autofill if you draw/analyze)
- `Parallel processes` : the number of processes to run seeding with
- `OMP Threads` : limits the amount of threads each process can use (recommend to set this to 1)
- `Intensity threshold` : mostly unused, can leave default
- `Peak detection window` : how closely each peak detected along a path/raycast can be
- `Expansion iterations` : how many iterations to run expansion mode on after initial seeding
- `Show Preview Points` : unused , can ignore
- `Clear` (both) : clears current points , paths, or raycasts
- `Analyze Paths` : after drawing paths, detect peaks along the line segment on which to place seed points
- `Run seeding` : creates segmentations from the current seed points
- `Expand seeds` : expands the current seed points to the given number of iterations
- `Clear all paths` : clears any user drawn paths / points (use this over the clear buttons most of the time)
- `Start label wraps` : this is used for absolute or relative wrap labels , is not used during the seeding step, will detail in another document
