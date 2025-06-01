---
title: "Tutorial: Segmentation"
sidebar_label: "Segmentation"
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

*Last updated: February 20, 2025*

This tutorial is a practical walkthrough for two of the methods reviewed in the [virtual unwrapping guide](unwrapping). Both of these methods are in active development and as such the content of this page may become outdated (let us know so we can fix it!). For a high-level introduction to the problem these approaches are aiming to solve, check out that [guide](unwrapping).

Both of these methods have pros and cons detailed in the previous walkthrough, in terms of current and future performance. Currently implemented, the primary benefit of the VC3D tracer solution is that the intermediate steps produce immediately usable partial segments, and the entire pipeline save ink detection is self-contained within the VC3D ecosystem. *If you are looking to simply produce exploratory segments for ink detection or other tasks, the tracer has the lowest "startup" time.*

:::tip
If you want to try out running traces only, and do not want to perform any of the seeding or expansion steps, you can download patches we have generated for some of the scrolls from the links in the [data section](#volume-data), place these in your paths directory, place the uint8 volumes and surface predictions in the volume directory, and skip to [running a trace](#running-a-trace).
:::

import TOCInline from '@theme/TOCInline';

**Table of Contents**
<TOCInline
  toc={toc}
  maxHeadingLevel={4}
/>

## Sheet Tracing (VC3D)
<div className="mb-4">
  <img src="/img/segmentation/vc3d_gui.png" className="w-[85%]"/>
</div>
___

### Installation
This guide will be written for Ubuntu 24.04, for other operating systems the details may vary slightly. Windows users may benefit from using Windows Subsystem For Linux(WSL2), and placing their volume data within the Linux filesystem. 

**Install Docker Engine for Linux:**

Follow the installation steps [here](https://docs.docker.com/engine/install/ubuntu/).

Ensure your installation is complete by entering. If you get the 'hello world' output from Docker, you're good to go. 

```bash
sudo docker run hello-world
```

**Pull the container image**

While it is possible to build this from source, we recommend just pulling the container image from our [volume-cartographer repository](https://github.com/ScrollPrize/volume-cartographer) by entering the following command:
```bash
sudo docker pull ghcr.io/scrollprize/volume-cartographer:edge
```

**Launch VC3D**

To launch the docker image, simply type 

```bash
xhost +local:docker
sudo docker run -it --rm \
  -v "/path/to/data/:/path/to/data/" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -e QT_QPA_PLATFORM=xcb \
  -e QT_X11_NO_MITSHM=1 \
ghcr.io/scrollprize/volume-cartographer:edge
```
The -v flag is used to mount a folder to the docker image, allowing you to use files that exist on your local filesystem. It will be mounted under the path following the ':'. You can place as many folders here as you would like.

:::tip
If you map the docker path exactly the same as it exists on your local system, you can use command snippets in either place without modification. This will also _greatly_ help troubleshooting path issues.
:::

You'll now have a terminal open in the running Docker container. To open the GUI type 
```bash
OMP_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE OMP_NESTED=FALSE nice ionice VC3D
```
Note that while the OMP environment variables and `nice ionice` are not necessary, for now these will make VC3D more responsive when working with very large amounts of patches.
___

### Preparing Data
VC3D requires a few changes to the data you may already have downloaded. All data must be in OME-Zarr format, of dtype uint8, and contain a meta.json file. To check if your zarr is in uint8 already, open a resolution group zarray file (located at /path/to.zarr/0/.zarray) look at the dtype field. "|u1" is uint8, and "|u2" is uint16. 

The meta.json contains the following information. The only real change from a standard VC meta.json is the inclusion of the `format:"zarr"` key.
```json
{"height":3550,"max":65535.0,"min":0.0,"name":"PHerc0332, 7.91 - zarr - s3_raw","slices":9778,"type":"vol","uuid":"20231117143551-zarr-s3_raw","voxelsize":7.91,"width":3400, "format":"zarr"}
```

To convert an already downloaded Zarr of dtype uint16 to uint8, download the zarr_to_ome.py file located in @khartes_chuck's scroll2zarr repository [here](https://github.com/KhartesViewer/scroll2zarr) and use the following command 
```bash
python zarr_to_ome.py /path/to/input.zarr /path/to/output.zarr --bytes 1
```
:::tip
if you are performing this on "indicator" data (such as binary predictions), use the flag `--algorithm max` or `--algorithm nearest`. When downsampling with the default 'mean' algorithm, the values of the neighboring pixels are considered when writing the lower resolution array. If we have a pixel of value 255, and the pixels around it are mostly valued 0, we might get a pixel of value 125. This obviously is not desired.
:::

If you do not wish to perform these, you can use the standardized volumes, links are below for these. I'd recommend Scroll 3 if you are looking to conserve storage, as it's only around 100gb all-in.

If you want to use the original data and do not want to run the processing steps, we have converted some of the scrolls into uint8 and placed the meta.json files within. These links will be updated as new predictions become available. You do not need both versions, you only need one version of the volume data.

The data should be placed within the volpkg of the respective scroll, in this format:
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
#### Volume Data
  * Scroll 1 
    * Raw uint8 not yet available, in progress.
    * [Standardized](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/)
    * [Surface Predictions](https://dl.ash2txt.org/community-uploads/bruniss/scrolls/s1/surfaces/full_scroll/050_entire_scroll1_ome.zarr.zip) - Updated 29 Dec 2024
    * [Patches](https://dl.ash2txt.org/community-uploads/bruniss/scrolls/s1/s1_patches.7z) - Updated 02 Feb 2025
  * Scroll 2 - Not available
    * [Standardized](https://dl.ash2txt.org/full-scrolls/Scroll2/PHercParis3.volpkg/volumes_zarr_standardized/)
  * Scroll 3  
    * Raw uint8 not yet available, in progress.
    * [Standardized](https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/volumes_zarr_standardized/)
    * [Surface Predictions](https://dl.ash2txt.org/community-uploads/bruniss/scrolls/s3/surfaces/s3_multi-ensemble_ome.7z) - Updated 26 Jan 2025
    * [Patches](https://dl.ash2txt.org/community-uploads/bruniss/scrolls/s3/s3_patches.7z) - Updated 06 Feb 2025
  * Scroll 4
    * Raw uint8 not yet available, in progress.
    * Standardized not yet available, in progress.
    * [Surface Predictions](https://dl.ash2txt.org/community-uploads/bruniss/scrolls/s4/surfaces/s4_multi-ensemble_ome.7z) - Updated 28 Jan 2025
    * [Patches](https://dl.ash2txt.org/community-uploads/bruniss/scrolls/s4/s4_patches.7z) - Updated 01 Feb 2025
  * Scroll 5  
    * [Raw uint8](https://dl.ash2txt.org/community-uploads/bruniss/scrolls/s5/s5_masked_ome.zarr/)
    * [Standardized](https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr_standardized/)
    * [Surface Predictions](https://dl.ash2txt.org/community-uploads/bruniss/scrolls/s5/surfaces/s5_055_surfaces.7z) - Updated 10 Feb 2025
    * [Patches](https://dl.ash2txt.org/community-uploads/bruniss/scrolls/s5/s5_patches.7z) - Updated 09 Feb 2025

___

### Seeding the Volume
:::warning
If you are running these steps in a docker container that was started using `sudo`, the files generated during these steps will be owned by root. If you encounter write errors in subsequent steps, change the ownership of the folder by typing 
```bash
sudo chown -R username /path/to/folder
```
If you encounter "json" errors, its likely you have entered your paths wrong, or forgot to include a meta.json! If you cannot view the zarr within the vc3d gui, you may be missing the `format:zarr` key!
:::

Much of this section borrows from Hendrik Schilling's repository [here](https://github.com/hendrikschilling/FASP), which also contains more detailed explanations of these processes.

The tracer operates by connecting overlapping patches based on a number of conditions. It chooses the initial candidates to connect based on patch overlap, and then uses some constraints defined within the cost functions to find the optimal patch to select. This step places our initial seeds, which are later expanded. The app that generates the seeds and the expanded patches is `vc_grow_seg_from_seed`.

During the tracing , a euclidean distance transform is performed, and this is stored within a cache, defined in the json params. This can occupy a non-neglible amount of space, so ensure your disk has enough storage -- my Scroll 1 cache is 376gb. This (and each subsequent step) involves _a ton_ of disk i/o , so a very fast disk here is a good choice.

The seed step uses a json file for configuration, which looks like this: 
```json
{
    "cache_root": "/mnt/raid_nvme/scroll3.volpkg/ensemble_cache", 
    "thread_limit": 1,
    "min_area_cm": 0.3, 
    "generations": 200
}
```

We begin the volume seeding by entering the following command , replacing the paths with your own. 
```bash
time seq 1 5000 |  xargs -i -P 8 bash -c 'nice ionice bin/vc_grow_seg_from_seed /mnt/raid_nvme/volpkgs/scroll3.volpkg/volumes/s3_multi_ensemble_ome.zarr /mnt/raid_nvme/volpkgs/scroll3.volpkg/paths /mnt/raid_nvme/volpkgs/scroll3.volpkg/seed.json' || true
```

This will run vc_grow_seg_from_seed five thousand times, concurrently using 8 processes. You can change these values if desired. This is ran 5,000 times because each run may not produce a segment that meets our min_area_cm defined in the json. Each time a succesful seed is generated, it will be placed in your /paths/ directory. We'd like to finish this seeding step with the following counts for seeds:

* Scroll 1, 2, and 5 (Entire Scroll) - 5,000
* Scroll 3 and 4 - 1000-1500

You can watch the progress of the seeding by opening another terminal window and entering:
```bash
watch -n 1 "ls -l /path/to/paths | wc -l"
```
This will refresh every 1 second, and count the number of folders in your /paths/ directory. 
:::tip
You may need to run the seed command multiple times to get a sufficient number of seeds before proceeding to the next step.
:::

___
### Expanding Seeds
:::warning
Seeds created using the mode "seed" do not contain overlap information and are not used by the tracer. Ensure that your params json 'mode' key is set to 'expansion' for this step.
:::
In this step, each initial seed is visited in an attempt to generate overlapping patches. The new patches will grow out from these initial seeds, and should eventually populate the entire scroll volume. Expansions are generated until each patch has a minimum number of overlapping patches associated with it. Overlaps are marked by creating a symlinked file in the /auto_grown_xxxx/overlaps directory of the respective patch. Expansion mode uses a config json, which contains the following keys:
```json
{
    "mode" : "expansion",
    "cache_root": "/mnt/raid_nvme/scroll3.volpkg/ensemble_cache",
    "tgt_overlap_count": 10,
    "min_area_cm": 0.30,
    "thread_limit": 1,
    "generations": 200
    
 }
```

Begin the expansion step by typing: 

```bash
time seq 1 10000000 |  xargs -i -P 8 bash -c 'nice ionice bin/vc_grow_seg_from_seed /mnt/raid_nvme/volpkgs/scroll3.volpkg/volumes/s3_multi_ensemble_ome.zarr /mnt/raid_nvme/volpkgs/scroll3.volpkg/paths /mnt/raid_nvme/volpkgs/scroll3.volpkg/seed.json' || true
```

You may need to repeat this command, until the volume is densely seeded. The number of patches required is somewhat difficult to approximate as better predictions generate larger patches, which require less seeds. You will need to open vc3d and inspect the volume to see it is sufficiently seeded. As an example, this Scroll 1 volume contains 68,000 seeds and could still use some more. 

<div className="mb-4">
  <img src="/img/segmentation/s1_seeded.png" className="w-[50%]"/>
  <figcaption className="mt-[-6px]">Scroll 1 with generated patches.</figcaption>
</div>

___

### Running a Trace
A "Trace" in the context of vc3d is essentially the same as a segmentation as we've come to know them. 

This step will finally create larger connected surfaces. The large surface tracer generates a single surface by tracing a "consensus" surface from the patches generated earlier. This processed can be influenced by annotating patches in several ways which allows to guide the tracer to avoid errors. Hence the process looks like this:

1. pick a seed patch
2. generate a trace
3. check for errors
4. annotation
5. and repeat step 2-4 or 1-4 several times

Keep previous trace rounds around as the fusion step can later join multiple traces and problematic areas of a trace can be masked out. A mask can be generated for example with VC3D, or by using any 8-bit grayscale image with the same aspect ratio as one of the tiffxyz channels.

**Locate a seed patch**
The choice here is somewhat arbitrary, but the initial condition is important. You want a seed here that does not skip sheets, and is of reasonable size.

<div className="mb-4">
  <img src="/img/segmentation/s3_good_seed.png" className="w-[100%]"/>
  <figcaption className="mt-[-6px]">A "good" seed".</figcaption>
</div>

By placing patches generated in the last step into the paths directory of the volpkg VC3D allows to inspect them. VC3D can also display the prediction ome-zarrs, which, together with fiber continuity allows to quickly scan for obvious errors. Useful tools for navigation:
* ctrl-click in any slice view to focus and slice on the clicked point
* shift-click to place a red POI
* shift-ctrl-click to place a blur POI
* filter by focus point & filter by POIs to narrow down the choice of patches

<div className="mb-4">
  <img src="/img/segmentation/s3_bad_seed.png" className="w-[100%]"/>
  <figcaption className="mt-[-6px]">A "bad" seed".</figcaption>
</div>

**Generate the trace**

The trace uses a config json with the following parameters:
```json
{
        "flip_x" : false,
        "global_steps_per_window" : 3,
        "step" : 10,
        "consensus_default_th" : 10,
        "consensus_limit_th" : 2
}
```
* flip_x determines the direction of the trace (it always grows to the right, but that can go to the inside or outside of the scroll, depending on seed location).
* global steps per window: number of global optimization steps per moving window. The tracer operates in a moving window fashion, once the global optimization steps were run per window and no new corners were added the window is moved to the right and the process repeated. At the beginning use 0 global steps to get a fast and long trace and see if there are any coarse errors. Set to 0 to get a purely greedy but quite fast trace.
* consensus_default_th: lowest number of inliers (patch "supports") required per corner to be considered an inlier. Note that a single well-connected patch gives more than a single support to a corner (so this is not the number of surfaces). Maximum of 20 to get only well-connected patches, minimum of 6 before there are a lot of errors. For the submission values of 6 and 10 were used.
* consesus_limit_th: if we could otherwise not proceed go down to this number of required inliers, this will only be used if otherwise the trace would end.

Begin the trace by typing:

```bash
bin/vc_grow_seg_from_segments /mnt/raid_nvme/volpkgs/scroll3.volpkg/volumes/s3_multi_ensemble_ome.zarr /mnt/raid_nvme/volpkgs/scroll3.volpkg/paths /mnt/raid_nvme/volpkgs/scroll3.volpkg/paths/ /mnt/raid_nvme/volpkgs/scroll3.volpkg/params.json /mnt/raid_nvme/volpkgs/scroll3.volpkg/paths/auto_grown_20250211031737772
```
The tracer will generate a debug/preview surface every 50 generations (labeled z_dbg_gen...) and in the end save a final surface, both in the output dir. You can watch the trace grow by opening and closing vc3d to repopulate the segments and selecting the highest number "dbg_gen" to view it in the 2d window in the top left. If the trace starts growing opposite of the direction you intended, change "flip_x" to the opposite of what you had set (for ex: true to false)

<div className="mb-4">
  <img src="/img/segmentation/s3_sample_trace.png" className="w-[100%]"/>
  <figcaption className="mt-[-6px]">Scroll 3 trace "in-progress".</figcaption>
</div>

**Check for errors**

Using the VC3D GUI, the traced surfaces can be inspected for errors. Inspecting with an ome-zarr volume containing the original scan data (to see fiber continuity) is suggested. You can switch your volume by selecting the drop-down below the segments.

The errors that need to be fixed are generally sheet jumps, where the surface jumps from one correct surface to another correct surface. Often these are visible by checking for gaps in the generated trace as a jump will normally not reconnect with the rest of the trace. 

Typically, the easiest method for fixing these is to locate where the sheet jump first occurs, and then ctrl+click on this location, filter segments by focus point, and then mask out the offending patches.

<div className="mb-4">
  <img src="/img/segmentation/sheet_jump.jpg" className="w-[75%]"/>
  <figcaption className="mt-[-6px]">"Sheet Jump" in a patch".</figcaption>
</div>

___ 

### Fixing Errors

VC3D allows to annotate patches as approved, defective, and to edit a patch mask which allows masking out areas of a patch that are problematic.

<div className="mb-4">
  <img src="/img/segmentation/vc3d_annotation.jpg" className="w-[50%]"/>
</div>

**Defective**

A patch can be marked as "defective" by checking the checkbox in the VC3D side panel. This is fast but not very useful as most patches have some good areas and also will have some amount of errors. Errors only matter if they fall at the same spot in multiple patches, so marking a whole patch as defective, which will make the tracer ignore it completely is not necessary. **_Use sparingly, most patches contain at least some good data_**

**Approved**

Checking the approved checkbox will mark a patch as manually "approved" which means the tracer will consider mesh corners coming from such a patch as good without checking for other patches. It is important that such a patch is error free (which is not necessary when creating a mask to only remove a problem area without checking "approved"). If you must mark one approved, the following process can be used: 

* ctrl-click on a point in the area that shall be "correct".
* follow along the two segment slices and place blue POIs at an interval wherever you are sure the trace follows the correct sheet.
* place red POIs where errors occur. This process will place a "cross" of two lines which show the good/bad areas of the patch
* ctrl-click to focus on a point on/between blue points to generate a second line, this way a whole grid of points is generated
* use this grid as orientation to create a mask in GIMP

**Masking**

By clicking on the "edit segment mask" button a mask image will be generated and saved as .tif in the segments directory. Then the systems default application for the filetype .tif will be launched. It is recommended to use GIMP for this. The mask file is a multi-layer tif file where the lowest layer is the actual binary mask and the second layer is the rendered surface using the currently selected volume.

* place one POI on one side of the jump and a second on the other.
* select "filter by POIs" to get a list of patches which contain this sheet jump, this probably list about 5-20 patches
* press "edit segment mask"
* GIMP will open an import dialog, the default settings are fine so just press import
* click on the layer transparency to make the upper layer around 50% transparent
* your default tool should be the pencil tool with black as color and a size of around 30-100 pixels. Use it to mask out offending areas on the lower layer (the actual mask), refer back to VC3D if you are unsure.
* save the mask by clicking "File->overwrite mask.tif"

With this process a mask can be generated using less than 10 clicks

<div className="mb-4">
  <img src="/img/segmentation/gimp_masking.jpg" className="w-[100%]"/>
</div>

___

### Inpainting

It is possible after the previous steps to simply render the trace and use it within downstream tasks. The traces at this stage however frequently contain holes and are not flattened as well as they could be. To improve the mesh, we can combine multiple traces and attempt to fill in the holes. 

The traces to be combined should be error free, and one can mask out bad regions of these larger traces using the same masking process as we used on the patches.

**Generating Winding Number Assignments**

<div className="mb-4">
  <img src="/img/segmentation/wind_vis.png" className="w-[100%]"/>
</div>

The first step in the fusion process is generating relative winding numbers of each trace by running:

```bash
OMP_WAIT_POLICY=PASSIVE OMP_NESTED=FALSE \
vc_tifxyz_winding /path/to/trace
```

Which will generate some debug images and the two files "winding.tif" and "winding_vis.tif". Check the winding vis for errors, it should be a smooth continuous rainbow going from left to right. If you see discontinuities here, there were some errors in the trace that must be fixed. Mask those errors in the source trace and re-run winding estimation until it works. This should not generally be necessary.

Copy the winding.tif and wind_vis.tif to the traces storage directory so it's all in one place and ready for the next step. The winding estimation should take about 10s.

**Trace Fusion and Inpainting**

Once you have estimated winding assignments for all the traces you intend on fusing, you can run this command with an arbitrary number of traces. Note that the first trace will be used as the seed and it will also define the size of the output trace and will generate normal constraints, so it should be the longest and most complete. The number after the trace is the weight of the trace when generating the surface, in all tests a weight of 1.0 was used and for the submission this params.json.

```bash
OMP_WAIT_POLICY=PASSIVE OMP_NESTED=FALSE time nice \
vc_fill_quadmesh params.json /path/to/trace1/ /path/to/trace1/winding.tif 1.0 /path/to/trace2/ /path/to/trace2/winding.tif 1.0
```
The params.json contains the following keys:
```json
{
    "trace_mul" : 5,
    "dist_w" : 1.5,
    "straight_w" : 0.005,
    "surf_w" : 0.05,
    "z_loc_loss_w" : 0.002,
    "wind_w" : 10.0,
    "wind_th" : 0.5,
    "inpaint_back_range" : 60,
    "opt_w" : 4
}
```

**Grounding and Upscaling**
The output of this fusion process is a lower resolution than the source traces. To upscale it back to its source resolution, run the following command

```bash
vc_tiffxyz_upscale_grounding <infill-tiffxyz> <infill-winding> 5 /path/to/trace1/ /path/to/trace1/winding.tif ...
```

___

### Rendering
Each stage of the seeding/expanding/tracing process uses the tifxyz format, and thus any output in these processes can be rendered by using the `vc_render_tifxyz`. To render your new trace, enter:
```bash
OMP_WAIT_POLICY=PASSIVE OMP_NESTED=FALSE time nice \
vc_render_tifxyz /path/to/volume/ome-zarr /output/path/%02d.tif /path/to/trace 0.5 1 21
```
where 0.5 is the "scale" (the final output is scaled down to this percentage, 50% in the example), 1 is the source resolution group (in this case, 1 , or the second resolution of the ome-zarr), and 21 is the number of "layers" from the center slice we want, equally in positive and negative directions. This will generate 21 slices from our traditionally understood "layer 22" to "layer 43".

:::tip
For the TimeSFormer model, the "1" resolution performs only slightly worse than the full resolution, "0". The i3d first letters model however, performs much worse on downsampled data. For models of this type, it is suggested to render the full resolution. This requires processing 8x more data, and will render significantly slower.
:::

___
### Ink Detection
Ink detection on the produced segmentations is no different at this stage than it was on previous segments, although some of the segmentations generated by the tracer can be quite large. The [ink-detection](https://github.com/ScrollPrize/villa/tree/main/ink-detection) page of the villa monorepo contains the 2023 Grand-Prize model, with some recent updates. If you have difficulty loading large segmentations with this version, a fork exists which was used to generate the ink detections for @waldkauz and @bruniss 2024 year-end submission , documented [here](https://discord.com/channels/1079907749569237093/1315006782191570975). 

<div className="mb-4">
  <img src="/img/segmentation/v4_hr.jpg" className="w-[100%]"/>
<figcaption className="mt-[-6px]">2023 grand prize region trace and ink detection.</figcaption>
</div>

___

### Common Issues
**When attempting to run one of these steps, you encounter "json" errors**
  * Fix: 99% of the time when this error is thrown, it is the result of an incorrect path in the command, or a missing meta.json file in the target volume. Check your paths.
<div className="ml-8">
  <img src="/img/segmentation/json_error.png" className="w-[75%]"/>
</div>

**When running subsequent traces after an initial trace , a write error is thrown**
    * Fix: The tracer cannot write to the folder because a folder with the same name exists. Clear your paths directory of "z_dbg_gen" files, and try again.

**When attempting to run a trace on a verified existing seed patch, and no z_dbg_gen files exist, a permissions or write error is thrown**
    * Fix: If you launched docker using sudo, your patches are owned by sudo and thus your user does not have permissions. Type `sudo chown -R username /path/to/paths/`

**Trace ends abruptly or before expected considering seed density**
    * Fix: If you are running into issues getting your trace to "take off", verify you are heading in the direction of a densely seeded region. If you are starting from the umbilicus or the outside of the scroll, verify you are not trying to trace outside of the volume. Change your `flip x:` key in the params.json.
    * Note: If you are sure your `flip x:` key is properly set, and still cannot get a trace to go from a good initial seed, you can try lowering the `step` parameter. This increases processing time significantly, but can help traces in regions of high curvature.

**Cannot view zarr in VC3D gui (may just show a black screen), or cannot select it in the volumes drop-down**
    * Fix: ensure that you have a meta.json in the zarr's root, and that it contains `"format":"zarr"`
    * Ensure that the volume is in uint8, or `|u1` if looking at the `.zarray` file

**Seed or Expand steps are running slow on VC3D built from source**
    * Fix: on most ubuntu linux installations, at least one of two file indexers are typically installed -- `tracker-miner-fs` or `rygel`. Either of these will attempt to index every single file you add, and during the seed/expansion steps, this is a very large number. You will want to "mask" or disable these services, as they will slow the seed/expansion steps to a crawl. 
___

### Tips and Tricks

**"Checkpointing"**

If you are attempting to create very large traces, it is generally smart to "checkpoint" your progress as you go by running a relatively long trace, allowing it to continue through errors, and then utilizing your dbg_gen files as "superpatches". The process looks like this:
  - Run a trace
  - Starting from your longest one, identify large contiguous regions of papyrus without errors
  - Edit the mask, and remove all the areas other than this
  - Rename the folder to "auto_grown_xxx"
  - Update the meta.json with this new name
  - Run `vc_seg_add_overlap /path/to/paths /path/to/trace`
  - Mark the segment as "approved" in the VC3D gui

**Manual Seeding**

It can be highly beneficial to manually seed volumes. The random seeding is just that, random. It places seeds at nonzero points throughout the volume, and these points can include the case, the lining, or regions that are far from the current location you're targeting. If you wish to manually seed, simply open the volume in VC3D, the simplest way is likely the following:
  - ctrl+click on the predicted surfaces around the area you're trying to segment. Ensure you don't miss any sheets. As you click these points, the 3d coordinates will be printed into the terminal. 
  - Copy the script below and paste it into a file called `create_seed.sh`, changing the paths to your own 
  - Run this script from a separate terminal
  - Paste the outputs of the coordinates (including the words before them) into the terminal running `create_seed`
  - The script will run vc_grow_seg_from_segments in explicit seed mode at these locations
  - Verify that a seed has been created on each surface around the target region, and that none have been missed
  - If necessary, repeat the process until you have a seed on each wrap. 
  - Run vc_grow_seg_from_seed in expansion mode as detailed earlier in the tutorial

The primary motivation for doing this is that if you run expansion mode on randomly created seeds, a significant portion of your compute time is spent expanding seeds into regions you might not care about, or expanding seeds along the case. While it's generally true that if you leave expansion mode running long enough, it will fill the volume, this may take a while. It is also not possible for a seed to expand if the surface itself is not sufficiently connected to other surfaces -- it is not uncommon to miss large parts of wraps if you do not verify and manually seed.


```bash
#!/bin/bash
# Configuration variables - update these as needed:
OME_ZARR_VOLUME="/home/sean/Documents/volpkgs/scroll3.volpkg/volumes/s3_059_medial_ome.zarr/"
TGT_DIR="/home/sean/Documents/volpkgs/scroll3.volpkg/paths/"
JSON_PARAMS="/home/sean/Documents/volpkgs/scroll3.volpkg/seed.json"
CMD="bin/vc_grow_seg_from_seed"

# Offset variable (in voxels)
OFFSET=30

# Maximum number of parallel jobs to run concurrently
MAX_JOBS=16

# Check for bc availability. If not available, fallback to awk.
if command -v bc >/dev/null 2>&1; then
    use_bc=1
else
    echo "bc command not found. Falling back to awk for arithmetic."
    use_bc=0
fi

# Function to wait until background jobs are below MAX_JOBS
wait_for_slot() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 0.5
    done
}

while true; do
    echo "Paste coordinate lines (end with an empty line):"
    lines=()
    while IFS= read -r line; do
        [[ -z "$line" ]] && break
        lines+=("$line")
    done

    if [ ${#lines[@]} -eq 0 ]; then
        echo "No coordinates provided. Try again."
        continue
    fi

    # Process each pasted line
    for line in "${lines[@]}"; do
        # Extract text between [ and ]
        coords=$(echo "$line" | grep -o '\[[^]]*\]')
        if [ -z "$coords" ]; then
            echo "No coordinates found in: $line"
            continue
        fi

        # Remove the square brackets and split into x, y, z
        coords=${coords#[}
        coords=${coords%]}
        IFS=',' read -r x y z <<< "$coords"
        x=$(echo "$x" | xargs)
        y=$(echo "$y" | xargs)
        z=$(echo "$z" | xargs)
        
        echo "Processing coordinate: x=$x, y=$y, z=$z"

        # Loop only over x offsets: -OFFSET, 0, and +OFFSET
        for dx in -$OFFSET 0 $OFFSET; do
            # Ensure we don't exceed the parallel job limit
            wait_for_slot

            # Calculate new seed position for x; y and z remain unchanged.
            if [ $use_bc -eq 1 ]; then
                seed_x=$(echo "$x + $dx" | bc)
            else
                seed_x=$(awk "BEGIN {print $x + $dx}")
            fi
            seed_y=$y
            seed_z=$z

            echo "Launching: nice ionice $CMD $OME_ZARR_VOLUME $TGT_DIR $JSON_PARAMS $seed_x $seed_y $seed_z"
            nice ionice $CMD "$OME_ZARR_VOLUME" "$TGT_DIR" "$JSON_PARAMS" "$seed_x" "$seed_y" "$seed_z" || true &
        done
    done

    # Wait for all background jobs to finish before prompting again
    wait
    echo "Finished processing these coordinates."
    echo
done

```



## Spiral Fitting (Coming Soon)
