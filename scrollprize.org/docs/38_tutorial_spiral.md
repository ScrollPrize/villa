---
title: "Tutorial: Spiral Fitting"
sidebar_label: "Spiral Fitting"
---

<head>
  <html data-theme="dark" />

  <meta
    name="description"
    content="Vesuvius Challenge spiral fitting tutorial: fit a single, globally coherent surface to an entire Herculaneum scroll by deforming an ideal spiral to match segments, fibers, and winding annotations."
  />

  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://scrollprize.org" />
  <meta property="og:title" content="Vesuvius Challenge" />
  <meta
    property="og:description"
    content="Vesuvius Challenge spiral fitting tutorial: fit a single, globally coherent surface to an entire Herculaneum scroll by deforming an ideal spiral to match segments, fibers, and winding annotations."
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
    content="Vesuvius Challenge spiral fitting tutorial: fit a single, globally coherent surface to an entire Herculaneum scroll by deforming an ideal spiral to match segments, fibers, and winding annotations."
  />
  <meta
    property="twitter:image"
    content="https://scrollprize.org/img/social/opengraph.jpg"
  />
</head>

import ChatCallout from '@site/src/components/ChatWidget/ChatCallout';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


*Last updated: September 16, 2026*

<ChatCallout prefill="Walk me through the spiral fitting tutorial" />

Most of our segmentation tools work bottom-up. [GrowPatch](2026_open_problems#normal-grids-growpatch-and-local-tracing), [lasagna](2026_open_problems#lasagna-smoother-optimization-of-one-or-more-sheets), and [manual segmentation in VC3D](tutorial_VC3D) all produce *patches* — pieces of papyrus surface that you grow bigger and bigger until they hit a tricky region and stall. Other tools trace individual fibers. Either way you end up with a big pile of small pieces: segments, fibers, point annotations. What we really want is the *whole scroll* — one surface covering every winding of the original papyrus sheet, from the center to the outer shell. However, gluing the pieces together directly is hard, especially where there are gaps between them. [^tracer]

That is what the spiral fit does. It takes the whole pile of partial evidence — surface patches, traced lines, winding annotations, volumetric predictions — and fits a single, globally coherent surface for the entire scroll that agrees with as much of that evidence as possible. Where the evidence is dense, the fitted surface follows it closely; where there are gaps, the spiral bridges them smoothly instead of stopping or leaving a gap.

<div className="mb-4 max-w-[720px] mx-auto">
  <img src="/img/tutorials/spiral-fit-paris4.webp" className="w-[100%]"/>
  <figcaption className="mt-[-6px]">The result of fitting a spiral to PHerc. Paris 4 (Scroll 1): the 130 fitted windings, overlaid on a horizontal slice through the scan.</figcaption>
</div>

The core idea: we know the scroll was originally one long rectangular sheet, rolled up into a neat spiral. The eruption of Vesuvius deformed that spiral into the crushed shape we see in the CT scan. Instead of reconstructing the surface piece by piece, we search for the combination of *ideal scroll shape* and *smooth deformation* that best explains everything we observe. Once we have those, virtual unrolling comes almost for free: any point in the scan can be mapped back onto the original flat sheet.

The [last section](#how-it-works) of this tutorial goes into how it works internally; first, the practical part — [what goes in](#what-goes-in), [what comes out](#what-comes-out), and [how to run it](#how-to-run-it).

[^tracer]: The [surface tracer](segmentation#growing-large-meshes-with-the-tracer-method) is an earlier attempt at this problem: it stitches overlapping patches into large segments automatically. But it requires the patches to physically overlap or touch, and it becomes unreliable at whole-scroll scale.

### What goes in

The spiral is flexible about its inputs: it consumes many kinds of evidence, in almost any combination, and each kind can be created manually or automatically.

- **Surface patches** — small pieces of scroll surface, stored as `tifxyz` meshes (the grid-of-3D-points format used by VC3D). These can come from [GrowPatch](2026_open_problems#normal-grids-growpatch-and-local-tracing), [lasagna](2026_open_problems#lasagna-smoother-optimization-of-one-or-more-sheets) (direct growth, or growth around fibers), neural [Copy In/Out](2026_open_problems#copy-outin-exploiting-neighboring-wraps), or any other segmentation method. Only human-checked (**verified**) patches are recommended; they are also used to calculate evaluation metrics.
- **Strips and lines of points** that follow the surface of a single sheet — either *point collections* drawn in VC3D, or *fibers* traced in VC3D.
- **Relative winding annotations** — sets of points lying on different windings, annotated with how many windings apart they are (e.g. "these two points are exactly one wrap apart"). Represented as VC3D point collections with relative-winding annotations.
- **Absolute winding annotations** — points annotated with the absolute winding number they lie on (e.g. "this patch is on winding 20"). Also VC3D point collections.
- **Coarser volumetric guidance** derived from machine-learning predictions: predicted surface normals (from lasagna, stored as zarr volumes), predicted gradient magnitude (which captures the local radial density of windings), and skeletonised surface-prediction *tracks* (created with `extract_surface_tracks.py`).
- **Scroll-level structure**: the *umbilicus* (the scroll's central axis, as a function of z — required), and optionally a mesh of the scroll's outermost surface, which pins down where the spiral must end.

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

None of these individually needs to cover the scroll. Sparse, scattered evidence — a patch in one region, a fiber in another, a few relative-winding annotations in an ambiguous area — is combined by the fit into one consistent global solution, and annotations placed where the scroll is most damaged contribute the most.

### What comes out

The output is **one `tifxyz` mesh per winding** of the scroll — a full set of surfaces that conform to the input constraints, covering the whole fitted region including places no patch ever reached. Two variants are written for each winding: `wNNN`, the pure fitted spiral surface, and `wNNN_spliced`, where the geometry of verified patches is spliced into the fitted surface wherever the fit and the patch agree — more locally accurate wherever trusted geometry exists.

Since these are ordinary `tifxyz` meshes, everything downstream works as usual: you can load them in VC3D, flatten them, and [render surface volumes for ink detection](tutorial5). The repo also includes a tool ([`render_ink.py`](https://github.com/ScrollPrize/villa/blob/main/spiral-fitting/render_ink.py)) that concatenates the windings into fixed-width chunks, flattens them, and renders ink predictions as a series of horizontal strips — more on that [below](#rendering-ink).

Alongside the meshes, a fit writes a model checkpoint, *satisfaction metrics* — per-input-type statistics of how much of the evidence the final surface actually honors — and, optionally, overlay images showing the fitted windings drawn over scan slices.

### How to run it

The code lives in the villa repository under [`spiral-fitting`](https://github.com/ScrollPrize/villa/tree/main/spiral-fitting); the main entry point is [`fit_spiral.py`](https://github.com/ScrollPrize/villa/blob/main/spiral-fitting/fit_spiral.py). You'll need Python ≥ 3.14 and an NVIDIA GPU.

```bash
git clone https://github.com/ScrollPrize/villa.git
cd villa/spiral-fitting
uv sync
```

Everything the fit needs is declared in the project's own [`pyproject.toml`](https://github.com/ScrollPrize/villa/blob/main/spiral-fitting/pyproject.toml), `torch` included — on Linux it comes from the CUDA 12.8 wheel index, so there is no separate torch install to get right. `uv sync` also builds `vc_spiral`, a small C++ extension the fit uses to link point annotations to patch surfaces, so the machine needs cmake and a C++ toolchain; you no longer need a volume-cartographer Python install.

On Windows, `uv sync` also installs `triton-windows`, a community build of Triton, because PyTorch publishes no `triton` wheel there and the fit's fused kernels need one. The first run compiles those kernels once.

:::warning

If you do not install through the pyproject.toml, ensure your torch version is `<2.13` versions `>= 2.13` will require _significantly_ more vram due to internal torch changes

:::




#### Get the dataset

Ready-made inputs are published in the [`spiral-input` dataset](data_datasets#spiral-input-2026-07), which lives on the dl.ash2txt.org data server : [Spiral Datasets](https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/) (~90 GB):

```bash
rclone copy :http: ./spiral_datasets/phercparis4 \
    --http-url https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/ \
    --transfers 32 -P
```

Note that re-running rclone resumes interrupted downloads. For PHerc Paris 4, the dataset contains verified and unverified patches, tracks, fibers, the outer shell, winding annotation JSONs, the umbilicus, and the volume inputs — see the [dataset README](pathname:///data/datasets/spiral-input-PHercParis4-README.md) for the exact layout.

#### Configure

Two separate things configure a run: **where the data is** — given on the command line, plus one file inside the dataset — and **how to fit it**, a flat set of named settings.

##### The dataset

`fit_spiral.py` takes a `--dataset` root and resolves the conventional layout underneath it: `umbilicus.json`, `verified_patches/`, `fibers/`, `fiber_directions.npz`, `outer_shell/`, `tracks/`, `winding_inference/`, the `lasagna_inputs/*.ome.zarr` volumes, and the point-collection documents `abs_winding.json`, `relative_windings.json`, `same_windings.json` and `drawn_control_points.json`. A download of the published dataset is already in that layout, so there are no paths to edit.

You also need to provide a **`spiral-scroll.json`** in the dataset root, recording the physical facts of the scroll:

```json
{
  "schema_version": 1,
  "name": "PHercParis4",
  "voxel_size_um": 9.6,
  "spiral_outward_sense": "CW"
}
```

`name` is free-form and is what appears in the generated run-folder name. `spiral_outward_sense` (`"CW"` or `"ACW"`) says which way the spiral turns as it winds outward. No automated method determines it: it is read off the CT data by a person in VC3D, or taken from an already-fitted spiral. The file can also carry a `paths` object naming individual inputs whose filenames don't match the conventional ones (`"tracks_dbm"` is the usual one), and `normal_zarr_group` / `lasagna_scale`, which choose the OME-Zarr pyramid level the lasagna normal stores are read at — these are easy to get wrong silently, so read the scale off the store's own `.zattrs` rather than copying another scroll's values. The [spiral-fitting README](https://github.com/ScrollPrize/villa/blob/main/spiral-fitting/README.md) documents the full schema.


#### Running the fit

There are two ways to run the spiral fitting, either through the CLI or through an interactive fit session in the VC3D spiral workspace. Both run the same fit_spiral.py script. They differ in two ways: 
- The VC3D workspace can be configured to render a flatten view at an interval or on the fits completion
- The VC3D workspace supports adding inputs to "live" fits, interactively

<div className="spiral-fit-tabs" style={{border: '1px solid var(--ifm-color-emphasis-300)', borderRadius: '8px', padding: '1.25rem', margin: '1.5rem 0', backgroundColor: 'var(--ifm-background-surface-color)'}}>

<Tabs block>
<TabItem value="vc3d-workspace" label="VC3D Workspace" className="spiral-workspace-tutorial" default>

##### Starting a fit session

Open VC3D, select the scroll's volume package in [VC3D](tutorial_VC3D), then select the **Spiral** workspace tab. It uses the same scan as the main workspace, with a flattened preview and CT slice views beside it. If you are planning on running the spiral service on the machine you're opening VC3D with , ensure you have the venv activated in the terminal before opening VC3D. 

The spiral workspace runs through a service whether you're using it on the machine you're viewing VC3D through or over a network connection (internet or local LAN). 


##### Navigating the UI 

The spiral workspace is composed of two "windows": the configuration dock, on the left-side of the UI, and the flattened spiral view, which occupies the majority of the view. When no fit has returned a "preview", the flattened view is just a black rectangle. 

The configuration dock for the most part should be used top-to-bottom, and all fits begin with a session "connection"

<div className="mb-4 max-w-[900px] mx-auto">
  <img src="/img/tutorials/spiral-workspace/workspace-overview.webp" alt="VC3D Spiral workspace with the configuration dock on the left, flattened preview in the center, and CT views on the right." className="w-[100%]" />
  <figcaption className="mt-[-6px]">The Spiral workspace: configuration dock, flattened preview, and CT views.</figcaption>
</div>

###### Connecting to a session

In **Spiral Service**, choose a connection:

<div className="mb-4 max-w-[720px] mx-auto">
  <img src="/img/tutorials/spiral-workspace/spiral-service-section.webp" alt="Spiral Service panel showing connection profiles, endpoint, SSH host, dataset, output, and cache fields." className="w-[100%]" />
  <figcaption className="mt-[-6px]">Connection profiles and service paths in the Spiral Service panel.</figcaption>
</div>

- **Local:** set **Dataset** and **Output**, then **Connect**. VC3D launches the service locally; its Python environment must have the spiral-fitting dependencies installed. Keep Output outside the dataset directory.
- **Remote (SSH):** start the service on the GPU machine using the command below, then click **+SSH** in VC3D. Enter `[user@]host` (an SSH-config alias also works) and port `8765`, then **Connect**. Use SSH keys or an agent; if the host is new, connect once with `ssh user@host` in a terminal to accept its host key. VC3D creates the tunnel and retrieves the API key automatically.
- **Remote (LAN):** start the same service with `--bind 0.0.0.0`, then click **+LAN**. Set **Endpoint** to `http://HOST:8765` and paste the API key printed by the service. Direct HTTP is unencrypted; use it on a trusted network, or use SSH.

For either remote connection, run this on the GPU host from `villa/spiral-fitting` with dependencies installed (add `--bind 0.0.0.0` for LAN):

```bash
uv run python spiral_service.py --port 8765 \
    --dataset /data/scrolls/s1 --output /data/spiral-output/s1 \
    --gpus 0 --session-name my-fit
```

Keep this process running in a persistent terminal such as `tmux`. Remote fits continue when VC3D disconnects. Previews and checkpoints transfer automatically; a shared filesystem is unnecessary. If you mount the remote dataset locally, set **Local dataset path** to its matching root to enable verified patch overlays.

###### Configuring and running the initial fit 

The default parameters used by fit_spiral.py also apply to fit sessions run through the workspace. The only fields necessary for you to fill in are **Output Directory**, **z begin** and **z end**. If you'd like to override the defaults, click **Open Spiral Configuration...** and edit the fields. Hovering the mouse over the text for each field will show a brief description of what it does. 

<div className="mb-4 max-w-[900px] mx-auto">
  <img src="/img/tutorials/spiral-workspace/spiral-configuration-panel.webp" alt="Spiral configuration dialog with sampling, loss, patch, and model settings." className="w-[100%]" />
  <figcaption className="mt-[-6px]">The Spiral configuration dialog exposes the fit settings and their descriptions.</figcaption>
</div>

Set **z begin / z end** in **Fit and output**; start with a small range. Click **Initialize Fit**, choose **Iterations**, then **Run**. **Stop after iteration** pauses at the next completed step; another Run continues the fit. Initialization becomes **Rebuild Fit** once a fit exists. To resume a saved model, select it under **Checkpoint** and click **Load**.

Once the configured number of steps completes, or on the interval set by **Background preview every..** spinbox when enabled, a flattened spiral surface will display in the viewer. 

Navigation in the spiral preview surface and the volume views is similar to the rest of VC3D
- `right-click + drag` to pan, 
- mouse-wheel to zoom, 
- shift+wheel to move through slices on the flattened view,
- `R` and `ctrl+c` over a point to show that area in all views, 
- `X` recenters the views on that focus if you have panned away 

The spiral surface also contains a "minimap". Click the winding minimap below the flattened view to jump along the scroll. Use **Min winding / Max winding** limit the displayed windings (`-1` means through the last).

Within the display section of the configuration dock is the **Volume** combobox, which allows you to select the primary displayed volume (Overlays are managed like regular VC3D, via the **Overlay** toolbar item). The spiral preview surface also has many additional overlays available. Use **Display →** to toggle output, input patches, fibers and point collections, surface intersections, winding boundaries, patch overlap, and run differences. Loss overlays require **Compute loss overlays with the next preview**, which roughly doubles preview cost. The fixed status area shows fit progress and preview age; **Logs** opens service messages.

<div className="mb-4 max-w-[378px] mx-auto">
  <img src="/img/tutorials/spiral-workspace/spiral-display-panel.webp" alt="Display dialog with controls for point collections, patch overlap, winding transitions, loss overlays, and input visibility." className="w-[100%]" />
  <figcaption className="mt-[-6px]">Display controls for the preview and its overlays.</figcaption>
</div>

##### Annotate and apply changes

When fitting a spiral using the spiral workspace, you can add constraints to a running fit. 

| Key or gesture | Action |
| --- | --- |
| Tap Ctrl | Toggle patch painting; left-drag paints, right-drag erases |
| Ctrl+wheel | Change brush size |
| Shift+right-drag | Draw a freehand control-point line |
| Ctrl + right-click -> 2d line annotation | place control points to use in the fiber annotation, press `E` to optimize and display the line annotation window | 
| Hold V + left-clicks | Draw a control-point line through chosen points; release V to finish |
| Q, then left-clicks | Place points belonging to the same winding |
| E, then left-clicks | Place relative-winding points numbered 0, 1, 2, …; click successive windings in order |
| F | Reverse the active point collection and its relative-winding ordering |
| Escape | Exit the current drawing/point-placement mode |
| Shift+E | Prepare drawing drafts for submission |

<div className="mb-4 max-w-[720px] mx-auto">
  <img src="/img/tutorials/spiral-workspace/surface-annotation-types.webp" alt="Flattened spiral surface with a purple painted patch and cyan point annotations." className="w-[100%]" />
  <figcaption className="mt-[-6px]">A flattened spiral preview with a painted patch and point annotations.</figcaption>
</div>

Use **Add/Apply changes** to submit ready drafts to the fit, including while it is running. **Commit** persists the selected changes into the dataset; applying alone does not. For existing inputs, enable **Show original dataset inputs**, then right-click an entry and choose **Edit**. Patch and fiber editors save working copies; use Add/Apply afterward. **Remove** stages a removal, **Restore** reverses it before Commit, and committing the removal deletes the managed dataset entry.

<div className="mb-4 max-w-[720px] mx-auto">
  <img src="/img/tutorials/spiral-workspace/pending-and-committed-inputs.webp" alt="Input list showing drawing changes and the Add/Apply changes, Commit, and Remove buttons." className="w-[100%]" />
  <figcaption className="mt-[-6px]">Drawing changes in the input list: Apply updates the fit; Commit writes them into the dataset.</figcaption>
</div>

##### Preview intervals and rendering

Enable **Background preview every** and choose an interval in **iterations** before clicking Run (default interval: 100). The service captures the fit at an iteration boundary, exports its surface, and flattens it through Lasagna; fitting resumes while flattening continues. VC3D downloads the finished geometry and displays the scan on that surface. This is a geometry preview; ink strips are produced separately by [Rendering ink](#rendering-ink).

A preview is also requested when a connected run finishes or is stopped. Exporting and flattening can take minutes, so the displayed preview may trail the fit; check its iteration and lag in the status area. Until a new preview succeeds, the previous one stays visible.

##### Where files live

Generated files live on the **service machine**, under its Output root; the example above uses `/data/spiral-output/s1/my-fit/`. This holds fit run directories, previews, uploads, and checkpoints, including `checkpoint_autosave.ckpt`. For local services, an empty Output field uses a per-dataset directory in VC3D's application-data folder; set it explicitly for an easy-to-find location. The derived cache defaults to `~/.cache/vc3d/spiral`.

**Checkpoint → Save on Service** saves remotely; **Download…** copies a checkpoint to your computer. VC3D also caches downloaded display artifacts locally. Only **Commit** writes your annotation changes back into the dataset. See the [service README](https://github.com/ScrollPrize/villa/blob/main/spiral-fitting/README.md#internet-flow-ssh-attach) for connection and storage details.

</TabItem>
<TabItem value="cli" label="CLI">

##### The fit configuration

Everything else — the fitted z-range, which inputs participate, loss weights, resolutions, step counts — is a flat dictionary of named settings defined in [`config.py`](https://github.com/ScrollPrize/villa/blob/main/spiral-fitting/config.py): one attribute of the `Config` class per knob, each with its default. You can pass overrides as JSON:

```bash
FIT_SPIRAL_CONFIG_OVERRIDES='{"z_begin": 10500, "z_end": 11500, "optimizer_num_training_steps": 10000}' \
    python fit_spiral.py --dataset ./spiral_datasets/phercparis4
```

The settings you are most likely to touch:

- `z_begin`, `z_end` — the slice range (in full-resolution voxels) to fit; the defaults, 4,000 and 17,000, span the whole written region of Scroll 1. **Consider starting with a small range**: fitting all of it needs a lot of GPU memory and time, and a ~1,000-slice range is a good first run on a smaller GPU. Per-step sample counts are scaled automatically to the size of the z-range, so the other hyperparameters don't need retuning when you change it.
- **`input_use_*` — whether an input that is present is actually used.** Each optional input source has its own toggle, independent of whether its file is there: `input_use_verified_patches`, `input_use_tracks`, `input_use_fibers`, `input_use_fiber_directions`, `input_use_normals`, `input_use_gradient_magnitude`, `input_use_winding_inference`, `input_use_outer_shell`, plus one per annotation role — `input_use_pcl_absolute`, `input_use_pcl_relative`, `input_use_pcl_same_winding`, `input_use_pcl_drawn_control_points`.Note that `input_use_tracks` defaults to `false`, so a dataset that includes a tracks DBM will not use it unless you turn it on:

  ```
  FIT_SPIRAL_CONFIG_OVERRIDES='{"input_use_tracks": true, ...}'
  ```

  A few ready-made override files live in [`configs/`](https://github.com/ScrollPrize/villa/tree/main/spiral-fitting/configs).

A few environment variables control the run itself rather than the fit:

| Variable | Effect |
| --- | --- |
| `FIT_SPIRAL_CONFIG_OVERRIDES` | JSON dict of `Config` overrides, e.g. `'{"optimizer_num_training_steps": 10000}'` |
| `FIT_SPIRAL_OUT_DIR` | Parent directory for the generated run folder (default `./out`) |
| `FIT_SPIRAL_RUN_DIR` | Use this exact directory as the run folder, instead of generating a name |
| `FIT_SPIRAL_RUN_TAG` | Tag appended to the output folder and mesh names |
| `FIT_SPIRAL_RESUME_PATH` / `FIT_SPIRAL_RESUME_STEP` | Resume from a checkpoint |
| `WANDB_MODE` | Set to `online` to log losses and visualizations to Weights & Biases (default `disabled`) |

The cache of preprocessed inputs — which speeds up subsequent runs a lot — defaults to `~/.cache/vc3d/spiral`, shared across datasets since its entries are content-addressed; `--cache DIR` (or `FIT_SPIRAL_CACHE_DIR`) moves it elsewhere.

#### Fit

```bash
python fit_spiral.py --dataset ./spiral_datasets/phercparis4
```

That's it — the script loads the inputs (caching the expensive preprocessing), then runs 30,000 optimization steps, printing the loss breakdown every 200 steps. Multi-GPU is supported via `torchrun --nproc-per-node=N fit_spiral.py --dataset ...`, which splits each step's work across GPUs.

To run the whole pipeline in one command — fit, then render ink, then score it — use [`runners/run_single.py`](https://github.com/ScrollPrize/villa/blob/main/spiral-fitting/runners/run_single.py) instead. It takes the same `--dataset`, plus an `--ink-volume`, and accepts the same configuration overrides as a `--config` JSON file; `runners/run_sweep.py` runs a whole folder of such configs concurrently across GPUs.

When it finishes, you get a self-contained run folder:

```
out/2026-07-08_s1_slice-10500-11500_27399-patch_<run-name>/
├── checkpoint_fitted.ckpt             # fitted model (resumable)
├── satisfied_fitted.json              # which individual inputs the fit honors
├── satisfaction_metrics_fitted.json   # and the summary of those, per input type
├── spiral_on_*_fitted.png             # fitted windings overlaid on inputs
└── meshes/mesh/
    ├── w010/                          # one tifxyz mesh per winding...
    ├── w010_spliced/                  # ...plus the patch-spliced variant
    ├── w011/
    └── ...
```

The overlay PNGs are only written when `output_save_png_visualizations` is on; it defaults to off, since rendering them means reading scan slices back at the end of the fit.

</TabItem>
</Tabs>

</div>

### Tips for getting better spiral fits
The goal of any spiral fit is to do _as little annotation as possible_ . This is, of course, a hard question to answer until you have a perfect fit and work backwards. However, we can get a rough guess if we consider how the spiral interpolates. The easiest way to think of this is to imagine the scroll as a set of local shapes glued together, perhaps one section of the scroll looks like a `C` or one a `J` or god forbid a `Z`. In each of these shaped sections, we need _something_ to inform the spiral how the local area should be deformed. 

<div className="mb-4 flex flex-wrap items-center justify-between sm:w-[108%] sm:ml-[-4%]">
  <figure className="w-[100%] sm:w-[71%] m-0">
    <img src="/img/tutorials/spiral-workspace/annotation-density-combined.webp" alt="A slab of scroll split into colored local shape sections, shown assembled on the left and exploded apart on the right, each marked with a green, yellow, or red annotation-density badge." className="w-[100%]" />
    <figcaption className="mt-0">The scroll as a set of local shapes glued together. Badges indicate roughly how much annotation each section needs, from low (green) to high (red).</figcaption>
  </figure>
  <figure className="w-[60%] my-0 mx-auto sm:mx-0 sm:w-[26%]">
    <img src="/img/tutorials/spiral-workspace/annotation-density-stack.webp" alt="Three stacked slabs of the scroll at different z heights, each divided into colored shape sections with annotation-density badges." className="w-[100%]" />
    <figcaption className="mt-0">The same scroll at three heights: density follows local shape complexity, so it changes along z.</figcaption>
  </figure>
</div>

These "descriptions" typically come in the form of constraints like patches, fibers, or relative winding annotations. The "density" of these shape descriptors required in a given area depends greatly on how uniform the deformation is. If a large number of windings all make mostly the same shape, it could be very few. However if an area makes something more `3C` than `C` (bear with me), you'll need a fair bit more annotation. The "complexity" of the local shape is the primary multiplier of annotation density, rather than the number of windings or the curvature alone. Once you've got a constraint that describes the neighborhood, there is little benefit in adding "more". You would not, for example, want to annotate multiple windings in a row radially if they are mostly the same shape and have somewhat regular spacing.

The other instance which typically requires more-than-usual amounts of annotation is the case of shears. In this context, a shear of the scroll is a location where the papyrus has not only _broken apart_ but also _shifted_ along an axis. These situations are particularly hard for a smooth field to interpolate (they are by nature unsmooth). In these types of areas it is best to try and find some portion of the scroll which you can follow through, and apply a fair bit of annotation on either side of the shear, as near as you can get to the actual missing papyrus. 

<div className="mb-4 flex flex-wrap items-start justify-between sm:w-[80%] sm:mx-auto">
  <figure className="w-[100%] sm:w-[53%] m-0">
    <video autoPlay playsInline loop muted className="w-[100%]" poster="/img/tutorials/spiral-workspace/shear-xy-poster.webp">
      <source src="/img/tutorials/spiral-workspace/shear-xy.webm" type="video/webm"/>
    </video>
    <figcaption className="mt-0">Scrolling through z 9680–9990 across a shear, where the papyrus breaks and shifts along an axis.</figcaption>
  </figure>
  <figure className="w-[100%] sm:w-[45%] m-0">
    <img src="/img/tutorials/spiral-workspace/shear-section-normal.webp" alt="Vertical CT section cut across the papyrus layers through the shear, showing a dark void at about z 9813 where the layers above and below are offset from each other." className="w-[100%]" />
    <figcaption className="mt-0">The same shear in a vertical section cut across the layers. The crosshair marks z 9813, where the layers break apart and shift.</figcaption>
  </figure>
</div>

Now that we've described the _bad_ areas, there are some shapes within a scroll which are "easier" (or at least as easy as unrolling an ancient carbonized scroll can reasonably be): 

- Because the gap expander prefers to push outwards from its initial gap, and due to it being a cumulative sum of the gaps _along_ the radial , the spiral will (for the most part) perform quite well in areas with regular spacing and curvature which happens smoothly over a large area. This _reduces_ the amount of annotation we need in these areas
- Areas which have large gaps between windings that are sustained for long runs reduce the amount of "uncertainty" in the fit, and are typically fit well very early
- Areas with minimal curvature (even highly compressed ones)

<div className="mb-4 max-w-[560px] mx-auto">
  <img src="/img/tutorials/spiral-workspace/annotation-density-easy-piece.webp" alt="A single green 3D piece of the scroll, its top face showing evenly spaced windings that curve smoothly, with one white annotation line." className="w-[100%]" />
  <figcaption className="mt-[-6px]">One of the easiest pieces from the slab above: regular spacing and smooth, gradual curvature mean a single annotation describes the whole neighborhood.</figcaption>
</div>



#### Rendering ink

To get from per-winding meshes to readable images, use `render_ink.py`. It groups the `_spliced` winding meshes into winding-range chunks, concatenates each chunk into a single mesh (written to a `concat/` folder — useful for loading the geometry behind each strip as one mesh), SLIM-flattens it, renders it through an ink-prediction volume with `vc_render_tifxyz`, and composites the result into one JPEG strip per chunk:

```bash
python render_ink.py /path/to/run/meshes/mesh --volume /path/to/ink_prediction.zarr
```

You'll need a [VC3D build](segmentation#installation-instructions) on your `PATH` for the rendering and flattening binaries (`vc_render_tifxyz`, `flatboi`, …), and an ink-prediction zarr for the scroll. The output `ink/` folder fills with strips named by winding range (e.g. `w010-027.jpg`).

#### Ink metrics

The script `get_ink_metrics.py` computes some metrics based on the amount of letter-like ink signal detected in the ink renders. By default it uses the model `scrollprize/ink-coverage-32um` from HuggingFace; this is a 2D nnUNet operating on small patches, trained to do binary segmentation of clearly-identifiable ink. The script measures the total area of ink detected, as well as evaluating whether columns are coherent and have approximately the expected width, and lines are locally coherent (based on sliding windows) and have approximately the expected pitch.

:::warning

The ink-coverage model was only trained on PHerc. Paris 4, so it may not give accurate results for other scrolls with significantly different writing styles.

:::


### How it works

Up to this point we treated the fit as a black box; here is what is actually inside it. (There are more math details in the paper [*Virtually Unrolling the Herculaneum Papyri by Diffeomorphic Spiral Fitting*](https://arxiv.org/abs/2512.04927), though for a slightly older version of the algorithm.)

#### An ideal scroll...

Originally, a scroll was one nearly rectangular sheet of papyrus, rolled up (often around a central rod). In cross-section that is a spiral — specifically, we model it as a perfect **archimedean spiral**, extruded into the plane. Treating it as arbitrarily large, the ideal scroll has just *one* free parameter: the tightness of its windings, $\omega$. A point on the ideal sheet is addressed by two curvilinear coordinates — the angle $\theta$ along the spiral and the height $z$ along the axis — and sits at radius

$$
r(\theta) = \tfrac{\omega}{2\pi}\,\theta,
$$

so each full turn moves the sheet outward by one sheet-to-sheet spacing $\omega$. Plug in any $(\theta, z)$ and you get a 3D point on the ideal sheet.

#### ...horribly deformed

The eruption turned that neat spiral into the crumpled shape in the scan. We model the damage as a **diffeomorphic transformation**: a smooth, differentiable, *invertible* map of 3D space. That choice buys us exactly the guarantees we need:

- It cannot tear the sheet, make it pass through itself, or squish it to a point — it preserves topology. If it starts as a spiral, after deformation it is still a spiral, just a messed-up one.
- It is invertible: a point on the ideal scroll maps to a point in the scan, and — just as importantly — any point in the scan maps back to a point on the ideal (i.e. flattened) scroll. That inverse map *is* the virtual unrolling.

The deformation is composed of three parts applied in sequence: a coarse global scale and shear, the integral of a stationary velocity field (the most important one), and a local scaling of the gap between windings, defined everywhere on the sheet (this lets windings locally squeeze together or spread apart without disturbing anything else). Each part is smooth and invertible, so the composition is too.

The middle term deserves a closer look. Imagine a little 3D arrow attached to every point in space — a **velocity field** $u$. Every point of the ideal spiral flows along these arrows, like dust in a (smooth, steady) wind. Mathematically, the trajectory $\phi_t(x)$ of a point $x$ is defined by the ODE

$$
\frac{\mathrm{d}\phi_t(x)}{\mathrm{d}t} = u\big(\phi_t(x)\big),
\qquad \phi_0(x) = x,
$$

and the transformation is where the flow ends up after one unit of time: $T_{\text{flow}}(x) = \phi_1(x)$. Don't worry too much about the equation — the intuition is what matters: every point rides smoothly along the flow, so the whole spiral deforms smoothly into a new shape, and running the flow backwards gives the exact inverse. This is the same machinery used in diffeomorphic medical image registration; in the code, the ODE is integrated with a few Runge–Kutta steps ([`flow_fields.py`](https://github.com/ScrollPrize/villa/blob/main/spiral-fitting/flow_fields.py), [`transforms.py`](https://github.com/ScrollPrize/villa/blob/main/spiral-fitting/transforms.py)).

<div className="mb-4">
  <img src="/img/ash2text/image16.png" className="w-[100%]"/>
  <figcaption className="mt-[-6px]">An idealized rolled scroll (left) is related to the deformed scroll observed in the scan (right) by a smooth spatial deformation — the transformation the spiral fit estimates.</figcaption>
</div>

#### Fitting as an inverse problem

Fitting is then an inverse problem: find the winding tightness $\omega$ and the deformation parameters (the velocity field, plus the scaling terms) such that the deformed spiral explains what we see in the scan. We don't fit to the raw CT intensities directly. Instead, every input from [What goes in](#what-goes-in) becomes a differentiable loss term saying what the deformed spiral should look like:

- points from a same-sheet strip should all land on *some* winding surface (and the same one);
- two points annotated as $k$ windings apart should land exactly $k$ windings apart;
- a verified patch should coincide with a single winding across its whole extent;
- tracks, normals, and gradient-magnitude volumes nudge the surface orientation and winding density;
- the innermost winding should wrap the umbilicus, and the outermost should follow the outer shell;
- and regularization terms keep the sheet parameterization from distorting.

All parameters are optimized *jointly*, with plain Adam, minimizing the weighted sum of these losses (the weights are the `loss_weight_*` entries in `default_config`). In effect, we tell the machine "here are all the constraints humans and models have gathered — find the deformation that squishes the ideal spiral so that they are all met", and gradient descent does the rest.

At the end, we sample each winding of the fitted ideal spiral on a regular $(\theta, z)$ grid, push the samples through the fitted deformation into scan coordinates, and write each winding out as a `tifxyz` mesh — the outputs described above.

One caveat when reading [the paper](https://arxiv.org/abs/2512.04927): it describes a fully automatic setup that fits only raw surface-prediction tracks and fields derived from them. The current code fits the much richer curated evidence described in this tutorial — verified patches, fibers, and winding annotations — which is what makes it accurate enough to target whole-scroll segmentation. The underlying model and optimization are still very similar.
