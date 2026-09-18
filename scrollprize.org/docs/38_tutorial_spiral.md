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


*Last updated: September 16, 2026*

<ChatCallout prefill="Walk me through the spiral fitting tutorial" />

Most of our segmentation tools work bottom-up. [GrowPatch](2026_open_problems#normal-grids-growpatch-and-local-tracing), [lasagna](2026_open_problems#lasagna-smoother-optimization-of-one-or-more-sheets), and [manual segmentation in VC3D](tutorial_VC3D) all produce *patches* — pieces of papyrus surface that you grow bigger and bigger until they hit a tricky region and stall. Other tools trace individual fibers. Either way you end up with a big pile of small pieces: segments, fibers, point annotations. What we really want is the *whole scroll* — one surface covering every winding of the original papyrus sheet, from the center to the outer shell. However, gluing the pieces together directly is hard, especially where there are gaps between them. [^tracer]

That is what the spiral fit does. It takes the whole pile of partial evidence — surface patches, traced lines, winding annotations, volumetric predictions — and fits a single, globally coherent surface for the entire scroll that agrees with as much of that evidence as possible. Where the evidence is dense, the fitted surface follows it closely; where there are gaps, the spiral bridges them smoothly instead of stopping or leaving a gap.

<div className="mb-4">
  <img src="/img/tutorials/spiral-fit-paris4.webp" className="w-[100%]"/>
  <figcaption className="mt-[-6px]">The result of fitting a spiral to PHerc. Paris 4 (Scroll 1): the 130 fitted windings, overlaid on a horizontal slice through the scan.</figcaption>
</div>

The core idea: we know the scroll was originally one long rectangular sheet, rolled up into a neat spiral. The eruption of Vesuvius deformed that spiral into the crushed shape we see in the CT scan. Instead of reconstructing the surface piece by piece, we search for the combination of *ideal scroll shape* and *smooth deformation* that best explains everything we observe. Once we have those, virtual unrolling comes almost for free: any point in the scan can be mapped back onto the original flat sheet.

The [last section](#how-it-works) of this tutorial goes into how it works internally; first, the practical part — [what goes in](#what-goes-in), [what comes out](#what-comes-out), and [how to run it](#how-to-run-it).

[^tracer]: The [surface tracer](segmentation#growing-large-meshes-with-the-tracer-method) is an earlier attempt at this problem: it stitches overlapping patches into large segments automatically. But it requires the patches to physically overlap or touch, and it becomes unreliable at whole-scroll scale.

### What goes in

The spiral is flexible about its inputs: it consumes many kinds of evidence, in almost any combination, and each kind can be created manually or automatically.

- **Surface patches** — small pieces of scroll surface, stored as `tifxyz` meshes (the grid-of-3D-points format used by VC3D). These can come from [GrowPatch](2026_open_problems#normal-grids-growpatch-and-local-tracing), [lasagna](2026_open_problems#lasagna-smoother-optimization-of-one-or-more-sheets) (direct growth, or growth around fibers), neural [Copy In/Out](2026_open_problems#copy-outin-exploiting-neighboring-wraps), or any other segmentation method. Patches are split into two groups, **verified** and **unverified**: the fit places strong weight on the human-checked verified patches, and treats the unverified ones as weaker hints. The verified patches are also used to calculate evaluation metrics.
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

Two notes for Windows, where the fit runs natively:

- PyPI marks PyTorch's CUDA runtime packages `platform_system == "Linux"`, so the line above installs a CPU-only build there. Take it from the CUDA index instead, e.g. `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.
- PyTorch also publishes no `triton` wheel for Windows, and the fit's fused RK4 and gap kernels quietly fall back to the eager path without it. Installing the community `triton-windows` build that matches the `triton` version your torch pins puts the fused path back; on one machine that was 3.8 vs 1.7 it/s, with the first run paying a one-off kernel compile.

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

`fit_spiral.py` takes a `--dataset` root and resolves the conventional layout underneath it: `umbilicus.json`, `verified_patches/`, `fibers/`, `fiber_directions.npz`, `outer_shell/`, `tracks/`, `winding_inference/`, the `lasagna_inputs/*.ome.zarr` volumes, and the point-collection documents `abs_winding.json`, `relative_windings.json`, `same_windings.json` and `drawn_control_points.json`. A download of the published dataset is already in that layout, so there are no paths to edit. The one input with no conventional location is `unverified_patches/`, which must be named explicitly in the `paths` block described below.

You also need to provide a **`spiral-scroll.json`** in the dataset root, recording the physical facts of the scroll:

```json
{
  "schema_version": 1,
  "name": "PHercParis4",
  "voxel_size_um": 9.6,
  "spiral_outward_sense": "CW"
}
```

`name` is free-form and is what appears in the generated run-folder name. `spiral_outward_sense` (`"CW"` or `"ACW"`) says which way the spiral turns as it winds outward. No automated method determines it: it is read off the CT data by a person in VC3D, or taken from an already-fitted spiral. The file can also carry a `paths` object naming individual inputs that have no conventional location (`"unverified_patches"`) or whose filenames don't match the conventional ones (`"tracks_dbm"` is the usual one), and `normal_zarr_group` / `lasagna_scale`, which choose the OME-Zarr pyramid level the lasagna normal stores are read at — these are easy to get wrong silently, so read the scale off the store's own `.zattrs` rather than copying another scroll's values. The [spiral-fitting README](https://github.com/ScrollPrize/villa/blob/main/spiral-fitting/README.md) documents the full schema.

##### The fit configuration

Everything else — the fitted z-range, which inputs participate, loss weights, resolutions, step counts — is a flat dictionary of named settings defined in [`config.py`](https://github.com/ScrollPrize/villa/blob/main/spiral-fitting/config.py): one attribute of the `Config` class per knob, each with its default. You can pass overrides as JSON:

```bash
FIT_SPIRAL_CONFIG_OVERRIDES='{"z_begin": 10500, "z_end": 11500, "optimizer_num_training_steps": 10000}' \
    python fit_spiral.py --dataset ./spiral_datasets/phercparis4
```

The settings you are most likely to touch:

- `z_begin`, `z_end` — the slice range (in full-resolution voxels) to fit; the defaults, 4,000 and 17,000, span the whole written region of Scroll 1. **Consider starting with a small range**: fitting all of it needs a lot of GPU memory and time, and a ~1,000-slice range is a good first run on a smaller GPU. Per-step sample counts are scaled automatically to the size of the z-range, so the other hyperparameters don't need retuning when you change it.
- **`input_use_*` — whether an input that is present is actually used.** Each optional input source has its own toggle, independent of whether its file is there: `input_use_verified_patches`, `input_use_unverified_patches`, `input_use_tracks`, `input_use_fibers`, `input_use_fiber_directions`, `input_use_normals`, `input_use_gradient_magnitude`, `input_use_surf_sdt`, `input_use_winding_inference`, `input_use_outer_shell`, plus one per annotation role — `input_use_pcl_absolute`, `input_use_pcl_relative`, `input_use_pcl_same_winding`, `input_use_pcl_drawn_control_points`.Note that `input_use_tracks` defaults to `false`, so a dataset that includes a tracks DBM will not use it unless you turn it on:

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
- a verified patch should coincide with a single winding across its whole extent — with unverified patches pulled in more gently;
- tracks, normals, and gradient-magnitude volumes nudge the surface orientation and winding density;
- the innermost winding should wrap the umbilicus, and the outermost should follow the outer shell;
- and regularization terms keep the sheet parameterization from distorting.

All parameters are optimized *jointly*, with plain Adam, minimizing the weighted sum of these losses (the weights are the `loss_weight_*` entries in `default_config`). In effect, we tell the machine "here are all the constraints humans and models have gathered — find the deformation that squishes the ideal spiral so that they are all met", and gradient descent does the rest.

At the end, we sample each winding of the fitted ideal spiral on a regular $(\theta, z)$ grid, push the samples through the fitted deformation into scan coordinates, and write each winding out as a `tifxyz` mesh — the outputs described above.

One caveat when reading [the paper](https://arxiv.org/abs/2512.04927): it describes a fully automatic setup that fits only raw surface-prediction tracks and fields derived from them. The current code fits the much richer curated evidence described in this tutorial — verified patches, fibers, and winding annotations — which is what makes it accurate enough to target whole-scroll segmentation. The underlying model and optimization are still very similar.
