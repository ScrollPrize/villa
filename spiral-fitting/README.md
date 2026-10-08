# Spiral fitting

Code and helpers to fit a canonical Archimedean spiral to deformed scrolls.
`spiral_service.py` hosts one persistent interactive fit session over HTTP for
the VC3D Spiral workspace; `fit_spiral.py` is the underlying fitter.

## CUDA startup check

VC3D fit sessions and command-line fits allocate one CUDA element and synchronize
before loading fit inputs. This creates the CUDA context before dataset reads
increase filesystem cache pressure, and reports startup failures independently
of atlas size. CPU-only `FitContext.load_host_inputs()` callers remain unchanged.
The GPU atlas progress messages appear when geometry is actually materialized.

On Linux NVIDIA GB10 systems, startup OOM diagnostics include host memory figures
and NVIDIA's manual cache-flush workaround. The application never flushes system
caches itself. Early initialization does not guarantee that the full fit will
fit in shared CPU/GPU memory.

Validate with the existing environment, from the repository root:

```sh
AGENTS_AGENT_MODE=1 PYTHONPATH=spiral-fitting spiral-fitting/.venv/bin/python -m pytest -q \
  spiral-fitting/tests/test_cuda_startup.py \
  spiral-fitting/tests/test_spiral_headless.py \
  spiral-fitting/tests/test_spiral_progress.py
AGENTS_AGENT_MODE=1 PYTHONPATH=spiral-fitting spiral-fitting/.venv/bin/python -c \
  'from types import SimpleNamespace; from fit_spiral import FitContext; FitContext.check_cuda_ready(SimpleNamespace(progress=None))'
```

See [NVIDIA's DGX Spark memory guidance](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).

## Editing input snapshots

The service snapshots patch directories, fiber JSON files and point collections
when claiming a dataset editing workspace. `input_snapshot.py` tries Linux
`FICLONE` or macOS `clonefile` for independent copy-on-write files. Unsupported
filesystems and cross-filesystem copies fall back to a streamed copy; other I/O
errors propagate. Hard links are never used.

The fallback hashes bytes while copying, verifies the destination, and compares
the final source fingerprint with the captured tree. Clones are hashed once and
compared with the final source. This reduces full content reads from four to
three for ordinary copies and two for clones. Snapshots retain the existing
SHA-256 format, sorted tree membership, empty directories and symlink rejection.
Changes during capture that leave the source different from the captured bytes
are rejected. No model inputs, precision or numerical algorithms change.

Run the focused tests using the existing environment, from the repository root:

```sh
PYTHONPATH=spiral-fitting spiral-fitting/.venv/bin/python -m pytest -q \
  spiral-fitting/tests/test_input_snapshot.py \
  spiral-fitting/tests/test_service_editing.py \
  spiral-fitting/tests/test_service_editing_http.py
```

Benchmark real patches (temporary copies are removed; sources stay unchanged):

```sh
PYTHONPATH=spiral-fitting spiral-fitting/.venv/bin/python \
  spiral-fitting/tests/benchmark_input_snapshots.py \
  /home/sean/Desktop/spiral_dataset/verified_patches --iterations 5 --profile
```

Use `--destination PATH` to test another filesystem. The default is `/tmp`.
`--count` selects the first N sorted patch directories (default 32).

Measured on Linux arm64, CPython 3.14, without Python optimization flags, with
`cProfile` enabled for both versions: 32 real verified patches, 55,302,864 bytes,
five iterations, ordinary-copy fallback on the local ext-family filesystem.
These are cache-warm measurements, not end-to-end fit initialization times.

| Snapshot implementation | Mean | Min | Median | Max |
| --- | ---: | ---: | ---: | ---: |
| Previous copy plus three hash passes | 166.8 ms | 160.8 ms | 162.2 ms | 181.0 ms |
| Streamed copy/hash plus verification | 152.3 ms | 148.9 ms | 153.2 ms | 154.0 ms |

Mean time decreased 8.7%. Before the change, fingerprint calls accounted for
0.649 s of 0.834 s total across all five iterations; file copying accounted for
0.177 s. Afterwards, hashing remains the largest cost (0.336 s in SHA-256 updates),
but the separate source pre-read is eliminated. Reflink throughput and actual
macOS cloning require validation on supporting filesystems; unit tests cover
the platform dispatch and fallback/error paths.

Filesystem API references: [Linux FICLONE](https://man7.org/linux/man-pages/man2/FICLONE.2const.html)
and [Apple file cloning support](https://developer.apple.com/documentation/foundation/urlresourcevalues/volumesupportsfilecloning).

## Scroll specification (spiral-scroll.json)

`fit_spiral.py` requires a `spiral-scroll.json` file in the dataset root.
This is not covered by scrollprize.org/tutorial_spiral. Required keys:

- `schema_version` — must equal `1`.
- `name`, `voxel_size_um` — required, no validation beyond presence.
- `z_direction_is_top_to_bottom`, `left_handed_coordinates` — the fitted
  volume's two properties from the open-data catalog (`metadata.json`), copied 
- verbatim as `true`/`false`. Together they fix the spiral's sense under the catalog convention that every scroll
  shows the same spiral seen from its top: `"CW"` when they are equal, `"ACW"`
  when they differ. The z direction also orients exported surfaces (below).
- `spiral_outward_sense` — `"CW"` or `"ACW"` (case-insensitive). Derived from
  the two catalog properties when both are present, and then only needed as a
  cross-check (a mismatch is an error). Without them it is required and is
  read off the CT data by a person in VC3D, or from an already-fitted spiral.

Exported surfaces (the per-winding `meshes/`, the combined preview and
`flatten_spiral_checkpoint.py`'s source surface) read like the scroll: column 0
is the outermost wrap, U running outside to inside as orient-segment
normalises it, and row 0 is the top of the scroll when
`z_direction_is_top_to_bottom` is known (rows stay in z order when it is
not). Each `meta.json` and preview manifest records the z direction the
layout was made with under `grid_orientation` (a grid without it is an older
one in sampling order: innermost first, rows by z); `winding_column_ranges`
are in the written grid, outermost winding first. `left_handed_coordinates` does not change the grid, only the
direction of its cross-product normal, which renders correct with
`flip-normals = not left_handed_coordinates`.

Optional `paths` object for per-input overrides when a dataset's file names
don't match the catalog's conventional defaults (e.g. `tracks_dbm`).

Also optional, and easy to get wrong silently: `normal_zarr_group`
(default `"4"`) and `lasagna_scale` (default `4`) select which OME-Zarr
pyramid level the `normal_x`/`normal_y` lasagna stores are read at.
**`lasagna_scale` must equal the actual downsample factor of whichever
group you pick for this specific scroll's lasagna store** — read it from
the store's own `.zattrs` multiscales metadata, do not assume it from
another scroll's example or from a generic recommendation. A mismatched
value either silently reads the wrong-resolution normal maps (no error) or
throws `RuntimeError: lasagna z-ROI [...] is empty` if the mismatch is
large enough to push the requested z-range outside the (wrongly-scaled)
store bounds.

Example (PHerc0826, where group `"2"` is a 4x downsample for this scroll
specifically):

```json
{
  "schema_version": 1,
  "name": "PHerc0826",
  "voxel_size_um": 9.362,
  "spiral_outward_sense": "CW",
  "normal_zarr_group": "2",
  "lasagna_scale": 4,
  "paths": {
    "tracks_dbm": "tracks/PHerc0826_20250821151701_surface_m7_L0_th0.2.dbm"
  }
}
```

## Lasagna inputs must be packed first

`fit_spiral.py` reads `normal_x`, `normal_y` and `gradient_magnitude` only
through the resident-pool sidecars that `pack_resident_pools.py` writes
(`lasagna_data.py`: there is no other loading path). Before the first fit on a
dataset, pack the stores at the group named by `normal_zarr_group` in
`spiral-scroll.json`:

```sh
python pack_resident_pools.py <dataset>/lasagna_inputs \
    --what normals,grad_mag --normal-group 2
```

The packer looks for `*_nx.ome.zarr`, `*_ny.ome.zarr` and `*_grad_mag.ome.zarr`
in that folder and writes `<store>.respool_g<group>_pair` (normals) and
`<store>.respool_g<group>` (grad_mag) next to them. Without the sidecars the fit
stops at input loading with

```
lasagna normals: resident-pool sidecar '.../PHercXXXX_nx.ome.zarr.respool_g2_pair' not found; build it with pack_resident_pools.py ...
```

The packer only iterates over chunk files that exist, so a store that holds only
the z chunk rows of your fit window packs correctly; rows outside it read back
as no-data. Pass `--ct <scroll zarr> --ct-group <group>` to drop bricks outside
the CT mask.

## Sweep runner output

`runners/run_sweep.py` prefixes each active fit's live `PROGRESS` and
every-200-step loss lines with its configuration name. Optimization progress
includes the average iteration rate for the current stage (`it/s`). Complete
combined stdout/stderr for every attempt remains available under
`<output>/.sweep/logs/<config>.log`.



## Flattening a fitted checkpoint

`flatten_spiral_checkpoint.py` is a standalone, one-shot exporter. It
reconstructs the combined surface from a fitted checkpoint, launches a private
Lasagna service, flattens with `flatten_fast_nofilter.json`, writes the final
TIFXYZ directory, and tears the service down even if the job fails or is
interrupted:

```sh
python flatten_spiral_checkpoint.py \
    /path/to/checkpoint_fitted.ckpt \
    /path/to/output.tifxyz
```

The checkpoint format does not embed the fixed umbilicus curve. The script
looks for `umbilicus.json` in the checkpoint's ancestors, in
`$SPIRAL_DATASET`, and in the standard local s1 dataset location. For other
layouts, pass `--umbilicus /path/to/umbilicus.json`. Use `--lasagna-dir` if
the Lasagna repository is not in its standard sibling or `~/villa` location.
An existing output path is never overwritten.

## Flow stages

`model_num_flow_stages` sets how many stationary velocity fields the flow
diffeomorphism composes. The stages are the slabs of the flow lattices'
leading axis (`flow_field.flows.{0,1}` are `[stages, 3, ...]`), integrated in
order by one fused kernel launch per direction; the inverse runs the slabs
backwards in reverse order. One stage is the original single-field model.

## Flow-gradient conditioning

These optional settings change how the flow lattices are optimized, without
adding loss terms or changing the model parameterization. The smoothing,
lazy-moment and shared-second-moment switches default to off; gradient
clipping defaults to disabled. The `optimizer_flow_*` settings apply at run
boundaries and are read every step.

The step order is: DDP gradient averaging, NaN/Inf detection and replacement
with zero, optional clipping, optional smoothing, then the optimizer update.
Sanitizing before smoothing prevents a single invalid entry from
contaminating its neighborhood.

- `optimizer_flow_grad_smoothing` Gaussian-smooths each flow lattice's
  gradient. `optimizer_flow_grad_smoothing_sigma_voxels` sets the standard
  deviation in scroll-voxel units of the flow coordinate frame. Cartesian
  smoothing is isotropic. Cylindrical smoothing runs along z and periodically
  around each ring, independently for the local z/radial/tangential components.
  `optimizer_flow_grad_smoothing_across_sigma_voxels` optionally smooths across
  rings at matching angles; its default of 0 disables this pass. These are
  lattice directions approximating along/across-sheet directions, not measured
  sheet tangents or winding boundaries. Kernels are truncated at three sigma
  and renormalized at nonperiodic borders to preserve constant gradients.
- `optimizer_flow_grad_smoothing_low_res_sigma_voxels` overrides the coarse
  lattice's along-sheet width; 0 uses the same width as the fine lattice. With
  the default fine spacing of 16 voxels and sixfold coarse spacing, a width of
  32 voxels is 2 fine cells but only about 0.33 coarse cells. Very small widths
  become identity kernels. The fitter reports effective widths at startup
  when smoothing is enabled. The across-ring width has no separate coarse
  override and is ignored for Cartesian fields.
- `optimizer_flow_lazy_moments` updates moments and applies the gradient step
  only to entries whose **processed gradient is nonzero**. Smoothing can make
  an entry active even without a direct sample there. Other entries retain
  their moments; configured AdamW weight decay still applies everywhere.
  This preserves gradient-scale history through unsampled steps and suppresses
  momentum-only movement there, but retains stale history too. It does not
  prevent large relative steps on a previously untouched smoothing tail.
- `optimizer_flow_shared_second_moment` uses one denominator across all cells
  and vector components of each lattice's leading slab, independently for
  each flow stage and for the coarse/fine lattices. Per-entry Adam scaling can
  make even a smooth gradient produce nearly sign-sized steps, particularly
  on first touch. A shared denominator preserves relative magnitudes and
  direction of the first moment before lazy masking and weight decay; it
  does not guarantee a smooth final displacement. The shared statistic is the
  mean of positive stored second moments, capped before averaging at
  `optimizer_flow_shared_second_moment_clip_quantile` (default 0.99). A value
  of 1 disables the cap. This limits outliers' effect on the common scale at
  the cost of local adaptivity. Full per-entry moments remain stored.
- `optimizer_flow_grad_clip_median_multiple` clips individual gradient
  components to this multiple of the median nonzero absolute component value,
  separately for each lattice slab. Zero disables it. Clipping before blur
  limits how far an extreme gradient can affect neighbors and optimizer
  moments, but can also suppress legitimate corrections and change vector
  direction. The median and second-moment cap are estimated from fixed-stride
  samples; the final averages and clipped fractions use the whole slab.
  Sampling is deterministic for identical inputs but may miss sparse support.
- `model_flow_field_low_res_lr_scale` multiplies the scheduled base learning
  rate for the coarse lattice independently of the fine lattice's existing
  ramp. It defaults to 1; 0 freezes coarse parameter values, including weight
  decay, although moments may still update. The fitter reads it every step,
  but the configuration catalog currently classifies it as a model-rebuild
  setting, like the other `model_` learning-rate controls.

`flow_grad_smoothing.py` provides Triton CUDA kernels and PyTorch reference/
fallback paths. `lazy_moment_adamw.LazyMomentAdamW` retains AdamW's state format
so its flags can be switched between runs without converting moment buffers.
For custom optimizer updates, diagnostics report the denominator, nonzero
update fraction, and per-component RMS over nonzero updates in voxel units.
These describe flow-parameter increments, excluding weight decay, not final
sheet displacement after integration. After both moment flags are disabled,
stored optimizer diagnostics can still describe the last custom step rather
than the current fused AdamW step. Clipping diagnostics report the bound
and fraction clipped. No full-scroll accuracy or throughput comparison is
established by the implementation tests.

### Rationale and references

Gaussian update smoothing has precedent in
[Vercauteren et al., *Diffeomorphic Demons*](https://www-sop.inria.fr/asclepios/Publications/Tom.Vercauteren/DiffeoDemons-NeuroImage08-Vercauteren.pdf).
Smooth velocity metrics are central to
[Beg et al., *Computing Large Deformation Metric Mappings*](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Beg05.pdf).
These motivate the preconditioning here; this discrete blur followed by Adam,
clipping and masking is not an implementation of classical Sobolev gradient
descent or LDDMM, and inherits no guarantee of the same fitted solution or
fold-free numerical integration. Direction-dependent smoothing has a related
motivation in [Pace et al.'s sliding-organ registration](https://pmc.ncbi.nlm.nih.gov/articles/PMC4112204/),
although this fitter uses cylindrical coordinates rather than detected sliding
interfaces.

Lazy moments follow the masked-update idea of
[PyTorch SparseAdam](https://docs.pytorch.org/docs/main/generated/torch.optim.SparseAdam.html),
using an explicit nonzero mask on dense gradients, a global per-parameter step
counter, AdamW's epsilon placement, and optional decoupled weight decay.
[Adam-mini](https://arxiv.org/abs/2406.16793) provides precedent for sharing
adaptive scales within parameter blocks; our grouping and winsorization are
custom choices, and retaining full moments does not provide its state-memory
saving. [Koloskova et al., *Revisiting Gradient Clipping*](https://proceedings.mlr.press/v202/koloskova23a/koloskova23a.pdf)
analyze clipping's stabilization and stochastic bias; their results do not
validate this particular sampled-median rule or its combination with Adam.

## Spiral service host setup

VC3D connects to a Spiral service in one of three modes, all speaking the same
authenticated HTTP protocol:

- **Localhost** — VC3D launches and owns the service on loopback. Nothing to
  set up beyond the Python environment; the dataset (plus optional output and
  cache roots) is chosen in the connection panel and VC3D launches the bound
  service with those values. Selecting a different dataset restarts the owned
  service — one service instance is bound to one dataset.
- **Remote (SSH)** — the supported internet flow. SSH access to the host is
  the only client-side prerequisite: VC3D opens and manages its own SSH
  tunnel, reads the service's auto-generated API key over SSH, and attaches to
  a persistent loopback service you start on the host. VC3D never starts the
  service on a remote host.
- **Remote (LAN)** — direct HTTP on a trusted network, authenticated with the
  service's auto-generated API key. No reverse proxies, VPNs, or manual
  tunnels are ever required.

In every mode the service — not the client — owns the base inputs: it is
launched with `--dataset` (inputs) and `--output` (all generated state),
resolves the dataset once at startup, and advertises the result through
`/dataset`. `--output` must resolve outside the dataset root; the optional
`--cache` (derived host caches) defaults to the documented user cache,
`$XDG_CACHE_HOME/vc3d/spiral` (`~/.cache/vc3d/spiral`). Clients can add
ephemeral inputs, commit them, and change run parameters, but cannot repoint
the session at different host paths.

### Creating the Spiral Python environment

The service host needs the Spiral environment (a CUDA-capable PyTorch plus the
dependencies in `pyproject.toml`, Python ≥ 3.14). With [uv](https://docs.astral.sh/uv/):

```sh
cd spiral-fitting
uv sync            # creates .venv from pyproject.toml
```

This also builds Spiral's native helpers as `vc_spiral.spiral_sampling`,
`vc_spiral.track_crossings`, `vc_spiral.track_store`, and
`vc_spiral.surface_index`. OpenMP is used when the toolchain provides it; the
same modules build with serial kernels when it does not.

or with conda/pip, install `torch` for your CUDA version and then
`pip install -e .` from `spiral-fitting/`.

On Linux x86-64, `uv sync` installs `brook-cu12`, CuPy and cuCIM for the GPU
backend of `extract_surface_tracks.py` instead of Kimimaro, its CPU backend;
on a machine without an NVIDIA GPU of compute capability 8.0 or newer, add
Kimimaro with `uv sync --extra cpu` (with pip: `pip install -e '.[cpu]'`), and
pass `--extra cpu` on later syncs too, since a plain `uv sync` removes it again.
Other platforms get Kimimaro by default.

### Resident sparse field pools

Normals and gradient magnitude samples are served by fully
resident device brick pools. Each store's occupied bricks are packed once
into a flat sidecar next to the source zarr by `pack_resident_pools.py`:

```sh
python pack_resident_pools.py /path/to/lasagna_inputs \
    --ct /path/to/<scroll>_ds2.zarr --ct-group 2 --verify 2000
```

`--ct` zeroes every voxel whose CT voxel reads 0 (the mask region around the
scroll) so those bricks drop out of the pool and sample as no-data. The
fitter loads the sidecars restricted to the configured z-ROI in one
sequential read per channel (for the full s1 ROI: ~10 GiB normals); after
that every gather is pure device indexing with no I/O and no
eviction. When a required sidecar is missing, the fitter builds it before GPU
loading and reports chunk progress. In DDP runs only rank 0 builds it. Manual
prepacking with `--ct` remains useful because the CT mask can substantially
reduce the resident pool size.
Set `FIT_SPIRAL_RESIDENT_BOUNDS_CHECK=1` to enable per-gather bounds
assertions when debugging new sampling code.

### Internet flow (SSH attach)

Start a persistent loopback service on the GPU host with its dataset. Give
each independently operated service a stable session name, port, and GPU.
Nothing is exposed on the network; VC3D tunnels to it over SSH:

```sh
tmux new -s spiral-alice 'python spiral_service.py --port 8765 \
    --dataset /data/scrolls/s1 --output /data/spiral-output/s1 \
    --gpus 0 --session-name alice'

tmux new -s spiral-bob 'python spiral_service.py --port 8766 \
    --dataset /data/scrolls/s1 --output /data/spiral-output/s1 \
    --gpus 1 --session-name bob'
```

The service uses only physical CUDA device `0` by default. Select a different
device or enable distributed fitting across several GPUs with a comma-separated
host-side list:

```sh
python spiral_service.py --port 8765 \
    --dataset /data/scrolls/s1 --output /data/spiral-output/s1 --gpus 0,1,2,3
```

Multi-GPU sessions run one fitter rank per listed device and split the configured
per-step sample counts across those ranks by default. The device list is fixed for
the lifetime of the service; restart it to change the selection.

A named service writes autosaves, previews, artifacts, uploaded checkpoints,
Lasagna output, and ephemeral inputs beneath `<output>/<session-name>/`, held
under an exclusive lease: two live services cannot own the same
output/session-name pair. Launches without `--session-name` use `<output>/`
directly. Permanent dataset inputs and the shared user cache stay untouched —
nothing generated is ever written under the dataset root.

Every completed Spiral preview is flattened by the host's Lasagna service
before it becomes downloadable in VC3D. The published grid uses a fixed
20-voxel output step: each dimension is
`ceil(((source_points - 1) * source_step) / 20) + 1`. Winding membership,
loss-map overlays, and run differences are transferred through Lasagna's
output-to-source correspondence so they remain aligned when the output grid
dimensions differ from the Spiral grid. If flattening or artifact mapping
fails, the service reports the publication error and VC3D keeps displaying the
previous successfully published preview.

Every checkpoint and every raw preview export carry a content digest of the
model state that placed the surface (`model_state_sha256`: live parameters,
frozen constraint-bake epochs and run window). The service records each
published preview under that digest in
`<output>/.spiral-published/preview-index.json`. When the fitter reports that
its resident model equals a checkpoint - after a save, an in-session load or a
startup resume - the service re-shows the flattened surface it already
published for that digest instead of waiting for a new export and flatten, and
pins that surface against the fixed-count preview retention for as long as the
checkpoint file exists. A checkpoint saved without a published preview, or one
whose preview has already been pruned, gets nothing back; request a preview as
before.

On first start the service generates a strong API key at
`~/.config/vc3d/spiral_api_key` (mode `0600`) and prints it to the console.
For an SSH profile you never copy it: VC3D reads that file over SSH.

In VC3D's Spiral workspace, add a *Remote (SSH)* profile with the
`[user@]host` destination (your `~/.ssh/config` aliases, agents, and jump
hosts work unchanged) and the service port (`8765` above), then Connect.
Non-interactive SSH authentication (keys or an agent) is required. If SSH does
not trust the host key yet, run `ssh <destination>` once in a terminal to
accept it — VC3D deliberately never auto-trusts host keys.

The fit survives viewer disconnects, laptop sleep, and network drops;
disconnecting or closing VC3D never terminates a service it did not launch.
The workspace reports the active loading, optimization, checkpoint, and
preview stage with elapsed time. Stages with a real work total also show a
counter and ETA; opaque native or CUDA operations deliberately use an
indeterminate bar instead of a guessed overall percentage. The same stage
updates are printed by standalone `fit_spiral.py`, with periodic elapsed-time
heartbeats when output is captured to a log.
While connected, the circular-arrow button beside the connection controls
restarts the remote service and reconnects automatically. The service replaces
its own process in place, so a containing `tmux` session remains alive and an
attached terminal is not disconnected.

### Trusted-LAN flow (direct HTTP)

```sh
python spiral_service.py --bind 0.0.0.0 --port 8765 \
    --dataset /data/scrolls/s1 --output /data/spiral-output/s1
```

Copy the API key printed at startup into the *Remote (LAN)* profile's API key
field (or export `SPIRAL_API_KEY` before starting VC3D). A non-loopback bind
always requires an API key (auto-generated when absent).

**Plaintext-HTTP risk note:** direct HTTP is not encrypted — on-path observers
can read the API key and the transferred data, so use it only on networks the
operator trusts. Over the internet, use an SSH profile instead. HTTPS
endpoints behind an existing TLS proxy also work; VC3D uses normal system CA
validation and never ignores certificate errors.

### API key file

- Location: `~/.config/vc3d/spiral_api_key` (respects `XDG_CONFIG_HOME`), or
  pass `--api-key-file PATH`.
- The key is created on first start (mode `0600`) and reused on later starts.
- To rotate it, delete the file and restart the service; reconnect clients
  with the new key. The key is never written to HTTP logs, responses, or the
  ready line — the console print at startup is the intended way to obtain it.
- `--nonce` is only for processes launched and owned by VC3D.

### Datasets, output, and cache

`--dataset` must point at a dataset root containing at least `umbilicus.json`
and `spiral-scroll.json`; the service refuses to start when either is missing.
Verified patches are required when their default-on input toggle is active,
but a patch-free fit can initialize with that source disabled. The dataset
holds inputs only.

`--output` is required and must resolve outside the dataset root. Every piece
of generated state — run directories, autosaves, previews, published
artifacts, ephemeral inputs, upload staging, and uploaded checkpoints — lives
under it (under `<output>/<session-name>` for a named service). Make sure its
filesystem has room for checkpoints and previews.

`--cache` holds derived host caches (content-addressed, shareable between
datasets). It defaults to `$XDG_CACHE_HOME/vc3d/spiral`
(`~/.cache/vc3d/spiral`) and must also resolve outside the dataset root. The
headless `fit_spiral.py` CLI accepts the same `--cache` with the same default
(`FIT_SPIRAL_CACHE_DIR` still overrides it for the CLI).

If the dataset root is read-only the fit still works, but *Commit current
inputs* is unavailable (committing writes inputs into the dataset).

### Connecting from VC3D

Open the Spiral workspace and pick the profile in the *Spiral Service*
section. For the local profile, set the dataset root (and optionally output
and cache roots) there — VC3D launches its owned service bound to those
values. Connection must succeed (an authenticated `/health` handshake and an
API-version check) before session controls enable. The base-input rows always
populate read-only from the service's advertised dataset resolution; run
parameters (z range, iterations, advanced config) stay editable and persist
per profile. Generated previews, geometry, and
checkpoints transfer through the artifact API into a local cache — no shared
filesystem is needed. Optional: set the profile's **Local dataset path** if
this machine mounts the same dataset, so input surface overlays
(verified patches/shell) can be displayed locally. It is assumed to
correspond to the dataset root the service advertises, which is the prefix
service paths are translated from; without it those overlays are simply marked
unavailable.

`spiral-scroll.json` in the dataset root is the only source of the scroll's
name and voxel resolution and of the Lasagna store layout (zarr groups,
coordinate scale). None of them are panel settings: the panel reports them
read-only, and the service rejects a session request that carries
`scroll_name`, `voxel_size_um`, `lasagna_group` or `lasagna_scale`.

Optional supervision sources have rebuild-scoped boolean switches in Advanced
config. Set an `input_use_*` key to `false` to skip validation, loading,
sampling, and losses for that source without changing its tuned weights or
sample counts. Available switches cover verified patches, tracks,
fibers, each PCL role (`absolute`, `relative`, `same_winding`, and
`drawn_control_points`), normals, gradient magnitude, winding inference, and
the outer shell. For example:

```json
{
  "input_use_tracks": false,
  "input_use_fibers": false,
  "input_use_pcl_drawn_control_points": false
}
```

Most role switches require a whole-fit rebuild; same-winding and relative PCL
switches apply at the next Run. Disabled roles retain their accepted workspace
content, and enabling a role restores that desired content. Disabling a
prerequisite also disables its dependent supervision: winding inference
needs the outer shell.

The API 33 client and service use one revisioned input workspace per dataset.
One service holds the dataset editing lease and one client owns that workspace;
other connections can observe. Disconnecting retains ownership, drafts and
command receipts. Reconnect with the same client resumes them. A fit rebuild
replays desired inputs without replacing the editing workspace. Reconnect and
rebuild also discover newly added dataset inputs, preserving their collection
IDs and any existing workspace edits. A restarted service replaces a clean
client catalog; pending edits or commands stay tied to their original workspace.

**Add/Apply changes** captures the selected local revisions and applies the
whole validated batch at a worker boundary, including while idle or after the
final step. **Commit current inputs** first applies the selected revisions,
then persists those exact revisions. Editing can continue during either action;
a response for an older revision leaves newer edits dirty. Failed transfers or
publication can be retried with the retained command and bytes.

In the flattened Spiral preview, tap `Ctrl` to toggle patch painting: left-drag
paints, right-drag erases, and `Escape` exits. Ctrl+wheel changes brush size
without toggling the mode; Shift+right-drag draws control-point lines. Starting
a stroke on a session-drawn selection extends that patch in its original color.
Drawn selections remain brush-editable after Apply and Commit, including across
preview updates when their original surface can be projected onto the new one.
Finalization keeps the largest edge-connected component of complete quads;
selections with no complete quad remain editable and report an error.

Drawn patches use the same editing workspace as other inputs. Add/Apply and
Commit capture their current selections; an edit made while an older snapshot
is being saved stays dirty. Erasing a previously staged patch completely stages
a deletion on the next Add/Apply. The input list shows local brush errors and
colors; Remove on a local brush edit discards that edit. Dataset patches use
the existing managed patch editor.

Brush regression checks use the existing build and Python environment:

```bash
AGENTS_AGENT_MODE=1 cmake --build volume-cartographer/build --target VC3D test_spiral_brush_patch test_spiral_input_workflow test_spiral_input_draft test_spiral_point_placement_mode test_spiral_point_collection_edit -j 4
AGENTS_AGENT_MODE=1 QT_QPA_PLATFORM=offscreen SPIRAL_TEST_PYTHON="$PWD/spiral-fitting/.venv/bin/python" ctest --test-dir volume-cartographer/build -R '^spiral_(brush_patch|input_workflow|input_draft|point_placement_mode|point_collection_edit)$' --output-on-failure
```

Set `SPIRAL_PATCH_REAL_INPUT` to a real tifxyz patch directory to include the
read-only selection check and draft-copy validation. `SPIRAL_PATCH_EVIDENCE_DIR`
optionally receives before/after selection-mask images. These checks cover CPU
geometry and the editing workflow; they do not run GPU fitting.

In the Spiral workspace, `Q` and `E` place same-winding and relative-winding
points. Relative annotations count 0, 1, 2, ... in placement order; `F` reverses
the chain and mirrors annotations. Existing collections remain editable after
Apply. Shift+E prepares local drafts. Managed patch and fiber editors save to
session working copies; successful saves update drafts and do not apply or
commit automatically. Linked-fiber saves use working copies for their peers.

The input list includes baseline dataset entries and additions, with a filter
and selection checkboxes. Edit, Remove, Restore, Retry and Discard Local Changes
operate on individual drafts. Applying Remove stops future supervision;
Commit deletes the managed dataset entry. Restore remains available before
committing a deletion. Baseline restores reference the retained service revision
and do not require access to the service filesystem. Removal cannot reverse
completed optimizer steps.
Invalid selected drafts keep the batch unapplied and remain editable. Scoped
external-source conflicts offer Use Current, Apply Local After Review, or Save
Local as New; unrelated PCL collection changes are merged during publication.

Editing workspaces are disposable session storage. **Discard and Exit** drops
local drafts and releases the workspace directly. **Commit** exits only after
publication succeeds, then releases temporary copies. Release and service
shutdown stop file users and remove the entire `editing-workspaces/<id>`
directory, including baseline snapshots, revisions and uploads. Reconnect,
Stop, and Rebuild keep the workspace and its drafts. The next claim snapshots
only committed dataset inputs.

New workspaces carry versioned ownership metadata and a lifetime advisory lock.
Startup reclaims marked directories whose owner has exited, including after a
crash or forced termination. Live owners, symlinks, unrecognized markers and
legacy folders without markers are left untouched; legacy storage requires
manual cleanup. Dataset inputs, saved fit outputs, exports and shared caches
are preserved. Interrupted Commit recovery directories (`.spiral-publication-*`
beside dataset targets) are retained and their paths logged. Cleanup errors are
reported with the workspace path; when a reader or worker cannot stop, files
and ownership remain intact for a release retry or later startup reclamation.

Cleanup regression checks (no installation needed):

```bash
AGENTS_AGENT_MODE=1 PYTHONPATH=spiral-fitting spiral-fitting/.venv/bin/python -m pytest -q \
  spiral-fitting/tests/test_workspace_cleanup.py \
  spiral-fitting/tests/test_service_editing.py \
  spiral-fitting/tests/test_service_editing_http.py \
  spiral-fitting/tests/test_spiral_service_v2.py
```

Set `SPIRAL_REAL_PCL=/path/to/real/abs_winding.json` for the real-scroll
close/reopen check. It edits a temporary dataset copy and verifies that both the
source bytes and committed copy remain unchanged; it uses a resident test double
and does not run GPU fitting.

Regression checks for this workflow (using existing environments/builds):

```bash
# From the repository root:
AGENTS_AGENT_MODE=1 spiral-fitting/.venv/bin/python -m pytest -q \
  spiral-fitting/tests/test_service_editing.py spiral-fitting/tests/test_service_editing_http.py \
  spiral-fitting/tests/test_revisioned_runtime.py spiral-fitting/tests/test_revisioned_geometry.py
AGENTS_AGENT_MODE=1 ninja -C volume-cartographer/build test_spiral_input_draft test_spiral_input_workflow -j 2
AGENTS_AGENT_MODE=1 QT_QPA_PLATFORM=offscreen SPIRAL_TEST_PYTHON="$PWD/spiral-fitting/.venv/bin/python" ctest --test-dir volume-cartographer/build -R '^spiral_input_(draft|workflow)$' --output-on-failure
```

The workflow tests start a loopback HTTP service and cover remote baseline
restore, service restart, and edits made while a submission is running.
The service suite includes transfer, revision, conflict, and publication recovery
checks; runtime and geometry tests cover worker boundaries and supervision.
For the retained CUDA check on temporary copies of a real dataset patch:

```bash
AGENTS_AGENT_MODE=1 SPIRAL_REVISION_LIVE_DATASET=/path/to/dataset \
  SPIRAL_REVISION_PATCH=patch-directory-name \
  spiral-fitting/.venv/bin/python -m pytest -q spiral-fitting/tests/test_revisioned_live_fit.py
```

Input uploads only transfer immutable bytes. The editing workspace owns
acceptance, application, and persistence; there is no separate ephemeral-input
ledger or automatic commit on editor save. Checkpoint uploads remain service-scoped.

Directional DT timing is an independent control on every interactive Run.
When **Restrict DT losses to final** is unchecked, the Run adds no DT gate.
When checked, the adjacent percentage is the eligible suffix of the originally
requested Run: the first eligible iteration is
`run_start + floor(iterations * (1 - percentage / 100))`. Thus 25% of a
10,000-iteration Run suppresses DT for 7,500 iterations and permits it for the
final 2,500; for small Runs the eligible step count is rounded up. Zero percent
suppresses DT for the whole Run and 100% adds no suppression. Stopping early
does not recalculate the original window. The schedule is transient Run state,
not advanced configuration or checkpoint state, and is cleared before the
Run's autosave.

Input-local validation failures reject the entire selected candidate without
changing active supervision. Distributed ranks prepare the same candidate and
must all agree before installation. Unexpected worker/device failures retain
fail-stop behavior. Publication prepares all outputs before changing targets;
a failure after publication starts retains the transaction and queues later
mutations behind recovery. Recovery covers the running service, not crashes,
and does not promise atomic visibility across multiple dataset files.

**Checkpoints** are one panel section, and loading one is one button. It lists
what the service advertises (checkpoints at the dataset root, and those under
the output directory such as the autosave) plus any **client-local `.ckpt`**
you browse for; a local file is uploaded to the service's
`<output>/uploaded-checkpoints/` directory on the way (the panel shows
progress and the transfer restarts if interrupted). Checkpoints are identified
by SHA-256, so choosing content the service already retains reuses it without
transferring the file again, and the service validates new archives and keeps
the newest few unique uploads.

Before the first fit is initialized, *Load* initializes it directly from the
selected checkpoint; it does not first construct a throwaway model. The
configuration profile becomes **Checkpoint** and displays the resolved
configuration carried by that checkpoint. *Initialize Fit* is the separate
from-scratch action.

With an existing fit, *Load* replaces the resident model's weights, optimiser
and RNG state in place. When the checkpoint does not match the live model the service refuses
it and says what a rebuild would have to replace: rebuilding the **model only**
keeps the loaded dataset inputs and everything already added to the fit, while
a **whole-fit** rebuild re-reads the dataset and replays the workspace's desired
revisions, including uncommitted additions. The panel reports the reasons and asks; a checkpoint no
rebuild can accept — one written against another dataset, or whose stored
configuration holds a value the schema cannot interpret — is reported and
nothing is offered. A checkpoint-backed session takes its durable configuration
from the checkpoint, so the local advanced-config profile does not override it.

A checkpoint's stored configuration is loaded tolerantly
(`checkpoint_migrations.tolerate_config`): keys the schema no longer has are
dropped and keys the checkpoint predates take their current defaults. Each
such edit is reported as a note or session warning. Only a stored value the
schema cannot validate (a retired enum member, an out-of-range number) refuses
the checkpoint. Tolerance covers configuration alone: a checkpoint whose
parameters do not fit the live model — a multi-stage flow saved in the
pre-slab `extra_flow_fields.*` layout, or an exponential-gap fit from before
late August 2026 — is still refused on tensor geometry.

The Iterations value on *Run* is a count added to the checkpoint's durable
iteration. The progress bar is local to that run and therefore starts at zero;
the session status line reports the global current and target iterations.

The section also holds *Save on Service* and *Download…*, and reports the
checkpoint the resident fit was actually built from. That report is read-only:
it is not a field, and a rebuild carries it forward by itself.

### Shutdown and logs

Stop the service with `Ctrl-C` or `SIGTERM` (`tmux kill-session -t spiral`);
it tears the fit session down at a safe boundary. Logs go to the service's
stdout/stderr on the host — for a `tmux` session, `tmux attach -t spiral`; for
an unowned service VC3D's Python-output dialog only reminds you of this. A
service started on an explicit port can be restarted immediately (the socket
uses `SO_REUSEADDR`). VC3D's remote restart control does not run
`tmux kill-session`; it gracefully closes the fit and re-executes the service
with the same interpreter, arguments, and process ID. Note that a large artifact
download during a running fit competes with the fitter for the Python
interpreter and can slow iterations somewhat.

### Optional systemd user unit

```ini
# ~/.config/systemd/user/spiral-service.service
[Unit]
Description=VC3D Spiral fitting service

[Service]
WorkingDirectory=%h/villa/spiral-fitting
ExecStart=%h/villa/spiral-fitting/.venv/bin/python \
    %h/villa/spiral-fitting/spiral_service.py \
    --port 8765 --dataset /data/scrolls/s1 \
    --output /data/spiral-output/s1 --gpus 0
Restart=on-failure

[Install]
WantedBy=default.target
```

```sh
systemctl --user daemon-reload
systemctl --user enable --now spiral-service
journalctl --user -u spiral-service -f     # logs (includes the API key print)
```

Direct command-line use remains fully supported; the unit is a convenience.

## Extracting surface tracks

`extract_surface_tracks.py` turns a surface-prediction volume into the tracks
DBM that the next sections pack, index and rasterize. The predictions are
binarized (`> 0`) and cut into thin slabs: horizontal ribbons 4 voxels deep
every 16 voxels of z, then vertical zx and zy slabs 4 voxels thick every 64
voxels across the occupied yx range. In every slab the connected components
are labelled, max-pooled by 4, filtered by size and skeletonized, and each
skeleton chain between branch or end points with at least 10 vertices becomes
a track.

```sh
python extract_surface_tracks.py \
    --predictions /path/to/<surface-predictions>.zarr/0 \
    --out /path/to/<dataset>/tracks/<name>.dbm \
    --z-min 10900 --z-max 11300
```

Every option defaults to the configuration at the top of the script, which
also holds the slab geometry, the size thresholds and `path_mode`. Its two
paths are placeholders: the script stops with an error until they are set or
passed as `--predictions` and `--out`. `--predictions` takes a local path or a
URL such as `s3://...`, opened with `open_zarr` from `../vesuvius/src`; the GPU
backend reads a local zarr v2 array in the layout of the surface predictions
(uint8, blosc, `/` chunk keys) straight from its chunk files instead. The z
range is half-open (`[z-min, z-max)`).

Two backends write the same keys in the same format:

- `gpu`: Brook (`brook-cu12`) and cuCIM on NVIDIA GPUs, one worker process
  per GPU. It needs Linux x86-64, GPUs of compute capability 8.0 or newer
  (Ampere or later) and an NVIDIA driver R545 or newer (R570 or newer where the
  driver JIT-compiles Brook's PTX). The CUDA 12 libraries come as pip wheels
  (`brook-cu12` bundles its runtime; CuPy and cuCIM use NVIDIA's CUDA wheels
  installed with them), so no system CUDA toolkit is needed. `--gpus 0,1`
  selects GPUs by their `nvidia-smi` index; the default is every visible GPU.
  Without `--gpus`, a set `CUDA_VISIBLE_DEVICES` is read the same way
  (`nvidia-smi` indices, PCI bus order); UUID and MIG entries are not accepted,
  so pass `--gpus` instead. With `--gpus`, `--backend auto` does not fall back
  to Kimimaro when the GPU backend cannot run.
- `cpu`: Kimimaro, one slab at a time. `uv sync` installs it on every platform
  except Linux x86-64, where it is the optional `cpu` extra
  (`uv sync --extra cpu`) for machines without a compatible GPU.

`--backend auto` (the default) picks the GPU backend when its check passes
(platform, packages, and the NVIDIA driver version and the compute capability
of the selected GPUs, as reported by `nvidia-smi`) and `path_mode` is
`'interjoint'`, and Kimimaro otherwise. The
script prints the backend it uses and why. `--backend gpu` and `--backend cpu`
stop with the reason when that backend cannot run. The GPU backend implements
only the `'interjoint'` path mode.

Brook implements Kimimaro's skeletonization on the GPU. Its skeletons are close
to Kimimaro's but not always identical, so the two backends' DBMs agree closely
rather than byte for byte (see the measurements below). Kimimaro collects the
skeletons of its 8 worker processes as they finish, so the order of the tracks
within a key can also differ between two CPU runs.

Every key (`h:{z}`, `vy:{y}`, `vx:{x}`) holds a pickled list of `(N, 3)` int32
ZYX arrays in full-resolution voxels, and an empty slab holds an empty list.
Keys already in the DBM are skipped, so an interrupted run resumes where it
stopped, with either backend; delete the DBM to recompute it. When
`write_native_packed_store` is set (the default), the script then writes the
packed store described in the next section. That step imports `tracks.py`,
which needs PyTorch; `--no-packed-store` skips it, and `convert_track_store.py`
can write the store later.

The GPU backend decodes the z range once into a bit-packed block in POSIX
shared memory (`/dev/shm`) of about (z-max − z-min) × Y × X / 8 bytes, where Y
and X are the full extent of the volume, plus at most a quarter of that again
(at the default slab spacing) for the byte columns of the vx slabs. For 400
slices of a 7888 × 8096 volume the block takes 3.2 GB. The script stops with an
error when `/dev/shm` has too little free space.

Measured on Scroll 1 surface predictions with the default slab geometry and
thresholds. Times run from the start of the extraction to the closed DBM,
reading and decoding included, interpreter start-up and the packed store not:

| Input (Y × X) | z range | Hardware | Backend | Runs | Time |
| :--- | :--- | :--- | :--- | ---: | ---: |
| public predictions, 7888 × 8096 | 10900–11300 | Intel Core i9-14900KF | Kimimaro, 8 workers | 1 | 336.9 s |
| same | 10900–11300 | NVIDIA RTX 4090, driver 615 | GPU | 3 | 4.1, 4.2, 4.2 s |
| predictions, 8174 × 8174 | 10900–11300 | 4 × NVIDIA H100 80GB, driver 570 | GPU | 3 | 4.4, 4.5, 4.6 s |
| same | 10752–14848 | 4 × NVIDIA H100 80GB | GPU | 3 | 15.7, 15.9, 19.6 s |
| same | 10752–14848 | 1 × NVIDIA H100 80GB | GPU | 1 | 40.0 s |

The 19.6 s run was the first in a new environment (numba compiles and caches
its kernels once). On z 10752–14848 (446 keys) the GPU backend wrote 1,006,831
tracks with 41,530,237 points, and a Kimimaro run on the same range wrote
1,006,740 tracks with 41,543,412 points (+0.01 % tracks, −0.03 % points). The
GPU DBMs were byte-identical across runs and between 1 and 4 GPUs, and the tests
compare them with the per-slab loop run with `brook.skeletonize`. These are
single inputs; speed and agreement depend on the data and the GPUs.

Run the tests from the repository root:

```sh
PYTHONPATH=spiral-fitting spiral-fitting/.venv/bin/python -m pytest -q \
  spiral-fitting/tests/test_extract_surface_tracks.py
```

They compare `fast_tracks.py` with the networkx walk of the CPU backend on
fixed and random graphs, check the backend selection and the command line
without a GPU or Kimimaro, and run the CPU backend on a small synthetic volume
when Kimimaro is installed. With Brook, CuPy, cuCIM and a GPU of compute
capability 8.0 or newer, they also run the GPU backend on one GPU on two small
synthetic zarr stores, one read by the direct chunk reader and one through the
fallback reader, and compare the DBMs with the per-slab loop run with
`brook.skeletonize`.

## Packing large track databases

Legacy track DBMs store a pickled list of NumPy arrays in every key. For large
datasets this spends minutes decoding millions of Python objects each time a
fit starts. Convert a DBM once to the adjacent packed format:

```sh
python convert_track_store.py \
    /data/tracks/2um_ds2_ps256_surf_v2.dbm
```

This writes `2um_ds2_ps256_surf_v2.dbm.vctracks/` atomically. The directory
contains contiguous coordinates, ragged offsets, source IDs, family codes,
Z bounds, arclengths, and tortuosities. `fit_spiral.py` automatically prefers
a current adjacent packed store while retaining the DBM as the authoritative
source and compatibility fallback. A source-file fingerprint prevents a stale
store from being used after the DBM changes; rerun with `--force` to replace it.

The native `vc_spiral.track_store` loader memory-maps the packed files, applies the Z
ROI from per-track metadata, and emits one compact float32 ragged array without
constructing per-track Python objects. The crossing builder also stages
directly from a current packed store, bypassing DBM and pickle decoding.

## Caching exact track crossings

Crossing-connected track sampling needs the exact shared voxels between the
horizontal and vertical track families. Build that index once as a CSR
sidecar instead of sorting every track point whenever a fit session loads:

```sh
python build_track_crossings.py \
    /data/tracks/2um_ds2_ps256_surf_v2.dbm \
    --z-min 4000 --z-max 17000 \
    --temp-dir /fast/disk/tmp
```

The optional Z range is half-open (`[z-min, z-max)`) and retains only tracks
entirely contained in that range. Omit both options to index the whole DBM.
The standalone builder uses a hybrid memory/disk index: it streams DBM tracks
into temporary coordinate and packed-voxel files, keeps the coordinates
memory-mapped, then loads and radix-sorts the packed keys in RAM. The native
`vc_spiral.track_crossings` kernel uses all requested workers for sorting,
exact-voxel discovery, arclength calculation, and pair consolidation. The
extension is built with the other Spiral native modules by `uv sync` from this
directory. A slower Python fallback remains available.

The builder needs roughly 20 bytes of temporary disk space per selected point.
The native radix sort temporarily holds about 32 RAM bytes per point; after the
sort, those arrays are released before the 8-byte-per-point arclength vector and
compact 16-byte crossing events are consolidated. This avoids retaining either
the selected track database or Python dictionaries of crossing pairs in RAM.
Temporary files are removed after the sidecar is written. Without `--temp-dir`,
the temporary workspace is created beside the tracks DBM rather than under the
system temporary directory.

The script writes
`/data/tracks/2um_ds2_ps256_surf_v2.dbm.crossings.npz` atomically.
`fit_spiral.py` finds it automatically from the configured tracks path. The
sidecar includes a fingerprint of every DBM backing file; a stale or malformed
file is ignored and the fitter falls back to its in-memory exact crossing
scan. Re-run the builder after changing the DBM (`--force` replaces a current
cache). A range-limited sidecar can serve the same or a narrower fitting Z
range; building another range replaces it. Point-level track exclusion also
uses the fallback because clipping a
track changes its crossing-local indices.

## Converting track DBMs to OME-Zarr

`tracks_to_ome_zarr.py` rasterizes the ZYX polylines produced by
`extract_surface_tracks.py` into a compressed `uint8` OME-Zarr. Value 0 is
background; values 1–255 are assigned with proximity-aware reuse and display
as categorical colors with VC3D's Glasbey colormap. Rasterization uses worker
processes, while independent Zarr chunks are compressed and written by a
thread pool using Zstandard level 3.

Use a paired OME-Zarr to copy the exact volume shape and physical geometry:

```sh
python tracks_to_ome_zarr.py \
    /data/tracks/2um_ds2_ps256_surf_v2.dbm \
    --out /data/tracks/2um_ds2_ps256_tracks.ome.zarr \
    --like /data/volumes/2um.ome.zarr \
    --like-group 0
```

Alternatively pass `--shape Z,Y,X`. If neither `--shape` nor `--like` is
given, the script first scans the DBM and uses the maximum track coordinate
plus one. The explicit forms avoid that extra pass for large databases.
`--resume` continues an interrupted conversion. Multiple positional DBMs are
combined into one output, so separate scrolls should be converted in separate
commands.

## Neural winding-inference losses

Set `dense_spacing_mode` to `winding_model` and provide the compact exported
crossing directory at the conventional `<dataset>/winding_inference` path or
override `paths.winding_inference` in `spiral-scroll.json`. Two vocabularies
deliberately coexist: `winding_model` names the fitting mode and its tunables
(`sample_count_winding_model_*`, `winding_model_relative_pair_delta`,
`winding_model_huber_delta`, the `dense_spacing_winding_model_*` losses),
while `winding_inference` names the exported artifact and everything tied to
its on-disk identity (the input path, the `winding_inference_crossings`
artifact type, and the checkpoint fingerprint field). The store is
checksum-verified and copied to each fitting GPU at startup; rays whose
crossings cannot intersect the configured z-range are excluded from sampling,
and optimisation then does no inference-store filesystem I/O. The default
24,000 samples per step are split evenly between long relative-winding pairs
(`sample_count_winding_model_relative_pairs`, index separation drawn from
`winding_model_relative_pair_delta`) and adjacent-passage density pairs
(`sample_count_winding_model_density_pairs`). The independent Lasagna normal
and native minimum-spacing losses remain available alongside it.

The compact store is created by the Vesuvius winding-model
`export_spiral_supervision.py` tool; see its `NATIVE_PHASE_CACHE.md` for the
exact export command and format.

For a headless fit, pass the dataset root with `--dataset` and select inference
mode (plus any independently disabled losses) through
`FIT_SPIRAL_CONFIG_OVERRIDES`. The dataset's `spiral-scroll.json` and the
declarative input catalog determine which conventional inputs are resolved.

## Fiber direction samples

The optional fiber-direction loss consumes one packed artifact extracted from a
remote Lasagna fiber prediction. Extraction downloads only chunks intersecting
the requested z ROI and keeps the highest-presence voxel in each fixed
prediction-space cell:

```bash
./.venv/bin/python fiber_direction_samples.py \
  https://example/fibers.lasagna.json \
  /path/to/dataset/fiber_directions.npz \
  --z-roi 10000,11000 --output-scale 4 \
  --presence-threshold 160 --cell-size 2
```

The z ROI is half-open and expressed in the output/fitter coordinate system;
`--output-scale 4` means one fitter `/2` coordinate is four base `/0` voxels. The
extractor always covers the fiber volume's complete XY extent.

Both `input_use_fiber_directions` and `loss_weight_fiber_directions` default
to off/zero; set the toggle true and the weight above zero to enable the
loss. The fitter then loads the conventional `fiber_directions.npz` artifact
and samples `sample_count_fiber_direction_points` observations per step.
Positions and directions constrain only local fitted-sheet orientation; they
do not attach a sample to a particular winding.

## Fiber classification

Each fiber strip is tagged at load time (`spiral_helpers.classify_fiber_hv`):
VC3D's manual H/V tag wins, then its automatic tag when the recorded certainty
reaches `pcl_vertical_fiber_min_auto_certainty`, then a geometric fallback that
calls a strip vertical when its z extent is at least
`pcl_vertical_fiber_min_z_fraction` of its path length. The fit log reports the
vertical / horizontal / untagged split.

## Vertical-fiber radial offset

Papyrus carries its horizontal fibers on the front face of the sheet (the face
toward the umbilicus and the lower-winding neighbour) and its vertical fibers
on the back face, so the two fiber classes sit a few voxels apart along the
sheet normal and at most one of them can lie on the face the fit targets. The
strip losses read every fiber as lying *on* the fitted winding, which leaves a
permanent residual on vertical strips and on every vertical-to-horizontal link
junction. With `pcl_vertical_fiber_radial_offset_enabled` set, vertical strips
(as classified by `spiral_helpers.classify_fiber_hv`, see above) carry a
per-point target offset of `pcl_vertical_fiber_radial_offset_voxels` (default
4, positive = outward) along the sheet normal of the fitted spiral (the
scan-space gradient of the fitted winding, not the straight line to the
umbilicus): the radius loss, the whole-strip DT target, the DT snap, and the
satisfaction metric all expect those points that far outside the winding, on
its back face, instead of on it. Horizontal strips and regular point
collections are never offset.

The offset is a physical scroll-space distance, not a spiral-space constant.
In spiral space the fitted sheet's normal is the radial direction, so the
offset acts on the shifted radius, but the scroll-to-spiral map is not an
isometry (the gap expander rescales the radial coordinate per gap and the flow
stretches locally), so each point's offset is multiplied by the transform's
local stretch along the sheet normal, `|J^T n|`
(`sample_spiral.get_radial_normal_stretch`; the same `J^T` covector transport
the dense-normals loss uses). `J^T n` is the scan-space gradient of the fitted
winding, so a positive offset displaces the expected fiber position along it:
the increasing-winding direction of the fitted spiral at that point, away from
the umbilicus, which coincides with the line from the umbilicus only where the
sheet happens to be perpendicular to it. The loss evaluates that stretch for its sampled
points each step under `no_grad`, the DT target cache and the satisfaction
metric for the points they read. After a constraint bake the resident geometry
lives in the frozen stack's output frame, so each vertical strip and fiber
catalog point also carries `radial_offset_bake_scale`, the product of every
frozen epoch's normal stretch at the pre-bake point
(`fit_spiral.accumulate_radial_offset_bake_scale`); ingested inputs pick it up
when they are pushed through the stack, re-materialised strips inherit it from
the catalog, and the final scroll-space export drops it because the composed
transform's Jacobian then carries the whole stretch. The offset travels with
the strip bundle (`radial_offsets`, resident-frame voxels after the bake
scale) next to the winding annotations, so linked components mixing both fiber
classes read as one winding. Unlike the other `pcl_` settings, both keys are
Run-scoped: a Run that changes them refills every retained strip's offsets from
its stored vertical/horizontal tag and rebuilds the strip bundle and DT target
caches, so no fit rebuild is needed. VC3D's spiral panel exposes them as a
checkbox and a voxel distance next to the fibers path.

## Point-to-patch linking

Every point of every point collection (regular PCLs and fibers) is attached to
the patch surface it lies on when the inputs load, and again for inputs added
to a running session. A point attaches when a patch surface is within
`pcl_link_distance_tolerance` scroll voxels (default 2.5). General collections
take the largest-area patch within tolerance, then the nearest;
`between_patches__A__B` collections take the nearest of their named pair.
Every linking setting applies at a Run boundary. The tolerance and window
settings re-link every resident collection, regular and fiber, from the
retained catalogs against the resident patches and re-derive all the
point-collection views (cross-patch groups and unattached strips); the fiber
side-rule settings below only affect fibers, so they re-link and
re-materialise the fibers alone.

`pcl_link_window_points` and `pcl_link_window_min_points` (both default 1)
gate candidates on their neighbours: a patch the point itself lies within
tolerance of is eligible only when at least `pcl_link_window_min_points` of the
centred window of `pcl_link_window_points` consecutive points (id order, the
point included) also lie within tolerance of it. Eligible candidates are ranked
as usual, largest area then nearest, so a fiber stays on the big patch its
neighbours share instead of hopping onto a patch only one point touches. Even
window counts round up to the next odd count, and at a collection's ends the
requirement is clipped to the window members available. Points already
attached to a patch count as window members when a session relinks.

`pcl_fiber_link_side_filter` keeps fibers off patches on the wrong side of the
sheet, which is how a fiber ends up on an adjacent winding when windings touch.
The front of a sheet faces inward, toward the umbilicus and the neighbouring
winding with the lower winding number; horizontal fibers lie on that front
face and vertical fibers on the back. A vertical fiber (classified as described above) therefore only attaches to a patch whose surface is
in front of it, a horizontal fiber only to one behind it, and untagged fibers
and regular collections are unrestricted. A hit is rejected when the projection
foot lies more than `pcl_fiber_link_side_margin_voxels` (default 0.5) on the
wrong side of the point along the inward direction; the margin absorbs points
lying on the traced surface itself.

No fitted transform exists when the inputs load, so the inward direction starts
as the line to the umbilicus at the point's z, which local deformation can turn
away from the true sheet normal. Once `pcl_fiber_link_model_direction_step`
steps (default 10000) have completed, whether run in the session or restored
from a checkpoint, every fiber is relinked once with the inward direction taken
from the fitted spiral's decreasing-winding direction (the negative scan-space
gradient of the fitted winding, `fit_spiral.inward_winding_direction`), and the
fiber training views are re-materialised. A checkpoint from before that step
loaded after the switch relinks back under the umbilicus direction at its first
step.
