# Spiral fitting

Code and helpers to fit a canonical Archimedean spiral to deformed scrolls.
`spiral_service.py` hosts one persistent interactive fit session over HTTP for
the VC3D Spiral workspace; `fit_spiral.py` is the underlying fitter.

## Scroll specification (spiral-scroll.json)

`fit_spiral.py` requires a `spiral-scroll.json` file in the dataset root.
This is not covered by scrollprize.org/tutorial_spiral. Required keys:

- `schema_version` — must equal `1`.
- `name`, `voxel_size_um` — required, no validation beyond presence.
- `spiral_outward_sense` — must be `"CW"` or `"ACW"` (case-insensitive).
  No automated method determines this; it is read off the CT data by a
  person in VC3D, or computed from an already-fitted spiral.

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

## Fitting a scroll that ships only tracks (the 2025-2026 scans)

The spiral datasets published for the 2025-2026 scans
(`dl.ash2txt.org/datasets/spiral_datasets/<scroll>/<volume>/`) contain a
`tracks/` directory and nothing else: no `umbilicus.json`, no `outer_shell/`,
no patches, no `lasagna_inputs/`. The headless fitter still needs three more
things. This is the layout that worked for PHerc0125, PHerc0826, PHerc0211 and
PHerc0358 (30,000-step fits on one RTX 3090 under WSL2):

```
<dataset>/
  spiral-scroll.json                  # see "Scroll specification" above
  umbilicus.json                      # published, or estimated (below)
  tracks/<volume>_surface_m7_L0_th0.2.dbm                # 9-13 GB, from spiral_datasets
  tracks/<volume>_surface_m7_L0_th0.2.dbm.crossings.npz  # 2-3 GB, same place
  tracks/<volume>_surface_m7_L0_th0.2.extract.json
  lasagna_inputs/las_008_nx.ome.zarr/2        # copied from the open-data bucket, z-slab only
  lasagna_inputs/las_008_ny.ome.zarr/2
  lasagna_inputs/las_008_grad_mag.ome.zarr/2
```

**Umbilicus.** The open-data bucket publishes one for some scrolls under
`<scroll>/representations/umbilicus/<volume>-umbilicus-<date>.json`
(PHerc0125, PHerc0139, PHerc0211, PHerc0332 and PHerc0826 at the time of
writing; PHerc0139's is annotated on a different scan from its only surface
store). For the others, `estimate_umbilicus.py` derives control points from
the organizers' surface prediction: at each of N heights it takes the largest
connected component of the sheet mask at pyramid level 3, fills it, and takes
as the core the point of the distance-to-boundary plateau (within 0.9 of the
maximum) nearest the centroid of the component. `core=argmax` selects the
previous behaviour, the plain argmax of the distance transform, and reproduces
files written before that option existed byte for byte. The plateau rule
exists because the distance transform is nearly flat over a wide region, so
its argmax picks one pixel out of many almost equal ones and jumps between
lobes of the section from one slice to the next.

Measured against the published PHerc0125 umbilicus at the default `n_z=12`,
over the 75 published control points inside the estimate's z coverage: plateau
0.1-6.5 mm (median 2.2), argmax 0.5-11.0 mm (median 3.0), with the plateau
point closer at 57 of the 75. Most of the gain is in the worst cases rather
than the median, which is what the lobe jumping predicts. Those 75 points fall
in 11 intervals between estimate knots and are not independent, so treat a
paired test across them as indicative rather than as a p-value.

A PHerc0358 fit from an estimated umbilicus (slices 8000-9500, 100 windings,
30,000 steps) ended with 50% of track points satisfied, against 12-38% for the
three scrolls fitted from published umbilici, so the estimate is good enough
for the tracks to pull the spiral into place, but it is a starting point for
the fit rather than a substitute for a published umbilicus.

```sh
python estimate_umbilicus.py PHerc0358 \
    PHerc0358/representations/predictions/surface/<volume>-surface-<run>-surface-m7-L0-th0.2.zarr/ \
    <dataset>/umbilicus.json
```

**Lasagna inputs.** The normal fields live at
`<scroll>/representations/predictions/lasagna/<volume>-lasagna-<run>/<scroll>_{nx,ny,grad_mag}.ome.zarr`;
for these scrolls group `"2"` is a 4x downsample, so `spiral-scroll.json`
needs `"normal_zarr_group": "2"` and `"lasagna_scale": 4`. The fitter only
reads the z-window it optimises, so copying the chunks for `[z_begin/4 - 50,
z_end/4 + 50)` of each store into `lasagna_inputs/las_008_<name>.ome.zarr/2`
(keeping the chunk grid and the store's `.zattrs`/`.zarray`) is enough; a
1,500-slice band is about 2 GB for the three stores.

**Configuration.** The headless CLI takes configuration overrides as JSON in
the `FIT_SPIRAL_CONFIG_OVERRIDES` environment variable and the output root in
`FIT_SPIRAL_OUT_DIR`. With tracks only, the switches that matter are:

```sh
export FIT_SPIRAL_OUT_DIR=/path/to/out
export FIT_SPIRAL_CONFIG_OVERRIDES='{
  "z_begin": 9000, "z_end": 10500, "optimizer_num_training_steps": 30000,
  "input_use_tracks": true,
  "input_disable_patches": true,
  "loss_weight_shell_outer": 0, "loss_weight_shell_patch_radius": 0,
  "dense_spacing_mode": "grad_mag", "loss_weight_dense_spacing": 0,
  "shell_outer_winding_idx": 90, "model_gap_expander_num_windings": 90
}'
python fit_spiral.py --dataset <dataset> --cache ~/spiral_cache
```

- `input_use_tracks` defaults to `false`; without it the fit silently
  optimises against the umbilicus and the normals only.
- The two shell-loss weights must be `0` because there is no outer shell (the
  winding model needs one too, so `dense_spacing_mode` cannot be
  `winding_model`); `phase` spacing needs a surf-SDT store these datasets do
  not have, hence `grad_mag` with weight `0`.
- `shell_outer_winding_idx` and `model_gap_expander_num_windings` default to
  130, which is Scroll 1's winding count. The exporter writes windings
  `[output_first_winding, shell_outer_winding_idx)`, so leaving 130 on a
  60-winding scroll exports dozens of windings that lie outside the papyrus.
  Set both to a little above the scroll's own count. Counting sheet crossings
  along radial rays through the organizers' surface prediction at three
  heights (`winding_counts.py`) gives:

  | scroll | median crossings per ray (three heights) | used |
  |---|---|---|
  | PHerc0125 | 58 | 90 |
  | PHerc0826 | 53-60 | 70 |
  | PHerc0211 | 60-77 | 90 |
  | PHerc0257 | 67-68 | 90 |
  | PHerc0358 | 69-82 | 100 |
  | PHerc0191 | 70-83 | |
  | PHerc0813 | 72-79 | |
  | PHerc0800 | 79-88 | |
  | PHerc0268 | 86-101 | |
The count is a lower bound where sheets are pressed together, so the value
  used should sit above the range.

**What to expect.** A 1,500-slice band runs at 7-8 it/s on an RTX 3090 (about
70 minutes for 30,000 steps) and reports 12-18% satisfied track points at the
end. Meshes are written to
`<out>/<date>_<scroll>_slice-<z_begin>-<z_end>_0-patch/meshes/fitted/w<NNN>`
as tifxyz at scale 0.05. Fits with tracks and no outer shell need #1732 (or
`"input_use_outer_shell": false`); a truncated `crossings.npz` from an
interrupted download needs #1735 (or delete the file so it is rebuilt).

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
dependencies in `pyproject.toml`, Python ≥ 3.14). With
[uv](https://docs.astral.sh/uv/):

```sh
cd spiral-fitting
uv sync            # creates .venv from pyproject.toml
```

This also builds Spiral's native helpers as `vc_spiral.spiral_sampling`,
`vc_spiral.track_crossings`, `vc_spiral.track_store`, and
`vc_spiral.surface_index`. OpenMP is used when the toolchain provides it; the
same modules build with serial kernels when it does not.

or with conda/pip, install `torch` for your CUDA version and then `pip install
-e .` from `spiral-fitting/`.

### Resident sparse field pools

Normals, gradient magnitude, and surf-SDT samples are served by fully resident
device brick pools. Each store's occupied bricks are packed once into a flat
sidecar next to the source zarr by `pack_resident_pools.py`:

```sh
python pack_resident_pools.py /path/to/lasagna_inputs \
    --ct /path/to/<scroll>_ds2.zarr --ct-group 2 --verify 2000
```

`--ct` zeroes every voxel whose CT voxel reads 0 (the mask region around the
scroll) so those bricks drop out of the pool and sample as no-data. The fitter
loads the sidecars restricted to the configured z-ROI in one sequential read
per channel (for the full s1 ROI: ~33 GiB SDT + ~10 GiB normals); after that
every gather is pure device indexing with no I/O and no eviction. When a
required sidecar is missing, the fitter builds it before GPU loading and
reports chunk progress. In DDP runs only rank 0 builds it. Manual prepacking
with `--ct` remains useful because the CT mask can substantially reduce the
resident pool size. Set `FIT_SPIRAL_RESIDENT_BOUNDS_CHECK=1` to enable
per-gather bounds assertions when debugging new sampling code.

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
device or enable distributed fitting across several GPUs with a
comma-separated host-side list:

```sh
python spiral_service.py --port 8765 \
    --dataset /data/scrolls/s1 --output /data/spiral-output/s1 --gpus 0,1,2,3
```

Multi-GPU sessions run one fitter rank per listed device and split the
configured per-step sample counts across those ranks by default. The device
list is fixed for the lifetime of the service; restart it to change the
selection.

A named service writes autosaves, previews, artifacts, uploaded checkpoints,
Lasagna output, and ephemeral inputs beneath `<output>/<session-name>/`, held
under an exclusive lease: two live services cannot own the same
output/session-name pair. Launches without `--session-name` use `<output>/`
directly. Permanent dataset inputs and the shared user cache stay untouched —
nothing generated is ever written under the dataset root.

Every completed Spiral preview is flattened by the host's Lasagna service
before it becomes downloadable in VC3D. The published grid uses a fixed
20-voxel output step: each dimension is `ceil(((source_points - 1) *
source_step) / 20) + 1`. Winding membership, loss-map overlays, and run
differences are transferred through Lasagna's output-to-source correspondence
so they remain aligned when the output grid dimensions differ from the Spiral
grid. If flattening or artifact mapping fails, the service reports the
publication error and VC3D keeps displaying the previous successfully
published preview.

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
heartbeats when output is captured to a log. While connected, the
circular-arrow button beside the connection controls restarts the remote
service and reconnects automatically. The service replaces its own process in
place, so a containing `tmux` session remains alive and an attached terminal
is not disconnected.

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
per profile. Generated previews, geometry, and checkpoints transfer through
the artifact API into a local cache — no shared filesystem is needed.
Optional: set the profile's **Local dataset path** if this machine mounts the
same dataset, so input surface overlays (verified/unverified/shell) can be
displayed locally. It is assumed to correspond to the dataset root the service
advertises, which is the prefix service paths are translated from; without it
those overlays are simply marked unavailable.

`spiral-scroll.json` in the dataset root is the only source of the scroll's
name and voxel resolution and of the Lasagna store layout (zarr groups,
coordinate scale). None of them are panel settings: the panel reports them
read-only, and the service rejects a session request that carries
`scroll_name`, `voxel_size_um`, `lasagna_group` or `lasagna_scale`.

Optional supervision sources have rebuild-scoped boolean switches in Advanced
config. Set an `input_use_*` key to `false` to skip validation, loading,
sampling, and losses for that source without changing its tuned weights or
sample counts. Available switches cover verified/unverified patches, tracks,
fibers, each PCL role (`absolute`, `relative`, `same_winding`, and
`drawn_control_points`), normals, surface SDT, gradient magnitude, winding
inference, and the outer shell. For example:

```json
{
  "input_use_tracks": false,
  "input_use_fibers": false,
  "input_use_pcl_drawn_control_points": false
}
```

Changing one requires a whole-fit rebuild. Disabling a prerequisite also
disables its dependent supervision: phase spacing needs normals and surface
SDT, while winding inference needs the outer shell.

While a session is active you can right-click a patch in the Surface panel or
a fiber in the Fibers panel and pick *Add to current spiral fit*. Added inputs
are uploaded into a session-scoped ephemeral folder, used from the next run
onward, and can be moved into the shared dataset with *Commit current inputs*.
Commits from multiple service processes are serialized; distinct inputs and
point collections are preserved, while an existing patch or fiber identifier
is reported as a conflict and is never overwritten.

Interactive influence settings are scoped to each **Run** request. The fitter
builds a fresh influence region from only the inputs pending for that run,
uses it for the requested iteration window, and discards it before autosaving.
Influence masks, limits, and controls are not checkpoint state. All
`interactive_influence_*` advanced settings can therefore change between runs
without reloading the resident session. The **Disable DT** percentage controls
how much of that run suppresses directional DT losses after incorporating its
pending inputs.

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
and RNG state in place. When the checkpoint does not match the live model the
service refuses it and says what a rebuild would have to replace: rebuilding
the **model only** keeps the loaded dataset inputs and everything already
added to the fit, while a **whole-fit** rebuild re-reads the dataset and
discards added inputs that were never committed. The panel reports the reasons
and asks; a checkpoint no rebuild can accept — one written against another
dataset, or against a configuration schema this service does not have — is
reported and nothing is offered. A checkpoint-backed session takes its durable
configuration from the checkpoint, so the local advanced-config profile does
not override it.

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
uses `SO_REUSEADDR`). VC3D's remote restart control does not run `tmux
kill-session`; it gracefully closes the fit and re-executes the service with
the same interpreter, arguments, and process ID. Note that a large artifact
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

## Packing large track databases

Legacy track DBMs store a pickled list of NumPy arrays in every key. For large
datasets this spends minutes decoding millions of Python objects each time a
fit starts. Convert a DBM once to the adjacent packed format:

```sh
python convert_track_store.py \
    /data/tracks/2um_ds2_ps256_surf_v2.dbm
```

This writes `2um_ds2_ps256_surf_v2.dbm.vctracks/` atomically. The directory
contains contiguous coordinates, ragged offsets, source IDs, family codes, Z
bounds, arclengths, and tortuosities. `fit_spiral.py` automatically prefers a
current adjacent packed store while retaining the DBM as the authoritative
source and compatibility fallback. A source-file fingerprint prevents a stale
store from being used after the DBM changes; rerun with `--force` to replace
it.

The native `vc_spiral.track_store` loader memory-maps the packed files,
applies the Z ROI from per-track metadata, and emits one compact float32
ragged array without constructing per-track Python objects. The crossing
builder also stages directly from a current packed store, bypassing DBM and
pickle decoding.

## Caching exact track crossings

Crossing-connected track sampling needs the exact shared voxels between the
horizontal and vertical track families. Build that index once as a CSR sidecar
instead of sorting every track point whenever a fit session loads:

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
The native radix sort temporarily holds about 32 RAM bytes per point; after
the sort, those arrays are released before the 8-byte-per-point arclength
vector and compact 16-byte crossing events are consolidated. This avoids
retaining either the selected track database or Python dictionaries of
crossing pairs in RAM. Temporary files are removed after the sidecar is
written. Without `--temp-dir`, the temporary workspace is created beside the
tracks DBM rather than under the system temporary directory.

The script writes `/data/tracks/2um_ds2_ps256_surf_v2.dbm.crossings.npz`
atomically. `fit_spiral.py` finds it automatically from the configured tracks
path. The sidecar includes a fingerprint of every DBM backing file; a stale or
malformed file is ignored and the fitter falls back to its in-memory exact
crossing scan. Re-run the builder after changing the DBM (`--force` replaces a
current cache). A range-limited sidecar can serve the same or a narrower
fitting Z range; building another range replaces it. Point-level track
exclusion also uses the fallback because clipping a track changes its
crossing-local indices.

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
(`sample_count_winding_model_density_pairs`). In this mode surf-SDT is neither
loaded nor required, while the independent Lasagna normal and native
minimum-spacing losses remain available.

The compact store is created by the Vesuvius winding-model
`export_spiral_supervision.py` tool; see its `NATIVE_PHASE_CACHE.md` for the
exact export command and format.

For a headless fit, pass the dataset root with `--dataset` and select
inference mode (plus any independently disabled losses) through
`FIT_SPIRAL_CONFIG_OVERRIDES`. The dataset's `spiral-scroll.json` and the
declarative input catalog determine which conventional inputs are resolved.

## Fiber direction samples

The optional fiber-direction loss consumes one packed artifact extracted from
a remote Lasagna fiber prediction. Extraction downloads only chunks
intersecting the requested z ROI and keeps the highest-presence voxel in each
fixed prediction-space cell:

```bash
./.venv/bin/python fiber_direction_samples.py \
  https://example/fibers.lasagna.json \
  /path/to/dataset/fiber_directions.npz \
  --z-roi 10000,11000 --output-scale 4 \
  --presence-threshold 160 --cell-size 2
```

The z ROI is half-open and expressed in the output/fitter coordinate system;
`--output-scale 4` means one fitter `/2` coordinate is four base `/0` voxels.
The extractor always covers the fiber volume's complete XY extent.

Both `input_use_fiber_directions` and `loss_weight_fiber_directions` default
to off/zero; set the toggle true and the weight above zero to enable the loss.
The fitter then loads the conventional `fiber_directions.npz` artifact and
samples `sample_count_fiber_direction_points` observations per step. Positions
and directions constrain only local fitted-sheet orientation; they do not
attach a sample to a particular winding.
