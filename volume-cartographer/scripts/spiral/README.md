# Spiral fitting

Code and helpers to fit a canonical Archimedean spiral to deformed scrolls.
`spiral_service.py` hosts one persistent interactive fit session over HTTP for
the VC3D Spiral workspace; `fit_spiral.py` is the underlying fitter.

## Spiral service host setup

VC3D connects to a Spiral service in one of three modes, all speaking the same
authenticated HTTP protocol:

- **Localhost** — VC3D launches and owns the service on loopback. Nothing to
  set up beyond the Python environment; this preserves the fully local
  workflow where every input path is editable.
- **Remote (SSH)** — the supported internet flow. SSH access to the host is
  the only client-side prerequisite: VC3D opens and manages its own SSH
  tunnel, reads the service's auto-generated API key over SSH, and attaches to
  a persistent loopback service you start on the host. VC3D never starts the
  service on a remote host.
- **Remote (LAN)** — direct HTTP on a trusted network, authenticated with the
  service's auto-generated API key. No reverse proxies, VPNs, or manual
  tunnels are ever required.

In both remote modes the service — not the client — owns the base inputs: it
is launched with `--dataset`, resolves it at startup, and advertises the
resolution to clients. Remote clients can add ephemeral inputs, commit them,
and change run parameters, but cannot repoint the session at different host
paths.

### Creating the Spiral Python environment

The service host needs the Spiral environment (a CUDA-capable PyTorch plus the
dependencies in `pyproject.toml`, Python ≥ 3.14). With [uv](https://docs.astral.sh/uv/):

```sh
cd scripts/spiral
uv sync            # creates .venv from pyproject.toml
```

or with conda/pip, install `torch` for your CUDA version and then
`pip install -e scripts/spiral`.

### Internet flow (SSH attach)

Start a persistent loopback service on the GPU host with its dataset. Nothing
is exposed on the network; VC3D tunnels to it over SSH:

```sh
tmux new -s spiral 'python scripts/spiral/spiral_service.py --port 8765 \
    --dataset /data/scrolls/s1 --gpus 0'
```

The service uses only physical CUDA device `0` by default. Select a different
device or enable distributed fitting across several GPUs with a comma-separated
host-side list:

```sh
python scripts/spiral/spiral_service.py --port 8765 \
    --dataset /data/scrolls/s1 --gpus 0,1,2,3
```

Multi-GPU sessions run one fitter rank per listed device and split the configured
per-step sample counts across those ranks by default. The device list is fixed for
the lifetime of the service; restart it to change the selection.

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
While connected, the circular-arrow button beside the connection controls
restarts the remote service and reconnects automatically. The service replaces
its own process in place, so a containing `tmux` session remains alive and an
attached terminal is not disconnected.

### Trusted-LAN flow (direct HTTP)

```sh
python scripts/spiral/spiral_service.py --bind 0.0.0.0 --port 8765 \
    --dataset /data/scrolls/s1
```

Copy the API key printed at startup into the *Remote (LAN)* profile's API key
field (or export `SPIRAL_API_KEY` before starting VC3D). A non-loopback bind
always requires both an API key (auto-generated when absent) and `--dataset`.

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

### Datasets and output

`--dataset` must point at a dataset root containing at least `umbilicus.json`
and `verified_patches/`; the service refuses to start when required entries
are missing and prints what was missing. Output goes to
`<dataset>/spiral_output` by default (from the same resolution VC3D shows).
Make sure that directory's filesystem has room for checkpoints and previews.
If the dataset root is read-only the fit still works, but *Commit current
inputs* is unavailable and the cache falls back to the user cache directory.

### Connecting from VC3D

Open the Spiral workspace and pick the profile in the *Spiral Service*
section. Connection must succeed (an authenticated `/health` handshake and an
API-version check) before session controls enable. In remote modes the
base-input rows populate read-only from the service's advertised dataset
resolution; run parameters (z range, iterations, advanced config) stay
editable and persist per profile. Generated previews, geometry, and
checkpoints transfer through the artifact API into a local cache — no shared
filesystem is needed. Optional: set the profile's path map
(service prefix → local prefix) if this machine mounts the same dataset, so
input surface overlays (verified/unverified/shell) can be displayed locally;
without a mapping those overlays are simply marked unavailable.

While a session is active you can right-click a patch in the Surface panel or
a fiber in the Fibers panel and pick *Add to current spiral fit*. Added inputs
are uploaded into a session-scoped ephemeral folder, used from the next run
onward, and can be moved into the dataset with *Commit current inputs*.

Interactive influence settings are scoped to each **Run** request. The fitter
builds a fresh influence region from only the inputs pending for that run,
uses it for the requested iteration window, and discards it before autosaving.
Influence masks, limits, and controls are not checkpoint state. All
`interactive_influence_*` advanced settings can therefore change between runs
without reloading the resident session. The **Disable DT** percentage controls
how much of that run suppresses directional DT losses after incorporating its
pending inputs.

**Resume checkpoints on a remote profile:** the Checkpoint field accepts a
service-advertised checkpoint (a `*.ckpt` at the dataset root), a service path
under the output directory (for example the autosave), or a **client-local
`.ckpt` file** — use the browse button. A local file is uploaded to the
service's `<output>/uploaded-checkpoints/` directory before the session loads
(the panel shows progress; the transfer restarts if interrupted). The service
validates the archive, never overwrites an existing upload, and keeps the
newest few uploaded checkpoints. To bring a fit result back to the client, use
*Download Checkpoint…*.

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
WorkingDirectory=%h/volume-cartographer
ExecStart=%h/volume-cartographer/scripts/spiral/.venv/bin/python \
    %h/volume-cartographer/scripts/spiral/spiral_service.py \
    --port 8765 --dataset /data/scrolls/s1 --gpus 0
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
python scripts/spiral/convert_track_store.py \
    /data/tracks/2um_ds2_ps256_surf_v2.dbm
```

This writes `2um_ds2_ps256_surf_v2.dbm.vctracks/` atomically. The directory
contains contiguous coordinates, ragged offsets, source IDs, family codes,
Z bounds, arclengths, and tortuosities. `fit_spiral.py` automatically prefers
a current adjacent packed store while retaining the DBM as the authoritative
source and compatibility fallback. A source-file fingerprint prevents a stale
store from being used after the DBM changes; rerun with `--force` to replace it.

The native `vc.track_store` loader memory-maps the packed files, applies the Z
ROI from per-track metadata, and emits one compact float32 ragged array without
constructing per-track Python objects. The crossing builder also stages
directly from a current packed store, bypassing DBM and pickle decoding.

## Caching exact track crossings

Crossing-connected track sampling needs the exact shared voxels between the
horizontal and vertical track families. Build that index once as a CSR
sidecar instead of sorting every track point whenever a fit session loads:

```sh
python scripts/spiral/build_track_crossings.py \
    /data/tracks/2um_ds2_ps256_surf_v2.dbm \
    --z-min 4000 --z-max 17000 \
    --temp-dir /fast/disk/tmp
```

The optional Z range is half-open (`[z-min, z-max)`) and retains only tracks
entirely contained in that range. Omit both options to index the whole DBM.
The standalone builder uses a hybrid memory/disk index: it streams DBM tracks
into temporary coordinate and packed-voxel files, keeps the coordinates
memory-mapped, then loads and radix-sorts the packed keys in RAM. The native
`vc.track_crossings` kernel uses all requested workers for sorting, exact-voxel
discovery, arclength calculation, and pair consolidation. Build it for a source
checkout with `ninja -C build vc_track_crossings`; installed VC Python packages
include it automatically. A slower Python fallback remains available.

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
python scripts/spiral/tracks_to_ome_zarr.py \
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
