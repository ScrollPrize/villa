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
uses `SO_REUSEADDR`). Note that a large artifact download during a running fit
competes with the fitter for the Python interpreter and can slow iterations
somewhat.

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
