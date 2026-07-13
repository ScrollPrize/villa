# Volume Cartographer MCP server (experimental)

`vc_mcp_server` exposes Volume Cartographer surface generation, bounded volume
analysis, registered-surface evidence, learned-model adapters, and human review
as [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) tools and
resources. It is implemented in C++23 with
[FastMCPP](https://github.com/0xeb/fastmcpp), pinned to commit
`9a3ee7125b1db9dc85a71f60661beb40ea1e7993` (upstream version 3.4.4).
Optional, isolated Python adapters perform Zarr staging and analysis.

The server is intentionally local and experimental. It supports stdio and
authenticated loopback Streamable HTTP; it is not a remotely deployable service.
Jobs and idempotency state live in the server process, while job artifacts are
written to persistent local directories.

## What this enables

An MCP-capable client can now run a review-gated workflow from a prediction to
inspectable evidence:

1. Inspect a remote prediction Zarr and search a bounded region for deterministic
   seed candidates.
2. Generate a surface with VC's real `vc_grow_seg_from_seed` executable using a
   fixed conservative profile.
3. Poll, cancel, inspect logs, preview the generated TIFXYZ, and verify every
   artifact with SHA-256.
4. Import a generated TIFXYZ into a canonical surface Zarr bundle and register a
   bounded raw-CT region against it.
5. Measure geometry, CT alignment, periodic/grid structure, epoch-fold structure,
   and stability across explicitly supplied perturbations.
6. Render model-compatible normal stacks and, when pinned assets are configured,
   run Dataset 058 nnU-Net, ResNet152 ink-model, DINOv3, or DinoVol adapters.
7. Preserve raw model components while fusing/ranking evidence for review.
8. Create immutable review queues and reviewer assessments, then evaluate supplied
   labels for coverage, ranking quality, calibration, and reviewer agreement.
9. View supported results in the embedded **Vellum Lens** MCP App, with text and
   structured-output fallbacks for clients without MCP Apps support.

This does **not** make generated surfaces correct by construction, classify ink
with certainty, or transcribe text. Geometry, alignment, structural, model, and
fusion scores are evidence or review-priority heuristics—not calibrated truth.

## Design and safety properties

- Real surface growth is launched with a fixed executable and argument vector;
  no shell is involved.
- Local inputs must be absolute and existing. Remote Zarr access is allowlisted,
  bounded, and staged by a separate fixed process.
- Chained operations accept immutable `{job_id, artifact_id}` references instead
  of caller-selected intermediate paths.
- Every async operation has a stable `client_request_id` and process-local
  idempotency.
- Jobs have bounded runtimes, per-job directories, manifests, captured/redacted
  logs, and immutable artifacts.
- HTTP binds only to loopback, requires a bearer token of at least 32 characters,
  applies Host/Origin checks, and returns `Cache-Control: no-store`.
- Optional tools are advertised only when all required adapter paths exist at
  server startup.
- CPU surface stacks are bounded to 129 slices, 8192 pixels per side, and 64
  million total pixels. Tool-specific schemas impose additional limits.
- The server does not implement PostgreSQL, OAuth, signed URLs, subscriptions,
  object storage, durable remote queues, or multi-user persistence.

## Quick start

### 1. Diagnose without installing

From the repository root:

```bash
.agents/skills/vc-generate-surface/scripts/vc-mcp-setup.sh doctor
```

The helper diagnoses, configures, builds, tests, prints environment templates,
and can create an optional Python environment. It never installs OS packages.
See [Agent workflow and safety](#agent-workflow-and-safety) before allowing any
install or model download.

### 2. Prerequisites and installation (only when approved)

The build requires CMake 3.28+, Ninja, a C++23 compiler, Qt 6, and the normal
Volume Cartographer dependencies. The optional Python adapters require `uv` or a
compatible Python environment plus the packages in `requirements-staging.txt`.
Learned tools additionally require their explicitly pinned repositories and
weights; the server never downloads or substitutes them.

On Ubuntu, inspect and use the repository dependency script:

```bash
sudo env AGENTS_AGENT_MODE=1 AGENTS_ALLOW_INSTALL=1 \
  bash scripts/install_build_deps.sh
```

On macOS, inspect and use the Homebrew build helper's install mode:

```bash
AGENTS_AGENT_MODE=1 AGENTS_ALLOW_INSTALL=1 \
  scripts/build_macos.sh --install-deps
```

For a manual local session the two `AGENTS_*` variables are unnecessary. They
are mandatory for agent-mode installs. Both commands have system side effects
and must not be run merely for discovery.

To create only the optional analysis Python environment after approval:

```bash
AGENTS_AGENT_MODE=1 AGENTS_ALLOW_INSTALL=1 \
  .agents/skills/vc-generate-surface/scripts/vc-mcp-setup.sh install-python
```

The default environment is `~/.cache/vc-mcp/venvs/analysis`; override it with
`VC_MCP_PYTHON_ENV=/absolute/path`.

### 3. Configure and build

Configuring the MCP target fetches pinned FastMCPP and `cpp-httplib` sources, so
a first configure needs network access unless FetchContent is already cached.

Linux, using the repository's Clang preset:

```bash
cmake --preset dev-clang \
  -DVC_BUILD_MCP_SERVER=ON \
  -DVC_TESTING=ON
cmake --build --preset dev-clang --target \
  vc_mcp_server vc_grow_seg_from_seed
```

macOS, after `scripts/build_macos.sh` has established the Homebrew LLVM build
cache:

```bash
scripts/build_macos.sh
cmake -S . -B build-macos \
  -DVC_BUILD_MCP_SERVER=ON \
  -DVC_TESTING=ON
cmake --build build-macos --target \
  vc_mcp_server vc_grow_seg_from_seed
```

Typical binaries are `build/dev-clang/bin/vc_mcp_server` on Linux and
`build-macos/bin/vc_mcp_server` on macOS. Adjust paths for another preset.

### 4. Start loopback Streamable HTTP (recommended)

Keep one server process alive for the whole session:

```bash
export VC_MCP_TRANSPORT=streamable-http
export VC_MCP_HOST=127.0.0.1
export VC_MCP_PORT=18080
export VC_MCP_AUTH_TOKEN="$(openssl rand -hex 32)"
export VC_MCP_WORK_ROOT="$HOME/.vc-mcp/jobs"
/path/to/bin/vc_mcp_server
```

Connect to `http://127.0.0.1:18080/mcp` and send:

```text
Authorization: Bearer <the value of VC_MCP_AUTH_TOKEN>
```

Generic MCP client configuration (field names vary by client):

```json
{
  "url": "http://127.0.0.1:18080/mcp",
  "headers": {
    "Authorization": "Bearer ${VC_MCP_AUTH_TOKEN}"
  }
}
```

Confirm that the client expands environment variables. If it does not, use its
secret facility; never commit the token or put it on the server command line.
The server rejects `0.0.0.0` and all non-loopback addresses.

### 5. Stdio compatibility mode

Use stdio when the client cannot use Streamable HTTP:

```json
{
  "command": "/absolute/path/to/bin/vc_mcp_server",
  "args": [],
  "env": {
    "VC_MCP_TRANSPORT": "stdio",
    "VC_MCP_WORK_ROOT": "/absolute/path/to/jobs",
    "VC_MCP_GROW_EXECUTABLE": "/absolute/path/to/bin/vc_grow_seg_from_seed"
  }
}
```

Use absolute paths and do not wrap the command in a shell. MCP protocol output is
stdout; diagnostics are stderr. Stdio is the binary's default transport, but
clients should set `VC_MCP_TRANSPORT=stdio` explicitly.

## Session bootstrap

At the beginning of every server process:

1. Initialize MCP and send `notifications/initialized` (normally handled by the
   client).
2. Call `tools/list` and `resources/list`/`resources/templates/list`.
3. Call `vc_capabilities` with `{}`.
4. Confirm the build commit, `execution_mode`, coordinate spaces, profile, and
   optional feature booleans.
5. Use the schemas returned by `tools/list` as the runtime source of truth.

Tool availability is fixed at process startup. Correct an adapter path and
restart the server if an expected optional tool is absent.

## Runtime configuration

### Core server

| Variable | Meaning | Default |
| --- | --- | --- |
| `VC_MCP_TRANSPORT` | `stdio` or `streamable-http` | `stdio` |
| `VC_MCP_HOST` | HTTP loopback host | `127.0.0.1` |
| `VC_MCP_PORT` | HTTP port, 1–65535 | `18080` |
| `VC_MCP_AUTH_TOKEN` | HTTP bearer token, at least 32 characters | required in HTTP mode |
| `VC_MCP_GROW_EXECUTABLE` | Absolute `vc_grow_seg_from_seed` path | sibling of server binary |
| `VC_MCP_WORK_ROOT` | Persistent per-job artifact root | `$HOME/.vc-mcp/jobs` |
| `VC_MCP_TIMEOUT_SECONDS` | Worker timeout, 1–604800 | `21600` |

### Analysis and staging adapters

These paths enable registered rendering and successive evidence families:

```bash
export VC_MCP_ANALYSIS_PYTHON="$HOME/.cache/vc-mcp/venvs/analysis/bin/python"
export VC_MCP_VOLUME_STAGER="$PWD/apps/VC3D/mcp/volume_stager.py"
export VC_MCP_SURFACE_BUNDLE_ADAPTER="$PWD/apps/VC3D/mcp/surface_bundle_adapter.py"
export VC_MCP_STRUCTURAL_EVIDENCE_ADAPTER="$PWD/apps/VC3D/mcp/structural_evidence_adapter.py"
export VC_MCP_EVIDENCE_FUSION_ADAPTER="$PWD/apps/VC3D/mcp/evidence_fusion_adapter.py"
export VC_MCP_REVIEW_ADAPTER="$PWD/apps/VC3D/mcp/review_adapter.py"
```

The packages required by the controlled staging/analysis adapters are pinned in
`requirements-staging.txt`. With an already-approved environment:

```bash
uv pip install --python "$VC_MCP_ANALYSIS_PYTHON" \
  -r apps/VC3D/mcp/requirements-staging.txt
```

This does not install model repositories or weights.

### Dataset 058 nnU-Net

```bash
export VC_MCP_NNUNET_PYTHON="$HOME/.cache/vc-mcp/venvs/nnunet/bin/python"
export VC_MCP_NNUNET_ADAPTER="$PWD/apps/VC3D/mcp/nnunet_segmentation_adapter.py"
export VC_MCP_VOLUME_STAGER="$PWD/apps/VC3D/mcp/volume_stager.py"
export VC_MCP_NNUNET_MODEL_DIR="/absolute/path/to/Dataset058/model"
```

### DINOv3

```bash
export VC_MCP_DINOV3_EXECUTABLE="$PWD/apps/VC3D/mcp/dinov3_headless_adapter.py"
```

The request itself must provide an absolute pinned repository, 40-character
commit, local weights, and 64-character weight SHA-256. The adapter is CPU-only,
limits crops to 2048×2048 / 4,194,304 pixels, and accepts official ViT-S/16 or
ViT-S+/16 architectures. The web-pretrained LVD-1689M ViT-S/16 weights are the
recommended first experiment.

### DinoVol

```bash
export VC_MCP_DINOVOL_PYTHON="$HOME/.cache/vc-mcp/venvs/nnunet/bin/python"
export VC_MCP_DINOVOL_ADAPTER="$PWD/apps/VC3D/mcp/dinovol_headless_adapter.py"
export VC_MCP_DINOVOL_REPOSITORY="/absolute/path/to/dinovol"
export VC_MCP_DINOVOL_REPOSITORY_COMMIT="965898b4a2de71f299304be8fc32344b0104be15"
export VC_MCP_DINOVOL_CHECKPOINT="/absolute/path/to/dinovol_teacher_backbone.pt"
```

The configured checkpoint must have SHA-256
`e041ca870dd2570f8a44d1dd26db1197b3f74121f62023bc774fbc9d40e51a59`.
The adapter runs the pinned EMA teacher backbone on MPS and projects 3D cosine
similarity back to the registered UV chart.

### ResNet152 ink-model inference

```bash
export VC_MCP_INK_MODEL_PYTHON="$HOME/.cache/vc-mcp/venvs/nnunet/bin/python"
export VC_MCP_INK_MODEL_ADAPTER="$PWD/apps/VC3D/mcp/resnet152_inference_adapter.py"
export VC_MCP_INK_MODEL_REPOSITORY="/absolute/path/to/ink-detection"
export VC_MCP_INK_MODEL_REPOSITORY_COMMIT="$(git -C "$VC_MCP_INK_MODEL_REPOSITORY" rev-parse HEAD)"
export VC_MCP_INK_MODEL_CHECKPOINT="/absolute/path/to/r152_3ddec_v2_l5_epoch13.ckpt"
```

The configured checkpoint must have SHA-256
`36dd0de84b7b7aa6590184192c7415466cd8a1ba7c1e59f42c6373846373c3e0`.
ResNet152 ink-model preprocessing is the pinned `[0,200] / 200` contract.

#### What the ink-model score means

For every valid pixel in a registered surface's UV chart, the adapter samples a
62-layer CT stack along the surface normal and runs the pinned ResNet152 +
3D-decoder checkpoint. It applies a sigmoid to the model logits, then combines
overlapping tiles using normalized Hann-window blending. The resulting
**ink-model score** is bounded to `[0,1]`.

The score is not calibrated on PHerc0332: `0.8` does not mean an 80% probability
of ink. It is learned evidence for prioritizing inspection, not proof of ink,
proof of text, a transcription, or a measure of surface quality. Its meaning is
specific to the pinned checkpoint, preprocessing, layer order, and input
registration recorded in the artifact manifest.

The API and artifacts call this the **ResNet152 ink-model score**. It is a
model output, not a repository-level score.

When `ink_fuse_registered_scores` is used, it computes a review-priority map:

```text
(w_ink_model * clip(ink_model_score, 0, 1)
 + w_dinovol * roi_percentile_normalized_dinovol
 + w_stability * clip(stability, 0, 1)) / sum(weights)
```

Default weights are `1` for the ink-model score, `1` for DinoVol, and `0.5` for
stability when stability is supplied. DinoVol normalization is ROI-relative,
and stability measures only the supplied perturbations. Consequently, the fused
value is also an uncalibrated review-priority heuristic, not an ink probability.
All raw and normalized components are preserved so reviewers do not have to
interpret only the combined value.

## Common tool conventions

### Asynchronous jobs

All generating/analysis tools return immediately with:

```json
{
  "job_id": "...",
  "state": "queued",
  "operation": "...",
  "job_resource": "vc://jobs/...",
  "log_resource": "vc://jobs/.../logs",
  "submitted_at": "..."
}
```

Use a unique, descriptive `client_request_id` of 1–128 characters for every new
submission. Reuse it only to retry the identical request in the same server
process. Poll `vc_get_job` until `succeeded`, `failed`, or `cancelled`; intermediate
states are `queued`, `starting`, and `running`. Submission is not completion.

On failure, read `vc://jobs/{job_id}/logs`. On success, inspect the job's
`artifacts` and use `vc_inspect_artifacts` for a SHA-256 inventory. Job state and
idempotency disappear on restart even though files under `VC_MCP_WORK_ROOT`
remain.

### Artifact references

Chained tools use immutable references:

```json
{"job_id": "upstream-job-id", "artifact_id": "registered-surface"}
```

Pass the exact artifact ID required by the live schema. Do not replace a
reference with an arbitrary local path. Common IDs are:

`surface`, `registered-surface`, `surface-volume`, `surface-geometry`,
`surface-ct-alignment`, `grid-coherence`, `structural-comparison`,
`surface-stability`, `ink-prediction`, `dinovol-exemplar`, `evidence-ranking`,
`review-queue`, and `review-assessment`.

### Coordinate contract

VC points are named `(x,y,z)`; NumPy volumes are indexed `(z,y,x)`. Every seed,
region, and prediction declares `ct_l0_xyz` or `ct_l2_xyz`. L0↔L2 has scale 4
and identity axis permutation, but clients must never infer or silently convert
the space. The server records submitted, normalized L0, and VC-input coordinates.
Remote growth requests require `voxel_size_um`.

## Complete tool reference

The concise signatures below show required fields and important options. All
object schemas reject unknown properties. Consult `tools/list` for exact live
JSON Schema, defaults, limits, and availability.

### Core lifecycle and growth (always available)

| Tool | Input | Purpose |
| --- | --- | --- |
| `vc_capabilities` | `{}` | Report build/version, execution mode, spaces, profiles, operations, and optional feature flags. |
| `vc_grow_surface` | growth request | Low-level asynchronous VC growth, retained for debugging. |
| `vc_generate_surface` | growth request | Preferred composition: grow, validate TIFXYZ, make generation preview, and inventory hashes. |
| `vc_get_job` | `job_id` | Read normalized input, progress, manifest, artifacts, timestamps, and terminal error. |
| `vc_cancel_job` | `job_id` | Request cancellation of a queued/running local job. |

A growth request contains exactly one of `prediction_path` (absolute existing
OME-Zarr directory) or `prediction_uri`, plus `prediction_space`, `seed`,
`profile: "scroll3-conservative-v1"`, and `client_request_id`. Optional `limits`
contains `max_generations` (1–10000, default 256) and `min_area_cm2` (0–100,
default 0.3). The profile fixes `step_size=20` and `use_cuda=false`.

```json
{
  "prediction_uri": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0332/representations/predictions/surfaces/example.zarr/",
  "prediction_space": "ct_l2_xyz",
  "voxel_size_um": 9.596,
  "seed": {"x": 2007, "y": 2009, "z": 2015, "space": "ct_l2_xyz"},
  "profile": "scroll3-conservative-v1",
  "limits": {"max_generations": 4, "min_area_cm2": 0.0},
  "client_request_id": "pherc0332-short-trial-1"
}
```

Remote growth currently permits the public Vesuvius S3 prefix and the Scroll 5
`dl.ash2txt.org` volume prefix. Queries, fragments, and `..` paths are rejected.

### Prediction and artifact inspection (always available)

| Tool | Input | Purpose |
| --- | --- | --- |
| `vc_inspect_prediction` | `prediction_uri`, `prediction_space` | Read Zarr shape/chunks/metadata and open the seed picker. |
| `vc_find_seed_candidates` | prediction, explicit `region` | Deterministically rank bounded surface candidates, optionally using a matching-grid ink prediction. |
| `vc_render_surface_preview` | `job_id` | Show TIFXYZ metadata and generation preview for completed growth. |
| `vc_inspect_artifacts` | `job_id` | Recursively list artifact size and SHA-256 plus the command manifest. |
| `volume_inspect_segmentation` | `job_id` | Show nnU-Net probability/mask previews and manifest. |
| `surface_inspect_registered_render` | `job_id` | Show registered intensity, coverage, and surface-depth previews. |

Candidate search `region` has `center` and `radius` XYZ integer points. Radius is
at most 192 on each axis; at most eight surface chunks and eight matching ink
chunks are read; at most 100 candidates are returned. Ink predictions must have
the same shape and chunk layout as the surface prediction.

```json
{
  "prediction_uri": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0332/representations/predictions/surfaces/20251211183505-surface-20260413222639-surface-m7-L2-th0.2.zarr/",
  "prediction_space": "ct_l2_xyz",
  "region": {
    "center": {"x": 2007, "y": 2009, "z": 2015},
    "radius": {"x": 64, "y": 64, "z": 64}
  },
  "max_candidates": 8,
  "minimum_separation_voxels": 16
}
```

Optional candidate parameters are `ink_prediction_uri`, `surface_threshold`,
`ink_threshold`, and `ink_weight` (0–1). Candidate selection is read-only and
deterministic.

### CPU discovery (always available)

These tools accept absolute existing local artifact paths and run asynchronously.
Their outputs are evidence for exploration, not calibrated ink probabilities.

| Tool | Required input | Important options / output |
| --- | --- | --- |
| `vc_render_surface_diagnostics` | `surface_volume_path`, `client_request_id` | Depth projections, deviation, gradient, depth-of-maximum, and persistence maps from TIFF/PNG stacks. |
| `ink_compute_classical_features` | `diagnostics_path`, `client_request_id` | CLAHE, morphology, DoG, LoG, Gabor, and heuristic candidate score. |
| `ink_find_candidate_regions` | `score_path`, `client_request_id` | Deterministic connected components; optional `threshold`, `min_area`, `max_candidates`. |
| `ink_render_candidate_report` | `candidate_set_path`, `client_request_id` | Context crops; optional `max_candidates` and `context_pixels`. |
| `text_analyze_layout` | `mask_path`, `client_request_id` | Components, skeleton, and line-layout evidence; optional threshold. No OCR. |

Typical path-based chain:

```text
vc_render_surface_diagnostics
  -> ink_compute_classical_features
  -> ink_find_candidate_regions
  -> ink_render_candidate_report / text_analyze_layout
```

Each result job reports the local output path needed by the next tool.

### Bounded Dataset 058 segmentation (optional)

`volume_run_segmentation` is enabled by the nnU-Net variables. It accepts either:

- `volume_path`: an absolute local NPY/TIFF/PNG file; or
- `source` plus `region`: a bounded local/remote Zarr ROI.

Required model fields are:

```json
{
  "model": "vc-surface-nnunet-058",
  "device": "cpu",
  "checkpoint_sha256": "8b90543a3b8063d1158467364fcf825527fb18edc3af852ffcb91906f0e3e763",
  "volume_path": "/absolute/path/to/input.npy",
  "client_request_id": "segment-local-1"
}
```

`device` is `cpu` or `mps`; options include `tile_size` (64/96/128), `overlap`
(0–0.75), `threshold` (0–1), and `cpu_threads` (1–16).

For Zarr, `source.kind` is `local_zarr` or `remote_zarr` and uses exactly one of
`path`/`uri`, with optional `array_path`, `scale`, `voxel_spacing` XYZ,
`voxel_spacing_unit: "um"`, and `origin_xyz`. `region` supplies XYZ origin,
width/height/depth (each at most 256), and coordinate space. Staging is limited
to 16M voxels, 64 touched chunks, and 64 MiB per uncompressed chunk. Inputs must
be rank-3 uint8/uint16. Public Vesuvius S3 and `dl.ash2txt.org` are allowlisted.

The stager strips object-store credentials before invoking the model and gives
the model only a local NPY. Manifests preserve XYZ region, ZYX slices, scale,
array path, spacing, and output origin. A `spatial-metadata.json` sidecar applies
that metadata to probability and mask NPY/TIFF outputs.

### Registered surface construction and geometry (optional)

These are enabled by the analysis Python, stager, and surface-bundle adapter.

| Tool | Required input | Purpose / options |
| --- | --- | --- |
| `surface_render_registered_roi` | `surface` (`surface` artifact), `volume`, `coordinate_space`, `client_request_id` | Import TIFXYZ, stage bounded CT, and register raw intensity. Optional `uv_region`, `normal_padding_voxels` (0–64). |
| `surface_validate_geometry` | `surface` (`registered-surface`), `client_request_id` | Local stretch, normal discontinuity, folds/degeneracy, components, holes, and boundaries. |
| `surface_measure_volume_alignment` | registered `surface`, `client_request_id` | Trilinear profiles along normals; optional `maximum_offset_voxels` (1–16). |
| `surface_render_normal_stack` | registered `surface`, `model_profile`, `client_request_id` | Render model-compatible uint8 H×W×C Zarr/TIFF; optional `reverse_layers`, `layer_step_voxels`. |

Example registration after a successful `vc_generate_surface` job:

```json
{
  "surface": {"job_id": "growth-job", "artifact_id": "surface"},
  "volume": {
    "kind": "remote_zarr",
    "uri": "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/example.zarr/",
    "array_path": "0",
    "scale": 0,
    "voxel_spacing": [3.24, 3.24, 3.24],
    "voxel_spacing_unit": "um",
    "origin_xyz": [0, 0, 0]
  },
  "coordinate_space": "ct_l0_xyz",
  "normal_padding_voxels": 32,
  "client_request_id": "register-growth-job-1"
}
```

The canonical `vc_surface_bundle` Zarr stores `geometry/xyz [v,u,xyz]`,
`geometry/valid [v,u]`, and registered raw intensity under `renders/raw`, with
NPY/TIFF/PNG previews, coordinate metadata, checksums, and a manifest. Registered
rendering uses deterministic nearest-neighbor sampling.

Geometry validation does not perform a global nonlocal triangle self-intersection
test. Alignment is limited to 1,048,576 UV pixels and cannot request more normal
offset than registration staged.

Normal-stack profiles are:

- `timesformer-26`: source layers `[17,43)`, offsets `[-15,+10]`;
- `resnet152-3d-decoder-62`: source layers `[1,63)`, offsets `[-31,+30]`.

Both follow a canonical 65-layer `[-32,+32]` source convention. Register with 32
voxels of normal padding for the 62-layer profile. Height and width must each
exceed the channel count so the optimized loader cannot ambiguously infer HWC versus CHW.

### Structural evidence (optional)

Enabled by `VC_MCP_STRUCTURAL_EVIDENCE_ADAPTER`.

| Tool | Required artifacts | Purpose / key options |
| --- | --- | --- |
| `text_measure_grid_coherence` | registered `surface` | Physical letter/line/column periodicity, cycle gating, windowed score, deterministic row-permutation/circular-shift null. |
| `ink_compare_registered_predictions` | `surface_a`, `surface_b` | Agreement, divergence, pixel difference, and review priority for identically registered surfaces. |
| `text_epoch_fold_structure` | registered `surface`, `grid` (`grid-coherence`) | Period search and 1D horizontal epoch fold with look-elsewhere-corrected permutation p-value. |

Grid options include `polarity` (`bright`/`dark`), physical periods and window
sizes in mm, `minimum_cycles` (2–10), `null_trials` (4–64), and `null_seed`.
Epoch-fold options include `period_tolerance`, `period_steps`, `phase_bins`, and
the null controls. Explicit physical voxel spacing in micrometres is required.
These tools detect structure; they cannot establish ink truth or transcribe text.

### Robustness, model fusion, and ranking (optional)

Enabled by `VC_MCP_EVIDENCE_FUSION_ADAPTER`.

| Tool | Required artifacts | Purpose / options |
| --- | --- | --- |
| `surface_test_stability` | `baseline` plus 1–7 registered `variants` | Validity IoU, physical displacement, sign-invariant normal angle, signal difference, and worst local stability. Optional explicit scales. |
| `ink_fuse_registered_scores` | `ink_model` (`ink-prediction`), `dinovol` (`dinovol-exemplar`) | Preserve raw/bounded components and calculate an uncalibrated review priority; optional `stability` and weights. |
| `surface_rank_evidence` | 1–16 candidate records | Weighted geometric mean of geometry, alignment, grid, and optional stability; optional explicit weights. |

A ranking candidate has a safe identifier plus `surface-geometry`,
`surface-ct-alignment`, and `grid-coherence` references; `surface-stability` is
optional. Every bounded component, weight, and formula is retained in JSON/Zarr,
CSV, NPY/TIFF/PNG outputs. Stability only evaluates the perturbations supplied;
it does not claim untested coverage. DinoVol normalization is ROI-relative.

### Learned evidence tools (optional)

| Tool | Required input | Purpose |
| --- | --- | --- |
| `dinov3_exemplar_search` | local image, pinned repository/commit/weights/hash, positive 2D points, request ID | CPU dense-feature reranking in an optional bbox; supports negative points, top-k, threads, ViT-S/16 or ViT-S+/16. |
| `dinovol_exemplar_search` | registered `surface`, `device: "mps"`, positive `(u,v,offset)` examples, request ID | 3D teacher-backbone exemplar similarity projected to UV; optional negative examples. |
| `ink_run_resnet152_inference` | `surface_volume`, `model_profile: "resnet152-3d-decoder-62"`, request ID | Produce an ink-model score using the pinned ResNet152/3D-decoder checkpoint; optional device, tile size, stride, reverse layers. |

`ink_run_resnet152_inference` only consumes a validated `surface-volume` artifact
from `surface_render_normal_stack`; it cannot consume an arbitrary path. Devices
are `cpu`, `mps`, or `cuda`, and tile size is 64/128/256. Learned outputs remain
uncalibrated evidence.

### Review and metrics (optional)

Enabled by `VC_MCP_REVIEW_ADAPTER`.

| Tool | Required artifacts | Purpose / options |
| --- | --- | --- |
| `review_create_queue` | `ranking` (`evidence-ranking`) | Immutable digest-addressed queue; optional structural `comparison`, `max_items` (1–100), divergence percentile. |
| `review_record_assessment` | `queue`, `reviewer_id`, 1–100 assessments | New immutable assessment; decisions are accept/reject/uncertain/defer with optional confidence, reason codes, and notes. |
| `metric_evaluate_labels` | 1–16 `review-assessment` references | Coverage, ROC AUC, average precision, five calibration bins, pairwise agreement, and Cohen kappa. |

All assessments must refer to the same queue. Reviewer decisions are supplied
labels, not objective ground truth. Recording an assessment never mutates the
queue.

### Evidence and review inspectors (always registered)

Each inspector takes `{"job_id":"..."}` for a succeeded job of the matching
kind and renders the manifest plus up to three previews:

- `surface_inspect_geometry`
- `surface_inspect_volume_alignment`
- `text_inspect_grid_coherence`
- `ink_inspect_registered_comparison`
- `text_inspect_epoch_fold`
- `surface_inspect_stability`
- `ink_inspect_registered_fusion`
- `surface_inspect_evidence_ranking`
- `review_inspect_queue`
- `review_inspect_assessment`
- `metric_inspect_label_evaluation`

These inspector names remain visible even when the corresponding generating
adapter is disabled; without a matching succeeded job they return an error.

## Resources and MCP App

| Resource | MIME type | Contents |
| --- | --- | --- |
| `vc://server/capabilities` | `application/json` | Build and enabled-operation snapshot. |
| `vc://jobs/{job_id}` | `application/json` | Complete current process-local job state. |
| `vc://jobs/{job_id}/logs` | `application/json` | At most the last 500 redacted log lines. |
| `ui://vc/inspector.html` | `text/html;profile=mcp-app` | Embedded Vellum Lens MCP App. |

The server advertises the MCP Apps `io.modelcontextprotocol/ui` extension.
Vellum Lens is attached to prediction, candidate, generation, segmentation,
registered-surface, evidence, and review results. Clients without MCP Apps still
receive text and `structuredContent`.

The checked-in single-file bundle is generated from `ui/src/`. After UI changes:

```bash
cd apps/VC3D/mcp/ui
npm ci
npm run build
```

Do not run `npm ci` merely for discovery; installation requires explicit
approval under this repository's agent policy.

## Recommended review-gated workflow

1. Call `vc_inspect_prediction`.
2. Call `vc_find_seed_candidates` with a small explicit region.
3. Present candidate coordinates, scores, bounds, and limitations for review.
4. After seed approval, call `vc_generate_surface` with a short generation limit.
5. Poll `vc_get_job`; inspect `vc://jobs/{id}/logs` if it fails.
6. On success, call `vc_render_surface_preview` and `vc_inspect_artifacts`.
7. Stop and review the geometry summary, previews, manifest, and hashes.
8. Only after explicit approval, proceed to registered rendering, normal stacks,
   learned inference, fusion/ranking, or review publication.

Do not silently choose a seed when review is possible, and do not treat a
successful process as scientific validation.

## Build and verification

Build the dependency-free MCP core tests:

```bash
cmake --build <build-directory> --target \
  vc_mcp_job_store_test \
  vc_mcp_local_worker_test \
  vc_mcp_cpu_discovery_test \
  vc_mcp_protocol_test \
  vc_mcp_http_transport_test

ctest --test-dir <build-directory> \
  -R '^(vc_mcp_job_store|vc_mcp_local_worker|vc_mcp_cpu_discovery|vc_mcp_protocol|vc_mcp_http_transport)$' \
  --output-on-failure
```

After all registered test executables are built:

```bash
ctest --test-dir <build-directory> -R '^vc_mcp_' --output-on-failure
```

When `VC_MCP_NNUNET_TEST_PYTHON` points to an existing interpreter, CMake also
registers tests for the stager, surface bundle, structural evidence, evidence
fusion, review adapter, registered-surface MCP chain, deterministic end-to-end
fixture, and adversarial boundaries. A real nnU-Net integration test additionally
requires `VC_MCP_NNUNET_TEST_MODEL` containing `fold_0/checkpoint_best.pth`.
Do not report optional tests as run when they were absent or skipped.

Finally verify with MCP Inspector or another real MCP client:

- initialize and list tools/resources;
- call `vc_capabilities`;
- read each resource type;
- submit and poll a small bounded test job;
- test cancellation;
- confirm Vellum Lens or structured fallback output.

Do not use a production-sized volume as a smoke test.

## Agent workflow and safety

The review-gated operating procedure is in
`.agents/skills/vc-generate-surface/SKILL.md`. Agents must diagnose first and may
not install packages, run bootstrap scripts, or download repositories/models
without explicit approval. In agent mode, installations additionally require:

```bash
AGENTS_AGENT_MODE=1 AGENTS_ALLOW_INSTALL=1 <approved command>
```

Ubuntu dependency source of truth: `scripts/install_build_deps.sh`.
macOS dependency/build source of truth: `scripts/build_macos.sh`.
Review either script before allowing its install mode.

Never expose `VC_MCP_AUTH_TOKEN`, credentials, private object-store URLs, or
unredacted logs. Never bind this experimental HTTP server to a LAN or public
interface.

## Troubleshooting

- **Optional tools are missing:** call `vc_capabilities`, verify every required
  interpreter/adapter/repository/checkpoint path, restart, then list tools again.
- **Growth executable is missing:** build `vc_grow_seg_from_seed` or set
  `VC_MCP_GROW_EXECUTABLE` to its absolute path.
- **Stdio exits immediately:** the client likely closed stdin, sent non-JSON to
  stdin, or launched a relative command from the wrong working directory.
- **HTTP returns unauthorized:** send a matching bearer token; it must contain at
  least 32 characters.
- **HTTP bind is rejected:** use only `127.0.0.1`, `::1`, or `localhost`.
- **A job vanished:** the server restarted; process-local state is gone even if
  artifact files remain.
- **A chained artifact is rejected:** use the upstream job and exact artifact ID
  required by `tools/list`, not a filesystem path.
- **Coordinate mismatch:** stop and explicitly resolve L0/L2 and XYZ/ZYX metadata.
- **Remote URI is rejected:** use an allowlisted bounded Vesuvius source; do not
  bypass validation.
- **Configure fails offline:** FastMCPP FetchContent is not cached; obtain approval
  before retrying with network access.
- **Pi shows no MCP tools:** Pi does not include a built-in MCP client. Use a
  reviewed MCP extension/package or connect through an external MCP-capable
  client; shell commands are not registered MCP tools.

## Source layout

- `main.cpp`: environment, worker configuration, and transports.
- `McpApplication.cpp`: core tools/resources and MCP handler.
- `JobStore.*`: async lifecycle, idempotency, artifact references, and logs.
- `VolumeCartographer.*`, `GrowRequest.cpp`: fixed-profile VC growth.
- `PredictionService.*`, `InspectionTools.*`: Zarr inspection, candidates,
  previews, artifact inspectors, and Vellum Lens resource.
- `CpuDiscovery.*`, `DiscoveryTools.*`: bounded CPU operations and optional
  adapter registration.
- `*_adapter.py`, `volume_stager.py`: isolated deterministic Python adapters.
- `ui/`: Vellum Lens source and embedded production bundle.
- `tests/`: C++ protocol/worker/transport tests and Python adapter/pipeline tests.

The server is versioned as experimental (`volume-cartographer` MCP server
`0.1.0`). Live `tools/list` schemas and generated manifests take precedence over
examples in this document.
