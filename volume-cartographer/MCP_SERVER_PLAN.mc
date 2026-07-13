# Plan: Volume Cartographer MCP Server

## 1. Scope

Build a remote MCP server that:

- speaks MCP over **Streamable HTTP**;
- advertises Volume Cartographer operations through `tools/list`;
- accepts operations through `tools/call`;
- exposes job state, logs, manifests, and previews as MCP resources;
- sends resource-update notifications when supported by the client;
- invokes a pinned, headless Volume Cartographer container;
- supports asynchronous work without keeping `tools/call` requests open;
- never exposes a shell or arbitrary command execution.

There does **not** need to be a separate public REST job API. The public interface is MCP:

```text
MCP client
    │ initialize
    │ tools/list
    │ tools/call
    │ resources/read
    │ resources/subscribe
    ▼
Volume Cartographer MCP server
    │
    ├── job database
    ├── queue
    ├── VC worker
    └── artifact storage
```

Only `/mcp` and operational endpoints such as `/healthz` are exposed.

## 2. MCP protocol surface

### Transport

Implement:

```text
POST   /mcp
GET    /mcp      # server stream, if required by negotiated transport
DELETE /mcp      # session termination, if sessions are enabled
```

Use the official MCP SDK rather than implementing JSON-RPC manually.

The server must handle the normal MCP lifecycle:

1. `initialize`
2. protocol-version and capability negotiation
3. `notifications/initialized`
4. `tools/list`
5. `tools/call`
6. `resources/list`
7. `resources/templates/list`
8. `resources/read`
9. optionally `resources/subscribe`
10. optionally `notifications/resources/updated`

Advertised capabilities:

```json
{
  "tools": {
    "listChanged": false
  },
  "resources": {
    "subscribe": true,
    "listChanged": false
  },
  "logging": {}
}
```

Do not advertise prompts, sampling, or elicitation unless they are actually implemented.

## 3. MCP tools

Use a small set of explicit tools. Each tool gets:

- an `inputSchema`;
- an `outputSchema`;
- structured output through `structuredContent`;
- tool annotations;
- a concise text fallback for clients that do not consume structured output.

### Discovery

#### `vc_capabilities`

Returns the installed VC build and supported operations.

```json
{
  "vc_version": "…",
  "vc_commit": "…",
  "container_digest": "sha256:…",
  "operations": [
    "grow_surface",
    "surface_metrics",
    "flatten_surface",
    "render_surface_volume"
  ],
  "coordinate_spaces": [
    "ct_l0_xyz",
    "ct_l2_xyz"
  ],
  "profiles": [
    "scroll3-conservative-v1"
  ]
}
```

Annotations:

```json
{
  "readOnlyHint": true,
  "idempotentHint": true,
  "openWorldHint": false
}
```

### Cost estimation

#### `vc_estimate_operation`

Validates inputs and estimates resources before submission.

Input:

```json
{
  "operation": "grow_surface",
  "input": {
    "volume_uri": "s3://allowed-bucket/PHerc0332/volume.zarr",
    "prediction_uri": "s3://allowed-bucket/PHerc0332/prediction.zarr",
    "seed": {
      "x": 2112,
      "y": 2304,
      "z": 4256,
      "space": "ct_l2_xyz"
    },
    "profile": "scroll3-conservative-v1"
  }
}
```

Output:

```json
{
  "estimate_id": "est_01...",
  "expires_at": "2026-07-12T20:00:00Z",
  "validated": true,
  "estimated_runtime_seconds": 5400,
  "cpu": 32,
  "memory_gib": 128,
  "scratch_gib": 500,
  "estimated_cost_usd": 9.4,
  "normalized_input": {}
}
```

This supplies the approval boundary. Expensive submission tools require a valid `estimate_id`; they should not depend on MCP elicitation support.

### VC operations

Implement separate tools rather than one arbitrary `vc_run` tool:

```text
vc_grow_surface
vc_compute_surface_metrics
vc_flatten_surface
vc_render_surface_volume
```

Each call validates the request, creates a job, and returns immediately.

#### Example: `vc_grow_surface`

Input schema, conceptually:

```json
{
  "estimate_id": "est_01...",
  "volume_uri": "s3://allowed-bucket/PHerc0332/volume.zarr",
  "prediction_uri": "s3://allowed-bucket/PHerc0332/prediction.zarr",
  "seed": {
    "x": 2112,
    "y": 2304,
    "z": 4256,
    "space": "ct_l2_xyz"
  },
  "profile": "scroll3-conservative-v1",
  "limits": {
    "max_generations": 256
  },
  "client_request_id": "scroll3-candidate-49-grow-1"
}
```

Output:

```json
{
  "job_id": "job_01...",
  "state": "queued",
  "operation": "grow_surface",
  "job_resource": "vc://jobs/job_01...",
  "log_resource": "vc://jobs/job_01.../logs",
  "submitted_at": "2026-07-12T19:20:00Z"
}
```

`client_request_id` provides idempotency. Repeating the same request returns the original job rather than launching duplicate compute.

Tool annotations:

```json
{
  "readOnlyHint": false,
  "destructiveHint": false,
  "idempotentHint": true,
  "openWorldHint": false
}
```

### CPU ink and text discovery

Add the following CPU-only tools after surface-volume rendering is available. They are
candidate-discovery and review tools, not proof that a feature is ink or a reliable
transcription. Every output must retain the source surface-volume artifact, coordinate
space, render offsets, parameter profile, and implementation version.

All tools that scan images or volumes are asynchronous jobs and use the same estimate,
idempotency, cancellation, artifact, and resource conventions as VC geometry jobs.
Small bounded metadata reads may remain synchronous. CPU workers must not silently use
CUDA, MPS, or another accelerator; record the selected backend as `cpu` in the command
manifest.

The selected initial tool set is:

```text
vc_render_surface_diagnostics
ink_compute_classical_features
ink_find_candidate_regions
ink_render_candidate_report
dinov3_exemplar_search        # preferred first learned reranker
dinovol_exemplar_search       # deferred optional 3D comparison
text_analyze_layout
```

#### `vc_render_surface_diagnostics`

Create a standardized diagnostic bundle from an existing registered surface volume, or
render the surface volume first from an approved CT volume and TIFXYZ artifact. This is
the only selected ink-discovery tool allowed to read the original CT volume directly.

Input, conceptually:

```json
{
  "estimate_id": "est_01...",
  "surface_artifact": "vc://jobs/job_01.../artifacts/surface-tifxyz",
  "volume_uri": "s3://allowed-bucket/PHerc0332/volume.zarr",
  "coordinate_space": "ct_l0_xyz",
  "offsets": {"start": -32, "end": 32, "step": 1},
  "render_scale": 1,
  "normal_direction": "both",
  "diagnostics": [
    "raw_stack",
    "mean",
    "median",
    "min",
    "max",
    "deviation_from_mean",
    "normal_gradient",
    "depth_of_max",
    "persistence"
  ],
  "profile": "cpu-diagnostics-v1",
  "client_request_id": "scroll3-candidate-49-diagnostics-1"
}
```

Requirements:

- require a symmetric, bounded normal-offset range unless a named profile explicitly
  says otherwise;
- permit `normal_direction: both` when surface orientation is uncertain;
- preserve raw values in the full artifact and apply display normalization only to
  previews;
- distinguish the surface-grid axes `(u, v, offset)` from CT `(x, y, z)` and NumPy
  `(z, y, x)`;
- produce validity masks so lacunae and out-of-volume samples cannot become candidates;
- reuse `vc_render_tifxyz` for sampling rather than implementing a second coordinate
  mapper;
- expose the existing VC compositors where applicable, including `devFromMean`,
  `gradientMag`, `gammaWeighted`, and `maxAboveIso`; and
- cap preview dimensions and raw-stack size independently.

Artifacts:

```text
surface-volume.zarr
validity-mask.zarr
diagnostics.zarr
mean.tif
median.tif
min.tif
max.tif
deviation-from-mean.tif
normal-gradient.tif
depth-of-max.tif
persistence.tif
depth-montage.jpg
diagnostic-contact-sheet.jpg
manifest.json
```

The manifest records exact offsets, normal orientation, interpolation mode, source and
output hashes, intensity dtype/range, and the VC executable/container identity.

#### `ink_compute_classical_features`

Compute deterministic, training-free CPU feature maps from a diagnostic surface volume.
The output is evidence for candidate generation, not a calibrated ink probability.

Input, conceptually:

```json
{
  "estimate_id": "est_01...",
  "diagnostics_artifact": "vc://jobs/job_02.../artifacts/diagnostics",
  "profile": "classical-ink-v1",
  "features": [
    "robust_depth_deviation",
    "depth_persistence",
    "local_robust_zscore",
    "difference_of_gaussians",
    "laplacian_of_gaussian",
    "multiscale_hessian_ridge",
    "oriented_gabor",
    "local_entropy",
    "black_hat",
    "top_hat"
  ],
  "client_request_id": "scroll3-candidate-49-features-1"
}
```

The `classical-ink-v1` profile pins all scales, kernels, border behavior, normalization,
and combination weights. Arbitrary kernels or Python expressions are forbidden. Reuse
CPU implementations from OpenCV, SciPy/scikit-image, and Villa where appropriate,
including:

```text
vesuvius/image_proc/features/curvature.py
vesuvius/image_proc/features/ridges_vessels.py
vesuvius/image_proc/features/skeletonization.py
```

Artifacts include each uncombined feature map, a validity mask, a heuristic combined
candidate score, per-feature display previews, and a manifest. Never label the combined
score `ink_probability` unless it has been calibrated and validated as such.

#### `ink_find_candidate_regions`

Convert one or more feature maps into a bounded, ranked set of review candidates.

Input, conceptually:

```json
{
  "estimate_id": "est_01...",
  "features_artifact": "vc://jobs/job_03.../artifacts/classical-features",
  "profile": "candidate-regions-v1",
  "max_candidates": 500,
  "minimum_separation_pixels": 16,
  "client_request_id": "scroll3-candidate-49-regions-1"
}
```

The pinned profile may perform robust normalization, fixed threshold sweeps, bounded
morphological closing, connected-component analysis, and skeletonization. Rank using
separately reported evidence such as area, elongation, skeleton length, stroke-width
consistency, branch count, continuity, depth persistence, and cross-feature agreement.
Do not hide these terms behind only one opaque score.

Output candidate coordinates in both surface UV pixels and, where the mapping is valid,
CT XYZ bounds. Candidates intersecting invalid surface pixels must be clipped or rejected.
The result includes a machine-readable `candidate-set.json`, score maps, threshold-sweep
previews, and candidate overlays. Candidate ordering must be deterministic.

#### `ink_render_candidate_report`

Render review artifacts for a candidate set. The report is evidence presentation and
must not modify candidate scores.

Input, conceptually:

```json
{
  "estimate_id": "est_01...",
  "candidate_set_artifact": "vc://jobs/job_04.../artifacts/candidate-set",
  "diagnostics_artifact": "vc://jobs/job_02.../artifacts/diagnostics",
  "candidate_ids": ["candidate-0001", "candidate-0002"],
  "context_scales": [1, 2, 4],
  "offsets": [-8, -4, 0, 4, 8],
  "profile": "candidate-report-v1",
  "client_request_id": "scroll3-candidate-49-report-1"
}
```

Produce per-candidate contact sheets containing raw offset slices, mean/median/max,
local-contrast and persistence views, feature overlays, coordinates, orientation, and at
least two context scales. Do not burn feature overlays into the only copy of a raw view.
Generate an HTML/JSON index and small JPEG/PNG previews; put the full-resolution report
bundle in artifact storage.

#### `dinov3_exemplar_search`

Run bounded, CPU-only dense exemplar retrieval over one 2D diagnostic representation.
This is the preferred first learned reranker. Start with the official web-pretrained
DINOv3 ViT-S/16 checkpoint; ViT-S+/16 is the only initial alternative. Require a pinned
local DINOv3 repository commit and checkpoint hash, a crop no larger than 2048×2048 or
4,194,304 pixels, and positive examples with optional fiber/crack negatives. Produce
positive, negative, and positive-minus-negative cosine-similarity maps plus top matches.
Call the result `exemplar_similarity`, never `ink_probability`. Advertise the tool only
when its CPU adapter is configured.

#### `dinovol_exemplar_search`

Run bounded, CPU-only exemplar retrieval with a pinned Dinovol checkpoint. This is a
deferred 3D comparison, optional at deployment time, and must only be advertised when a
validated checkpoint and runtime are installed.

Input, conceptually:

```json
{
  "estimate_id": "est_01...",
  "surface_volume_artifact": "vc://jobs/job_02.../artifacts/surface-volume",
  "checkpoint": "dinovol-scroll-v2-pinned",
  "search_bbox": {
    "u": 0, "v": 0, "offset": 0,
    "width": 2048, "height": 2048, "depth": 64
  },
  "positive_examples": [
    {"u": 812, "v": 934, "offset": 31}
  ],
  "negative_examples": [
    {"u": 620, "v": 730, "offset": 27}
  ],
  "profile": "dinovol-exemplar-cpu-v1",
  "client_request_id": "scroll3-candidate-49-dinovol-1"
}
```

Requirements:

- extract a headless library/CLI from Dinovol's Napari visualizer logic; never automate
  the GUI;
- pin the Dinovol commit, checkpoint hash, checkpoint configuration, normalization,
  patch size, crop/window size, and overlap;
- require a bounded search box and enforce voxel, patch, memory, and wall-time limits;
- force the Torch CPU device and record thread counts;
- compute normalized patch embeddings once per artifact/profile and cache them by input
  and checkpoint hash;
- score deterministically as positive-example cosine similarity minus negative-example
  cosine similarity, retaining the individual similarity maps; and
- describe output as `exemplar_similarity`, never `ink_probability`.

Artifacts include positive, negative, and combined similarity volumes/maps; top matches;
PCA previews when requested; the exact exemplar coordinates; checkpoint provenance; and
a manifest. A deployment without the pinned runtime returns this omission through
`ink_capabilities` rather than registering a tool that always fails.

#### `text_analyze_layout`

Analyze whether a candidate mask has text-like organization using deterministic CPU
geometry. This is not OCR and does not return a transcription.

Input, conceptually:

```json
{
  "estimate_id": "est_01...",
  "candidate_set_artifact": "vc://jobs/job_04.../artifacts/candidate-set",
  "candidate_ids": ["candidate-0001"],
  "mask_source": "combined_candidate_score",
  "profile": "text-layout-v1",
  "client_request_id": "scroll3-candidate-49-layout-1"
}
```

The pinned profile may use threshold sweeps, connected components, 2D skeletonization,
stroke-width estimates, dominant orientation, baseline/line grouping, component spacing,
repeated component scale, and branch/junction statistics. It should explicitly penalize
very long parallel structures likely to be papyrus fibers. Reuse Villa's skeleton and
component primitives where possible.

Return per-candidate evidence fields rather than only a textness score, including line
hypotheses and component bounding boxes. Artifacts include masks, skeletons, component
and baseline overlays, `layout-analysis.json`, and a review contact sheet. Use terms such
as `text_like_score` and `line_hypothesis`; never claim that text was read.

All six tools use these annotations:

```json
{
  "readOnlyHint": false,
  "destructiveHint": false,
  "idempotentHint": true,
  "openWorldHint": false
}
```

They create derived artifacts but do not mutate their inputs. Require an unexpired
`estimate_id` for full-surface work; deployments may permit a named, tightly bounded
preview profile without an estimate.

### Job control

```text
vc_get_job
vc_list_jobs
vc_cancel_job
vc_retry_job
```

`vc_get_job` is useful for clients that do not support resources. It returns the same representation as `vc://jobs/{job_id}`.

`vc_cancel_job` must only cancel jobs owned by the authenticated principal.

### Artifact access

Prefer MCP resources for metadata and small previews, but provide one tool for large objects.

#### `vc_get_artifact_url`

Input:

```json
{
  "job_id": "job_01...",
  "artifact_id": "surface-volume",
  "expires_in_seconds": 900
}
```

Output:

```json
{
  "url": "https://storage.example/...",
  "expires_at": "2026-07-12T19:45:00Z",
  "media_type": "application/vnd+zarr",
  "size_bytes": 18439288320,
  "sha256": "..."
}
```

Do not return large TIFF, Zarr, OBJ, or TIFXYZ data inside MCP content blocks.

## 4. MCP resources

Register resource templates:

```text
vc://jobs/{job_id}
vc://jobs/{job_id}/logs
vc://jobs/{job_id}/manifest
vc://jobs/{job_id}/artifacts
vc://jobs/{job_id}/previews/{preview_name}
vc://profiles/{profile_name}
vc://server/capabilities
```

### Job resource

Reading `vc://jobs/job_01...` returns:

```json
{
  "job_id": "job_01...",
  "operation": "grow_surface",
  "state": "running",
  "progress": {
    "completed": 91,
    "total": 256,
    "unit": "generations",
    "message": "Growing surface"
  },
  "created_at": "...",
  "started_at": "...",
  "finished_at": null,
  "job_revision": 14
}
```

Use a monotonically increasing `job_revision`.

### Subscription behavior

If a client calls `resources/subscribe` for a job URI, emit `notifications/resources/updated` when the job changes state or crosses a meaningful progress threshold.

Do not send an update for every output line. Suggested thresholds:

- state transitions;
- every 5% progress;
- new preview available;
- terminal success, failure, or cancellation.

Clients without resource subscriptions can use `vc_get_job`.

### Log resource

`vc://jobs/{id}/logs` returns bounded log output:

- last 500 lines by default;
- redacted environment and credentials;
- a cursor for subsequent reads;
- a signed URL for the complete compressed log if necessary.

### Preview resources

Small PNG/JPEG diagnostics may be returned directly as MCP blob resources. Examples:

```text
normal-preview.png
intersection-preview.png
flattening-distortion.png
surface-depth-montage.jpg
ink-probability-preview.png
```

Set the correct MIME type and impose a small maximum size, such as 5 MB.

## 5. Job state machine

Use a defined state machine:

```text
validating
    ├── rejected
    └── queued
          ├── cancelled
          └── starting
                ├── failed
                └── running
                      ├── cancelling → cancelled
                      ├── failed
                      └── succeeded
```

Pipeline stages should be separate jobs connected by parent/output references:

```text
grow job
  → metrics job
  → flatten job
  → render job
  → surface diagnostics job
  → classical features job
  → candidate regions job
       ├── candidate report job
       ├── optional Dinovol exemplar-search job
       └── text-layout analysis job
```

Do not hide all stages in one opaque job initially. Separate jobs provide:

- explicit agent decisions;
- review gates;
- independent retries;
- clearer provenance;
- lower risk of automatically processing a bad surface.

A later `vc_create_workflow` tool can compose these stages after individual operations are reliable.

## 6. Headless Volume Cartographer adapter

The MCP server must not construct command lines from arbitrary user strings.

Define an internal typed adapter:

```python
class VolumeCartographer:
    async def capabilities(self) -> VCCapabilities: ...
    async def grow_surface(self, request: GrowSurfaceRequest, workdir: Path): ...
    async def surface_metrics(self, request: MetricsRequest, workdir: Path): ...
    async def flatten_surface(self, request: FlattenRequest, workdir: Path): ...
    async def render_surface_volume(self, request: RenderRequest, workdir: Path): ...
```

Each method maps typed fields to a fixed executable and fixed flags.

Example mapping:

```python
argv = [
    "/opt/volume-cartographer/bin/vc_grow_seg_from_seed",
    "--volume", mounted_volume_path,
    "--prediction", mounted_prediction_path,
    "--seed-x", str(request.seed.x),
    "--seed-y", str(request.seed.y),
    "--seed-z", str(request.seed.z),
    "--parameters", generated_profile_path,
    "--output", output_path,
]
```

The exact flags must be taken from the pinned VC build's `--help`; do not assume the names above are correct.

Before MCP implementation, run a headless compatibility spike for every executable:

1. Record `--help` and version output.
2. Verify it runs without Qt, OpenGL, X11, or an interactive terminal.
3. If a binary is GUI-bound, extract or add a CLI entry point rather than treating `xvfb` as the permanent solution.
4. Verify exit codes and signal handling.
5. Identify progress output that can be parsed.
6. Verify SIGTERM cancels cleanly.
7. Record required working-directory layout and file naming.
8. Pin the tested OCI image by digest.

## 7. Coordinate contract

Never accept an unqualified array like:

```json
"seed_xyz": [2112, 2304, 4256]
```

Use named fields and a required coordinate space:

```json
{
  "x": 2112,
  "y": 2304,
  "z": 4256,
  "space": "ct_l2_xyz"
}
```

Persist both submitted and normalized coordinates:

```json
{
  "submitted": {
    "x": 2112,
    "y": 2304,
    "z": 4256,
    "space": "ct_l2_xyz"
  },
  "vc_input": {
    "x": 8448,
    "y": 9216,
    "z": 17024,
    "space": "ct_l0_xyz"
  },
  "transform": {
    "scale": 4,
    "permutation": [0, 1, 2]
  }
}
```

For this repository, also document that NumPy reads volumes as `(z, y, x)`, while VC coordinates are `(x, y, z)`.

## 8. Server implementation layout

Because this repository is Python 3.12, use the official Python MCP SDK and put the service in a separate package area:

```text
services/vc_mcp/
├── pyproject.toml
├── Dockerfile
├── src/vc_mcp/
│   ├── server.py
│   ├── tools.py
│   ├── resources.py
│   ├── schemas.py
│   ├── auth.py
│   ├── config.py
│   ├── store.py
│   ├── queue.py
│   ├── notifications.py
│   └── vc_adapter.py
├── migrations/
└── tests/
    ├── test_initialize.py
    ├── test_tools_list.py
    ├── test_tools_call.py
    ├── test_resources.py
    ├── test_auth.py
    ├── test_idempotency.py
    └── test_cancellation.py

workers/vc/
├── Dockerfile
├── worker.py
└── profiles/
    └── scroll3-conservative-v1.json
```

Responsibilities:

- `server.py`: MCP instance and Streamable HTTP transport.
- `tools.py`: registered MCP tool handlers.
- `resources.py`: resource readers and templates.
- `schemas.py`: typed request/output models.
- `auth.py`: bearer-token validation and authorization.
- `store.py`: jobs, artifacts, estimates, and subscriptions.
- `queue.py`: internal worker dispatch.
- `notifications.py`: resource-update publication.
- `vc_adapter.py`: safe mapping to the VC CLI.

## 9. Persistence model

Minimum tables follow.

### `jobs`

```text
id
owner_id
operation
state
input_json
normalized_input_json
client_request_id
estimate_id
progress_json
error_json
container_digest
command_manifest_json
created_at
started_at
finished_at
revision
```

Unique constraint:

```text
(owner_id, client_request_id)
```

### `artifacts`

```text
id
job_id
name
kind
storage_uri
media_type
size_bytes
sha256
metadata_json
created_at
```

### `estimates`

```text
id
owner_id
operation
normalized_input_hash
estimated_resources_json
estimated_cost
expires_at
consumed_at
```

### `job_events`

Append-only audit trail:

```text
job_id
sequence
event_type
payload_json
created_at
```

The MCP server reads this state to serve resources. Workers update it through the internal queue/store interface.

## 10. Authentication and authorization

For a remote MCP server:

- require TLS;
- validate OAuth access tokens or short-lived service tokens;
- bind every job to the authenticated subject;
- validate token audience for this MCP server;
- authorize every tool call and resource read;
- reject token forwarding to storage or worker processes;
- use workload identities for S3/R2 access;
- validate `Origin` and permitted hosts;
- rate-limit by principal;
- log tool name, subject, job ID, and outcome.

Scopes could be:

```text
vc:read
vc:submit
vc:cancel
vc:admin
```

`vc_get_artifact_url` must confirm artifact ownership before signing.

## 11. URI and input validation

Allow:

```text
s3://vesuvius-challenge-open-data/...
s3://approved-private-bucket/...
https://approved-data-host/...
```

Reject:

```text
file://...
ftp://...
http://169.254.169.254/...
http://localhost/...
../../...
arbitrary presigned URLs to unknown hosts
```

For Zarr:

- cap metadata size;
- cap array rank and dimensions;
- cap chunk count touched by a job;
- set network and decompression limits;
- reject object references outside the allowed prefix;
- do not trust metadata-reported codecs blindly.

## 12. Error semantics

Distinguish MCP/tool failure from asynchronous VC failure.

### Tool-call failure

Use an MCP tool error when:

- input is invalid;
- authorization fails;
- estimate expired;
- URI is disallowed;
- job cannot be created.

### Job failure

If a job was successfully created and VC later fails, `tools/call` was successful. The job resource becomes:

```json
{
  "state": "failed",
  "error": {
    "code": "VC_PROCESS_EXITED",
    "message": "Surface growth exited with code 2",
    "retryable": false,
    "log_resource": "vc://jobs/job_01.../logs"
  }
}
```

Do not report a completed submission call itself as failed merely because the background operation later failed.

## 13. Testing

### Protocol conformance

Test with an MCP Inspector and at least one real remote client:

- initialization and capability negotiation;
- `tools/list`;
- schema validation;
- structured tool output;
- resource template discovery;
- `resources/read`;
- resource subscriptions;
- session termination;
- unsupported method behavior;
- authentication failures.

### Functional tests

Use a fake VC executable that:

- reports deterministic progress;
- writes known artifacts;
- supports cancellation;
- exits with configurable codes.

Then test the real container against small fixtures.

### Security tests

Verify:

- shell metacharacters remain inert;
- path traversal is rejected;
- unknown URI hosts are rejected;
- one user cannot read another user's job;
- duplicate `client_request_id` does not duplicate work;
- signed artifact URLs expire;
- cancellation targets only the requested container;
- secrets are absent from MCP logs and resources.

## 14. Delivery phases

### Phase 0: Headless VC verification

Deliver:

- pinned VC container;
- command inventory;
- exact CLI contracts;
- one tiny fixture per operation;
- confirmed cancellation behavior.

### Phase 1: Local MCP server

Deliver:

- stdio transport for rapid testing;
- `vc_capabilities`;
- fake `vc_grow_surface`;
- `vc_get_job`;
- job and log resources;
- MCP Inspector tests.

This proves the MCP protocol surface before adding remote deployment.

### Phase 2: Remote MCP transport

Deliver:

- Streamable HTTP `/mcp`;
- authentication;
- PostgreSQL job persistence;
- resource subscriptions;
- deployment behind TLS.

### Phase 3: Real VC worker

Deliver:

- queue-backed execution;
- per-job sandbox directories;
- fixed VC command adapter;
- artifacts uploaded to object storage;
- cancellation and timeout enforcement.

### Phase 4: Geometry pipeline

Deliver:

```text
vc_grow_surface
vc_compute_surface_metrics
vc_flatten_surface
vc_render_surface_volume
```

Each produces MCP-readable manifests and previews.

### Phase 5: CPU ink and text discovery

Add a separate namespace or server, while retaining `vc_render_surface_diagnostics` in
the VC adapter because it owns CT/TIFXYZ coordinate sampling:

```text
ink_capabilities
vc_render_surface_diagnostics
ink_compute_classical_features
ink_find_candidate_regions
ink_render_candidate_report
dinov3_exemplar_search        # preferred; advertised only with pinned runtime/checkpoint
dinovol_exemplar_search       # deferred 3D comparison
text_analyze_layout
```

Deliver in this order:

1. diagnostic stacks, projections, validity masks, and bounded previews;
2. individually inspectable classical feature maps;
3. deterministic ranked candidate regions;
4. candidate review reports;
5. text-layout evidence; and
6. bounded DINOv3 ViT-S/16 CPU exemplar retrieval and candidate reranking; and
7. Dinovol only if later depth-specific evidence justifies a 3D comparison.

Phase 5 acceptance requires a held-out labeled-fragment evaluation reporting candidate
recall at fixed review budgets (at least top 100 and top 500), false-positive area,
runtime, peak RAM, CPU model/architecture, and exact profile hashes. DINOv3 is accepted
only if it improves candidate retrieval over the classical baseline enough to justify its
runtime and memory cost. Dinovol is not operationalized until a suitable Vesuvius-domain
checkpoint, a depth-dependent failure case, and affordable bounded CPU inference are all
demonstrated.

Keeping discovery separate avoids baking model and scientific-interpretation contracts
into the VC geometry server. A client can connect to both servers, or the VC server can
later expose a deliberately composed workflow. A future calibrated ink model remains a
separate phase and must not reuse the training-free tools' heuristic scores as if they
were probabilities.

## 15. First end-to-end acceptance criterion

A remote MCP client must be able to:

1. connect to `/mcp`;
2. initialize and discover `vc_render_surface_volume`;
3. submit a trusted Paris 4 TIFXYZ render;
4. receive a job resource URI;
5. subscribe to that resource;
6. observe queued, running, and succeeded states;
7. read a small depth-montage preview through MCP;
8. request a signed URL for the full surface volume;
9. verify the manifest contains the VC image digest, arguments, input hashes, coordinate space, layer order, spacing, and output checksum.

That is the first complete MCP server milestone. Scroll 3 growth should only be enabled after this path and the underlying VC headless executable are proven.
