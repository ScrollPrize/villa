---
name: vc-generate-surface
description: Install, build, configure, connect to, and safely operate the experimental Volume Cartographer (VC3D) MCP server. Use when an agent needs to enable the VC MCP server, diagnose its requirements, configure stdio or loopback Streamable HTTP, discover VC tools/resources, grow or inspect a surface, poll asynchronous jobs, or run the review-gated surface workflow.
compatibility: macOS or Ubuntu on amd64/arm64; CMake 3.28+, Ninja, C++23, Qt 6, and the Volume Cartographer source tree. Optional Python adapters need uv/Python and model-specific assets.
metadata:
  project: volume-cartographer
  server: vc_mcp_server
  transport: loopback-streamable-http-default
---

# Volume Cartographer MCP Server

Use this skill for the experimental server in `apps/VC3D/mcp/`. It covers setup and safe operation. Treat repository discovery as read-only. **Never install packages, download models, or run bootstrap scripts unless the user explicitly approves it.** In agent mode, installs additionally require `AGENTS_AGENT_MODE=1 AGENTS_ALLOW_INSTALL=1`.

Resolve paths relative to this skill directory when invoking bundled scripts. Resolve repository paths from the repository root, not from the skill directory.

## Non-negotiable safety rules

1. Start with `doctor`; do not install by default.
2. Use loopback Streamable HTTP by default so the process-local job session persists across agent calls. Use stdio only when the client cannot use Streamable HTTP. HTTP must remain loopback-only.
3. Never expose `VC_MCP_AUTH_TOKEN`, credentials, private object-store URLs, or unredacted logs.
4. Never bind HTTP to `0.0.0.0` or a LAN address. The server rejects it by design.
5. Call `vc_capabilities` and inspect `tools/list` at the beginning of each server session. Optional tools are configuration-dependent.
6. Preserve explicit coordinate spaces. VC points are `(x,y,z)`; NumPy arrays are `(z,y,x)`. Never infer `ct_l0_xyz` versus `ct_l2_xyz`.
7. Every asynchronous submission gets a unique, stable `client_request_id`. Reuse it only to retry the same request in the same server process.
8. Poll `vc_get_job`; on failure, read `vc://jobs/{job_id}/logs`. Never assume submission means completion.
9. Keep the server process alive while using jobs: state and idempotency are process-local and disappear on restart.
10. Surface/ink scores are evidence and review priorities, not proof, calibrated probabilities, or transcription.
11. Stop after generation review unless the user explicitly approves registered rendering, normal stacks, model inference, evidence fusion, or downstream publication.

## Locate the repository and server

From a checkout, the root contains `CMakeLists.txt`, `CMakePresets.json`, and `apps/VC3D/mcp/README.md`.

```bash
# From the skill directory:
./scripts/vc-mcp-setup.sh doctor

# Or point at another checkout:
VC_REPO_ROOT=/absolute/path/to/volume-cartographer \
  ./scripts/vc-mcp-setup.sh doctor
```

The setup helper only diagnoses, configures, builds, tests, creates an optional Python analysis environment, and prints configuration. It does not install OS packages.

## Install requirements (only after explicit approval)

### Ubuntu

The repository source of truth is `scripts/install_build_deps.sh`. It runs `apt-get`, adds an apt repository, and installs AWS CLI; it must run as root and is intentionally not called by this skill automatically.

```bash
cd /absolute/path/to/volume-cartographer
sudo env AGENTS_AGENT_MODE=1 AGENTS_ALLOW_INSTALL=1 \
  bash scripts/install_build_deps.sh
```

For a manual/local session, the two `AGENTS_*` variables are unnecessary. Review the script before running it.

### macOS

The repository source of truth is `scripts/build_macos.sh`. It checks Homebrew requirements; `--install-deps` installs missing formulae and then configures/builds the project.

```bash
cd /absolute/path/to/volume-cartographer
AGENTS_AGENT_MODE=1 AGENTS_ALLOW_INSTALL=1 \
  scripts/build_macos.sh --install-deps
```

Without install permission, run `scripts/build_macos.sh` only to report missing formulae. The required Apple SDK must be available through Xcode Command Line Tools.

### Optional Python analysis/staging adapters

The core growth server is C++; Python is optional. The controlled Zarr stager and deterministic analysis adapters need the packages pinned by `apps/VC3D/mcp/requirements-staging.txt`. With explicit approval:

```bash
AGENTS_AGENT_MODE=1 AGENTS_ALLOW_INSTALL=1 \
  .agents/skills/vc-generate-surface/scripts/vc-mcp-setup.sh install-python
```

Default environment: `~/.cache/vc-mcp/venvs/analysis`. Override with `VC_MCP_PYTHON_ENV=/absolute/path`. This does **not** install nnU-Net, DINOv3, DinoVol, Villa, repositories, checkpoints, or model weights. Configure those only when the user supplies and approves pinned assets. Never silently substitute an unpinned model.

## Configure and build

Configuration fetches pinned FastMCPP and cpp-httplib sources, so it requires network access. Ask before a first configure if the dependencies are not already cached.

### Linux preset build

```bash
cd /absolute/path/to/volume-cartographer
cmake --preset dev-clang \
  -DVC_BUILD_MCP_SERVER=ON \
  -DVC_TESTING=ON
cmake --build --preset dev-clang --target \
  vc_mcp_server vc_grow_seg_from_seed
```

Expected binaries: `build/dev-clang/bin/vc_mcp_server` and `build/dev-clang/bin/vc_grow_seg_from_seed`.

### macOS Homebrew LLVM build

First let `scripts/build_macos.sh` establish the Homebrew LLVM cache. Then enable the MCP target in that cache:

```bash
cd /absolute/path/to/volume-cartographer
scripts/build_macos.sh
cmake -S . -B build-macos \
  -DVC_BUILD_MCP_SERVER=ON \
  -DVC_TESTING=ON
cmake --build build-macos --target \
  vc_mcp_server vc_grow_seg_from_seed
```

Expected binaries are under `build-macos/bin/`.

### Helper equivalent

```bash
# Linux defaults to dev-clang; set VC_MCP_PRESET to override.
./scripts/vc-mcp-setup.sh configure
./scripts/vc-mcp-setup.sh build
./scripts/vc-mcp-setup.sh test
# After reviewing and sourcing print-env:
./scripts/vc-mcp-setup.sh serve
```

On macOS, run the repository's Homebrew build helper at least once before the skill's `configure` action.

## Configure runtime environment

Print a conservative environment template and edit it before sourcing:

```bash
./scripts/vc-mcp-setup.sh print-env > /tmp/vc-mcp.env
# Review it. It contains no secret.
. /tmp/vc-mcp.env
```

Core variables:

- `VC_MCP_GROW_EXECUTABLE`: absolute path to `vc_grow_seg_from_seed`; defaults to the server's sibling binary.
- `VC_MCP_WORK_ROOT`: absolute persistent job directory; default is `~/.vc-mcp/jobs`.
- `VC_MCP_TIMEOUT_SECONDS`: `1..604800`; default `21600`.
- `VC_MCP_TRANSPORT`: set this explicitly to `streamable-http` (the skill default) or `stdio` for compatibility. Do not depend on an older binary's implicit transport fallback.
- `VC_MCP_HOST`, `VC_MCP_PORT`, `VC_MCP_AUTH_TOKEN`: Streamable HTTP settings. Token must contain at least 32 characters.

Optional adapters are enabled only when their required paths exist. Common variables are:

- Analysis/staging: `VC_MCP_ANALYSIS_PYTHON`, `VC_MCP_VOLUME_STAGER`, `VC_MCP_SURFACE_BUNDLE_ADAPTER`, `VC_MCP_STRUCTURAL_EVIDENCE_ADAPTER`, `VC_MCP_EVIDENCE_FUSION_ADAPTER`, `VC_MCP_REVIEW_ADAPTER`.
- nnU-Net: `VC_MCP_NNUNET_PYTHON`, `VC_MCP_NNUNET_ADAPTER`, `VC_MCP_NNUNET_MODEL_DIR`.
- DINO/Villa: the `VC_MCP_DINOV3_*`, `VC_MCP_DINOVOL_*`, and `VC_MCP_VILLA_*` families documented in `apps/VC3D/mcp/README.md`.

Do not advertise an optional adapter until `vc_capabilities` confirms it.

## Connect an MCP client

### Loopback Streamable HTTP (default)

Start one persistent server process for the agent session:

```bash
. /tmp/vc-mcp.env  # generated and reviewed with print-env
/absolute/path/to/bin/vc_mcp_server
```

Equivalently, set the environment explicitly:

```bash
export VC_MCP_AUTH_TOKEN="${VC_MCP_AUTH_TOKEN:-$(openssl rand -hex 32)}"
export VC_MCP_TRANSPORT=streamable-http
export VC_MCP_HOST=127.0.0.1
export VC_MCP_PORT=18080
/absolute/path/to/bin/vc_mcp_server
```

Connect to `http://127.0.0.1:18080/mcp` with `Authorization: Bearer $VC_MCP_AUTH_TOKEN`. Generic client shape (adapt field names to the client's current MCP configuration schema):

```json
{
  "url": "http://127.0.0.1:18080/mcp",
  "headers": {
    "Authorization": "Bearer ${VC_MCP_AUTH_TOKEN}"
  }
}
```

Confirm whether the client expands environment placeholders. If it does not, inject the token through that client's secret/environment facility rather than committing it to a config file. Never write the token into source control or pass it on the process command line.

### Stdio (compatibility fallback)

Use stdio only if the MCP client cannot use Streamable HTTP. Configure an **absolute** executable and explicit environment:

```json
{
  "command": "/absolute/path/to/build/bin/vc_mcp_server",
  "args": [],
  "env": {
    "VC_MCP_TRANSPORT": "stdio",
    "VC_MCP_WORK_ROOT": "/absolute/path/to/jobs",
    "VC_MCP_GROW_EXECUTABLE": "/absolute/path/to/build/bin/vc_grow_seg_from_seed"
  }
}
```

Protocol output is stdout; diagnostics are stderr. Do not wrap the command in a shell or append arbitrary user strings.

### Pi-specific note

Pi does not ship a built-in MCP client. This skill is still discoverable by Pi, but direct MCP tool registration requires a reviewed MCP extension/package or an external MCP-capable client. Do not pretend shell commands are registered MCP tools. If no MCP tools appear, explain this limitation and offer the exact stdio/HTTP configuration to the user's chosen client.

## Required session bootstrap

After connection:

1. Initialize MCP and send the initialized notification (normally handled by the client).
2. List tools and resources.
3. Call `vc_capabilities` with `{}`.
4. Confirm `execution_mode`, build commit, coordinate spaces, profile, and optional feature booleans.
5. Read `vc://server/capabilities` if a resource view is useful.
6. If expected optional tools are absent, fix environment paths and restart; tool availability is determined at process start.

Always use the schema returned by `tools/list`; source documentation can lag the running binary.

## Review-gated surface workflow

### 1. Inspect (read-only)

Call `vc_inspect_prediction`:

```json
{
  "prediction_uri": "https://allowed.example/prediction.zarr/",
  "prediction_space": "ct_l2_xyz"
}
```

Confirm shape/chunks/metadata and retain the declared coordinate space.

### 2. Find bounded candidates (read-only)

Call `vc_find_seed_candidates` with an explicit bounded region. Maximum radius is 192 and maximum returned candidates is 100.

```json
{
  "prediction_uri": "https://allowed.example/prediction.zarr/",
  "prediction_space": "ct_l2_xyz",
  "region": {
    "center": {"x": 2007, "y": 2009, "z": 2015},
    "radius": {"x": 64, "y": 64, "z": 64}
  },
  "max_candidates": 8,
  "minimum_separation_voxels": 16
}
```

Present candidate coordinates, scores, bounds, and limitations to the user. Do not silently choose a seed when review is possible.

### 3. Submit a short generation trial

After seed approval, call `vc_generate_surface` (preferred) rather than low-level `vc_grow_surface`. Begin with a small `max_generations`.

```json
{
  "prediction_uri": "https://allowed.example/prediction.zarr/",
  "prediction_space": "ct_l2_xyz",
  "voxel_size_um": 9.596,
  "seed": {"x": 2007, "y": 2009, "z": 2015, "space": "ct_l2_xyz"},
  "profile": "scroll3-conservative-v1",
  "limits": {"max_generations": 4, "min_area_cm2": 0.0},
  "client_request_id": "descriptive-stable-id-1"
}
```

Use exactly one of `prediction_uri` or absolute existing `prediction_path`. The only profile is `scroll3-conservative-v1`; it fixes `step_size=20` and `use_cuda=false`.

### 4. Poll and diagnose

Poll `vc_get_job` with `{"job_id":"..."}` at a moderate interval until `succeeded`, `failed`, or `cancelled`. States may pass through `queued`, `starting`, and `running`.

- Read `vc://jobs/{job_id}` for complete state.
- Read `vc://jobs/{job_id}/logs` for at most 500 redacted lines.
- Use `vc_cancel_job` if the user requests cancellation or limits are being exceeded.
- Report normalized input, command manifest, progress, terminal error, and artifacts. Do not expose secrets.

### 5. Inspect output and stop for approval

On success:

1. Call `vc_render_surface_preview` with the job ID.
2. Call `vc_inspect_artifacts` with the job ID.
3. Review geometry summary, generation preview, manifests, and SHA-256 inventory.
4. Clearly state that a generated surface is not automatically correct.
5. **Stop and request approval before** `surface_render_registered_roi`, normal-stack generation, Villa/DinoVol inference, evidence fusion/ranking, or review publication.

## Tool families

Tool names are dynamic; verify them with `tools/list`.

- Core: `vc_capabilities`, `vc_grow_surface`, `vc_generate_surface`, `vc_get_job`, `vc_cancel_job`.
- Inspection: `vc_inspect_prediction`, `vc_find_seed_candidates`, `vc_render_surface_preview`, `vc_inspect_artifacts`, plus artifact-specific inspectors.
- CPU discovery: `vc_render_surface_diagnostics`, `ink_compute_classical_features`, `ink_find_candidate_regions`, `ink_render_candidate_report`, `text_analyze_layout`.
- Optional segmentation/registration: `volume_run_segmentation`, `surface_render_registered_roi`, `surface_validate_geometry`, `surface_measure_volume_alignment`, `surface_render_normal_stack`.
- Optional evidence/review: grid/epoch tools, stability, fusion/ranking, queue/assessment/metrics tools.
- Optional learned models: `dinov3_exemplar_search`, `dinovol_exemplar_search`, `ink_run_villa_inference`.

For chained tools, pass immutable job artifact references exactly as required by the live schema. Never replace an artifact reference with a caller-selected filesystem path.

## Verification

Build and run the dependency-free MCP core tests:

```bash
cmake --build <build-directory> --target \
  vc_mcp_job_store_test vc_mcp_local_worker_test vc_mcp_cpu_discovery_test \
  vc_mcp_protocol_test vc_mcp_http_transport_test
ctest --test-dir <build-directory> \
  -R '^(vc_mcp_job_store|vc_mcp_local_worker|vc_mcp_cpu_discovery|vc_mcp_protocol|vc_mcp_http_transport)$' \
  --output-on-failure
```

After building all registered test executables, run the complete MCP set with `ctest --test-dir <build-directory> -R '^vc_mcp_' --output-on-failure`. Optional Python integration tests are registered only when `VC_MCP_NNUNET_TEST_PYTHON` points to an existing interpreter. A real model test also needs `VC_MCP_NNUNET_TEST_MODEL` with the expected checkpoint. Do not claim optional tests ran if they were skipped, not built, or absent.

Then use MCP Inspector or one real client to verify initialize, `tools/list`, `vc_capabilities`, resource reads, a bounded test job, polling, and cancellation. Do not use production-sized data as a smoke test.

## Troubleshooting

- **Server exits immediately under stdio:** the client likely closed stdin, emitted non-JSON text to stdin, or used a relative path with the wrong cwd.
- **Growth executable missing:** build `vc_grow_seg_from_seed` or set its absolute path.
- **Optional tools absent:** verify every configured interpreter/adapter/model path, then restart and re-run `vc_capabilities`.
- **HTTP unauthorized:** send a Bearer token matching the server environment; token length must be at least 32.
- **HTTP bind rejected:** use only `127.0.0.1`, `::1`, or `localhost`.
- **Job vanished:** the server restarted; jobs are process-local even if artifact files remain.
- **Coordinate mismatch:** stop; inspect metadata and explicitly resolve L0/L2. L0↔L2 scale is 4 with identity axis permutation, but never convert implicitly in agent reasoning.
- **Remote URI rejected:** the server intentionally allowlists bounded Vesuvius sources. Do not bypass validation.
- **Configure fails offline:** FastMCPP FetchContent was not cached. Report the network requirement and ask before retrying.

## Source of truth

Before changing setup or orchestration behavior, read:

- `apps/VC3D/mcp/README.md`
- `apps/VC3D/mcp/CMakeLists.txt`
- `apps/VC3D/mcp/main.cpp`
- `apps/VC3D/mcp/requirements-staging.txt`
- `scripts/install_build_deps.sh` (Ubuntu)
- `scripts/build_macos.sh` (macOS)

Prefer live `tools/list` schemas over examples in this skill. Keep this skill synchronized when environment variables, tool names, limits, or install scripts change.
