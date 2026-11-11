# AGENTS.md

This file defines how automated agents (Codex, Claude, chat-based coding agents, CI bots, etc.) should operate inside this monorepo.

The repo contains multiple subprojects with different languages, runtimes, and constraints. **Do not assume one “global” build/run workflow applies everywhere.** Instead:

1. **Identify the target subproject(s)** from the user prompt and/or file paths you’re asked to touch.
2. Follow the **Monorepo-wide rules** below.
3. Then apply the matching **Subproject playbook** (e.g., `volume-cartographer/`, `vesuvius/`) only if the prompt targets it.


**If you are working on a PR or issue for this repository, refer to CONTRIBUTING.md for guidelines**  

For work involving papyrus structure, surface tracing, windings, or CT interpretation,
read [SCROLL_ANATOMY.md](SCROLL_ANATOMY.md) for shared terminology and anatomical context.

---

## 1) Monorepo-wide rules

### 1.0 Portal startup policy (important)
- Treat discovery/exploration runs as **read-only** unless the user explicitly asks for environment setup.
- **Do not run installation/bootstrap commands by default** when starting work in this repo.
- Skip side-effect scripts until explicitly requested by the user:
  - `build_dependencies.sh`
  - `install_dependencies.sh`
  - `install_repositories.sh`
  - `setup_user.sh`
  - `setup_sudo.sh`
  - `npm install`, `yarn install`, `pip install`, `poetry install`, `conda env` creation, `uv sync`, Docker build/pull
- If dependencies are needed, report the exact minimal install command per target subproject and ask for confirmation.
- For agent-mode runs (Codex/CI), skip install/bootstrap side effects unless explicitly allowed:
  - Set `AGENTS_AGENT_MODE=1` for that session/run.
  - Then explicitly set `AGENTS_ALLOW_INSTALL=1` to run installs.
- For local/manual usage, no extra env var is required; run installs directly.

### 1.1 Scope first, then act
- Treat each top-level folder as an independent product unless proven otherwise.
- Make the smallest change that solves the requested task.
- If the task spans multiple subprojects, split your work into clearly separated commits/patches.
- Do not execute setup, install, or build scripts in non-target subprojects.

### 1.2 Don’t guess build systems or dependencies
Before changing code:
- Look for **subproject-local** docs and scripts:
  - `README*`, `docs/`, `scripts/`, `Makefile`, `CMakeLists.txt`, `pyproject.toml`, `requirements.txt`, `environment.yml`, `package.json`, `Dockerfile`
- Prefer **existing scripts** over inventing new commands.
- If the target subproject is not explicit, ask the user once for scope before running any install/build/discovery script.

### 1.3 Default to correctness and reproducibility
Unless the prompt explicitly says otherwise:
- Preserve behavior and outputs.
- Avoid nondeterminism (race conditions, unordered iteration affecting results, data-loader shuffles without fixed seeds, etc.).
- Avoid changes that silently relax numerical guarantees, precision, or error bounds.

### 1.4 Performance work must be measured
If the prompt is about performance:
- Establish a baseline.
- Use a profiler appropriate for the platform and language.
- Report before/after results with:
  - command line
  - dataset/input
  - build type
  - iteration counts and summary stats (mean + p50/p95 or min/median/max)

### 1.5 Portability is a hard requirement
The repo targets **Ubuntu and macOS**, across **amd64 and arm64** (where applicable).
- Avoid OS-specific code without guards.
- If adding SIMD/intrinsics, gate it and provide a safe fallback.
- Avoid toolchain-specific flags unless justified and documented.

### 1.6 Keep changes reviewable
- Prefer small, focused diffs.
- Avoid large refactors unless the prompt explicitly requests them.
- If you must refactor, do it in two steps:
  1) mechanical refactor with no behavior change
  2) functional/performance change with measurements

### 1.7 Tests are not optional
- Run the subproject’s tests (or at minimum its smoke/run steps) before claiming success.
- If no tests exist for the touched logic, add a minimal regression test or a lightweight validation harness.

---

## 2) How to decide which playbook to use

Use a subproject playbook when **any** of the following is true:
- The user prompt names the folder (e.g., “work on `volume-cartographer`”).
- The files you’re editing are under that folder.
- You’re asked to run a binary/script that clearly belongs to that folder.

If the prompt is ambiguous:
- Start by mapping the repo structure and identifying candidate entrypoints.
- Propose a plan that separates “discovery” from “changes”.
- Avoid risky changes until scope is clear.

---

## 3) Subproject playbooks

### 3.1 `volume-cartographer/` playbook (activate only when prompted)

**What it is (from repo context):**
- A CPU-based computational geometry / volumetric pipeline project.
- **Language:** C++
- **Build:** CMake
- **Key script:** `volume-cartographer/scripts/build_dependencies.sh` is the source of truth for dependencies.
- **Platforms:** Ubuntu + macOS, amd64 + arm64
- **Current optimization constraint (from prompt context):** focus on speedups **without numeric changes**.

#### Build & test (discover actual entrypoint first)
1) Read and follow:
   - `volume-cartographer/scripts/build_dependencies.sh`
2) Locate the correct CMake entrypoint:
   - repo root `CMakeLists.txt` vs `volume-cartographer/CMakeLists.txt`
3) Prefer:
   - `RelWithDebInfo` for profiling
   - `Release` for final performance numbers
4) Export compile commands where possible:
   - `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`

#### Performance constraints (strict)
- **No numeric changes**:
  - no `-ffast-math`, `-Ofast`, “fast” approximations, reduced precision, epsilon relaxations, etc.
- Avoid nondeterminism in results:
  - be careful with parallelism and iteration order changes
- Favor improvements that preserve exact math:
  - fewer allocations
  - better cache locality / data layout
  - pruning and early-out logic that is mathematically equivalent
  - algorithmic broad-phase that does not change accepted/rejected sets

#### Workload inputs
- A representative dataset may be provided as `<folder.volpkg>`.
- Treat that as the canonical perf workload unless instructed otherwise.

#### Required deliverables for perf PRs
- Profiler hotspot summary (top functions by time)
- Before/after benchmark table (command, dataset, build type, iterations)
- Explanation of why numerics are unchanged
- Minimal regression test or validation step if the hotspot lacked coverage

---

### 3.2 `vesuvius/` playbook (activate only when prompted)

**What it is (from repo context):**
- Deep Learning pipelines for **3D computer vision**.

Because ML stacks vary, **do not assume** the framework or environment manager. You must detect it from repo files.

#### Environment & reproducibility
1) Identify environment definition:
   - `pyproject.toml`, `requirements*.txt`, `environment.yml`, `poetry.lock`, `uv.lock`, `Dockerfile`, etc.
2) Follow project-provided commands/scripts for setup and running.
3) Preserve reproducibility by default:
   - fixed seeds where used
   - stable evaluation protocols
   - avoid silently changing preprocessing, augmentations, normalization, or label semantics

#### Performance and “speedups” in ML context
Unless the prompt allows numerical changes, do **not**:
- change model precision (fp32 → fp16/bf16)
- change kernels, quantization, approximations
- change batch sizing or input resolution to “cheat” throughput

Safe speedups (often no numeric change) can include:
- removing data-loading bottlenecks (caching, prefetching, pinned memory where applicable)
- reducing redundant preprocessing
- improving I/O (sharding, memory mapping) while preserving exact bytes/values
- eliminating unnecessary tensor copies/conversions
- batching and vectorizing CPU-side preprocessing deterministically

#### Required deliverables for ML changes
- Exact run command(s) used
- Metric comparison (before/after) for correctness-sensitive changes
- Throughput/latency measurements (before/after) for performance work
- Notes about determinism/reproducibility impact

---

### 3.3 Template playbook for other subprojects (fill in when you encounter them)

When the prompt targets a different folder, create a mini playbook in your notes (or extend this file if requested) with:

- **Folder:** `<name>/`
- **Purpose:** what the subproject does
- **Language(s):** `<...>`
- **Build/run:** `<commands or scripts>`
- **Tests:** `<how to run>`
- **Platforms:** `<os/arch constraints>`
- **Non-negotiable constraints:** `<numerics, determinism, backwards compatibility, etc.>`
- **Typical inputs/datasets:** `<paths, formats>`
- **Perf protocol (if relevant):** `<how to measure>`

---

## 4) Cross-cutting implementation guidelines

### 4.1 Avoid “hidden” behavior changes
Even if output files look similar, changes in:
- iteration order
- concurrency scheduling
- floating-point accumulation order
- dataset shuffling
can alter results. Keep this stable unless explicitly allowed.

### 4.2 Prefer clear, local improvements
High-ROI improvements that are usually safe across projects:
- eliminate repeated allocations in hot loops
- reuse buffers
- improve data locality (SoA, contiguous arrays)
- reduce needless copies and conversions
- hoist invariants out of loops
- add early-outs that are logically equivalent

Do not copy an existing implementation into a new module just because the
existing code is private to another translation unit or package. Extract the
shared behavior into a reusable helper/library first, then make both callers use
that shared implementation. If extraction is genuinely impossible in the current
task, treat that as an explicit deviation and report it before proceeding.

Before writing code that reads, samples, interpolates, parses, caches or writes
Vesuvius data (Zarr, tifxyz surfaces, images, catalog metadata, display kernels,
transforms), check the [`vc_reader` capability map](vc_reader/README.md#capability-map)
and call what it lists.

### 4.3 Document anything that affects developer workflow
If you add:
- new scripts
- new dependencies
- new benchmark harnesses
document how to use them and how they’re validated.

### 4.4 vcobj readers and writers (including consumers outside `vc_object_store/`)

Use the [single versioned-reference contract](vc_object_store/docs/versioned-reference-contract.md): carry the shared reference unchanged. Consumer-specific tuples or flattened reference fields are not an internal API; extra application data wraps the reference instead.

Read `vc_object_store/docs/reading.md` and the contract in `vc_object_store/docs/coherent-store-api-cleanup.md` before changing vcobj handling. Physical placement (local, referenced, sharded or packed) is resolved by the store layer, never by consumer-specific SQL or fallback branches.

For reviews, use the [vcobj review contract](vc_object_store/docs/review-contract.md): trace each invariant through all sibling entry points and test the same case at their boundaries. Python bindings adapt Python types; the shared native implementation owns schema interpretation and validation.

- Resolve once, then carry the complete `ObjectRevision`/`ObjectVersion` through geometry, transforms, adapters and caches. Never replace its metadata with a fresh current-object lookup or infer its owner from the attachment. Exact reads must not require a live current object. Compare complete identities, not just version numbers.
- Transaction preparations, fetch callbacks, artifact uploads and deferred edit accumulators carry the same full references. Use `RecordSelection` and shared expectation helpers; numeric counters are storage projections or explicit current-selection inputs, never internal guards. Transaction commands use format 3 only: do not restore automatic format selection or numeric/full sidechannels. Reject incomplete saved identities rather than infer owners. Preserve the distinction between no live record and an exact selected tombstone, as well as loaded mapping/membership and active-write-lineage guards.
- Use shared identity, endpoint and tombstone resolution for writes as well as reads. Local SQL uniqueness checks do not enforce collection-wide identity. Canonicalize UUIDs through the shared API, not ad-hoc string comparisons.
- For edits of loaded data, use `expectObject(request, selected.reference())`, not a freshly read numeric version. Retain the loaded coordinate-mapping guard with editable points; an unchanged object reference does not imply unchanged volume registrations.
- Decide whether an edit changes anything before creating an override. No-op edits must preserve ownership and versions; required relationship revalidation must still run.
- Repin edits with the receipt's complete object references, not its committing store plus numeric versions: receipts also contain unchanged inherited inputs. Allocate structural point IDs through the shared retained-allocation API, never a restorable metadata counter.
- Use shared selection for geometry and artifacts from the same version. Query optional history through the owning record's policy, not the attachment's history flag. Copy/export collections through the shared closure APIs rather than backing up only their physical root.
- History policy controls retention, not identity or routing. Do not add history-off alternatives, proxy rows, per-object directories or copies of referenced payloads.
- Persisted caches and solved evidence must identify the selected source revision, not just its UUID or row ordinals. Geometry-derived counts, lengths and bounds come from the selected geometry, not ingestion summaries. Editors retain the loaded membership and transform baseline as well as object references.
- Public batch writes must roll back the entire operation even if the caller catches an error and commits its surrounding transaction. Physical copy/pack/conversion must preserve external locator targets through the shared relocation helper, including retained historical records.
- Extend a shared API when it lacks a capability; migrate existing callers instead of adding another partial resolver. Test local and referenced cases, history on/off, retained exact reads after override/deletion, and no-op writes. Keep production-code growth separate from regression-test/documentation growth when reporting refactors.

---

## 5) What to include in your final response (agent output format)

When you complete a task, include:
- **What you changed** (files + brief rationale)
- **How to build and run** (exact commands)
- **How you verified** (tests + dataset/inputs)
- **Perf results** (if applicable) with before/after numbers and methodology
- **Risks/limitations** (what might break on other OS/arch or edge cases)

---

## 6) Quick reminder: when to be specific

- If the prompt says “work on `volume-cartographer`”: apply §3.1.
- If it says “work on `vesuvius`”: apply §3.2.
- If it names another folder: create a lightweight playbook using §3.3 and proceed cautiously.
