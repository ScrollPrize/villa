# Agent Bridge — working principles

Design judgment for agents working on the bridge, the MCP layer, and their VC3D
integration. Process and verification (test layers, container workflow, commit
authorship) live in the maintainer's working notes and per-task specs — read those
first. This document is about *how to decide*, not how to build.

## Boundaries: optimize for the maintainers

- Optimize the **cumulative diff against `origin/main`** on maintainer-owned files,
  not the latest commit. Before finishing, re-read that diff for every established
  file you touched and grep for leftover migration types, flags, helpers, comments.
- Keep the bridge thin and domain-local. Reuse VC3D operation cores; don't mirror
  UI behavior inside the bridge.
- If both GUI and bridge need the same orchestration, put it in a **focused domain
  controller with neutral vocabulary** (the house idiom — see `ViewerManager`,
  `VolumeAttachmentController`). Never bury bridge lifecycle machinery in a large
  UI controller. The GUI should be the new component's first consumer; an API that
  exists only for the bridge is a smell.
- Follow existing architectural idioms. No opportunistic refactors of sibling
  workflows, no broad facades, no parallel controller architectures.
- **Stop rule:** if clean integration starts to require more than a few
  ownership/wiring lines in maintainer-owned files, stop and flag it. Thin
  duplication in bridge-owned code is an acceptable fallback; churn in
  maintainer-owned files is not.
- Fewer total lines is not the goal. A small cohesive domain component that removes
  churn and responsibility from established files beats a smaller diff that leaves
  machinery in them.

## Contract: descriptors are the source of truth

- C++ method descriptors own the wire contract: params, error codes, MCP mapping.
  Handlers hold semantics and cross-field rules only. Never add another
  hand-maintained contract artifact — the live snapshot and conformance tests
  check Python and SPEC against the descriptors.
- Descriptors must not silently tighten semantics: declare an enum or bound only
  if the handler truly rejects other inputs (see `viewer.rotate`'s `plane`).
- When SPEC and behavior disagree, **adjudicate and surface it** — never assume
  either side is right by default.
- Every new RPC method must justify existing instead of composing existing
  methods (`project.create` + `volume.open` is the model). The surface grows by
  default; resist it.

## Correctness: persistence and async

- Every input a shared API *accepts* must behave consistently across validation,
  loading, identity comparison, persistence, reload, and retry — even if today's
  callers don't exercise it. Watch relative paths, canonical remote locators, and
  collection-versus-item identity.
- Design mutations around **reopen semantics**: persistence is correct only if the
  same state is recovered after close and reopen. Test idempotency after reload,
  not just within the process.
- Treat multi-file persistence as a transaction: validate first, mutate once, roll
  back memory on failure, and log whatever disk state could not be restored.
- Asynchronous behavior is bounded, cancellation-safe, and observational —
  progress reporting never alters terminal outcomes. Workers capture values, GUI
  mutation returns to the GUI thread, callbacks are lifetime-guarded, and
  completion re-verifies the target project hasn't changed.
- No ambient mutable flags. Immutable per-operation options, typed requests,
  typed outcomes.

## Python MCP layer

- Tools stay explicit, typed, and readable top-to-bottom — no wrapper-framework
  cleverness. Tool docstrings are agent-facing UX and the most valuable
  hand-authored content in this layer: write them for the calling agent, and
  never flatten them into schema one-liners.

## Code and commits

- Comments state current invariants and non-obvious policy — never migration
  history, implementation stages, or review discussion.
- Split work into independently understandable, buildable, revertible commits:
  core behavior + tests; behavior-preserving extraction (a move reviewable as a
  move); bridge contract + integration coverage. Use the `vc3d:` /
  `agent_bridge:` / `vc3d-mcp:` prefixes.
- Add focused boundary tests: identity, reload, duplicate retries, conflicts,
  rollback, cancellation, object lifetime. Preserve Ubuntu/macOS and
  amd64/arm64 portability.

## The three checks that matter most

1. Inspect the **cumulative diff against `origin/main`**, not the latest commit.
2. Reason from **reopen semantics** for every mutation.
3. Verify that **every accepted input behaves consistently** across the whole
   lifecycle — even inputs no current caller sends.
