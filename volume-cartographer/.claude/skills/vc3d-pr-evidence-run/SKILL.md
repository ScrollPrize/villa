---
name: vc3d-pr-evidence-run
description: Producing before/after evidence for a volume-cartographer change on real scroll data — checking the scenario actually exercises the fix, building and identifying both revisions, holding inputs fixed, and writing a log that says what was and was not proven. Load when asked to validate, evaluate, or gather CONTRIBUTING-required evidence for a PR, branch, or bugfix.
---

Assumes `vc3d-bridge-session`; the task skills (`vc3d-open-data`,
`vc3d-fiber-tracing`, `vc3d-visual-evidence`) cover the workflow itself.

`CONTRIBUTING.md` requires this on **real scroll data** — synthetic or toy
examples are not accepted — with a screenshot of the failure and of the tool
working afterwards, plus metrics where applicable.

## 1. Check the scenario before you run it

Most wasted evaluation runs are mis-specified, not mis-executed. Before
touching the app:

- **Read the diff and find what actually changed.** The commit subjects and PR
  body are a summary, not the contract. Locate the specific behavior, and its
  regression test if one exists — that test usually names the exact inputs that
  reproduce the bug.
- **Confirm the chosen sample can exhibit it.** The failure mode has to be
  reachable with that data. A rendering fix for placeholder surfaces cannot be
  demonstrated on a sample that publishes zero segments; a cross-level
  coordinate fix needs a sample that actually publishes two representations at
  different levels.
- **Predict both halves.** Write down, before running, what you expect to see
  on the old revision and on the new one. If you cannot state the difference in
  a sentence, the scenario is not ready and no amount of screenshots will fix
  that. Say so and propose one that is.

If the scenario you were handed cannot produce the evidence asked for, say that
first, then run the scenario that can.

## 2. Identify both revisions, explicitly

```
git worktree add <dir> <base-revision>       # or check out and build in place
cmake -S . -B <build-dir> -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo
ninja -C <build-dir> VC3D
```

Re-run `cmake <build-dir>` before **every** build whose identity you will
quote. The git revision is embedded at configure time, not at build time, so
committing and running `ninja` alone produces a binary with new code and the
previous commit's sha — `vc3d_ping` will then name the wrong revision with
complete confidence, which is the exact failure this check exists to catch.

Launch each build **by explicit path** (`--launch <path>` or `VC3D_BINARY=`).
Auto-launch scans a fixed preset list and can start a different, older build.
The discovery record has process/socket metadata but no executable path or
revision.

Call `vc3d_ping` on each session and record `gitSha` and `executablePath`. An
evaluation that does not name the two revisions it compared has not compared
anything, and this is the failure mode that silently invalidates whole runs.

## 3. Hold everything else fixed

Same sample, same attached representations, same seed coordinate, same camera
(center + zoom), same viewer selected by `surfName`, same render and overlay
settings. Note them all. Where the workflow has a fixture (a saved fiber, a
`seed.json`, a cached manifest), use it rather than hand-placed inputs — it is
the only way the run is repeatable.

Prefer a **script** over hand-driven calls for anything you will run twice:
`apps/VC3D/agent_bridge/test/` has `bridge_client.py`, `vc3d_process.py`, and
`png_utils.py`, which let a driver speak raw JSON-RPC in a few dozen lines.
Note that scripts there use RPC names (`canvas.click`) while MCP tools use
snake_case (`vc3d_click`).

## 4. Capture the failing half too

The "before" is the point. If the old revision crashes, capture the crash — the
terminal output, the last screenshot, the state that triggered it. If it
returns a wrong number rather than failing, capture the number. A bundle
containing only working screenshots does not show a fix.

Follow `vc3d-visual-evidence` for capture rules, and checksum the whole bundle:
two identical files under two different claims means the run failed.

## 5. Write the log

Structure it as `apps/VC3D/agent_bridge/test/flatten_render_demo/log.md` does:

- what was run, on which sample, with which two `gitSha`s;
- **proven** — with the artifact and metric that proves it;
- **found but not fixed** — real gaps encountered, flagged rather than worked
  around;
- **unconfirmed** — what the run could not establish and why.

That last section is not a failure to report; it is the part that makes the
rest trustworthy. A reviewer who finds an overclaim stops believing the whole
bundle.

Keep the artifacts next to the log (`wip/<sample>/` is the local convention)
and include, per artifact: the pane, pyramid level and scale, overlay volume,
and revision.
