# Cross-call VC3D bridge footguns

These notes preserve behavior from the original `vc3d-agent-bridge` skill.
The focused workflow skill takes precedence when the two overlap.

## Editing

- `vc3d_grow_segment(method="manual_add")` is invalid; use
  `vc3d_manual_add_begin` and `vc3d_manual_add_finish`.
- Correction-point mode persists until explicitly disabled.
- A correction drag longer than about one voxel starts a `growth` job. Wait for
  it before issuing another editing mutation.
- Manual add, correction-point mode, and push/pull require editing enabled, an
  active editable segment, and no active `growth` job. Missing sessions have
  historically returned `-32007` with `data.kind="session"`; a busy growth
  source returns `-32004`.
- `vc3d_push_pull_start` uses the last pointer position. Move there first with
  `vc3d_drag(button="none")`, then start and later stop push/pull.

## Project replacement and asynchronous calls

- `vc3d_open_catalog_sample` is explicit consent to replace the current
  project; preserve any work that matters before calling it.
- A result with `jobId` is polled through `vc3d_job_status` or
  `vc3d_wait_job`. Deferred Lasagna and Spiral calls may instead keep the RPC
  open until an application signal arrives and have no job to poll.
- Examples of deferred Lasagna calls are dataset listing, job listing, and
  external-service readiness; measured replies can take roughly 10–15 seconds.
- Job concurrency is per source. The current sources are `tool`, `growth`,
  `lasagna`, `atlas`, `catalog`, `volume`, `flatten`, `seeding`, and `autosave`.

## Fiber and screenshots

- Fiber launch requires a compatible Lasagna dataset for the active volume;
  catalog `normal_grids` is not a substitute.
- A historical catalog check found no compatible Lasagna representation for
  `PHercParis4`. Catalog contents change, so rediscover rather than treating
  that observation as a current inventory.
- In a fiber pane, a plain click adds a control point; shift-click requests a
  predicted snap point.
- Fiber workspace viewer ids can change after an edit. Re-read state before
  targeting a rebuilt pane.
- `vc3d_fiber_launch(replace_owning=true)` replaces unsaved control points in
  the owning session. Set it false when building several fibers before one
  save.
- Screenshot capture rejects a hidden viewer with `-32009`. Switch to the
  viewer's workspace first. Earlier builds could instead return a degenerate
  image (one observed example was 15×50 pixels), which is not valid evidence.

## Seeding configuration

`vc3d_seeding_run` and `vc3d_seeding_expand` require a real local `paths`
configuration (`seed.json` or `expand.json`) resolved by the VC3D process. The
configuration's `normal_grid_path` is a local path and does not resolve a
remote store directly. Use the focused `vc3d-seeding` skill for the full
workflow and do not infer cache paths.

The preserved `seed.json` fields are `cache_root`, `thread_limit`,
`normal_grid_path`, `min_area_cm`, and `generations`. For a catalog resource,
the materialized normal-grid cache has historically appeared below
`~/.VC3D/remote_cache/normal_grids/<sampleId>/<volumeId>/L<level>-<hash>/`;
record the path returned by the live workflow rather than synthesizing it.
Remote CT volume resolution itself uses `Volume::remoteLocator()`. The seeding
source collection is the collection most recently committed through the point
workflow, so derive its points from inspected image/cursor state.
