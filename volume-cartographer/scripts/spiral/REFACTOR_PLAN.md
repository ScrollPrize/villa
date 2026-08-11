# Spiral: one rebuild cut point, one checkpoint-load path

Two small changes. Together they remove more code than they add.

**1. A rebuild that only changes the model should not reload the data.**
`config.py::_runtime_impact` sorts every setting into `run_boundary` or
`new_fit`, and `new_fit` means one thing: tear down the whole `FitContext` and
build another. Changing `model_num_flow_integration_steps` therefore re-runs
`load_host_inputs()` — patch loading from disk, PCL linking, sampling caches,
host patch atlases, the trusted-geometry index, the track graph — and
re-materialises the Lasagna and surf-SDT brick pools, before passing one
different argument to the model constructor. `SpiralAndTransform` is built from
`umbilicus_zyx`, the flow-field corners, `config` and `spiral_outward_sense`;
of the host inputs that is the umbilicus alone.

**2. Loading a checkpoint has two unrelated controls.** A `checkpoint` path row
in *Fit and output* feeds `resume_path` on the next `POST /session/rebuild`; a
*Load Checkpoint into Fit…* button calls `POST /session/load-checkpoint`, which
keeps the session and 409s anything that is not an exact match for the live
model. The path row is a load input wearing a save-shaped label, next to
*Output directory*; it silently suppresses the local advanced-config profile
when non-empty; and after a successful in-session load the client writes the
loaded path into it, so it ends up displaying a load it had no part in, after
which typing there does nothing until the next rebuild.

They connect: the only reason the client must expose the rebuild path for
checkpoints is that a domain-mismatched checkpoint cannot be applied in place.
Once a rebuild has a cheap stage, "this checkpoint needs a rebuild" is a
bounded operation that can hide behind the same button.

## Shape

The build is already a linear pipeline — host inputs, output path, dense
stores, model/optimiser, shell and track device state — and dependencies run
one way through it. So the plan is **one ordinal, not a graph**: rebuild from
stage N onward. Two stages:

```
all     everything, exactly as today
model   retain host inputs and the dense stores; rebuild the model,
        optimiser, scheduler and everything constructed after them
```

Which stage a change gets comes from **one explicit allowlist of keys**, not
from key prefixes. Prefix-derived scoping has already been caught wrong twice:

- `optimizer_random_seed` is grouped with `model_*` by `_runtime_impact`, but it
  seeds `np.random` and `torch.random` at the top of `load_host_inputs()` and
  the pool generator during host preparation, so it reaches every
  RNG-order-sensitive host decision.
- `model_flow_bounds_z_margin` sizes the host-side `ShellPolarMap` used for
  track filtering (`fit_spiral.py:1329`).

18 of the 19 `model_*` keys never appear in `load_host_inputs` at all. The
allowlist starts as exactly those, and anything not on it is `all`. Safe by
construction: a key nobody has audited gets today's behaviour.

There is **one stage function and two call sites** — `ServiceState.rebuild` and
the checkpoint preflight — both reducing their situation to a set of changed
config keys and asking `rebuild_stage()`. If that logic ends up expressed twice,
something has drifted.

What participates in the rebuild diff is stated as an exclusion, not a scan:
everything outside `run.config` is `all` — paths, preview, `run_tag`, the z
window, and `defaults: true` — with `paths.checkpoint` the single exception,
routed through the cfg-diff rule in commit 4.

Deliberately **not** part of this: the phase-dependency machinery. `dependencies`,
`_dependencies()`, `input_change_impact()` and `FitInputSpec.runtime_impact` are
dead code — a fossil of the run-plan handshake that `0acd44257` removed — and
commit 0 deletes them rather than reviving them. One allowlist replaces the lot.

## Commits

Six commits, one `API_VERSION` bump (28), one stack. Prepare with `gh stack`
locally; do not push or submit.

**0. Delete the phase-dependency fossil.**
`dependencies` and `_dependencies()` from `config.py`; `input_change_impact()`,
`input_path_schema()` and `FitInputSpec.runtime_impact` / `.dependencies` /
`FULL_REBUILD_DEPENDENCIES` from `fit_session.py`. Nothing else in this plan
touches them, and reviewing them beside the new stage logic obscures both.
`Config.catalog()["schema"]["paths"]` keeps emitting an empty object, which is
what it emits today, so no client notices and no `API_VERSION` bump is needed.

**1. Split `build_device_state()` at the model boundary.**
Two methods: the store stage (Lasagna + surf-SDT pools) and the model stage
(umbilicus device tensors, flow-field corners, model, shell setup, optimiser,
scheduler, prepared device track tables). The composed call order is unchanged,
so the RNG and resume ordering the current comments flag as load-bearing is
unchanged. The shell and track device structures sit after the model today and
stay there: rebuilding them is cheap next to host loading and the brick pools,
and keeping the cut single is worth more than shaving them.

One thing does move, and it is what makes the model stage re-runnable at all.
Today the tail of `build_device_state()` *consumes* host data and then releases
it: `influence_anchor_geometry` is subsampled from
`verified_patches_and_pcls_cpu` (`fit_spiral.py:1899`) which is then set to
`None` along with `verified_patches_and_pcls_np`, so a second pass would call
`subsample_rows(None, ...)`. That use-then-release tail is host work with no
device or model dependency, as is `prepare_patch_dt_target_samples`; both move
above the cut into the host stage. The model stage then owns only what it
constructs, which is what commit 3 needs to be able to say.

The golden-run harness must be untouched by this commit. If it moves, the
decomposition is wrong.

**2. `MODEL_STAGE_KEYS` and the stage function.**
An explicit frozenset in `config.py` (the 18 audited `model_*` keys) and
`rebuild_stage(changed_keys) -> "model" | "all"`, returning `all` unless every
changed key is on the list. `_runtime_impact` keeps returning `run_boundary` /
`new_fit` so the configuration catalog and its wire format do not move.

Two tests. A source-scan test asserting no allowlisted key appears in
`inspect.getsource(FitContext.load_host_inputs)` — cheap, and it catches
exactly the leak that bit `model_flow_bounds_z_margin`. And a
rebuild-equivalence test in the golden-run harness: build, change one
allowlisted key, rebuild, and compare the checkpoint payload's key set, tensor
shapes and durable `cfg` against a session built from scratch with that value.
Structural equality is exact and is what this test asserts; parameter values are
not compared, because model construction draws from the global torch RNG and a
rebuild starts from wherever the session's training left that stream.

**3. `FitContext.rebuild_model_state()` and a runtime command.**
Release exactly what the model stage owns and re-run it. CUDA frees stay on the
owning fitter thread. `ServiceState.rebuild` derives the stage by diffing
`run.config` against `self.session_request`, with everything outside
`run.config` forcing `all` per the exclusion rule above; `stage == "all"` routes
through the existing teardown unchanged, so there is one construction path and
the new one is a strict subset.

A model-stage rebuild retains host inputs, so the ephemeral inputs incorporated
into them are still live — condition `_build`'s
`shutil.rmtree(previous_ephemeral)` on the stage, and make the panel's
uncommitted-inputs warning conditional on the same fact. No re-arming logic is
needed: either host inputs are retained and nothing changed, or the rebuild is
`all` and behaves exactly as it does now.

**4. `allow_rebuild` on `/session/load-checkpoint`.**
One verb, three outcomes. Called without the flag it preflights with the
existing `inspect_checkpoint()` (CPU-only, mutates nothing, reports every
failed invariant) and either applies the checkpoint in place as today, or
refuses with `409` carrying:

- `stage`: `rebuild_stage(keys whose value in the checkpoint's stored `cfg`
  differs from the live durable config)` — the same function commit 2 adds, not
  a second rule;
- `reasons`: the verdict's own text, unmodified;
- `refused: true` instead of a stage when no rebuild can help — a checkpoint
  written against a different `dataset_root`, or a `cfg` key set that does not
  match the current schema.

The stage comes from the **whole** cfg diff, not from the mismatches
`inspect_checkpoint` reported. Two reasons. `CHECKPOINT_MODEL_SHAPE_KEYS`
contains `model_flow_bounds_z_margin`, the one leaky key, so "only shape keys
mismatched" does not imply a model-stage rebuild. And `spiral_runtime.py:727`
lets a checkpoint's stored `cfg` override host-affecting keys the preflight
never flags, so a checkpoint differing in, say, a `track_*` key is `all` even
though it reported no incompatibility. A model z-domain mismatch reaches `all`
through `z_begin`/`z_end` on the same path, not through a special case.

`paths.checkpoint` is therefore the one path change that is not automatically
`all`: its stage is the cfg diff above. That is sound where it would not be for
host inputs — the checkpoint's contents are read at build time
(`spiral_runtime.py:686`) and re-read here, so there is nothing cached to grow
stale, and the diff is taken against the bytes that will actually be applied.

Re-POSTed with `allow_rebuild: true`, the service performs the rebuild itself
with that checkpoint as `paths.checkpoint`. The client never calls
`/session/rebuild` for a checkpoint. An uploaded checkpoint is already a legal
`paths.checkpoint` — `_dataset_session_request` accepts anything under the
output root and uploads land in `<output>/uploaded-checkpoints/` — so one
handle serves both routes and nothing is re-uploaded.

The escalated rebuild carries **no advanced-config overrides**.
`spiral_runtime.py:761` applies `run_config.config` on top of the checkpoint's
`cfg`, so a request that resends the live profile re-imposes exactly the
mismatching keys and fails the preflight a second time, now inside a rebuild.
The service builds the escalated request from `self.session_request` with the
checkpoint set and `run.config` emptied, and a request that names a checkpoint
while overriding a key that checkpoint's `cfg` disagrees with is a `400`, not a
silently mis-built session.

The preflight runs twice on the escalation path. That is a real cost, it only
happens on refusal, and it buys a single client-side code path; note it in the
docstring rather than caching a verdict.

`API_VERSION` 28, with commit 5 in the same PR.

**5. One Checkpoint sub-panel in VC3D.**
A collapsible *Checkpoint* section owning every checkpoint operation: the list
the service advertises (`serviceCheckpoints()` already unions
`session_checkpoints` and `detected_checkpoints`), a browse button for a local
`.ckpt`, and one **Load** button. Move *Save* and *Download* here too, so
checkpoints are one place instead of three.

Load is one flow: upload a local file first if needed (existing
`uploadCheckpointForResume`, which already reuses an identical checkpoint on the
host), POST once, and on a `409` show the reasons and the stage — naming what
will be rebuilt, and that uncommitted inputs will be discarded only when the
stage is `all` — then re-POST with `allow_rebuild` on confirmation. A `refused`
response reports the reasons and offers nothing.

Delete the editable `checkpoint` row from *Fit and output*; in its place a
read-only line reporting `session_paths.checkpoint`, which is what the live fit
was actually loaded from. Delete the *Load Checkpoint into Fit…* button and its
`QInputDialog`, and fold `loadServiceCheckpoint` / `loadLocalCheckpointFile`
into the one flow. `sessionAdvancedConfig()`'s suppression of the local profile
when a checkpoint is named stays — it is correct, because a checkpoint-backed
session takes its durable configuration from the checkpoint's stored `cfg` — and
stops being a surprise that follows from typing in a text box.

The panel keeps `_reloadRequired` as a boolean and its current message. Nothing
here needs per-field scopes.

## Explicitly out of scope

- Any third stage, and any scoping of *input path* changes. A config value is in
  the request and can be trusted; a host input path's contents live on a
  filesystem another process can mutate, so retaining a phase across such a
  change needs a content fingerprint. Every path stays `all` except
  `paths.checkpoint`, whose bytes are read as part of the operation itself (see
  commit 4).
- Reviving `dependencies` / `input_change_impact` / the run-plan handshake in any
  form. The stage is derived and reported inside `/session/rebuild`, never handed
  out as a token to be presented back.
- Per-field scopes on the run block. `run_tag` and `render_volume_scale` force a
  rebuild for no reason — a run tag is an output-directory label and the render
  scale is a preview parameter — but fixing that needs the run request to carry
  them, which is an independent change. Separate PR if wanted.
- Reusing host-prepared inputs across service restarts or between sessions.
- Transparent model-domain rebuilding without asking: the escalation is always
  the client's decision after a warning.
- Restarting the service as a substitute for a rebuild. It is not equivalent — a
  fresh process runs `startup_session_request(resume=True)` and resumes the
  autosave rather than applying a request — and it does not exist for remote
  profiles, where the service is persistent and operator-owned
  (`POST /service/restart` was removed in v23).
- Loss-function or intended numerical behaviour changes.

## Risks and gates

- **Retaining something that should have been rebuilt produces a wrong fit with
  no error.** The allowlist is the defence and it must stay small: a key goes on
  it only after its consumers are audited the way `optimizer_random_seed` and
  `model_flow_bounds_z_margin` were. The source-scan test and the equivalence
  test in commit 2 are the mechanism, not the intention.
- Commit 1 must not perturb RNG or resume ordering; the golden-run harness is
  the detector. Moving the `influence_anchor_geometry` subsample above the cut
  keeps its own explicitly seeded generator, so it is a reorder of host work
  only — but it is the one part of commit 1 the harness has to clear.
- **A model-stage rebuild re-runs the model stage against retained host state.**
  Anything in that stage which mutates or releases host structures breaks on the
  second pass; commit 1 moves today's instances above the cut, and anything
  added to the model stage later has to stay re-runnable.
- `rebuild_stage` fails safe by construction — anything not allowlisted is
  `all` — and must stay that way if a third stage is ever added.
- The checkpoint warning must state the real stage. If the stage is always `all`
  the warning is honest but useless, so commit 5 depends on commits 1–3 landing,
  not merely on being stacked after them — and on `paths.checkpoint` not being
  classified `all` by the path rule, which is the whole reason for its exception.
- `find_inconsistent_windings.py` and `phase_tuning.py` consume `fit_spiral` as
  a library (host-input loading + checkpoint model reconstruction) and must work
  at every commit; any commit that moves or renames something they touch ports
  them in the same commit.
