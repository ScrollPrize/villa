# Plan: managed Fiberlet preprocessing and Atlas publication

## Proposed user interface

```bash
las_manager fiberlet run <fiber-inference-run> <normal-inference-run> [--threads 32] [-- VC_OPTIONS...]
las_manager fiberlet resume <fiberlet-run>
las_manager fiberlet ls
las_manager run ls
las_manager tmux attach <fiberlet-run>
las_manager open-data validate <fiberlet-run>
las_manager open-data upload <fiberlet-run>
```

The first selector must resolve to a completed manager Fiber 3D inference; the
second must resolve to a completed regular Lasagna normal inference. Raw
manifest-path inputs are deliberately outside the initial interface because
manager run selectors provide the source-volume, model, scale, license, hash,
and portable provenance required for safe publication.

`config init` will expose:

```toml
fiberlet_binary = ""                 # empty means resolve vc_fiberlets on PATH
fiberlet_threads = 32                 # portable default; this machine uses 128
fiberlet_params = []                  # additional native arguments
```

The manager always passes `--threads <fiberlet_threads>`. An explicit
`--threads` or argument after `--` occurs later and overrides the configured
default. The manager must resolve and record the exact executable path and
hash; it must not depend on `$SRC` or an interactive shell activation.

## Phase 0: establish the shared branch baseline

1. Bring current `origin/main` into `fiber-lets3` before implementing. The
   Fiberlet branch contains `vc_fiberlets preprocess-volume`, while its current
   tracked tree predates `lasagna/manager`; `las_manager` must be integrated by
   merging/rebasing the real main implementation, not copied from another
   branch or reconstructed from bytecode.
2. Resolve conflicts without changing existing inference, rolling-cache,
   tmux, completion, or Atlas behavior. Keep the native Fiberlet commits as a
   reviewable series above the shared main baseline.
3. Run the existing manager and focused Fiberlet tests after the branch update
   to establish the functional baseline.

## Phase 1: generalize managed jobs without duplicating inference orchestration

1. Extract the inference-specific launch/runner assumptions into one shared
   managed-job contract. The shared code owns run-directory reservation,
   atomic `metadata.json`/`command.json`, tmux creation, byte-for-byte teeing to
   `run.log`, signal forwarding, child process groups, PID reconciliation,
   status transitions, and completion validation.
2. Preserve existing inference record fields and lifecycle values. Add an
   explicit `job_kind` and a job-specific active lifecycle phase so legacy
   inference records remain readable while Fiberlet jobs use
   `fiberlet_preprocess` rather than pretending to be neural inference.
3. Keep `run ls`, selector resolution, stale-process reconciliation, and
   `tmux attach` generic over both job kinds. Keep `inference ls` filtered to
   inference jobs and add `fiberlet ls`/`fiberlet resume` for Fiberlet jobs.
4. Use the same runner module and tmux implementation for both job types. Do
   not add a second process supervisor, log copier, or tmux launcher.

## Phase 2: resolve and validate Fiberlet inputs

1. Resolve both positional selectors from the existing manager run index and
   require completed portable bundles. Reject active, failed, incomplete, or
   host-only runs before creating a Fiberlet run directory.
2. Validate the Fiber input as `fiber3d-prediction` with canonical
   `presence/nx/ny` products. Validate the normal input as regular Lasagna
   geometry with a complete `nx/ny` normal bundle; Fiber `nx/ny` must never be
   substituted for surface normals.
3. Require both inputs to identify the same sample and source volume. Compare
   the manifests' base shape, coordinate frame, group scaledowns, and
   `source_to_base` mappings so the native normal sampler sees the exact
   compatible base coordinate frame. Different stored levels are allowed only
   when their declared mappings are compatible.
4. Bind each dependency by manager run UUID, artifact kind, manifest-relative
   path and SHA-256, upstream model ID/hash/run/snapshot, source level, and
   source volume identity. Rehash the actual manifests at launch rather than
   trusting cached metadata.
5. Reject cropped or partial prediction bundles for the initial whole-volume
   command unless both inputs and the native sparse scan prove they cover the
   same complete base volume. This matches `preprocess-volume` semantics and
   avoids publishing an artifact that appears whole-volume but is not.

## Phase 3: launch and track native preprocessing

1. Add a focused Fiberlet launcher that builds exactly:

   ```text
   <resolved-vc_fiberlets> preprocess-volume <fiber-manifest>
     <run>/artifacts/fiberlets.zarr
     --normal-manifest <normal-manifest>
     --anchor-cache <run>/cache/fiberlets.anchors.zarr
     <configured-and-explicit-options>
   ```

2. Keep `cache/fiberlets.anchors.zarr` outside `artifacts/`. It remains durable
   and resumable on the local machine, but the final combined Zarr already
   contains compact anchors, prefixes, and routes, so uploading the float
   intermediate would duplicate a much larger implementation cache.
3. Pass stable, machine-readable source context to the native command: source
   sample/volume ID, Fiber and normal artifact/model/level identities, and the
   freshly computed manifest hashes. These values describe the data; local
   manifest paths remain input locations only.
4. Reserve a concise run name such as
   `<sample>-<volume>-fiberlets-<uuid8>` and a short tmux window such as
   `flt-<sample>-<uuid4>`. Record the original argv, exact resolved argv,
   working directory, executable SHA-256, Villa/VC commit and dirty state,
   effective options, input identities, timestamps, and host-private details.
5. Stream native carriage-return progress unchanged through tmux and
   `run.log`. A zero native exit is not sufficient: completion requires a
   machine-readable native completion check of the final combined dataset plus
   a completed portable provenance document. The checker must share the native
   schema/coordinate implementation used by preprocessing and return nonzero
   on an invalid or incomplete dataset.
6. Add `fiberlet resume <run>`. It reruns the recorded effective native command
   against the same output and local anchor cache after revalidating both
   inputs. Native payload tuples and the persistent anchor cache remain the
   only work-completion state; the manager adds no second checkpoint format.

## Phase 4: make the final Fiberlet artifact portable

1. Make the combined Fiberlet Zarr's root metadata the authoritative,
   self-describing reader contract. Store structured, versioned fields for:
   stable source volume and upstream Fiber/normal identities and hashes; the
   complete effective anchor/path/selection settings; prediction-to-base and
   grid/coordinate semantics; sparse layout; chunk geometry; storage profile;
   quantization; route-cost schema; and every codec parameter needed to decode
   chunks. Do not rely on undocumented defaults or a caller supplying matching
   settings merely to open the dataset.
2. Add a native read-only open path that parses this metadata, validates that
   all required fields are present and internally consistent, and constructs
   the compatible chunk decoders/readers from the stored values. A separate
   compatibility check may compare requested settings with stored settings for
   generation/resume, but that comparison is not the read protocol.
3. Split runtime locations from persisted identity. Positional Fiber/normal
   manifest paths, output directories, and cache paths are used to execute the
   job and may be retained only in private manager `command.json`; they must
   never appear in the public Zarr metadata or either fingerprint.
4. Canonically serialize the self-describing structured metadata and derive:
   - `algorithm_fingerprint` from the schema/algorithm version and complete
     effective scientific processing, coordinate, layout, storage, and codec
     settings;
   - `dataset_fingerprint` from that algorithm contract plus stable source
     sample/volume, upstream artifact/model/level identities, and manifest
     hashes. It remains embedded in every chunk as the dataset-integrity guard.
   The exact executable hash, host, platform, and build path are audit
   provenance, not scientific identity.
5. Bump the unpublished Fiberlet metadata schema/identity version so existing
   path-dependent outputs fail explicitly instead of being silently read under
   the new contract.
6. Keep whole-dataset sparse validation explicit: use the persisted stable
   source references to resolve the actual Fiber prediction, then perform the
   mandatory fresh presence scan to reconstruct the expected active chunk set.
   This resolution must not require or recover a stored producer filesystem
   path.
7. Add `artifacts/fiberlets.json` as bounded publication provenance for
   `artifact_kind = "fiberlets"`. It references the Zarr's authoritative
   metadata and adds source license, manager run IDs, executable/code audit
   provenance, and a structural inventory. It must not duplicate algorithm
   defaults, enumerate sparse payload files, or contain credentials/host-local
   paths.
8. Validate every expected final active tuple using the native
   dataset reader/checks before marking the job complete. Verify the portable
   bundle contains only `fiberlets.json` and the final combined Zarr; exclude
   locks, temporary files, the float anchor cache, logs, command records, and
   private manager metadata.

## Phase 5: command completion and operator ergonomics

1. Add unique-prefix command handling and contextual help for `fiberlet run`
   and `fiberlet ls` through the existing command registry.
2. Complete the first selector from completed Fiber 3D inference runs and the
   second from completed regular Lasagna runs; descriptions should show
   sample/volume, model, source level, and status. Complete only manager-owned
   options after the two selectors and stop interpreting tokens after `--`;
   the native binary remains the authority for its full option set.
3. Extend human run output with job kind and, for Fiberlets, the two upstream
   run names instead of a nonexistent single snapshot. JSON/durable metadata
   remains the authoritative interface.
4. Update configuration rendering/validation so older config files work by
   defaults and new `config init` files include the binary and argument keys.

## Phase 6: reuse staging upload for Fiberlet bundles

1. Generalize the existing inference-only upload plan into artifact-kind
   dispatch while retaining the same `_INCOMPLETE` marker, fixed bundle
   inventory, `rclone` defaults, retry behavior, and lifecycle recording.
   Existing Fiber/Lasagna uploads must remain byte-for-byte compatible.
2. Stage Fiberlet bundles below a distinct run-UUID namespace such as
   `<upload_staging_s3>/fiberlets/<run_uuid>/`; never mix them into the existing
   `inference/` namespace.
3. `open-data validate` must run all local structural and dependency checks
   before uploading. `open-data upload` stages only `artifacts/`, invokes the
   Atlas Fiberlet ingester, and records the returned Atlas entry just as it
   does for inference.
4. Walk/hash/upload sparse payloads as a bounded-memory stream. Do not reuse an
   inference helper that materializes a tuple of millions of paths.
5. Keep upload independent of local anchor-cache availability after successful
   preprocessing; only the portable final bundle is required.

## Phase 7: minimal Atlas representation and publication support

1. Add one volume data type, `fiberlets`, rather than registering a fake
   trained model. Its identity parameters reference the existing Fiber model,
   normal Lasagna model, their input levels, and a deterministic processing ID
   derived from the two manifest hashes and the canonical Fiberlet dataset
   fingerprint. Executable/build provenance remains audit metadata and does
   not split scientifically identical representations.
2. Add a deterministic canonical location under the volume's representations,
   for example:

   ```text
   <sample>/representations/fiberlets/
     <volume>-fiberlets-<fiber-model>-<normal-model>-<processing-id>/
   ```

   The exact template is fixed in the implementation/spec update and covered
   by tests. It must distinguish materially different settings without using a
   random manager run UUID as scientific identity.
3. Extract and reuse the existing Atlas ingestion/origin/dependency helpers;
   do not copy the inference-bundle implementation. Add only the Fiberlet
   structure and parameter specialization required by the new data type.
4. Add a small Fiberlet bundle adapter alongside the existing inference-bundle
   adapter. It verifies source volume/license, both upstream model/data
   entries, levels, hashes, processing ID, combined-Zarr structure, and
   collision/idempotency before appending the minimal `fiberlets` data entry to
   the volume. It creates no model record and does not duplicate inference
   ingestion logic.
5. Add exact `fiberlets` parameter constraints, including both upstream data
   IDs/model IDs/levels/hashes and the Fiberlet dataset fingerprint. Add Zarr
   structure checks for the
   root plus `anchors`, `prefix`, and `routes`, temp/lock exclusions, a
   copy-first canonical target, and the existing CC-licensed public secondary
   target. No derive action or Kubernetes processing is needed; Atlas only
   copies the already-produced static artifact.
6. Ensure public metadata export retains the public origin and dependency
   parameters while excluding the private staging origin according to the
   existing Atlas convention. Confirm data-sync plans exactly one canonical
   copy and one public copy for the Fiberlet entry.

## Phase 8: validation

1. Manager unit tests with synthetic completed run bundles:
   - command parsing, unique abbreviations, help, and completion filtering;
   - exact native argv/config override behavior and binary discovery;
   - rejection of passthrough overrides for manager-owned input, output,
     normal-manifest, anchor-cache, and source-context options;
   - Fiber/normal role rejection, incomplete runs, source mismatch, coordinate
     mismatch, crop rejection, and manifest rehashing;
   - generic runner lifecycle compatibility for old inference records and new
     Fiberlet records;
   - tmux creation/attachment, log teeing, failure/signal reconciliation, and
     post-exit structural validation;
   - portable inventory exclusion of the anchor cache and all host-local paths;
   - `fiberlet resume` reuses the exact output/cache and revalidates inputs;
   - upload dispatch, staging prefix, marker/rclone behavior, and retries.
2. Native focused tests:
   - dataset fingerprints are identical after relocating identical input
     manifests;
   - stable source references and hashes round-trip with no absolute path;
   - a reader configures itself entirely from stored processing/coordinate/
     layout/storage/codec metadata, with no caller-supplied defaults;
   - moving identical inputs leaves both canonical metadata and fingerprints
     unchanged;
   - wrong references/hashes or old metadata fail loudly;
   - existing presence-derived sparse completeness and resumability remain
     unchanged.
3. Atlas tests:
   - exact parameter matching and canonical path rendering;
   - Fiber and normal dependency validation;
   - idempotent ingest and collision rejection;
   - combined-Zarr validation and private-origin filtering;
   - data-sync planning produces only canonical/public copy actions;
   - metadata export includes the new representation.
4. Run the existing Lasagna manager/inference/open-data suites unchanged, the
   focused Fiberlet C++ tests, and the Atlas test suite. Build the native target
   on the project-supported build tree; do not install/bootstrap dependencies
   without explicit permission.
5. Run one tiny synthetic end-to-end smoke: create two compatible prediction
   manifests, launch `fiberlet run` through fake or real tmux, wait for
   completion, validate the portable bundle, run upload against a fake object
   store/temporary Atlas checkout, and verify the planned canonical/public
   copies. A production whole-volume run remains an operator test because it
   is unbounded and expensive.

## Commit structure

1. Bring the Fiberlet branch onto current main without content duplication.
2. Generalize manager job orchestration with no inference behavior change.
3. Add native portable Fiberlet source identity and focused tests.
4. Add `las_manager fiberlet` launch, provenance, completion, and docs.
5. Generalize staging upload and add Fiberlet upload handling.
6. Add the minimal Atlas `fiberlets` data type, ingester, publication config,
   and tests.

## Spec Update

- Add the managed Fiberlet CLI syntax, completed-run selector rules, exact
  Fiber/normal role and coordinate validation, durable local anchor-cache
  placement, portable final bundle layout, and generic manager lifecycle.
- Specify the complete self-describing Fiberlet read contract,
  location-independent dataset/algorithm identity, stable source references,
  manifest hashes, and explicit source resolution for sparse validation.
- Specify `fiberlets` as a derived volume representation with two upstream
  Lasagna dependencies, deterministic processing identity, canonical location,
  validation, and copy-first public publication.
- Preserve every existing whole-volume sparse eligibility, scheduling,
  resumability, atomic publication, and progress requirement.

## Docs Updates

- Extend `lasagna/docs/manager.md` with config keys, `fiberlet run/ls`, input
  compatibility rules, tmux/log/status behavior, artifact/cache layout,
  validation/upload commands, and the Atlas publication sequence.
- Extend `volume-cartographer/docs/fiberlets.md` with stable source-reference
  metadata, location-independent fingerprints, and the manager invocation.
- Extend the Atlas operator/runbook documentation with the `fiberlets` data
  type, canonical path, dependency requirements, plan/execute/status commands,
  and expected two-copy plan.

## Changelog Update

- Record managed whole-volume Fiberlet preprocessing, portable Fiberlet bundle
  provenance, shared manager lifecycle/upload support, and Atlas publication as
  one dated cross-project feature entry after implementation is validated.
