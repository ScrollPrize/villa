# Full embedding evaluation 2026-09-14

## Latest visualization correction and eligibility follow-up

- Corrected v9pilot completed exit0 in8.180279s coordinator time,0 inferred tiles/
  0published chunks. Paris4 short reference retained5.57785mm and excluded5.76mm;
  1 restart total, mean3.718567mm across both directions (F1/R0). Forward combined
  image visually inspected: cropped valid span only, paired CT/presence and aligned
  R0-/R0+ plus dense transverse sections. Full v9 now submitted through approved
  runner, requiring this successful filtered pilot; no old coverage comparison.

- Canonical-path follow-up verified: approved test runner passed18 native cases
  and71 Python tests (18.39s), including file/directory alias equality, changed
  content rejection and missing target rejection; diff check clean. v9pilot has
  acquired all8GPUs via manager86eb8bf9-e6c3-4ca5-86d3-3262c3ae6837. Training
  now advertises pausable=true/priority0, so normal high-priority preemption works.

- v8 pilot acquired resources but failed native preflight: native plan retained
  split/test symlink spelling while Python binding resolved to all/. Content and
  selected spans were unchanged. Fixed native shared referencePlan to canonicalize
  existing files, matching Path.resolve. Independent narrow-plan review approved;
  file and directory symlink regressions also check content drift and broken links.
  Failed v8 artifacts preserved; v9 pilot/full retain existing inference namespace.

- Final build and approved test runner passed18 native cases +69 Python tests
  in14.70s; git diff --check clean. Reference plan refreshed with actual per-CP modes,
  original arcs and SHA audit; per-run case/image links retained in final reports.
- Filtered v8_pilot is queued as manager task2a50a63c-3d4b-481a-a7a3-d7421f205790
  (root1471052, exec session59574). It is blocked by non-pausable training task
  9ca62151-ab05-492b-a865-d8cd3f7b677c, label5scroll_json_eligible_128_sd1_16x8_ema,
  which holds all8GPUs exclusively atpriority100. No training interruption authorized
  or attempted. Temporarily controller-paused OUR queued task for safe rebuild,
  then released that hold after tests. Full v8 cohort run is prepared, not launched;
  no filtered real-run compression/tracing metrics are claimed yet.

- Eligibility implementation: shared exact CP-index helper extracted from existing
  loadFiberJson; native maximal eligible-run selection reused by export and replay.
  Python reference prefetch consumes that export, records SHA256s and checks before
  launch; native verifies the full semantic plan. Separate crops keep source_index
  and run_index, original arcs and policy provenance. No fabricated skipped edges.
- Cohort audit:0139 10 fibers/31 runs,0332 3/3,1299(label1229)5/5,1451 2/2,
  Paris4_eval24/41,Paris4_fibers6/8. Total50 fibers/90 runs/180 cases;none all-excluded.
  Native reference_plan.json saved in classic_reference_vis_20260914_v8.
- Approved no-argument test runner:18 native cases and64 Python tests passed15.49s,
  including mixed actual-mode filtering, extrapolation, drift rejection, all-excluded,
  reverse original arcs and multiple-run denominator. One raw sandbox pytest attempt
  stalled on Zarr fixture setup; superseded by successful approved runner, not a
  code workaround. No full-cohort run launched before filtered pilot verification.

- Reconfirmed approved test/eval runners after user interrupted a raw pytest
  escalation. Native replay runner now supports a no-argument config-driven
  inspection suite; explicit arguments preserve its original full suite. Reuse
  that runner and run_classic_fiber_evaluation.py, not raw test commands with new
  environment prefixes. No new runner approval is necessary for these existing
  approved prefixes.

- v7 uses 2 base voxels/output pixel, scale1, CT level1 when available with
  trilinear/fine-to-coarse sampling labeled on rows. Shared inverse line-strip map
  selects actual traced points after dense layout positions; every restart has
  committed old endpoint and actual new start, extra lanes preserve alignment.
- Native test_fiber_replay passed0.60s;18 native CLI tests passed13.44s including
  unchanged traces, paired restart endpoints, exact center-column/page mapping.
  Independent review passed; generic unusually tall repeated-context pagination
  remains a bounded-layout limitation, irrelevant to current threshold20 strips.
- Real v7_pilot completed successfully: short Paris4 fiber,17 failures and
  1.193457985mm mean unchanged. Reverse image inspected;95 dense sections and
  10 old/new pairs correctly aligned. Eight extra tiles needed for new sampling
  coverage; historical Classic inference key reused, not invalidated.
- User added full-evaluation exclusion of extrapolation and Lasagna CP spans.
  Full v7 cohort render held before launch; implementing eligibility before new
  full50 run. Existing v5/v6/v7_pilot outputs are unfiltered historical evidence,
  not corrected eligibility results.

- Branch fiber-vischeck at3b2f79701; native policy83e695caf. Previous log archived docs/embedding_refinement_task_log_20260914.md.
- Reference SHA0a2651b7 step6500/832000;64 restarts is only hard-fiber tuning result.
- User requests partial allocation:one nonexclusive20GiB GPU,batch1,32CPU planned. Prior peak4.74GiB. Float32 accumulation matches tuning; historical full evaluation used8GPU/batch8/float16, disclose differences.
- Independent review approved matrix; report Classic-normalized equal-scroll score plus all per-scroll metrics, including newer finetune table. Own native build required because villa binary was rebuilt from different dirty sources. Existing /tmp/vc-live-native-cmake targets villa2; build exposed missing standard <cstring> include in ChunkCache.cpp, added include only, no runtime change. Initial guessed test target names did not exist; corrected to test_fiber_trace3d and test_fiber_reference_replay_benchmark.
- Frozen best99 update9900/samples1267200 SHA8f2b330d96401cd3fcb8cde232bebad1dec988b7c58e98a1c7f9d7885c6a9d91; current99 update32000/samples4096000 SHAd1c74163d47e331ec2e0235ec41060f49ddf363a69e8b4c896644aa5032a8e2f. Both decay.99 verified with shared freeze_snapshot_selection helper, frozen before evaluations.
- Own RelWithDebInfo native build complete;2ctest suites passed0.27s. Binary SHA4af38bdf8ac04500d5b54cab6ad97a0e6aafd69c4bd83c2ee94e4424d83b0fd6. Premature prepare during libvc_core linking failed before output; rerun after build passed. All six configs prepared50fibers100directions. First validate-full passed.
- First full reference6500/tuned active manager task14f760e8-814f-4145-89ff-349469d64724, nonexclusive32CPU/20GiB GPU. About4–5tiles/s initial steady state;31853 initial tiles. Native results pending, not claim completed.
- Python tests in sandbox stalled at signed-float32 Zarr storage test; interrupted original and diagnostic retry. Faulthandler showed Zarr asyncio future wait. Rerun outside sandbox using approved pytest entrypoint; results pending. No production behavior change to work around environment.
- Outside-sandbox rerun:23passed1skipped in1.78s. Skip is native diagnostic test hardcoded to absent checkout QuickBuild path; actual /tmp native suites passed. Sandbox-only Zarr hang is not present in real runner.
- Independent final review approved all six configs: exact classic source definitions/counts and tuned refine-0118 args plus explicit nearest touch. Created read-only report_full_embedding_matrix.py reusing existing comparison/metrics;26 historical audits passed. Strict pending rejection verified;3local report regression tests passed1.52s. Manifest records all6 configs,26historical roots and frozen metadata.
- User corrected allocation: use all8GPUs. Stopped sequential supervisor and single-GPU v1 attempt at~4150tiles/1422published, exit130; artifacts retained, no completed result. Manager confirms terminal and0heldCPU/GPU, no matching child processes. Multiprocessing resource tracker cleaned objects on interrupt (warnings retained). New v2 configs use8GPUs,20GiB each nonexclusive,96CPU,32readers,128inputslots,64resultslots/traceworkers; numerical/model/policy settings unchanged. Fresh v2 prediction namespaces, no inference-cache reuse from aborted run. v1 prepared configs retained as provenance only.
- Reference6500 tuned v2 completed100directions,exit0:2181restarts, pooled mean3.542989066mm; Classic-normalized score4.694521405 versus5.722681385 Classic. Native1054.2259s/coordinator1102.6301s,33970tiles. Subsequent process exit took several minutes; GPUs were already released from CUDA while manager still waited for process termination. Do not equate coordinator time with total lifecycle. Saved exact metrics/provenance in docs/benchmarks/embedding_full_reference6500_tuned_20260914.json. Matching off control active next; all GPU/model/policy settings preserved.
# Concurrent visualization task (2026-09-14)

Combined-layout revision: user rejected separate CT/presence images and requested quarter resolution, transverse sections/acceptance ellipses, restart IDs+arc metadata. Stopped v5 coordinator1238161 with SIGINT; exit130, outputs/cache retained. At least100trace.json plus partial images saved. Added report-progress mode to approved runner and generated trace_progress.md/csv/json for v5 all50pairs. Arithmetic mean of per-fiber means7.213334495mm. Short bad example Paris4 kb_20260729T182039882_000211:11.33785085mm,17restarts,1.193457985mm mean, cases92/93. No trace/inference rerun for this report. Live reporting now refreshes on trace/image-count changes, independent of final render completion, with explicit single-direction vs paired formula.

Implemented shared combined inspection renderer at scale2 versus legacy8 (quarter linear CT resolution), CT/presence paired on common CT-coordinate rasters. Added regular+endpoint+every-restart cross-section galleries with metric-derived plane/ellipsoid radii, long axis vertical, faint ellipse, reference center and rejected projection. Metadata includes restart index, directional/original arcs and mm, XYZ, geometry axes/radii/extents, projection/out-of-plane offset and cross-section pixel placements. Existing component placements retained. Original CT-only posthoc API unchanged; paired report accepts both formats. Cross-section extent interpreted as twice each acceptance radius (ellipse diameter).

Validation: new native test verifies exact metric boundary for ideal/oblique/parallel normals and invalid-normal isotropic fallback, reverse arcs and row modalities; ctest passed0.64s.18native Python integration tests passed14.53s, including live cold/warm and unchanged traces. Partial-report regression3passed1.60s. First revision missed source shape metadata, fixed; test also caught producer_implementation NameError in coordinator scope from previous cache-provenance change, fixed. Legacy longitudinal strips still reject invalid/parallel anchor normals even though transverse geometry is defined; tests separate those contracts. Rendering requires positive threshold. Restart text is on reference strips and every gallery station; dense coincident strip labels can overlap, but gallery IDs and JSON retain each restart (no claim of independently readable labels at identical pixels). Exact strip-marker pixel positions not yet included; arc/XYZ identity and cross-section placements are included as requested.

Prepared v6_pilot on that single Paris4 fiber and submitted through approved full runner, same namespace/model/settings and128publication workers. Full50v6 config ready; real visual audit and full completion pending.

v6_pilot completed2directions in34.9079s with ZERO inferred tiles,65warm chunks and no model-worker pool started.17restarts total,1.193457985mm mean unchanged. Combined real image visually inspected (CT/presence paired, transverse galleries); full per-fiber index produced. Full50v6 prepared and launched with approved runner, task9d93cfd1-0fcd-4fe3-b885-1203f476f071, exclusive8GPU priority100. Existing historical cache327bb949 reopened. No further native binary changes while this run is active.

User corrected the cache contract: checkpoint/model and inference parameters only, no code-file/runtime hashing. Implemented parameter-based keys and legacy metadata lookup retaining all other model/source/geometry/layout checks. Shared demand.cache_identity avoids duplicating the store contract. Legacy provenance and manifest are preserved; current producer hashes recorded in evaluation summary only. Missing old embedding_channels normalizes to disabled. Independent review approved. 88 identity/namespace/demand tests passed29.77s. v4 had already exited with native evaluator disconnected at~424s before attempted interruption (kill reported no such process); published chunks retained. v5 prepared50/100 with unchanged namespace/settings and submitted task23725e34-c516-4d09-9699-a2f230c4a65c; manager autostarted after unavailable socket and granted all8GPUs. Actual reuse verification pending.

v5 startup VERIFIED historical model-key327bb949 reuse for all5scrolls, all3existing prediction products opened; no new model arrays created. Original historical source/model parameters and full store identity checks passed. Visualization remains running; output classic_reference_vis_20260914_v5, per-fiber index generated only on completion. Prior blanket claim that old chunks were incompatible was incorrect under the requested parameter-only contract; only implementation metadata had prevented their selection.

Resume correction: v3 failed before inference with FileExistsError because inference_cache_root deliberately rejected existing namespaces. Removed that single-use restriction; validated namespaces now resume, with symlink/containment protections and unchanged per-model writer locking/identity validation. No chunks deleted or identities relabeled. Independent review approved. Tests: namespace/report/runner 32 passed1.72s; demand identity/reuse 3 passed1.26s. Config v4 retains classic_full_dynamic256_b8_20260911_v3_heap namespace and all numerical settings, sets flush_workers128 through existing config, writes separate audited visualization output classic_reference_vis_20260914_v4. Prepared50fibers100directions; submitted approved full runner, manager task0f2631df-69bc-4a74-8e46-d7164b8d5ff0, priority100 exclusive8GPU. Initially waiting behind final current32000/off evaluation.

v4 admitted with all8GPUs, passed namespace reopening and started inference. New model-key arrays were created (historical cache not reused); existing files untouched. At128.3s:4548tiles,1229published, publication queue0 and25 submissions observed concurrently (old cap4 removed via config). Full demand-inference tests57passed27.72s. Final current32000/off completed, and the complete32-row historical/new matrix is saved in docs/benchmarks/embedding_full_comparison_complete_20260914.md.

Important correction to earlier replies: same namespace does not imply matching model keys. Historical Classic hashes differ in numerical modules, and the whole-file runner hash also changes for this orchestration fix. Exact compatible predictions reuse automatically (tested); do not claim historical cache hits without observing them. Existing identity protections retained, no forced compatibility. Future improvement: separate numerical inference identity from orchestration hashing via explicit reviewed contract; not silently migrated here.

Classic reference visualization requested next: configured dedicated classic_reference_vis_20260914_v1 output with exact Classic best91.5k SHA f389da7914a6da34506f92204bf5441964e96599339dfe79dfc9c48b67165e17, baseline batch8/bf16/float16 accumulation, full50/100 directions. All8GPUs via nonexclusive manager60GiB each,64CPU,32trace permits; lower priority leaves ongoing runs alone. Historical reserved peak54.7GiB justifies allocation. Reuse historical namespace only under exact normal cache identity matching; no cache metadata rewriting. Prepared and audited all six collections/five scrolls. Manager task d1eefeee-297e-46e0-bab5-7b434133c293 initially queued insufficient_resources behind active newer evaluation.

First admission failed closed before worker startup: manager offered a partial grant. No inference/render artifacts were generated. An unrelated one-GPU20GiB comparison prevents a60GiB allocation on that GPU (inventory78.65GiB). Revised v2 config/output uses58GiB/GPU, still above observed54.7GiB peak plus2GiB context allowance; wait for all8 devices to have capacity before submitting to avoid another partial grant. Existing v1 prepared evidence retained, no task cancellation or manager-policy change. Active output target is classic_reference_vis_20260914_v2.

User then explicitly requested high-priority admission instead of idle-window polling. Stopped unsubmitted v2 polling; prepared/submitted v3 with priority100/exclusive all8GPU (manager task5142235d-865b-4cc2-b7cb-c57d1f91c800). Pending exclusive priority bars later lower-priority non-pausable evaluations from jumping ahead; current two non-pausable tasks must finish. Active output is classic_reference_vis_20260914_v3. Cache audit: historical Classic checkpoint matches, but implementation hashes differ for tiled_predict3d, inference_adapter, model and runner; old model identity also lacks embedding_channels. Thus existing runner does not automatically reuse historical chunks. Do not rewrite cache identities to pretend compatibility.

User confirmed: do not relabel; run again. Proceed with fresh current-code inference under its distinct model key, leaving historical cache files untouched. High-priority v3 remains submitted; supervisor continues through completion and automatic per-fiber report.

Added visualization_report.py, automatically invoked by approved classic runner after a render-enabled completed run. Writes root README.md/per_fiber.csv/per_fiber.json, grouping collection+fiber with both directions, mean=2L/(Fforward+Freverse+2), shortest-mean and most-restarts rankings, all CT/presence page links. Validates summary coverage, unique direction pairs, equal forward/reverse lengths, artifact restart counts and complete image pages. Independent plan review approved; new reporting + existing runner tests:7passed1.80s.

User requested inline many-fiber reference/traced top/side strips for CT and presence while evaluations continue. Existing replay-set renderer already reuses the old overview but is CT-only and rejects remote CT; extending that shared path, not inventing another geometry renderer. Preserve running evaluation binaries and use a separate build. Planning includes readiness for presence strip samples outside the traced centerline.

Implementation: extracted texture callback in shared FiberReplay overview, retained legacy Volume overloads, added bounded-batch presence sampling through the case-local readiness gate. Batch render emits CT and presence four-row page sets plus source/component/page metadata. Live runner binds input CT pyramid automatically; explicit local selected-array transform support preserved. Render-mutex wait suspends CPU permit to avoid deadlock with inference wait/resume. Independent reviewer approved coordinate/readiness behavior and flagged generic presence-only limitations; canonical-grid capability is now preflighted.

Validation: isolated `/tmp/vc-strip-vis-cmake` RelWithDebInfo/GCC13 build with separate shared libraries verified via ldd. Native test_fiber_replay passed (0.60s); Python test_native_replay_evaluation.py 17 passed,1 skipped (16.49s), including local HTTP remote CT, bidirectionality, both modalities, cold/warm identity-model inference, missing prediction chunks with one CPU permit, CT inline/posthoc byte identity and unchanged trace artifacts. Existing skip requires separately built vc_cache_root. Updated stale synthetic adapter constructor to accept the production emit_embeddings argument after first regression run failed before inference (16 passed before that failure).

Build setup initially selected a vendored libbacktrace download (blocked, no install); corrected pkg-config discovery to reuse `/tmp/vc-live-native-pkgconfig` and existing installed library. No dependencies installed and no ongoing evaluation executable/library changed.

Scope/limitations: no new full-cohort render run launched; synthetic presence image visually inspected. Presence uses full canonical prediction products and max over options; offline missing nx/ny can suppress available presence, unlike coherently published live products. Remote CT bindings use explicit base pyramids, not independently selected remote arrays. Existing post-hoc render-replay remains CT-only; requested inline flag supports both. Rendering can add inference requests and affect asynchronous accumulation scheduling; fixed-input traces are unchanged. Usage/spec/changelog updated in docs/batched_trace_strips.md.
# Traced restart label follow-up

Completed v10 full run, exit0,180/180 rendered cases and README/per_fiber index.
Zero newly inferred tiles or published chunks. Visually inspected case30:
R0-/R0+ visible on traced CT/presence top/side rows. Sorted {case_id,replay}
payload SHA256 matches v9 exactly:
3f12c0d3ca82fd248d0dce3c8b3672ff8f01fc26bbad41e34726a81853dc922e.
Results remain720 restarts and7.630697mm pooled mean. Gallery:
/ephemeral/hendrik/fiber3d_train2/evaluations/classic_reference_vis_20260914_v10/README.md.

Validation: cmake --build /tmp/vc-strip-vis-cmake --target vc_fiber_trace_metric
test_fiber_replay -j8; bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests:
19 native tests and71 Python tests passed (15.47s Python); git diff --check clean.
Independent code review passed. Known readability limits: raster labels may be
split at panel boundaries; unusually short strip heights may clip text. No
geometry/metric changes. V9 completed180 galleries before replacement run;
v10 config preserves checkpoint/cache/selection and regenerates through the
approved run_classic_fiber_evaluation.py prepare/full --config ...v10.json.

User found traced strips missing IDs. Root cause: labelRestarts drew text only
in the reference-frame path; traced-frame path drew marker bands but no text.
Added exact-segment endpoint Rn-/Rn+ labels after top/side raster assembly.
Independent plan review approved; no inference/geometry/metric changes.
Regression compares labeled vs unlabeled pixels in both traced views.
Initial synthetic test omitted referenceArcFraction; corrected the fixture to
match its failure arc. Existing saved JPEGs remain unchanged until regenerated.
# Restart episode reporting update

Final validation: approved native replay test runner passed19 native +77 Python
tests (17.90s Python); git diff --check clean. Coverage includes chained gaps,
exact80 boundary, first event, reverse, crop isolation, invalid events and
summary idempotence, plus published-cache missing-channel failure.

Implemented shared80-base-voxel grouping from preceding actual event; first
event counts, exact80 begins new episode, reset per directed run. Preserve raw
trace events and visual IDs; normalize reported summary/source counts, means,
quantiles/reasons and remove competing legacy seeded/failed means. Add episode
event-index/span metadata; rendering validation uses same filtered semantics.
Remove raw native progress count. Added CPU-only cached-full and report-metrics
to the existing approved runner; reuse PublishedStore/coordinate_native, no
inference pool, no GPU allocation, fail closed on any absent chunk.
Plan/code review approved after adding saved-normal-hash validation.
Classic full180 completed34.7212358s,0 inference tiles,0 published chunks;
reported510 restarts,mean9.953083mm (previous mean7.630697).
Current32000 tuned attempted same eligible-span policy but failed on Paris4
chunk[213,57,78] under model2b48bd699f99a46ae75ab011bae740e0feb70eb5e3354f02f473850349ed3532.
No inference was enabled. Recomputed original whole-fiber100-direction saved
traces instead:757 reported restarts,mean9.430056mm (previous3.926899).
Not a matched model comparison; current rerun remains blocked. Direct CPU mode
deviates from manager allocation because existing launcher mandates GPUs; no
manager code changed and no isolated throughput claim. Full numbers/commands:
docs/benchmarks/restart_episodes80_20260914.md. Existing vis unchanged.
