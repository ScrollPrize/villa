# Short failure benchmark — 2026-09-16

User requests sorting work by model. Removed four-batch fairness quantum: keep
each lane on its resident model while work is available, switching only when its
queue is drained. Other lanes remain concurrent and deterministic selection is
unchanged. Briefly interrupting66891 to apply scheduler change and resume state.
Resident-model regression passed9tests; no native processes remained after
interruption (resource_tracker emitted cleanup warnings on group SIGINT).
Resumed approved command session60249. Confirmed previous persistent run emitted
reused_worker_pool=True and kept~48GiB model allocator reservations, avoiding
batch reloads; observed4simultaneous native processes totaling~8268%CPU in one
snapshot (not a sustained average). Selected count can lag trial completion due
to deterministic original-order selection frontier; do not equate trials/cases.

Persistent queue validation passed254Python/8skipped and native23/84/11. Warm
borrowed-pool replay preserves failures and emits perbatch counter deltas; failure
closes borrowed resources. Prior real batch8 GPU peaks were39.38GBallocated and
49.93–50.57GBreserved: do NOT divide GPU budget between2active models. Corrected
policy retains2CPU/mmap sessions, only1CUDAmodel perlane, fullgrantbudgetguard;
model switches stop/restartGPU only. Fourbatch warm-model quantum avoids switching
eachbatch. Restarted sameapproved extraction command session66891, single grant.

Independent review + native reuse tests caught a real warm-session output race:
Python opened the reused pool's queue log immediately after connection, before
native exclusively created its output directory. Both new warm replay tests
failed with evaluation output already exists. Fixed by waiting for first demand
(or completed native process when no demand) before opening the log; Python no
longer pre-creates native output. Rerunning full native suite before real queue.

Current user request: persistent queue, not repeated per-batch inference setup.
Implemented fixed spawned coordinator per GPU lane, canonical cache-overlap
affinity, up to2LRUproducer sessions with conflicting-writer eviction. Each
session retains native input readers, mmap stores and existing DemandWorkerPool;
borrowed coordinate_native loop recreates only native replay/control per batch.
Per-session VRAM budget is granted per-GPU budget divided by retained-session
limit, enforced by existing allocator guard. Without explicit budget retain1.
Shared multiprocessing Event cancels coordinators; explicit close commands and
Finalize fallback release GPU pools before stores. Source catalog initialized
once on first actualcachemiss; checkpointSHA computed once per session. First
batch may use PublishedStore cached-only, subsequent batches reuse demand store.
Per-batch source counters are deltas; worker stats explicitly session-cumulative.
Native process/cache remains perbatch (not yet persistent).22focused tests pass;
native borrowed-pool reuse/failure cleanup suite running. Earlier corrected
parallel test completed2719trial audits (6626->9345),selected53->59 in180s plus
cleanup; these are trials, NOT2719benchmark cases or59newcases. No final ETA.

Parallel runtime validation: first60second probe ran8snapshot jobs and completed
253additional trial audits (6278->6531); selected52->53. Not a matched-input
speedup measurement. Broad speculative startup opened many source caches, so
limited rolling work to128unresolved candidates (refilled individually).
Next180second probe found an actual concurrency bug: select() rejected control
socket FD>1023 after ~66s; sibling tasks cancelled/drained. Replaced with portable
DefaultSelector (epoll/kqueue) and added native integration coverage holding
FDs through1100 open. Repeated bounded180second probe after fix. Auxiliary
worker pools are independently bounded, not included in128native worker cap.
gdb attach was denied by host ptrace policy; no profiling privileges changed.

User stopped session59796 due to low utilization; exit130 confirmed, no native
processes remaining. Previous1–3hour ETA was not measured and is withdrawn.
Implemented rolling asynchronous snapshot jobs rather than sequential model
batches and32-candidate tail barriers. Original-order commit frontier preserves
quotas/dedup,100voxel grid and confirmation rules. Single outer grant; disjoint
GPU lanes; canonical cache-set exclusion;128native trace workers total and64GiB
decoded cache total. Auxiliary pools have separate caps, not a strict aggregate
128thread cap (many loader threads are I/O waiters).
Bounded loader/accumulator/publication workers and buffers per job. Coordinator
owns state/geometry hashes; workers write distinct trials. Shared coordinator
accepts cancellation event; shutdown drains workers before releasing allocation.
Added bounded measurement flag; no unbounded rerun planned before measurement.
Known remaining overhead: inference workers still start lazily per batch, not
persistent across batches.18focused tests passed before final full suite.

User correction: acquire one exclusive priority100 GPU grant for the entire
extraction, not one per cache-extension batch. Stopped session8109; retained
trial audits and winner checkpoint. Runner now wraps the entire extraction
using existing launch_managed, passes all granted local device ordinals to the
shared extension helper, and prevents nested demand grants. Scientific settings
unchanged; resume v3 with existing evidence. Regression checks all8 device
ordinals forwarded and stale demand-manager config removed.

Validated: reusable regression suite244passed/8skipped, native23/84/11cases;
new outer-grant entrypoint test passed separately. Independent focused review
found no blocking ownership/device/resume regressions. Resumed same v3 approved
command under session59796; no orphan native tracer remained after stopping8109.

Best phase completed with inference gaps filled:41unique reproducible cases after
whole-fiber exclusion,200voxel target dedup and true-grace reproduction checks.
Padded total50738.225basevoxels/121.749003mm; lengths min300,median900,max10300.
Search now on other-model failures outside Classic raw failure windows, then
Classic backfill. These are partial benchmark results, not final200case set.

First v3 inference-backed retry succeeded:24tiles/3publishedchunks, original cache
reused. Lazy manager task9d2f3a16-70a7-4474-8f42-708552ae72f5 acquired all8GPUs
exclusively atpriority100, then released. Subsequent misses use the same path.
No missing trial is now excluded from the search. Best-phase currently22selected,
remaining directions in progress; final200benchmark not yet complete.

User corrected cache-only policy: missing inference must be produced so shorter
valid cases are not excluded. Stopped v2 search; new v3 config/output imports all
scientifically compatible valid/nonfailure trial audits but deliberately does NOT
import uncached_dependency trials or candidate/winner decisions. Rewalk distances
and retry unknowns. On cache miss use existing extend_cached_replay (shared demand
loader/accumulator/native bridge and lazy exclusive high-priority GPU manager).
All14 producer snapshots resolved from historical full_bindings and verified
against original cache model SHA. Added duplicate-source cache sharing to shared
extension helper (both Paris4 collections can share a cache). No new inference
implementation. Single reusable runner still performs entire workflow. v3 session8109.

Reusable-entrypoint regression rerun passes243Python/8skipped plus native suites.
Real shortened starts occasionally leave historical inference corridors: explicit
uncached_dependency trials are unknown, not successful/nonfailing. They cannot
be selected; shortness claims are shortest verified cached grid result, not proof
that all missing smaller starts would succeed. Full200extraction remains ongoing.

User requests one reusable extraction script. Entire workflow is one approved
runner mode short-failures, with shared Python library/native code internally.
Added minimal standalone workload config support (no unrelated pilot/source
bindings required), configurable baseline/best producer/evaluation labels and
total/best/other quotas. Input comparison uses existing report schema. Allpaths
come from config/history, no hardcoded dataset paths in extraction code. Documented
minimal config and command. Active v2 run uses same default semantics and continues.

Self-proximity preflight completed:50fibers,359265indexed subsegments,
25581narrowphase checks,0.570935s summed native checks. Four whole fibers excluded:
PHerc0139/kb_20260828T155910844_000043,PHerc0139/kb_20260905T035141671_000093,
PHerc0332/dj_20260810T113807458_000001,PHercParis4_fibers/kb_20260729T173847941_000210.
All hit normals valid; report stores source arcs/points/threshold ratio. Candidate
pool after exclusion:64best,213Classic,403other, before dedup/reproduction. Reduced
best quota backfills from Classic under approved policy. Regression suite now
243Python passed/8skipped plus23replay/84trace/11reference native. New tests confirm
straight/folded geometry screening and explicit short-piece interpolation eligibility.

User added whole-fiber self-proximity exclusion before failure selection. Paused
partial extraction (no final benchmark exported). Native spatial-grid segment
index,<=4base query spacing,500base nonlocal arc separation,same40/80normal-aware
metric. Eligible spans only; compare across pieces,flag original fiber globally.
Review caught constrained-minimum boundary at500; fixed permitted t intervals.
Sampling is not a certified continuous curve intersection. New output/config
short_failures_200_20260916_v2 binds exclusion/native hash. Session14878 real
prefilter/extraction; regression56762.

Pilot found28/first32candidates with verified direction; padded300–5100basevoxels,
median500before dedup. Added speculative batches up to128trials while retaining
sequential100grid decisions. Implementation-only resume retains old revision.
Review fixes: isolate speculative cached misses into explicit rejected trials
by recursively splitting batches; preserve interrupted outputs with attempt paths.

First real batch revealed generated annotations defaulted to Lasagna interpolation
when optimization_mode was omitted; native selection correctly excluded all116.
Fixed explicit native_fiber_trace3d mode for these already-eligible GT subpieces,
preserving coordinates and source provenance. Failed batch retained, no inference
performed. New protocol-hashed batch paths avoid overwriting old evidence.
Session80046 now performs actual cached native trials; first497audits include96
reproductions and reject3terminal-only excursions. Search continues; counts are
trial observations, not final distinct benchmark cases.

Implemented single-direction opt-in annotations (regular both-direction default retained), shared PublishedStore cached replay helper, approved-runner short-failures mode and checkpointed100-voxel grid search. Independent review required first-failure-only and actual outside-grace exhaustion audit to prevent terminal truncation manufacturing a failure; both implemented. Classic success requires completed both-direction eligible-piece coverage and no raw event within200basevoxels. Dedup after validation, compare padded total lengths including clipped tails. Preserve snapshot/config/native/protocol/cache/normal hashes and consumed historical artifact hashes. Inventory:14 untuned improved/baseline models,99best/292Classic/538other candidates before reproduction/dedup.

Approved workspace build succeeded. Approved regression suite:240Python passed/8skipped,23replay/84trace/11reference native cases. Four new pure-unit tests cover slicing/dedup/firstfailure/grace; two native directed cases pass. A standalone pytest invocation used the wrong installed checkout and failed import; approved runner sets workspace PYTHONPATH and passes. Full real search launched session23914 via existing approved run_classic_fiber_evaluation.py short-failures --config .../eval_short_failures_200_20260916_v1.json. Output /ephemeral/hendrik/fiber3d_train2/evaluations/short_failures_200_20260916_v1; no GPUs/inference enabled.

User approved up to99best,50other-only,51Classic with Classic backfill; match±100basevoxels. Cancelled fiberlet fullv2 session16292. Exit1 with manager cleanup timeout/shared-memory warnings; escalated process check confirms no native evaluator remains. Published caches untouched. Last observed progress9968.8s,174131tiles,214885publishedchunks. Full result incomplete, archived, not a completed quality benchmark. No128-chunk restart launched.

Plan uses100-voxel start search resolution and existing acceptance/grace. Changed short traces may request uncached neighbors; shared PublishedStore must enforce readiness, no silently missing inference.
## Accepted benchmark and common best-model evaluation

Approved regression runner completed23replay/85trace/11reference native tests,
258Pythonpassed/8skipped. Includes owner-coalescing on syntheticcold fiberlets,
warm payload hashes, empty coverage, permits and grouping. Latest userrequires
on-demand GPUlease: benchmark-full now passes existing DemandGpuLease to shared
session; no outer grant,30second idle release/reacquire policy unchanged and
covered by existing tests. CPUtracing continues withoutGPUownership.

User requested fix for underutilization, then clarified scroll-by-scroll dynamic
parallelism. Stopped session70013 exit130; published caches retained; final seen
531s/4931tiles/2961publications, not a completed quality score. Replaced64hash
stripes with exact-owner paired futures and RAM empty-owner memoization. Added
shared principal-compute semaphore for trace/anchor/path kernels, after path
dependencies; prediction readiness temporarily releases permits. Separate waiting
cache service threads remain to avoid dependency deadlock. This is a measured
first correction, NOT the full-volume finite-Z-slab scheduler. Ancillary halo
assembly/anchor-view work is outside permits;224is not a total thread-count cap.
Independent reviewer found no deadlock/lifetime blocker, noted this accounting
caveat. No numerical kernels, batching math, cached payload identity changes.
Scroll grouping retains bothParis collections and one common inference session.
New v2config224logicalCPU/8GPU. Originalapprovedtests255passed/8skipped;
added owner-coalescing, sharedpermit exception/reacquisition, grouping regressions.
Use ONLY existingapproved build/test/eval runners. Extraneous directpytestapproval
was interrupted; do not repeat it. Incrementalapprovedbuild passed.

Follow-up Classic completed session22549:153restarts,4.299285mm mean,.720median,
30.403s setup/replay/cleanup,8newtiles. Full same200pieces/662.089848mm validated.
Fiberlet128run launched session70013 with managed8GPU/128CPU grant82590ab3-a394-4e97-8c45-fc51421afd2b.
Historical beam config lacked canonical fiberlet fields. Extracted shared binder
from live_evaluate and reused with validated persisted cache identity; no duplicate
identity logic or new prediction namespace. Added backend/fiberlet-only override,
canonical field checks and persisted graph coverage summaries.2focusedtests pass.
Plan independently reviewed. Full results docs/benchmarks/short200_classic_fiberlet_20260916.md.

Completed session69464 exit0:200cases/662.089848mm,62reported restarts,
10.509363mm agreed aggregate mean,median1.650440mm,p05.161748mm,p957.907840mm.
Per-scroll restarts0139:27,0332:0,1299:9,1451:0,Paris4:26.
59pieces fail,141finish without reported restart. All41best-origin pieces
reproduce (41restarts); other50yield6,Classic109yield15across12pieces.
46.200s cache-extension setup/replay/cleanup,5.146s native coordinator;
zero new inference/zero publications,1653warmchunks. No cold-inference speed claim.
Artifacts/provenance/metrics preserved; full report docs/benchmarks/short200_best_u22000_20260916.md.
Coverage and exact length validated; git diff --check passed. No dataset changes.

Final extraction session60249 exited0:200cases,41best/50other/109Classic,
662.089848mm; scheduler1293.876s. Validated all annotation hashes, forward-only
directions, confirming trials and lengths. User accepted composition/shortfall.
Added benchmark-full mode to approved runner, reusing extend_cached_replay and
complete_report, no new tracer/inference implementation. Independent plan review
required200directed cases and exact dataset length, source/model provenance and
unchanged protocol; implemented. Two focused tests passed with workspace import
path (initial installed-checkout import failed; rerun with pytest pythonpath).
Launched session69464, managera6aa538a-a32f-4e7b-99ae-f571896e8100,
all8GPUs/128CPUcores, frozen best EMA999u22000 withoutembedding. Results pending.
