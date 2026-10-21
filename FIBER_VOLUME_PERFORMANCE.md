# Paris4 volume performance task

Goal: remove measured CPU bottlenecks in the resumable neural follower, on GPU5
only. Preserve checkpoint, precision, crop sizes, seed ordering, tracing policy
and completed tiles. No other GPU jobs may be interrupted.

Plan: verify resume; remove repeated NPZ member materialization during restore;
restart with bounded main-thread cProfile; rank costs and optimize the largest;
regression-test unchanged outputs and compare live decision throughput/profile.
Continue until the dominant avoidable costs are addressed or a blocker is found.

Baseline: ~96 decisions/s (current-band cumulative), GPU utilization observed3%,
28.49 CPU-seconds in30s in main process;6MiB physical disk reads. External perf
is blocked; Python remote unwinder cannot find target runtime. Use in-process
profiling at restart, not privileged attachment. Last saved coverage ~49%.

Resume contract: atomic tiles/*.npz are authoritative; partial tiles are ignored,
their unfinished JSON directories rewritten, saved fibers reconstruct ownership.
In-flight tiles restart; this is not exact mid-trace continuation.

Approved launcher: /home/hendrik/villa/vesuvius/scripts/run_fiber3d_experiment
with paris4-volume-{stop,start,status,tests}. Artifacts stay under
/ephemeral/hendrik/fiber_single_small/paris4, including profiles.

## Restore result

`paris4-volume-tests`:1 passed. It verifies each NPZ member is read once,
partial archives ignored, point values/IDs preserved, and next_id never reused.
`paris4-volume-restore-bench`: actual saved tilez04096_y02048_x01792,451 fibers,
three trials, same float64 extraction/checksum. Old median0.48049s
(min0.47377,max0.54895), new median0.003593s (min0.002559,max0.003875).
This134x is array extraction only, not total ownership restore or tracing.
No arithmetic change. Runtime profiler will rank tracing costs separately.

## Warm profiling and candidates (2026-10-19)

Artifacts: profiles/internal_20261019_203634 (cProfile, includes startup;
cross-thread accounting overlaps, so do not sum cumulative entries) and
internal_20261019_204648 (three warmed steps, torch profiler + Chrome trace).
The latter records ~2.44s CPU in cuDNN attention for tail sizes25/52 versus
~7ms GPU time; these shapes still incur first-use setup. Full64-row calls are
much cheaper on CPU. There are62139 cudaLaunchKernel calls across those steps.
Keep the same backend, precision and row count. Testing optional single-shape
CUDA graph replay for the existing64-row regression forwards; other shapes
remain eager. Compare all outputs exactly on three real batches before trusting
replay; benchmark three warmed eager/replay trials on identical inputs. Do not
claim the first-use cuDNN setup explains the long-running baseline by itself.

Recent pre-graph interval throughput:239,238,203,201 decisions/s onGPU5
(exclude104/s during profiler collection/export). This is a resumed region,
not a matched-input comparison against the old96/s cumulative figure.

`paris4-volume-axis-bench`: one CPU core (OpenBLAS/OMP1),128 deterministic
synthetic35-point continuations, three trials. Original predicted_axis median
0.147635s (min0.147109,max0.149991); cached invariant cone geometry0.063815s
(min0.063365,max0.064530):2.31x. Every resulting axis bit-identical. Also test
64 random centres at both cone resolutions, all candidate directions exactly
equal. This is not an end-to-end speedup claim.

`paris4-volume-tests`:4 passed (NPZ restore, occupied LRU equivalence across
300 requests, exact cone values, CPU graph fallback). GPU replay validation
runs inside the existing GPU5 manager allocation, not an unmanaged test.

Extended regression command also includes test_policy.py and
test_trace_sampling.py:13 passed in1.89s. The manager-launched process already
uses OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1; thread
oversubscription is not the identified cause here.

Graph replay checked three different live inputs exactly (all output tensors).
Same-input64-row forward benchmark, GPU5, three warmed synchronized trials:
eager0.043819/0.043951/0.043991s (min/median/max), graph including input copies
and output clones0.041324/0.041331/0.041360s. Only~6% faster: not the main fix.
Graph profile internal_20261019_205822 reduces CPU cudaLaunchKernel calls to
9871 but still includes first-use tail cuDNN setup. No precision/backend change.

Added stage wall timers (no added CUDA synchronization) for feed/checks/inputs/
forward/advance/heading/axes/finish, and runtime graph-on/graph-off controls.
Next: measure small-cache input cost then use paris4-volume-cache-grow to
increase GPU5 cache1024->8192 slots (2->16GiB), preserving resident slot/data
and free-list order. The old long-running log reported~509 chunk loads/s.
This is a hypothesis to test, not a measured cache speedup yet.14 tests pass.

## Continued measurements, 2026-10-19 21:20 UTC

After live cache growth, successive minute rates were321 (transition),377,395,
373,369,397,419,397,388 decisions/s, versus230 just before growth. This is not
a matched trajectory comparison: model warm-up and open-tile working sets
change. GPU5 utilization remains low:30 one-second readings averaged28.3%,
median3%, zero33.3%. CPU input construction21–22%, seed feed16–19%,
advance/checks18%, heading/axes14%, forward24–27% of latest step wall time.
Timers do not add CUDA synchronization; asynchronous work can cross stages.

Approved `paris4-volume-seed-bench`:64 actual seed locations from completed
tile z04096_y01792_x02816,3 repetitions,1 CPU core.13.067s total:12.931s CT
normal estimation,0.102s CT read. Saved cProfile/cubes under
profiles/seed-heading. Optimize the measured arithmetic/allocation cost, not I/O.

`paris4-volume-tensor-bench`:8 real77³ cubes,4 repeats/trial,5 alternating
old/new trials,1 CPU core. Original min/median/max1.74355/1.78768/2.09617s;
reused buffers/shared identical axis-0 smoothing1.19207/1.21264/1.37968s.
Median1.474x; every tensor exactly equal. No precision or reduction-order
change. All9 tensor elements still independently computed in original order.

Crop input path additionally stages metadata once per call with pinned
nonblocking transfers (formerly3 blocking copies per16 rows), and caches
volume bounds. Original row grouping, LRU decisions, sample arithmetic and
normalization unchanged.16 CPU regressions pass in2.12s; GPU exactness and
matched-input comparison configured inside the existing GPU5 grant on restart.
No claim of end-to-end benefit for these new changes until measured live.

Limitations: restarted in-flight tiles must be recomputed; completed atomic
tile archives retained. Current initial2GiB GPU cache grows to16GiB at the
first step boundary using runtime.json. No other GPU or training job touched.

GPU5 matched-input check passed bitwise for64 heading crops and498 actual
144x104x104 tracing crops, with16-row gather groups and the grown16GiB cache.
Tracing crop old min/median/max59.547/59.599/59.817ms, batched56.215/56.234/
56.273ms:5.65% improvement. Heading crop median2.804->2.616ms. These are
crop-only warmed timings, not end-to-end claims. CUDA graph replay again
passed3 live exact-output checks.16 CPU tests pass, including Fortran and
strided arrays and SciPy's tiny-sigma skip behavior. Independent review found
no concrete defect; concurrent cross-stream cache use remains unsupported.

Post-deployment minute rates127/286/281 during initial warmup, then380/410/
431/437 decisions/s. Latest30 one-second GPU5 readings:mean36.47%,median11%,
zero23.3%, compared with28.3%/3%/33.3% before deployment. These sequential
production windows are not matched trajectory trials. GPU is still not
saturated; the remaining work spans crop/history construction, live ownership
checks, per-path advancement, seed feeding and heading updates. Do not claim
that every remaining idle interval is fixed, or that another4x gain follows.

CPU pool scaling (approved runner `paris4-volume-pool-bench` and
`paris4-volume-pool-fine-bench`): 8 real 77³ CT cubes, repeated through
32/64/256-item batches, 3 timed trials after separate process startup/warmup.
Single-threaded NumPy/SciPy per process, CPU only; original tensor arithmetic,
every output bitwise checked. This isolates the dominant seed tensor stage,
not CT fetching, eigendecomposition or full tracing throughput.

At 256 items, fine-grained scheduling (one item per work packet) yields:
8 workers min/median/max 1.47274/1.47339/1.47517s (174 tensors/s);
16 workers .73808/.74214/.74483s (345/s);
32 workers .43744/.44998/.45273s (569/s);
64 workers .44159/.44651/.44663s (573/s).
At 64 items, median latency is .36814/.19698/.13252/.13627s respectively.
Thus 32 is the measured useful pool size here; 64 adds no meaningful gain.
The existing chunksize=n//32 prevents more than about 32 concurrent packets;
fine scheduling improves 32-worker 256-item throughput from 475 to 569/s.
No live pool or manager reservation changed by these CPU-only benchmarks.
Full results: profiles/seed-heading/pool[-fine]-benchmark.json under the run root.

Deployment: explicit --axis-workers 32, CPU manager request 40 (32 compute
workers plus main/tile/I/O allowance), GPU5 only as before. Seed tasks dispatch
individually, ordered map preserves filtering and seed order. 17 tests passed.

Further CPU measurement (`paris4-volume-geometry-bench`): 256 actual saved
paths, history/reference/path features, one core median46ms over5 trials.
Process pools 4/8/16/32 median17.57/11.32/9.95/7.40ms over3 trials, exact
arrays. Production now shares the existing axis pool for path_geometry_samples;
history and reference layout remain serial. This is deliberately a smaller
change than the benchmark, so the full benchmark gain is not claimed for it.
Spawned-process regression compares singleton/short/long paths exactly.
18 tests passed in4.72s. Live same-item serial/parallel path feature timings
and bitwise checks are scheduled once after reaching256 active rows.
Restarted before restoration finished to include both measured improvements;
no additional completed tiles were lost. In-flight tiles still restart by design.

Live production geometry check: 498 actual active rows, three alternating
same-input trials, serial62.469/63.032/63.798ms min/median/max versus
parallel25.924/27.561/27.682ms (2.29x), all outputs exactly equal. This measures
the deployed path-feature-only change, not the broader offline benchmark.
Reviewer found no blocking issue. Expanded regression to30 paths across
multiple process packets, with rotated frames and valid seeds;18 tests pass
in4.52s. Startup utilization is excluded from steady-state comparisons.

After warm-up, minute rates471/500/517/528/508/489/477 decisions/s versus
the earlier roughly380–440/s windows. Matched-path timings remain the
causal evidence: these live windows are different trajectories. GPU5 at
21:56 averaged63.33% (median83.5%, zero10%) over30 one-second readings.

Path advancement is now a measured12–13% of wall time. Approved
`paris4-volume-advance-bench` on256 actual saved-path continuations,3 trials,
including complete path serialization/return: single-core min/median/max
41.166/43.394/60.737ms;8-process8.492/8.619/11.552ms;
32-process6.704/7.253/8.016ms. Exact returned arrays/scalars/reasons.
Reuse axis_pool for accepted rows when no on_decision callback exists; apply
ordered results in the original serial loop. Callback mode retains its original
ordering. Independent review found no blocking defect. Full multi-step trace
regression covers changing active rows, confidence/bounds/loop/max-length stops,
and callback fallback;19 tests pass in7.32s. A first version of the test fixture
hit the model recovery gate before its intended bounds/loop cases; corrected
the fixture (not the acceptance policy) and reran successfully.

Longer utilization check after advancement parallelization:180 one-second
GPU5 readings averaged44.03%, median35.5%, zero17.2%; the earlier30-second
63.33% window was not sustained. Busy minute rates533/545/569/563/548/529
decisions/s; advancement now4% wall. Tiles subsequently completed and saved.
Heading inference remains11–15% wall, with a CPU wait after each64-row
forward. Queue those same forwards, combine small output transfers, preserve
64-row CPU frame arithmetic; CPU callers keep their original path.
20 CPU tests pass in7.71s, including130-row exact old/new heading comparison.
Independent review found no numerical/lifetime defect; pending outputs and
pinned copies scale with the caller's bounded512-row batch. Actual GPU exact
comparison and three alternating timing trials run inside the existing GPU5
grant; runner falls back to original scheduling if queueing is slower.
GPU validation:458 real paths (64-row chunks plus10-row tail) bit-identical.
Original heading seconds min/median/max=.057690/.057808/.085343;
queued=.056663/.057350/.058172, three alternating trials. Median gain0.8%
is too small to claim a meaningful speedup.20 tests pass in6.98s.
The first180-second post-start window averaged26.43% GPU utilization as
throughput rose293→388→503 decisions/s. The next180-second window averaged
32.48% and included draining the last paths of Z=4096. That band completed,
saved49.40% of total presence,934674 fibers/3265.48m retained. Z=4352 then
ran532→612 decisions/s. Source explicitly serializes Z bands, so overlapping
them would change duplicate-suppression ordering and is not an exact-output
optimization; it has not been implemented. A separate active-band utilization
window is being collected. ETA varies with band drain;13h is approximate,
not a guaranteed completion bound.
