# Task Log: restore baseline fiberlet search around weighted lookahead

## Initial findings

- The current implementation changed ranking semantics beyond the requested
  forward integration: it drops all route cost before the current checkpoint.
  The old scorer ranked cumulative cost from the segment seed through the common
  horizon.
- The current `W=1` path repeatedly partitions and rescans independently
  quantized segment densities. Small codec/FP differences are acceptable, but
  the repeated work and large observed route changes are not.
- Exact search removed the prior relaxed cost-to-go DP. Its replacement uses
  zero future cost and repeatedly calls full interval scoring while queueing
  states, causing weak pruning and route-history/profile rescans.
- The previous 5k `W=1` result was a new decoded-profile control, not an actual
  pre-change baseline. It cannot establish regression-free quality or speed.
- The repair will preserve old prefix/search semantics and change only
  checkpoint-forward integration.
- Independent review required exact half-open checkpoint ownership, a pinned
  baseline SHA, a formally conservative
  later-start distance-bin heuristic, deterministic all-or-zero on-demand graph
  enumeration, bounded per-decision scalar memoization, exact preservation of
  bounded-search label semantics, concrete incremental state, and quantitative
  hot-cache acceptance. The plan now includes all of these.

## User clarifications

- Decoded subsegment costs are already additive and must be used directly.
  Do not rescale them to the separately stored whole-edge cost and do not
  synthesize a replacement profile.
- `W=1` is not a compatibility branch. The same linear integration-grid
  algorithm applies for every weight, and `--cost-step` remains active.
  Interpolation and accumulation may cause small differences, but material
  route or failure-count changes indicate a defect or an extreme near tie.

## Implementation

- Restored ranking as an authoritative unweighted prefix from the segment seed
  through the checkpoint plus decoded-profile integration from the checkpoint
  through the common horizon. Edge and join ownership is half open at the
  checkpoint and horizon.
- Removed the remaining profile correction in graph construction. Segment
  density is now each stored segment cost divided by its segment length; no
  aggregate whole-edge scale factor is applied.
- One linear boundary-walking integrator handles all weights and integration
  spacings, including `W=1`. The scorer carries prefix-edge, prefix-join,
  forward-edge, and forward-join values incrementally.
- Restored decision-local relaxed cost-to-go memoization for exact search and
  exposed memo state, hit, and zero-fallback counts in decision diagnostics.
- Split lightweight cached graph adjacency from explicit route-profile access.
  A bounded 4096-entry decision-local profile cache is shared by initialization,
  successor scoring, and relaxed bounds. This removed route reconstruction and
  profile copying from ordinary arc lookup.
- Added focused tests proving that decoded profile costs are not corrected to a
  conflicting aggregate edge total and that checkpoint advancement retains
  different prefix costs when future costs are equal.

## Validation

Build command:

```bash
cmake --build volume-cartographer/build --target vc_fiberlets test_fiberlet_paths test_fiber_replay test_fiberlet_storage -j32
```

Focused executable results:

- `test_fiber_replay`: 12 test cases passed.
- `test_fiberlet_storage`: 17 test cases passed.
- `test_fiberlet_paths`: the two added scorer cases pass. The executable still
  reports the same 298 existing checks as before this repair: legacy bitwise
  fixtures at line 414 and three pre-existing Q4 expectations at lines
  1026-1028. No new failure was introduced.
- A hot-cache 5k replay with 1 thread and another with 32 threads produced
  identical serialized fiberlet segments and search-decision diagnostics, with
  zero fiberlet failures in both.

Representative candidate command, with the populated cache paths omitted here
only because they are temporary run artifacts:

```bash
volume-cartographer/build/bin/vc_fiberlets fiberlet-replay /home/hendrik/business/aiconsulting/vesuviuschallenge/data/s1/PHercParis4.volpkg/volumes/fiber_s1_002.lasagna.json /home/hendrik/business/aiconsulting/vesuviuschallenge/data/fibers/david/Paris4_fibers/dj_20260805T025256484_000003.json /tmp/fiberlet-score-repair-5k --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json --threads 32 --length 5000 --cost-weight 1 --cost-step 16
```

The immutable baseline was revision
`64e534183d6ee4a9c0c09aa08f046463464ab7fb`, built in a separate Release
worktree. Five serial hot-cache repetitions on the same 5,000-base-voxel
Paris4 interval produced:

| Mode | Wall mean/p50/p95 | CPU mean/p50/p95 | Peak RSS | Fiberlet failures |
| --- | --- | --- | --- | --- |
| Pinned baseline | 0.558/0.550/0.640 s | 15.684/15.480/18.550 s | 96.4 MiB | 0 |
| Repaired W=1, step 16 | 0.666/0.650/0.800 s | 18.992/18.800/23.260 s | 97.6 MiB | 0 |
| W=0.99, delay 0, step 16 | 0.642/0.630/0.710 s | 18.584/18.410/20.800 s | 97.1 MiB | 0 |
| W=0.99, delay 192, step 16 | 0.616/0.610/0.650 s | 17.076/17.430/17.800 s | 97.4 MiB | 0 |

The repaired W=1 p50 wall and CPU ratios are 1.18x and 1.21x the pinned
baseline, within the 1.25x acceptance target. A diagnostic run recorded 103
decisions, 11,761 generated states, 645 expanded states, 1,273 cost-pruned
states, 4,101 relaxed-bound states, 15,141 relaxed-bound hits, and zero
relaxed-bound fallbacks. W=1 step 8 and step 32 selected the same canonical
55-edge route on this interval.

An earlier four-process concurrent timing experiment was discarded because CPU
oversubscription made its wall and CPU numbers invalid. No profiler was run:
`perf` was unavailable in this environment, so the leading CPU-function report
from the full plan remains outstanding. The representative longer regression
corridor also remains outstanding and is kept unchecked in `status.md`.

## Long-route regression found after 5k validation

- A user run slowed sharply near 65 percent despite fixed checkpoint and
  lookahead distances.
- `scorePersistentRouteForDecision()` materialized `persistentRouteHistory()`
  from the segment seed for every retained beam at every checkpoint. Its work
  therefore grew linearly with the committed prefix and quadratically over the
  complete trace.
- `PersistentLogicalRouteRegistry::pruneExpired()` also scanned every interned
  node after every checkpoint. Permanently live nodes along the committed route
  made this another quadratic prefix-dependent path.
- The repair must initialize prefix score from cumulative history scalars and
  visit only the checkpoint-to-horizon suffix. Registry cleanup must use a
  bounded incremental cursor rather than a full-table sweep.

## Long-route repair and validation

- Score initialization now walks backward only to the edge crossing the current
  checkpoint. It initializes the earlier prefix from cumulative scalar edge and
  join costs and integrates that bounded suffix forward through the horizon.
- Logical-route cleanup now retains a stable ordered-map cursor and inspects at
  most 4,096 entries per checkpoint. Replacing an expired interning entry keeps
  the map node in place, so insertion and replacement cannot invalidate that
  cursor.
- A 96-edge linear regression records every decision and verifies score
  initialization visits at most five history nodes for a four-voxel lookahead;
  cleanup work remains within its fixed budget.
- The replay CLI previously assigned all 32 requested workers independently to
  both concurrent evaluators. A hot-cache sweep exposed a sharp SMT cliff:
  `--threads 1` took 0.37 seconds while `--threads 32` took 8.2 seconds on the
  16-core/32-thread Ryzen 5950X. The CLI now divides the requested process
  budget between the greedy and fiberlet evaluators.
- After both repairs, the same hot-cache 5k command at `--threads 32` completed
  in 0.39 seconds wall time with 1.88 seconds user, 0.58 seconds system, 92.8
  MiB peak RSS, one greedy failure, and zero fiberlet failures.
- The representative full 46,148-base-voxel interval reused the existing
  radius-768 anchor and fiberlet caches. It reached 64.7 percent at 11 seconds,
  78.5 percent at 14 seconds, and completed in 22.34 seconds wall time with
  41.27 seconds user, 4.97 seconds system, and 1.19 GiB peak RSS. It reported 14
  greedy failures and 5 fiberlet failures. The former run had reached only 65.1
  percent after 8m53s with a 37m58s current-speed ETA.

Full validation command:

```bash
/usr/bin/time -f 'wall=%e user=%U sys=%S maxrss_kib=%M' volume-cartographer/build/bin/vc_fiberlets fiberlet-replay /home/hendrik/business/aiconsulting/vesuviuschallenge/data/s1/PHercParis4.volpkg/volumes/fiber_s1_002.lasagna.json /home/hendrik/business/aiconsulting/vesuviuschallenge/data/fibers/david/Paris4_fibers/dj_20260805T025256484_000003.json /tmp/fiberlet-prefix-fix-full-r768 --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json --threads 32 --radius 768 --anchor-cache /home/hendrik/business/aiconsulting/vesuviuschallenge/data/workdir3/fiberlet-replay-full/cache/fnv1a64-28da47830cd793ba/anchors.zarr --fiberlet-cache /home/hendrik/business/aiconsulting/vesuviuschallenge/data/workdir3/fiberlet-replay-full/cache/fnv1a64-92a6fb9b1512f01b/fiberlets.zarr
```

## W=1 quality-regression diagnosis

- The pinned aggregate-cost baseline has two full-corridor fiberlet failures at
  reference arcs `42747.297592184739` and `44748.20886432947`; direct decoded
  profile scoring at `W=1` has five failures.
- Decision diagnostics locate the first geometry divergence at checkpoint six.
  At checkpoint zero both implementations commit the same prefix, but profile
  integration changes candidate scores by roughly `0.15` to `0.34`, far beyond
  uint16 codec or floating-point error. A route needed by the aggregate scorer
  at the next horizon falls outside the profile-ranked retained set, and the
  profile scorer then commits a longer one-fiberlet shortcut.
- Increasing the retained beam from 16 to 64 does not change that first wrong
  committed prefix. This is not ordinary beam-width truncation.
- A temporary diagnostic build kept the repaired persistent search and cache
  path unchanged but replaced all `W=1` profile integration with uniform
  whole-edge density. It reproduced the pinned baseline's exact two failure
  arcs (`20.84 s` wall, `38.48 s` user, `5.70 s` system, `1,223,900 KiB` peak
  RSS).
- A second temporary build used decoded profiles for every complete fiberlet
  and uniform whole-edge density only for checkpoint- or horizon-cut partial
  fiberlets. It again reproduced the exact two baseline failure arcs (`21.18
  s` wall, `38.99 s` user, `5.45 s` system, `1,269,548 KiB` peak RSS).
- Therefore the complete decoded profiles, cost codec, persistent checkpoint
  state, pruning, and cached graph reconstruction are not responsible. The
  regression is caused specifically by raw subsegment cost distribution on
  partial boundary fiberlets. At `W=1`, a terminal fiberlet with a cheap prefix
  and expensive tail beyond the common horizon is ranked substantially better
  than the aggregate baseline, effectively shortening and destabilizing the
  usable lookahead.
- Both diagnostic substitutions were removed after measurement and the normal
  binary was rebuilt from the committed source. A production boundary policy
  remains to be selected; silently retaining uniform partial-edge scoring would
  conflict with the requested subsegment-resolved integration semantics.
