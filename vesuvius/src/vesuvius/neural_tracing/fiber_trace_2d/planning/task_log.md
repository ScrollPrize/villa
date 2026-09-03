# Task Log

## 2026-09-03

- The first staged crop artifact was capped by `--max-fibers 2000` and left 65
  candidates. It is not the requested complete benchmark cohort.
- Started a new immutable artifact at
  `data/workdir3/fiber-crop-1024-staged-full/crop_traces.zarr` with the same
  three filter stages and no fiber or attempt limit.
- Independent review required explicit scientific defaults, distinct cohort
  labeling, durable hashes, and disclosure that the source is an unverified
  `build_state=partial` local mirror. These corrections were incorporated.
- The uncapped trace completed all 26,640 candidates: 2,062 attempted and
  accepted, 24,578 covered, and zero remaining. Its 512/343/27 stage aggregate
  Fiberlet counts were `5903905 -> 4011983`, `2082011 -> 1292725`, and
  `632703 -> 448512`.
- The staged oracle-pruning benchmark started with 1,450 pieces and 78,925
  unique constraints. It removed 308 pieces; 45,996 constraints were
  problematic versus 32,929 retained fulfilled (`139.68%`). The final solve
  converged with 1,134 active pieces, zero retained sign conflicts, and 24
  exact / 0 wrong / 2 missing reference windings. Two runs took 79.33 and
  83.34 seconds.
- The staged reference replay evaluated all 48 directed cases and 101.036 mm.
  It found 7 failures in 6 cases: 14.434 mm per failure and 14.286% distance
  per failure. The full command, including preprocessing, took 243.75 seconds
  and 8,366,164 KiB peak RSS.
- Added separate reproducibility records and distinct staged-uncapped points
  to the benchmark result index and plot data. The older pruning point remains
  labeled as capped and unstaged; no causal stage-filtering claim is made.
- Regenerated both benchmark SVGs, validated the plot-data JSON, and passed all
  41 Fiberlet storage and 81 crop-trace focused test cases. No transient stage
  directories remained after either benchmark command.
