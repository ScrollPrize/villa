# Plan: Native 3D Trace2CP GPU-Centric Beam Acceleration

## Implementation

1. Reconfirm the current baseline.
   - Run the same approved whole-fiber Trace2CP benchmark command.
   - Record metric quality, total trace wall/CPU time, and top profile rows in
     `task_log.md`.

2. Batch field lookup work.
   - Keep query points as torch tensors.
   - Group the requested points by inference block once per query batch.
   - Stack the unique block tensors and sample all unique blocks with one
     batched `grid_sample` call instead of looping one `grid_sample` per block.
   - Sample output channels and valid-mask channel together.
   - Decode direction/presence once for the full sampled query batch.
   - Benchmark immediately and keep the change only if quality is unchanged.

3. Reduce beam-loop Python rebuilds and synchronizations.
   - Keep the live beam state in tensors across committed beam expansions.
   - Reconstruct Python `NativeTraceResult` nodes only for the final selected
     path or explicit failure path.
   - Avoid per-expansion CPU scalar conversions where the result can stay on
     device until pruning/termination.
   - Benchmark immediately and verify the restart count remains unchanged.

4. Move cached-block routing toward a GPU-resident path.
   - Add a GPU-side cache index for already inferred blocks so point lookup can
     map point coordinates to block slots without NumPy grouping when all
     requested blocks are resident.
   - Fall back to CPU block discovery only when a query references a missing
     block, then infer/store the missing blocks and retry the GPU path.
   - Preserve the existing cache size accounting and deterministic block
     origins.
   - Benchmark immediately.

5. Fuse broader trace work where it remains profitable.
   - After the previous steps, use the profiler to identify any remaining
     candidate-scoring substage that still launches many small ops.
   - Batch/fuse only the measured hot pieces, preserving the scoring formula and
     beam semantics.

## Spec Update

- Native 3D Trace2CP hot-path sampling should keep inferred field blocks
  device-resident and should not route resident point lookups through NumPy.
- Field lookup should batch unique-block `grid_sample` work and decode sampled
  choices once per query batch.
- Beam tracing should keep beam state tensorized across expansions where
  possible and only materialize Python result nodes for outputs.

## Docs Updates

- Update `specs.md` with the GPU-centric lookup/beam requirements.
- Update `task_log.md` with each benchmark command/result.
- Add a changelog note if the final implementation produces a sustained speed
  improvement.

## Tests

- Run focused Trace2CP unit tests after each code step.
- Run `git diff --check`.
- Run the approved whole-fiber benchmark after each implementation step and
  compare restarts plus profile timing.
