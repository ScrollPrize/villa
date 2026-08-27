# Plan: BP-aligned Lasagna normal samples

## Scope and contracts

- Operate only on regular Lasagna normal manifests through
  `LasagnaDataset`/`LasagnaNormalSampler`; do not route through
  `NormalGridVolume` or `vc_ngrids`.
- Align a finite set of sampled axes. Do not rewrite the source Lasagna Zarr or
  imply whole-volume persistence in this first standalone diagnostic.
- Preserve current fiber BP results while extracting its binary sum-product
  message update into a shared solver: factor/message order, probabilities,
  log odds, iteration count, residual, convergence, and disconnected-component
  gauge behavior must remain unchanged.
- Treat the absolute sign as gauge freedom. Fix one deterministic node per
  connected component and align all other nodes relative to it.

## Implementation

1. Extract a reusable binary pairwise sum-product solver with normalized
   log-odds messages, damping, residual convergence, arbitrary fixed binary
   nodes, and per-node state probabilities. Port current binary fiber
   sum-product BP to this solver without changing graph construction or report
   semantics.
2. Add reusable Lasagna normal alignment code:
   - accept finite base-voxel positions and normalized ambiguous normal axes;
   - build deterministic local lattice-neighbor factors;
   - clamp signed normal dot `d` to `[-1,1]`, then use
     `sameCost=(1-d)/2` and `differentCost=(1+d)/2`; exact `d=0` links are
     neutral and omitted, while temperature remains solely a BP setting;
   - omit exactly neutral factors, identify components, fix one deterministic
     gauge node per component, solve, and flip normals whose posterior favors
     the opposite sign;
   - return original/aligned normals, flip probabilities, topology, and BP
     convergence diagnostics.
3. Add `vc_lasagna_normal_align`:
   - positional Lasagna manifest plus required base-voxel `--bbox` and
     `--output` basename;
   - optional remote cache, decoded-cache size, worker count, lattice spacing,
     neighborhood radius, and shared BP controls;
   - validate the regular normal schema and open with `workingToBaseScale=1`;
   - interpret `--bbox` as finite half-open base-voxel XYZ bounds and reject
     bounds outside the declared base shape or with no lattice points;
   - default lattice spacing from the equal physical Lasagna `nx/ny` channel
     spacing in base voxels;
   - globally anchor each coordinate as integer index `ceil(min/spacing)`
     through the last index whose product is strictly below `max`, so
     overlapping bboxes sample identical points;
   - batch sample through `LasagnaNormalSampler`, then compact invalid samples
     stably while retaining their original lattice indices.
   - connect every retained pair within the configured positive Chebyshev
     lattice radius exactly once, with clipping at bbox boundaries and no
     distance weighting. The default radius is one (26-neighborhood).
4. Write `<base>_unaligned.obj` and `<base>_aligned.obj` from the exact same
   positions. Each glyph contains two short undirected perpendicular base
   strokes and one longer center-to-normal directed stroke, grouped separately
   in OBJ.
5. Print sample/factor/component/flip/convergence counts and output paths.
   Exact posterior ties do not flip. A finite `message_limit` result is emitted
   and reported rather than discarded; malformed input remains a hard failure.

The standalone spatial graph can connect nearby parallel sheets and currently
treats every `grad_mag > 0` sample as valid because the normal sampler does not
expose magnitude confidence. These are explicit diagnostic limitations. The
later H/V integration should provide its own topology/evidence to the shared
alignment/BP APIs rather than inheriting this lattice graph blindly.

## Spec update

Add a Lasagna-normal sign-alignment contract: regular manifest input, globally
anchored sampling, full signed-dot pair evidence, deterministic per-component
gauge fixing, shared binary BP implementation, no source-Zarr rewrite, and
paired exact-sample OBJ diagnostics.

## Docs updates

Add a standalone command example and explain sampling coordinates, gauge
freedom, factor meaning, outputs, and how the reusable result is intended for
later H/V optimization.

## Testing

- Generic binary BP: exact small trees, fixed-node gauges, disconnected
  components, convergence, and malformed input rejection.
- Normal alignment: alternating signs become mutually aligned; weak/neutral
  links do not invent evidence; disconnected components are independently
  gauged; input positions remain unchanged.
- OBJ: each retained sample emits exactly two base lines and one directed line,
  and aligned/unaligned files use identical centers.
- Existing fiber BP focused tests remain green after solver extraction.
- Add a minimal regular Lasagna manifest fixture integration test covering
  schema validation, source/base scale, batch sampling, stable invalid-sample
  compaction, default spacing, and paired OBJ output.
- Verify half-open XYZ bounds, out-of-range and empty bounds, overlapping-bbox
  lattice identity, exact factor equations, 26-neighborhood de-duplication,
  posterior ties, and finite nonconverged output.
- Build `vc_lasagna_normal_align`, `vc_fiber_trace_chunk`, and focused tests;
  run `git diff --check`.

## Changelog

Record reusable BP sign alignment and standalone Lasagna normal visualization.

## Parallel BP follow-up

1. Add an explicit worker count to the reusable binary BP configuration and
   pass the standalone command's existing `--threads` value to both Lasagna
   sampling and BP. Preserve one worker for existing callers unless they opt
   in, append the field to preserve aggregate initialization, cap it to the
   OpenMP/runtime and useful graph limits, and bypass OpenMP overhead below a
   documented graph-size threshold. Report the effective worker count.
2. Build a deterministic CSR incident-factor index once per solve. Store each
   node's incident factors in original factor order, then compute node totals
   independently so parallel execution retains the serial summation order.
3. Run message iterations inside one persistent OpenMP region:
   - static parallel node-total evaluation;
   - static parallel factor-message updates into disjoint output slots;
   - max reduction for residual;
   - one synchronized message-buffer swap and convergence decision.
   Retain barriers after totals, after updates/reduction, and after the
   single-threaded swap/stop decision; no dependent phase may use `nowait`.
   Parallelize final totals and posterior conversion through the same ordered
   node representation.
4. Add parity tests comparing serial and parallel reports, including fixed and
   disconnected nodes, exact iteration/residual status, log odds, and
   probabilities. Retain existing fiber BP regression coverage.
5. Add separate BP setup, iterative-solve, and total timing to
   `vc_lasagna_normal_align` output. Measure a fixed large real-manifest bbox
   in GCC Release after a warmup and at least five times with one worker and
   the selected parallel count. Record compiler, command/input, node/factor/
   iteration counts, min/median/max, total CLI time, and whether the requested
   0.5--1.0 second BP target is reached.

### Spec update

Document deterministic opt-in BP parallelism: node sums retain factor order,
message updates remain synchronous Jacobi iterations, and worker count changes
execution only, not graph or inference semantics.

### Docs updates

Document that `--threads` controls both regular Lasagna batch sampling and BP,
and report standalone BP timing for performance diagnosis.

### Testing and changelog

Build the Release CLI and focused Clang/GCC tests. The GCC parity case must be
above the serial threshold and assert an effective worker count greater than
one; the Clang OpenMP-shim case must assert serial fallback. Run exact
serial-versus-parallel report parity plus the existing fiber BP suite, run
`git diff --check`, and record profiler evidence and measured speedup in the
changelog/task log.
