# Task Log: Signed winding belief propagation

## Findings

- Existing constraint extraction stores only nonnegative normal-modulated
  winding magnitude because `LasagnaNormalSampler` uses `abs(dot)`.
- `fiber-lets-align` provides reusable axis BP primitives but the standalone CLI
  still owns crop sampling/compaction. That behavior must be extracted rather
  than copied into crop BP.
- Normal alignment fixes one deterministic gauge per connected valid lattice
  component; absolute physical outward/inward sign remains unobservable.
- Current soft H/V output is a posterior over discrete binary/ternary states.
  Winding factors do not depend on those states, so a separate categorical
  winding solver is equivalent to a joint Cartesian model and substantially
  smaller.
- The old unsigned winding field is consumed by parity MILP and OBJ diagnostics;
  changing its sign would be a hidden behavior regression. Signed BP evidence
  therefore needs its own optional field.
- Independent review found that independently gauged normal components cannot
  share signed evidence, every winding component needs a crop-central gauge,
  aligned lookup must be exact, and adaptive support must not stop at a small
  fixed radius. The revised plan fails on incomparable gauges, defines
  three-point nearest-lattice lookup, extracts shared topology preparation, and
  expands until resolved or an explicit resource guard fails.

## Deviations

- None.

## Implementation

- Extracted reusable manifest-backed lattice sampling and alignment into
  `sampleAndAlignLasagnaNormalLattice`; the standalone normal command and crop
  BP now share the implementation.
- Added stable aligned-normal component identities and exact nearest-lattice
  lookup. Constraint signing checks A, midpoint, and B and preserves the old
  unsigned winding magnitude.
- Extracted strict BP topology preparation from the H/V solver. Both H/V and
  winding inference now share piece, continuity, source geometry, crop-center,
  and normalized arc validation.
- Added a sparse continuous solve and adaptive categorical sum-product winding
  BP. Split pieces remain distinct variables joined by their zero-difference
  continuity factors. Gauges are selected independently per winding component;
  mixed normal-sign gauges fail explicitly.
- Integrated winding inference into both final cohort BP paths. Existing H/V
  output is unchanged; its CSV gains winding columns and receives a separate
  factor CSV plus integer MAP OBJ layers.
- Shortened BP artifacts to content-only names and added explicit H/V/error/tie
  partitions inside every integer winding layer.
- Preserved aggregate initialization compatibility by appending new optional
  signed fields after the existing `hardContinuity` member.

## Validation

- Core and crop command compile successfully in the existing build.
- `test_fiber_trace_winding_bp`: 5 cases passed under GCC Release and the
  Clang debug/system-dependency build, including parallel/serial equality.
- `test_lasagna_normal_alignment`: 8 cases passed.
- `test_fiberlet_crop_trace`: 74 cases passed after retaining the legacy
  validation order and aggregate field layout.
- Release targets `vc_fiber_trace_chunk` and `vc_lasagna_normal_align` built
  with `-j32`. Focused CTest ran all three binaries in 0.74 seconds with no
  failures. `git diff --check` passed.
- A real 384-base crop smoke run used 17,576 aligned normal samples in one
  component and 3,259 selected constraints over 192 pieces. Continuous plus
  integer winding inference converged, expanded support in six rounds, used 32
  workers, produced labels from -2 through 2, and wrote the augmented piece
  CSV, factor CSV, and five integer winding OBJ layers under
  `/tmp/winding-bp-smoke`.
- A follow-up 384-base smoke run wrote short current-result names and the full
  H/V/error/tie cross partition. Published winding layers are normalized to
  consecutive nonnegative indices while the CSV retains the solver-relative
  label.
