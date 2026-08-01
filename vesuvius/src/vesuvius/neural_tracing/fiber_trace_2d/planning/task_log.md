# Vesuvius Python CI Compatibility

## Findings

- The workflow's default working directory is `vesuvius/`, and the test step
  does not add the checkout root to Python's module search path.
- Fiber 2D/3D modules deliberately share `omezarr_pyramid`,
  `normal_encoding`, and tiled inference code from the sibling `lasagna/`
  source tree. Collection therefore fails before any tests run.
- The original workflow path filter also omitted `lasagna/**`, despite that
  source being a test dependency; commit `eacce1a89` contains that workflow
  correction.
- The committed workflow fix resolves collection, after which the Zarr 3.2.1
  matrix reports 94 failures. Roughly 90 come from one fixture using the
  removed `Group.create_dataset`; three use `dimension_separator` while
  implicitly creating v3 arrays; one creates v3 input although the tested
  inference path deliberately consumes v2 `.zarray` metadata.
- Once fixture creation is repaired, prefetch exposes two runtime API gaps:
  Zarr 3 removes private `Array._chunk_key`, and stores use `get_sync()` plus a
  buffer result instead of mapping subscription.
- The local system Python cannot wake background asyncio loops from other
  threads. Zarr 3 therefore needs a periodic event-loop tick in the local test
  harness; GitHub's clean uv Python completed the original suite and does not
  show this host defect.

## Plan Review

- Supplying the monorepo root preserves the existing shared implementation and
  avoids copying private helpers into Vesuvius.
- Explicit v2 fixture creation preserves the format the tests intend to model;
  it does not convert production output to Zarr 3.
- Chunk-key encoding and raw-byte reads belong in one shared neural-tracing
  helper so Fiber and Fiber 2D do not grow separate version branches.
- Independent review was not used because delegation is prohibited unless the
  user explicitly requests subagents; the plan was reviewed locally instead.

## Implementation

- Added `lasagna/**` to the workflow trigger paths.
- Set the Vesuvius test step's `PYTHONPATH` to the GitHub checkout root, making
  the sibling namespace importable without changing runtime modules or
  duplicating helpers.
- Added shared test constructors for explicit v2 standalone/group arrays and
  migrated the three failing fixture families.
- Added shared runtime helpers for Zarr 2/3 store-relative chunk keys and raw
  store reads. Both prefetch paths use the key helper, and the older Fiber
  prefetch path uses the byte reader.
- Documented and specified the cross-version cache-key contract.

## Validation

- Zarr 2.18.7, all three modules named by CI: 509 passed, 2 skipped.
- Isolated Zarr 3.2.1, all three modules named by CI: 509 passed, 2 skipped.
- The complete test selection cannot collect in the current host environment
  because optional CI extras are absent (`fft_conv_pytorch`, `nnunetv2`, and
  `pytorch_optimizer`). The clean workflow installs them through
  `uv sync --extra all`.
- Python compilation of every changed production/test module: passed.
- `git diff --check`: passed.
