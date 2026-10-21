# Reviewed annotation training sets

The HTTP collector uses Python's standard library. It fetches current directory
listings and JSON annotations from the configured source, selecting only pieces
tagged `reviewed` (including reviewed pieces also tagged `rework` or `tricky`).

From a villa checkout:

```sh
python vesuvius/scripts/collect_reviewed_fibers.py \
  --originals /data/fibers/all --split /data/fibers/existing-split \
  --output /data/fibers/reviewed-new --bake-config /data/fibers/baking.json
```

All paths are explicit. Existing outputs are never overwritten. Only `train/`
is training input; `raw_reviewed/` and source listings are provenance, not more
training examples. Collection does not generate or change a train/test split.

## Parent labels and provenance

Parent matching removes the crop's numeric prefix and `_runN` suffix and checks
the original scroll folder. PHerc1299 accepts the historical PHerc1229 folder;
Paris4 accepts its two original collection folders. The full parent's effective
manual H/V tag (otherwise its automatic tag) becomes the piece's manual tag.
Never estimate a parent's orientation from a short piece. Unknown H/V, missing
or ambiguous parents, changed coordinate identity, held-out parents and output
collisions fail publication. The existing split determines membership.

The manifest records source URLs, tags, original piece/parent/output SHA256,
both classifications, inherited labels and normal-source recovery. A missing
normal manifest is recoverable only when other reviewed pieces in the same
scroll unanimously identify the same normal artifact and coordinate space.
Explicit conflicting references are never replaced. Metadata/geometry and the
original inputs remain unchanged except for the recorded labels/baked fields.

## Dense normal baking

Build the existing native target in an already configured VC build:

```sh
cmake --build volume-cartographer/build/your-build --target vc_fiber_bake_annotations
```

A baking config provides `binary`, a `manifests` mapping from original manifest
basenames to relocated paths, optional `artifact_identities` and `cache_root`.
Relative binary/manifest paths resolve beside that config. The baker speaks one
JSON object per line on stdin/stdout. It reuses VC3D's shared
`LasagnaNormalSampler` and `fiberDisplayField`, including interpolated
control-point rotations; there is no Python copy of the interpolation.

Width-annotated pieces receive `baked_annotations` v1: a unit XYZ normal axis
and validity flag per original dense line point, coordinate identity, source
paths, resolved manifest content and SHA256. Missing/zero-validity normals remain
invalid. Training reads these baked fields without the original Lasagna inputs.
`--no-bake` explicitly collects labels only; `--audit-only` saves raw reviewed
pieces and reports unresolved provenance without accepting a training dataset.

## Preserve an existing test cohort

```sh
python vesuvius/scripts/collect_reviewed_fibers.py \
  --originals /data/fibers/all --split /data/fibers/existing-split \
  --test-source /data/fibers/fixed-test --output /data/fibers/fixed-test-baked \
  --bake-config /data/fibers/baking.json
```

The config's `test_coordinates` maps source names to exact scan coordinate
identities. This selects exactly the existing manifest entries, verifies their
hashes and full-parent test membership, inherits parent H/V and bakes normals.
No test selection, geometry, width or control-point corrections are changed.

Directory ordering is deterministic; the HTTP remote is not transactionally
frozen, so recorded hashes identify the exact downloaded bytes. Publication is
atomic after every selected piece succeeds. Prior datasets remain untouched.

Validation:

```sh
python -m pytest -q vesuvius/tests/scripts/test_collect_reviewed_fibers.py
```
