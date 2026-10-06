# Reviewed annotation training sets

From the villa checkout, run:

```sh
python3 vesuvius/scripts/collect_reviewed_fibers.py
```

The HTTP collector uses Python's standard library; normal baking also requires
the native `vc_fiber_bake_annotations` executable. Fetches the current directory listings and every
JSON under https://dl.ash2txt.org/other/annotators/ (no stale download cache).
Defaults use this host's original collection and train/test split under
`/ephemeral/hendrik/fiber3d_train/2026-09-09_x4_collection/`.
Output is a new timestamped `*_reviewed_annotators_*` directory under
`/ephemeral/hendrik/fiber3d_train/`. Override with `--output`, `--originals`,
`--split`, or `--source`; existing outputs are never overwritten.

Only files tagged `reviewed` are copied to `train/PHerc*/`. This includes
`reviewed` files also tagged `rework` or `tricky`, which remain tagged and are
reported. Do not use a recursive glob across the dataset root: `raw_reviewed/`
contains untouched downloaded copies for provenance, not training inputs.

Parent matching removes the crop's numeric prefix and `_runN` suffix, then
resolves within the original scroll folder. PHerc1299 maps to the legacy
PHerc1229 source folder. Exact source coordinate spaces must match. Missing
parents, unknown H/V, ambiguous train/test membership, held-out parents, invalid
JSON, and output-name collisions fail the whole publication without a partial
training dataset. No geometry-based label fallback exists.

The parent's effective `manual_tag` (if set), otherwise `automatic_tag`, is put
into the piece's **manual_tag**. Piece-local automatic scores remain intact.
This preserves the inherited effective H/V through VC3D recalculation; copying
parent geometric statistics into piece-local automatic fields would not.
`manifest.json` records both classifications, source URLs, parent paths and
source/parent/output checksums. Original geometry and CP corrections are unchanged.

## Dense normal baking

By default, width-annotated pieces receive `baked_annotations` schema v1:
one unit XYZ normal axis and validity flag per original dense line point,
coordinate identity, original manifest paths, resolved manifest contents and SHA256.
Missing/zero-validity source normals stay zero/invalid, never fabricated. Invalid
CP corrections or unresolved/ambiguous manifests fail the atomic publication.

The native executable calls the **same** `LasagnaNormalSampler` and
`fiberDisplayField` used by VC3D, including interpolated CP normal corrections.
It keeps native datasets/cache warm across pieces. Training reads the baked data;
it does not need the original Lasagna dataset or annotator filesystem paths.

`reviewed_normal_baking.json` maps recorded manifest basenames to explicit local
copies of those manifests and configures the existing manager cache location.
Change that mapping for another host with `--bake-config PATH`. Scroll and base
volume identity are checked. `--no-bake` explicitly produces the old labels-only
dataset, unsuitable for width/normal-head training on width-annotated pieces.
Build the native target in an already configured VC build:

```sh
cmake --build volume-cartographer/build/fiber-evaluation --target vc_fiber_bake_annotations -j 8
```

The collector writes every reviewed piece under `train/`; it does not create a
new train/test split. Training validation remains the established reviewed failure
set, configured independently.

Reruns create a fresh complete snapshot, including newly reviewed files and
excluding files whose review tag was removed. Earlier datasets are untouched.
Directory discovery and output ordering are deterministic; the remote is not
transactionally frozen, so recorded hashes identify the exact bytes fetched.

Tests: `python -m pytest -q vesuvius/tests/scripts/test_collect_reviewed_fibers.py`
# Bake an existing fixed test set

Use the same collector/native baker without downloading or creating any split:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa/vesuvius/scripts/collect_reviewed_fibers.py \
  --test-source /home/hendrik/data/2026-10-01_reviewed_failure_union_v1 \
  --output /ephemeral/hendrik/fiber3d_train/2026-10-05_reviewed_failure_union_baked_v1
```

The output must not already exist. This mode selects exactly the input manifest
files, checks hashes, requires full parents exclusively in the original **test**
split, and copies their effective H/V to each piece's manual_tag. Training mode
still requires train parents. No piece geometry, width or CP corrections change.
Missing coordinate IDs use explicit `test_coordinates` from the baking config;
existing IDs must agree and normal manifest sources must match the scan. Both
source and output hashes, parent provenance and valid-normal counts are retained.
Publication is atomic; originals remain untouched. Native baking uses the shared
VC3D normal sampler/display correction code, not a Python interpolation copy.

Prepared2026-10-05:32 pieces (11 PHerc0139,6 PHerc1299,15 Paris4),71,165 valid
point normals,23H/9V. Paris4 piece0050 inherited V instead of its local H estimate.
The two annotated training configs point dense/trace tests at this copy.
