# Global fiber assignment with PyTorch SGD

Current direction passes: [H](h_direction.md) and [V consistent up](v_direction.md),
with whole-fiber breakage. H checkpoint`3f5b443e8`; V keeps that solution frozen.
[V results](v_direction_results.md):77newly disabled V, zero remaining parallel
direction conflicts,99.953354%Vfiber arc retained. Full H/global state preserved;
global U/V assignment remains PAUSED pending further constraints.

Current stage3: [H-fiber direction alignment with breakage](h_direction.md), following
normal signs and baked H/V. [Breakage sweep](h_direction_breakage_results.md)
replaces the [sign-only baseline](h_direction_results.md), which had0.047735%
weighted parallel disagreement with no additional rejections. Global U/V is PAUSED
pending further constraints; this direction stage has not yet been wired into it.

Paused experimental stage: [probabilistic global U/V layout](uv_layout.md),
following accepted inward normal and baked H/V solutions.100winding hypotheses
plus rejection; immutable constraint events, eligible1cmcuts and continuation.
See [global results](uv_layout_results.md) for full-graph diagnostics and limitations.

Initial data reconnaissance:2026-10-08 (historical details below).

Follow-up: shared BP code merged from `fiber-exp5`; see
[Lasagna constraint extraction overview](bp_constraints.md) for the reusable
pipeline, modes, merge resolution and validation limitations.

Full-volume reusable constraint export: see [database schema and commands](constraint_database.md).
This stores unsigned local events and both endpoint sheet-normal axes; source
geometry remains in the original AFV.

Second pass: [fiber-local signed normal sweep](constraint_database.md#second-pass-fiber-local-signed-normals-schema2)
stores signed endpoint normals plus a separate+1fiber sign. No cross-fiber
alignment or winding assignment is performed yet.

Third stage: [whole-fiber normal sign optimization](sign_resolution.md) adds
PyTorch sign/broken probabilities, perpendicular normal agreement and a weak
catalogue-umbilicus prior. Whole fibers remain intact; winding solving is separate.

Reciprocal-filter follow-up: [parallel removal and perpendicular iteration
comparison](parallel_reciprocal_results.md). The existing exported database is
the historical reciprocal-on baseline; these comparison runs do not overwrite it.

Follow-up correction: [moving-window perpendicular refinement](moving_projection_results.md)
removes the original seed-window restriction while retaining local support checks.

Current extraction semantics: [continuous winding measurements](raw_winding_results.md),
with median parallel estimates and winding-value processing filters disabled.
See [parallel/perpendicular rejection tables](relation_rejection_results.md)
for the current per-type outcome audit and separately unclassified seed failures.

## Local-event extraction benchmark (no fiber cutting)

The adapter selects source IDs by exact intersection with a spatial box, then
loads each selected fiber in full. It calls the shared native local-event
extractor with `localKeepWholeFibers=true`; endpoint certification is not relaxed.
It performs three unsigned extraction repetitions. **No normal alignment, dense
normal lattice, tracing, inference, winding assignment solve or PyTorch SGD runs.**
The sampler's `normalAlignedWindingDistance` name refers to integrating the
absolute normal projection, not to an additional sign-alignment operation.

Reusable build/test/benchmark commands, from this checkout:

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_benchmark_unsigned.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_benchmark_unsigned_scaling.json
```

The first command includes the regression suite and a 256-base-voxel full-XY
Z slab; the second measures thicker slabs. JSON summaries are under
`data/unsigned_z_slab_v1/`; command logs, binary/input hashes and peak RSS are under
`data/unsigned_z_slab_v1/runner/`. Configs are versionable and data outputs ignored.
Whole selected curves extend beyond the selecting slab. Scaling estimates use
their actual arc length, not total volume height divided by slab thickness.
The older `benchmark_v1` alignment experiment was stopped and is not this benchmark.

Recreate the compact aggregate after running the benchmarks:

```bash
/home/hendrik/.venv_las/bin/python global-fiber-sgd/summarize_benchmark.py --output global-fiber-sgd/benchmark_results.json
```

The extrapolation fields are explicitly conditional estimates, not measured
whole-volume runtimes. Missing cross-boundary neighbors can underestimate work.

## Working data

`data/Paris4_v1.afv` (ignored by Git), downloaded from
[the provided AFV](https://dl.ash2txt.org/community-uploads/qual/AutomatedFiberVolumes/Paris4_v1.afv).

- Download completed: 2026-10-08 09:02:00 UTC.
- Size: 412,086,272 bytes (393.00 MiB).
- SHA256: `05fa4606d6403d5e3b417afd1d6c1def17cb4c42b598f6b18b17fe58ce347504`.
- HTTP Last-Modified: 2026-10-05 18:14:53 GMT; ETag: `6ac3e91d-188ff000`.
- UUID: `b06d22a5-8628-4fd3-8b56-86bcfe506506`.
- Full embedded metadata: [metadata.json](metadata.json).

Download again, if needed (do not overwrite a file currently open in a viewer):

```bash
curl -fsSL https://dl.ash2txt.org/community-uploads/qual/AutomatedFiberVolumes/Paris4_v1.afv -o data/Paris4_v1.afv.part
sha256sum data/Paris4_v1.afv.part
# Verify the hash above before renaming the completed download.
mv data/Paris4_v1.afv.part data/Paris4_v1.afv
```

## Which scan?

**The April 2026 Paris4 2.4 micrometre scan, not the older 9 micrometre scan.**

- Coordinate space: `PHercParis4/20260411134726`.
- CT: `PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr`.
- Stored geometry: native L0 XYZ voxels, scale factor 1, 2.4 micrometres/voxel.
- Native shape ZYX: `[75784, 32693, 32693]`, independently confirmed against
  [the source level-0 metadata](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/0/.zarray).

The tracing input was substantially downsampled: provenance identifies prediction
level **4**, scale factor **16**, or **38.4 micrometres/prediction voxel**.
This does not change the stored geometry's native coordinate units.

Prediction prefix:
`PHercParis4/representations/predictions/fibers/20260411134726-fibers-20260915212757-L1/`.
The presence Zarr's `.zattrs` independently confirms scale 16 for level 4;
its shape is `[4737, 2044, 2044]`, chunks `[64,64,64]`, uint8, Blosc Zstd3.
The adjacent Lasagna manifest defaults to **level 3**, whereas this AFV explicitly
records **level 4**: do not silently substitute the manifest's default when
reproducing this tracing input. A GET of the prefix itself returns 404 because
it is not an object; S3 prefix listing and the actual metadata objects work.

## What AFV contains

This is **a collection of traced polylines, not a dense voxel prediction volume**.
The existing [AFV specification](../volume-cartographer/docs/fiber-collections.md)
and C++ `vc::fibers::FiberCollection` reader already support it.

- SQLite database, application ID `0x56434643` (`VCFC`), format version 1.
- JSON metadata and annotations; fiber catalog; binary geometry blocks; RTree.
- Geometry blocks: little-endian float64 XYZ triples, at most 257 points per
  block, with one shared endpoint between successive blocks of a fiber.
- 512,698 fibers, 5,265,142 unique stored points, 512,698 geometry blocks.
- All catalog families are `unknown`; no supplied H/V assignment.
- Mean length: 1,402.64 native voxels = 3.366 mm.
- Length range: 123.26–18,627.66 native voxels = 0.296–44.706 mm.

Embedded provenance describes bidirectional midpoint CUDA integration of
unoriented fiber-axis tensors, a 16-native-voxel integration step, CT exterior
cleanup and overlap-chain merging. Simplification tolerance was 8 native voxels
(19.2 micrometres). These are provenance claims, not independently reproduced
algorithm results. The collection explicitly warns that curves are estimated
trajectories, not unique physical fibers or validated merges.

For SGD this can provide initial candidate curves plus fast spatial lookup.
Presence/direction evidence would come from the separate prediction Zarrs; AFV
alone is not that evidence. Avoid treating overlap, proximity or the existing
merged trajectories as ground-truth identity. No dense prediction download or
optimization has been started in this first step.

## Validation

- HTTP byte count matched local file; SHA256 recorded (no Paris4 publisher hash
  sidecar was listed, so this is a local reproducibility hash).
- SQLite read-only inspection: `quick_check=ok`, `foreign_key_check=[]`, complete
  flag true, expected application ID and user version.
- Catalog counts matched metadata. All geometry blocks checked for valid point
  count, finite float64 coordinates, contiguous segment indices, shared endpoints,
  and reconstructed per-fiber point count. All passed.
- RTree row count: 512,698. No exhaustive spatial-index bounds audit performed.
- Source CT shape and prediction scale verified from their remote Zarr metadata.
- No software build or inference required; no runtime code changed.

## Branch and preserved work

`global-fiber-sgd` starts at fetched `origin/main`
`e0bbb8b40a2db58b1d71864f286eb85717e59e64`.
Previous uncommitted work from `fiber-skeleton-experiments` is preserved in stash
`fc24d1323af2504b00095944765820283788d739`, named
`fiber-skeleton-experiments before global-fiber-sgd 2026-10-08`.
It includes tracked changes and 91 untracked files. The unrelated untracked
`.pytest_tmp_private_hot/` remains untouched. Do not apply that stash onto this
new branch; restore it on the old experimental branch if needed.
