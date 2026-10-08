# Global fiber assignment with PyTorch SGD

Initial data reconnaissance, 2026-10-08. No optimizer implemented yet.

Follow-up: shared BP code merged from `fiber-exp5`; see
[Lasagna constraint extraction overview](bp_constraints.md) for the reusable
pipeline, modes, merge resolution and validation limitations.

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
