# Auditing predictions against CT support

Surface predictions can contain *phantom* positives: voxels marked foreground
where the masked CT reads exactly 0, i.e. outside the scroll
([#1114](https://github.com/ScrollPrize/villa/issues/1114)). Preventing them at
inference time and masking them out of finalized artifacts are separate,
already-tracked problems. `vesuvius.audit_ct_support` answers the remaining
question: **for a prediction volume that already exists, how contaminated is it,
and is the contamination confined to the blend margin?**

Both modes are read-only. Point `--ct` at the CT pyramid level whose voxel grid
matches the prediction array; the tool refuses to run on mismatched grids rather
than silently comparing different resolutions.

## `chunks` — zero-download triage

Zarr does not write all-zero chunks, so the set of stored chunk keys is an exact
map of where a volume holds data. Listing keys for the predictions and for the
CT classifies every prediction chunk without fetching a voxel:

- `supported` — the chunk's voxel box overlaps a CT-bearing chunk;
- `halo_within_1_chunk` — it sits at most one prediction chunk away, the reach
  of a blend window that runs one chunk past the data;
- `beyond_blend_margin` — farther than that, which a blend-margin artifact
  cannot explain and points at a second mechanism. Up to 20 offending chunk
  indices are listed in `beyond_examples` for follow-up.

```bash
vesuvius.audit_ct_support chunks \
  --predictions s3://vesuvius-challenge-open-data/PHerc1203/representations/predictions/surfaces/20260319130212-surface-20260413222639-surface-m7-L2-th0.2.zarr/0 \
  --ct s3://vesuvius-challenge-open-data/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr/2 \
  --anon --output audit_pherc1203.json
```

```
prediction chunks: 7,830 | supported 0.9079 | one-chunk halo 0.0921 | beyond blend margin 0.0000 (0 chunks)
```

That run takes about ten seconds and transfers only key listings, which makes it
practical to sweep a whole batch of published volumes, or to confirm after a fix
that no phantom-bearing chunks remain.

## `voxels` — exact phantom fraction

Reads chunk-aligned z-slabs of both volumes in bounded-memory stripes along Y and
reports per-plane positives and phantoms plus the totals. Every plane inside a
sampled slab is measured exactly; `--slab_stride` controls how many slabs are
visited (every 12th by default). Slab reads use every transferred byte, whereas
sampling individual planes from a remote store pays chunk-depth amplification.

```bash
vesuvius.audit_ct_support voxels \
  --predictions preds.zarr/0 --ct ct.zarr/2 \
  --threshold 127 --slab_stride 12 --output survey.json
```

```
planes measured: 1,536 | positives 9,077,181,876 | phantom 3,936,984,632 (0.4337) | support 0.5663
```

The JSON report carries the per-plane rows, so contamination can be plotted
against z to see where along the scroll it concentrates.

## Options

| Flag | Modes | Meaning |
| --- | --- | --- |
| `--predictions` | both | prediction array (a single pyramid level) |
| `--ct` | both | masked CT array on the same voxel grid |
| `--output` | both | write the JSON report to this path |
| `--anon` | both | access object storage anonymously (public buckets) |
| `--threshold` | `voxels` | value above which a prediction voxel counts as foreground (default 127) |
| `--slab_stride` | `voxels` | measure every Nth chunk-aligned z-slab (default 12) |
| `--quiet` | `voxels` | suppress per-slab progress output |

The audit functions are importable directly (`audit_chunks`, `audit_voxels`,
`classify_chunks`) if you want the reports as dictionaries rather than JSON on
disk.
