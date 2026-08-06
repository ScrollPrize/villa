# Plan: grouped volume listing table

## Implementation

1. Replace per-record `_print_volume` output in the human `volume ls` path with
   one table renderer over the complete filtered record list.
2. Use columns `SCROLL`, `VOLUME`, `SHAPE`, `VOXEL`, `FORMAT`,
   `PREFETCHED`, and `ORIGINS`. Compute deterministic widths from headers and rendered values,
   separate columns with two spaces, and avoid terminal-width-dependent output.
3. Group the already deterministically sorted records by `sample_id`. Print the
   scroll and first volume on the same row, then put `├─`/`└─` in the `SCROLL`
   column for additional volumes while keeping long names in `VOLUME`. A
   one-volume group has no branch marker. Omit the redundant catalog `ID` column because `long_id` begins with
   the same ID; the JSON record continues to retain both fields.
   The renderer sorts defensively by `(sample_id, long_id)` so direct callers
   cannot produce repeated interleaved groups. An empty result prints the
   header and separator only.
4. Preserve unknown values as `-`, origin deduplication/sorting, filtering, and
   `--json` output exactly. Do not introduce a table-formatting dependency.
5. Populate `PREFETCHED` from numeric local OME group directories that contain
   `.zarray` plus at least one non-metadata chunk file. Do not treat root
   `.zattrs` advertisements or empty group metadata as downloaded data. Probe
   lazily and stop at the first chunk per group so listing does not traverse
   complete volumes.
6. Render Unicode branches when the output encoding supports them and use
   deterministic ASCII `|-`/`\-` branches otherwise. Do not inspect terminal
   width, truncate, colorize, or otherwise make redirected output unstable.
7. Render three-component catalog shapes in existing depth/height/width order
   as `D×H×W`, right-aligning depth to 6 characters and height/width to 5 with
   spaces. Preserve `-` for unknown shape and use the generic representation
   for non-3D future records.

## Tests

- Assert exact headers, alignment, sample grouping, branch markers, unknown
  shape rendering, deterministic order, and single-record groups.
- Assert prefetched scales are numerically ordered, metadata-only groups are
  excluded, chunked groups are included, and absent cache roots render `-`.
- Assert header-only empty output, interleaved-input sorting, and ASCII fallback.
- Assert fixed 6/5/5 space padding for 3D depth/height/width components.
- Assert filtered output and `--json` remain valid.
- Run focused manager tests, the real cached `volume ls`, Python compilation,
  and `git diff --check`.

## Spec update

Specify the grouped human table and unchanged JSON contract.

## Docs updates

Add a compact output example to `lasagna/docs/manager.md` and note `--json` for
machine consumers.

## Changelog

Add a dated entry for the grouped volume table.
