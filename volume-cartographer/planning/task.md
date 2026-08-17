# Direct Zarr mirror disk cache

Change persistent remote-volume caching so a newly encountered remote Zarr is
cached as an incomplete but otherwise exact local mirror of the source Zarr.

Requirements:

- Select the legacy cache layout whenever an existing cache directory contains
  the legacy cache footprint, including for a remote volume newly added to a
  project.
- Keep legacy per-chunk reading and writing so existing mixed `.bin`, `.zst`,
  `.c3d`, `.source`, and `.empty` caches remain usable.
- Select the direct-mirror layout for a remote volume with no legacy footprint.
- Mirror required Zarr metadata and store every downloaded encoded chunk at its
  exact source-relative Zarr object key. The resulting directory must be
  readable as an incomplete native Zarr volume.
- For sharded arrays, download, deduplicate, persist, and account the complete
  outer shard object while decoding only requested logical inner chunks.
- Continue writing missing-chunk markers as an adjacent `<chunk-key>.empty`
  file; native Zarr readers must continue to see the original chunk key as
  absent and ignore the marker.
- Ordinary remote reads, Open Data prefill, and cache redownload must all use
  the selected layout and the shared source scheduler.
- Remove VC3D options and production paths that recompress decoded cache data.
  Keep legacy compressed-cache decoding, but do not create new recompressed
  cache entries.
- Preserve exact downloaded bytes and existing rendering values.
