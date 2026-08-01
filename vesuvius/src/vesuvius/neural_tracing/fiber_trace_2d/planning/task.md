# Vesuvius Python CI Compatibility

- The Vesuvius Python CI job runs from `vesuvius/`, but the fiber tracing tests
  import shared modules through the sibling top-level `lasagna` namespace.
- Keep the previously applied workflow import-path and trigger fix.
- Fix the Zarr 3.2.1 matrix failures in `/tmp/job-logs.txt` without breaking
  Zarr 2.18.7.
- Test fixtures that model v2 OME-Zarr input must create v2 metadata and
  slash-separated chunk keys through APIs supported by both Zarr versions.
- Runtime prefetch must generate the same store-relative chunk keys and read
  raw store bytes through either supported Zarr store API.
- Do not duplicate Lasagna helpers or add fallback implementations.
