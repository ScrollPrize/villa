# Task log: portable cache manifests (2026-09-22)

- Reused authoritative source inference.lasagna.json and inference.json for all five scrolls, including source metadata for the Paris4 subset.
- Generated files in vc3d_channels/<scroll>/ with relative Zarr references. Verified all 15 group array shapes at scaledown=1 against exact source base dimensions.
- Added --export-manifests-only to approved failure-union runner; regular exports also generate manifests. No inference required.
- Documentation: docs/benchmarks/failure_union_vc3d_manifests.md.
- Repeated metadata-only export succeeded for all five volumes. Independent
  review confirmed portable bindings and identified the second generated-folder
  guard for Paris4 conversion; both paths now share the updated guard.
