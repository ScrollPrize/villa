# Plan: portable cache manifests

Reuse original inference manifests and provenance. Validate relative channel paths, exact base dimensions and array shapes. Use Paris4 subset source metadata with local subset channels. Integrate into normal export and expose metadata-only refresh through the approved runner. Preserve fiber filenames.

## Spec update

The exported channel view includes portable inference.lasagna.json and inference.json in each scroll directory.

## Docs updates

Document use and regeneration in docs/benchmarks/failure_union_vc3d_manifests.md; record results and durable changelog.

## Validation

Generate all five files using approved runner; validate 15 array bindings; rerun metadata export to verify repeatability. Independent review covers portability and rerun behavior.
