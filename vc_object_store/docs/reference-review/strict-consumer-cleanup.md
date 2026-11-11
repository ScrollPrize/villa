# Strict consumer reference cleanup

This follow-up supersedes the compatibility choices recorded in the earlier review: unsupported saved partial identities now require explicit regeneration instead of reconstruction or weaker comparison.

Implemented consumer changes:

- `InputPins` rejects numeric `objects` maps and selected entries without complete references. Position-only requests remain valid discovery requests; freshly selected records still capture their actual owners.
- Saved object, outer-shell and relationship manifest entries require complete references before Python digest/comparison/resume processing. The allow-changed-inputs option cannot authorize missing identity. Removed the geometry-only legacy resume comparison and obsolete relationship-generation suppression.
- C++ Spiral input reads validate every saved relationship at the load boundary, including dropped and out-of-scope entries. Decoded `RelationshipVersion` values are carried into comparison rather than parsing again. Missing evidence reports regeneration instead of weaker equality.
- Link requests use command 3 with complete object references in `read_dependencies`; no numeric/full sidechannel is produced.
- Project umbilici require `{reference, points_zyx}`. Removed `legacy_source`, optional identity and flattened-field reconstruction. Complete references accompany materialized point files and source metadata unchanged.

Content digests remain application projections: complete publication references identify inputs, while geometry and consumed content determine whether content changed. Current UUID discovery is not treated as an invalid historical reference.

Compatibility limitation: old numeric input pins, incomplete manifests/checkpoints and flattened/ownerless project-umbilicus requests must be regenerated from an explicitly selected source. No historical owner is inferred from the current attachment. The shared test project fixture and targeted tests now use canonical reference-bearing input.

Validation: source edits and targeted regression updates only; this agent ran no tests or builds. The parent implementation owns the fixed approved runner executions and the shared `relationship_version_from_json` binding. `git diff --check` passed for these consumer files. Earlier test results do not validate this stricter pass.

Parent integration validation is now complete and recorded in [legacy removal](legacy-removal.md#validation-and-size), including migrated headless/service/Lasagna fixtures, consumer runtime/syntax checks and the three unchanged extended-service failures. The paragraph above describes the independent agent's source-only lane, not the final integrated validation.
