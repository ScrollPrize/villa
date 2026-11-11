# Consumer duplication cleanup

Baseline: `43a2b576b`. Remove obsolete identity branches and duplicated reference/relationship policy rather than preserve competing implementations. Keep application configuration, content digests, draft/mapping/membership guards and intentional external API boundaries. No schema or solver changes.

- [x] Remove `legacy_source` consumption from SpiralPanel and Spiral service commit context; only the producer's complete `reference` is valid.
- [x] Consolidate manifest reference validation and retain decoded handles instead of decoding the same entries repeatedly in pin construction and resume comparison.
- [x] Share relationship eligibility and endpoint ordering between native Spiral input comparison and Python input materialization. Python wraps `fittingRelationshipInput` through the batch binding; neither consumer has its own family table or endpoint-ordering policy. The fitting vocabulary is separate from store identity/routing.
- [x] Remove redundant numeric/content identity checks where the complete relationship reference already determines the selected publication; preserve independently meaningful content-based fit digests.
- [ ] Follow-up: remove the separate Lasagna surface-reference classifiers (`fit_service._is_surface_ref` and `stored_surfaces.is_surface_ref`) in favor of shared classification. First establish the intended acceptance rules for canonical references, application envelopes, `vcobj:` locators and Lasagna's distinct content-addressed artifacts. Do not accidentally remove the latter or broaden an API silently.
- [x] Add regression coverage, run the existing approved argument-free runners, record production versus test/documentation line deltas, and commit the fixes.

The Lasagna classifier follow-up is recorded explicitly at the user's request; it is not considered fixed by removing Spiral legacy branches.

## Implementation and validation

`objects/RelationshipInputs.hpp` and its native implementation define the geometry-fitting family/eligibility/order policy once. Python only adapts inputs/results; native comparison keeps the original `RelationshipVersion` plus display fields, not a reconstructed JSON identity. The binding reads only the needed scalar fields, not anchor/metadata payloads. No extra database reads, schema changes, identity formats or routing paths are introduced.

`validated_manifest_references` returns decoded object handles for pin construction. Resume comparison no longer validates the recorded manifest once before validating it again inside comparison. Independent public digest/comparison boundaries still validate their inputs: there is no cache that trusts a potentially mutated raw dictionary. Regression instrumentation verifies one decode per entry during construction and one per entry per compared manifest during resume checks.

Validation used only `bash local/scripts/run_vcobj_pr_tests`, `bash local/scripts/run_vcobj_consumer_checks` and `bash local/scripts/check_vcobj_consumers`. Native suite: 53 passed, optional SSH skipped. Python store: 223 passed. Spiral: 353 passed plus 54 subtests, one optional data skip, and the same three previously reproduced service failures listed in [legacy-removal validation](legacy-removal.md#validation-and-size). The runner retains exit status 1. Lasagna: 34 passed plus six subtests. Consumer runtime and affected GUI/CLI syntax checks passed, including the newly rebuilt native policy tests. No full GUI, macOS/arm64 or full-volume performance claim is made.

Production delta from `43a2b576b`: consumers 31 added / 133 removed (net −102), shared implementation/binding/build registration 66 added (including the new files); total 97 added / 133 removed, net −36. Regression code is 47 added / one removed (net +46); documentation is separate. Native tests cover supported families, canonical endpoint order and exclusion precedence; existing real-store consumer tests verify the integration.
