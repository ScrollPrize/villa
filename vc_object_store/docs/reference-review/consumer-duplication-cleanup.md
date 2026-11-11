# Consumer duplication cleanup

Baseline: `43a2b576b`. Remove obsolete identity branches and duplicated reference/relationship policy rather than preserve competing implementations. Keep application configuration, content digests, draft/mapping/membership guards and intentional external API boundaries. No schema or solver changes.

- [x] Remove `legacy_source` consumption from SpiralPanel and Spiral service commit context; only the producer's complete `reference` is valid.
- [x] Consolidate manifest reference validation and retain decoded handles instead of decoding the same entries repeatedly in pin construction and resume comparison.
- [x] Share relationship eligibility and endpoint ordering between native Spiral input comparison and Python input materialization. Python wraps `fittingRelationshipInput` through the batch binding; neither consumer has its own family table or endpoint-ordering policy. The fitting vocabulary is separate from store identity/routing.
- [x] Remove redundant numeric/content identity checks where the complete relationship reference already determines the selected publication; preserve independently meaningful content-based fit digests.
- [x] Lasagna classifier follow-up: removed `fit_service._is_surface_ref`; all four service routes use `stored_surfaces.is_surface_ref`, as do tifxyz and approval readers. One dispatch rule recognizes explicit vcobj inputs (canonical/wrapped references, untyped UUID selectors, `type=surface`, `vcobj:` locators); paths/path-only dictionaries and Lasagna's distinct type/name/hash artifacts do not qualify. Native code still validates identity. Existing object-only `external_surfaces` and atlas-base envelope validation remains unchanged.
- [x] Add regression coverage, run the existing approved argument-free runners, record production versus test/documentation line deltas, and commit the fixes.

The Lasagna classifier follow-up is separate from the earlier Spiral cleanup; its implementation and validation are recorded below.

## Implementation and validation

`objects/RelationshipInputs.hpp` and its native implementation define the geometry-fitting family/eligibility/order policy once. Python only adapts inputs/results; native comparison keeps the original `RelationshipVersion` plus display fields, not a reconstructed JSON identity. The binding reads only the needed scalar fields, not anchor/metadata payloads. No extra database reads, schema changes, identity formats or routing paths are introduced.

`validated_manifest_references` returns decoded object handles for pin construction. Resume comparison no longer validates the recorded manifest once before validating it again inside comparison. Independent public digest/comparison boundaries still validate their inputs: there is no cache that trusts a potentially mutated raw dictionary. Regression instrumentation verifies one decode per entry during construction and one per entry per compared manifest during resume checks.

Validation used only `bash local/scripts/run_vcobj_pr_tests`, `bash local/scripts/run_vcobj_consumer_checks` and `bash local/scripts/check_vcobj_consumers`. Native suite: 53 passed, optional SSH skipped. Python store: 223 passed. Spiral: 353 passed plus 54 subtests, one optional data skip, and the same three previously reproduced service failures listed in [legacy-removal validation](legacy-removal.md#validation-and-size). The runner retains exit status 1. Lasagna: 34 passed plus six subtests. Consumer runtime and affected GUI/CLI syntax checks passed, including the newly rebuilt native policy tests. No full GUI, macOS/arm64 or full-volume performance claim is made.

Production delta from `43a2b576b`: consumers 31 added / 133 removed (net −102), shared implementation/binding/build registration 66 added (including the new files); total 97 added / 133 removed, net −36. Regression code is 47 added / one removed (net +46); documentation is separate. Native tests cover supported families, canonical endpoint order and exclusion precedence; existing real-store consumer tests verify the integration.

## Lasagna classifier follow-up

Baseline: `6944d335c`. Removed the private service classifier instead of retaining an alias. Atlas-base resolution, external surface preparation, model loading and visualization export delegate to the same helper used by the geometry and approval readers. The helper classifies request intent only; complete identity validation/resolution stays in native code. Untyped UUID selectors and vcobj model locators now follow that shared route; typed content-addressed artifacts retain their own route even when carrying an incidental UUID. Arbitrary dictionaries are no longer assumed to be vcobj references. Missing-model errors use the decoded reference's object UUID, so string locators do not trigger dictionary-method errors. No schema, numerical/model, history or core reference-decoder changes.

Regressions cover canonical/wrapped/current/locator inputs, malformed vcobj intent, paths/path-only configuration, typed artifacts with and without an incidental UUID, actual model-artifact reads and a content-addressed model upload/read. `bash local/scripts/run_vcobj_pr_tests` passed 53 native tests (optional SSH skipped), 223 Python store tests and all 36 Lasagna tests plus 26 subtests. Spiral retains the same three known service failures alongside 353 passes, 54 subtests and one optional data skip; the overall runner correctly returns 1. The initial sandbox run could not bind loopback sockets; the approved runner was rerun outside the sandbox. `git diff --check` passed. No native/GUI code or dependencies changed.

Production delta for this follow-up: 17 added / 14 removed, net +3 (including the shared predicate's explanatory docstring); tests add 32 lines, with documentation separate. There is one classifier implementation, not two wrappers around competing rules.
