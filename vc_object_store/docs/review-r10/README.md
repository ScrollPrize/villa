# v21 closing review

The review scope is the v21 store and directly affected, supported consumers. Earlier rounds are preserved as [R6](../review-r6/README.md), [R7](../review-r7/adjudication.md), [R8](../review-r8/adjudication.md) and [R9](../review-r9/adjudication.md). Findings and review limitations stay separate from test results.

- [Core](../review-r9/core.md) and [storage](storage.md): no store or reader implementation changed after the R9 scope review. No schema changes, new persistent tables, payload copies, or history-layout redesign were added by the fixes.
- [Consumer lane](consumers.md): the ordinary neighbor expansion reads the saved parent signature and uses its pin for catalog and curve reads. The final delta `2748bd5c1` rejects expansion outputs without a certified pin before republishing evidence. Older outputs without evidence must be regenerated for a new solve; existing report-only paths do not reconstruct evidence.
- [Independent Astra gate](adjudication.md): decides whether remaining observations affect the v21 release scope.

Runtime validation uses the existing no-argument `bash local/scripts/run_vcobj_pr_tests` runner; focused consumer checks use `bash local/scripts/run_vcobj_consumer_checks`. The focused Vesuvius tests exercise reference comparison, piece splitting, UV layout and subset, H/V, selected and joint evaluation, and neighbor expansion/pairing. The paired and expansion fixtures test provenance handoff and rejection, not a real-data solve. No interactive GUI, full-volume, or macOS run is claimed.

Final Linux validation passed: 53 native tests (optional SSH skipped), 219 Python store tests, 122 Spiral tests (optional real-data test skipped), 14 Lasagna tests plus six subtests, 63 focused Vesuvius tests, and the native consumer checks. The existing Spiral `PatchSamplingAtlas` nanobind shutdown warning remains. The final store suite ran against the unchanged runtime from `8859fc945`; the focused Vesuvius tests include both subsequent producer commits.
