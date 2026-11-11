# Post-fusion review

Frozen baseline: `153c9a10f1cca90f57a2e70dc5458d1bd5bad2a4`. Review only: no production changes. Follow the invariant/sibling-call review contract, with independent Astra review and no finding quota. Findings below are candidates until their dispositions are recorded; a keyword/file screen is not line-by-line semantic verification.

This is the frozen review record. Subsequent implementation and regression results are tracked in [fixes](fixes.md); the source-review wording in individual findings describes their discovery stage, not the final implementation status.

Native/core/hosted/bindings inventory and findings are in [Astra's inventory](pf-astra-inventory.md). Consumer screening and traced flows are in [the consumer inventory](consumer-inventory.md). Root candidates receive independent [Astra adjudication](root-adjudication.md); runtime diagnostics, when available, are recorded here rather than implied by unrelated suite counts.

| ID | Candidate | Status |
| --- | --- | --- |
| PF-A01 | SurfaceGrid accepts a transformed view from a different selected metadata version sharing one payload. | Accepted; runtime reproduced |
| PF-A02 | expected_commit_sequence JSON silently coerces fractional values. | Accepted; runtime reproduced |
| PF-A03 | Hosted current-object response internally serializes/parses JSON values again. | Simplification only |
| PF-A04 | Exact graph projection validation executes dummy current-object SQL. | Simplification only |
| PF-A05 | Generation-only selectors silently ignore the supplied lineage. | Accepted; runtime reproduced |
| PF-C01 | Bound-only Vesuvius volume discovery uses a registration-aware count and can certify mixed bindings. | Accepted narrowly; runtime reproduced |
| PF-C02 | StoreSurfaces no-op return skips loaded-reference/mapping guard validation. | Accepted; runtime reproduced; no stored-data corruption |
| PF-C03 | ToolObjects retains loaded pins after surface update/artifact publication. | Not a confirmed defect: read-pin contract, not an automatic reusable editor |
| PF-C04 | Spiral object manifest references silently overwrite duplicate UUID entries. | Accepted; runtime reproduced |
| PF-C05 | Different native coordinate bindings collapse to one untransformed frame in ToolObjects. | Accepted; runtime reproduced; shipped multi-source exposure not demonstrated |

## Final disposition and verification

Ten candidates: seven correctness/input-contract gaps, two simplification-only opportunities, one rejected required fix (PF-C03). Independent Astra adjudication and root counterexample checks agree on these dispositions. Source history traces these omissions to before the latest API-fusion pass; they are not identified as regressions introduced by that pass.

The root ran isolated native/Python diagnostic fixtures through the fixed approved command `bash /home/hendrik/villa3/local/scripts/run_vcobj_review_checks`; exit status 0. Environment: Linux, GCC 13, Release consumer targets with the existing RelWithDebInfo core library, Python 3.14. No timing/performance claims. Observed results:

| Finding | Counterexample result |
| --- | --- |
| PF-A01 | Metadata A plus transformed view B, sharing geometry, accepted: x=1 instead of A's x=2. |
| PF-A02 | Fractional expected commit sequence N+0.5 accepted as N; write committed. |
| PF-A05 | A generation-only selection naming a different lineage returned current. |
| PF-C01 | Mixed direct volume bindings with a registration inferred volume a; native catalog selected 2 fibers but default bound-only reader returned 1. |
| PF-C02 | Loaded version 3, external edit to version 4: unchanged update returned 3 without validating the stale guard. |
| PF-C04 | Duplicate UUID entries at versions 1 and 2 accepted; retained version 2. |
| PF-C05 | Native bindings with scale factors 2 and 4 accepted together; output binding silently used scale 2. |

The same runner passed its focused core/repeat checks, source-repair tests, ten native consumer regressions, ten selected-value/transformed-view checks and consumer syntax checks. The diagnostic fixtures used automatically cleaned local temporary databases, not production datasets. Their temporary runner hooks and source files were removed afterward so future fixes do not leave bug-expecting probes in the approved runner. During implementation, promote each counterexample to a checked-in regression asserting corrected behavior.

These new runtime cases cover local history-on stores only, not a full hosted/sharded/packed/history-policy matrix. Individual finding files retain their initial source-review notes; this section records the subsequent runtime verification. The 128-file consumer inventory is a screen with detailed traces of listed flows, not a claim of exhaustive semantic proof.

No claim of zero undiscovered defects, complete line-by-line coverage or a percentage of all defects found. Explicit discovery, payload-only reads, detached source-import formats, immutable payload identifiers and offline conversion are not findings by themselves. Do not revive deleted legacy APIs or propose a second local resolver as a fix.
