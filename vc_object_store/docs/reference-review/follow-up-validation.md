# Follow-up unification: implementation and verification

Historical report: the compatibility choices below are superseded by [legacy removal](legacy-removal.md). Current commands use format 3 only, with full-reference guards; incomplete manifests no longer receive weaker comparison.

Baseline: `5920625cc`. Astra [adjudicated](follow-up-astra.md) all four [finding groups](follow-up.md), with F-02 classified as consolidation and F-03 qualified for legacy requests and deletion semantics. Its [implementation review](follow-up-implementation-review.md) has no remaining source blocker after the mixed-case command-2 and current-tombstone selection fixes. Runtime results below are implementation-owner evidence, not tests executed by that reviewer.

## Completed implementation

- [x] F-01: fiber-map controller/cache/job state retains an optional `ObjectVersion` alongside unavailable status. Existing application counters, volume/mapping invalidation and stale-job handling remain separate. Same-counter/different-owner/lineage and unavailable-status regressions added.
- [x] F-02: scalar Qt/nanobind consumers use shared scalar codecs, not artificial one-element batch envelopes. Actual batch/endpoint operations remain batch operations.
- [x] F-03: `TransactionRequest`, `TransactionContext`, `PreparedOperation`, linker snapshots, fetch callbacks and document helpers carry one `RecordSelection`. `RecordExpectation` contains the shared `ObjectVersion` or `RelationshipVersion`, with explicit no-live and exact-tombstone states; legacy numbered inputs remain boundary-only compatibility values. The parallel `expectedObjectReferences` map is removed. Read/write/preparation merging preserves full references and conflicts across owners/lineages, including equal counters. Numeric receipt/SQL projections remain at their owning boundary.
- [x] F-03 consumer closure: artifact publication retains the selected reference across upload; native loaded tools preserve mapping/write-lineage guards. Grouped fiber edits accumulate complete handles. Growth annotations use the editor's complete loaded guard without a redundant current-head read; direct callers use shared selection. Project/resource, attached-document, transform, ingest endpoint and link-marker guards retain their selected identities.
- [x] F-03 versioning exceptions: no live record is not an exact deleted version. Shared current/exact resolution can describe the winning tombstone with either history policy, including referenced sources; ordinary reader lookup still hides it. History-off does not acquire old-version retention or resurrect deleted geometry. Deferred drafts, membership/mapping guards, active write lineage and pending-submission recovery remain independent state.
- [x] F-03 compatibility: old command-1/2 canonical encodings and retry digests are retained at serialization. Command 3 directly represents complete write/read/relationship/tombstone guards, so old readers reject rather than silently discard stronger checks. Legacy mixed-field disagreement and duplicate canonical keys reject; saved receipts replay before current-input checks. Schema 21 and existing binary UUID/reference storage are unchanged; no tables or persisted identity directories are added.
- [x] F-04: relationship provenance carries a complete shared `RelationshipVersion` beside materialized content. Native manifest comparison uses it for new manifests; old manifests retain their explicitly weaker prior comparison without inventing an owner. Content-based fit digests keep their existing semantics.

## Verification commands

Only the fixed, argument-free approved runners are used:

```sh
bash local/scripts/run_vcobj_pr_tests
bash local/scripts/run_vcobj_consumer_checks
bash local/scripts/check_vcobj_consumers
```

The existing local consumer runner was extended to execute the checked-in fiber-map staleness regression; the syntax runner also checks `FiberMapWorkspace.cpp`. These local environment/build wrappers remain outside the submitted source. No dependency installation or ad-hoc test command was used.

## Results

Initial integrated run: 53 native tests passed, optional SSH skipped; 223 Python store tests passed; 352 Spiral tests and 54 subtests passed, one optional data test skipped. The same three unrelated extended-service failures recorded in [the preceding validation](validation.md) reproduce against baseline Python source under the same native/dependency environment. The runner preserves its nonzero status. Lasagna passed 34 tests and six subtests.

Final approved PR rerun repeated those totals: 53 native passes plus the optional SSH skip, 223 Python store passes, 352 Spiral passes plus 54 subtests and one optional data skip, the same three baseline-reproduced failures, and 34 Lasagna passes plus six subtests. The preserved overall exit status is 1 because of those three service failures, not a green full-suite claim.

Consumer runtime runner passed twice, including source-repair Python tests, selected geometry/editor/export consumers, relationship manifest tests and all 20 fiber-map staleness tests. Its last rerun rebuilt and passed the latest coherent-store regressions, including mixed-case UUID rejection, exact live/deleted/no-live states, inherited tombstones under both history policies, complete read/relationship guards and stale-input receipt replay. Additional command-1/2 wire/digest assertions use an explicit frozen pre-unification field layout checked against baseline source, rather than deriving expected values through the new encoder. This is not a separate baseline native rebuild. The final affected native GUI/CLI syntax runner also passed.

No interactive GUI session, full-volume solver benchmark or macOS/arm64 build is claimed. Existing optional SSH/data skips and dependency shutdown/deprecation diagnostics are not silently suppressed.

## Size

Relative to `5920625cc`, production code adds 449 lines and removes 239 (net +210); regression code adds 195 and removes 28 (net +167). Documentation and local runner configuration are excluded. This is not a net line reduction: complete dependency/relationship/tombstone guards and replay-compatible command handling add shared capabilities while removing the competing numeric/full maps and consumer encodings. No compressed formatting is used to claim savings.
