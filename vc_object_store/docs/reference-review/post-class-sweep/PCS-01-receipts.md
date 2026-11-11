# PCS-01: receipt admission differs from the shared identity codecs

Priority: low. Status: implemented in the subsequent requested fix; validation is recorded below. Classification at the review freeze: pre-existing omission; `Versions.cpp` had no changes in the reviewed implementation patch, and the same decoder exists at frozen HEAD `d13d33629`. Native and Astra reviewers agree this is one decoder issue with several manifestations, not one finding per caller or input variation.

## Owning boundary and callers

`vc_object_store/src/Versions.cpp:237–242` (`selectionFromJson`) iterates `j.items()` without requiring a map, retains raw UUID keys, and reads non-null versions with `get<int64_t>()`. `receiptFromJson` at lines 262–279 uses it for `objects` and `relationships`, and reads nullable `commit_sequence` with `get<int64_t>()` rather than the shared nonnegative integer codec. Complete `object_references` are correctly required to be a map and their scalar references use the shared native decoder.

`TransactionReceipt::objectReference` at lines 282–292 canonicalizes its requested UUID, then looks it up in the unnormalized numeric map. An uppercase key therefore appears absent, even when the matching complete reference remains in `objectReferences`. The consistency calls inside `receiptFromJson` return early on that absence instead of detecting or normalizing the mismatch.

The public `transactionReceiptFromJson` in `Commands.cpp` delegates to this decoder and wraps actual decoding failures as `ReceiptDecodeError`. Hosted submit and receipt recovery (`client/HostedStore.cpp`), saved receipt replay (`Versions.cpp::savedReceipt`) and Python `receipt_object_reference` (`python/objects.cpp`) share it. The corrected Spiral marker path now delegates to that public reference accessor; it must not acquire its own compensating parser.

## Reproduction

The coordinator ran the fixed ignored `local/scripts/vcobj_post_review_probes.py` through `bash local/scripts/run_vcobj_review_checks` (runner session 12181). The probe creates a disposable store, publishes a note with an alphabetic UUID, obtains its complete reference, and obtains a real no-op checked-submit receipt expecting that note. Each case mutates only one part of a copy of that valid receipt, then calls public `receipt_object_reference(receipt, uuid)`. History off and history on produce identical results:

| Receipt variation | Observed result |
| --- | --- |
| Valid producer receipt | Accepted; original complete reference returned |
| `objects: []` or `objects: null` | Accepted; reference reported absent |
| `relationships: []` or `relationships: null` | Accepted; original object reference returned |
| Numeric object version `1.5`, original reference version 1 | Accepted; original reference returned after integer truncation |
| `commit_sequence: 1.5` or `commit_sequence: -1` | Accepted |
| Uppercase UUID key in `objects`, canonical matching complete reference unchanged | Accepted; reference reported absent |
| Object version `true` | Rejected by numeric type decoding |
| Object version 0 with complete reference version 1 | Rejected as inconsistent identity |
| Commit sequence `true` | Rejected by numeric type decoding |
| Scalar empty-string maps | Rejected; not a silent empty-map case |

Do not claim booleans are accepted, a wrong-owner object is read, or that normal canonical server-produced receipts fail. No end-to-end incorrect geometry/repinning workflow was demonstrated. The concrete failure is inconsistent/malformed receipt admission and a lost lookup result for an equivalent UUID spelling. The complete-reference codec itself remains strict.

## Shared fix direction

At the native receipt boundary, require the declared maps, canonicalize their UUID keys with collision detection, decode nullable positive version projections and nullable nonnegative commit sequences with the existing shared scalar codecs, and check complete-reference consistency against the normalized maps. Preserve `ReceiptDecodeError` and uncertain-request recovery behavior. Test local/saved/hosted/Python sibling use through the same codec, including valid producer receipts and optional null **values**. Do not add a new format, schema table, legacy repair path or per-consumer parser.

No production or permanent regression test was changed in this review. The two local history modes were exercised through the public binding, not an injected malformed HTTP server or corrupted saved-receipt database. Those transport/persistence entry points are source-traced to the same decoder.

## Requested fix

- [x] Require receipt version maps and canonicalize UUID keys with duplicate detection.
- [x] Reuse strict positive-version and nonnegative-commit scalar codecs; retain nullable values.
- [x] Reject duplicate/extra/inconsistent complete references through native receipt admission.
- [x] Add native round-trip/replay and Python public-boundary regressions, including history on/off and unchanged inherited references; run the approved checks.

Keep all consumers unchanged. No new reference representation, schema or compatibility repair is needed. Validation results follow when complete.

Implementation: `Versions.cpp` adds eight lines and removes four (net +4), using the existing `versionNumberFrom`, `commitSequenceFrom` and canonical UUID handling. The native `completeReceipts` regression now also normalizes upper-case projection/reference-map keys while retaining both committing-store and inherited-store identities. Python receipt regressions use actual submitted/replayed no-op receipts in both history modes and run through `receipt_object_reference`, not another parser.

The approved focused runner (`bash local/scripts/run_vcobj_review_checks`) exited 0: eleven native targets, the two repeat/umbilicus checks, 25 source-import tests, ten consumer runtime executables, 102 selected Python checks, all consumer syntax checks and 85 combined consumers passed. The full PR runner (`bash local/scripts/run_vcobj_pr_tests`) passed 54 native tests (optional SSH skipped), 333 store Python tests, 369 Spiral tests plus 54 subtests (one optional skip), and 37 Lasagna tests plus 31 subtests. Its nonzero exit is solely the same three pre-existing Spiral failures listed in the class-sweep fix tracker; no new failures appeared. `git diff --check` passed. No dependencies were installed and consumers/schema/storage layout are unchanged. No macOS, interactive GUI or full-volume performance validation is claimed.
