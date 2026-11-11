# Targeted failure-class sweep before fixes

Frozen production commit: `d13d33629`. User requested a pattern-wide investigation before changing the three open findings. No production or permanent regression-test changes in this pass. The scope is explicitly these failure classes, not another claim that unrelated behavior has been proved correct.

This is the pre-fix evidence record. The subsequent requested implementation and validation are tracked in [fixes.md](fixes.md).

- [x] Enumerate numeric/partial-field → complete object/relationship reference construction in native code and Python/native consumers; trace where each field originates and whether a complete reference was already available.
- [x] Enumerate repair/acceptance of obsolete reference encodings; distinguish current storage layouts, supported independent source formats, explicit conversion tools and failure messages that reject old forms.
- [x] Enumerate JSON fields specified as arrays that are iterated without container validation; trace ingress and sibling passes before classifying a site. Keep deliberate maps, optional-null semantics and typed vector decoders distinct.
- [x] Independently adjudicate candidates with Astra, reproduce representative cases through the existing approved runner, and report class-wide site lists plus exclusions and limits.

The searches use actual construction/iteration sites rather than a finding quota or a few chosen functions. Each potential reconstruction must be classified as retained-reference duplication, legitimate storage/wire/feed/CLI boundary reconstruction, display-only numeric output, or a false positive. A legitimate numeric boundary must not be proposed for removal merely because it contains four reference fields.

## Evidence lanes

- [Python references](python-references.md): literal dictionaries, keyword constructors, numeric receipt maps, helper/caller chains and facade/workflow users.
- [Native references](native-references.md): aggregate and field-by-field construction, request/status/receipt combinations, native GUI/CLI/storage sources and selected-state accessors.
- [Codecs and independent adjudication](codecs-astra.md): legacy reference handling, declared-array ingress and persisted metadata, accepted site ledger and representative diagnostic results.

Root checks the overlap between lanes and the known seed findings NX-A01/NX-C01/NX-P01. New occurrences of one mechanism are counted as sites of that class, not inflated into unrelated architecture features.

## Diagnostics

Use the same approved argument-free command `bash /home/hendrik/villa3/local/scripts/run_vcobj_review_checks`. A fixed ignored local probe (`local/scripts/vcobj_class_sweep_probes.py`) runs within that runner's automatically cleaned fixture directory, creates only isolated disposable stores, and prints actual acceptance/rejection alongside valid controls. Acceptance is diagnostic evidence, not a passing assertion of intended behavior. No arguments, external services, installed dependencies, user dataset mutations or custom competing native build are required. The normal regression checks remain in the runner.

## Results and interpretation

The numeric-reference sweep confirms the known Spiral marker reconstruction (NX-P01); it did not establish another such defect. The native ledger dispositions 28 construction groups, and its Qt supplement covers 29 `toArray` sites in 115 store-connected native files. Legitimate storage/feed boundary reconstruction, display-only numbers and unstored draft sentinels are not proposed for removal.

The obsolete-reference sweep confirms the known wire relationship revision-array repair (NX-A01). Supported independent source formats and explicit conversion are not legacy vcobj fallbacks.

The array finding is missing schema admission at decoding boundaries, not ordinary iteration. An array is the correct representation. Reject a non-array once when reading the file/request/response; use a runtime error there, since debug assertions disappear in release builds. Internally validated arrays and typed vectors should be used directly, without repeated consumer guards. See Astra's exact ingress/field ledger for the broader native sites and its independently accepted Python saved-checkpoint/provenance sites. Deliberate maps, optional-null normalization and APIs intentionally accepting Python iterables remain distinct.

The approved argument-free review runner completed successfully, including native checks, consumer syntax checks, 33 selected Python checks and 66 combined consumer checks. The additional diagnostic matrix demonstrates malformed-container acceptance with valid controls: notably an object-valued command operation collection executes its contained write, and null becomes a no-op. The observations hold for both local history policies. These are diagnostic observations of missing validation, not tests asserting that acceptance is correct. Details and limitations are in the codec ledger.

No production code or checked-in tests were changed. No fix authorization is inferred from a new review finding. This is a documented failure-class search, not a proof that all unrelated defects are absent; the lane reports retain unconfirmed candidates and unexecuted cases explicitly.
