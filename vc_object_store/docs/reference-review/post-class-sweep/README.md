# Review after shared decoding fixes

Reviewed HEAD: `d13d33629c3ccee94fa61a9ef873ec82ea0b9db9` plus the uncommitted implementation patch. Freeze command: `git diff --binary -- vc_object_store/src vc_object_store/include vc_object_store/python spiral-fitting lasagna volume-cartographer vesuvius | sha256sum`; result `90ad71c5ac3c79abcff42509dd0d2c7584d0a43e645180cd13983e23ad01662f`. The new untracked native test `vc_object_store/tests/cpp/test_json_boundaries.cpp` is separately frozen at SHA-256 `508749f9df3843001544c0b173fe94161651b8c1612f3db7f7132521f4402cf2`. Production and checked-in tests were not edited during this review.

## Verdict

One confirmed finding: [PCS-01 — receipt schema admission and UUID map keys](PCS-01-receipts.md), low severity and pre-existing. Astra independently accepted it after source/caller review and the coordinator's public-API probes. No regression in the shared-decoding patch or additional consumer defect was confirmed. This is not a claim that every possible bug is absent.

Subsequent requested implementation: PCS-01 is now fixed and regression-tested in the native receipt decoder. Its linked record preserves the pre-fix evidence and documents the fix and final verification separately.

The previously logged array-boundary fixes remain valid. The new issue is a different native boundary: receipt **maps and numeric projections**. It belongs in the common receipt decoder, not another consumer workaround or reference representation.

## Coverage and evidence

- [Native lane](native.md): reopened the changed JSON/command/wire/selection/snapshot/source/geometry/extension/packed admission paths and their entry-point siblings; cross-checked selected identities, visibility, history and atomic writes against the existing per-file evidence at the same HEAD.
- [Consumer lane](consumers.md): repeated direct consumer discovery (109 native and 24 Python callers), checked the changed application paths, bindings/facades and source-format siblings, and documented exclusions and unchanged-file evidence.
- [Independent Astra adjudication](astra.md): independently reviewed the fix diff, shared boundary/caller chains and both lane reports, and adjudicated every candidate without a finding quota.
- Coordinator: traced native receipt parsing, receipt-reference projection, hosted submit/recovery, saved replay and Python use; ran malformed receipts derived from a real `ObjectStore.submit` receipt through `receipt_object_reference` under both history modes. Valid controls and rejected false positives are retained in PCS-01.

Discovery counted 255 files under native source/public-header/Python-package roots and 133 direct repository consumer files. Those counts are discovery inventory, not a claim that all 388 files received a fresh independent line-by-line review. The lanes explicitly distinguish reopened code from source-equivalent prior ledger evidence, indirect siblings, deliberate boundary adaptation and unchecked numerical/platform behavior. No unrelated solver/model tuning or schema redesign was added.

`ObjectCopy::copied_from` was also considered: it is a documented, externally visible informational provenance object with no operational reference reconstruction found. It is not counted as a wrong-reference bug. Making that metadata structurally uniform would be an explicit metadata API change, not grounds for another hidden compatibility branch.

## Validation

The diagnostic uses the existing argument-free approved command `bash local/scripts/run_vcobj_review_checks`, with an ignored fixed local probe and automatically cleaned disposable stores. Its observations are diagnostic acceptance/rejection, not tests asserting malformed acceptance is correct. That runner completed with exit 0: ten focused native targets, two repeat/umbilicus checks, 25 source-import tests, ten consumer runtime executables, 48 selected Python tests, consumer syntax checks and 85 combined consumer/position tests passed. The production patch hash and new test hash remain unchanged. `git diff --check` passed.

No dependency installation, external service, user-dataset write or independent agent build was performed. Prior full-suite results remain in [the fix tracker](../class-sweep/fixes.md); no fresh full-suite result is claimed here. No production fixes were made in this review.
