# Review after the post-closure fixes

Source baseline `b73a31f33` plus the implementation/test patch SHA-256 `89d698050e0e41d28aa8ec2acb55129048467147b2e40a5f27feb1094fe4daab` (`git diff --binary -- vc_object_store/src vc_object_store/include vc_object_store/python vc_object_store/tests`). The initial review freeze was `eb68039f0c6f248274a32885417698098b2b9fb319593273024ebf48c34f1ced`; selection scalar obligations were explicitly reopened when a related count-coercion case was reproduced and fixed. Other production sources did not change between those freezes. Final commit mapping and test completion are recorded below when available.

Commit mapping: import `56dc9c792`, shared scalar validation `395e86ace`, stale scope `9223b5b22`. Comparing `b73a31f33` to this final code snapshot yields exactly the reviewed source/test patch hash above; no source change was hidden by committing it.

## Astra-adjudicated findings

| ID | Priority | Remaining issue and evidence |
| --- | --- | --- |
| NX-A01 | Low | `Wire.cpp::relationshipFrom` still clears array-prefixed archive selector strings for older-server compatibility. Current producers emit UUIDs or empty strings; ordinary ingress should not repair obsolete forms. Complete endpoint references validate first, so no wrong-owner or incomplete-reference acceptance is claimed. Source-traced through both relationship and selected-input result decoding/caches. |
| NX-C01 | Low | `Selections.cpp::decodeSelectionRecord` does not require `transforms` and `origins` to be arrays. Null/object containers can be interpreted as empty lists or iteration over their values. Astra independently reproduced this structural validation gap. This is distinct from the now-fixed numeric position/count coercion; no selected identity loss or wrong geometry demonstrated. |
| NX-P01 | Low | Spiral's link-marker read rebuilds a complete reference using a numeric receipt projection plus the request owner/generation instead of carrying the receipt's existing complete reference. Ordinary marker producers publish a new committing-root version, so no wrong-owner runtime counterexample is claimed. The redundant reconstruction violates the shared-reference contract and should use the existing receipt reference API. |

All three predate this implementation patch. They remain open for the next requested fix pass; they were not silently changed during this review. No new regression in the inspected fixes was confirmed. This does not imply absence of other undiscovered issues.

Follow-up: the requested failure-class sweep expanded the container-admission sites; the subsequent implementation fixes NX-A01/NX-C01/NX-P01 and those sibling sites. See [the shared-decoder fix tracker](../class-sweep/fixes.md) for current status and validation; the table above records the pre-fix review.

## Coverage and obligations

- [Native core](core.md): full enumerated per-file ledger, current/exact, read/write, identity/retention/placement and scalar-ingress obligations; source-only and numerical exclusions are explicit.
- [Consumers and formats](consumers.md): the established 109 direct native, 24 Python and nine format-file ledger, reopened through actual shared API/caller obligations. Unchanged per-file dispositions refer to their prior explicit source evidence, not an assertion of newly proved unrelated algorithms.
- [Independent Astra review](astra.md): storage/transport/import/export/bindings source traces, independent candidate adjudication and diagnostic scope.

Root checked the edited scalar codecs, metadata import admission, both stale-scope public overloads, native/Python selection ingress and their regression integration. Bindings still adapt language values and delegate schema interpretation to native code. The expanded selection scalar check now uses the byte-derived cardinality plus integer-type/exact equality rather than another permissive numeric getter. No additional concrete root finding beyond the three listed above.

This review has no arbitrary finding quota. Its coverage is still source/caller review plus specified tests, not proof of every malformed-input combination, numerical method, asynchronous interleaving, low-level VFS failure or platform. No interactive GUI, macOS/arm64 or full-volume performance result is newly claimed. Current task and regression results belong to [the fix tracker](../post-closure/fixes.md).

## Final verification

The approved review runner exits 0 on the mapped committed source: nine focused native tests, two repeat/umbilicus checks, five source-repair tests, ten native consumer executables, 33 selector/reference/extension tests, consumer syntax checks and 66 combined fiber/surface/manifest/grower tests pass. The full PR runner passes 53 native, 244 store Python, 358 Spiral plus 54 subtests, and 37 Lasagna plus 26 subtests, with the same three baseline Spiral failures described in the fix tracker. This is current runtime evidence, not a restatement of a prior unchanged test run. Existing warnings and optional skips remain. `git diff --check` passes; only review/task documentation changed after the source freeze.
