# Independent review of the three post-closure fixes

Baseline: `b73a31f33`. Reviewed the working production/test delta for PC-A01,
PC-CORE-01 and PC-S01. No production edits or competing builds/test runners by
this reviewer. This is delta review, not the requested later full review; that
requires completed tests and a frozen implementation commit.

## Production dispositions

- **PC-A01:** `Snapshots.cpp::importJson` now uses existing closure-aware
  `objectSource` / `edgeSource` with `includeDeleted=true` for reserved-identity
  admission. Live same-content checks still precede the same-kind tombstone
  rejection, retaining idempotence and inherited ownership. Cross-kind rejection
  is also shared/tombstone-aware. The existing whole-operation savepoint remains
  around all object/edge insertions. Intentional low-level resurrection APIs are
  unchanged. No new local SQL resolver or copied dependency data.
- **PC-CORE-01:** selection artifact decoder, native encoder and Python producer
  reuse the shared store-position codec. Native encoding canonicalizes position
  identity before serialization; binary entry representation is unchanged.
  `commonFromJson`, projected `presentCommonFromJson`, and `Wire::versionFromJson`
  reuse shared scalar helpers. Checked `RecordFields.hpp`: the common visitor's
  sole `int64_t` is commit sequence and its optional integers are version/approved
  version, so the new type-based dispatch does not accidentally apply version
  positivity to unrelated numeric metadata. Metadata imports still publish local
  versions via `restoreCommon`. No second scalar validation table added.
  The follow-up `Wire::revisionFrom` / `relationshipFrom` envelope versions use
  the same positive-version helper, and the optional object-envelope commit
  sequence uses the shared nonnegative helper while preserving null. No valid
  unnumbered-wire regression found: `versionFromJson` retains its absent-field
  early return; present null already failed the previous integer extraction.
  Published selected revisions carry positive versions independently of history
  retention; empty archive UUIDs in history-off results are not version zero.
- **PC-S01:** both stale-scope UUID sets are canonicalized through the existing
  UUID API before selection/comparison. Both public commit overloads and stored
  detector publication share this admission. Edge-kind strings and empty-scope,
  ownership/review-state and candidate semantics are unchanged. Invalid inputs
  fail under the operation savepoint before mutation.

No blocking production defect or regression established by this delta trace.

## Regression inspection and resolved correction

Metadata-import tests cover local history on/off, all four referenced source/root
history pairs, live idempotence/owner retention, content conflict, same-kind
tombstones, both cross-kind directions and caught failure followed by committing
unrelated outer work. They inspect original tombstone owner/version after failure.
The scalar tests exercise native binary encoder/decoder and Python ingress plus
snapshot/projected common record boundaries using malformed type/range inputs.
The revised tests use public `objectRevisionFromJson` and
`objectQueryResultFromJson` boundaries, not an unexported internal header, and
also mutate outer object-envelope version/commit fields.

Scope tests span local/referenced inputs, history policies, both public overloads,
empty scopes, malformed UUID rollback and approved/rejected preservation. However,
the initial touching regression used `{upper(a), a}`. That also contained the
canonical endpoint, so the old broken implementation still passes that individual
assertion. The correction is now verified in source: touching is uppercase-only
and fixture UUIDs deterministically contain hexadecimal letters, avoiding the
random all-digit UUID case. The object scope independently requires the uppercase
second endpoint. This was a regression-coverage correction, not a discovered
production bug.

Runtime results belong to the approved main runner. This reviewer has not claimed
the edited tests passed or independently executed GUI/macOS/arm64/concurrency
workloads. No schema or persisted-format migration is required by these fixes.

## Deferred full-review obligation

The untouched `Wire.cpp::relationshipFrom` branch clearing array-prefixed
`a_revision`/`b_revision` strings is explicitly older-server compatibility and
should be evaluated in the next full review, separately from this implementation
delta. Trace current relationship producers, cached wire values and consumers;
preserve legitimate empty archive fields for current-only records and distinguish
them from malformed legacy array strings. No current producer/consumer failure
or new regression is asserted here, and no fix is silently requested in this
three-finding implementation scope.

Also carry forward binary selection header `count` extraction with
`get<uint64_t>()`: a fractional `0.5` may coerce to zero for an empty-entry
artifact. This unchanged scalar boundary needs independent reproduction and
contract/severity adjudication in the full review; it is not asserted to lose
selected objects or silently included in the current fix scope.

The later full review should reopen shared import admission and write counterparts,
selection/record/wire scalar ingress siblings, detector scope producers and public
overloads, plus the existing storage/consumer inventory. This note is preparation,
not premature full-coverage closure.

Follow-up disposition: the next review independently reproduced count coercion and the root folded it into PC-CORE-01 before final validation, superseding the earlier deferral of that scalar check. The final reopened source hash is `89d698050e0e41d28aa8ec2acb55129048467147b2e40a5f27feb1094fe4daab`; [the independent next review](../iteration-next/astra.md) records the source check of integer/exact byte-count equality. Legacy archive handling remains NX-A01, and the distinct selection-list shape gap is NX-C01. The three implemented fixes are now commits `56dc9c792`, `395e86ace` and `9223b5b22`, with unchanged reviewed source content.
