# PF-A02 — Write commit-position guard still truncates JSON numbers

Baseline: `153c9a10f1cca90f57a2e70dc5458d1bd5bad2a4`. Candidate correctness/input-validation inconsistency, medium priority; source-traced. Older omission: `0c7e4e521:src/Commands.cpp::requestHeader` has the same conversion.

Owner: `src/Commands.cpp::requestHeader`, `expected_commit_sequence`. A present non-null value is decoded with `get<int64_t>()`, which accepts floating JSON numbers by numeric conversion. The newly consolidated read-position codec `internal/ObjectVersionJson.hpp::storePositionFrom` correctly rejects bool/fractional/overflow/negative values instead.

Counterexample: at head N, a command with `expected_commit_sequence: N+0.5` is converted to integer N and can pass `Versions.cpp::checkTransactionContext`. An invalid snapshot guard is silently changed into a valid guard rather than rejected. This is not a request to remove explicit numeric commit positions: they are legitimate store-wide guards, but must be exact integers.

Sibling ingress: local Python `submit`, hosted Python `submit`, session request adaptation, server command decoding, and saved-request validation all route through `transactionRequestFromJson` / `requestHeader`. Direct native requests already have an `optional<int64_t>` and cannot carry a fraction. Complete-reference object/relationship guards already reject fractional versions through the shared reference codec. History policy and physical placement do not affect this scalar decode.

Recommendation: extract one small strict nonnegative-int64 boundary helper used by both read position and write expected position; preserve null as no requested commit guard. Do not duplicate another number policy in Python or introduce another position type.

Required checks: reject fractional/bool/unsigned overflow/negative guards before execution via local/session/hosted commands, accept integer zero/N and null, preserve conflict behavior for a valid but stale integer. Source-only; no runtime proof claimed.
