# PF-A03 — Hosted current-object envelope retains internal JSON text round trips

Baseline: `153c9a10f1cca90f57a2e70dc5458d1bd5bad2a4`. Simplification-only candidate, low priority; no demonstrated correctness defect. Older leftover: identical code at `0c7e4e521`.

Owner: `src/client/HostedStore.cpp::currentObjectRevision`. After parsing its HTTP response once, it dumps `status` and `revision` subobjects to strings and calls `storeStatusFromJson` / `objectRevisionFromJson`, which parse them again. This is an internal conversion, not another transport boundary.

Sibling comparison: `Wire.cpp::envelope/unwrap` now use JSON-value status codecs directly. Exact revision endpoints parse the response text once through the scalar revision codec and do not need changing merely because their public function accepts transport text. Hosted session current-object reads delegate this current endpoint; Python hosted/session readers inherit it.

Recommendation: extract/reuse one native current-object response/value decoder with the same revision and exact-metadata validation. Do not copy `revisionFrom`, move schema interpretation into the HTTP client, or broaden this into a general JSON framework. Keep the response string boundary and exact metadata text unchanged.

Verification after an optional cleanup: current live/tombstone/missing results and status output, with exact metadata text; no expected numerical or persistent changes. Not performance-measured in this review.
