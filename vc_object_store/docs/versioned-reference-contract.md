# One versioned reference representation

Use the shared `ObjectVersion` value in C++ and its standard Python/wire representation `{store, generation, object, version}`. Carry that complete value unchanged through store APIs, Lasagna, Spiral, GUI adapters, caches, geometry and artifact access. A consumer must not invent another tuple, flattened field set, reordered representation or partial reference and reconstruct the original later.

Extra application data belongs beside the reference: for example `{reference: <ObjectVersion>, type: "surface", ...}`. The reference is one component of that record, not redefined by its application. A selected `ObjectRevision` already contains the reference and selected metadata; retain it when available.

Current-object discovery is an explicit UUID-based selection, not an incomplete historical reference. Resolve it once when an operation needs a stable version and carry the resulting reference to all associated reads. Do not infer an exact owner from the attached store or from a later current-object lookup.

Transaction/preparation guards carry one `RecordSelection` of shared object/relationship references. Numeric counters are boundary/storage projections, not a second synchronized handle map. `RecordExpectation` distinguishes no live record, an exact live reference and an exact tombstone reference; explicitly numbered legacy requests retain their original meaning until shared validation. Full-reference merges never downgrade to numbers, and equal counters do not make different owners/lineages equivalent. Deferred editing retains the loaded reference alongside draft, mapping and pending-submission state.

Only an external input/output boundary may parse or serialize a reference, using shared native codecs. This includes JSON requests, persisted older formats, CLI selectors and textual locators. Decode legacy inputs once into the shared representation; keep any unavoidable legacy selection resolution in the store layer. Internal calls do not reparse or normalize already-selected references. Existing external data compatibility is not a reason to preserve a second internal representation.

## Review and implementation task

Baseline: `38c72d972`. Follow the [review contract](review-contract.md) across producers and sibling consumers, not only the Lasagna example. Log concrete alternate representations, duplicated field interpretation and unnecessary conversions, and distinguish legitimate wire serialization from internal policy. Astra independently adjudicates the candidates before implementation. Fix by sharing native handling and deleting local adapters; do not redesign schema 21, add persistent identity tables, or broaden this into an unrelated model/solver audit. Record production additions/removals separately from tests/docs and run the existing approved test runners.
