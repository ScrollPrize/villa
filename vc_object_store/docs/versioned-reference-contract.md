# One versioned reference representation

Use the shared `ObjectVersion` value in C++ and its standard Python/wire representation `{store, generation, object, version}`. Carry that complete value unchanged through store APIs, Lasagna, Spiral, GUI adapters, caches, geometry and artifact access. A consumer must not invent another tuple, flattened field set, reordered representation or partial reference and reconstruct the original later.

Extra application data belongs beside the reference: for example `{reference: <ObjectVersion>, type: "surface", ...}`. The reference is one component of that record, not redefined by its application. A selected `ObjectRevision` already contains the reference and selected metadata; retain it when available.

Current-object discovery is an explicit UUID-based selection, not an incomplete historical reference. Resolve it once when an operation needs a stable version and carry the resulting reference to all associated reads. Do not infer an exact owner from the attached store or from a later current-object lookup.

Transaction/preparation guards carry one `RecordSelection` of shared object/relationship references. Numeric counters are boundary/storage projections, not a second synchronized handle map. `RecordExpectation` distinguishes no live record, an exact live reference and an exact tombstone reference; it cannot contain a bare counter. Explicit numeric selection APIs check requested live counters in one shared snapshot and immediately return full references. Full-reference merges never downgrade to numbers, and equal counters do not make different owners/lineages equivalent. Deferred editing retains the loaded reference alongside draft, mapping and pending-submission state.

Only an external input/output boundary may parse or serialize a reference, using shared native codecs. Explicit CLI selectors and textual locators resolve once in the store layer. Internal calls do not reparse or normalize already-selected references. Transaction commands have one representation: format 3 with full references. Old command formats, array references and incomplete application pins/manifests are rejected rather than reconstructed. Resolve uncertain prior request outcomes before regenerating requests; a failed decode does not prove that the original request never committed.

Resolution, editing, evidence and validation requirements are defined in the [shared store API contract](store-api-contract.md).
