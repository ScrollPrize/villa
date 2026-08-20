# Task: cache decoded fiberlet graph chunks

Fix on-demand fiberlet replay so the existing `ChunkCache` and its LRU own the
decoded, indexed anchor and fiberlet chunk representations.

- Keep anchors and fiberlets in separate caches with independent local limits
  and their existing shared decoded-byte budget.
- A cache miss may read or generate serialized storage, but it must decode and
  index the chunk once before publishing the cache entry.
- A cache hit must return a lease to the decoded chunk. Graph queries must not
  deserialize whole chunks per anchor, edge, or beam candidate.
- Incident-edge lookup must prefetch the complete neighboring owner-chunk halo
  through the fiberlet cache, then use per-chunk endpoint indices.
- Beam lookahead must use cached prefix/connectivity data only. The separate
  route level in the fiberlet cache is loaded for the committed edge, not every
  candidate expansion.
- The existing cache LRU must account for and evict the decoded objects. Do not
  add a second graph or fiberlet LRU.
- Preserve stable ordering, costs, replay choices, and numeric behavior.
