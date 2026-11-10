# R6 fixes

Scope: finalize the v21 store and the consumers directly affected by its identity, geometry, and evidence contracts. No schema change, payload duplication, new persistent metadata, or unrelated solver change.

| Findings | Change and regression |
| --- | --- |
| C-001, C-004 | Order project records and resources by the creation timestamp preserved through overlays, without querying history. New project/resource writes record fractional UTC; historical whole-second stamps sort after precise stamps in their second, with UUID as the deterministic tie-breaker. Mixed history and overlay override regressions. |
| C-002, C-003, S-001 | Keep owner immutable in the resource metadata API, share native resource preparation between local Python and checked sessions, and restore retired rows when adding. Resource-specific edits accept a matching new deterministic key for a retained legacy ID; generic object APIs continue to use the stored UUID. Owner, Python retirement, and native legacy edit tests. |
| C-005 | Honor explicit validity when inverse-transforming a full surface, even at the displayed missing sentinel. Coordinate-only edits retain their existing deletion convention. Test both forms. |
| X-001, X-002 | Load the saved summary in metrics-only dispatch; reject nonintegral and out-of-range indices in the shared anchor codec. Regressions exercise metrics-only dispatch and the Python branch-anchor caller. |
| X-003 | Pin direction preparation to extraction evidence; carry selected and joint dataset pins into split and sampling reads; verify a prepared solve's saved source stamp before sampling. Standalone derivation callers without such evidence are not redesigned as part of v21 finalization. |

S-002 was rejected by independent review. See [adjudication](adjudication/S-002.md).
