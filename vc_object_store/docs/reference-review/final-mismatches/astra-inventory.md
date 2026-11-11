# Final-mismatch review — native/hosted/bindings coverage

Baseline: HEAD `153c9a10f1cca90f57a2e70dc5458d1bd5bad2a4` plus working-tree post-fusion fixes inspected during this pass. No production/test edits, installations or runtime tests; only requested review documents. Review/versioned-reference/coherent-store contracts reread. Main agent's reported test totals are not independent runtime evidence from this lane.

## Latest working delta

| Paths / symbols | Disposition |
| --- | --- |
| SurfaceGrid.cpp selected transformed pairing | Full canonical-source reference and geometry checks close PF-A01; no new issue found. Payload-only reads remain available through their separate APIs. |
| Commands.cpp::requestHeader, internal/ObjectVersionJson.hpp | Shared strict commitSequenceFrom closes read/write scalar mismatch. No extra position type or relaxed reference policy. |
| StoreReader.cpp::selectedObject | Generation-only rejection closes ignored qualifier; current and complete exact branches retained. |
| Queries.cpp field validators, ObjectStore.hpp, ObjectReadBindings.hpp | Native accepted-field policy reused without dummy database queries. Both current/exact projections validate; no binding-owned schema table added. |
| Wire.cpp current-result codecs, HostedStore.cpp, Server.cpp | Shared envelope removes internal text round trips; matching exact metadata and status codecs retained. |
| ToolObjects.cpp::coordinatesFor | Loaded frames resolved through retained references and shared geometryCoordinates; mapped-frame declarations remain explicit. No implicit successful-write repinning introduced. |
| StoreSurfaces.cpp::update | No-op requests now reach native guards instead of returning early; no separate fresh numeric expectation added. |

## Additional bounded invariant tracing

| Paths / symbols / sibling boundary | Disposition |
| --- | --- |
| History.cpp::relationshipsAtSelected; Versions.cpp::relationshipsAt; Traversal.cpp exact/around graph paths | FM-A02 alternate internal membership-key opportunity; no supported collision established. Existing endpoint/reference applicability and retention behavior not labeled defective. |
| ObjectStoreSession.cpp artifact selection/materialization/publication; Artifacts.cpp writer/export; HostedSession.cpp::writeArtifact; HostedStore.cpp::fetchArtifact; SourcePaths.cpp closure protection; Python session artifact bindings | FM-A01 concrete missing local output-target protection. Current/exact identity selection and checked publication otherwise delegate shared paths. Copy-sidecar writer is a justified internal destination. |
| GeometryViews.cpp transform construction, validateForWrite, mapping operation and canonical data access | Selected identity, payload-only non-authority, mapping guard and data freshness retained; no new finding from the bounded inspection. Numerical conversion algorithms not re-audited. |
| ObjectStoreSession.cpp receipt/deferred submission; Versions.cpp saved receipts; Commands.cpp public decode; HostedSession.cpp submit | Previous ReceiptDecodeError pending-preservation fix remains intact. No additional legacy receipt acceptance found. |
| StoreReader.cpp::changesTouchKinds; HostedSession.cpp current/exact caches and transforms; ObjectReadBindings.hpp graph materialization | Conservative invalidation and pinned/current graph differences are legitimate. Not every concurrency interleaving verified. |
| Repository runtime `legacy`/`compatibility` search in vcobj native/Python | Matches were explicit rejection, source-file formats, offline conversion, projection defaults or comments; no additional runtime old-schema decoder proved. Keyword absence is not exhaustive API verification. |

Independent root-candidate adjudication is in `astra-root-adjudication.md`; root owns wider consumer coverage. All newly logged native candidates predate the latest fixes. No new working-diff regression established.

Unchecked / limitations: no exhaustive reread of every mutation family, every HTTP route, extension schema operation, copy/packing branch, all GUI editors or ML consumers; no macOS, runtime race, retained-history matrix or performance verification by this agent. Source attribution and explicit counterexamples distinguish FM-A01 from optional FM-A02. This bounded review does not claim zero defects or use finding counts as coverage estimates.
