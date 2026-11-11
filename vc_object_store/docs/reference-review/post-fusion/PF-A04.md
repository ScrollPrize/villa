# PF-A04 — Exact graph projection validation invokes dummy current queries

Baseline: `153c9a10f1cca90f57a2e70dc5458d1bd5bad2a4`. Simplification/wrong-layer-plumbing candidate, low priority; no demonstrated incorrect graph data. Older leftover present at `0c7e4e521`.

Owner: `python/ObjectReadBindings.hpp`, `GraphReadView.objects(fields)` and `.edges(fields)`. For an exact graph with retained native selected records, each binding constructs a fresh selection with `limit=0` and executes `selectObjects` / `selectEdges` solely to validate projection names; it discards the result and projects the retained records in Python adaptation.

Native policy already resides in `internal/RecordSql.cpp::projectedObjectColumns` and `internal/EdgeSql.cpp::projectedEdgeColumns`, but field validation is coupled to SQL column generation. This does not currently replace exact records with moving heads: the returned data correctly comes from `selectedObjects` / `selectedRelationships`. Current graph queries legitimately use the retained transaction to fetch records and are not this finding.

Recommendation: expose shared native projection validation or projection of an already-selected record, extracting the existing accepted-field policy rather than duplicating lists in bindings. Both query construction and retained-record projection should delegate. Do not introduce another query merely for validation, and do not remove invalid-field rejection.

Required checks: identical accepted/rejected fields for current and exact objects/edges, including empty result sets, owner identity fields and retained inherited history. Source review only; no latency claim.
