# MeshLab OBJ segment compatibility fix

User sees only first segment per fiber plus isolated vertices. Existing exporter
writes one multi-index OBJ l record per fiber. Replace with explicit two-index l
records for every consecutive pair, preserving original vertices/order/groups.
Share helper for ordinary and broken-overlay exports. Regression tests verify
all edges and no cross-fiber connection. Repair existing two OBJ files in place
programmatically via approved runner, validating against index.json ranges before
publication. PLY arrows/metadata positions/IDs stay unchanged; no GPU/inference.

Discovered during repair:shared raw reader dropped first point via setdefault
evaluation order. Correct reader and regenerate OBJ from original AFV; update
OBJ vertex ranges in index.json. Verify original coordinates in regression tests.
Spec update: output compatibility only. Update docs/status/task log/changelog;
independent review per AGENTS. MeshLab GUI verification unavailable locally.
