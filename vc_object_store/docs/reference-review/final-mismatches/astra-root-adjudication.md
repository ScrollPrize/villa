# Independent adjudication — additional root candidates

Baseline: `153c9a10f1cca90f57a2e70dc5458d1bd5bad2a4` plus current working fixes. Source only; no production/test edits or execution.

## grow_track_graph retirement guards

Accept as a concrete current-API mismatch. `spiral-fitting/grow_track_graph.py::retire_surfaces` builds `{uuid: db.current_version(uuid)}` and passes the integer values to `db.write(expected_versions=...)`. `python/objects.cpp` routes that argument through shared `recordSelectionFromJson`; `Commands.cpp::selectionFrom` accepts a complete reference/null and rejects a numeric value. Thus the live-surface retirement path raises before executing retirement. This is not an external numeric discovery API: the integer is being used as the actual write guard after discovery.

The exact code exists at `0c7e4e521`, so it is an older missed consumer migration, not introduced by the latest working patch. Migrate the selected records to complete references through shared APIs and retain normal no-live/tombstone semantics; do not restore numeric guard compatibility. The root agent owns caller reachability and runtime proof for the enabled repair workflow.

## InputPosition.from_provenance numeric coercion

Accept as a malformed-input policy mismatch. `spiral-fitting/object_store_inputs.py::InputPosition.from_provenance` calls `int(value['commit_sequence'])`, accepting/truncating fractions, bools and numeric strings before the strict native position decoder can inspect the original value. A saved position 1.5 becomes position 1, and `True` becomes 1. This duplicates and weakens the shared position boundary policy. Store/generation are also coerced with `str` rather than initially validated as native identifiers.

The implementation exists at `0c7e4e521`. It is an older ingress omission, not a new selected-input codec regression. Use the shared native position validation rather than a second Python numeric-rule table. `of_status` receives trusted native status and is a less risky adaptation, but can share the same representation boundary. Preserve the legitimate wrapper containing `position`; do not mistake that application wrapper for a second reference format. Test invalid fraction/bool/string/overflow, valid zero/integer, and saved pins/manifests/configuration paths.

PF-C03 automatic repinning is not reopened: ToolObjects retains read-baseline semantics as previously adjudicated.
