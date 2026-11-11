# FM-P02 — Saved input positions weaken native validation through coercion

Older omission present at `0c7e4e521`, not a PF working-diff regression. Owner: `spiral-fitting/object_store_inputs.py::InputPosition.from_provenance`; independently accepted by Astra.

This saved/configuration boundary coerces `commit_sequence` with `int`, and store/lineage fields with `str`, before native selected-input validation sees them. Runtime probes show `1.5`, `True` and `'1'` all become commit sequence 1. The shared native `storePositionFrom` codec instead requires a nonnegative int64 and native UUID validation. The native strict codec cannot recover the original malformed value after Python has changed it.

Callers include `InputPins.from_provenance`, `InputPins.from_manifest`, configured `object_store_input_position`, resumed input status and `_startup_phase`. The sibling `spiral_service.py::_inputs_position` has its own stricter Python integer check plus legitimate application checks for service binding, requested minimum and receipt existence. Its semantic checks are justified; its basic position decoding should share the native boundary instead of being another schema interpreter. `of_status` receives trusted native data; casts of native receipts/statuses are not proof of malformed-ingress defects.

Expose/reuse one strict native position codec at external ingress, then retain the position intact with any application methods. Preserve the application wrapper containing `position`. Regression cases: fractions, booleans, strings, int64 overflow, malformed UUIDs and valid zero/integer sequences; saved pins/manifests and service request paths.
