# PF-C04: Object manifest duplicates silently overwrite selected references

`spiral-fitting/object_store_inputs.py::validated_manifest_references` concatenates the objects and optional outer shell, then builds `{entry.uuid: decoded_reference}` with a dictionary comprehension. Repeated object UUIDs overwrite earlier entries, even when their complete source references differ. The same function explicitly rejects duplicate relationship UUIDs, so the two selected-evidence families have inconsistent validation.

Counterexample: a saved manifest contains two entries for the same object with different owning-store/version references (or reuses its object entry as the outer shell). Validation returns only the last reference rather than rejecting the ambiguous evidence. `InputPins.from_manifest` then uses that reduced mapping. Ordinary generated snapshots are unique; this is malformed saved-input rejection, not evidence that normal runs read wrong geometry.

Independent Astra adjudication accepts the duplicate-policy gap. Do not solve by another owner inference, arbitrary first/last choice or duplicated reference parser; use the existing shared reference validation and explicit unique-membership checks at manifest ingress. Older omission; runtime verification pending.
