# PF-A05 — Generation-only numbered selector silently becomes current discovery

Baseline: `153c9a10f1cca90f57a2e70dc5458d1bd5bad2a4`. Candidate input-validation inconsistency, medium priority; source-traced. Older omission: unchanged `StoreReader::selectedObject` behavior at `0c7e4e521`.

Owner: `src/StoreReader.cpp::selectedObject(uuid, version, generation, reference)`. With no complete reference, it resolves `currentObject(uuid)` and uses the supplied generation only if `version` is present. A generation with no version is silently ignored, even if malformed or naming a different lineage.

Counterexample: call native selectedObject(id, nullopt, unrelatedGeneration), or Python `export_geometry(id, destination, generation=unrelatedGeneration)` without a version. The export binding enters the explicit-selection branch because generation is present, but the shared selector returns the current object. The output does not correspond to the caller-supplied lineage and there is no error. Surface-grid numeric convenience selection delegates the same selector. This is not a complaint about permitted explicit version-only CLI selectors: generation alone is not a usable numbered selector.

Sibling comparison: complete-reference selection rejects mixing version/generation, and `decodeObjectReference` rejects generation-only structured inputs. Current discovery with neither version nor generation is legitimate and should remain. Exact complete references must continue working after current deletion. History-on/off and local/hosted readers share this base method, so the rejection belongs here, before current lookup.

Recommendation: reject generation without version centrally; retain current discovery, explicit numbered-current-owner selection, and exact-reference selection as distinct valid cases. Tests should exercise native, surface-grid and export boundaries, checking no file is written for an invalid selector. No runtime checks performed.
