# PF-A01 — Surface-grid adapter accepts a differently selected transformed view

Baseline: `153c9a10f1cca90f57a2e70dc5458d1bd5bad2a4`. Candidate correctness issue, medium priority; source-traced, not runtime reproduced. Older omission: identical geometry-only check exists at `0c7e4e521:src/SurfaceGrid.cpp`.

Owner: `src/SurfaceGrid.cpp::readSurfaceGrid(reader, ObjectRevision, TransformedGeometryView*, bool)`. It checks only `transformed->revision().uuid == canonical.revision().uuid`, copies points from the supplied transformed view, then labels them with `selected.record`.

Counterexample: select object revision A with coordinate scale 1, then metadata-only revision B with coordinate scale 2; both retain geometry G. Construct `snapshot.surface(B).viewTransformedAt(volume)` and pass that view with A into `readSurfaceGrid`. The check accepts G==G, returns B-transformed points but A's metadata/reference. This also applies to inherited metadata-only overrides sharing a payload, including owners with equal counters. It does not require stale or malformed data.

Sibling paths: current/full-reference convenience overloads in the same file construct a matching view and are safe by construction. Python session `read_surface_grid` delegates those convenience paths. `volume-cartographer/core/src/QuadSurfaceStore.cpp::load_quad_from_store` forwards externally supplied selected state and transformed view to this unchecked boundary; SurfaceCatalog and SpiralStoreInputs currently construct matching pairs, but the shared API does not enforce its invariant. Stored-coordinate reads with no transformed view are unaffected. Inverse-edit validation is separate and does not validate this read result.

Recommendation: compare the complete selected identity exposed by the transformed view's canonical source, not only its geometry revision. All just-migrated production selected-view callers now retain that identity. Do not invent a parallel tuple. Decide explicitly whether an anonymous payload-only transformed view is allowed at this selected-metadata API; lack of identity must not be treated as proof it used the supplied metadata. Keep payload-only reads themselves available elsewhere.

Required checks: A/B same-payload coordinate revisions; differing owners with equal numeric versions; valid same-selection reads; retained A after override/deletion; history-on/off where the selected versions remain available; native wrapper boundary as well as direct core API. No runtime cases verified in this review.
