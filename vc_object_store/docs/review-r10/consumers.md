# R10 — Consumer release sweep

Frozen commit `433eed807`; workflow delta compared with `8859fc945`. Source review only. The R9 stale-source case is closed for newly produced expansion datasets, but remains reachable through existing expansion datasets at the paired-to-joint boundary.

## X-001 — Paired preparation upgrades an unverified legacy expansion stamp

`fiber_neighbor_expansion.run` now loads the parent tensor sidecar, checks its saved signature before sampling, records the sidecar hash and verified source pin, and supplies that pin to the source catalog and polyline reads. A same-UUID change before fresh expansion fails verification; a change between verification and geometry loading fails the pinned read. That fixes the fresh production path from R9.

`fiber_neighbor_pair.py:35–39,56` still accepts an existing expansion `inputs.json` with just the older live digest. Its `evidence.get('fiber_source')` yields `None`; `fiber_store.py::verify_fiber_source` then captures the **current** source position and compares the older expansion's digest against the current fibers. Pre-fix expansion wrote that digest from current geometry independently of its saved parent tensors. Thus an expanded dataset generated before this fix after a same-UUID edit can still pass paired preparation today if the store has not changed since that expansion. Paired writes a new complete pin to `prepared/inputs.json` and joint trusts it. The paired regression fixture explicitly accepts a digest-only input; this is not simply missing coverage.

This trust gap pre-dates `433eed807`; its persistence now conflicts with the documented requirement to regenerate preparation artifacts that did not record verified evidence. Fresh expansion, old expanded artifact consumption, and joint report-only are distinct paths. Require certified parent-derived evidence when preparing a **new** paired/joint solve from an expanded dataset; reject the unverified legacy stamp with an actionable regeneration message, and retain report-only behavior. No new solver semantics are needed.

## Other supported consumer families

- Selected-dataset direct samples, split and staged soft solve carry the selected source pin. Reference metrics and stamped prepared soft solves use the shared signature adapter. Joint uses verified parent positions and checks reused combined-store evidence; direct and split-search callers reach the same validator. Report-only dispatch remains ahead of new source checks.
- VC3D selected grid/fiber readers preserve exact references through transforms; geometry-only reads intentionally select through the shared retained-geometry API. Native grid read and Python/Lasagna/Vesuvius adapters preserve explicit validity; lazy channels carry selected geometry, and the displacement workflow records the loaded reference. Placement remains in shared readers.
- Checked VC3D surface/fiber/correction/Spiral save paths remain separate boundaries for loaded identity, mapping, membership and receipt references. Python resource and anchor adapters delegate validation to the native APIs. These families are unchanged from the R9 source review; no new blocker was established there.
