# Task Log: managed Fiberlet preprocessing and publication

## 2026-08-23

- Planning only; no implementation has been started.
- The active `fiber-lets3` branch contains the native whole-volume command but
  currently predates the tracked `lasagna/manager` sources. Current
  `origin/main` contains the manager implementation. The implementation must
  integrate main first rather than copying manager code or using untracked
  bytecode.
- Existing manager orchestration is reusable but still names its generic
  process state, completion validation, and upload APIs as inference. The plan
  extracts one shared managed-job path and keeps inference records compatible.
- The native final combined Zarr already contains compact anchors, prefixes,
  and routes. Its separate float anchor Zarr is a durable resume cache and does
  not need to be uploaded.
- The native dataset fingerprint currently includes absolute local manifest
  paths and persists those paths in metadata. That is acceptable for an
  experimental local cache but not for a portable/open-data artifact. The plan
  separates runtime locations from stable manifest-hash/source identity before
  publication support is enabled.
- Atlas has no existing Fiberlet data type. Existing `lasagna` represents dense
  neural prediction bundles and requires a trained model ID plus Lasagna
  manifest/OME-Zarr layout, so a combined sparse Fiberlet graph must not be
  forced through that contract. The plan adds one minimal derived volume data
  type and no fake model.
- Local Fiberlet processing may precede publication of its inputs. Atlas
  validation/upload will require the referenced Fiber and normal prediction
  identities to exist so the published dependency graph is resolvable.
- Existing untracked compiled Python extensions under `lasagna/` were observed
  and left untouched.
- Independent review identified missing resume semantics, native completion
  validation, bounded-memory sparse upload, passthrough ownership checks, and
  several places where existing manager/Atlas helpers must be extracted rather
  than copied. The plan now includes those corrections.
- The portable metadata contract was clarified: it is not merely a settings
  fingerprint checked against caller configuration. The final Zarr must store
  every effective setting needed by a reader to construct compatible decoders
  and interpret coordinates. Canonical global settings and stable source
  identities feed the fingerprints; runtime filesystem paths do not. Paths
  remain only in private manager execution records.
