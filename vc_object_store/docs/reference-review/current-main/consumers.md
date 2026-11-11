# Current-main merge: consumers and boundary review

Frozen HEAD: `7b08dfebd030168a6127f43e060301c55c7c20c3`; merged upstream `f0e441016`; cumulative comparison baseline `371961148`. Read-only production/test review. No build, installation, test run or environment change performed by this lane.

## Result

No concrete new vcobj consumer/reference defect or merge regression established. The cumulative merge changes no `vc_object_store` or `volume-cartographer` production file. Upstream Spiral memory changes alter tensor placement, not selected input identity; reader additions retain fixed-budget defaults; #2506 adapts source geometry/loader APIs, not vcobj schema or references. Tests remain coordinator-owned execution evidence.

## Fresh inventory and evidence reuse

Repeated direct discovery:

```sh
rg -l 'vc_object_store|vc::objects|ObjectStoreSession|ObjectRevision|ObjectVersion' volume-cartographer --glob '*.{cpp,hpp,h}' --glob '!**/test/**' --glob '!**/tests/**' --glob '!**/build/**'
rg -l 'vc_object_store|stored_surfaces|object_store_inputs|surface_store' spiral-fitting lasagna vesuvius/src --glob '*.py' --glob '!**/tests/**' --glob '!**/test_*.py'
git diff --name-only 371961148..HEAD -- vc_object_store volume-cartographer
git diff 371961148..HEAD -- spiral-fitting lasagna vesuvius vc_reader
rg -n 'request_channel_cache|MemoryCache\(.auto|MemoryCache\("auto' vc_object_store volume-cartographer spiral-fitting lasagna vesuvius/src --glob '*.{py,cpp,hpp}' --glob '!**/tests/**' --glob '!**/test/**' --glob '!**/build/**'
rg -n 'legacy_source|external:|sqlite3|SELECT.*objects|SELECT.*edges' volume-cartographer/apps/VC3D vc_object_store/python spiral-fitting/object_store_inputs.py lasagna/stored_surfaces.py --glob '*.{py,cpp,hpp}' --glob '!**/tests/**' --glob '!**/test/**'
rg -n 'patch_evaluation\.(center_theta|target_winding_indices|.*device)|evaluate_patch_satisfaction_packed\(' spiral-fitting --glob '*.py' --glob '!**/tests/**'
```

The seed inventory remains 109 native VC files and 24 direct Python files, with no newly added store-sensitive caller. Per-file/symbol dispositions from [post-closure/consumers.md](../post-closure/consumers.md), complete binding/facade inventory from [post-class-sweep/consumers.md](../post-class-sweep/consumers.md), and the broader call-chain review in [post-receipt/consumers.md](../post-receipt/consumers.md) are explicitly **reused unchanged source evidence**. They are not represented as a fresh independent line-by-line review of all rendering or solver algorithms. Current/exact selection, checked edits, mapping and membership guards, artifact payload routing, pending request recovery, complete receipt repinning and source-format siblings keep those recorded owning APIs.

## Cumulative changes reopened

| File / family | Source trace and disposition |
| --- | --- |
| `spiral-fitting/losses.py::_random_patch_pairs/get_patch_rel_winding_loss` | Direct distinct unordered pair sampling replaces constructing every combination; adjacent mode retains neighbor pairing. It consumes already materialized `points_by_patch`, not a database table or unversioned replacement source. Sampling/RNG consumption can differ from the old implementation; no selected-reference format, endpoint routing or saved cache identity is changed. Numerical optimizer redesign is outside this integration review. |
| `satisfaction_metrics.py::evaluate_patch_satisfaction_packed/_evaluate_patch_satisfaction_packed` | Public wrapper retries a bounded inference evaluation with per-quad tensors on CPU after device OOM. Model batches remain on fit device; `dr_quad`, theta, shifted radius, targets, reductions and residual outputs consistently use their declared storage device. No current-object read, repinning or metadata replacement occurs in the retry. CPU failure propagates, not a second source/data fallback. |
| `satisfaction_metrics.py` sibling entrypoints | Reopened `get_patch_satisfied_areas`, satisfaction reporting and direct fit evaluation callers: they call the same wrapper. Returned packed indices/corners and CPU profiles remain the existing application representation; only large evaluation tensors may change device. No serialization of an attachment-derived source identity was introduced. |
| `spiral_helpers.py::_build_spliced_overlay` | Gathers `target_winding_indices` and `center_theta` on the owning evaluation tensor device and transfers only the batch to model device. Eligibility/corner tables are CPU application indices. Its direct consumption of the evaluation result handles both normal and OOM-retry paths; it does not rebuild vcobj pins from patch ordinals. |
| `theta_crossing_map.py::invalidate/_refresh` and registered potential-source callbacks | Refresh drains prior checks, finalizes immutable source topology, invalidates freshness and releases old device tables before constructing replacements. Event-prefix buffers use `topology_device` (CPU); theta/gathers and published node-potential tables remain on model device, with chunk transfers explicit. Failure leaves refresh due. Source node IDs are resident model/topology indexing, not persistent object version identifiers. |
| `vc_reader` Python `MemoryCache`, `ZarrReader`, native bindings | Fixed numeric memory budgets still construct the existing native cache; auto budgeting and channel selection are opt-in. Pickle/fork reconstruction retains requested configuration and effective channel registration; wrappers do not infer dataset identity or modify vcobj references. Existing vcobj CT normal samplers and VC/Lasagna callers found by the cache search retain fixed budgets, so no silent auto-mode adoption occurs. Deep shared-cache concurrency/numerics belongs to the reader/native review lane, not this consumer report. |
| `lasagna/tiled_predict3d.py::_omezarr_chunk_exists` | Adds `FileNotFoundError` to absent-source handling alongside RuntimeError; source writer/open path remains `vc_reader`. Does not catch malformed vcobj receipt/reference errors or redirect store artifact resolution. |
| `vesuvius/.../ink_detection/data/geometry.py` | `resample` normal/point requests move from removed `smooth` spelling to shared `bicubic`; source patch reader owns interpolation and validity. No store metadata/owner interpretation added. |
| `vesuvius/.../neural_tracing/datasets/common.py`, `inference/common.py`, `scripts/labels_to_zarr.py` | Same shared source surface interpolation spelling update at points_at/resample_full boundaries. Source geometry sampling remains common reader functionality, not consumer-local interpolation or linked-store handling. |
| `vesuvius/.../fiber_trace_3d/loader.py` | Replaces einsum with fixed-order products implementing the same inverse-affine row multiplication, so tiling no longer picks a different BLAS reduction. Reopened following smoothing/source-coordinate call; tensor dimensions remain `(…,3)` with matrix columns broadcast to output coordinates. Does not alter selected-source reference, normal channel identity, or store persistence. Broader loader numerical accuracy is not re-proven by source review. |
| `vesuvius/.../ink_detection/preprocessing/download_required_zarr_chunks.py` | `_build_segment_download_patches` likewise changes `resample_full` interpolation spelling to `bicubic` before computing source crop bounds. Uses shared surface sampling rather than local interpolation or store SQL; does not change reference selection. |

## Existing shared contracts re-screened

- Native/Python receipt helpers remain the single strict decoder and `ReceiptDecodeError` outcome boundary. No merge hunk touches their map/number/canonical-UUID admission or full-reference receipt pins.
- Spiral `InputPins`, manifest input capture, fitting relationship policy, shell/full/incremental builders, Lasagna saved reference arrays and Vesuvius derived source metadata remain unchanged. Application roles/data wrap complete references rather than flattening them.
- Current-object discovery remains explicit; exact selected metadata and geometry use the same record. Geometry payload IDs and model node ordinals are not reference alternatives. Deliberate explicit CLI selectors and feed projection adaptation remain accepted boundaries.
- GUI checked edits continue carrying loaded `PinnedObject`, mapping and attached membership evidence; uncertain requests stay pending rather than being regenerated. No placement-specific Python or consumer SQL fallback surfaced in the repeated search.
- Independent source fiber/tifxyz/AFV formats and geometry algorithms are not legacy vcobj database schemas. Existing documented informational copy metadata remains the previously adjudicated external metadata API, not a new wrong-reference finding.

## Limitations

This is a source integration review, not a new runtime claim. Interactive GUI ordering, malformed hosted service injection, macOS/arm64 builds, full-volume performance, deep cache synchronization correctness and model convergence/sampling accuracy remain unverified here. Full prior file inventories are explicitly reused for unchanged files; the fresh scope is the complete cumulative merge delta and affected sibling boundary call chains, plus repeated repository discovery/contract screening. No new runtime tests or hidden compatibility workaround were added.
