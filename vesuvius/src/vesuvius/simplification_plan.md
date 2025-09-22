### IMPORTANT NOTES: 
- our focus is always on _clean , compact, and production ready code_ 
- we always pay attention to shapes and dtypes
- we focus on modularity where possible
- when creating new features , we do not leave compatibility layers or versioned functions around 
- before making changes, we first analyze what kinds of impacts this change will have downstream of it
- we always update the plan each time we complete a step, with a summary of what we changed (and why) and specific line numbers / examples
- we always read the plan or consult before continuing with next steps
- if something is ambiguous we ask followup questions
- WE DO NOT USE FALLBACKS, OR SILENTLY HANDLE ERRORS, WE FAIL HARD AND FAST IF SOMETHING IS UNEXPECTED
- we log extensively with the python standard logging module


### plan :
- [ ] **0. Package import cleanup** — align module layout and discovery with standard `src/vesuvius` packaging to eliminate path hacks and ambiguous imports
  - Map all import entry points and pain points (e.g. path mutations in `structure_tensor/run_create_st.py:19`, `models/training/loss/betti_losses.py:30`, `utils/napari_trainer/test_integration.py:10`) and document which modules expect top-level `vesuvius` vs relative imports.
  - Audit exported surface for collisions and implicit namespace packages, including the dual `utils` definitions (`utils/__init__.py:1`, `utils.py:1`) and root `__init__.py:5`, and decide the authoritative module structure.
  - Implement the reorganized package: relocate modules as needed, update import statements to absolute `vesuvius.*`, remove `sys.path` mutations, and ensure `pyproject.toml:1` / `setup.py:1` metadata matches the final layout.
  - Add enforcement and regression coverage (ruff import rules, `pytest -m importsmoke`, `python -m compileall src/vesuvius`) and document the import contract so future modules don't regress.
- This roadmap assumes a feature-forward overhaul: no compatibility shims, no fallbacks, and hard failures whenever invariants are violated. Legacy dataset code will be replaced outright.
- All new modules must emit actionable, structured logs via the standard `logging` package (no print statements), following project logging conventions and ensuring debug-level diagnostics are available during data pipeline builds.
- [ ] **1. Dataset simplification** — datasets are too complex; factor responsibilities out of monoliths
  - [x] Drafted refactor outline covering slicer abstractions and a unified dataset entrypoint; references `models/datasets/base_dataset.py:42`, `models/datasets/adapters/image_io.py:1`, `models/datasets/adapters/zarr_io.py:1`, `data/vc_dataset.py:20`
  - [x] Inventory current dataset responsibilities and patch sampling flows (BaseDataset, ImageAdapter, ZarrAdapter, NapariAdapter, VCDataset)
    - BaseDataset centralizes config ingestion, valid patch discovery (grid generation, caching), and slice-plane tooling before sampling weights are computed; see `models/datasets/base_dataset.py:49`, `models/datasets/base_dataset.py:290`, `models/datasets/base_dataset.py:390`, `models/datasets/base_dataset.py:458`, `models/datasets/base_dataset.py:512`.
    - BaseDataset also owns patch extraction, normalization, auxiliary target synthesis, augmentation pipelines, and `__getitem__`; see `models/datasets/base_dataset.py:1439`, `models/datasets/base_dataset.py:1464`, `models/datasets/base_dataset.py:1746`, `models/datasets/base_dataset.py:2022`.
    - ImageAdapter streams TIFF/PNG volumes directly from disk, handling metadata validation and label availability per target; see `models/datasets/adapters/image_io.py:23`, `models/datasets/adapters/image_io.py:72`, `models/datasets/adapters/image_io.py:186`.
    - ZarrAdapter loads pre-existing Zarr/OME-Zarr hierarchies and populates BaseDataset structures with optional unlabeled fallbacks; see `models/datasets/adapters/zarr_io.py:25`, `models/datasets/adapters/zarr_io.py:119`, `models/datasets/adapters/zarr_io.py:209`.
    - NapariAdapter hydrates BaseDataset directly from interactive napari viewers, cloning image/label/mask layers without disk I/O; see `models/datasets/adapters/napari_io.py:30`, `models/datasets/adapters/napari_io.py:90`, `models/datasets/adapters/napari_io.py:205`.
    - VCDataset provides a separate inference-only path backed by `Volume`, handling sliding-window splits, normalization, and empty patch skipping; see `data/vc_dataset.py:20`, `data/vc_dataset.py:77`, `data/vc_dataset.py:167`, `data/vc_dataset.py:186`, `data/vc_dataset.py:204`.
  - [x] Specify `PlaneSlicer` API for 2D/channel slices, including arbitrary plane extraction, caching hooks, and transform integration
    - Define `PlaneSliceConfig` to lift all current slice-related knobs off the dataset (axes, weights, per-plane patch sizes, mask mode, label thresholds, rotation/tilt policies, caching flags) drawn from `mgr` fields now read inside `models/datasets/base_dataset.py:120`.
    - `PlaneSlicer.register_volume(volume: PlaneSliceVolume)` ingests image arrays plus per-target labels/masks and stores lightweight metadata required for validation, reusing shape introspection currently embedded in `_collect_slice_patches_for_volume` (`models/datasets/base_dataset.py:546`) and `_extract_spatial_shape` (`models/datasets/base_dataset.py:602`).
    - `PlaneSlicer.build_index(validate: bool, cache: CacheHandle | None)` enumerates candidate patches per axis, applies label ratio/bbox thresholds, and returns `(patches, weights)`; this repackages `_collect_slice_patches_for_volume`, `_iter_plane_positions`, and the weighting loop in `models/datasets/base_dataset.py:522-545`.
    - `PlaneSlicer.extract(patch: PlaneSlicePatch, *, normalize: bool = True)` produces image/label tensors, masks, and metadata; it orchestrates axis-aligned vs rotated sampling (`models/datasets/base_dataset.py:873`, `models/datasets/base_dataset.py:919`), mask preparation (`models/datasets/base_dataset.py:1108`), and normalization/mask application from `_extract_slice_patch` (`models/datasets/base_dataset.py:1220`).
    - Expose `PlaneSlicePatch`/`PlaneSliceResult` dataclasses carrying orientation vectors and angle bookkeeping so downstream augmentation/logging can remain untouched, aligning with the payload built in `_extract_slice_patch` (`models/datasets/base_dataset.py:1391`).
    - Surface callbacks for cache hydration and normalizer injection so the dataset orchestrator can manage persistence and intensity transforms without the slicer reaching back into the ConfigManager.
    - Implemented the slicer module and dataset wiring; see `models/datasets/slicers/plane.py:1` and integration points in `models/datasets/base_dataset.py:90`, `models/datasets/base_dataset.py:280`, `models/datasets/base_dataset.py:465`, `models/datasets/base_dataset.py:1122`.
    - Added initial pytest coverage for slicer enumeration/extraction in `vesuvius/tests/models/datasets/test_plane_slicer.py:1`.
  - [x] Specify `ChunkSlicer` API for 3D/channel chunks, including tiling, stride controls, and patch weighting metadata
    - Introduce `ChunkSliceConfig` capturing patch size, optional strides, bbox/label thresholds, downsample level, caching knobs, and unlabeled allowances that BaseDataset currently pulls from `mgr` inside `_get_all_sliding_window_positions` and `_get_valid_patches` (`models/datasets/base_dataset.py:290`, `models/datasets/base_dataset.py:390`).
    - `ChunkSlicer.register_volume(volume: ChunkVolume)` records per-target zarr/numpy handles plus label presence so validation and extraction can reuse metadata now tracked implicitly via `self.target_volumes` (`models/datasets/base_dataset.py:406`, `models/datasets/base_dataset.py:418`).
    - `ChunkSlicer.build_index(validate: bool, cache: CacheHandle | None)` orchestrates cache lookup, parallel label screening with `find_valid_patches`, and cache persistence—mirroring the logic around `load_cached_patches`, `find_valid_patches`, and `save_valid_patches` (`models/datasets/base_dataset.py:406`, `models/datasets/base_dataset.py:418`, `models/datasets/find_valid_patches.py:50`, `models/datasets/save_valid_patches.py:47`).
    - `ChunkSlicer.enumerate(volume: ChunkVolume, stride: tuple | None)` provides the raw sliding-window iterator used when validation is disabled or labels are absent, lifting `_get_all_sliding_window_positions` (`models/datasets/base_dataset.py:290`).
    - `ChunkSlicer.extract(patch: ChunkPatch, *, normalizer: Callable | None)` returns image/label tensors plus `is_unlabeled` flags, handling padding, channel bookkeeping, and normalization exactly as `_extract_patch` and `__getitem__` do today (`models/datasets/base_dataset.py:1464`, `models/datasets/base_dataset.py:2022`).
    - Define `ChunkPatch` / `ChunkResult` dataclasses with volume id, start position, patch size, and optional weights so downstream trainers keep the same metadata contract currently embedded in BaseDataset dictionaries (`models/datasets/base_dataset.py:432`, `models/datasets/base_dataset.py:1473`).
  - [x] Design a single dataset orchestrator that composes slicers with pluggable I/O providers (TIFF, Zarr, Napari) in dedicated modules
    - Create a `DatasetOrchestrator` facade responsible for reading config knobs, instantiating data providers, wiring Plane/Chunk slicers, and exposing the PyTorch `Dataset` surface; this consolidates code now scattered across BaseDataset initialization (`models/datasets/base_dataset.py:120`), dataset factories (`models/datasets/sample_slice_batches.py:58`), and module exports (`models/datasets/__init__.py:1`).
    - Define `DataSourceAdapter` implementations (`ImageAdapter`, `ZarrAdapter`, `NapariAdapter`) that encapsulate the former `_initialize_volumes` logic from the legacy datasets (now implemented in `models/datasets/adapters/image_io.py:100`, `models/datasets/adapters/zarr_io.py:92`, `models/datasets/adapters/napari_io.py:82`), returning standardized `LoadedVolume` objects with image/label handles and metadata.
      - `datasets/adapters/base_io.py` exposes `DataSourceAdapter` (lifecycle: `discover() -> prepare() -> volumes`) plus shared dataclasses:
        - `AdapterConfig` captures mgr-derived knobs the adapter cares about (paths, cache flags, dtype policies) so adapters stay pure data-driven components.
        - `VolumeMetadata` records origin, spatial shape, channels, and label availability. This is logged when adapters emit volumes and allows orchestrator assertions without poking into backends.
        - `LoadedVolume` carries the `VolumeMetadata` alongside `ImageHandle`/`LabelHandle` callables for streaming reads. For TIFF streaming, `ImageHandle.read_window(start, size)` returns a NumPy view backed by `tifffile.TiffArray` without copying entire stacks.
      - Adapter workflow:
        1. `discover()` scans filesystem (or live sources) and yields `DiscoveredItem` records (path, target mapping, label path, errors). Fail fast when expectations are unmet (missing labels, duplicate ids).
        2. `prepare(items)` performs any one-off preprocessing (e.g., intensity stats gathering, metadata caching). For TIFF streaming we only open headers via `tifffile.TiffFile` to capture shapes/tiling without materializing pixels.
        3. `iter_volumes()` yields `LoadedVolume` objects wired to lazily open tiles on demand. No Zarr conversion occurs; `ImageHandle` delegates to `tifffile.asarray`/`aszarr` windows, and label handles use identical semantics.
      - Orchestrator integration contract:
        - Orchestrator passes mgr/config into adapter constructor, calls `discover()` immediately, and aborts if any entry reports `status='error'`.
        - After `prepare()`, orchestrator registers adapter-emitted `LoadedVolume` instances with slicers by constructing `PlaneSliceVolume` / `ChunkVolume` using the streaming handles and metadata.
        - Adapters never cache slices or touch slicer internals; they only own I/O and metadata validation. All patch enumeration stays inside slicers.
    - The orchestrator owns intensity statistics + normalization setup (reuse `initialize_intensity_properties` / `get_normalization`) and passes the resulting normalizer into slicers before index construction, removing direct `mgr` dependencies from the slicers (`models/datasets/base_dataset.py:189`, `models/datasets/base_dataset.py:200`).
    - `DatasetOrchestrator.build_index()` decides whether to invoke `PlaneSlicer` or `ChunkSlicer` (or both) based on config flags like `slice_sampling_enabled`, manages cache hydration through the shared cache API, and records per-patch sampling weights just as `_get_valid_patches`/`_get_valid_slice_patches` do today (`models/datasets/base_dataset.py:390`, `models/datasets/base_dataset.py:458`, `models/datasets/base_dataset.py:520`).
    - `DatasetOrchestrator.__getitem__` delegates to the appropriate slicer’s `extract` method, merges auxiliary targets, and attaches metadata so downstream trainers continue seeing the same dictionary layout (`models/datasets/base_dataset.py:1391`, `models/datasets/base_dataset.py:2022`).
    - Provide a thin factory (`build_dataset` replacement) that maps `mgr.data_format` to adapters and feeds the orchestrator, replacing the branching in `models/datasets/sample_slice_batches.py:58` and the legacy dataset classes once migrations complete.
  - [x] Decide on direct TIFF streaming strategy versus Zarr conversion; prototype to measure I/O latency, memory pressure, and caching needs
    - Image adapter now streams via `TiffArrayHandle` windows, respecting optional chunk-shape hints (`models/datasets/adapters/base_io.py:88`, `models/datasets/adapters/image_io.py:197`) and bypassing `.zarr` conversion entirely. Unit coverage verifies labeled/unlabeled flows plus handle chunking (`tests/models/datasets/test_image_adapter.py:13`).
    - Chunk and plane slicers pull patches directly from adapter handles without materialising full arrays, keeping cache/validation semantics intact (`models/datasets/slicers/chunk.py:327`, `models/datasets/slicers/plane.py:100`).
    - Follow-up: benchmark streaming vs legacy conversion on representative workloads and document any remaining performance trade-offs before deleting the old image/zarr pipelines.
  - [ ] Plan migration steps, regression tests, and documentation updates before switching consumers to the new dataset stack
    - Implementation status:
      1. `PlaneSlicer` / `ChunkSlicer` extracted and now own streaming patch extraction (`models/datasets/slicers/plane.py:100`, `models/datasets/slicers/chunk.py:327`).
      2. Image/Zarr/Napari adapters live under `models/datasets/adapters/` and feed the orchestrator (`models/datasets/adapters/image_io.py:193`). `models/datasets/__init__.py:1` now exports only `BaseDataset` and `DatasetOrchestrator`, with legacy names raising informative ImportError guidance.
      3. Orchestrator is wired through dataset factories and training entrypoints (`models/training/train.py:177`, `tests/models/datasets/test_dataset_orchestrator.py:8`), and Napari trainer hooks reference the napari adapter (`utils/napari_trainer/main_window.py:118`).
      4. CLI sampling and trainer now instantiate the orchestrator; README training section references the orchestrator-based flow (`README.md:17`, `README.md:86`).
      5. Streaming prototype complete; next step is benchmark write-up and decision record citing performance numbers (TODO).
      6. Cleanup backlog: refresh docs/notebooks post-removal and add cache round-trip regression once cache schema is finalized.
    - Test coverage to maintain/extend:
      - `tests/models/datasets/test_plane_slicer.py` (TODO to port to new streaming fixtures),
      - `tests/models/datasets/test_chunk_slicer.py`,
      - `tests/models/datasets/test_image_adapter.py`, `tests/models/datasets/test_zarr_adapter.py`, `tests/models/datasets/test_napari_adapter.py`,
      - `tests/models/datasets/test_dataset_orchestrator.py`,
      - Future: `tests/models/datasets/test_patch_cache_roundtrip.py` once cache schema solidified, plus perf benchmark hooks.
- [ ] **2. Aux task handling** — auxiliary task handling should move out of datasets
  - [x] Replace dataset-driven aux generation with trainer specialisations (`models/training/trainers/auxiliary/base_aux_trainer.py:1`). Each aux task now has its own lightweight trainer that injects derived targets post-augmentation, keeping datasets agnostic.
  - [x] Hook BaseTrainer with `_prepare_sample` / `_prepare_batch` so subclasses can compute auxiliary supervision without forking datasets (`models/training/train.py:214`, `models/training/train.py:901`).
  - [x] Simplify loss routing; BaseTrainer now evaluates losses directly while auxiliary trainers override `_compute_loss_value` to pass extra context (`models/training/train.py:924`, `models/training/trainers/auxiliary/base_aux_trainer.py:118`).
- [ ] **3. Config manager simplification** — trim bloat after dataset refactor clarifies dependencies
- [ ] **4. W&B logging cleanup** — centralize logging in a best-practice module



- **Completed**
  - Plane slicing extracted into `models/datasets/slicers/plane.py:1` with dataclasses/configuration and integrated logging.
  - Base dataset now delegates slice enumeration/extraction to the slicer (`models/datasets/base_dataset.py:95`, `models/datasets/base_dataset.py:280`, `models/datasets/base_dataset.py:465`, `models/datasets/base_dataset.py:1122`).
  - Added pytest coverage scaffold in `vesuvius/tests/models/datasets/test_plane_slicer.py:1` to exercise slicer index/extraction logic.
  - Validation guard now checks for present label arrays without triggering NumPy truthiness errors, restoring unlabeled-volume handling during index builds (`models/datasets/slicers/plane.py:180`).
  - `_slice_array_patch` no longer relies on channel-count heuristics; volumetric slices are normalized to channel-first patches and the plane slicer unit tests now pass cleanly (`models/datasets/slicers/plane.py:535`, `tests/models/datasets/test_plane_slicer.py:1`).
  - Chunk slicing implemented in `models/datasets/slicers/chunk.py:1`, including sequential validation fallback for constrained runtimes and integration with the shared cache API.
  - Base dataset now constructs and consumes chunk slicer patches instead of hand-rolled extraction logic (`models/datasets/base_dataset.py:361`, `models/datasets/base_dataset.py:405`, `models/datasets/base_dataset.py:483`).
  - Added chunk slicer pytest coverage exercising validated and unlabeled paths (`tests/models/datasets/test_chunk_slicer.py:1`).
  - Streaming dataset adapter scaffolding in `models/datasets/adapters/` with the TIFF-backed `ImageAdapter` and stubs for Napari/Zarr sources.
  - Adapter pytest coverage validating TIFF streaming, unlabeled allowances, and error paths (`tests/models/datasets/test_image_adapter.py:1`).
  - Zarr adapter streams OME/vanilla zarr hierarchies with windowed reads and metadata validation (`models/datasets/adapters/zarr_io.py:1`, `tests/models/datasets/test_zarr_adapter.py:1`).
  - Napari adapter hydrates datasets from viewer layers without GUI dependencies and carries contract tests (`models/datasets/adapters/napari_io.py:1`, `tests/models/datasets/test_napari_adapter.py:1`).
  - Added `DatasetOrchestrator` that instantiates adapters based on dataset format, reusing BaseDataset machinery while adapter tests and integration coverage lock in the new entrypoint (`models/datasets/orchestrator.py:1`, `tests/models/datasets/test_dataset_orchestrator.py:1`).
  - Chunk/plane slicers now consume streaming handles directly, removing eager NumPy materialisation from the dataset path and keeping caching/validation intact (`models/datasets/slicers/chunk.py:1`, `models/datasets/slicers/plane.py:1`).
  - Adapter knobs such as TIFF chunk shape are configurable through the ConfigManager and exercised via unit tests (`models/datasets/adapters/base_io.py:15`, `tests/models/datasets/test_image_adapter.py:59`).
  - Legacy dataset classes (`ImageDataset`, `ZarrDataset`, `NapariDataset`) have been removed; `models/datasets/__init__.py:1` now exposes only `BaseDataset` and `DatasetOrchestrator`, with legacy names guiding callers toward the appropriate adapter.
  - Image adapter supports PNG/JPG inputs through OpenCV-backed raster loading (`models/datasets/adapters/image_io.py:23`, `tests/models/datasets/test_image_adapter.py:134`).
