# Dataset Pipeline Guide

This guide explains how the dataset subsystem is structured, how the adapter and slicer abstractions cooperate with `BaseDataset`, and how to extend the pipeline with additional adapters or slicing strategies.

## High-Level Flow

- A dataset configuration object (`mgr`) is passed into `BaseDataset` (`src/vesuvius/models/datasets/base_dataset.py:45`), which stores runtime options such as patch size, augmentation flags, target definitions, and sampling preferences.
- `_initialize_volumes()` is invoked during construction. In `DatasetOrchestrator` (`src/vesuvius/models/datasets/orchestrator.py:23`) this method loads data by running a configurable adapter and populating `self.target_volumes` with image/label handles.
- Once volumes are registered, `BaseDataset` decides whether to instantiate a `ChunkSlicer` or `PlaneSlicer` based on `slice_sampling_enabled`. Each slicer enumerates valid patches and exposes extraction helpers.
- Intensity statistics are computed (or reused from cache) to initialise a normaliser. Augmentation pipelines are created if `is_training=True`.
- During iteration, `__getitem__` pulls a patch description from the slicer, extracts image/label tensors, runs optional augmentations, and returns a dictionary ready for a model.

## BaseDataset Responsibilities

`BaseDataset` implements the shared mechanics for all dataset variants.

- **Volume registry**: `_initialize_volumes()` must populate `self.target_volumes` with dictionaries containing `image`, `label`, `label_path`, and related metadata (`src/vesuvius/models/datasets/base_dataset.py:125`).
- **Slice strategy selection**: `_setup_plane_slicer()` configures planar sampling, while `_setup_chunk_slicer()` configures volume chunks (`base_dataset.py:214` and `base_dataset.py:248`). Only one slicer is active at a time.
- **Patch validation**: `_get_valid_patches()` triggers the chosen slicer to enumerate patches and enforces label coverage rules such as `min_labeled_ratio` and `min_bbox_percent` (see `base_dataset.py:363`).
- **Normalisation & augmentation**: Intensity properties are gathered via `initialize_intensity_properties`, then `get_normalization()` produces a callable used by slicers. Augmentation pipelines are composed in `_create_training_transforms()` and optionally `_create_validation_transforms()`.
- **Runtime accessors**: Helper methods (e.g. `_get_entry_image`, `_get_entry_label`) provide a uniform view over the per-volume dictionaries. `__getitem__` orchestrates extraction, augmentation, and metadata packaging (`base_dataset.py:629`).

`DatasetOrchestrator` extends `BaseDataset` and provides a ready-made `_initialize_volumes()` that talks to adapters. Subclasses can override `_initialize_volumes()` if they need different sourcing behaviour.

## Adapter Abstractions

Adapters translate external storage (filesystems, zarr stores, viewers) into the in-memory structures `BaseDataset` expects.

### Core data structures

Defined in `src/vesuvius/models/datasets/adapters/base_io.py`.

- `AdapterConfig`: immutable, validated configuration shared with all adapters (data root, label policy, file naming, resolution hints).
- `DiscoveredItem`: a lightweight record pointing at an image path and optional per-target label paths.
- `VolumeMetadata`: detailed metadata gathered during preparation (spatial shape, dtypes, axis labels, label presence).
- `ArrayHandle` and its concrete subclasses (`TiffArrayHandle`, `ZarrArrayHandle`, `NumpyArrayHandle`): thin wrappers that expose `.read()` and `.read_window()` so slicers can operate lazily without loading entire volumes.
- `LoadedVolume`: the normalised container emitted by adapters, bundling metadata with image/label handles.
- Mesh-specific helpers live under `models/datasets/mesh/`: `MeshMetadata` and `MeshPayload` describe polygonal surfaces, `MeshHandle` lazy-loads them, `LoadedMesh` couples metadata with handles, and `mesh_to_binary_voxels` converts surfaces into occupancy grids when needed.

### DataSourceAdapter contract

Every adapter subclasses `DataSourceAdapter` (`base_io.py:208`) and implements three phases:

1. `discover()`: scan backing storage and yield `DiscoveredItem` entries. Fail early when files are missing or malformed.
2. `prepare(discovered)`: validate discovered items, probe array metadata, and populate an internal metadata map.
3. `iter_volumes()`: instantiate array handles and yield fully prepared `LoadedVolume` objects.

`run()` orchestrates the phases for convenience but `DatasetOrchestrator` calls each method explicitly to retain access to intermediate data.

### Built-in adapters

| Adapter | Purpose | Notable behaviour |
| --- | --- | --- |
| `ImageAdapter` (`adapters/image_io.py`) | Streams TIFF or raster stacks from `images/` plus per-target labels from `labels/`. | Uses `TiffArrayHandle` for windowed reads when files are TIFF; accepts grayscale PNG/JPEG fallbacks. Enforces spatial shape parity between images and labels. |
| `ZarrAdapter` (`adapters/zarr_io.py`) | Reads OME-Zarr or plain zarr hierarchies. | Supports per-target zarr groups suffixed with `_target`. Respects `ome_zarr_resolution` to pick pyramid levels. Reuses zarr objects for zero-copy slicing. |
| `NapariAdapter` (`adapters/napari_io.py`) | Pulls data directly from an in-memory napari viewer. | Uses faux paths to satisfy `LoadedVolume` but keeps arrays in RAM. Matches label layers named `{image_layer}_{target}`. |
| `MeshAdapter` (`models/datasets/mesh/filesystem.py`) | Loads `.ply` / `.obj` meshes alongside raster volumes. | Supports JSON/YAML manifests, per-mesh transforms, and mappings that associate each mesh with its source volume so downstream slices can surface the geometry. |

### Mesh and vector payloads

Meshes are optional. Enable them via `dataset_config.meshes` (or `mesh_config` when constructing `DatasetOrchestrator`). Example:

```yaml
dataset_config:
  meshes:
    enabled: true
    adapter: mesh
    dirname: meshes
    manifest: meshes/index.yaml
    default_source_volume: volume_A
    source_map:
      relief_mesh: volume_B
    transform_map:
      relief_mesh: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
```

The orchestrator instantiates the requested mesh adapter, links meshes to raster volumes, and injects `MeshHandle` objects into each `target_volumes` entry. Dataset samples then expose a `meshes` dictionary alongside images/labels, and `mesh_to_binary_voxels` can turn any handle into a boolean occupancy grid for auxiliary tasks.

Vector-valued rasters (normals, tangent frames, UV directions, etc.) can be declared through `dataset_config.vector_targets`, `ConfigManager.vector_targets`, or per-target flags like `vector: true` / `type: vector`. Targets whose names include `normal`, `t_u`, `t_v`, `uv`, or `frame` are treated as vectors automatically.

Whenever meshes or vector labels are present in a patch, spatial augmentations (`SpatialTransform`, `MirrorTransform`, `TransposeAxesTransform`, `SmearTransform`, and their `RandomTransform` wrappers) are skipped so orientation-dependent signals remain valid. Intensity-only transforms still execute.

## Slicer Strategies

Slicers take registered volumes and generate patch descriptors plus extraction helpers.

### ChunkSlicer (volumetric patches)

Located at `src/vesuvius/models/datasets/slicers/chunk.py`.

- **Configuration**: `ChunkSliceConfig` defines patch size, stride, label coverage thresholds, downsampling for validation, worker count, and cache parameters.
- **Volume registration**: `ChunkVolume` stores lazy image handles, optional label handles, and paths used for caching. Registration enforces target availability (`ChunkSlicer.register_volume`).
- **Index building**: `build_index(validate=...)` decides whether to run `find_valid_patches` (parallel or cached) or enumerate every position. Results are stored as `ChunkPatch` objects with resolved coordinates and optional weights.
- **Extraction**: `extract()` reads windows with `read_window` when available, applies normalisation, and returns a `ChunkResult` containing tensors, label dict, `is_unlabeled` flag, and metadata (plane, angles, etc.). Padding utilities ensure patches fit requested shapes.

Chunk slicing is used whenever `slice_sampling_enabled` is false. Patch caches are persisted under `.patches_cache/` inside the data directory when enabled.

### PlaneSlicer (planar sampling from 3D)

Located at `src/vesuvius/models/datasets/slicers/plane.py`.

- **Configuration**: `PlaneSliceConfig` specifies which axes to sample (`sample_planes`), patch sizes per plane, weighting, and optional random yaw/tilt parameters. Label interpolation policies control whether bilinear interpolation is applied to labels per target.
- **Volume registration**: `PlaneSliceVolume` stores lazy handles and caches numpy arrays on demand. Plane masks can be accumulated either per-plane or across a volume depending on `plane_mask_mode`.
- **Index building**: For each registered volume and plane, `_collect_patches_for_volume()` enumerates slice indices and in-plane offsets. Optional validation ensures patches satisfy bounding box and label ratio thresholds.
- **Extraction**: `extract()` slices axis-aligned planes or samples rotated/tilted planes using `torch.nn.functional.grid_sample` under the hood. It returns `PlaneSliceResult` with image tensor, per-target labels, `is_unlabeled`, optional plane mask, and metadata describing plane orientation.

Plane slicing is enabled when `mgr.slice_sampling_enabled` is true. `BaseDataset` supplies per-plane patch sizes, weighting, and tilt/rotation options from the config manager.

## Patch Validation & Augmentation

- The slicers emit `valid_patches` records that `BaseDataset` exposes publicly. Each entry includes volume index/name, position, and patch size (plus plane metadata for slice mode).
- When validation is enabled, only patches exceeding the configured label density and bounding box coverage are retained. Otherwise, every enumerated chunk or plane is accepted.
- Augmentations run in `__getitem__` unless `augment_on_device` is set. Spatial/intensity pipelines include rotations, mirroring, contrast, blur, noise, and optional skeleton transforms for losses that require distance fields.

## Implementing a New Adapter

1. **Create a subclass** of `DataSourceAdapter` in `src/vesuvius/models/datasets/adapters/`. Start by copying the structure of `ImageAdapter` and tailor `discover/prepare/iter_volumes` to your source.
2. **Accept configuration** by extending `AdapterConfig` if you need additional parameters (e.g., credentials, dataset IDs). Prefer optional fields so existing adapters remain compatible.
3. **Produce `LoadedVolume` records** that wrap images/labels in `ArrayHandle` implementations. Implement a new handle if your storage supports efficient windowing.
4. **Register the adapter** in `DatasetOrchestrator._ADAPTERS` (`orchestrator.py:23`) so it can be selected via configuration, e.g. `adapter="my_new_source"`.
5. **Expose config knobs** through the training/config manager (`mgr`). The orchestrator forwards `adapter_kwargs` when instantiating the adapter, so any runtime-only arguments can be supplied there.

Skeleton:

```python
from .base_io import AdapterConfig, DataSourceAdapter, DiscoveredItem, LoadedVolume

class MyAdapter(DataSourceAdapter):
    def discover(self) -> Sequence[DiscoveredItem]:
        # inspect storage, build DiscoveredItem entries
        ...

    def prepare(self, discovered: Sequence[DiscoveredItem]) -> None:
        # validate shapes/dtypes and cache metadata
        ...

    def iter_volumes(self) -> Iterator[LoadedVolume]:
        # wrap arrays in ArrayHandle implementations and yield LoadedVolume
        ...
```

## Implementing a New Slicer

1. **Define configuration and data containers** similar to `ChunkSliceConfig`/`ChunkPatch` or `PlaneSliceConfig`/`PlaneSlicePatch`. Include all metadata the dataset will need downstream.
2. **Implement a slicer class** with methods to `register_volume`, `build_index`, and `extract`. Normalisation is provided by `BaseDataset.set_normalizer`.
3. **Integrate with `BaseDataset`** by subclassing `DatasetOrchestrator` (or another `BaseDataset` derivative) and overriding `_setup_chunk_slicer`/`_setup_plane_slicer` (or `_initialize_volumes`) to instantiate your new slicer when applicable.
4. **Return patch metadata** that mirrors the existing structure so downstream dataloader logic (e.g., caching, weighting) works unchanged. Update `BaseDataset.__getitem__` or your subclass if additional outputs are required.
5. **Add tests** under `tests/models/datasets` to cover enumeration and extraction edge cases (empty volumes, non-square patches, tilt angles, etc.).

Example integration sketch:

```python
class MyDataset(DatasetOrchestrator):
    def _setup_chunk_slicer(self) -> None:
        if self.slice_sampling_enabled:
            return  # keep plane behaviour
        config = MySliceConfig(...)
        slicer = MySlicer(config=config, target_names=list(self.target_volumes))
        for idx, info in enumerate(self.target_volumes[first_target]):
            slicer.register_volume(convert_volume(idx, info))
        self.chunk_slicer = slicer
```

## Troubleshooting Tips

- Enable logging at DEBUG level for adapters and slicers to inspect discovery and patch enumeration steps.
- If validation returns zero patches, verify label coverage thresholds and ensure `allow_unlabeled_data` is set appropriately.
- When adding adapters, confirm `VolumeMetadata.spatial_shape` matches across image and label sources; slicers rely on consistent dimensions.
- Use the cached `.patches_cache/` directory to persist expensive validation runs; clear it when underlying labels change.
