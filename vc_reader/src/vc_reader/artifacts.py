"""Data entries ("artifacts") and where to find them (``vcr::catalog``).

Every entity in metadata.json carries a ``data`` list. Each entry names an
artifact ``type``, one or more ``origins`` (alternative locations of the same
content), the ``parameters`` that identify it among artifacts of the same type,
and free-form ``properties`` / ``creation_info``.
"""
from ._catalog import AccessRoot, Artifact, Origin, join_url, parse_artifacts, to_https, to_s3


class ArtifactType:
    """Artifact ``type`` strings documented by metadata.schema.json.

    New types appear over time; unknown types are kept as plain artifacts.
    """

    # sample
    PHOTO = "photo"
    PHOTO_MASK = "photo-mask"
    # volume
    OME_ZARR = "ome-zarr"
    SURFACE_PREDICTION_ZARR = "surface-prediction-zarr"
    NORMAL_GRIDS = "normal-grids"
    LASAGNA = "lasagna"
    FIBERLETS = "fiberlets"
    INK_DETECTION_3D_ZARR = "ink-detection-3d-zarr"
    UMBILICUS = "umbilicus"
    # segment meshes
    TIFXYZ = "tifxyz"
    TIFXYZ_FLATTENED = "tifxyz-flattened"
    TIFXYZ_NORMALIZED = "tifxyz-normalized"
    TIFXYZ_TRANSFORMED = "tifxyz-transformed"
    OBJ = "obj"
    OBJ_LEGACY_UNITS = "obj-legacy-units"
    OBJ_FLATTENED = "obj-flattened"
    OBJ_NORMALIZED = "obj-normalized"
    # segment renders / predictions / labels (all keyed by target_volume)
    LAYERS_ZARR = "layers-zarr"
    LAYERS_TIF = "layers-tif"
    ALPHA_RENDER = "alpha-render"
    ALPHA_RENDER_DOWNSAMPLED = "alpha-render-downsampled"
    INK_DETECTION = "ink-detection"
    INK_DETECTION_DOWNSAMPLED = "ink-detection-downsampled"
    INK_3D_PREDICTION = "3d-ink-prediction"
    INK_LABELS = "ink-labels"
