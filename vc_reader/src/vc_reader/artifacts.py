"""Data entries ("artifacts") and where to find them.

Every entity in metadata.json carries a ``data`` list. Each entry names an
artifact ``type``, one or more ``origins`` (alternative locations of the same
content), the ``parameters`` that identify it among artifacts of the same type,
and free-form ``properties`` / ``creation_info``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

OPEN_DATA_S3_ROOT = "s3://vesuvius-challenge-open-data"
OPEN_DATA_HTTPS_ROOT = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"

# s3:// <-> https:// equivalences for the two public channels. These match the
# rewrites used by the scrollprize.org data browser (src/data/atlasDataAccess.json).
HTTPS_REWRITES: tuple[tuple[str, str], ...] = (
    (OPEN_DATA_S3_ROOT + "/", OPEN_DATA_HTTPS_ROOT + "/"),
    ("s3://vesuvius-challenge/", "https://data.aws.ash2txt.org/samples/"),
)


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


# Volume-level artifacts that are OME-Zarr pyramids on a grid of the source
# volume's pyramid. The value says which source level the artifact's own level 0
# sits on; None means "read it from parameters.level".
_PYRAMID_GRID_LEVEL: dict[str, int | None] = {
    ArtifactType.OME_ZARR: 0,
    ArtifactType.INK_DETECTION_3D_ZARR: 0,
    ArtifactType.SURFACE_PREDICTION_ZARR: None,
}


def pyramid_sublevel(grid_level: int | None, volume_level: int) -> int | None:
    """Catalog-convention sublevel; negative means finer than the artifact."""
    return None if grid_level is None else volume_level - grid_level


def join_url(root: str, path: str) -> str:
    return root.rstrip("/") + "/" + path.lstrip("/")


def to_https(url: str) -> str | None:
    """HTTPS form of ``url``, or None if no public HTTPS equivalent is known."""
    if url.startswith(("https://", "http://")):
        return url
    for s3, https in HTTPS_REWRITES:
        if url.startswith(s3):
            return https + url[len(s3):]
    return None


def to_s3(url: str) -> str | None:
    """``s3://`` form of ``url``, or None if it is not under a known bucket."""
    if url.startswith("s3://"):
        return url
    for s3, https in HTTPS_REWRITES:
        if url.startswith(https):
            return s3 + url[len(https):]
    return None


@dataclass(frozen=True)
class AccessRoot:
    type: str
    url: str
    usage: str

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> AccessRoot:
        return cls(type=d.get("type") or "", url=d.get("url") or "", usage=d.get("usage") or "")


@dataclass(frozen=True)
class Origin:
    """One location of an artifact. All origins of an artifact hold the same content."""

    path: str
    access_roots: tuple[AccessRoot, ...]
    copy_info: Mapping[str, Any] | None = field(default=None, compare=False, repr=False)

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Origin:
        return cls(
            path=d.get("path") or "",
            access_roots=tuple(AccessRoot.from_json(r) for r in d.get("access_roots") or ()),
            copy_info=d.get("copy_info"),
        )

    def urls(self) -> tuple[str, ...]:
        return tuple(join_url(r.url, self.path) for r in self.access_roots)


@dataclass(frozen=True, eq=False)
class Artifact:
    """One data entry of a sample, volume or segment."""

    type: str
    origins: tuple[Origin, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict, repr=False)
    properties: Mapping[str, Any] = field(default_factory=dict, repr=False)
    creation_info: Mapping[str, Any] | None = field(default=None, repr=False)
    id: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Artifact:
        return cls(
            type=d.get("type") or "",
            origins=tuple(Origin.from_json(o) for o in d.get("origins") or ()),
            parameters=d.get("parameters") or {},
            properties=d.get("properties") or {},
            creation_info=d.get("creation_info"),
            id=d.get("id"),
            raw=d,
        )

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.parameters.items())
        return f"Artifact({self.type!r}{', ' + params if params else ''}, path={self.path!r})"

    # -- identity -----------------------------------------------------------

    @property
    def path(self) -> str | None:
        """Path of the first origin (relative to its access root)."""
        return self.origins[0].path if self.origins else None

    @property
    def is_directory(self) -> bool:
        """True for prefix artifacts (zarr stores, tifxyz folders, bundles)."""
        return bool(self.path and self.path.endswith("/"))

    @property
    def target_volume_id(self) -> str | None:
        return self.parameters.get("target_volume")

    @property
    def model_id(self) -> str | None:
        return self.parameters.get("model_id")

    @property
    def label_set(self) -> str | None:
        return self.parameters.get("label_set")

    @property
    def render_level(self) -> int:
        """Pyramid level of ``target_volume`` a render was computed from (0 when absent)."""
        return int(self.parameters.get("render_level") or 0)

    @property
    def level(self) -> int | None:
        """``parameters.level``: the source pyramid level a volume prediction ran on."""
        level = self.parameters.get("level")
        return None if level is None else int(level)

    @property
    def grid_level(self) -> int | None:
        """Source-volume pyramid level that this artifact's own level 0 lies on.

        Defined for OME-Zarr pyramids on the volume grid (``ome-zarr``,
        ``ink-detection-3d-zarr``, ``surface-prediction-zarr``); None otherwise,
        and for legacy surface predictions whose level is unknown.
        """
        if self.type not in _PYRAMID_GRID_LEVEL:
            return None
        fixed = _PYRAMID_GRID_LEVEL[self.type]
        return fixed if fixed is not None else self.level

    # -- locations ----------------------------------------------------------

    def urls(self) -> tuple[str, ...]:
        """Every location as listed, in metadata order."""
        return tuple(u for o in self.origins for u in o.urls())

    def url(self, scheme: str | None = "https") -> str | None:
        """Preferred location of the artifact.

        ``scheme`` is ``"https"`` (default; anonymous HTTP access), ``"s3"``, or
        None to keep the location's own scheme. The open-data bucket is
        preferred over other channels; otherwise metadata order is kept.
        """
        convert = {"https": to_https, "s3": to_s3, None: lambda u: u}.get(scheme)
        if convert is None:
            raise ValueError(f"unknown scheme {scheme!r}; use 'https', 's3' or None")
        ranked = []
        for i, o in enumerate(self.origins):
            for r in o.access_roots:
                url = convert(join_url(r.url, o.path))
                if url is not None:
                    open_data = r.url.rstrip("/") in (OPEN_DATA_S3_ROOT, OPEN_DATA_HTTPS_ROOT)
                    ranked.append((not open_data, i, url))
        return min(ranked)[2] if ranked else None

    @property
    def usage(self) -> tuple[str, ...]:
        """Distinct access classes of the artifact's roots (``public-read`` in published data)."""
        return tuple(sorted({r.usage for o in self.origins for r in o.access_roots}))


def parse_artifacts(entries: Any) -> tuple[Artifact, ...]:
    return tuple(Artifact.from_json(e) for e in entries or ())
