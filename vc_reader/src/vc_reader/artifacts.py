"""Data entries ("artifacts") and where to find them.

Every entity in metadata.json carries a ``data`` list. Each entry names an
artifact ``type``, one or more ``origins`` (alternative locations of the same
content), the ``parameters`` that identify it among artifacts of the same type,
and free-form ``properties`` / ``creation_info``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
from . import _catalog

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
    return _catalog.join_url(root, path)


def to_https(url: str) -> str | None:
    """Public HTTPS location resolved by the shared native catalog."""
    return _catalog.to_https(url)


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
    _parsed: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def _native_values(self):
        if not self._parsed:
            native = _catalog.parse_artifacts([self.raw or dict(type=self.type,
                parameters=dict(self.parameters), properties=dict(self.properties), origins=[
                dict(path=o.path, access_roots=[dict(type=r.type, url=r.url, usage=r.usage) for r in o.access_roots])
                for o in self.origins])])[0]
            object.__setattr__(self, '_parsed', dict(level=native.level, target=native.target_volume_id,
                model=native.model_id, url=native.url, source_path=native.source_path))
        return self._parsed

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Artifact:
        return cls.from_native(_catalog.parse_artifacts([d])[0])

    @classmethod
    def from_native(cls, native) -> Artifact:
        d = native.raw
        return cls(type=native.type,
                   origins=tuple(Origin(o.path, tuple(AccessRoot(r.type, r.url, r.usage) for r in o.access_roots),
                                        o.raw.get('copy_info')) for o in native.origins),
                   parameters=native.parameters, properties=native.properties,
                   creation_info=native.creation_info or None, id=d.get('id'), raw=d,
                   _parsed=dict(level=native.level, target=native.target_volume_id, model=native.model_id,
                                url=native.url, source_path=native.source_path))

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
        return self._native_values()["target"]

    @property
    def model_id(self) -> str | None:
        return self._native_values()["model"]

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
        return self._native_values()["level"]

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
        if scheme not in ('https', 's3', None):
            raise ValueError(f"unknown scheme {scheme!r}; use 'https', 's3' or None")
        values = self._native_values()
        url = values['url'] if scheme == 'https' else values['source_path']
        return (to_s3(url) if scheme == 's3' else url) or None

    @property
    def usage(self) -> tuple[str, ...]:
        """Distinct access classes of the artifact's roots (``public-read`` in published data)."""
        return tuple(sorted({r.usage for o in self.origins for r in o.access_roots}))


def parse_artifacts(entries: Any) -> tuple[Artifact, ...]:
    return tuple(Artifact.from_native(a) for a in _catalog.parse_artifacts(entries or []))
