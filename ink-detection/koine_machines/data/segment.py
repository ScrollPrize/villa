from pathlib import Path

_LABEL_SUFFIXES = (
    ("_inklabels", "inklabels"),
    ("_supervision_mask", "supervision_mask"),
    ("_validation_mask", "validation_mask"),
)
_LABEL_EXTENSIONS = {".tif", ".tiff", ".zarr"}


def _normalize_label_version(label_version):
    if label_version in (None, ""):
        return None
    if isinstance(label_version, str):
        value = label_version.strip().lower()
        if value in {"", "auto", "latest"}:
            return None
        if value in {"base", "unversioned", "v1"}:
            return 1
        if value.startswith("v") and value[1:].isdigit():
            version_num = int(value[1:])
            if version_num < 1:
                raise ValueError(f"label_version must be >= v1, got {label_version!r}")
            return version_num
        raise ValueError(
            f"label_version must be one of None/'auto', 'base', or 'vN', got {label_version!r}"
        )
    if isinstance(label_version, int):
        if label_version < 1:
            raise ValueError(f"label_version must be >= 1, got {label_version!r}")
        return label_version
    raise ValueError(
        f"label_version must be None, a string like 'v2', or an integer, got {type(label_version).__name__}"
    )


def _parse_label_asset_path(name):
    path = Path(name)
    extension = path.suffix.lower()
    if extension not in _LABEL_EXTENSIONS:
        return None

    stem = path.stem
    version_num = 1
    stem_parts = stem.rsplit("_", 1)
    if len(stem_parts) == 2 and stem_parts[1].startswith("v") and stem_parts[1][1:].isdigit():
        version_num = int(stem_parts[1][1:])
        stem = stem_parts[0]

    for suffix, label_kind in _LABEL_SUFFIXES:
        if stem.endswith(suffix):
            return {
                "prefix": stem[: -len(suffix)],
                "label_kind": label_kind,
                "version_num": version_num,
                "extension": extension,
            }
    return None

class Segment:
    def __init__(
        self,
        config,
        image_volume=None,
        supervision_mask=None,
        validation_mask=None,
        inklabels=None,
        scale=None,
        dataset_idx=None,
        segment_relpath=None,
        segment_dir=None,
        segment_name=None,
    ):
        self.config = config
        self.scale = scale
        self.image_volume = image_volume
        self.supervision_mask = supervision_mask
        self.validation_mask = validation_mask
        self.inklabels = inklabels
        self.dataset_idx = dataset_idx
        self.segment_relpath = segment_relpath
        self.segment_dir = None if segment_dir is None else Path(segment_dir)
        self.segment_name = segment_name
        self.patch_size = config["patch_size"]

    @property
    def cache_key(self):
        return (
            int(self.dataset_idx),
            str(self.segment_relpath),
            self.scale,
            str(self.inklabels),
            str(self.supervision_mask),
            "" if self.validation_mask is None else str(self.validation_mask),
        )
    
    def discover_labels(self, *, label_version=None, extension=".zarr"):
        if self.segment_dir is None:
            raise ValueError("segment_dir is required for local label discovery")

        segment_name = self.segment_name or self.segment_dir.name
        resolved_label_version = self.config.get("label_version") if label_version is None else label_version
        normalized_extension = str(extension).lower()
        requested_version = _normalize_label_version(resolved_label_version)
        candidates_by_version = {}
        candidates_by_kind = {
            "inklabels": {},
            "supervision_mask": {},
            "validation_mask": {},
        }

        for path in self.segment_dir.iterdir():
            parsed = _parse_label_asset_path(path.name)
            if parsed is None:
                continue
            if parsed["prefix"] != str(segment_name):
                continue
            if parsed["extension"].lower() != normalized_extension:
                continue
            version_num = int(parsed["version_num"])
            version_entry = candidates_by_version.setdefault(version_num, {})
            version_entry[parsed["label_kind"]] = path
            candidates_by_kind[parsed["label_kind"]][version_num] = path

        if requested_version is not None:
            available_versions = sorted(
                version_num
                for version_num, record in candidates_by_version.items()
                if "inklabels" in record and "supervision_mask" in record
            )
            requested_name = "base" if requested_version <= 1 else f"v{requested_version}"
            assert requested_version in available_versions, (
                f"{self.segment_dir} does not contain matching {normalized_extension} labels "
                f"for version {requested_name}."
            )
            selected = candidates_by_version[requested_version]
            inklabels = selected["inklabels"]
            supervision_mask = selected["supervision_mask"]
            validation_mask = selected.get("validation_mask")
        else:
            assert candidates_by_kind["inklabels"], (
                f"{self.segment_dir} must contain at least one inklabels {normalized_extension} asset."
            )
            assert candidates_by_kind["supervision_mask"], (
                f"{self.segment_dir} must contain at least one supervision_mask {normalized_extension} asset."
            )
            validation_candidates = candidates_by_kind["validation_mask"]
            inklabels = candidates_by_kind["inklabels"][max(candidates_by_kind["inklabels"])]
            supervision_mask = candidates_by_kind["supervision_mask"][max(candidates_by_kind["supervision_mask"])]
            validation_mask = (
                validation_candidates[max(validation_candidates)] if validation_candidates else None
            )

        self.inklabels = inklabels
        self.supervision_mask = supervision_mask
        self.validation_mask = validation_mask
        return inklabels, supervision_mask, validation_mask
    
    def _find_patches(self):
        try:
            from koine_machines.data.patch_finding.default import find_segment_patches
        except ImportError:
            from data.patch_finding.default import find_segment_patches

        try:
            from .patch import Patch
        except ImportError:
            from patch import Patch

        training_patches, validation_patches = find_segment_patches(self, Patch)
        self.training_patches = training_patches
        self.validation_patches = validation_patches
        self.patches = training_patches + validation_patches
