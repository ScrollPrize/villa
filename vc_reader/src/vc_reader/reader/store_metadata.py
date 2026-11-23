"""What is known about a volume store: its own ``meta.json`` / ``metadata.json`` and the
Open Data catalog, merged once when the store is first opened and kept from then on
(see ``vcr::zarr::volumeMetadata``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StoreMetadata:
    voxel_size_um: float | None = None  # level 0
    energy_kev: float | None = None
    detector_distance_mm: float | None = None
    beamline: str = ""
    scan_name: str = ""
    acquisition_date: str = ""
    # The source values the stored range spans, and the source Z range kept (zarr_export).
    value_window: tuple[float, float] | None = None
    z_crop: tuple[int, int] | None = None
    # From the Open Data catalog.
    sample_id: str = ""
    volume_id: str = ""
    scan_id: str = ""
    name: str = ""
    created_at: str = ""
    data_format: str = ""
    # The sources as published: file names (.zattrs too), and "catalog".
    documents: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, text: str) -> StoreMetadata:
        values = json.loads(text)
        for key in ("value_window", "z_crop"):
            if values[key] is not None:
                values[key] = tuple(values[key])
        return cls(**values)
