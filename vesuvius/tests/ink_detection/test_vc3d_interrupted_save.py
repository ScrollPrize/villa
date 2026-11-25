"""Label TIFFs that an interrupted VC3D save left from two saves are refused."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vesuvius.ink_detection.data.segment import interrupted_vc3d_save


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_store(directory: Path, files: dict[str, bytes]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for kind, data in files.items():
        (directory / f"seg_{kind}.tif").write_bytes(data)


def _write_record(directory: Path, before: dict[str, bytes], after: dict[str, bytes]) -> None:
    record = {
        "schema": "vc3d.ink_labels.saving/1",
        "sidecar": {
            "files": {
                kind: {"name": f"seg_{kind}.tif", "sha256": _sha(data)}
                for kind, data in after.items()
            }
        },
        "replaces": {kind: {"sha256": _sha(data)} for kind, data in before.items()},
    }
    (directory / "seg_inklabels.json.saving").write_text(json.dumps(record))


OLD = {"supervision_mask": b"old supervision", "inklabels": b"old ink"}
NEW = {"supervision_mask": b"new supervision", "inklabels": b"new ink"}


def test_no_record_means_a_settled_store(tmp_path):
    _write_store(tmp_path, OLD)
    assert interrupted_vc3d_save(tmp_path / "seg_inklabels.tif") is None


@pytest.mark.parametrize("files", [OLD, NEW])
def test_one_complete_generation_reads(tmp_path, files):
    _write_store(tmp_path, files)
    _write_record(tmp_path, OLD, NEW)
    assert interrupted_vc3d_save(tmp_path / "seg_inklabels.tif") is None
    assert interrupted_vc3d_save(tmp_path / "seg_supervision_mask.tif") is None


def test_masks_from_two_saves_are_refused(tmp_path):
    _write_store(tmp_path, {"supervision_mask": NEW["supervision_mask"], "inklabels": OLD["inklabels"]})
    _write_record(tmp_path, OLD, NEW)
    problem = interrupted_vc3d_save(tmp_path / "seg_inklabels.tif")
    assert problem is not None and "interrupted VC3D save" in problem


def test_unreadable_record_is_refused(tmp_path):
    _write_store(tmp_path, OLD)
    (tmp_path / "seg_inklabels.json.saving").write_text("{")
    assert interrupted_vc3d_save(tmp_path / "seg_inklabels.tif") is not None


def test_versioned_files_use_their_own_record(tmp_path):
    for kind, data in OLD.items():
        (tmp_path / f"seg_{kind}_v2.tif").write_bytes(data)
    # A record for version 1 says nothing about version 2.
    _write_record(tmp_path, {"inklabels": b"x"}, {"inklabels": b"y"})
    assert interrupted_vc3d_save(tmp_path / "seg_inklabels_v2.tif") is None


def _tiff_segment(tmp_path: Path):
    from vesuvius.ink_detection.config import InkDataConfig
    from vesuvius.ink_detection.types import Segment

    config = InkDataConfig.from_mapping(
        {
            "mode": "flat",
            "patch_size": [3, 2, 2],
            "patch_overlap": 0.5,
            "patch_min_labeled_coverage": 0.0,
            "image_normalization": "none",
            "out_dir": str(tmp_path),
            "dataloader_workers": 1,
            "datasets": [{"segments_path": str(tmp_path), "volume_scale": 0}],
        }
    )
    return Segment(
        data_config=config,
        source=config.datasets[0],
        dataset_idx=0,
        segment_relpath="seg",
        segment_dir=tmp_path / "seg",
        segment_name="seg",
        image_volume="image.zarr",
    )


def test_trainer_discovery_refuses_masks_from_two_saves(tmp_path):
    from vesuvius.ink_detection.data.segment import discover_segment_labels

    store = tmp_path / "seg"
    _write_store(store, {"supervision_mask": NEW["supervision_mask"], "inklabels": OLD["inklabels"]})
    _write_record(store, OLD, NEW)
    segment = _tiff_segment(tmp_path)
    with pytest.raises(ValueError, match="interrupted VC3D save"):
        discover_segment_labels(segment, extension=".tif")

    # A complete save reads.
    _write_store(store, NEW)
    discovered = discover_segment_labels(segment, extension=".tif")
    assert discovered.inklabels == store / "seg_inklabels.tif"
