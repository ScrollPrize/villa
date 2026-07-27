"""The output store should record the settings needed to reproduce it.

patch_size and overlap were already stored. TTA was not, and it is the setting that
changes the output most, so a stored prediction could not be reproduced from itself
without guessing whether augmentation had been used.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import zarr

from vesuvius.models.run.inference import Inferer


def _inferer(tmp_path, *, do_tta: bool, tta_type: str = "mirroring") -> Inferer:
    """An Inferer with just enough state for _create_output_stores to run.

    __init__ builds a model and a dataset, neither of which this behaviour depends on,
    so the instance is assembled directly instead.
    """
    inf = Inferer.__new__(Inferer)
    inf.num_classes = 2
    inf.patch_size = (4, 4, 4)
    inf.num_total_patches = 1
    inf.output_dir = str(tmp_path)
    inf.part_id = 0
    inf.num_parts = 1
    inf.overlap = 0.5
    inf.verbose = False
    # 'none' short-circuits _get_zarr_compressor, which reaches for zarr.Blosc - absent in
    # zarr 3. Compression is irrelevant to the attributes under test.
    inf.compressor_name = "none"
    inf.compression_level = 1
    inf.bbox = None
    inf.is_multi_task = False
    inf.target_info = None
    inf.dataset = SimpleNamespace(input_shape=(8, 8, 8))
    inf.patch_start_coords_list = [(0, 0, 0)]
    inf.do_tta = do_tta
    inf.tta_type = tta_type
    return inf


def _attrs(tmp_path):
    return dict(zarr.open(str(tmp_path / "logits_part_0.zarr"), mode="r").attrs)


def test_records_tta_enabled(tmp_path):
    _inferer(tmp_path, do_tta=True, tta_type="mirroring")._create_output_stores()
    attrs = _attrs(tmp_path)
    assert attrs["tta"] is True
    assert attrs["tta_type"] == "mirroring"


def test_records_tta_disabled(tmp_path):
    _inferer(tmp_path, do_tta=False)._create_output_stores()
    attrs = _attrs(tmp_path)
    assert attrs["tta"] is False
    # tta_type is meaningless when no augmentation ran, so it is not claimed
    assert "tta_type" not in attrs


def test_records_rotation_tta(tmp_path):
    _inferer(tmp_path, do_tta=True, tta_type="rotation")._create_output_stores()
    assert _attrs(tmp_path)["tta_type"] == "rotation"


def test_tta_is_json_serialisable_bool(tmp_path):
    """zarr writes .zattrs as JSON, so a numpy bool would not round-trip."""
    inf = _inferer(tmp_path, do_tta=True)
    inf.do_tta = 1  # truthy non-bool, as an argparse/config path could supply
    inf._create_output_stores()
    assert _attrs(tmp_path)["tta"] is True


def test_existing_attrs_still_written(tmp_path):
    """The new keys must not displace what consumers already read."""
    _inferer(tmp_path, do_tta=False)._create_output_stores()
    attrs = _attrs(tmp_path)
    assert attrs["patch_size"] == [4, 4, 4]
    assert attrs["overlap"] == 0.5
    assert attrs["part_id"] == 0
    assert attrs["num_parts"] == 1
    assert attrs["original_volume_shape"] == [8, 8, 8]
