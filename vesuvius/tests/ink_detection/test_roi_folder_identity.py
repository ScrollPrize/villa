"""Folder ROI requests retain their own outputs on the saved real CT case."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch
from vc_reader.reader import ZarrWriter, open_node

from vesuvius.ink_detection.inference import infer

from .test_roi_output_metadata import SavedFixtureModel


class ReplayFixtureModel(SavedFixtureModel):
    def forward(self, image):
        # Inference may sigmoid its output in place; keep the saved logits fixed.
        return super().forward(image).clone()


@pytest.fixture
def folder_case(tmp_path):
    with np.load(Path(__file__).parent / "fixtures/ink_roi_case.npz") as saved:
        ct, probability = saved["ct"], saved["probability"]
    segment = tmp_path / "segment"
    segment.mkdir()
    input_path = segment / "segment.zarr"
    volume = ZarrWriter.create(
        input_path, shape=ct.shape, chunks=ct.shape, dtype=ct.dtype
    )
    volume[:] = ct
    read_ct = np.asarray(open_node(input_path)[:])
    assert read_ct.dtype == ct.dtype
    np.testing.assert_array_equal(read_ct, ct)
    tile = np.zeros((128, 128), np.float32)
    tile[:80, :80] = probability
    configured = infer.ConfiguredModel(
        model=ReplayFixtureModel(tile), patch_size=128, input_depth=17,
        preprocessing="tifxyz_robust", amp_dtype=None,
    )
    args = infer.normalize_inference_paths(infer.parse_args([
        "--folder", str(segment), "weights.pth", "--num-workers", "0",
        "--no-compile", "--stride", "64",
    ]))
    reference_path = tmp_path / "reference.tif"
    infer.infer_single_zarr(
        args=args, input_zarr=input_path, configured_model=configured,
        device=torch.device("cpu"), output_tiff=reference_path,
    )
    return segment, configured, args, tifffile.imread(reference_path)


def _run(configured, args, roi):
    args.roi = roi
    infer.infer_folder(args, configured, device=torch.device("cpu"))


@pytest.mark.parametrize("first_roi", [None, [3, 66, 5, 73]])
def test_different_folder_roi_runs_and_keeps_previous_output(folder_case, first_roi):
    segment, configured, args, reference = folder_case
    _run(configured, args, first_roi)
    first = next((segment / "preds").glob("*.tif"))
    first_bytes = first.read_bytes()
    expected_first = reference if first_roi is None else reference[3:66, 5:73]
    np.testing.assert_array_equal(tifffile.imread(first), expected_first)
    next_roi = [2, 70, 4, 76]

    _run(configured, args, next_roi)

    outputs = sorted((segment / "preds").glob("*.tif"))
    assert len(outputs) == 2
    assert first.read_bytes() == first_bytes
    second = next(path for path in outputs if path != first)
    assert "_roi_r2-70_c4-76_forward_" in second.name
    assert tifffile.imread(second).shape == (68, 72)
    np.testing.assert_array_equal(tifffile.imread(second), reference[2:70, 4:76])
    sidecar = json.loads(second.with_suffix(".tif.roi.json").read_text())
    assert sidecar["roi_row_start_stop_col_start_stop"] == next_roi
    assert sidecar["parent_shape_yx"] == [80, 80]


@pytest.mark.parametrize("roi", [None, [3, 66, 5, 73]])
def test_identical_folder_request_is_still_skipped(folder_case, monkeypatch, roi):
    segment, configured, args, reference = folder_case
    _run(configured, args, roi)
    output = next((segment / "preds").glob("*.tif"))
    before = output.read_bytes()
    expected = reference if roi is None else reference[3:66, 5:73]
    np.testing.assert_array_equal(tifffile.imread(output), expected)
    monkeypatch.setattr(
        infer, "infer_single_zarr",
        lambda **kwargs: pytest.fail("identical request was rerun"),
    )

    _run(configured, args, roi)

    assert list((segment / "preds").glob("*.tif")) == [output]
    assert output.read_bytes() == before
    if roi is None:
        assert output.name.startswith("segment_weights_forward_")


def test_folder_full_image_is_not_skipped_after_roi(folder_case):
    segment, configured, args, reference = folder_case
    _run(configured, args, [3, 66, 5, 73])

    _run(configured, args, None)

    full = list((segment / "preds").glob("segment_weights_forward_*.tif"))
    assert len(full) == 1
    assert tifffile.imread(full[0]).shape == (80, 80)
    np.testing.assert_array_equal(tifffile.imread(full[0]), reference)
    assert not full[0].with_suffix(".tif.roi.json").exists()


@pytest.mark.parametrize("malformed_sidecar", [False, True])
@pytest.mark.parametrize("old_date", ["010101", "041126"])
def test_old_untagged_roi_does_not_satisfy_full_image_request(
    folder_case, monkeypatch, malformed_sidecar, old_date
):
    class FixedDate:
        @staticmethod
        def now():
            return datetime(2026, 11, 4)

    monkeypatch.setattr(infer, "datetime", FixedDate)
    segment, configured, args, reference = folder_case
    _run(configured, args, [3, 66, 5, 73])
    roi = next((segment / "preds").glob("*.tif"))
    old = roi.with_name(f"segment_weights_forward_{old_date}.tif")
    roi.rename(old)
    roi.with_suffix(".tif.roi.json").rename(old.with_suffix(".tif.roi.json"))
    if malformed_sidecar:
        old.with_suffix(".tif.roi.json").write_text("{malformed", encoding="utf-8")
    old_bytes = old.read_bytes()
    old_metadata = old.with_suffix(".tif.roi.json").read_bytes()
    np.testing.assert_array_equal(tifffile.imread(old), reference[3:66, 5:73])

    _run(configured, args, None)

    outputs = list((segment / "preds").glob("segment_weights_forward_*.tif"))
    assert len(outputs) == 2
    full = next(path for path in outputs if path != old)
    assert tifffile.imread(full).shape == (80, 80)
    np.testing.assert_array_equal(tifffile.imread(full), reference)
    assert not full.with_suffix(".tif.roi.json").exists()
    assert old.read_bytes() == old_bytes
    assert old.with_suffix(".tif.roi.json").read_bytes() == old_metadata
    if old_date == "041126":
        assert full.name == "segment_weights_forward_full1_041126.tif"
    monkeypatch.setattr(
        infer, "infer_single_zarr",
        lambda **kwargs: pytest.fail("genuine full-frame prediction was rerun"),
    )
    _run(configured, args, None)


def test_folder_roi_prefix_and_both_directions_have_distinct_outputs(folder_case):
    segment, configured, args, reference = folder_case
    args.output_prefix = "probe"
    args.direction = "both"

    _run(configured, args, [3, 66, 5, 73])

    outputs = list((segment / "preds").glob("*.tif"))
    assert len(outputs) == 2
    assert {path.name.split("_roi_r3-66_c5-73_")[1].split("_")[0]
            for path in outputs} == {"forward", "reverse"}
    assert all(path.name.startswith("probe_segment_weights_roi_") for path in outputs)
    for output in outputs:
        np.testing.assert_array_equal(tifffile.imread(output), reference[3:66, 5:73])


def test_invalid_folder_roi_is_rejected_even_when_prediction_exists(folder_case):
    _, configured, args, _ = folder_case
    _run(configured, args, None)

    with pytest.raises(ValueError, match="--roi"):
        _run(configured, args, [0, 81, 0, 80])
