"""A missing physical scale must be visible, not silently replaced by 1.0."""

from __future__ import annotations

from pathlib import Path

import pytest

zarr_threshold_masks = pytest.importorskip("lasagna.scripts.zarr_threshold_masks")


def test_infer_zyx_scale_warns_when_no_scale_is_available(monkeypatch, capsys):
    monkeypatch.setattr(zarr_threshold_masks, "_scale_from_preprocess_params", lambda arr: None)
    monkeypatch.setattr(zarr_threshold_masks, "_scale_from_ome_parent", lambda path: None)

    scale = zarr_threshold_masks._infer_zyx_scale(None, Path("/nonexistent/level0"))

    assert scale == (1.0, 1.0, 1.0)
    err = capsys.readouterr().err
    assert "no physical scale found" in err
    assert "assuming 1.0 per voxel" in err


def test_infer_zyx_scale_prefers_metadata_without_warning(monkeypatch, capsys):
    monkeypatch.setattr(zarr_threshold_masks, "_scale_from_preprocess_params", lambda arr: None)
    monkeypatch.setattr(zarr_threshold_masks, "_scale_from_ome_parent", lambda path: (8.64, 8.64, 8.64))

    assert zarr_threshold_masks._infer_zyx_scale(None, Path("/nonexistent/level0")) == (
        8.64,
        8.64,
        8.64,
    )
    assert "no physical scale found" not in capsys.readouterr().err


def test_a_111_parent_scale_on_a_named_pitch_warns(monkeypatch, capsys):
    """The audited shape: parent OME metadata returns [1,1,1] while the path states 8.64um."""
    monkeypatch.setattr(zarr_threshold_masks, "_scale_from_preprocess_params", lambda arr: None)
    monkeypatch.setattr(
        zarr_threshold_masks, "_scale_from_ome_parent", lambda path: (1.0, 1.0, 1.0)
    )
    scale = zarr_threshold_masks._infer_zyx_scale(
        None, Path("/data/PHerc0009B/volumes/20250521125136-8.640um-masked.zarr/0")
    )
    assert scale == (1.0, 1.0, 1.0)
    err = capsys.readouterr().err
    assert "8.640um" in err
    assert "[1, 1, 1]" in err


def test_a_111_parent_scale_without_a_stated_pitch_stays_quiet(monkeypatch, capsys):
    monkeypatch.setattr(zarr_threshold_masks, "_scale_from_preprocess_params", lambda arr: None)
    monkeypatch.setattr(
        zarr_threshold_masks, "_scale_from_ome_parent", lambda path: (1.0, 1.0, 1.0)
    )
    zarr_threshold_masks._infer_zyx_scale(None, Path("/data/store/zarr/0"))
    assert capsys.readouterr().err == ""


def test_a_real_scale_is_returned_without_a_warning(monkeypatch, capsys):
    monkeypatch.setattr(zarr_threshold_masks, "_scale_from_preprocess_params", lambda arr: None)
    monkeypatch.setattr(
        zarr_threshold_masks, "_scale_from_ome_parent", lambda path: (2.4, 2.4, 2.4)
    )
    scale = zarr_threshold_masks._infer_zyx_scale(None, Path("/data/x-8.64um.zarr/0"))
    assert scale == (2.4, 2.4, 2.4)
    assert capsys.readouterr().err == ""
