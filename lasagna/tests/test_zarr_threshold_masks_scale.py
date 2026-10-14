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
