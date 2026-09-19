import json
from pathlib import Path

import pytest

from flatten_spiral_checkpoint import (
    MODEL_CONFIG_KEYS,
    _checkpoint_config,
    _parse_args,
    _resolve_lasagna,
    _resolve_umbilicus,
    _resolve_voxel_size_um,
)


def test_legacy_checkpoint_model_config_gets_current_aliases():
    legacy = {key: index for index, key in enumerate(MODEL_CONFIG_KEYS)}
    config = _checkpoint_config({"cfg": legacy})
    for key in MODEL_CONFIG_KEYS:
        assert config[f"model_{key}"] == legacy[key]


def test_current_checkpoint_model_config_is_preserved():
    current = {
        f"model_{key}": index for index, key in enumerate(MODEL_CONFIG_KEYS)
    }
    config = _checkpoint_config({"cfg": current})
    assert config == current


def test_resolve_umbilicus_prefers_explicit_path(tmp_path):
    path = tmp_path / "umbilicus.json"
    path.write_text(json.dumps({"control_points": []}))
    checkpoint = tmp_path / "checkpoint.ckpt"
    assert _resolve_umbilicus(checkpoint, path) == path.resolve()


def test_resolve_umbilicus_finds_checkpoint_ancestor(tmp_path):
    path = tmp_path / "umbilicus.json"
    path.write_text(json.dumps({"control_points": []}))
    checkpoint = tmp_path / "spiral_output" / "run" / "checkpoint.ckpt"
    assert _resolve_umbilicus(checkpoint, None) == path.resolve()


def test_resolve_lasagna_requires_service_and_config(tmp_path):
    (tmp_path / "fit_service.py").write_text("")
    config = tmp_path / "configs" / "flatten_fast_nofilter.json"
    config.parent.mkdir()
    config.write_text("{}")
    assert _resolve_lasagna(tmp_path) == (
        (tmp_path / "fit_service.py").resolve(),
        config.resolve(),
    )


def test_checkpoint_config_reports_missing_fields():
    with pytest.raises(ValueError, match="missing model configuration"):
        _checkpoint_config({"cfg": {}})


# PHerc0826 as the repository's own scroll specification states it, beside the
# scan it is fitted from: 20250821151701-9.362um-1.2m-113keV.
PHERC0826_VOXEL_SIZE_UM = 9.362


def _dataset(root, voxel_size_um=PHERC0826_VOXEL_SIZE_UM):
    """A dataset root as fit_spiral requires it, with a run under it."""
    (root / "umbilicus.json").write_text(json.dumps({"control_points": []}))
    (root / "spiral-scroll.json").write_text(json.dumps({
        "schema_version": 1,
        "name": "PHerc0826",
        "voxel_size_um": voxel_size_um,
        "spiral_outward_sense": "CW",
    }))
    checkpoint = root / "spiral_output" / "run" / "checkpoint_fitted.ckpt"
    checkpoint.parent.mkdir(parents=True)
    return checkpoint


def test_voxel_size_comes_from_the_scroll_specification(tmp_path):
    checkpoint = _dataset(tmp_path)
    assert _resolve_voxel_size_um(checkpoint, None) == PHERC0826_VOXEL_SIZE_UM


def test_voxel_size_is_not_guessed_when_the_dataset_is_silent(tmp_path):
    checkpoint = tmp_path / "run" / "checkpoint_fitted.ckpt"
    checkpoint.parent.mkdir()
    with pytest.raises(FileNotFoundError, match="spiral-scroll.json"):
        _resolve_voxel_size_um(checkpoint, None)


def test_an_explicit_voxel_size_overrides_the_specification(tmp_path):
    checkpoint = _dataset(tmp_path)
    assert _resolve_voxel_size_um(checkpoint, 7.91) == 7.91


def test_a_non_positive_voxel_size_is_refused(tmp_path):
    checkpoint = _dataset(tmp_path)
    with pytest.raises(ValueError, match="must be positive"):
        _resolve_voxel_size_um(checkpoint, 0.0)


def test_the_exported_mesh_states_the_area_of_the_scroll_it_came_from(tmp_path):
    """The whole point of the number: it squares into every area written.

    save_combined_tifxyz records area_cm2 as area_vx2 * voxel_size_um ** 2 /
    1e8, and render_ink recovers the voxel size back out of that ratio, so a
    voxel size wrong by one part rescales every published area by two.
    """
    checkpoint = _dataset(tmp_path)
    args = _parse_args([str(checkpoint), str(tmp_path / "out.tifxyz")])
    resolved = _resolve_voxel_size_um(checkpoint, args.voxel_size_um)
    area_vx2 = 1_000_000
    stated_cm2 = area_vx2 * PHERC0826_VOXEL_SIZE_UM ** 2 / 1e8
    written_cm2 = area_vx2 * resolved ** 2 / 1e8
    assert written_cm2 == pytest.approx(stated_cm2, rel=1e-12), (
        f"the mesh would report {written_cm2:.6f} cm2 where the scroll the "
        f"dataset names measures {stated_cm2:.6f} cm2")
