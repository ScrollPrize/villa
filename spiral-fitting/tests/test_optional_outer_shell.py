"""A dataset that ships tracks but no outer_shell/ must still load.

The conventional layout resolves an outer_shell path without probing for it,
so before this guard load_host_inputs() opened that path for track filtering
even when no loss required the shell, and the fit died on the missing
outer_shell/meta.json.
"""
import dbm
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

SPIRAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPIRAL_DIR))

from config import Config, FitConfig  # noqa: E402
from fit_session import ScrollSpec, SpiralInputPaths  # noqa: E402
from fit_spiral import FitContext  # noqa: E402

Z_BEGIN, Z_END = 1000, 2000


def tracks_only_dataset(root: Path) -> SpiralInputPaths:
    """A dataset root with umbilicus + tracks and deliberately no outer_shell."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "umbilicus.json").write_text(json.dumps({
        "control_points": [
            {"x": 3000, "y": 3000, "z": z, "score": 100}
            for z in range(0, 4001, 500)
        ]
    }))

    (root / "tracks").mkdir()
    dbm_path = root / "tracks" / "surf.dbm"
    track = np.stack([
        np.arange(Z_BEGIN + 100, Z_BEGIN + 132),
        np.full(32, 3200),
        np.arange(3300, 3332),
    ], axis=-1).astype(np.int32)
    with dbm.open(str(dbm_path), "c") as db:
        db[f"h:{Z_BEGIN}"] = pickle.dumps([track])

    assert not (root / "outer_shell").exists()
    return SpiralInputPaths(
        dataset_root=str(root),
        umbilicus=str(root / "umbilicus.json"),
        tracks_dbm=str(dbm_path),
        # exactly what conventional_input_paths() hands over: a path that was
        # composed from the dataset root, never probed for
        outer_shell=str(root / "outer_shell"),
    )


def tracks_only_config() -> dict:
    return Config({
        "z_begin": Z_BEGIN,
        "z_end": Z_END,
        "loss_weight_shell_outer": 0.0,
        "input_use_verified_patches": False,
        "input_use_unverified_patches": False,
        "input_use_fibers": False,
        "input_use_normals": False,
        "input_use_surf_sdt": False,
        "input_use_gradient_magnitude": False,
        "input_use_winding_inference": False,
    }).as_dict()


def make_context(config, paths):
    return FitContext(
        FitConfig(config),
        scroll=ScrollSpec(name="test", voxel_size_um=1.0,
                          spiral_outward_sense="CW"),
        paths=paths,
    )


def test_absent_outer_shell_does_not_block_track_loading(tmp_path):
    context = make_context(tracks_only_config(), tracks_only_dataset(tmp_path / "ds"))

    context.load_host_inputs()

    assert context.filter_tracks_by_shell is False
    assert context.shell_patch is None
    assert len(context.tracks) == 1


def test_a_required_outer_shell_still_fails_when_it_is_missing(tmp_path):
    config = tracks_only_config()
    config["loss_weight_shell_outer"] = 1.0  # shell losses want the shell
    context = make_context(config, tracks_only_dataset(tmp_path / "ds"))

    assert context.outer_shell_required()
    with pytest.raises(FileNotFoundError):
        context.load_host_inputs()
