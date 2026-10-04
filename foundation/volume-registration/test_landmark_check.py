"""Run: python -m pytest test_landmark_check.py (from foundation/volume-registration)."""
import json
from pathlib import Path

import numpy as np

from check_transform import main
from transform_utils import fit_affine_transform_from_points, landmark_report

# Published PHerc1667 1.129 um -> 2.399 um transform (villa #1843), copied from the open-data bucket.
PHERC1667 = Path(__file__).parent / "testdata" / "pherc1667_1.129um_to_2.399um_transform.json"


def _load():
    d = json.loads(PHERC1667.read_text())
    return d["transformation_matrix"], d["fixed_landmarks"], d["moving_landmarks"]


def test_published_pherc1667_matrix_is_flagged():
    matrix, fixed, moving = _load()
    rep = landmark_report(matrix, fixed, moving)
    assert rep["n_landmarks"] == 6
    assert abs(rep["matrix_rms"] - 51.39) < 0.05
    assert rep["lsq_rms"] < 1.0
    assert rep["flagged"]


def test_least_squares_refit_is_not_flagged():
    _, fixed, moving = _load()
    refit = fit_affine_transform_from_points(fixed, moving)
    rep = landmark_report(refit, fixed, moving)
    assert not rep["flagged"]
    assert rep["matrix_rms"] == rep["lsq_rms"] or abs(rep["matrix_rms"] - rep["lsq_rms"]) < 1e-6
    # held-out landmarks stay within a few voxels, so the refit is constrained by the landmarks
    assert rep["lsq_loo_rms"] < 5.0


def test_cli_exit_code_and_refit(tmp_path, capsys):
    assert main([str(PHERC1667)]) == 1
    out = tmp_path / "refit.json"
    main([str(PHERC1667), "--refit", str(out)])
    assert main([str(out)]) == 0
    refit = json.loads(out.read_text())
    assert np.asarray(refit["transformation_matrix"]).shape == (3, 4)
    assert "WARNING" in capsys.readouterr().out
