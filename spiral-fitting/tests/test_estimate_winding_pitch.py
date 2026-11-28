import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estimate_winding_pitch import radial_gaps  # noqa: E402


def test_radial_gaps_of_an_archimedean_spiral_are_its_pitch():
    pitch, size = 18.0, 801
    yy, xx = np.mgrid[:size, :size] - size // 2
    radius = np.hypot(yy, xx)
    theta = np.mod(np.arctan2(yy, xx), 2 * np.pi)
    # A two-voxel-thick sheet at r = pitch * (k + theta / 2pi).
    phase = np.mod(radius - pitch * theta / (2 * np.pi), pitch)
    mask = (phase < 2) & (radius > 30) & (radius < 380)
    gaps = radial_gaps(mask, (size // 2, size // 2))
    assert abs(np.median(gaps) - pitch) <= 0.5


def test_refit_scales_the_start_by_the_measured_windings_per_turn(capsys):
    from estimate_winding_pitch import main
    main(["--refit", "18.5", "1.14"])
    assert json.loads(capsys.readouterr().out) == {"model_initial_dr_per_winding": 21.1}
