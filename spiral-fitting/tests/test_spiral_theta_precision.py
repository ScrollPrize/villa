"""The spiral's winding angle, and what float32 costs on the outer windings.

``get_spiral_yxs`` builds the angle of winding k as ``k * 2 * pi + phi``. The
winding offset is the large part and phi the small one, so the sum spends its
precision on the offset: at winding 250 the angle is around 1571 rad, where a
float32 step is 1.2e-4 rad, and sin and cos of it times a radius of 4000 voxels
move the written node by half a voxel. These tests compare the shipped
construction against the same formula evaluated in float64.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sample_spiral


# A 256 winding spiral at 16 voxels to the winding reaches a radius of 4096
# voxels, which is the scale of a full scroll cross section, and 20 voxels of
# point spacing is the shipped output step size.
NUM_WINDINGS = 256
DR_PER_WINDING = 16.0
INTER_POINT_SPACING = 20.0


def _reference_windings_float64():
    """The same formula as get_spiral_yxs, evaluated in float64 throughout."""
    windings = []
    for winding_idx in range(NUM_WINDINGS):
        step = INTER_POINT_SPACING / (winding_idx + 0.5) / DR_PER_WINDING
        theta = winding_idx * 2 * np.pi + torch.arange(
            0, 2 * np.pi, step=step, dtype=torch.float64)
        radius = DR_PER_WINDING * theta / (2 * np.pi)
        windings.append(
            torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)
            * radius[:, None])
    return windings


def test_spiral_nodes_stay_within_a_float32_ulp_of_the_exact_spiral():
    # The tolerance is the ulp of a float32 position at this radius, which is
    # what a float32 output can carry. Building the angles in float32 misses it
    # by more than three orders of magnitude on the outer windings: the measured
    # displacement is 0.485 voxels at winding 250, with a median of 0.241.
    shipped = sample_spiral.get_spiral_yxs(
        NUM_WINDINGS, DR_PER_WINDING, INTER_POINT_SPACING,
        group_by_winding=True, device='cpu')
    reference = _reference_windings_float64()

    worst = 0.0
    for winding_idx, (got, want) in enumerate(zip(shipped, reference)):
        displacement = (got.to(torch.float64) - want).norm(dim=-1)
        worst = max(worst, float(displacement.max()))
    assert worst < 1e-3, (
        f'outermost node displaced {worst:.4f} voxels from the exact spiral')


def test_outer_winding_nodes_are_not_displaced_by_half_a_voxel():
    # The same statement read on the single winding the measurement names, so a
    # regression says which winding it broke rather than only that one did.
    shipped = sample_spiral.get_spiral_yxs(
        NUM_WINDINGS, DR_PER_WINDING, INTER_POINT_SPACING,
        group_by_winding=True, device='cpu')
    reference = _reference_windings_float64()
    displacement = (shipped[250].to(torch.float64) - reference[250]).norm(dim=-1)
    assert float(displacement.max()) < 1e-3
    assert float(displacement.median()) < 1e-3


@pytest.mark.parametrize('radius', [20.0, 100.0, 500.0, 4000.0])
def test_theta_just_below_the_seam_keeps_its_winding(radius):
    # A point a hair below the positive x axis has a winding angle just under
    # 2 * pi, so it belongs to the end of its winding and its shifted radius is
    # one winding less than its radius. Folding an angle that rounds up to
    # exactly 2 * pi back to 0 would move it forward by a whole winding, which
    # is the opposite of the half-open range it looks like it is enforcing.
    relative_yx = torch.tensor([[-1e-9, radius]], dtype=torch.float32)
    _theta, _radius, shifted_radius = sample_spiral.get_theta_and_radii(
        relative_yx, DR_PER_WINDING)

    exact = torch.tensor([[-1e-9, radius]], dtype=torch.float64)
    exact_theta = torch.arctan2(exact[..., 0], exact[..., 1]) % (2 * np.pi)
    exact_shifted = (torch.linalg.norm(exact, dim=-1)
                     - exact_theta / (2 * np.pi) * DR_PER_WINDING).clamp(min=0.)

    assert float(shifted_radius[0]) == pytest.approx(
        float(exact_shifted[0]), abs=1e-3)
