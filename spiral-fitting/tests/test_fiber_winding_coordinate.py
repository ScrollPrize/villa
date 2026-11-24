"""The fiber winding coordinate w (fiber_solve_inputs) is vc_object_store's
seam winding: W + turns + the fraction of a turn past the seam."""
import numpy as np
import pytest

from fiber_solve_inputs import (WindingSolveError, fiber_winding_coordinate,
                                packed_winding_coordinate)

vc_object_store = pytest.importorskip('vc_object_store')

# A tilted, piecewise-linear umbilicus (XYZ), deliberately not sorted by z.
AXIS = np.array([[120.0, 80.0, 300.0], [100.0, 100.0, 0.0], [110.0, 85.0, 150.0]])


def centre(z):
    order = np.argsort(AXIS[:, 2])
    return np.stack([np.interp(z, AXIS[order, 2], AXIS[order, 0]),
                     np.interp(z, AXIS[order, 2], AXIS[order, 1])], axis=-1)


def helix(turns, radius, z0, direction, start=0.37, steps_per_turn=60):
    """A helix about the axis rising from z0, ``direction`` +1 anticlockwise
    in XY; no vertex on the seam ray (-pi/2)."""
    angle = start + direction * np.linspace(0, 2 * np.pi * turns,
                                            int(turns * steps_per_turn) + 1)
    z = z0 + np.linspace(0, 40 * turns, len(angle))
    return np.concatenate([centre(z) + radius * np.stack(
        [np.cos(angle), np.sin(angle)], axis=-1), z[:, None]], axis=-1)


def native_winding(offsets, xyz, W, handedness):
    """W + turns + phi_seam from vc_object_store's WindingSeam."""
    turns = vc_object_store.pack_winding_geometry(
        offsets, xyz, AXIS, handedness=handedness)['turns']
    relative = vc_object_store.NativeWindingSeam(AXIS, handedness=handedness).relative(xyz)
    phase = np.mod((np.arctan2(relative[:, 1], relative[:, 0]) + np.pi / 2) / (2 * np.pi), 1.0)
    phi = phase if handedness == 1 else np.mod(-phase, 1.0)
    return np.repeat(W, np.diff(offsets)) + turns + phi


@pytest.mark.parametrize('handedness', [1, -1])
def test_w_is_the_native_seam_winding(handedness):
    fibers = [helix(3.4, 30.0, 10.0, 1), helix(2.7, 55.0, 60.0, -1, start=2.1),
              helix(0.6, 18.0, 250.0, 1, start=-1.2)]
    W = np.array([4, -7, 0])
    offsets = np.concatenate([[0], np.cumsum([len(fiber) for fiber in fibers])])
    xyz = np.ascontiguousarray(np.concatenate(fibers))
    expected = native_winding(offsets, xyz, W, handedness)
    np.testing.assert_allclose(packed_winding_coordinate(offsets, xyz, W, AXIS, handedness),
                               expected, rtol=0, atol=1e-9)
    for index, fiber in enumerate(fibers):
        np.testing.assert_allclose(
            fiber_winding_coordinate(fiber, W[index], AXIS, handedness),
            expected[offsets[index]:offsets[index + 1]], rtol=0, atol=1e-9)
    # The first helix gains one winding per turn in the scroll's direction.
    first = fiber_winding_coordinate(fibers[0], 4, AXIS, handedness)
    assert first[-1] - first[0] == pytest.approx(3.4 * handedness)


@pytest.mark.parametrize('change, message', [
    (lambda xyz: xyz.__setitem__((5, 1), np.nan), 'not finite'),
    (lambda xyz: xyz.__setitem__(5, [*centre(xyz[5, 2]), xyz[5, 2]]), 'within 1e-06'),
    (lambda xyz: _rotate_after(xyz, 5), 'turns apart'),
])
def test_the_load_guard_names_the_fiber(change, message):
    fibers = [helix(1.0, 30.0, 10.0, 1), helix(1.0, 30.0, 10.0, -1)]
    change(fibers[1])
    offsets = np.array([0, len(fibers[0]), len(fibers[0]) + len(fibers[1])])
    with pytest.raises(WindingSolveError, match=f'fiber second: .*{message}'):
        packed_winding_coordinate(offsets, np.concatenate(fibers), [0, 0], AXIS, -1,
                                  name=['first', 'second'].__getitem__)


def _rotate_after(xyz, index):
    """Turn every vertex after ``index`` half a turn about the axis (in
    place): the step from ``index`` sweeps about 0.5 turns."""
    after = slice(index + 1, None)
    xyz[after, :2] = 2 * centre(xyz[after, 2]) - xyz[after, :2]


@pytest.mark.parametrize('sense', ['CW', 'ACW'])
def test_sheet_label_and_winding_offset_are_constant_along_a_sheet(sense):
    """A fiber following one sheet of a deformed model over several turns:
    its sheet label n steps exactly where Spiral's winding does, at theta =
    0, so s = shifted / dr - n is constant along it, for both handedness
    signs (fiber_cc_loss's sheet rule on fiber_solve_inputs' w_s). The
    fiber CC loss built on it is zero at the right gauge and pulls the sheet
    outward (inward) when its target is one winding out (in)."""
    from types import SimpleNamespace
    import torch
    from config import Config
    from fiber_cc_loss import FiberCCLoss, branch_target
    from fiber_solve_inputs import circular_median, handedness_for, undeformed_theta
    from sample_spiral import get_theta_and_radii
    from transforms import SpiralAndTransform
    torch.manual_seed(0)
    config = {
        'model_initial_dr_per_winding': 16., 'model_flow_voxel_resolution': 8,
        'model_flow_field_type': 'cartesian', 'model_num_flow_stages': 2,
        'model_linear_z_resolution': 48, 'model_gap_expander_logit_resolution': 24,
        'model_gap_expander_num_windings': 6, 'model_gap_expander_capacity_windings': 6,
        'model_gap_expander_min_gap': 1.0, 'model_gap_expander_softplus_bias': 4.0,
        'model_gap_expander_lr_scale': 0.3, 'output_first_winding': 1}
    centre_yx = (5.0, -3.0)
    umbilicus = torch.tensor([[z, *centre_yx] for z in np.linspace(0., 192., 5)],
                             dtype=torch.float32)
    model = SpiralAndTransform(
        flow_integration_steps=3, flow_integration_solver='rk4',
        flow_min_corner_zyx=torch.tensor([0, -96, -96]),
        flow_max_corner_zyx=torch.tensor([192, 96, 96]),
        umbilicus_zyx=umbilicus, config=config, spiral_outward_sense=sense)
    # A visible deformation: the model's angle differs from the raw angle
    # around the axis by up to a few degrees (checked below).
    with torch.no_grad():
        for flow in model.flow_field.flows:
            flow.normal_(std=2e-2)
        model.linear_logits.normal_(std=2e-3)
    transform = model.get_slice_to_spiral_transform()
    dr = model.get_dr_per_winding().detach()
    # Sheet 1 of the spiral frame over 2.6 turns, rising in z: winding
    # k = 1 + floor(turns) at radius (k + theta / 2pi) dr.
    turns = np.linspace(0.13, 2.73, 400)
    theta = 2 * np.pi * turns
    radius = float(dr) * (1 + turns)
    spiral = torch.tensor(np.stack([np.linspace(60., 130., len(turns)),
                                    radius * np.sin(theta), radius * np.cos(theta)], -1),
                          dtype=torch.float32)
    with torch.no_grad():
        zyx = transform.inv(spiral).numpy().astype(np.float64)
    axis_xyz = umbilicus.numpy()[:, ::-1].astype(np.float64)
    w = fiber_winding_coordinate(zyx[:, ::-1], 3, axis_xyz, handedness_for(sense))
    c, deviation = circular_median(
        np.mod(w, 1.0) - undeformed_theta(model, zyx) / (2 * np.pi))
    assert np.abs(deviation).max() < 0.03
    with torch.no_grad():
        model_theta, _, shifted = get_theta_and_radii(
            transform(torch.tensor(zyx, dtype=torch.float32))[:, 1:], dr)
    n, rho = branch_target(torch.tensor(w - c, dtype=torch.float32), model_theta)
    s = (shifted / dr - n).numpy()
    # The deformation moves the angle, never near the half-turn rejection.
    assert np.abs(rho.numpy()).max() < 0.2
    # The label steps twice (one per theta = 0 crossing), s never does.
    assert len(np.unique(n.numpy())) == 3
    assert np.ptp(s) < 0.02
    raw = np.arctan2(zyx[:, 1] - centre_yx[0], zyx[:, 2] - centre_yx[1])
    warp = np.mod(model_theta.numpy() - raw + np.pi, 2 * np.pi) - np.pi
    assert np.ptp(warp) > 0.02

    # The loss itself, over every point of the sheet as one solve CC.
    count = len(zyx)
    solve = SimpleNamespace(
        c=c, w_min=np.array([w.min()]), w_max=np.array([w.max()]),
        fiber_cc=np.array([0]), largest_cc=0, point_fiber=np.zeros(count, dtype=np.int64),
        piece_offsets=np.array([0, count]), zyx=zyx.astype(np.float32), w_s=w - c,
        family=np.array([0]), fiber_uuid=np.zeros((1, 16), dtype=np.uint8),
        cc_component=np.array([0]), run_uuid='run',
        n0=np.round(w - c - undeformed_theta(model, zyx) / (2 * np.pi)).astype(np.int64))
    offset = int(np.round(np.median(s)))
    n_low = int(np.floor(w.min() - c))

    def fiber_loss(target_shift):
        config = Config({'fiber_largest_cc_min_winding': offset + n_low + target_shift,
                         'sample_count_fiber_ccs': 1,
                         'sample_count_points_per_fiber_cc': 256}).as_dict()
        loss = FiberCCLoss(solve, config, torch.device('cpu'), log=lambda line: None)
        assert loss.c0 == offset + target_shift
        # _evaluate's s at every point is the sheet's constant offset.
        with torch.no_grad():
            _, _, _, _, s_vox = loss._evaluate(transform, dr, torch.arange(count))
        np.testing.assert_allclose((s_vox / dr).numpy(), s, atol=1e-4)
        # A radial scale of the spiral frame, to read the loss's pull.
        scale = torch.ones((), requires_grad=True)
        fresh = model.get_slice_to_spiral_transform()

        def scaled(points):
            spiral = fresh(points)
            return torch.cat([spiral[:, :1], spiral[:, 1:] * scale], dim=-1)
        value = loss.loss(scaled, dr, config, iteration=0, dt_active=False)
        value.backward()
        return float(value), float(scale.grad)

    torch.manual_seed(1)
    at_gauge, _ = fiber_loss(0)
    assert at_gauge < 0.01 * float(dr)
    outward, outward_grad = fiber_loss(1)
    inward, inward_grad = fiber_loss(-1)
    # One winding off is within the quadratic zone: 0.5 dr^2 / (1.5 dr).
    assert outward == pytest.approx(float(dr) / 3, rel=0.05)
    assert inward == pytest.approx(float(dr) / 3, rel=0.05)
    assert outward_grad < 0 < inward_grad
