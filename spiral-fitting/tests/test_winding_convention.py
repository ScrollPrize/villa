"""Sheet-convention absolute anchors (winding_convention.py) and the held-out patch exclusion."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import fit_spiral
import losses
import winding_convention as wc
from config import CHECKPOINT_SUPERVISION_KEYS, NEW_FIT_KEYS, Config
from fit_session import PclRole, pcl_input_enabled
from sample_spiral import get_theta_and_radii
from tests.test_winding_pcl_unwrap import (DR, _Chain, _IdentityTransform,
                                           _setup, _spiral_point)

TWO_PI = 2 * math.pi


def _q_t(phi, radius, w_cont, dr=20.0):
    """(q, t) the fitter / the sheet rule assign to a spiral-space point at (phi, radius)."""
    yx = torch.tensor([[math.sin(phi) * radius, math.cos(phi) * radius]], dtype=torch.float32)
    theta, _, shifted = get_theta_and_radii(yx, torch.tensor(dr))
    t, _ = wc.branch_target(w_cont, theta.numpy())
    return float(shifted / dr), int(t[0])


def _cfg(**overrides):
    return {
        'sample_count_points_per_patch': 4,
        'sample_count_relative_winding_pcls': 1,
        'sample_count_relative_winding_patch_pairs_per_pcl': 1,
        'sample_count_absolute_winding_pcls': 1,
        'sample_count_absolute_winding_points_per_pcl': 1,
        'pcl_rel_winding_adjacent_patches_only': True,
        'pcl_sampling_weights': None,
        'pcl_stratified_pcl_sampling': False,
        'patch_loss_z_margin': 0.0,
        **overrides,
    }


def _sheet_meta():
    return {'winding_is_absolute': True, 'winding_convention': 'sheet', 'input_role': 'absolute'}


class _Rotate:
    """Spiral-space rotation by alpha at fixed radius (stands in for a stale cache)."""

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, points):
        c, s = math.cos(self.alpha), math.sin(self.alpha)
        y, x = points[..., 1], points[..., 2]
        return torch.stack([points[..., 0], c * y + s * x, -s * y + c * x], dim=-1)


# --- branch rule -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize('alpha, expected', [
    (0.4, (10, 10)), (-0.25, (9, 9)), (TWO_PI + 0.1, (11, 10)), (2 * TWO_PI + 0.4, (12, 10))])
def test_travel_along_own_sheet_uses_the_endpoint_angle(alpha, expected):
    # Round-5 cases: a point on sheet n = 10 at phi_0 = 0.2 carried along its own sheet by alpha ends at
    # radius dr * (n + (phi_0 + alpha)/2pi); the anchor keeps w = n + phi_0/2pi, so full turns are real errors.
    phi0, n = 0.2, 10
    w = n + phi0 / TWO_PI
    q, t = _q_t((phi0 + alpha) % TWO_PI, 20.0 * (n + (phi0 + alpha) / TWO_PI), w)
    assert (round(q, 3), t) == expected


def test_rigid_rotation_by_full_turns_is_invisible():
    w = 10 + 0.2 / TWO_PI
    radius = 20.0 * w
    a = _q_t((0.2 + 0.4) % TWO_PI, radius, w)
    b = _q_t((0.2 + 2 * TWO_PI + 0.4) % TWO_PI, radius, w)
    assert a[1] == b[1] == 10 and a[0] == pytest.approx(b[0], abs=1e-4)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('phi0, phi_e, expected', [
    (0.02, 0.0, 10), (0.02, 1e-7, 10), (0.02, TWO_PI - 1e-7, 9),
    (TWO_PI - 0.02, TWO_PI - 1e-7, 10), (TWO_PI - 0.02, 0.0, 11), (TWO_PI - 0.02, 1e-7, 11)])
def test_seam_endpoints_round_to_the_analytic_label(dtype, phi0, phi_e, expected):
    w = dtype(10 + phi0 / TWO_PI)
    t, rho = wc.branch_target(w, dtype(phi_e))
    assert int(t) == expected and abs(float(rho)) < 0.01


def test_acceptance_boundary_is_inclusive():
    assert wc.accepted(np.float64(0.4)) and wc.accepted(np.float64(-0.4))
    assert wc.accepted(np.float64(0.4 - 1e-6))
    assert not wc.accepted(np.float64(0.4 + 1e-6))
    # geometric: a point turned by just under / over 0.8 pi relative to its anchor
    w = 10 + 0.2 / TWO_PI
    for alpha, ok in ((0.79 * math.pi, True), (0.81 * math.pi, False)):
        _, rho = wc.branch_target(w, (0.2 + alpha) % TWO_PI)
        assert bool(wc.accepted(rho)) is ok


def test_float32_seam_angle_validates():
    # get_theta's float32 % 2pi can return float32(2pi) for a true angle just below 2pi.
    phi = float(np.float32(TWO_PI - 1e-7) % np.float32(TWO_PI))
    assert phi <= wc.PHI_MAX
    record = json.loads(json.dumps({'wind_a': 3, 'phi_ref': wc.PHI_MAX, 'w_cont': 3 + wc.PHI_MAX / TWO_PI}))
    point = {'winding_annotation': record['wind_a'], 'phi_ref': record['phi_ref'], 'w_cont': record['w_cont']}
    assert wc.record_problems(point) == []


# --- validation ------------------------------------------------------------------------------------------------

def _sheet_pcl(points, meta=None):
    return {'name': 'abs_x', 'metadata': meta or _sheet_meta(), 'points': dict(enumerate(points))}


def _good_point(n=5, phi=1.0):
    return {'id': 0, 'winding_annotation': float(n), 'phi_ref': phi, 'w_cont': n + phi / TWO_PI}


def test_validation_accepts_consistent_records_and_rejects_bad_ones():
    cfg = {'abs_winding_target_convention': 'sheet'}
    assert wc.validate_sheet_collections({0: _sheet_pcl([_good_point()])}, cfg) == (1, 1)
    bad = [dict(_good_point(), winding_annotation=6.0),           # wind_a changed alone
           dict(_good_point(), winding_annotation=5.5),           # non-integer
           dict(_good_point(n=0, phi=0.5)),                       # n < 1
           {k: v for k, v in _good_point().items() if k != 'phi_ref'},
           dict(_good_point(), phi_ref='1.0'),                    # malformed type reaches the validator
           dict(_good_point(), phi_ref=7.0)]
    for point in bad:
        with pytest.raises(ValueError):
            wc.validate_sheet_collections({0: _sheet_pcl([point])}, cfg)
    # a deliberately wrong anchor (w_cont and wind_a both +1) is still a valid record
    wrong = dict(_good_point(), winding_annotation=6.0, w_cont=_good_point()['w_cont'] + 1)
    assert wc.validate_sheet_collections({0: _sheet_pcl([wrong])}, cfg) == (1, 1)


def test_validation_refuses_convention_role_and_mixing_errors():
    with pytest.raises(ValueError, match='legacy'):
        wc.validate_sheet_collections({0: _sheet_pcl([_good_point()])},
                                      {'abs_winding_target_convention': 'legacy'})
    relative = dict(_sheet_meta(), winding_is_absolute=False, input_role='relative')
    with pytest.raises(ValueError, match='absolute input role'):
        wc.validate_sheet_collections({0: _sheet_pcl([_good_point()], relative)},
                                      {'abs_winding_target_convention': 'sheet'})
    legacy_abs = {'name': 'old', 'metadata': {'winding_is_absolute': True}, 'points': {}}
    mixed = {0: _sheet_pcl([_good_point()]), 1: legacy_abs}
    assert wc.validate_sheet_collections(mixed, {'abs_winding_target_convention': 'sheet'}) == (1, 1)
    with pytest.raises(ValueError, match='mixed'):
        wc.validate_sheet_collections(mixed, {'abs_winding_target_convention': 'sheet',
                                              'abs_winding_eval_at_point': True})


def test_config_lifecycle_and_input_gate():
    assert set(CHECKPOINT_SUPERVISION_KEYS) <= NEW_FIT_KEYS
    with pytest.raises(ValueError, match='requires'):
        Config({'abs_winding_eval_at_point': True})
    Config({'abs_winding_eval_at_point': True, 'abs_winding_target_convention': 'sheet'})
    for patches_off in ({'input_use_verified_patches': False}, {'input_disable_patches': True}):
        assert pcl_input_enabled(patches_off, PclRole.ABSOLUTE) is False
        assert pcl_input_enabled({**patches_off, 'abs_winding_eval_at_point': True}, PclRole.ABSOLUTE) is True


def test_loader_keeps_sheet_fields(tmp_path):
    from point_collection import load_point_collection
    doc = {'vc_pointcollections_json_version': '1', 'collections': {'1': {
        'name': 'abs_f', 'metadata': {'winding_convention': 'sheet', 'source_fiber': 'f.json'},
        'points': {'1': {'p': [1.0, 2.0, 3.0], 'wind_a': 4, 'phi_ref': wc.PHI_MAX, 'w_cont': 5.0,
                         'source_fiber': 'f.json', 'src_index': 7}}}}}
    path = tmp_path / 'abs_winding.json'
    path.write_text(json.dumps(doc))
    pcl = load_point_collection(str(path))[1]
    point = pcl['points'][1]
    assert pcl['metadata']['winding_convention'] == 'sheet'
    assert (point['phi_ref'], point['w_cont'], point['source_fiber'], point['src_index']) == (wc.PHI_MAX, 5.0, 'f.json', 7)


# --- absolute loss, patch path ---------------------------------------------------------------------------------

def _abs_case(ref_theta, ref_winding, patch_points, w_cont, cfg_overrides=(), refresh=None):
    crossing_map, atlas, patches = _setup([_spiral_point(ref_theta, ref_winding)], patch_points)
    if refresh is not None:
        crossing_map.force_refresh(refresh)
    point = {'_theta_node_id': 0, 'winding_annotation': 1.0, 'w_cont': w_cont,
             'on_patch': {'id': 'p1', 'ij': [0.0, 0.0]}}
    pcl = {'metadata': _sheet_meta(), 'points_by_patch': {'p1': [point]}}
    cfg = _cfg(abs_winding_target_convention='sheet', **dict(cfg_overrides))
    return losses.get_patch_abs_winding_loss(
        _IdentityTransform(), torch.tensor(DR), patches, atlas, [pcl],
        crossing_map=crossing_map, cfg=cfg, z_begin=-1, z_end=1).item()


@pytest.mark.parametrize('mode', [(), (('abs_winding_local_to_attachment', True),)])
@pytest.mark.parametrize('ref_theta', [TWO_PI - 0.02, 0.02, 6.0])
def test_sheet_anchor_patch_path_is_zero_on_correct_geometry(mode, ref_theta):
    # The anchor's sheet is raw winding 4 at ref_theta; the patch continues that sheet across the seam
    # (raw winding 5 just after phi = 0), as in the legacy fixture.
    w = 4 + ref_theta / TWO_PI
    patch = [[_spiral_point(0.1, 5), _spiral_point(0.2, 5)]] if ref_theta > math.pi else \
        [[_spiral_point(0.1, 4), _spiral_point(0.2, 4)]]
    assert _abs_case(ref_theta, 4, patch, w, mode) < 1e-4
    # a deliberately wrong anchor (w_cont + 1) costs exactly one winding
    assert _abs_case(ref_theta, 4, patch, w + 1, mode) == pytest.approx(DR, rel=1e-4)


@pytest.mark.parametrize('alpha', [0.05, -0.05])
def test_sheet_anchor_patch_path_with_stale_cached_theta(alpha):
    # Cached thetas refreshed under a slightly rotated transform (the reference crosses the seam in the
    # cache but not now, or vice versa): the target follows the cached reference frame.
    ref_theta = TWO_PI - 0.02 if alpha > 0 else 0.02
    w = 10 + ref_theta / TWO_PI
    if alpha > 0:
        patch = [[_spiral_point(TWO_PI - 0.015, 10), _spiral_point(TWO_PI - 0.01, 10)]]
    else:
        patch = [[_spiral_point(0.025, 10), _spiral_point(0.03, 10)]]
    assert _abs_case(ref_theta, 10, patch, w, refresh=_Rotate(alpha)) < 1e-4


def test_sheet_rows_are_rejected_when_the_phase_residual_is_too_large():
    ref_theta = 1.0
    patch = [[_spiral_point(1.1, 4), _spiral_point(1.2, 4)]]
    # w_cont half a winding off: |rho| = 0.5 > 0.4, row masked, loss 0 (finite)
    assert _abs_case(ref_theta, 4, patch, 4 + ref_theta / TWO_PI + 0.5) == 0.0


def test_legacy_rows_are_unchanged_beside_sheet_rows():
    crossing_map, atlas, patches = _setup([_spiral_point(6.0, 4)], [[_spiral_point(0.1, 5), _spiral_point(0.2, 5)]])
    point = {'_theta_node_id': 0, 'winding_annotation': 4.0, 'on_patch': {'id': 'p1', 'ij': [0.0, 0.0]}}
    pcl = {'metadata': {'winding_is_absolute': True}, 'points_by_patch': {'p1': [point]}}
    for convention in ('legacy', 'sheet'):
        loss = losses.get_patch_abs_winding_loss(
            _IdentityTransform(), torch.tensor(DR), patches, atlas, [pcl], crossing_map=crossing_map,
            cfg=_cfg(abs_winding_target_convention=convention), z_begin=-1, z_end=1)
        assert loss.item() < 1e-5


def test_point_mode_leaves_sheet_anchors_out_of_the_patch_path():
    assert _abs_case(1.0, 4, [[_spiral_point(1.1, 4), _spiral_point(1.2, 4)]], 4 + 1.0 / TWO_PI + 1,
                     (('abs_winding_eval_at_point', True),)) == 0.0


# --- relative loss, sheet pairs --------------------------------------------------------------------------------

def _rel_case(ref_specs, patch_points, w_conts, chain_extra=()):
    refs = [_spiral_point(theta, winding) for theta, winding in ref_specs]
    crossing_map, atlas, patches = _setup(refs, patch_points, [[0, 1]])
    p1 = {'_theta_node_id': 0, 'winding_annotation': 1.0, 'w_cont': w_conts[0],
          'on_patch': {'id': 'p1', 'ij': [0.0, 0.0]}}
    p2 = {'_theta_node_id': 1, 'winding_annotation': 1.0, 'w_cont': w_conts[1],
          'on_patch': {'id': 'p2', 'ij': [0.0, 0.0]}}
    pcl = {'metadata': _sheet_meta(), 'points_by_patch': {'p1': [p1], 'p2': [p2]},
           'chain': _Chain([p1, *chain_extra, p2])}
    cfg = _cfg(abs_winding_target_convention='sheet')
    strata = losses.build_pcl_sampling_strata(['absolute'], cfg)
    return losses.get_patch_rel_winding_loss(
        _IdentityTransform(), torch.tensor(DR), patches, atlas, [pcl], strata,
        crossing_map=crossing_map, cfg=cfg, z_begin=-1, z_end=1).item()


def test_sheet_relative_pair_across_the_seam():
    # One sheet: raw winding 4 at theta 6.0 continues as raw winding 5 at 0.3 (after the seam).
    specs = [(6.0, 4), (0.3, 5)]
    patches = [[_spiral_point(5.9, 4), _spiral_point(5.95, 4)], [_spiral_point(0.35, 5), _spiral_point(0.4, 5)]]
    w1 = 4 + 6.0 / TWO_PI
    w2 = 4 + (TWO_PI + 0.3) / TWO_PI       # continuous winding one seam later on the same sheet
    assert _rel_case(specs, patches, (w1, w2)) < 1e-4
    assert _rel_case(specs, patches, (w1, w2 + 1)) == pytest.approx(DR, rel=1e-4)


def test_sheet_relative_pair_one_turn_apart_and_on_adjacent_sheets():
    patches = [[_spiral_point(1.0, 4), _spiral_point(1.05, 4)], [_spiral_point(1.0, 5), _spiral_point(1.05, 5)]]
    # same sheet one turn apart: continuous winding differs by exactly one
    assert _rel_case([(1.0, 4), (1.0, 5)], patches, (4 + 1 / TWO_PI, 5 + 1 / TWO_PI)) < 1e-4


@pytest.mark.parametrize('direction', [1, -1])
def test_sheet_relative_pair_with_intermediate_chain_nodes_crossing_the_seam(direction):
    # The fiber runs one full turn (positive or negative seam crossing) through unannotated chain nodes;
    # sheet rows use (t2 - t1) and never subtract the chain's seam crossings.
    path = [(1.0, 4), (3.0, 4), (5.5, 4), (0.5, 5), (1.0, 5)]
    if direction < 0:
        path = path[::-1]
    refs = [_spiral_point(theta, winding) for theta, winding in path]
    edges = [[i, i + 1] for i in range(len(refs) - 1)]
    first, last = path[0], path[-1]
    patches_pts = [[_spiral_point(first[0] + 0.05, first[1]), _spiral_point(first[0] + 0.1, first[1])],
                   [_spiral_point(last[0] + 0.05, last[1]), _spiral_point(last[0] + 0.1, last[1])]]
    crossing_map, atlas, patches = _setup(refs, patches_pts, edges)
    w = lambda theta, winding: winding + theta / TWO_PI
    p1 = {'_theta_node_id': 0, 'winding_annotation': 1.0, 'w_cont': w(*first),
          'on_patch': {'id': 'p1', 'ij': [0.0, 0.0]}}
    p2 = {'_theta_node_id': len(refs) - 1, 'winding_annotation': 1.0, 'w_cont': w(*last),
          'on_patch': {'id': 'p2', 'ij': [0.0, 0.0]}}
    middle = [{'_theta_node_id': i} for i in range(1, len(refs) - 1)]
    pcl = {'metadata': _sheet_meta(), 'points_by_patch': {'p1': [p1], 'p2': [p2]},
           'chain': _Chain([p1, *middle, p2])}
    cfg = _cfg(abs_winding_target_convention='sheet')
    strata = losses.build_pcl_sampling_strata(['absolute'], cfg)
    loss = losses.get_patch_rel_winding_loss(
        _IdentityTransform(), torch.tensor(DR), patches, atlas, [pcl], strata,
        crossing_map=crossing_map, cfg=cfg, z_begin=-1, z_end=1).item()
    assert loss < 1e-4


# --- point loss ------------------------------------------------------------------------------------------------

def _point_pcl(points):
    return {'metadata': _sheet_meta(), 'points': {i: p for i, p in enumerate(points)}}


def _point_loss(pcls, transform=None, **cfg):
    return losses.get_point_abs_winding_loss(
        transform or _IdentityTransform(), torch.tensor(DR), pcls,
        cfg=_cfg(**cfg), z_begin=-1, z_end=1)


@pytest.mark.parametrize('alpha', [0.0, 0.6, -0.6, 2.5, -2.5, TWO_PI + 0.1])
@pytest.mark.parametrize('phi0', [0.2, TWO_PI - 0.02])
def test_point_loss_matches_the_analytic_oracle(alpha, phi0):
    n = 6
    w = n + phi0 / TWO_PI
    lifted = phi0 + alpha
    zyx = [0.0, math.sin(lifted % TWO_PI) * DR * (n + lifted / TWO_PI), math.cos(lifted % TWO_PI) * DR * (n + lifted / TWO_PI)]
    q_true = n + math.floor(lifted / TWO_PI)
    t_true = round(w - (lifted % TWO_PI) / TWO_PI)
    loss = _point_loss([_point_pcl([{'p': zyx[::-1], 'w_cont': w}])]).item()
    assert loss == pytest.approx(abs(q_true - t_true) * DR, abs=1e-3)


def test_point_loss_wrong_anchor_gradient_and_masks():
    n, phi0 = 6, 1.0
    zyx = [0.0, math.sin(phi0) * DR * (n + phi0 / TWO_PI), math.cos(phi0) * DR * (n + phi0 / TWO_PI)]
    w = n + phi0 / TWO_PI
    assert _point_loss([_point_pcl([{'p': zyx[::-1], 'w_cont': w + 1}])]).item() == pytest.approx(DR, rel=1e-4)
    # radially 0.4 winding too far out: gradient pushes the radius inward
    far = torch.tensor([0.0, math.sin(phi0), math.cos(phi0)]) * DR * (n + 0.4 + phi0 / TWO_PI)

    class _Shift:
        def __init__(self):
            self.offset = torch.zeros(3, requires_grad=True)

        def __call__(self, points):
            return points + self.offset

    shift = _Shift()
    pcl = _point_pcl([{'p': far.tolist()[::-1], 'w_cont': w}])
    _point_loss([pcl], transform=shift).backward()
    radial = torch.tensor([0.0, math.sin(phi0), math.cos(phi0)])
    assert float(shift.offset.grad @ radial) > 0      # increasing the radius increases the loss
    # outside z, or rejected: masked to an exact, finite zero
    assert _point_loss([_point_pcl([{'p': [zyx[2], zyx[1], 5.0], 'w_cont': w}])]).item() == 0.0
    assert _point_loss([_point_pcl([{'p': zyx[::-1], 'w_cont': w + 0.5}])]).item() == 0.0


# --- held-out patches, non-finite abort, audit -------------------------------------------------------------------

def _patch(name, holdout, weight=1.0):
    return SimpleNamespace(entry_name=name, area=np.float32(10.0), loss_weight_scale=weight, holdout=holdout,
                           inner_core_radius=None, source_fiber=None)


def _fake_fit(**config):
    full = Config().as_dict()
    full.update(config)
    return SimpleNamespace(config=full)


def test_held_patch_gets_no_sampling_and_no_attachment():
    patches = [_patch('band-a', False), _patch('band-b', True, weight=1.0)]
    fit = _fake_fit(exclude_holdout_patches_from_training=True)
    fit._held_out_of_training = lambda patch: fit_spiral.FitContext._held_out_of_training(fit, patch)
    probabilities = fit_spiral.FitContext._patch_sampling_probabilities(fit, patches)
    assert probabilities.tolist() == [1.0, 0.0]
    excluded, _ = fit_spiral.FitContext._fiber_link_exclusions(fit, {}, {'a': patches[0], 'b': patches[1]})
    assert excluded == frozenset({'b'})
    off = _fake_fit()
    assert fit_spiral.FitContext._patch_sampling_probabilities(off, patches).tolist() == [0.5, 0.5]
    assert fit_spiral.FitContext._fiber_link_exclusions(off, {}, {'a': patches[0], 'b': patches[1]})[0] == frozenset()


def test_holdout_flag_round_trips_through_the_worker_payload(tmp_path):
    from tifxyz import Patch, patch_from_payload, patch_to_payload
    zyxs = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3) + 1
    patch = Patch(zyxs, torch.tensor([1.0, 1.0]), None, None, holdout=True)
    assert patch_from_payload(patch_to_payload(patch)).holdout is True


def test_sanitize_reports_nonfinite_steps():
    grad = torch.tensor([1.0, float('nan')])
    param = SimpleNamespace(grad=grad)
    fit = SimpleNamespace(dist_grad_named=[('w', param)], nonfinite_grad_steps=torch.zeros(()),
                          nonfinite_grad_by_param={'w': torch.zeros(())})
    assert bool(fit_spiral.FitContext._sanitize_nonfinite_grads_(fit))
    assert torch.isfinite(param.grad).all()
    assert not bool(fit_spiral.FitContext._sanitize_nonfinite_grads_(fit))


def test_audit_counts_records(tmp_path):
    n, phi0 = 6, 1.0
    good = [0.0, math.sin(phi0) * DR * (n + phi0 / TWO_PI), math.cos(phi0) * DR * (n + phi0 / TWO_PI)]
    records = [
        {'p': good[::-1], 'w_cont': n + phi0 / TWO_PI},          # correct
        {'p': good[::-1], 'w_cont': n + 1 + phi0 / TWO_PI},      # off by one
        {'p': good[::-1], 'w_cont': n + phi0 / TWO_PI + 0.5},    # rejected
        {'p': [good[2], good[1], 50.0], 'w_cont': 1.0},          # outside z
    ]
    path = tmp_path / 'records.json'
    path.write_text(json.dumps(records))
    fit = SimpleNamespace(
        config={'abs_winding_audit_records': str(path), 'abs_winding_audit_interval': 1000},
        dist=SimpleNamespace(is_main_process=True), z_begin=-1, z_end=1, device=torch.device('cpu'),
        out_path=str(tmp_path),
        spiral_and_transform=SimpleNamespace(get_slice_to_spiral_transform=lambda: _IdentityTransform(),
                                             get_dr_per_winding=lambda: torch.tensor(DR)))
    assert fit_spiral.FitContext._maybe_abs_winding_audit(fit, 999) is None
    entry = fit_spiral.FitContext._maybe_abs_winding_audit(fit, 1000)
    assert (entry['attempted'], entry['off'], entry['rejected'], entry['nonfinite'], entry['outside_z']) == (3, 2, 1, 0, 1)
    final = fit_spiral.FitContext._maybe_abs_winding_audit(fit, 1234, final=True)
    assert final['final'] and len((tmp_path / 'abs_winding_audit.jsonl').read_text().splitlines()) == 2
