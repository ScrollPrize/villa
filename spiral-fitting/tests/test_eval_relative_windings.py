"""eval_relative_windings scores relative windings against hand-traced w-meshes."""

import math

import numpy as np
import torch

import eval_relative_windings as ev
from tifxyz import Patch

RADIUS, DR = 800.0, 50.0


def _xyz(theta, winding):
    r = RADIUS + DR * (winding - 10)
    return [math.cos(theta) * r, math.sin(theta) * r, 1.0]


def _arc_patch(winding):
    r = RADIUS + DR * (winding - 10)
    thetas = np.linspace(-0.5, 0.5, 40)
    grid = np.stack([np.stack([[float(i), math.sin(t) * r, math.cos(t) * r] for t in thetas])
                     for i in range(3)]).astype(np.float32)
    return Patch(zyxs=torch.from_numpy(grid), scale=torch.tensor([1.0, 1.0]),
                 overlapping_ids=None, winding=None)


MESHES = [(w, _arc_patch(w)) for w in (10, 11, 12)]


def _collection(cid, windings, annotated=None, thetas=None):
    annotated = annotated if annotated is not None else windings
    thetas = thetas or [0.0, 0.05, -0.05, 0.1, -0.1, 0.15][:len(windings)]
    pts = {i: {'p': _xyz(t, w), 'winding_annotation': float(a), 'creation_time': i}
           for i, (w, a, t) in enumerate(zip(windings, annotated, thetas))}
    return {'id': cid, 'name': f'c{cid}', 'points': pts}


def test_perfect_annotation_scores_one_with_degenerate_interval():
    """A correct annotation must score exactly 1.0 / 0 error, else the scorer cannot be trusted as a control."""
    cols = {i: _collection(i, [10, 11, 12, 11]) for i in range(5)}
    r = ev.evaluate(cols, MESHES)
    assert r['exact_step_share'] == 1.0 and r['mean_abs_err'] == 0.0
    assert r['exact_step_share_ci95'] == [1.0, 1.0] and r['n_pairs'] == 15


def test_fractional_wind_a_step_is_rounded_not_truncated():
    """A 0.9-turn or 1.9-turn annotated step is a 1- or 2-turn step; truncation (int()) would halve it (jo-tud, PR #2448)."""
    pts = [{'winding_annotation': w, 'truth': t, 'creation_time': i}
           for i, (w, t) in enumerate([(0.0, 10), (1.9, 12), (2.8, 13), (2.9, 13)])]
    r = ev.score_collection(pts)
    assert r['n_pairs'] == 3 and r['n_exact'] == 3 and r['sum_abs_err'] == 0


def test_shifted_chain_lowers_exact_share_and_reversed_step_is_a_sign_violation():
    """A winding slip inside one chain is the error the tool exists to find; a reversed step is worse than a wrong magnitude."""
    good = {i: _collection(i, [10, 11, 12, 11]) for i in range(3)}
    shifted = _collection(9, [10, 11, 12, 11], annotated=[10, 11, 13, 12])  # +1 after position 2
    reversed_ = _collection(10, [10, 12, 11], annotated=[10, 12, 13])       # true -1 predicted +1
    r = ev.evaluate({**good, 9: shifted, 10: reversed_}, MESHES)
    assert r['exact_step_share'] < 1.0 and r['mean_abs_err'] > 0
    assert r['sign_violations'] == 1


def test_unlinked_point_is_counted_and_its_pairs_skipped():
    """A point far from every mesh must not be scored against a wrong sheet; silently linking it would inflate errors."""
    col = _collection(0, [10, 11, 12])
    col['points'][1]['p'] = [0.0, 0.0, 1.0]  # nowhere near any arc
    r = ev.evaluate({0: col}, MESHES)
    assert r['n_unlinked'] == 1 and r['n_pairs'] == 0


def test_bootstrap_is_deterministic_for_fixed_seed():
    """Intervals must reproduce between runs, otherwise two scorings of the same annotation disagree."""
    cols = {i: _collection(i, [10, 11, 12], annotated=[10, 11, 12 + (i % 2)]) for i in range(8)}
    rows = [ev.score_collection([{**p, 'truth': w, 'excluded': False}
                                 for p, w in zip(c['points'].values(), [10, 11, 12])])
            for c in cols.values()]
    a, b = ev.pooled_with_interval(rows), ev.pooled_with_interval(rows)
    assert a == b and a['exact_step_share_ci95'][0] < a['exact_step_share_ci95'][1]
    assert ev.pooled_with_interval(rows, seed=1) != a


def test_exclude_sector_drops_points_on_that_mesh_only():
    """The exclusion sector removes known-bad annotator regions; it must act on the named winding and angle only."""
    umb = {'control_points': [{'x': 0, 'y': 0, 'z': 0, 'score': 100}, {'x': 0, 'y': 0, 'z': 5, 'score': 100}]}
    col = _collection(0, [10, 11, 12], thetas=[0.0, 0.0, 0.0])  # theta = 0 deg for all
    ex = [{'winding': 11, 'theta_deg': [350, 10]}]  # wraps through 0
    r = ev.evaluate({0: col}, MESHES, umbilicus=umb, exclude=ex)
    assert r['n_excluded'] == 1 and r['n_pairs'] == 0


def test_prefilter_links_match_brute_force_projection():
    """The KD-tree prefilter is a speed-up only: every audit number rests on it giving the same links as
    projecting onto every face, including points at the edge of the tolerance and points on no mesh."""
    rng = np.random.default_rng(0)
    pts = []
    for w in (10, 11, 12, 13):  # 13 is the holed mesh below; its points also fall over the hole
        for t in rng.uniform(-0.45, 0.45, 25):
            x, y, z = _xyz(t, w)
            off = rng.uniform(-14.0, 14.0)  # radial offset spanning both sides of tolerance 10
            r = math.hypot(x, y) + off
            pts.append([math.cos(t) * r, math.sin(t) * r, rng.uniform(-2.0, 4.0)])
    pts += rng.uniform(-900, 900, (40, 3)).tolist()  # mostly nowhere near the arcs
    pts.append(_xyz(0.7, 11))  # past the arc ends
    grid = _arc_patch(13).zyxs.clone()
    grid[1, 15:25] = -1.0  # a strip of invalid vertices in the middle row (masks are set at construction)
    holed = Patch(zyxs=grid, scale=torch.tensor([1.0, 1.0]), overlapping_ids=None, winding=None)
    meshes = MESHES + [(13, holed)]
    for tol in (3.0, 10.0, 25.0):
        fast = ev.link_points(pts, meshes, tol, prefilter=True)
        slow = ev.link_points(pts, meshes, tol, prefilter=False)
        assert fast == slow, tol
    assert any(l is None for l in slow) and {l for l in slow if l is not None} == {10, 11, 12, 13}
