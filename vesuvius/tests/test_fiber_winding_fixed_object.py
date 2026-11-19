"""Rigid groups in the winding stage: contraction, rewriting and expansion."""
import json

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_trace_3d.fiber_wgraph_geometry import (
    _contract_fixed_groups, fixed_object_groups, fixed_object_settings,
)


def windings(tmp_path, ids, selected, component):
    path = tmp_path / 'windings.json'
    path.write_text(json.dumps(dict(ids=list(ids), selected_winding=list(selected), component=list(component))))
    return dict(windings=str(path))


def test_groups_need_two_present_fibers(tmp_path):
    ids = ['a', 'b', 'c', 'd']
    settings = dict(fixed_object=windings(tmp_path, ['a', 'b', 'x'], [3, 5, 9], [0, 0, 1]))
    root, offset, report = fixed_object_groups(settings, ids)
    # a and b are one group with b two windings outside a; x is not in this
    # store, so its component fixes nothing.
    assert list(root) == [0, 0, -1, -1]
    assert list(offset) == [0, 2, 0, 0]
    assert report['groups'] == 1 and report['fixed_fibers'] == 2
    assert report['present'] == 2 and report['solve_components'] == 2

    alone = dict(fixed_object=windings(tmp_path, ['a', 'x'], [1, 2], [0, 1]))
    root, offset, report = fixed_object_groups(alone, ids)
    assert root is None and report['groups'] == 0 and report['single_fiber_components'] == 1


def test_arcs_move_to_the_root_and_internal_ones_are_dropped(tmp_path):
    ids = ['a', 'b', 'c']
    settings = dict(fixed_object=windings(tmp_path, ['a', 'b'], [10, 12], [0, 0]))
    root, offset, report = fixed_object_groups(settings, ids)
    merged = dict(a=np.array([0, 1, 0, 1], np.int64), b=np.array([1, 2, 2, 0], np.int64),
                  target=np.array([2, 1, 4, -3], np.int64), weight=np.array([1., 2., 3., 4.]))
    out = _contract_fixed_groups(merged, root, offset, report)
    # The a-b arcs are internal: the group already fixes that relation, and
    # one of them disagrees with it by one turn.
    assert report['internal_arcs'] == 2 and report['internal_weight'] == 5.
    assert report['internal_disagreeing_arcs'] == 1 and report['internal_disagreement_turns'] == 1.
    # b -> c and a -> c both leave the group: b's arc moves to a (the root)
    # with its target shifted by b's offset.
    assert list(out['a']) == [0, 0] and list(out['b']) == [2, 2]
    assert list(out['target']) == [1 + 2, 4]
    assert list(out['weight']) == [2., 3.]
    assert report['rewritten_arcs'] == 2 and report['remaining_arcs'] == 2


def test_a_group_the_store_does_not_hold_is_no_group(tmp_path):
    settings = dict(fixed_object=windings(tmp_path, ['x', 'y'], [0, 1], [0, 0]))
    root, offset, report = fixed_object_groups(settings, ['a', 'b'])
    assert root is None and offset is None and report['groups'] == 0


def test_settings_reject_an_unknown_key_and_a_missing_path():
    assert fixed_object_settings({}) is None
    with pytest.raises(ValueError):
        fixed_object_settings(dict(fixed_object=dict(windings='w.json', extra=1)))
    with pytest.raises(ValueError):
        fixed_object_settings(dict(fixed_object=dict()))


def test_classification_modes_are_validated():
    """run+record is accepted; the sign-free mode still needs the tags alone."""
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_wgraph_geometry import winding_settings
    base = dict(weights=dict(parallel=1, perpendicular=1, enclosure=0, manual=0))
    assert winding_settings(dict(base, classification='run+record'))['classification'] == 'run+record'
    assert winding_settings(dict(base))['classification'] == 'run'
    with pytest.raises(ValueError):
        winding_settings(dict(base, classification='tags'))
    with pytest.raises(ValueError):
        # Without a classification run there is nothing to pin, so a run that
        # solves no signs needs the tags alone.
        winding_settings(dict(weights=dict(parallel=0, perpendicular=0, enclosure=0, manual=1),
                              classification='run+record', normal_sign='none'))
