"""The constraint cut (fiber_constraint_cut) on small synthetic graphs: a
bridge crossing above the threshold splits an island off the largest
component, one below does not, a satisfied parallel crossing keeps a pair
together; with nothing cut the solve's components come back; crossings
between solve components never merge them; connectivity the crossings do not
explain is kept joined and counted, while a cut inside it still splits. Then what the cut does to the largest component's gauge
and outlier backstop, and the checkpoint identity a resume compares."""
from types import SimpleNamespace

import numpy as np
import pytest


def cut(component, edges, cut_turns):
    """cut_components over ``edges`` = [(a, b, violation), ...]."""
    from fiber_constraint_cut import cut_components
    a, b, violation = (np.array(column) for column in zip(*edges)) if edges else (
        np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0))
    return cut_components(np.asarray(component), a, b, violation, cut_turns)


# Fibers 0-3 a chain, 4-5 an island tied to it by one crossing 3-4; fiber 6
# a component of its own in the solve.
SOLVE = [7, 7, 7, 7, 7, 7, 9]
CHAIN = [(0, 1, 0.0), (1, 2, 0.5), (2, 3, 1.0), (4, 5, 0.0)]


@pytest.mark.parametrize('bridge, splits', [(5.0, True), (4.0, False), (3.5, False)])
def test_a_bridge_violated_above_the_threshold_splits_the_island_off(bridge, splits):
    result = cut(SOLVE, CHAIN + [(3, 4, bridge)], 4.0)
    if splits:
        np.testing.assert_array_equal(result['component'], [0, 0, 0, 0, 1, 1, 2])
        assert (result['cut'], result['components'], result['largest']) == (1, 3, 4)
        assert result['left_largest'] == 2
    else:
        np.testing.assert_array_equal(result['component'], [0, 0, 0, 0, 0, 0, 1])
        assert (result['cut'], result['left_largest']) == (0, 0)
    assert result['reproduces'] and result['joined_fibers'] == 0
    assert (result['solve_components'], result['solve_largest']) == (2, 6)
    assert result['histogram'][2] == (4, int(bridge > 4))


def test_a_pair_stays_joined_while_one_of_its_crossings_holds():
    result = cut(SOLVE, CHAIN + [(3, 4, 9.0), (4, 3, 0.2)], 4.0)
    assert result['cut'] == 1
    np.testing.assert_array_equal(result['component'], [0, 0, 0, 0, 0, 0, 1])


def test_nothing_cut_reproduces_the_solve_components():
    from fiber_constraint_cut import cut_enabled, partition_sha256
    assert not any(cut_enabled(value) for value in (None, 0, 0.0, -1))
    assert cut_enabled(4.0)
    result = cut(SOLVE, CHAIN + [(3, 4, 30.0)], 1e9)
    assert result['reproduces'] and result['cut'] == 0
    assert partition_sha256(result['component']) == partition_sha256(SOLVE)


def test_crossings_between_solve_components_are_dropped_not_merged():
    result = cut(SOLVE, CHAIN + [(3, 4, 0.0), (5, 6, 0.0)], 4.0)
    assert result['between_components'] == 1 and result['reproduces']
    np.testing.assert_array_equal(result['component'], [0, 0, 0, 0, 0, 0, 1])


def test_connectivity_the_crossings_do_not_explain_is_kept_joined():
    # The solve's component 7 reads as two pieces, 0-3 and 4-5: something not
    # in the store (a scaffold reference, a fixed group) joined them.
    for cut_turns in (1e9, 4.0):
        result = cut(SOLVE, CHAIN, cut_turns)
        np.testing.assert_array_equal(result['component'], [0, 0, 0, 0, 0, 0, 1])
        assert not result['reproduces']
        assert (result['joined_pieces'], result['joined_fibers']) == (1, 2)
        assert (result['cut'], result['left_largest']) == (0, 0)


def test_a_cut_inside_a_joined_piece_still_splits_it():
    # Piece 4-5-6-7 (joined to the main piece only by unread links) loses
    # fiber 7 to a cut crossing; the rest of it stays attached.
    solve = [1] * 8 + [2]
    edges = [(0, 1, 0), (1, 2, 0), (2, 3, 0), (4, 5, 0), (5, 6, 0), (6, 7, 9.0)]
    result = cut(solve, edges, 4.0)
    np.testing.assert_array_equal(result['component'], [0, 0, 0, 0, 0, 0, 0, 1, 2])
    assert result['left_largest'] == 1
    # A cut inside the main piece: its larger part keeps the joined piece.
    edges = [(0, 1, 0), (1, 2, 0), (2, 3, 9.0), (4, 5, 0)]
    result = cut([1] * 6, edges, 4.0)
    np.testing.assert_array_equal(result['component'], [0, 0, 0, 1, 0, 0])


def test_the_partition_digest_ignores_label_values():
    from fiber_constraint_cut import canonical_labels, partition_sha256, same_partition
    np.testing.assert_array_equal(canonical_labels([5, 5, 2, 9, 2]), [0, 0, 1, 2, 1])
    assert partition_sha256([5, 5, 2, 9, 2]) == partition_sha256([0, 0, 3, 1, 3])
    assert partition_sha256([5, 5, 2, 9, 2]) != partition_sha256([5, 5, 2, 2, 2])
    assert same_partition([1, 1, 2], [4, 4, 0]) and not same_partition([1, 1, 2], [1, 2, 2])


def test_the_violation_is_the_solve_step_against_the_crossing_target():
    from fiber_constraint_cut import violation
    np.testing.assert_allclose(
        violation([0, 3, -2], np.array([0, 1]), np.array([1, 2]), [2.25, -9.5]), [0.75, 4.5])


def test_the_report_names_the_cut_and_the_components():
    from fiber_constraint_cut import cut_report
    lines = cut_report(cut(SOLVE, CHAIN + [(3, 4, 5.0)], 4.0), seconds=1.5)
    assert 'cut 1 above 4 turns' in lines[0]
    assert 'reproduce the solve\'s 2 components exactly' in lines[1]
    assert lines[2].startswith('fiber constraints: components 2 -> 3, largest 6 -> 4 fibers; '
                               '2 fibers left the largest component')


def gauge_solve(fiber_cc, w_min, w_max):
    fiber_cc = np.asarray(fiber_cc, dtype=np.int64)
    return SimpleNamespace(c=0.0, fiber_cc=fiber_cc, largest_cc=0,
                           w_min=np.asarray(w_min, float), w_max=np.asarray(w_max, float))


def test_a_cut_island_no_longer_sets_the_gauge_and_the_backstop_drops_nothing():
    """An island whose labels sit five windings below the rest is tied to the
    largest component by one badly violated crossing. Uncut, it sets n_low
    (the backstop's margin does not reach it); cut, it is a component of its
    own and the gauge rests on the correctly labelled fibers."""
    from config import Config
    from fiber_cc_loss import largest_cc_gauge
    config = Config({'fiber_largest_cc_min_percentile': 20.0}).as_dict()
    labels_low = [10, 10, 11, 12, 13, 13, 5, 5]
    labels_high = [20, 20, 21, 22, 23, 23, 6, 6]
    solve = [0] * 8
    edges = [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0), (4, 5, 0), (6, 7, 0), (5, 6, 5.0)]
    for cut_turns, n_low in ((1e9, 5), (4.0, 10)):
        component = cut(solve, edges, cut_turns)['component']
        gauge = largest_cc_gauge(gauge_solve(component, labels_low, labels_high), config)
        assert gauge['n_low'] == n_low and not gauge['dropped'].any()
    # A fiber whose labels are wrong far beyond the margin, with no crossing
    # violated enough to cut it off, is still dropped by the backstop.
    config = Config({'fiber_largest_cc_min_percentile': 20.0,
                     'fiber_outlier_margin_turns': 1.0}).as_dict()
    gauge = largest_cc_gauge(gauge_solve(cut(solve, edges, 1e9)['component'],
                                         labels_low, labels_high), config)
    assert gauge['dropped'].tolist() == [False] * 6 + [True, True] and gauge['n_low'] == 10


def test_a_resume_compares_the_cut_threshold_and_components():
    from fit_spiral import winding_solve_identity
    base = dict(run_uuid='run', fiber_store_position={'commit_sequence': 1},
                run_store_position={'commit_sequence': 2})
    assert winding_solve_identity(SimpleNamespace(**base, constraint_cut=None)) == base
    identities = [winding_solve_identity(SimpleNamespace(**base, constraint_cut={
        'identity': {'turns': turns, 'components_sha256': digest}}))
        for turns, digest in ((4.0, 'a'), (4.0, 'b'), (3.0, 'a'), (4.0, 'a'))]
    assert identities[0]['constraint_cut'] == {'turns': 4.0, 'components_sha256': 'a'}
    assert identities[0] != identities[1] and identities[0] != identities[2]
    assert identities[0] == identities[3] and identities[0] != base


def test_the_setting_is_registered_as_a_new_fit_input():
    from config import Config, _PREPARED_INPUT_FIELDS, config_descriptions
    assert Config().fiber_constraint_cut_turns == 4.0
    assert Config({'fiber_constraint_cut_turns': None}).fiber_constraint_cut_turns is None
    assert 'fiber_constraint_cut_turns' in _PREPARED_INPUT_FIELDS
    assert 'fiber_constraint_cut_turns' in config_descriptions()
    with pytest.raises(ValueError):
        Config({'fiber_constraint_cut_turns': -1.0})
