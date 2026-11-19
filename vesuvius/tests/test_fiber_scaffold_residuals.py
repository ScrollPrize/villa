"""The scaffold solve's residual rows, read back as the approval ranking sees them."""
import uuid

import pytest

from vc_object_store import (
    extension_uuid, fiber_graph_solve_row, fiber_graph_solve_table, import_initial_records,
)

from vesuvius.neural_tracing.fiber_trace_3d.fiber_wgraph_stages import (
    solve_residual_pieces, solve_residuals,
)


def _store(tmp_path, run, fibers):
    """One run of two violated constraints, the second with two stretches."""
    residuals = fiber_graph_solve_table('residuals')
    pieces = fiber_graph_solve_table('residual_pieces')
    rows = {residuals['name']: [], pieces['name']: []}
    rows[residuals['name']].append(fiber_graph_solve_row('residuals', dict(
        run_uuid=extension_uuid(run), residual_index=1, kind=1,
        fiber_a=extension_uuid(fibers[0]), fiber_b=extension_uuid(fibers[1]),
        target=1, weight=2.0, residual=0.5, arc_a=10.0, arc_b=20.0,
        cycle_hops=3, cycle_piece_length=120.0, redundancy=2))['values'])
    rows[residuals['name']].append(fiber_graph_solve_row('residuals', dict(
        run_uuid=extension_uuid(run), residual_index=0, kind=2,
        fiber_a=extension_uuid(fibers[1]), fiber_b=extension_uuid(fibers[0]),
        target=0, weight=1.0, residual=1.0))['values'])
    rows[pieces['name']].append(fiber_graph_solve_row('residual_pieces', dict(
        run_uuid=extension_uuid(run), residual_index=1, member_index=1,
        fiber_uuid=extension_uuid(fibers[1]), arc_from=5.0, arc_to=35.0,
        control_from=2, control_to=3, score=0.25))['values'])
    rows[pieces['name']].append(fiber_graph_solve_row('residual_pieces', dict(
        run_uuid=extension_uuid(run), residual_index=1, member_index=0,
        fiber_uuid=extension_uuid(fibers[0]), arc_from=0.0, arc_to=10.0,
        score=0.75))['values'])
    database = tmp_path / 'scaffold.vcobj'
    import_initial_records(
        database,
        [dict(record=dict(uuid=fiber, kind='fiber'), extensions=[]) for fiber in fibers],
        extensions=[residuals, pieces])
    from vc_object_store import write_extension_store
    overlay = tmp_path / 'runs.vcobj'
    write_extension_store(overlay, [database], [residuals, pieces], rows)
    return overlay


def test_residuals_read_back_in_index_order_with_uuid_text(tmp_path):
    run = str(uuid.uuid4())
    fibers = sorted(str(uuid.uuid4()) for _ in range(2))
    database = _store(tmp_path, run, fibers)
    read = solve_residuals(database, run)
    assert [entry['residual_index'] for entry in read] == [0, 1]
    assert read[0]['run_uuid'] == run and read[0]['fiber_a'] == fibers[1]
    assert read[0]['edge_uuid'] is None and read[0]['cycle_hops'] is None
    violated = read[1]
    assert violated['weight'] == 2.0 and violated['residual'] == 0.5
    assert violated['cycle_hops'] == 3 and violated['redundancy'] == 2
    assert violated['cycle_piece_length'] == 120.0


def test_residual_pieces_are_grouped_by_residual_and_ordered_by_member(tmp_path):
    run = str(uuid.uuid4())
    fibers = sorted(str(uuid.uuid4()) for _ in range(2))
    database = _store(tmp_path, run, fibers)
    pieces = solve_residual_pieces(database, run)
    assert set(pieces) == {1}
    members = pieces[1]
    assert [entry['member_index'] for entry in members] == [0, 1]
    assert [entry['fiber_uuid'] for entry in members] == [fibers[0], fibers[1]]
    # A stretch whose fiber carries no controls there states none.
    assert members[0]['control_from'] is None and members[0]['control_to'] is None
    assert members[1]['control_from'] == 2 and members[1]['control_to'] == 3
    assert sum(entry['score'] for entry in members) == pytest.approx(1.0)


def test_a_store_without_the_tables_is_an_error_not_an_empty_answer(tmp_path):
    database = tmp_path / 'bare.vcobj'
    import_initial_records(database, [dict(record=dict(uuid=str(uuid.uuid4()), kind='fiber'), extensions=[])])
    with pytest.raises(ValueError):
        solve_residuals(database, str(uuid.uuid4()))
    with pytest.raises(ValueError):
        solve_residual_pieces(database, str(uuid.uuid4()))
