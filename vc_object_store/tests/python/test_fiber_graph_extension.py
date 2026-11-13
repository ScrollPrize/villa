"""One typed extension stores independent stages/runs without copying geometry."""
import json
import uuid

import pytest

from vc_object_store import (
    extension_uuid, fiber_source_extension, fiber_constraint_extension,
    fiber_graph_solve_table, fiber_graph_solve_row, fiber_graph_run_row,
    fiber_graph_input_row,
)


ROLES = ("runs", "inputs", "fibers", "samples", "hypotheses", "widths",
         "positions", "relations", "metrics")


def unpack(spec, row):
    return dict(zip((c["name"] for c in spec["columns"]), row))


def append(tx, row):
    tx.append_extension_rows(row["table"], [row["values"]])


def test_workflow_two_runs_share_graph(db):
    fiber = db.upsert_object(dict(kind="fiber"))
    other = db.upsert_object(dict(kind="fiber"))
    edge = db.add_edge(dict(kind="test_link", a=fiber, b=other,
                            a_anchor=dict(whole_geometry=True), b_anchor=dict(whole_geometry=True)))
    source_state = db.status()
    source_fiber_version = db.current_version(fiber)
    source = dict(store_uuid=source_state["store"], generation=source_state["generation"],
                  commit_sequence=source_state["commit_sequence"])
    specs = {role: fiber_graph_solve_table(role) for role in ROLES}
    runs = [str(uuid.uuid4()), str(uuid.uuid4())]
    f, e = extension_uuid(fiber), extension_uuid(edge)
    with db.write() as tx:
        for spec in specs.values():
            assert spec["extension"] == "ext_fiber_graph_solve" and spec["format_version"] == 1
            assert len(json.dumps(spec["provisioning"])) < 256
            assert spec["based_on"] == []  # Inputs differ by run, not by physical table.
            tx.create_extension_table(spec)
        for i, run in enumerate(runs):
            r = extension_uuid(run)
            append(tx, fiber_graph_run_row(run, "winding", "irls", {"max_iterations": 100, "robust": True}))
            append(tx, fiber_graph_input_row(run, "graph", source))
            if i:
                append(tx, fiber_graph_input_row(run, "initialization", source, runs[0]))
            append(tx, fiber_graph_solve_row("fibers", dict(
                fiber_uuid=f, run_uuid=r, family=0, enabled=1, p_h=0.85, p_v=0.05, p_broken=0.1,
                normal_sign=-1, sign_probability=0.2, winding_mean=5.5, winding_sigma=0.7,
                selected_winding=5+i, selected_probability=0.6, joint_probability=0.54,
                anchor_arc=0.0, anchor_u_turns=2.3, anchor_v_vx=30000.0)))
            for k in (5, 6):
                append(tx, fiber_graph_solve_row("hypotheses", dict(
                    fiber_uuid=f, run_uuid=r, winding=k, probability=0.5)))
                append(tx, fiber_graph_solve_row("widths", dict(run_uuid=r, winding=k, width_vx=12000.0+k)))
            append(tx, fiber_graph_solve_row("samples", dict(
                fiber_uuid=f, run_uuid=r, sample_index=0, arc=100.0, direction_x=1.0,
                direction_y=0.0, direction_z=0.0, normal_valid=1, normal_x=0.0, normal_y=1.0, normal_z=0.0)))
            append(tx, fiber_graph_solve_row("samples", dict(
                fiber_uuid=f, run_uuid=r, sample_index=1, arc=200.0, normal_valid=0)))
            for layer in range(4):
                append(tx, fiber_graph_solve_row("positions", dict(
                    fiber_uuid=f, run_uuid=r, layer=layer, hypothesis=0 if layer == 0 else 5+i,
                    node_index=0, arc=100.0, edge_uuid=e, endpoint=0,
                    u_turns=2.3, u_vx=None if layer < 2 else 70000.0, v_vx=30001.0)))
            append(tx, fiber_graph_solve_row("relations", dict(
                edge_uuid=e, run_uuid=r, layer=2, evaluation_index=0, included=1,
                hypothesis_a=5+i, hypothesis_b=5+i, fulfilled=1, sign_violation=0,
                probability=0.7, same_sheet_probability=0.65, meeting_weight=0.5,
                required_step=0.0, realized_step=0.01, winding_error=0.01,
                u_error_vx=0.6, v_error_vx=0.4, loss=0.52)))
            for it in (0, 100):
                append(tx, fiber_graph_solve_row("metrics", dict(
                    run_uuid=r, iteration=it, scope="gt/perpendicular", name="sign_violations",
                    count=0, units="constraints")))
                append(tx, fiber_graph_solve_row("metrics", dict(
                    run_uuid=r, iteration=it, scope="gt/perpendicular", name="l1",
                    value=0.125+i*0.01, units="turns")))
    for i, run in enumerate(runs):
        r = extension_uuid(run)
        row, = db.extension_rows(specs["fibers"]["name"], [f, r])
        state = unpack(specs["fibers"], row)
        assert state["winding_mean"] == 5.5 and state["anchor_u_turns"] == 2.3
        assert state["selected_winding"] == 5+i and state["component"] is None
        positions = db.extension_rows(specs["positions"]["name"], [f, r])
        assert len(positions) == 4
        assert all(unpack(specs["positions"], p)["u_turns"] == 2.3 for p in positions)
        metric_rows = db.extension_rows(specs["metrics"]["name"], [r, 100])
        assert len(metric_rows) == 2
        inputs = db.extension_rows(specs["inputs"]["name"], [r])
        assert len(inputs) == 1+i
        assert all(unpack(specs["inputs"], x)["commit_sequence"] == source["commit_sequence"] for x in inputs)
    assert db.current_version(fiber) == source_fiber_version  # Results never edit the input fiber.
    assert db.status()["commit_sequence"] > source["commit_sequence"]


def test_metadata_is_constructed_not_copied():
    source = str(uuid.uuid4())
    info = fiber_source_extension(source, fiber_count=100, point_count=2000)
    assert info["provisioning"] == dict(source_uuid=source, fiber_count=100, point_count=2000)
    assert fiber_source_extension("")["provisioning"] == {}
    constraints = fiber_constraint_extension(source, constraint_count=12)
    assert constraints["provisioning"] == dict(fiber_source_uuid=source, constraint_count=12,
        length_units="base_voxels", winding_units="turns", endpoint_order="source")
    with pytest.raises(TypeError):
        fiber_source_extension({"uuid": source, "selected_ids": list(range(100)), "path": "/not/copied"})
    run = str(uuid.uuid4())
    for settings in ({"selected_ids": [1, 2]}, {"source_path": "/not/copied"}, {"report": {"value": 4}}):
        with pytest.raises(ValueError):
            fiber_graph_run_row(run, "winding", "irls", settings)


def test_fiber_graph_rejects_invalid_rows():
    f, r = extension_uuid(str(uuid.uuid4())), extension_uuid(str(uuid.uuid4()))
    for fields in (dict(p_h=1.1), dict(winding_mean=float("nan")), dict(normal_sign=0),
                   dict(family=3), dict(enabled=2), dict(unknown=1)):
        with pytest.raises(ValueError):
            fiber_graph_solve_row("fibers", dict(fiber_uuid=f, run_uuid=r, **fields))
    with pytest.raises(ValueError):
        fiber_graph_solve_row("samples", dict(fiber_uuid=f, run_uuid=r, sample_index=0, arc=0.0, normal_valid=1))
    with pytest.raises(ValueError):
        fiber_graph_solve_row("metrics", dict(run_uuid=r, iteration=0, scope="all", name="bad", units="turns"))
    with pytest.raises(ValueError):
        fiber_graph_solve_table("unknown")
