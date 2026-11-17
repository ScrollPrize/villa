"""Piece-wise fiber approval helpers share the native fingerprint and crop semantics."""
import uuid

import pytest

from vc_object_store import (
    FIBER_LOCK_CONTROL_POINT_TAG, FIBER_PIECE_APPROVAL_TABLE, approve_fiber_piece,
    crop_fiber_piece_approvals, extension_uuid, fiber_piece_approval_from_row,
    fiber_piece_approval_states, fiber_piece_approval_table, fiber_piece_approval_to_row,
    fingerprint_fiber_piece, revoke_fiber_piece_approval,
)


def fiber(controls=8, locks=()):
    """Controls ids 10..80 on the x axis, one line point per unit."""
    annotations = []
    for i in range(controls):
        a = {"direction": [1.0, 0.0, 0.0], "tags": [FIBER_LOCK_CONTROL_POINT_TAG] if i in locks else []}
        if i + 1 < controls:
            a["segment_to_next"] = {"goal": "trace", "mode": "auto"}
        annotations.append(a)
    return dict(control_ids=[10 * (i + 1) for i in range(controls)],
                control_points=[[10.0 * i, 0.0, 0.0] for i in range(controls)],
                line_points=[[float(x), 0.0, 0.0] for x in range(10 * (controls - 1) + 1)],
                annotations=annotations)


def approve(f, a, b, inside, aspect="geometry", author="alice"):
    return approve_fiber_piece(f, str(uuid.uuid4()), a, b, inside, aspect, 3, author, "2026-11-17T10:00:00Z", "host")


def test_table_and_row_codec(db):
    f = fiber(locks=(3, 5))
    row = approve(f, 40, 60, 50)
    assert [c[0] for c in row["controls"]] == [40, 50, 60]
    assert [(s[0], s[1]) for s in row["spans"]] == [(40, 50), (50, 60)]
    values = fiber_piece_approval_to_row(row)
    assert len(values) == len(fiber_piece_approval_table()["columns"])
    assert fiber_piece_approval_from_row(values) == row
    with pytest.raises(ValueError):
        fiber_piece_approval_to_row(dict(row, inside_cp=60))
    with db.write() as tx:
        tx.create_extension_table(fiber_piece_approval_table())
        tx.append_extension_rows(FIBER_PIECE_APPROVAL_TABLE, [values])
    stored = db.extension_rows(FIBER_PIECE_APPROVAL_TABLE, key_prefix=[extension_uuid(row["fiber_uuid"])])
    assert [fiber_piece_approval_from_row(r) for r in stored] == [row]


def test_fingerprints_follow_the_native_rules():
    f = fiber()
    base = fingerprint_fiber_piece(f, 40, 60, 50)
    moved = fiber(); moved["line_points"][45][1] = 0.5
    changed = fingerprint_fiber_piece(moved, 40, 60, 50)
    assert changed["spans"][0] == base["spans"][0] and changed["spans"][1] != base["spans"][1]
    assert changed["controls"] == base["controls"]
    locked = fiber(locks=(4,))
    assert fingerprint_fiber_piece(locked, 40, 60, 50) == base
    normals = fingerprint_fiber_piece(f, 40, 60, 50, "normals")
    assert normals["spans"] == [] and len(normals["controls"]) == 3
    assert fingerprint_fiber_piece(moved, 40, 60, 50, "normals") == normals
    with pytest.raises(ValueError):
        fingerprint_fiber_piece(f, 40, 60, 30)


def test_states_revocation_and_crop():
    f = fiber(locks=(0, 4, 5, 7))
    a = approve(f, 10, 60, 20)
    c = approve(f, 60, None, 70, author="carol")
    states = fiber_piece_approval_states(f, [a, c])
    assert states["approved_spans"]["geometry"] == [0, 1, 2, 3, 4, 5, 6]
    assert states["row_current"] == [True, True]
    assert sorted(states["lock_cp_ids"]) == [10, 60]
    # B opens [4..7]: A is cropped to [0..4], C is revoked.
    crops = crop_fiber_piece_approvals(f, [a, c], 4, 7, "bob", "2026-11-17T12:00:00Z")
    by_approval = {r["approval_uuid"]: r for r in crops}
    cropped = by_approval[a["approval_uuid"]]
    assert cropped["state"] == "approved" and cropped["lock_cp_b"] == 50 and cropped["supersedes"] == a["row_uuid"]
    assert cropped["author"] == "alice" and cropped["changed_by"] == "bob"
    assert by_approval[c["approval_uuid"]]["state"] == "revoked"
    rows = [a, c] + crops
    states = fiber_piece_approval_states(f, rows)
    assert states["approved_spans"]["geometry"] == [0, 1, 2, 3]
    assert states["releasable_lock_cp_ids"] == [60]
    rows.append(approve(f, 50, 80, 60, author="bob"))
    assert fiber_piece_approval_states(f, rows)["approved_spans"]["geometry"] == [0, 1, 2, 3, 4, 5, 6]
    revoked = revoke_fiber_piece_approval(rows[-1], "bob", "t")
    assert revoked["state"] == "revoked" and revoked["spans"] == []
    rows.append(revoked)
    assert fiber_piece_approval_states(f, rows)["approved_spans"]["geometry"] == [0, 1, 2, 3]
