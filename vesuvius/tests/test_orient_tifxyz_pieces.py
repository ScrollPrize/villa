"""orient_tifxyz_pieces on a grid of three pieces around an axis at (500, 500):
A turns with increasing theta (N away from the axis), B against it (N toward
the axis) and sits next to A in the grid but far from it in space, C is flat
(N along z, so it gets no votes)."""
import json

import numpy as np
from vc_reader import tifxyz

from vesuvius.scripts import orient_tifxyz_pieces as op

STEP, R, CENTRE = 16.0, 400.0, (500.0, 500.0)


def _mosaic():
    rows, cols = 21, 40
    xyz = np.full((rows, cols, 3), -1, np.float32)
    for j in range(cols):
        theta = 0.04 * j if j < 20 else 3.0 - 0.04 * (j - 20)
        for i in range(10):
            xyz[i, j] = (CENTRE[0] + R * np.cos(theta), CENTRE[1] + R * np.sin(theta), 100 + STEP * i)
    for i in range(12, rows):
        for j in range(cols):
            xyz[i, j] = (100 + STEP * j, 900 + STEP * i, 2000)
    return xyz


def _pieces():
    xyz = _mosaic()
    valid = xyz[..., 0] != -1
    labels, count = op.label_pieces(xyz, valid, 2 * STEP)
    return xyz, valid, labels, count


def test_pieces_are_cut_at_long_edges_and_keep_their_edge_vertices():
    xyz, valid, labels, count = _pieces()
    assert count == 3
    a, b, c = labels[0, 0], labels[0, 39], labels[20, 0]
    assert len({a, b, c}) == 3
    assert (labels[:10, :20] == a).all() and (labels[:10, 20:] == b).all() and (labels[12:] == c).all()
    assert (labels[~valid] == 0).all()


def test_votes_decide_each_piece():
    xyz, valid, labels, count = _pieces()
    away, total = op.radial_votes(xyz, valid, labels, count, op.read_axis("500,500"), max_len=2 * STEP)
    side = op.decide(away, total)
    assert side[labels[0, 0]] == 1 and side[labels[0, 39]] == -1 and side[labels[20, 0]] == 0


def test_split_copies_vertices_and_reverses_only_the_inward_pieces():
    xyz, valid, labels, count = _pieces()
    centre = op.read_axis("500,500")
    side = op.decide(*op.radial_votes(xyz, valid, labels, count, centre, max_len=2 * STEP))
    move = (side == -1)[labels]
    (kept, k_rev), (mirrored, m_rev) = op.split(xyz, move)
    assert not k_rev and m_rev
    assert np.array_equal(kept[~move], xyz[~move]) and (kept[move] == -1).all()
    back = mirrored[:, ::-1]
    assert np.array_equal(back[move], xyz[move]) and (back[~move] == -1).all()
    assert op.away_share(xyz, valid, centre, max_len=2 * STEP)[0] == 0.5
    for grid in (kept, mirrored):
        assert op.away_share(grid, grid[..., 0] != -1, centre, max_len=2 * STEP)[0] == 1.0


def test_cli_writes_both_surfaces(tmp_path, capsys):
    tifxyz.write(tmp_path / "in", tifxyz.Surface(_mosaic(), scale=(1 / STEP, 1 / STEP)))
    assert op.main([str(tmp_path / "in"), str(tmp_path / "out"), "--umbilicus", "500,500"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["pieces"] == {"away": 1, "toward": 1, "undecided": 1}
    assert report["away_share_before"] == 0.5 and report["away_share_after"] == 1.0
    kept, mirrored = tifxyz.read(tmp_path / "out"), tifxyz.read(tmp_path / "out_mirrored")
    assert kept.extra["piece_orientation"]["columns_reversed"] is False
    assert mirrored.extra["piece_orientation"]["columns_reversed"] is True
    assert np.array_equal(mirrored.xyz[:10, :20][:, ::-1], _mosaic()[:10, 20:])
