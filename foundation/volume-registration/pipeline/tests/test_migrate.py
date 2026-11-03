"""Migration of a first-run session, on a small synthetic one."""

import json

import numpy as np

from volreg import migrate, record as rec
from volreg.cli import check

REVIEWER = {"handle": "j", "email": "j@x"}


def session(tmp_path):
    directory = tmp_path / "s"
    (directory / "fits").mkdir(parents=True)
    rng = np.random.default_rng(0)
    patches, fixed, moving = [], [], []
    for index in range(30):
        centre = rng.uniform(100, 900, 3)
        fine = centre * 4 + rng.normal(0, 0.5, 3)          # coarse 4x the fine voxel
        patches.append({"id": f"p{index + 1:04d}", "origin_zyx": [0, 0, 0], "round": 0,
                        "auto": "ok", "human": None, "test": index < 24,
                        "measurement": {"under": "start", "centre_zyx": centre.tolist(),
                                        "fine_zyx": fine.tolist(), "match": 0.9},
                        "test_answer": {"answer": "lines-up", "at": "2026-10-01T00:00:00+00:00"}
                        if index < 22 else None})
        fixed.append(centre[::-1].tolist()); moving.append(fine[::-1].tolist())
    matrix = [[0.25, 0, 0, 0], [0, 0.25, 0, 0], [0, 0, 0.25, 0]]       # fine -> coarse, XYZ
    with open(directory / "fits" / "f001.json", "w") as handle:
        json.dump({"transformation_matrix": matrix, "fixed_landmarks": fixed,
                   "moving_landmarks": moving}, handle)
    state = {"schema": migrate.SESSION_SCHEMA, "created": "2026-10-01T00:00:00+00:00",
             "fixed": {"url": "coarse", "voxel_um": 8.0}, "moving": {"url": "fine", "voxel_um": 2.0},
             "coarse_side": "fixed", "start": {"source": "dense/PAIR.json", "matrix_xyz": matrix},
             "sampling": {"block": 128, "sign": 1.0, "rounds": [{"along": None}]},
             "patches": patches,
             "fits": [{"id": "f001", "matrix_xyz": matrix, "transform": "fits/f001.json",
                       "used": [patch["id"] for patch in patches], "model": "affine12",
                       "bend": None, "hold_out": "patches", "robust": "gate", "unit": "patches"}],
             "current": "f001",
             "comparisons": [{"id": "c001", "at": "2026-10-02T00:00:00+00:00",
                              "candidates": ["f001", "f001/affine"], "test_only": True,
                              "items": [{"patch": "p0001", "spread": 0.4,
                                         "options": [["f001"], ["f001/affine"]], "choice": 1}]}]}
    with open(directory / "session.json", "w") as handle:
        json.dump(state, handle)
    return directory


def test_migrate_keeps_every_answer_and_scores(tmp_path):
    record = migrate.migrate(session(tmp_path), tmp_path / "out", REVIEWER)
    held = record.read(rec.HOLDOUT)
    assert len(held["patches"]) == 24
    assert sum(bool(patch["answer"]) for patch in held["patches"]) == 22
    assert all(patch["answer"]["by"] == REVIEWER for patch in held["patches"] if patch["answer"])
    kinds = [event["kind"] for event in record.events()]
    assert kinds.count("holdout-answer") == 22 and kinds.count("blind-pick") == 1
    bend = record.read(rec.FIT)["scores"]["bend"]
    assert bend["lines_up"] == 22 and bend["median_fine_voxels"] < 1.5
    assert record.read(rec.FINAL_REVIEW)["decision"] is None


def test_check_reproduces_a_decided_record(tmp_path):
    record = migrate.migrate(session(tmp_path), tmp_path / "out", REVIEWER)
    final = record.read(rec.FINAL_REVIEW)
    record.write(rec.FINAL_REVIEW, final | {"decision": {"fit": "f001", "publish": "bend",
                                                         "reason": "test", "override": None}})
    class Args: records = [str(tmp_path / "out")]
    assert check(Args) == 0
