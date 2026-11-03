"""A pair through every stage, on synthetic stores with a planted offset."""

import json

import numpy as np
import pytest
import zarr
from scipy import ndimage

from volreg import record as rec
from volreg import stages as pipeline_stages
from volreg.engine import DONE, FAILED, READY, STALE, WAITING

PAD = 32
# Tall enough along z for each of the hold-out set's five stretches to hold whole blocks.
COARSE_SHAPE = (769, 257, 385)
PLANTED_XYZ = np.array([1.4, -0.8, 2.2])
REVIEWER = {"handle": "test", "email": "test@example.org"}


def write_store(path, data):
    group = zarr.open_group(str(path), mode="w")
    group.create_dataset("0", data=data, chunks=(128, 128, 128), compressor=None)
    group.attrs["multiscales"] = [{"datasets": [{
        "path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}]}]}]
    return str(path)


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    rng = np.random.default_rng(3)
    shape = tuple(size + 2 * PAD for size in COARSE_SHAPE)
    texture = ndimage.gaussian_filter(rng.normal(size=shape).astype(np.float32), 1.5)
    texture = np.clip(128 + 40 * texture / texture.std(), 1, 255).astype(np.uint8)
    root = tmp_path_factory.mktemp("stores")
    noisy = np.clip(texture + rng.normal(scale=12.0, size=shape), 1, 255).astype(np.uint8)
    return (write_store(root / "coarse.zarr", texture[PAD:-PAD, PAD:-PAD, PAD:-PAD]),
            write_store(root / "fine.zarr", noisy))


def new_pair(tmp_path, stores, offset_xyz):
    coarse, fine = stores
    record = rec.Record(tmp_path / "pair")
    # moving (fine) -> fixed (coarse): a fine point sits PAD further along every axis.
    matrix = np.hstack([np.eye(3), np.asarray(offset_xyz, float)[:, None]])
    record.write(rec.PAIR, {"pair": "synthetic", "fixed": {"url": coarse, "voxel_um": 1.0},
                            "moving": {"url": fine, "voxel_um": 1.0}, "coarse_side": "fixed",
                            "start": {"source": "planted", "matrix_xyz": matrix.tolist()}})
    record.write(pipeline_stages.PARAMS, {"capture": {"probes": 6, "overview": False},
                                          "holdout": {"patches": 10},
                                          "fit": {"patches": 16, "passes": 2}})
    return record


def test_a_pair_through_every_stage(tmp_path, stores):
    record = new_pair(tmp_path, stores, np.full(3, -PAD) + PLANTED_XYZ)
    pipeline = pipeline_stages.pipeline(record.directory)
    for name in ("capture", "holdout"):
        assert pipeline.run_stage(name)["status"] == "ok", pipeline.stamp(name).get("traceback")
    states = pipeline.states()
    assert states["judge-holdout"] == WAITING and states["fit"] == READY

    held = record.read(rec.HOLDOUT)["patches"]
    for patch in held:      # every unit rendered once, its digests on the record
        assert set(patch["image_sha256"]) == {"coarse", "own"}
        assert (pipeline.work("images", "holdout", patch["id"], "own.png")).exists()
    # Never more than asked for. Only the two outer stretches of this small store hold whole
    # blocks, so it gives 4 of the 10.
    assert 4 <= len(held) <= 10 and all("centre_zyx" in patch for patch in held)
    frozen = {patch["id"]: patch["fine_zyx"] for patch in held}

    assert pipeline.run_stage("fit")["status"] == "ok", pipeline.stamp("fit").get("traceback")
    # The fit stage never re-measures the hold-out set.
    assert {patch["id"]: patch["fine_zyx"] for patch in record.read(rec.HOLDOUT)["patches"]} == frozen
    fit = json.loads(pipeline.work("fit", "fit.json").read_text())
    assert not set(fit["points"]["ids"]) & set(frozen)

    for patch in held:
        record.answer(patch["id"], "lines-up", REVIEWER)
    assert pipeline.states()["judge-holdout"] == DONE
    assert pipeline.run_stage("recheck")["status"] == "ok", pipeline.stamp("recheck").get("traceback")
    recheck = json.loads(pipeline.work("recheck.json").read_text())
    assert recheck["max"] < 0.5 and not recheck["flagged"]     # a planted offset: nothing drifts
    assert pipeline.run_stage("score")["status"] == "ok"
    scores = json.loads(pipeline.work("scores.json").read_text())["scores"]
    assert scores["bend"]["lines_up"] == len(held) and scores["bend"]["median_fine_voxels"] < 1.0
    assert json.loads(pipeline.work("scores.json").read_text())["drift"]["judge_again"] == []

    # A new answer makes the score stale; scoring again picks it up.
    record.answer(held[0]["id"], "partly", REVIEWER)
    assert pipeline.states()["score"] == STALE
    pipeline.run_stage("score")
    assert json.loads(pipeline.work("scores.json").read_text())["scores"]["bend"]["lines_up"] == len(held) - 1

    assert pipeline.states()["blind"] == READY
    assert pipeline.run_stage("blind")["status"] == "ok", pipeline.stamp("blind").get("traceback")
    assert pipeline.states()["final-review"] == WAITING
    round_ = record.read(rec.FINAL_REVIEW)["rounds"][0]
    assert round_["candidates"] == [fit["fit"], fit["fit"] + "/affine"]
    for index, item in enumerate(round_["items"]):
        assert pipeline.work("images", "blind", round_["id"], str(index), "1.png").exists()
        record.blind_pick(round_["id"], index, "unsure", REVIEWER)
    assert all(item["choice"] == "unsure" for item in record.read(rec.FINAL_REVIEW)["rounds"][0]["items"])
    # Running the stage again keeps the round of this fit, and the picks.
    pipeline.redo("blind"); pipeline.run_stage("blind")
    assert len(record.read(rec.FINAL_REVIEW)["rounds"]) == 1

    record.decide({"fit": fit["fit"], "publish": "affine", "reason": "test", "override": None},
                  REVIEWER)
    assert pipeline.states()["final-review"] == DONE
    assert pipeline.run_stage("store")["status"] == "ok"
    stored = record.read(rec.TRANSFORM)
    assert "deformation" not in stored and stored["fixed_volume"] == "coarse.zarr"
    assert set(record.read(rec.FIT)["code"]) == {"commit", "dirty"}
    assert all(state == DONE for state in pipeline.states().values())


def test_asking_for_more_makes_the_holdout_stale_and_a_rerun_keeps_the_answers(tmp_path, stores):
    record = new_pair(tmp_path, stores, np.full(3, -PAD) + PLANTED_XYZ)
    pipeline = pipeline_stages.pipeline(record.directory)
    pipeline.run_stage("capture"); pipeline.run_stage("holdout")
    before = record.read(rec.HOLDOUT)["patches"]
    record.answer(before[0]["id"], "lines-up", REVIEWER)
    record.write(pipeline_stages.PARAMS, record.read(pipeline_stages.PARAMS)
                 | {"holdout": {"patches": 14}})
    assert pipeline_stages.pipeline(record.directory).states()["holdout"] == STALE
    # This small store has no room for more, so the rerun is made with what is there: a
    # rerun of the stage must keep every answer either way.
    record.write(pipeline_stages.PARAMS, record.read(pipeline_stages.PARAMS)
                 | {"holdout": {"patches": len(before)}})
    pipeline = pipeline_stages.pipeline(record.directory)
    pipeline.redo("holdout")
    assert pipeline.run_stage("holdout")["status"] == "ok"
    held = {patch["id"]: patch for patch in record.read(rec.HOLDOUT)["patches"]}
    assert held[before[0]["id"]]["answer"]["answer"] == "lines-up"
    assert {patch["id"] for patch in before} <= set(held)


def test_a_start_that_does_not_capture_fails_before_anyone_judges(tmp_path, stores):
    record = new_pair(tmp_path, stores, np.full(3, -PAD) + [40.0, -35.0, 30.0])
    pipeline = pipeline_stages.pipeline(record.directory)
    stamp = pipeline.run_stage("capture")
    assert stamp["status"] == "failed" and "does not capture" in stamp["error"]
    assert pipeline.states()["capture"] == FAILED
