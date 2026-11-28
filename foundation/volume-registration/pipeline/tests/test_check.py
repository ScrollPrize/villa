import json

from volreg import cli
from volreg import record as rec
from volreg import stages as pipeline_stages

REVIEWER = {"handle": "test", "email": "test@example.org"}
SCORE = {"lines_up": 25, "median_fine_voxels": 1.0, "p90_fine_voxels": 2.0, "worst_fine_voxels": 3.0}


def test_check_passes_over_a_pair_decided_to_publish_nothing(tmp_path):
    record = rec.Record(tmp_path / "pairs" / "a")
    record.write(rec.PAIR, {"pair": "A", "coarse_side": "fixed"})
    record.write(rec.HOLDOUT, {"patches": []})
    pipeline = pipeline_stages.pipeline(record.directory)
    pipeline.work("fit").mkdir(parents=True)
    pipeline.work("fit", "fit.json").write_text(json.dumps({"fit": "f001", "fine_um": 2.0}))
    pipeline.work("fit", "transform.json").write_text(json.dumps({"schema_version": "1.0.0"}))
    pipeline.work("scores.json").write_text(json.dumps({"scores": {"bend": SCORE, "affine": SCORE}}))
    record.decide({"fit": "f001", "publish": "nothing", "reason": "test", "override": None}, REVIEWER)
    pipeline_stages.run_store(pipeline, {})
    assert record.exists(rec.FIT) and not record.exists(rec.TRANSFORM)

    assert cli.main(["check", str(tmp_path / "pairs")]) == 0
