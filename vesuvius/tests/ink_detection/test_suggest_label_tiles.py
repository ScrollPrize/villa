"""CPU checks for suggest_label_tiles: planted disagreement is found, controls are not drawn to it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import tifffile

from vesuvius.ink_detection.preprocessing import suggest_label_tiles as slt


BLOCK = (slice(64, 128), slice(128, 192))      # four 32 px tiles where the predictions disagree


def _predictions(tmp_path: Path, *, disagree: bool = True) -> Path:
    """Three uint8 predictions of a 256 x 320 segment that agree everywhere except BLOCK."""
    rng = np.random.default_rng(7)
    base = np.zeros((256, 320), np.float32)
    base[:, 160:] = 0.9                           # one clean ink region with a straight edge at x = 160
    preds = tmp_path / "preds"
    preds.mkdir()
    for i in range(3):
        p = base + rng.normal(0.0, 0.02, base.shape).astype(np.float32)
        if disagree:
            p[BLOCK] = rng.uniform(0.0, 1.0, (64, 64))
        p = np.clip(p, 0.01, 1.0)
        Image.fromarray(np.rint(p * 255).astype(np.uint8)).save(preds / f"seg_model{i}_forward.png")
    Image.fromarray(np.full((256, 320), 200, np.uint8)).save(preds / "merged_confidence_vote_forward.png")
    return preds


@pytest.mark.parametrize("measure", slt.MEASURE_CHOICES)
def test_planted_disagreement_is_suggested_and_the_controls_are_not(tmp_path, measure):
    preds = _predictions(tmp_path)
    out = tmp_path / "out"
    assert slt.main([str(preds), "--output-dir", str(out), "--tile", "32", "--top", "4", "--measure", measure]) == 0
    result = json.loads((out / "suggestions.json").read_text())
    assert len(result["inputs"]) == 3                       # the merged output is not a reader
    in_block = [t for t in result["suggested"] if 64 <= t["y"] < 128 and 128 <= t["x"] < 192]
    assert len(in_block) == 4
    assert result["shuffle_check"]["verdict"] == "targeted"
    mask = np.asarray(Image.open(out / "label_suggestions.png"))
    assert mask[BLOCK].min() == 255 and mask.sum() == 255 * 64 * 64
    random_mask = np.asarray(Image.open(out / "random_tiles.png"))
    assert (random_mask == 255).sum() == 4 * 32 * 32


def test_agreeing_predictions_are_flat_under_vote(tmp_path):
    preds = _predictions(tmp_path, disagree=False)
    result, _, _ = slt.suggest(slt.collect_inputs([preds]), measure="vote", tile=32, top=4)
    assert result["shuffle_check"]["verdict"] == "flat"


def test_edge_scores_the_committee_ink_edge(tmp_path):
    preds = _predictions(tmp_path, disagree=False)
    result, _, _ = slt.suggest(slt.collect_inputs([preds]), measure="edge", tile=32, top=8)
    assert {t["x"] for t in result["suggested"]} <= {128, 160}   # the tiles on each side of x = 160


def test_mask_limits_the_eligible_tiles(tmp_path):
    preds = _predictions(tmp_path)
    mask = np.zeros((256, 320), np.uint8)
    mask[:, :128] = 1
    Image.fromarray(mask).save(tmp_path / "mask.png")
    result, chosen, _ = slt.suggest(slt.collect_inputs([preds]), mask=tmp_path / "mask.png", tile=32, top=4,
                                    measure="margin")
    assert all(t["x"] < 128 for t in result["suggested"] + result["random"])
    assert chosen[:, 128:].max() == 0


def test_integer_predictions_use_their_type_maximum_and_floats_must_be_probabilities(tmp_path):
    a = np.zeros((64, 64), np.float32)
    a[:, 32:] = 0.9
    tifffile.imwrite(tmp_path / "a.tif", a)
    tifffile.imwrite(tmp_path / "b.tif", np.rint(a * 65535).astype(np.uint16))
    tifffile.imwrite(tmp_path / "small16.tif", np.full((64, 64), 199, np.uint16))
    mean, vote_share, valid = slt.committee([tmp_path / "a.tif", tmp_path / "b.tif"], 0.5)
    assert np.allclose(mean[:, 32:], 0.9, atol=1e-4) and vote_share[:, 32:].min() == 1.0
    assert valid[:, :32].sum() == 0
    mean, _, _ = slt.committee([tmp_path / "small16.tif"], 0.5)
    assert mean.max() < 0.01                                    # 199 of 65535, not 199 of 255
    tifffile.imwrite(tmp_path / "wide.tif", a * 255)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        slt.committee([tmp_path / "wide.tif"], 0.5)


def test_direction_filter_and_one_prediction(tmp_path):
    preds = _predictions(tmp_path)
    Image.open(preds / "seg_model0_forward.png").save(preds / "seg_model0_reverse_0326.png")
    assert len(slt.collect_inputs([preds])) == 4
    assert len(slt.collect_inputs([preds], direction="forward")) == 3
    assert len(slt.collect_inputs([preds], direction="reverse", measure="edge")) == 1
    with pytest.raises(ValueError, match="at least 2"):
        slt.collect_inputs([preds], direction="reverse", measure="vote")


def test_edge_and_margin_cluster_even_where_predictions_agree(tmp_path):
    preds = _predictions(tmp_path, disagree=False)
    paths = slt.collect_inputs([preds])
    for measure in ("edge", "margin"):
        result, _, _ = slt.suggest(paths, measure=measure, tile=32, top=4)
        assert result["shuffle_check"]["verdict"] == "targeted"   # the ink outline, documented as such
    result, _, _ = slt.suggest(paths, measure="vote", tile=32, top=4, nulls=0)
    assert result["shuffle_check"]["verdict"] == "unchecked"


def test_refusals(tmp_path):
    preds = _predictions(tmp_path)
    one = sorted(preds.glob("seg_*.png"))[0]
    with pytest.raises(ValueError, match="at least 2"):
        slt.collect_inputs([one], measure="vote")
    Image.fromarray(np.zeros((100, 100), np.uint8)).save(tmp_path / "small.png")
    with pytest.raises(ValueError, match="one grid"):
        slt.committee([one, tmp_path / "small.png"], 0.5)
    with pytest.raises(ValueError, match="--mask is"):
        slt.suggest([one], mask=tmp_path / "small.png", tile=32)
    out = tmp_path / "out"
    slt.main([str(preds), "--output-dir", str(out), "--tile", "32", "--top", "2"])
    assert sorted(p.name for p in out.iterdir()) == sorted(slt.OUTPUT_NAMES.values())   # no staged leftovers
    with pytest.raises(SystemExit, match="overwrite"):
        slt.main([str(preds), "--output-dir", str(out), "--tile", "32", "--top", "2"])
    assert slt.main([str(preds), "--output-dir", str(out), "--tile", "32", "--top", "2", "--overwrite"]) == 0
    with pytest.raises(SystemExit):
        slt.parse_args([str(preds), "--output-dir", str(out), "--cut", "1.5"])
