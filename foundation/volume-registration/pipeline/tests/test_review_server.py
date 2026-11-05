"""The review server on a hand-made pair: no volume is read, only the record and stored pictures."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from volreg import record as rec
from volreg.review.server import Reviews, handler_for

REVIEWER = {"handle": "test", "email": "test@example.org"}


def scores(bend_median, affine_median):
    def one(median):
        passes = median <= 2
        return {"lines_up": 25, "median_fine_voxels": median, "p90_fine_voxels": 3.0, "far": 0,
                "median_um": median, "p90_um": 3.0,
                "bar": {"passes": passes, "checks": {"enough": True, "median": passes,
                                                     "p90": True, "far": True}}}
    return {"bar": {"version": "test", "far_voxels": 5, "median_max": 2, "p90_max": 4,
                    "min_lines_up": 20},
            "scores": {"bend": one(bend_median), "affine": one(affine_median)}}


@pytest.fixture
def pairs(tmp_path):
    record = rec.Record(tmp_path / "pair-a")
    record.write(rec.PAIR, {"pair": "A-to-B", "coarse_side": "fixed"})
    record.refresh_holdout([{"id": f"p{index:04d}", "stretch": index % 5,
                             "origin_zyx": [index * 128, 0, 0], "centre_zyx": [0, 0, 0],
                             "fine_zyx": [0, 0, 0],
                             "octants": [{"index": 0, "corner_zyx": [0, 0, 0],
                                          "centre_zyx": [0, 0, 0], "fine_zyx": [0, 0, 0]}]}
                            for index in range(6)], patch_coarse_voxels=128, stretches=5)
    images = tmp_path / "pair-a" / "work" / "images" / "holdout" / "p0000"
    images.mkdir(parents=True)
    (images / "coarse.png").write_bytes(b"png")
    work = tmp_path / "pair-a" / "work"
    (work / "fit").mkdir()
    (work / "fit" / "fit.json").write_text(json.dumps({"fit": "f002"}))
    (work / "scores.json").write_text(json.dumps(scores(1.5, 2.3)))
    record.refresh_final([{"id": "c001", "candidates": ["f002", "f002/affine"], "items": [
        {"patch": "p0001", "options": [["f002/affine"], ["f002"]], "choice": None},
        {"patch": "p0002", "options": [["f002"], ["f002/affine"]], "choice": None}]}])
    return Reviews(tmp_path, REVIEWER, auto_advance=False), record


def test_holdout_answers_and_notes_go_to_the_log(pairs):
    reviews, record = pairs
    state = reviews.holdout("pair-a")
    assert len(state["units"]) == 6 and state["next"] is not None and not state["done"]
    reviews.answer("pair-a", "p0000", "lines-up")
    reviews.note("pair-a", "p0000", "a crack at the top")
    state = reviews.holdout("pair-a")
    unit = next(u for u in state["units"] if u["id"] == "p0000")
    assert unit["answer"]["answer"] == "lines-up" and unit["answer"]["by"] == REVIEWER
    assert unit["note"] == "a crack at the top"
    assert state["progress"]["lines_up"] == 1
    assert [e["kind"] for e in record.events()][-2:] == ["holdout-answer", "note"]
    octants = reviews.holdout("pair-a", octants=True)
    assert {u["id"] for u in octants["units"]} == {f"p{i:04d}.0" for i in range(6)}
    with pytest.raises(ValueError):
        reviews.answer("pair-a", "p9999", "lines-up")


def test_the_next_unit_comes_from_the_stretch_with_fewest_lining_up(pairs):
    reviews, _record = pairs
    reviews.answer("pair-a", "p0000", "lines-up")          # stretch 0
    assert reviews.holdout("pair-a")["next"] == "p0001"     # stretch 1 has none yet


def test_the_blind_mapping_stays_hidden_until_every_item_is_answered(pairs):
    reviews, _record = pairs
    state = reviews.blind("pair-a")
    assert not state["done"] and state["tally"] is None and state["scores"] is None
    assert all("revealed" not in item for item in state["round"]["items"])
    reviews.pick("pair-a", "c001", 0, 1)        # item 0, option B: the bend
    with pytest.raises(ValueError):
        reviews.decide("pair-a", "bend", "")    # the round is not finished
    reviews.pick("pair-a", "c001", 1, "unsure")
    state = reviews.blind("pair-a")
    assert state["done"] and state["tally"] == {"f002": 1, "f002/affine": 0, "unsure": 1, "none": 0}
    assert state["round"]["items"][0]["revealed"] == [["f002/affine"], ["f002"]]


def test_publishing_what_misses_the_bar_needs_a_reason_and_is_an_override(pairs):
    reviews, record = pairs
    reviews.pick("pair-a", "c001", 0, 1)
    reviews.pick("pair-a", "c001", 1, 0)
    with pytest.raises(ValueError, match="misses the bar"):
        reviews.decide("pair-a", "affine", "")
    reviews.decide("pair-a", "affine", "the bend is not visible")
    decision = record.read(rec.FINAL_REVIEW)["decision"]
    assert decision["publish"] == "affine" and decision["override"]["bar_missed"] == ["median"]
    reviews.decide("pair-a", "bend", "")       # passes: no reason needed, no override
    assert record.read(rec.FINAL_REVIEW)["decision"]["override"] is None


def test_over_http_and_no_way_out_of_the_images(pairs):
    reviews, _record = pairs
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(reviews))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/img/pair-a/holdout/p0000/coarse.png") as response:
            assert response.read() == b"png"
        for path in ("/img/pair-a/../pair.json", "/img/pair-a/holdout/../../pair.json",
                     "/api/pairs/../holdout", "/pairs/nope/holdout"):
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(base + path)
            assert error.value.code == 404
        request = urllib.request.Request(f"{base}/api/pairs/pair-a/answer", method="POST",
                                         data=json.dumps({"unit": "p0003", "answer": "no"}).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            assert json.load(response) == {"ok": True}
        with urllib.request.urlopen(f"{base}/") as response:
            assert b"Registration pairs" in response.read()
    finally:
        server.shutdown()


def test_settings_change_a_stage_and_are_logged(pairs):
    reviews, record = pairs
    reviews.set_params("pair-a", "fit", {"patches": 300, "passes": 3})
    assert record.read("params.json")["fit"] == {"patches": 300, "passes": 3}
    assert record.events()[-1]["kind"] == "params"
    for stage, values in (("fit", {"seed": 1}), ("capture", {"probes": 5}), ("fit", {"patches": 0})):
        with pytest.raises(ValueError):
            reviews.set_params("pair-a", stage, values)


def test_the_fits_page_lists_every_record_with_its_scores_and_tally(pairs, tmp_path):
    reviews, record = pairs
    elsewhere = tmp_path.parent / (tmp_path.name + "-records")
    other = rec.Record(elsewhere / "pair-b")
    other.write(rec.PAIR, {"pair": "C-to-D", "coarse_side": "fixed",
                           "fixed": {"voxel_um": 8.0}, "moving": {"voxel_um": 2.0}})
    record.write(rec.PAIR, record.read(rec.PAIR) | {"fixed": {"voxel_um": 9.4}, "moving": {"voxel_um": 2.4}})
    (record.directory / "work" / "fit" / "fit.json").write_text(json.dumps(
        {"fit": "f002", "model": "affine12", "points": {"ids": ["p0009"] * 3}, "bend": None}))
    reviews.pick("pair-a", "c001", 0, 1)
    reviews.pick("pair-a", "c001", 1, "unsure")
    reviews.records = [elsewhere]
    rows = {row["name"]: row for row in reviews.fits()["fits"]}
    a = rows["pair-a"]
    assert a["source"] == "pipeline" and a["fit"]["points"] == 3
    assert a["scores"]["bend"]["passes"] and not a["scores"]["affine"]["passes"]
    assert a["blind"]["tally"]["f002"] == 1 and a["blind"]["tally"]["unsure"] == 1
    assert rows["pair-b"]["source"] == "first runs" and rows["pair-b"]["state"] == "needs a hold-out set"
