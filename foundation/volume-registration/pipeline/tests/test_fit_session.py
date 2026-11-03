"""What a registration session has to get right.

Two things are pinned. A person's verdict must reach the fit, in both directions: a patch
marked bad is never used, and a patch marked good is used even where the robust fit would
drop it. And the whole loop -- sample, calibrate, correlate, fit -- must find a planted
offset on volumes whose answer is known, whichever side is called fixed, because the
direction is the mistake this work has made most often and it never shows in a summary.

Nothing here touches the network: the volumes are small zarr stores written to a
temporary directory.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from scipy import ndimage

from volreg.cores import dense_correlation as dense
from volreg.cores.fit_session import (START, TEST_ALLOWED_MISSES, TEST_STRETCHES, TEST_TARGET, Session,
                         VERDICTS, octant_summary)
from volreg.cores.transform_io import read_transform_json

# How much larger the fine store is than the coarse one on every side, in voxels. The
# coarse store is the middle of the fine one, so the true transform is a pure offset.
PAD = 32
# One voxel over whole chunks: `candidate_origins` draws below `shape - BLOCK`, so a store
# that is an exact multiple of the chunk never offers its last row of chunks.
COARSE_SHAPE = (257, 257, 385)
PLANTED_XYZ = np.array([1.4, -0.8, 2.2])


def write_store(path, data):
    group = zarr.open_group(str(path), mode="w")
    group.create_dataset("0", data=data, chunks=(128, 128, 128), compressor=None)
    group.attrs["multiscales"] = [{"datasets": [{
        "path": "0",
        "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}],
    }]}]
    return str(path)


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    """A textured fine store, and a coarse store cut from its middle."""
    rng = np.random.default_rng(3)
    shape = tuple(size + 2 * PAD for size in COARSE_SHAPE)
    texture = ndimage.gaussian_filter(rng.normal(size=shape).astype(np.float32), 1.5)
    texture = np.clip(128 + 40 * texture / texture.std(), 1, 255).astype(np.uint8)
    root = tmp_path_factory.mktemp("stores")
    # Noise of its own on the fine side, as a second scan has. Without it every correlation
    # is exact, every model on the ladder fits to 1e-13, and which one is chosen is decided
    # by rounding -- which differs from run to run.
    noisy = texture + rng.normal(scale=12.0, size=shape)
    fine = write_store(root / "fine.zarr", np.clip(noisy, 1, 255).astype(np.uint8))
    coarse = write_store(root / "coarse.zarr", texture[PAD:-PAD, PAD:-PAD, PAD:-PAD])
    return coarse, fine


def start_matrix(offset_xyz):
    return np.hstack([np.eye(3), np.asarray(offset_xyz, float)[:, None]])


@pytest.mark.parametrize("fixed_is_coarse", [True, False])
def test_the_loop_finds_a_planted_offset(stores, tmp_path, fixed_is_coarse):
    coarse, fine = stores
    if fixed_is_coarse:
        # moving (fine) -> fixed (coarse): a fine point sits PAD further along every axis.
        fixed, moving, fixed_um, moving_um, truth = coarse, fine, 1.0, 1.0, -PAD
    else:
        # The moving side is made marginally coarser so it is the one blocks are cut from.
        fixed, moving, fixed_um, moving_um, truth = fine, coarse, 1.0, 1.0001, PAD
    session = Session.create(tmp_path / "s", fixed, moving, fixed_um, moving_um,
                             start_matrix(np.full(3, truth) + PLANTED_XYZ), "planted")
    assert session.fixed_is_coarse == fixed_is_coarse

    session.sample(12)
    origins = [tuple(patch["origin_zyx"]) for patch in session.state["patches"]]
    session.sample(12)
    again = [tuple(patch["origin_zyx"]) for patch in session.state["patches"]]
    assert len(set(again)) == len(again), "a block was drawn twice"
    assert again[:len(origins)] == origins

    session.refine(passes=2)
    # Judged where the patches are. The origin is far outside them, so the offset alone
    # carries any small error in the linear part times that distance.
    matrix = np.array(session.fit_record()["matrix_xyz"])
    moving_xyz = np.array([[x, y, z] for z, y, x in (
        np.array(patch["origin_zyx"]) + 64 + (PAD if fixed_is_coarse else 0)
        for patch in session.state["patches"])], float)
    placed = moving_xyz @ matrix[:, :3].T + matrix[:, 3]
    assert np.abs(placed - (moving_xyz + truth)).max() < 0.1

    # The written fit is a valid transform file, and the session reloads to the same state.
    read_transform_json(str(session.directory / session.fit_record()["transform"]))
    reloaded = Session.load(session.directory)
    assert reloaded.state == json.loads(json.dumps(session.state))
    np.testing.assert_array_equal(reloaded.coarse_block("p0001"),
                                  session.coarse_block("p0001"))


def measured_session(tmp_path, count=24, outlier=None):
    """A session holding measurements only: a pure offset, plus one patch that is off."""
    rng = np.random.default_rng(1)
    directory = tmp_path / "m"
    (directory / "fits").mkdir(parents=True)
    patches = []
    for index in range(count):
        centre = rng.uniform(0, 2000, size=3)
        fine = centre + PAD + rng.normal(scale=0.02, size=3)
        if index == outlier:
            fine = fine + np.array([0.0, 6.0, 0.0])
        patches.append({
            "id": f"p{index + 1:04d}", "origin_zyx": [0, 0, 0], "round": 0, "auto": "ok",
            "human": None,
            "measurement": {"under": START, "centre_zyx": centre.tolist(),
                            "fine_zyx": fine.tolist(), "shift_zyx": [0.0, 0.0, 0.0],
                            "error": 0.1},
        })
    state = {
        "schema": "volreg-session/0.1", "created": "",
        "fixed": {"url": "coarse", "voxel_um": 9.0},
        "moving": {"url": "fine", "voxel_um": 2.25},
        "coarse_side": "fixed",
        "start": {"source": "planted", "matrix_xyz": start_matrix(np.full(3, -PAD))[:3]
                  .tolist()},
        # The one round every patch was drawn in, over the whole scroll.
        "sampling": {"block": 128, "sign": 1.0, "rounds": [{"along": None}]},
        "patches": patches, "fits": [], "current": START,
    }
    return Session(directory, state)


def test_a_patch_marked_bad_is_never_used(tmp_path):
    session = measured_session(tmp_path)
    session.set_verdict("p0003", "debris")
    fit = session.fit_record(session.fit())
    assert "p0003" not in fit["candidates"]
    assert "p0003" not in fit["used"]
    session.set_verdict("p0003", None)
    assert "p0003" in session.fit_record(session.fit())["used"]


def test_a_patch_marked_good_overrides_the_robust_fit(tmp_path):
    session = measured_session(tmp_path, outlier=4)
    fit = session.fit_record(session.fit())
    assert fit["ransac_dropped"] == ["p0005"]
    assert "p0005" not in fit["used"]
    session.set_verdict("p0005", "good")
    fit = session.fit_record(session.fit())
    assert "p0005" in fit["used"]
    assert fit["parent"] == "f001"


def test_a_patch_can_lend_the_fit_its_octants_instead(tmp_path):
    # p0005's whole-patch shift is off; its octants, measured on their own, are right.
    session = measured_session(tmp_path, outlier=4)
    measurement = session.patch("p0005")["measurement"]
    centre = np.asarray(measurement["centre_zyx"])
    octants = []
    for index, corner in enumerate(itertools.product((-32, 32), repeat=3)):
        octant = {"corner_zyx": list(corner), "centre_zyx": (centre + corner).tolist(),
                  "check": {"peak_zyx": [0.0, 0.0, 0.0], "peak_match": 0.95}}
        if index == 6:
            octant["check"]["peak_match"] = 0.5
        if index == 7:
            octant = {"corner_zyx": list(corner), "rejected": "no-texture"}
        octants.append(octant)
    measurement["octants"] = octants
    assert octant_summary(measurement) == pytest.approx(
        {"octants": 6, "gradient": 0.0, "scatter": 0.0}, abs=1e-9)

    session.set_verdict("p0005", "use-octants")
    fit = session.fit_record(session.fit())
    assert "p0005" not in fit["used"]
    assert sorted(point for point in fit["used"] if point.startswith("p0005")) == \
        [f"p0005.{index}" for index in range(6)]
    assert "leverage" in fit["diagnostics"]["p0005.0"]
    # Placed at the octants, the patch is no longer an outlier.
    to_fine, _bent = session.coarse_to_fine()
    placed = to_fine(np.array([octant["centre_zyx"] for octant in octants[:6]]))
    np.testing.assert_allclose(placed, np.array([o["centre_zyx"] for o in octants[:6]]) + PAD,
                               atol=0.1)


def test_a_blind_round_asks_only_where_the_candidates_part(tmp_path):
    # One patch measured six fine voxels (1.5 coarse) off the offset every other follows:
    # the fit and that patch's own measurement place it apart, and nowhere else.
    session = measured_session(tmp_path, outlier=4)
    fit_id = session.fit()
    round_id = session.start_comparison([fit_id, "own"], count=10, seed=0)
    items = session.comparison(round_id)["items"]
    assert [item["patch"] for item in items] == ["p0005"]
    assert items[0]["spread"] == pytest.approx(1.5, abs=0.05)
    assert sorted(items[0]["options"]) == [[fit_id], ["own"]]

    own = items[0]["options"].index(["own"])
    session.choose(round_id, 0, own)
    tally = session.tally(round_id)
    assert tally["candidates"]["own"] == {"won": 1, "won_alone": 1}
    assert tally["candidates"][fit_id]["won"] == 0
    assert tally["unanswered"] == 0
    with pytest.raises(ValueError):
        session.choose(round_id, 0, 2)
    session.choose(round_id, 0, "none")
    assert session.tally(round_id)["none"] == 1
    assert session.tally(round_id)["candidates"]["own"]["won"] == 0

    # A patch judged to hold something other than scroll judges no fit.
    session.set_verdict("p0005", "debris")
    assert session.comparison(session.start_comparison([fit_id, "own"]))["items"] == []


def test_a_patch_can_be_drawn_at_its_ncc_optimum(tmp_path):
    session = measured_session(tmp_path)
    patch = session.patch("p0002")
    measurement = patch["measurement"]
    measurement["check"] = {"peak_zyx": [0.0, 2.0, -1.0], "offset": 2.2, "peak_match": 0.9}
    # Under the start transform a fine point is the coarse point plus PAD; the optimum
    # moves it by its shift on top of the measured centre.
    np.testing.assert_allclose(session.placement("peak", patch),
                               np.asarray(measurement["centre_zyx"]) + PAD + [0.0, 2.0, -1.0])
    fit_id = session.fit()
    items = session.comparison(session.start_comparison([fit_id, "peak"]))["items"]
    # Only patches with a peak check can be asked about, and only p0002's is apart.
    assert [item["patch"] for item in items] == ["p0002"]


def test_an_unknown_verdict_is_refused(tmp_path):
    session = measured_session(tmp_path)
    with pytest.raises(ValueError):
        session.set_verdict("p0001", "looks-fine")
    assert "good" in VERDICTS


def bent_session(tmp_path, spike=None):
    """Measurements carrying a smooth field the size of the real ones, plus one spike."""
    session = measured_session(tmp_path, count=80)
    rng = np.random.default_rng(5)
    for index, patch in enumerate(session.state["patches"]):
        centre = rng.uniform(0, 4000, size=3)
        field = 1.5 * np.sin(centre / 1200.0)
        fine = centre + PAD + field + rng.normal(scale=0.02, size=3)
        if index == spike:
            fine = fine + np.array([0.0, 1.0, 0.0])
        patch["measurement"].update(centre_zyx=centre.tolist(), fine_zyx=fine.tolist())
    return session


def test_every_used_patch_gets_a_leverage_and_every_measured_one_a_move(tmp_path):
    session = measured_session(tmp_path)
    first = session.fit_record(session.fit())
    assert set(first["used"]) <= set(first["diagnostics"])
    assert all("leverage" in first["diagnostics"][patch] for patch in first["used"])
    session.set_verdict("p0007", "debris")
    second = session.fit_record(session.fit())
    moved = [entry["moved"] for entry in second["diagnostics"].values()]
    assert len(moved) == 24 and max(moved) < 0.05


def test_the_patch_a_bend_rests_on_has_the_largest_influence(tmp_path):
    session = bent_session(tmp_path, spike=11)
    session.set_verdict("p0012", "good")
    fit = session.fit_record(session.fit())
    assert fit["bend"] is not None, "the planted field should be worth a bend"
    influence = {patch: entry["influence"] for patch, entry in fit["diagnostics"].items()
                 if "influence" in entry}
    assert max(influence, key=influence.get) == "p0012"
    _map, bent = session.coarse_to_fine()
    assert bent


def test_a_draw_can_be_kept_to_a_stretch_along_the_scroll(stores, tmp_path):
    coarse, fine = stores
    session = Session.create(tmp_path / "s", coarse, fine, 1.0, 1.0,
                             start_matrix(np.full(3, -PAD) + PLANTED_XYZ), "planted")
    low, high = session.region()
    start, end = 0.0, 0.5
    added = session.sample(4, along=(start, end))
    assert added
    bottom = low[0] + start * (high[0] - low[0])
    top = low[0] + end * (high[0] - low[0])
    for patch_id in added:
        z = session.patch(patch_id)["origin_zyx"][0]
        assert bottom <= z and z + dense.BLOCK - 1 <= top
    assert session.state["sampling"]["rounds"][-1]["along"] == [start, end]
    with pytest.raises(ValueError):
        session.sample(4, along=(0.5, 0.5))


def bumped_session(tmp_path, spike=None):
    """A smooth field plus a bump that only one stretch shows, carried by many patches."""
    session = measured_session(tmp_path, count=120)
    rng = np.random.default_rng(7)
    bump_at = np.array([500.0, 2000.0, 2000.0])
    for index, patch in enumerate(session.state["patches"]):
        # One in ten patches sits where the bump is, as a stretch sampled on purpose would.
        centre = (bump_at + rng.uniform(-400, 400, size=3) if index % 10 == 5
                  else rng.uniform(0, 4000, size=3))
        field = 0.5 * np.sin(centre / 1500.0)
        bump = 2.0 * np.exp(-np.sum((centre - bump_at) ** 2) / (2 * 600.0 ** 2))
        fine = centre + PAD + field + np.array([0.0, bump, 0.0]) + rng.normal(scale=0.02, size=3)
        if index == spike:
            fine = fine + np.array([0.0, 1.5, 0.0])
        patch["measurement"].update(centre_zyx=centre.tolist(), fine_zyx=fine.tolist())
        patch["human"] = {"verdict": "good", "at": ""}
    return session


def misses(session, patch_ids):
    to_fine, _bent = session.coarse_to_fine()
    patches = [session.patch(patch_id) for patch_id in patch_ids]
    centres = np.array([patch["measurement"]["centre_zyx"] for patch in patches])
    fine = np.array([patch["measurement"]["fine_zyx"] for patch in patches])
    return np.linalg.norm(to_fine(centres) - fine, axis=1)


def test_holding_out_patches_follows_a_bump_several_patches_show(tmp_path):
    session = bumped_session(tmp_path)
    near = [patch["id"] for patch in session.state["patches"]
            if np.linalg.norm(np.asarray(patch["measurement"]["centre_zyx"])
                              - [500.0, 2000.0, 2000.0]) < 600]
    assert len(near) >= 5, "the bump needs patches to show it"
    by_regions = session.fit_record(session.fit("regions"))
    regions_miss = misses(session, near)
    by_patches = session.fit_record(session.fit("patches"))
    patches_miss = misses(session, near)
    assert by_patches["hold_out"] == "patches"
    assert patches_miss.max() < regions_miss.max()
    assert patches_miss.max() < 0.3
    # A refit keeps the hold-out it was given last.
    assert session.fit_record(session.fit())["hold_out"] == "patches"


def test_holding_out_patches_still_does_not_follow_one_patch(tmp_path):
    session = bumped_session(tmp_path, spike=60)
    session.fit("patches")
    # Capped at half: the fit may follow a lone patch at most half of the way.
    assert misses(session, ["p0061"])[0] > 0.7, "one patch alone bent the fit to it"


def test_a_gate_keeps_the_patches_that_match_and_drops_the_rest(tmp_path):
    session = measured_session(tmp_path, outlier=4)
    for index, patch in enumerate(session.state["patches"]):
        # The outlier matches well -- the end of a scroll bending away -- and p0010 not.
        patch["measurement"]["match"] = 0.5 if index == 9 else 0.95
    by_ransac = session.fit_record(session.fit(robust="ransac"))
    assert "p0005" not in by_ransac["used"] and "p0010" in by_ransac["used"]
    by_gate = session.fit_record(session.fit(robust="gate"))
    assert by_gate["robust"] == "gate"
    assert "p0005" in by_gate["used"] and "p0010" not in by_gate["used"]
    # A person's verdict still decides over the gate.
    session.set_verdict("p0010", "good")
    assert "p0010" in session.fit_record(session.fit())["used"]


def test_a_gate_splits_a_patch_that_deformed_inside_into_octants(tmp_path):
    session = measured_session(tmp_path)
    measurement = session.patch("p0005")["measurement"]
    measurement["match"] = 0.95
    centre = np.asarray(measurement["centre_zyx"])
    # Eight octants about a gradient of a voxel across half a patch, agreeing exactly.
    measurement["octants"] = [
        {"corner_zyx": list(corner), "centre_zyx": (centre + corner).tolist(),
         "check": {"peak_zyx": [0.0, corner[0] / 32.0, 0.0], "peak_match": 0.95}}
        for corner in itertools.product((-32, 32), repeat=3)]
    for patch in session.state["patches"]:
        patch["measurement"].setdefault("match", 0.95)
    fit = session.fit_record(session.fit(robust="gate"))
    assert "p0005" not in fit["used"]
    assert sum(point.startswith("p0005.") for point in fit["used"]) == 8


def test_a_test_patch_is_never_fitted_and_scores_the_fit(tmp_path):
    session = measured_session(tmp_path, outlier=4)
    for index, patch in enumerate(session.state["patches"][:8]):
        patch["test"] = True
        patch["measurement"]["match"] = 0.5 if index == 7 else 0.95
    for patch_id in ("p0001", "p0002", "p0005"):
        session.set_test_answer(patch_id, "lines-up")
    session.set_test_answer("p0003", "not-scroll")
    # A verdict is about fitting, not testing: it neither scores a patch nor fits it.
    session.set_verdict("p0004", "good")
    fit = session.fit_record(session.fit())
    tested = {patch["id"] for patch in session.state["patches"][:8]}
    assert not tested & {point.split(".")[0] for point in fit["used"]}
    assert not tested & set(fit["candidates"])

    score = session.score()
    assert score["fit"] == fit["id"]
    # Lining up: p0001, p0002 and p0005, which is 6 fine voxels -- 1.5 coarse -- off.
    assert score["checked"]["patches"] == 3 and score["checked"]["missed"] == 1
    assert score["checked"]["worst"][0][0] == "p0005"
    assert abs(score["checked"]["worst"][0][1] - 1.5) < 0.05
    assert score["checked"]["median"] < 0.05
    # Unjudged and passing the gate: p0004, p0006, p0007; p0008 matches too poorly and
    # p0003 is not scroll.
    assert score["unchecked"]["patches"] == 3 and score["unchecked"]["missed"] == 0
    assert session.summary()["test_patches"] == 8
    with pytest.raises(ValueError):
        session.set_test_answer("p0009", "lines-up")


def test_a_test_review_decides_as_soon_as_it_can(tmp_path):
    session = measured_session(tmp_path, count=60, outlier=40)
    rounds = session.state["sampling"]["rounds"]
    for index in range(TEST_STRETCHES):
        rounds.append({"along": [index / TEST_STRETCHES, (index + 1) / TEST_STRETCHES]})
    for index, patch in enumerate(session.state["patches"][20:]):
        patch["test"], patch["round"] = True, 1 + index % TEST_STRETCHES
    session.fit()

    # The next patch always comes from the stretch with the fewest lining up.
    seen = []
    for _ in range(TEST_STRETCHES):
        patch_id = session.next_test()
        seen.append(session.stretch(session.patch(patch_id)))
        session.set_test_answer(patch_id, "lines-up")
    assert sorted(seen) == list(range(TEST_STRETCHES))
    partly = session.next_test()
    session.set_test_answer(partly, "partly")
    assert session.score()["partly"] == 1 / (TEST_STRETCHES + 1)
    # A patch that does not line up at its own measurement is counted, never scored.
    wrong = session.next_test()
    session.set_test_answer(wrong, "no")
    assert session.test_progress()["answers"]["no"] == 1
    assert wrong not in {patch_id for patch_id, _ in session.score()["checked"]["worst"]}
    assert session.score()["decision"] == "continue"

    # p0041 lines up at its own measurement, 6 fine voxels from where the fit can go.
    session.set_test_answer("p0041", "lines-up")
    while session.test_progress()["lines_up"] < TEST_TARGET:
        session.set_test_answer(session.next_test(), "lines-up")
    score = session.score()
    assert score["checked"]["missed"] == 1 <= TEST_ALLOWED_MISSES
    assert score["decision"] == "pass"
    # A second miss fails the fit, however many line up.
    other = session.patch(session.state["patches"][20]["id"])
    other["measurement"]["fine_zyx"] = (np.asarray(other["measurement"]["fine_zyx"]) + 8).tolist()
    assert session.score()["decision"] == "fail"


def test_a_long_job_saves_without_undoing_what_a_person_answered(tmp_path):
    measured_session(tmp_path).save()
    directory = tmp_path / "m"
    job, page = Session.load(directory), Session.load(directory)
    for patch_id in ("p0001", "p0002"):
        page.patch(patch_id)["test"] = True
    page.save()
    # The page answers and judges while the job, holding its older copy, works on.
    page = Session.load(directory)
    page.set_test_answer("p0001", "lines-up")
    page.set_verdict("p0003", "debris")
    page.set_note("p0004", "a seam")
    job.patch("p0005")["measurement"]["error"] = 0.5
    job.set_verdict("p0006", "good")
    job.save()

    saved = Session.load(directory)
    assert saved.patch("p0001")["test_answer"]["answer"] == "lines-up"
    assert saved.patch("p0003")["human"]["verdict"] == "debris"
    assert saved.patch("p0004")["note"]["text"] == "a seam"
    # The job's own work stands, a verdict it set itself included.
    assert saved.patch("p0005")["measurement"]["error"] == 0.5
    assert saved.patch("p0006")["human"]["verdict"] == "good"


def test_a_test_set_can_be_judged_octant_by_octant(tmp_path):
    session = measured_session(tmp_path)
    tested = session.state["patches"][:4]
    for patch in tested:
        patch["test"] = True
        centre = np.asarray(patch["measurement"]["centre_zyx"])
        patch["measurement"]["octants"] = [
            {"corner_zyx": list(corner), "centre_zyx": (centre + corner).tolist(),
             "check": {"peak_zyx": [0.0, 0.0, 0.0], "peak_match": 0.95 if corner[0] else 0.5}}
            for corner in itertools.product((0, 64), repeat=3)]
    # One octant measured 8 fine voxels -- 2 coarse -- from where the rest of the scan puts it.
    tested[1]["measurement"]["octants"][5]["check"]["peak_zyx"] = [0.0, 8.0, 0.0]
    session.fit()

    # Only the octants that measured well are units, each placed at its own peak.
    units = session.test_units(octants=True)
    assert [unit.id for unit in units[:4]] == [f"p0001.{index}" for index in range(4, 8)]
    assert len(units) == 16 and all(unit.size == 64 for unit in units)
    assert units[1].corner_zyx == (64, 0, 64)
    with pytest.raises(ValueError):
        session.set_test_answer("p0001.0", "lines-up")

    # An octant's answer is its own, apart from its patch's.
    session.set_test_answer("p0001.4", "partly")
    assert session.patch("p0001").get("test_answer") is None
    assert session.test_progress(octants=True)["answers"]["partly"] == 1
    assert session.test_progress()["answers"]["partly"] == 0
    session.set_test_answer("p0001.4", None)
    assert session.next_test(octants=True) == "p0001.4"
    while (unit_id := session.next_test(octants=True)) is not None:
        session.set_test_answer(unit_id, "lines-up")
    score = session.score(octants=True)
    assert score["checked"]["patches"] == 16 and score["checked"]["missed"] == 1
    assert score["checked"]["worst"][0][0] == "p0002.5"
    assert abs(score["checked"]["worst"][0][1] - 2.0) < 0.05
    assert session.score()["checked"]["patches"] == 0


def test_a_blind_round_can_pit_a_bend_against_its_affine_on_the_test_set(tmp_path):
    session = bumped_session(tmp_path)
    # Half the patches on the bump, so the bend has patches to rest on and to be tested at.
    tested = [patch["id"] for patch in session.state["patches"][5::20]]
    for patch_id in tested:
        patch = session.patch(patch_id)
        patch["test"], patch["human"] = True, None
    fit_id = session.fit("patches")
    assert session.fit_record()["bend"], "the bump needs a bend"
    near = session.patch(tested[0])
    bent, affine = (session.placement(c, near) for c in (fit_id, fit_id + "/affine"))
    to_fine = np.linalg.inv(session.fine_to_coarse(fit_id))
    np.testing.assert_allclose(affine, dense.apply(to_fine, near["measurement"]["centre_zyx"]))
    assert not np.allclose(bent, affine)

    round_id = session.start_comparison([fit_id, fit_id + "/affine"], count=50, seed=1,
                                        test_only=True)
    record = session.comparison(round_id)
    assert record["test_only"] and record["items"]
    assert {item["patch"] for item in record["items"]} <= set(tested)


def test_a_fit_can_be_made_of_every_patch_s_octants(tmp_path):
    session = measured_session(tmp_path)
    for patch in session.state["patches"]:
        measurement = patch["measurement"]
        measurement["match"] = 0.95
        centre = np.asarray(measurement["centre_zyx"])
        measurement["octants"] = [
            {"corner_zyx": list(corner), "centre_zyx": (centre + corner).tolist(),
             "check": {"peak_zyx": [0.0, 0.0, 0.0], "peak_match": 0.95 if corner[0] > 0 else 0.5}}
            for corner in itertools.product((-32, 32), repeat=3)]
    session.set_verdict("p0001", "good")
    fit = session.fit_record(session.fit(robust="gate", unit="octants"))
    assert fit["unit"] == "octants"
    # A person's "good" keeps its patch whole; every other patch lends its four octants
    # that matched well, and not the four that did not.
    assert "p0001" in fit["used"]
    assert sum(point.startswith("p0002.") for point in fit["used"]) == 4
    assert not any("." not in point for point in fit["used"] if point != "p0001")
    # The unit sticks to the next fit, as the hold-out and the robust rule do.
    assert session.fit_record(session.fit())["unit"] == "octants"


def test_a_fit_exports_with_or_without_its_bend(tmp_path):
    session = bumped_session(tmp_path)
    fit_id = session.fit("patches")
    assert session.fit_record()["bend"], "the bump needs a bend"
    session.export(tmp_path / "bent.json")
    session.export(tmp_path / "affine.json", affine_only=True)
    bent = json.loads((tmp_path / "bent.json").read_text())
    affine = json.loads((tmp_path / "affine.json").read_text())
    assert "deformation" in bent and "deformation" not in affine
    assert affine["transformation_matrix"] == bent["transformation_matrix"]
    assert affine["fixed_landmarks"] == bent["fixed_landmarks"]
    assert bent["fixed_volume"] == affine["fixed_volume"] == "coarse"
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((Path(__file__).resolve().parent.parent / "volreg" / "cores" / "transform_schema.json").read_text())
    jsonschema.validate(affine, schema)
    matrix, _fixed, _moving = read_transform_json(str(tmp_path / "affine.json"))
    np.testing.assert_allclose(np.asarray(matrix)[:3],
                               np.asarray(session.fit_record(fit_id)["matrix_xyz"]))
