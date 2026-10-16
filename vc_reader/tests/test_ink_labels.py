from __future__ import annotations

import pytest

from vc_reader import Affine

VOL_78KEV = "20260411134726"
VOL_137KEV = "20260323153942"
VOL_1UM = "20260608103018"


def test_direct_labels(paris4):
    vol = paris4.volume(VOL_78KEV)
    matches = vol.ink_labels()
    assert [m.segment.id for m in matches] == ["20230702185753", "20231016151002", "20231106155351"]
    for m in matches:
        assert m.relation == "direct" and m.same_frame
        assert m.transform == Affine.identity()
        assert m.labels.target_volume_id == VOL_78KEV
        assert m.labels.label_set == "20260918"
        assert m.coverage.overlap_ratio == 1.0


def test_label_set_fields(paris4):
    labels = paris4.volume(VOL_78KEV).ink_labels()[0].labels
    assert labels.registration == "transferred"
    assert labels.arrays == ("inklabels", "supervision")
    assert labels.url() == (
        "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
        "PHercParis4/segments/20230702185753/ink-labels/2.4um-volume-20260411134726/20260918/"
    )
    assert labels.array_url("supervision").endswith("/20260918/supervision.zarr/")
    assert labels.mesh.path.endswith("/mesh/20230702185753-on-20260411134726-2.4um.tifxyz/")
    assert labels.render.path.endswith("/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/")


def test_mappable_labels_on_other_volume(paris4):
    vol = paris4.volume(VOL_1UM)
    assert vol.ink_labels(mappable=False) == []
    matches = vol.ink_labels()
    assert len(matches) == 3
    for m in matches:
        assert m.relation == "transform" and not m.same_frame
        assert m.labels.target_volume_id == VOL_78KEV
        assert m.transform == paris4.transform(VOL_78KEV, VOL_1UM)
        assert m.coverage.volume_id == VOL_1UM
    # Not every covering segment has a published mesh on the requested volume;
    # those labels are only reachable through the transform.
    assert [m.mesh_on_volume is not None for m in matches] == [True, False, True]


def test_mapped_bbox_lands_near_recorded_coverage(paris4):
    # The labels' mesh bbox pushed through the transform should land close to the
    # segment's independently recorded coverage bbox in the requested volume.
    m = paris4.volume(VOL_1UM).ink_labels()[0]
    src = m.segment.coverage[m.labels.target_volume_id]
    dst = m.coverage
    lo, hi = m.transform.apply(src.bbox_min), m.transform.apply(src.bbox_max)
    extent = [b - a for a, b in zip(dst.bbox_min, dst.bbox_max)]
    for got, want, ext in zip(lo + hi, dst.bbox_min + dst.bbox_max, extent + extent):
        assert abs(got - want) < 0.05 * ext


def test_min_coverage(paris4):
    vol = paris4.volume(VOL_137KEV)  # small ROI: segments only partly inside
    ratios = [m.coverage.overlap_ratio for m in vol.ink_labels()]
    assert ratios and all(0 < r < 0.1 for r in ratios)
    assert vol.ink_labels(min_coverage=0.5) == []


def test_dedupe_prefers_same_frame_then_native(pherc139):
    new = pherc139.volume("20260930100252")  # successor of the superseded 2.399um volume
    matches = new.ink_labels()
    assert len(matches) == len({m.segment.id for m in matches}) == 3
    for m in matches:
        assert m.relation == "transform"
        assert m.labels.target_volume_id == "20260102150214" and m.same_frame
    all_matches = new.ink_labels(dedupe=False)
    assert {m.labels.target_volume_id for m in all_matches} == {"20260102150214", "20250728140407"}
    assert len(all_matches) == 5


def test_direct_native_labels(pherc139):
    vol = pherc139.volume("20250728140407")  # 9.362um, where some labels were drawn
    by_rel = {}
    for m in vol.ink_labels():
        by_rel.setdefault(m.relation, []).append(m)
    assert all(m.labels.registration == "native" for m in by_rel["direct"])
    assert len(by_rel["direct"]) == 2 and len(by_rel["transform"]) == 1


def test_sample_level_listing(pherc139):
    assert len(pherc139.ink_labels()) == 5
    assert len(pherc139.ink_labels(target_volume="20250728140407")) == 2
    assert pherc139.ink_labels(label_set="19990101") == []


def test_latest_only_drops_superseded_sets(paris4):
    seg = paris4.segment("20230702185753")
    entry = seg.artifacts_of("ink-labels")[0]
    newer = dict(entry.raw, parameters={**entry.parameters, "label_set": "20270101"})
    newer["properties"] = {**entry.properties, "label_set": "20270101", "supersedes": "20260918"}
    from vc_reader.catalog import Segment

    raw = dict(seg.raw, data=[*seg.raw["data"], newer])
    patched = Segment(paris4, raw)
    assert [s.label_set for s in patched.ink_labels()] == ["20270101"]
    assert [s.label_set for s in patched.ink_labels(latest_only=False)] == ["20260918", "20270101"]


def test_segment_on_volume(paris4):
    vol = paris4.volume(VOL_78KEV)
    view = paris4.segment("20230702185753").on(vol)
    assert view.mesh.path.endswith("-on-20260411134726-2.4um.tifxyz/")
    assert [a.type for a in view.layers] == ["layers-zarr"]
    assert len(view.ink_labels) == 1
    (ink3d,) = view.ink_3d_projections
    assert ink3d.render_level == 2
    assert view.render_pixel_size_um(ink3d) == pytest.approx(9.6)
    assert all(a.target_volume_id == VOL_78KEV for a in view.artifacts)
