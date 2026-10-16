from __future__ import annotations

import pytest

from vc_reader import Affine, AmbiguousMatchError, NotFoundError

VOL_78KEV = "20260411134726"  # PHercParis4 2.4um 78keV, full scroll
VOL_137KEV = "20260323153942"  # PHercParis4 2.4um 137keV ROI
VOL_1UM = "20260608103018"  # PHercParis4 1.129um


def test_volume_by_id_and_long_id(paris4):
    v = paris4.volume(VOL_78KEV)
    assert paris4.volume("20260411134726-2.400um-0.2m-78keV-masked.zarr") is v
    assert paris4.volume("20260411134726-2.400um-0.2m-78keV-masked") is v
    assert v.pixel_size_um == 2.4 and v.energy_kev == 78.0
    assert v.shape_zyx == (75784, 32693, 32693)
    assert v.license.name == "CC BY-NC 4.0"
    assert v.scan.long_id == "20260311024914-2.400um-0.2m-78keV"


def test_resolution_is_ambiguous_without_energy(paris4):
    with pytest.raises(AmbiguousMatchError) as err:
        paris4.volume(resolution=2.399)
    assert {v.id for v in err.value.candidates} == {VOL_78KEV, VOL_137KEV}
    assert "energy_kev" in str(err.value)


def test_resolution_and_energy(paris4):
    assert paris4.volume(resolution=2.399, energy_kev=78).id == VOL_78KEV
    assert paris4.volume(resolution=2.4, energy_kev=137).id == VOL_137KEV
    assert paris4.volume(resolution=1.13).id == VOL_1UM


def test_resolution_tolerance(paris4):
    assert paris4.find_volumes(resolution=2.5) == []
    assert len(paris4.find_volumes(resolution=2.5, rel_tol=0.05)) == 2


def test_no_match_lists_volumes(paris4):
    with pytest.raises(NotFoundError, match="20260411134726-2.400um"):
        paris4.volume(resolution=3.0)


def test_superseded_excluded_unless_named(pherc139):
    old, new = "20260102150214", "20260930100252"
    assert pherc139.volume(resolution=2.399).id == new
    assert {v.id for v in pherc139.find_volumes(resolution=2.399, include_superseded=True)} == {old, new}
    v = pherc139.volume(old)
    assert v.is_superseded and v.superseded_by.id == new


def test_resolution_results_ordered_by_distance(pherc139):
    vols = pherc139.find_volumes(resolution=2.4)
    assert vols[0].pixel_size_um == 2.399
    assert {v.pixel_size_um for v in vols[1:]} == {2.403}


def test_catalog_volume_global_lookup(catalog):
    assert catalog.volume(VOL_78KEV).sample.id == "PHercParis4"
    with pytest.raises(NotFoundError):
        catalog.volume("19700101000000")


def test_volume_artifacts(paris4):
    v = paris4.volume(VOL_78KEV)
    assert v.url() == (
        "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
        "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/"
    )
    assert v.url("s3") == "s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/"
    assert len(v.surface_predictions()) == 2
    assert len(v.lasagna()) == 3
    assert len(v.ink_3d_predictions()) == 1
    assert v.umbilicus()[0].parameters["revision"] == "20260524235033"


def test_educelab_volume_is_only_on_canonical_bucket(paris4):
    v = paris4.volume("20230205180739")
    assert v.license.name == "EduceLab Data License"
    assert v.url() == "https://data.aws.ash2txt.org/samples/PHercParis4/volumes/20230205180739-7.910um-54keV-masked.zarr/"
    assert v.url("s3") == "s3://vesuvius-challenge/PHercParis4/volumes/20230205180739-7.910um-54keV-masked.zarr/"


def test_transforms(paris4):
    a, b = VOL_78KEV, VOL_1UM
    tf = paris4.transform(a, b)
    group = next(g for g in paris4.properties["volume_transforms"] if g["from_volume_id"] == a)
    listed = next(t for t in group["transforms"] if t["to_volume_id"] == b)
    assert [list(r) for r in tf.matrix] == listed["matrix"]
    assert paris4.transform(a, a) == Affine.identity()
    assert paris4.volume(a).transform_to(b) == tf
    p = (10000.0, 12000.0, 30000.0)
    back = paris4.transform(b, a).apply(tf.apply(p))
    assert back == pytest.approx(p, abs=1e-6)


def test_superseded_pair_shares_frame(pherc139):
    assert pherc139.same_frame("20260102150214", "20260930100252")
    assert not pherc139.same_frame("20250728140407", "20260930100252")


# Several volumes at one resolution and energy. Different scans cover
# different regions; several exports of one scan are variants of the same data.


def test_scan_parts_need_scan(paris4):
    part1, part2 = "20230205180739", "20230206171837"  # stitched part 1 / part 2, 7.91um 54keV
    with pytest.raises(AmbiguousMatchError) as err:
        paris4.volume(resolution=7.91, energy_kev=54)
    assert {v.id for v in err.value.candidates} == {part1, part2}
    assert "narrow it with scan= or a volume id" in str(err.value)
    assert paris4.volume(resolution=7.91, scan=part2).id == part2
    assert paris4.volume(resolution=7.91, scan=paris4.scans[part1]).id == part1


def test_scan_by_long_id(paris4):
    assert paris4.volume(scan="20260311024914-2.400um-0.2m-78keV").id == VOL_78KEV
    assert paris4.find_volumes(scan="19700101000000") == []


def test_secondary_export_of_same_scan_dropped(pherc139, catalog):
    main = "20250728140407"  # 9.362um, alongside its pag0/pag50 exports
    assert len(pherc139.find_volumes(resolution=9.362)) == 3
    assert {v.is_secondary for v in pherc139.find_volumes(resolution=9.362)} == {False, True}
    assert pherc139.volume(resolution=9.362).id == main
    assert pherc139.volume("20251107132835").is_secondary  # still reachable by id
    pherc172 = catalog.sample("PHerc0172")
    assert pherc172.volume(resolution=7.91).id == "20241024131838"


def test_secondary_never_dropped_for_another_scan(pherc139):
    # 2.4um: the main 2.399um volume (78keV scan) and four secondary 2.403um
    # exports of two other scans. Nothing is preferred across scans.
    with pytest.raises(AmbiguousMatchError) as err:
        pherc139.volume(resolution=2.4)
    assert len(err.value.candidates) == 5


def test_two_secondary_exports_of_one_scan_need_id(pherc139):
    with pytest.raises(AmbiguousMatchError) as err:
        pherc139.volume(resolution=2.403, energy_kev=77)
    assert len(err.value.candidates) == 4
    assert "scan=" in str(err.value)
    with pytest.raises(AmbiguousMatchError) as err:
        pherc139.volume(resolution=2.403, scan="20250722040659")  # ROI2: Aug 2025 and Mar 2026 exports
    assert {v.id for v in err.value.candidates} == {"20250820105138", "20260319133554"}
    assert "narrow it with a volume id" in str(err.value)
    assert "secondary" in str(err.value) and "scan=20250722040659" in str(err.value)


def test_two_main_exports_of_one_scan_need_id(catalog):
    pherc9b = catalog.sample("PHerc0009B")
    with pytest.raises(AmbiguousMatchError) as err:
        pherc9b.volume(resolution=2.401)
    assert {v.id for v in err.value.candidates} == {"20250820154339", "20260319104112"}
    assert "narrow it with a volume id" in str(err.value)


def test_scan_names(paris4, pherc139):
    # DLS records scan_name, ESRF the acquisition's scanRadix
    assert paris4.scans["20230206171837"].name.startswith("PHercParis4 54keV stitched_part_2")
    assert pherc139.scans["20250722040659"].name == "SCROLLS_HEL_2.403um_77keV_0.22m_PHerc_0139_ROI2_HA"
    assert paris4.volume(scan="2.4um_137keV_PHerc-Paris4_ROI1").id == VOL_137KEV
    with pytest.raises(AmbiguousMatchError, match=r"scan=20230206171837 +\(PHercParis4 54keV stitched_part_2"):
        paris4.volume(resolution=7.91)
