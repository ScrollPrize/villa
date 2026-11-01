from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

from vesuvius import gp_submission


def write_surface(root: Path, *, width: int = 12, stretch: float = 1.0) -> Path:
    root.mkdir()
    rows, cols = np.mgrid[0:8, 0:width].astype(np.float32)
    tifffile.imwrite(root / "x.tif", cols * stretch + 10)
    tifffile.imwrite(root / "y.tif", rows + 10)
    tifffile.imwrite(root / "z.tif", np.full((8, width), 5, dtype=np.float32))
    meta = {"uuid": "fixture", "type": "seg", "format": "tifxyz", "scale": [1.0, 1.0]}
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return root


def cut(root: Path, rows: str, *extra: str) -> int:
    cuts = root / "cuts.csv"
    cuts.write_text("column,u0,u1,kind\n" + rows, encoding="utf-8")
    surface = root / "flat.tifxyz"
    args = ["--submission", str(root / "sub"), "--surface", str(surface), "--cuts", str(cuts)]
    return gp_submission.main(["cut-columns", *args, *extra])


def check(root: Path) -> tuple[int, dict[str, dict]]:
    code = gp_submission.main(["check", "--submission", str(root / "sub")])
    report = json.loads((root / "sub" / "check_report.json").read_text(encoding="utf-8"))
    return code, {gate["name"]: gate for gate in report["gates"]}


def add_images(root: Path, stems: list[str]) -> None:
    (root / "sub" / "images").mkdir(exist_ok=True)
    for stem in stems:
        Image.new("L", (4, 4)).save(root / "sub" / "images" / f"{stem}.png")


def test_complete_submission_passes(tmp_path) -> None:
    write_surface(tmp_path / "flat.tifxyz")
    assert cut(tmp_path, "1,0,6,column\n2,6,12,column\n") == 0
    add_images(tmp_path, ["column_01", "column_02"])
    code, gates = check(tmp_path)
    assert code == 0
    assert all(gate["passed"] for gate in gates.values())


def test_check_fails_without_images_and_on_stretch(tmp_path) -> None:
    write_surface(tmp_path / "flat.tifxyz", stretch=1.3)
    assert cut(tmp_path, "1,0,12,column\n") == 0
    code, gates = check(tmp_path)
    assert code == 2
    assert gates["image_names_match_meshes"]["passed"] is False
    assert gates["mesh_isometry"]["observed"]["failing_stretch_p05_p95"] == {
        "column_01.tifxyz": [1.0, 1.3]
    }
    # The rules give no numeric tolerance, so stretch alone only warns.
    add_images(tmp_path, ["column_01"])
    code, gates = check(tmp_path)
    assert code == 0
    assert (gates["mesh_isometry"]["required"], gates["mesh_isometry"]["passed"]) == (False, False)


def test_parts_numbered_from_one(tmp_path) -> None:
    write_surface(tmp_path / "flat.tifxyz")
    assert cut(tmp_path, "1,0,4,column\n1,4,6,break\n1,6,8,patch\n2,8,12,column\n") == 0
    meshes = tmp_path / "sub" / "meshes"
    assert sorted(path.name for path in meshes.iterdir()) == [
        "column_01.tifxyz",
        "column_01_part1.tifxyz",
        "column_01_part2.tifxyz",
        "column_02.tifxyz",
    ]
    meta = json.loads((meshes / "column_01_part1.tifxyz" / "meta.json").read_text())
    assert (meta["kind"], meta["u0"], meta["u1"]) == ("break", 4, 6)
    assert tifffile.imread(meshes / "column_01_part1.tifxyz" / "x.tif").shape == (8, 2)
    assert check(tmp_path)[1]["column_naming"]["passed"] is True


def test_bad_cuts_write_nothing(tmp_path, capsys) -> None:
    write_surface(tmp_path / "flat.tifxyz")
    for rows, expected in [
        ("1,0,6,column\n2,-5,10,column\n", "cuts.csv:3: need 0 <= u0 < u1 <= 12"),
        ("1,6,6,column\n", "need 0 <= u0 < u1"),
        ("1,0,13,column\n", "need 0 <= u0 < u1 <= 12"),
        ("1,0,6,column\n1,6,12,column\n", "more than one cut for column_01.tifxyz"),
        ("1,0,x,column\n", "must be integers"),
        ("1,0,6,tear\n", "kind 'tear'"),
    ]:
        assert cut(tmp_path, rows) == 2
        assert expected in capsys.readouterr().err
        assert not (tmp_path / "sub" / "meshes").exists()


def test_empty_cut_writes_nothing(tmp_path, capsys) -> None:
    surface = write_surface(tmp_path / "flat.tifxyz")
    for name in ("x", "y", "z"):
        values = tifffile.imread(surface / f"{name}.tif")
        values[:, 6:] = -1
        tifffile.imwrite(surface / f"{name}.tif", values)
    assert cut(tmp_path, "1,0,6,column\n2,6,12,column\n") == 2
    err = capsys.readouterr().err
    assert "column_02.tifxyz (u 6-12); flat.tifxyz has valid points in u 0-6" in err
    assert not (tmp_path / "sub" / "meshes").exists()


def test_failed_recut_keeps_old_meshes(tmp_path, monkeypatch) -> None:
    write_surface(tmp_path / "flat.tifxyz")
    assert cut(tmp_path, "1,0,4\n2,4,8\n3,8,12\n") == 0

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(gp_submission, "write_tifxyz", fail)
    assert cut(tmp_path, "1,0,6\n2,6,12\n", "--overwrite") == 2
    assert sorted(path.name for path in (tmp_path / "sub").iterdir()) == ["meshes"]
    assert len(list((tmp_path / "sub" / "meshes").iterdir())) == 3


def test_recut_needs_overwrite_and_replaces_all(tmp_path, capsys) -> None:
    write_surface(tmp_path / "flat.tifxyz")
    assert cut(tmp_path, "1,0,4\n2,4,8\n3,8,12\n") == 0
    assert cut(tmp_path, "1,0,6\n2,6,12\n") == 2
    assert "--overwrite" in capsys.readouterr().err
    assert cut(tmp_path, "1,0,6\n2,6,12\n", "--overwrite") == 0
    names = sorted(path.name for path in (tmp_path / "sub" / "meshes").iterdir())
    assert names == ["column_01.tifxyz", "column_02.tifxyz"]


def test_naming_gaps_overlaps_and_orphan_images_fail(tmp_path) -> None:
    write_surface(tmp_path / "flat.tifxyz")
    assert cut(tmp_path, "1,0,6,column\n3,4,12,column\n") == 0
    (tmp_path / "sub" / "meshes" / "column_04.tifxyz").write_text("not a mesh")
    add_images(tmp_path, ["column_01", "column_03", "column_04", "column_09"])
    code, gates = check(tmp_path)
    assert code == 2
    assert gates["tifxyz_format"]["passed"] is False
    assert gates["column_naming"]["observed"]["problems"] == ["column 02 missing"]
    assert gates["no_overlapping_cuts"]["observed"]["problems"] == [
        "column_01.tifxyz and column_03.tifxyz overlap"
    ]
    problems = gates["image_names_match_meshes"]["observed"]["problems"]
    assert problems == ["column_09.png has no mesh"]


def test_fails_closed_on_missing_folder_and_bad_tolerance(tmp_path, capsys) -> None:
    assert gp_submission.main(["check", "--submission", str(tmp_path / "nope")]) == 2
    assert "run cut-columns first" in capsys.readouterr().err
    args = ["check", "--submission", str(tmp_path), "--iso-tolerance", "nan"]
    assert gp_submission.main(args) == 2
    assert "--iso-tolerance must be between 0 and 1" in capsys.readouterr().err


def test_isometry_uses_scale(tmp_path) -> None:
    # Grid points 20 voxels apart with scale 0.05 are unstretched.
    surface = write_surface(tmp_path / "flat.tifxyz", stretch=20.0)
    tifffile.imwrite(surface / "y.tif", tifffile.imread(surface / "y.tif") * 20)
    meta = json.loads((surface / "meta.json").read_text())
    (surface / "meta.json").write_text(json.dumps({**meta, "scale": [0.05, 0.05]}))
    assert cut(tmp_path, "1,0,12\n") == 0
    assert check(tmp_path)[1]["mesh_isometry"]["passed"] is True


def test_hand_made_part_passes_and_one_image_per_mesh(tmp_path) -> None:
    write_surface(tmp_path / "flat.tifxyz")
    assert cut(tmp_path, "1,0,6\n2,6,12\n") == 0
    meshes = tmp_path / "sub" / "meshes"
    (meshes / "column_02.tifxyz").rename(meshes / "column_01_part1.tifxyz")
    add_images(tmp_path, ["column_01", "column_01_part1"])
    Image.new("L", (4, 4)).save(tmp_path / "sub" / "images" / "column_01.jpg")
    code, gates = check(tmp_path)
    assert code == 2
    # The rules allow detached patches traced by hand, so any _partK name passes.
    assert gates["column_naming"]["passed"] is True
    assert gates["image_names_match_meshes"]["observed"]["problems"] == ["column_01 has 2 images"]


def test_overlaps_and_wrong_meta_types(tmp_path) -> None:
    write_surface(tmp_path / "flat.tifxyz")
    assert cut(tmp_path, "1,0,12\n2,2,4\n3,6,8\n") == 0
    add_images(tmp_path, ["column_01", "column_02", "column_03"])
    code, gates = check(tmp_path)
    # Overlap only warns, and every overlapping pair is named, not only neighbours.
    assert code == 0
    assert gates["no_overlapping_cuts"]["observed"]["problems"] == [
        "column_01.tifxyz and column_02.tifxyz overlap",
        "column_01.tifxyz and column_03.tifxyz overlap",
    ]
    path = tmp_path / "sub" / "meshes" / "column_02.tifxyz" / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**meta, "u0": "2", "source": 3, "kind": [1]}), encoding="utf-8")
    code, gates = check(tmp_path)
    assert code == 0
    assert gates["no_overlapping_cuts"]["observed"]["problems"] == [
        "column_01.tifxyz and column_03.tifxyz overlap"
    ]


def test_padding_aliases_junk_and_part_without_image(tmp_path) -> None:
    write_surface(tmp_path / "flat.tifxyz")
    assert cut(tmp_path, "1,0,4\n2,4,8\n2,8,12,patch\n") == 0
    meshes = tmp_path / "sub" / "meshes"
    shutil.copytree(meshes / "column_01.tifxyz", meshes / "column_001.tifxyz")
    shutil.copytree(meshes / "column_01.tifxyz", meshes / "column_00.tifxyz")
    (meshes / "notes.txt").write_text("x")
    (meshes / ".DS_Store").write_text("x")
    add_images(tmp_path, ["column_01", "column_001", "column_00", "column_02"])
    code, gates = check(tmp_path)
    assert code == 2
    assert gates["column_naming"]["observed"]["problems"] == [
        "bad name column_00.tifxyz",
        "bad name column_001.tifxyz",
        "notes.txt is not a tifxyz mesh",
    ]
    # A part may share its column's image, so a missing image only warns.
    assert gates["every_mesh_has_image"]["observed"]["meshes_without_image"] == ["column_02_part1"]
    assert gates["every_mesh_has_image"]["required"] is False


def test_failed_swap_restores_old_meshes(tmp_path, monkeypatch) -> None:
    write_surface(tmp_path / "flat.tifxyz")
    assert cut(tmp_path, "1,0,4\n2,4,8\n3,8,12\n") == 0
    replace = gp_submission.os.replace
    calls = []

    def flaky(src, dst):
        calls.append(src)
        if len(calls) == 2:
            raise OSError("busy")
        replace(src, dst)

    monkeypatch.setattr(gp_submission.os, "replace", flaky)
    assert cut(tmp_path, "1,0,6\n2,6,12\n", "--overwrite") == 2
    assert sorted(path.name for path in (tmp_path / "sub").iterdir()) == ["meshes"]
    assert len(list((tmp_path / "sub" / "meshes").iterdir())) == 3


def test_every_bad_row_is_reported(tmp_path, capsys) -> None:
    write_surface(tmp_path / "flat.tifxyz")
    assert cut(tmp_path, "1,0,13\n2,x,4\n0,4,6,tear\n5,0,2\n5,2,4\n") == 2
    err = capsys.readouterr().err
    assert "more than one cut for column_05.tifxyz (lines 5, 6)" in err
    assert "cuts.csv:2: need 0 <= u0 < u1" in err
    assert "cuts.csv:3: column, u0 and u1 must be integers" in err
    assert "cuts.csv:4: kind 'tear'" in err and "cuts.csv:4: column must be 1-999" in err


def test_unreadable_surface_is_bad_input(tmp_path, capsys) -> None:
    surface = write_surface(tmp_path / "flat.tifxyz")
    (surface / "z.tif").unlink()
    assert cut(tmp_path, "1,0,6\n") == 2
    assert "does not read as tifxyz" in capsys.readouterr().err
