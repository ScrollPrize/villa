"""Run the built CLI writers on a real segment fixture with catalog metadata."""

import json
from pathlib import Path
import shutil
import subprocess
from uuid import UUID

import numpy as np
import pytest
import tifffile


ROOT = Path(__file__).resolve().parents[2]
CATALOG_UUID = "catalog-representation-id"


@pytest.fixture
def catalog_surface(tmp_path):
    source = tmp_path / "source"
    shutil.copytree(ROOT / "core/test/data/segments/20241113070770", source)
    # Pad the real geometry with invalid cells so trimming must write an output.
    for band in ("x", "y", "z"):
        path = source / f"{band}.tif"
        values = tifffile.imread(path)
        tifffile.imwrite(path, np.pad(values, 1, constant_values=-1))
    meta_path = source / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update(uuid=CATALOG_UUID,
                vc_open_data_representation="published-transformed",
                vc_open_data_coordinate_space="sample/volume@L0")
    meta_path.write_text(json.dumps(meta))
    (source / "catalog-origin.json").write_text("{}")
    return source


@pytest.mark.parametrize("tool,args,in_place", [
    ("vc_straighten", ["--overlap-pairs", "0"], False),
    ("vc_tifxyz_trim", [], False),
    ("vc_tifxyz_trim", [], True),
])
def test_catalog_identity_for_cli_outputs(catalog_surface, tmp_path, tool, args, in_place):
    binary = ROOT / "build/bin" / tool
    if not binary.is_file():
        pytest.skip(f"Build {tool} with ninja -C build first")
    destination = catalog_surface if in_place else tmp_path / "derived"
    subprocess.run([str(binary), str(catalog_surface), str(destination), *args],
                   check=True, capture_output=True, text=True)
    meta = json.loads((destination / "meta.json").read_text())
    assert meta["vc_open_data_coordinate_space"] == "sample/volume@L0"
    if in_place:
        assert meta["uuid"] == CATALOG_UUID
        assert meta["vc_open_data_representation"] == "published-transformed"
        assert (destination / "catalog-origin.json").exists()
    else:
        assert UUID(meta["uuid"]).version == 4
        assert "vc_open_data_representation" not in meta
        assert not (destination / "catalog-origin.json").exists()
        source_meta = json.loads((catalog_surface / "meta.json").read_text())
        assert source_meta["uuid"] == CATALOG_UUID
        assert source_meta["vc_open_data_representation"] == "published-transformed"
