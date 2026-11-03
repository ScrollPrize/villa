"""Chunk benchmark contracts use generated fixtures, not archived experiments."""
import hashlib
import json

import numpy as np
import pytest

from test_fiber_chunk_trace import ft, inputs, handles, chunk_config, trace_config
from vc.fiber_chunk_benchmark import import_legacy_cache, time_filter, time_construction


def payload():
    lines = []
    for line in range(3):
        lines.append([[x, 16, 16 + (max(0, x-96)*.5 if line == 1 else 80 if line == 2 else 0)]
                      for x in range(0, 241, 8)])
    return dict(key=[0,0,0], lo=[-64,-64,-64], hi=[576,576,576],
                maxima=3, seeds=3, lines=lines,
                end_reasons=[["chunk_boundary","chunk_boundary"]]*3)


def test_filter_immutable_and_parallel(inputs):
    _, normals = handles(inputs)
    path = inputs / "paths.json"
    original = json.dumps(payload())
    path.write_text(original)
    paths = ft.load_chunk_paths(str(path))
    cfg = ft.ChunkConfig()
    cfg.conflict_normal_base = 10
    cfg.conflict_tangent_base = 20
    # Normals in this fixture point along Y; the divergent path separates in Z.
    serial, serial_report = ft.filter_chunk_paths(paths, cfg, normals, 1)
    parallel, parallel_report = ft.filter_chunk_paths(paths, cfg, normals, 1, 4)
    a, b = json.loads(serial_report), json.loads(parallel_report)
    assert a["rejected"] == b["rejected"]
    assert len(paths.lines_base[0]) == len(payload()["lines"][0])
    assert path.read_text() == original
    for x, y in zip(serial.lines_base, parallel.lines_base):
        np.testing.assert_array_equal(x, y)


def test_legacy_import_checks_hash_and_keeps_source(inputs):
    path = inputs / "legacy.json"
    path.write_text(json.dumps(payload()))
    original = path.read_bytes()
    spec = json.loads(ft.chunk_cache_specification(ft.ChunkConfig(), trace_config(), 1, "explicit-model"))
    manifest = inputs / "import.json"
    data = dict(version=1, specification=spec,
                files=[dict(path=path.name, sha256=hashlib.sha256(original).hexdigest())])
    manifest.write_text(json.dumps(data))
    imported = import_legacy_cache(manifest, inputs / "imported")
    assert imported["chunks"] == 1
    assert json.loads((inputs / "imported/0_0_0.json").read_text())["specification"] == spec
    assert path.read_bytes() == original
    with pytest.raises(FileExistsError):
        import_legacy_cache(manifest, inputs / "imported")
    data["files"][0]["sha256"] = "0"*64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash mismatch"):
        import_legacy_cache(manifest, inputs / "rejected")
    assert not (inputs / "rejected").exists()


def test_cache_only_and_configuration_bindings(inputs):
    field, normal = handles(inputs)
    cfg = chunk_config()
    cfg.cache_only = True
    cfg.initial_alignment_power = 2
    cfg.initial_normal_radius_base, cfg.initial_tangential_radius_base = 10,40
    roundtrip = ft.ChunkConfig.from_json(cfg.to_json())
    assert json.loads(roundtrip.to_json()) == json.loads(cfg.to_json())
    cache = ft.ChunkCache(field, normal, trace_config(), cfg, str(inputs/"missing"), "test")
    with pytest.raises(RuntimeError, match="cache-only"):
        cache.get([0,0,0])
    assert not (inputs/"missing").exists()


def test_construction_timer_is_uncached_and_parallel(inputs):
    field, normals = handles(inputs)
    report = time_construction(field, normals, trace_config(), chunk_config(), [1,1,1],
                               workers=(1,4), repeats=1)
    assert [row["workers"] for row in report["timings"]] == [1,4]
    assert all(row["min_seconds"] > 0 for row in report["timings"])
