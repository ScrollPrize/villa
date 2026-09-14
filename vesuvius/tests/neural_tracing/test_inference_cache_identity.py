import json

import pytest

from lasagna.demand_predict3d import DemandGeometry, cache_identity
from vesuvius.neural_tracing.fiber_trace_3d.inference_cache import resolve_prediction_cache


def fixture_cache(tmp_path, **changes):
    geometry = DemandGeometry((8, 8, 8), 4, 0, 0, 1, 4)
    model = {"checkpoint_sha256": "checkpoint", "precision": "bf16", "tile": 4,
             "implementation": {"runner": "old", "runtime": "old"}, **changes}
    root = tmp_path / "old-hashed-key"
    root.mkdir()
    identity = cache_identity({"source": {"id": "source"}, "model": model}, geometry,
                              "float16", "flat", False)
    (root / "cache_identity.json").write_text(json.dumps(identity))
    (root / "chunk").write_bytes(b"unchanged")
    return root, geometry, model


def test_code_hashes_do_not_invalidate_existing_cache(tmp_path):
    root, geometry, old = fixture_cache(tmp_path)
    before = (root / "cache_identity.json").read_bytes()
    selected, stored = resolve_prediction_cache(tmp_path, {"id": "source"},
        {**old, "implementation": {"runner": "new"}, "embedding_channels": None}, geometry, {})
    assert selected == root and stored == old
    assert (root / "cache_identity.json").read_bytes() == before
    assert (root / "chunk").read_bytes() == b"unchanged"


@pytest.mark.parametrize("change", [{"checkpoint_sha256": "other"}, {"precision": "float32"},
                                  {"tile": 8}, {"embedding_channels": "signed-float32-v2"}])
def test_inference_parameters_remain_part_of_identity(tmp_path, change):
    root, geometry, model = fixture_cache(tmp_path)
    selected, stored = resolve_prediction_cache(tmp_path, {"id": "source"},
                                                {**model, **change}, geometry, {})
    assert selected != root and "implementation" not in stored
    assert not selected.exists()


def test_source_and_layout_must_match(tmp_path):
    root, geometry, model = fixture_cache(tmp_path)
    assert resolve_prediction_cache(tmp_path, {"id": "other"}, model, geometry, {})[0] != root
    assert resolve_prediction_cache(tmp_path, {"id": "source"}, model, geometry,
                                    {"accumulator_layout": "chunk-major"})[0] != root
