"""Named prediction caches resume without invalidating or relocating data."""
from pathlib import Path

import pytest

from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import inference_cache_root


@pytest.mark.parametrize("cap", [0, -1, True, 1.5, "4"])
def test_coordinator_rejects_invalid_batch_cap_before_starting_native(cap):
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import coordinate_native
    with pytest.raises(ValueError, match="max_batch_size"):
        coordinate_native({}, "unused", {}, {}, {}, {}, {}, [], {"max_batch_size": cap})


def test_default_path_unchanged_and_not_created(tmp_path):
    cache = tmp_path / "cache"
    assert inference_cache_root(cache) == cache / "live_inference"
    assert not cache.exists()


def test_namespace_is_fresh_and_preserves_existing_data(tmp_path):
    cache = tmp_path / "cache"
    source = cache / "open_data" / "source"
    old_prediction = cache / "live_inference" / "source" / "model" / "chunk"
    for path in (source, old_prediction):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unchanged")
    namespace = inference_cache_root(cache, "warm_inputs_v1")
    assert namespace == cache / "live_inference" / "_namespaces" / "warm_inputs_v1"
    assert namespace.is_dir() and not list(namespace.iterdir())
    assert source.read_bytes() == old_prediction.read_bytes() == b"unchanged"
    chunk = namespace / "published_chunk"
    chunk.write_bytes(b"cached prediction")
    assert inference_cache_root(cache, "warm_inputs_v1") == namespace
    assert chunk.read_bytes() == b"cached prediction"
    assert inference_cache_root(cache, "warm_inputs_v2") != namespace


@pytest.mark.parametrize("namespace", ["", ".", "..", "../escape", "a/b", "a\\b",
    "/absolute", " spaced", "a.b", "a" * 65, 0, True, [], {}, Path("name")])
def test_invalid_names_never_create_cache(tmp_path, namespace):
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="safe single component"):
        inference_cache_root(cache, namespace)
    assert not cache.exists()


def test_namespace_parent_symlink_cannot_escape(tmp_path):
    cache = tmp_path / "cache"
    root = cache / "live_inference"
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "_namespaces").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        inference_cache_root(cache, "warm_inputs")
    assert not list(outside.iterdir())


def test_existing_namespace_symlink_is_not_reused(tmp_path):
    cache = tmp_path / "cache"
    root = cache / "live_inference"
    parent = root / "_namespaces"
    parent.mkdir(parents=True)
    (parent / "warm_inputs").symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        inference_cache_root(cache, "warm_inputs")


def test_namespace_file_is_not_replaced(tmp_path):
    namespace = inference_cache_root(tmp_path, "test")
    namespace.rmdir()
    namespace.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        inference_cache_root(tmp_path, "test")
    assert namespace.read_bytes() == b"keep"
def test_failure_search_summary_accounts_for_cancelled_trials():
    import pytest
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import finalize_native_summary
    good = dict(backend='failure_search', completed=True, cases=100,
                evaluated_cases=20, cancelled_cases=80)
    assert finalize_native_summary(good) == good
    with pytest.raises(ValueError, match='incomplete'):
        finalize_native_summary(dict(good, cancelled_cases=79))
