from types import SimpleNamespace

from vesuvius.neural_tracing.fiber_trace_3d import train
from vesuvius.neural_tracing.fiber_trace_3d.sample_pipeline import EvaluationSampleIndex, _pipeline_config
from tests.neural_tracing.test_fiber_trace_3d import _loader_with_mapping
import torch


def test_explicit_evaluation_indices_preserve_remainder_and_offset():
    mapping = EvaluationSampleIndex(3, (1, 4, 9), 17)
    assert [mapping(i) for i in range(9)] == [20, 21, 22, 29, 30, 31, 44, 45, 46]


def test_pipeline_uses_resolved_test_config_and_bounded_workers(monkeypatch):
    config = _loader_with_mapping({"augment_enabled": False, "batch_size": 2}).config
    assert _pipeline_config(config) is config
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr(train, "FiberTrace3DTrainingBatchPipeline", lambda **kw: SimpleNamespace(**kw))
    pipeline = train._make_test_sample_pipeline(config,
        raw_config={"training": {"sample_pipeline_workers": 128}},
        global_batch_indices=(2, 7), start_sample_index=13, sample_mode="random")
    assert pipeline.config_path is config
    assert not pipeline.config_path.augment_enabled
    assert pipeline.workers == 16
    assert pipeline.slot_count == 4
    assert pipeline.index_map(2) == 27
    assert train._make_test_sample_pipeline(config, raw_config={}, global_batch_indices=()) is None


def test_real_sample_pool_matches_serial_test_batches(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    loader = _loader_with_mapping({"augment_enabled": False, "batch_size": 2,
                                   "patch_shape_zyx": [8, 8, 8]})
    pipeline = train._make_test_sample_pipeline(loader.config,
        raw_config={"training": {"sample_pipeline_workers": 2, "sample_pipeline_slots": 4}},
        global_batch_indices=(0, 1), sample_mode="random")
    try:
        for i in range(2):
            expected = loader.load_batch(i * 2, sample_mode="random", device="cpu")
            actual = next(pipeline)
            assert torch.equal(actual.stream_indices, expected.stream_indices)
            assert actual.fiber_paths == expected.fiber_paths
            torch.testing.assert_close(actual.volume, expected.volume)
            pipeline.release_current()
    finally:
        pipeline.close()
