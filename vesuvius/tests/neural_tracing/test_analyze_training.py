import pytest
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary

from vesuvius.neural_tracing.fiber_trace_3d.analyze_training import summarize


def scalar(tag, step, wall, value):
    return Event(step=step, wall_time=wall,
                 summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)]))


def test_threshold_and_startup_and_timing_filter():
    events = [Event(wall_time=100), scalar("test/loss_total", 0, 110, 2),
              scalar("timing/load_ms", 1, 120, 999),
              scalar("timing/load_ms", 200, 130, 5),
              scalar("train/loss_total", 200, 130, .5),
              scalar("test/loss_total", 200, 150, .9)]
    result = summarize(events)
    assert result["first_threshold_crossing"]["elapsed_seconds"] == 50
    assert result["timing_ms"]["timing/load_ms"]["mean"] == 5
    assert result["paired_train_to_test_gap_sum_seconds"] == 20


def test_no_crossing_and_missing_timings():
    result = summarize([scalar("test/loss_total", 0, 100, 1)])
    assert result["first_threshold_crossing"] is None
    assert result["timing_ms"] == {}
    assert result["paired_train_to_test_gap_seconds"] is None
    with pytest.raises(ValueError, match="No timestamped"):
        summarize([])


def test_saved_config_preserved():
    value = Summary.Value(tag="config/json/text_summary")
    value.tensor.string_val.append(b'{"training":{"learning_rate":0.01}}')
    result = summarize([Event(wall_time=100, summary=Summary(value=[value]))])
    assert result["config"]["training"]["learning_rate"] == .01
