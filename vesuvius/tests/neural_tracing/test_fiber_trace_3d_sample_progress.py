import pytest
import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from vesuvius.neural_tracing.fiber_trace_3d.train import (
    _SampleProgress, _SampleAxisWriter, _resume_sample_count,
    _save_snapshot, _load_snapshot, _report_test_timing,
)
from vesuvius.neural_tracing.fiber_trace_3d.profile_experiment import _TEST_METRIC_PATTERN


def test_tensorboard_sample_axis_includes_resumed_eval_images_and_timing(tmp_path, capsys):
    writer = _SampleAxisWriter(SummaryWriter(str(tmp_path)), _SampleProgress(10, 640, 128))
    writer.add_scalar('test/loss_total', 1.0, 10)
    writer.add_scalar('train/loss_total', 0.9, 11)
    writer.add_image('train/image', torch.zeros(3, 2, 2), 11)
    writer.add_text('config/units', 'samples', 0)
    _report_test_timing(step=11, total_seconds=2.0, writer=writer, samples_seen=768)
    writer.flush()
    writer.close()
    events = EventAccumulator(str(tmp_path)).Reload()
    assert events.Scalars('test/loss_total')[0].step == 640
    assert events.Scalars('train/loss_total')[0].step == 768
    assert events.Scalars('timing/test_total_seconds')[0].step == 768
    assert events.Images('train/image')[0].step == 768
    assert events.Tensors('config/units/text_summary')[0].step == 0
    assert 'test_timing samples=768 update=11' in capsys.readouterr().out


def test_checkpoint_sample_count_survives_multiple_batch_changes(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / 'snapshot.pt'
    progress = _SampleProgress(0, 0, 64)
    for update, expected_samples, next_batch in [(10, 640, 128), (12, 896, 32)]:
        _save_snapshot(path, model=model, optimizer=optimizer, step=update,
                       config={}, metric=None, sample_progress=progress)
        metadata = {}
        assert _load_snapshot(path, model=model, progress_metadata=metadata) == update
        assert metadata['samples_seen'] == expected_samples
        assert metadata['optimizer_global_batch_size'] == progress.global_batch_size
        progress = _SampleProgress(update, _resume_sample_count(metadata, update), next_batch)
    assert progress.at(13) == 928


def test_legacy_checkpoint_sample_count_requires_known_global_batch():
    assert _resume_sample_count({}, 0) == 0
    assert _resume_sample_count({'config': {'training': {'optimizer_global_batch_size': 64}}}, 10) == 640
    with pytest.raises(ValueError, match='resume_samples_seen'):
        _resume_sample_count({'config': {'batch_size': 8}}, 10)
    assert _resume_sample_count({}, 10, 640) == 640
    with pytest.raises(ValueError, match='nonnegative'):
        _resume_sample_count({}, 10, -1)


@pytest.mark.parametrize('prefix', ['test step=10 ', 'test samples=1280 update=10 '])
def test_benchmark_parser_retains_optimizer_update_units(prefix):
    match = _TEST_METRIC_PATTERN.match(prefix + 'loss_total=1.0 loss_direction=0.2 loss_presence=0.8 angle_mean_deg=5.0')
    assert match is not None
    assert int(match.group('step')) == 10
