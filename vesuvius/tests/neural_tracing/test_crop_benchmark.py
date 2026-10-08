import json
import subprocess
from pathlib import Path

import pytest

from vesuvius.neural_tracing.fiber_trace_3d import crop_benchmark


def config(tmp_path, binary='vc_fiber_trace_chunk'):
    return {'output_root': str(tmp_path / 'out'), 'reference_dir': str(tmp_path),
            'commands': [{'name': 'case', 'binary': binary, 'args': ['--help']}]}


def mock_tools(monkeypatch, returncode=0):
    monkeypatch.setattr(crop_benchmark, 'sha256_file', lambda _: 'sha256')
    monkeypatch.setattr(crop_benchmark.subprocess, 'check_output',
                        lambda *a, **kw: 'revision' if kw.get('text') else b'patch')
    monkeypatch.setattr(crop_benchmark.subprocess, 'run',
                        lambda args, **kw: subprocess.CompletedProcess(args, returncode))


def test_records_and_archives_attempts(tmp_path, monkeypatch):
    mock_tools(monkeypatch)
    cfg = config(tmp_path)
    crop_benchmark.run(cfg)
    record = json.loads((tmp_path / 'out/case.run.json').read_text())
    assert record['returncode'] == 0
    assert record['command'][-1] == '--help'
    assert record['binary_sha256'] == 'sha256'
    crop_benchmark.run(cfg)
    assert len(list((tmp_path / 'out/attempts').glob('*/case.run.json'))) == 1


def test_failed_command_is_recorded(tmp_path, monkeypatch):
    mock_tools(monkeypatch, 7)
    with pytest.raises(subprocess.CalledProcessError):
        crop_benchmark.run(config(tmp_path))
    assert json.loads((tmp_path / 'out/case.run.json').read_text())['returncode'] == 7


def test_rejects_other_executables(tmp_path, monkeypatch):
    mock_tools(monkeypatch)
    with pytest.raises(ValueError, match='unsupported'):
        crop_benchmark.run(config(tmp_path, 'unrelated'))


def test_allows_native_winding_regression_runner(tmp_path, monkeypatch):
    mock_tools(monkeypatch)
    crop_benchmark.run(config(tmp_path, 'test_fiber_trace_winding_bp'))
    record = json.loads((tmp_path / 'out/case.run.json').read_text())
    assert record['command'][0].endswith('/test_fiber_trace_winding_bp')


def test_restores_historical_metric_without_changing_cases(tmp_path):
    path = tmp_path / 'replay.json'
    path.write_text(json.dumps({'metric_definition': 'complete-reference-partition-v1',
        'summary': {'directed_reference_length_base_voxels': 200,
                    'total_failures': 1, 'base_voxel_size_um': 5},
        'cases': [{'failure': 'evidence'}]}))
    crop_benchmark.restore_historical_metrics(tmp_path)
    result = json.loads(path.read_text())
    assert result['summary']['mean_segment_length_percent'] == 50
    assert result['summary']['mean_segment_length_mm'] == 0.5
    assert result['cases'] == [{'failure': 'evidence'}]
    assert result['metric_definition'] == 'corpus-failures-plus-one-v1'


def test_bp_metrics_use_raw_counts_and_all_references():
    log = ('constraint pruning benchmark\ninitial removed infringed defect problematic fulfilled pct ratio\n'
           '31891 7769 6829 0 14598 17293 45.77 84.42\n'
           'reference oracle rounds\nround pieces removed exact wrong missing rest\n'
           '0 818 0 22 3 1 5794 6769 converged 8.20 accepted\n')
    result = crop_benchmark.bp_metrics(log)
    assert result['constraint_score_percent'] == pytest.approx(-100 * 14598 / 17293)
    assert result['reference_score_percent'] == pytest.approx(100 * 22 / 26)
    with pytest.raises(ValueError, match='lacks completed'):
        crop_benchmark.bp_metrics('unfinished')


@pytest.mark.parametrize('date', [None, '2026-10-02'])
def test_publish_custom_run_record_preserves_plot_exclusions(tmp_path, monkeypatch, date):
    mock_tools(monkeypatch)
    cfg = config(tmp_path)
    cfg['commands'] = []
    if date:
        cfg['measurement_date'] = date
    cfg['publish_replay_plots'] = [dict(id='external', label='External fibers',
        result='replay.json', run_record='docs/custom.md')]
    output = Path(cfg['output_root'])
    output.mkdir()
    (output / 'replay.json').write_text(json.dumps({
        'summary': {'directed_reference_length_base_voxels': 100,
                    'base_voxel_size_um': 2.4, 'total_failures': 2},
        'inputs': {'normal_manifest': 'normal.json'}}))
    original_read = Path.read_text
    def read(path, *args, **kwargs):
        if path.name == 'fiber_benchmark_plot_data.json':
            return json.dumps({'benchmarks': {'reference_replay': {'points': []}},
                               'plot_excluded_measurements': {'old': 'reproduction'}})
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    written = {}
    monkeypatch.setattr(crop_benchmark, 'atomic_write',
                        lambda path, data, **kw: written.update({path.name: data}))
    crop_benchmark.run(cfg)
    document = written['fiber_benchmark_plot_data.json']
    assert document['plot_excluded_measurements'] == {'old': 'reproduction'}
    assert document['benchmarks']['reference_replay']['points'][0]['run_record'] == 'docs/custom.md'
    point = document['benchmarks']['reference_replay']['points'][0]
    assert point['measurement_date'] == (date or '2026-09-23')
    assert point['algorithm_date'] == (date or '2026-09-23')
