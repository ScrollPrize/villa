"""CPU-only controller checks: do not initialize GPUs or signal real tasks."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[2] / 'scripts/fiber3d_experiment.py'
    spec = importlib.util.spec_from_file_location('experiment_stop_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_completed_test_lines(runner):
    assert runner.completed_test_samples('samples=601600 update=4700 loss_total=1\n') == 0
    assert runner.completed_test_samples('test_timing samples=601600 update=4700 total_seconds=5\n') == 601600


@pytest.mark.parametrize('state', ['dead', 'unknown'])
def test_never_signal_dead_or_unknown(runner, monkeypatch, state):
    import villa_gpu_manager.processes as processes
    monkeypatch.setattr(processes, 'process_state', lambda *a: state)
    kill = Mock()
    monkeypatch.setattr(runner.os, 'kill', kill)
    if state == 'unknown':
        with pytest.raises(RuntimeError, match='verify'):
            runner.stop_production_identity({'pid':123})
    else:
        runner.stop_production_identity({'pid':123})
    kill.assert_not_called()


def test_monitor_waits_for_ema_best_and_binds_launch(runner, monkeypatch, tmp_path):
    import torch
    import villa_gpu_manager.processes as processes
    launch = tmp_path / 'launch'
    launch.mkdir()
    run = tmp_path / 'run'
    identity = {'pid':123, 'start_us':456, 'boot_us':789}
    (launch / 'controller.json').write_text(json.dumps(identity))
    (launch / 'train.log').write_text(f'run_dir={run}\ntest_timing samples=601600 update=4700 total_seconds=5\n')
    control = tmp_path / 'control.json'
    control.write_text(json.dumps({'production_stop':{'launch':str(launch), 'samples':600000}}))
    monkeypatch.setattr(runner, 'CONTROL', control)
    monkeypatch.setattr(processes, 'process_state', lambda *a: 'alive')
    sleeps = []
    monkeypatch.setattr(runner.time, 'sleep', lambda seconds: sleeps.append(seconds))
    stopped = Mock()
    monkeypatch.setattr(runner, 'stop_production_identity', stopped)
    def load(path, **kwargs):
        assert kwargs['map_location'] == 'cpu' and kwargs['mmap']
        if path.name == 'current.pt':
            return dict(step=4700, samples_seen=601600, metric=2,
                        ema={'tracks':{'ema_0p99':{'best_metric':1}}})
        if path.name == 'best.pt':
            return {'metric':1}
        if path.name == 'current_ema_0p99.pt':
            return {'step':4700}
        return {'metric':1 if sleeps else 3}
    monkeypatch.setattr(torch, 'load', load)
    runner.production_watch()
    assert sleeps == [2]
    stopped.assert_called_once_with(identity)
    result = json.loads((launch / 'sample_stop_result.json').read_text())
    assert result['snapshot_samples'] == 601600


def test_campaign_failure_stops_owned_detached_training(runner, monkeypatch, tmp_path):
    initial = tmp_path / 'raw'
    initial.mkdir()
    (initial / 'selection.json').write_text(json.dumps({'run':'s1a'}))
    control = tmp_path / 'control.json'
    control.write_text(json.dumps({'comparison':{
        'output':str(tmp_path / 'campaign'), 's1a_raw_evaluation':str(initial),
        'stages':[{'name':'s1a', 'existing_run':'s1a', 'config':'s1.json', 'variants':[]},
                  {'name':'json', 'config':'json.json', 'variants':[]}]
    }}))
    monkeypatch.setattr(runner, 'CONTROL', control)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **kw: json.dumps({'global_result':{}}))
    folder = tmp_path / 'launch'
    folder.mkdir()
    identity = {'pid':123, 'start_us':456, 'boot_us':789}
    (folder / 'controller.json').write_text(json.dumps(identity))
    process = Mock()
    process.poll.side_effect = RuntimeError('injected failure')
    monkeypatch.setattr(runner, 'production_start', lambda *a: (folder, process))
    stopped = Mock()
    monkeypatch.setattr(runner, 'stop_production_identity', stopped)
    with pytest.raises(RuntimeError, match='injected'):
        runner.comparison()
    stopped.assert_called_once_with(identity)
    assert 'injected' in json.loads((tmp_path / 'campaign/comparison.json').read_text())['error']


@pytest.mark.parametrize('variant', ['raw', 'ema'])
def test_followup_selection_includes_parent_and_enforces_cutoff(runner, monkeypatch, tmp_path, variant):
    import torch
    from vesuvius.neural_tracing.fiber_trace_3d import live_evaluate
    name = 'best.pt' if variant == 'raw' else 'best_ema_0p99.pt'
    for source in ('parent', 'continued'):
        snapshots = tmp_path / source / 'snapshots'
        snapshots.mkdir(parents=True)
        (snapshots / name).touch()
    template = tmp_path / 'template.json'
    template.write_text(json.dumps({'evaluation':{}}))
    settings = dict(run=str(tmp_path / 'continued'), additional_runs=[str(tmp_path / 'parent')],
        variant=variant, max_samples=601600, template=str(template), output=str(tmp_path / 'eval'))
    def load(path, **kwargs):
        parent = path.parent.parent.name == 'parent'
        return dict(metric=1 if parent else 2, step=3900 if parent else 4700,
            samples_seen=499200 if parent else 601600,
            metric_name='test/loss_total' if variant == 'raw' else 'test_ema_0p99/loss_total')
    monkeypatch.setattr(torch, 'load', load)
    monkeypatch.setattr(live_evaluate, 'freeze_checkpoint', lambda path, dest: (path, 'sha'))
    monkeypatch.setattr(runner.subprocess, 'run', Mock())
    runner.trace_evaluate(settings)
    selection = json.loads((tmp_path / 'eval/selection.json').read_text())
    assert len(selection['candidates']) == 2
    assert selection['selected']['samples'] == 499200
    settings['max_samples'] = 600000
    with pytest.raises(ValueError, match='cutoff'):
        runner.trace_evaluate(settings)


def test_followup_resumes_raw_current_only_after_both_audits(runner, monkeypatch, tmp_path):
    import villa_gpu_manager.processes as processes
    launch = tmp_path / 'launch'
    launch.mkdir()
    (launch / 'sample_stop_result.json').write_text(json.dumps(
        dict(snapshot_samples=601600, run=str(tmp_path / 'continued'), identity={'pid':123})))
    settings = dict(launch=str(launch), max_samples=601600, parent_run='parent',
        template='template', output=str(tmp_path / 'eval'), continuation_config='continue.json')
    control = tmp_path / 'control.json'
    control.write_text(json.dumps({'comparison_followup':settings}))
    monkeypatch.setattr(runner, 'CONTROL', control)
    monkeypatch.setattr(processes, 'process_state', lambda *a: 'dead')
    calls = []
    def evaluate(options):
        calls.append(options['variant'])
        output = Path(options['output'])
        output.mkdir()
        (output / 'selection.json').write_text('{}')
        return output
    monkeypatch.setattr(runner, 'trace_evaluate', evaluate)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **kw: '{}')
    def start(config, resume):
        assert calls == ['raw', 'ema']
        assert resume == tmp_path / 'continued/snapshots/current.pt'
        return tmp_path / 'next', Mock(pid=456)
    monkeypatch.setattr(runner, 'production_start', start)
    runner.comparison_followup()


def test_exact_ema_freeze_reads_metadata_from_frozen_copy(runner, monkeypatch, tmp_path):
    import torch
    from vesuvius.neural_tracing.fiber_trace_3d import live_evaluate
    frozen = tmp_path / 'frozen.pt'
    monkeypatch.setattr(live_evaluate, 'freeze_checkpoint', lambda *a: (frozen, 'digest'))
    def load(path, **kwargs):
        assert path == frozen and kwargs['map_location'] == 'cpu'
        return dict(snapshot_kind='ema', ema_metadata={'decay':.999},
                    metric_name='test_ema_0p999/loss_total', metric=.3, step=100, samples_seen=12800)
    monkeypatch.setattr(torch, 'load', load)
    result = runner.freeze_snapshot_selection(tmp_path, 'current_ema_0p999.pt', .999, tmp_path)
    assert result['selected']['step'] == 100 and result['sha256'] == 'digest'
    with pytest.raises(ValueError, match='metadata'):
        runner.freeze_snapshot_selection(tmp_path, 'best_ema_0p99.pt', .99, tmp_path)


def test_exact_raw_freeze_accepts_training_ema_state_not_ema_export(runner, monkeypatch, tmp_path):
    import torch
    from vesuvius.neural_tracing.fiber_trace_3d import live_evaluate
    frozen = tmp_path / 'frozen.pt'
    monkeypatch.setattr(live_evaluate, 'freeze_checkpoint', lambda *a: (frozen, 'digest'))
    payload = dict(ema={'tracks':{}}, metric_name='test/loss_total', metric=.4,
                   step=200, samples_seen=25600)
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: payload)
    result = runner.freeze_snapshot_selection(tmp_path, 'current.pt', None, tmp_path)
    assert result['selected']['decay'] is None
    assert result['selected']['samples'] == 25600
    payload['snapshot_kind'] = 'ema'
    with pytest.raises(ValueError, match='metadata'):
        runner.freeze_snapshot_selection(tmp_path, 'current.pt', None, tmp_path)


def test_exact_trace_best_freeze_requires_restart_metric(runner, monkeypatch, tmp_path):
    import torch
    from vesuvius.neural_tracing.fiber_trace_3d import live_evaluate
    frozen = tmp_path / 'frozen.pt'
    monkeypatch.setattr(live_evaluate, 'freeze_checkpoint', lambda *a: (frozen, 'digest'))
    payload = dict(snapshot_kind='raw', metric_name='reviewed_trace/restarts_combined',
                   metric=17, step=300, samples_seen=38400)
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: payload)
    result = runner.freeze_snapshot_selection(
        tmp_path, 'best_reviewed_trace.pt', None, tmp_path
    )
    assert result['selected']['metric'] == 17
    payload['metric_name'] = 'test/loss_total'
    with pytest.raises(ValueError, match='metadata'):
        runner.freeze_snapshot_selection(
            tmp_path, 'best_reviewed_trace.pt', None, tmp_path
        )


@pytest.mark.parametrize('variants', [None, ['best99', 'best999'], ['current999', 'current99', 'current'], ['current999', 'current99', 'current', 'best']])
def test_series_prepares_all_exact_variants_without_training_mutations(runner, monkeypatch, tmp_path, variants):
    import villa_gpu_manager.processes as processes
    launch = tmp_path / 'launch'
    launch.mkdir()
    (launch / 'controller.json').write_text('{"pid":123}')
    control = tmp_path / 'control.json'
    settings = dict(launch=str(launch), run='run', output=str(tmp_path / 'series'))
    if variants is not None:
        settings['variants'] = variants
    control.write_text(json.dumps({'trace_series':settings}))
    monkeypatch.setattr(runner, 'CONTROL', control)
    monkeypatch.setattr(processes, 'process_state', lambda *a: 'alive')
    seen = []
    def freeze(run, filename, decay, cache):
        seen.append((filename, decay))
        return {'frozen':filename}
    monkeypatch.setattr(runner, 'freeze_snapshot_selection', freeze)
    stop, start = Mock(side_effect=AssertionError('must not stop')), Mock(side_effect=AssertionError('must not restart'))
    monkeypatch.setattr(runner, 'stop_production_identity', stop)
    monkeypatch.setattr(runner, 'production_start', start)
    runner.trace_series_prepare()
    expected = ([('current_ema_0p999.pt', .999), ('best_ema_0p999.pt', .999), ('best_ema_0p99.pt', .99)]
                if variants is None else
                ([('best_ema_0p99.pt', .99), ('best_ema_0p999.pt', .999)] if variants[0] == 'best99' else
                 [('current_ema_0p999.pt', .999), ('current_ema_0p99.pt', .99), ('current.pt', None)]))
    if variants and 'best' in variants:
        expected.append(('best.pt', None))
    assert seen == expected
    assert [e['status'] for e in json.loads((tmp_path / 'series/series.json').read_text())['entries']] == ['pending'] * len(expected)
    stop.assert_not_called()
    start.assert_not_called()


@pytest.mark.parametrize('variants', [[], ['best99', 'best99'], ['typo'], 'best99', [[]]])
def test_series_invalid_variants_fail_before_freezing(runner, monkeypatch, tmp_path, variants):
    control = tmp_path / 'control.json'
    control.write_text(json.dumps({'trace_series':{'variants':variants}}))
    monkeypatch.setattr(runner, 'CONTROL', control)
    freeze = Mock(side_effect=AssertionError('must not freeze'))
    monkeypatch.setattr(runner, 'freeze_snapshot_selection', freeze)
    with pytest.raises(ValueError, match='variants'):
        runner.trace_series_prepare()
    freeze.assert_not_called()


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('training_state', ['alive', 'dead'])
def test_series_next_order_and_failure_never_skips(runner, monkeypatch, tmp_path, fail, training_state):
    import villa_gpu_manager.processes as processes
    monkeypatch.syspath_prepend(str(Path(runner.__file__).parent))
    import fiber3d_comparison_report as reporting
    control = tmp_path / 'control.json'
    control.write_text(json.dumps({'trace_series':{'output':str(tmp_path)}}))
    monkeypatch.setattr(runner, 'CONTROL', control)
    monkeypatch.setattr(processes, 'process_state', lambda *a: training_state)
    state = dict(training_identity={'pid':123}, settings={'template':'template', 'expected_training_state': training_state},
                 entries=[dict(name=n, status='pending', selection={}) for n, _, _ in runner.EMA_SERIES])
    path = tmp_path / 'series.json'
    path.write_text(json.dumps(state))
    called = []
    def evaluate(selection, template, output):
        called.append(output.name)
        if fail:
            raise RuntimeError('injected')
    monkeypatch.setattr(runner, 'evaluate_frozen', evaluate)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: '{}')
    monkeypatch.setattr(reporting, 'series_report', lambda s: 'report')
    monkeypatch.setattr(runner, 'stop_production_identity', Mock(side_effect=AssertionError('stop')))
    monkeypatch.setattr(runner, 'production_start', Mock(side_effect=AssertionError('start')))
    if fail:
        with pytest.raises(RuntimeError, match='injected'):
            runner.trace_series_next()
        with pytest.raises(RuntimeError, match='interrupted/failed'):
            runner.trace_series_next()
        assert [e['status'] for e in json.loads(path.read_text())['entries']] == ['failed', 'pending', 'pending']
        fail = False
        runner.trace_series_next(retry_failed=True)
        recovered = json.loads(path.read_text())['entries'][0]
        assert recovered['status'] == 'done'
        assert recovered['selection'] == state['entries'][0]['selection']
        assert recovered['failed_attempts'][0]['output'] == str(tmp_path / 'current999')
        assert recovered['output'] == str(tmp_path / 'current999-retry1')
        assert (tmp_path / 'current999').is_dir()
        with pytest.raises(RuntimeError, match='requires a failed'):
            runner.trace_series_next(retry_failed=True)
    else:
        runner.trace_series_next()
        runner.trace_series_next()
        assert called == ['current999', 'best999']
        assert [e['status'] for e in json.loads(path.read_text())['entries']] == ['done', 'done', 'pending']


@pytest.mark.parametrize('expected,observed', [('alive','dead'), ('dead','alive'), ('dead','unknown'), ('invalid','alive')])
def test_series_training_state_fails_closed(runner, monkeypatch, expected, observed):
    import villa_gpu_manager.processes as processes
    monkeypatch.setattr(processes, 'process_state', lambda *a: observed)
    with pytest.raises((ValueError, RuntimeError)):
        runner.check_series_training_state({'expected_training_state':expected}, {})


def test_evaluation_cache_names_include_series_directory(runner, monkeypatch, tmp_path):
    template = tmp_path / 'template.json'
    template.write_text(json.dumps({'evaluation': {}}))
    monkeypatch.setattr(runner.subprocess, 'run', Mock())
    selection = dict(frozen='snapshot.pt', sha256='digest', selected={})
    namespaces = []
    for series in ('first', 'second'):
        output = tmp_path / series / 'best99'
        output.mkdir(parents=True)
        runner.evaluate_frozen(selection, str(template), output)
        namespaces.append(json.loads((output / 'workload.json').read_text())['evaluation']['inference_cache_namespace'])
    assert namespaces[0] != namespaces[1]
    assert all(len(n) <= 64 for n in namespaces)


def test_equal_scroll_report_and_bolding(runner, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(runner.__file__).parent))
    from fiber3d_comparison_report import SCROLLS, comparison_markdown, equal_scroll_score
    rows = []
    for name, failures in [('Classic', 99), ('B', 49)]:
        scrolls = {s:dict(directions=1, failures=failures, directed_reference_mm=2000,
                         mean_segment_length_mm=2000/(failures+1), median_segment_length_mm=1,
                         p05_segment_length_mm=.1, p95_segment_length_mm=10, zero_length_segments=0) for s in SCROLLS}
        rows.append(dict(name=name, output=name, metrics={'scrolls':scrolls}))
    assert equal_scroll_score(rows[0]['metrics']) == 20
    doc = comparison_markdown(rows)
    assert '| B | **40.000** |' in doc
    assert '| B | **49** | **40.000** | **1.000** | **0.1000** | **10.000** | **0** |' in doc
    rows[0]['selection'] = dict(selected=dict(step=1, samples=128, metric=.4), sha256='hash')
    assert '| Classic | 1 | 128 | unknown | unspecified | unknown | 0.400000 |' in comparison_markdown(rows)
    rows[0]['selection']['selected']['decay'] = None
    assert '| Classic | 1 | 128 | raw | unspecified | unknown | 0.400000 |' in comparison_markdown(rows)
    rows[1]['metrics']['scrolls'][SCROLLS[0]]['directions'] = 2
    with pytest.raises(ValueError, match='coverage'):
        comparison_markdown(rows)


def test_snapshot_labels(runner, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(runner.__file__).parent))
    from fiber3d_comparison_report import snapshot_label
    row = {'name':'New current99', 'selection':{'run':'/runs/finetune_5scroll_s1a',
           'selected':{'path':'/snapshots/current_ema_0p99.pt', 'samples':6336000, 'step':49500, 'metric':.455648}}}
    assert snapshot_label(row) == 'Combined finetune EMA99 (latest/current; test=0.455648) — 6,336,000 samples / u49,500'
    row['selection']['run'] = '/runs/s1a_nml_test'
    assert snapshot_label(row).startswith('NML EMA99 (latest/current; ')
    row['selection']['run'] = '/runs/compare_5scroll_test'
    row['selection']['selected']['path'] = '/snapshots/best.pt'
    assert snapshot_label(row).startswith('JSON raw (best test loss; ')
    assert snapshot_label({'name':'Classic'}) == 'Classic'


def test_classic_normalization(runner, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(runner.__file__).parent))
    from fiber3d_comparison_report import SCROLLS, classic_normalized_score, comparison_markdown
    def metrics(values):
        return {'scrolls': {s: {'mean_segment_length_mm': v} for s, v in zip(SCROLLS, values)}}
    base = metrics([18, 1.8, 3, 4, 5])
    scale = 31.8 / 5
    assert classic_normalized_score(base, base) == pytest.approx(scale)
    assert classic_normalized_score(metrics([19.8, 1.98, 3.3, 4.4, 5.5]), base) == pytest.approx(scale * 1.1)
    a = classic_normalized_score(metrics([22, 1.8, 3, 4, 5]), base)
    b = classic_normalized_score(metrics([18, 2.2, 3, 4, 5]), base)
    assert a == pytest.approx(b)
    scaled_base = metrics([180, 1.8, 3, 4, 5])
    scaled_model = metrics([220, 1.8, 3, 4, 5])
    assert classic_normalized_score(scaled_model, scaled_base) / classic_normalized_score(
        scaled_base, scaled_base) == pytest.approx(a / scale)
    assert classic_normalized_score(metrics([0, 1, 1, 1, 1]), base) == 0
    with pytest.raises(ValueError, match='positive'):
        classic_normalized_score(base, metrics([0, 1, 1, 1, 1]))
    for bad in [-1, float('nan'), float('inf')]:
        with pytest.raises(ValueError, match='invalid'):
            classic_normalized_score(metrics([bad, 1, 1, 1, 1]), base)
    for rows in [[], [{'name':'A', 'metrics':base}], [{'name':'Classic','metrics':base}] * 2]:
        with pytest.raises(ValueError):
            comparison_markdown(rows)
