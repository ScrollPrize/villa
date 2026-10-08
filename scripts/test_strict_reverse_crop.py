from strict_reverse_crop import configured, VARIANTS


def test_configured_preserves_forward_and_bp_settings():
    template = dict(output_root='old', input_files=['normal'],
        commands=[dict(name='old', binary='vc_fiber_trace_chunk', args=[
            'reference-replay-benchmark','-','--output','old.json',
            '--replay-reverse-interval-base','512','--replay-reverse-to-start',
            '--replay-reverse-all-endpoints','--step-voxels','4'])])
    for name,n,t in VARIANTS:
        config = configured(template,name,'.json',
            {'--replay-reverse-normal-base':n,'--replay-reverse-tangential-base':t})
        args = config['commands'][0]['args']
        assert args[args.index('--replay-reverse-normal-base')+1] == str(n)
        assert args[args.index('--replay-reverse-tangential-base')+1] == str(t)
        assert args[args.index('--step-voxels')+1] == '4'
        assert args[args.index('--replay-reverse-interval-base')+1] == '512'
        assert '--replay-reverse-to-start' in args
        assert '--replay-reverse-all-endpoints' in args
        assert template['commands'][0]['args'][3] == 'old.json'


def test_matrix_contains_half_quarter_and_equal_radii():
    pairs = {(n,t) for _,n,t in VARIANTS}
    assert pairs == {(10,20),(5,10),(20,20),(10,10),(5,5)}


def test_report_preserves_history_and_refreshes_without_duplicate_rows(tmp_path, monkeypatch):
    import json
    import strict_reverse_crop as sweep
    base = tmp_path/'baseline'
    base.mkdir()
    out = tmp_path/'output'
    out.mkdir()
    docs = tmp_path/'volume-cartographer/docs/fiber_benchmark_runs'
    docs.mkdir(parents=True)
    historical = docs/'2026-10-02-prefix512.md'
    historical.write_text('| older result | 1 |\n| raw40500 beam8, full-prefix512 filter | 100 |\n')
    for prefix in ('beam-prefix512','replay-prefix512','bp-prefix512','bp-prefix512-independent'):
        (base/f'{prefix}.run.json').write_text(json.dumps({'wall_seconds':1}))
    monkeypatch.setattr(sweep,'BASE',base)
    monkeypatch.setattr(sweep,'OUT',out)
    monkeypatch.setattr(sweep,'REPO',tmp_path)
    metrics = dict(replay_percent=100, constraint_score_percent=-45,
        reference_score_percent=76, independent_percent=50,
        observed_ladder_shift=dict(ladder_shift_percent=83,ladder_shifts=4,references=24),
        failures=0,filtered_stops=23,retained_percent=73,dense_retained_percent=31,
        trace_count=2227,independent_missing=2,tied_rows=0,unbounded_rows=0)
    monkeypatch.setattr(sweep,'detailed_result',lambda root,name,timings:dict(metrics,timings=timings))
    rows = {'ellipse5x10':dict(normal_radius_base=5,tangential_radius_base=10,timings={'trace':1})}
    sweep.report(rows)
    first = historical.read_text()
    sweep.report(rows)
    assert historical.read_text() == first
    assert first.count('| raw40500 strict ellipse5x10,') == 1
    assert '| older result | 1 |' in first
    result = json.loads((out/'results.json').read_text())
    assert list(result) == ['baseline20x40','ellipse5x10']
