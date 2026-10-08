import json
from reviewed_trace_tuning import CHANGES, variant, nondominated
from reviewed_presence_sweep import MODELS


def test_tuning_only_changes_selected_trace_args():
    folder, _ = MODELS['ema999_12100']
    base = json.loads((folder/'workload.json').read_text())
    saved = json.loads(json.dumps(base))
    for label, changes in CHANGES.items():
        cfg = variant(base, label, changes)
        assert cfg['checkpoint'] == base['checkpoint']
        expected = json.loads(json.dumps(base))
        expected['output_root'] = cfg['output_root']
        expected['evaluation']['gpu_manager']['label'] = label
        expected['evaluation']['trace_args'] = cfg['evaluation']['trace_args']
        assert cfg == expected
        args = cfg['evaluation']['trace_args']
        def strip_changed(values):
            result = []
            i = 0
            while i < len(values):
                if values[i] in {'--'+k for k in changes}:
                    i += 2
                else:
                    result.append(values[i])
                    i += 1
            return result
        assert strip_changed(args) == strip_changed(base['evaluation']['trace_args'])
        for key, value in changes.items():
            assert float(args[args.index('--'+key)+1]) == value
        assert args[args.index('--beam-width')+1] == '8'
    assert base == saved


def test_pareto_includes_coverage_and_distinct_tradeoffs():
    def row(errors,pieces,coverage):
        return {'metrics': {'512': {'combined': dict(ordinary_restarts=errors,kept_segments=pieces,retained_percent=coverage)}}}
    rows = dict(a=row(2,200,80),b=row(1,210,80),c=row(3,210,70),empty=row(0,0,0))
    assert set(nondominated(rows)) == {'a','b','empty'}
    # Pareto membership is not automatic promotion: an empty/low-coverage
    # endpoint must remain visibly distinct from useful retained output.
