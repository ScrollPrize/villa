import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reviewed_reverse_prefix import workload, VARIANTS


def test_reverse_matrix_preserves_inference_and_parallelism():
    base = {'checkpoint': 'frozen.pt', 'checkpoint_sha256': 'fixed', 'devices': '0,1,2,3,4,5',
            'evaluation': {'trace_args': ['--beam-width', '8'], 'workers': 192,
                           'gpu_manager': {'label': 'original', 'cpu_cores': 192},
                           'inference_cache_namespace': 'existing', 'inference': {'max_batch_size': 64}}}
    original = copy.deepcopy(base)
    for name, interval, prefix in VARIANTS:
        result = workload(base, name, interval, prefix)
        assert base == original
        assert result['checkpoint'] == base['checkpoint']
        assert result['evaluation']['inference'] == base['evaluation']['inference']
        assert result['evaluation']['inference_cache_namespace'] == 'existing'
        assert result['evaluation']['workers'] == 192
        args = result['evaluation']['trace_args']
        assert args[args.index('--replay-reverse-interval-base')+1] == str(interval)
        assert '--replay-reverse-all-endpoints' in args
        assert ('--replay-reverse-to-start' in args) == prefix


def test_history_variant_and_selection():
    from reviewed_reverse_history import config_for, choose_history
    base = {'evaluation': {'trace_args': [], 'gpu_manager': {}, 'workers': 192,
                           'inference_cache_namespace': 'unchanged'}}
    name, config = config_for(base, 256, 4)
    assert name == 'prefix256_history4'
    assert config['evaluation']['trace_args'][-2:] == ['--replay-reverse-history-steps', '4']
    assert config['evaluation']['workers'] == 192
    assert config['evaluation']['inference_cache_namespace'] == 'unchanged'
    results = {f'prefix256_history{h}': {'combined': dict(ordinary_restarts=f,
               filtered_stops=s, retained_percent=p)} for h,f,s,p in [(1,6,1,99), (4,5,50,98), (8,5,50,97)]}
    assert choose_history(results) == 4
    for r in results.values():
        r['combined']['ordinary_restarts'] = 6
    assert choose_history(results) is None
