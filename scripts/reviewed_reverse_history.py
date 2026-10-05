"""Directional-history reverse-seed pilot, using the existing full-prefix runner."""
import json
from reviewed_reverse_prefix import ROOT, MODEL, EXPERIMENT as PREVIOUS, workload, evaluate_config, report
from lasagna.inference_provenance import atomic_write, sha256_file

EXPERIMENT = ROOT/'reverse_history_40500_v1'
NOTE = ('History H is a directional prior: negative last H forward edges, oldest to newest, '
        'folded with the existing rolling smoothness rule (window4). The most recent edge seeds '
        'previous-step direction; model direction remains separate. H0 preserves the cold model-direction seed. '
        'History samples do not consume search depth or fabricate path points. Every reverse step is freely inferred '
        'and checked. Checkpoints use actual committed vertices. This is not exact reverse curvature history '
        '(future forward points beyond terminal endpoints are unavailable). H1 isolates actual-edge initialization. '
        'Lower filtered counts may weaken rejection; choose only among H1/4/8 with <=5ordinary failures, '
        'then minimize filtered stops, tie-breaking by retained coverage. If none qualifies at256, test the least-error '
        'candidate at128/512 as a bounded follow-up, without calling it a winner. Selection is tuned on this cohort, not independent validation.')


def config_for(base, interval, history):
    name = f'prefix{interval}_history{history}'
    config = workload(base, name, interval, True, EXPERIMENT)
    config['evaluation']['trace_args'] += ['--replay-reverse-history-steps', str(history)]
    return name, config


def choose_history(results):
    viable = [(h, results[f'prefix256_history{h}']['combined']) for h in (1,4,8)
              if results[f'prefix256_history{h}']['combined']['ordinary_restarts'] <= 5]
    return min(viable, key=lambda item: (item[1]['filtered_stops'], -item[1]['retained_percent'], item[0]))[0] if viable else None


def main():
    base = json.loads((ROOT/MODEL/'workload.json').read_text())
    if sha256_file(base['checkpoint']) != base['checkpoint_sha256']:
        raise ValueError('checkpoint changed')
    done = []
    def run(interval, history):
        name, config = config_for(base, interval, history)
        evaluate_config(config, name)
        done.append((name, interval, True))
        result = report(EXPERIMENT, done,
                        [(f'old_prefix{i}', PREVIOUS/f'prefix{i}_terminal') for i in (128,256,512)],
                        'reviewed_reverse_history_40500_20261002', NOTE, 'reviewed_reverse_history')
        print('HISTORY RESULT ' + name + ': ' + json.dumps(result[name]), flush=True)
        return result
    results = run(256, 0)
    # Strict geometry/result guard: new history counter must preserve old math
    # when no prior is supplied; ignore only provenance metadata.
    old_cases = sorted((PREVIOUS/'prefix256_terminal/full/cases').glob('*/trace.json'))
    new_cases = sorted((EXPERIMENT/'prefix256_history0/full/cases').glob('*/trace.json'))
    if len(old_cases) != 182 or {p.parent.name for p in old_cases} != {p.parent.name for p in new_cases}:
        raise RuntimeError('cold comparison needs exactly182 matching case artifacts')
    for old in old_cases:
        new = EXPERIMENT/'prefix256_history0/full/cases'/old.parent.name/'trace.json'
        a, b = json.loads(old.read_text()), json.loads(new.read_text())
        if (a['replay'] != b['replay'] or a['reference_points_base_xyz'] != b['reference_points_base_xyz']
                or any(a['provenance'][k] != b['provenance'][k] for k in ('coordinate_space', 'case_index'))):
            raise RuntimeError(f'cold reverse replay changed for case {old.parent.name}')
    print('Cold replay equality verified:182 matching cases including reference geometry/coordinate space.', flush=True)
    for h in (1,4,8):
        results = run(256, h)
    winner = choose_history(results)
    candidate = winner if winner is not None else min((1,4,8), key=lambda h: (
        results[f'prefix256_history{h}']['combined']['ordinary_restarts'],
        results[f'prefix256_history{h}']['combined']['filtered_stops'],
        -results[f'prefix256_history{h}']['combined']['retained_percent'], h))
    atomic_write(EXPERIMENT/'selection.json', dict(history_steps=winner,
                 followup_history_steps=candidate,
                 improves_filtered_at_256=(winner is not None and results[f'prefix256_history{winner}']['combined']['filtered_stops'] <
                                            results['prefix256_history0']['combined']['filtered_stops']),
                 criterion='ordinary<=5, minimum filtered, maximum retained'), durable=False)
    for interval in (128,512):
        results = run(interval, candidate)
    qualified = {name: r['combined'] for name, r in results.items()
                 if name.startswith(('prefix', 'old_prefix')) and r['combined']['ordinary_restarts'] <= 5}
    key = lambda name: (qualified[name]['filtered_stops'], -qualified[name]['retained_percent'])
    new_qualified = [n for n in qualified if n.startswith('prefix') and not n.endswith('history0')]
    atomic_write(EXPERIMENT/'final_selection.json', dict(
        best_qualifying_new=min(new_qualified, key=key) if new_qualified else None,
        fewest_filtered_including_baselines=min(qualified, key=key),
        note='All selections tuned on this same cohort; defaults unchanged.'), durable=False)


if __name__ == '__main__':
    main()
