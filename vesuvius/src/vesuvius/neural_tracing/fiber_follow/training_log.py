"""Terminal formatting for follower training; JSON remains the analysis format."""
import json


def _rate(n, d):
    n, d = int(n), int(d)
    return f'{n}/{d} ({n/d:.1%})' if d else 'n/a (0 known)'


def _number(value, spec='.3f'):
    return format(value, spec) if value is not None else '--'


def _decision_lines(decisions):
    bands = decisions['by_drift']
    stats = bands['all']
    if not stats['states']:
        return ['  decisions: no states']
    lines = [f"  {decisions['n_commit']}-point correctness: "
             +_rate(stats['commit_correct'], stats['commit_known'])]
    for key, gate in stats.items():
        if not key.startswith('gate_'):
            continue
        lines.append(f"  gate @ {float(key[5:]):.2f}: false stops "
                     +_rate(gate['false_stops'], stats['first_correct'])
                     +' | accepted wrong '+_rate(gate['accepted_wrong'], gate['accepted_known'])
                     +f" | accepted unknown {gate['accepted_unknown']} | departed continues "
                     +_rate(gate['departed_continues'], bands['departed']['states']))
    lines.append('  refinement: mean lateral error in voxels')
    for title, groups in (('drift', bands), ('history', decisions.get('by_history', {}))):
        if not groups:
            continue
        lines.append(f'    {title:<10} states  GT pts  initial    final')
        for name, group in groups.items():
            if not group['states']:
                continue
            lines.append(f"    {name:<10} {group['states']:6d} {group['final_error_count']:7d}"
                         f"  {_number(group['initial_error_mean']):>7} {_number(group['final_error_mean']):>8}")
    lines.append('    improved/worsened/compared states: '
                 +f"{stats['correction_improved']}/{stats['correction_worsened']}/{stats['correction_comparable']}")
    if stats['initial_nonfinite_count'] or stats['final_nonfinite_count']:
        lines.append(f"    nonfinite predictions: initial {stats['initial_nonfinite_count']}"
                     f" | final {stats['final_nonfinite_count']}")
    return lines


def _direct_training_lines(row):
    lines = [f"  geometry {row['geometry']:.4f} | confidence {row['confidence_loss']:.4f}"
             f" | mean error {row['error_mean']:.3f} voxels | prefix correct {row['prefix_correct_fraction']:.1%}",
             f"  data: fresh {row['fresh_fraction']:.0%} | fixed {row['fixed_fraction']:.0%}"
             f" | recent {row['recent_fraction']:.0%}"]
    if 'judge_loss' in row:
        lines.append(f"  judge: loss {row['judge_loss']:.4f} | sequences {row['judge_sequences']}"
                     f" | positive {row['judge_positive']} | departed {row['judge_departed']}"
                     f" | unknown {row['judge_unknown']}")
        lines.append('    sources: '+' | '.join(f"{key.replace('_', ' ')} {value:.0%}"
                     for key, value in row['judge_source_fractions'].items()))
        lines.append(f"    synthetic: attempts {row['judge_synthetic_attempts']}"
                     f" | fallback {row['judge_synthetic_fallback']}")
        if 'judge_replay_requested' in row:
            lines.append(f"    departed replay: added {row['judge_replay_added']}/{row['judge_replay_requested']}"
                         f" | attempts {row['judge_replay_attempts']} | pool {row['judge_replay_pool_rows']} rows"
                         f" / {row['judge_replay_pool_fibers']} fibers")
    if 'decisions' in row:
        lines.extend(_decision_lines(row['decisions']))
    return lines


def format_training_log(row):
    step = f"Step {row['step']:,}" if 'step' in row else 'Training'
    if 'beam_rollout' in row:
        threshold = row['threshold']
        gate = f" @ {threshold:.2f}" if threshold is not None else ''
        return (f"\n{step} | beam rollout {row['beam_rollout']}{gate} | {row['n']} seeds\n"
                f"  coverage {row['coverage_mean']:.1%} | precision {row['length_precision']:.1%}"
                f" | diverged {row['diverged']:.1%}\n"
                f"  image: {row['image']}")
    if 'span_fibers' in row:
        if not row['span_fibers']:
            return f'\n{step} | beam spans | no held-out fibers'
        lines = [f"\n{step} | beam spans | {row['span_fibers']} fibers"
                 f" | {row['span_segments']} spans | {row['span_length_grid']:.1f} voxels"]
        for method in ('hand', 'model'):
            if f'span_success_{method}' in row:
                lines.append(f"  {method}: success {row[f'span_success_{method}']:.1%}"
                             f" | restarts / 1,000 voxels {row[f'span_restarts_kvx_{method}']:.3f}")
        return '\n'.join(lines)
    if 'recovery' in row:
        report = row['recovery']
        lines = [f"\n{step} | monitor recovery | {report['states']} states"]
        for threshold, groups in report['thresholds'].items():
            lines.append(f'  recovery @ {float(threshold):.2f}:')
            for name, stats in groups.items():
                if stats['states']:
                    lines.append(f'    {name}: recovered '+_rate(stats['recovered'], stats['recovery_observed'])
                                 +f" | false stops {stats['false_stops']}")
        lines.extend(_decision_lines(report['decisions']))
        return '\n'.join(lines)
    if 'judge' in row and isinstance(row['judge'], dict):
        report = row['judge']
        return (f"\n{step} | monitor judge @ {row['threshold']:.2f}\n"
                f"  precision {_number(report['precision'], '.1%')}"
                f" | wrong reduction {_number(report['wrong_reduction'], '.1%')}"
                f" | correct loss {_number(report['correct_loss'], '.1%')}\n"
                f"  false stops / 10,000 voxels {_number(report['false_stops_per_10000'])}"
                f" | unknown stops {report['unknown_stops']} | missing source {report['missing_source']}")
    if 'length_weighted_coverage' in row:
        return (f"\n{step} | monitor rollout @ {row['threshold']:.2f}\n"
                f"  coverage {row['length_weighted_coverage']:.1%} | precision {row['length_precision']:.1%}"
                f" | diverged {row['diverged']:.1%}")
    if 'roll_coverage' in row:
        return (f"\n{step} | rollout @ {row['threshold']:.2f}\n"
                f"  coverage {row['roll_coverage']:.1%} | precision {row['roll_precision']:.1%}"
                f" | diverged {row['roll_diverged']:.1%}")
    if 'loss' not in row:
        details = ' | '.join(f'{key}: {json.dumps(value)}' for key,value in row.items() if key != 'step')
        return f'{step} | {details}'

    if 'ranking' in row:
        return '\n'.join([
            f"\n{step} | loss {row['loss']:.4f} | lr {row['lr']:.2e}"
            f" | {row['samples_per_second']:.2f} samples/s",
            f"  ranking {row['ranking']:.4f} | on-fiber {row['onfiber']:.4f}",
            f"  candidate correctness: model top-1 {row['model_top1_onfiber']:.1%}"
            f" | oracle {row['oracle_onfiber']:.1%}"
            f" | labeled candidates/state {row['labeled_candidates']:.1f}",
            f"  data: hard {row['hard_fraction']:.0%} | mined or hard {row['mined_fraction']:.0%}",
        ])

    if 'geometry' in row:
        return '\n'.join([f"\n{step} | loss {row['loss']:.4f} | lr {row['lr']:.2e}"
                          f" | {row['samples_per_second']:.2f} samples/s", *_direct_training_lines(row)])

    lines = [f"\n{step} | loss {row['loss']:.4f} | lr {row['lr']:.2e}"
             f" | {row['samples_per_second']:.2f} samples/s",
             f"  flow {row['flow']:.4f} | confidence {row['confidence_loss']:.4f}"
             f" (weight {row['confidence_coefficient']:.2f})",
             f"  data: fresh {row['fresh_fraction']:.0%} | fixed {row['fixed_fraction']:.0%}"
             f" | recent {row['recent_fraction']:.0%} | replay seen {row['replay_samples_seen']:,}"]

    def rate(numerator, denominator):
        return _rate(row[numerator], row[denominator])

    lines.append(f"  {int(row.get('commit_window',0)) or 'commit'}-point correctness: "+rate('commit_correct_count','commit_known_count'))
    for threshold in (.5,.85):
        lines.append(f'  gate @ {threshold:.2f}: false stops '
                     +rate(f'false_stop_count_{threshold}',f'correct_first_count_{threshold}')
                     +' | departed continues '
                     +rate(f'departed_continue_count_{threshold}',f'departed_count_{threshold}'))
    if 'refinement' in row:
        refinement = row['refinement']
        bands = refinement['by_drift']
        stages = len(bands['all']['error_mean'])
        lines.append(f"  refinement: first {refinement['first_n']} points, mean lateral error in voxels"
                     f" ({refinement['departed_count']} departed excluded)")
        lines.append('    drift      states  GT pts  '+' '.join(f'{name:>7}' for name in
                     ['initial']+[f'step {i}' for i in range(1,stages)]))
        for name,stats in bands.items():
            if name != 'all' and not stats['state_count']:
                continue
            values = ' '.join(f'{v:7.3f}' if v is not None else f'{"--":>7}' for v in stats['error_mean'])
            lines.append(f"    {name:<10} {stats['state_count']:6d} {stats['known_point_count']:7d}  {values}")
            if any(stats['nonfinite_point_count']):
                lines.append(f"      nonfinite predictions by step: {stats['nonfinite_point_count']}")
        stats = bands['all']
        changes = ' | '.join(f'{i}: {better}/{worse}/{count}' for i,(better,worse,count) in
                            enumerate(zip(stats['improved_count'],stats['worsened_count'],stats['comparison_count']),1))
        lines.append('    improved/worsened/compared states by update: '+changes)
    return '\n'.join(lines)
