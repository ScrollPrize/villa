"""Terminal formatting for follower training; JSON remains the analysis format."""
import json

from vesuvius.neural_tracing.fiber_follow.shared.policy import DIAGNOSTIC_THRESHOLDS


def _rate(n, d):
    n, d = int(n), int(d)
    return f'{n}/{d} ({n/d:.1%})' if d else 'n/a (0 known)'


def _number(value, spec='.3f'):
    return format(value, spec) if value is not None else '--'


class DirectTrainingInterval:
    """Pool every update's counts; weight per-crop means by actual crop count."""
    means = ('loss', 'geometry', 'confidence_loss', 'replay_loss', 'fresh_fraction',
             'recent_fraction', 'bank_wrong_continuation_fraction',
             'bank_following_fraction', 'decision_pair_fraction')
    counts = ('error_sum', 'geometry_count', 'point_correct_count', 'point_wrong_count',
              'point_unknown_count', 'replay_endpoints', 'replay_observations', 'replay_encoder_crops',
              'confidence_labeled_states', 'confidence_departed_states')
    nested_counts = {'identity': ('identity_count', 'identity_rank_correct', 'candidate_states'),
                     'memory': ('identity_count', 'identity_correct', 'departed_count',
                                'departed_correct', 'offset_count', 'offset_error_sum')}
    nested_means = {'identity': ('identity_loss', 'candidate_loss'),
                    'memory': ('probe_identity_loss', 'probe_offset_loss')}

    def __init__(self):
        self.values = dict(updates=0, crops=0)

    def add(self, row):
        crops = row['observed_states']
        self.values['updates'] += 1
        self.values['crops'] += crops
        for key in self.means + self.counts:
            self.values[key] = self.values.get(key, 0.)+row.get(key, 0.)*(crops if key in self.means else 1)
        for section in self.nested_counts:
            source = row.get(section, {})
            for key in self.nested_counts[section] + self.nested_means[section]:
                name = section+'_'+key
                weight = crops if key in self.nested_means[section] else 1
                self.values[name] = self.values.get(name, 0.)+source.get(key, 0.)*weight
        for group in ('memory', 'rest'):
            key = group+'_grad_norm'
            self.values[key+'_max'] = max(self.values.get(key+'_max', 0.), row.get(key, 0.))
            key = group+'_clipped_updates'
            self.values[key] = self.values.get(key, 0)+int(row.get(group+'_grad_clip_scale', 1.) < 1.)

    def summary(self):
        result = dict(self.values)
        means = list(self.means)+[s+'_'+k for s, keys in self.nested_means.items() for k in keys]
        for key in means:
            result[key] = result.get(key, 0.)/max(1, result['crops'])
        result['error_mean'] = result.get('error_sum', 0.)/result['geometry_count'] if result.get('geometry_count') else None
        return result


def _interval_training_lines(row):
    m = row['interval']
    right, wrong, unknown = (int(m[k]) for k in ('point_correct_count', 'point_wrong_count', 'point_unknown_count'))
    score = f'{right/(right+wrong):.1%}' if right+wrong else 'n/a'
    updates = m['updates']
    lines = [f"  points ({row['n_future']}/curve, tolerance {row['tolerance']:g} vox): "
             f"{right:,} right / {wrong:,} wrong | {score} correct | {unknown:,} unknown",
             f"  loss {m['loss']:.4f} | geometry {m['geometry']:.4f} | confidence {m['confidence_loss']:.4f}"
             f" | mean error {_number(m['error_mean'])} vox",
             f"  speed: {1000*row['interval_update_seconds']/updates:.0f} ms/update"
             f" | data wait {1000*row['interval_data_seconds']/updates:.0f} ms/update"
             f" | {row['interval_samples_per_second']:.1f} crops/s"]
    if row.get('cuda_peak_allocated_gib') is not None:
        lines[-1] += f" | peak allocated VRAM {row['cuda_peak_allocated_gib']:.2f} GiB (session)"
    if m['memory_identity_count']:
        lines.append(f"  memory probe: correct {_rate(m['memory_identity_correct'], m['memory_identity_count'])}"
                 f" | departed recall {_rate(m['memory_departed_correct'], m['memory_departed_count'])}"
                 f" | departed loss weight {row.get('memory_departed_weight', 1.):g}x")
        lines.append(f"  probe labels: departed {_rate(m['memory_departed_count'], m['memory_identity_count'])}"
                     f" | false departure {_rate(m['memory_identity_count']-m['memory_departed_count']-(m['memory_identity_correct']-m['memory_departed_correct']), m['memory_identity_count']-m['memory_departed_count'])}")
    lines.append(f"  replay: {int(m['replay_endpoints'])} endpoints / {int(m['replay_encoder_crops'])} encoder crops"
                 f" (loss {m['replay_loss']:.4f})")
    lines.append(f"  scored crops: candidates {_rate(m['identity_candidate_states'], m['crops'])}"
                 f" | departed {_rate(m.get('confidence_departed_states', 0), m.get('confidence_labeled_states', 0))}")
    lines.append(f"  identity: rank {_rate(m['identity_identity_rank_correct'], m['identity_identity_count'])}"
                 f" | InfoNCE {m['identity_identity_loss']:.4f} | candidate BCE {m['identity_candidate_loss']:.4f}"
                 f" ({int(m['identity_candidate_states'])} eligible crops)")
    lines.append('  data: '+' / '.join(f'{name} {m[key]:.0%}' for name, key in
                 (('fresh','fresh_fraction'), ('recent','recent_fraction'),
                  ('switch streams','bank_wrong_continuation_fraction'), ('following','bank_following_fraction'),
                  ('pairs','decision_pair_fraction'))))
    if row.get('feature_switch_crop_fraction', -1.) >= 0:
        lines[-1] += f" | switch crop budget {row['feature_switch_crop_fraction']:.0%} (cumulative)"
    lines.append('  gradients: '+' | '.join(
        f"{name} max {m[name+'_grad_norm_max']:.2g}, clipped {m[name+'_clipped_updates']}/{updates} updates"
        for name in ('memory', 'rest')))
    return lines


def _decision_lines(decisions):
    bands = decisions['by_drift']
    stats = bands['all']
    if not stats['states']:
        return ['  decisions: no states']
    lines = [f"  entire proposed {decisions['n_commit']}-point prefix correct (distance): "
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
             f"  data: fresh {row['fresh_fraction']:.0%}"
             f" | recent {row['recent_fraction']:.0%}"]
    if 'interval_samples_per_second' in row:
        lines.insert(0, f"  recent speed {row['interval_samples_per_second']:.2f} samples/s"
                        f" | data wait {row['interval_data_seconds']:.2f}s"
                        f" | optimizer {row['interval_update_seconds']:.2f}s per logging interval")
    if 'bank_wrong_continuation_fraction' in row:
        lines[-1] += f" | bank departures {row['bank_wrong_continuation_fraction']:.0%}"
    if 'bank_following_fraction' in row:
        lines[-1] += f" | bank following {row['bank_following_fraction']:.0%}"
    if 'decision_pair_fraction' in row:
        lines[-1] += f" | identity pairs {row['decision_pair_fraction']:.0%} (requested {row.get('decision_requested_fraction', 0.):.0%})"
    if 'memory_grad_norm' in row:
        lines.append(f"  gradients before clipping: memory {row['memory_grad_norm']:.3g}"
                     f" (scale {row['memory_grad_clip_scale']:.3g})"
                     f" | rest {row['rest_grad_norm']:.3g} (scale {row['rest_grad_clip_scale']:.3g})")
    if 'identity' in row:
        lines.extend(_identity_lines(row['identity']))
    if 'memory' in row:
        m = row['memory']
        lines.append(f"  memory probe: identity {m.get('probe_identity_loss', 0.):.4f}"
                     f" (acc {m.get('probe_identity_accuracy', 0.):.1%}, departed recall "
                     +_rate(m.get('departed_correct', 0), m.get('departed_count', 0))+")"
                     f" | offset {m.get('probe_offset_loss', 0.):.4f} ({m.get('probe_offset_error_mean', 0.):.2f} vox)"
                     f" | labeled writes/state {m.get('labeled_writes_per_state', 0.):.1f}"
                     f" | departed states {m.get('departed_state_fraction', 0.):.0%}")
    if 'decisions' in row:
        lines.extend(_decision_lines(row['decisions']))
    return lines


def _identity_lines(stats):
    lines = [f"  identity: InfoNCE {stats.get('identity_loss', 0.):.4f} | rank "
             +_rate(stats.get('identity_rank_correct', 0), stats.get('identity_count', 0))
             +f" | scored states {int(stats.get('identity_states', 0))}"
             +f" ({stats.get('eligible_fraction', 0.):.1%} eligible)"
             +f" | identity-aware prefix correct {stats.get('identity_prefix_correct_fraction', 0.):.1%}"
             +f" | labels flipped {int(stats.get('identity_flipped_count', 0))}",
             f"  identity data: presence dropped {stats.get('presence_dropped_fraction', 0.):.0%}"
             f" | neighbor coverage {stats.get('foreign_components_fraction', 0.):.0%}"
             +''.join(f" | {key[9:-9]} {value:.0%}" for key, value in stats.items()
                      if key.startswith('location_') and key.endswith('_fraction'))]
    if 'blurred_fraction' in stats:
        lines[-1] += f" | blurred {stats['blurred_fraction']:.0%}"
    if 'ranking' in stats:
        lines.append('  history-vs-neighbor ranking: '+' | '.join(
            f"{name} {_rate(v['correct'], v['pairs'])}" for name, v in stats['ranking'].items() if v['pairs']))
    if 'identity_loss_eligible' in stats:
        lines.append(f"  InfoNCE per eligible state {stats['identity_loss_eligible']:.4f}"
                     +f" | observed seed present {stats.get('seed_present_fraction', 0.):.0%}")
    if 'candidate_loss' in stats:
        lines.append(f"  candidate BCE {stats['candidate_loss']:.4f}"
                     +f" | per eligible state {stats['candidate_loss_eligible']:.4f}")
    for name, group in stats.get('candidate_decisions', {}).items():
        if group['states']:
            lines.append(f"  identity decisions {name}: correct accepted "
                         +_rate(group['accepted_positive'], group['positive'])
                         +" | wrong rejected "+_rate(group['rejected_negative'], group['negative']))
    for name,group in stats.get('training_groups',{}).items():
        if group['states']:
            lines.append(f"  identity {name}: {_rate(group['eligible_states'],group['states'])} eligible"
                         f" | paths/positive {group['distinct_paths_per_positive']:.2f}"
                         f" | recent/seed {group['recent_states']}/{group['seed_states']}"
                         f" | geometry {group['geometry_mean']:.4f} | confidence {group['confidence_mean']:.4f}"
                         f" | InfoNCE {group['identity_mean']:.4f}")
    return lines


def format_training_log(row):
    step = f"Step {row['step']:,}" if 'step' in row else 'Training'
    if row.get('event') == 'resume_configuration':
        options, cfg = row.get('training_options', {}), row.get('model_cfg', {})
        memory = 'observation memory | causal survival confidence'
        return (f"{step} | resumed {row['checkpoint']}\n"
                f"  commit {options.get('n_commit', '?')} | history spacing {cfg.get('memory_stride', '?')} vox"
                f" | batch {options.get('batch', '?')} / microbatch {options.get('microbatch', '?')}"
                f"\n  {memory}"
                f" | switch crop budget {cfg.get('feature_switch_crop_fraction', -1.):g}")
    if row.get('event') == 'identity_sampling':
        bank = row.get('negative_bank_provenance') or {}
        return (f"{step} | identity sampling v{row.get('pair_sampling_version', '?')}"
                f" | bank {row.get('negative_bank_path', '(see run configuration)')}"
                f" | {len(bank.get('shard_hashes', {})):,} published shards"
                f" | run {bank.get('run_digest', 'unknown')[:12]}")
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

    if 'geometry' in row:
        if 'interval' in row:
            m = row['interval']
            return '\n'.join([f"\n{step} | last {m['updates']} updates / {m['crops']:,} crops"
                              f" | lr {row['lr']:.2e}", *_interval_training_lines(row)])
        return '\n'.join([f"\n{step} | loss {row['loss']:.4f} | lr {row['lr']:.2e}"
                          f" | {row['samples_per_second']:.2f} samples/s", *_direct_training_lines(row)])

    lines = [f"\n{step} | loss {row['loss']:.4f} | lr {row['lr']:.2e}"
             f" | {row['samples_per_second']:.2f} samples/s",
             f"  flow {row['flow']:.4f} | confidence {row['confidence_loss']:.4f}"
             f" (weight {row['confidence_coefficient']:.2f})",
             f"  data: fresh {row['fresh_fraction']:.0%}"
             f" | recent {row['recent_fraction']:.0%} | replay seen {row['replay_samples_seen']:,}"]

    def rate(numerator, denominator):
        return _rate(row[numerator], row[denominator])

    lines.append(f"  {int(row.get('commit_window',0)) or 'commit'}-point correctness: "+rate('commit_correct_count','commit_known_count'))
    if 'candidate_states' in row:
        lines.append('  candidates: zero '+rate('candidate_zero_correct','candidate_states')
                     +' | selected '+rate('candidate_selected_correct','candidate_states')
                     +' | oracle '+rate('candidate_oracle_correct','candidate_states'))
        lines.append(f"    rescued {int(row['candidate_rescues'])} | spoiled {int(row['candidate_spoiled'])}")
    for threshold in DIAGNOSTIC_THRESHOLDS:
        lines.append(f'  gate @ {threshold:.2f}: false stops '
                     +rate(f'false_stop_count_{threshold}',f'correct_first_count_{threshold}')
                     +' | departed continues '
                     +rate(f'departed_continue_count_{threshold}',f'departed_count_{threshold}'))
        if f'candidate_fallback_count_{threshold}' in row:
            lines.append(f"    alternative-prefix fallbacks {int(row[f'candidate_fallback_count_{threshold}'])}"
                         +' | correct first point '+rate(f'candidate_fallback_correct_{threshold}',
                                                        f'candidate_fallback_known_{threshold}'))
    if 'refinement' in row:
        refinement = row['refinement']
        bands = refinement['by_drift']
        stages = len(bands['all']['error_mean'])
        label = 'candidate-zero refinement' if 'candidate_states' in row else 'refinement'
        lines.append(f"  {label}: first {refinement['first_n']} points, mean lateral error in voxels"
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
