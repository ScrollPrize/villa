"""Terminal formatting for follower training; JSON remains the analysis format."""
import json

from vesuvius.neural_tracing.fiber_follow.shared.policy import DIAGNOSTIC_THRESHOLDS


def _rate(n, d):
    n, d = int(n), int(d)
    return f'{n}/{d} ({n/d:.1%})' if d else 'n/a (0 known)'


def _number(value, spec='.3f'):
    return format(value, spec) if value is not None else '--'


class DirectTrainingInterval:
    """Pool counts and weight decision-normalized means by supervised decisions."""
    means = ('loss', 'geometry', 'confidence_loss', 'fresh_fraction',
             'recent_fraction', 'bank_wrong_continuation_fraction',
             'bank_following_fraction', 'decision_pair_fraction', 'refinement_attempts_mean')
    counts = ('error_sum', 'geometry_count', 'point_correct_count', 'point_wrong_count',
              'point_unknown_count', 'supervised_states', 'observation_only_states', 'history_valid_slabs', 'history_age_sum', 'history_overlap_sum', 'history_load_seconds', 'history_encode_seconds',
              'confidence_labeled_states', 'confidence_departed_states', 'refinement_attempts_sum',
              'supervision_weight', 'endpoint_weight', 'matched_endpoint_weight', 'choice_endpoint_weight',
              'endpoint_states', 'matched_endpoint_states', 'choice_endpoint_states',
              'replay_bank_switch_endpoints', 'replay_premature_stop_endpoints',
              'replay_endpoint_overshoot_endpoints', 'replay_pre_switch_endpoints')
    nested_counts = {'identity': ('candidate_states', 'candidate_intervals', 'candidate_late_failures',
                                 'candidate_first_failures', 'candidate_supervision_weight')}
    nested_means = {'identity': ('candidate_loss',)}

    def __init__(self):
        self.values = dict(updates=0, crops=0, decisions=0)

    def add(self, row):
        if 'dataset_counts' in row:
            counts = self.values.setdefault('dataset_counts', {})
            for key, value in row['dataset_counts'].items():
                counts[key] = counts.get(key, 0)+value
        crops = row['observed_states']
        self.values['updates'] += 1
        self.values['crops'] += crops
        decisions = row['supervised_states']
        self.values['decisions'] += decisions
        for key in self.means + self.counts:
            self.values[key] = self.values.get(key, 0.)+row.get(key, 0.)*(decisions if key in self.means else 1)
        for section in self.nested_counts:
            source = row.get(section, {})
            for key in self.nested_counts[section] + self.nested_means[section]:
                name = section+'_'+key
                weight = decisions if key in self.nested_means[section] else 1
                self.values[name] = self.values.get(name, 0.)+source.get(key, 0.)*weight
        for group in ('history', 'rest'):
            key = group+'_grad_norm'
            self.values[key+'_max'] = max(self.values.get(key+'_max', 0.), row.get(key, 0.))
            key = group+'_clipped_updates'
            self.values[key] = self.values.get(key, 0)+int(row.get(group+'_grad_clip_scale', 1.) < 1.)

    def summary(self):
        result = dict(self.values)
        means = list(self.means)+[s+'_'+k for s, keys in self.nested_means.items() for k in keys]
        for key in means:
            result[key] = result.get(key, 0.)/max(1, result['decisions'])
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
             f" | {row['interval_samples_per_second']:.1f} crops/s"
             f" | {row['interval_samples_per_second']*m['decisions']/max(1, m['crops']):.1f} decisions/s"
             f" | {m['refinement_attempts_mean']:.2f} attempts/decision"]
    if row.get('cuda_peak_allocated_gib') is not None:
        lines[-1] += f" | peak allocated VRAM {row['cuda_peak_allocated_gib']:.2f} GiB (session)"
    slabs = max(1., m.get('history_valid_slabs', 0.))
    lines.append(f"  historical slabs: {m.get('history_valid_slabs', 0.)/max(1, m['decisions']):.2f}/decision"
                 f" | age {m.get('history_age_sum', 0.)/slabs:.1f} voxels"
                 f" | current-crop overlap {m.get('history_overlap_sum', 0.)/slabs:.1%}"
                 f" | load {m.get('history_load_seconds', 0.):.3f}s"
                 f" | encode {m.get('history_encode_seconds', 0.):.3f}s")
    lines.append(f"  supervision: {int(m['decisions'])} decisions / {int(m['crops'])} observations")
    lines.append(f"  scored crops: candidates {_rate(m['identity_candidate_states'], m['decisions'])}"
                 f" | departed {_rate(m.get('confidence_departed_states', 0), m.get('confidence_labeled_states', 0))}")
    if 'dataset_counts' in m:
        lines.append('  datasets (IDs from dataset_configuration): '+', '.join(
            f'{key}: {value}/{int(m["decisions"])} ({value/max(1,m["decisions"]):.1%})'
            for key,value in sorted(m['dataset_counts'].items())))
    if 'remote_prefetch' in row:
        p = row['remote_prefetch']
        lines.append(f"  remote CT prefetch (cumulative): alive={p['alive']} | active {p['active']}"
                     f" | completed {p['completed']} chunks / {p['completed_bytes']/2**20:.1f} MiB"
                     f" | promoted {p['promoted']} | preempted {p['preempted']}"
                     f" | deferred {p['deferred']} | errors {p['errors']}")
        if 'lookahead_chunks' in p:
            lines[-1] += f" | lookahead {p['lookahead_chunks']} chunk references / {p['lookahead_windows']} windows"
    if m.get('endpoint_states'):
        lines.append(f"  supervised endpoints: matched {_rate(m['matched_endpoint_states'], m['endpoint_states'])}"
                     f" | geometry choices {_rate(m['choice_endpoint_states'], m['endpoint_states'])}")
        weight = max(m['supervision_weight'], 1e-12)
        lines.append(f"  task loss budget: endpoints {m['endpoint_weight']/weight:.1%}"
                     f" | matched {m['matched_endpoint_weight']/weight:.1%}"
                     f" | geometry choices {m['choice_endpoint_weight']/weight:.1%}")
        lines.append(f"  candidate hazard targets: {int(m['identity_candidate_intervals'])} intervals"
                     f" | {int(m['identity_candidate_first_failures'])} first-segment failures"
                     f" | {int(m['identity_candidate_late_failures'])} later failures")
    lines.append(f"  candidate survival loss {m['identity_candidate_loss']:.4f}"
                 f" ({int(m['identity_candidate_states'])} eligible crops)")
    lines.append('  data: '+' / '.join(f'{name} {m[key]:.0%}' for name, key in
                 (('fresh','fresh_fraction'), ('recent','recent_fraction'),
                  ('wrong turns','bank_wrong_continuation_fraction'), ('following','bank_following_fraction'),
                  ('pairs','decision_pair_fraction'))))

    failures = ('bank_switch', 'pre_switch', 'premature_stop', 'endpoint_overshoot')
    if any(m.get('replay_'+name+'_endpoints', 0) for name in failures):
        lines.append('  replay failure endpoints: '+' / '.join(
            f'{name.replace("_", " ")} {int(m.get("replay_"+name+"_endpoints", 0))}' for name in failures))
    lines.append('  gradients: '+' | '.join(
        f"{name} max {m[name+'_grad_norm_max']:.2g}, clipped {m[name+'_clipped_updates']}/{updates} updates"
        for name in ('history', 'rest')))
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
    if 'history_grad_norm' in row:
        lines.append(f"  gradients before clipping: history {row['history_grad_norm']:.3g}"
                     f" (scale {row['history_grad_clip_scale']:.3g})"
                     f" | rest {row['rest_grad_norm']:.3g} (scale {row['rest_grad_clip_scale']:.3g})")
    if 'identity' in row:
        lines.extend(_identity_lines(row['identity']))
    if 'decisions' in row:
        lines.extend(_decision_lines(row['decisions']))
    return lines


def _identity_lines(stats):
    lines = [f"  identity-aware prefix correct {stats.get('identity_prefix_correct_fraction', 0.):.1%}"
             +f" | labels flipped {int(stats.get('identity_flipped_count', 0))}",
             f"  identity data: presence dropped {stats.get('presence_dropped_fraction', 0.):.0%}"
             f" | neighbor coverage {stats.get('foreign_components_fraction', 0.):.0%}"
             +''.join(f" | {key[9:-9]} {value:.0%}" for key, value in stats.items()
                      if key.startswith('location_') and key.endswith('_fraction'))]
    if 'blurred_fraction' in stats:
        lines[-1] += f" | blurred {stats['blurred_fraction']:.0%}"
    if 'candidate_loss' in stats:
        lines.append(f"  candidate survival loss {stats['candidate_loss']:.4f}"
                     +f" | per eligible state {stats['candidate_loss_eligible']:.4f}")
    for name, group in stats.get('candidate_decisions', {}).items():
        if group['states']:
            lines.append(f"  identity decisions {name}: correct accepted "
                         +_rate(group['accepted_positive'], group['positive'])
                         +" | wrong rejected "+_rate(group['rejected_negative'], group['negative']))
    return lines


def format_training_log(row):
    step = f"Step {row['step']:,}" if 'step' in row else 'Training'
    if row.get('event') == 'resume_configuration':
        options, cfg = row.get('training_options', {}), row.get('model_cfg', {})
        return (f"{step} | resumed {row['checkpoint']}\n"
                f"  commit {options.get('n_commit', '?')} | historical slabs: 8 slots, minimum spacing 32 vox"
                f" | batch {options.get('batch', '?')} / microbatch {options.get('microbatch', '?')}"
                f"\n  live CT/path slabs | causal survival confidence")
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
