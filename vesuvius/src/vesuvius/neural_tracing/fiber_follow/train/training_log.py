"""Terminal formatting for follower training; JSON remains the analysis format."""
import json


from vesuvius.neural_tracing.fiber_follow.tracing.policy import DIAGNOSTIC_THRESHOLDS
from vesuvius.neural_tracing.fiber_follow.models.identity_verifier import VERIFY_METRICS


def _rate(n, d):
    n, d = int(n), int(d)
    return f'{n}/{d} ({n/d:.1%})' if d else 'n/a (0 known)'


def _number(value, spec='.3f'):
    return format(value, spec) if value is not None else '--'


class SamplingLedger:
    """Per dataset source: requested and delivered task shares, fallbacks and supply.

    Also label availability, positive/negative confidence targets, distinct fibers,
    episodes and events, replay/live source age, startup draws and realized seed ages,
    and replay travel strata. Accumulates CPU batch metadata between log lines.
    """
    def __init__(self):
        self.sources = {}

    def add(self, cpu, step, targets=None):
        import numpy as np
        import torch
        from vesuvius.neural_tracing.fiber_follow.data.data import FALLBACKS, SEED_AGE_STRATA, SOURCES, STARTUP_CATEGORIES, TASKS, TRAVEL_STRATA, seed_age_stratum
        from vesuvius.neural_tracing.fiber_follow.data.state_labels import REASONS, REPLAY_CLASSES, SUPERVISION
        n = len(cpu['hist'])
        ids = cpu.get('dataset_id', torch.zeros(n, dtype=torch.long)).tolist()
        column = lambda key, default=-1: (cpu[key].tolist() if key in cpu else [default]*n)
        requested, delivered, fallback = column('task_requested'), column('task_delivered'), column('task_fallback', 0)
        supervision, reasons = column('supervision'), column('supervision_reason')
        geometry, confidence = column('geometry_valid', False), column('confidence_valid', False)
        sources, steps = column('source'), column('source_step')
        fibers, episodes, events = column('fiber_id'), column('replay_episode'), column('replay_event')
        startup, ages, travel = column('startup'), column('seed_age', 0.), column('travelled', 0.)
        excursions, classes = column('excursion', False), column('replay_class')
        for row, source in enumerate(ids):
            entry = self.sources.setdefault(str(source), dict(
                rows=0, requested={}, delivered={}, fallbacks={}, supervision={}, reasons={}, sources={},
                replay_classes={}, geometry_valid=0, confidence_valid=0, positive_targets=0., negative_targets=0.,
                geometry_states=0, fibers=set(), episodes=set(), events=set(), source_age_sum=0, source_age_count=0,
                source_age_max=0, startup_requested={}, seed_age={}, replay_travel={}, excursions=0, counters={}))
            entry['rows'] += 1
            count = lambda table, key: table.__setitem__(key, table.get(key, 0)+1)
            if requested[row] >= 0:
                count(entry['requested'], TASKS[requested[row]])
            if delivered[row] >= 0:
                count(entry['delivered'], TASKS[delivered[row]])
            if fallback[row] > 0 and requested[row] >= 0:
                count(entry['fallbacks'], f'{TASKS[requested[row]]}->{FALLBACKS[fallback[row]]}')
            if supervision[row] >= 0:
                count(entry['supervision'], SUPERVISION[supervision[row]])
                count(entry['reasons'], REASONS[reasons[row]])
            if sources[row] >= 0:
                count(entry['sources'], SOURCES[sources[row]])
            if classes[row] >= 0:
                count(entry['replay_classes'], REPLAY_CLASSES[classes[row]])
            entry['geometry_valid'] += int(geometry[row])
            entry['confidence_valid'] += int(confidence[row])
            if fibers[row] >= 0:
                entry['fibers'].add(fibers[row])
            if episodes[row] >= 0:
                entry['episodes'].add(episodes[row])
                entry['events'].add(events[row])
            if steps[row] >= 0:
                age = int(step)-steps[row]
                entry['source_age_sum'] += age
                entry['source_age_count'] += 1
                entry['source_age_max'] = max(entry['source_age_max'], age)
            if startup[row] >= 0:
                count(entry['startup_requested'], STARTUP_CATEGORIES[startup[row]])
                count(entry['seed_age'], SEED_AGE_STRATA[seed_age_stratum(ages[row])][0])
                entry['excursions'] += int(excursions[row])
            if sources[row] == SOURCES.index('replay'):
                stratum = int(np.digitize(travel[row], TRAVEL_STRATA[1:-1]))
                count(entry['replay_travel'], f'{TRAVEL_STRATA[stratum]:g}-{TRAVEL_STRATA[stratum+1]:g}')
            if targets is not None:
                entry['positive_targets'] += float(targets[0][row])
                entry['negative_targets'] += float(targets[1][row])
                entry['geometry_states'] += int(targets[2][row])
        # Row-0 totals from the loader (seed rejections, live outcomes, maximum reuse).
        for key in [k for k in cpu if k.startswith(('ct_seed_', 'live_advanced', 'live_stale', 'live_censored',
                                                     'live_empty', 'replay_max_event_reuse'))]:
            counters = self.sources.setdefault(str(ids[0]), {}).setdefault('counters', {})
            value = int(cpu[key][0])
            counters[key] = max(counters.get(key, 0), value) if key == 'replay_max_event_reuse' else counters.get(key, 0)+value

    def summary(self):
        result = {}
        for source, entry in self.sources.items():
            entry = dict(entry)
            rows = max(1, entry.get('rows', 0))
            for key in ('fibers', 'episodes', 'events'):
                entry[key] = len(entry.get(key, ()))
            entry['requested_share'] = {k: v/rows for k, v in entry.get('requested', {}).items()}
            entry['delivered_share'] = {k: v/rows for k, v in entry.get('delivered', {}).items()}
            entry['source_age_mean'] = (entry['source_age_sum']/entry['source_age_count']
                                        if entry.get('source_age_count') else None)
            result[source] = entry
        return result


def _verification_lines(m):
    """Accuracy with memory / shuffled memory / no memory: head-axis groups, decisive pairs, dense samples
    by path stratum and the model's own predicted points."""
    def rate(name):
        count = m.get(f'verify_{name}_count', 0)
        return f"{m.get(f'verify_{name}_correct', 0)/max(1, count):.1%}" if count else 'n/a'

    lines = [f"  identity verification: loss {m.get('verify_loss_sum', 0.)/max(1, m['verify_states']):.3f}"
             f" over {int(m['verify_states'])} states | original fiber ranks first in"
             f" {m.get('verify_listwise_correct', 0)/max(1, m.get('verify_listwise_planes', 0)):.1%}"
             f" of {int(m.get('verify_listwise_planes', 0))} planes"]
    for name, label in (('own', 'on original'), ('switched_recent', 'switched <=64 vox'),
                        ('switched_old', 'switched >64 real'), ('switched_old_synthetic', 'switched >64 synthetic'),
                        ('pair_old', 'original beats path >64 real'),
                        ('pair_old_synthetic', 'original beats path >64 synthetic'),
                        ('dense_path_on', 'dense on path, on original'), ('dense_path_off', 'dense on path, off original'),
                        ('dense_away_on', 'dense off path, on original'), ('dense_away_off', 'dense off path, off original'),
                        ('predicted_on', 'own prediction on original'), ('predicted_off', 'own prediction off original')):
        lines.append(f"    {label} ({int(m.get(f'verify_{name}_count', 0))}): memory {rate(name)}"
                     f" | shuffled memory {rate('shuffled_'+name)} | no memory {rate('empty_'+name)}")
    return lines


class TrainingInterval:
    """Pool counts and weight decision-normalized means by supervised decisions."""
    means = ('loss', 'geometry', 'confidence_loss', 'refinement_attempts_mean')
    counts = tuple(p+'_'+s for p in ('ct_frame', 'history_frame')
                   for s in ('count', 'transported', 'deterministic', 'learned', 'energy_sum', 'gap_sum')) + (
              'live_depth_sum', 'live_travelled_sum', 'live_rows', 'live_terminal_rows',
              'ct_frame_rejected_batches', 'error_sum', 'geometry_count', 'point_correct_count', 'point_wrong_count',
              'point_unknown_count', 'supervised_states', 'observation_only_states', 'history_valid_slabs', 'history_age_sum', 'history_overlap_sum', 'history_load_seconds', 'history_encode_seconds',
              'confidence_labeled_states', 'confidence_terminal_states', 'confidence_recoverable_states',
              'connector_rejected_targets', 'refinement_attempts_sum',
              'memory_recorded', 'memory_missing', 'memory_encoded', 'memory_identity_loss_sum',
              'memory_identity_states', 'memory_identity_pairs', 'memory_identity_correct',
              'memory_identity_anchor_pairs', 'memory_identity_anchor_correct',
              'memory_identity_control_pairs', 'memory_identity_control_correct',
              *(f'memory_identity_{group}_{kind}' for group in ('departed_recent', 'departed_old', 'departed_old_afv')
                for kind in ('pairs', 'correct')), *VERIFY_METRICS)

    def __init__(self):
        self.values = dict(updates=0, crops=0, decisions=0)

    def add(self, row):
        for name in ('live_depth_counts', 'live_start_limit_counts'):
            if name in row:
                counts = self.values.setdefault(name, {})
                for key, value in row[name].items():
                    counts[key] = counts.get(key, 0)+value
        if 'dataset_counts' in row:
            counts = self.values.setdefault('dataset_counts', {})
            for key, value in row['dataset_counts'].items():
                counts[key] = counts.get(key, 0)+value
        self.values['prediction_loss_type'] = row.get('prediction_loss_type', 'geometry')
        crops = row['observed_states']
        self.values['updates'] += 1
        self.values['crops'] += crops
        decisions = row['supervised_states']
        self.values['decisions'] += decisions
        for key in self.means + self.counts:
            self.values[key] = self.values.get(key, 0.)+row.get(key, 0.)*(decisions if key in self.means else 1)
        for group in ('history', 'rest'):
            if group+'_grad_norm' not in row:  # no memory encoder: no history clipping group
                continue
            key = group+'_grad_norm'
            self.values[key+'_max'] = max(self.values.get(key+'_max', 0.), row.get(key, 0.))
            key = group+'_clipped_updates'
            self.values[key] = self.values.get(key, 0)+int(row.get(group+'_grad_clip_scale', 1.) < 1.)

    def summary(self):
        result = dict(self.values)
        for key in self.means:
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
             f"  loss {m['loss']:.4f} | {m.get('prediction_loss_type', 'geometry')} {m['geometry']:.4f} | confidence {m['confidence_loss']:.4f}"
             f" | mean error {_number(m['error_mean'])} vox",
             f"  speed: {1000*row['interval_update_seconds']/updates:.0f} ms/update"
             f" | data wait {1000*row['interval_data_seconds']/updates:.0f} ms/update"
             f" | {row['interval_samples_per_second']:.1f} crops/s"
             f" | {row['interval_samples_per_second']*m['decisions']/max(1, m['crops']):.1f} decisions/s"
             f" | {m['refinement_attempts_mean']:.2f} attempts/decision"]
    if row.get('cuda_peak_allocated_gib') is not None:
        lines[-1] += f" | peak allocated VRAM {row['cuda_peak_allocated_gib']:.2f} GiB (session)"
    if m.get('history_valid_slabs') or m.get('history_encode_seconds'):  # memory models only
        slabs = max(1., m.get('history_valid_slabs', 0.))
        lines.append(f"  historical slabs: {m.get('history_valid_slabs', 0.)/max(1, m['decisions']):.2f}/decision"
                     f" | age {m.get('history_age_sum', 0.)/slabs:.1f} voxels"
                     f" | current-crop overlap {m.get('history_overlap_sum', 0.)/slabs:.1%}"
                     f" | load {m.get('history_load_seconds', 0.):.3f}s"
                     f" | encode {m.get('history_encode_seconds', 0.):.3f}s")
    if any(m.get(key) for key in ('memory_recorded', 'memory_missing', 'memory_encoded')):
        lines.append(f"  decision memory: recorded {int(m.get('memory_recorded', 0))}"
                     f" | encoded from crops {int(m.get('memory_encoded', 0))}"
                     f" | missing {int(m.get('memory_missing', 0))}")
    if m.get('memory_identity_pairs') or m.get('memory_identity_anchor_pairs'):
        rate = lambda name: (f"{m.get(f'memory_identity_{name}correct', 0)/max(1, m.get(f'memory_identity_{name}pairs', 0)):.1%}"
                             f" of {int(m.get(f'memory_identity_{name}pairs', 0))}")
        lines.append(f"  memory identity: InfoNCE {m['memory_identity_loss_sum']/max(1, m['memory_identity_states']):.3f}"
                     f" | query rank {rate('')} | anchor rank {rate('anchor_')} | shuffled-anchor control {rate('control_')}")
        lines.append(f"    departed query rank: recent (<=24 vox) {rate('departed_recent_')}"
                     f" | old (>24) {rate('departed_old_')} | old AFV {rate('departed_old_afv_')}")
    if m.get('verify_states'):
        lines.extend(_verification_lines(m))
    lines.append(f"  supervision: {int(m['decisions'])} decisions / {int(m['crops'])} observations")
    frames = []
    for prefix, label in (('ct_frame', 'current'), ('history_frame', 'history')):
        count = m.get(prefix+'_count', 0)
        if count:
            transported, deterministic = (int(m.get(prefix+'_'+key, 0)) for key in ('transported', 'deterministic'))
            frames.append(f"{label} {transported+deterministic}/{int(count)} fallbacks"
                          f" ({transported} transported, {deterministic} deterministic)"
                          f"; mean gap {m.get(prefix+'_gap_sum', 0)/count:.3f}"
                          +(f"; {int(m[prefix+'_learned'])} learned" if m.get(prefix+'_learned', 0) else ''))
    if frames:
        lines.append('  CT frames: '+' | '.join(frames))
    if m.get('ct_frame_rejected_batches', 0):
        lines.append(f"  CT frames: {int(m['ct_frame_rejected_batches'])} unusable batch plans rejected; retried within source")
    lines.append(f"  confidence-labeled crops: terminal {_rate(m.get('confidence_terminal_states', 0), m.get('confidence_labeled_states', 0))}"
                 f" | recoverable {_rate(m.get('confidence_recoverable_states', 0), m.get('confidence_labeled_states', 0))}"
                 f" | target connections crossing a neighbor {int(m.get('connector_rejected_targets', 0))}")
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
    if m.get('live_rows'):
        lines.append(f"  live continuation: {int(m['live_rows'])} rows | terminal {int(m.get('live_terminal_rows', 0))}"
                     f" | depth {m['live_depth_sum']/m['live_rows']:.2f}"
                     f" | mean travel {m.get('live_travelled_sum', 0)/m['live_rows']:.1f} voxels"
                     f" | chain starts by limit {m.get('live_start_limit_counts', {})}"
                     f" | reached depths {m.get('live_depth_counts', {})}")
    for source, entry in sorted(row.get('sampling', {}).items()):
        lines.append(f"  source {source}: requested "+', '.join(f'{k} {v:.0%}' for k, v in sorted(entry['requested_share'].items()))
                     +' | delivered '+', '.join(f'{k} {v:.0%}' for k, v in sorted(entry['delivered_share'].items())))
        lines.append(f"    fallbacks {entry['fallbacks'] or 'none'} | supervision {entry['supervision']}"
                     f" | geometry {entry['geometry_valid']}/{entry['rows']} | confidence {entry['confidence_valid']}/{entry['rows']}"
                     f" | targets +{entry['positive_targets']:.0f}/-{entry['negative_targets']:.0f}")
        lines.append(f"    fibers {entry['fibers']} | episodes {entry['episodes']} | events {entry['events']}"
                     f" | source age mean {_number(entry['source_age_mean'], '.0f')} max {entry['source_age_max']}"
                     f" | startup {entry['startup_requested']} -> seed ages {entry['seed_age']}"
                     f" | replay travel {entry['replay_travel']} | {entry['counters']}")
    lines.append('  gradients: '+' | '.join(
        f"{name} max {m[name+'_grad_norm_max']:.2g}, clipped {m[name+'_clipped_updates']}/{updates} updates"
        for name in ('history', 'rest') if name+'_grad_norm_max' in m))
    return lines


def _decision_lines(decisions):
    bands = decisions['by_state']
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
                     +f" | accepted unknown {gate['accepted_unknown']} | terminal continues "
                     +_rate(gate['terminal_continues'], bands['terminal']['states']))
    lines.append('  refinement: mean lateral error in voxels')
    for title, groups in (('state', bands), ('displaced', decisions.get('by_displacement', {})),
                          ('history', decisions.get('by_history', {}))):
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
    lines = [f"  {row.get('prediction_loss_type', 'geometry')} {row['geometry']:.4f} | confidence {row['confidence_loss']:.4f}"
             f" | mean error {_number(row['error_mean'])} voxels | prefix correct {row['prefix_correct_fraction']:.1%}"]
    if 'interval_samples_per_second' in row:
        lines.insert(0, f"  recent speed {row['interval_samples_per_second']:.2f} samples/s"
                        f" | data wait {row['interval_data_seconds']:.2f}s"
                        f" | optimizer {row['interval_update_seconds']:.2f}s per logging interval")
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
             f"  identity data: neighbor coverage {stats.get('foreign_components_fraction', 0.):.0%}"
             +''.join(f" | {key[9:-9]} {value:.0%}" for key, value in stats.items()
                      if key.startswith('location_') and key.endswith('_fraction'))]
    if 'blurred_fraction' in stats:
        lines[-1] += f" | blurred {stats['blurred_fraction']:.0%}"
    return lines


def format_training_log(row):
    step = f"Step {row['step']:,}" if 'step' in row else 'Training'
    if row.get('event') == 'resume_configuration':
        options = row.get('training_options', {})
        return (f"{step} | resumed {row['checkpoint']}\n"
                f"  commit {options.get('n_commit', '?')} | historical slabs: 8 slots, minimum spacing 32 vox"
                f" | batch {options.get('batch', '?')} / grad steps {options.get('grad_steps', '?')}"
                f"\n  live CT/path slabs | causal survival confidence")
    if row.get('event') == 'identity_sampling':
        bank = row.get('negative_bank_provenance') or {}
        return (f"{step} | identity sampling"
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
