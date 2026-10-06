"""Terminal formatting for follower training; JSON remains the analysis format."""
import json


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


class TrainingInterval:
    """Pool counts and weight decision-normalized means by supervised decisions."""
    means = ('loss', 'geometry', 'confidence_loss', 'refinement_attempts_mean')
    counts = tuple(p+'_'+s for p in ('ct_frame',)
                   for s in ('count', 'transported', 'deterministic', 'learned', 'energy_sum', 'gap_sum')) + (
              'live_depth_sum', 'live_travelled_sum', 'live_rows', 'live_terminal_rows',
              'ct_frame_rejected_batches', 'error_sum', 'geometry_count', 'point_correct_count', 'point_wrong_count',
              'point_unknown_count', 'supervised_states', 'observation_only_states',
              'confidence_labeled_states', 'confidence_terminal_states', 'confidence_recoverable_states',
              'connector_rejected_targets', 'refinement_attempts_sum')

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
        if 'grad_norm' in row:
            self.values['grad_norm_max'] = max(self.values.get('grad_norm_max', 0.), row['grad_norm'])
            self.values['clipped_updates'] = self.values.get('clipped_updates', 0)+int(row.get('grad_clip_scale', 1.) < 1.)

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
    lines.append(f"  supervision: {int(m['decisions'])} decisions / {int(m['crops'])} observations")
    frames = []
    for prefix, label in (('ct_frame', 'current'),):
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
        lines.append(f"  live chains: {int(m['live_rows'])} rows | mean depth {m['live_depth_sum']/m['live_rows']:.0f} decisions"
                     f" | mean travel {m.get('live_travelled_sum', 0)/m['live_rows']:.0f} vox"
                     f" | {int(m.get('live_terminal_rows', 0))} terminal")
    lines.extend(_sampling_table(row))
    if 'grad_norm_max' in m:
        lines.append(f"  gradients: max {m['grad_norm_max']:.2g}, clipped {m['clipped_updates']}/{updates} updates")
    return lines


def _sampling_table(row):
    """One row per source: task mix (DAgger requested -> delivered), replay supply, label mix, confidence targets and
    live-chain outcomes. Histograms (startup/seed ages, replay travel, chain limits) stay in the JSON log."""
    sampling = row.get('sampling', {})
    if not sampling:
        return []
    names = row.get('dataset_names', [])
    pct = lambda n, d: f'{n/d:.0%}' if d else '--'
    lines = ['  sampling (dagger = replayed rollout states; fallback = nothing to replay -> fresh; age = steps since the model that made the data)',
             f"    {'source':<20}{'rows':>5}{'fresh':>7}{'live':>6}{'dagger req>got':>16}{'fallback':>10}"
             f"{'replay ep':>11}{'age':>6}   {'labels follow/recov/term/unk':<30}{'hazard tgts':>12}"
             "   live adv/cens/stale/empty"]
    for source, entry in sorted(sampling.items(), key=lambda item: int(item[0])):
        rows = entry['rows']
        name = names[int(source)] if int(source) < len(names) else source
        req, got = entry['requested_share'], entry['delivered_share']
        dagger = lambda shares: sum(v for k, v in shares.items() if k.startswith('dagger'))
        fallback = sum(entry['fallbacks'].values())
        sup = entry['supervision']; labeled = sum(sup.values())
        labels = '/'.join(pct(sup.get(k, 0), labeled) for k in ('following', 'recoverable', 'terminal', 'unknown'))
        targets = entry['positive_targets']+entry['negative_targets']
        c = entry.get('counters', {})
        live = '/'.join(str(c.get('live_'+k, 0)) for k in ('advanced', 'censored', 'stale', 'empty'))
        lines.append(f"    {name[:19]:<20}{rows:>5}{got.get('fresh', 0):>7.0%}{got.get('live', 0):>6.0%}"
                     f"{f'{dagger(req):.0%}>{dagger(got):.0%}':>16}{pct(fallback, rows):>10}"
                     f"{entry['episodes']:>11}{_number(entry['source_age_mean'], '.0f'):>6}   {labels:<30}"
                     f"{pct(entry['negative_targets'], targets):>12}   {live}")
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


def format_training_log(row):
    step = f"Step {row['step']:,}" if 'step' in row else 'Training'
    if row.get('event') == 'resume_configuration':
        options = row.get('run_config', {}).get('training', {})
        return (f"{step} | resumed {row['checkpoint']}\n"
                f"  commit {options.get('n_commit', '?')}"
                f" | batch {options.get('batch', '?')} / grad steps {options.get('grad_steps', '?')}"
                f" | causal survival confidence")
    if row.get('event') == 'identity_sampling':
        return f"{step} | identity sampling (settings in the run configuration)"
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
    if 'loss' not in row:
        details = ' | '.join(f'{key}: {json.dumps(value)}' for key,value in row.items() if key != 'step')
        return f'{step} | {details}'
    m = row['interval']
    return '\n'.join([f"\n{step} | last {m['updates']} updates / {m['crops']:,} crops"
                      f" | lr {row['lr']:.2e}", *_interval_training_lines(row)])
