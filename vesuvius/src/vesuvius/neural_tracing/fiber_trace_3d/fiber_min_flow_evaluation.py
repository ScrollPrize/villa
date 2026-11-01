"""Seam-aware evaluation of integer winding solutions against explicit references."""
import numpy as np
from .fiber_min_flow import integer_targets


def reference_offsets(reference):
    """Reject conflicting exclusions and fractional labels instead of truncating."""
    offsets = {str(k): v for k, v in reference['fiber_offsets'].items()}
    if set(offsets).intersection(map(str, reference.get('excluded_source_ids', []))):
        raise ValueError('Excluded reference IDs must be omitted from fiber_offsets')
    if any(not np.isfinite(v) or v != int(v) for v in offsets.values()):
        raise ValueError('Reference offsets must be finite integers')
    return offsets


def summarize_errors(error, parallel):
    error, parallel = np.asarray(error), np.asarray(parallel, bool)
    def part(mask):
        v = abs(error[mask])
        return dict(events=int(len(v)), l1=float(v.mean()) if len(v) else None,
            correct_percent=float(100*np.mean(v == 0)) if len(v) else None,
            max_error=int(v.max()) if len(v) else None)
    return dict(all=part(np.ones(len(error), bool)), perpendicular=part(~parallel), parallel=part(parallel))


def small_reference(d, solution, reference, transport):
    offsets = reference_offsets(reference)
    use = np.array([str(d['source_id'][a]) in offsets and str(d['source_id'][b]) in offsets
                    for a, b in zip(d['i'], d['j'])], dtype=bool)
    covered = use.copy()
    same_component = solution['component'][d['i']] == solution['component'][d['j']]
    use &= same_component
    # Reference = fiber offset + original UNWRAPPED angle. Convert that to
    # the same seam chart as our endpoint labels; fractional phase is absent.
    anchor = np.floor(d['node_phase'][:, 0]+.25).astype(int)
    refa = np.array([offsets.get(str(d['source_id'][p]), 0) for p in d['i']])+anchor[d['i']]+transport['fiber_a']
    refb = np.array([offsets.get(str(d['source_id'][p]), 0) for p in d['j']])+anchor[d['j']]+transport['fiber_b']
    local = solution['winding'][d['j']]-solution['winding'][d['i']]+transport['fiber_b']-transport['fiber_a']
    result = summarize_errors((local-(refb-refa))[use], d['parallel'][use])
    result.update(excluded_reference_source_ids=reference.get('excluded_source_ids', []),
                  scored_events=int(use.sum()), unscored_events=int((~use).sum()),
                  unidentifiable_events=int(np.sum(covered & ~same_component)),
                  missing_reference_events=int(np.sum(~covered)))
    return result


def cross_events(document):
    """Accept generated event documents and historical bare lists."""
    events = document.get('events') if isinstance(document, dict) else document
    if not isinstance(events, list) or any(not isinstance(e, dict) for e in events):
        raise ValueError('Cross events must be a list of records or an object containing an events list')
    return events


def large_reference(d, solution, paths, reference, events, gt_data, gt_paths, details=None, pieces=None):
    events = cross_events(events)
    gd, gp = gt_data, gt_paths
    offsets = reference_offsets(reference)
    picks = {str(s): i for i, s in enumerate(d['source_id'])}
    required, par, components, orders, signs, margins = [], [], [], [], [], []
    skipped = 0
    scored = []
    for e in events:
        piece = e.get('gt_piece')
        if (isinstance(piece, bool) or not isinstance(piece, (int, np.integer))
                or not 0 <= piece < len(gd['source_id'])):
            raise ValueError(f'Invalid GT piece index: {piece!r}')
        if str(gd['source_id'][piece]) != e['gt_source']:
            raise ValueError(f'GT piece/source mismatch: row {piece} is '
                             f"{gd['source_id'][piece]}, event declares {e['gt_source']}")
        if e['gt_source'] not in offsets or e['test_source'] not in picks:
            skipped += 1
            continue
        m, valid = integer_targets(abs(e['signed_distance']), np.sign(e['signed_distance']),
            e['family_delta'], e['parallel'])
        if not valid:
            skipped += 1
            continue
        p = (picks[e['test_source']] if pieces is None else
             pieces.locate(e['test_source'], e['test_arc']))
        x, t = paths.at(e['test_source'], e['test_arc'])
        y, gt_transport = gp.at(e['gt_source'], e['gt_arc'])
        c = paths.seam.segment(x, y)
        truth = offsets[e['gt_source']]+int(np.floor(gd['node_phase'][e['gt_piece'], 0]+.25))+gt_transport
        delta = truth-(solution['winding'][p]+t)-c
        required.append(int(delta-m))
        par.append(e['parallel'])
        components.append(int(solution['component'][p]))
        signs.append(np.sign(e['signed_distance']))
        orders.append(delta)
        margins.append(int(not e['parallel'] and np.sign(e['signed_distance'])*e['family_delta'] < 0))
        if details is not None:
            scored.append(dict(e, connector_crossings=int(c), fiber_crossings=int(t),
                gt_fiber_crossings=int(gt_transport), measured_integer_target=int(m)))
    required = np.asarray(required)
    components = np.asarray(components)
    # Each disconnected component has an unidentifiable additive integer gauge.
    # Align separately for diagnostic scoring, explicitly reporting each gauge.
    aligned = required.copy()
    alignment = {}
    for component in np.unique(components):
        mask = components == component
        shift = int(np.floor(np.median(required[mask])))
        aligned[mask] -= shift
        alignment[str(component)] = dict(offset=shift, events=int(mask.sum()))
    result = summarize_errors(aligned, par)
    result.update(available_events=len(events), scored_events=len(required), unscored_events=skipped,
        component_gauges=alignment, gauge_policy='per-component integer median; GT used only for evaluation',
        excluded_reference_source_ids=reference.get('excluded_source_ids', []))
    global_shift = int(np.floor(np.median(required))) if len(required) else 0
    result['single_global_gauge'] = dict(offset=global_shift, metrics=summarize_errors(required-global_shift, par))
    per_shift = np.array([alignment[str(c)]['offset'] for c in components])
    wrong = np.asarray(signs)*(np.asarray(orders)-per_shift) < np.asarray(margins)
    result['signs'] = sign_counts(
        np.asarray(par, bool), np.asarray(signs), np.asarray(orders)-per_shift, np.asarray(margins))
    result['single_global_gauge']['signs'] = sign_counts(
        np.asarray(par, bool), np.asarray(signs), np.asarray(orders)-global_shift, np.asarray(margins))
    if details is not None:
        for index, e in enumerate(scored):
            delta = int(orders[index]-per_shift[index])
            deficit = int(margins[index]-signs[index]*delta)
            details.append(dict(e, component=int(components[index]), gauge=int(per_shift[index]),
                predicted_gt_minus_test=delta, required_signed_minimum=int(margins[index]),
                sign_violation=bool(wrong[index]), violation_steps=max(0, deficit),
                integer_residual=int(aligned[index]),
                global_gauge_sign_violation=bool(signs[index]*(orders[index]-global_shift) < margins[index])))
    return result


def sign_counts(parallel, signs, differences, margins):
    """Count known-sign violations separately for the two constraint families."""
    parallel = np.asarray(parallel, bool)
    known = np.asarray(signs) != 0
    wrong = known & (np.asarray(signs)*differences < margins)
    return {name: dict(events=int(np.sum(mask & known)), violations=int(np.sum(mask & wrong)))
            for name, mask in [('parallel', parallel), ('perpendicular', ~parallel)]}


def boundary_diagnostics(pieces, solution):
    """Report jumps only where both pieces share an identifiable gauge."""
    links = np.flatnonzero(np.diff(pieces.parent) == 0)
    same = solution['component'][links] == solution['component'][links+1]
    jumps = solution['winding'][links+1]-solution['winding'][links]
    return dict(boundaries=len(links), unidentifiable_boundaries=int((~same).sum()),
                identifiable_nonzero_boundary_jumps=int(np.sum((jumps != 0) & same)),
                identifiable_max_jump=int(np.max(abs(jumps[same]), initial=0)))


def compare_solution(current, baseline):
    """Compare matching topology, allowing an additive gauge per component."""
    keys = ['source_id', 'a', 'b', 'target', 'weight', 'one_sided']
    if 'parent' in current or 'parent' in baseline:
        keys += ['parent', 'start', 'end']
    mismatches = [k for k in keys if k not in current or k not in baseline
                  or not np.array_equal(current[k], baseline[k])]
    if mismatches:
        return dict(comparable=False, mismatched_arrays=mismatches)
    # Component IDs may be permuted, but their partition must agree.
    a, b = current['component'], baseline['component']
    pairs = np.unique(np.column_stack([a, b]), axis=0)
    if len(pairs) != len(np.unique(a)) or len(pairs) != len(np.unique(b)):
        return dict(comparable=False, mismatched_arrays=['component_partition'])
    delta = current['winding']-baseline['winding']
    aligned = delta.copy()
    for c in np.unique(a):
        mask = a == c
        aligned[mask] -= int(np.floor(np.median(delta[mask])))
    return dict(comparable=True, raw_label_changes=int(np.count_nonzero(delta)),
                aligned_label_changes=int(np.count_nonzero(aligned)),
                max_aligned_change=int(np.max(abs(aligned), initial=0)))
