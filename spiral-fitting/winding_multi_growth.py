"""Independent fiber-seeded growth with certified, transactional region merges.

Disconnected regions keep separate integer gauges and separate workspaces.
Evidence IDs belong to one append-only catalog shared by all seed runs.
"""
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path


def merge_regions(first, second, edges, incident):
    """Return a certified union or a reason; never mutate either input."""
    from grow_winding_region import audit_accepted_topology, hypotheses
    a, b = first['offsets'], second['offsets']
    shifts = {a[n]-b[n] for n in a.keys() & b.keys()}
    shared = bool(shifts)
    supported = shared
    for node in sorted(b):
        if node in a:
            continue
        proposals = hypotheses(node, a, edges, incident)
        for proposal in proposals:
            shifts.add(proposal['offset']-b[node])
            supported |= proposal['support'] >= 1.
    if not shifts:
        return None, dict(reason='no_connection')
    if len(shifts) != 1:
        return None, dict(reason='incompatible_winding_shifts', shifts=sorted(shifts))
    if not supported:
        return None, dict(reason='insufficient_support')
    shift = next(iter(shifts))
    offsets = dict(a)
    offsets.update({n: k+shift for n, k in b.items()})
    # Includes every eligible edge within the union, not just chosen bridges.
    topology = audit_accepted_topology(first['seed'], offsets, edges)
    guard = first['guard'].fork()
    guard.patch_nodes = {**first['guard'].patch_nodes, **second['guard'].patch_nodes}
    for node in sorted(b.keys()-a.keys()):
        if node not in second['guard'].surfaces:
            continue  # Fiber components carry graph constraints, not surface area.
        surface, old_offset = second['guard'].surfaces[node]
        assert old_offset == b[node]
        conflict = guard.check_surface(node, offsets[node], surface)
        if conflict:
            return None, dict(reason='geometric_conflict', conflict=conflict)
        guard.accept(node, offsets[node])
    return dict(seed=first['seed'], offsets=offsets, guard=guard, topology=topology,
                runs=first['runs']+second['runs']), dict(reason='merged', shift=shift,
                                                     shared_nodes=len(a.keys() & b.keys()))


def grow_all_components(components, initial_edges, cfg, expand, guard_factory, progress=None):
    """Seed each consistent component unless an earlier growth already reached it.

    Expensive incident discovery is shared across runs, while acceptance and
    coincidence decisions are independent. Once all seeds have been covered,
    repeatedly merge regions whose full topology and geometry permit it.
    """
    from grow_winding_region import grow, audit_accepted_topology
    seeds = sorted((c for c in components if not c['conflicts']),
                   key=lambda c: (-c['length'], c['key']))
    edges, regions, runs, covered = list(initial_edges), [], [], set()
    prepared = set()

    def cached_expand(node):
        if node in prepared:
            return []
        result = list(expand(node)) if expand else []
        prepared.add(node)
        return result

    for component in seeds:
        seed = component['key']
        if seed in covered:
            continue
        run_id = len(runs)
        print(f'Independent region {run_id}: seed {seed}; '
              f'{len(covered & {c["key"] for c in seeds})}/{len(seeds)} fiber components covered', flush=True)
        guard = guard_factory()
        result, edges = grow(seed, edges, cfg, cached_expand, progress, guard)
        covered.update(result['offsets'])
        # Pending includes the entire shared catalog, mostly unrelated to this
        # seed. Keep counts rather than duplicating that catalog for every run.
        from collections import Counter
        result['pending_counts'] = dict(Counter(p['reason'] for p in result.pop('pending')))
        runs.append(result)
        regions.append(dict(seed=seed, offsets=result['offsets'], guard=guard,
                            topology=result['topology'], runs=[run_id]))
        print(f'Independent region {run_id} finished: {len(result["offsets"])} nodes', flush=True)

    incident = defaultdict(list)
    for eid, edge in enumerate(edges):
        incident[edge.a].append(eid)
        if edge.b != edge.a:
            incident[edge.b].append(eid)
    merges = []
    # Earlier (larger-seeded) regions keep their gauge. Restart after each
    # successful union because its additional constraints affect later merges.
    while True:
        changed = False
        failures = []
        owners = defaultdict(set)
        for i, region in enumerate(regions):
            for node in region['offsets']:
                owners[node].add(i)
        candidates = set()
        for ids in owners.values():
            for i in ids:
                candidates.update((i, j) for j in ids if i < j)
        for edge in edges:
            for i in owners.get(edge.a, ()):
                for j in owners.get(edge.b, ()):
                    if i != j:
                        candidates.add(tuple(sorted((i, j))))
        for i, j in sorted(candidates):
            union, outcome = merge_regions(regions[i], regions[j], edges, incident)
            record = dict(first_seed=regions[i]['seed'], second_seed=regions[j]['seed'], **outcome)
            if union is None:
                failures.append(record)
                continue
            print(f'Merged regions {i} and {j}: shift {outcome["shift"]}, '
                  f'{len(union["offsets"])} nodes', flush=True)
            merges.append(record)
            regions[i] = union
            del regions[j]
            changed = True
            break
        if not changed:
            break
    for region in regions:
        region['topology'] = audit_accepted_topology(region['seed'], region['offsets'], edges)
    return regions, edges, runs, dict(accepted=merges, remaining_connections=failures)


def publish_regions(output, regions, edges, runs, merges, members, fields, frame,
                    raw, cfg, components, excluded_links, linking, preprocessing,
                    candidate_counts, config_path):
    """Publish one VC3D workspace per connected region plus a collection index."""
    import math
    import copy
    import find_inconsistent_windings as fiw
    from grow_winding_region import save_fields
    from winding_workspace import export_workspace
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve solved placements before any workspace export, so an export
    # failure cannot discard the expensive independent growth and merge work.
    state_path = output.with_suffix('.regions.json')
    state_path.write_text(json.dumps(dict(regions=[
        {**{k: v for k, v in region.items() if k != 'guard'},
         'geometric_consistency': region['guard'].report()} for region in regions],
        merges=merges), default=fiw._to_py)+'\n')
    # Store the shared evidence once. Per-workspace reports contain only the
    # accepted edges that the straightening solver needs.
    evidence = [dict(id=i, a=e.a, b=e.b, delta=e.delta,
                     support=math.fsum(e.weights), **e.detail) for i, e in enumerate(edges)]
    evidence_path = output.with_suffix('.evidence.json')
    evidence_path.write_text(json.dumps(evidence, default=fiw._to_py)+'\n')
    run_path = output.with_suffix('.runs.json')
    run_path.write_text(json.dumps(runs, default=fiw._to_py)+'\n')
    entries = []
    for i, region in enumerate(regions):
        path = output.with_name(f'{output.stem}.region-{i:03d}.json')
        offsets = region['offsets']
        # Export scans patch fields; restrict it to this region rather than
        # rescanning every discovered patch for each disconnected workspace.
        selected_fields = copy.copy(fields)
        patch_ids = {n.removeprefix('patch:').rsplit(':', 1)[0]
                     for n in offsets if n.startswith('patch:')}
        selected_fields.fields = {pid: field for pid, field in fields.fields.items() if pid in patch_ids}
        accepted = [dict(e, status='consistent', residual=0) for e in evidence
                    if e['a'] in offsets and e['b'] in offsets]
        fields_path = path.with_suffix('.potentials.npz')
        counts = dict(accepted_patches=sum(n.startswith('patch:') for n in offsets),
                      accepted_fiber_components=sum(n.startswith('fiber:') for n in offsets))
        report = dict(schema_version=4, seed=region['seed'], offsets=offsets,
            acceptance_rule=('all_eligible_connections_and_local_normal_grid_thickness'
                             if cfg.coincidence_metric == 'normal_grid'
                             else 'all_eligible_connections_and_theta_height_coincidence'),
            topology=region['topology'], geometric_consistency=region['guard'].report(),
            seed_runs=region['runs'], seed_runs_file=str(run_path.resolve()),
            history=[], pending=[], history_scope='see seed_runs in the shared seed_runs_file; each run has its own gauge',
            stopped_at_limit=any(runs[r]['stopped_at_limit'] for r in region['runs']),
            config=str(config_path), settings=asdict(cfg), linking=asdict(linking),
            patch_preprocessing=preprocessing, candidate_counts=dict(candidate_counts),
            theta_frame='checkpoint', gauge='region seed component root has turn count zero; other regions are independent',
            potential_convention='unwrapped theta in turns = theta_fraction + turns; add an unknown integer for absolute winding',
            fiber_components=components, excluded_branch_links=excluded_links,
            fields_file=str(fields_path), fields=save_fields(fields_path, members, selected_fields, offsets),
            evidence=accepted, shared_evidence_file=str(evidence_path.resolve()), counts=counts)
        workspace = export_workspace(path, members, selected_fields, offsets, region['seed'], frame, raw)
        report['workspace_file'] = str(workspace)
        path.write_text(json.dumps(report, indent=2, default=fiw._to_py)+'\n')
        entries.append(dict(region=i, seed=region['seed'], report_file=str(path.resolve()),
                            workspace_file=str(workspace.resolve()), seed_runs=region['runs'], **counts))
        print(f'Published region {i}: {counts}; {workspace}', flush=True)
    unique = set().union(*(set(r['offsets']) for r in regions))
    index = dict(type='winding_growth_collection', schema_version=1, settings=asdict(cfg),
        regions=entries, merges=merges, seed_runs_file=str(run_path.resolve()),
        shared_evidence_file=str(evidence_path.resolve()), independent_growth_runs=len(runs),
        region_state_file=str(state_path.resolve()),
        eligible_fiber_components=sum(not c['conflicts'] for c in components),
        excluded_inconsistent_fiber_components=sum(bool(c['conflicts']) for c in components),
        unique_patches=sum(n.startswith('patch:') for n in unique),
        unique_fiber_components=sum(n.startswith('fiber:') for n in unique),
        stopped_at_limit=any(r['stopped_at_limit'] for r in runs),
        gauge='Each listed region is independent; never overlay their theta origins without a certified connection.')
    output.write_text(json.dumps(index, indent=2, default=fiw._to_py)+'\n')
    print(f'Collection: {output}; {len(regions)} regions, {index["unique_patches"]} unique patches', flush=True)
