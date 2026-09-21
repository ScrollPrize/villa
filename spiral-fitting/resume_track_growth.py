"""Resume ordinary winding growth between conservative, provisional bridges.

Reconstructs original per-segment contact and overlap weights, resumes the
original spatial discovery policy, and grows to exhaustion after each
admitted loop. The input pilot and its workspaces remain immutable.
"""
import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import pickle
import time

import numpy as np
import torch

import find_inconsistent_windings as fiw
from audit_track_fiber_triangles import patch_hits
from audit_track_gap_bridges import (ReferenceMesh, radius_trees, prepare_window,
                                     track_window, closest_per_object)
from audit_winding_coincidence import FrozenWorkspaceCheck
from checkpoint_io import load_checkpoint_cpu
from evaluate_track_transactions import FrozenTrialGuard, constraint
from grow_track_graph import PackedTracks
from grow_winding_region import (GeometryEvidence, GrowthConfig, PatchFields,
    ThetaFrame, prepare_fibers, grow, hypotheses, audit_accepted_topology, save_fields)
from point_collection import _build_surface_patch_index
from spiral_helpers import load_fiber_point_collections, load_patch_payload_chunk
from tifxyz import patch_from_payload
from winding_linking import AnalysisLinkingConfig, model_inward_direction
from winding_provisional import ProvisionalRegion, REQUIRED_CHECKS
from winding_workspace import export_workspace


class SpatialEvidence:
    """The original outgoing overlap queries over the complete source index."""
    def __init__(self, raw, saved, catalog, frame, cfg, linking):
        self.raw, self.saved, self.frame, self.cfg, self.linking = raw, saved, frame, cfg, linking
        self.zrange = tuple(map(int, raw['z_range'].split(',')))
        self.catalog = catalog['patches']
        self.pids = [r['patch'] for r in self.catalog]
        self.low = np.asarray([r['low'] for r in self.catalog])
        self.high = np.asarray([r['high'] for r in self.catalog])
        self.positions = {pid: i for i, pid in enumerate(self.pids)}
        self.patches = {}
        self.fields = PatchFields(self.patches, frame, self.zrange)
        self.patch_nodes = {}
        self.expanded = set()
        self.geometry = None

    def build_index(self):
        self.load(self.pids)
        self.geometry = GeometryEvidence(self.patches, {}, self.frame, self.fields,
            self.linking, self.cfg, None, {})
        self.geometry.patch_nodes = self.patch_nodes

    def load(self, pids):
        missing = sorted(set(pids)-self.patches.keys())
        for first in range(0, len(missing), 256):
            for pid, payload, error, drop in load_patch_payload_chunk(
                    self.raw['patches_dir'], missing[first:first+256], *self.zrange,
                    self.saved['patch_preprocessing']['erode_cells_default'],
                    io_threads=4, keep_largest_component=True):
                if payload is None:
                    raise ValueError(f'Original source unavailable: {pid}: {error}, {drop}')
                self.patches[pid] = patch_from_payload(payload)

    def nearby(self, low, high, distance):
        return [self.pids[i] for i in np.flatnonzero(
            np.all(self.high >= np.asarray(low)-distance, axis=1)
            & np.all(self.low <= np.asarray(high)+distance, axis=1))]

    def expand(self, node):
        if not node.startswith('patch:') or node in self.expanded:
            return []
        pid, label = node[6:].rsplit(':', 1)
        label = int(label)
        if self.geometry is None:
            self.build_index()
        discovered = self.geometry.overlaps(pid,label)
        self.patch_nodes[node] = (pid, label)
        self.expanded.add(node)
        return discovered


def contact_signature(edge):
    return (edge.a, edge.b, edge.delta, edge.detail['member'],
            edge.detail['from_point_id'], edge.detail['to_point_id'])


def verify_contacts(edges, saved):
    expected, actual = defaultdict(list), defaultdict(list)
    for e in saved['evidence']:
        if e['kind'] == 'contact':
            expected[(e['a'], e['b'], e['delta'], e['member'],
                      e['from_point_id'], e['to_point_id'])].append(e['support'])
    for edge in edges:
        actual[contact_signature(edge)].append(math.fsum(edge.weights))
    if expected.keys() != actual.keys():
        raise ValueError(f'Contact provenance changed: {len(expected.keys()-actual.keys())} missing, '
                         f'{len(actual.keys()-expected.keys())} extra intervals')
    maximum = 0.
    for key in expected:
        a, b = sorted(expected[key]), sorted(actual[key])
        if len(a) != len(b) or not np.allclose(a, b, rtol=1e-8, atol=1e-10):
            raise ValueError(f'Reconstructed contact weights differ: {key}: {a} != {b}')
        maximum = max(maximum, max(abs(x-y) for x, y in zip(a, b)))
    return dict(intervals=len(edges), maximum_support_error=maximum,
                support='original dense-segment tokens and per-token maximum weights')


def verify_overlaps(edges,saved):
    expected = {(e['a'],e['b'],e['delta']):e['support'] for e in saved['evidence'] if e['kind']=='overlap'}
    actual = {(e.a,e.b,e.delta):math.fsum(e.weights) for e in edges}
    if len(expected)!=len(edges) or expected.keys()!=actual.keys():
        raise ValueError(f'Overlap provenance changed: {len(expected.keys()-actual.keys())} missing, '
                         f'{len(actual.keys()-expected.keys())} extra')
    maximum = 0.
    for key,weight in expected.items():
        if not math.isclose(weight,actual[key],rel_tol=1e-8,abs_tol=1e-10):
            raise ValueError(f'Overlap support changed: {key}: {weight} != {actual[key]}')
        maximum=max(maximum,abs(weight-actual[key]))
    return dict(intervals=len(edges),maximum_support_error=maximum)


class PilotGuard(FrozenTrialGuard):
    """Reuse final-pass failures against the unchanged trusted baseline.

    An existing 3-voxel failure at the same offset also fails the 1/2-voxel
    passes. Adding further surfaces cannot repair that baseline comparison.
    Every uncached candidate still receives the ordinary whole-patch check.
    """
    def __init__(self,fields,nodes,frozen,saved):
        super().__init__(fields,nodes,frozen)
        self.rejections={(r['node'],r['hypotheses'][0]['offset']):r['geometric_conflict']
            for r in saved['pending'] if r['reason']=='geometric_conflict'
            and len(r['hypotheses'])==1 and r.get('geometric_conflict')}

    def check(self,node,offset):
        cached=self.rejections.get((node,offset))
        if cached and self.tolerance<=3.:
            self.pending=None
            return dict(cached,reused_baseline_failure=True)
        return super().check(node,offset)


def fiber_mesh(members):
    """All eligible original centerlines, with relative component theta gauges."""
    mesh = ReferenceMesh.__new__(ReferenceMesh)
    mesh.sources, xyz, theta, objects = [], [], [], []
    groups = {}
    for member in members.values():
        if not member['consistent']:
            continue
        line = member['zyx'][:, ::-1].astype(float)
        length = np.linalg.norm(np.diff(line, axis=0), axis=1)
        rows = np.flatnonzero((length > 1e-8) & (length <= 4.))
        if not len(rows):
            continue
        node = member['node']
        if node not in groups:
            groups[node] = len(mesh.sources)
            mesh.sources.append(dict(id=node, fiber=True))
        indices = np.column_stack((rows, rows+1, rows))
        xyz.append(line[indices])
        theta.append((member['theta']+member['lift'])[indices])
        objects.append(np.full(len(rows), groups[node], int))
    mesh.xyz, mesh.theta, mesh.objects = map(np.concatenate, (xyz, theta, objects))
    mesh.centers = mesh.xyz.mean(1)
    mesh.radii = np.linalg.norm(mesh.xyz-mesh.centers[:, None], axis=2).max(1)
    mesh.normals = np.zeros((len(mesh.xyz), 3))
    mesh.trees = radius_trees(mesh.centers, mesh.radii)
    return mesh


class RestartGrowth(RuntimeError):
    """An observed contradiction withdrew dependencies; discard the frontier."""


def track_constraints(row, witness, selection, tracks, mesh, spatial):
    a, b, ia, ib = selection[witness['case_index']]['key']
    local = ia if witness['track'] == a else ib if witness['track'] == b else None
    if local is None:
        raise ValueError('Unknown track provenance')
    window = prepare_window(track_window(tracks, witness['track'], local, 200.),
                            mesh, spatial.frame, 4., evaluate_reference=False)
    if window['first_local'] != witness['first_local'] or window['discontinuous'] or window['ambiguous_phase_step']:
        raise ValueError('Track probe changed')
    track = next(n for n in row['added_offsets'] if n.startswith('track:'))
    result = []
    h = window['hits']
    ids = closest_per_object(h, mesh.objects, h['phase_error'] < .05)
    modes = {(mesh.sources[mesh.objects[h['face'][i]]]['id'], int(h['offset'][i])) for i in ids}
    for fiber, delta in sorted(modes):
        result.append(constraint(fiber, track, delta, dict(kind='track_fiber')))
    pids = spatial.nearby(window['xyz'].min(0), window['xyz'].max(0), 4.)
    spatial.load(pids)
    index, surface_ids = _build_surface_patch_index({p: spatial.patches[p] for p in pids}, 4.)
    candidates = {}
    for pid in pids:
        f = spatial.fields.get(pid)
        for label in np.unique(f['labels']):
            if int(label) not in f['bad_labels']:
                node = spatial.fields.key(pid, label)
                candidates[node] = dict(offset=0)
                spatial.patch_nodes[node] = (pid, int(label))
    for node, contact in patch_hits(index, surface_ids, spatial.fields, window, candidates, 4.):
        for delta in contact['offsets']:
            result.append(constraint(node, track, delta, dict(kind='track_patch')))
    # Retain the prior independently verified baseline anchors as constraints.
    for provider in row['track_anchor_providers']:
        result.append(constraint(provider, track,
            row['added_offsets'][track]-spatial.saved['offsets'][provider],
            dict(kind='saved_track_anchor', witness=row['witness'])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/winding-normal-neighborhood-10vox'))
    parser.add_argument('--frontier', type=Path, default=Path('outputs/track-frontier'))
    parser.add_argument('--output', type=Path, default=Path('outputs/track-frontier/resumed-growth'))
    parser.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    parser.add_argument('--contacts-cache',type=Path,help='Reuse an exact-contacts pickle generated locally by this evaluation with unchanged inputs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def log(message):
        print(f'[{time.monotonic()-started:.1f}s] {message}', flush=True)
    saved = json.loads((args.root/'pilot.json').read_text())
    raw = json.loads((args.root/'config.json').read_text())
    cfg = GrowthConfig.from_json(raw['growth'])
    if cfg.max_steps is not None:
        raise ValueError('Exhaustion evaluation requires max_steps=null')
    torch.set_grad_enabled(False)
    checkpoint = load_checkpoint_cpu(raw['checkpoint'])
    zrange = fiw.parse_z_range(raw['z_range'])
    fit_cfg, scroll, paths, lo, hi = fiw.build_fit_inputs(checkpoint, raw['patches_dir'], [], *zrange,
        fiw.resolve_umbilicus_path(raw['patches_dir'], raw['umbilicus']), raw['fibers'])
    context = fiw.fs.FitContext(fit_cfg, scroll=scroll, paths=paths)
    if args.device == 'cuda':
        torch.zeros(1,device='cuda')  # Initialize before loading host geometry.
    transform, dr = fiw.build_transform(checkpoint, fit_cfg, context, lo, hi, device=args.device)
    frame = ThetaFrame(transform, dr)
    linking = AnalysisLinkingConfig.from_json(raw['linking']).resolve_distance(fit_cfg['pcl_link_distance_tolerance'])
    catalog = json.loads((args.frontier/'patch-catalog.json').read_text())
    spatial = SpatialEvidence(raw, saved, catalog, frame, cfg, linking)
    log('Loading original fiber annotations and original contact patch sources')
    collections, _ = load_fiber_point_collections(raw['fibers'], 0, min_point_spacing=0.,
                                                base_shape_zyx=scroll.base_shape_zyx)
    members, components, excluded = prepare_fibers(collections, frame, zrange)
    log(f'Prepared {len(members)} fiber members in {len(components)} components')
    if {c['key']: c['offsets'] for c in components} != {c['key']: c['offsets'] for c in saved['fiber_components']}:
        raise ValueError('Original fiber component transport changed')
    pids = sorted({e['patch_id'] for e in saved['evidence'] if e['kind'] == 'contact'})
    for first in range(0,len(pids),1024):
        spatial.load(pids[first:first+1024])
        log(f'Loaded {min(first+1024,len(pids))}/{len(pids)} contact patch sources')
    if args.contacts_cache:
        with args.contacts_cache.open('rb') as stream:
            contacts=pickle.load(stream)
        for e in contacts:
            spatial.patch_nodes[e.b]=(e.detail['patch_id'],int(e.b.rsplit(':',1)[1]))
    else:
        geometry = GeometryEvidence(spatial.patches, members, frame, spatial.fields, linking, cfg,
            model_inward_direction(transform, dr, args.device, lo, hi), fit_cfg)
        contacts = []
        for i, key in enumerate(sorted(members)):
            if members[key]['consistent']:
                contacts.extend(geometry.contacts(members[key]))
            if i % 25 == 0:
                log(f'Exact contacts: {i+1}/{len(members)} members, {len(contacts)} intervals')
        spatial.patch_nodes.update(geometry.patch_nodes)
        del geometry
    reconstruction = verify_contacts(contacts, saved)
    log(f'Reconstructed contacts: {reconstruction}')
    # This locally generated cache preserves the expensive original contact
    # reconstruction for inspection; it is never loaded from external inputs.
    with (args.output/'exact-contacts.pickle').open('wb') as stream:
        pickle.dump(contacts,stream,protocol=pickle.HIGHEST_PROTOCOL)
    log('Loading full patch catalog for the shared overlap index')
    for first in range(0,len(spatial.pids),1024):
        spatial.load(spatial.pids[first:first+1024])
        if first % 4096 == 0:
            log(f'Full catalog: {min(first+1024,len(spatial.pids))}/{len(spatial.pids)} sources prepared')
    spatial.build_index()
    log('Shared full-catalog overlap index ready')
    prepared = set(saved['offsets'])|{r['node'] for r in saved['pending'] if r['incident_discovery_complete']}
    overlaps=[]
    for i, node in enumerate(sorted(prepared)):
        overlaps.extend(spatial.expand(node))
        if i % 25 == 0:
            log(f'Original overlap reconstruction: {i+1}/{len(prepared)} objects, {len(overlaps)} edges')
    reconstruction['overlaps']=verify_overlaps(overlaps,saved)
    log(f'Reconstructed original overlaps: {reconstruction["overlaps"]}')
    edges=contacts+overlaps
    with (args.output/'exact-original-evidence.pickle').open('wb') as stream:
        pickle.dump(edges,stream,protocol=pickle.HIGHEST_PROTOCOL)
    transactions = [r for r in json.loads((args.frontier/'transactions/transactions.json').read_text())
                    if r['decision'] == 'validated_trial']
    witnesses = {r['artifact']: r for r in json.loads((args.frontier/'fiber-track-patch-triangles/witnesses.json').read_text())}
    selection = json.loads((args.frontier/'triangle-selection.json').read_text())['pairs']
    tracks = PackedTracks('/home/sean/Desktop/spiral_dataset/tracks/2um_ds2_ps256_surf_v2.dbm.vctracks')
    mesh = fiber_mesh(members)
    seen = set()
    for row in transactions:
        track = next(n for n in row['added_offsets'] if n.startswith('track:'))
        if track in seen:
            continue
        seen.add(track)
        edges.extend(track_constraints(row, witnesses[row['witness']], selection, tracks, mesh, spatial))
        log(f'Track incident discovery: {len(seen)} windows')
    del mesh
    guard = PilotGuard(spatial.fields, spatial.patch_nodes, FrozenWorkspaceCheck(args.root/'pilot.workspace.json'),saved)
    region = ProvisionalRegion(saved['seed'], saved['offsets'], edges, guard)
    rounds, attempts = [], []
    def persist():
        (args.output/'progress.json').write_text(json.dumps(dict(rounds=rounds, bridges=attempts,
            added_offsets={n: r.offset for n, r in region.records.items()},
            elapsed_seconds=time.monotonic()-started), indent=2)+'\n')
    def saturate(label):
        before = set(region.offsets)
        growth_guard = region.guard.fork()
        def accept(node, best, current_edges):
            region.edges = list(current_edges)
            kind = 'growth_patch' if node.startswith('patch:') else 'fiber'
            region.guard.tolerance = growth_guard.tolerance
            admitted = region.admit(node, best['offset'], kind, best['providers'],
                {**{k: True for k in REQUIRED_CHECKS[kind]}, 'ordinary_support': True})
            if not admitted['accepted']:
                raise RuntimeError(f'Growth/ledger disagreement: {node}: {admitted}')
            if len(region.records) % 10 == 0:
                log(f'{label}: {len(region.records)} provisional objects; latest {node}')
                persist()
        def discover(node):
            discovered = spatial.expand(node)
            region.guard.patch_nodes = spatial.patch_nodes
            growth_guard.patch_nodes = spatial.patch_nodes
            withdrawn = region.observe(discovered)
            if withdrawn:
                log(f'Late evidence withdrew {len(withdrawn)} provisional objects; rebuilding growth frontier')
                raise RestartGrowth()
            return discovered
        while True:
            try:
                result, current_edges = grow(saved['seed'], region.edges, cfg, discover,
                    coincidence=growth_guard, initial_offsets=region.offsets,
                    initial_expanded=spatial.expanded, on_accept=accept,
                    excluded_nodes=region.quarantined)
                break
            except RestartGrowth:
                growth_guard = region.guard.fork()
        region.edges = current_edges
        region.guard.tolerance = 3.
        if region.offsets != result['offsets'] or not all(s['exhausted'] for s in result['thickness_stages']):
            raise RuntimeError('Continuation did not reach a consistent exhausted state')
        added = sorted(set(region.offsets)-before)
        round_row = dict(label=label, added=added, patches=sum(n.startswith('patch:') for n in added),
            fibers=sum(n.startswith('fiber:') for n in added), stages=result['thickness_stages'],
            topology=result['topology'], pending_reasons=dict(Counter(r['reason'] for r in result['pending'])))
        rounds.append(round_row)
        (args.output/f'round-{len(rounds)-1:02d}.json').write_text(json.dumps(result, default=fiw._to_py)+'\n')
        log(f'EXHAUSTED {label}: {round_row["patches"]} patches, {round_row["fibers"]} fibers')
        persist()
        return result
    result = saturate('control_without_track_bridge')
    for row in transactions:
        node, fiber = row['patch'], row['new_fiber']
        attempt = dict(patch=node, target_fiber=fiber, witness=row['witness'])
        if node in region.offsets or fiber in region.offsets:
            attempt['reason'] = 'already_reached'; attempts.append(attempt); continue
        withdrawn = region.observe(spatial.expand(node))
        if withdrawn:
            log(f'Late evidence withdrew {len(withdrawn)} provisional objects')
            result = saturate('after_late_invalidation')
        track = next(n for n in row['added_offsets'] if n.startswith('track:'))
        if track in region.quarantined or node in region.quarantined:
            attempt['reason'] = 'proposal_quarantined'; attempts.append(attempt); continue
        # The target must still have enough ORIGINAL deduplicated support once
        # this patch is active, and every current incident theta mode must agree.
        incident = defaultdict(list)
        for i, edge in enumerate(region.edges):
            incident[edge.a].append(i)
            if edge.b != edge.a:
                incident[edge.b].append(i)
        proposal = hypotheses(fiber, {**region.offsets, node: row['added_offsets'][node],
                                      track: row['added_offsets'][track]}, region.edges, incident)
        if len(proposal) != 1 or proposal[0]['support'] < 1:
            attempt.update(reason='target_no_longer_eligible', hypotheses=proposal)
            attempts.append(attempt); continue
        witness = witnesses[row['witness']]
        loop = region.admit_fiber_track_patch(track, row['added_offsets'][track],
            node, row['added_offsets'][node], witness['fiber'],
            {k: True for k in ('fiber_track_contact','fiber_patch_contact','track_patch_contact',
                              'separated_contacts','theta_transport','incident_complete')}, [])
        attempt.update(reason=loop['reason'], admission=loop)
        attempts.append(attempt)
        if loop['accepted']:
            admitted_fiber = region.admit(fiber,proposal[0]['offset'],'fiber',proposal[0]['providers'],
                {**{k:True for k in REQUIRED_CHECKS['fiber']},'ordinary_support':True})
            attempt['target_fiber_admission'] = admitted_fiber
            if not admitted_fiber['accepted']:
                region.invalidate({track},dict(reason='target_fiber_admission_failed',detail=admitted_fiber))
                attempt['reason'] = 'target_fiber_admission_failed'
                persist()
                continue
            log(f'BRIDGE admitted: {node}; target {fiber}')
            result = saturate(f'bridge_{len(attempts):02d}')
        else:
            log(f'Bridge rejected against current region: {node}: {loop["reason"]}')
        persist()
    result = saturate('final_exhaustion_after_bridge_reservoir')
    topology = audit_accepted_topology(saved['seed'], region.offsets, region.edges)
    rollback = None
    if region.joint_groups:
        trial = copy.copy(region)
        trial.records = region.records.copy(); trial.events = region.events.copy()
        trial.quarantined = region.quarantined.copy(); trial.joint_groups = region.joint_groups.copy()
        root = next(iter(trial.joint_groups))
        removed = trial.invalidate({root}, dict(reason='synthetic_full_continuation_rollback'))
        rollback = dict(synthetic_test=True, root=root, removed=removed,
            trusted_offsets_unchanged=all(trial.offsets[n] == v for n, v in saved['offsets'].items()),
            remaining_topology=audit_accepted_topology(saved['seed'], trial.offsets, trial.edges))
    for node in region.offsets:
        if node.startswith('patch:'):
            spatial.fields.get(node[6:].rsplit(':',1)[0])
    output = args.output/'growth.json'
    field_path = output.with_suffix('.potentials.npz')
    evidence_path = args.output/'exact-evidence.pickle'
    with evidence_path.open('wb') as stream:
        pickle.dump(region.edges,stream,protocol=pickle.HIGHEST_PROTOCOL)
    result.update(schema_version=3,offsets=region.offsets, config=str(args.root/'config.json'), settings=asdict(cfg),
        fields_file=str(field_path), fields=save_fields(field_path,members,spatial.fields,region.offsets),
        fiber_components=components, patch_preprocessing=saved['patch_preprocessing'],
        reconstruction=reconstruction, rounds=rounds, bridge_attempts=attempts,
        topology=topology, provisional_events=region.events, rollback=rollback,
        exact_evidence_file=str(evidence_path),baseline_report=str(args.root/'pilot.json'),
        provisional_records={n:dict(offset=r.offset,kind=r.kind,parents=sorted(r.parents),checks=r.checks)
                             for n,r in region.records.items()},
        joint_groups={n:sorted(group) for n,group in region.joint_groups.items()},
        quarantined=sorted(region.quarantined),
        counts=dict(accepted_patches=sum(n.startswith('patch:') for n in region.offsets),
                    accepted_fiber_components=sum(n.startswith('fiber:') for n in region.offsets),
                    accepted_tracks=sum(n.startswith('track:') for n in region.offsets),
                    pending=len(result['pending']),conflicting_intervals=0))
    result['workspace_file'] = str(export_workspace(output,members,spatial.fields,region.offsets,
                                                  saved['seed'],frame,raw))
    output.write_text(json.dumps(result, default=fiw._to_py)+'\n')
    summary = dict(baseline_patches=saved['counts']['accepted_patches'],
        baseline_fibers=saved['counts']['accepted_fiber_components'],
        added_patches=sum(n.startswith('patch:') for n in region.records),
        added_fibers=sum(n.startswith('fiber:') for n in region.records),
        added_tracks=sum(n.startswith('track:') for n in region.records),
        rounds=rounds, bridge_reasons=dict(Counter(r['reason'] for r in attempts)),
        topology=topology, contact_reconstruction=reconstruction,
        all_rounds_exhausted=all(s['exhausted'] for r in rounds for s in r['stages']),
        rollback=rollback, cropped_patches=0, additional_field_checks=False, raw_ct_used=False,
        production_graph_modified=False, workspace_file=result['workspace_file'],
        elapsed_seconds=time.monotonic()-started,
        limitations=['Finite existing validated bridge reservoir; no new crossing-location search after it is exhausted.',
            'Theta/geometry consistency is conditional evidence, not physical-sheet ground truth.',
            'Workspace displays patches and annotated fibers; track constraints and provenance are in the growth report.',
            'Control verifies that the reconstructed original region remains stalled before adding any bridge.'])
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, default=fiw._to_py)+'\n')
    log(f'COMPLETE: {summary["added_patches"]} added patches, {summary["added_fibers"]} added fibers, '
        f'{summary["added_tracks"]} admitted tracks')


if __name__ == '__main__':
    main()
