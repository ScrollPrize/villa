"""Measure immediate ordinary fiber-to-patch growth unlocked by trial bridges.

Each follow-on patch is tested independently against baseline + its bridge.
This is a one-step yield measurement, not a saturated production growth run.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import time

from audit_frontier_patches import existing_constraints,disjoint_contact_support,fresh_patch_constraints
from audit_track_gap_bridges import load_frame
from audit_winding_coincidence import FrozenWorkspaceCheck
from evaluate_track_transactions import FrozenTrialGuard
from grow_winding_region import PatchFields
from spiral_helpers import load_patch_payload_chunk
from tifxyz import patch_from_payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/winding-normal-neighborhood-10vox'))
    p.add_argument('--transactions',type=Path,default=Path('outputs/track-frontier/transactions/transactions.json'))
    p.add_argument('--output',type=Path,default=Path('outputs/track-frontier/followon'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    saved=json.loads((args.root/'pilot.json').read_text());raw=json.loads((args.root/'config.json').read_text())
    catalog=json.loads(Path('outputs/track-frontier/patch-catalog.json').read_text())
    selected={}
    for r in json.loads(args.transactions.read_text()):
        if r['decision']=='validated_trial':selected.setdefault(r['new_fiber'],r)
    incident=defaultdict(list)
    for e in saved['evidence']:
        incident[e['a']].append(e)
        if e['a']!=e['b']:incident[e['b']].append(e)
    pending={r['node']:r for r in saved['pending']}
    rows=[];nodes=set()
    for fiber,trial in selected.items():
        active={**saved['offsets'],**trial['added_offsets']};nodes.add(trial['patch'])
        neighbours=sorted({e['b'] for e in incident[fiber] if e['kind']=='contact' and e['b'] not in active})
        for node in neighbours:
            edges=[e for e in incident[node] if e['kind']=='contact' and e['a'] in active]
            offsets={active[e['a']]+e['delta'] for e in edges}
            support=disjoint_contact_support(edges)
            row=dict(fiber=fiber,node=node,support_lower_bound=support['support'],support_edges=support['edges'],decision='reject')
            if len(offsets)!=1:row['reason']='theta_conflict'
            else:
                offset=next(iter(offsets));row['offset']=offset
                if existing_constraints(node,offset,incident,active):row['reason']='theta_conflict'
                elif pending.get(node,{}).get('reason')=='geometric_conflict':row['reason']='existing_geometry_conflict'
                elif support['support']<1.:row['reason']='insufficient_certified_support'
                else:row['reason']='needs_geometry';nodes.add(node)
            rows.append(row)
    pids=sorted({n[6:].rsplit(':',1)[0] for n in nodes});zrange=tuple(map(int,raw['z_range'].split(',')))
    loaded=load_patch_payload_chunk(raw['patches_dir'],pids,*zrange,saved['patch_preprocessing']['erode_cells_default'],io_threads=4,keep_largest_component=True)
    patches={pid:patch_from_payload(payload) for pid,payload,error,drop in loaded if payload is not None}
    if len(patches)!=len(pids):raise ValueError('Follow-on geometry loading incomplete')
    frame=load_frame(args.root/'config.json','cpu');fields=PatchFields(patches,frame,zrange)
    mapping={n:(n[6:].rsplit(':',1)[0],int(n.rsplit(':',1)[1])) for n in nodes}
    frozen=FrozenWorkspaceCheck(args.root/'pilot.workspace.json');guards={}
    for fiber,trial in selected.items():
        guard=FrozenTrialGuard(fields,mapping,frozen);bridge=trial['patch'];offset=trial['added_offsets'][bridge]
        conflict=guard.check(bridge,offset)
        if conflict:raise ValueError(f'Bridge geometry recheck failed: {conflict}')
        guard.accept(bridge,offset);guards[fiber]=guard
    for row in rows:
        if row['reason']!='needs_geometry':continue
        trial=selected[row['fiber']];active={**saved['offsets'],**trial['added_offsets']}
        node=row['node'];pid,label=mapping[node]
        if label in fields.get(pid)['bad_labels']:row['reason']='internal_theta_cycle';continue
        overlaps=fresh_patch_constraints(node,row['offset'],patches,fields,frame,{**saved,'offsets':active},raw,catalog)
        if overlaps['conflicts']:row.update(reason='fresh_overlap_theta_conflict',conflicts=overlaps['conflicts']);continue
        conflict=guards[row['fiber']].fork().check(node,row['offset'])
        if conflict:row.update(reason='geometry_conflict',geometry=conflict)
        else:row.update(decision='eligible_followon',reason='ordinary_contact_theta_geometry_pass')
        print(f'{node}: {row["reason"]}; {time.monotonic()-started:.1f}s',flush=True)
    (args.output/'patches.json').write_text(json.dumps(rows,indent=2)+'\n')
    summary=dict(fibers_evaluated=len(selected),candidate_fiber_patch_pairs=len(rows),
        reasons=dict(Counter(r['reason'] for r in rows)),
        eligible_distinct_patches=sorted({r['node'] for r in rows if r['decision']=='eligible_followon'}),
        per_fiber={f:sum(r['fiber']==f and r['decision']=='eligible_followon' for r in rows) for f in selected},
        raw_ct_used=False,additional_field_checks=False,cropped_patches=0,graph_modified=False,
        elapsed_seconds=time.monotonic()-started,
        limitations=['One deterministic representative bridge per reached fiber.',
            'Each follow-on candidate is independent against baseline plus that bridge; mutual compatibility is not implied.',
            'Uses ordinary saved contact eligibility, theta constraints, fresh overlaps, original geometry, and conservative disjoint-interval support.',
            'This is immediate follow-on yield, not full growth to exhaustion.'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
