"""Validate real loop/new-fiber transactions and exercise provisional rollback.

Uses a certified lower bound from disjoint saved contact intervals; no support
is invented for tracks. Every transaction is independent against the frozen
baseline. Production workspaces and ordinary-growth state are not mutated.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import time

import numpy as np

from audit_frontier_patches import fresh_patch_constraints,validate_fiber_target
from audit_track_gap_bridges import load_frame, track_window, prepare_window, closest_per_object
from audit_track_frontier import natural_bracket
from audit_winding_coincidence import FrozenWorkspaceCheck
from grow_winding_region import Evidence,PatchFields,audit_accepted_topology
from spiral_helpers import load_patch_payload_chunk
from tifxyz import patch_from_payload
from winding_coincidence import CoincidenceGuard,patch_surface
from winding_provisional import ProvisionalRegion,REQUIRED_CHECKS
from grow_track_graph import PackedTracks
from track_fiber_anchors import PhysicalAnchorMesh


class FrozenTrialGuard(CoincidenceGuard):
    """Original geometry guard, with lazy immutable baseline comparisons."""
    def __init__(self,fields,nodes,frozen):
        super().__init__(fields,nodes,3.,'normal_grid',3,10.)
        self.frozen=frozen

    def check(self,node,offset):
        self.pending=None
        if node not in self.patch_nodes:return None
        pid,label=self.patch_nodes[node]
        surface=patch_surface(self.fields,pid,label,'normal_grid',3,10.)
        check=self.frozen.check(surface,offset,tolerance=self.tolerance,node=node)
        if check['decision']!='pass':return check
        return self.check_surface(node,offset,surface)


def constraint(a,b,delta,detail):
    return Evidence(a,b,int(delta),(),(),(),detail)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/winding-normal-neighborhood-10vox'))
    p.add_argument('--loops',type=Path,default=Path('outputs/track-frontier/fiber-track-patch-triangles'))
    p.add_argument('--catalog',type=Path,default=Path('outputs/track-frontier/patch-catalog.json'))
    p.add_argument('--selection',type=Path,default=Path('outputs/track-frontier/triangle-selection.json'))
    p.add_argument('--output',type=Path,default=Path('outputs/track-frontier/transactions'))
    p.add_argument('--allow-incomplete-loops',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    complete=(args.loops/'summary.json').exists()
    if not complete and not args.allow_incomplete_loops:raise ValueError('Loop evaluation incomplete')
    witnesses=json.loads((args.loops/'witnesses.json').read_text())
    # One shortest completed loop per patch/fiber hypothesis, fixed tie-breaks.
    chosen={}
    for row in sorted(witnesses,key=lambda r:(r['length'],r['track'],r['case_index'],r['node'])):
        for target in row['fiber_targets']:chosen.setdefault((row['node'],target['node']),row)
    saved=json.loads((args.root/'pilot.json').read_text());raw=json.loads((args.root/'config.json').read_text())
    catalog=json.loads(args.catalog.read_text());incident=defaultdict(list)
    for edge in saved['evidence']:
        incident[edge['a']].append(edge)
        if edge['a']!=edge['b']:incident[edge['b']].append(edge)
    edges=[constraint(e['a'],e['b'],e['delta'],dict(saved_edge=e['id'])) for e in saved['evidence']]
    pids=sorted({n[6:].rsplit(':',1)[0] for n,f in chosen});zrange=tuple(map(int,raw['z_range'].split(',')))
    loaded=load_patch_payload_chunk(raw['patches_dir'],pids,*zrange,saved['patch_preprocessing']['erode_cells_default'],io_threads=4,keep_largest_component=True)
    patches={pid:patch_from_payload(payload) for pid,payload,error,drop in loaded if payload is not None}
    if len(patches)!=len(pids):raise ValueError('Incomplete original patch loading')
    frame=load_frame(args.root/'config.json','cpu');fields=PatchFields(patches,frame,zrange)
    nodes={n:(n[6:].rsplit(':',1)[0],int(n.rsplit(':',1)[1])) for n,f in chosen}
    guard=FrozenTrialGuard(fields,nodes,FrozenWorkspaceCheck(args.root/'pilot.workspace.json'))
    mesh=PhysicalAnchorMesh(args.root/'pilot.workspace.json')
    tracks=PackedTracks('/home/sean/Desktop/spiral_dataset/tracks/2um_ds2_ps256_surf_v2.dbm.vctracks')
    selection=json.loads(args.selection.read_text())['pairs'];anchor_cache={}
    overlap_cache={};results=[]
    for (node,fiber),row in sorted(chosen.items()):
        result=dict(patch=node,new_fiber=fiber,witness=row['artifact'],track=row['track'],decision='reject')
        check=validate_fiber_target(fiber,{node:row['patch_offset']},saved,incident)
        result['fiber_check']=check
        if check['reason']!='validated_fiber_target':result['reason']=check['reason']
        else:
            if node not in overlap_cache:
                overlap_cache[node]=fresh_patch_constraints(node,row['patch_offset'],patches,fields,frame,saved,raw,catalog)
            overlaps=overlap_cache[node]
            if overlaps['conflicts']:result['reason']='fresh_overlap_theta_conflict'
            else:
                trial=ProvisionalRegion(saved['seed'],saved['offsets'],edges,guard)
                track=f'track:{row["track"]}:{row["case_index"]}'
                anchor=row['fiber'];to=row['track_offset'];po=row['patch_offset']
                cache_key=(row['track'],row['case_index'])
                if cache_key not in anchor_cache:
                    a,b,ia,ib=selection[row['case_index']]['key']
                    local=ia if row['track']==a else ib if row['track']==b else None
                    if local is None:raise ValueError('Loop selection provenance mismatch')
                    window=track_window(tracks,row['track'],local,200.)
                    if window['first_local']!=row['first_local']:raise ValueError('Loop probe extent mismatch')
                    window=prepare_window(window,mesh,frame,4.,evaluate_reference=False)
                    bracket=natural_bracket(window,mesh);h=window['hits']
                    is_fiber=np.array([mesh.sources[o].get('fiber',False) for o in mesh.objects[h['face']]],bool)
                    ids=closest_per_object(h,mesh.objects,is_fiber&(h['phase_error']<.05))
                    modes=set(bracket['offsets'])|set(h['offset'][ids].tolist())
                    if modes!={to}:raise ValueError('Recomputed track theta differs from saved witness')
                    providers=set(bracket['anchors'])|{mesh.sources[mesh.objects[h['face'][i]]]['id'] for i in ids}
                    anchor_cache[cache_key]=sorted(providers)
                new=[constraint(provider,track,to-saved['offsets'][provider],dict(kind='track_anchor',witness=row['artifact']))
                     for provider in anchor_cache[cache_key]]
                new.append(constraint(track,node,po-to,dict(kind='track_patch',witness=row['artifact'])))
                result['track_anchor_providers']=anchor_cache[cache_key]
                new.extend(constraint(e['a'],e['b'],e['delta'],dict(kind='fresh_overlap')) for e in overlaps['eligible_edges'])
                loop_checks={k:True for k in ('fiber_track_contact','fiber_patch_contact','track_patch_contact',
                                             'separated_contacts','theta_transport','incident_complete')}
                admitted=trial.admit_fiber_track_patch(track,to,node,po,anchor,loop_checks,new)
                result['loop_admission']=admitted
                if not admitted['accepted']:result['reason']=admitted['reason']
                else:
                    parents={e['b'] for e in incident[fiber] if e['kind']=='contact' and e['b'] in trial.offsets}
                    admitted=trial.admit(fiber,check['offsets'][0],'fiber',parents,
                                         {**{k:True for k in REQUIRED_CHECKS['fiber']},'certified_support':True})
                    result['fiber_admission']=admitted
                    if not admitted['accepted']:result['reason']=admitted['reason']
                    else:
                        result['topology']=audit_accepted_topology(saved['seed'],trial.offsets,trial.edges)
                        result['added_offsets']={n:trial.offsets[n] for n in (track,node,fiber)}
                        # Deliberate rollback exercise, NOT evidence that this
                        # real patch has actually failed a physical check.
                        withdrawn=trial.invalidate({node},dict(reason='synthetic_rollback_test'))
                        assert set(withdrawn)=={track,node,fiber}
                        assert trial.offsets==saved['offsets'] and not trial.guard.surfaces
                        result.update(decision='validated_trial',reason='new_fiber_reached',
                                      rollback=dict(synthetic_test=True,withdrawn=withdrawn,baseline_restored=True))
        results.append(result)
        print(f'{len(results)}/{len(chosen)}: {result["reason"]}; {time.monotonic()-started:.1f}s',flush=True)
        (args.output/'transactions.json').write_text(json.dumps(results,indent=2)+'\n')
    valid=[r for r in results if r['decision']=='validated_trial']
    summary=dict(input_loops_complete=complete,input_witnesses=len(witnesses),patch_fiber_hypotheses=len(chosen),
        reasons=dict(Counter(r['reason'] for r in results)),validated_transactions=len(valid),
        distinct_new_fibers=sorted({r['new_fiber'] for r in valid}),distinct_patches=sorted({r['patch'] for r in valid}),
        all_trial_rollbacks_restore_baseline=all(r['rollback']['baseline_restored'] for r in valid),
        raw_ct_used=False,additional_field_checks=False,cropped_patches=0,production_graph_modified=False,
        elapsed_seconds=time.monotonic()-started,
        limitations=['Each transaction is a separate trial against the same baseline; counts do not imply simultaneous compatibility.',
            'No ordinary-growth continuation is run here; this verifies reaching new eligible fibers.',
            'Saved contact intervals give a conservative support lower bound without double counting.',
            'Topology and geometry consistency are not a physical-sheet guarantee.'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
