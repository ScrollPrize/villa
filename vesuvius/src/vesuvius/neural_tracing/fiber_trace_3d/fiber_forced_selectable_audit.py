"""Read-only checkpoint audit; writes a separate report, never changes a solve."""
import json
from pathlib import Path
import numpy as np
import torch
from vesuvius.neural_tracing.fiber_trace_3d import fiber_soft_winding as soft
from vesuvius.neural_tracing.fiber_trace_3d import fiber_implicit_winding as density
from vesuvius.neural_tracing.fiber_trace_3d.fiber_bucket_uv import BucketUV
from vesuvius.neural_tracing.fiber_trace_3d.fiber_gt_constraint_score import arrays


def load(root):
    cfg=json.loads((root/'solve/config.json').read_text())
    d=arrays(cfg['tensor_cache']);samples=arrays(cfg['soft_samples'])
    saved=arrays(root/'solve/solution.npz')
    ext=BucketUV(d,cfg,'cpu');ext.prepare_fields(d,samples);ext.restore(saved)
    data,nodes=soft.tensor_inputs(d,samples,cfg,'cpu')
    return cfg,d,saved,ext,data,nodes


def evaluate(bundle,state=None,widths=None):
    cfg,d,saved,ext,data,nodes=bundle
    state=torch.tensor(saved['state']) if state is None else state
    widths=torch.tensor(saved['widths']) if widths is None else widths
    initial=torch.tensor(arrays(cfg['initial_solution'])['widths'])
    with torch.no_grad():
        loss,terms=soft.diagnostic_objective(state,soft.encode_widths(widths,cfg),initial,
            data,nodes,cfg,density.pair_cost,ext)
    return dict(loss=float(loss),terms={k:float(v) for k,v in terms.items()})


def transfer_parent_geometry(source, target, parent, signed_start_offset):
    """Keep the original anchor; translate query arcs, never rebase at MAP.

    This is an evaluation view of the parent solution on the child graph, not
    a new independently anchored child checkpoint. Every integer hypothesis
    and continuous soft-loss query retains the parent's physical geometry.
    """
    if not source.curve.straight or not target.curve.straight:
        raise ValueError('Exact audit currently supports fixed-length curves only')
    parent=torch.as_tensor(parent,dtype=torch.long)
    shift=torch.as_tensor(signed_start_offset)
    with torch.no_grad():
        target.curve.u.copy_(source.curve.u[parent])
        target.curve.v.copy_(source.curve.v[parent])
        target.cavity.values=source.cavity.values[parent].clone()
    # add_plan also rebuilds packed pair plans used by optimized evaluators.
    for key,plan in list(target.curve.plans.items()):
        ids,ds=plan[:2]
        target.curve.add_plan(key,ids.cpu().numpy(),(ds+shift[ids]).cpu().numpy())
    return shift


def run(settings):
    torch.set_num_threads(4)
    base=Path(settings['output_root']);base.mkdir(parents=True,exist_ok=True)
    output={}
    for path in settings['roots']:
        root=Path(path);result_key=root.name
        joint=load(root/'joint');select=load(root/'selectable')
        row={'joint':evaluate(joint),'selectable':evaluate(select)}
        joint[0].pop('force_keep_source_ids',None)
        row['joint_unlocked_same_graph']=evaluate(joint)
        assert abs(row['joint']['loss']-row['joint_unlocked_same_graph']['loss'])<1e-10
        jd,sd=joint[1],select[1];js=joint[2]
        parent=[]
        for source,a,b in zip(sd['source_id'],sd['start'],sd['end']):
            found=np.flatnonzero((jd['source_id']==source)&(jd['start']<=a+1e-7)&(jd['end']>=b-1e-7))
            assert len(found)==1
            parent.append(found[0])
        parent=np.asarray(parent)
        state=torch.tensor(js['state'][parent]);widths=torch.tensor(js['widths'])
        ds=torch.tensor((sd['start']-jd['start'][parent])*np.sign(jd['node_ds'][parent,-1]))
        ids=torch.tensor(parent)
        original_plans=dict(select[3].curve.plans)
        transfer_parent_geometry(joint[3],select[3],parent,ds)
        with torch.no_grad():
            h,lp=density.hypotheses(state,select[3].cavity.values,select[0]['sigma_max'])
            source_h,source_lp=density.hypotheses(torch.tensor(js['state']),joint[3].cavity.values,joint[0]['sigma_max'])
            torch.testing.assert_close(h,source_h[ids],rtol=0,atol=0)
            torch.testing.assert_close(lp,source_lp[ids],rtol=0,atol=0)
            errors={}
            for key,plan in original_plans.items():
                child,arc=plan[:2]
                labels=h[child,:,0]
                expected=joint[3].curve.hypothesis_queries(torch.tensor(js['state']),widths,
                    (ids[child],arc+ds[child],None,None,None),labels)
                actual=select[3].curve.hypothesis_queries(state,widths,select[3].curve.plans[key],labels)
                errors[key]=max(float((a-b).abs().max()) if a.numel() else 0. for a,b in zip(actual,expected))
                for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
            row['transfer_validation']={'max_coordinate_error_by_query':errors,'max_probability_error':float((lp-source_lp[ids]).abs().max())}
        row['joint_transferred_exact_selectable_graph']=evaluate(select,state,widths)
        row['cost_gap_direct_minus_transfer']=(row['selectable']['loss']-
            row['joint_transferred_exact_selectable_graph']['loss'])
        row['sigma_policy']='Each candidate retains its complete saved distribution: transferred children inherit parent sigma. No uncertainty update.'
        for name,b in [('joint',joint),('selectable',select)]:
            gt=json.loads((root/name/'dataset.json').read_text())['gt_source_id_map'].values()
            mask=np.isin(b[1]['source_id'],list(gt))
            row[name]['graph']={'pieces':len(mask),'GT_pieces':int(mask.sum()),'continuations':len(b[1]['links']),
                'constraints':len(b[1]['i'])}
            row[name]['GT_by_source']=[dict(source=int(src),pieces=int((b[1]['source_id']==src).sum()),
                retained_vx=float(b[1]['lengths'][(b[1]['source_id']==src)&(b[2]['state'][:,3]>=0)].sum()))
                for src in sorted(set(b[1]['source_id'][mask]))]
        output[result_key]=row
        print(result_key,json.dumps(row),flush=True)
    (base/'forced_selectable_objective_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    return output
