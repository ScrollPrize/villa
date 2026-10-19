"""Detailed GT-reference sign audit of the unchanged saved flow solutions."""
from collections import Counter
import csv
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vesuvius/src'))
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_benchmark import arrays,read,write,digest,Paths,large_reference

def run(source=None, output=None):
    source=Path(source) if source is not None else ROOT/'global-fiber-min-flow/data/seven_fractional_v1'
    output=Path(output) if output is not None else ROOT/'global-fiber-min-flow/data/gt_sign_audit_fractional_v1'
    output.mkdir(parents=True,exist_ok=True)
    project=Path('/home/hendrik/villa2/global-fiber-sgd')
    manifest=read(source/'manifest.json');reference=read(project/'gt_winding_reference.json')
    summary={}
    for case in manifest['cases']:
        name=case['name']
        if name not in ('free','joint'):continue
        cfg=case['config'];path=source/name/'solution.npz';before=digest(path)
        assert digest(cfg['tensor_cache'])==case['tensor_sha256']
        d=arrays(cfg['tensor_cache']);s=arrays(path);details=[]
        result=large_reference(project,d,s,Paths(cfg,d),reference,details=details)
        assert result==read(source/name/'metrics.json')['reference']
        bad=[e for e in details if e['sign_violation']]
        columns=['event_id','test_source','gt_source','test_arc','gt_arc','parallel','signed_distance',
            'family_delta','normal_cosine','predicted_gt_minus_test','required_signed_minimum',
            'violation_steps','integer_residual','component','gauge','connector_crossings','fiber_crossings',
            'gt_fiber_crossings','global_gauge_sign_violation']
        with (output/(name+'_gt_violations.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(bad)
        write(output/(name+'_gt_violations.json'),bad)
        result['violation_detail']=dict(events=len(bad),ordinary_fibers=len(set(e['test_source'] for e in bad)),
            gt_fibers=dict(Counter(e['gt_source'] for e in bad)),
            steps=dict(Counter(e['violation_steps'] for e in bad)),
            reversals=sum(np.sign(e['signed_distance'])*e['predicted_gt_minus_test']<0 for e in bad),
            missing_next_winding=sum(e['predicted_gt_minus_test']==0 for e in bad),
            ordinary_fiber_counts=dict(Counter(e['test_source'] for e in bad)),
            normal_abs_cosine_min=min((abs(e['normal_cosine']) for e in bad),default=None),
            normal_abs_cosine_max=max((abs(e['normal_cosine']) for e in bad),default=None))
        result['violation_detail']['reversals']=int(result['violation_detail']['reversals'])
        result['single_global_gauge']['signs']={kind:dict(events=sum(e['parallel']==par for e in details),
            violations=sum(e['global_gauge_sign_violation'] for e in details if e['parallel']==par))
            for kind,par in [('parallel',True),('perpendicular',False)]}
        result['solution_sha256']=before
        assert digest(path)==before
        summary[name]=result
        write(output/'summary.json',summary)
        print(name,result['signs'],result['violation_detail'],flush=True)
    return summary
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    args=parser.parse_args()
    run(args.source, args.output)
