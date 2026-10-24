"""Read-only matched-cohort inspection of saved UV surface refinement."""
import csv
import json
from pathlib import Path

import numpy as np

from . import fiber_uv_surface_refinement as refine
from . import fiber_uv_layout as uv
from .fiber_uv_visualization import TiledRaster, VectorPDF, DrawingSinks


def stats(values):
    a=np.asarray(values)
    return dict(mean=float(a.mean()),median=float(np.median(a)),rms=float(np.sqrt(np.mean(a*a))),
                p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)),maximum=float(a.max()),
                over100_percent=float(100*np.mean(a>100)),over1000_percent=float(100*np.mean(a>1000)))


def winding_positions(points,widths):
    """Physical U inverted through fitted widths; no H/V ordering half-step."""
    n=len(points)
    return uv.geometry_points({'family':np.ones(n,dtype=int)},points[:,0],points[:,1],
        np.r_[0.,np.cumsum(widths)],widths,np.arange(n),np.zeros(n))[0]


def run(config):
    root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    path=Path(config['solution'])
    with np.load(path) as f:after=dict(f)
    parent_path=Path(str(after['parent_solution']))
    parent=json.loads((parent_path.parent/'config.json').read_text())
    with np.load(parent_path) as f:s=dict(f)
    with np.load(parent['tensor_cache']) as f:d=dict(f)
    refinement_config=json.loads((path.parent/'summary.json').read_text())['config']
    model=refine.initialize(d,s,parent,refinement_config)
    np.testing.assert_array_equal(model['piece'],after['original_piece_index'])
    for key in ('source','arc','piece','kind','constraint_id','side'):
        np.testing.assert_array_equal(model['meta'][key],after['endpoint_'+key])
    before=dict(after,piece_uv_base=model['anchors'],endpoint_uv_offset_base=model['seed'])
    piece=after['endpoint_piece'];arc=after['endpoint_arc'];ds=after['endpoint_ds']
    a=refine.evaluate_points(before,piece,arc,ds);b=refine.evaluate_points(after,piece,arc,ds)
    np.testing.assert_allclose(a,model['base']+model['seed'],atol=1e-7,rtol=0)
    delta=b-a
    report=dict(parent=str(parent_path),solution=str(path),units='base voxels in UV, not 3D movement',
                endpoint_movement={name:stats(v) for name,v in [('u',abs(delta[:,0])),('v',abs(delta[:,1])),('uv',np.linalg.norm(delta,axis=1))]},
                anchor_movement=stats(np.linalg.norm(after['piece_delta_uv_base'],axis=1)))
    qa=winding_positions(a,after['widths']);qb=winding_positions(b,after['widths'])
    report['endpoint_movement']['windings']=stats(abs(qb-qa))
    e=len(model['edges']);same=after['same_sheet_perpendicular']
    report['same_sheet_contact_mismatch']={}
    for name,points,q in [('before',a,qa),('after',b,qb)]:
        diff=points[e:2*e][same]-points[:e][same]
        report['same_sheet_contact_mismatch'][name]=dict(events=int(same.sum()),
            windings=stats(abs(q[e:2*e][same]-q[:e][same])),u=stats(abs(diff[:,0])),v=stats(abs(diff[:,1])))
    curves=[];rows=[]
    for k,original in enumerate(model['piece']):
        # Each source's signed longitudinal coordinate is affine in true arc.
        local=np.flatnonzero(piece==k)
        lo=float(after['start'][k]);hi=float(after['end'][k])
        travel=d['node_ds'][original]
        direction=float((travel[2]-travel[0])/(hi-lo))
        arcs=np.unique(np.r_[lo,arc[local],hi])
        signed=travel[0]+direction*(arcs-lo)
        owner=np.full(len(arcs),k)
        ca=refine.evaluate_points(before,owner,arcs,signed)
        cb=refine.evaluate_points(after,owner,arcs,signed)
        curves.append((ca,cb))
        motion=np.linalg.norm(cb-ca,axis=1)
        rows.append(dict(piece=int(original),source=int(after['source_id'][k]),family='H' if after['family'][k]==0 else 'V',
                         maximum_movement=float(motion.max()),median_node_movement=float(np.median(motion))))
    with (root/'piece_movements.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(sorted(rows,key=lambda r:-r['maximum_movement']))
    allpoints=np.concatenate([c for pair in curves for c in pair]);low=allpoints.min(0);high=allpoints.max(0)
    height=1000;scale=(height-1)/max(high[1]-low[1],1);width=int(np.ceil((high[0]-low[0])*scale))+1
    report['shared_bounds']=dict(minimum=low.tolist(),maximum=high.tolist(),width=width,height=height,scale=scale)
    def xy(points):return np.column_stack([(points[:,0]-low[0])*scale,(high[1]-points[:,1])*scale])
    for name in ('before','after','overlay'):
        draw=DrawingSinks(TiledRaster(width,height),VectorPDF(width,height))
        for boundary in np.r_[0,np.cumsum(after['widths'])]:
            x=(boundary-low[0])*scale
            if 0<=x<width:draw.line((x,0,x,height-1),fill=(220,220,220,255),width=1)
        for k,(ca,cb) in enumerate(curves):
            color=(27,108,168,255) if after['family'][k]==0 else (208,105,22,255)
            selected=[(ca,color)] if name=='before' else [(cb,color)] if name=='after' else [(ca,(160,160,160,120)),(cb,color)]
            for curve,rgb in selected:
                points=xy(curve)
                for start,end in zip(points[:-1],points[1:]):draw.line(tuple(np.r_[start,end]),fill=rgb,width=2)
        draw.sinks[0].save(root/(name+'.tif'));draw.sinks[1].save(root/(name+'.pdf'))
    (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report
