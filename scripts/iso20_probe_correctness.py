"""Read-only cached factor audit against strict reference-curve correspondences.

No BP state is used as ground truth. Spherical matching is deliberately a
conservative audit subset, not a replacement for the acceptance ellipse.
"""
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vesuvius/src'))
import numpy as np
from scipy.spatial import cKDTree
from vesuvius.scripts.ordered_polyline_obj import read_ordered_polyline_obj

ROOT = Path('/ephemeral/hendrik/fiber_crop_bench/20261002_iso20_orthogonal_probes_v1')
REPORT = Path(__file__).resolve().parents[1] / 'volume-cartographer/docs/fiber_benchmark_runs/2026-10-03-probe-correctness.json'
NAMES = ('winner-repeat', 'winner-median64', 'winner-consensus32')


def curves(path):
    return [g.points_xyz for g in read_ordered_polyline_obj(
        path, container_records=('o',), allow_singletons=True,
        require_segment_lines=False).groups]


def sample_at(points, query):
    points = np.asarray(points, dtype=float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.r_[True, lengths > 1e-9]
    points = points[keep]
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    arc = np.r_[0, np.cumsum(lengths)]
    query = np.asarray(query)
    if query.max() > arc[-1] + 1 or query.min() < 0:
        raise ValueError('piece/source arc mismatch')
    idx = np.clip(np.searchsorted(arc, query, side='right')-1, 0, len(lengths)-1)
    return points[idx]+delta[idx]*((query-arc[idx])/lengths[idx])[:, None], delta[idx]/lengths[idx, None]


def sample_curve(points, begin, end, spacing=4):
    if end <= begin:
        raise ValueError('invalid interval')
    n = max(1, int(np.ceil((end-begin)/spacing)))
    return sample_at(points, begin + (np.arange(n)+.5)*(end-begin)/n)


def point_votes(positions, tangents, refs, radii=(4,8,16)):
    starts, vectors, labels = [], [], []
    for label, lines in refs.items():
        for line in lines:
            d = np.diff(line, axis=0)
            ok = np.linalg.norm(d, axis=1) > 1e-9
            starts.extend(line[:-1][ok]); vectors.extend(d[ok]); labels.extend([label]*int(ok.sum()))
    a, d, labels = np.array(starts), np.array(vectors), np.array(labels)
    lens = np.linalg.norm(d, axis=1)
    tree = cKDTree(a+d*.5)
    positions,tangents=np.asarray(positions),np.asarray(tangents)
    neighbours = tree.query_ball_point(positions, max(radii)+lens.max()/2, workers=32)
    sizes = np.fromiter(map(len, neighbours), int)
    qi = np.repeat(np.arange(len(positions)), sizes)
    si = np.concatenate([np.asarray(v,dtype=int) for v in neighbours])
    t = np.clip(np.einsum('ij,ij->i',positions[qi]-a[si],d[si])/lens[si]**2,0,1)
    dist = np.linalg.norm(positions[qi]-a[si]-t[:,None]*d[si],axis=1)
    aligned = np.abs(np.einsum('ij,ij->i',tangents[qi],d[si])/lens[si]) >= np.cos(np.deg2rad(25))
    # One vote per sample and reference, regardless of segment tessellation.
    result = {}
    nlabel = int(labels.max())+1
    for radius in radii:
        ok = aligned & (dist <= radius)
        unique = np.unique(qi[ok]*nlabel+labels[si[ok]])
        hits = np.zeros((len(positions),nlabel),dtype=bool)
        hits[unique//nlabel,unique%nlabel]=True
        result[radius] = hits
    return result


def match_pieces(sources, pieces, refs, radii=(4,8,16)):
    positions,tangents,owners=[],[],[]
    for i,p in enumerate(pieces):
        x,t=sample_curve(sources[int(p['original_trace'])],float(p['begin_arc_base']),float(p['end_arc_base']))
        positions.extend(x);tangents.extend(t);owners.extend([i]*len(x))
    owners=np.array(owners)
    counts=np.bincount(owners,minlength=len(pieces))
    result={}
    for radius,votes in point_votes(positions,tangents,refs,radii).items():
        hits=np.zeros((len(pieces),votes.shape[1]),dtype=int)
        np.add.at(hits,owners,votes)
        admitted=hits>=.9*counts[:,None]
        result[radius]=np.where((admitted.sum(axis=1)==1)&((hits>0).sum(axis=1)==1),admitted.argmax(axis=1),-1)
    return result


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def score(factors, labels, polarity=None, contact_labels=None, kind='orthogonal'):
    value_key='original_signed_delta' if kind=='orthogonal' else 'original_signed_parallel_delta'
    flag='hard_perpendicular_sign' if kind=='orthogonal' else 'hard_parallel_sign'
    eligible = []
    for r in factors:
        a,b = labels[int(r['piece_a'])],labels[int(r['piece_b'])]
        if a < 0 or b < 0 or r[value_key]=='NA' or r[flag]!='1':
            continue
        if contact_labels is not None and contact_labels[r['constraint']] != (a,b):
            continue
        eligible.append((r,float(b-a)/2,float(r[value_key])))
    if polarity is None:
        polarity = {}
        for component in {r['normal_component'] for r,_,_ in eligible}:
            pairs = [(truth,value) for r,truth,value in eligible if r['normal_component']==component and truth!=0 and value!=0]
            votes = sum(np.sign(t)*np.sign(v) for t,v in pairs)
            polarity[component] = int(np.sign(votes))
    errors = {}
    magnitude = []
    for r,truth,value in eligible:
        if polarity[r['normal_component']]==0:
            continue
        errors[r['constraint']] = bool(truth==0 or polarity[r['normal_component']]*value*truth<=0)
        magnitude.append(abs(abs(value)-abs(truth)))
    return dict(assessable=len(errors),wrong=sum(errors.values()),
        wrong_percent=100*sum(errors.values())/len(errors) if errors else None,
        same_winding_pairs=sum(t==0 for _,t,_ in eligible),
        raw_magnitude_mae=float(np.mean(magnitude)) if magnitude else None), errors, polarity


def main():
    pieces = read_csv(ROOT/'winner-repeat_consistency.csv')
    sources = curves(ROOT/'winner-repeat_initial.obj')
    refs = {int(p.stem.rsplit('_',1)[1]):curves(p) for p in ROOT.glob('winner-repeat_reference_hs_*.obj')}
    assigned = match_pieces(sources,pieces,refs)
    factors = {name:read_csv(ROOT/f'{name}_winding_factors.csv') for name in NAMES}
    keys = ('constraint','piece_a','piece_b','arc_a','arc_b','original_signed_parallel_delta')
    for name in NAMES[1:]:
        assert [[r[k] for k in keys] for r in factors[name]]==[[r[k] for k in keys] for r in factors[NAMES[0]]]
        other = read_csv(ROOT/f'{name}_consistency.csv')
        assert [(r['original_trace'],r['begin_arc_base'],r['end_arc_base']) for r in pieces]==[(r['original_trace'],r['begin_arc_base'],r['end_arc_base']) for r in other]
    contacts={}
    contact_keys=[]
    positions,tangents=[],[]
    for r in factors[NAMES[0]]:
        pair=[]
        for side in ('a','b'):
            source=int(pieces[int(r['piece_'+side])]['original_trace'])
            key=(source,float(r['arc_'+side]))
            if key not in contacts:
                contacts[key]=len(positions)
                x,t=sample_at(sources[source],[key[1]])
                positions.append(x[0]);tangents.append(t[0])
            pair.append(contacts[key])
        contact_keys.append(pair)
    contact_keys=np.asarray(contact_keys)
    contact_votes=point_votes(positions,tangents,refs)
    output = {'method':'unique >=90% arc samples within spherical radius, tangent <=25 degrees; midpoint spacing <=4vx; strict subset, NOT acceptance ellipse',
              'polarity':'majority sign per normal component on baseline matched subset; frozen for probes; label uncertainty and calibration can bias absolute error rates',
              'input_root':str(ROOT),'pieces':len(pieces),'radii':{}}
    for radius,labels in assigned.items():
        votes=contact_votes[radius]
        cl=np.where(votes.sum(axis=1)==1,votes.argmax(axis=1),-1)
        per_factor={r['constraint']:tuple(cl[key]) for r,key in zip(factors[NAMES[0]],contact_keys)}
        row = {'matched_pieces':int((labels>=0).sum()),'variants':{}}
        base,base_errors,polarity = score(factors[NAMES[0]],labels,contact_labels=per_factor)
        row['polarity']=polarity
        _,_,parallel_polarity=score(factors[NAMES[0]],labels,contact_labels=per_factor,kind='parallel')
        row['parallel_polarity']=parallel_polarity
        for name in NAMES:
            stats,errors,_ = score(factors[name],labels,polarity,per_factor)
            common=errors.keys() & base_errors.keys()
            stats.update(corrected=sum(base_errors[k] and not errors[k] for k in common),
                         damaged=sum(not base_errors[k] and errors[k] for k in common),
                         common=len(common),total_hard_orthogonal=sum(r['hard_perpendicular_sign']=='1' for r in factors[name]))
            stats['changed_targets_assessable']=sum(a['constraint'] in common and a['original_signed_delta']!=b['original_signed_delta'] for a,b in zip(factors[NAMES[0]],factors[name]))
            stats['parallel'],_,_=score(factors[name],labels,parallel_polarity,per_factor,kind='parallel')
            row['variants'][name]=stats
        output['radii'][radius]=row
    REPORT.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':
    main()
