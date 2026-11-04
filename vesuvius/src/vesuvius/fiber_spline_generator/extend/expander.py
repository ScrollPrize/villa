"""Iterative, reciprocal endpoint expansion; catalogues remain unchanged.

Safety gates are experimental criteria, not calibrated physical guarantees.
Both seed endpoints grow until their local choice is insufficient or ambiguous.
"""
import math
import time

import numpy as np
from scipy.spatial import cKDTree

from .continuation import COST_MARGIN, clipped, covered
from .overlap import _at, _curve, _has_loop, _project

MIN_MARGIN = .15
MAX_LINKS = 1000
MAX_SECONDS = 300.
MAX_POINTS = 2_000_000
STOP_LABELS = {
    'no_candidate':'no continuation',
    'low_score':'score too low',
    'ambiguous':'several possible continuations',
    'not_reciprocal':'join is not reciprocal',
    'reverse_ambiguous':'reverse join is ambiguous',
    'out_of_domain':'fragment too short or join outside the model domain',
    'cycle':'returns into the existing chain',
    'obstruction':'geometric obstruction',
    'candidate_limit':'incomplete neighbourhood',
    'ct_unavailable':'CT unavailable',
    'not_requested':'endpoint not requested',
    'limit':'computation limit reached',
    'overlap_ambiguous':'several overlapping paths possible',
    'overlap_conflict':'incompatible overlaps',
}


def arc(points):
    return float(np.linalg.norm(np.diff(np.asarray(points,dtype=float),axis=0),axis=1).sum())


def _stop(code,**details):
    return dict(code=code,label=STOP_LABELS[code],**details)


def _sample_local(points,boxlo,boxhi,step=1.):
    """Dense points from only segments near a bridge, with bounded allocations."""
    p=np.asarray(points,dtype=float)
    if len(p)<2:
        return np.empty((0,3))
    hit=(np.minimum(p[:-1],p[1:])<=boxhi).all(1)&(np.maximum(p[:-1],p[1:])>=boxlo).all(1)
    pieces=[]
    for index in np.flatnonzero(hit):
        a,b=p[index:index+2]
        count=min(256,max(2,int(math.ceil(np.linalg.norm(b-a)/step))+1))
        pieces.append(np.linspace(a,b,count))
    return np.concatenate(pieces) if pieces else np.empty((0,3))


def obstruction(proposal,bridge,accepted):
    """Do not cross same-family support or return into an already used curve."""
    b=np.asarray(bridge,dtype=float)
    dense=_sample_local(b,b.min(0)-1,b.max(0)+1,step=.75)
    if not len(dense):
        return False
    # Endpoints naturally touch their two source curves. Only the interior gap
    # tests third-party geometry; long gaps retain a two-voxel end exclusion.
    interior=dense[(np.linalg.norm(dense-b[0],axis=1)>3)&(np.linalg.norm(dense-b[-1],axis=1)>3)]
    if not len(interior):
        return False
    lo,hi=interior.min(0)-2.5,interior.max(0)+2.5
    curves=[entry['points'] for entry in proposal.get('context',[])
            if int(entry.get('family',proposal['row']['family']))==int(proposal['row']['family'])]
    curves += [row['points'] for row in accepted.values()]
    curves.append(proposal['row']['points'])
    for points in curves:
        local=_sample_local(points,lo,hi)
        if len(local) and np.any(cKDTree(local).query(interior)[0]<2.):
            return True
    return False


def retained_ranges(seed,steps):
    rows={int(seed['id']):seed,**{int(s['row']['id']):s['row'] for s in steps}}
    ranges={key:[0.,arc(row['points'])] for key,row in rows.items()}
    for step in steps:
        if step.get('kind')!='overlap':
            continue
        for key,end,cut in ((step['referenceId'],step['referenceEnd'],step['referenceCut']),
                            (step['row']['id'],step['candidateEnd'],step['candidateCut'])):
            bounds=ranges[int(key)]
            bounds[end]=max(bounds[0],cut) if end==0 else min(bounds[1],cut)
            if bounds[1]-bounds[0]<=1e-5:
                raise ValueError('incompatible overlaps on one spline')
    return rows,ranges


def assemble(seed,steps):
    """Keep one copy of shared intervals, retaining source geometry at a seam."""
    rows,ranges=retained_ranges(seed,steps)
    def points(row):
        low,high=ranges[int(row['id'])]
        if low==0 and abs(high-arc(row['points']))<1e-5:
            return np.asarray(row['points'],dtype=np.float64)
        return clipped(row['points'],low,high)
    left=[s for s in steps if s['side']==0]
    right=[s for s in steps if s['side']==1]
    pieces=[]
    for step in reversed(left):
        p=points(step['row'])
        pieces.extend([p if step['candidateEnd']==1 else p[::-1],
                       np.asarray(step['bridge'],dtype=np.float64)[::-1]])
    pieces.append(points(seed))
    for step in right:
        p=points(step['row'])
        pieces.extend([np.asarray(step['bridge'],dtype=np.float64),
                       p if step['candidateEnd']==0 else p[::-1]])
    p=np.concatenate(pieces)
    keep=np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-6]
    return p[keep]


class Expander:
    """Small graph walk with injected reads/scoring for deterministic tests."""
    def __init__(self,seed,fetch,score,check,bridge,minimum=.9,margin=MIN_MARGIN,
                 endpoint='both',measure=None,blocked=obstruction,progress=None,
                 max_links=MAX_LINKS,max_seconds=MAX_SECONDS,max_points=MAX_POINTS):
        self.seed=seed
        self.fetch,self.score,self.check,self.bridge=fetch,score,check,bridge
        self.minimum,self.margin=float(minimum),float(margin)
        self.measure,self.blocked,self.progress=measure,blocked,progress
        self.max_links,self.max_seconds,self.max_points=max_links,max_seconds,max_points
        self.started=time.monotonic()
        self.accepted={int(seed['id']):seed}
        self.steps=[]
        self.fronts={0:(int(seed['id']),0),1:(int(seed['id']),1)}
        self.stops={}
        self.tests=0
        self.cache={}
        self.truncated=False
        if endpoint!='both':
            self.stops[1 if endpoint=='start' else 0]=_stop('not_requested')

    def ranked(self,identity,end):
        key=(identity,end)
        self.check()
        if key not in self.cache:
            row,proposals,warnings=self.fetch(identity,end)
            if any('dense neighbourhood' in warning or 'search limited' in warning for warning in warnings):
                return [],_stop('candidate_limit')
            proposals=[p for p in proposals if p['referenceEnd']==end]
            if not proposals:
                self.cache[key]=[]
            else:
                gaps=[p for p in proposals if p.get('kind')!='overlap']
                scores=np.asarray(self.score(gaps,row) if gaps else [],dtype=float).reshape(-1)
                if len(scores)!=len(gaps) or not np.isfinite(scores).all():
                    raise ValueError('invalid expansion scores')
                self.tests+=len(proposals)
                scored=[dict(p,score=None,inTrainingDomain=False) for p in proposals if p.get('kind')=='overlap']
                for p,score in zip(gaps,scores):
                    support=min(arc(p['a']),arc(p['b']))
                    scored.append(dict(p,score=float(score),inTrainingDomain=(2<=p['distanceVoxels']<=45 and support>=39.5)))
                self.cache[key]=scored
        return self.cache[key],None

    def select(self,ranked):
        if not ranked:
            return None,_stop('no_candidate')
        overlaps=[p for p in ranked if p.get('kind')=='overlap']
        if overlaps:
            distinct=[p for p in overlaps if not any(
                p is not other and covered(p['row']['points'],other['row']['points']) for other in overlaps)]
            distinct.sort(key=lambda p:p['geometricCost'])
            if len(distinct)>1 and distinct[1]['geometricCost']-distinct[0]['geometricCost']<COST_MARGIN:
                return None,_stop('overlap_ambiguous',candidateIds=[int(p['row']['id']) for p in distinct])
            top=distinct[0]
            return top,None
        ranked=sorted(ranked,key=lambda p:p['score'],reverse=True)
        top=ranked[0]
        details=dict(bestScore=top['score'],candidateId=int(top['row']['id']))
        if not top['inTrainingDomain']:
            return None,_stop('out_of_domain',**details)
        if top['score']<self.minimum:
            return None,_stop('low_score',**details)
        second=ranked[1]['score'] if len(ranked)>1 else 0.
        details['secondScore']=second
        if top['score']-second<self.margin:
            return None,_stop('ambiguous',**details)
        return top,None

    def advance(self,side):
        identity,end=self.fronts[side]
        ranked,reason=self.ranked(identity,end)
        if reason is None:
            chosen,reason=self.select(ranked)
        if reason is not None:
            self.stops[side]=reason
            return
        candidate=int(chosen['row']['id']);ce=int(chosen['candidateEnd'])
        if candidate in self.accepted:
            self.stops[side]=_stop('cycle',candidateId=candidate)
            return
        reverse,reason=self.ranked(candidate,ce)
        if reason is None:
            back,reason=self.select(reverse)
        if reason is not None:
            self.stops[side]=_stop('reverse_ambiguous',candidateId=candidate,reverseStop=reason)
            return
        if int(back['row']['id'])!=identity or back['candidateEnd']!=end:
            self.stops[side]=_stop('not_reciprocal',candidateId=candidate)
            return
        kind=chosen.get('kind','gap')
        bridge=chosen['bridge'] if kind=='overlap' else self.bridge(chosen['a'],chosen['b'])
        step=dict(side=side,referenceId=identity,referenceEnd=end,candidateEnd=ce,
                  row=chosen['row'],bridge=bridge,kind=kind,score=chosen['score'],
                  reverseScore=back['score'],distanceVoxels=chosen['distanceVoxels'])
        if kind=='overlap':
            step.update({key:chosen[key] for key in ('referenceCut','candidateCut','overlapLength','geometricCost','metrics')})
            try:
                combined=assemble(self.seed,self.steps+[step])
            except ValueError:
                self.stops[side]=_stop('overlap_conflict',candidateId=candidate)
                return
            if _has_loop(combined):
                self.stops[side]=_stop('cycle',candidateId=candidate)
                return
        if self.blocked(chosen,bridge,self.accepted):
            self.stops[side]=_stop('obstruction',candidateId=candidate)
            return
        self.check()
        ct={'state':'not_requested'}
        if self.measure is not None:
            ct=self.measure(chosen['a'],chosen['b'])
            self.check()
            if ct.get('state')!='available':
                self.stops[side]=_stop('ct_unavailable',candidateId=candidate,ct=ct)
                return
        step['ct']=ct
        self.steps.append(step)
        self.accepted[candidate]=chosen['row']
        self.fronts[side]=(candidate,1-ce)

    def run(self):
        while len(self.stops)<2:
            self.check()
            for side in (0,1):
                if side in self.stops:
                    continue
                if (len(self.steps)>=self.max_links or time.monotonic()-self.started>=self.max_seconds
                        or sum(len(row['points']) for row in self.accepted.values())>=self.max_points):
                    self.truncated=True
                    self.stops[side]=_stop('limit')
                    continue
                self.advance(side)
                if self.progress:
                    self.progress(self)
        return self

    def result(self,include_geometry=True):
        new_gap=sum(arc(step['bridge']) for step in self.steps if step.get('kind')!='overlap')
        combined=assemble(self.seed,self.steps)
        length=arc(combined)
        return dict(seedId=int(self.seed['id']),curveCount=len(self.accepted),
                    addedCurveCount=len(self.steps),linkCount=len(self.steps),
                    curveIds=list(self.accepted),lengthVoxels=length,
                    addedLengthVoxels=length-arc(self.seed['points']),
                    overlapCount=sum(step.get('kind')=='overlap' for step in self.steps),
                    gapCount=sum(step.get('kind')!='overlap' for step in self.steps),
                    newGapLengthVoxels=new_gap,stops={str(k):v for k,v in self.stops.items()},
                    fronts={str(k):dict(curveId=v[0],end=v[1]) for k,v in self.fronts.items()},
                    minimumScore=self.minimum,minimumMargin=self.margin,
                    candidateTests=self.tests,truncated=self.truncated,
                    points=combined.tolist() if include_geometry else [])

    def ribbon_curve(self,result):
        # Preserve inherited gap provenance only over each retained source range.
        rows,ranges=retained_ranges(self.seed,self.steps)
        gaps=[]
        for identity,row in rows.items():
            p,s=_curve(row['points']);low,high=ranges[identity]
            for gap in row.get('gaps',[]):
                if not gap.get('points'):
                    continue
                _,at=_project(np.asarray(gap['points'])[ [0,-1] ],p,s)
                begin,end=max(low,float(min(at))),min(high,float(max(at)))
                if end>begin+1e-6:
                    gaps.append(dict(gap,points=_at(p,s,[begin,end]).tolist(),lengthVoxels=end-begin))
        for step in self.steps:
            gaps.append(dict(kind='overlap-seam' if step.get('kind')=='overlap' else 'gap',
                             points=step['bridge'],lengthVoxels=arc(step['bridge']),
                             inferred=True,sourceCurveIds=[step['referenceId'],step['row']['id']]))
        gap_length=sum(g['lengthVoxels'] for g in gaps if g.get('kind')=='gap')
        seam_length=sum(g['lengthVoxels'] for g in gaps if g.get('kind')=='overlap-seam')
        return dict(id=int(self.seed['id']),family=int(self.seed['family']),points=result['points'],
                    pointCount=len(result['points']),lengthVoxels=result['lengthVoxels'],gaps=gaps,
                    gapLengthVoxels=gap_length,seamLengthVoxels=seam_length,
                    observedLengthVoxels=max(0.,result['lengthVoxels']-gap_length-seam_length),
                    sourceCurveIds=result['curveIds'],curveCount=result['curveCount'])
