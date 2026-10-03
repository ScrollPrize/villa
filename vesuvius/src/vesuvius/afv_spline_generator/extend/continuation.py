"""Geometric suffix/head continuations, separate from the learned gap scorer.

Uses the catalogue's continuous 3-D projections. Tangents are compared inside
the shared interval, at matching support scales, rather than at truncated tips.
No catalogue is modified and no geometric cost is a model probability.
"""
import numpy as np

from . import overlap as ov

MIN_OVERLAP = 24.
MAX_MEDIAN = 1.5
MAX_P90 = 3.
MIN_UNIQUE = 8.
COST_MARGIN = .35


def compare(A, B, reference_end):
    original_a, original_sa = ov._curve(A)
    original_b, original_sb = ov._curve(B)
    flip_a = reference_end == 0
    a, sa = ov._curve(original_a[::-1] if flip_a else original_a)
    a, sa = ov._sample(a, sa)
    matches = []
    for flip_b in (False, True):
        b, sb = ov._curve(original_b[::-1] if flip_b else original_b)
        b, sb = ov._sample(b, sb)
        da, ab = ov._project(a, b, sb)
        db, ba = ov._project(b, a, sa)
        if da[-1] > MAX_P90 or db[0] > MAX_P90:
            continue
        start, finish = float(ba[0]), float(ab[-1])
        overlap_a, overlap_b = float(sa[-1] - start), finish
        if min(overlap_a, overlap_b) < MIN_OVERLAP:
            continue
        # Fully contained duplicates add no safe outward continuation here.
        if start <= MIN_UNIQUE or sb[-1] - finish <= MIN_UNIQUE:
            continue
        if max(overlap_a, overlap_b) / min(overlap_a, overlap_b) > 1.25:
            continue
        ia, ib = sa >= start, sb <= finish
        ca, cb = ab[ia], ba[ib]
        if not ov._monotonic(ca) or not ov._monotonic(cb):
            continue
        if ca[-1]-ca[0] < .8*overlap_b or cb[-1]-cb[0] < .8*overlap_a:
            continue
        if np.any(ca > finish+2) or np.any(cb < start-2):
            continue
        median = max(float(np.median(da[ia])), float(np.median(db[ib])))
        p90 = max(float(np.percentile(da[ia],90)), float(np.percentile(db[ib],90)))
        # A short distant excursion must not disappear inside a percentile.
        maximum = max(float(da[ia].max()), float(db[ib].max()))
        if median > MAX_MEDIAN or p90 > MAX_P90 or maximum > 4.:
            continue
        # Same support on both sides, entirely inside the observed overlap.
        angles = []
        for s in np.linspace(start+7, sa[-1]-7, 9):
            t = float(np.interp(s,sa,ab))
            radius = min(6., s-start, sa[-1]-s, t, finish-t)
            if radius >= 3.:
                angles.append(ov._angle(ov._tangent(a,sa,s,radius),ov._tangent(b,sb,t,radius)))
        if len(angles) < 5:
            continue
        angle50, angle90 = np.percentile(angles,[50,90])
        if angle50 > 10. or angle90 > 15. or max(angles) > 25.:
            continue
        margin = min(8., .2*min(overlap_a,overlap_b))
        options = np.flatnonzero((sa>=start+margin)&(sa<=sa[-1]-margin)&
                                  (ab>=margin)&(ab<=finish-margin)&(da<=MAX_MEDIAN))
        if not len(options):
            continue
        centre = (start+sa[-1])/2
        at = min(options,key=lambda i:(round(float(da[i]),8),abs(float(sa[i])-centre)))
        cut_a, cut_b = float(sa[at]), float(ab[at])
        original_cut_a = float(original_sa[-1]-cut_a if flip_a else cut_a)
        original_cut_b = float(original_sb[-1]-cut_b if flip_b else cut_b)
        point_a = ov._at(original_a,original_sa,original_cut_a)
        point_b = ov._at(original_b,original_sb,original_cut_b)
        separation = float(np.linalg.norm(point_a-point_b))
        if separation > MAX_MEDIAN:
            continue
        matches.append(dict(kind='overlap',referenceEnd=reference_end,candidateEnd=1 if flip_b else 0,
            referenceCut=original_cut_a,candidateCut=original_cut_b,
            bridge=[point_a.tolist(),point_b.tolist()],distanceVoxels=separation,
            overlapLength=min(overlap_a,overlap_b),
            geometricCost=median+.25*p90+.02*float(angle90),
            metrics=dict(medianDistance=median,p90Distance=p90,maxDistance=maximum,
                         tangentMedianDegrees=float(angle50),tangentP90Degrees=float(angle90),
                         overlapA=overlap_a,overlapB=overlap_b,monotonic=True)))
    return min(matches,key=lambda m:m['geometricCost']) if matches else None


def clipped(points, low=0., high=None):
    p,s = ov._curve(points)
    high = float(s[-1]) if high is None else float(high)
    low = float(low)
    if not 0 <= low < high <= s[-1]+1e-4:
        raise ValueError('incompatible overlaps on one spline')
    return np.concatenate((ov._at(p,s,low)[None],p[(s>low+1e-7)&(s<high-1e-7)],ov._at(p,s,high)[None]))


def covered(short, long):
    """Strictly redundant alternative, not a reason to erase a source curve.

    A short trace wholly following a longer trace is not a second branch.
    Bounds are tighter than continuation matching; close parallel neighbours
    with a persistent displacement remain distinct competing alternatives.
    """
    a,sa=ov._curve(short);b,sb=ov._curve(long)
    if sa[-1]<MIN_OVERLAP or sa[-1]+MIN_UNIQUE>=sb[-1]:
        return False
    a,sa=ov._sample(a,sa);b,sb=ov._sample(b,sb)
    d,t=ov._project(a,b,sb)
    reverse=t[-1]<t[0]
    if not ov._monotonic(-t if reverse else t) or abs(t[-1]-t[0])<.9*sa[-1]:
        return False
    core=(sa>=8)&(sa<=sa[-1]-8)
    if (np.median(d)>.65 or np.percentile(d,90)>1. or d.max()>4.
            or not core.any() or d[core].max()>1.5):
        return False
    angles=[ov._angle(ov._tangent(a,sa,s),ov._tangent(b,sb,u)*(-1 if reverse else 1))
            for s,u in zip(sa[core],t[core])]
    return bool(np.percentile(angles,90)<=10. and max(angles)<=20.)


def proposals(reference, rows, endpoint, blocked=None, check=lambda:None):
    blocked = blocked or {}
    ends = (0,1) if endpoint=='both' else (0,) if endpoint=='start' else (1,)
    for end in ends:
        if end in blocked.get(reference['id'],set()):
            continue
        tip = np.asarray(reference['points'][0 if end==0 else -1])
        for row in rows:
            check()
            if row['id']==reference['id'] or row['family']!=reference['family']:
                continue
            p=np.asarray(row['points'])
            if np.any(tip<p.min(0)-MAX_P90) or np.any(tip>p.max(0)+MAX_P90):
                continue
            match=compare(reference['points'],p,end)
            if not match or match['candidateEnd'] in blocked.get(row['id'],set()):
                continue
            # CT context is measured at the seam on the retained, observed sides.
            a=clipped(reference['points'],0.,match['referenceCut']) if end==1 else clipped(reference['points'],match['referenceCut'])
            b=clipped(p,match['candidateCut']) if match['candidateEnd']==0 else clipped(p,0.,match['candidateCut'])
            yield dict(match,row=row,a=a[::-1] if end==1 else a,
                       b=b if match['candidateEnd']==0 else b[::-1])
