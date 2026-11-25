"""Continuation candidates around one chain endpoint, read from the catalogue."""
import math

import numpy as np

from .continuation import proposals as overlap_proposals

MAX_ENDPOINTS = 1024
MAX_CHAINS = 192


def _near_endpoints(catalog, point, radius, family):
    low, high = point-radius, point+radius
    sql = '''SELECT e.id,s.id source_id,m.chain,
        (e.x0-?)*(e.x0-?)+(e.y0-?)*(e.y0-?)+(e.z0-?)*(e.z0-?) d2
        FROM endpoint_bbox e CROSS JOIN sources s ON s.id=CAST(e.id/2 AS INTEGER)
        CROSS JOIN members m ON m.source=s.id
        WHERE e.x0<=? AND e.x1>=? AND e.y0<=? AND e.y1>=? AND e.z0<=? AND e.z1>=?
        AND s.family=? AND s.alias IS NULL
        ORDER BY d2 LIMIT ?'''
    return catalog.db.execute(sql, [float(v) for v in np.repeat(point, 2)] +
        [float(high[0]),float(low[0]),float(high[1]),float(low[1]),float(high[2]),float(low[2]),
         int(family),MAX_ENDPOINTS+1]).fetchall()


def _context(catalog, point, exclude):
    radius = 128.
    low, high = point-radius, point+radius
    sql = '''SELECT s.id,s.family,
        ((r.x0+r.x1)*.5-?)*((r.x0+r.x1)*.5-?)+
        ((r.y0+r.y1)*.5-?)*((r.y0+r.y1)*.5-?)+
        ((r.z0+r.z1)*.5-?)*((r.z0+r.z1)*.5-?) d2
        FROM source_bbox r CROSS JOIN sources s ON s.id=r.id
        WHERE r.x0<=? AND r.x1>=? AND r.y0<=? AND r.y1>=? AND r.z0<=? AND r.z1>=?
        AND s.alias IS NULL ORDER BY d2 LIMIT 256'''
    found = catalog.db.execute(sql,[float(v) for v in np.repeat(point, 2)] +
        [float(high[0]),float(low[0]),float(high[1]),float(low[1]),float(high[2]),float(low[2])]).fetchall()
    result = []
    for row in found:
        if row['id'] in exclude:
            continue
        points = np.asarray(catalog.source(row['id'])['geometry'], dtype=np.float32)
        inside = np.flatnonzero(np.linalg.norm(points-point,axis=1) <= radius)
        if len(inside) < 2:
            continue
        points = points[max(0,inside[0]-1):min(len(points),inside[-1]+2)]
        result.append(dict(points=points, family=int(row['family']), sourceId=int(row['id']),
                           distance=float(np.linalg.norm(points-point,axis=1).min())))
    return sorted(result,key=lambda row:row['distance'])[:64]


def _near_overlap_chains(catalog, point, family):
    # An endpoint may meet the *interior* of another curve, hundreds of voxels
    # from its tips. Endpoint-only lookup cannot discover these overlaps.
    low,high=point-3.,point+3.
    return catalog.db.execute('''SELECT DISTINCT m.chain
        FROM source_bbox r CROSS JOIN sources s ON s.id=r.id
        CROSS JOIN members m ON m.source=s.id
        WHERE r.x0<=? AND r.x1>=? AND r.y0<=? AND r.y1>=? AND r.z0<=? AND r.z1>=?
        AND s.family=? AND s.alias IS NULL LIMIT ?''',
        [float(high[0]),float(low[0]),float(high[1]),float(low[1]),float(high[2]),float(low[2]),
         int(family),MAX_CHAINS+1]).fetchall()


def _members(catalog, chain):
    return {int(row[0]) for row in catalog.db.execute('SELECT source FROM members WHERE chain=?',(int(chain),))}


def _endpoint_sample(points, end, length=160.):
    points = np.asarray(points,dtype=np.float32)
    if end == 1:
        points = points[::-1]
    arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    stop = min(len(points),int(np.searchsorted(arc,length))+2)
    return np.ascontiguousarray(points[:stop])


def _proposals(reference, rows, endpoint, radius, blocked=None):
    blocked = blocked or {}
    requested = (0,1) if endpoint=='both' else (0,) if endpoint=='start' else (1,)
    for re in requested:
        if re in blocked.get(reference['id'],set()):
            continue
        a = _endpoint_sample(reference['points'],re)
        if len(a) < 2:
            continue
        for row in rows:
            if row['id'] == reference['id'] or row['family'] != reference['family']:
                continue
            for ce in (0,1):
                if ce in blocked.get(row['id'],set()):
                    continue
                b = _endpoint_sample(row['points'],ce)
                distance = float(np.linalg.norm(a[0]-b[0]))
                if len(b)<2 or not .5 <= distance <= radius:
                    continue
                # Broad proposal cone only; the learned scorer orders candidates.
                # Exact opposite direction or an overlap is not a gap continuation.
                direction = (b[0]-a[0])/distance
                ta = a[0]-a[min(len(a)-1,3)]
                tb = b[0]-b[min(len(b)-1,3)]
                if np.dot(ta,direction) <= -0.2*np.linalg.norm(ta):
                    continue
                if np.dot(tb,-direction) <= -0.2*np.linalg.norm(tb):
                    continue
                yield dict(row=row,a=a,b=b,referenceEnd=re,candidateEnd=ce,distanceVoxels=distance)


def collect(catalog, curve_id, end, radius, check):
    """Gap and overlap proposals for one endpoint (``end`` 0 = start, 1 = end)."""
    warnings = []
    reference = catalog.get_curve(curve_id,include_provenance=False)
    if reference is None:
        raise ValueError(f'chain {curve_id} is not in the catalogue')
    endpoint = 'start' if end == 0 else 'end'
    ref_members = _members(catalog,reference['id'])
    ids = {}
    overlap_ids = set()
    check()
    tip = np.asarray(reference['points'][0 if end==0 else -1],dtype=float)
    found = _near_endpoints(catalog,tip,radius,reference['family'])
    if len(found)>MAX_ENDPOINTS:
        warnings.append('dense neighbourhood: proposals limited to the nearest endpoints')
    for row in found[:MAX_ENDPOINTS]:
        cid = int(row['chain'])
        if cid != reference['id']:
            ids[cid] = min(ids.get(cid,math.inf),float(row['d2']))
    nearby=_near_overlap_chains(catalog,tip,reference['family'])
    if len(nearby)>MAX_CHAINS:
        warnings.append('dense neighbourhood: overlap search incomplete')
    overlap_ids.update(int(row['chain']) for row in nearby if int(row['chain'])!=reference['id'])
    context = _context(catalog,tip,ref_members)
    if len(set(ids)|overlap_ids)>MAX_CHAINS:
        warnings.append(f'search limited to the {MAX_CHAINS} nearest chains')
    rows, members = [], {}
    candidate_ids=sorted(overlap_ids)+[cid for cid in sorted(ids,key=ids.get) if cid not in overlap_ids]
    for cid in candidate_ids[:MAX_CHAINS]:
        check()
        row = catalog.get_curve(cid,include_provenance=False)
        if row is None:
            continue
        rows.append(row)
        members[cid] = _members(catalog,cid)
    proposals = list(_proposals(reference,rows,endpoint,radius))
    proposals.extend(overlap_proposals(reference,rows,endpoint,None,check))
    for proposal in proposals:
        proposal['context'] = [c for c in context if c['sourceId'] not in members[proposal['row']['id']]]
    return reference,proposals,warnings


def bridge(a,b):
    # Hermite hypothesis between two tips; these points are not observed CT.
    p,q = a[0].astype(float),b[0].astype(float)
    distance = float(np.linalg.norm(q-p))
    ta,tb = p-a[min(3,len(a)-1)],b[min(3,len(b)-1)]-q
    ta /= max(float(np.linalg.norm(ta)),1e-8)
    tb /= max(float(np.linalg.norm(tb)),1e-8)
    t = np.linspace(0,1,max(3,min(32,int(distance/3)+2)))[:,None]
    return ((2*t**3-3*t**2+1)*p+(t**3-2*t**2+t)*ta*distance+
            (-2*t**3+3*t**2)*q+(t**3-t**2)*tb*distance).tolist()
