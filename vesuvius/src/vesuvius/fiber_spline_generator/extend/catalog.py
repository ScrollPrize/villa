"""Incremental, reversible catalogue of source splines and the chains they form.

Only unquantized spline coordinates and provenance survive ingestion. Spatial
R-trees restrict geometric reconciliation to the new block frontier. Immutable
source fragments permit overlap/gap links to be withdrawn when a later block
introduces an alternative; cached chain geometry is never the source of truth.

A geometric bridge is a hypothesis, NOT CT-verified fibre identity.
"""
from __future__ import annotations

from collections import deque
import hashlib
from functools import wraps
import json
import math
from pathlib import Path
import sqlite3
import zlib

import numpy as np
from scipy.spatial import cKDTree

from . import overlap as ov
from .endpoints import endpoints

SPACING = 4.
GAP_CAP = 30.
VERSION = 1


def jdump(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False)


GEOMETRY_TAG=b'FBS2'
ENDPOINT_TAG=b'EJZ1'


def pack(points):
    # Reversible operations on IEEE754 *bits*, not floating-point deltas.
    # uint32 modular arithmetic preserves every sign/exponent/mantissa bit.
    words=np.asarray(points,dtype='<f4').reshape(-1,3).view('<u4').copy()
    for _ in range(2): words[1:]=words[1:]-words[:-1]
    words=(words<<1)^((words.view('<i4')>>31).view('<u4'))
    shuffled=words.view('u1').reshape(-1,12).T.copy().tobytes()
    return GEOMETRY_TAG+zlib.compress(shuffled,6)


def unpack(blob):
    if not blob.startswith(GEOMETRY_TAG):
        # Original catalogue payloads remain readable during/after migration.
        return np.frombuffer(zlib.decompress(blob),dtype='<f4').reshape(-1,3).copy()
    raw=np.frombuffer(zlib.decompress(blob[len(GEOMETRY_TAG):]),'u1')
    words=raw.reshape(12,-1).T.copy().reshape(-1).view('<u4').reshape(-1,3)
    words=(words>>1)^(np.uint32(0)-(words&1))
    for _ in range(2): words=np.cumsum(words,axis=0,dtype=np.uint32)
    return words.view('<f4')


def pack_endpoints(value):
    raw=value.encode() if isinstance(value,str) else jdump(value).encode()
    return ENDPOINT_TAG+zlib.compress(raw,6)


def unpack_endpoints(value):
    if isinstance(value,bytes) and value.startswith(ENDPOINT_TAG):
        value=zlib.decompress(value[len(ENDPOINT_TAG):])
    return json.loads(value)


def clean(points):
    return ov._curve(points)


def point_count(length):
    # Endpoint-exclusive arange(0, length, 4), with tolerance for float32 noise.
    return max(1, int(math.ceil(float(length) / SPACING - 1e-7)))


def bbox(points):
    return points.min(axis=0), points.max(axis=0)


def strict_duplicate(a, b):
    """Subvoxel full containment, substantially stricter than continuation.

    No transitive clustering: a suppressed fragment must directly match the
    retained representative. Adjacent fibres separated by one voxel survive.
    """
    a, sa = clean(a); b, sb = clean(b)
    if sa[-1] > sb[-1] + .15:
        return None
    sample, ss = ov._sample(a, sa)
    target, st = ov._sample(b, sb)
    d, projected = ov._project(sample, target, st)
    if np.median(d) > .20 or np.percentile(d, 90) > .40 or d.max() > .65:
        return None
    if abs(projected[-1] - projected[0]) < .97 * sa[-1]:
        return None
    reverse = bool(projected[-1] < projected[0])
    signed = -projected if reverse else projected
    if not ov._monotonic(signed):
        return None
    angle = max(ov._angle(ov._tangent(a, sa, s),
                          ov._tangent(b, sb, t) * (-1 if reverse else 1))
                for s, t in ((0, projected[0]), (sa[-1], projected[-1])))
    if angle > 8:
        return None
    return dict(median=float(np.median(d)), p90=float(np.percentile(d, 90)),
                maximum=float(d.max()), reverse=reverse, angle=angle)


SCHEMA = '''
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,sha TEXT NOT NULL,receipt TEXT NOT NULL,
 imported REAL NOT NULL,curve_count INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS sources(id INTEGER PRIMARY KEY,job_id TEXT NOT NULL,
 source_index INTEGER NOT NULL,family INTEGER NOT NULL,points BLOB NOT NULL,
 length REAL NOT NULL,count INTEGER NOT NULL,original_count INTEGER NOT NULL,
 endpoint_data TEXT NOT NULL,alias INTEGER,UNIQUE(job_id,source_index));
CREATE INDEX IF NOT EXISTS source_alias ON sources(alias);
CREATE VIRTUAL TABLE IF NOT EXISTS source_bbox USING rtree(id,x0,x1,y0,y1,z0,z1);
CREATE VIRTUAL TABLE IF NOT EXISTS endpoint_bbox USING rtree(id,x0,x1,y0,y1,z0,z1);
CREATE TABLE IF NOT EXISTS duplicates(a INTEGER NOT NULL,b INTEGER NOT NULL,info TEXT NOT NULL,
 PRIMARY KEY(a,b));
CREATE INDEX IF NOT EXISTS duplicate_b ON duplicates(b);
CREATE TABLE IF NOT EXISTS candidates(id INTEGER PRIMARY KEY,a INTEGER NOT NULL,ea INTEGER NOT NULL,
 b INTEGER NOT NULL,eb INTEGER NOT NULL,kind TEXT NOT NULL,score REAL NOT NULL,
 strict INTEGER NOT NULL,info TEXT NOT NULL,UNIQUE(a,ea,b,eb,kind));
CREATE INDEX IF NOT EXISTS candidates_a ON candidates(a,ea);
CREATE INDEX IF NOT EXISTS candidates_b ON candidates(b,eb);
CREATE TABLE IF NOT EXISTS links(id INTEGER PRIMARY KEY REFERENCES candidates(id),
 a INTEGER NOT NULL,ea INTEGER NOT NULL,b INTEGER NOT NULL,eb INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS links_a ON links(a,ea);
CREATE INDEX IF NOT EXISTS links_b ON links(b,eb);
CREATE TABLE IF NOT EXISTS chains(id INTEGER PRIMARY KEY AUTOINCREMENT,family INTEGER NOT NULL,
 point_count INTEGER NOT NULL,length REAL NOT NULL,observed REAL NOT NULL,gap REAL NOT NULL,
 seam REAL NOT NULL,source_count INTEGER NOT NULL,bbox TEXT NOT NULL,signature TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS long_chains ON chains(point_count DESC,id);
CREATE TABLE IF NOT EXISTS members(source INTEGER PRIMARY KEY,chain INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS member_chain ON members(chain);
'''


def read_snapshot(method):
    @wraps(method)
    def wrapped(self,*args,**kwargs):
        owned=not self.db.in_transaction
        if owned: self.db.execute('BEGIN')
        try:
            return method(self,*args,**kwargs)
        finally:
            if owned: self.db.rollback()
    return wrapped


class Catalog:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'catalog.sqlite3'
        self.db = sqlite3.connect(self.path, timeout=120.)
        self.db.row_factory = sqlite3.Row
        # A private scratch catalogue: it is rebuilt from scratch on every run.
        self.db.execute('PRAGMA journal_mode=MEMORY')
        self.db.execute('PRAGMA synchronous=OFF')
        self.db.execute('PRAGMA foreign_keys=ON')
        if not self.db.execute("SELECT 1 FROM sqlite_master WHERE name='meta'").fetchone():
            self.db.executescript(SCHEMA)
        for key, value in dict(version=VERSION, pointSpacingVoxels=SPACING,
                               mergeGapVoxels=GAP_CAP).items():
            old = self.db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
            if old and json.loads(old[0]) != value:
                raise ValueError(f'Catalogue configuration mismatch: {key}')
            if not old:
                self.db.execute('INSERT INTO meta VALUES(?,?)', (key, jdump(value)))
        self.db.commit()
        self._cache = {}

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def source(self, sid):
        sid = int(sid)
        if sid not in self._cache:
            row = self.db.execute('SELECT * FROM sources WHERE id=?', (sid,)).fetchone()
            if row is None:
                raise KeyError(sid)
            value = dict(row)
            value['geometry'] = unpack(row['points'])
            value['endpoint_data'] = unpack_endpoints(row['endpoint_data'])
            self._cache[sid] = value
        return self._cache[sid]

    def nearby(self, low, high, family=None, representatives=False):
        params = [float(high[0]),float(low[0]),float(high[1]),float(low[1]),float(high[2]),float(low[2])]
        where = 'r.x0<=? AND r.x1>=? AND r.y0<=? AND r.y1>=? AND r.z0<=? AND r.z1>=?'
        if family is not None:
            where += ' AND s.family=?'; params.append(int(family))
        if representatives:
            where += ' AND s.alias IS NULL'
        # CROSS JOIN fixes the traversal order: filter the 3D R-tree first.
        # A reorderable JOIN makes SQLite start with source_alias(NULL), which
        # scans nearly every source before testing one tiny bridge bbox.
        return [int(r[0]) for r in self.db.execute(
            'SELECT s.id FROM source_bbox r CROSS JOIN sources s ON s.id=r.id WHERE '+where, params)]

    def nearby_endpoints(self, point, radius, family):
        low=np.asarray(point)-radius; high=np.asarray(point)+radius
        params=[float(high[0]),float(low[0]),float(high[1]),float(low[1]),float(high[2]),float(low[2]),int(family)]
        return {int(row[0])//2 for row in self.db.execute(
            'SELECT e.id FROM endpoint_bbox e JOIN sources s ON s.id=CAST(e.id/2 AS INTEGER) '
            'WHERE e.x0<=? AND e.x1>=? AND e.y0<=? AND e.y1>=? AND e.z0<=? AND e.z1>=? AND s.family=?',params)}

    def _candidate(self, a, ea, b, eb, kind, score, strict, info):
        assert a < b
        self.db.execute('INSERT OR REPLACE INTO candidates(a,ea,b,eb,kind,score,strict,info) '
                        'VALUES(?,?,?,?,?,?,?,?)', (a,ea,b,eb,kind,float(score),int(strict),jdump(info)))

    def _compare_pair(self, a, b):
        if a > b:
            a, b = b, a
        aa, bb = self.source(a), self.source(b)
        pa, pb = aa['geometry'], bb['geometry']
        if aa['family'] != bb['family']:
            return
        # No duplicate/overlap reconciliation inside one extractor result: those
        # source curves may be distinct fibres. Gaps can occur within a block.
        if aa['job_id'] != bb['job_id']:
            short, long = (a,b) if (aa['length'], -a) < (bb['length'], -b) else (b,a)
            sp=self.source(short)['geometry']; lp=self.source(long)['geometry']
            slo,shi=bbox(sp); llo,lhi=bbox(lp)
            contained_bbox=bool(np.all(slo>=llo-.65) and np.all(shi<=lhi+.65))
            dup = strict_duplicate(sp,lp) if contained_bbox else None
            if dup is not None:
                self.db.execute('INSERT OR REPLACE INTO duplicates VALUES(?,?,?)',
                                (short, long, jdump(dup)))
                return
            # Reject proximity-only contained neighbours; compare() classifies
            # these as duplicate but only strict_duplicate may suppress them.
            # Any suffix/head overlap must reach one endpoint of each source.
            # The cheap bbox gate avoids constructing projection trees for most
            # spatial neighbours and for endpoint-only gap proposals.
            alo,ahi=bbox(pa); blo,bhi=bbox(pb)
            ends_a=pa[[0,-1]]; ends_b=pb[[0,-1]]
            mutual_tip_bbox=(np.any(np.all((ends_a>=blo-3)&(ends_a<=bhi+3),axis=1)) and
                            np.any(np.all((ends_b>=alo-3)&(ends_b<=ahi+3),axis=1)))
            match = ov.compare(pa, pb) if mutual_tip_bbox else None
            if match and match['kind'] == 'overlap':
                ea, eb = (0 if match['flipA'] else 1), (1 if match['flipB'] else 0)
                match['aCut'] = aa['length']-match['spliceA'] if match['flipA'] else match['spliceA']
                match['bCut'] = bb['length']-match['spliceB'] if match['flipB'] else match['spliceB']
                self._candidate(a,ea,b,eb,'overlap',match['score'],True,match)
        ta, tb = aa['endpoint_data'], bb['endpoint_data']
        for ea in (0,1):
            for eb in (0,1):
                delta = np.asarray(tb[eb]['point']) - ta[ea]['point']
                distance = float(np.linalg.norm(delta))
                if not .25 <= distance <= GAP_CAP:
                    continue
                direction = delta / distance
                ca = float(np.dot(ta[ea]['tangent'], direction))
                cb = float(np.dot(tb[eb]['tangent'], -direction))
                cosine = min(ca, cb)
                angle = float(np.rad2deg(np.arccos(np.clip(cosine,-1,1))))
                lateral = distance * math.sqrt(max(0,1-cosine*cosine))
                mutual = -float(np.dot(ta[ea]['tangent'],tb[eb]['tangent']))
                if angle > 25 or lateral > 4:
                    continue
                score = distance + 8*lateral + .5*angle
                strict = angle <= 12 and lateral <= 2 and mutual >= math.cos(math.radians(15))
                strict = strict and ta[ea]['reliable'] and tb[eb]['reliable']
                info = dict(distance=distance,angleDegrees=angle,lateralVoxels=lateral,
                            points=[ta[ea]['point'],tb[eb]['point']],score=score)
                self._candidate(a,ea,b,eb,'gap',score,strict,info)

    def _aliases(self, dirty):
        # Follow only the spatially discovered duplicate frontier. Direct match
        # to the chosen representative is required (no proximity transitivity).
        component = set(dirty); pending = deque(dirty)
        while pending:
            sid = pending.popleft()
            for row in self.db.execute('SELECT a,b FROM duplicates WHERE a=? OR b=?',(sid,sid)):
                for peer in row:
                    if peer not in component:
                        component.add(peer); pending.append(peer)
        ordered = sorted(component, key=lambda i:(-self.source(i)['length'],i))
        representative = {}; changed = set()
        for sid in ordered:
            options = [int(r[0]) for r in self.db.execute('SELECT b FROM duplicates WHERE a=?',(sid,))
                       if representative.get(int(r[0]),int(r[0])) == int(r[0])]
            parent = min(options,key=lambda i:(-self.source(i)['length'],i)) if options else None
            representative[sid] = sid if parent is None else parent
            old = self.source(sid)['alias']
            if old != parent:
                self.db.execute('UPDATE sources SET alias=? WHERE id=?',(parent,sid))
                self.source(sid)['alias'] = parent
                changed.add(sid)
        return changed

    def _rows_for(self, table, nodes):
        rows = {}
        for sid in nodes:
            for row in self.db.execute(f'SELECT * FROM {table} WHERE a=? OR b=?',(sid,sid)):
                rows[row['id']] = dict(row)
        return rows

    def _options(self, sid, end):
        rows = self.db.execute('SELECT c.* FROM candidates c JOIN sources a ON a.id=c.a '
            'JOIN sources b ON b.id=c.b WHERE a.alias IS NULL AND b.alias IS NULL '
            'AND ((c.a=? AND c.ea=?) OR (c.b=? AND c.eb=?))', (sid,end,sid,end)).fetchall()
        # A measured shared interval takes priority over extrapolated endpoint
        # continuation. Every measured alternative participates in ambiguity.
        overlap = [dict(r) for r in rows if r['kind']=='overlap']
        return sorted(overlap or [dict(r) for r in rows],key=lambda r:(r['score'],r['id']))

    def _chosen(self, row, sid, end):
        options = self._options(sid,end)
        if not options or options[0]['id'] != row['id']:
            return False
        if len(options) == 1:
            return True
        # Two geometrically admissible shared intervals are ambiguous even if
        # one happens to be closer. A later block may reveal a neighbouring
        # fibre; retract rather than treating distance as identity confidence.
        if row['kind']=='overlap':
            return False
        margin=max(4.,.35*row['score'])
        return options[1]['score']-options[0]['score'] >= margin

    def _component(self, start):
        found = {int(start)}; pending = deque(found)
        while pending:
            sid = pending.popleft()
            for row in self.db.execute('SELECT a,b FROM links WHERE a=? OR b=?',(sid,sid)):
                peer = int(row['b'] if row['a']==sid else row['a'])
                if peer not in found:
                    found.add(peer); pending.append(peer)
        return found

    def _gap_clear(self, row):
        info = json.loads(row['info']); points = np.asarray(info['points'])
        bridge = np.linspace(points[0],points[1],max(3,int(math.ceil(info['distance']))+1))
        family = self.source(row['a'])['family']
        lo, hi = bbox(points)
        for sid in self.nearby(lo-3,hi+3,family,True):
            p, s = clean(self.source(sid)['geometry'])
            values = np.linspace(0,s[-1],max(2,int(math.ceil(s[-1]/2))+1))
            if sid == row['a']:
                d = values if row['ea']==0 else s[-1]-values
                values = values[d>4]
            elif sid == row['b']:
                d = values if row['eb']==0 else s[-1]-values
                values = values[d>4]
            if len(values) and np.any(cKDTree(ov._at(p,s,values)).query(bridge)[0] <= 2.5):
                return False
        return True

    def _compatible_cut(self,row,sid,end):
        # Avoid two accepted overlaps consuming a source in reversed order.
        source = self.source(sid)
        incoming = json.loads(row['info'])
        cut = incoming['aCut' if row['a']==sid else 'bCut'] if row['kind']=='overlap' else (0 if end==0 else source['length'])
        other = self.db.execute('SELECT c.* FROM links l JOIN candidates c ON c.id=l.id '
            'WHERE (l.a=? AND l.ea=?) OR (l.b=? AND l.eb=?)',(sid,1-end,sid,1-end)).fetchone()
        other_cut = source['length'] if end==0 else 0.
        if other and other['kind']=='overlap':
            other_info = json.loads(other['info'])
            other_cut = other_info['aCut' if other['a']==sid else 'bCut']
        return other_cut-cut >= 1e-5 if end==0 else cut-other_cut >= 1e-5

    def _reconcile(self, dirty):
        candidate_rows = self._rows_for('candidates',dirty)
        # Choices at the opposite endpoint can change when a new alternative
        # appears. One-hop endpoint expansion, not whole-volume geometry reload.
        affected = set(dirty)
        for row in candidate_rows.values():
            affected.update((row['a'],row['b']))
        candidate_rows = self._rows_for('candidates',affected)
        affected_chains = {r[0] for sid in affected for r in self.db.execute('SELECT chain FROM members WHERE source=?',(sid,))}
        old_members = {r[0] for chain in affected_chains for r in self.db.execute('SELECT source FROM members WHERE chain=?',(chain,))}
        for sid in affected:
            self.db.execute('DELETE FROM links WHERE a=? OR b=?',(sid,sid))
        ordered = sorted(candidate_rows.values(),key=lambda r:(r['kind']!='overlap',r['score'],r['id']))
        possible = []
        for row in ordered:
            if not row['strict'] or self.source(row['a'])['alias'] is not None or self.source(row['b'])['alias'] is not None:
                continue
            if not self._chosen(row,row['a'],row['ea']) or not self._chosen(row,row['b'],row['eb']):
                continue
            if row['kind']=='gap' and not self._gap_clear(row):
                continue
            possible.append(row)
        # Inferred bridges crossing one another are both rejected. Include
        # accepted bridges outside this frontier for consistent incremental use.
        gaps = [r for r in possible if r['kind']=='gap']
        conflict = set(); gap_ids={r['id'] for r in gaps}; external={}
        for row in gaps:
            info=json.loads(row['info']); points=np.asarray(info['points'])
            nearby=self.nearby_endpoints(points.mean(axis=0),GAP_CAP+info['distance']/2+2,
                                          self.source(row['a'])['family'])
            for link in self._rows_for('links',nearby).values():
                if link['id'] in gap_ids or link['id'] in external:
                    continue
                candidate=dict(self.db.execute('SELECT * FROM candidates WHERE id=?',(link['id'],)).fetchone())
                if candidate['kind']=='gap': external[link['id']]=candidate
        for family in (0,1):
            samples=[]; owners=[]
            for row in gaps+list(external.values()):
                if self.source(row['a'])['family']!=family: continue
                info=json.loads(row['info']); points=np.asarray(info['points'])
                samples.append(np.linspace(points[0],points[1],max(3,int(np.ceil(info['distance']))+1)))
                owners.extend([row['id']]*len(samples[-1]))
            if samples:
                ids=np.asarray(owners,dtype='i8')
                pairs=cKDTree(np.concatenate(samples)).query_pairs(2.,output_type='ndarray')
                if len(pairs):
                    different=ids[pairs[:,0]]!=ids[pairs[:,1]]
                    conflict.update(int(i) for i in ids[pairs[different]].ravel())
        for linkid,row in external.items():
            if linkid in conflict:
                self.db.execute('DELETE FROM links WHERE id=?',(linkid,))
                affected.update((row['a'],row['b']))
        for row in possible:
            if row['id'] in conflict:
                continue
            used = self.db.execute('SELECT 1 FROM links WHERE (a=? AND ea=?) OR (b=? AND eb=?) '
                'OR (a=? AND ea=?) OR (b=? AND eb=?) LIMIT 1',
                (row['a'],row['ea'],row['a'],row['ea'],row['b'],row['eb'],row['b'],row['eb'])).fetchone()
            if used or row['b'] in self._component(row['a']):
                continue
            if not self._compatible_cut(row,row['a'],row['ea']) or not self._compatible_cut(row,row['b'],row['eb']):
                continue
            self.db.execute('INSERT OR IGNORE INTO links VALUES(?,?,?,?,?)',
                            (row['id'],row['a'],row['ea'],row['b'],row['eb']))
        todo = old_members | affected
        for sid in list(todo):
            todo.update(self._component(sid))
        # Include previous chains of newly reached neighbours to remove stale
        # summaries before splitting/rejoining them.
        stale_chains = {r[0] for sid in todo for r in self.db.execute('SELECT chain FROM members WHERE source=?',(sid,))}
        prior_ids={row['signature']:int(row['id']) for chain in stale_chains
                   for row in self.db.execute('SELECT id,signature FROM chains WHERE id=?',(chain,))}
        for chain in stale_chains:
            todo.update(r[0] for r in self.db.execute('SELECT source FROM members WHERE chain=?',(chain,)))
            self.db.execute('DELETE FROM members WHERE chain=?',(chain,))
            self.db.execute('DELETE FROM chains WHERE id=?',(chain,))
        done = set()
        for sid in sorted(todo):
            if sid in done or self.source(sid)['alias'] is not None:
                continue
            component = self._component(sid); done.update(component)
            assembled = self._assemble(component,geometry=False)
            signature = hashlib.sha256(jdump([assembled['sourceIds'],sorted(self._rows_for('links',component))]).encode()).hexdigest()
            cursor = self.db.execute('INSERT INTO chains(id,family,point_count,length,observed,gap,seam,source_count,bbox,signature) '
                'VALUES(?,?,?,?,?,?,?,?,?,?)',(prior_ids.get(signature),assembled['family'],assembled['pointCount'],assembled['lengthVoxels'],
                assembled['observedLengthVoxels'],assembled['gapLengthVoxels'],assembled['seamLengthVoxels'],
                len(component),jdump(assembled['bboxXYZ']),signature))
            chainid = cursor.lastrowid
            self.db.executemany('INSERT INTO members VALUES(?,?)',[(member,chainid) for member in component])

    def _assemble(self, component, geometry=True):
        edges = self._rows_for('links',component)
        adjacency = {}
        for row in edges.values():
            candidate = dict(self.db.execute('SELECT * FROM candidates WHERE id=?',(row['id'],)).fetchone())
            candidate['parsed'] = json.loads(candidate['info'])
            adjacency[(row['a'],row['ea'])] = candidate
            adjacency[(row['b'],row['eb'])] = candidate
        tips = [(sid,e) for sid in component for e in (0,1) if (sid,e) not in adjacency]
        if len(tips) != 2:
            raise ValueError('Accepted component is not a simple open chain')
        sid, entering = min(tips)
        pieces=[]; bridges=[]; sourceids=[]; observed=0.; gap=0.; seam=0.; incoming=None
        low=np.full(3,np.inf); high=np.full(3,-np.inf)
        while True:
            source = self.source(sid); p,s = clean(source['geometry'])
            outgoing = adjacency.get((sid,1-entering))
            start = 0. if entering==0 else s[-1]
            stop = s[-1] if entering==0 else 0.
            if incoming and incoming['kind']=='overlap':
                start=incoming['parsed']['aCut' if incoming['a']==sid else 'bCut']
            if outgoing and outgoing['kind']=='overlap':
                stop=outgoing['parsed']['aCut' if outgoing['a']==sid else 'bCut']
            a,b=sorted((start,stop)); values=np.r_[a,s[(s>a+1e-7)&(s<b-1e-7)],b]
            piece=ov._at(p,s,values)
            if entering==1:
                piece=piece[::-1]
            observed += b-a; low=np.minimum(low,piece.min(axis=0)); high=np.maximum(high,piece.max(axis=0))
            if pieces:
                previous=pieces[-1][-1]; distance=float(np.linalg.norm(piece[0]-previous))
                kind='gap' if incoming['kind']=='gap' else 'overlap-seam'
                bridges.append(dict(kind=kind,points=[previous.tolist(),piece[0].tolist()],lengthVoxels=distance,
                                    sourceIds=[sourceids[-1],sid],evidence=incoming['parsed']))
                if kind=='gap': gap+=distance
                else: seam+=distance
            pieces.append(piece); sourceids.append(sid)
            if outgoing is None:
                break
            nextsid=outgoing['b'] if outgoing['a']==sid else outgoing['a']
            entering=outgoing['eb'] if outgoing['b']==nextsid else outgoing['ea']
            sid=nextsid; incoming=outgoing
            if len(sourceids)>len(component):
                raise ValueError('Cycle in accepted component')
        total=observed+gap+seam
        result=dict(family=self.source(sid)['family'],pointCount=point_count(total),
                    observedPointCount=point_count(observed),lengthVoxels=total,
                    observedLengthVoxels=observed,gapLengthVoxels=gap,seamLengthVoxels=seam,
                    sourceIds=sourceids,bboxXYZ=[low.tolist(),high.tolist()],gaps=bridges)
        if geometry:
            result['points']=np.concatenate(pieces).astype('f4').tolist()
        return result

    def ingest(self, job_id, origin_xyz, shape_zyx, curves):
        """Import one block atomically.

        ``curves`` holds ``(family, points, point_count)`` with ``points`` in
        block-local XYZ voxels; ``origin_xyz`` places the block in the volume.
        """
        self._cache.clear()
        jobid=str(job_id)
        n=len(curves)
        points=[np.asarray(c[1],dtype='<f4').reshape(-1,3) for c in curves]
        families=np.asarray([int(c[0]) for c in curves],dtype=int)
        counts=np.asarray([int(c[2]) for c in curves],dtype=int)
        if any(len(p)<2 or not np.isfinite(p).all() for p in points):
            raise ValueError('Invalid points')
        if np.any(~np.isin(families,[0,1])):
            raise ValueError('Invalid curve metadata')
        origin=np.asarray(origin_xyz,dtype=float)
        shape=np.asarray(shape_zyx,dtype=int)[::-1]
        if origin.shape!=(3,) or shape.shape!=(3,) or np.any(shape<1) or not np.isfinite(origin).all():
            raise ValueError('Invalid input coordinate frame')
        if n and (min(float(p.min()) for p in points)<-.01 or
                  np.any(np.max([p.max(axis=0) for p in points],axis=0)>shape-1+.01)):
            raise ValueError('Result points escape declared input bounds')
        receipt=dict(job=dict(id=jobid,inputOriginXYZ=origin.tolist(),shapeZYX=[int(v) for v in shape_zyx]))
        inserted=[]; dirty=set(); pairs=set()
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.db.execute('SELECT 1 FROM jobs WHERE id=?',(jobid,)).fetchone():
                raise ValueError(f'Block {jobid} was already imported')
            digest=hashlib.sha256(b''.join(p.tobytes() for p in points)).hexdigest()
            self.db.execute('INSERT INTO jobs VALUES(?,?,?,?,?)',(jobid,digest,jdump(receipt),0.,n))
            for i,local in enumerate(points):
                p,arc=clean(local.astype('f8')+origin)
                length=float(arc[-1]); count=point_count(length)
                if count<5:
                    continue
                tips,tangents,reliable=endpoints([p])
                edata=[dict(point=tips[e].tolist(),tangent=tangents[e].tolist(),reliable=bool(reliable[e])) for e in (0,1)]
                cur=self.db.execute('INSERT INTO sources(job_id,source_index,family,points,length,count,original_count,endpoint_data) '
                    'VALUES(?,?,?,?,?,?,?,?)',(jobid,i,int(families[i]),pack(p),length,count,int(counts[i]),pack_endpoints(edata)))
                sid=int(cur.lastrowid); inserted.append(sid)
                low,high=bbox(p)
                self.db.execute('INSERT INTO source_bbox VALUES(?,?,?,?,?,?,?)',
                    (sid,float(low[0]),float(high[0]),float(low[1]),float(high[1]),float(low[2]),float(high[2])))
                for e in (0,1):
                    x,y,z=map(float,tips[e]); self.db.execute('INSERT INTO endpoint_bbox VALUES(?,?,?,?,?,?,?)',(2*sid+e,x,x,y,y,z,z))
                neighbors=self.nearby(low-GAP_CAP-3,high+GAP_CAP+3,int(families[i]))
                dirty.update(neighbors)
                # Candidate geometry is substantially narrower than the dirty
                # obstruction frontier: shared bboxes or two nearby endpoints.
                compare_peers=set(self.nearby(low-3,high+3,int(families[i])))
                for tip in tips:
                    compare_peers.update(self.nearby_endpoints(tip,GAP_CAP,int(families[i])))
                for peer in compare_peers:
                    if peer<sid:
                        pairs.add((peer,sid))
            for a,b in sorted(pairs):
                self._compare_pair(a,b)
            dirty.update(self._aliases(set(inserted)))
            self._reconcile(dirty)
            self.db.commit()
        except BaseException:
            self.db.rollback(); self._cache.clear(); raise
        self._cache.clear()
        return dict(jobId=jobid,insertedFragments=len(inserted),comparedPairs=len(pairs),frontierFragments=len(dirty))

    @staticmethod
    def _summary(row):
        return dict(id=int(row['id']),family=int(row['family']),pointCount=int(row['point_count']),
                    observedPointCount=point_count(row['observed']),lengthVoxels=row['length'],
                    observedLengthVoxels=row['observed'],gapLengthVoxels=row['gap'],seamLengthVoxels=row['seam'],
                    sourceCount=int(row['source_count']),bboxXYZ=json.loads(row['bbox']))

    @read_snapshot
    def get_curve(self,chain_id,include_provenance=True):
        row=self.db.execute('SELECT * FROM chains WHERE id=?',(int(chain_id),)).fetchone()
        if row is None: return None
        members={int(r[0]) for r in self.db.execute('SELECT source FROM members WHERE chain=?',(int(chain_id),))}
        result=self._summary(row); result.update(self._assemble(members))
        if include_provenance:
            result['sources']=[dict(id=sid,jobId=self.source(sid)['job_id'],sourceIndex=self.source(sid)['source_index'],
                                   pointCount=self.source(sid)['count']) for sid in result.pop('sourceIds')]
            result['duplicateSources']=[dict(id=int(r['id']),jobId=r['job_id'],sourceIndex=int(r['source_index']),
                                             representativeId=sid)
                                        for sid in members for r in self.db.execute(
                                            'SELECT id,job_id,source_index FROM sources WHERE alias=?',(sid,))]
            job_ids={s['jobId'] for s in result['sources']+result['duplicateSources']}
            result['provenance']=[json.loads(r[0]) for jid in sorted(job_ids)
                                  for r in self.db.execute('SELECT receipt FROM jobs WHERE id=?',(jid,))]
        else:
            result.pop('sourceIds')
        self._cache.clear()
        return result
