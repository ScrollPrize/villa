"""CPU-only original-fiber OBJ and signed-normal mesh PLY export."""
import json
from pathlib import Path
import sqlite3
import numpy as np
from lasagna.inference_provenance import atomic_write, sha256_file
from vesuvius.neural_tracing.fiber_follow.data.afv import read_afv_polylines
from .fiber_normal_convention import inward_solution


def spatial_subset(ids, centers, count):
    """Deterministic coverage sample, equal axis weight (not random statistics)."""
    ids=np.asarray(ids,dtype=np.int64);centers=np.asarray(centers,dtype=float)
    if count<1 or not len(ids) or centers.shape!=(len(ids),3) or not np.isfinite(centers).all():
        raise ValueError('Invalid spatial subset inputs')
    extent=np.ptp(centers,axis=0)
    points=(centers-centers.min(axis=0))/np.where(extent>0,extent,1.)
    chosen=[];distance=np.full(len(ids),np.inf)
    next_index=int(np.argmax(np.sum((points-.5)**2,axis=1)))
    for _ in range(min(count,len(ids))):
        chosen.append(next_index)
        distance=np.minimum(distance,np.sum((points-points[next_index])**2,axis=1))
        distance[chosen]=-1
        next_index=int(np.argmax(distance))
    return ids[chosen].tolist()


def obj_edges(first_vertex, count):
    """Pairwise OBJ lines: MeshLab may only load the first pair of a long l row."""
    return ''.join(f'l {i} {i+1}\n' for i in range(first_vertex,first_vertex+count-1))


def obj_fiber(fid, points, first_vertex, broken):
    return (f'o fiber_{fid}\nusemtl {"broken" if broken else "retained"}\n'
            + ''.join('v '+' '.join(f'{x:.9f}' for x in p)+'\n' for p in points)
            + obj_edges(first_vertex,len(points)))


def repair_obj_lines(config):
    root=Path(config['output_root'])
    document=json.loads((root/'index.json').read_text());records=document['records']
    source=document.get('config',{}).get('afv')
    raw=None
    if source:
        raw={int(row['id']):points for row,points in read_afv_polylines(source,[r['fiber_id'] for r in records])}
        first=1
        for r in records:
            r['obj_first_vertex']=first;r['obj_vertex_count']=len(raw[r['fiber_id']]);first+=r['obj_vertex_count']
    result={}
    for name,chosen in [('fibers.obj',records),('broken_fibers.obj',[r for r in records if r['broken']])]:
        path=root/name;lines=path.read_text().splitlines(keepends=True)
        if raw is not None:
            first=1;parts=['# Original full AFV curves; explicit pairwise edges\nmtllib fibers.mtl\n']
            for r in chosen:
                points=raw[r['fiber_id']]
                parts.append(obj_fiber(r['fiber_id'],points,first,r['broken']));first+=len(points)
            lines=''.join(parts).splitlines(keepends=True)
        output=[];edges=[]
        for line in lines:
            fields=line.split()
            if fields and fields[0]=='l':
                indices=list(map(int,fields[1:]))
                if len(indices)<2:raise ValueError('Invalid OBJ line')
                for a,b in zip(indices,indices[1:]):
                    output.append(f'l {a} {b}\n');edges.append((a,b))
            else:output.append(line)
        expected=[];first=1
        for r in chosen:
            count=r['obj_vertex_count']
            expected.extend((i,i+1) for i in range(first,first+count-1));first+=count
        if edges!=expected or sum(line.startswith('v ') for line in lines)!=first-1:
            raise ValueError(f'{name}: geometry differs from recorded fiber ranges')
        temporary=path.with_suffix('.obj.partial')
        if temporary.exists():raise FileExistsError(temporary)
        temporary.write_text(''.join(output));temporary.replace(path)
        result[name]=dict(fibers=len(chosen),vertices=first-1,segments=len(edges))
    if raw is not None:atomic_write(root/'index.json',document,durable=False)
    atomic_write(root/'obj_segment_validation.json',result,durable=False)
    print(json.dumps(result,indent=2),flush=True)
    return result


def arrow_mesh(point, normal, length=64., radius=2., sides=6):
    p=np.asarray(point,dtype=float);n=np.asarray(normal,dtype=float)
    norm=np.linalg.norm(n)
    if not np.isfinite(norm) or norm<=0 or length<=0 or radius<=0:
        raise ValueError('Invalid arrow geometry')
    n=n/norm
    axis=np.eye(3)[np.argmin(abs(n))]
    u=np.cross(n,axis);u/=np.linalg.norm(u);v=np.cross(n,u)
    angles=np.arange(sides)*2*np.pi/sides
    ring=np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v
    shoulder=p+.75*length*n
    vertices=np.concatenate([p+radius*ring,shoulder+radius*ring,
                             shoulder+3*radius*ring,(p+length*n)[None,:]])
    faces=[]
    for k in range(sides):
        t=(k+1)%sides
        faces.extend([(k,t,sides+k),(t,sides+t,sides+k),(2*sides+k,2*sides+t,3*sides)])
    return vertices,np.asarray(faces,dtype=np.int32)


def write_ply(path, vertices, faces, colors):
    v=np.empty(len(vertices),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for d,key in enumerate(('x','y','z')):v[key]=vertices[:,d]
    for d,key in enumerate(('red','green','blue')):v[key]=colors[:,d]
    f=np.empty(len(faces),dtype=[('count','u1'),('indices','<i4',(3,))]);f['count']=3;f['indices']=faces
    header=(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n'
            'property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n'
            f'element face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n')
    with path.open('wb') as stream:
        stream.write(header.encode());stream.write(v.tobytes());stream.write(f.tobytes())


def update_arrows(config):
    """Update existing preview only; reuse indexed positions, not the database scan."""
    root=Path(config['output_root']);document=json.loads((root/'index.json').read_text())
    source=Path(document['config']['solution'])
    if sha256_file(source)!=document['solution_sha256']:raise ValueError('Solution changed')
    solution=inward_solution(np.load(source,allow_pickle=False))
    target=source if source.stem.endswith('_inward') else source.with_name(source.stem+'_inward.npz')
    if target!=source:
        if target.exists():raise FileExistsError(target)
        with target.with_suffix('.npz.partial').open('wb') as stream:np.savez(stream,**solution)
        target.with_suffix('.npz.partial').replace(target)
    lookup={int(fid):i for i,fid in enumerate(solution['ids'])}
    old_convention=document.get('normal_convention','outward')
    if old_convention not in ('inward','outward'):raise ValueError('Unknown indexed convention')
    length=float(config['arrow_length']);radius=float(config['arrow_radius'])
    verts=[];faces=[];colors=[];offset=0
    for record in document['records']:
        k=lookup[record['fiber_id']]
        record['normal_sign']=int(solution['normal_sign'][k])
        record['sign_probability']=float(solution['sign_probability'][k])
        for arrow in record['arrows']:
            n=np.asarray(arrow['normal'])*(1 if old_convention=='inward' else -1)
            arrow['normal']=n.tolist();arrow['display_normal']=(-n).tolist()
            vv,ff=arrow_mesh(arrow['point'],-n,length,radius)
            arrow['ply_first_vertex']=offset;arrow['ply_vertex_count']=len(vv)
            verts.append(vv);faces.append(ff+offset);offset+=len(vv)
            colors.append(np.tile([255,30,30] if record['broken'] else [30,190,245],(len(vv),1)))
    if not verts:raise ValueError('No arrows to update')
    temporary=root/'signed_normals.ply.partial'
    write_ply(temporary,np.concatenate(verts),np.concatenate(faces),np.concatenate(colors))
    temporary.replace(root/'signed_normals.ply')
    document['normal_convention']='inward';document['display_normal_convention']='outward'
    document['config'].update(solution=str(target),arrow_length=length,arrow_radius=radius)
    document['solution_sha256']=sha256_file(target)
    atomic_write(root/'index.json',document,durable=False)
    (root/'README.txt').write_text('Open fibers.obj and signed_normals.ply together; keep fibers.mtl beside OBJ.\n'
        'Signed normals in index/solution point inward; displayed arrows point outward (negative stored normal).\n'
        f'Arrow length {length:g}, shaft radius {radius:g} base voxels. Fiber geometry unchanged.\n'
        'Orientation is inferred using a weak radial prior, not guaranteed pointwise towards the umbilicus in folds.\n')
    result=dict(fibers=document['fibers'],arrows=document['arrows'],arrow_length=length,arrow_radius=radius,
                normal_convention='inward',display_normal_convention='outward',solution=str(target))
    print(json.dumps(result,indent=2),flush=True)
    return result


def export(config):
    root=Path(config['output_root'])
    if root.exists():raise FileExistsError('Choose a new visualization directory')
    solution_path=Path(config['solution']);solution=inward_solution(np.load(solution_path,allow_pickle=False))
    ids=solution['ids'];lookup={int(fid):i for i,fid in enumerate(ids)}
    if len(lookup)!=len(ids) or not np.isin(solution['normal_sign'],[-1,1]).all():
        raise ValueError('Duplicate solution IDs or invalid signs')
    dbpath=Path(config['database'])
    summary=json.loads((solution_path.parent/'summary.json').read_text())
    if sha256_file(dbpath)!=summary['input']['signature']['database_sha256']:
        raise ValueError('Solution and normal database do not match')
    db=sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro',uri=True)
    afvpath=Path(config['afv'])
    afv=sqlite3.connect(afvpath.resolve().as_uri()+'?mode=ro',uri=True)
    get_meta=lambda c,k:json.loads(c.execute('SELECT value FROM metadata WHERE key=?',(k,)).fetchone()[0])
    if get_meta(db,'provenance')['afv_uuid']!=get_meta(afv,'uuid'):
        raise ValueError('Wrong source AFV')
    selected=config.get('fiber_ids')
    bounds=config.get('bounds_xyz')
    focus=None
    selection_semantics='AFV block-AABB intersection candidates; full source curves retained, NOT clipped'
    if config.get('sample_count') is not None:
        if selected is not None or bounds is not None:raise ValueError('Use sample_count OR explicit IDs/bounds')
        rows=afv.execute('SELECT id,(min_x+max_x)/2.,(min_y+max_y)/2.,(min_z+max_z)/2.,length FROM fibers ORDER BY id').fetchall()
        rows=[r for r in rows if r[0] in lookup and (not config.get('exclude_broken',False) or not solution['broken'][lookup[r[0]]])]
        pool=config.get('longest_candidate_count')
        if pool is not None:
            if int(pool)<int(config['sample_count']):raise ValueError('Longest candidate count must cover requested sample count')
            rows=sorted(sorted(rows,key=lambda r:(-r[4],r[0]))[:int(pool)],key=lambda r:r[0])
        selected=spatial_subset([r[0] for r in rows],[r[1:4] for r in rows],int(config['sample_count']))
        selection_semantics='Whole-scroll deterministic farthest-point sample of fiber bbox centers, normalized per XYZ axis; whole fibers, no clipping'
        if pool is not None:selection_semantics+=f'; candidates restricted to {int(pool)} longest eligible fibers by AFV arc length'
    if selected is None:
        if bounds is None:
            # Highest-degree broken fiber provides a reproducible diagnostic region.
            tensors=np.load(summary['config']['tensor_cache'],allow_pickle=False)
            degree=np.bincount(np.concatenate([tensors['i'],tensors['j']]),minlength=len(ids))
            candidates=np.flatnonzero(solution['broken'] & (degree>0))
            if not len(candidates):candidates=np.flatnonzero(degree>0)
            focus=int(ids[candidates[np.argmax(degree[candidates])]])
            pos=db.execute('SELECT ax,ay,az FROM constraints WHERE fiber_a=? UNION ALL SELECT bx,by,bz FROM constraints WHERE fiber_b=? LIMIT 1',(focus,focus)).fetchone()
            center=np.asarray(pos);half=float(config.get('crop_size',512))/2
            bounds=[(center-half).tolist(),(center+half).tolist()]
        lo,hi=np.asarray(bounds,dtype=float)
        if not np.isfinite([lo,hi]).all() or np.any(lo>=hi):raise ValueError('Invalid bbox')
        selected=[r[0] for r in afv.execute('SELECT DISTINCT b.fiber_id FROM block_bounds r JOIN blocks b ON b.id=r.id '
            'WHERE r.max_x>=? AND r.min_x<=? AND r.max_y>=? AND r.min_y<=? AND r.max_z>=? AND r.min_z<=? ORDER BY b.fiber_id',
            (lo[0],hi[0],lo[1],hi[1],lo[2],hi[2]))]
    selected=sorted(set(map(int,selected)))
    if config.get('exclude_broken',False):
        selected=[fid for fid in selected if fid in lookup and not solution['broken'][lookup[fid]]]
    if not selected or len(selected)>config.get('max_fibers',10000):raise ValueError(f'Invalid/oversized selection: {len(selected)} fibers')
    if not all(fid in lookup for fid in selected):raise ValueError('Selection contains fibers absent from solution')
    spacing=float(config.get('arrow_spacing',256));length=float(config.get('arrow_length',64))
    if spacing<=0 or length<=0:raise ValueError('Positive arrow spacing/length required')
    raw=read_afv_polylines(afvpath,selected)
    if sum(len(points) for _,points in raw)>config.get('max_points',1000000):
        raise ValueError('Selection exceeds point limit; use fewer fibers')
    root.mkdir(parents=True)
    (root/'fibers.mtl').write_text('newmtl retained\nKd 0.55 0.65 0.75\nnewmtl broken\nKd 1.0 0.1 0.1\n')
    verts=[];triangles=[];colors=[];records=[];offset=0;line_vertex=1
    with (root/'fibers.obj').open('w') as obj, (root/'broken_fibers.obj').open('w') as broken_obj:
        broken_vertex=1
        broken_obj.write('# Broken fiber overlay; manually color this layer red if viewer ignores OBJ line materials\nmtllib fibers.mtl\nusemtl broken\n')
        obj.write('# Original full AFV curves; native L0 XYZ/base voxels\nmtllib fibers.mtl\n')
        for row,points in raw:
            fid=int(row['id']);k=lookup[fid];broken=bool(solution['broken'][k]);sign=int(solution['normal_sign'][k])
            obj.write(obj_fiber(fid,points,line_vertex,broken))
            if broken:
                broken_obj.write(obj_fiber(fid,points,broken_vertex,True))
                broken_vertex+=len(points)
            rec=dict(fiber_id=fid,broken=broken,normal_sign=sign,
                sign_probability=float(solution['sign_probability'][k]),broken_probability=float(solution['broken_probability'][k]),
                obj_first_vertex=line_vertex,obj_vertex_count=len(points),arrows=[])
            line_vertex+=len(points)
            rows=db.execute('SELECT id,arc_a,ax,ay,az,normal_a FROM constraints WHERE fiber_a=? '
                'UNION ALL SELECT id,arc_b,bx,by,bz,normal_b FROM constraints WHERE fiber_b=? ORDER BY 2,1',(fid,fid))
            last=-np.inf
            for event,arc,x,y,z,normal_json in rows:
                normal=json.loads(normal_json)
                if not normal['valid'] or arc-last<spacing:continue
                last=arc;p=np.asarray([x,y,z]);n=sign*np.asarray(normal['xyz'])
                if len(verts)>=config.get('max_arrows',50000):raise ValueError('Too many arrows; increase spacing or reduce selection')
                vv,ff=arrow_mesh(p,-n,length,float(config.get('arrow_radius',2)))
                color=np.array([255,30,30] if broken else [30,190,245],dtype=np.uint8)
                rec['arrows'].append(dict(event_id=event,arc=arc,point=p.tolist(),normal=n.tolist(),display_normal=(-n).tolist(),
                    ply_first_vertex=offset,ply_vertex_count=len(vv)))
                verts.append(vv);triangles.append(ff+offset);colors.append(np.tile(color,(len(vv),1)));offset+=len(vv)
            records.append(rec)
    vertices=np.concatenate(verts) if verts else np.empty((0,3))
    faces=np.concatenate(triangles) if triangles else np.empty((0,3),dtype=np.int32)
    color=np.concatenate(colors) if colors else np.empty((0,3),dtype=np.uint8)
    write_ply(root/'signed_normals.ply',vertices,faces,color)
    report=dict(config=config,solution_sha256=sha256_file(solution_path),focus_fiber=focus,selection_bbox=bounds,
        selection_semantics=selection_semantics,
        normal_convention='inward',display_normal_convention='outward',
        units='native L0 XYZ/base voxels',fibers=len(records),broken_fibers=sum(r['broken'] for r in records),
        arrows=sum(len(r['arrows']) for r in records),mesh_vertices=len(vertices),mesh_faces=len(faces),records=records)
    atomic_write(root/'index.json',report,durable=False)
    (root/'README.txt').write_text('Open fibers.obj and signed_normals.ply together in MeshLab. Keep fibers.mtl beside OBJ.\n'
        'Optional broken_fibers.obj overlay isolates broken fibers if your viewer ignores OBJ line material colors.\n'
        'Blue arrows: retained fibers. Red lines/arrows: model-marked broken fibers. Arrows are sheet normals, not tangents.\n'
        'Stored normals point inward; displayed arrows point outward (negative stored normal).\n'
        'Arrow length is display-only. Coordinates are native XYZ/base voxels. Full selected curves, no clipping.\n'
        'index.json maps original fiber IDs, probabilities, arcs and mesh/line indices.\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','config')},indent=2),flush=True)
    return report
