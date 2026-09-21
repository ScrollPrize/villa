"""SLIM chart construction and constraint assembly for winding workspaces.

Each source keeps its parameter-domain connectivity. Shared physical contacts
couple charts; vertical centerlines share one horizontal degree of freedom.
"""
from dataclasses import dataclass
from pathlib import Path
import os
import json
import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay, ConvexHull, QhullError, cKDTree
from winding_arclength import source_grid
from winding_topology import source_quad_mask, polish_uv_orientation
from winding_straightening import collect_guides
from winding_contacts import closest_triangle, overlap_witnesses


@dataclass
class Chart:
    source: dict
    start: int
    points: np.ndarray
    domain: np.ndarray
    triangulation: object
    faces: np.ndarray
    initial: np.ndarray
    rest: np.ndarray
    arc: object = None
    columns: object = None
    frames: object = None
    mass: object = None
    correction_field: object = None

    def weights(self, q):
        q = np.atleast_2d(q)
        if self.source['fiber']:
            row = np.clip(q[:, 0], 0, 2)
            ri = np.minimum(row.astype(int), 1)
            column = np.clip(q[:, 1], 0, self.source['cols']-1)
            arc = np.interp(column, np.arange(len(self.arc)), self.arc)
            knot = self.arc[self.columns]
            ci = np.searchsorted(knot, arc, side='right').clip(1, len(knot)-1)-1
            t = (arc-knot[ci])/np.maximum(knot[ci+1]-knot[ci], 1e-20)
            r = row-ri
            ix = np.stack((ri*len(knot)+ci,ri*len(knot)+ci+1,(ri+1)*len(knot)+ci,(ri+1)*len(knot)+ci+1),axis=1)
            diagonal=np.maximum(r+t-1.,0.)
            w = np.stack((np.maximum(1-r-t,0.),t-diagonal,r-diagonal,diagonal),axis=1)
        else:
            if self.triangulation is None:
                ix=cKDTree(self.domain).query(q)[1][:,None]
                return ix+self.start,np.ones_like(ix,dtype=float)
            simplex = self.triangulation.find_simplex(q, tol=1e-7)
            if np.any(simplex < 0):
                # The complete valid-domain convex hull is retained. Roundoff
                # or attachment observations half a source cell outside it use
                # the nearest boundary triangle for a local extrapolation.
                missing = simplex < 0
                tree = cKDTree(self.domain[self.triangulation.simplices].mean(axis=1))
                simplex[missing] = tree.query(q[missing])[1]
            transform = self.triangulation.transform[simplex]
            ab = np.einsum('nij,nj->ni',transform[:,:2],q-transform[:,2])
            w = np.column_stack((ab,1-ab.sum(axis=1)))
            ix = self.triangulation.simplices[simplex]
        return ix+self.start,w

    def map(self, uv, q):
        ix,w=self.weights(q)
        placed=np.sum(uv[ix]*w[:,:,None],axis=1)
        if self.correction_field is not None:
            q=np.atleast_2d(q);shape=np.array(self.correction_field.shape[:2]);q=np.clip(q,0,shape-1)
            cell=np.minimum(q.astype(int),shape-2);r,t=(q-cell).T;diagonal=np.maximum(r+t-1.,0.)
            weights=np.column_stack((np.maximum(1-r-t,0.),t-diagonal,r-diagonal,diagonal))
            for k,offset in enumerate(((0,0),(0,1),(1,0),(1,1))):
                ij=cell+offset;placed+=self.correction_field[ij[:,0],ij[:,1]]*weights[:,k,None]
        return placed


def make_charts(payload, sources, spacing=80., progress=print, initialize_from_saved=False):
    guides, guide_ids, missing = collect_guides(payload,sources,allow_source_z_reference=True)
    guide_lookup = dict(zip(guide_ids, guides.heights))
    charts=[]; omitted=[];count=0
    for number,source in enumerate(sources):
        if source.get('quarantined', False):
            continue
        grid=source_grid(payload,source)
        if source['fiber']:
            center=grid[1]
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(center[:,:3],axis=0),axis=1))]
            cols=np.unique(np.r_[np.searchsorted(arc,np.arange(0.,arc[-1],spacing)),len(arc)-1])
            if len(cols)<2:
                omitted.append(source['id']);continue
            points=grid[:,cols].reshape(-1,5).copy()
            rr,cc=np.meshgrid(np.arange(3),cols,indexing='ij');domain=np.column_stack((rr.ravel(),cc.ravel()))
            width=np.median(np.linalg.norm(grid[0,:,:3]-grid[1,:,:3],axis=1))
            width=max(width,.01)
            rest=np.zeros((3,len(cols),3));rest[:,:,0]=np.arange(-1,2)[:,None]*width;rest[:,:,1]=arc[cols][None,:]
            vertical=source.get('direction')=='V'
            if vertical:
                target=guide_lookup[source['id']]
                if center[-1,2]<center[0,2]:target=target[::-1]
                uv=np.empty((3,len(cols),2));uv[:,:,0]=np.median(center[:,3])+np.arange(-1,2)[:,None]*width
                uv[:,:,1]=target[cols][None,:]
            else:
                uv=np.empty((3,len(cols),2));sign=1 if center[-1,3]>=center[0,3] else -1
                uv[:,:,0]=np.median(center[:,3])+sign*(arc[cols]-np.median(arc))[None,:]
                uv[:,:,1]=np.median(guides.row_reference(center[:,2]))+np.arange(-1,2)[:,None]*width
            triangles=[]
            for row in range(2):
                a=row*len(cols)+np.arange(len(cols)-1);b=a+1;c=a+len(cols);d=c+1
                triangles.extend([np.column_stack((a,b,c)),np.column_stack((b,d,c))])
            faces=np.concatenate(triangles);uv=uv.reshape(-1,2);rest=rest.reshape(-1,3)
            chart=Chart(source,count,points,domain,None,faces,uv,rest,arc,cols)
        else:
            valid=grid[:,:,0]!=-1;ij=np.argwhere(valid)
            if not len(ij):raise ValueError('Empty workspace source')
            try:hull=ConvexHull(ij).vertices
            except QhullError:
                points=grid[ij[:,0],ij[:,1]].copy()
                uv=np.column_stack((points[:,3],guides.row_reference(points[:,2])))
                charts.append(Chart(source,count,points,ij,None,np.empty((0,3),dtype=int),uv,points[:,:3].copy()))
                count+=len(points);continue
            lengths=[]
            for axis in (0,1):
                d=np.diff(grid[:,:,:3],axis=axis)
                mask=(valid[1:]&valid[:-1]) if axis==0 else (valid[:,1:]&valid[:,:-1])
                length=np.linalg.norm(d[mask],axis=1)
                lengths.append(float(np.median(length[length>1e-6])) if np.any(length>1e-6) else 1.)
            steps=np.maximum(1,np.round(spacing/np.array(lengths)).astype(int))
            keep=((ij[:,0]%steps[0])==0)&((ij[:,1]%steps[1])==0);keep[hull]=True
            selected=ij[keep]
            # A control at the domain center limits long skinny hull triangles.
            center=np.mean(ij,axis=0);keep[np.argmin(np.sum((ij-center)**2,axis=1))]=True;selected=ij[keep]
            try:triangulation=Delaunay(selected)
            except QhullError:
                omitted.append(source['id']);continue
            points=grid[selected[:,0],selected[:,1]].copy();faces=triangulation.simplices.copy()
            target=(points[:,3:].copy() if initialize_from_saved else
                    np.column_stack((points[:,3],guides.row_reference(points[:,2]))))
            local=(selected-selected.mean(axis=0))*np.array(lengths)
            u,_,vt=np.linalg.svd(local.T@(target-target.mean(axis=0)))
            uv=local@(u@vt)+target.mean(axis=0)
            chart=Chart(source,count,points,selected,triangulation,faces,uv,points[:,:3].copy())
        if initialize_from_saved:
            saved=chart.points[:,3:].copy()
            f=chart.faces;da=saved[f[:,1]]-saved[f[:,0]];db=saved[f[:,2]]-saved[f[:,0]]
            determinants=da[:,0]*db[:,1]-da[:,1]*db[:,0]
            # Fine-grid validity does not guarantee a nonfolded coarse chord
            # mesh across holes. Keep the safe rigid initialization there.
            if len(determinants) and (np.all(determinants>1e-8) or np.all(determinants<-1e-8)):
                chart.initial=saved
            else:
                # The safe rigid chart still belongs at its saved position.
                # Source Z is only a weak row prior, not a placement reset.
                chart.initial+=saved.mean(axis=0)-chart.initial.mean(axis=0)
        f=chart.faces;a=chart.initial[f[:,1]]-chart.initial[f[:,0]];b=chart.initial[f[:,2]]-chart.initial[f[:,0]]
        area=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        flip=area<0;chart.faces[flip]=chart.faces[flip][:,[0,2,1]]
        charts.append(chart);count+=len(chart.points)
        if number%3000==0:progress(f'Built {number:,}/{len(sources):,} charts; {count:,} controls')
    return charts,guides,dict(omitted_sources=omitted,missing_classifications=missing,spacing=spacing)


def prepare_constraints(charts,guides,evidence=None,contact_weight=1000.,progress=print,payload=None,row_band=40.,overlap_distance=1.,overlap_normal_cos=.95):
    points=np.concatenate([c.points for c in charts]);initial=np.concatenate([c.initial for c in charts]);n=len(points)
    owners=np.concatenate([np.full(len(c.points),i) for i,c in enumerate(charts)])
    faces=np.concatenate([c.faces+c.start for c in charts if not c.source['fiber']])
    face_owner=owners[faces[:,0]]
    centers=np.column_stack((points[faces,:3].mean(axis=1),points[faces,3].mean(axis=1)))
    tree=cKDTree(centers)
    # Limit links to relationships actually accepted by the growth graph when
    # its evidence is available. The phase coordinate also separates windings.
    lookup={c.source['id']:i for i,c in enumerate(charts)}
    component_lookup={}
    for i,c in enumerate(charts):component_lookup.setdefault(c.source.get('component',c.source['id']),[]).append(i)
    allowed=None
    if evidence is not None:
        allowed=set()
        for edge in evidence:
            if edge.get('status')=='consistent':
                for i in component_lookup.get(edge['a'],[]):
                    for j in component_lookup.get(edge['b'],[]):
                        allowed.add(min(i,j)*len(charts)+max(i,j))
    rows=[];cols=[];values=[];targets=[];links=[];nc=0
    def add(ix,w,jx,v,weight):
        nonlocal nc
        # Every attachment is imposed on both chart coordinates.
        for axis in range(2):
            rows.extend([nc]*(len(ix)+len(jx)));cols.extend((np.r_[ix,jx]+axis*n).tolist())
            values.extend((np.r_[w,-v]*weight).tolist());targets.append(0.);nc+=1
    eligible=np.ones(n,dtype=bool)
    for chart in charts:
        if chart.source['fiber']:
            eligible[chart.start:chart.start+len(chart.points)]=False
            eligible[chart.start+len(chart.columns):chart.start+2*len(chart.columns)]=True
    measured_overlaps=evidence is not None and payload is not None
    if measured_overlaps:
        # Patch contacts below come from accepted fine faces, not coarse chords.
        eligible &= np.array([charts[i].source["fiber"] for i in owners])
    ids=np.flatnonzero(eligible)
    for start in range(0,len(ids),4000):
        indices=ids[start:start+4000];query=points[indices];source=owners[indices]
        if not len(faces):raise ValueError('SLIM requires supported patch triangles')
        _,neighbor=tree.query(np.column_stack((query[:,:3],query[:,3])),k=list(range(1,min(32,len(faces))+1)),workers=4)
        other=face_owner[neighbor];valid=other!=source[:,None]
        if allowed is not None:
            codes=np.minimum(source[:,None],other)*len(charts)+np.maximum(source[:,None],other)
            valid &= np.fromiter((int(k) in allowed for k in codes.ravel()),bool,count=codes.size).reshape(codes.shape)
        d2,w=closest_triangle(query[:,:3],points[faces[neighbor],:3])
        near_u=np.sum(points[faces[neighbor],3]*w,axis=2)
        valid &= (d2<16.)&(np.abs(near_u-query[:,3,None])<16.)
        d2[~valid]=np.inf
        # At most three other charts per control, avoiding redundant cliques.
        for _ in range(3):
            best=d2.argmin(axis=1);take=np.flatnonzero(np.isfinite(d2[np.arange(len(indices)),best]))
            for r in take:
                face=neighbor[r,best[r]];other_source=face_owner[face]
                add(np.array([indices[r]]),np.ones(1),faces[face],w[r,best[r]],np.sqrt(contact_weight))
                links.append((source[r],other_source))
                d2[r,other[r]==other_source]=np.inf
        if start%100000==0:progress(f'Attached {start:,}/{len(ids):,} controls; {nc//2:,} contacts')
    overlap_pairs=0;missing_overlap_pairs=[];overlap_contacts=0
    if measured_overlaps:
        for aid,bid,qa,qb,distances in overlap_witnesses(payload,[c.source for c in charts],evidence,
                distance=overlap_distance,normal_cos=overlap_normal_cos,progress=progress,workers=min(8,os.cpu_count() or 1)):
            if not len(qa):
                missing_overlap_pairs.append([aid,bid]);continue
            a,b=charts[lookup[aid]],charts[lookup[bid]]
            ix,w=a.weights(qa);jx,v=b.weights(qb)
            for k in range(len(qa)):
                add(ix[k],w[k],jx[k],v[k],np.sqrt(contact_weight))
            links.append((lookup[aid],lookup[bid]))
            overlap_pairs+=1;overlap_contacts+=len(qa)
    if evidence is not None:
        # Exact fiber/patch witnesses retain narrow contacts missed by a coarse
        # proximity sample. Acquisition indices preserve reversed-Z traversal.
        for edge in evidence:
            if edge.get('kind')!='contact' or edge.get('status')!='consistent':continue
            member='fiber:'+edge['member']
            if member not in lookup or edge['b'] not in lookup:continue
            a,b=charts[lookup[member]],charts[lookup[edge['b']]]
            if not a.source['fiber'] or b.source['fiber']:continue
            for end in ('from','to'):
                column=edge[end+'_point_id']-a.source['first_point_id']
                if not 0<=column<a.source['cols']:continue
                q=np.asarray(edge[end+'_ij'])-b.source['source_ij_origin']
                ix,w=a.weights([[1.,column]]);jx,v=b.weights(q)
                add(ix[0],w[0],jx[0],v[0],np.sqrt(contact_weight))
                links.append((lookup[member],lookup[edge['b']]))
    if payload is not None:
        fiber_ids=[i for i,c in enumerate(charts) if c.source['fiber']]
        lines=[source_grid(payload,charts[i].source)[1] for i in fiber_ids]
        full=np.concatenate(lines);full_owner=np.repeat(fiber_ids,[len(line) for line in lines])
        full_column=np.concatenate([np.arange(len(line)) for line in lines])
        coordinate=np.column_stack((full[:,:3],full[:,3]));fiber_tree=cKDTree(coordinate)
        components=[c.source.get('component',c.source['id']) for c in charts]
        seen=set()
        for start in range(0,len(full),20000):
            distance,index=fiber_tree.query(coordinate[start:start+20000],k=8,workers=4)
            for row in range(len(distance)):
                p=start+row;i=int(full_owner[p])
                candidates=np.flatnonzero((distance[row]<1.5)&(full_owner[index[row]]!=i))
                for choice in candidates:
                    q=index[row,choice];j=int(full_owner[q])
                    if i>=j or components[i]!=components[j]:continue
                    key=(i,j,int(full_column[p]//32),int(full_column[q]//32))
                    if key in seen:continue
                    seen.add(key)
                    ix,w=charts[i].weights([[1.,full_column[p]]]);jx,v=charts[j].weights([[1.,full_column[q]]])
                    add(ix[0],w[0],jx[0],v[0],np.sqrt(contact_weight));links.append((i,j))
        progress(f'Added {len(seen):,} contacts within accepted fiber components')
    contacts=nc//2
    # Weak height reference; genuine bending may move well outside this row.
    reference=guides.row_reference(points[:,2]);weight=.4/row_band
    rows.extend(range(nc,nc+n));cols.extend(range(n,2*n));values.extend(np.full(n,weight));targets.extend(reference*weight);nc+=n
    C=sparse.coo_matrix((values,(rows,cols)),shape=(nc,2*n)).tocsr()
    dofs=np.arange(2*n,dtype=np.int64)
    for chart in charts:
        if chart.source.get('direction')=='V' and chart.source['fiber']:
            center=chart.start+len(chart.columns)+np.arange(len(chart.columns));dofs[center]=center[0]
    _,dofs=np.unique(dofs,return_inverse=True)
    links=np.asarray(links,dtype=np.int32).reshape(-1,2)
    from scipy.sparse.csgraph import connected_components
    adjacency=sparse.coo_matrix((np.ones(len(links)),(links[:,0],links[:,1])),shape=(len(charts),len(charts)))
    components,labels=connected_components(adjacency,directed=False)
    return C,np.asarray(targets),dofs.astype(np.int32),dict(contacts=contacts,measured_overlap_pairs=overlap_pairs,measured_overlap_contacts=overlap_contacts,missing_overlap_pairs=missing_overlap_pairs,connected_components=int(components),component_sizes=np.bincount(labels).tolist(),chart_components=labels.tolist())


def write_input(path,charts,C,target,dofs):
    vertices=np.concatenate([c.rest for c in charts]);initial=np.concatenate([c.initial for c in charts]);faces=np.concatenate([c.faces+c.start for c in charts])
    coo=C.tocoo()
    with Path(path).open('wb') as out:
        np.array([0x534c494d57494e44,len(vertices),len(faces),C.shape[0],C.nnz,dofs.max()+1],dtype='<u8').tofile(out)
        for array,dtype in ((vertices,'<f8'),(faces,'<i4'),(initial,'<f8'),(dofs,'<i4'),(coo.row,'<i4'),(coo.col,'<i4'),(coo.data,'<f8'),(target,'<f8')):
            np.asarray(array,dtype=dtype).tofile(out)
        np.concatenate([c.frames for c in charts]).astype('<f8').tofile(out)
        mass=np.concatenate([c.mass for c in charts])
        frames=np.concatenate([c.frames for c in charts])
        # Continue the coordinate field weakly across unobserved cells. This
        # keeps their boundaries ordered without exporting material in holes.
        # Zero stiffness lets such cells collapse and block global line search.
        empty=mass==0
        mass[empty]=.05*frames[empty,0]*frames[empty,2]
        mass.astype('<f8').tofile(out)


def measure_chart_metrics(charts,payload,progress=print):
    """Integrate the observed fine-grid metric, not chords across holes/bends."""
    for number,chart in enumerate(charts):
        if not chart.source['fiber'] and chart.triangulation is not None:
            chart.faces=chart.triangulation.simplices.copy()
            f=chart.faces;a=chart.initial[f[:,1]]-chart.initial[f[:,0]];b=chart.initial[f[:,2]]-chart.initial[f[:,0]]
            flip=a[:,0]*b[:,1]-a[:,1]*b[:,0]<0
            chart.faces[flip]=chart.faces[flip][:,[0,2,1]]
        f=chart.faces
        if not len(f):chart.frames=np.empty((0,3));chart.mass=np.empty(0);continue
        if chart.source['fiber']:
            a=chart.rest[f[:,1]]-chart.rest[f[:,0]];b=chart.rest[f[:,2]]-chart.rest[f[:,0]]
            length=np.linalg.norm(a,axis=1);along=np.sum(a*b,axis=1)/length;across=np.linalg.norm(np.cross(a,b),axis=1)/length
            chart.frames=np.column_stack((length,along,across));chart.mass=length*across;continue
        grid=source_grid(payload,chart.source);a,b,c,d=grid[:-1,:-1],grid[:-1,1:],grid[1:,:-1],grid[1:,1:]
        values=np.zeros((len(f),3));counts=np.zeros(len(f));mass=np.zeros(len(f))
        quad_valid=source_quad_mask(grid,chart.source)
        for dr,dc,quadrature in ((c[:,:,:3]-a[:,:,:3],b[:,:,:3]-a[:,:,:3],
                                   ((1/6,1/6),(1/6,2/3),(2/3,1/6))),
                                (d[:,:,:3]-b[:,:,:3],d[:,:,:3]-c[:,:,:3],
                                   ((1/3,5/6),(5/6,5/6),(5/6,1/3)))):
            q=np.argwhere(quad_valid);r=dr[quad_valid];s=dc[quad_valid]
            metric=np.column_stack((np.sum(r*r,axis=1),np.sum(r*s,axis=1),np.sum(s*s,axis=1)))
            area=np.linalg.norm(np.cross(r,s),axis=1)
            # A centroid alone can lie exactly on the other diagonal of a
            # Delaunay cell and leave half of a fully observed cell unweighted.
            for offset in quadrature:
                ids=chart.triangulation.find_simplex(q+offset,tol=1e-7);good=ids>=0;ids=ids[good]
                counts+=np.bincount(ids,weights=np.full(len(ids),1/3),minlength=len(f))
                mass+=np.bincount(ids,weights=area[good]/3,minlength=len(f))
                for k in range(3):values[:,k]+=np.bincount(ids,weights=metric[good,k]/3,minlength=len(f))
        observed=counts>0
        if observed.any():
            fallback=values.sum(axis=0)/counts.sum();values[observed]/=counts[observed,None];values[~observed]=fallback
        else:
            values[:]=[1.,0.,1.]
        g=np.empty((len(f),2,2));g[:,0,0]=values[:,0];g[:,0,1]=g[:,1,0]=values[:,1];g[:,1,1]=values[:,2]
        a=chart.domain[f[:,1]]-chart.domain[f[:,0]];b=chart.domain[f[:,2]]-chart.domain[f[:,0]]
        length=np.sqrt(np.maximum(np.einsum('ni,nij,nj->n',a,g,a),1e-12));along=np.einsum('ni,nij,nj->n',a,g,b)/length
        across=np.sqrt(np.maximum(np.einsum('ni,nij,nj->n',b,g,b)-along**2,1e-12))
        chart.frames=np.column_stack((length,along,across));chart.mass=mass
        if number%5000==0:progress(f'Measured fine-grid metric for {number:,}/{len(charts):,} charts')


def read_constraints(path):
    """Read a prepared driver input to refine its contacts without resampling."""
    with Path(path).open('rb') as stream:
        magic,n,nf,nr,nnz,nd=np.fromfile(stream,dtype='<u8',count=6).tolist()
        if magic!=0x534c494d57494e44:raise ValueError('Invalid SLIM input')
        stream.seek(n*3*8+nf*3*4+n*2*8,1)
        dofs=np.fromfile(stream,dtype='<i4',count=2*n)
        rows=np.fromfile(stream,dtype='<i4',count=nnz);cols=np.fromfile(stream,dtype='<i4',count=nnz)
        values=np.fromfile(stream,dtype='<f8',count=nnz);target=np.fromfile(stream,dtype='<f8',count=nr)
        return sparse.coo_matrix((values,(rows,cols)),shape=(nr,2*n)).tocsr(),target,dofs


def bridge_components(charts,payload,evidence,labels,weight=1000.,progress=print,row_reference=None):
    """Recover accepted overlap witnesses missed by coarse surface sampling."""
    lookup={c.source['id']:i for i,c in enumerate(charts)};parent=np.arange(len(charts))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    groups={}
    for i,label in enumerate(labels):
        if label in groups:parent[i]=groups[label]
        else:groups[label]=i
    rows=[];cols=[];values=[];targets=[];n=sum(len(c.points) for c in charts);nc=0;distances=[];extended=0;phase_links=0
    if row_reference is None:row_reference=lambda z:z
    def connect(i,j,qa,qb,distance,offset=None):
        nonlocal nc
        ix,aw=charts[i].weights(qa);jx,bw=charts[j].weights(qb)
        for axis in (0,1):
            rows.extend([nc]*(ix.size+jx.size));cols.extend((np.r_[ix[0],jx[0]]+axis*n).tolist())
            values.extend((np.r_[aw[0],-bw[0]]*np.sqrt(weight)).tolist());targets.append(0. if offset is None else offset[axis]*np.sqrt(weight));nc+=1
        parent[root(j)]=root(i)
        if offset is None:distances.append(distance)
    fiber_groups={}
    for i,c in enumerate(charts):
        if c.source['fiber']:fiber_groups.setdefault(c.source.get('component',c.source['id']),[]).append(i)
    for group in fiber_groups.values():
        for at,i in enumerate(group):
            a=source_grid(payload,charts[i].source)[1]
            for j in group[at+1:]:
                if root(i)==root(j):continue
                b=source_grid(payload,charts[j].source)[1]
                _,indices=cKDTree(b[:,:3]).query(a[:,:3],k=list(range(1,min(8,len(b))+1)))
                indices=np.clip(np.concatenate((indices-1,indices),axis=1),0,len(b)-2)
                delta=b[indices+1,:3]-b[indices,:3]
                t=np.clip(np.sum((a[:,None,:3]-b[indices,:3])*delta,axis=2)/np.maximum(np.sum(delta*delta,axis=2),1e-20),0,1)
                d2=np.sum((a[:,None,:3]-b[indices,:3]-t[:,:,None]*delta)**2,axis=2)
                phase=b[indices,3]*(1-t)+b[indices+1,3]*t
                d2[np.abs(phase-a[:,None,3])>10]=np.inf
                r,k=np.unravel_index(np.argmin(d2),d2.shape);distance=float(np.sqrt(d2[r,k]))
                if distance<=1.5+1e-6:
                    connect(i,j,[1.,r],[1.,indices[r,k]+t[r,k]],distance)
    progress(f'After tracing accepted fiber components: {len({root(k) for k in range(len(charts))})} components')
    candidates=[e for e in evidence if e.get('kind')=='overlap' and e.get('status')=='consistent' and e['a'] in lookup and e['b'] in lookup]
    candidates.sort(key=lambda e:-e['support'])
    for edge in candidates:
        i,j=lookup[edge['a']],lookup[edge['b']]
        if root(i)==root(j):continue
        a,b=charts[i],charts[j];ga=source_grid(payload,a.source);gb=source_grid(payload,b.source)
        ia=np.argwhere(ga[:,:,0]!=-1);ib=np.argwhere(gb[:,:,0]!=-1)
        pa=ga[ia[:,0],ia[:,1],:3];pb=gb[ib[:,0],ib[:,1],:3]
        distance,index=cKDTree(pb).query(pa);order=np.argsort(distance)[:128]
        q=pa[order];near=ib[index[order]];cells=[];domains=[]
        for dr in (-1,0):
            for dc in (-1,0):
                base=near+[dr,dc]
                for offsets in (((0,0),(0,1),(1,0)),((0,1),(1,1),(1,0))):
                    ij=base[:,None]+np.array(offsets)[None]
                    valid=np.all((ij>=0)&(ij<np.array(gb.shape[:2])),axis=(1,2))
                    clipped=np.minimum(np.maximum(ij,0),np.array(gb.shape[:2])-1)
                    xyz=gb[clipped[:,:,0],clipped[:,:,1],:3].copy();valid &= np.all(xyz[:,:,0]!=-1,axis=1)
                    xyz[~valid]=1e20;cells.append(xyz);domains.append(ij)
        triangles=np.stack(cells,axis=1);domain=np.stack(domains,axis=1)
        d2,w=closest_triangle(q,triangles);flat=np.argmin(d2);r,k=np.unravel_index(flat,d2.shape)
        distance=float(np.sqrt(d2[r,k]))
        if distance>2.:
            # The workspace stores accepted quad CENTERS, while growth tests
            # complete source quads. At a narrow accepted overlap the displayed
            # center meshes can stop half a cell short on either side. Use a
            # bounded local continuation of a real source triangle; do not
            # collapse the physical gap by identifying its two endpoints.
            base=triangles[:,:,0];ab=triangles[:,:,1]-base;ac=triangles[:,:,2]-base;delta=q[:,None]-base
            aa=np.sum(ab*ab,axis=2);bb=np.sum(ab*ac,axis=2);cc=np.sum(ac*ac,axis=2);det=aa*cc-bb*bb
            aq=np.sum(delta*ab,axis=2);cq=np.sum(delta*ac,axis=2)
            s=(cc*aq-bb*cq)/np.maximum(det,1e-20);t=(aa*cq-bb*aq)/np.maximum(det,1e-20)
            planar=np.stack((1-s-t,s,t),axis=2)
            normal=np.sum((delta-s[:,:,None]*ab-t[:,:,None]*ac)**2,axis=2)
            valid=(det>1e-8)&np.all((planar>=-1.)&(planar<=2.),axis=2)&np.all(triangles[:,:,:,0]<1e19,axis=2)
            normal[~valid]=np.inf;r,k=np.unravel_index(np.argmin(normal),normal.shape)
            distance=float(np.sqrt(normal[r,k]))
            if distance>2.:continue
            w=planar;extended+=1
        qa=ia[order[r]];qb=np.sum(domain[r,k]*w[r,k,:,None],axis=0)
        connect(i,j,qa,qb,distance)
        progress(f'Recovered accepted overlap {len(distances)}: {len({root(k) for k in range(len(charts))})} components')
    components=len({root(k) for k in range(len(charts))})
    def preserve_gap(i,j):
        nonlocal phase_links
        a,b=charts[i],charts[j]
        ca=np.column_stack((a.points[:,:3],a.points[:,3]));cb=np.column_stack((b.points[:,:3],b.points[:,3]))
        distance,index=cKDTree(cb).query(ca);p=int(np.argmin(distance));q=int(index[p])
        offset=np.array([a.points[p,3]-b.points[q,3],row_reference(a.points[p,2])-row_reference(b.points[q,2])])
        connect(i,j,a.domain[p],b.domain[q],float(distance[p]),offset);phase_links+=1
    # Branch links are SAME-WINDING statements; they need not be coincident
    # points. Retain measured relative placement across unobserved coverage,
    # rather than collapsing those gaps or inventing coincident geometry.
    for group in fiber_groups.values():
        for j in group[1:]:
            if root(group[0])!=root(j):preserve_gap(group[0],j)
    for edge in candidates:
        i,j=lookup[edge['a']],lookup[edge['b']]
        if root(i)!=root(j):preserve_gap(i,j)
    representatives={}
    for i,c in enumerate(charts):representatives.setdefault(c.source.get('component',c.source['id']),i)
    for edge in evidence:
        if edge.get('status')!='consistent' or edge['a'] not in representatives or edge['b'] not in representatives:continue
        i,j=representatives[edge['a']],representatives[edge['b']]
        if root(i)!=root(j):preserve_gap(i,j)
    components=len({root(k) for k in range(len(charts))})
    return sparse.coo_matrix((values,(rows,cols)),shape=(nc,2*n)).tocsr(),np.asarray(targets),dict(bridge_contacts=len(distances),boundary_continuations=extended,phase_gap_links=phase_links,connected_components=components,component_sizes=np.unique([root(k) for k in range(len(charts))],return_counts=True)[1].tolist(),max_bridge_distance=max(distances,default=0.),chart_components=[int(root(k)) for k in range(len(charts))])

def validate_solution(charts, payload, uv, progress=print, samples=32):
    """Audit every rendered triangle for folds and sample its physical metric."""
    from winding_straightening import distortion_report
    triangles=[];mapped=[];orientations=[]
    folds=0;checked=0;max_column_span=0.;fiber_ratios=[];foldover_sources=[]
    for number,chart in enumerate(charts):
        grid=source_grid(payload,chart.source);valid=grid[:,:,0]!=-1
        ij=np.argwhere(valid);placed=np.zeros(grid.shape[:2]+(2,))
        placed[valid]=chart.map(uv,ij)
        if not np.isfinite(placed[valid]).all():raise ValueError('Nonfinite SLIM placement')
        if chart.source['fiber']:
            if chart.source.get('direction')=='V':
                max_column_span=max(max_column_span,float(np.ptp(placed[1,:,0])))
                lengths=np.linalg.norm(np.diff(grid[1,:,:3],axis=0),axis=1)
                fiber_ratios.extend((np.linalg.norm(np.diff(placed[1],axis=0),axis=1)[lengths>1e-9]/lengths[lengths>1e-9]).tolist())
            continue
        if chart.triangulation is None:continue
        f=chart.faces[0];a,b=chart.domain[f[1:]]-chart.domain[f[0]]
        # Original fine-grid triangles (a,b,c) have negative IJ orientation.
        sign=-np.sign(a[0]*b[1]-a[1]*b[0])
        for offsets in (((0,0),(0,1),(1,0)),((0,1),(1,1),(1,0))):
            shape=(grid.shape[0]-1,grid.shape[1]-1)
            blocks=[grid[r:r+shape[0],c:c+shape[1]] for r,c in offsets]
            uvs=[placed[r:r+shape[0],c:c+shape[1]] for r,c in offsets]
            ok=source_quad_mask(grid,chart.source)
            p=np.stack([block[ok] for block in blocks],axis=1)
            u=np.stack([block[ok] for block in uvs],axis=1)
            if not len(p):continue
            cross=np.linalg.norm(np.cross(p[:,1,:3]-p[:,0,:3],p[:,2,:3]-p[:,0,:3]),axis=1)
            a=u[:,1]-u[:,0];b=u[:,2]-u[:,0];area=sign*(a[:,0]*b[:,1]-a[:,1]*b[:,0])
            usable=cross>1e-8;bad=int(np.count_nonzero((area<=0)&usable));folds+=bad;checked+=int(usable.sum())
            if bad and chart.source['id'] not in foldover_sources:foldover_sources.append(chart.source['id'])
            take=np.flatnonzero(usable)
            take=take[np.linspace(0,len(take)-1,min(samples,len(take))).astype(int)] if len(take) else take
            triangles.append(p[take]);mapped.append(u[take]);orientations.append(np.full(len(take),sign))
        if number%5000==0:progress(f'Validated {number:,}/{len(charts):,} sources')
    triangles=np.concatenate(triangles);mapped=np.concatenate(mapped);orientation=np.concatenate(orientations)
    before=distortion_report(triangles,triangles[:,:,3:],orientation)
    after=distortion_report(triangles,mapped,orientation)
    fiber_percentiles=np.percentile(fiber_ratios,[1,5,50,95,99]).tolist() if fiber_ratios else []
    accepted=(folds==0 and max_column_span<1e-6 and after['mean_squared_log_stretch']<=before['mean_squared_log_stretch']+.01)
    return dict(accepted=bool(accepted),before=before,after=after,rendered_triangles=checked,
                rendered_foldovers=folds,foldover_sources=foldover_sources,max_vertical_column_span=max_column_span,
                vertical_length_ratio_percentiles=fiber_percentiles)


def drop_rendered_foldovers(charts,payload,uv,source_ids):
    """Omit only accepted quads with an inverted rendered triangle.

    This changes the candidate workspace's face mask, never its source XYZ,
    UV, vertex layout, or the patch files on disk.
    """
    from winding_topology import quad_runs
    wanted=set(source_ids);removed=[]
    for chart in charts:
        if chart.source['id'] not in wanted:continue
        if chart.source['fiber'] or chart.triangulation is None:
            raise ValueError('Cannot mask foldovers for this source')
        grid=source_grid(payload,chart.source);valid=grid[:,:,0]!=-1
        placed=np.zeros(grid.shape[:2]+(2,));placed[valid]=chart.map(uv,np.argwhere(valid))
        mask=source_quad_mask(grid,chart.source);cells=np.argwhere(mask)
        f=chart.faces[0];a,b=chart.domain[f[1:]]-chart.domain[f[0]]
        sign=-np.sign(a[0]*b[1]-a[1]*b[0]);bad=np.zeros(len(cells),bool)
        for offsets in (((0,0),(0,1),(1,0)),((1,1),(1,0),(0,1))):
            ij=cells[:,None]+np.array(offsets)[None]
            xyz=grid[ij[:,:,0],ij[:,:,1],:3];xy=placed[ij[:,:,0],ij[:,:,1]]
            physical=np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)
            a=xy[:,1]-xy[:,0];b=xy[:,2]-xy[:,0]
            bad |= (physical>1e-8)&(sign*(a[:,0]*b[:,1]-a[:,1]*b[:,0])<=0)
        if bad.any():
            discarded=cells[bad];mask[discarded[:,0],discarded[:,1]]=False
            chart.source['quad_runs']=quad_runs(mask)
            removed.append(dict(source=chart.source['id'],quads=int(bad.sum()),
                                quad_indices=(discarded[:,0]*mask.shape[1]+discarded[:,1]).tolist()))
    return dict(quads=sum(item['quads'] for item in removed),sources=removed)


def apply_solution(charts,payload,uv,progress=print):
    lo,hi=np.full(2,np.inf),np.full(2,-np.inf)
    for number,chart in enumerate(charts):
        grid=source_grid(payload,chart.source);valid=grid[:,:,0]!=-1
        placed=chart.map(uv,np.argwhere(valid))
        if not np.isfinite(placed).all():raise ValueError('Nonfinite SLIM placement')
        grid[:,:,3:][valid]=placed
        lo=np.minimum(lo,placed.min(axis=0));hi=np.maximum(hi,placed.max(axis=0))
        if number%5000==0:progress(f'Straightened {number:,}/{len(charts):,} sources')
    return [lo.tolist(),hi.tolist()]


def fit_workspace(payload,sources,*,evidence=None,row_band=40.,spacing=100.,iterations=50,
                  executable=None,progress=print,initialize_from_saved=False,overlap_distance=1.,overlap_normal_cos=.95,work_dir=None,drop_folded_quads=False):
    """Run the existing flatboi SLIM/Pastix solver and validate before export."""
    import os
    import subprocess
    import tempfile
    from build_winding_slim import build_driver
    if not np.isfinite(row_band) or row_band<=0 or not np.isfinite(spacing) or spacing<=0 or iterations<1:
        raise ValueError('Invalid SLIM straightening settings')
    executable=Path(executable or Path(__file__).parent/'outputs'/'winding_slim_build'/'winding_slim').resolve()
    if not executable.is_file():executable=build_driver(output_dir=executable.parent)
    charts,guides,report=make_charts(payload,sources,spacing=spacing,progress=progress,initialize_from_saved=initialize_from_saved)
    if report['omitted_sources']:raise ValueError('Cannot straighten sources with degenerate parameter domains')
    measure_chart_metrics(charts,payload,progress=progress)
    C,target,dofs,contacts=prepare_constraints(charts,guides,evidence=evidence,payload=payload,progress=progress,row_band=row_band,overlap_distance=overlap_distance,overlap_normal_cos=overlap_normal_cos)
    if evidence is not None:
        extra,targets,bridges=bridge_components(charts,payload,evidence,contacts['chart_components'],progress=progress,row_reference=guides.row_reference)
        C=sparse.vstack((C,extra),format='csr');target=np.r_[target,targets]
        contacts.update(bridges)
    contacts.pop('chart_components',None)
    from contextlib import nullcontext
    if work_dir is not None:
        Path(work_dir).mkdir(parents=True,exist_ok=True)
        Path(work_dir,'contacts.json').write_text(json.dumps(contacts,indent=2)+'\n')
    with (tempfile.TemporaryDirectory(prefix='winding-slim-') if work_dir is None else nullcontext(str(work_dir))) as temporary:
        folder=Path(temporary);input_path=folder/'input.bin';output=folder/'uv.bin'
        write_input(input_path,charts,C,target,dofs)
        del C,target,dofs
        env=os.environ.copy();env.setdefault('PASTIX_NUM_THREADS',str(min(16,os.cpu_count() or 1)))
        env.setdefault('OPENBLAS_NUM_THREADS','1')
        log=[]
        with subprocess.Popen([str(executable),str(input_path),str(output),str(iterations)],
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env) as process:
            for line in process.stdout:
                line=line.rstrip();log.append(line);progress(line)
            if process.wait():raise ValueError('SLIM/Pastix failed: '+'\n'.join(log[-5:]))
        uv=np.fromfile(output,dtype='<f8').reshape(-1,2)
        validation=validate_solution(charts,payload,uv,progress=progress)
        if work_dir is not None:
            Path(work_dir,'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
        corrections=[];omitted=dict(quads=0,sources=[])
        if validation['rendered_foldovers'] and drop_folded_quads:
            omitted=drop_rendered_foldovers(charts,payload,uv,validation['foldover_sources'])
            progress(f"Omitted {omitted['quads']} folded quads from the candidate workspace")
            validation=validate_solution(charts,payload,uv,progress=progress)
        if validation['rendered_foldovers']:
            uv,corrections=polish_rendered_orientation(charts,payload,uv,validation['foldover_sources'])
            validation=validate_solution(charts,payload,uv,progress=progress)
        validation['resampling_corrections']=corrections
        validation['omitted_folded_quads']=omitted
        C,target,_=read_constraints(input_path)
        attachments=contact_report(C,target,charts,uv,contacts['contacts'])
        del C,target
    validation['contacts']=attachments
    validation['accepted'] &= attachments['accepted']
    if work_dir is not None:
        Path(work_dir,'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
        Path(work_dir,'contacts.json').write_text(json.dumps(contacts,indent=2)+'\n')
    if not validation['accepted']:
        raise ValueError('Straightening rejected; original workspace preserved. '+json.dumps(validation))
    report.update(contacts,method='SLIM symmetric Dirichlet with Pastix',validation=validation,
                  intrinsic_quadrature_points=3,hole_continuation_weight=.05,
                  overlap_distance=overlap_distance,overlap_normal_cos=overlap_normal_cos,
                  row_band=row_band,solver_log=log,
                  vertical_columns='hard equality' if guides.lines else 'none (no vertical fibers)',
                  vertical_fibers=len(guides.lines),
                  row_reference='measured_vertical_arclength' if guides.lines else 'source_z',
                  row_z=guides.row_z.tolist(),row_v=guides.row_v.tolist())
    return charts,uv,report


def contact_report(C,target,charts,uv,count,weight=1000.):
    """Compare physical attachment residuals against the original placement."""
    constraint=C[:2*count];target=target[:2*count]
    before=np.concatenate([c.points[:,3:] for c in charts])
    reports=[]
    corrected=uv.copy()
    for chart in charts:
        if chart.correction_field is not None:
            corrected[chart.start:chart.start+len(chart.points)]=chart.map(uv,chart.domain)
    for placement in (before,corrected):
        residual=(constraint@placement.T.ravel()-target).reshape(-1,2)/np.sqrt(weight)
        distances=np.linalg.norm(residual,axis=1)
        reports.append(dict(percentiles=np.percentile(distances,[1,5,50,95,99]).tolist(),
                            rms=float(np.sqrt(np.mean(distances**2)))))
    a,b=reports
    return dict(before=a,after=b,accepted=bool(b['percentiles'][2]<=max(2.,a['percentiles'][2]) and
                                             b['percentiles'][3]<=max(5.,a['percentiles'][3])))


def polish_rendered_orientation(charts,payload,uv,source_ids,max_displacement=2.):
    """Refine a few original grid vertices when reconnecting samples folds a face.

    Keep the global SLIM field and add minimum-displacement nodal corrections,
    interpolated continuously on the native grid. The caller rechecks all
    metric and attachment diagnostics; changes above two voxels are rejected.
    """
    corrections=[];wanted=set(source_ids)
    for chart in charts:
        if chart.source['id'] not in wanted:continue
        if chart.source['fiber'] or chart.triangulation is None:raise ValueError('Cannot refine this source')
        grid=source_grid(payload,chart.source);valid=grid[:,:,0]!=-1;q=np.argwhere(valid)
        original=chart.map(uv,q);index=np.full(valid.shape,-1,dtype=int);index[valid]=np.arange(len(q))
        quads=source_quad_mask(grid,chart.source);cells=np.argwhere(quads)
        f=chart.faces[0];a,b=chart.domain[f[1:]]-chart.domain[f[0]];sign=-np.sign(a[0]*b[1]-a[1]*b[0])
        faces=[];scales=[]
        for offsets in (((0,0),(0,1),(1,0)),((0,1),(1,1),(1,0))):
            ij=cells[:,None]+np.array(offsets)[None];xyz=grid[ij[:,:,0],ij[:,:,1],:3]
            area=np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1);good=area>1e-8
            faces.append(index[ij[:,:,0],ij[:,:,1]][good]);scales.append(area[good])
        faces=np.concatenate(faces);scale=np.concatenate(scales)
        delta,displacement=polish_uv_orientation(original,faces,scale,sign,max_displacement)
        field=np.zeros(grid.shape[:2]+(2,));field[valid]=delta
        chart.correction_field=field if chart.correction_field is None else chart.correction_field+field
        corrections.append(dict(source=chart.source['id'],vertices=int(np.count_nonzero(np.linalg.norm(delta,axis=1)>1e-8)),max_displacement=displacement))
    return uv,corrections
