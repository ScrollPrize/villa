"""Read-only, approximate reference winding association and cross-colored maps.

No correspondence optimization: nearest local surface, original-fiber weighted
median winding, then nearest surface restricted to that winding. Surface cells
are approximated by triangles; this is deliberately a rough diagnostic.
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import fiber_uv_layout as uv
from .fiber_uv_visualization import TiledRaster


def landmarks(bounds):
    rows = sorted(bounds, key=lambda r: r['winding'])
    ids = np.array([r['winding'] for r in rows])
    if len(ids) < 2 or np.any(np.diff(ids) != 1):
        raise ValueError('Requires consecutive reference winding IDs')
    x = []
    for a, b in zip(rows[:-1], rows[1:]):
        left = max(a['column_begin'], b['column_begin'])
        right = min(a['column_end'], b['column_end'])
        if left >= right:
            raise ValueError('Adjacent winding rectangles do not overlap')
        x.append((left + right) / 2.)
    if np.any(np.diff(x) <= 0):
        raise ValueError('Winding landmarks are not increasing')
    return np.asarray(x), ids[:-1] + .5


def winding_at(columns, x, q):
    columns = np.asarray(columns)
    if len(x) < 2:
        raise ValueError('Need two landmarks to extrapolate')
    i = np.searchsorted(x, columns, side='right').clip(1, len(x)-1)-1
    return q[i] + (columns-x[i])*(q[i+1]-q[i])/(x[i+1]-x[i])


def palette(q, seed=20261011):
    """Golden-angle saturated landmarks, smoothly interpolated in OKLab.

    Matrices: https://bottosson.github.io/posts/oklab/ (public-domain code).
    The seed rotates the entire palette; integer identity is range-independent.
    """
    from matplotlib.colors import hsv_to_rgb
    q = np.asarray(q)
    safe = np.where(np.isfinite(q), q, 0.)
    k = np.floor(safe).astype(np.int64)
    ids, inverse = np.unique(np.r_[k.ravel(), (k+1).ravel()], return_inverse=True)
    hsv = np.empty((len(ids), 3))
    hsv[:, 0] = (ids * ((3-np.sqrt(5))/2) + (seed-20261011)*.01) % 1
    hsv[:, 1:] = [.9, .95]
    rgb = hsv_to_rgb(hsv)
    linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
    lms = linear @ np.array([[.4122214708,.5363325363,.0514459929],
        [.2119034982,.6806995451,.1073969566],[.0883024619,.2817188376,.6299787005]]).T
    lab = np.cbrt(lms) @ np.array([[.2104542553,.7936177850,-.0040720468],
        [1.9779984951,-2.4285922050,.4505937099],[.0259040371,.7827717662,-.8086757660]]).T
    a, b = inverse.reshape(2, -1)
    f = (safe-k).ravel()[:, None]
    mixed = lab[a]*(1-f)+lab[b]*f
    lms = (mixed @ np.array([[1,.3963377774,.2158037573],
        [1,-.1055613458,-.0638541728],[1,-.0894841775,-1.2914855480]]).T)**3
    linear = lms @ np.array([[4.0767416621,-3.3077115913,.2309699292],
        [-1.2684380046,2.6097574011,-.3413193965],[-.0041960863,-.7034186147,1.7076147010]]).T
    linear = np.clip(linear, 0, 1)
    rgb = np.where(linear <= .0031308, 12.92*linear, 1.055*linear**(1/2.4)-.055)
    out = np.rint(rgb*255).astype(np.uint8).reshape(q.shape+(3,))
    return np.where(np.isfinite(q)[..., None], out, 180)


def weighted_median(values, weights):
    values = np.asarray(values); weights = np.asarray(weights)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not ok.any():
        return np.nan
    order = np.argsort(values[ok], kind='stable')
    a = values[ok][order]; w = weights[ok][order]
    return a[np.searchsorted(np.cumsum(w), w.sum()/2, side='left')]


def grouped_weighted_median(values, weights, offsets):
    """Exact constant-group fast path; mixed groups use the canonical routine.

    No cross-group cumulative sums (which can perturb half-weight ties).
    Empty groups and groups without positive finite weights return NaN.
    """
    values=np.asarray(values);weights=np.asarray(weights);offsets=np.asarray(offsets)
    if (values.ndim!=1 or weights.shape!=values.shape or offsets.ndim!=1 or
            len(offsets)<1 or offsets[0]!=0 or offsets[-1]!=len(values) or
            np.any(np.diff(offsets)<0)):
        raise ValueError('Invalid grouped median arrays/offsets')
    result=np.full(len(offsets)-1,np.nan)
    groups=np.flatnonzero(np.diff(offsets)>0)
    if not len(groups):return result
    valid=np.isfinite(values)&np.isfinite(weights)&(weights>0)
    starts=offsets[groups]
    low=np.minimum.reduceat(np.where(valid,values,np.inf),starts)
    high=np.maximum.reduceat(np.where(valid,values,-np.inf),starts)
    same=np.isfinite(low)&(low==high)
    result[groups[same]]=low[same]
    for group in groups[(low<high)&np.isfinite(low)]:
        a,b=offsets[group:group+2]
        result[group]=weighted_median(values[a:b],weights[a:b])
    return result


def discrete_winding(q, limits=None):
    labels=np.floor(np.asarray(q)+.5)
    return labels if limits is None else np.clip(labels,*limits)


def winding_error_metrics(fitted, reference):
    """Equal-fiber scores, with one L1-optimal global translation (no rescaling)."""
    fitted=np.asarray(fitted);reference=np.asarray(reference)
    valid=np.isfinite(fitted)&np.isfinite(reference)
    if not valid.any():
        return dict(matched_fibers=0,global_offset=None,mean_absolute_winding_error=None,
                    within_two_windings_percent=None)
    offset=float(np.median(reference[valid]-fitted[valid]))
    error=np.abs(fitted[valid]+offset-reference[valid])
    return dict(matched_fibers=int(valid.sum()),global_offset=offset,
                mean_absolute_winding_error=float(error.mean()),
                fitted_winding_q10_q90=np.quantile(fitted[valid]+offset,[.1,.9]).tolist(),
                reference_winding_q10_q90=np.quantile(reference[valid],[.1,.9]).tolist(),
                within_two_windings_percent=float(100*np.mean(error<=2)))


def visualization_offset(settings,reference,fitted,weights):
    if 'global_winding_offset' in settings:
        value=float(settings['global_winding_offset'])
        if not np.isfinite(value):raise ValueError('Nonfinite visualization offset')
        return value
    return int(np.rint(weighted_median(reference-fitted,weights)))


def load_metric_projection(path,d,config=None):
    """Load existing geometry-only projection; fail on mismatched piece identity."""
    if config is not None:
        signature=json.loads(Path(path).with_name('projection_signature.json').read_text())
        if signature['afv']!=uv.sha256_file(config['afv']):
            raise ValueError('Reference projection AFV provenance mismatch')
    with np.load(path) as src:
        for key in ('source_id','start','end'):
            if not np.array_equal(src[key],d[key]):raise ValueError('Reference projection identity mismatch')
        return {key:src[key] for key in ('owner','arc','weight','offset','chosen_winding')}


def prepare_metric_cache(d,projection):
    """Build once before parallel readouts; arrays must remain immutable."""
    owner=projection['owner']
    identity=tuple(id(d[key]) for key in ('node_ds','lengths','start','source_id','family'))+tuple(
        id(projection[key]) for key in ('owner','arc','offset'))
    cache=projection.get('_metric_cache')
    if cache is None or cache['identity']!=identity:
        direction=(d['node_ds'][:,2]-d['node_ds'][:,0])/d['lengths']
        ds=(projection['arc']-d['start'][owner])*direction[owner]
        starts=np.r_[0,np.flatnonzero(np.diff(d['source_id']))+1,len(d['family'])]
        cache=projection['_metric_cache']=dict(starts=starts,identity=identity,
            displacement=(d['family'][owner]==0)*ds,offsets=projection['offset'][starts])
    return cache


def reference_metric_inputs(d,state,widths,config,projection,geometry=None):
    """Shared sample readout; projection and source geometry are immutable."""
    _,_,keep,_,u,_,_=(uv.hard_geometry(d,state,widths,config) if geometry is None else geometry)
    owner=projection['owner'];cache=prepare_metric_cache(d,projection);starts=cache['starts']
    fitted=discrete_winding(inverse_u(u[owner]+cache['displacement'],widths))
    weights=projection['weight']*keep[owner]
    return keep,fitted,weights,starts,cache['offsets']


def reference_winding_metrics(d,state,widths,config,projection,geometry=None):
    """Original-source median winding, aggregating retained actual-curve samples."""
    keep,fitted,weights,starts,offsets=reference_metric_inputs(d,state,widths,config,projection,geometry)
    if config.get('diagnostic_backend','optimized')=='reference':
        predictions=[];references=[];retained=0
        for a,b in zip(starts[:-1],starts[1:]):
            if not keep[a:b].any():continue
            retained+=1
            lo,hi=projection['offset'][[a,b]]
            predictions.append(weighted_median(fitted[lo:hi],weights[lo:hi]))
            references.append(projection['chosen_winding'][a])
    else:
        retained_sources=np.logical_or.reduceat(keep,starts[:-1])
        retained=int(retained_sources.sum())
        predictions=grouped_weighted_median(fitted,weights,offsets)[retained_sources]
        references=projection['chosen_winding'][starts[:-1]][retained_sources]
    result=winding_error_metrics(predictions,references)
    result.update(total_source_fibers=len(starts)-1,retained_source_fibers=retained,
                  unmatched_retained_fibers=retained-result['matched_fibers'],
                  matched_source_percent=100*result['matched_fibers']/max(len(starts)-1,1),
                  weighting='equal original fibers; retained sample length median within each fiber')
    return result


def triangle_projection(p, vertices, coords):
    """Closest point on candidate triangles, including edges; vectorized batches."""
    a, b, c = np.moveaxis(vertices, -2, 0)
    ab = b-a; ac = c-a; ap = p[:, None, :]-a
    dot = lambda x, y: np.sum(x*y, axis=-1)
    aa = dot(ab, ab); bb = dot(ac, ac); cc = dot(ab, ac)
    det = aa*bb-cc*cc
    good = det > 1e-12
    den = np.where(good, det, 1)
    s = (dot(ap, ab)*bb-dot(ap, ac)*cc)/den
    t = (dot(ap, ac)*aa-dot(ap, ab)*cc)/den
    weights = np.stack((1-s-t, s, t), axis=-1)
    point = a + s[..., None]*ab + t[..., None]*ac
    dist = dot(point-p[:, None, :], point-p[:, None, :])
    dist = np.where(good & (s >= 0) & (t >= 0) & (s+t <= 1), dist, np.inf)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        start = vertices[..., i, :]; edge = vertices[..., j, :]-start
        f = np.clip(dot(p[:, None, :]-start, edge)/np.maximum(dot(edge, edge), 1e-20), 0, 1)
        pos = start+f[..., None]*edge
        ds = dot(pos-p[:, None, :], pos-p[:, None, :])
        use = good & (ds < dist)
        ew = np.zeros_like(weights); ew[..., i] = 1-f; ew[..., j] = f
        weights = np.where(use[..., None], ew, weights); dist = np.where(use, ds, dist)
    rc = np.sum(coords*weights[..., None], axis=-2)
    return dist, rc


class Surface:
    def __init__(self, root, scale=4., stride=2, workers=32, candidates=4):
        from vesuvius.tifxyz.reader import TifxyzReader
        self.root = Path(root); self.meta = json.loads((self.root/'meta.json').read_text())
        self.x, self.q = landmarks(self.meta['winding_bounds'])
        self.limits=(self.q[0]-.5,self.q[-1]+.5)
        reader = TifxyzReader(root)
        self.full = [reader.read_coordinate(c) for c in 'xyz']
        self.stride = stride; self.workers = workers; self.candidates = candidates
        self.xyz = np.stack([a[::stride, ::stride] for a in self.full], axis=-1)*scale
        # Every original grid vertex inside each coarse cell must be valid.
        valid = np.ones(np.array(self.xyz.shape[:2])-1, bool)
        h, w = valid.shape
        for dy in range(stride+1):
            for dx in range(stride+1):
                for a in self.full:
                    sub = a[dy:dy+h*stride:stride, dx:dx+w*stride:stride]
                    valid &= np.isfinite(sub) & (sub != -1)
        corners = [self.xyz[:-1, :-1], self.xyz[:-1, 1:], self.xyz[1:, :-1], self.xyz[1:, 1:]]
        # Avoid damaged/folded-grid long bridges (native voxels).
        for a, b in ((0, 1), (0, 2), (1, 3), (2, 3)):
            valid &= np.linalg.norm(corners[a]-corners[b], axis=-1) <= scale*self.meta['output_step_vx']*stride*4
        self.r, self.c = np.nonzero(valid)
        self.valid_cells = valid
        self.centers = sum(a[valid] for a in corners)/4
        self.tree = cKDTree(self.centers)
        self.labels = discrete_winding(winding_at((self.c+.5)*stride, self.x, self.q),self.limits).astype(int)
        print(f'reference: {len(self.r):,} valid surface cells, stride={stride}', flush=True)

    def project(self, points, indices=None, batch=8192):
        tree = self.tree if indices is None else cKDTree(self.centers[indices])
        count = len(self.r) if indices is None else len(indices)
        result = np.full((len(points), 2), np.nan); distances = np.full(len(points), np.inf)
        if not count:
            return result, distances
        for lo in range(0, len(points), batch):
            p = points[lo:lo+batch]
            _, ix = tree.query(p, k=min(self.candidates, count), workers=self.workers)
            if ix.ndim == 1: ix = ix[:, None]
            if indices is not None: ix = indices[ix]
            r = self.r[ix]; c = self.c[ix]
            vertices = np.stack((self.xyz[r,c], self.xyz[r,c+1], self.xyz[r+1,c], self.xyz[r+1,c+1]), axis=-2)
            rc = np.stack((np.stack((r,c),-1),np.stack((r,c+1),-1),np.stack((r+1,c),-1),np.stack((r+1,c+1),-1)), axis=-2)*self.stride
            triangles = np.concatenate((vertices[...,[0,1,3],:], vertices[...,[0,3,2],:]), axis=1)
            tc = np.concatenate((rc[...,[0,1,3],:], rc[...,[0,3,2],:]), axis=1)
            dist, uvrc = triangle_projection(p, triangles, tc)
            pick = dist.argmin(1); rows = np.arange(len(p))
            result[lo:lo+len(p)] = uvrc[rows,pick]
            distances[lo:lo+len(p)] = np.sqrt(dist[rows,pick])
            if indices is None and lo % (batch*32) == 0:
                print(f'reference projection {lo+len(p):,}/{len(points):,}', flush=True)
        return result, distances


def samples(d, cfg, spacing):
    from vesuvius.neural_tracing.fiber_follow.data.afv import read_afv_polylines
    source_ids = np.unique(d['source_id']); curves = read_afv_polylines(cfg['afv'], source_ids)
    counts=[]; xyzs=[]; arcs=[]; weights=[]
    # Source pieces are sorted by source and arc in the tensor cache.
    if np.any(np.diff(d['source_id']) < 0): raise ValueError('Unsorted source pieces')
    for row, xyz in curves:
        sid = row['id']; arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        first = np.searchsorted(d['source_id'], sid, side='left')
        last = np.searchsorted(d['source_id'], sid, side='right')
        for i in range(first, last):
            a,b = d['start'][i],d['end'][i]
            # Retain original curve vertices as well as uniform arc samples.
            s = np.unique(np.r_[np.linspace(a,b,max(2,int(np.ceil((b-a)/spacing))+1)),arc[(arc>a)&(arc<b)]])
            counts.append(len(s)); arcs.append(s)
            xyzs.append(np.stack([np.interp(s,arc,xyz[:,axis]) for axis in range(3)],axis=1))
            delta=np.diff(s); weights.append(np.r_[delta[0]/2,(delta[:-1]+delta[1:])/2,delta[-1]/2])
    counts=np.asarray(counts)
    if len(counts)!=len(d['source_id']):raise ValueError('Missing AFV geometry')
    return dict(xyz=np.concatenate(xyzs).astype(np.float32),arc=np.concatenate(arcs),weight=np.concatenate(weights),
                owner=np.repeat(np.arange(len(counts)),counts),offset=np.r_[0,np.cumsum(counts)])


def assign_sources(d, sample, reference_q, distance, maximum, limits=None):
    winding = discrete_winding(reference_q,limits)
    winding = np.where(distance<=maximum,winding,np.nan)
    chosen = np.full(len(d['source_id']),np.nan)
    starts = np.r_[0,np.flatnonzero(np.diff(d['source_id']))+1,len(d['source_id'])]
    for a,b in zip(starts[:-1],starts[1:]):
        lo,hi=sample['offset'][[a,b]]
        chosen[a:b]=weighted_median(winding[lo:hi],sample['weight'][lo:hi])
    return chosen


def inverse_u(u,widths):
    b=np.r_[0.,np.cumsum(widths)]
    i=(np.searchsorted(b,u,side='right')-1).clip(0,len(widths)-1)
    return i+(u-b[i])/widths[i]


def surface_edges(surface,rc,valid,owner,arc,pitch):
    """Don't draw UV jumps or straight connectors across missing surface cells."""
    length=np.linalg.norm(np.diff(rc,axis=0),axis=1)
    ok=valid[:-1]&valid[1:]&(owner[:-1]==owner[1:])
    ok &= length <= 4*np.maximum(np.diff(arc),0)/pitch+4
    ids=np.flatnonzero(ok)
    for first in range(0,len(ids),65536):
        ix=ids[first:first+65536]
        steps=np.maximum(1,np.ceil(length[ix]/surface.stride).astype(int))
        for n in range(int(steps.max(initial=0))+1):
            active=n<=steps
            pos=rc[ix[active]]+(rc[ix[active]+1]-rc[ix[active]])*(n/steps[active])[:,None]
            cell=np.floor(pos/surface.stride).astype(int)
            inside=(cell>=0).all(1)&(cell<np.array(surface.valid_cells.shape)).all(1)
            good=np.zeros(len(cell),bool)
            good[inside]=surface.valid_cells[cell[inside,0],cell[inside,1]]
            ok[ix[active]]&=good
    return ok


def render_curves(path, xy, colors, owner, valid, alpha, height=1000, bounds=None,edges=None):
    """Equal-scale map-only TIFF and streamed SVG, no edges across missing points."""
    use=valid & np.isfinite(xy).all(1)
    if not use.any():raise ValueError('No renderable fiber samples')
    low=xy[use].min(0) if bounds is None else np.asarray(bounds[:2])
    high=xy[use].max(0) if bounds is None else np.asarray(bounds[2:])
    scale=(height-1)/max(high[1]-low[1],1.)
    width=int(np.ceil((high[0]-low[0])*scale))+1
    pix=(xy-low)*scale
    raster=TiledRaster(width,height)
    connected=use[:-1]&use[1:]&(owner[:-1]==owner[1:])
    if edges is not None: connected &= edges
    lines=np.flatnonzero(connected)
    with path.with_suffix('.svg').open('w') as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>\n')
        for i in lines:
            coords=tuple(pix[i])+tuple(pix[i+1]);rgb=tuple(int(v) for v in colors[i])
            raster.line(coords,fill=rgb+(255,),width=2)
            f.write(f'<path d="M{coords[0]:.2f},{coords[1]:.2f} L{coords[2]:.2f},{coords[3]:.2f}" fill="none" stroke="rgb{rgb}" stroke-opacity="1" stroke-width="2"/>\n')
        f.write('</svg>')
    raster.save(path.with_suffix('.tif'))
    return dict(width=width,height=height,segments=len(lines),bounds=[float(v) for v in (*low,*high)])


def projected_surface_z(surface,rc):
    """Same coarse-cell diagonal and piecewise-linear interpolation as project()."""
    result=np.full(len(rc),np.nan);ok=np.isfinite(rc).all(1)
    pos=rc[ok]/surface.stride
    cell=np.floor(pos).astype(int)
    cell=np.clip(cell,0,np.array(surface.xyz.shape[:2])-2)
    fy,fx=(pos-cell).T;r,c=cell.T
    a=surface.xyz[r,c,2];b=surface.xyz[r,c+1,2]
    d=surface.xyz[r+1,c,2];e=surface.xyz[r+1,c+1,2]
    result[ok]=np.where(fy<=fx,a+fx*(b-a)+fy*(e-b),a+fx*(e-d)+fy*(d-a))
    return result


def physical_z_column_mask(z,valid,z_range,height):
    """Rasterize unions of valid consecutive-row Z intervals, without filling gaps."""
    low,high=z_range
    if high<=low or height<2:raise ValueError('Invalid physical Z mapping')
    y=(np.where(valid,z,low)-low)*(height-1)/(high-low)
    lower=np.ceil(np.minimum(y[:-1],y[1:])).astype(np.int64)
    upper=np.floor(np.maximum(y[:-1],y[1:])).astype(np.int64)
    ok=valid[:-1]&valid[1:]&(lower<=upper)&(upper>=0)&(lower<height)
    rows,cols=np.nonzero(ok)
    diff=np.zeros((height+1,z.shape[1]),np.int32)
    np.add.at(diff,(np.clip(lower[rows,cols],0,height-1),cols),1)
    np.add.at(diff,(np.clip(upper[rows,cols],0,height-1)+1,cols),-1)
    return np.cumsum(diff[:-1],axis=0)>0


def fit_vertical_affine(v,z):
    v=np.asarray(v,dtype=float);z=np.asarray(z,dtype=float)
    valid=np.isfinite(v)&np.isfinite(z);v=v[valid];z=z[valid]
    if len(v)<2 or np.var(v)<1e-12:raise ValueError('Degenerate vertical alignment')
    a=float(np.mean((v-v.mean())*(z-z.mean()))/np.var(v));b=float(z.mean()-a*v.mean())
    if not np.isfinite(a+b) or abs(a)<1e-8:raise ValueError('Degenerate vertical alignment')
    return [a,b]


def affine_vertical_setup(settings,surface,sample):
    pitch=surface.meta['output_step_vx']*settings.get('xyz_scale',4.)
    z=surface.full[2][::16,::64]*settings.get('xyz_scale',4.)
    v=np.broadcast_to(np.arange(0,surface.full[2].shape[0],16)[:,None]*pitch,z.shape)
    valid=np.isfinite(z)&(z>0)
    ref=fit_vertical_affine(v[valid],z[valid]);fits={}
    endpoints=[0.,80000.,ref[1],ref[0]*(surface.full[0].shape[0]-1)*pitch+ref[1]]
    for entry in settings['solutions']:
        root=Path(entry['run_root']);cfg=json.loads((root/'summary.json').read_text())['config']
        with np.load(cfg['tensor_cache']) as src:d={k:src[k] for k in ('phase','z','family','node_ds','lengths','start')}
        with np.load(root/'solution.npz') as src:
            _,_,keep,_,_,anchor_v,_=uv.hard_geometry(d,src['state'],src['widths'].astype(float),cfg)
        owner=sample['owner'];direction=(d['node_ds'][:,2]-d['node_ds'][:,0])/d['lengths']
        vv=anchor_v[owner]+(d['family'][owner]==1)*(sample['arc']-d['start'][owner])*direction[owner]
        retained=keep[owner];ix=np.flatnonzero(retained)[::32]
        fit=fit_vertical_affine(vv[ix],sample['xyz'][ix,2]);fits[entry['label']]=fit
        endpoints.extend([fit[0]*vv[retained].min()+fit[1],fit[0]*vv[retained].max()+fit[1]])
    return [float(min(endpoints)),float(max(endpoints))],ref,fits


def render_reference(surface,root,height,z_range=None,xyz_scale=4.,v_transform=None):
    import tifffile
    if v_transform is not None:
        pitch=surface.meta['output_step_vx']*xyz_scale
        factor=(height-1)/(z_range[1]-z_range[0]);a,b=v_transform
        h,w=surface.full[0].shape;width=int(np.ceil((w-1)*pitch*factor))+1;tile=512
        def tiles_affine():
            for top in range(0,height,tile):
                vertical=(np.arange(top,min(top+tile,height))/factor+z_range[0]-b)/a
                r=np.floor(vertical/pitch).astype(int);inside=(r>=0)&(r<h);r=r.clip(0,h-1)
                for left in range(0,width,tile):
                    c=np.minimum((np.arange(left,min(left+tile,width))/(factor*pitch)).astype(int),w-1)
                    valid=np.broadcast_to(inside[:,None],(len(r),len(c))).copy()
                    for channel in surface.full:
                        value=channel[np.ix_(r,c)];valid&=np.isfinite(value)&(value!=-1)
                    image=np.full((tile,tile,3),255,np.uint8)
                    image[:len(r),:len(c)]=np.where(valid[...,None],palette(winding_at(c,surface.x,surface.q))[None],255)
                    yield image
        tifffile.imwrite(root/'reference_own.tif',tiles_affine(),shape=(height,width,3),dtype=np.uint8,
                         tile=(tile,tile),photometric='rgb',compression='lzw',bigtiff=True,metadata=None)
        return
    if z_range is not None:
        pitch=surface.meta['output_step_vx']*xyz_scale
        factor=(height-1)/(z_range[1]-z_range[0])
        width=int(np.ceil((surface.full[0].shape[1]-1)*pitch*factor))+1
        # Strips remain bounded in memory; output tiles are emitted row-major.
        strip=np.memmap(root/'reference_z_raster.tmp',mode='w+',dtype=np.uint8,shape=(height,width,3))
        try:
            for left in range(0,width,512):
                cols=np.minimum((np.arange(left,min(left+512,width))/(factor*pitch)).astype(int),surface.full[0].shape[1]-1)
                valid=np.ones((surface.full[0].shape[0],len(cols)),bool)
                for a in surface.full:valid&=np.isfinite(a[:,cols])&(a[:,cols]!=-1)
                mask=physical_z_column_mask(surface.full[2][:,cols]*xyz_scale,valid,z_range,height)
                colors=palette(winding_at(cols,surface.x,surface.q))
                strip[:,left:left+len(cols)]=np.where(mask[...,None],colors[None],255)
            tifffile.imwrite(root/'reference_own.tif',strip,tile=(512,512),photometric='rgb',compression='lzw',bigtiff=True,metadata=None)
        finally:
            del strip
            (root/'reference_z_raster.tmp').unlink(missing_ok=True)
        return
    h,w=surface.full[0].shape; scale=height/h; width=int(np.ceil(w*scale)); tile=512
    def tiles():
        for y in range(0,height,tile):
            r=np.minimum((np.arange(y,min(y+tile,height))/scale).astype(int),h-1)
            for x in range(0,width,tile):
                c=np.minimum((np.arange(x,min(x+tile,width))/scale).astype(int),w-1)
                valid=np.ones((len(r),len(c)),bool)
                for a in surface.full:
                    values=a[np.ix_(r,c)];valid&=np.isfinite(values)&(values!=-1)
                image=np.full((tile,tile,3),255,np.uint8)
                color=palette(winding_at(c,surface.x,surface.q))
                image[:len(r),:len(c)]=np.where(valid[...,None],color[None],255)
                yield image
    tifffile.imwrite(root/'reference_own.tif',tiles(),shape=(height,width,3),dtype=np.uint8,
                     tile=(tile,tile),photometric='rgb',compression='lzw',bigtiff=True,metadata=None)


def preview(path):
    """Small central crop for checking output; full map remains TIFF/SVG."""
    import tifffile
    from PIL import Image
    image=tifffile.imread(path)
    h,w=image.shape[:2];left=max(0,w//2-h//2)
    detail=np.asarray(Image.fromarray(image[:,left:min(left+h,w)]).resize((1000,1000)))
    tifffile.imwrite(path.with_name(path.stem+'_detail.tif'),detail,
                     photometric='rgb',compression='lzw',metadata=None)


def convert_detail_pngs(root):
    """Migrate only known comparison detail crops; verify before removing PNG."""
    import tifffile
    from PIL import Image
    root=Path(root); converted=[]
    paths=[root/'reference_own_detail.png']
    paths.extend(root/label/(name+'_detail.png') for label in ('ccw','cw')
                 for name in ('fit_own','fit_reference','reference_fit'))
    for path in paths:
        if not path.exists():continue
        if path.is_symlink():raise ValueError('Refusing symlink detail migration')
        with Image.open(path) as image: pixels=np.array(image.convert('RGB'))
        target=path.with_suffix('.tif'); temporary=target.with_suffix('.partial.tif')
        if target.is_symlink() or temporary.exists() or temporary.is_symlink():
            raise ValueError('Unsafe or unfinished detail TIFF target')
        if target.exists() and not np.array_equal(tifffile.imread(target),pixels):
            raise ValueError('Existing detail TIFF has different pixels')
        tifffile.imwrite(temporary,pixels,photometric='rgb',compression='lzw',metadata=None)
        with tifffile.TiffFile(temporary) as image:
            if image.pages[0].compression.name!='LZW' or not np.array_equal(image.asarray(),pixels):
                raise ValueError('Detail TIFF verification failed')
        temporary.replace(target)
        path.unlink()
        converted.append(str(target))
    report=dict(converted=converted,count=len(converted),note='PNG removed only after pixel-identical LZW TIFF verified')
    uv.atomic_write(root/'tiff_detail_conversion.json',report,durable=False)
    return report


def write_palette(root,low,high):
    with (root/'palette.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['winding','red','green','blue'])
        ids=np.arange(int(low),int(high)+1)
        writer.writerows(zip(ids,*palette(ids).T))


def metric_report(settings):
    """CPU-only saved-state diagnostics; no rendering or optimization."""
    result={'solutions':[]}
    if settings.get('reference'):
        import tifffile
        xyz=np.stack([tifffile.imread(Path(settings['reference'])/(c+'.tif'))[::32,::8] for c in 'xyz'],axis=-1)
        valid=np.isfinite(xyz).all(-1)&(xyz>=0).all(-1)
        xyz=xyz*settings.get('xyz_scale',4.)
        control=json.loads(Path(settings['umbilicus']).read_text())['control_points']
        axis=np.array([[p['x'],p['y'],p['z']] for p in control]);axis=axis[np.argsort(axis[:,2])]
        delta=np.diff(xyz,axis=1);mid=(xyz[:,1:]+xyz[:,:-1])/2
        rx=mid[...,0]-np.interp(mid[...,2],axis[:,2],axis[:,0])
        ry=mid[...,1]-np.interp(mid[...,2],axis[:,2],axis[:,1])
        tangent=rx*delta[...,1]-ry*delta[...,0]
        ok=valid[:,1:]&valid[:,:-1]&(np.abs(tangent)>1e-8)
        positive=np.abs(tangent[ok&(tangent>0)]).sum()/np.abs(tangent[ok]).sum()
        result['reference_direction']=dict(convention='positive atan2(y-cy,x-cx) is CCW; increasing reference column',
            ccw_tangential_weight_percent=float(100*positive),direction='ccw' if positive>.5 else 'cw',
            sampled_valid_segments=int(ok.sum()))
    for entry in settings.get('solutions',[]):
        root=Path(entry['run_root']);cfg=json.loads((root/'summary.json').read_text())['config']
        with np.load(cfg['tensor_cache']) as src:d={k:src[k] for k in src.files}
        projection=load_metric_projection(settings['projection_cache'],d,cfg)
        if entry.get('initial_state',False):
            state,widths=uv.fresh_initial_state(d,cfg)
            metrics=reference_winding_metrics(d,state,widths,cfg,projection)
        else:
            with np.load(root/'solution.npz') as src:
                metrics=reference_winding_metrics(d,src['state'],src['widths'],cfg,projection)
        result['solutions'].append(dict(label=entry['label'],**metrics))
    uv.atomic_write(Path(settings['output_root'])/'metrics.json',result,durable=False)
    print(json.dumps(result),flush=True)
    return result


def projection_coverage_report(settings):
    """Read-only diagnosis of gray samples, in physical base-voxel Z."""
    import tifffile
    root=Path(settings['run_root']);cfg=json.loads((root/'summary.json').read_text())['config']
    with np.load(cfg['tensor_cache']) as src:d={k:src[k] for k in src.files}
    with np.load(root/'solution.npz') as src:
        _,_,keep,*_=uv.hard_geometry(d,src['state'],src['widths'],cfg)
    z=tifffile.imread(Path(settings['reference'])/'z.tif')
    valid=np.isfinite(z)&(z>0)
    zmin=float(z[valid].min()*settings.get('xyz_scale',4.))
    zmax=float(z[valid].max()*settings.get('xyz_scale',4.))
    with np.load(settings['projection_cache']) as src:
        for key in ('source_id','start','end'):
            if not np.array_equal(src[key],d[key]):raise ValueError('Projection identity mismatch')
        physical_z=src['xyz'][:,2];weight=src['weight']*keep[src['owner']]
        accepted=src['placed_valid'];nearest=src['nearest_distance']<=settings.get('max_distance_base',200.)
        chosen=src['chosen_winding'][src['owner']]
    gray=~accepted;outside=(physical_z<zmin)|(physical_z>zmax)
    total=weight.sum();gray_total=weight[gray].sum()
    result=dict(reference_physical_z=[zmin,zmax],fiber_physical_z=[float(physical_z[weight>0].min()),float(physical_z[weight>0].max())],
        retained_length_base=float(total),gray_length_percent=float(100*gray_total/total),
        gray_outside_reference_global_z_percent=float(100*weight[gray&outside].sum()/gray_total),
        gray_nearest_surface_within_threshold_percent=float(100*weight[gray&nearest].sum()/gray_total),
        gray_without_source_winding_percent=float(100*weight[gray&~np.isfinite(chosen)].sum()/gray_total),
        z_bins=[])
    for lo,hi in zip(range(0,80000,8000),range(8000,88000,8000)):
        mask=(physical_z>=lo)&(physical_z<hi);den=weight[mask].sum()
        if not den:continue
        result['z_bins'].append(dict(z=[lo,hi],retained_length_percent=float(100*den/total),
            nearest_accepted_percent=float(100*weight[mask&nearest].sum()/den),
            chosen_accepted_percent=float(100*weight[mask&accepted].sum()/den)))
    uv.atomic_write(Path(settings['output_root'])/'coverage.json',result,durable=False)
    print(json.dumps(result),flush=True)
    return result


def run(settings):
    if settings.get('projection_coverage_only'):return projection_coverage_report(settings)
    if settings.get('metrics_only'):return metric_report(settings)
    if settings.get('convert_detail_pngs_only',False):
        return convert_detail_pngs(settings['output_root'])
    root=Path(settings['output_root']);root.mkdir(parents=True,exist_ok=True)
    started=time.monotonic(); maximum=settings.get('max_distance_base',200.)
    surface=Surface(settings['reference'],settings.get('xyz_scale',4.),settings.get('surface_stride',2),
                    settings.get('workers',32),settings.get('candidates',4))
    first=Path(settings['solutions'][0]['run_root']);cfg=json.loads((first/'summary.json').read_text())['config']
    if not uv.anchored(cfg):raise ValueError('Reference comparison requires shared-anchor geometry')
    keys=('source_id','start','end','family','node_ds','lengths','phase','z')
    with np.load(cfg['tensor_cache']) as src:d={k:src[k] for k in keys}
    signature=dict(afv=uv.sha256_file(cfg['afv']),reference={c:uv.sha256_file(surface.root/(c+'.tif')) for c in 'xyz'},
                   metadata=uv.sha256_file(surface.root/'meta.json'),
                   parameters={k:settings.get(k) for k in ('xyz_scale','surface_stride','sample_spacing_base','max_distance_base','candidates')},
                   version=2)
    projection_root=Path(settings.get('projection_cache_root',root))
    stamp=projection_root/'projection_signature.json'; cache=projection_root/'projection.npz'
    reuse=stamp.exists() and cache.exists() and json.loads(stamp.read_text())==signature
    if projection_root!=root and not reuse:
        raise ValueError('External projection cache is missing or incompatible')
    if reuse:
        print('reference: reusing validated projection cache',flush=True)
        with np.load(cache) as src:
            for k in ('source_id','start','end'):
                if not np.array_equal(src[k],d[k]):raise ValueError('Projection piece identity mismatch')
            sample={k:src[k] for k in ('xyz','arc','weight','owner','offset')}
            rc=src['nearest_rc'];distance=src['nearest_distance'];ref_q=src['nearest_q'];chosen=src['chosen_winding']
            placed=src['placed_rc'];placed_dist=src['placed_distance']
    else:
        print('reference: sampling original AFV curves',flush=True)
        sample=samples(d,cfg,settings.get('sample_spacing_base',200.))
        print(f'reference: {len(sample["xyz"]):,} curve samples',flush=True)
        rc,distance=surface.project(sample['xyz'])
        ref_q=winding_at(rc[:,1],surface.x,surface.q)
        chosen=assign_sources(d,sample,ref_q,distance,maximum,surface.limits)
        target=chosen[sample['owner']]
        placed=np.full_like(rc,np.nan); placed_dist=np.full(len(rc),np.inf)
        for number in np.unique(target[np.isfinite(target)]):
            ids=np.flatnonzero(target==number); cells=np.flatnonzero(surface.labels==number)
            placed[ids],placed_dist[ids]=surface.project(sample['xyz'][ids],cells)
            print(f'reference chosen-winding projection {number:g}: {len(ids):,} samples',flush=True)
    target=chosen[sample['owner']]
    placed_q=winding_at(placed[:,1],surface.x,surface.q)
    valid=(placed_dist<=maximum)&(discrete_winding(placed_q,surface.limits)==target)
    if not reuse:
        temporary=root/'projection.partial.npz'
        np.savez_compressed(temporary,**sample,nearest_rc=rc,nearest_distance=distance,
                            nearest_q=ref_q,chosen_winding=chosen,placed_rc=placed,placed_distance=placed_dist,placed_valid=valid,
                            source_id=d['source_id'],start=d['start'],end=d['end'])
        temporary.replace(cache)
        uv.atomic_write(stamp,signature,durable=False)
    pitch=surface.meta['output_step_vx']*settings.get('xyz_scale',4.)
    edges=surface_edges(surface,placed,valid,sample['owner'],sample['arc'],pitch)
    height=settings.get('render_height',1000)
    z_range=settings.get('physical_z_range_base')
    reference_affine=None;layout_affines={}
    if settings.get('vertical_alignment')=='affine_z':
        z_range,reference_affine,layout_affines=affine_vertical_setup(settings,surface,sample)
    render_reference(surface,root,height,z_range,settings.get('xyz_scale',4.),reference_affine)
    preview(root/'reference_own.tif')
    write_palette(root,*surface.limits)
    reports=[]
    for entry in settings['solutions']:
        source=Path(entry['run_root']);dest=root/entry['label'].lower();dest.mkdir(exist_ok=True)
        cfg=json.loads((source/'summary.json').read_text())['config']
        if not uv.anchored(cfg):raise ValueError('Reference comparison requires shared-anchor geometry')
        with np.load(cfg['tensor_cache']) as src:
            for key in ('source_id','start','end','family'):
                if not np.array_equal(d[key],src[key]):raise ValueError('Solution source identity mismatch')
            local={k:src[k] for k in keys}
        with np.load(source/'solution.npz') as src:
            state=src['state'];widths=src['widths'].astype(float)
            for key in ('source_id','start','end','family'):
                if not np.array_equal(src[key],d[key]):raise ValueError('Solution piece identity mismatch')
        p,_,keep,pick,u,v,_=uv.hard_geometry(local,state,widths,cfg)
        owner=sample['owner'];direction=(local['node_ds'][:,2]-local['node_ds'][:,0])/local['lengths']
        ds=(sample['arc']-local['start'][owner])*direction[owner]
        xx=u[owner]+(local['family'][owner]==0)*ds;yy=v[owner]+(local['family'][owner]==1)*ds
        if reference_affine is not None:
            a,b=layout_affines[entry['label']];yy=a*yy+b
        elif z_range is not None:yy=sample['xyz'][:,2]
        fit_q=inverse_u(xx,widths)
        matched=(distance<=maximum)&keep[owner]
        if not matched.any():raise ValueError('No retained fiber samples matched reference within distance threshold')
        offset=visualization_offset(settings,ref_q[matched],fit_q[matched],sample['weight'][matched])
        fit_q+=offset
        alpha=p[np.arange(len(p)),pick][owner]
        bounds=[float(xx[keep[owner]].min()),float(yy[keep[owner]].min()),float(xx[keep[owner]].max()),float(yy[keep[owner]].max())]
        if z_range is not None:bounds[1],bounds[3]=z_range
        maps={}
        reference_pitch=surface.meta['output_step_vx']*settings.get('xyz_scale',4.)
        refcoords=placed[:,::-1]*reference_pitch
        refbounds=[0,0,(surface.full[0].shape[1]-1)*reference_pitch,(surface.full[0].shape[0]-1)*reference_pitch]
        if z_range is not None:
            if reference_affine is not None:refcoords[:,1]=reference_affine[0]*refcoords[:,1]+reference_affine[1]
            else:refcoords[:,1]=projected_surface_z(surface,placed)
            refbounds[1],refbounds[3]=z_range
        for name,coords,colorq,ok,bbox in (
            ('fit_own',np.c_[xx,yy],fit_q,keep[owner],bounds),
            ('fit_reference',np.c_[xx,yy],np.where(valid,placed_q,np.nan),keep[owner],bounds),
            ('reference_fit',refcoords,fit_q,valid&keep[owner],refbounds)):
            print(f'reference: rendering {entry["label"]} {name}',flush=True)
            maps[name]=render_curves(dest/name,coords,palette(colorq),owner,ok,alpha,height,bbox,
                                     edges if name=='reference_fit' else None)
            preview((dest/name).with_suffix('.tif'))
        weights=sample['weight'];total=weights[keep[owner]].sum();err=fit_q-ref_q
        report=dict(label=entry['label'],global_integer_offset=offset,retained_pieces=int(keep.sum()),
                    vertical_mapping=(dict(coordinate='affine_Z_aligned_V' if reference_affine is not None else 'physical_base_Z',
                        reference_affine=reference_affine,fit_affine=layout_affines.get(entry['label']),range=z_range,height=height,
                        pixels_per_base_voxel=(height-1)/(z_range[1]-z_range[0])) if z_range is not None else dict(coordinate='layout_V')),
                    retained_source_fibers=int(len(np.unique(d['source_id'][keep]))),
                    nearest_matched_length_percent=float(100*weights[matched].sum()/total),
                    chosen_winding_projected_length_percent=float(100*weights[valid&keep[owner]].sum()/total),
                    nearest_signed_error_median=float(weighted_median(err[matched],weights[matched])),
                    nearest_absolute_error_median=float(weighted_median(abs(err[matched]),weights[matched])),
                    nearest_within_one_percent=float(100*weights[matched&(abs(err)<=1)].sum()/weights[matched].sum()),
                    matched_distance_quantiles=np.quantile(distance[matched],[0,.5,.95,.99,1]).tolist(),
                    projected_source_fibers=int(len(np.unique(d['source_id'][owner[valid&keep[owner]]]))),
                    maps=maps,solution_sha256=uv.sha256_file(source/'solution.npz'))
        report['fiber_winding_accuracy']=reference_winding_metrics(local,state,widths,cfg,
            dict(sample,chosen_winding=chosen))
        with (dest/'fibers.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['piece','source_id','start','end','retained','reference_winding','fit_anchor_winding','joint_probability'])
            writer.writerows(zip(range(len(keep)),d['source_id'],d['start'],d['end'],keep,chosen,inverse_u(u,widths)+offset,p[np.arange(len(p)),pick]))
        uv.atomic_write(dest/'summary.json',report,durable=False);reports.append(report)
    uv.atomic_write(root/'summary.json',dict(config=settings,results=reports,seconds=time.monotonic()-started,
        reference_metadata_sha256=uv.sha256_file(surface.root/'meta.json'),afv_sha256=uv.sha256_file(cfg['afv']),
        implementation_sha256=uv.sha256_file(__file__),landmark_columns=surface.x.tolist(),landmark_windings=surface.q.tolist(),
        notes='Approximate triangle patches; bounded nearest-center candidates; rectangle winding labels; no path optimization; original-source median before restricted projection'),durable=False)
    return reports
