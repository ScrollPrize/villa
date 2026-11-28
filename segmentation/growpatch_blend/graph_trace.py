"""Graph trace of one papyrus layer from a surface-prediction cube, around a seed.

Vendored from the GrowPatch blend benchmark (growbench_recover.py, MIT). The only change is that the traced
field is returned in memory instead of written to .npz files; the numbers are identical.

The trace works in a 193 x 193 chart (2-voxel spacing, 96-voxel radius) laid on the two axes orthogonal to the
seed normal's dominant axis. It finds ridges (local maxima above 0.15 of the cube maximum) along that axis, links
them using local orientation, and keeps the ridge path connected to the seed. ValueError means it abstains.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

STEP = 2
RADIUS = 96
DEPTH = 111


def evidence(volume, site):
    """Resample a height chart and locate local maxima across candidate sheets."""
    seed = np.asarray(site["seed_xyz"], dtype=np.float64)
    normal = np.asarray(site["normal_xyz"], dtype=np.float64)
    dep = int(np.argmax(np.abs(normal)))
    u, v = [axis for axis in range(3) if axis != dep]
    depth = np.arange(-DEPTH, DEPTH + 1, dtype=np.float32)
    sides = np.arange(-RADIUS, RADIUS + 1, STEP, dtype=np.float32)
    vv, uu, dd = np.meshgrid(sides, sides, depth, indexing="ij")
    xyz = np.empty((3, *vv.shape), np.float32)
    for dim, offsets in ((u, uu), (v, vv), (dep, dd)):
        xyz[dim] = offsets + seed[dim] - site["origin_xyz"][dim]
    sampled = map_coordinates(volume, xyz[::-1], order=1, output=np.float32,
                              mode="constant", cval=0.0, prefilter=False)
    if not sampled.max():
        raise ValueError("No surface prediction in the chart")
    sampled /= float(volume.max())
    # The chart spans the central portion of the crop; all samples are within
    # its provided volume. Normal and transverse smoothing differ in voxel units.
    signal = gaussian_filter(sampled, (0.55,0.55,0.85))
    peaks = np.zeros(signal.shape, dtype=bool)
    peaks[...,1:-1] = ((signal[...,1:-1] >= signal[...,:-2]) &
                       (signal[...,1:-1] > signal[...,2:]) &
                       (signal[...,1:-1] > .15))
    iy, ix, iz = np.where(peaks)
    n, m = signal.shape[:2]
    before, middle, after = signal[iy,ix,iz-1],signal[iy,ix,iz],signal[iy,ix,iz+1]
    denom = np.maximum(2*middle-before-after,1e-6)
    dz = np.clip((after-before)/(2*denom),-.5,.5)
    peak_z = (depth[iz]+dz).astype(np.float32)
    # Principal direction of the structure tensor in the native orthogonal
    # basis (u,v,depth). Evaluate only on candidate ridges.
    dv, du, dn = np.gradient(signal, STEP, STEP, 1)
    grads = (du, dv, dn)
    tensor = np.zeros((len(iy),3,3),np.float32)
    for a in range(3):
        for b in range(a+1):
            val = gaussian_filter(grads[a]*grads[b], (1.25,1.25,1.8))[iy,ix,iz]
            tensor[:,a,b] = tensor[:,b,a] = val
    eig, vec = np.linalg.eigh(tensor)
    nor = vec[:,:,-1]
    norm_depth = np.maximum(np.abs(nor[:,2]), .24)
    slopes = -nor[:,:2] * np.where(nor[:,2:3]>=0,1.,-1.) / norm_depth[:,None]
    slopes = np.clip(slopes,-2.8,2.8).astype(np.float32)
    coherence = np.clip((eig[:,-1]-eig[:,-2])/(eig[:,-1]+1e-12),0,1)
    ray = iy*m+ix
    counts = np.bincount(ray,minlength=n*m)
    offsets = np.r_[0,np.cumsum(counts)]
    max_count = int(counts.max())
    ids = np.full((n*m,max_count),-1,np.int32)
    ids[ray,np.arange(len(ray))-offsets[ray]] = np.arange(len(ray),dtype=np.int32)
    return dict(seed=seed,axes=(u,v,dep),sides=sides,iy=iy,ix=ix,iz=iz,
                height=peak_z,slopes=slopes,coherence=coherence,
                density=middle,ids=ids,n=n,m=m,signal=signal)


def connect(ev):
    """Use tangent extrapolation and intermediate ridge support to link nearby peaks."""
    iy,ix,rz=ev["iy"],ev["ix"],ev["height"]
    slopes,coherence,density=ev["slopes"],ev["coherence"],ev["density"]
    n,m=ev["n"],ev["m"]
    source_parts, target_parts, weights = [],[],[]
    cardinal = {direction: np.full(len(iy),-1,np.int32)
                for direction in ((1,0),(-1,0),(0,1),(0,-1))}
    for dy, dx in ((0,1),(1,0),(1,1),(1,-1),(0,3),(3,0)):
        y,x=iy+dy,ix+dx
        src=np.flatnonzero((y>=0)&(x>=0)&(y<n)&(x<m))
        if not len(src):
            continue
        neighbors=ev["ids"][y[src]*m+x[src]]
        a,b=np.where(neighbors>=0)
        src=src[a].astype(np.int32)
        dest=neighbors[a,b]
        run=STEP*np.hypot(dx,dy)
        rise=rz[dest]-rz[src]
        keep=np.abs(rise)<=1.9*run+1.7
        src,dest,rise=src[keep],dest[keep],rise[keep]
        slope0=STEP*(dx*slopes[src,0]+dy*slopes[src,1])
        slope1=STEP*(dx*slopes[dest,0]+dy*slopes[dest,1])
        error0=np.abs(rise-slope0)
        error1=np.abs(rise-slope1)
        residual=np.abs(rise-(slope0+slope1)*.5)
        quality=np.minimum(coherence[src],coherence[dest])
        # Limit excessive hops even where tangent estimates are unreliable.
        keep=(residual <= 1.5+.22*run+1.1*(1-quality)) & (np.maximum(error0,error1) < 3.0+.55*run)
        src,dest,rise,residual,quality=src[keep],dest[keep],rise[keep],residual[keep],quality[keep]
        if not len(src):
            continue
        cost=run*(1+.24*(1-quality)+.20*(1-np.minimum(density[src],density[dest]))) + 1.6*residual
        if max(abs(dx),abs(dy))>1:
            # Longer links can reconnect short breaks, but a fully observed
            # local route should win over a speculative bridge.
            py=iy[src]+.5*dy
            px=ix[src]+.5*dx
            pd=DEPTH+rz[src]+.5*rise
            mid=map_coordinates(ev["signal"],[py,px,pd],order=1,prefilter=False)
            cost += 1.8 + 3.4*(1-mid)
        source_parts.append(src);target_parts.append(dest);weights.append(cost)
        if (dy,dx) in cardinal:
            for direction,start,end in [((dy,dx),src,dest),((-dy,-dx),dest,src)]:
                # Prefer the cheapest candidate among all immediate neighbors;
                # use its table for the two axis-order paths.
                order=np.lexsort((end,cost,start))
                first=np.r_[True,start[order[1:]]!=start[order[:-1]]]
                take=order[first]
                cardinal[direction][start[take]] = end[take]
    if not source_parts:
        raise ValueError("The seed chart has no connected ridges")
    s,t,w=map(np.concatenate,(source_parts,target_parts,weights))
    nodes=len(iy)
    graph=coo_matrix((w,(s,t)),shape=(nodes,nodes)).tocsr()
    return graph,cardinal


def axis_routes(ev, anchor, cardinal):
    mid=ev["n"]//2
    def route(primary):
        rays=np.full((ev["n"],ev["m"]),-1,np.int32)
        rays[mid,mid]=anchor
        def extend(y,x,direction):
            dy,dx=direction
            ident=rays[y,x]
            while ident>=0 and 0<=y+dy<ev["n"] and 0<=x+dx<ev["m"]:
                ident=cardinal[direction][ident]
                y+=dy;x+=dx
                rays[y,x]=ident
        for sign in (-1,1):
            extend(mid,mid,(0,sign) if primary=="u" else (sign,0))
        for idx in range(ev["n"]):
            for sign in (-1,1):
                if primary=="u":
                    extend(mid,idx,(sign,0))
                else:
                    extend(idx,mid,(0,sign))
        return rays
    return route("u"),route("v")


def make_field(ev,chosen,quality):
    ok=chosen>=0
    vv,uu=np.meshgrid(ev["sides"],ev["sides"],indexing="ij")
    u,v,d=ev["axes"]
    xyz=np.empty((*ok.shape,3),np.float32)
    xyz[...,u]=ev["seed"][u]+uu
    xyz[...,v]=ev["seed"][v]+vv
    xyz[...,d]=ev["seed"][d]+ev["height"][np.maximum(chosen,0)]
    xyz[~ok]=np.nan
    return dict(xyz=xyz,valid=ok,quality=np.where(ok,quality,0).astype(np.float32))


def solve(volume,site):
    ev=evidence(volume,site)
    graph,cardinal=connect(ev)
    mid=ev["n"]//2
    center=ev["ids"][mid*ev["m"]+mid]
    center=center[center>=0]
    if not len(center):
        raise ValueError("No M7 ridge on seed ray")
    h=ev["height"]
    anchor=int(center[np.argmin(np.abs(h[center]))])
    if abs(h[anchor])>6:
        raise ValueError(f"Seed is {abs(h[anchor]):.3f} cue voxels from closest ridge")
    paths=axis_routes(ev,anchor,cardinal)
    left,right=paths
    agreed=(left>=0)&(right>=0)&(np.abs(h[np.maximum(left,0)]-h[np.maximum(right,0)])<=2)
    two=np.where(agreed,left,-1)
    simple_q=np.sqrt(ev["density"][np.maximum(two,0)]*ev["coherence"][np.maximum(two,0)])
    stats={}
    pos=dijkstra(graph,directed=False,indices=anchor)
    other=center[(center!=anchor)&(np.abs(h[center]-h[anchor])<65)]
    neg=(dijkstra(graph,directed=False,indices=other,min_only=True)
         if len(other) else np.full_like(pos,np.inf))
    clearance=np.zeros_like(pos)
    finite=np.isfinite(pos)&np.isfinite(neg)
    clearance[finite]=np.clip((neg[finite]-pos[finite])/8.,-5,5)
    clearance[np.isfinite(pos)&~np.isfinite(neg)]=5
    ids=ev["ids"];safe=np.maximum(ids,0)
    costs=np.where(ids>=0,pos[safe]+8*np.maximum(0,0.6-clearance[safe]),np.inf)
    best_col=np.argmin(costs,axis=1)
    best=ids[np.arange(len(ids)),best_col].reshape(ev["n"],ev["m"])
    reachable=np.isfinite(np.min(costs,axis=1)).reshape(best.shape)
    best=np.where(reachable,best,-1)
    # Locally consistent, independently arrived axis traces temporarily outrank
    # the cheapest longer crossing, but never overrule an adjacent seeded sheet.
    keep_prior=agreed&(clearance[np.maximum(two,0)]>-.15)
    best=np.where(keep_prior,two,best)
    safe=np.maximum(best,0)
    axis_votes=((left==best)&(left>=0)).astype(float)+((right==best)&(right>=0))
    margin=1/(1+np.exp(-clearance[safe]))
    local_confirm=np.zeros_like(margin)
    for dy,dx in ((0,1),(0,-1),(1,0),(-1,0)):
        nbr=np.roll(best,(-dy,-dx),axis=(0,1))
        if dx>0:nbr[:,-1]=-1
        if dx<0:nbr[:,0]=-1
        if dy>0:nbr[-1,:]=-1
        if dy<0:nbr[0,:]=-1
        local_confirm+=(cardinal[(dy,dx)][safe]==nbr)&(nbr>=0)
    quality=(.4*margin + .20*axis_votes/2 + .25*local_confirm/4
             + .15*np.sqrt(ev["density"][safe]*ev["coherence"][safe]))
    quality[mid,mid]=1
    stats["field"]=make_field(ev,best,quality)
    stats["graph"]=int(stats["field"]["valid"].sum())
    stats.update(anchor_offset=float(h[anchor]),depth_axis_xyz=ev["axes"][-1],
                 peak_nodes=int(len(h)),neighbor_origin_count=int(len(other)))
    return stats
