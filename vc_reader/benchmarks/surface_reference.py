"""Slow independent float64 oracle for zero-thickness projection acceptance.

This deliberately reads small recorded geometry/label fixtures directly and
uses NumPy scalar geometry, never the production clipping or raster kernels.
Double precision is confined to this validation oracle.
"""
import itertools
import numpy as np


def crop_triangles(fields, placement, grid_to_label, origin, shape):
    """Stream raw geometry; retain complete triangles touching the 3D crop."""
    height,width=fields[0].shape
    lower=np.asarray(origin[::-1],np.float64)-.5
    upper=lower+np.asarray(shape[::-1])
    for y in range(0,height-1,256):
        for x in range(0,width-1,256):
            size=(min(257,height-y),min(257,width-x))
            xyz=np.stack([r.read((y,x),size) for r in fields],axis=-1).astype(np.float64)
            valid=np.isfinite(xyz).all(axis=-1)&(xyz>=0).all(axis=-1)
            iy,ix=np.nonzero(valid[:-1,:-1]&valid[1:,:-1]&valid[:-1,1:]&valid[1:,1:])
            xyz=xyz@np.asarray(placement.linear).T+np.asarray(placement.translation)
            for ordinal,indices in enumerate((((0,0),(0,1),(1,1)),((0,0),(1,1),(1,0)))):
                points=np.stack([xyz[iy+dy,ix+dx] for dy,dx in indices],axis=1)
                keep=(points.max(axis=1)>=lower).all(axis=1)&(points.min(axis=1)<=upper).all(axis=1)
                for row,col,triangle in zip(iy[keep],ix[keep],points[keep]):
                    if not np.any(np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])):
                        continue
                    pixels=np.array([grid_to_label.apply((x+col+dx,y+row+dy,0))[:2] for dy,dx in indices])
                    yield ((y+row)*(width-1)+x+col)*2+ordinal,np.concatenate((triangle,pixels),axis=1)


def label_sampler(reader):
    """Independent bounded nearest-pixel reads, including distant folded regions."""
    from collections import OrderedDict
    tiles=OrderedDict()
    def sample(y,x):
        if not (0<=y<reader.shape[0] and 0<=x<reader.shape[1]):
            return None
        start=(y//128*128,x//128*128)
        if start not in tiles:
            if len(tiles)==32:
                tiles.popitem(last=False)
            tiles[start]=reader.read(start,tuple(min(128,n-a) for n,a in zip(reader.shape,start)))
        tiles.move_to_end(start)
        return tiles[start][y-start[0],x-start[1]]
    return sample


def nearest(p,triangle):
    a,b,c=triangle
    ab,ac=b-a,c-a
    basis=np.column_stack((ab,ac))
    bary=np.linalg.solve(basis.T@basis,basis.T@(p-a))
    if min(bary)>=0 and sum(bary)<=1:
        return a+basis@bary,np.array((1-sum(bary),*bary))
    candidates=[]
    for i,j in ((0,1),(1,2),(2,0)):
        d=triangle[j]-triangle[i]
        t=np.clip(np.dot(p-triangle[i],d)/np.dot(d,d),0,1)
        q=triangle[i]+d*t
        w=np.zeros(3);w[i]=1-t;w[j]=t
        candidates.append((np.dot(q-p,q-p),q,w))
    _,q,w=min(candidates,key=lambda item:item[0])
    return q,w


def intersects(relative):
    basis=np.eye(3)
    edges=np.roll(relative,-1,axis=0)-relative
    axes=[*basis,np.cross(edges[0],edges[1]),*[np.cross(e,b) for e in edges for b in basis]]
    for axis in axes:
        values=relative@axis
        radius=np.abs(axis).sum()/2
        if values.min()>radius or values.max()<-radius:return False
    return True


def project(triangles,label_at,dtype,origin,shape,spacing):
    output=np.zeros(shape,dtype);valid=np.zeros(shape,bool);conflict=np.zeros(shape,bool)
    coverage=np.zeros(shape,bool)
    distance=np.full(shape,np.inf);winners=np.full(shape,np.iinfo(np.int64).max,np.int64)
    winner_pixels=np.full((*shape,2),np.nan)
    for identifier,triangle in triangles:
        points=triangle[:,:3]-np.array(origin[::-1])
        lower=np.maximum(0,np.ceil(points.min(0)-.5).astype(int))[::-1]
        upper=np.minimum(np.array(shape[::-1])-1,np.floor(points.max(0)+.5).astype(int))[::-1]
        for index in itertools.product(*(range(a,b+1) for a,b in zip(lower,upper))):
            xyz=np.array(index[::-1])
            if not intersects(points-xyz):continue
            coverage[index]=True
            q,w=nearest(xyz*np.array(spacing),points*np.array(spacing))
            active=np.flatnonzero(w)
            if len(active)==1:
                pixel=triangle[active[0],3:]
            elif len(active)==2:
                a,b=active
                pixel=triangle[a,3:]+w[b]*(triangle[b,3:]-triangle[a,3:])
            else:
                pixel=triangle[0,3:]+w[1]*(triangle[1,3:]-triangle[0,3:])+w[2]*(triangle[2,3:]-triangle[0,3:])
            # Surface intersections can land on exact half-pixel ties. Remove float64
            # interpolation roundoff before applying nearest-even categorical sampling.
            half=np.round(pixel*2)/2
            pixel=np.where(np.abs(pixel-half)<1e-9,half,pixel)
            ix,iy=np.rint(pixel).astype(int)
            value=label_at(iy,ix)
            if value is None:continue
            dist=np.sum((q-xyz*np.array(spacing))**2)
            if valid[index] and output[index]!=value:conflict[index]=True
            if not valid[index] or (dist,identifier)<(distance[index],winners[index]):
                output[index]=value;distance[index]=dist;winners[index]=identifier;winner_pixels[index]=pixel
            valid[index]=True
    return output,valid,conflict,winner_pixels,coverage
