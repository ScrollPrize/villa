"""Slow independent float64 oracle for zero-thickness projection acceptance.

This deliberately reads small recorded geometry/label fixtures directly and
uses NumPy scalar geometry, never the production clipping or raster kernels.
Double precision is confined to this validation oracle.
"""
import itertools
import numpy as np


def clipped_triangles(xyz, grid_to_label, start, size):
    height,width=xyz.shape[:2]
    lower=np.array(start[::-1],np.float64)-.5
    upper=lower+np.array(size[::-1])
    for y in range(height-1):
        for x in range(width-1):
            for ordinal,indices in enumerate((((y,x),(y,x+1),(y+1,x+1)),((y,x),(y+1,x+1),(y+1,x)))):
                points=np.array([xyz[index] for index in indices],np.float64)
                if not np.isfinite(points).all() or (points<0).any():continue
                pixels=np.array([grid_to_label.apply((col,row,0))[:2] for row,col in indices])
                if np.any(pixels.max(0)<lower) or np.any(pixels.min(0)>upper):continue
                polygon=list(np.concatenate((points,pixels),axis=1))
                for axis,side in itertools.product((0,1),(0,1)):
                    edge=(lower if side==0 else upper)[axis]
                    sign=1 if side==0 else -1
                    old=polygon;polygon=[]
                    if not old:break
                    for a,b in zip(old,old[1:]+old[:1]):
                        da,db=sign*(a[axis+3]-edge),sign*(b[axis+3]-edge)
                        if da>=0:polygon.append(a)
                        if (da<0)!=(db<0):
                            intersection=(a*db-b*da)/(db-da)
                            intersection[axis+3]=edge
                            polygon.append(intersection)
                for j in range(1,len(polygon)-1):
                    tri=np.array((polygon[0],polygon[j],polygon[j+1]))
                    if np.linalg.norm(np.cross(tri[1,:3]-tri[0,:3],tri[2,:3]-tri[0,:3]))>0:
                        yield ((y*(width-1)+x)*2+ordinal),tri


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


def project(triangles,labels,label_origin,start,size,origin,shape,spacing):
    output=np.zeros(shape,labels.dtype);coverage=np.zeros(shape,bool);conflict=np.zeros(shape,bool)
    distance=np.full(shape,np.inf);winners=np.full(shape,np.iinfo(np.int64).max,np.int64)
    winner_pixels=np.full((*shape,2),np.nan)
    for identifier,triangle in triangles:
        points=triangle[:,:3]-np.array(origin[::-1])
        lower=np.maximum(0,np.ceil(points.min(0)-.5).astype(int))[::-1]
        upper=np.minimum(np.array(shape[::-1])-1,np.floor(points.max(0)+.5).astype(int))[::-1]
        for index in itertools.product(*(range(a,b+1) for a,b in zip(lower,upper))):
            xyz=np.array(index[::-1])
            if not intersects(points-xyz):continue
            q,w=nearest(xyz*np.array(spacing),points*np.array(spacing))
            active=np.flatnonzero(w)
            if len(active)==1:
                pixel=triangle[active[0],3:]
            elif len(active)==2:
                a,b=active
                pixel=triangle[a,3:]+w[b]*(triangle[b,3:]-triangle[a,3:])
            else:
                pixel=triangle[0,3:]+w[1]*(triangle[1,3:]-triangle[0,3:])+w[2]*(triangle[2,3:]-triangle[0,3:])
            ix,iy=np.rint(pixel).astype(int)
            if not(start[0]<=iy<start[0]+size[0] and start[1]<=ix<start[1]+size[1]):continue
            value=labels[iy-label_origin[0],ix-label_origin[1]]
            dist=np.sum((q-xyz*np.array(spacing))**2)
            if coverage[index] and output[index]!=value:conflict[index]=True
            if not coverage[index] or (dist,identifier)<(distance[index],winners[index]):
                output[index]=value;distance[index]=dist;winners[index]=identifier;winner_pixels[index]=pixel
            coverage[index]=True
    return output,coverage,conflict,winner_pixels
