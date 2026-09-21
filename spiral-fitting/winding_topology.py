"""Original patch faces and migration of legacy quad-center workspaces."""
import copy
import json
import os
from pathlib import Path
import uuid

import numpy as np


def quad_runs(mask):
    """Sorted [start, length] runs in row-major *cell* order."""
    edges = np.diff(np.r_[False, np.asarray(mask, bool).ravel(), False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return np.column_stack((starts, ends - starts)).tolist()


def source_quad_mask(grid, source=None):
    valid = grid[..., 0] != -1
    mask = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    if source is not None and 'quad_runs' in source:
        explicit = np.zeros(mask.size, bool)
        end = 0
        for start, length in source['quad_runs']:
            if (type(start) is not int or type(length) is not int or
                    start < end or length <= 0 or start + length > mask.size):
                raise ValueError('Invalid workspace quad runs')
            explicit[start:start + length] = True
            end = start + length
        explicit = explicit.reshape(mask.shape)
        if np.any(explicit & ~mask):
            raise ValueError('Accepted quad has a missing vertex')
        mask = explicit
    return mask


def patch_corners(xyz, rows, cols):
    """Crop original vertices, retaining precisely the selected source cells."""
    r0, c0 = int(rows.min()), int(cols.min())
    r, c = rows - r0, cols - c0
    mask = np.zeros((int(r.max()) + 1, int(c.max()) + 1), bool)
    mask[r, c] = True
    used = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), bool)
    for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)):
        used[r + dr, c + dc] = True
    points = np.asarray(xyz[r0:r0 + used.shape[0], c0:c0 + used.shape[1]], float)
    if points.shape[:2] != used.shape or not np.isfinite(points[used]).all() or np.any(points[used, 0] == -1):
        raise ValueError('Original patch no longer contains the accepted quads')
    grid = np.zeros((*used.shape, 5))
    grid[..., :3] = -1
    grid[used, :3] = points[used]
    return grid, mask, (r0, c0)


def lift_center_uv(centers, corners, mask):
    """Extend the existing placement half a cell onto original shared corners.

    Local affine derivatives preserve smooth fitted UV, including its tilt.
    Where a one-cell-wide component cannot measure both derivatives, the
    unobserved direction comes from the original 3D tangent metric.
    """
    r, c = np.where(mask)
    a, b, d, e = (corners[r + dr, c + dc, :3] for dr, dc in
                  ((0, 0), (0, 1), (1, 0), (1, 1)))
    tangent = np.stack(((d + e - a - b) / 2, (b + e - a - d) / 2), axis=1)
    normal = np.cross(tangent[:, 0], tangent[:, 1])
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-20)
    vertical = np.array([0., 0., 1.]) - normal * normal[:, 2, None]
    weak = np.linalg.norm(vertical, axis=1) < 1e-8
    vertical[weak] = tangent[weak, 0]
    vertical /= np.maximum(np.linalg.norm(vertical, axis=1)[:, None], 1e-20)
    horizontal = np.cross(vertical, normal)
    prior = np.stack((np.einsum('nij,nj->ni', tangent, horizontal),
                      np.einsum('nij,nj->ni', tangent, vertical)), axis=2)
    lhs = np.zeros((len(r), 2, 2)); rhs = np.zeros((len(r), 2, 2))
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == dc == 0:
                continue
            rr, cc = r + dr, c + dc
            ok = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
            ok[ok] &= mask[rr[ok], cc[ok]]
            delta = centers[rr[ok], cc[ok], 3:] - centers[r[ok], c[ok], 3:]
            step = np.array([dr, dc])
            lhs[ok] += np.outer(step, step)
            rhs[ok] += step[None, :, None] * delta[:, None, :]
    inverse = np.linalg.pinv(lhs, hermitian=True)
    measured = inverse @ rhs
    # Orient the metric completion consistently with the observed horizontal
    # direction. Do not assume source row/column ordering or handedness.
    if np.sum(prior[:, :, 0] * measured[:, :, 0]) < 0:
        prior[:, :, 0] *= -1
    derivative = measured + (np.eye(2) - inverse @ lhs) @ prior
    # A strip's unobserved direction must use the same handedness as its
    # neighboring two-dimensional region. A generic Z-based completion can
    # point backwards at a bent, one-quad-wide neck.
    eigenvalues, eigenvectors = np.linalg.eigh(lhs)
    full = eigenvalues[:, 0] > 1e-8
    determinant = np.linalg.det(measured[full])
    sign = np.sign(np.median(determinant)) if len(determinant) else np.sign(np.median(np.linalg.det(prior)))
    strip = (~full) & (eigenvalues[:, 1] > 1e-8)
    if strip.any():
        along = eigenvectors[strip, :, 1]
        across = np.column_stack((-along[:, 1], along[:, 0]))
        physical = np.einsum('ni,nij->nj', along, tangent[strip])
        lateral = np.einsum('ni,nij->nj', across, tangent[strip])
        length = np.maximum(np.linalg.norm(physical, axis=1), 1e-20)
        parallel = np.sum(physical*lateral, axis=1) / length
        perpendicular = np.linalg.norm(np.cross(physical, lateral), axis=1) / length
        direction = np.einsum('ni,nij->nj', along, measured[strip])
        direction /= np.maximum(np.linalg.norm(direction, axis=1)[:, None], 1e-20)
        sideways = np.column_stack((-direction[:, 1], direction[:, 0]))
        completion = direction*parallel[:, None] + sideways*(sign*perpendicular[:, None])
        derivative[strip] = measured[strip] + across[:, :, None]*completion[:, None, :]
    total = np.zeros(corners.shape[:2] + (2,)); count = np.zeros(corners.shape[:2])
    for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)):
        placed = centers[r, c, 3:] + np.einsum('i,nij->nj', [dr - .5, dc - .5], derivative)
        total[r + dr, c + dc] += placed
        count[r + dr, c + dc] += 1
    used = count > 0
    corners[used, 3:] = total[used] / count[used, None]



def polish_uv_orientation(original, faces, scale, sign, max_displacement=2.):
    """Minimum-displacement correction of resampling inversions on a fixed mesh."""
    from scipy.optimize import minimize
    minimum=scale*.01
    def areas(placement):
        p=placement[faces];a=p[:,1]-p[:,0];b=p[:,2]-p[:,0]
        return sign*(a[:,0]*b[:,1]-a[:,1]*b[:,0])
    active=np.unique(faces[areas(original)<minimum*1.01]);displaced=original.copy()
    for _ in range(12):
        local=np.full(len(original),-1,dtype=int);local[active]=np.arange(len(active))
        affected=np.any(local[faces]>=0,axis=1);triples=faces[affected];indices=local[triples]
        denom=scale[affected];limit=minimum[affected]*1.01
        def placement(vector):
            points=original.copy();points[active]+=vector.reshape(-1,2);return points
        def constraint(vector):
            p=placement(vector)[triples];a=p[:,1]-p[:,0];b=p[:,2]-p[:,0]
            return (sign*(a[:,0]*b[:,1]-a[:,1]*b[:,0])-limit)/denom
        def jacobian(vector):
            p=placement(vector)[triples];jac=np.zeros((len(triples),2*len(active)))
            for corner in range(3):
                j,k=(corner+1)%3,(corner+2)%3
                gradient=np.column_stack((p[:,j,1]-p[:,k,1],p[:,k,0]-p[:,j,0]))*sign/denom[:,None]
                rows=np.flatnonzero(indices[:,corner]>=0);cols=indices[rows,corner]
                jac[rows,2*cols]=gradient[rows,0];jac[rows,2*cols+1]=gradient[rows,1]
            return jac
        fit=minimize(lambda x:.5*np.dot(x,x),(displaced[active]-original[active]).ravel(),jac=lambda x:x,
                     method='SLSQP',constraints=[dict(type='ineq',fun=constraint,jac=jacobian)],
                     options=dict(maxiter=100,ftol=1e-12))
        displaced=placement(fit.x)
        if not np.isfinite(displaced).all():
            raise ValueError('Nonfinite resampling correction')
        bad=np.flatnonzero(areas(displaced)<minimum)
        if not len(bad):break
        active=np.union1d(active,np.unique(faces[bad]))
    else:raise ValueError('Cannot refine resampling inversions')
    delta=displaced-original;displacement=float(np.max(np.linalg.norm(delta,axis=1)))
    if displacement>max_displacement:
        raise ValueError(f'Resampling correction requires {displacement:.3g} voxels; refine controls')
    return delta,displacement


def validate_corner_uv(grid, mask, max_displacement=None):
    """Check every restored face and correct small resampling inversions."""
    used=grid[...,0]!=-1
    index=np.full(used.shape,-1,dtype=int);index[used]=np.arange(used.sum())
    cells=np.argwhere(mask);faces=[];scales=[]
    for offsets in (((0,0),(0,1),(1,0)),((1,1),(1,0),(0,1))):
        ij=cells[:,None]+np.asarray(offsets)[None]
        xyz=grid[ij[:,:,0],ij[:,:,1],:3]
        area=np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)
        good=area>1e-8
        faces.append(index[ij[:,:,0],ij[:,:,1]][good]);scales.append(area[good])
    faces=np.concatenate(faces);scale=np.concatenate(scales)
    if not len(faces):
        return dict(triangles=0,corrected_vertices=0,max_displacement=0.)
    original=grid[used,3:];uv=original[faces];a,b=uv[:,1]-uv[:,0],uv[:,2]-uv[:,0]
    area=a[:,0]*b[:,1]-a[:,1]*b[:,0];sign=np.sign(np.median(area))
    if not np.isfinite(original).all() or sign==0:
        raise ValueError('Restored UV has nonfinite or collapsed faces')
    if np.any(sign*area<=1e-8):
        if max_displacement is None:
            ij=cells[:,None]+np.array(((0,0),(0,1),(1,0),(1,1)))[None]
            xyz=grid[ij[:,:,0],ij[:,:,1],:3]
            # These are newly restored corners, half a source cell from the
            # old centers. Bound correction by that reconstruction distance.
            max_displacement=float(np.median(np.max(np.linalg.norm(xyz-xyz.mean(axis=1)[:,None],axis=2),axis=1)))
        delta,displacement=polish_uv_orientation(original,faces,scale,sign,max_displacement)
        grid[used,3:]=original+delta
        return dict(triangles=len(faces),corrected_vertices=int(np.count_nonzero(np.linalg.norm(delta,axis=1)>1e-8)),
                    max_displacement=displacement)
    return dict(triangles=len(faces),corrected_vertices=0,max_displacement=0.)


def restore_workspace(path, output, progress=print):
    """Upgrade a center-grid artifact atomically, without rerunning growth."""
    from scipy.ndimage import label
    from tifxyz import load_tifxyz
    from winding_arclength import source_grid
    path, output = Path(path), Path(output)
    document = json.loads(path.read_text())
    if document.get('type') != 'vc3d_winding_layout' or document.get('version') != 1:
        raise ValueError('Expected a legacy version-1 winding workspace')
    name = Path(document['geometry_file'])
    if name.is_absolute() or name.name != str(name):
        raise ValueError('Workspace geometry must be a sibling file')
    if (path.parent / name).stat().st_size != document['geometry_bytes']:
        raise ValueError('Workspace geometry size mismatch')
    payload = np.memmap(path.parent / name, dtype='<f8', mode='r').reshape(-1, 5)
    binary = output.with_name(output.stem + '.' + uuid.uuid4().hex + '.bin')
    temporary = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
    result = copy.deepcopy(document)
    low, high = np.full(2, np.inf), np.full(2, -np.inf)
    report = dict(patch_sources=0, accepted_quads=0, old_rendered_quads=0,
                  old_face_components=0, restored_face_components=0, exact_source_vertices=True,
                  rendered_triangles=0, corrected_sources=[],
                  correction_limit='median original quad center-to-corner radius')
    previous_path, original = None, None
    try:
        with binary.open('wb') as stream:
            for number, (source, target) in enumerate(zip(document['sources'], result['sources'])):
                old = source_grid(payload, source)
                if source['fiber'] or source.get('quarantined', False):
                    grid = old
                else:
                    if source['path'] != previous_path:
                        original = load_tifxyz(source['path']).zyxs.numpy()[..., ::-1]
                        previous_path = source['path']
                    mask = old[..., 0] != -1
                    origin = np.asarray(source['source_ij_origin']) - .5
                    if not np.array_equal(origin, np.round(origin)):
                        raise ValueError('Expected quad-center source coordinates')
                    r, c = np.where(mask)
                    grid, restored, ij_origin = patch_corners(original, r + int(origin[0]), c + int(origin[1]))
                    if not np.array_equal(mask, restored):
                        raise ValueError('Legacy source has empty outer rows or columns')
                    # Detect a changed or mismatched source before publishing.
                    expected = sum(grid[r + dr, c + dc, :3] for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1))) / 4
                    if not np.allclose(expected, old[r, c, :3], atol=.002, rtol=0):
                        raise ValueError(f'Original geometry changed: {source["id"]}')
                    lift_center_uv(old, grid, mask)
                    try:
                        validation=validate_corner_uv(grid,mask)
                    except ValueError as error:
                        raise ValueError(f'{source["id"]}: {error}') from error
                    report['rendered_triangles']+=validation['triangles']
                    if validation['corrected_vertices']:
                        report['corrected_sources'].append(dict(source=source['id'],**validation))
                    target.update(source_ij_origin=list(ij_origin), quad_runs=quad_runs(mask))
                    before = source_quad_mask(old)
                    report['patch_sources'] += 1
                    report['accepted_quads'] += int(mask.sum())
                    report['old_rendered_quads'] += int(before.sum())
                    report['old_face_components'] += label(before)[1]
                    report['restored_face_components'] += label(mask)[1]
                target.update(rows=grid.shape[0], cols=grid.shape[1], byte_offset=stream.tell())
                stream.write(np.asarray(grid, dtype='<f8').tobytes())
                valid = grid[..., 0] != -1
                low = np.minimum(low, grid[valid, 3:].min(axis=0))
                high = np.maximum(high, grid[valid, 3:].max(axis=0))
                if number % 2000 == 0:
                    progress(f'Restored {number:,}/{len(document["sources"]):,} sources')
        result.update(version=2, geometry_file=binary.name, geometry_bytes=binary.stat().st_size)
        result['layout'].update(patch_samples='original quad vertices with explicit accepted faces',
                                bounds=[low.tolist(), high.tolist()], topology_restoration=report)
        report['rendered_foldovers']=0
        if 'settings' in result:
            spacing=result['settings']['spacing']
            while np.prod(np.ceil((high-low)/spacing)+4)>16_000_000:
                spacing*=2
            result['settings']['spacing']=spacing
            result['layout']['raster_spacing']=spacing
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        os.replace(temporary, output)
    except BaseException:
        binary.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        raise
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(restore_workspace(args.workspace, args.output, progress=lambda s: print(s, flush=True)))
