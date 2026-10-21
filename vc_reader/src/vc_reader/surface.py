"""Registered coordinate grids and bounded surface geometry operations.

NumPy, image decoders and compiled kernels load only when opening geometry.
"""
from dataclasses import replace
import math
import operator
import time
from .affine import Affine
from .sources import Source, Storage, Surface, SurfaceRegistration, SampleResult, TargetResult
from .reader.source_resolution import pixel_affine


def _matrix(affine):
    return tuple(v for row in affine.matrix for v in row)


def _scale(values):
    return Affine.from_rows(tuple(tuple(v if i == j else 0 for j in range(3)) + (0,)
                                  for i, v in enumerate(values)))


def _shift(xyz):
    return Affine.from_rows(tuple((*row[:3], float(v)) for row, v in zip(Affine.identity().matrix, xyz)))


def _transform(points, mapping, origin=(0, 0, 0)):
    """Split the small transformed origin before doing bulk float32 arithmetic."""
    import numpy as np
    at = mapping.apply(origin)
    base = tuple(math.floor(v) for v in at)
    local = np.asarray(points, dtype=np.float32) @ np.asarray(mapping.linear, dtype=np.float32).T
    local += np.asarray(tuple(v-b for v, b in zip(at, base)), dtype=np.float32)
    return np.ascontiguousarray(local), base


def _pixel_points(uv, mapping, origin=(0,0,0)):
    import numpy as np
    xy, base = _transform(np.column_stack((uv, np.zeros(len(uv), np.float32))), mapping, origin)
    return np.ascontiguousarray(xy[:, ::-1]), tuple(reversed(base))


def _region(start, size):
    start, size = tuple(map(operator.index, start)), tuple(map(operator.index, size))
    if len(start) != 2 or len(size) != 2 or any(n <= 0 or n > 2**24 for n in size):
        raise ValueError('surface start/size must be YX pairs with positive sizes <= 2**24')
    if any(abs(n) > 2**40 for n in start):
        raise ValueError('surface start is outside supported coordinate range')
    return start, size


def _distance(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def _surface(reader, surface):
    if isinstance(surface, Surface):
        return surface
    source = getattr(surface, 'source', surface)
    registration = getattr(source, 'registration', None)
    if isinstance(registration, SurfaceRegistration):
        return registration.surface
    identifier = getattr(surface, 'id', surface)
    candidates = [t['reader'].registration.surface for t in reader._surface_targets.values()
                  if t['reader'].registration.surface.id == identifier]
    if not candidates or any(s != candidates[0] for s in candidates):
        raise ValueError('select an explicit registered Surface; geometry relationship is missing or ambiguous')
    return candidates[0]


class Geometry:
    """Coordinate-grid metadata and bounded tiles using shared Source readers."""
    def __init__(self, owner, surface):
        import json
        import numpy as np
        from .reader.encoded_files import EncodedFiles
        from .errors import CacheMissError
        if surface.xyz_to_volume is None or surface.component_order is None:
            raise ValueError(f'{surface.id}: unknown geometry coordinate frame/components')
        self.owner, self.surface = owner, surface
        self.coordinate_readers = []
        self.mask = None
        grid = surface.pixel_to_uv
        if surface.geometry.format == 'tifxyz':
            root = surface.geometry.location
            if root is None:
                raise ValueError('surface geometry location is unresolved')
            files = EncodedFiles(**{k: v for k, v in owner._options.items()
                                   if k in {'auth', 'cache_dir', 'cache_writes', 'cache_max_bytes', 'offline'}})
            try:
                for candidate in (root,*surface.geometry.alternatives):
                    try:
                        with files.open(candidate.rstrip('/')+'/meta.json') as (handle, _):
                            meta = json.load(handle)
                        root=candidate
                        break
                    except (FileNotFoundError,CacheMissError):
                        if candidate == (surface.geometry.alternatives[-1] if surface.geometry.alternatives else root):
                            raise
            finally:
                files.close()
            scale = meta.get('scale')
            if len(scale or ()) != 2 or any(not math.isfinite(v) or v <= 0 for v in scale):
                raise ValueError('tifxyz scale must contain two positive finite UV scales')
            observed = ((1/scale[0], 0, 0), (0, 1/scale[1], 0))
            if grid is not None and grid != observed:
                raise ValueError('surface pixel_to_uv contradicts tifxyz scale')
            grid = observed
            for axis in 'xyz':
                source = Source(surface.id+'/'+axis, Storage(root.rstrip('/')+'/'+axis+'.tif', 'tiff'),
                                role='scalar', axes=('y', 'x'))
                self.coordinate_readers.append(owner._open(source))
            if surface.tifxyz_mask:
                try:
                    self.mask = owner._open(Source(surface.id+'/geometry-mask', Storage(root.rstrip('/')+'/mask.tif', 'tiff'),
                                                   role='label', axes=('y', 'x')))
                except FileNotFoundError:
                    pass
        else:
            if grid is None:
                raise ValueError('coordinate grid requires explicit pixel_to_uv')
            source = Source(surface.id+'/coordinates', surface.geometry, role='scalar', axes=surface.axes)
            self.coordinate_readers.append(owner._open(source))
        first = self.coordinate_readers[0]
        self.shape = tuple(first.shape[first.axes.index(a)] for a in ('y', 'x'))
        for r in self.coordinate_readers:
            if r.spatial_axes != ('y', 'x') or tuple(r.shape[r.axes.index(a)] for a in ('y', 'x')) != self.shape:
                raise ValueError('surface coordinates require matching YX grids')
        if len(self.coordinate_readers) == 1 and ('c' not in first.axes or first.shape[first.axes.index('c')] != 3):
            raise ValueError('coordinate grid must contain three components')
        if self.mask is not None and self.mask.shape != self.shape:
            raise ValueError('tifxyz mask shape does not match coordinate grid')
        self.to_uv = pixel_affine(grid)
        self.from_uv = self.to_uv.inverse()
        self.orientation = 1 if grid[0][0]*grid[1][1]-grid[0][1]*grid[1][0] > 0 else -1
        self.explicit_mask = self._auxiliary(surface.mask)
        self.normals = self._auxiliary(surface.normals)
        self.normal_transform = None
        if self.normals is not None:
            value = surface.normals.value_frame
            if surface.normals.role != 'normal' or value is None or value.kind != 'volume' or surface.normals.component_order is None:
                raise ValueError('surface normals require a declared volume value frame and XYZ components')
            mapping = owner._value_placement(value)
            if surface.spacing_um is None:
                raise ValueError('normal values require surface physical spacing')
            if (value.sample_id, value.volume_id) != (surface.sample_id, surface.volume_id):
                if not hasattr(owner, '_between'):
                    raise ValueError('cross-volume normals need a volume-aligned reader with a known frame relationship')
                source_transform = owner._between(surface.normals,(surface.sample_id,surface.volume_id))
                values = owner._between(surface.normals,(value.sample_id,value.volume_id))
                mapping = source_transform.inverse() @ values @ mapping
            inv = (_scale(surface.spacing_um) @ surface.xyz_to_volume.inverse() @ mapping).inverse()
            self.normal_transform = np.asarray(inv.linear, np.float32).T

    def _auxiliary(self, source):
        if source is None:
            return None
        r = self.owner._open(source)
        reg = r.registration
        if not isinstance(reg, SurfaceRegistration) or reg.surface.id != self.surface.id or reg.pixel_to_uv is None:
            raise ValueError('geometry masks/normals require explicit registration to this surface')
        target = self.owner._sampling_target(source, r, pixel_affine(reg.pixel_to_uv).inverse(), plane=True, planar_grid=True)
        if source.role == 'normal':
            target['order'] = 1
        return target

    def tile(self, start, size):
        import numpy as np
        key = ('geometry', self.surface, start, size)
        cached = self.owner._tiles.get(key)
        if cached is not None:
            return cached
        if math.prod(size)*80 > self.owner._work_bytes:
            raise MemoryError('geometry tile exceeds sampling_work_bytes')
        arrays = [self.owner._canonical(r.read(start, size), r)[..., 0, :, :] for r in self.coordinate_readers]
        self.owner._surface_metrics['geometry_read_bytes'] += sum(a.nbytes for a in arrays)
        xyz = np.moveaxis(np.concatenate(arrays, axis=0), 0, -1)
        if len(self.coordinate_readers) == 1:
            xyz = xyz[..., [self.surface.component_order.index(a) for a in 'xyz']]
        xyz = np.ascontiguousarray(xyz, np.float32)
        valid = np.isfinite(xyz).all(axis=-1)
        if self.surface.geometry.format == 'tifxyz':
            valid &= (xyz >= 0).all(axis=-1)
        if self.mask is not None:
            valid &= self.mask.read(start, size) != 0
        normals = np.empty((0,), np.float32)
        if self.explicit_mask is not None or self.normals is not None:
            y, x = np.indices(size, dtype=np.float32)
            coords = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size, np.float32)))
            uv, base = _transform(coords, self.to_uv, (start[1], start[0], 0))
            for target in (self.explicit_mask, self.normals):
                if target is None:
                    continue
                local, origin = _transform(uv, target['mapping'], base)
                values, support = self.owner._sample_points(target, np.ascontiguousarray(local[:, ::-1]), tuple(reversed(origin)))
                if self.owner.backend == 'cuda':
                    values, support = values.numpy(), support.numpy()
                valid &= support.reshape(size)
                if target is self.explicit_mask:
                    if values.shape[0] != 1:
                        raise ValueError('geometry mask must have one channel')
                    valid &= values.reshape(size) != 0
                else:
                    order = [self.surface.normals.component_order.index(a) for a in 'xyz']
                    normals = np.ascontiguousarray(values[order].T @ self.normal_transform.T).reshape(*size, 3)
        result = xyz, valid, normals
        self.owner._tiles.put(key, result, xyz.nbytes+valid.nbytes+normals.nbytes)
        return result

    def sample(self, uv, offsets, destination, uv_origin=(0,0,0)):
        """Interpolate a bounded UV block; outputs are XYZ in destination grid."""
        import numpy as np
        from . import _reader
        coords, base = _transform(np.column_stack((uv, np.zeros(len(uv), np.float32))), self.from_uv, uv_origin)
        coords[:, :2] += np.asarray(base[:2], np.float32)
        finite = np.isfinite(coords[:, :2]).all(axis=1)
        inside = finite & (coords[:, 0]>=0) & (coords[:, 1]>=0) & (coords[:, 0]<=self.shape[1]-1) & (coords[:, 1]<=self.shape[0]-1)
        output = np.full((len(offsets), len(uv), 3), np.nan, np.float32)
        output_base = None
        support = np.zeros((len(offsets), len(uv)), bool)
        if not inside.any():
            return output, support, (0, 0, 0)
        # Group queries by stored coordinate tile; halos supply complete quads.
        keys = np.floor(coords[inside, :2]/32).astype(np.int64)
        indices = np.flatnonzero(inside)
        packed = keys[:, 1]*((self.shape[1]+31)//32)+keys[:, 0]
        order = np.argsort(packed, kind='stable')
        groups = np.split(order, np.flatnonzero(np.diff(packed[order]))+1)
        spacing = self.surface.spacing_um
        if spacing is None:
            raise ValueError('surface reads require physical coordinate spacing')
        for group in groups:
            at = indices[group]
            x, y = (int(v)*32 for v in keys[group[0]])
            x, y = min(x, max(0,self.shape[1]-2)), min(y,max(0,self.shape[0]-2))
            size = (min(33,self.shape[0]-y),min(33,self.shape[1]-x))
            xyz, valid, normals = self.tile((y,x),size)
            if not valid.any():
                continue
            anchor = tuple(math.floor(float(v)) for v in xyz[valid][0])
            physical_origin = tuple(v*s for v,s in zip(anchor,spacing))
            tile_destination = destination @ _shift(physical_origin)
            if output_base is None:
                output_base = tuple(math.floor(v) for v in tile_destination.translation)
            tile_destination = _shift(tuple(-v for v in output_base)) @ tile_destination
            xyz = np.ascontiguousarray(xyz-np.asarray(anchor,np.float32))
            local_queries = np.ascontiguousarray(coords[at,:2]-np.array([x,y],np.float32))
            for depth, offset in enumerate(offsets):
                values = np.empty((len(at),3),np.float32); good = np.empty(len(at),bool)
                if self.owner.backend == 'cuda':
                    self.owner._surface_metrics['geometry_upload_bytes'] += xyz.nbytes+valid.nbytes+local_queries.nbytes+normals.nbytes
                    self.owner._surface_metrics['geometry_download_bytes'] += values.nbytes+good.nbytes
                _reader._surface_interpolate(xyz,valid.view(np.uint8).ravel(),local_queries,normals,values,good.view(np.uint8),
                                             _matrix(_scale(spacing)),_matrix(tile_destination),self.orientation,offset,
                                             self.owner.device if self.owner.backend=='cuda' else -1)
                output[depth,at],support[depth,at] = values,good
        return output,support,output_base or (0,0,0)


def geometry(reader, surface):
    surface = _surface(reader, surface)
    key = ('surface-reader', surface)
    # The metadata objects contain only opened-reader references, never tile data.
    if key not in reader._geometries:
        reader._geometries[key] = Geometry(reader, surface)
    return reader._geometries[key]


def prepare_target(reader, source, opened):
    reg = opened.registration
    if opened.spatial_axes != ('y', 'x') or reg.pixel_to_uv is None:
        raise ValueError(f'{source.id}: surface targets require YX axes and explicit pixel_to_uv registration')
    if source.role != 'label':
        raise ValueError('surface targets must be categorical labels')
    target = reader._sampling_target(source, opened, pixel_affine(reg.pixel_to_uv).inverse(), plane=True, planar_grid=True)
    geometry(reader, reg.surface)  # Fail unknown frames while opening aligned data.
    reader._between(source,(reg.surface.sample_id,reg.surface.volume_id))
    target['companions'] = {}
    for companion in source.companions:
        child = opened.companions[companion.name]
        child_reg = child.registration
        if not isinstance(child_reg, SurfaceRegistration) or child_reg.surface.id != reg.surface.id or child_reg.pixel_to_uv is None:
            raise ValueError('surface companions require an explicit shared surface registration')
        target['companions'][companion.name] = reader._sampling_target(
            companion.source, child, pixel_affine(child_reg.pixel_to_uv).inverse(), plane=True, planar_grid=True)
    return target


def _matching(reader, surface):
    targets = reader._surface_targets
    if any(t['reader'].registration.surface != surface for t in targets.values()):
        raise ValueError('all selected surface targets must register to the requested surface descriptor')
    return targets


def _check_budget(reader, shape, surface_shape=None):
    import numpy as np
    size = math.prod(shape)
    estimate = 0
    for target in reader._targets.values():
        estimate += size*(target['channels']*max(4,target['dtype'].itemsize)+1)
        estimate += sum(size*(t['channels']*max(4,t['dtype'].itemsize)) for t in target['companions'].values())
    for target in reader._surface_targets.values():
        count = math.prod(surface_shape) if surface_shape is not None else size
        estimate += count*(target['channels']*target['dtype'].itemsize+3)
        estimate += sum(count*t['channels']*max(4,t['dtype'].itemsize) for t in target['companions'].values())
    if estimate > reader._surface_output_bytes:
        raise MemoryError(f'derived output shape {shape} requires approximately {estimate} bytes; '
                          'increase surface_output_bytes or use iter_project_surface')


def _step(step):
    step = (step, step) if isinstance(step, int) else tuple(step)
    if len(step) != 2 or any(type(v) is not int or v <= 0 for v in step):
        raise ValueError('step must be a positive integer or a YX pair of positive integers')
    return step


def render_surface(reader, surface, start, size, offsets, step=1):
    import numpy as np
    start,size = _region(start,size)
    step = _step(step)
    offsets = tuple(float(v) for v in offsets)
    if not offsets or any(not math.isfinite(v) for v in offsets):
        raise ValueError('offsets_um must be nonempty and finite')
    geom = geometry(reader,surface);surface=geom.surface
    targets = _matching(reader,surface)
    _check_budget(reader,(len(offsets),*size),size)
    if surface.spacing_um is None:
        raise ValueError('surface physical spacing is unknown')
    between = reader._between(next(iter(targets.values()))['source'] if targets else reader.reference,
                              (surface.sample_id,surface.volume_id))
    destination = reader._reference_map.inverse() @ between @ surface.xyz_to_volume @ _scale(surface.spacing_um).inverse()
    result = SampleResult({}, dict(reader.metadata or {}))
    result.metadata['surface'] = dict(id=surface.id,uv_start=start,uv_size=size,uv_step=step,offsets_um=offsets,
        coordinate_frame=(surface.sample_id,surface.volume_id),reference_frame=reader._reference_frame,
        array_to_level0=reader._reference_map.matrix)
    n = math.prod(size)
    # Output rows are contiguous UV strips, keeping scratch independent of H/W.
    for name,target in {**reader._targets,**targets}.items():
        depth = len(offsets) if name in reader._targets else 1
        dtype = np.dtype('float32') if target['order'] else target['native_endian_dtype']
        if reader.backend == 'cpu' and not target['order']:
            dtype = target['dtype']
        result.targets[name] = TargetResult(
            reader._owned_empty((target['channels'],depth*n),dtype),
            reader._owned_empty((1,depth*n),np.dtype('bool')))
        result[name].masks = {k:reader._owned_empty((t['channels'],depth*n),np.dtype('float32') if t['order'] else
                              t['dtype'] if reader.backend == 'cpu' else t['native_endian_dtype'])
                              for k,t in target['companions'].items()}
    for first in range(0,n,8192):
        index=np.arange(first,min(first+8192,n))
        # Output pixel (row, col) samples UV (start + row*step_y, start + col*step_x).
        uv=np.column_stack((index%size[1]*step[1],index//size[1]*step[0])).astype(np.float32)
        uv_origin=(start[1],start[0],0)
        points,support,point_base=geom.sample(uv,offsets,destination,uv_origin)
        for name,target in {**reader._targets,**targets}.items():
            volume=name in reader._targets
            for d in range(len(offsets) if volume else 1):
                if volume:
                    local,base=_transform(points[d],target['mapping'],point_base)
                    local=np.ascontiguousarray(local[:,::-1]);base=tuple(reversed(base))
                else:
                    local,base=_pixel_points(uv,target['mapping'],uv_origin)
                values,valid=reader._sample_points(target,local,base)
                reader._copy_columns(values,result[name].data,d*n+first)
                reader._copy_columns(valid,result[name].valid,d*n+first)
                for key,child in target['companions'].items():
                    if volume:
                        p,b=_transform(points[d],child['mapping'],point_base);p=np.ascontiguousarray(p[:,::-1]);b=tuple(reversed(b))
                    else:
                        p,b=_pixel_points(uv,child['mapping'],uv_origin)
                    values,_=reader._sample_points(child,p,b)
                    reader._copy_columns(values,result[name].masks[key],d*n+first)
    for name,target in {**reader._targets,**targets}.items():
        shape=(len(offsets),*size) if name in reader._targets else size
        result[name].data=result[name].data.reshape((target['channels'],*shape))
        result[name].valid=result[name].valid.reshape((1,*shape))
        result[name].masks={k:v.reshape((target['companions'][k]['channels'],*shape)) for k,v in result[name].masks.items()}
    return result


def _clip(triangle, lower, upper):
    """Clip XYZ/pixel attributes together to a convex label-pixel rectangle."""
    import numpy as np
    polygon = list(triangle)
    for axis in (3,4):
        for limit,sign in ((lower[axis-3],1),(upper[axis-3],-1)):
            if not polygon:
                return []
            result=[]
            prev=polygon[-1];pdist=(prev[axis]-limit)*sign
            for point in polygon:
                dist=(point[axis]-limit)*sign
                if (dist>=0)!=(pdist>=0):
                    t=pdist/(pdist-dist)
                    intersection=prev+(point-prev)*t
                    intersection[axis]=limit
                    result.append(intersection)
                if dist>=0:
                    result.append(point)
                prev,pdist=point,dist
            polygon=result
    return [np.stack((polygon[0],polygon[i],polygon[i+1])) for i in range(1,len(polygon)-1)]


class _SurfaceIndex:
    """Bounded BVH of tile metadata; decoded geometry stays in the tile cache."""
    def __init__(self, entries):
        self.bounds=(tuple(min(b[0][a] for _,b in entries) for a in range(3)),
                     tuple(max(b[1][a] for _,b in entries) for a in range(3)))
        self.children=()
        self.entries=()
        if len(entries)<=8:
            self.entries=tuple(entries)
        else:
            axis=max(range(3),key=lambda a:self.bounds[1][a]-self.bounds[0][a])
            entries=sorted(entries,key=lambda item:item[1][0][axis]+item[1][1][axis])
            mid=len(entries)//2
            self.children=(_SurfaceIndex(entries[:mid]),_SurfaceIndex(entries[mid:]))

    def query(self,lower,upper,expansion):
        def overlaps(bounds):
            return all(b+e+.5>=lo and a-e-.5<=hi for a,b,e,lo,hi in zip(bounds[0],bounds[1],expansion,lower,upper))
        if not overlaps(self.bounds):
            return
        for tile,bounds in self.entries:
            if overlaps(bounds):
                yield tile
        for child in self.children:
            yield from child.query(lower,upper,expansion)


class Projection:
    def __init__(self, reader, surface, start, size, label_target, thickness, padding):
        import numpy as np
        self.reader=reader
        self.geom=geometry(reader,surface);self.surface=self.geom.surface
        self.targets=_matching(reader,self.surface)
        if label_target not in self.targets:
            raise ValueError('label_target must name a configured surface label')
        self.label_target=label_target;self.selecting=self.targets[label_target]
        self.start,self.size=_region(start,size)
        if any(s<0 or s+n>extent for s,n,extent in zip(self.start,self.size,self.selecting['shape'][1:])):
            raise ValueError('selected label patch must lie inside its opened pixel grid')
        self.thickness=_distance(thickness,'thickness_um');self.padding=_distance(padding,'padding_um')
        spacing=reader._reference_reader.registration.volume_spacing_um
        if spacing is None:
            raise ValueError('projection requires known reference volume physical spacing')
        self.metric=_scale(spacing) @ reader._reference_map
        # Only the small affine uses Python double; bulk vertices stay float32.
        inv=self.metric.inverse().linear
        self.radius=self.thickness/2
        self.expansion=tuple(self.radius*math.sqrt(sum(v*v for v in row)) for row in inv)
        self.margin=tuple(math.ceil(self.padding*math.sqrt(sum(v*v for v in row))) for row in inv)
        between=reader._between(self.selecting['source'],(self.surface.sample_id,self.surface.volume_id))
        self.destination=reader._reference_map.inverse() @ between @ self.surface.xyz_to_volume
        self.label_to_uv=pixel_affine(self.selecting['reader'].registration.pixel_to_uv)
        self.uv_origin=self.label_to_uv.apply((self.start[1],self.start[0],0))
        self.grid_to_label=self.label_to_uv.inverse() @ self.geom.to_uv
        corners=[(self.start[1]-.5,self.start[0]-.5,0),
                 (self.start[1]+self.size[1]-.5,self.start[0]-.5,0),
                 (self.start[1]+self.size[1]-.5,self.start[0]+self.size[0]-.5,0),
                 (self.start[1]-.5,self.start[0]+self.size[0]-.5,0)]
        grid_corners=[(self.geom.from_uv @ self.label_to_uv).apply(p) for p in corners]
        self.grid_start=tuple(max(0,math.floor(min(p[a] for p in grid_corners))) for a in (1,0))
        self.grid_end=tuple(min(extent-1,math.ceil(max(p[a] for p in grid_corners))) for a,extent in zip((1,0),self.geom.shape))
        self.key=('projection',self.surface,self.start,self.size,self.label_to_uv.matrix,self.destination.matrix)
        before=time.perf_counter()
        self.index=reader._plans.get(('surface-index',self.key))
        low=[None]*3;high=[None]*3
        if self.index is not None:
            low,high=self.index.bounds
        else:
            count=math.prod(max(0,(b-a+31)//32) for a,b in zip(self.grid_start,self.grid_end))
            # Python metadata/node overhead is deliberately budgeted conservatively.
            entries=[] if count*2048<=reader._plans.limit else None
            for tile in self.tiles():
                bounds=reader._plans.get(('surface-bounds',(*self.key,tile,True)))
                if bounds is None:
                    _,_,bounds=self.triangles(tile,clip=True)
                if bounds is None:
                    continue
                if entries is not None:
                    entries.append((tile,bounds))
                for a in range(3):
                    low[a]=bounds[0][a] if low[a] is None else min(low[a],bounds[0][a])
                    high[a]=bounds[1][a] if high[a] is None else max(high[a],bounds[1][a])
            if entries:
                self.index=_SurfaceIndex(entries)
                reader._plans.put(('surface-index',self.key),self.index,count*2048)
        if any(v is None for v in low):
            raise ValueError('selected label patch has no valid nondegenerate surface geometry')
        origin=tuple(math.ceil(v-.5-e)-m for v,e,m in zip(low,self.expansion,self.margin))
        end=tuple(math.floor(v+.5+e)+m+1 for v,e,m in zip(high,self.expansion,self.margin))
        self.origin=tuple(reversed(origin));self.shape=tuple(b-a for a,b in zip(reversed(origin),reversed(end)))
        if any(n>2**24 for n in self.shape):
            raise ValueError('derived crop extent exceeds supported coordinate range')
        # The label rectangle selects the crop, not the extent of its sheet.
        self.index=None
        self.preparation_seconds=time.perf_counter()-before
        reader._surface_metrics['preparation_seconds'] += self.preparation_seconds

    def tiles(self):
        for y in range(self.grid_start[0],self.grid_end[0],32):
            for x in range(self.grid_start[1],self.grid_end[1],32):
                yield (y,x),(min(32,self.grid_end[0]-y)+1,min(32,self.grid_end[1]-x)+1)

    def spatial_index(self):
        """Index the whole registered sheet once, including folds returning to a crop."""
        import numpy as np
        # Bulk coordinate reads amortize Python/decoder overhead. Index leaves are
        # smaller; only intersecting leaves are triangulated and read as labels.
        block=256
        while (block+1)**2*80>self.reader._work_bytes//2 and block>1:
            block//=2
        step=min(64,block)
        while math.prod((n-1+step-1)//step for n in self.geom.shape)*2048>self.reader._work_bytes//2 and step<block:
            step*=2
        count=math.prod((n-1+step-1)//step for n in self.geom.shape)
        if count*2048>self.reader._work_bytes//2:
            raise MemoryError('surface spatial index exceeds sampling_work_bytes')
        key=('surface-spatial-index',self.surface,self.destination.matrix,step)
        cached=self.reader._plans.get(key)
        if cached is not None:
            return cached
        entries=[]
        for y in range(0,self.geom.shape[0]-1,block):
            for x in range(0,self.geom.shape[1]-1,block):
                size=(min(block,self.geom.shape[0]-1-y)+1,min(block,self.geom.shape[1]-1-x)+1)
                xyz,valid,_=self.geom.tile((y,x),size)
                if not valid.any():
                    continue
                anchor=tuple(math.floor(float(v)) for v in xyz[valid][0])
                points,base=_transform(xyz-np.asarray(anchor,np.float32),self.destination,anchor)
                for iy in range(0,size[0]-1,step):
                    for ix in range(0,size[1]-1,step):
                        leaf=(min(step,size[0]-1-iy)+1,min(step,size[1]-1-ix)+1)
                        mask=valid[iy:iy+leaf[0],ix:ix+leaf[1]]
                        if not mask.any():
                            continue
                        vertices=points[iy:iy+leaf[0],ix:ix+leaf[1]][mask]
                        lo,hi=vertices.min(axis=0),vertices.max(axis=0)
                        bounds=(tuple(float(v)+b for v,b in zip(lo,base)),
                                tuple(float(v)+b for v,b in zip(hi,base)))
                        entries.append((((y+iy,x+ix),leaf),bounds))
        index=_SurfaceIndex(entries)
        self.reader._plans.put(key,index,count*2048)
        self.reader._peak_work=max(self.reader._peak_work,count*2048+(block+1)**2*80)
        return index

    def triangles(self,tile,*,clip=False):
        import numpy as np
        key=(*self.key,tile,clip)
        cached=self.reader._tiles.get(key)
        if cached is not None:
            return cached
        start,size=tile
        xyz,valid,_=self.geom.tile(start,size)
        good=valid[:-1,:-1]&valid[1:,:-1]&valid[:-1,1:]&valid[1:,1:]
        iy,ix=np.nonzero(good)
        if not len(ix):
            return np.empty((0,3,5),np.float32),np.empty(0,np.int64),None
        p0=xyz[iy,ix];p1=xyz[iy,ix+1];p2=xyz[iy+1,ix+1];p3=xyz[iy+1,ix]
        points=np.stack((np.stack((p0,p1,p2),axis=1),np.stack((p0,p2,p3),axis=1)),axis=1).reshape(-1,3,3)
        anchor=tuple(math.floor(float(v)) for v in points[0,0])
        local,base=_transform(points-np.asarray(anchor,np.float32),self.destination,anchor)
        grid=np.stack((np.stack((ix,iy),axis=-1),np.stack((ix+1,iy),axis=-1),np.stack((ix+1,iy+1),axis=-1),np.stack((ix,iy+1),axis=-1)),axis=1).astype(np.float32)
        grid=np.stack((grid[:,[0,1,2]],grid[:,[0,2,3]]),axis=1).reshape(-1,3,2)
        pix,pixbase=_transform(np.concatenate((grid,np.zeros((*grid.shape[:-1],1),np.float32)),axis=-1),self.grid_to_label,(start[1],start[0],0))
        # Clip in patch-local label pixels to preserve boundary ties at large origins.
        pix[...,:2]+=np.asarray((pixbase[0]-self.start[1],pixbase[1]-self.start[0]),np.float32)
        triangles=np.concatenate((local,pix[...,:2]),axis=-1)
        ids=((iy.astype(np.int64)+start[0])*(self.geom.shape[1]-1)+ix+start[1])*2
        ids=np.stack((ids,ids+1),axis=1).ravel()
        if clip:
            lower=(-.5,-.5);upper=(self.size[1]-.5,self.size[0]-.5)
            inside=(triangles[...,3:]>=np.array(lower,np.float32)).all(axis=(1,2))&(triangles[...,3:]<=np.array(upper,np.float32)).all(axis=(1,2))
            clipped=[triangles[inside]];identifiers=[ids[inside]]
            for t,identifier in zip(triangles[~inside],ids[~inside]):
                pieces=_clip(t,lower,upper)
                if pieces:
                    clipped.append(np.asarray(pieces,np.float32));identifiers.append(np.full(len(pieces),identifier,np.int64))
            triangles=np.concatenate(clipped);ids=np.concatenate(identifiers)
        if len(triangles):
            a=triangles[:,1,:3]-triangles[:,0,:3];b=triangles[:,2,:3]-triangles[:,0,:3]
            normal=np.cross(a,b);keep=np.any(normal!=0,axis=1)&np.isfinite(normal).all(axis=1)
            triangles,ids=triangles[keep],ids[keep]
        if not len(triangles):
            return np.empty((0,3,5),np.float32),np.empty(0,np.int64),None
        # UV values are local to the selecting patch's mapped origin. All target
        # mappings below include this same integer label-pixel origin explicitly.
        local_uv=np.concatenate((triangles[...,3:],np.zeros((*triangles.shape[:2],1),np.float32)),axis=-1)
        uv_linear=Affine.from_rows(tuple((*row[:3],0) for row in self.label_to_uv.matrix))
        uv,uvbase=_transform(local_uv,uv_linear)
        triangles[...,3:]=uv[...,:2]+np.asarray(uvbase[:2],np.float32)
        lo=triangles[...,:3].min(axis=(0,1));hi=triangles[...,:3].max(axis=(0,1))
        bounds=(tuple(float(v)+b for v,b in zip(lo,base)),tuple(float(v)+b for v,b in zip(hi,base)))
        result=(np.ascontiguousarray(triangles),ids,(bounds[0],bounds[1],base))
        if not clip:
            self.reader._tiles.put(key,result,triangles.nbytes+ids.nbytes+128)
        self.reader._plans.put(('surface-bounds',key),result[2],512)
        return result

    def metadata(self,start,size):
        return dict(label_target=self.label_target,label_id=self.selecting['source'].id,label_start_yx=self.start,
                    label_size_yx=self.size,surface_id=self.surface.id,pixel_to_uv=self.selecting['reader'].registration.pixel_to_uv,
                    reference_id=self.reader.reference.id,reference_frame=self.reader._reference_frame,
                    selection=self.reader._reference_reader.resolved.selection.__dict__,origin_zyx=start,shape_zyx=size,
                    crop_to_reference=_shift(tuple(reversed(start))).matrix,array_to_level0=self.reader._reference_map.matrix,
                    full_origin_zyx=self.origin,full_shape_zyx=self.shape,thickness_um=self.thickness,padding_um=self.padding,
                    annotation_extent='crop',preparation_seconds=self.preparation_seconds)

    def _state(self,target,shape):
        import numpy as np
        r=self.reader;n=math.prod(shape)
        before=time.perf_counter()
        arrays=[np.full((target['channels'],n),target['reader'].fill_value,target['native_endian_dtype']),
                np.zeros((1,n),bool),np.zeros((1,n),bool),np.full(n,np.inf,np.float32),
                np.full(n,np.iinfo(np.int64).max,np.int64),np.full((n,2),np.nan,np.float32),
                np.zeros((1,n),bool)]
        if r.backend=='cuda':
            from .cuda import asarray
            r._uploads+=sum(a.nbytes for a in arrays)
            arrays=[asarray(a,device=r.device) for a in arrays]
        r._surface_metrics['output_allocation_seconds'] += time.perf_counter()-before
        return arrays

    def _raster(self,triangles,ids,origin,shape,states):
        import numpy as np
        from . import _reader
        r=self.reader
        before=time.perf_counter()
        try:
            plan=_reader.SurfacePlan(triangles,ids,shape,_matrix(self.metric),self.radius,self.expansion,origin,r._work_bytes//3)
        except (ValueError,MemoryError) as error:
            if 'batch exceeds' not in str(error) or len(ids)<=1:
                raise
            middle=len(ids)//2
            self._raster(triangles[:middle],ids[:middle],origin,shape,states)
            self._raster(triangles[middle:],ids[middle:],origin,shape,states)
            return
        r._surface_metrics['candidate_index_seconds'] += time.perf_counter()-before
        r._peak_work=max(r._peak_work,plan.nbytes*3)
        uv=triangles[...,3:].reshape(-1,2)
        # triangles contain UV relative to the selecting label patch origin.
        patch_shift=_shift(self.uv_origin)
        for name,target in self.targets.items():
            mapping=target['mapping'] @ patch_shift
            p,base=_pixel_points(uv,mapping)
            lo=tuple(max(0,math.floor(float(p[:,a].min()))+base[a]-1) for a in (1,2))
            hi=tuple(min(target['shape'][a],math.ceil(float(p[:,a].max()))+base[a]+2) for a in (1,2))
            size=tuple(max(0,b-a) for a,b in zip(lo,hi))
            if not all(size):
                lo=(0,0);size=(0,0)
            if math.prod(size)*target['channels']*target['dtype'].itemsize*3>r._work_bytes:
                if len(ids)<=1:
                    raise MemoryError('one triangle label footprint exceeds sampling_work_bytes')
                middle=len(ids)//2
                # Recurse for this target only to avoid processing earlier targets twice.
                saved=self.targets
                try:
                    self.targets={name:target}
                    self._raster(triangles[:middle],ids[:middle],origin,shape,states)
                    self._raster(triangles[middle:],ids[middle:],origin,shape,states)
                finally:
                    self.targets=saved
                continue
            key=('surface-label',target['storage_key'],lo,size,r.backend)
            label=r._tiles.get(key)
            if label is None:
                label=(r._canonical(target['reader'].read(lo,size),target['reader'])[:,0] if all(size)
                       else np.empty((target['channels'],0,0),target['native_endian_dtype']))
                r._tile_reads+=1
                if r.backend=='cuda':
                    from .cuda import asarray
                    r._uploads+=label.nbytes;label=asarray(label,device=r.device)
                r._tiles.put(key,label,label.nbytes)
            matrix=_shift((-lo[1],-lo[0],0)) @ mapping
            rows=matrix.matrix
            config=plan.configure(tuple(rows[a][b] for a in (0,1) for b in (0,1,3)),lo,size,target['channels'])
            state=states[name]
            before=time.perf_counter()
            if r.backend=='cpu':
                plan.cpu(config,r._bytes(label),target['dtype'].kind,target['dtype'].itemsize,*[r._bytes(a) for a in state])
            else:
                _reader._surface_raster_cuda(plan,config,label._array,target['dtype'].kind,target['dtype'].itemsize,*[a._array for a in state])
                r._surface_metrics['candidate_upload_bytes'] += plan.upload_bytes
            r._surface_metrics['raster_call_seconds'] += time.perf_counter()-before

    def read(self,start,size):
        import numpy as np
        r=self.reader
        _check_budget(r,size)
        if self.index is None:
            before=time.perf_counter()
            self.index=self.spatial_index()
            elapsed=time.perf_counter()-before
            self.preparation_seconds+=elapsed
            r._surface_metrics['preparation_seconds']+=elapsed
        # These are ordinary volume reads; surface targets are explicitly handled below.
        result=SampleResult({}, dict(r.metadata or {}))
        result.metadata['projection']=self.metadata(start,size)
        for name,target in r._targets.items():
            result.targets[name]=TargetResult(*r._sample_many(target,[(start,size)])[0])
            result[name].masks={k:r._sample_many(t,[(start,size)])[0][0] for k,t in target['companions'].items()}
        states={name:self._state(t,size) for name,t in self.targets.items()}
        request_lo=tuple(reversed(start));request_hi=tuple(s+n-1 for s,n in zip(request_lo,reversed(size)))
        tiles=self.index.query(request_lo,request_hi,self.expansion)
        for tile in tiles:
            key=('surface-bounds',(*self.key,tile,False))
            bounds=r._plans.get(key)
            if bounds is not None and any(b+e+.5<l or a-e-.5>h for a,b,e,l,h in zip(bounds[0],bounds[1],self.expansion,request_lo,request_hi)):
                continue
            triangles,ids,bounds=self.triangles(tile)
            if bounds is None or any(b+e+.5<l or a-e-.5>h for a,b,e,l,h in zip(bounds[0],bounds[1],self.expansion,request_lo,request_hi)):
                continue
            origin=tuple(s-b for b,s in zip(bounds[2],request_lo))
            # Reject triangles before label reads and candidate construction. A
            # spatial-index leaf can be much larger than a small training crop.
            lower=np.asarray(origin,np.float32)-np.float32(.5)-np.asarray(self.expansion,np.float32)
            upper=lower+np.asarray(tuple(reversed(size)),np.float32)+2*np.asarray(self.expansion,np.float32)
            lower=np.nextafter(lower,np.float32(-np.inf));upper=np.nextafter(upper,np.float32(np.inf))
            keep=(triangles[...,:3].max(axis=1)>=lower).all(axis=1)&(triangles[...,:3].min(axis=1)<=upper).all(axis=1)
            if not keep.any():
                continue
            triangles,ids=triangles[keep],ids[keep]
            self._raster(triangles,ids,origin,size,states)
        patch_shift=_shift(self.uv_origin)
        for name,target in self.targets.items():
            values,valid,conflict,_,_,uv,coverage=states[name]
            if r.backend == 'cpu':
                values=values.astype(target['dtype'],copy=False)
            result.targets[name]=TargetResult(values.reshape((target['channels'],*size)),
                                              valid.reshape((1,*size)))
            result[name].masks={'coverage':coverage.reshape((1,*size)),'conflict':conflict.reshape((1,*size))}
            if target['companions']:
                uv=uv.numpy() if r.backend=='cuda' else uv
                for key,child in target['companions'].items():
                    p,b=_pixel_points(uv,child['mapping'] @ patch_shift)
                    mask,_=r._sample_points(child,p,b)
                    result[name].masks[key]=mask.reshape((child['channels'],*size))
        return result


def _tile_regions(origin,shape,tile_size):
    for z in range(0,shape[0],tile_size[0]):
        for y in range(0,shape[1],tile_size[1]):
            for x in range(0,shape[2],tile_size[2]):
                offset=(z,y,x)
                yield tuple(a+b for a,b in zip(origin,offset)),tuple(min(t,n-a) for t,n,a in zip(tile_size,shape,offset))


def iter_project_surface(reader,surface,start,size,label_target,tile_size,thickness,padding):
    tile_size=tuple(map(operator.index,tile_size))
    if len(tile_size)!=3 or any(n<=0 or n>2**24 for n in tile_size):
        raise ValueError('tile_size must be three positive ZYX dimensions')
    reader._ready()
    with reader._lock:
        projection=Projection(reader,surface,start,size,label_target,thickness,padding)
    for origin,shape in _tile_regions(projection.origin,projection.shape,tile_size):
        reader._ready()
        with reader._lock:
            result = _read_crop(projection,origin,shape)
        yield result


def _copy_region(reader,src,dst,offset):
    import numpy as np
    if reader.backend=='cpu':
        dst[(slice(None),*(slice(a,a+n) for a,n in zip(offset,src.shape[1:])))] = src
    else:
        from . import _reader
        _reader._surface_region_copy_cuda(src._array,dst._array,src.shape[1:],dst.shape[1:],offset,src.shape[0],src.dtype.itemsize)


def _read_crop(projection,origin,shape):
    r=projection.reader
    _check_budget(r,shape)
    result=None
    for start,size in _tile_regions(origin,shape,(16,32,64)):
        block=projection.read(start,size)
        if size==shape:
            return block
        if result is None:
            result=SampleResult({}, block.metadata)
            result.metadata['projection']=projection.metadata(origin,shape)
            for name,target in block.targets.items():
                result.targets[name]=TargetResult(
                    r._owned_empty((target.data.shape[0],*shape),target.data.dtype),
                    r._owned_empty((target.valid.shape[0],*shape),target.valid.dtype),
                    {key:r._owned_empty((array.shape[0],*shape),array.dtype) for key,array in target.masks.items()})
        offset=tuple(a-b for a,b in zip(start,origin))
        for name,target in block.targets.items():
            dest=result[name]
            _copy_region(r,target.data,dest.data,offset)
            _copy_region(r,target.valid,dest.valid,offset)
            for key,array in target.masks.items():
                _copy_region(r,array,dest.masks[key],offset)
    return result


def project_surface(reader,surface,start,size,label_target,thickness,padding):
    projection=Projection(reader,surface,start,size,label_target,thickness,padding)
    return _read_crop(projection,projection.origin,projection.shape)


from ._sampling_engine import SamplingEngine


class SurfaceGeometry(SamplingEngine):
    """Reusable geometry-only UV interpolation, including the benchmark path.

    ``read_points`` returns local XYZ float32 coordinates, validity, and an
    integer XYZ origin. Coordinates are in the surface coordinate-value frame;
    offsets are physical micrometers. It does not sample a volume.
    """
    def __init__(self,surface,*,backend='cpu',device=0,sampling_cache_bytes=64<<20,
                 sampling_work_bytes=64<<20,**reader_options):
        limits=(operator.index(sampling_cache_bytes),operator.index(sampling_work_bytes),0)
        self._configuration=(surface,dict(reader_options),backend,device,limits)
        self._initialize_runtime(backend,device,limits,reader_options)
        try:
            self.geometry=Geometry(self,surface)
        except BaseException:
            self.close();raise

    def read_points(self,start,size,*,offsets_um=(0,)):
        import numpy as np
        self._ready();start,size=_region(start,size)
        offsets=tuple(float(v) for v in offsets_um)
        if not offsets or any(not math.isfinite(v) for v in offsets):
            raise ValueError('offsets_um must be nonempty and finite')
        if math.prod(size)*len(offsets)*32>self._work_bytes:
            raise MemoryError('geometry point request exceeds sampling_work_bytes; request smaller UV tiles')
        if self.geometry.surface.spacing_um is None:
            raise ValueError('surface physical spacing is unknown')
        with self._lock:
            y,x=np.indices(size,dtype=np.float32)
            uv=np.column_stack((x.ravel(),y.ravel()))
            points,valid,base=self.geometry.sample(uv,offsets,_scale(self.geometry.surface.spacing_um).inverse(),(start[1],start[0],0))
            return points.reshape(len(offsets),*size,3),valid.reshape(len(offsets),*size),base

    def _restore(self,state):
        surface,options,backend,device,limits=state
        self.__init__(surface,backend=backend,device=device,sampling_cache_bytes=limits[0],
                      sampling_work_bytes=limits[1],**options)
