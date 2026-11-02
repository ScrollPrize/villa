"""Native sampling of resident float32 arrays; coordinates are array indices (ZYX/YX)."""
import numpy as np
from . import _reader
from .geometry import _device, _input, _array_view, _empty


def sample_array(values, points, *, order=1, boundary="zeros", fill=0):
    """Sample a 2D/3D scalar array at (...,D) indices, on CPU or CUDA.

    ``zeros`` interpolates with the fill value outside the array; ``constant``
    returns fill for any out-of-bounds query; ``border`` replicates edge values.
    Nearest ties round up. Use vc_reader.torch.sample_array for batched tensors
    and autograd. Float32 output; no implicit CUDA upload.
    """
    device = _device(values)
    values = _input(values, np.float32, device)
    points = _input(points, np.float32, device)
    if len(values.shape) not in (2, 3) or points.shape[-1] != len(values.shape):
        raise ValueError('expected 2D/3D scalar array and (...,D) points')
    shape = points.shape[:-1]
    n = int(np.prod(shape))
    # Host ndarray reshape and device array views are array integration only.
    if device >= 0:
        values = values.reshape((1, 1, *values.shape))
        points = points.reshape((1, n, points.shape[-1]))
    else:
        values = values.reshape(1, 1, *values.shape)
        points = points.reshape(1, n, points.shape[-1])
    out = _empty((1, 1, n), np.float32, device)
    _reader._sample_dense(_array_view(values), _array_view(points), _array_view(out), order, boundary, float(fill))
    return out.reshape(shape)


class GridField:
    """A fixed H,W,C field (C<=3) with explicit validity and shared surface interpolation.

    Values may be XYZ or arbitrary continuous fields. No coordinate sentinel is
    inferred. Bicubic is the default; use linear for UV maps whose overshoot
    would cross a seam, triangles for maps defined on a triangle mesh. Queries
    use (...,2) row,column coordinates. Outputs have three components, padded
    with zeros for fields with fewer channels, on both CPU and CUDA.
    The optional CUDA copy remains resident.
    """
    def __init__(self, values, valid, *, interpolation="bicubic"):
        values = np.asarray(values, np.float32)
        if values.ndim == 2:
            values = values[..., None]
        if values.ndim != 3 or not 1 <= values.shape[-1] <= 3:
            raise ValueError('field must have shape H,W,C with 1<=C<=3')
        self.channels = values.shape[-1]
        self.mode = {"linear": 1, "bicubic": 2, "triangles": 3}[interpolation]
        self.xyz = np.zeros((1, *values.shape[:2], 3), np.float32)
        self.xyz[0, ..., :self.channels] = values
        self.valid = np.array(valid, dtype=bool, order="C", copy=True)[None]
        self.derivatives = np.empty((1, 3, *self.xyz.shape[1:]), np.float32)
        self.device = -1
        _reader._prepare_surfaces(self.xyz, self.valid, self.derivatives)

    def cuda(self, device=0):
        from . import cuda
        result = object.__new__(type(self))
        result.channels, result.mode, result.device = self.channels, self.mode, int(device)
        result.xyz = cuda.asarray(self.xyz, device=device)
        result.valid = cuda.asarray(self.valid, device=device)
        result.derivatives = cuda.asarray(self.derivatives, device=device)
        return result

    def sample(self, rc, *, border=False, tangents=False):
        rc = _input(rc, np.float32, self.device)
        shape = rc.shape[:-1]
        if rc.shape[-1] != 2:
            raise ValueError('queries must have shape (...,2)')
        rc = rc.reshape((1, int(np.prod(shape)), 2))
        p, row, col = (_empty((1, rc.shape[1], 3), np.float32, self.device) for _ in range(3))
        ok = _empty((1, rc.shape[1]), np.bool_, self.device)
        _reader._sample_surfaces(*map(_array_view, (self.xyz, self.valid, self.derivatives, rc, p, row, col, ok)), self.mode, border)
        values = tuple(v.reshape((*shape, 3)) for v in (p, row, col))
        return (*values, ok.reshape(shape)) if tangents else (values[0], ok.reshape(shape))


class SurfaceTransfer:
    """Prepared surface-to-surface label lookup, with bicubic target geometry.

    Source coordinates follow the mapper's triangle split. UV and seam maps
    remain linear to avoid interpolation overshoot across correspondence seams.
    The returned indices select categorical labels without smoothing them.
    """
    def __init__(self, *, target_fields, target_valid, uv_rows, uv_cols, uv_valid,
                 source_fields, source_valid, label_shape, output_shape,
                 label_offset_yx=(0, 0), max_distance, filled_uv_rows=None,
                 filled_uv_cols=None, filled_uv_anchor_distance=None,
                 max_seam_distance=float('inf'), source_validity=None):
        target = GridField(np.stack(target_fields, -1), target_valid)
        source = GridField(np.stack(source_fields, -1), source_valid, interpolation='triangles')
        array = lambda a, dtype: np.empty(0, dtype) if a is None else np.ascontiguousarray(a, dtype)
        self.arrays = [target.xyz, target.valid, target.derivatives, source.xyz, source.valid,
                       array(uv_rows, np.float32), array(uv_cols, np.float32), array(uv_valid, bool),
                       array(filled_uv_rows, np.float32), array(filled_uv_cols, np.float32),
                       array(filled_uv_anchor_distance, np.float32), array(source_validity, np.uint8)]
        self.label_shape, self.output_shape = tuple(label_shape), tuple(output_shape)
        self.offset, self.distance, self.seam = tuple(label_offset_yx), float(max_distance), float(max_seam_distance)
        self.device = -1

    def cuda(self, device=0):
        from . import cuda
        result = object.__new__(type(self))
        result.__dict__.update(self.__dict__)
        result.arrays = [cuda.asarray(a, device=device) for a in self.arrays]
        result.device = int(device)
        return result

    def rasterize(self, bounds):
        shape = (bounds[1]-bounds[0], bounds[3]-bounds[2])
        indices = _empty(shape, np.int64, self.device)
        valid = _empty(shape, np.uint8, self.device)
        distances = _empty(shape, np.float32, self.device)
        stats = _reader._surface_transfer(list(map(_array_view, self.arrays)), self.label_shape,
            self.output_shape, tuple(bounds), self.offset, self.distance, self.seam,
            *map(_array_view, (indices, valid, distances)))
        return indices, valid, distances, stats


def sample_field_tile(field, valid, output_shape, row_start, row_end, col_start, col_end, *, interpolation='linear'):
    """Stored-grid continuous field at output pixel centres, with edge replication."""
    if interpolation == "linear":
        device = _device(field)
        field, valid = _input(field, np.float32, device), _input(valid, np.bool_, device)
        shape = (row_end-row_start, col_end-col_start)
        out, supported = _empty(shape, np.float32, device), _empty(shape, np.bool_, device)
        _reader._linear_field_tile(_array_view(field), _array_view(valid), tuple(output_shape),
                                   (row_start, row_end, col_start, col_end), _array_view(out), _array_view(supported))
        return out, supported
    field = GridField(field, valid, interpolation=interpolation)
    # Coordinate generation belongs to C++ so its convention is shared by C++ callers.
    return _reader._field_tile(field.xyz, field.valid, field.derivatives, tuple(output_shape),
                               (row_start, row_end, col_start, col_end), field.mode)


def project_mask(mask, positions, normals, valid, origin, shape, *, half_thickness):
    """Rasterize binary surface support through a normal band into a ZYX crop.

    Positions/normals are H,W,3 in ZYX; output is uint32 (zero or one), on the
    inputs' device. Uses the label projector's nearest-voxel placement and
    normal layers at most half a voxel apart, including the band endpoints.
    """
    device = _device(positions)
    positions, normals = (_input(a, np.float32, device) for a in (positions, normals))
    mask, valid = (_input(a, np.bool_, device) for a in (mask, valid))
    output = _empty(tuple(shape), np.uint32, device)
    _reader._project_mask(*map(_array_view, (positions, normals, mask, valid)), tuple(origin), float(half_thickness), _array_view(output))
    return output
