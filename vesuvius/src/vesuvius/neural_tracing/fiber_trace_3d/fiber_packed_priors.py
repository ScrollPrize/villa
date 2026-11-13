"""Thin array bindings for parallel native measured-geometry preparation."""
import numpy as np


def _geometry(paths):
    from vc_object_store import PackedWindingGeometry
    # Retain one native view, not copies of the packed geometry. It owns the
    # NumPy buffer references while kernels release the GIL.
    if not hasattr(paths,'_native_winding_geometry'):
        paths._native_winding_geometry=PackedWindingGeometry(
            paths.offset,paths.keys['arc'],paths.xyz,paths.turns,
            np.ascontiguousarray(paths.seam.axis,dtype=np.float64),float(paths.seam.angle))
    return paths._native_winding_geometry


def geometry_priors(paths, ids, family, spacing=1000., batch_fibers=4096, batch_samples=65536, *, workers=32):
    """Return owned native prior columns; no Python fiber/point/batch loops.

    Legacy batch arguments are accepted for call compatibility only. Work is
    scheduled in coarse native tasks, independent of Python batch settings.
    """
    if spacing<=0 or batch_fibers<1 or batch_samples<1:raise ValueError('Invalid geometry sampling settings')
    return _geometry(paths).geometry_priors(
        np.ascontiguousarray(paths.indices(ids),dtype=np.int64),
        np.ascontiguousarray(family,dtype=np.int64),float(spacing),int(workers))


def _labels(labels):
    values=np.asarray(labels)
    dtype=np.int64 if np.issubdtype(values.dtype,np.integer) else np.float64
    return np.ascontiguousarray(values,dtype=dtype)


def _targets(paths,data,labels,directions,workers):
    return _geometry(paths).fixed_target_columns(
        np.ascontiguousarray(paths.indices(data['source_id']),dtype=np.int64),_labels(labels),
        np.ascontiguousarray(data['i'],dtype=np.int64),np.ascontiguousarray(data['j'],dtype=np.int64),
        np.ascontiguousarray(data['ds_a'],dtype=np.float64),np.ascontiguousarray(data['ds_b'],dtype=np.float64),
        None if directions is None else np.ascontiguousarray(directions,dtype=np.float64),int(workers))


def fixed_targets(paths, data, labels, batch_size=65536, *, workers=32):
    """Evaluate exact endpoint transport in one parallel native operation."""
    if batch_size<1:raise ValueError('Invalid endpoint batch size')
    return _targets(paths,data,labels,None,workers)['target']


def fixed_target_columns(paths, data, labels, directions, *, workers=32):
    """Fuse transport targets and final signed arcs into owned native buffers."""
    return _targets(paths,data,labels,directions,workers)


def width_initial(labels,radii,bins,*,workers=32):
    """Group once natively; only the small width-knot interpolation is NumPy."""
    from vc_object_store import winding_grouped_medians
    labels=np.asarray(labels)
    if not np.isfinite(labels).all() or np.any(labels!=np.rint(labels)):
        raise ValueError('Width initialization needs integer winding labels')
    grouped=winding_grouped_medians(np.ascontiguousarray(labels,dtype=np.int64),
        np.ascontiguousarray(radii,dtype=np.float64),int(workers))
    if not len(grouped['groups']):raise ValueError('Width initialization needs fibers')
    return np.maximum.accumulate(np.interp(bins,grouped['groups'],grouped['medians']*2*np.pi))
