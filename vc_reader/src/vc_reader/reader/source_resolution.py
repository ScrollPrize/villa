"""Resolve selected storage grids without mutating their source declarations."""

from dataclasses import dataclass, replace
import math

from ..affine import Affine
from ..sources import Plane, Selection, SurfaceRegistration, VolumeRegistration

# Spatial storage transforms use XYZ column vectors and micrometers for
# dimensional frames. No bulk floating-point coordinate arrays are created.
_UNITS = {'meter': 1e6, 'metre': 1e6, 'm': 1e6, 'millimeter': 1e3, 'mm': 1e3,
          'micrometer': 1., 'micrometre': 1., 'um': 1., 'µm': 1.,
          'nanometer': .001, 'nm': .001}


@dataclass(frozen=True)
class ResolvedSource:
    location: str
    selection: Selection
    axes: tuple
    shape: tuple
    dtype: str
    registration: object
    storage_from_array: Affine | None = None
    storage_units: str | None = None


def spatial_selection(selection):
    return replace(selection, channels=None, time=None)


def axes_metadata(value):
    if value is None:
        return None, None
    names = tuple(a['name'] if isinstance(a, dict) else a for a in value)
    units = tuple(a.get('unit') if isinstance(a, dict) else None for a in value)
    return names, units


def storage_transform(axes, units, groups):
    """NGFF transformations in application order, then conversion to um."""
    transforms = [t for group in groups for t in (group or ())]
    if not transforms:
        return None, None
    result = Affine.identity()
    for transform in transforms:
        kind = transform.get('type')
        if kind not in ('scale', 'translation', 'identity'):
            raise ValueError(f'unsupported storage transformation {kind!r}; declare a supported grid')
        if kind == 'identity':
            continue
        values = transform.get(kind, ())
        if len(values) != len(axes) or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError('storage transformation rank/nonfinite values contradict storage axes')
        rows = [list(row) for row in Affine.identity().matrix]
        for axis, value in zip(axes, values):
            if axis in 'xyz':
                i = 'xyz'.index(axis)
                if kind == 'scale':
                    if value <= 0:
                        raise ValueError('storage scale must be positive')
                    rows[i][i] = value
                else:
                    rows[i][3] = value
        result = Affine.from_rows(rows) @ result
    spatial_units = {unit for axis, unit in zip(axes, units or (None,)*len(axes)) if axis in 'xyz'}
    dimensional = spatial_units != {None}
    if dimensional:
        if None in spatial_units or not spatial_units <= _UNITS.keys():
            raise ValueError(f'unknown or mixed dimensional/unitless storage units {spatial_units}')
        factors = {axis: _UNITS[unit] for axis, unit in zip(axes, units) if axis in 'xyz'}
        rows = [[v * factors.get(axis, 1.) for v in row] for axis, row in zip('xyz', result.matrix)]
        result = Affine.from_rows(rows)
    return result, 'micrometer' if dimensional else None


def regrid(registration, original_from_selected, selection):
    if registration is None:
        return None
    if isinstance(registration, SurfaceRegistration):
        if registration.pixel_to_uv is None:
            return replace(registration, declared_grid=selection)
        r = original_from_selected.matrix
        uv = registration.pixel_to_uv
        mapped = tuple((row[0]*r[0][0]+row[1]*r[1][0], row[0]*r[0][1]+row[1]*r[1][1],
                        row[0]*r[0][3]+row[1]*r[1][3]+row[2]) for row in uv)
        return replace(registration, pixel_to_uv=mapped, declared_grid=selection)
    if registration.plane is not None:
        plane, r = registration.plane, original_from_selected.matrix
        origin = tuple(o + r[0][3]*x + r[1][3]*y for o, x, y in zip(plane.origin, plane.basis_x, plane.basis_y))
        x = tuple(r[0][0]*a + r[1][0]*b for a, b in zip(plane.basis_x, plane.basis_y))
        y = tuple(r[0][1]*a + r[1][1]*b for a, b in zip(plane.basis_x, plane.basis_y))
        return replace(registration, plane=Plane(origin, x, y, plane.thickness), declared_grid=selection)
    return replace(registration, affine=registration.affine @ original_from_selected if registration.affine else None,
                   declared_grid=selection)


def resolve_registration(source, selection, selected_transform=None, units=None, declared_transform=None,
                         declared_units=None, relative=None):
    reg = source.registration
    if reg is None:
        return None
    if isinstance(reg, VolumeRegistration) and reg.mode == 'storage':
        if reg.affine is None or selected_transform is None:
            raise ValueError('storage placement requires both frame placement and array-to-storage metadata')
        if units is not None:
            if reg.storage_units not in _UNITS:
                raise ValueError('dimensional storage placement requires explicit known storage_units')
            selected_transform = Affine.scale(1 / _UNITS[reg.storage_units]) @ selected_transform
        elif reg.storage_units is not None:
            raise ValueError('unitless storage metadata cannot resolve dimensional storage placement')
        return replace(reg, affine=reg.affine @ selected_transform, mode='array', storage_frame=None,
                       storage_units=None, declared_grid=selection)
    if isinstance(reg, VolumeRegistration) and reg.origin == 'catalog-fallback' and selected_transform is not None:
        if units is not None:
            if reg.volume_spacing_um is None:
                raise ValueError('physical storage metadata needs volume_spacing_um; cannot use a catalog estimate')
            rows = tuple(tuple(v / spacing for v in row) for row, spacing in zip(selected_transform.matrix, reg.volume_spacing_um))
            affine = Affine.from_rows(rows)
        else:
            # Unitless metadata only establishes relative pyramid grids. Anchor
            # it at the explicitly known base grid, not an invented global frame.
            if declared_transform is None or reg.affine is None:
                raise ValueError('storage metadata has no connection to the declared volume grid')
            affine = reg.affine @ declared_transform.inverse() @ selected_transform
        return replace(reg, affine=affine, declared_grid=selection, origin='storage-metadata')
    if spatial_selection(reg.declared_grid or Selection()) == spatial_selection(selection):
        return replace(reg, declared_grid=selection)
    if relative is None and selected_transform is not None and declared_transform is not None:
        if units != declared_units:
            raise ValueError('selected and declared grids have incompatible units')
        relative = declared_transform.inverse() @ selected_transform
    if relative is None:
        raise ValueError('selected grid has no known relationship to declared_grid; supply explicit placement')
    return regrid(reg, relative, selection)


def matches_reference(source, registration, reference, axes=None):
    """Exact voxel resolution after the published volume-to-volume mapping."""
    if not isinstance(registration, VolumeRegistration):
        return False
    transform = registration.affine
    if registration.plane is not None:
        plane = registration.plane
        transform = Affine.from_rows(tuple((x, y, 0., o) for x, y, o in
                                          zip(plane.basis_x, plane.basis_y, plane.origin)))
    if transform is None:
        return False
    if (reference.sample_id, reference.volume_id) != (registration.sample_id, registration.volume_id):
        query = source.query
        if query is None or query.reference != reference:
            raise ValueError('mapped corresponding level needs its published query transform')
        transform = query.transform @ transform
    linear = transform.linear
    # Gram matrices accept exact reflections/permutations, but not shears.
    dimensions = ['xyz'.index(a) for a in (axes or source.axes or ('z', 'y', 'x')) if a in 'xyz']
    for i in dimensions:
        for j in dimensions:
            if sum(linear[k][i]*linear[k][j] for k in range(3)) != (float(2**(2*reference.level)) if i == j else 0.):
                return False
    return True


def array_placement(reg):
    if not isinstance(reg, VolumeRegistration):
        raise ValueError('volume registration is required; surface targets require surface projection')
    if reg.plane is not None:
        p = reg.plane
        affine = Affine.from_rows(tuple((x, y, n, o) for x, y, n, o in
                                       zip(p.basis_x, p.basis_y, p.thickness, p.origin)))
    else:
        affine = reg.affine
    if affine is None or reg.mode != 'array':
        raise ValueError('complete resolved array placement is required')
    inverse = affine.inverse()
    if not all(math.isfinite(v) for row in (*affine.matrix, *inverse.matrix) for v in row):
        raise ValueError('registration and its inverse must be finite')
    return affine

