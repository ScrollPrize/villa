"""Source descriptors over the existing Zarr group/array reader."""

from dataclasses import replace

from ..affine import Affine
from ..errors import CacheMissError
from ..sources import Selection, VolumeRegistration
from .image_files import FileInfo, _axes
from .source_resolution import (ResolvedSource, axes_metadata, resolve_registration,
                                spatial_selection, storage_transform, matches_reference)
from .zarr_group import ZarrGroup
from .zarr_reader import ZarrReader


class ZarrSource:
    def __init__(self, source, options):
        self.source, self.options = source, options
        self.group = self.reader = None
        self._grids = {}
        try:
            self._open()
        except BaseException:
            self.close()
            raise

    def _open(self):
        source = self.source
        selection = source.storage.selection
        location = source.storage.location
        if source.storage.pyramid:
            self.group = ZarrGroup.open(location, **self.options)
        else:
            try:
                self.reader = ZarrReader.open(location, **self.options)
            except (FileNotFoundError, CacheMissError):
                self.group = ZarrGroup.open(location, **self.options)
        if self.group is None:
            if any(v is not None for v in (selection.array_path, selection.level, selection.series, selection.corresponding)):
                raise ValueError('an array location cannot also select a group child, series or pyramid level')
            self.series, self.datasets = {}, []
            path = None
            self.series_index = None
        else:
            multiscales = self.group.multiscales
            index = selection.series
            if multiscales:
                if index is None:
                    if len(multiscales) != 1:
                        raise ValueError('multiple multiscale series; select series explicitly')
                    index = 0
                if isinstance(index, str):
                    matches = [i for i, ms in enumerate(multiscales) if ms.get('name') == index]
                    if len(matches) != 1:
                        raise ValueError(f'absent or ambiguous multiscale series {index!r}')
                    index = matches[0]
                if not 0 <= index < len(multiscales):
                    raise ValueError(f'multiscale series {index} does not exist')
                self.series = multiscales[index]
                self.datasets = self.series.get('datasets', [])
            else:
                if index is not None:
                    raise ValueError('series selection requires multiscale series metadata')
                self.series, self.datasets = {}, []
            self.series_index = index
            if selection.corresponding is not None:
                path = self._corresponding(selection)
            else:
                path = self._path(selection)
            self.reader = self.group.array(path)
        axes, units, transform, transform_units = self._grid(path)
        self.info = FileInfo(self.reader.shape, axes, self.reader.dtype, self.reader.shape, axes)
        self.registration = self._registration(path, selection)
        self.fill_value = self.reader.fill_value
        attrs = self.reader.attrs
        root_attrs = self.group.attrs if self.group else attrs
        self.channel_names = attrs.get('channel_names') or [c.get('label') for c in root_attrs.get('omero', {}).get('channels', [])]
        self.time_names = attrs.get('time_names', [])
        self.resolved = ResolvedSource(self.reader.location, self._canonical(path, selection), axes,
                                       self.reader.shape, str(self.reader.dtype), self.registration,
                                       transform, transform_units)
        self.path = path

    def _path(self, selection):
        if selection.array_path is not None:
            return selection.array_path.strip('/')
        if selection.level is not None:
            if self.datasets:
                if selection.level >= len(self.datasets):
                    raise ValueError(f'requested source level {selection.level} does not exist')
                return self.datasets[selection.level]['path']
            return str(selection.level)
        raise ValueError('group child is ambiguous; select array_path or a pyramid level explicitly')

    def _canonical(self, path, selection):
        return Selection(array_path=path, series=self.series_index, channels=selection.channels, time=selection.time)

    def _grid(self, path):
        if path in self._grids:
            return self._grids[path]
        reader = self.group.array(path) if self.group is not None else self.reader
        attrs = reader.attrs
        group_axes, group_units = axes_metadata(self.series.get('axes'))
        array_axes, array_units = axes_metadata(attrs.get('axes') or attrs.get('_ARRAY_DIMENSIONS'))
        if group_axes and array_axes and group_axes != array_axes:
            raise ValueError(f'array axes {array_axes} contradict group axes {group_axes}')
        observed = array_axes or group_axes or (None,) * len(reader.shape)
        axes = _axes(self.source.axes, observed, reader.shape)
        if group_units and array_units and any(u is not None for u in array_units):
            for axis, a, b in zip(axes, group_units, array_units):
                if a is not None and b is not None and a != b:
                    from .source_resolution import _UNITS
                    if _UNITS.get(a) != _UNITS.get(b) or a not in _UNITS:
                        raise ValueError(f'array and group unit declarations conflict for {axis}: {a}, {b}')
        units = array_units if array_units and any(u is not None for u in array_units) else group_units
        dataset = next((d for d in self.datasets if d['path'] == path), {})
        dataset_transforms = dataset.get('coordinateTransformations')
        array_transforms = attrs.get('coordinateTransformations')
        if dataset_transforms and array_transforms and dataset_transforms != array_transforms:
            raise ValueError('array and multiscale dataset declare conflicting transformations for the same grid')
        transform, transform_units = storage_transform(axes, units, [
            array_transforms or dataset_transforms, self.series.get('coordinateTransformations')])
        result = axes, units, transform, transform_units
        self._grids[path] = result
        return result

    def _registration(self, path, selection):
        source, reg = self.source, self.source.registration
        if reg is None:
            return None
        canonical = self._canonical(path, selection)
        original = reg.declared_grid or source.storage.selection
        if self.group is not None and self.group.multiscales:
            old_series = original.series
            if isinstance(old_series, str):
                candidates = [i for i, ms in enumerate(self.group.multiscales) if ms.get('name') == old_series]
                old_series = candidates[0] if len(candidates) == 1 else -1
            elif old_series is None and len(self.group.multiscales) == 1:
                old_series = 0
            if old_series != self.series_index:
                raise ValueError('selected series has no declared connection to the original series; supply placement')
        if spatial_selection(original) == spatial_selection(source.storage.selection) and original.corresponding is None:
            old_path = path
        elif self.group is not None:
            old_path = self._path(original)
        else:
            old_path = None
        declared = self._canonical(old_path, original)
        axes, _, transform, units = self._grid(path)
        _, _, old_transform, old_units = self._grid(old_path)
        reg = replace(reg, declared_grid=declared)
        source = replace(source, registration=reg)
        if isinstance(reg, VolumeRegistration) and reg.mode == 'storage':
            selected = self.group.array(path) if self.group is not None else self.reader
            frames = [selected.attrs.get('coordinateSystem')]
            if self.group is not None:
                frames.append(self.group.attrs.get('coordinateSystem'))
            for frame in frames:
                if frame is not None and frame != reg.storage_frame:
                    raise ValueError(f'storage frame {frame!r} contradicts declared frame {reg.storage_frame!r}')
        relative = None
        if (isinstance(reg, VolumeRegistration) and reg.origin == 'catalog-fallback'
                and transform is None and old_transform is None):
            if self.datasets:
                old_level = next((i for i, d in enumerate(self.datasets) if d['path'] == old_path), None)
                level = next((i for i, d in enumerate(self.datasets) if d['path'] == path), None)
            else:
                old_level = int(old_path) if old_path is not None and str(old_path).isdigit() else None
                level = int(path) if path is not None and str(path).isdigit() else None
            if old_level is not None and level is not None:
                relative = Affine.scale(2**(level-old_level))
        return resolve_registration(source, canonical, transform, units, old_transform, old_units, relative)

    def _corresponding(self, selection):
        request = selection.corresponding
        reference = request.reference
        reg = self.source.registration
        if not isinstance(reg, VolumeRegistration):
            raise ValueError('corresponding level requires volume registration; select a source level explicitly')
        paths = [d['path'] for d in self.datasets]
        have_transforms = any(self._grid(p)[2] is not None for p in paths)
        if not have_transforms:
            if request.sublevel is None or request.sublevel < 0:
                raise ValueError(f'no exact correspondence for requested reference level {reference.level}; select a source level explicitly')
            path = self._path(replace(selection, corresponding=None, level=request.sublevel))
            self.group.array(path)  # No fallback when the array does not exist.
            if not matches_reference(self.source, self._registration(path, selection), reference, self._grid(path)[0]):
                raise ValueError(f'no exact correspondence for requested reference level {reference.level}; select a source level explicitly')
            return path
        matches = []
        for path in paths:
            resolved = self._registration(path, selection)
            if matches_reference(self.source, resolved, reference, self._grid(path)[0]):
                matches.append(path)
        if len(matches) != 1:
            raise ValueError(f'requested reference level {reference.level} has {len(matches)} exact source levels; select a source level explicitly')
        return matches[0]

    def close(self):
        if self.group is not None:
            self.group.close()
        elif self.reader is not None:
            self.reader.close()
