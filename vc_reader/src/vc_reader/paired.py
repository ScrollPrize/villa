"""Read published or explicitly registered flattened image/label pairs."""
from dataclasses import replace
import math
import operator

from .affine import Affine
from ._sampling_engine import SamplingEngine
from .browsing import _select_level
from .sources import SampleResult, TargetResult, SurfaceRegistration, VolumeRegistration
from .reader.source_resolution import array_placement, pixel_affine as _pixel_affine


def _companion_to_label(label, mask):
    if isinstance(label, SurfaceRegistration) and isinstance(mask, SurfaceRegistration):
        if label.surface.id != mask.surface.id or label.pixel_to_uv is None or mask.pixel_to_uv is None:
            raise ValueError('companion surface registration has no shared pixel frame')
        return _pixel_affine(label.pixel_to_uv).inverse() @ _pixel_affine(mask.pixel_to_uv)
    if isinstance(label, VolumeRegistration) and isinstance(mask, VolumeRegistration):
        if (label.sample_id, label.volume_id) != (mask.sample_id, mask.volume_id):
            raise ValueError('flattened companion registration requires a shared volume frame')
        mapping = array_placement(label).inverse() @ array_placement(mask)
        if any(abs(mapping.matrix[2][i]) > 1e-7 for i in (0, 1, 3)):
            raise ValueError('flattened companion registration must lie in the label plane')
        return _xy(mapping)
    raise ValueError('flattened companion needs same_grid or complete pixel registration')


def _xy(matrix):
    """Keep the pixel plane; the image's stored z enumerates sampled layers."""
    r = matrix.matrix
    return Affine.from_rows(((r[0][0], r[0][1], 0, r[0][3]),
                             (r[1][0], r[1][1], 0, r[1][3]), (0, 0, 1, 0)))


def _same(a, b):
    # A small coefficient difference can move pixels on a large canvas.
    return a.matrix == b.matrix


def _shape_yx(reader):
    return tuple(reader.shape[reader.axes.index(a)] for a in ('y', 'x'))


def _relative(base, selected):
    a, b = base.resolved, selected.resolved
    if a.location == b.location and a.selection == b.selection:
        return Affine.identity()
    if a.storage_from_array is None or b.storage_from_array is None or a.storage_units != b.storage_units:
        raise ValueError('selected pixel grid has no known transform to its base grid')
    original, current = _xy(a.storage_from_array), _xy(b.storage_from_array)
    return Affine.identity() if _same(original, current) else original.inverse() @ current


class _PairedTarget:
    def __init__(self, reader, name):
        self.reader, self.name = reader, name

    @property
    def shape(self):
        self.reader._ready()
        target = self.reader._targets[self.name]
        depth = (self.reader._layer_count,) if self.name == 'image' and not self.reader._drop_layer else ()
        return (target['channels'], *depth, *self.reader.shape)

    def __getitem__(self, key):
        from .reader._indexing import parse_index
        r = self.reader
        r._ready()
        with r._lock:
            start, size, squeeze = parse_index(key, self.shape)
            target = r._targets[self.name]
            z = r._layer_start if self.name == 'image' else 0
            depth = r._layer_count if self.name == 'image' else 1
            if self.name == 'image' and not r._drop_layer:
                z += start[1]
                depth = size[1]
            # A zero channel request needs no I/O either.
            region_size = (depth, *size[-2:]) if size[0] else (0, *size[-2:])
            array, _ = r._sample_many(target, [((z, *start[-2:]), region_size)])[0]
            shape = tuple(n for n, drop in zip(size, squeeze) if not drop)
            if r.backend == 'cuda':
                offset = start[0] * math.prod(region_size) if all(size) else 0
                return array._view(shape, offset)
            array = array[start[0]:start[0]+size[0]].reshape(shape)
            return array[()] if not shape else array


class PairedReader(SamplingEngine):
    """Shared indexing is YX on the selected image, with independent layer depth.

    label/mask grids are selected independently. A matching stored grid is used
    when available; otherwise categorical values use exact nearest-neighbor
    sampling. CPU workers reopen descriptors; CUDA readers remain process-local.
    """
    def __init__(self, label, image, pairing, *, level=0, layers=slice(None), backend='cpu',
                 sampling_cache_bytes=64 << 20, sampling_work_bytes=64 << 20,
                 plan_cache_bytes=8 << 20, device=0, **reader_options):
        limits = tuple(map(operator.index, (sampling_cache_bytes, sampling_work_bytes, plan_cache_bytes)))
        self._configuration = (label, image, backend, pairing, level, layers, limits, device, dict(reader_options))
        self._initialize_runtime(backend, device, limits, reader_options)
        try:
            if label.role != 'label' or image.role != 'image':
                raise ValueError('paired reading requires label and image source roles')
            # The original registration remains on public descriptors. A flattened
            # read uses its explicit pixel relationship, without projecting to 3D.
            raw_image = replace(image, registration=None, companions=())
            raw_label = replace(label, registration=None, companions=())
            image_base = self._open(_select_level(raw_image, 0))
            selected_image = self._open(_select_level(raw_image, level))
            if selected_image.spatial_axes != ('z', 'y', 'x'):
                raise ValueError('paired image must have sampled layers and YX spatial axes')
            self._shape = _shape_yx(selected_image)
            from .reader._indexing import parse_index
            layer_start, layer_size, layer_drop = parse_index(layers, (selected_image.shape[selected_image.axes.index('z')],))
            self._layer_start, self._layer_count, self._drop_layer = layer_start[0], layer_size[0], layer_drop[0]
            self._targets['image'] = self._sampling_target(raw_image, selected_image, Affine.identity())
            label_base = self._open(_select_level(raw_label, 0))
            if label_base.spatial_axes != ('y', 'x'):
                raise ValueError('paired labels must have YX spatial axes')
            if pairing.pixel_affine is None:
                raise ValueError('label-pixel-to-image-pixel mapping is unresolved')
            rows = pairing.pixel_affine
            mapping = _pixel_affine(rows)
            # Published target-canvas relationships are expressed in the shared
            # storage frame when both arrays publish dimensional transforms.
            a, b = image_base.resolved, label_base.resolved
            shared_metadata = a.storage_from_array is not None and b.storage_from_array is not None
            if shared_metadata and a.storage_units != b.storage_units and pairing.evidence != 'explicit user pixel mapping':
                raise ValueError('paired pixel grids have incompatible storage units')
            if shared_metadata and a.storage_units == b.storage_units:
                original, current = _xy(a.storage_from_array), _xy(b.storage_from_array)
                observed = Affine.identity() if _same(original, current) else original.inverse() @ current
                if pairing.evidence.startswith('published') or pairing.evidence.startswith('segment/'):
                    mapping = observed
                elif pairing.evidence == 'explicit same grid' and not _same(observed, mapping):
                    raise ValueError('storage transforms contradict same_grid pairing')
            elif pairing.evidence.startswith(('published', 'segment/')) and _shape_yx(image_base) != _shape_yx(label_base):
                raise ValueError('published pixel grids differ without storage transforms; pairing is unresolved')
            if pairing.evidence == 'explicit same grid' and _shape_yx(image_base) != _shape_yx(label_base):
                raise ValueError('same_grid pairing has different base shapes')
            image_from_selected = _relative(image_base, selected_image)
            label_to_selected = image_from_selected.inverse() @ mapping
            self._targets['label'] = self._planar_target(raw_label, label_base, label_to_selected, level)
            self._masks = {}
            for companion in label.companions:
                mask_source = replace(companion.source, registration=None, companions=())
                base = self._open(_select_level(mask_source, 0))
                if base.spatial_axes != ('y', 'x'):
                    raise ValueError('flattened masks require YX spatial axes')
                if 'c' in base.axes and base.shape[base.axes.index('c')] != 1:
                    raise ValueError('companion masks require one channel')
                mask_to_label = Affine.identity()
                if companion.same_grid:
                    if _shape_yx(base) != _shape_yx(label_base):
                        raise ValueError(f'{companion.name}: companion base shape contradicts same_grid')
                    a, b = base.resolved, label_base.resolved
                    if a.storage_from_array is not None and b.storage_from_array is not None and (
                            a.storage_units != b.storage_units or not _same(_xy(a.storage_from_array), _xy(b.storage_from_array))):
                        raise ValueError(f'{companion.name}: companion storage transforms contradict same_grid')
                else:
                    # Resolve each declared grid before comparing placements.
                    full_resolution_label = self._open(_select_level(replace(label, companions=()), 0))
                    full_resolution_mask = self._open(_select_level(replace(companion.source, companions=()), 0))
                    mask_to_label = _companion_to_label(full_resolution_label.registration, full_resolution_mask.registration)
                self._masks[companion.name] = self._planar_target(mask_source, base, label_to_selected @ mask_to_label, level)
            self.metadata = dict(image_id=image.id, label_id=label.id, image_selection=selected_image.resolved.selection,
                                 label_selection=self._targets['label']['reader'].resolved.selection,
                                 depth_semantics='sampled_layers', pairing_evidence=pairing.evidence)
        except BaseException:
            self.close()
            raise

    def _planar_target(self, source, base, base_to_image, preferred_level):
        selected, selected_to_image = base, base_to_image
        if not _same(selected_to_image, Affine.identity()) and source.storage.pyramid:
            # Header-declared levels are candidates, not an assumed shared level
            # count. Never request a nonexistent label/mask level.
            backend = base._backend
            datasets = backend.datasets if backend is not None else ()
            levels = list(range(len(datasets)))
            if preferred_level in levels:
                levels.remove(preferred_level)
                levels.insert(0, preferred_level)
            for level in levels:
                candidate = self._open(_select_level(source, level))
                try:
                    mapping = base_to_image @ _relative(base, candidate)
                except ValueError:
                    continue
                if _same(mapping, Affine.identity()):
                    selected, selected_to_image = candidate, Affine.identity()
                    break
        if _same(selected_to_image, Affine.identity()):
            selected_to_image = Affine.identity()
        target = self._sampling_target(source, selected, selected_to_image.inverse(), plane=True, planar_grid=True)
        return target

    @property
    def shape(self):
        self._ready()
        return self._shape

    def __getitem__(self, key):
        from .reader._indexing import parse_index
        self._ready()
        if isinstance(key, str):
            if key not in self._targets:
                raise KeyError(key)
            return _PairedTarget(self, key)
        start, size, squeeze = parse_index(key, self.shape)
        result = self.read(start, size)
        if any(squeeze):
            spatial = tuple(n for n, drop in zip(size, squeeze) if not drop)
            for target in result.targets.values():
                target.data = target.data.reshape((*target.data.shape[:-2], *spatial))
                target.valid = target.valid.reshape((*target.valid.shape[:-2], *spatial))
                for name, array in target.masks.items():
                    target.masks[name] = array.reshape((*array.shape[:-2], *spatial))
        return result

    def read(self, start, size):
        return self.read_many([(start, size)])[0]

    def read_many(self, regions):
        self._ready()
        with self._lock:
            normalized = []
            for start, size in regions:
                start, size = tuple(map(operator.index, start)), tuple(map(operator.index, size))
                if len(start) != 2 or len(size) != 2 or any(n < 0 or n > 2**24 for n in size) or any(abs(v) > 2**40 for v in start):
                    raise ValueError('paired regions require YX start/size, nonnegative bounded sizes')
                normalized.append((start, size))
            results = [SampleResult({}, dict(self.metadata)) for _ in normalized]
            for name, target in self._targets.items():
                z, depth = (self._layer_start, self._layer_count) if name == 'image' else (0, 1)
                requests = [((z, *start), (depth, *size)) for start, size in normalized]
                for result, (array, valid) in zip(results, self._sample_many(target, requests)):
                    if name != 'image' or self._drop_layer:
                        array = array.reshape((array.shape[0], *array.shape[-2:]))
                        valid = valid.reshape((1, *valid.shape[-2:]))
                    result.targets[name] = TargetResult(array, valid)
            requests = [((0, *start), (1, *size)) for start, size in normalized]
            for name, target in self._masks.items():
                for result, (array, _) in zip(results, self._sample_many(target, requests)):
                    result['label'].masks[name] = array.reshape((array.shape[0], *array.shape[-2:]))
            return results

    def stats(self):
        result = super().stats()
        result['masks'] = {name: target['reader'].stats() for name, target in self._masks.items()}
        return result

    def _restore(self, state):
        label, image, backend, pairing, level, layers, limits, device, options = state
        self.__init__(label, image, pairing, backend=backend, level=level, layers=layers,
                      sampling_cache_bytes=limits[0], sampling_work_bytes=limits[1], plan_cache_bytes=limits[2],
                      device=device, **options)
