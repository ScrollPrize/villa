"""Explicit input fixtures for the training benchmark and its validation tests.

Local analytic arrays exercise registration; they are never reported as real
annotation acceptance. Published selections pin IDs and caller-owned regions.
"""
from dataclasses import replace
import json
from training_support import Array, ReadSpec, image_arrays, label_arrays

PAIRED_STARTS = ((18576, 7984), (19344, 14192), (19856, 15696),
                 (20496, 13232), (22128, 11472), (23152, 7984))


def write_array(path, values, axes):
    """Write tiny uncompressed deterministic benchmark inputs, one chunk each."""
    path.mkdir(parents=True, exist_ok=True)
    (path / '.zarray').write_text(json.dumps(dict(zarr_format=2, shape=values.shape,
        chunks=values.shape, dtype=values.dtype.str, compressor=None, filters=None,
        fill_value=0, order='C')))
    (path / '.zattrs').write_text(json.dumps({'_ARRAY_DIMENSIONS': axes}))
    (path / '.'.join('0' for _ in values.shape)).write_bytes(values.tobytes())


def local_inputs(root):
    import numpy as np
    from vc_reader import (Affine, Catalog, Companion, Source, Storage, Surface,
                           SurfaceRegistration, ValueFrame, VolumeRegistration)
    z, y, x = np.indices((12, 14, 16), dtype=np.float32)
    ct = (z * 100 + y * 10 + x).astype(np.uint16)
    transform = Affine.from_rows(((1, 0, 0, 2), (0, 1, 0, 0), (0, 0, 1, 0)))
    volumes = {}
    for name, values in [('a', ct), ('b', ct + 100)]:
        path = root / name
        write_array(path / '0', values, ('z', 'y', 'x'))
        (path / '.zgroup').write_text('{"zarr_format":2}')
        (path / '.zattrs').write_text(json.dumps(dict(multiscales=[dict(axes=[dict(name=a, unit='micrometer') for a in 'zyx'],
            datasets=[dict(path='0', coordinateTransformations=[dict(type='scale', scale=[9.6]*3)])])])))
        volumes[name] = dict(id=name, properties=dict(pixel_size_um=9.6, shape=list(ct.shape)),
            data=[dict(type='ome-zarr', id=name, origins=[dict(path=str(path), access_roots=[dict(url='')])])])
    scroll = Catalog(dict(samples=dict(example=dict(sample=dict(id='example', properties=dict(type='scroll',
        volume_transforms=[dict(from_volume_id='a', transforms=[dict(to_volume_id='b', matrix=transform.matrix)])])),
        volumes=volumes)))).scrolls['example']
    registration = VolumeRegistration('example', 'a', affine=Affine.identity(), volume_spacing_um=(9.6,) * 3)

    def source(name, values, axes, **options):
        path = root / name
        write_array(path, values, axes)
        return Source(name, Storage(str(path), 'zarr'), axes=axes, **options)

    mask = source('quality', np.where(x % 2 == 0, 128, 255).astype(np.uint8), ('z', 'y', 'x'),
                  role='label', registration=registration)
    label = source('classes', (x % 3).astype(np.uint8), ('z', 'y', 'x'), role='label',
                   registration=registration, companions=(Companion('quality', mask),))
    rotation = Affine.from_rows(((0, -1, 0, 0), (1, 0, 0, 0), (0, 0, 1, 0)))
    field = source('directions', np.broadcast_to(np.array([1, 2, 3], np.float32)[:, None, None, None], (3, *ct.shape)),
                   ('c', 'z', 'y', 'x'), role='vector', registration=registration,
                   component_order=('x', 'y', 'z'), normalize=False,
                   value_frame=ValueFrame(sample_id='example', volume_id='a', affine=rotation))
    scroll = scroll.with_source(label, kind='label', volume='a').with_source(field, kind='representation', volume='a')
    layers = source('layers', np.stack((ct[3:6], ct[3:6] + 1000)), ('c', 'z', 'y', 'x'), role='image')
    ink = source('ink', (x[0] % 3).astype(np.uint8), ('y', 'x'), role='label')
    supervision = source('supervision', np.where(x[0] % 2 == 0, 128, 255).astype(np.uint8), ('y', 'x'), role='label')
    ink = replace(ink, companions=(Companion('supervision', supervision, True),))
    scroll = scroll.with_source(layers, kind='image', volume='a').with_source(ink, kind='label', volume='a', image=layers.id, same_grid=True)
    yy, xx = np.indices((8, 9), dtype=np.float32)
    xyz = source('geometry', np.stack((xx + 3, yy + 2, np.full_like(xx, 5))), ('c', 'y', 'x'), role='scalar')
    surface = Surface('plane', xyz.storage, 'example', 'a', axes=('c', 'y', 'x'), component_order=('x', 'y', 'z'),
                      xyz_to_volume=Affine.identity(), pixel_to_uv=((1, 0, 0), (0, 1, 0)), spacing_um=(9.6,) * 3)
    surface_label = source('surface-label', (xx % 3).astype(np.uint8), ('y', 'x'), role='label',
                           registration=SurfaceRegistration(surface, ((1, 0, 0), (0, 1, 0))))
    scroll = scroll.with_source(surface_label, kind='label', volume='a', segment='user:plane')
    return scroll, dict(label=label.id, field=field.id, ink=ink.id, surface_label=surface_label.id)


def build_workflow(name, root, metadata=None):
    from vc_reader import load
    if name in ('custom', 'registered', 'transfer', 'local-render', 'local-project'):
        scroll, ids = local_inputs(root)
        provenance = dict(fixture='analytic local arrays; not real annotation acceptance', workflow=name,
                          scroll=scroll.id, native_volume='a', reference_volume='b' if name == 'transfer' else 'a')
        if name == 'custom':
            label = scroll.labels[ids['ink']]
            spec = ReadSpec(label, mode='paired', image=label.images.zarr())
            arrays = {**image_arrays(layers=True), **label_arrays(masks=('supervision',))}
            requests = [(0, (i, i), (4, 5)) for i in range(6)]
            return [spec], requests, arrays, 2, provenance
        if name in ('local-render', 'local-project'):
            label = scroll.labels[ids['surface_label']]
            spec = ReadSpec(scroll.volumes['a'], mode='render' if name == 'local-render' else 'project',
                            targets={'label': label}, surface=label.source.registration.surface)
            arrays = {**image_arrays(layers=spec.mode == 'render'), **label_arrays(masks=('conflict',) if spec.mode == 'project' else ())}
            return [spec], [(0, (1, i), (3, 3)) for i in range(6)], arrays, 2 if spec.mode == 'render' else 3, provenance
        label, features = scroll.labels[ids['label']], scroll.representations[ids['field']]
        volume = scroll.volumes['a']
        targets = {'label': label, 'features': features}
        arrays = {**image_arrays(), **label_arrays(masks=('quality',)),
                  'features': Array('features', role='vector'), 'features_valid': Array('features', 'valid', 'valid')}
        if name == 'transfer':
            mapping = label.mappable_volumes['b']
            if 'surface_projection' in mapping.required_operations:
                raise ValueError('this selection requires explicit surface projection')
            volume = mapping.volume
            targets['scan_a'] = scroll.volumes['a'].level(0).source
            arrays['scan_a'] = Array('scan_a', valid='scan_a_valid')
            arrays['scan_a_valid'] = Array('scan_a', 'valid', 'valid')
            provenance.update(required_operations=mapping.required_operations, transform=mapping.transform.matrix,
                              transform_provenance='explicit analytic A→B translation, x + 2',
                              known_coverage='A shape ZYX 12,14,16 mapped to B x in [2,18); B x in [0,16)')
        provenance.update(label=label.id, features=features.id, reference_level=0,
                          native_registration=label.source.to_dict(), component_frame='selected reference volume XYZ voxels')
        return [ReadSpec(volume, targets=targets)], [(0, (1, 2, i), (3, 4, 5)) for i in range(6)], arrays, 3, provenance

    catalog = load(metadata) if metadata else load()
    if name == 'paired':
        from bench_paired import discover
        label, image = discover(catalog)
        return ([ReadSpec(label, mode='paired', image=image, layers=slice(40, 65))],
                [(0, start, (128, 128)) for start in PAIRED_STARTS],
                {**image_arrays(layers=True), **label_arrays(masks=('supervision', 'validation'))}, 2,
                dict(image=image.id, label=label.id, scroll=label.scroll.id, layers=[40, 65]))
    if name == 'published-target':
        from vc_reader import Selection
        volume = catalog.scrolls['PHercParis4'].volumes['20260411134726']
        identifier = ('catalog:PHercParis4/volume/20260411134726/snapshot/'
                      'de38ba456eedfa4e0c4c6d6b4e093f4664d635eae551bfbe923bbdf6dc4ec58d/data/2')
        asset = volume.representations[identifier]
        source = replace(asset.source, storage=replace(asset.source.storage, selection=Selection(level=0)))
        return ([ReadSpec(volume, level=2, targets={'features': source})],
                [(0, (9000 + i*2, 3600, 3600), (8, 8, 8)) for i in range(6)],
                {**image_arrays(), 'features': Array('features', role='scalar'),
                 'features_valid': Array('features', 'valid', 'valid')}, 3,
                dict(reference=volume.id, reference_level=2, target=source.to_dict(),
                     target_level=0, kind='published surface prediction, not a categorical annotation'))
    if name == 'ct':
        candidates = catalog.scrolls['PHercParis4'].volumes.at_resolution(8., rel_tol=.02)
        selected = candidates.get(volume='20230206171837')
        return ([ReadSpec(selected.volume, level=selected.level, resolution=selected)],
                [(0, (5264 + i, 3906, 4158), (4, 4, 4)) for i in range(6)], image_arrays(), 3,
                dict(volume=selected.volume.id, level=selected.level, candidates=[c.volume.id for c in candidates],
                     resolution_um=8., relative_tolerance=.02))
    if name in ('render', 'project', 'published-transfer'):
        # Discover ownership/geometry/affines first; declare only the canvas
        # convention missing from metadata. The optional TIFF mask is absent.
        scroll = catalog.samples['PHerc0009B']
        published = scroll.segments['20250919125754'].labels.ink(label_set='20260918')
        surface = replace(published.source.registration.surface, tifxyz_mask=False)
        registration = replace(published.source.registration, surface=surface,
                               pixel_to_uv=((1, 0, .5), (0, 1, .5)), origin='explicit')
        label = replace(published.source, id='user:training-surface-label', registration=registration)
        scroll = scroll.with_source(label, kind='label', volume=published.volume.id,
                                    segment=published.segment.id)
        asset = scroll.labels[label.id]
        destination = '20250521125136' if name == 'published-transfer' else published.volume.id
        mapping = asset.mappable_volumes[destination]
        targets = {'label': asset}
        mode = 'render' if name == 'render' else 'project'
        arrays = {**image_arrays(layers=mode == 'render'),
                  **label_arrays(masks=('supervision', 'validation', 'conflict') if mode == 'project' else ('supervision', 'validation'))}
        if name == 'published-transfer':
            targets['scan_a'] = scroll.volumes[published.volume.id].level(0).source
            arrays.update(scan_a=Array('scan_a', valid='scan_a_valid'),
                          scan_a_valid=Array('scan_a', 'valid', 'valid'))
        spec = ReadSpec(mapping.volume, mode=mode, targets=targets, surface=surface)
        # Projection bounds depend on geometry: use batch_size=1 or explicitly pad.
        return ([spec], [(0, (18576 + 2*i, 7984 + 2*i), (16, 16)) for i in range(6)], arrays,
                2 if mode == 'render' else 3, dict(label=label.to_dict(), workflow=name,
                published=published.summary(), native_volume=published.volume.id,
                reference_volume=destination, native_level=0, reference_level=0,
                required_operations=mapping.required_operations, affine=mapping.transform.matrix,
                derivation_path=mapping.transform.derivation_path,
                registration_basis='declared vc_render_tifxyz canvas convention; volume affine from catalog'))
    raise ValueError(f'unknown workflow {name}')
