"""Independent array/geometry checks for training integration (outside timing)."""
from contextlib import ExitStack
import json
import numpy as np


def verify_surface_images(spec, requests, samples, *, cache_dir, offline):
    """Check projected CT boxes or the generated zero-offset layer via raw reads.

    Float64 is deliberate in this oracle: it exposes errors in the production
    float32 geometry arithmetic. Geometry and CT are interpolated by SciPy.
    """
    from scipy.ndimage import map_coordinates
    from vc_reader import Source, Storage
    from vc_reader.reader.encoded_files import EncodedFiles
    options = dict(cache_dir=cache_dir, offline=offline)
    surface = spec.surface
    with ExitStack() as stack:
        volume = stack.enter_context(spec.reference.open(level=spec.level, **options))
        if spec.mode == 'project':
            for sample in samples:
                metadata = sample.metadata['projection']
                expected = volume.read(tuple(metadata['origin_zyx']), tuple(metadata['shape_zyx']))
                np.testing.assert_array_equal(sample.image.data[0], expected)
            return 'all six projected CT boxes match independent volume reads'
        files = EncodedFiles(**options)
        try:
            root = surface.geometry.location.rstrip('/')
            with files.open(root + '/meta.json') as (handle, _):
                scale = json.load(handle)['scale']
        finally:
            files.close()
        coordinates = [stack.enter_context(Source('oracle/' + axis, Storage(root + '/' + axis + '.tif', 'tiff'),
                       role='scalar', axes=('y', 'x')).open(**options)) for axis in 'xyz']
        # These pinned inputs use native-volume coordinates at reference level 0.
        assert spec.level == 0
        matrix = np.asarray(surface.xyz_to_volume.linear, dtype=np.float64)
        translation = np.asarray(surface.xyz_to_volume.translation, dtype=np.float64)
        depth = spec.offsets_um.index(0)
        for (_, start, size), sample in zip(requests, samples):
            y, x = np.indices(size, dtype=np.float64)
            grid = np.stack(((y + start[0]) * scale[1], (x + start[1]) * scale[0]))
            low = np.floor(grid.reshape(2, -1).min(axis=1)).astype(int)
            high = np.ceil(grid.reshape(2, -1).max(axis=1)).astype(int) + 1
            xyz = np.stack([map_coordinates(reader.read(tuple(low), tuple(high-low)).astype(np.float64),
                                grid - low[:, None, None], order=1, prefilter=False) for reader in coordinates], axis=-1)
            xyz = xyz @ matrix.T + translation
            low = np.floor(xyz.reshape(-1, 3).min(axis=0)).astype(int)[::-1]
            high = np.ceil(xyz.reshape(-1, 3).max(axis=0)).astype(int)[::-1] + 1
            data = volume.read(tuple(low), tuple(high-low)).astype(np.float32)
            expected = map_coordinates(data, np.moveaxis(xyz[..., ::-1]-low, -1, 0), order=1, prefilter=False)
            valid = sample.image.valid[0, depth]
            assert valid.any()
            np.testing.assert_allclose(sample.image.data[0, depth][valid], expected[valid], atol=.03, rtol=2e-5)
        return 'all six zero-offset CT layers match raw TIFF geometry + SciPy interpolation; physical offsets covered by analytic tests'


def verify_analytic(name, requests, samples):
    """Scalar affine expectations independent of the aligned-reader machinery."""
    for (_, start, size), sample in zip(requests, samples):
        if name in ('registered', 'transfer'):
            z, y, x = np.indices(size, dtype=np.float32)
            z, y, x = z + start[0], y + start[1], x + start[2]
            shift = 2 if name == 'transfer' else 0
            np.testing.assert_array_equal(sample.image.data[0], z*100 + y*10 + x + (100 if shift else 0))
            support = (x >= shift) & (x < 16 + shift)
            categories = ((x-shift) % 3).astype(np.uint8)
            categories[~support] = 0
            np.testing.assert_array_equal(sample.label.data[0], categories)
            np.testing.assert_array_equal(sample.label.valid[0], support)
            quality = np.where((x-shift) % 2 == 0, 128, 255).astype(np.uint8)
            quality[~support] = 0
            np.testing.assert_array_equal(sample.label.masks['quality'][0], quality)
            field = np.broadcast_to(np.array([-2, 1, 3], np.float32)[:, None, None, None], (3, *size)).copy()
            field[:, ~support] = 0
            np.testing.assert_array_equal(sample['features'].data, field)
            if shift:
                expected = z*100 + y*10 + x-shift
                expected[~support] = 0
                np.testing.assert_array_equal(sample['scan_a'].data[0], expected)
        elif name == 'local-render':
            y, x = np.indices(size, dtype=np.float32)
            y, x = y+start[0], x+start[1]
            expected = np.stack([z*100 + (y+2)*10 + x+3 for z in (4, 5, 6)])[None]
            np.testing.assert_array_equal(sample.image.data, expected)
            np.testing.assert_array_equal(sample.label.data[0], x.astype(np.uint8) % 3)
        else:
            metadata = sample.metadata['projection']
            z, y, x = np.indices(metadata['shape_zyx'], dtype=np.float32)
            sz, sy, sx = metadata['origin_zyx']
            np.testing.assert_array_equal(sample.image.data[0], (z+sz)*100 + (y+sy)*10 + x+sx)
            assert sample.label.valid.any()
    return 'all six requests match analytic affine/plane expectations; categorical projection covered by independent surface oracle suite'


def verify_projected_labels(spec, requests, reader, *, cache_dir, offline):
    """Independent zero-thickness categorical projection in the reference volume.

    Read raw TIFF coordinates and label pixels, compose the catalog volume
    affine, then use the float64 scalar oracle instead of production geometry.
    Positive-thickness CPU/CUDA equality is checked separately by the harness.
    """
    from surface_reference import crop_triangles, label_sampler, project
    from vc_reader import Affine, Source, Storage
    from vc_reader.reader.encoded_files import EncodedFiles
    options = dict(cache_dir=cache_dir, offline=offline)
    label = spec.targets['label'].source
    surface = spec.surface
    root = surface.geometry.location.rstrip('/')
    files = EncodedFiles(**options)
    try:
        with files.open(root + '/meta.json') as (handle, _):
            scale = json.load(handle)['scale']
    finally:
        files.close()
    uv = Affine.from_rows(((1/scale[0], 0, 0, 0), (0, 1/scale[1], 0, 0), (0, 0, 1, 0)))
    a, b = label.registration.pixel_to_uv
    pixels = Affine.from_rows(((a[0], a[1], 0, a[2]), (b[0], b[1], 0, b[2]), (0, 0, 1, 0)))
    placement = spec.reference.scroll.transform(surface.volume_id, spec.reference.id) @ surface.xyz_to_volume
    assert spec.level == 0
    with ExitStack() as stack:
        fields = [stack.enter_context(Source('oracle/'+axis, Storage(root+'/'+axis+'.tif', 'tiff'),
                  axes=('y', 'x'), role='scalar').open(**options)) for axis in 'xyz']
        raw_label = stack.enter_context(label.open(**options))
        for _, start, size in requests:
            actual = reader.project_surface(surface, start, size, label_target='label', thickness_um=0)
            meta = actual.metadata['projection']
            triangles = crop_triangles(fields, placement, pixels.inverse() @ uv,
                                       meta['origin_zyx'], meta['shape_zyx'])
            expected = project(triangles, label_sampler(raw_label), raw_label.dtype,
                               meta['origin_zyx'], meta['shape_zyx'], (spec.reference.pixel_size_um,)*3)
            assert actual.label.valid.any()
            for name, value, oracle in zip(('data', 'valid', 'conflict', 'coverage'),
                    (actual.label.data, actual.label.valid, actual.label.masks['conflict'], actual.label.masks['coverage']),
                    (*expected[:3], expected[4])):
                np.testing.assert_array_equal(value[0], oracle, err_msg=f'start={start}, field={name}')
    return 'six zero-thickness labels/coverage/conflicts match independent raw TIFF + catalog affine + scalar float64 rasterization'


def verify_scan_transfer(spec, samples, *, cache_dir, offline):
    """Sample A at B grid centers using raw A chunks and SciPy interpolation."""
    from scipy.ndimage import map_coordinates
    source = spec.targets['scan_a']
    transform = spec.reference.scroll.transform(spec.reference.id, source.registration.volume_id)
    with source.open(cache_dir=cache_dir, offline=offline) as raw:
        for sample in samples:
            meta = sample.metadata['projection']
            # Float64 oracle coordinates test the production float32 sampling.
            zyx = np.indices(meta['shape_zyx'], dtype=np.float64)
            zyx += np.asarray(meta['origin_zyx'])[:, None, None, None]
            xyz = np.moveaxis(zyx[::-1], 0, -1) @ np.asarray(transform.linear).T + np.asarray(transform.translation)
            low = np.floor(xyz.reshape(-1, 3).min(0)).astype(int)[::-1]
            high = np.ceil(xyz.reshape(-1, 3).max(0)).astype(int)[::-1] + 1
            data = raw.read(tuple(low), tuple(high-low)).astype(np.float32)
            expected = map_coordinates(data, np.moveaxis(xyz[..., ::-1]-low, -1, 0), order=1, prefilter=False)
            assert sample['scan_a'].valid.all()
            np.testing.assert_allclose(sample['scan_a'].data[0], expected, atol=.08, rtol=2e-5)
    return 'six source-scan arrays match independent inverse catalog affine + raw A chunks + SciPy interpolation'
