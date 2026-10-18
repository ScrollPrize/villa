"""Relationship and resolution contracts, independent of optional readers."""
from dataclasses import replace
import json
import pickle
import unittest
from unittest.mock import patch

from vc_reader import (Affine, AmbiguousMatchError, Catalog, NotFoundError, ScrollSource,
                       Source, Storage, VolumeRegistration)


def artifact(kind, identifier, target=None, **parameters):
    return dict(type=kind, id=identifier, parameters={**({'target_volume': target} if target else {}), **parameters})


def catalog():
    older = artifact('ink-labels', 'old', 'a', label_set='2020')
    newer = artifact('ink-labels', 'new', 'a', label_set='2021')
    newer['properties'] = dict(supersedes='2020', arrays=['inklabels', 'supervision', 'validation'])
    return Catalog(dict(samples=dict(s=dict(
        sample=dict(id='s', properties=dict(type='scroll', volume_transforms=[dict(from_volume_id='a', transforms=[dict(to_volume_id='b', matrix=Affine.scale(2).matrix)])])),
        volumes=dict(a=dict(id='a', properties=dict(pixel_size_um=4), data=[artifact('ome-zarr', 'ct'), artifact('surface-prediction-zarr', 'rep', level=1)]),
                     b=dict(id='b', properties=dict(pixel_size_um=8), data=[artifact('ome-zarr', 'ct')]),
                     c=dict(id='c', properties=dict(pixel_size_um=7.91), data=[artifact('ome-zarr', 'ct')]),
                     historical=dict(id='historical', properties=dict(pixel_size_um=8, superseded_by=dict(volume_id='b')), data=[artifact('ome-zarr', 'ct')])),
        segments=dict(seg=dict(id='seg', original_volume_id='b', data=[older, newer, artifact('ink-labels', 'other', 'b', label_set='2021'),
            artifact('layers-zarr', 'render1', 'a'), artifact('layers-zarr', 'render2', 'a'), artifact('new-kind', 'mystery', 'a')]),
                      missing=dict(id='missing', original_volume_id='b', data=[artifact('ink-labels', 'missing', 'b', label_set='2021')]))),
        fragment=dict(sample=dict(id='fragment', properties=dict(type='fragment'))))))


class BrowsingTests(unittest.TestCase):
    def test_shared_records_direct_context_and_reverse_links(self):
        c = catalog()
        scroll = ScrollSource('s', 'a', catalog=c)
        label = scroll.segments['seg'].labels.ink(label_set='2021')
        self.assertEqual(label.id, scroll.labels.ink().id)
        self.assertIs(label.source, c.samples['s'].labels[label.id].source)
        self.assertEqual(label.volume.id, 'a')
        self.assertEqual(label.segment.id, 'seg')
        self.assertEqual([s.id for s in scroll.segments], ['seg'])  # no mesh required
        self.assertEqual([l.artifact_id for l in scroll.labels], ['new'])
        self.assertEqual([v.id for v in label.mappable_volumes], ['a', 'b'])
        mapping = label.mappable_volumes['b']
        self.assertEqual(mapping.transform, Affine.scale(2))
        self.assertEqual(mapping.required_operations, ('surface_projection', 'volume_transform'))
        self.assertEqual(len(label.images.filter(format='zarr')), 2)
        with self.assertRaisesRegex(AmbiguousMatchError, 'render1'):
            label.images.zarr()
        self.assertEqual(label.images.zarr(artifact_id='render1').artifact_id, 'render1')
        with self.assertRaisesRegex(NotFoundError, 'candidates'):
            label.images.zarr(artifact_id='absent')
        for render in label.images:
            self.assertEqual(render.labels.ink().id, label.id)
            self.assertEqual(render.volume.id, 'a')
            self.assertEqual(render.depth_semantics, 'sampled_layers')
            self.assertEqual(render.source.axes, ('z', 'y', 'x'))
            self.assertIsNone(render.source.registration)
        self.assertEqual(len(c.samples['s'].on_volume('b').segments['missing'].labels.ink().images), 0)
        self.assertEqual(len(scroll.artifacts.filter(artifact_kind='new-kind')), 1)
        self.assertEqual(len(scroll.representations), 1)
        self.assertEqual(len(c.scrolls), 1)
        self.assertEqual(c.samples['fragment'].type, 'fragment')

    def test_supersession_history_and_candidates(self):
        s = catalog().samples['s']
        labels = s.on_volume('a').labels
        old = labels.ink(include_superseded=True, label_set='2020')
        self.assertEqual(labels[old.id].id, old.id)
        with self.assertRaisesRegex(NotFoundError, 'candidates'):
            labels.ink(label_set='2020')
        self.assertEqual([v.artifact_id for v in labels.filter(artifact_kind='ink-labels')], ['new'])
        self.assertEqual(len(labels.filter(include_superseded=True)), 2)
        self.assertEqual(len(s.volumes), 3)
        self.assertTrue(s.volumes['historical'].is_superseded)
        self.assertEqual(len(s.volumes.filter(include_superseded=True)), 4)
        with self.assertRaises(AmbiguousMatchError):
            s.labels.ink(label_set='2021')  # distinct targets stay visible

    def test_resolution_all_matches_stable_and_metadata_only(self):
        c = catalog()
        with patch('urllib.request.urlopen', side_effect=AssertionError('network request')):
            matches = c.scrolls.volumes.at_resolution(8, rel_tol=.02)
            self.assertEqual([(m.volume.id, m.level) for m in matches], [('a', 1), ('b', 0), ('c', 0)])
            self.assertFalse(any(m.verified for m in matches))
            self.assertEqual(len(c.scrolls.volumes.at_resolution(8)), 2)
        for um, tolerance in ((0, .01), (float('nan'), .01), (8, -.1)):
            with self.assertRaises(ValueError):
                c.scrolls.volumes.at_resolution(um, rel_tol=tolerance)

    def test_overlays_explicit_pairing_serialization_and_native_registration(self):
        c = catalog()
        before = json.dumps(c.raw)
        s = c.samples['s']
        native = s.labels.ink(artifact_id='new').source
        render = Source('render', Storage('/tmp/custom.zarr', 'zarr'), role='image', axes=('z', 'y', 'x'))
        label = Source('label', Storage('/tmp/label.png', 'png'), role='label', axes=('y', 'x'), artifact_kind='ink-labels')
        s = s.with_source(render, kind='image', volume='a', segment='user:seg')
        s = s.with_source(label, kind='label', volume='a', segment='user:seg', image=render.id, same_grid=True)
        self.assertIs(s.labels[native.id].source, native)
        self.assertEqual(s.labels[label.id].images.get().id, render.id)
        self.assertEqual(s.images[render.id].labels.get().id, label.id)
        self.assertEqual(s.labels[label.id].segment.id, 'user:seg')
        self.assertEqual(json.dumps(c.raw), before)
        self.assertNotIn(label.id, [l.id for l in c.samples['s'].labels])
        for restored in (pickle.loads(pickle.dumps(s)), ScrollSource.from_dict(json.loads(json.dumps(s.to_dict())))):
            self.assertEqual(restored.labels[label.id].images.get().id, render.id)
            self.assertEqual(restored.labels[native.id].source.registration, native.registration)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            c.samples['s'].with_source(label, kind='label', image=render.id)
        unknown = c.samples['s'].with_source(label, kind='label')
        self.assertFalse(unknown.labels[label.id].images)
        self.assertFalse(unknown.labels[label.id].mappable_volumes)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            s.with_source(label, kind='label')
        local = Source('local-volume', Storage('/tmp/volume.zarr', 'zarr'), role='image',
                       registration=VolumeRegistration('s', 'user:local-volume', affine=Affine.identity(), volume_spacing_um=(8.,)*3))
        s = s.with_source(local, kind='volume')
        self.assertEqual(s.volumes['user:local-volume'].source.id, local.id)
        self.assertEqual(len(s.volumes.at_resolution(8)), 3)
        rep = Source('rep', Storage('/tmp/rep.zarr', 'zarr'), role='scalar', registration=VolumeRegistration('s', 'a', affine=Affine.scale(2)))
        s = s.with_source(rep, kind='representation', volume='a')
        self.assertEqual(s.on_volume('a').representations[rep.id].source.registration, rep.registration)
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            s.with_source(replace(rep, id='bad'), kind='representation', volume='b')

    def test_unpublished_native_target_and_historical_mapped_destination(self):
        from test_sources import catalog as source_catalog
        c = source_catalog()
        label = c.samples['s'].labels.ink(artifact_id='absent')
        self.assertEqual(label.volume.id, 'unpublished')
        self.assertIsNone(label.volume.record)
        self.assertEqual(label.mappable_volumes['a'].transform, Affine.scale(2))
        self.assertEqual([m.id for m in label.mappable_volumes.filter(include_superseded=True)], ['a'])

    def test_cycles_and_malformed_overlay_relationships(self):
        c = catalog()
        s = c.samples['s']
        first = Source('first', role='label', supersedes=('user:second',))
        s = s.with_source(first, kind='label')
        with self.assertRaisesRegex(ValueError, 'cycle'):
            s.with_source(Source('second', role='label', supersedes=('user:first',)), kind='label')
        raw = s.to_dict()
        raw['catalog']['samples']['s']['sample']['id'] = 'changed'
        self.assertEqual(c.raw['samples']['s']['sample']['id'], 's')
        raw = s.to_dict()
        raw['scroll'] = 'absent'
        with self.assertRaisesRegex(ValueError, 'unknown sample'):
            ScrollSource.from_dict(raw)
        raw = s.to_dict()
        raw['overlays'][0]['pairs'] = [dict(label=first.id, image='absent', affine=((1,0,0),(0,1,0)))]
        with self.assertRaisesRegex(ValueError, 'known label and image'):
            ScrollSource.from_dict(raw)

    def test_explicit_published_reference_narrows_convention(self):
        c = catalog()
        tree = c.raw
        segment = tree['samples']['s']['segments']['seg']
        for a in segment['data']:
            if a['id'] in ('render1', 'render2'):
                a['origins'] = [dict(path=a['id'], access_roots=[dict(url='https://example.test')])]
            if a['id'] == 'new':
                a['creation_info'] = dict(target=dict(render='https://example.test/render2'))
        label = Catalog(tree).samples['s'].labels.ink(artifact_id='new')
        self.assertEqual(label.images.zarr().artifact_id, 'render2')


class PublishedLabelMappingTests(unittest.TestCase):
    def test_volume_affine_is_visible_with_unresolved_surface_pixels(self):
        from pathlib import Path
        fixture = Path(__file__).resolve().parents[1] / 'benchmarks/fixtures/training/catalog.json'
        with patch('urllib.request.urlopen', side_effect=AssertionError('metadata browsing requested I/O')):
            c = Catalog.from_file(fixture)
            label = c.samples['PHerc0009B'].segments['20250919125754'].labels.ink(label_set='20260918')
            summary = label.summary()
            self.assertEqual(summary['native_volume'], '20250820154339')
            self.assertIn('registration.pixel_to_uv', summary['unresolved_fields'])
            mapping = label.mappable_volumes['20250521125136']
            self.assertEqual(mapping.required_operations, ('surface_projection', 'volume_transform'))
            self.assertEqual(mapping.coverage.overlap_ratio, 1.)
            self.assertEqual(mapping.transform.derivation_path, '20250820154339-20250521125136⁻¹')
            self.assertEqual(mapping.transform.matrix, (
                (0.2776774047309435, 0.0004236055284734014, 0.010623949057789617, -50.119403010604934),
                (-0.000621669391020947, 0.27727002476200535, -0.008217722069624534, 133.0058074289684),
                (-0.010596257965952037, 0.007226232168178798, 0.27734094887564803, 250.4726348647675)))
            self.assertIn('20250521125136', repr(label))
            restored = json.loads(json.dumps(summary))
            self.assertEqual(len(restored['mappable_volumes']), 2)
            self.assertEqual(label.volume.id, summary['native_volume'])
            # Native ownership never moves the label into B's direct-label list.
            self.assertFalse(mapping.volume.labels)
