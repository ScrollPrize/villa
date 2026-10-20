"""VC3D loading, coordinate units and all-sample training inputs (no CT access)."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vesuvius.ink_detection.paris4_3d_gambiarra.labels import Points
from ._scribbles import RELEASE_JSON, synthetic_scribbles
from vesuvius.ink_detection.paris4_3d_gambiarra import train_teacher, train_student


class Scribbles(unittest.TestCase):
    def setUp(self):
        self.doc = synthetic_scribbles()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'scribbles.json'

    def load(self, doc):
        self.path.write_text(json.dumps(doc))
        return Points(self.path)

    @unittest.skipUnless(RELEASE_JSON, "set PARIS4_INK_SCRIBBLES for full-dataset checks")
    def test_complete_dataset_and_stable_coordinates(self):
        p = Points(RELEASE_JSON)
        self.assertEqual((len(p.y), np.bincount(p.y).tolist()), (9627, [6698, 2929]))
        np.testing.assert_array_equal(p.train, np.arange(9627))
        self.assertEqual(p.heldout.shape, (0,))
        np.testing.assert_array_equal(p.parents, np.floor(p.q).astype(np.int64) - 128)
        # Frozen from the original all-label CSV: exact ZYX float64 coordinates and labels in source order.
        digest = hashlib.sha256(p.q.astype('<f8').tobytes() + p.y.astype('<i8').tobytes()).hexdigest()
        self.assertEqual(digest, '85b9325ff5382754c679c2102ad28b146bfca4b0c569c312de0910298deb2438')
        self.assertTrue(all(s['labels'] == 'all' for s in train_teacher.STAGES.values()))
        self.assertTrue(all(s['labels'] == 'all' for s in train_student.STAGES.values()))

    def test_json_order_and_viewer_links_do_not_change_training_samples(self):
        expected = self.load(self.doc)
        self.doc['collections'] = dict(reversed(list(self.doc['collections'].items())))
        for c in self.doc['collections'].values():
            c['points'] = dict(reversed(list(c['points'].items())))
            for point in c['points'].values():
                point.pop('links', None)
        actual = self.load(self.doc)
        np.testing.assert_array_equal(actual.q, expected.q)
        np.testing.assert_array_equal(actual.y, expected.y)

    def test_reject_wrong_coordinate_metadata(self):
        for key, value in [('vc_pointcollections_json_version', '2'),
                           ('vc_open_data_source_coordinate_level', 1),
                           ('vc_open_data_source_coordinate_scale_factor', 2),
                           ('vc_open_data_coordinate_space', 'another scan')]:
            with self.subTest(key=key):
                doc = copy.deepcopy(self.doc)
                doc[key] = value
                with self.assertRaises(ValueError):
                    self.load(doc)

    def test_reject_bad_labels_coordinates_and_ids(self):
        for kind in ('label', 'nonfinite', 'axis_count', 'duplicate_id', 'duplicate_coordinate', 'outside_ct'):
            with self.subTest(kind=kind):
                doc = copy.deepcopy(self.doc)
                ink, noink = doc['collections'].values()
                key = next(iter(ink['points']))
                point = ink['points'][key]
                if kind == 'label':
                    ink['name'] = 'unknown'
                elif kind == 'nonfinite':
                    point['p'][0] = float('nan')
                elif kind == 'axis_count':
                    point['p'] = [1, 2]
                elif kind == 'duplicate_id':
                    noink['points'][key] = {'p': [1000, 1000, 1000]}
                elif kind == 'duplicate_coordinate':
                    next(iter(noink['points'].values()))['p'] = point['p']
                else:
                    point['p'] = [0, 0, 0]
                with self.assertRaises(ValueError):
                    self.load(doc)

    def test_reject_malformed_collections(self):
        for doc in ([], dict(self.doc, collections=[]), dict(self.doc, collections={'1': []}),
                    dict(self.doc, collections={'1': {'name': 'ink', 'points': {}}})):
            with self.subTest(doc_type=type(doc)):
                with self.assertRaises(ValueError):
                    self.load(doc)


if __name__ == '__main__':
    unittest.main()
