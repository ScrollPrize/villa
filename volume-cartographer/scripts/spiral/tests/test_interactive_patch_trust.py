from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interactive_patch_trust import (
    collect_trusted_collection_points,
    select_interactive_patch_pool,
)


def _select(record, verified=None, unverified=None,
            pending_verified=None, pending_unverified=None):
    verified = {} if verified is None else verified
    unverified = {} if unverified is None else unverified
    pending_verified = {} if pending_verified is None else pending_verified
    pending_unverified = {} if pending_unverified is None else pending_unverified
    return select_interactive_patch_pool(
        record, verified, unverified, pending_verified, pending_unverified)


class InteractivePatchTrustTests(unittest.TestCase):
    def test_legacy_record_defaults_only_to_verified_pool(self):
        pending_verified = {}
        pending_unverified = {}

        classification, target = _select(
            {'id': 'legacy-patch'}, pending_verified=pending_verified,
            pending_unverified=pending_unverified)

        self.assertEqual(classification, 'verified')
        self.assertIs(target, pending_verified)
        self.assertIsNot(target, pending_unverified)

    def test_unverified_record_selects_only_unverified_pool(self):
        pending_verified = {}
        pending_unverified = {}

        classification, target = _select(
            {'id': 'scrollfiesta-hint', 'classification': 'unverified'},
            pending_verified=pending_verified,
            pending_unverified=pending_unverified)

        self.assertEqual(classification, 'unverified')
        self.assertIs(target, pending_unverified)
        self.assertIsNot(target, pending_verified)

    def test_fitter_rejects_unknown_explicit_classification(self):
        for classification in (None, '', 'trusted', 'UNVERIFIED', 1):
            with self.subTest(classification=classification):
                with self.assertRaisesRegex(RuntimeError, 'invalid classification'):
                    _select({'id': 'hint', 'classification': classification})

    def test_duplicate_identifier_is_rejected_across_every_trust_pool(self):
        for pool_name in (
                'verified', 'unverified',
                'pending_verified', 'pending_unverified'):
            with self.subTest(pool_name=pool_name):
                pools = {
                    'verified': {},
                    'unverified': {},
                    'pending_verified': {},
                    'pending_unverified': {},
                }
                pools[pool_name]['shared-id'] = object()
                with self.assertRaisesRegex(
                        RuntimeError, 'already part of this session'):
                    _select(
                        {'id': 'shared-id', 'classification': 'unverified'},
                        **pools)

    def test_filtered_source_identifier_remains_reserved(self):
        with self.assertRaisesRegex(
                RuntimeError, 'already part of this session'):
            _select(
                {'id': 'outside-roi', 'classification': 'unverified'},
                verified={'outside-roi'})

    def test_new_point_collections_extend_the_trusted_exclusion_cloud(self):
        collections = {
            1: {'points': {
                '0': {'p': [10.0, 20.0, 30.0]},
                '1': {'p': [11.0, 21.0, 31.0]},
                '2': {'p': [12.0, 22.0, 99.0]},
                '3': {'p': [float('nan'), 23.0, 32.0]},
            }},
            2: {'points': {}},
        }

        points = collect_trusted_collection_points(collections, 25, 40)

        np.testing.assert_array_equal(
            points,
            np.array([[30.0, 20.0, 10.0],
                      [31.0, 21.0, 11.0]], dtype=np.float32))

    def test_malformed_trusted_collection_point_is_rejected(self):
        collections = {1: {'points': {'0': {'p': [1.0, 2.0]}}}}
        with self.assertRaisesRegex(RuntimeError, 'x-y-z triples'):
            collect_trusted_collection_points(collections, 0, 10)


if __name__ == '__main__':
    unittest.main()
