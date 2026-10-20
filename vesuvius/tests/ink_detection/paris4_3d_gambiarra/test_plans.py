"""All-label plan streams and final-stage geometry/KD regression values."""
import hashlib
import json
import math
import unittest
from pathlib import Path

import numpy as np

from vesuvius.ink_detection.paris4_3d_gambiarra import rotation_aug, sampling
from vesuvius.ink_detection.paris4_3d_gambiarra.data import ClickData
from vesuvius.ink_detection.paris4_3d_gambiarra.labels import read_pool

from ._scribbles import RELEASE_JSON

REF = json.loads((Path(__file__).parent / 'fixtures' / 'reference.json').read_text())


def assert_same_plan(case, plan, ref):
    case.assertEqual(plan.get('anchors'), ref.get('anchors'))
    case.assertEqual(len(plan['plans']), len(ref['plans']))
    for a, b in zip(plan['plans'], ref['plans']):
        ga, gb = a['photo']['gamma'], b['photo']['gamma']
        case.assertLessEqual(abs(ga - gb), math.ulp(gb))
        case.assertEqual(dict(a, photo=dict(a['photo'], gamma=0)), dict(b, photo=dict(b['photo'], gamma=0)))


def kd_counts(data, plan):
    return [int(sampling.lattice(item, data.mask_clicks(item, data.transform(item)))[1].sum()) for item in plan['plans']]


class Data:
    all = None

    @classmethod
    def get(cls):
        if RELEASE_JSON is None:
            raise unittest.SkipTest("set PARIS4_INK_SCRIBBLES for recorded-plan checks")
        if cls.all is None:
            cls.all = ClickData(None, weak=True, points_json=RELEASE_JSON)
        return cls.all


class TeacherStreams(unittest.TestCase):
    def test_all_label_plans(self):
        d = Data.get()
        self.assertEqual((len(d.train), len(d.heldout)), (9627, 0))
        for draw in (sampling.draw_cube, sampling.draw_rotated):
            rng = np.random.default_rng(2027)
            deck = sampling.ClassDeck(d.train, d.y, rng)
            plans = [draw(rng, deck, d.q, d.parents, d.eval_parents) for _ in range(3)]
            replay_rng = np.random.default_rng(2027)
            replay_deck = sampling.ClassDeck(d.train, d.y, replay_rng)
            for plan in plans:
                self.assertEqual(plan, draw(replay_rng, replay_deck, d.q, d.parents, d.eval_parents))
                anchors = np.asarray(plan['anchors'])[::2]
                self.assertEqual(d.y[anchors].tolist(), [0] * 4 + [1] * 4)
                for item in plan['plans']:
                    ids, coordinates = d.clicks(item, d.transform(item))
                    self.assertIn(item['anchor'], ids)
                    self.assertTrue(((coordinates >= 0) & (coordinates < 128)).all())

    def test_all_aug_update1(self):
        d = Data.get()
        self.assertEqual(d.weak_audit, REF['all_tubes_audit_port'])  # regression value (no original record)
        geo = rotation_aug.geometry(d, REF['all_aug_plan1'], 1)
        expected = REF['all_aug_step1']
        counts = dict(labels=sum(q['labels'] for q in geo), weak_labels=sum(q['weak'] for q in geo),
                      plan_labels=sum(q['plan_labels'] for q in geo), tube_plan=sum(q['tube_plan'] for q in geo),
                      tube_rotated=sum(q['tube_rotated'] for q in geo), zoom_on=sum(q['zoom'] != 1. for q in geo),
                      unique_points=len(set(np.concatenate([q['ids'] for q in geo]).tolist())),
                      weak_count=len(set(np.concatenate([q['weak_ids'] for q in geo]).tolist())))
        self.assertEqual(counts, {k: expected[k] for k in counts})
        self.assertAlmostEqual(float(np.mean([q['angle'] for q in geo])), expected['angle_mean'], places=9)
        self.assertEqual(float(np.mean([q['attempts'] for q in geo])), expected['attempts_mean'])
        self.assertAlmostEqual(float(np.mean([q['zoom'] for q in geo])), expected['zoom_mean'], places=12)


class StudentStreams(unittest.TestCase):
    def setUp(self):
        pool = read_pool()
        self.origins = {r['index']: r['origin'] for r in pool}
        self.train = [r['index'] for r in pool if r['split'] == 'train']
        self.assertEqual(len(self.train), 16091)

    def check(self, data, rows, rng, deck, urng, udeck, skip_unlabelled=0):
        for _ in range(skip_unlabelled):
            sampling.draw_unlabelled(urng, udeck, self.origins)
        for row in rows:
            plan = sampling.draw_cube(rng, deck, data.q, data.parents, data.eval_parents)
            uplan = sampling.draw_unlabelled(urng, udeck, self.origins)
            assert_same_plan(self, plan, row['plan'])
            assert_same_plan(self, uplan, row['unlabelled_plan'])
            self.assertEqual(sampling.plan_sha(row['plan']), row['plan_sha256'])
            self.assertEqual(sampling.plan_sha(row['unlabelled_plan']), row['unlabelled_plan_sha256'])
            self.assertEqual(kd_counts(data, row['plan']), row['counts'])
            for item in row['unlabelled_plan']['plans']:
                self.assertTrue(sampling.lattice(item, np.empty((0, 3), np.float32))[1].all())

    def test_stage1_lattice_masks_all_scribbles(self):
        d = Data.get()
        rng = np.random.default_rng(2027)
        deck = sampling.ClassDeck(d.train, d.y, rng)
        plan = sampling.draw_cube(rng, deck, d.q, d.parents, d.eval_parents)
        for item in plan['plans']:
            clicks = d.mask_clicks(item, d.transform(item))
            sites, keep = sampling.lattice(item, clicks)
            # Independent distance calculation on the 32^3 lattice.
            nearest = np.full(len(sites), np.inf)
            for click in clicks:
                nearest = np.minimum(nearest, np.linalg.norm(sites.astype(np.float64) - click, axis=1))
            np.testing.assert_array_equal(keep, nearest > 1)

    def test_stage2(self):
        d = Data.get()
        rng, urng = np.random.default_rng(2026092612), np.random.default_rng(2026092302)
        # the unlabelled stream continues from stage 1: plans 1..20,000 come first
        self.check(d, REF['student2'], rng, sampling.ClassDeck(d.train, d.y, rng), urng, sampling.PoolDeck(self.train, urng),
                   skip_unlabelled=20000)


class Decks(unittest.TestCase):
    def test_class_deck(self):
        y = np.array([0] * 10 + [1] * 5)
        deck = sampling.ClassDeck(np.arange(15), y, np.random.default_rng(0))
        seen = []
        for _ in range(5):
            out = deck.take()
            self.assertEqual(len(set(out)), 8)
            self.assertTrue(all(y[i] == 0 for i in out[:4]) and all(y[i] == 1 for i in out[4:]))
            seen += out
        self.assertEqual(set(seen), set(range(15)))


if __name__ == '__main__':
    unittest.main()
