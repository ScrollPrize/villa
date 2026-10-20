"""Checkpoint/control-flow tests using tiny CPU stand-ins, never the released training networks."""
import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from vesuvius.ink_detection.paris4_3d_gambiarra import runtime as rt, sampling
from vesuvius.ink_detection.paris4_3d_gambiarra.labels import Points
from vesuvius.ink_detection.paris4_3d_gambiarra import train_teacher as T
from vesuvius.ink_detection.paris4_3d_gambiarra import train_student as S


from ._scribbles import synthetic_scribbles


class FakeStore:
    zarray = dict(shape=[8, 8, 8], chunks=[128] * 3)
    opened = closed = 0
    def __init__(self, source, *args, **kwargs):
        self.source = str(source)
        FakeStore.opened += 1
    def identity(self):
        return dict(source=self.source, zarray=copy.deepcopy(FakeStore.zarray))
    def close(self):
        FakeStore.closed += 1


class FakeData:
    def __init__(self, store, *, points_json, **kwargs):
        p = Points(points_json)
        self.y, self.train, self.heldout = p.y, p.train, p.heldout
        self.q, self.parents = p.q, p.parents
        self.eval_parents = self.parents[self.heldout]


class FakeTeacher(nn.Module):
    def __init__(self, backbone=None, residual=True, checkpoint_mode=None, seed=27):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.backbone = nn.Linear(1, 1)
            self.head = nn.Linear(1, 1)
            self.residual = nn.Linear(1, 1) if residual else None
    def trainable_state(self):
        return {n: p.detach().clone() for n, p in self.named_parameters()}
    def load_trainable(self, state):
        self.load_state_dict(state, strict=True)
    def targets(self, x, sites):
        return [torch.zeros(len(q)) for q in sites]


class FakeStudent(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.network = nn.Linear(1, 1)


def draw(rng, deck, *args):
    return dict(anchors=deck.take(), plans=[], random=float(rng.random()))


def teacher_loss(model, x, packet, labels):
    value = sum(p.square().sum() for p in model.parameters())
    return value, dict(bce=value.detach(), weak_bce=value.detach() * 0, weak_count=0, unique=[0, 1])


def never(*args, **kwargs):
    raise AssertionError('a terminal resume must not build training data or run an update')


def curve_steps(out):
    return [json.loads(line)['step'] for line in (out / 'curve.jsonl').read_text().splitlines()]


def student_update(model, opt, *args):
    opt.zero_grad(set_to_none=True)
    value = sum(p.square().sum() for p in model.parameters())
    value.backward()
    opt.step()
    return dict(loss=float(value.detach()))


class Checkpoints(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        self.addCleanup(torch.use_deterministic_algorithms, torch.are_deterministic_algorithms_enabled(),
                        warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
        tf32, benchmark = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.benchmark
        self.addCleanup(setattr, torch.backends.cuda.matmul, 'allow_tf32', tf32)
        self.addCleanup(setattr, torch.backends.cudnn, 'benchmark', benchmark)
        # These stand-ins are CPU-only, including configure_torch's seeding calls.
        self.stack.enter_context(patch('torch.cuda.manual_seed_all'))
        self.tmp = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.scribbles = self.tmp / 'scribbles.json'
        self.scribbles.write_text(json.dumps(synthetic_scribbles()))
        self.stack.enter_context(patch('torch.cuda.get_rng_state', side_effect=torch.get_rng_state))
        self.stack.enter_context(patch('torch.cuda.set_rng_state', side_effect=torch.set_rng_state))
        for module in (T, S):
            for name, value in [('ChunkStore', FakeStore), ('ClickData', FakeData), ('Teacher', FakeTeacher),
                                ('load_backbone', lambda path: None)]:
                self.stack.enter_context(patch.object(module, name, value))
        self.stack.enter_context(patch.object(sampling, 'draw_cube', draw))
        self.stack.enter_context(patch.object(sampling, 'draw_rotated', draw))

    def args(self, **kw):
        return SimpleNamespace(ct='fake', ssl_checkpoint='fake', cache_gib=0, chunk_cache_gib=0,
                               resume=None, init=None, stop_at=None, scribbles=self.scribbles, **kw)

    def teacher_setup(self):
        self.stack.enter_context(patch.object(T, 'Loader', lambda *args: lambda plan, step: dict(plan=plan)))
        self.stack.enter_context(patch.object(T, 'make_views', lambda *args: (None, None)))
        self.stack.enter_context(patch.object(T, 'linear_loss', teacher_loss))
        self.stack.enter_context(patch.object(T, 'decoder_loss', teacher_loss))
        self.stack.enter_context(patch.object(T, 'aug_row', lambda *args: {}))

    def check_teacher_stage_transitions(self, original=False):
        self.teacher_setup()
        # Only one stand-in update per stage, placed at the real save/transition boundary.
        specs = copy.deepcopy(T.STAGES)
        for name in specs:
            specs[name]['start'] = specs[name]['end'] - 1
        self.stack.enter_context(patch.object(T, 'STAGES', specs))
        previous = None
        for stage in specs:
            args = self.args(stage=stage, out=self.tmp / stage)
            args.init = previous
            T.run(args)
            previous = args.out / f'step-{specs[stage]["end"]:06d}.pth'
            state = rt.load(previous)
            self.assertEqual(state['stage'], stage)
            started = json.loads((args.out / 'STARTED.json').read_text())
            self.assertEqual((started['train_points'], started['heldout_points']), (8, 0))
            self.assertEqual(state['labels'], 'all')
            self.assertIsNotNone(state['pending_plan']) if stage == 'linear20k' else self.assertIsNone(state['pending_plan'])
            if stage == 'all_aug':
                self.assertEqual({int(v['step']) for v in state['optimizer']['state'].values()}, {3})
            if original:
                state.pop('stage')
                state['fold'] = 2
                if stage == 'linear20k':
                    state['head_type'] = 'linear'
                    state['delta'] = state.pop('trainable')
                else:
                    state['arm'] = 'all_aug' if stage == 'all_aug' else 'scribble'
                    state['trainable'] = {('native.' + k if k.startswith(('head.', 'backbone.')) else k): v
                                          for k, v in state['trainable'].items()}
                rt.save(previous, state)

    def test_teacher_all_stage_transitions(self):
        self.check_teacher_stage_transitions()

    def test_teacher_original_checkpoint_transitions(self):
        self.check_teacher_stage_transitions(original=True)

    def test_teacher_pending_plan_resume_matches_uninterrupted(self):
        self.teacher_setup()
        specs = copy.deepcopy(T.STAGES)
        specs['linear20k'].update(start=0, end=3, saves=(1, 3))
        self.stack.enter_context(patch.object(T, 'STAGES', specs))
        args = self.args(stage='linear20k', out=self.tmp / 'teacher')
        T.run(args)
        expected = rt.load(args.out / 'step-000003.pth')
        with self.assertRaises(FileExistsError):
            T.run(args)
        args.resume = args.out / 'step-000003.pth'
        args.stop_at = 2
        curve = (args.out / 'curve.jsonl').read_bytes()
        with self.assertRaisesRegex(ValueError, 'nothing to do'):
            T.run(args)
        args.resume = args.out / 'step-000001.pth'
        args.stop_at = 1  # the checkpoint is the stopping update: nothing to train, the later progress is kept
        T.run(args)
        self.assertEqual((args.out / 'curve.jsonl').read_bytes(), curve)
        self.assertTrue((args.out / 'COMPLETE.json').exists())
        args.stop_at = 2
        T.run(args)
        self.assertFalse((args.out / 'COMPLETE.json').exists())
        args.resume = args.out / 'step-000002.pth'
        args.stop_at = None
        T.run(args)
        actual = rt.load(args.out / 'step-000003.pth')
        for key in ('numpy_rng', 'decks', 'pending_plan'):
            self.assertEqual(actual[key], expected[key])
        for key in expected['trainable']:
            self.assertTrue(torch.equal(actual['trainable'][key], expected['trainable'][key]))
        self.assertEqual([__import__('json').loads(s)['step'] for s in (args.out / 'curve.jsonl').read_text().splitlines()], [1, 2, 3])

    def student_setup(self):
        self.stack.enter_context(patch.object(S, 'Student', FakeStudent))
        self.stack.enter_context(patch.object(S, 'update', student_update))
        self.stack.enter_context(patch.object(S, 'save_weights', lambda network, path: rt.save(path, network.state_dict())))
        self.stack.enter_context(patch.object(S.views, 'cube_views', lambda *a: torch.zeros(1)))
        self.stack.enter_context(patch.object(S, 'Loader', lambda *a: lambda plan, uplan: dict(
            plan=plan, uplan=uplan, crops=[], norm=[], ucrops=[], unorm=[], sites=[], usites=[])))
        self.stack.enter_context(patch.object(sampling, 'draw_unlabelled', lambda rng, deck, origins: dict(
            plans=[], index=deck.take(), random=float(rng.random()))))

    def test_student_save_resume_and_stage2_continued_pool(self):
        self.student_setup()
        specs = copy.deepcopy(S.STAGES)
        specs[1].update(start=19999)
        specs[2].update(end=20001)
        self.stack.enter_context(patch.object(S, 'STAGES', specs))
        checkpoint = self.tmp / 'teacher.pth'
        rt.save(checkpoint, dict(stage='scribble10k', step=10000, trainable=FakeTeacher().trainable_state()))
        args = self.args(stage=1, out=self.tmp / 'student1', teacher=checkpoint, pool=S.POOL_CSV)
        S.run(args)
        state_path = args.out / 'train-state-020000.pth'
        state = rt.load(state_path)
        self.assertTrue((args.out / 'step-020000.safetensors').exists())
        urng = np.random.default_rng()
        rt.set_rng_state(urng, state['unlabelled_rng'])
        pool = S.read_pool()
        udeck = sampling.PoolDeck([r['index'] for r in pool if r['split'] == 'train'], urng)
        udeck.left = copy.deepcopy(state['unlabelled_deck'])
        expected = sampling.draw_unlabelled(urng, udeck, {})
        rt.save(checkpoint, dict(stage='all_aug', step=5000, trainable=FakeTeacher().trainable_state()))
        args = self.args(stage=2, out=self.tmp / 'student2', teacher=checkpoint, pool=S.POOL_CSV)
        args.init = state_path
        S.run(args)
        actual = rt.load(args.out / 'train-state-020001.pth')
        self.assertEqual(actual['pending_unlabelled'], expected)
        for out in (self.tmp / 'student1', self.tmp / 'student2'):
            self.assertEqual(json.loads((out / 'STARTED.json').read_text())['train_points'], 8)
        self.assertEqual({int(v['step']) for v in actual['optimizer']['state'].values()}, {2})

    def test_student_pending_plans_resume_matches_uninterrupted(self):
        self.student_setup()
        specs = copy.deepcopy(S.STAGES)
        specs[1].update(end=3, saves=(1, 3))
        self.stack.enter_context(patch.object(S, 'STAGES', specs))
        checkpoint = self.tmp / 'teacher.pth'
        rt.save(checkpoint, dict(stage='scribble10k', step=10000, trainable=FakeTeacher().trainable_state()))
        args = self.args(stage=1, out=self.tmp / 'student', teacher=checkpoint, pool=S.POOL_CSV)
        S.run(args)
        expected = rt.load(args.out / 'train-state-000003.pth')
        with self.assertRaises(FileExistsError):
            S.run(args)
        args.resume, args.stop_at = args.out / 'train-state-000003.pth', 2
        curve = (args.out / 'curve.jsonl').read_bytes()
        with self.assertRaisesRegex(ValueError, 'nothing to do'):
            S.run(args)
        args.resume, args.stop_at = args.out / 'train-state-000001.pth', 1
        S.run(args)
        self.assertEqual((args.out / 'curve.jsonl').read_bytes(), curve)
        self.assertTrue((args.out / 'COMPLETE.json').exists())
        args.stop_at = 2
        S.run(args)
        self.assertFalse((args.out / 'COMPLETE.json').exists())
        args.resume, args.stop_at = args.out / 'train-state-000002.pth', None
        S.run(args)
        actual = rt.load(args.out / 'train-state-000003.pth')
        for key in ('numpy_rng', 'decks', 'pending_plan', 'unlabelled_rng', 'unlabelled_deck', 'pending_unlabelled'):
            self.assertEqual(actual[key], expected[key])
        for key in expected['network']:
            self.assertTrue(torch.equal(actual['network'][key], expected['network'][key]))
        for key, state in expected['optimizer']['state'].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(actual['optimizer']['state'][key][name], value))
        self.assertEqual([__import__('json').loads(s)['step'] for s in (args.out / 'curve.jsonl').read_text().splitlines()], [1, 2, 3])
        self.assertTrue((args.out / 'COMPLETE.json').exists())

    def teacher_run(self):
        """linear20k cut to updates 1-3 (saves 1 and 3); returns its args."""
        self.teacher_setup()
        specs = copy.deepcopy(T.STAGES)
        specs['linear20k'].update(start=0, end=3, saves=(1, 3))
        self.stack.enter_context(patch.object(T, 'STAGES', specs))
        args = self.args(stage='linear20k', out=self.tmp / 'teacher')
        T.run(args)
        return args

    def student_run(self):
        """Student stage 1 cut to updates 1-3 (saves 1 and 3); returns its args."""
        self.student_setup()
        specs = copy.deepcopy(S.STAGES)
        specs[1].update(end=3, saves=(1, 3))
        self.stack.enter_context(patch.object(S, 'STAGES', specs))
        checkpoint = self.tmp / 'teacher.pth'
        rt.save(checkpoint, dict(stage='scribble10k', step=10000, trainable=FakeTeacher().trainable_state()))
        args = self.args(stage=1, out=self.tmp / 'student', teacher=checkpoint, pool=S.POOL_CSV)
        S.run(args)
        return args

    def changed_file(self, path, name):
        changed = self.tmp / name
        changed.write_bytes(Path(path).read_bytes() + b'\n')
        return changed

    def assert_refused(self, module, args, pattern):
        """The resume fails before touching the run directory, and the CT store is closed."""
        before = {p: p.read_bytes() for p in args.out.rglob('*') if p.is_file()}
        opened, closed = FakeStore.opened, FakeStore.closed
        with self.assertRaisesRegex(ValueError, pattern):
            module.run(args)
        self.assertEqual((FakeStore.opened - opened, FakeStore.closed - closed), (1, 1))
        self.assertEqual({p: p.read_bytes() for p in args.out.rglob('*') if p.is_file()}, before)

    def test_teacher_resume_refuses_changed_inputs(self):
        args = self.teacher_run()
        args.resume = args.out / 'step-000001.pth'
        self.assertEqual(rt.load(args.resume)['inputs']['ct'], dict(source='fake', zarray=FakeStore.zarray))
        args.ct = 'other'
        self.assert_refused(T, args, r"different inputs.*'ct'")
        args.ct = 'fake'
        with patch.object(FakeStore, 'zarray', dict(FakeStore.zarray, shape=[8, 8, 9])):
            self.assert_refused(T, args, r"different inputs.*'ct'")
        original = args.scribbles
        args.scribbles = self.changed_file(original, 'points.json')
        self.assert_refused(T, args, r"different inputs.*'points_sha256'")
        args.scribbles = original
        legacy = rt.load(args.resume)
        del legacy['inputs']
        args.resume = self.tmp / 'legacy.pth'
        rt.save(args.resume, legacy)
        self.assert_refused(T, args, 'records no input identities')
        args.resume = args.out / 'step-000001.pth'  # unchanged inputs: resumes
        T.run(args)
        self.assertEqual(curve_steps(args.out), [1, 2, 3])

    def test_student_resume_refuses_changed_inputs(self):
        args = self.student_run()
        args.resume = args.out / 'train-state-000001.pth'
        for name, value, key in (('pool', self.changed_file(S.POOL_CSV, 'pool.csv'), 'pool_sha256'),
                                 ('teacher', self.changed_file(args.teacher, 'other-teacher.pth'), 'teacher_sha256'),
                                 ('ct', 'other', 'ct')):
            original = getattr(args, name)
            setattr(args, name, value)
            self.assert_refused(S, args, rf"different inputs.*'{key}'")
            setattr(args, name, original)
        original = args.scribbles
        args.scribbles = self.changed_file(original, 'points.json')
        self.assert_refused(S, args, r"different inputs.*'points_sha256'")
        args.scribbles = original
        S.run(args)
        self.assertEqual(curve_steps(args.out), [1, 2, 3])

    def test_teacher_terminal_resume_writes_completion_without_update(self):
        args = self.teacher_run()
        final = args.out / 'step-000003.pth'
        files = {p: p.read_bytes() for p in args.out.rglob('*') if p.is_file() and p.name != 'COMPLETE.json'}
        (args.out / 'COMPLETE.json').unlink()
        args.resume = final
        with patch.object(T, 'ClickData', never), patch.object(T, 'Teacher', never), patch.object(T, 'linear_loss', never):
            T.run(args)
        complete = json.loads((args.out / 'COMPLETE.json').read_text())
        self.assertEqual((complete['step'], complete['checkpoint'], complete['repaired']), (3, str(final), True))
        self.assertEqual(json.loads((args.out / 'RESUMED.json').read_text())['updates'], 0)
        for path, data in files.items():
            if path.name != 'RESUMED.json':
                self.assertEqual(path.read_bytes(), data, path)

    def test_student_terminal_resume_rewrites_export_and_completion_without_update(self):
        args = self.student_run()
        final, export = args.out / 'train-state-000003.pth', args.out / 'step-000003.safetensors'
        state, weights, curve = final.read_bytes(), export.read_bytes(), (args.out / 'curve.jsonl').read_bytes()
        expected = json.loads((args.out / 'COMPLETE.json').read_text())
        (args.out / 'COMPLETE.json').unlink()
        export.write_bytes(b'\0' * len(weights))  # a damaged export of the same size
        (args.out / 'step-000001.safetensors').unlink()
        args.resume = final
        with patch.object(S, 'ClickData', never), patch.object(S, 'Teacher', never), patch.object(S, 'update', never):
            S.run(args)
            self.assertEqual(export.read_bytes(), weights)
            complete = json.loads((args.out / 'COMPLETE.json').read_text())
            self.assertEqual((complete['step'], complete['checkpoint'], complete['repaired']), (3, str(final), True))
            self.assertEqual(complete['export_sha256'], expected['export_sha256'])
            self.assertEqual((final.read_bytes(), (args.out / 'curve.jsonl').read_bytes()), (state, curve))
            self.assertFalse((args.out / 'step-000001.safetensors').exists())  # only the resumed step is exported
        # A resume that continues training first restores the export of its own checkpoint.
        args.resume = args.out / 'train-state-000001.pth'
        S.run(args)
        self.assertTrue((args.out / 'step-000001.safetensors').exists())
        self.assertEqual(export.read_bytes(), weights)
        self.assertEqual(curve_steps(args.out), [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
