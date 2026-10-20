"""Training boundaries: accumulation, method state, and deterministic resume."""
import copy
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dinovol_2.methods.base import StepOutput, TrainingMethod
from dinovol_2.pretrain import DinoIBOTPretrainer
from dinovol_2.training.trainer import Trainer
from dinovol_2.ops.collate import build_dino_ibot_collate_fn


def tiny_config(directory, device='cpu'):
    return dict(device=device, use_amp=False, max_iterations=5, output_dir=str(directory),
        freeze_last_layer_steps=0, warmup_steps=0, lr=1e-4, lr_schedule='constant',
        model=dict(model_type='v3', embed_dim=36, depth=1, num_heads=3,
            global_crops_size=[8]*3, local_crops_size=[4]*3, patch_size=[4]*3,
            drop_path_rate=0., rope_rescale_coords=None,
            dino_out_dim=16, ibot_out_dim=16, dino_head_hidden_dim=16,
            dino_head_bottleneck_dim=8, ibot_head_hidden_dim=16, ibot_head_bottleneck_dim=8),
        dataset=dict(global_crop_size=[8]*3, local_crop_size=[4]*3, num_local_crops=2))


def tiny_batch(local=2):
    collate = build_dino_ibot_collate_fn(dict(global_crop_size=[8]*3, patch_size=[4]*3))
    return collate([dict(global_views=[torch.randn(1,8,8,8) for _ in range(2)],
        local_views=[torch.randn(1,4,4,4) for _ in range(local)]) for _ in range(2)])


class RegressionMethod(TrainingMethod):
    """A teacher-free method proves the trainer has no DINO assumptions."""
    name = 'regression_test'

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 2)
        self.updates = 0

    def forward(self, batch, step):
        loss = self.encoder(batch).square().mean()
        return StepOutput(loss, {'loss': loss.detach()})

    def parameter_groups(self, **kwargs):
        return [{'params': self.encoder.parameters()}]

    def after_optimizer_step(self, step):
        self.updates += 1

    def evaluation_backbone(self):
        return self.encoder


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        random.seed(15); np.random.seed(15); torch.manual_seed(15)

    def test_teacher_free_method_and_accumulation(self):
        with tempfile.TemporaryDirectory() as folder:
            first = RegressionMethod()
            second = copy.deepcopy(first)
            config = dict(device='cpu', use_amp=False, max_iterations=3, warmup_steps=0,
                          output_dir=folder, clip_grad=3.)
            accumulated = Trainer(config, method=first)
            combined = Trainer(config, method=second)
            a, b = torch.randn(2,3), torch.randn(2,3)
            accumulated.train_step([a,b], 0)
            combined.train_step(torch.cat([a,b]), 0)
            self.assertEqual(first.updates, 1)
            for x,y in zip(first.parameters(), second.parameters()):
                torch.testing.assert_close(x,y)

    def test_dino_checkpoint_and_next_update(self):
        with tempfile.TemporaryDirectory() as folder:
            config = tiny_config(folder)
            trainer = DinoIBOTPretrainer(config)
            batch = tiny_batch()
            trainer.train_step(batch, 0)
            path = trainer.save_checkpoint(0)
            resumed = DinoIBOTPretrainer(config)
            self.assertEqual(resumed.load_checkpoint(path), 0)
            for key, value in trainer.method.checkpoint_state()['student'].items():
                torch.testing.assert_close(value, resumed.method.checkpoint_state()['student'][key], rtol=0, atol=0)
            # Fix forward RNG independently of model construction.
            state = trainer._capture_rng_state()
            expected = trainer.train_step(batch, 1)
            trainer._restore_rng_state(state)
            actual = resumed.train_step(batch, 1)
            self.assertEqual(expected, actual)
            for a,b in zip(trainer.model_module.parameters(), resumed.model_module.parameters()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_dino_diagnostics_and_validation_do_not_update_teacher(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = DinoIBOTPretrainer(tiny_config(folder))
            batch = tiny_batch()
            report = trainer.verify_train_step(batch)
            self.assertTrue(report['train_step']['checks']['all_passed'], report)
            state = copy.deepcopy(trainer.method.checkpoint_state())
            trainer.validate(batch, 0)
            for key in ('student','teacher','dino_loss','ibot_patch_loss'):
                for name, value in state[key].items():
                    torch.testing.assert_close(value, trainer.method.checkpoint_state()[key][name],rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
