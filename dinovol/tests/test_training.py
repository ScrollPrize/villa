"""Training boundaries: accumulation, method state, and deterministic resume."""
import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_runtime_num_threads_applies_before_model_setup(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        with tempfile.TemporaryDirectory() as folder:
            config = dict(runtime=dict(device='cpu', use_amp=False, output_dir=folder,
                                       max_iterations=1, num_threads=1))

            def build(*args, **kwargs):
                self.assertEqual(torch.get_num_threads(), 1)
                return RegressionMethod()

            with patch('dinovol_2.training.trainer.build_method', side_effect=build):
                trainer = Trainer(config)
            self.assertEqual(trainer.runtime_config['num_threads'], 1)
            self.assertTrue(np.isfinite(trainer.train_step(torch.randn(2, 3), 0)['loss']))

    def test_runtime_num_threads_unset_preserves_current_setting(self):
        with tempfile.TemporaryDirectory() as folder:
            for setting in ({}, {'num_threads': None}):
                with self.subTest(setting=setting), patch('torch.set_num_threads') as set_threads:
                    Trainer(dict(device='cpu', output_dir=folder, max_iterations=1,
                                 runtime=setting), method=RegressionMethod())
                    set_threads.assert_not_called()

    def test_runtime_num_threads_rejects_invalid_values_before_setup(self):
        for value in (0, -1, 1.5, True, '2'):
            with self.subTest(value=value), patch('dinovol_2.training.trainer.build_method') as build:
                with self.assertRaisesRegex(ValueError, 'runtime.num_threads'):
                    Trainer({'runtime': {'num_threads': value}})
                build.assert_not_called()

    def test_monitor_subset_preserves_device_and_view_order(self):
        devices = ['cpu', *[f'cuda:{i}' for i in range(torch.cuda.device_count())]]
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(dict(device='cpu', output_dir=folder, max_iterations=1,
                                   monitor_batch_size=5), method=RegressionMethod())
            for device in devices:
                for batch_size, expected_rows in ((7, [0, 1, 2, 3, 4, 7, 8, 9, 10, 11]),
                                                  (3, [0, 1, 2, 3, 4, 5])):
                    with self.subTest(device=device, batch_size=batch_size):
                        crops = torch.arange(2 * batch_size, device=device, dtype=torch.float32)
                        crops = crops[:, None, None, None, None].expand(-1, 1, 2, 3, 4).clone()
                        subset = trainer._monitor_subset(dict(collated_global_crops=crops,
                                                              batch_size=batch_size, n_global_views=2))
                        self.assertEqual(subset['collated_global_crops'].device, crops.device)
                        self.assertEqual(subset['batch_size'], min(batch_size, 5))
                        self.assertEqual(subset['n_global_views'], 2)
                        torch.testing.assert_close(subset['collated_global_crops'].cpu(),
                                                   crops.cpu()[expected_rows], rtol=0, atol=0)

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

    def test_teacher_free_fit_with_validation_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            config = dict(device='cpu', use_amp=False, max_iterations=3, warmup_steps=0,
                          output_dir=folder, val_every_n=1, save_every_n=1)
            method = RegressionMethod()
            trainer = Trainer(config, method=method)
            loader = torch.utils.data.DataLoader(torch.randn(4,3), batch_size=2)
            with patch.object(trainer, 'build_dataloader', return_value=loader), \
                 patch.object(trainer, 'build_val_dataloader', return_value=loader):
                trainer.fit()
            self.assertEqual(method.updates, 3)
            self.assertTrue((Path(folder)/'checkpoint_step_000002.pt').exists())

    def test_validation_policy_with_cross_view_and_fused_training(self):
        with tempfile.TemporaryDirectory() as folder:
            config = tiny_config(folder)
            config.update(ibot_targets='cross_view', ibot_fused_last_layer=True)
            trainer = DinoIBOTPretrainer(config)
            batch = tiny_batch(local=0)
            batch['global_token_targets'] = torch.zeros(4,8,3)
            trainer.train_step(batch,0)
            with patch.object(trainer.method, '_cross_view_ibot_selection', side_effect=AssertionError('training only')), \
                 patch.object(trainer.method.ibot_patch_loss, 'forward_masked_from_features', side_effect=AssertionError('training only')):
                metrics = trainer.validate(batch,0)
            self.assertNotIn('object_loss',metrics)
            self.assertEqual(metrics['dino_local_loss'],0.)
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_object_supervision_can_request_teacher_without_dino_ibot(self):
        with tempfile.TemporaryDirectory() as folder:
            config = tiny_config(folder)
            config['model'].update(embedding_type='deeper',deeper_embed_stage_channels=[4,8],
                deeper_embed_stage_strides=[2,2],deeper_embed_depth_per_level=[1,1])
            config['object_losses'] = dict(enabled=True,head=dict(hidden_dim=8,out_dim=8,ink_dim=4,fpn_dim=4))
            config.update(dino_loss_weight=0.,ibot_loss_weight=0.)
            trainer = Trainer(config)
            batch = tiny_batch()
            batch.update(object_point_coordinates=torch.rand(4,4,3)*6,
                object_point_valid=torch.ones(4,4,dtype=torch.bool),
                object_point_objects=torch.tensor([[1,1,2,2]]*2),
                object_point_ink=torch.tensor([[1,1,-1,-1]]*2),
                object_negatives=torch.tensor([[[False,False,False],[False,False,True],[False,True,False]]]*2))
            metrics = trainer.train_step(batch,0)
            self.assertGreater(metrics['object_loss'],0.)
            self.assertEqual(metrics['dino_global_loss'],0.)
            self.assertEqual(metrics['ibot_loss'],0.)
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

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
