import copy
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dinovol_2.training.trainer import Trainer
from dinovol_2.training.config import normalize_config, as_legacy_config
from dinovol_2.training.checkpoint import method_state, backbone_state
from dinovol_2.eval.napari_visualizer import load_backbone_from_checkpoint
from test_training import tiny_config, tiny_batch, RegressionMethod


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_centering_rng_and_next_update_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            config = tiny_config(folder)
            config['centering'] = 'centering'
            trainer = Trainer(config)
            batch = tiny_batch()
            trainer.train_step(batch, 0)
            self.assertFalse(trainer.method.dino_loss.updated)
            trainer._train_sampler_epoch = 3
            trainer._next_monitor_seed()
            path = trainer.save_checkpoint(0)
            saved = torch.load(path, weights_only=False)
            self.assertEqual(saved['format_version'], 2)
            self.assertNotIn('student', saved)
            self.assertTrue(trainer.method.dino_loss.updated)
            rng_draws = (random.random(), np.random.rand(), torch.rand(2))
            monitor_seed = trainer._next_monitor_seed()
            expected = trainer.train_step(batch, 1)
            expected_state = copy.deepcopy(trainer.method.checkpoint_state())
            resumed = Trainer(config)
            resumed.load_checkpoint(path)
            self.assertEqual(resumed._train_sampler_epoch, 3)
            self.assertEqual(random.random(),rng_draws[0])
            self.assertEqual(np.random.rand(),rng_draws[1])
            torch.testing.assert_close(torch.rand(2),rng_draws[2],rtol=0,atol=0)
            self.assertEqual(resumed._next_monitor_seed(),monitor_seed)
            self.assertEqual(expected,resumed.train_step(batch,1))
            for key in ('student','teacher','dino_loss','ibot_patch_loss'):
                for name,value in expected_state[key].items():
                    torch.testing.assert_close(value,resumed.method.checkpoint_state()[key][name],rtol=0,atol=0)

    def test_legacy_loading_and_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            config = tiny_config(folder)
            trainer = Trainer(config)
            trainer.train_step(tiny_batch(),0)
            new_path = trainer.save_checkpoint(0)
            legacy_path = Path(folder)/'legacy.pt'
            torch.save(dict(step=0, config=as_legacy_config(trainer.config),
                **trainer.method.checkpoint_state(), optimizer=trainer.optimizer.state_dict()),legacy_path)
            for path in (new_path,legacy_path):
                resumed = Trainer(config)
                self.assertEqual(resumed.load_checkpoint(path),0)
                loaded = load_backbone_from_checkpoint(path,device=torch.device('cpu'))
                for name,value in trainer.method.evaluation_backbone().state_dict().items():
                    torch.testing.assert_close(value,loaded.backbone.state_dict()[name],rtol=0,atol=0)
                # Pretrained backbone initialization shares the versioned reader.
                resumed.method.model_module.student.backbone.load_pretrained_weights(str(path),backbone_only=True)

    def test_teacher_free_checkpoint_and_wrong_method(self):
        with tempfile.TemporaryDirectory() as folder:
            config = dict(device='cpu',use_amp=False,output_dir=folder,warmup_steps=0)
            trainer = Trainer(config,method=RegressionMethod())
            trainer.train_step(torch.randn(2,3),0)
            path = trainer.save_checkpoint(0)
            saved = torch.load(path,weights_only=False)
            state,source = backbone_state(saved)
            self.assertEqual(source,'evaluation_backbone')
            self.assertEqual(set(state),{'weight','bias'})
            restored = Trainer(config,method=RegressionMethod())
            restored.load_checkpoint(path)
            for a,b in zip(trainer.method.parameters(),restored.method.parameters()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'does not match'):
                Trainer(tiny_config(folder)).load_checkpoint(path)
            with self.assertRaisesRegex(ValueError,'Unsupported checkpoint'):
                method_state({'format_version': 999})

    def test_gram_teacher_ramp_and_ink_queues(self):
        with tempfile.TemporaryDirectory() as folder:
            config = tiny_config(folder)
            config['model'].update(embedding_type='deeper',deeper_embed_stage_channels=[4,8],
                deeper_embed_stage_strides=[2,2],deeper_embed_depth_per_level=[1,1])
            config['object_losses'] = dict(enabled=True,head=dict(hidden_dim=8,out_dim=8,ink_dim=4,fpn_dim=4))
            config['gram'] = dict(enabled=True)
            trainer = Trainer(config)
            trainer.method.object_heads_start_step = 7
            trainer.method.object_loss.ink_pos_queue = torch.randn(3,4)
            trainer.method.object_loss.ink_neg_queue = torch.randn(2,4)
            trainer.method.gram_teacher_backbone.eval()
            path = trainer.save_checkpoint(8)
            restored = Trainer(config)
            restored.load_checkpoint(path)
            self.assertEqual(restored.method.object_heads_start_step,7)
            for name in ('ink_pos_queue','ink_neg_queue'):
                torch.testing.assert_close(getattr(trainer.method.object_loss,name),getattr(restored.method.object_loss,name),rtol=0,atol=0)
            restored.method.train()
            self.assertFalse(restored.method.gram_teacher_backbone.training)
            for a,b in zip(trainer.method.gram_teacher_backbone.parameters(),restored.method.gram_teacher_backbone.parameters()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
