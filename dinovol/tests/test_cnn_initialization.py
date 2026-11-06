import copy
from pathlib import Path
import tempfile
import unittest

import torch

from dinovol_2.model.model import DinoVitStudentTeacher
from dinovol_2.training.config import normalize_config, dino_model_config
from dinovol_2.training.trainer import Trainer
from test_training import tiny_config


CNN_PREFIXES = ('backbone.down_projection.stem.', 'backbone.down_projection.stages.')


class CnnInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def configs(self, folder):
        source = tiny_config(folder)
        source['model'].update(embedding_type='deeper',
            deeper_embed_stage_channels=[4, 8, 12], deeper_embed_stage_strides=[1, 2, 2],
            deeper_embed_depth_per_level=[1, 1, 1], deeper_embed_stem_channels=4,
            deeper_embed_stem_type='conv')
        target = copy.deepcopy(source)
        target.update(resume=False, auto_resume=False)
        target['model'].update(embed_dim=48, num_heads=4, patch_size=[8, 4, 4],
            local_crops_size=[8, 4, 4], deeper_embed_stage_strides=[[2, 1, 1], 2, 2])
        target['dataset']['local_crop_size'] = [8, 4, 4]
        return source, target

    def test_teacher_cnn_only_with_changed_width_and_stride(self):
        with tempfile.TemporaryDirectory() as folder:
            source_config, target_config = self.configs(folder)
            source = DinoVitStudentTeacher(dino_model_config(normalize_config(source_config)))
            with torch.no_grad():
                for parameter in source.teacher.parameters():
                    parameter.fill_(0.125)
                for parameter in source.student.parameters():
                    parameter.fill_(-0.25)
            teacher = source.teacher.state_dict()
            for version in ('legacy', 'v2'):
                with self.subTest(version=version):
                    path = Path(folder) / f'{version}.pt'
                    branches = dict(teacher=teacher, student=source.student.state_dict())
                    checkpoint = (dict(step=615000, **branches) if version == 'legacy' else
                                  dict(format_version=2, step=615000,
                                       method=dict(name='dino', state=branches),
                                       runtime_states=[{'rng_state': Trainer._capture_rng_state()}]))
                    torch.save(checkpoint, path)
                    torch.manual_seed(42)
                    baseline = Trainer(target_config)
                    initialized_config = copy.deepcopy(target_config)
                    initialized_config['method'] = {'pretrained_cnn_weights': str(path)}
                    torch.manual_seed(42)
                    initialized = Trainer(initialized_config)
                    baseline_state = baseline.method.model_module.student.state_dict()
                    for name, tensor in initialized.method.model_module.student.state_dict().items():
                        expected = teacher[name] if name.startswith(CNN_PREFIXES) else baseline_state[name]
                        torch.testing.assert_close(tensor, expected, rtol=0, atol=0, msg=name)
                        torch.testing.assert_close(tensor, initialized.method.model_module.teacher.state_dict()[name],
                                                   rtol=0, atol=0, msg=name)
                    self.assertTrue(all(not p.requires_grad for p in initialized.method.model_module.teacher.parameters()))
                    self.assertFalse(initialized.optimizer.state)
                    self.assertIsNone(initialized.resume_path)
                    self.assertEqual(initialized.warmup_steps, baseline.warmup_steps)

    def test_incompatible_cnn_is_rejected_before_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            source_config, target_config = self.configs(folder)
            source = DinoVitStudentTeacher(dino_model_config(normalize_config(source_config)))
            target = DinoVitStudentTeacher(dino_model_config(normalize_config(target_config)))
            before = copy.deepcopy(target.student.state_dict())
            teacher = source.teacher.state_dict()
            name = next(name for name in teacher if name.startswith(CNN_PREFIXES))
            for invalid in ('missing', 'shape'):
                with self.subTest(invalid=invalid):
                    broken = dict(teacher)
                    if invalid == 'missing':
                        del broken[name]
                    else:
                        broken[name] = torch.zeros(1)
                    path = Path(folder) / 'broken.pt'
                    torch.save({'teacher': broken}, path)
                    with self.assertRaisesRegex(RuntimeError, 'Incompatible pretrained CNN'):
                        target.load_pretrained_cnn_weights(str(path))
                    for key, tensor in target.student.state_dict().items():
                        torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)

    def test_conflicting_initializers_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            _, config = self.configs(folder)
            model_config = dino_model_config(normalize_config(config))
            model_config.update(pretrained_weights='full.pt', pretrained_cnn_weights='cnn.pt')
            with self.assertRaisesRegex(ValueError, 'only one'):
                DinoVitStudentTeacher(model_config)
