import copy
import json
from pathlib import Path
import tempfile
import unittest

from dinovol_2.training.config import normalize_config, as_legacy_config
from dinovol_2.model import materialize_backbone_config
from dinovol_2.training.trainer import Trainer
from test_training import tiny_config


class ConfigTests(unittest.TestCase):
    def test_current_defaults_without_model_type(self):
        config = normalize_config({})
        model = config['backbone']
        self.assertFalse(model['qkv_bias'])
        self.assertFalse(model['scale_mlp'])
        self.assertEqual(model['mlp_ratio'], 2.)
        self.assertTrue(model['untie_global_and_local_cls_norm'])
        self.assertEqual(config['optimizer']['lr_schedule'], 'constant')
        self.assertEqual(config['method']['momentum_teacher'], .994)
        self.assertNotIn('model_type', model)
        self.assertEqual(materialize_backbone_config({}),
            {k:v for k,v in materialize_backbone_config({'model_type':'v3'}).items() if k != 'model_type'})

    def test_legacy_and_sectioned_configs_build_identical_models(self):
        with tempfile.TemporaryDirectory() as folder:
            import torch
            old = tiny_config(folder)
            old['model'].pop('model_type')
            old_copy = copy.deepcopy(old)
            new = normalize_config(old)
            self.assertEqual(old, old_copy)
            self.assertEqual(new, normalize_config(new))
            self.assertEqual(new, normalize_config(as_legacy_config(new)))
            torch.manual_seed(5); first = Trainer(old)
            torch.manual_seed(5); second = Trainer(new)
            for a,b in zip(first.method.parameters(),second.method.parameters()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_architecture_does_not_choose_schedule(self):
        old = normalize_config({'model': {'model_type':'v2'}, 'optimizer': {'lr_schedule':'constant'}})
        self.assertEqual(old['optimizer']['lr_schedule'], 'constant')
        self.assertTrue(old['backbone']['qkv_bias'])
        new = normalize_config({'backbone': {'qkv_bias': True, 'mlp_ratio': 8./3.}})
        self.assertEqual(new['optimizer']['lr_schedule'], 'constant')
        self.assertEqual(new['method']['momentum_teacher'], .994)

    def test_recipes_are_valid_without_model_type(self):
        root = Path(__file__).resolve().parents[1] / 'dinovol_2'
        for path in [root/'example_config.json', *(root/'configs').glob('*.json')]:
            raw = json.loads(path.read_text())
            for config in [raw, *raw.get('recipes',{}).values()]:
                result = normalize_config(config)
                self.assertNotIn('model_type', result['backbone'])
                self.assertEqual(set(result), {'data','backbone','method','optimizer','runtime'})
