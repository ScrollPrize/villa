"""Prepared PBS-only ConfigManager/YAML/CLI roundtrips,not full training."""
from __future__ import annotations
import ast
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import yaml
import vesuvius.models.configuration.config_manager as config_module
from vesuvius.models.configuration.config_manager import ConfigManager
from vesuvius.models.training.cli import build_parser
from vesuvius.models.utilities.cli_utils import update_config_from_args


def effective_policy(manager):
    # Execute the exact whole precision-selection block in source order with
    # real Torch dtype objects. No model initialization,autocast or scaler runs.
    path=Path(config_module.__file__).resolve().parents[1]/'training/train.py'
    module=ast.parse(path.read_text(encoding='utf8'))
    cls=next(n for n in module.body if isinstance(n,ast.ClassDef)and n.name=='BaseTrainer')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='_initialize_training')
    first=next(i for i,n in enumerate(method.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='use_amp'for t in n.targets))
    last=next(i for i,n in enumerate(method.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Attribute)and ast.unparse(t)=='self.amp_dtype_str'for t in n.targets))
    helper=SimpleNamespace(mgr=manager,is_distributed=False,rank=0)
    ns={'self':helper,'torch':torch}
    exec(compile(ast.Module(body=deepcopy(method.body[first:last+1]),type_ignores=[]),str(path),'exec'),ns)
    return ns['use_amp'],helper.amp_dtype_str


class AmpPolicyRoundtripTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.initial=self.root/'initial.yaml'

    def config(self,*,tr=None,extra=None):
        config={'tr_setup':{'model_name':'amp_policy_fixture','ckpt_out_base':str(self.root/'out')},
            'tr_config':{'patch_size':[8,8,8],'num_dataloader_workers':0,**(tr or {})},
            'model_config':{},'dataset_config':{'data_path':str(self.root),'targets':{'ink':{'out_channels':2}}}}
        config.update(extra or {});self.initial.write_text(yaml.safe_dump(config),encoding='utf8')
        return self.initial

    def load(self,path,flags=(),*,merge=True):
        manager=ConfigManager(verbose=False);manager.load_config(path)
        if merge:
            args=build_parser().parse_args(['--config',str(path),'--output',str(self.root/'out'),*flags])
            update_config_from_args(manager,args)
        return manager

    def save(self,manager):
        manager.save_config()
        path=Path(manager.ckpt_out_base)/manager.model_name/f'{manager.model_name}_config.yaml'
        self.assertTrue(path.is_file())
        return path,yaml.safe_load(path.read_text(encoding='utf8'))

    def test_saved_bfloat16_survives_reload_without_dtype_flag(self):
        original=self.load(self.config(),['--amp-dtype','bfloat16'])
        self.assertEqual(effective_policy(original),(True,'bfloat16'))
        saved,payload=self.save(original);self.assertEqual(payload['tr_config']['amp_dtype'],'bfloat16')
        before=saved.read_bytes();reloaded=self.load(saved)
        self.assertEqual(effective_policy(reloaded),(True,'bfloat16'))
        self.assertEqual(saved.read_bytes(),before)

    def test_saved_no_amp_survives_reload(self):
        original=self.load(self.config(),['--no-amp'])
        self.assertEqual(effective_policy(original),(False,'float16'))
        saved,payload=self.save(original);self.assertIs(payload['tr_config']['no_amp'],True)
        self.assertEqual(effective_policy(self.load(saved)),(False,'float16'))

    def test_explicit_dtype_does_not_erase_saved_disabled_policy(self):
        original=self.load(self.config(),['--no-amp','--amp-dtype','bfloat16'])
        saved,payload=self.save(original)
        self.assertEqual((payload['tr_config']['no_amp'],payload['tr_config']['amp_dtype']),(True,'bfloat16'))
        reloaded=self.load(saved,['--amp-dtype','float16'])
        self.assertEqual(effective_policy(reloaded),(False,'float16'))
        self.assertEqual(reloaded.tr_configs['amp_dtype'],'float16')

    def test_legacy_missing_policy_keeps_enabled_float16_default(self):
        self.assertEqual(effective_policy(self.load(self.config())),(True,'float16'))

    def test_explicit_dtype_overrides_config_control(self):
        manager=self.load(self.config(tr={'amp_dtype':'float16','no_amp':False}),['--amp-dtype','bfloat16'])
        self.assertEqual(effective_policy(manager),(True,'bfloat16'))
        self.assertEqual(manager.tr_configs['amp_dtype'],'bfloat16')

    def test_explicit_no_amp_overrides_enabled_config_control(self):
        manager=self.load(self.config(tr={'amp_dtype':'float16','no_amp':False}),['--no-amp'])
        self.assertEqual(effective_policy(manager),(False,'float16'))
        self.assertIs(manager.tr_configs['no_amp'],True)

    def test_existing_flattened_precision_attributes_control(self):
        path=self.config(extra={'mean_teacher_config':{'amp_dtype':'bfloat16','no_amp':True}})
        # Existing config API behavior before any CLI merge;absence of canonical
        # keys must not reset attributes supplied by that existing flattening.
        self.assertEqual(effective_policy(self.load(path,merge=False)),(False,'bfloat16'))


if __name__=='__main__':unittest.main(verbosity=2)
