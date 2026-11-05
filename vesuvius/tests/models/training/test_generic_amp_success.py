"""PBS-only actual generic step/EMA/consumer blocks with CPU GradScaler.

The complete _train_step, _get_scaler and _update_ema_model methods are extracted
from the selected arm. Forward/loss/batch preparation use supplied tiny inputs;
actual backward,GradScaler,SGD,EMA and scheduler run. No full trainer or CUDA.
"""
from __future__ import annotations
import ast
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
from time import perf_counter
from types import MethodType,SimpleNamespace
import unittest

import torch


def source_methods():
    source=Path(os.environ['GENERIC_AMP_TRAIN_SOURCE'])
    module=ast.parse(source.read_text(encoding='utf8'))
    cls=next(n for n in module.body if isinstance(n,ast.ClassDef)and n.name=='BaseTrainer')
    names=['_train_step','_get_scaler','_update_ema_model']
    methods=[deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name==name))for name in names]
    consumers=[n for n in ast.walk(module)if isinstance(n,ast.If)and ast.unparse(n.test)=='optimizer_stepped and is_per_iteration_scheduler']
    assert len(consumers)==1
    wrapper=ast.parse('def consume(optimizer_stepped,is_per_iteration_scheduler,scheduler):\n    pass\n').body[0]
    wrapper.body=[deepcopy(consumers[0])]
    code=ast.Module(body=methods+[wrapper],type_ignores=[]);ast.fix_missing_locations(code)
    namespace={'torch':torch,'perf_counter':perf_counter}
    exec(compile(code,str(source),'exec'),namespace)
    return namespace


class ScalarModel(torch.nn.Module):
    def __init__(self,value=2.0,integer=7):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(value))
        self.register_buffer('integer',torch.tensor(integer,dtype=torch.int64))

    def forward(self,slope):return self.weight*slope


class GenericAmpSuccessTests(unittest.TestCase):
    def run_sequence(self,slopes,*,accumulation=1,ema=True,start=0,every=1,
                     scaler_kind='active',growth_interval=100,dummy_options=None):
        methods=source_methods();model=ScalarModel();optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=1.0)
        initial_scheduler=scheduler.last_epoch
        helper=SimpleNamespace(mgr=SimpleNamespace(gradient_clip=12.0,augment_on_device=False),
            device=torch.device('cpu'),startup_timing=False,_ema_optimizer_step=0,
            ema_start_step=start,ema_update_every_steps=every,ema_decay=0.5,
            ema_model=ScalarModel(-2.0,-3)if ema else None,
            _unwrap_model=lambda value:value,_accumulate_aug_timers=lambda data:None,
            _prepare_batch=lambda data,is_training:data)
        for name in ['_train_step','_get_scaler','_update_ema_model']:
            setattr(helper,name,MethodType(methods[name],helper))
        def forward(actual_model,data):
            inputs=data['slope'];return inputs,{}, {'ink':actual_model(inputs)}
        helper._get_model_outputs=forward
        helper._compute_train_loss=lambda outputs,targets,loss_fns:(outputs['ink'],{'ink':outputs['ink']})
        if helper.ema_model is not None:
            # The real _create_ema_model disables parameter gradients as well.
            helper.ema_model.requires_grad_(False).eval()
        if scaler_kind=='dummy':
            scaler=helper._get_scaler(**(dummy_options or {'device_type':'cpu','use_amp':False}))
        else:
            scaler=torch.amp.GradScaler('cpu',enabled=scaler_kind!='disabled',init_scale=8.0,
                growth_factor=2.0,backoff_factor=0.5,growth_interval=growth_interval)
        records=[]
        for step,slope in enumerate(slopes):
            # Same zeroing/window contract as the actual caller;the complete
            # production step and exact scheduler If run below.
            if step%accumulation==0:optimizer.zero_grad(set_to_none=True)
            result=helper._train_step(model,{'slope':torch.tensor(float(slope))},{},False,
                nullcontext(),epoch=1,step=step,scaler=scaler,optimizer=optimizer,
                num_iters=len(slopes),grad_accumulate_n=accumulation)
            methods['consume'](result[-1],True,scheduler)
            records.append({'stepped':result[-1],'weight':model.weight.detach().item(),
                'ema_step':helper._ema_optimizer_step,
                'ema_weight':None if helper.ema_model is None else helper.ema_model.weight.detach().item(),
                'ema_integer':None if helper.ema_model is None else helper.ema_model.integer.item(),
                'scheduler_steps':scheduler.last_epoch-initial_scheduler,
                'scale':scaler.get_scale()if hasattr(scaler,'get_scale')else 1.0})
        print('GENERIC_AMP_CASE='+json.dumps({'test':self.id(),'kind':scaler_kind,
            'ema':ema,'accumulation':accumulation,'start':start,'every':every,'records':records}))
        return records

    def test_overflow_preserves_flag_scheduler_and_ema(self):
        for ema in [True,False]:
            r=self.run_sequence([float('inf')],ema=ema)[0]
            self.assertEqual((r['stepped'],r['scheduler_steps'],r['ema_step']),(False,0,0))
            self.assertEqual((r['weight'],r['scale']),(2.0,4.0))
            if ema:self.assertEqual((r['ema_weight'],r['ema_integer']),(-2.0,-3))

    def test_success_skip_success_keeps_update_history(self):
        r=self.run_sequence([1.0,float('inf'),1.0])
        self.assertEqual([x['stepped']for x in r],[True,False,True])
        self.assertEqual([x['ema_step']for x in r],[1,1,2])
        self.assertEqual([x['scheduler_steps']for x in r],[1,1,2])
        self.assertAlmostEqual(r[-1]['weight'],1.8,places=6)
        self.assertAlmostEqual(r[0]['ema_weight'],-0.05,places=6)
        self.assertEqual(r[1]['ema_weight'],r[0]['ema_weight'])
        self.assertAlmostEqual(r[-1]['ema_weight'],0.875,places=6)
        self.assertEqual(r[-1]['ema_integer'],7)

    def test_overflow_at_accumulation_boundary(self):
        r=self.run_sequence([1.0,float('inf'),1.0,1.0],accumulation=2)
        self.assertEqual([x['stepped']for x in r],[False,False,False,True])
        self.assertEqual([x['ema_step']for x in r],[0,0,0,1])
        self.assertEqual([x['scheduler_steps']for x in r],[0,0,0,1])
        self.assertAlmostEqual(r[-1]['weight'],1.9,places=6)

    def test_ema_warmup_and_cadence_use_successes(self):
        r=self.run_sequence([1.0,float('inf'),1.0],start=2,every=2)
        self.assertEqual([x['ema_step']for x in r],[1,1,2])
        self.assertEqual([x['ema_weight']for x in r[:2]],[-2.0,-2.0])
        self.assertAlmostEqual(r[-1]['ema_weight'],-0.1,places=6)
        self.assertEqual(r[-1]['ema_integer'],7)

    def test_scale_growth_is_a_success_control(self):
        r=self.run_sequence([1.0,1.0],growth_interval=1)
        self.assertEqual([x['scale']for x in r],[16.0,32.0])
        self.assertEqual([x['stepped']for x in r],[True,True])
        self.assertEqual((r[-1]['scheduler_steps'],r[-1]['ema_step']),(2,2))
        self.assertAlmostEqual(r[-1]['weight'],1.8,places=6)

    def test_actual_dummy_scaler_controls(self):
        options=[{'device_type':'cpu','use_amp':False},
            {'device_type':'cpu','use_amp':True},
            {'device_type':'cuda','use_amp':True,'amp_dtype':torch.bfloat16}]
        for option in options:
            r=self.run_sequence([1.0],scaler_kind='dummy',dummy_options=option)[0]
            self.assertEqual((r['stepped'],r['scheduler_steps'],r['ema_step']),(True,1,1))
            self.assertAlmostEqual(r['weight'],1.9,places=6)
            self.assertEqual(r['ema_integer'],7)

    def test_disabled_real_scaler_control(self):
        r=self.run_sequence([1.0],scaler_kind='disabled')[0]
        self.assertEqual((r['stepped'],r['scheduler_steps'],r['ema_step'],r['scale']),(True,1,1,1.0))
        self.assertAlmostEqual(r['weight'],1.9,places=6)

    def test_finite_accumulation_and_final_flush_control(self):
        r=self.run_sequence([1.0,1.0,1.0],accumulation=2)
        self.assertEqual([x['stepped']for x in r],[False,True,True])
        self.assertEqual([x['scheduler_steps']for x in r],[0,1,2])
        self.assertEqual([x['ema_step']for x in r],[0,1,2])
        self.assertAlmostEqual(r[-1]['weight'],1.85,places=6)


if __name__=='__main__':unittest.main(verbosity=2)
