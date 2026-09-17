from copy import deepcopy
import random
import numpy as np
import pytest
import torch
from torch import nn
from vesuvius.ink_detection.training.ema_batchnorm import recalibrate_batchnorm,batchnorm_digest,verify_calibrated_ema


def test_calibration_changes_only_bn_buffers_and_restores_rng_modes():
    model=nn.Sequential(nn.Conv3d(1,2,1),nn.BatchNorm3d(2)).eval()
    batches=[torch.ones(2,1,3,3,3)*v for v in (4.,8.)]
    parameters={n:p.clone() for n,p in model.named_parameters()}
    before=batchnorm_digest(model.state_dict())
    torch.manual_seed(11);random.seed(11);np.random.seed(11)
    torch_rng=torch.get_rng_state().clone();py_rng=random.getstate();np_rng=np.random.get_state()
    def forward(m,b):
        torch.rand(3);random.random();np.random.rand();return m(b)
    result=recalibrate_batchnorm(model,batches,forward)
    assert result['batches_per_rank']==2 and result['buffer_sha256']!=before
    assert model[1].num_batches_tracked==2 and model[1].momentum==.1
    assert all(not m.training for m in model.modules())
    for n,p in model.named_parameters():torch.testing.assert_close(p,parameters[n],rtol=0,atol=0)
    assert torch.equal(torch_rng,torch.get_rng_state()) and random.getstate()==py_rng
    assert np.array_equal(np_rng[1],np.random.get_state()[1])
    expected=deepcopy(model)
    recalibrate_batchnorm(model,batches,forward)
    for n,v in model.state_dict().items():torch.testing.assert_close(v,expected.state_dict()[n],rtol=0,atol=0)


def test_calibration_failure_restores_original_buffers_and_modes():
    model=nn.Sequential(nn.BatchNorm3d(1)).train()
    before=deepcopy(model.state_dict())
    with pytest.raises(ValueError,match='Empty'):
        recalibrate_batchnorm(model,[],lambda m,b:m(b))
    assert model.training and model[0].training and model[0].momentum==.1
    for n,v in model.state_dict().items():torch.testing.assert_close(v,before[n],rtol=0,atol=0)


def test_export_rejects_missing_stale_and_corrupt_calibration():
    recipe={'method':'training_only_cumulative','global_samples':512,'draw_start':0,'batch_size_per_gpu':4}
    state={'bn.running_mean':torch.zeros(1),'bn.running_var':torch.ones(1)}
    cfg={'ema_batchnorm_calibration':recipe,'manifest_sha256':'data','patches_sha256':'split'}
    record={'method':'training_only_cumulative','optimizer_update':12,'recipe':recipe,
            'manifest_sha256':'data','patches_sha256':'split','buffer_sha256':batchnorm_digest(state)}
    payload={'config':cfg,'ema_model':state,'optimizer_step':12,'ema_batchnorm_calibration':record}
    verify_calibrated_ema(payload)
    for change in ({'optimizer_step':13},{'ema_batchnorm_calibration':None}):
        with pytest.raises(ValueError):verify_calibrated_ema({**payload,**change})
    state['bn.running_mean'].add_(1)
    with pytest.raises(ValueError):verify_calibrated_ema(payload)
