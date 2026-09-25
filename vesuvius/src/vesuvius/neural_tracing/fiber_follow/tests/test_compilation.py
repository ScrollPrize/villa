"""Production compilation entry points, preflight matching and checkpoint resume."""
import copy
import functools
import json

import pytest
import torch

from test_single_path import config,batch
from vesuvius.neural_tracing.fiber_follow.model import ARCHITECTURE,FollowNet,prepare_model
from vesuvius.neural_tracing.fiber_follow.train import (
    compile_training_model,main,optimizer_update,read_checkpoint,resume_training,
    save_checkpoint,training_rng_state,
)
from vesuvius.neural_tracing.fiber_follow.data import SampleConfig
from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolumeSpec


def test_compile_entrypoints_preserve_parameters_and_eager_ema(monkeypatch):
    import torch._functorch.config
    monkeypatch.setattr(torch._functorch.config,'backward_pass_autocast','same_as_forward')
    model=FollowNet(config());ema=copy.deepcopy(model)
    parameters=list(model.parameters());keys=list(model.state_dict())
    compiled=[];called=[]
    def compile_method(fn,**kwargs):
        assert kwargs==dict(options={'fallback_random':True})
        compiled.append(fn.__name__)
        @functools.wraps(fn)
        def wrapped(*args,**kw):
            called.append(fn.__name__)
            assert torch._functorch.config.backward_pass_autocast=='off'
            return fn(*args,**kw)
        return wrapped
    monkeypatch.setattr(torch,'compile',compile_method)
    assert compile_training_model(model) is model
    assert compiled==['encode_conditioning','generate_training_curve','training_forward']
    assert list(model.state_dict())==keys
    assert all(a is b for a,b in zip(parameters,model.parameters()))
    assert all(name not in ema.__dict__ for name in compiled)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    optimizer_update(model,ema,opt,[batch(model.cfg)],2000,1e-3,device='cpu')
    assert set(called)==set(compiled)


@pytest.mark.parametrize('compiled_benchmark',[False,True])
def test_preflight_requires_matching_compilation(tmp_path,monkeypatch,compiled_benchmark):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    benchmark=dict(architecture=ARCHITECTURE,passed=True,microbatch=2,flow_draws=64,
                   crop=[176,96,96],cache_training_encoding=True)
    if compiled_benchmark: benchmark['compile_model']=True
    path=tmp_path/'benchmark.json';path.write_text(json.dumps(benchmark))
    args=['--fiber-zarrs','unused','--ct','unused','--fibers','unused','--name','unused',
          '--fixed-bank','unused','--manifest','unused','--benchmark',str(path)]
    if compiled_benchmark: args.append('--no-compile')
    with pytest.raises(ValueError,match='matching full-crop benchmark'):
        main(args)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable in this environment')
def test_cuda_compiled_training_checkpoint_resume(tmp_path):
    torch.manual_seed(73)
    cfg=config();model=prepare_model(FollowNet(cfg),'cuda')
    ema=copy.deepcopy(model).requires_grad_(False).eval()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    compile_training_model(model)
    data=batch(cfg)
    data['hmask'][0].zero_();data['plane_mask'][1,2:]=0
    for step,logged in ((2000,False),(2001,True)):
        loss,metrics,_=optimizer_update(model,ema,opt,[data],step,1e-3,device='cuda',compute_metrics=logged)
        assert torch.isfinite(torch.tensor(loss))
        assert ('refinement' in metrics)==logged
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    sample=SampleConfig(crop=CropSpec(depth=20,width=12,behind=10),n_history=8,recent_history_points=8,n_future=4)
    spec=FiberVolumeSpec('unused',ct_zarr='unused',ct_level=1,ct_grid_scale=8,inputs='ct+presence')
    path=tmp_path/'last.pt'
    save_checkpoint(path,model,ema,spec,sample,dict(step=2001,replay_seen=0,
                    optimizer=opt.state_dict(),rng=training_rng_state()))
    checkpoint=read_checkpoint(path,'cuda')
    resumed=prepare_model(FollowNet(cfg),'cuda');resumed_ema=copy.deepcopy(resumed).requires_grad_(False).eval()
    resumed_opt=torch.optim.AdamW(resumed.parameters(),lr=1e-3)
    assert resume_training(checkpoint,resumed,resumed_ema,resumed_opt)==(2001,0)
    assert model.state_dict().keys()==resumed.state_dict().keys()
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
    for a,b in zip(ema.parameters(),resumed_ema.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert resumed_opt.state_dict()['state']
    compile_training_model(resumed)
    loss,_,_=optimizer_update(resumed,resumed_ema,resumed_opt,[data],2002,1e-3,device='cuda')
    assert torch.isfinite(torch.tensor(loss))
