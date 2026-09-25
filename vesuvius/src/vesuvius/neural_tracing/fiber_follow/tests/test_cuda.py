"""BF16/channels-last regressions on the actual accelerator."""
import pytest
import torch
from test_single_path import config,batch,run
from vesuvius.neural_tracing.fiber_follow.model import FollowNet,prepare_model
from vesuvius.neural_tracing.fiber_follow.supervision import loss_fn


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable in this environment')
def test_cuda_bf16_determinism_gradients_and_missing_history():
    torch.manual_seed(123)
    cfg=config();m=prepare_model(FollowNet(cfg),'cuda');b={k:v.cuda() for k,v in batch(cfg).items()}
    b['hmask'][0].zero_();b['hmask'][1,4:]=0;b['plane_mask'][0,2:]=0
    assert m.encoders[0][0].weight.is_contiguous(memory_format=torch.channels_last_3d)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=run(m,b,targets=b)
        other=run(m,b)
        loss,_=loss_fn(out,b,cfg,update=2000)
    torch.testing.assert_close(out['points'],other['points'],rtol=0,atol=0)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.encoders[0][0].weight.grad.abs().sum()>0
    assert m.flow.velocity.weight.grad.abs().sum()>0
    assert m.flow.confidence_head.weight.grad.abs().sum()>0
