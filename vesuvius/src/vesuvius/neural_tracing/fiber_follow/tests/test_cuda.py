"""BF16 and convolution-layout regressions on the actual accelerator."""
import copy
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
    assert m.encoders[0][0].weight.is_contiguous()
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


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable in this environment')
def test_contiguous_layout_matches_channels_last_predictions_and_loss():
    torch.manual_seed(123)
    cfg=config();m=prepare_model(FollowNet(cfg),'cuda')
    reference=copy.deepcopy(m)
    torch.nn.utils.convert_conv3d_weight_memory_format(reference,torch.channels_last_3d)
    assert m.state_dict().keys()==reference.state_dict().keys()
    b={k:v.cuda() for k,v in batch(cfg).items()}
    b['hmask'][0].zero_();b['hmask'][1,4:]=0;b['plane_mask'][0,2:]=0
    outputs=[]
    for model in (reference,m):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=run(model,b,targets=b,return_steps=True,
                    generator=torch.Generator(device='cuda').manual_seed(17))
            loss,_=loss_fn(out,b,cfg,update=2000)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        outputs.append(out)
    # Backend choices can differ by device. BF16 rounding is unchanged; the
    # full-crop RTX 5090 experiment also checks exact forward equality.
    for key in ('points','confidence','confidence_logits','denoising_steps','flow_loss'):
        torch.testing.assert_close(outputs[0][key],outputs[1][key],rtol=1e-3,atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable in this environment')
def test_models_without_layout_override_keep_channels_last():
    model=prepare_model(torch.nn.Sequential(torch.nn.Conv3d(3,4,3)),'cuda')
    assert model[0].weight.is_contiguous(memory_format=torch.channels_last_3d)
